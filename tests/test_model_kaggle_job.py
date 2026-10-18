"""Kaggle training: metadata keys, auth and token masking, dry run without any network, verification and promotion
refusals, and the wait/back-off logic. The end-to-end kernel runs are in test_model_kaggle_kernel.py."""
from __future__ import annotations

import ast
import importlib
import json
import re
import socket
import subprocess
import sys
import time
import types
import zipfile
from pathlib import Path

import numpy as np
import lightgbm as lgb
import pandas as pd
import pytest

import ipo_radar
from ipo_radar import config
from ipo_radar.model import kaggle, kaggle_job as job, tune
from ipo_radar.model.kaggle_job import Kaggle, KaggleJobError, Result
from model_helpers import install_stubs_in_process, make_ipos, write_inputs

FAKE_TOKEN = "KGAT_fake_token_value_123"


def point_config_at(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(config, "MODELS", root / "models")
    monkeypatch.setattr(config, "IPOS_PARQUET", root / "data" / "ipos.parquet")
    monkeypatch.setattr(config, "INDICES_PARQUET", root / "data" / "indices.parquet")
    write_inputs(root / "data")


def no_network_or_subprocess(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("this step must not start a process or open a network connection")

    for name in ("run", "Popen", "check_output", "call", "check_call"):
        monkeypatch.setattr(subprocess, name, boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket.socket, "connect", boom)


# ---- metadata -------------------------------------------------------------------------------------
def test_kernel_metadata_has_exact_snake_case_keys_and_cpu_no_internet():
    meta = job.kernel_metadata("alice")
    assert set(meta) == {"id", "title", "code_file", "language", "kernel_type", "is_private", "enable_gpu",
                         "enable_tpu", "enable_internet", "dataset_sources", "competition_sources",
                         "kernel_sources", "model_sources"}
    assert meta["id"] == "alice/ipo-radar-train" and meta["title"] == "ipo-radar-train"
    assert meta["code_file"] == "kernel.py" and meta["language"] == "python" and meta["kernel_type"] == "script"
    assert meta["is_private"] is True
    assert meta["enable_gpu"] is False and meta["enable_tpu"] is False and meta["enable_internet"] is False
    assert meta["dataset_sources"] == ["alice/ipo-radar-train-inputs"]
    assert "machine_shape" not in meta  # default machine is CPU
    assert not [k for k in meta if re.search(r"[a-z][A-Z]", k)], "keys must be snake_case, camelCase is ignored"


def test_dataset_metadata_is_private_other_licence():
    meta = job.dataset_metadata("alice")
    assert meta["id"] == "alice/ipo-radar-train-inputs"
    assert meta["licenses"] == [{"name": "other"}]
    assert set(meta) == {"id", "title", "subtitle", "description", "licenses"}  # private is the default for create


def test_kernel_imports_only_what_kaggle_preinstalls():
    tree = ast.parse(job.kernel_source("r1", job.run_profile(False, 90)))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module.split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"pandas", "numpy", "joblib", "lightgbm", "sklearn", "optuna",
                                              "ipo_radar"}
    assert names <= allowed, names - allowed
    assert not names & {"urllib", "http", "socket", "requests", "subprocess", "ssl"}


def test_kernel_source_has_no_install_or_network_calls():
    source = job.kernel_source("r1", job.run_profile(False, 90))
    assert not re.search(r"\b(pip|urllib|requests|socket|subprocess|httpx)\b", source)


def test_kernel_source_embeds_run_id_and_profile_and_no_placeholders_remain():
    profile = job.run_profile(True, 90)
    source = job.kernel_source("20261015-101010", profile)
    assert "RUN_ID = '20261015-101010'" in source
    assert f"PROFILE = {profile!r}" in source
    assert "__RUN_ID__" not in source and "__PROFILE__" not in source


def test_run_profiles():
    quick = job.run_profile(True, 90)
    assert quick["quick"] is True and quick["n_trials"] == 20 and quick["final_seeds"] == 1
    full = job.run_profile(False, 90)
    assert full["quick"] is False and full["n_trials"] == 300 and full["final_seeds"] == 5
    assert full["search_budget_s"] == 90 * 60 - job.SEARCH_RESERVE_S
    assert job.run_profile(False, 5)["search_budget_s"] == 600  # never below ten minutes


def test_copy_package_ships_plain_python_files_and_no_zip(tmp_path):
    pkg = Path(ipo_radar.__file__).resolve().parent
    names = job.copy_package(tmp_path / "up", pkg)
    assert names and all(n.startswith("ipo_radar_src/ipo_radar/") and n.endswith(".py") for n in names)
    assert "ipo_radar_src/ipo_radar/model/tune.py" in names
    assert not any("__pycache__" in n for n in names)
    assert not list((tmp_path / "up").rglob("*.zip"))
    for n in names:
        assert (tmp_path / "up" / n).read_bytes() == (pkg / n[len("ipo_radar_src/ipo_radar/"):]).read_bytes()
    with pytest.raises(ValueError, match="ipo_radar folder"):
        job.copy_package(tmp_path / "c", tmp_path)

def test_manifest_keys_are_the_paths_a_service_that_extracts_zips_would_mount(tmp_path):
    """The guard for the first real-run failure: a manifest written against a zip that the service unpacks into a
    folder must not pass as a match. Mounts the upload through the service emulation and checks every key."""
    from model_helpers import kaggle_service_transform

    ipos, indices = write_inputs(tmp_path / "data")
    upload = tmp_path / "up"
    manifest = job.write_upload(upload, run_id="r1", username="alice", ipos=ipos, indices=indices,
                                package_dir=Path(ipo_radar.__file__).resolve().parent)
    mount = tmp_path / "mount"
    kaggle_service_transform(upload, mount)
    for name, meta in manifest["files"].items():
        path = mount.joinpath(*name.split("/"))
        assert path.is_file(), f"{name} is not where the manifest says after the service"
        assert job.sha256_file(path) == meta["sha256"], name
    assert not list(mount.rglob("*.zip"))


def test_manifest_lists_checksums_of_every_file_that_ships(tmp_path):
    ipos, indices = write_inputs(tmp_path / "data")
    upload = tmp_path / "up"
    manifest = job.write_upload(upload, run_id="r1", username="alice", ipos=ipos, indices=indices,
                                package_dir=Path(ipo_radar.__file__).resolve().parent)
    assert manifest["run_id"] == "r1" and manifest["source_dir"] == job.SOURCE_DIR
    assert "source_zip" not in manifest
    shipped = {p.relative_to(upload).as_posix() for p in upload.rglob("*") if p.is_file()}
    shipped -= {job.MANIFEST, "dataset-metadata.json"}
    assert set(manifest["files"]) == shipped, "every shipped file, and only those, must be in the manifest"
    for name, meta in manifest["files"].items():
        assert meta["sha256"] == job.sha256_file(upload.joinpath(*name.split("/")))
        assert meta["bytes"] == upload.joinpath(*name.split("/")).stat().st_size
    assert json.loads((upload / "manifest.json").read_text())["run_id"] == "r1"

# ---- auth and masking ------------------------------------------------------------------------------
def test_token_from_environment_wins(tmp_path, monkeypatch):
    (tmp_path / ".kaggle").mkdir()
    (tmp_path / ".kaggle" / "kaggle.json").write_text("{}")
    auth = job.resolve_auth({job.TOKEN_ENV: FAKE_TOKEN}, home=tmp_path)
    assert auth.token == FAKE_TOKEN and FAKE_TOKEN not in auth.source


def test_existing_cli_login_is_accepted_by_existence_only(tmp_path):
    (tmp_path / ".kaggle").mkdir()
    (tmp_path / ".kaggle" / "access_token").write_text("not read by us")
    auth = job.resolve_auth({}, home=tmp_path)
    assert auth.token is None and "access_token" in auth.source


def test_no_login_gives_a_clear_error_with_both_fixes(tmp_path):
    with pytest.raises(KaggleJobError) as err:
        job.resolve_auth({}, home=tmp_path)
    assert "kaggle auth login" in str(err.value) and job.TOKEN_ENV in str(err.value)


def test_mask_secret_removes_every_occurrence_and_handles_no_secret():
    assert job.mask_secret(f"a {FAKE_TOKEN} b {FAKE_TOKEN}", FAKE_TOKEN) == "a *** b ***"
    assert job.mask_secret("plain", None) == "plain"
    assert job.mask_secret("plain", "") == "plain"


def test_token_goes_in_the_child_environment_never_in_arguments():
    seen = {}

    def runner(args, **kwargs):
        seen["args"], seen["env"] = args, kwargs["env"]
        return subprocess.CompletedProcess(args, 0, stdout=f"ok {FAKE_TOKEN}\n", stderr="")

    client = Kaggle("kaggle", FAKE_TOKEN, runner=runner)
    res = client.run("datasets", "status", "a/b")
    assert FAKE_TOKEN not in " ".join(seen["args"])
    assert seen["env"][job.TOKEN_ENV] == FAKE_TOKEN
    assert FAKE_TOKEN not in res.text and "***" in res.text


def test_errors_built_from_cli_output_never_show_the_token(tmp_path):
    def runner(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr=f"403 for token {FAKE_TOKEN}")

    client = Kaggle("kaggle", FAKE_TOKEN, runner=runner)
    with pytest.raises(KaggleJobError) as err:
        job.ensure_dataset(client, tmp_path, "alice/ipo-radar-train-inputs", "msg")
    assert FAKE_TOKEN not in str(err.value) and "***" in str(err.value)

def test_timeout_and_missing_binary_are_clear_errors():
    def slow(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 1)

    def missing(args, **kwargs):
        raise FileNotFoundError(args[0])

    with pytest.raises(KaggleJobError, match="did not answer"):
        Kaggle("kaggle", None, runner=slow, timeout_s=1).run("kernels", "status", "a/b")
    with pytest.raises(KaggleJobError, match="pip install kaggle"):
        Kaggle("kaggle", None, runner=missing).run("kernels", "status", "a/b")


def test_kernel_state_reads_only_the_first_line():
    class Client:
        def run(self, *args):
            return Result(0, 'a/b has status "KernelWorkerStatus.COMPLETE"\nFailure message: "RUNNING later"')

    assert job.kernel_state(Client(), "a/b") == "COMPLETE"
    assert job.kernel_state(types.SimpleNamespace(run=lambda *a: Result(1, "404")), "a/b") == "UNKNOWN"


@pytest.mark.parametrize("env, config_text, expected", [
    ({job.USER_ENV: "alice"}, None, "alice"),
    ({}, "Configuration values from /x\n- username: bob\n- auth_method: y", "bob"),
])
def test_username_from_env_or_config_view(env, config_text, expected):
    class Client:
        def run(self, *args):
            return Result(0, config_text or "")

    assert job.resolve_username(Client(), env) == expected


def test_username_missing_or_bad_is_refused():
    class Client:
        def run(self, *args):
            return Result(0, "Configuration values from /x")

    with pytest.raises(KaggleJobError, match=job.USER_ENV):
        job.resolve_username(Client(), {})
    with pytest.raises(KaggleJobError, match="slug"):
        job.resolve_username(Client(), {job.USER_ENV: "bad name/with slash"})


# ---- dry run and refusals before any upload ------------------------------------------------------
def test_dry_run_builds_everything_and_makes_no_process_or_network_call(tmp_path, monkeypatch, capsys):
    point_config_at(monkeypatch, tmp_path)
    monkeypatch.setenv(job.TOKEN_ENV, FAKE_TOKEN)
    monkeypatch.delenv(job.USER_ENV, raising=False)
    no_network_or_subprocess(monkeypatch)

    run_dir = kaggle.run_train(dry_run=True)

    out = capsys.readouterr().out
    assert "dry run" in out and "nothing was sent" in out
    assert FAKE_TOKEN not in out
    assert f"set {job.USER_ENV}" in out  # username unknown without the network
    # the listing is what would be sent: parquets and every package file, with sizes and checksum prefixes
    for name in ("ipos.parquet", "indices.parquet", "ipo_radar_src/ipo_radar/model/tune.py",
                 "ipo_radar_src/ipo_radar/__init__.py"):
        assert name in out, name
    assert "ipo_radar_src.zip" not in out and not list((run_dir / "upload").rglob("*.zip"))
    assert (run_dir / "upload" / "manifest.json").is_file()
    assert (run_dir / "upload" / "dataset-metadata.json").is_file()
    assert (run_dir / "kernel" / job.KERNEL_FILE).is_file()
    meta = json.loads((run_dir / "kernel" / "kernel-metadata.json").read_text())
    assert meta["id"] == f"{job.CONFIG_TEMPLATE_NAME}/ipo-radar-train"
    for path in run_dir.rglob("*"):
        if path.is_file() and path.suffix not in (".parquet",):
            assert FAKE_TOKEN not in path.read_text(errors="ignore"), path

def test_dry_run_with_username_env_uses_it_in_the_metadata(tmp_path, monkeypatch):
    point_config_at(monkeypatch, tmp_path)
    monkeypatch.setenv(job.USER_ENV, "alice")
    monkeypatch.delenv(job.TOKEN_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    no_network_or_subprocess(monkeypatch)
    run_dir = kaggle.run_train(dry_run=True)
    meta = json.loads((run_dir / "upload" / "dataset-metadata.json").read_text())
    assert meta["id"] == "alice/ipo-radar-train-inputs"


def test_real_run_without_login_uploads_nothing(tmp_path, monkeypatch):
    point_config_at(monkeypatch, tmp_path)
    monkeypatch.delenv(job.TOKEN_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    no_network_or_subprocess(monkeypatch)
    with pytest.raises(KaggleJobError, match="kaggle auth login"):
        kaggle.run_train()
    assert not (tmp_path / "models").exists()


def test_run_without_built_data_says_how_to_fix(tmp_path, monkeypatch):
    point_config_at(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "IPOS_PARQUET", tmp_path / "missing" / "ipos.parquet")
    with pytest.raises(KaggleJobError, match="ipo data build"):
        kaggle.run_train(dry_run=True)


@pytest.mark.parametrize("kwargs", [{"poll_s": 0}, {"timeout_min": 0}])
def test_bad_timing_arguments_are_refused(tmp_path, monkeypatch, kwargs):
    point_config_at(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        kaggle.run_train(dry_run=True, **kwargs)


# ---- verification -----------------------------------------------------------------------------------
def make_output(out: Path, ipos: Path, run_id: str = "r1") -> Path:
    """What a successful kernel leaves in /kaggle/working, produced with the in-process stand-ins."""
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(ipos)
    best, table = tune.search(df, n_trials=2, seed=1, n_folds=3)
    table.to_csv(out / "tune_trials.csv", index=False)
    train = importlib.import_module("ipo_radar.model.train")
    train.train(df, out_dir=out, seed=1, quick=True, params=best, seeds=[1])
    bundle = out / "bundle.joblib"
    metrics = {"status": "ok", "run_id": run_id, "bundle_sha256": job.sha256_file(bundle)}
    (out / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (out / "log.txt").write_text("done\n", encoding="utf-8")
    return out


@pytest.fixture
def stubs(monkeypatch):
    install_stubs_in_process(monkeypatch)


def test_verify_accepts_a_good_output(tmp_path, stubs):
    ipos, _ = write_inputs(tmp_path / "data")
    out = make_output(tmp_path / "out", ipos)
    verdict = job.verify_outputs(out, run_id="r1", ipos_path=ipos, uploaded_ipos_sha256=job.sha256_file(ipos))
    assert verdict.ok, verdict.errors
    assert any("finite" in c for c in verdict.checks)


def test_verify_refuses_a_tampered_bundle(tmp_path, stubs):
    ipos, _ = write_inputs(tmp_path / "data")
    out = make_output(tmp_path / "out", ipos)
    with (out / "bundle.joblib").open("ab") as handle:
        handle.write(b"x")
    verdict = job.verify_outputs(out, run_id="r1", ipos_path=ipos, uploaded_ipos_sha256=job.sha256_file(ipos))
    assert not verdict.ok and any("checksum" in e for e in verdict.errors)


def test_verify_refuses_a_failed_kernel_and_a_wrong_run(tmp_path, stubs):
    ipos, _ = write_inputs(tmp_path / "data")
    out = make_output(tmp_path / "out", ipos, run_id="r1")
    metrics = json.loads((out / "metrics.json").read_text())
    metrics["status"], metrics["error"] = "error", "boom"
    (out / "metrics.json").write_text(json.dumps(metrics))
    verdict = job.verify_outputs(out, run_id="r2", ipos_path=ipos, uploaded_ipos_sha256=job.sha256_file(ipos))
    assert not verdict.ok
    assert any("status 'error'" in e for e in verdict.errors) and any("run 'r1'" in e for e in verdict.errors)


def test_verify_refuses_when_local_data_changed_since_upload(tmp_path, stubs):
    ipos, _ = write_inputs(tmp_path / "data")
    out = make_output(tmp_path / "out", ipos)
    uploaded = job.sha256_file(ipos)
    make_ipos(200, seed=99).to_parquet(ipos)
    verdict = job.verify_outputs(out, run_id="r1", ipos_path=ipos, uploaded_ipos_sha256=uploaded)
    assert not verdict.ok and any("changed since the upload" in e for e in verdict.errors)


def test_verify_refuses_non_finite_predictions(tmp_path, stubs, monkeypatch):
    ipos, _ = write_inputs(tmp_path / "data")
    out = make_output(tmp_path / "out", ipos)
    monkeypatch.setattr(lgb.LGBMRegressor, "predict", lambda self, X, **kwargs: np.full(len(X), np.nan))
    verdict = job.verify_outputs(out, run_id="r1", ipos_path=ipos, uploaded_ipos_sha256=job.sha256_file(ipos))
    assert not verdict.ok and any("non-finite" in e for e in verdict.errors)


def test_verify_refuses_a_file_that_is_not_a_bundle(tmp_path, stubs):
    ipos, _ = write_inputs(tmp_path / "data")
    out = make_output(tmp_path / "out", ipos)
    import joblib

    joblib.dump({"not": "a bundle"}, out / "bundle.joblib")
    metrics = json.loads((out / "metrics.json").read_text())
    metrics["bundle_sha256"] = job.sha256_file(out / "bundle.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics))
    verdict = job.verify_outputs(out, run_id="r1", ipos_path=ipos, uploaded_ipos_sha256=job.sha256_file(ipos))
    assert not verdict.ok and any("not a model Bundle" in e for e in verdict.errors)


def test_verify_refuses_a_bundle_that_does_not_load(tmp_path, stubs):
    ipos, _ = write_inputs(tmp_path / "data")
    out = make_output(tmp_path / "out", ipos)
    (out / "bundle.joblib").write_bytes(b"not a pickle")
    metrics = json.loads((out / "metrics.json").read_text())
    metrics["bundle_sha256"] = job.sha256_file(out / "bundle.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics))
    verdict = job.verify_outputs(out, run_id="r1", ipos_path=ipos, uploaded_ipos_sha256=job.sha256_file(ipos))
    assert not verdict.ok and any("bundle check failed" in e for e in verdict.errors)


def test_verify_lists_missing_outputs(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    verdict = job.verify_outputs(out, run_id="r1", ipos_path=tmp_path, uploaded_ipos_sha256="")
    assert not verdict.ok and "bundle.joblib" in verdict.errors[0]


# ---- promotion ------------------------------------------------------------------------------------
def test_promote_keeps_the_previous_bundle(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "bundle.joblib").write_bytes(b"old bundle")
    new = tmp_path / "new.joblib"
    new.write_bytes(b"new bundle")
    job.promote(new, models, job.sha256_file(new))
    assert (models / "bundle.joblib").read_bytes() == b"new bundle"
    assert (models / "bundle.prev.joblib").read_bytes() == b"old bundle"
    assert not list(models.glob(".bundle*"))


def test_promote_refuses_a_copy_that_does_not_match_and_changes_nothing(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "bundle.joblib").write_bytes(b"working")
    new = tmp_path / "new.joblib"
    new.write_bytes(b"new")
    with pytest.raises(KaggleJobError, match="was not changed"):
        job.promote(new, models, "0" * 64)
    assert (models / "bundle.joblib").read_bytes() == b"working"
    assert not (models / "bundle.prev.joblib").exists()
    assert not list(models.glob(".bundle*"))


# ---- waiting, back-off and the push guard -----------------------------------------------------------
class Scripted:
    """A client whose kernel status follows a script. Records every call."""

    def __init__(self, states: list[str]):
        self.states = list(states)
        self.calls: list[tuple[str, ...]] = []

    def run(self, *args: str) -> Result:
        self.calls.append(args)
        if args[:2] == ("kernels", "status"):
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            if state == "UNKNOWN":
                return Result(1, "404")
            return Result(0, f'{args[2]} has status "KernelWorkerStatus.{state}"')
        return Result(0, "ok")


def test_wait_backs_off_and_caps_the_interval(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    client = Scripted(["RUNNING"] * 7 + ["COMPLETE"])
    assert job.wait_kernel(client, "a/b", poll_s=10, timeout_min=60, say=lambda m: None) == "COMPLETE"
    assert sleeps[0] == 10
    assert sleeps == sorted(sleeps) and max(sleeps) <= job.POLL_MAX_S
    assert sleeps[1] == pytest.approx(15)


def test_wait_past_the_deadline_leaves_the_kernel_running(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    with pytest.raises(KaggleJobError, match="keeps running on Kaggle"):
        job.wait_kernel(Scripted(["RUNNING"]), "a/b", poll_s=1, timeout_min=0.0001, say=lambda m: None)


def test_push_is_refused_while_a_session_is_queued_or_running():
    client = Scripted(["QUEUED"])
    with pytest.raises(KaggleJobError, match="already QUEUED"):
        job.push_kernel(client, Path("."), "a/b")
    assert not [c for c in client.calls if c[:2] == ("kernels", "push")]


def test_dataset_wait_reports_an_error_state(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)

    class Client:
        def run(self, *args):
            return Result(0, 'a/b has status "error"')

    with pytest.raises(KaggleJobError, match="failed to process"):
        job.wait_dataset_ready(Client(), "a/b", "R1", 1, 1, lambda m: None)

def test_finish_run_with_an_errored_kernel_keeps_the_working_bundle(tmp_path, monkeypatch):
    point_config_at(monkeypatch, tmp_path)
    models = config.MODELS
    models.mkdir(parents=True)
    (models / "bundle.joblib").write_bytes(b"working bundle")
    run_dir = tmp_path / "models" / "kaggle-x"
    run_dir.mkdir()
    job.write_run_state(run_dir, {"run_id": "x", "kernel_ref": "a/b", "uploaded_ipos_sha256": "",
                                  "status": "pushed"})
    client = Scripted(["ERROR"])
    with pytest.raises(KaggleJobError, match="kernel ended in ERROR"):
        kaggle.finish_run(run_dir, client=client, poll_s=0.01, timeout_min=1, say=lambda m: None)
    assert (models / "bundle.joblib").read_bytes() == b"working bundle"
    assert job.read_run_state(run_dir)["status"] == "kernel-error"


def test_cli_visible_error_names_the_problem_without_a_traceback(tmp_path, monkeypatch):
    """The user runs `ipo train --where kaggle`; a missing login must come out as one line, exit 1."""
    from typer.testing import CliRunner

    from ipo_radar.cli.app import app

    point_config_at(monkeypatch, tmp_path)
    monkeypatch.delenv(job.TOKEN_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    no_network_or_subprocess(monkeypatch)
    result = CliRunner().invoke(app, ["--plain", "train", "--where", "kaggle"])
    assert result.exit_code == 1, result.output
    assert "kaggle auth login" in result.output
    assert "Traceback" not in result.output


# ---- dataset version race: create-or-version, failure text, and the run marker -----------------------
class DatasetCLI:
    """Answers `datasets` calls like the real CLI. `states` gives one entry per listing read (no page token): None
    for a missing dataset, else a list of pages of file names (the last state repeats). `status` gives one line per
    status read (the last repeats). create and version return the given replies."""

    def __init__(self, states=(None,), status=("ready",), create=Result(0, "Dataset created"),
                 version=Result(0, "Dataset version created")):
        self.states, self.statuses = list(states), list(status)
        self.create_reply, self.version_reply = create, version
        self.calls: list[tuple[str, ...]] = []
        self.listing_reads = 0
        self.status_reads = 0
        self._state = None

    def run(self, *args: str) -> Result:
        self.calls.append(args)
        if args[:2] == ("datasets", "files"):
            if "--page-token" not in args:
                self._state = self.states[min(self.listing_reads, len(self.states) - 1)]
                self.listing_reads += 1
            if self._state is None:
                return Result(0, "Dataset not found")
            page = int(args[args.index("--page-token") + 1][1:]) if "--page-token" in args else 0
            names = self._state[page]
            text = "name  size  creationDate\n" + "\n".join(f"{n}  1  2026-10-15" for n in names)
            if page + 1 < len(self._state):
                text += f"\nNext Page Token = t{page + 1}"
            return Result(0, text)
        if args[:2] == ("datasets", "status"):
            line = self.statuses[min(self.status_reads, len(self.statuses) - 1)]
            self.status_reads += 1
            return Result(0, f'{args[2]} has status "{line}"')
        if args[:2] == ("datasets", "create"):
            return self.create_reply
        if args[:2] == ("datasets", "version"):
            return self.version_reply
        return Result(1, f"unknown command {args}")

    def writes(self) -> list[str]:
        return [c[1] for c in self.calls if c[0] == "datasets" and c[1] in ("create", "version")]


class ErrorListing:
    def __init__(self):
        self.calls: list[tuple[str, ...]] = []

    def run(self, *args: str) -> Result:
        self.calls.append(args)
        return Result(1, "403 Forbidden")


def sleeps_recorded(monkeypatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    return sleeps


def test_existing_dataset_gets_exactly_one_version_and_no_create(tmp_path):
    cli = DatasetCLI(states=[[["manifest.json", "ipos.parquet"]]])
    assert job.ensure_dataset(cli, tmp_path, "alice/ipo-radar-train-inputs", "run R1") == "new version"
    assert cli.writes() == ["version"]


def test_missing_dataset_gets_exactly_one_create_and_no_version(tmp_path):
    cli = DatasetCLI(states=[None])
    assert job.ensure_dataset(cli, tmp_path, "alice/ipo-radar-train-inputs", "run R1") == "created"
    assert cli.writes() == ["create"]


def test_exit_zero_with_already_exists_warning_is_a_failure(tmp_path):
    cli = DatasetCLI(states=[None], create=Result(0, "Warning: dataset already exists; nothing was uploaded"))
    with pytest.raises(KaggleJobError, match="dataset upload failed \\(created\\).*already exists"):
        job.ensure_dataset(cli, tmp_path, "alice/ipo-radar-train-inputs", "run R1")


def test_exit_zero_with_error_text_on_version_is_a_failure(tmp_path):
    cli = DatasetCLI(states=[[["manifest.json"]]], version=Result(0, "Error: could not create the version"))
    with pytest.raises(KaggleJobError, match="new version"):
        job.ensure_dataset(cli, tmp_path, "alice/ipo-radar-train-inputs", "run R1")


def test_exit_zero_with_unsuccessful_upload_is_a_failure(tmp_path):
    cli = DatasetCLI(states=[None], create=Result(0, "Starting upload for file ipos.parquet\nUpload unsuccessful"))
    with pytest.raises(KaggleJobError, match="unsuccessful"):
        job.ensure_dataset(cli, tmp_path, "alice/ipo-radar-train-inputs", "run R1")


def test_listing_error_stops_before_any_upload(tmp_path):
    client = ErrorListing()
    with pytest.raises(KaggleJobError, match="could not list dataset"):
        job.ensure_dataset(client, tmp_path, "alice/ipo-radar-train-inputs", "run R1")
    assert not [c for c in client.calls if c[1] in ("create", "version")]


def test_listing_follows_page_tokens_to_a_marker_on_page_two(monkeypatch):
    sleeps = sleeps_recorded(monkeypatch)
    cli = DatasetCLI(states=[[["manifest.json"], ["ipo_radar_src/ipo_radar/__init__.py", "run-R1.txt"]]])
    job.wait_dataset_ready(cli, "alice/ipo-radar-train-inputs", "R1", 0.01, 1, lambda m: None)
    assert sleeps == []
    assert any(c[:2] == ("datasets", "files") and "--page-token" in c for c in cli.calls)


def test_ready_but_marker_missing_keeps_waiting(monkeypatch):
    sleeps = sleeps_recorded(monkeypatch)
    said: list[str] = []
    cli = DatasetCLI(states=[[["manifest.json"]], [["manifest.json"]], [["manifest.json", "run-R1.txt"]]],
                     status=("ready",))
    job.wait_dataset_ready(cli, "alice/ipo-radar-train-inputs", "R1", 0.01, 1, said.append)
    assert len(sleeps) == 2
    assert sum("waiting for run-R1.txt" in m for m in said) == 2


def test_marker_listed_but_status_not_ready_keeps_waiting(monkeypatch):
    sleeps = sleeps_recorded(monkeypatch)
    cli = DatasetCLI(states=[[["manifest.json", "run-R1.txt"]]], status=("processing", "processing", "ready"))
    job.wait_dataset_ready(cli, "alice/ipo-radar-train-inputs", "R1", 0.01, 1, lambda m: None)
    assert len(sleeps) == 2


def test_marker_and_ready_proceed_without_waiting(monkeypatch):
    sleeps = sleeps_recorded(monkeypatch)
    cli = DatasetCLI(states=[[["run-R1.txt"]]], status=("ready",))
    job.wait_dataset_ready(cli, "alice/ipo-radar-train-inputs", "R1", 0.01, 1, lambda m: None)
    assert sleeps == []


def test_timeout_message_names_the_marker_and_says_to_wait(monkeypatch):
    sleeps_recorded(monkeypatch)
    cli = DatasetCLI(states=[[["manifest.json"]]], status=("ready",))
    with pytest.raises(KaggleJobError) as err:
        job.wait_dataset_ready(cli, "alice/ipo-radar-train-inputs", "R1", 0.01, 0.000001, lambda m: None)
    message = str(err.value)
    assert "run-R1.txt" in message and "status ready" in message and "run again" in message


def test_marker_is_written_and_listed_in_the_manifest(tmp_path):
    ipos, indices = write_inputs(tmp_path / "data")
    manifest = job.write_upload(tmp_path / "up", run_id="R1", username="alice", ipos=ipos, indices=indices,
                                package_dir=Path(ipo_radar.__file__).resolve().parent)
    assert (tmp_path / "up" / "run-R1.txt").read_text() == "run R1\n"
    assert "run-R1.txt" in manifest["files"]


def test_older_version_error_gets_the_wait_hint_and_keeps_the_run_folder(tmp_path, monkeypatch):
    point_config_at(monkeypatch, tmp_path)
    run_dir = config.MODELS / "kaggle-x"
    (run_dir / "output").mkdir(parents=True)
    (run_dir / "output" / "metrics.json").write_text(json.dumps({
        "status": "error",
        "error": "ValueError: dataset is run 1 but this kernel is run 2: Kaggle mounted an older dataset version",
    }))
    job.write_run_state(run_dir, {"run_id": "x", "kernel_ref": "a/b", "uploaded_ipos_sha256": "", "status": "pushed"})
    with pytest.raises(KaggleJobError, match="Wait a minute and run again"):
        kaggle.finish_run(run_dir, client=Scripted(["ERROR"]), poll_s=0.01, timeout_min=1, say=lambda m: None)
    assert run_dir.is_dir() and (run_dir / "run.json").is_file()

