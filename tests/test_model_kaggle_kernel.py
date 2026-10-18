"""The Kaggle kernel run end to end on this machine.

The kernel is the real generated script, run as a subprocess with IPO_KAGGLE_INPUT and IPO_KAGGLE_WORKING standing
in for /kaggle/input and /kaggle/working. The package it ships is a copy whose train.py is the stand-in; tune,
features and bundle run for real. The kaggle CLI is replaced by FakeKaggle (see model_helpers.py)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

import ipo_radar
from ipo_radar import config
from ipo_radar.model import kaggle, kaggle_job as job
from ipo_radar.model.kaggle_job import KaggleJobError
from model_helpers import (FakeKaggle, install_stubs_in_process, kaggle_service_transform, kernel_env,
                           package_copy, write_inputs)

FAKE_TOKEN = "KGAT_kernel_test_token_456"
SMALL_PROFILE = {"quick": True, "n_trials": 4, "final_seeds": 2, "search_budget_s": 60, "seed": 1}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Config paths, the stand-in package, the in-process stand-ins, and a FakeKaggle wired in as the client."""
    data = tmp_path / "data"
    ipos, indices = write_inputs(data)
    monkeypatch.setattr(config, "MODELS", tmp_path / "models")
    monkeypatch.setattr(config, "IPOS_PARQUET", ipos)
    monkeypatch.setattr(config, "INDICES_PARQUET", indices)
    pkg = package_copy(tmp_path / "shipped")
    monkeypatch.setattr(ipo_radar, "__file__", str(pkg / "__init__.py"))
    install_stubs_in_process(monkeypatch)
    monkeypatch.setattr(job, "run_profile", lambda quick, timeout_min: dict(SMALL_PROFILE))
    monkeypatch.setenv(job.USER_ENV, "tester")
    monkeypatch.setenv(job.TOKEN_ENV, FAKE_TOKEN)
    fake = FakeKaggle(tmp_path / "cloud")
    monkeypatch.setattr(job, "connect", lambda: fake)
    return {"tmp": tmp_path, "fake": fake, "pkg": pkg, "ipos": ipos, "indices": indices}


def secret_free(*roots: Path) -> None:
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix not in (".zip", ".parquet", ".joblib"):
                assert FAKE_TOKEN not in path.read_text(errors="ignore"), path


def run_full(sleep_first: bool = False) -> Path:
    if sleep_first:
        time.sleep(1.1)  # run folders are named by the second
    return kaggle.run_train(wait=True, poll_s=0.01, timeout_min=5)


def test_full_run_ships_a_verified_bundle_and_keeps_secrets_out(world):
    run_dir = run_full()
    models = config.MODELS
    out = run_dir / "output"
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["status"] == "ok" and metrics["run_id"] == job.read_run_state(run_dir)["run_id"]
    assert metrics["completed_trials"] == 4 and metrics["trial_rows"] == 4
    assert metrics["train_report"]["params"] == metrics["best_params"], "the searched params must reach train()"
    assert metrics["train_report"]["stages"], "train() must have fitted at least one stage"
    assert metrics["train_report"]["seeds"] == [SMALL_PROFILE["seed"], SMALL_PROFILE["seed"] + 1]
    assert {f["market"] for f in metrics["folds"]} >= {"IN_MAIN", "US"}
    assert metrics["versions"]["pandas"] and metrics["versions"]["optuna"]
    assert metrics["bundle_sha256"] == job.sha256_file(models / "bundle.joblib")
    assert job.read_run_state(run_dir)["status"] == "promoted"
    assert not (models / "bundle.prev.joblib").exists()  # nothing to keep on the first run
    trials = (out / "tune_trials.csv").read_text().splitlines()
    assert len(trials) == 5 and trials[0].startswith("trial,state,score")  # header + 4 trials
    log_text = (out / "log.txt").read_text()
    assert "search: " in log_text and "done:" in log_text
    # guard for a service that changes uploads: the first log line is the tree Kaggle mounted, and it lists the
    # manifest and every package file
    lines = log_text.splitlines()
    assert "dataset tree under" in lines[0]
    tree = "\n".join(lines[1:200])
    assert "manifest.json" in tree and "ipo_radar_src/ipo_radar/model/tune.py" in tree
    fake = world["fake"]
    assert fake.kernel_exit == 0 and fake.pushes == 1
    assert any(c[:2] == ["datasets", "create"] and "-t" in c for c in fake.calls)
    secret_free(run_dir, models)


def test_a_second_run_keeps_the_previous_bundle_as_prev(world):
    first_run = run_full()
    first = job.sha256_file(config.MODELS / "bundle.joblib")
    second_run = run_full(sleep_first=True)
    assert job.sha256_file(config.MODELS / "bundle.prev.joblib") == first
    new_metrics = json.loads((second_run / "output" / "metrics.json").read_text())
    assert job.sha256_file(config.MODELS / "bundle.joblib") == new_metrics["bundle_sha256"]
    assert first_run != second_run


def test_a_second_run_waits_until_its_own_version_is_listed(world):
    """The service lists the old version for a few polls after a new upload, while status already says ready.
    The run must not proceed on that: it proceeds only once its own marker is listed."""
    run_full()
    fake = world["fake"]
    fake.processing_polls = 3
    start = len(fake.calls)
    second = run_full(sleep_first=True)
    calls = fake.calls[start:]
    version_at = next(i for i, c in enumerate(calls) if c[:2] == ["datasets", "version"])
    listings_after_upload = [c for c in calls[version_at:] if c[:2] == ["datasets", "files"]]
    assert len(listings_after_upload) >= 4, "the wait should read the old version several times before the new one"
    assert job.read_run_state(second)["status"] == "promoted"


def test_a_kernel_error_keeps_the_working_bundle(world):
    run_full()
    working = job.sha256_file(config.MODELS / "bundle.joblib")
    world["fake"].corrupt_mount = True  # Kaggle mounts a file that no longer matches the manifest
    with pytest.raises(KaggleJobError) as err:
        run_full(sleep_first=True)
    assert "kernel ended in ERROR" in str(err.value) and "does not match the manifest" in str(err.value)
    assert job.sha256_file(config.MODELS / "bundle.joblib") == working
    run_dirs = sorted(config.MODELS.glob("kaggle-*"))
    assert job.read_run_state(run_dirs[-1])["status"] == "kernel-error"
    secret_free(config.MODELS)


def test_a_tampered_output_is_refused_and_the_working_bundle_kept(world):
    run_full()
    working = job.sha256_file(config.MODELS / "bundle.joblib")
    world["fake"].tamper_output = True
    with pytest.raises(KaggleJobError, match="verification failed, models/bundle.joblib was NOT changed"):
        run_full(sleep_first=True)
    assert job.sha256_file(config.MODELS / "bundle.joblib") == working
    assert job.read_run_state(sorted(config.MODELS.glob("kaggle-*"))[-1])["status"] == "rejected"


def test_a_failing_upload_stops_before_the_kernel_is_pushed(world, monkeypatch):
    fake = world["fake"]
    original = fake.run

    def refuse_uploads(*args):
        if args[:2] in (("datasets", "create"), ("datasets", "version")):
            return job.Result(1, "403 Forbidden")
        return original(*args)

    monkeypatch.setattr(fake, "run", refuse_uploads)
    with pytest.raises(KaggleJobError, match="dataset upload failed"):
        run_full()
    assert fake.pushes == 0 and not config.MODELS.joinpath("bundle.joblib").exists()


def run_kernel(folder: Path, input_root: Path, working: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(folder / job.KERNEL_FILE)], cwd=folder, env=kernel_env(input_root,
                          working), capture_output=True, text=True, timeout=300)


def mounted_run(tmp_path: Path, world: dict, run_id: str = "run-A") -> tuple[Path, Path]:
    """Upload a run, mount it the way the service does, and write its kernel. Returns (kernel dir, working dir)."""
    upload = tmp_path / "upload"
    job.write_upload(upload, run_id=run_id, username="tester", ipos=world["ipos"], indices=world["indices"],
                     package_dir=world["pkg"])
    kaggle_service_transform(upload, tmp_path / "input" / job.DATASET_SLUG)
    (tmp_path / job.KERNEL_FILE).write_text(job.kernel_source(run_id, SMALL_PROFILE), encoding="utf-8")
    return tmp_path, tmp_path / "working"


def test_kernel_refuses_a_dataset_that_belongs_to_another_run(tmp_path, world):
    folder, working = mounted_run(tmp_path, world, run_id="run-A")
    (folder / job.KERNEL_FILE).write_text(job.kernel_source("run-B", SMALL_PROFILE), encoding="utf-8")
    proc = run_kernel(folder, tmp_path / "input", working)
    assert proc.returncode != 0
    metrics = json.loads((working / "metrics.json").read_text())
    assert metrics["status"] == "error" and "older dataset version" in metrics["error"]
    assert not (working / "bundle.joblib").exists()


def _change_file(mount: Path) -> None:
    tune_file = mount / "ipo_radar_src" / "ipo_radar" / "model" / "tune.py"
    tune_file.write_bytes(tune_file.read_bytes() + b"# edited after upload\n")


def _remove_file(mount: Path) -> None:
    (mount / "ipo_radar_src" / "ipo_radar" / "model" / "tune.py").unlink()


def _add_extra_file(mount: Path) -> None:
    (mount / "ipo_radar_src" / "ipo_radar" / "model" / "rogue.py").write_text("x = 1\n", encoding="utf-8")


@pytest.mark.parametrize("tamper, message", [
    (_change_file, "does not match the manifest"),
    (_remove_file, "is missing from the dataset"),
    (_add_extra_file, "files not in the manifest"),
])
def test_kernel_refuses_a_dataset_that_differs_from_its_manifest(tmp_path, world, tamper, message):
    folder, working = mounted_run(tmp_path, world)
    tamper(tmp_path / "input" / job.DATASET_SLUG)
    proc = run_kernel(folder, tmp_path / "input", working)
    assert proc.returncode != 0
    metrics = json.loads((working / "metrics.json").read_text())
    assert metrics["status"] == "error" and message in metrics["error"], metrics.get("error")
    first_line = (working / "log.txt").read_text().splitlines()[0]
    assert "dataset tree under" in first_line, "the tree must be logged before any check can fail"


def test_kernel_with_no_dataset_mounted_says_so(tmp_path, world):
    (tmp_path / "kernel.py").write_text(job.kernel_source("run-A", SMALL_PROFILE), encoding="utf-8")
    (tmp_path / "input").mkdir()
    working = tmp_path / "working"
    proc = run_kernel(tmp_path, tmp_path / "input", working)
    assert proc.returncode != 0
    assert "expected one manifest.json" in json.loads((working / "metrics.json").read_text())["error"]


def test_kernel_error_hint_is_printed_as_text_only(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "metrics.json").write_text(json.dumps({"error": "x" * 1000}))
    hint = kaggle._error_hint(out)
    assert hint.startswith("Kernel error: ") and len(hint) < 330
