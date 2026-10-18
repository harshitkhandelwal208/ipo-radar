"""Packaging, kaggle CLI calls and output checks for `ipo train --where kaggle`. Public entry: model/kaggle.py.

What the Kaggle run is for: the table is a few thousand IPOs, so a GBM fits in seconds on this machine. The
value of a free Kaggle CPU session is the large Optuna search (many walk-forward folds, many trials) and
multi-seed ensembling, not speed. Local training stays the default.

Kaggle facts this module relies on (patchwork docs/KAGGLE.md, learned the hard way):
- Kernel metadata keys are snake_case. camelCase keys are silently ignored.
- Internet is silently off on accounts that are not phone-verified, so the kernel installs nothing and uses only
  the libraries preinstalled on Kaggle's images.
- There is no cancel command. `kaggle kernels delete -y <ref>` stops a kernel.
- Datasets mount under /kaggle/input/<dataset-slug>. A fresh dataset version can lag behind the upload, so the
  kernel refuses any dataset whose manifest is not for its own run id.

Secrets: the API token is read only from the KAGGLE_API_TOKEN environment variable (or an existing ~/.kaggle
login, whose files are checked for existence and never read). It is passed to the kaggle CLI in the child's
environment, never in arguments, and every captured output is masked before it is used or shown.

Username (owner of the dataset and kernel): the KAGGLE_USERNAME environment variable, or the `- username:` line
that `kaggle config view` prints once you are logged in. It is also the name in your kaggle.com profile URL.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

TOKEN_ENV = "KAGGLE_API_TOKEN"
USER_ENV = "KAGGLE_USERNAME"
NO_AUTH = ("No Kaggle login found. Run `kaggle auth login` once, or set the environment variable "
           f"{TOKEN_ENV} to an API token from https://www.kaggle.com/settings. Nothing was uploaded.")

DATASET_SLUG = "ipo-radar-train-inputs"
KERNEL_SLUG = "ipo-radar-train"
DATASET_LICENCE = "other"            # contains third-party data, so not CC0; the dataset stays private
KERNEL_FILE = "kernel.py"
SOURCE_DIR = "ipo_radar_src"          # the package ships as plain files here (no zip: Kaggle extracts zips)
MANIFEST = "manifest.json"
RUN_STATE = "run.json"
INPUT_FILES = ("ipos.parquet", "indices.parquet")
OUTPUT_FILES = ("bundle.joblib", "tune_trials.csv", "metrics.json", "log.txt")
SEARCH_SEED = 2026
SEARCH_RESERVE_S = 900               # kept back from the local wait for the final fit and the output
POLL_MAX_S = 300
SAMPLE_ROWS = 50                     # rows the local sanity check predicts on
KERNEL_TERMINAL = ("COMPLETE", "ERROR", "CANCEL")
USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,50}$")
CONFIG_USER_RE = re.compile(r"^\s*-\s*username:\s*(\S+)\s*$", re.MULTILINE)
CONFIG_TEMPLATE_NAME = "<your-kaggle-username>"


class KaggleJobError(RuntimeError):
    """A Kaggle step failed or was refused. The message is safe to show the user (token already masked)."""


# ---- run profile ----------------------------------------------------------------------------------
def run_profile(quick: bool, timeout_min: float) -> dict[str, Any]:
    """What the kernel does. Full: a long search, then the final fit. Quick: a smoke run of a few minutes."""
    if quick:
        return {"quick": True, "n_trials": 20, "final_seeds": 1, "search_budget_s": 180, "seed": SEARCH_SEED}
    budget = max(600.0, timeout_min * 60 - SEARCH_RESERVE_S)
    return {"quick": False, "n_trials": 300, "final_seeds": 5, "search_budget_s": budget, "seed": SEARCH_SEED}


# ---- auth ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Auth:
    token: str | None      # set only when KAGGLE_API_TOKEN is the source
    source: str            # human text for logs and the dry run; never contains the token


def resolve_auth(environ: Mapping[str, str] | None = None, home: Path | None = None) -> Auth:
    """KAGGLE_API_TOKEN first, then an existing kaggle CLI login. Raises KaggleJobError when neither exists."""
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else home
    token = environ.get(TOKEN_ENV, "").strip()
    if token:
        return Auth(token, f"{TOKEN_ENV} (environment variable, value hidden)")
    for name in ("access_token", "kaggle.json"):
        if (home / ".kaggle" / name).is_file():
            return Auth(None, f"kaggle CLI login (~/.kaggle/{name})")
    raise KaggleJobError(NO_AUTH)


def mask_secret(text: str, secret: str | None) -> str:
    """Replace every occurrence of the secret. Used on all captured CLI output before it is shown or stored."""
    return text.replace(secret, "***") if secret else text


def resolve_username(client: "Kaggle", environ: Mapping[str, str] | None = None) -> str:
    """Owner name for the dataset and kernel refs. From KAGGLE_USERNAME, else `kaggle config view`."""
    environ = os.environ if environ is None else environ
    name = environ.get(USER_ENV, "").strip()
    if not name:
        res = client.run("config", "view")
        match = CONFIG_USER_RE.search(res.text)
        if res.code != 0 or not match:
            raise KaggleJobError(f"cannot read your Kaggle username from `kaggle config view`; set {USER_ENV}")
        name = match.group(1)
    if not USERNAME_RE.match(name):
        raise KaggleJobError(f"the Kaggle username {name!r} has characters a Kaggle slug cannot use")
    return name


# ---- kaggle CLI ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Result:
    code: int
    text: str              # masked, stripped

    @property
    def last_line(self) -> str:
        lines = self.text.splitlines()
        return lines[-1][:300] if lines else f"exit code {self.code}"


class Kaggle:
    """The kaggle CLI with the token only in the child's environment and all output masked."""

    def __init__(self, binary: str, token: str | None,
                 runner: Callable[..., subprocess.CompletedProcess] | None = None, timeout_s: float = 900):
        self.binary = binary
        self._token = token or ""
        self._runner = runner or subprocess.run
        self._timeout = timeout_s
        self._env = dict(os.environ)
        if token:
            self._env[TOKEN_ENV] = token

    def run(self, *args: str) -> Result:
        try:
            proc = self._runner([self.binary, *args], env=self._env, capture_output=True, text=True,
                                timeout=self._timeout)
        except subprocess.TimeoutExpired:
            raise KaggleJobError(f"kaggle {' '.join(args[:2])} did not answer within {self._timeout:.0f} s") from None
        except FileNotFoundError:
            raise KaggleJobError("the kaggle CLI is not installed; install it with `pip install kaggle`") from None
        text = mask_secret((proc.stdout or "") + (proc.stderr or ""), self._token).strip()
        return Result(proc.returncode, text)


def connect() -> Kaggle:
    """A client for the real CLI. Raises KaggleJobError when there is no login or no kaggle binary."""
    auth = resolve_auth()
    binary = shutil.which("kaggle")
    if binary is None:
        raise KaggleJobError("the kaggle CLI is not on PATH; install it with `pip install kaggle`")
    return Kaggle(binary, auth.token)


def kernel_state(client: "Kaggle", ref: str) -> str:
    """RUNNING, QUEUED, COMPLETE, ERROR, CANCEL or UNKNOWN. Only the first line is read: the failure text
    that follows is free text and must not be mistaken for a state."""
    res = client.run("kernels", "status", ref)
    if res.code != 0:
        return "UNKNOWN"
    first = res.text.splitlines()[0].upper() if res.text else ""
    for name in ("RUNNING", "QUEUED", "COMPLETE", "ERROR", "CANCEL"):
        if name in first:
            return name
    return "UNKNOWN"


# ---- packaging -----------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_package(upload_dir: Path, package_dir: Path) -> list[str]:
    """Copy the package's .py files as plain files to upload_dir/ipo_radar_src/ipo_radar/.

    Plain files, not a zip: a zip uploaded with `datasets create -r zip` is extracted by Kaggle into a folder, so
    a zip in the manifest would never match the mounted dataset (seen on the first real run, 2026-10-15). Returns
    the relative posix paths copied, which become the manifest keys.
    """
    if package_dir.name != "ipo_radar":
        raise ValueError(f"package_dir must be the ipo_radar folder, got {package_dir}")
    copied = []
    for path in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(package_dir)
        target = upload_dir / SOURCE_DIR / "ipo_radar" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        copied.append(f"{SOURCE_DIR}/ipo_radar/{rel.as_posix()}")
    return copied


def dataset_ref(username: str) -> str:
    return f"{username}/{DATASET_SLUG}"


def kernel_ref(username: str) -> str:
    return f"{username}/{KERNEL_SLUG}"


def dataset_metadata(username: str) -> dict[str, Any]:
    return {
        "id": dataset_ref(username),
        "title": "IPO radar train inputs",
        "subtitle": "Private training inputs for ipo-radar (one kernel run)",
        "description": ("IPO table, index closes, the ipo_radar package as plain files and a manifest with checksums. "
                        "Third-party data: see the sources in the project. Private."),
        "licenses": [{"name": DATASET_LICENCE}],
    }


def kernel_metadata(username: str) -> dict[str, Any]:
    """Exact keys the kaggle CLI reads for a script kernel. No machine_shape: the default machine is CPU."""
    return {
        "id": kernel_ref(username),
        "title": KERNEL_SLUG,
        "code_file": KERNEL_FILE,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [dataset_ref(username)],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def write_upload(upload_dir: Path, *, run_id: str, username: str, ipos: Path, indices: Path,
                 package_dir: Path) -> dict[str, Any]:
    """Dataset folder: the inputs, the package as plain files, manifest.json (sha256 and size of EVERY file that
    ships, keyed by relative path, plus the run id) and dataset-metadata.json."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ipos, upload_dir / "ipos.parquet")
    shutil.copyfile(indices, upload_dir / "indices.parquet")
    marker = marker_name(run_id)
    (upload_dir / marker).write_text(f"run {run_id}\n", encoding="utf-8")
    names = [*INPUT_FILES, marker, *copy_package(upload_dir, package_dir)]
    files = {}
    for name in names:
        path = upload_dir.joinpath(*name.split("/"))
        files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest = {"run_id": run_id, "source_dir": SOURCE_DIR, "files": files}
    (upload_dir / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (upload_dir / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata(username), indent=2) + "\n", encoding="utf-8")
    return manifest


# The kernel is generated per run: RUN_ID and PROFILE are literals, so the kernel knows which dataset it expects.
KERNEL_TEMPLATE = '''"""ipo-radar training kernel: Optuna search, then the final bundle. CPU only, internet off.

Generated by ipo_radar.model.kaggle_job for one run. The first log lines list the dataset tree as Kaggle mounted it.
Then every file in manifest.json is checked (present, size, sha256), and no file may sit in the package folder that
the manifest does not list. Only then does it import tune and train from that folder. Writes bundle.joblib,
train_report.json, tune_trials.csv, metrics.json and log.txt to WORKING. Every file it reads is data.
"""
import hashlib
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from importlib import metadata
from pathlib import Path

import pandas as pd

RUN_ID = __RUN_ID__
PROFILE = __PROFILE__
INPUT_ROOT = Path(os.environ.get("IPO_KAGGLE_INPUT", "/kaggle/input"))
WORKING = Path(os.environ.get("IPO_KAGGLE_WORKING", "/kaggle/working"))
MANIFEST = "manifest.json"
PACKAGES = ("lightgbm", "optuna", "scikit-learn", "pandas", "numpy", "joblib")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_dataset(root):
    hits = sorted(root.rglob(MANIFEST)) if root.is_dir() else []
    if len(hits) != 1:
        raise RuntimeError(f"expected one {MANIFEST} under {root}, found {len(hits)}")
    return hits[0].parent


def dataset_tree(root):
    if not root.is_dir():
        return []
    return [(p.relative_to(root).as_posix(), p.stat().st_size) for p in sorted(root.rglob("*")) if p.is_file()]


def log_tree(log):
    """First thing logged: what Kaggle actually mounted. A service that changes the layout shows up here."""
    rows = dataset_tree(INPUT_ROOT)
    log.info("dataset tree under %s: %d files", INPUT_ROOT, len(rows))
    for name, size in rows[:500]:
        log.info("  %10d  %s", size, name)


def check_inputs(base):
    manifest = json.loads((base / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("run_id") != RUN_ID:
        raise RuntimeError(f"dataset is run {manifest.get('run_id')} but this kernel is run {RUN_ID}: "
                           "Kaggle mounted an older dataset version; push the kernel again")
    expected = manifest["files"]
    for name, meta in expected.items():
        path = base.joinpath(*name.split("/"))
        if not path.is_file():
            raise RuntimeError(f"{name} is missing from the dataset")
        if path.stat().st_size != meta["bytes"] or sha256_file(path) != meta["sha256"]:
            raise RuntimeError(f"{name} does not match the manifest (corrupted or changed after upload)")
    source = base / manifest["source_dir"]
    present = {p.relative_to(base).as_posix() for p in source.rglob("*") if p.is_file()}
    extra = sorted(present - set(expected))
    if extra:
        raise RuntimeError(f"files not in the manifest inside {manifest['source_dir']}: {extra[:5]}")
    return manifest


def setup_logging(path):
    root = logging.getLogger()  # the root logger, so messages from ipo_radar.model.tune reach log.txt too
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    return logging.getLogger("ipo_radar.kernel")


def package_versions():
    out = {"python": sys.version.split()[0]}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def write_metrics(payload):
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\\n"
    (WORKING / "metrics.json").write_text(text, encoding="utf-8")


def main():
    WORKING.mkdir(parents=True, exist_ok=True)
    log = setup_logging(WORKING / "log.txt")
    started = time.time()
    payload = {"run_id": RUN_ID, "profile": PROFILE, "versions": package_versions()}
    try:
        log_tree(log)
        base = find_dataset(INPUT_ROOT)
        manifest = check_inputs(base)
        sys.dont_write_bytecode = True  # the input mount is read-only and must not gain __pycache__ folders
        sys.path.insert(0, str(base / manifest["source_dir"]))
        from ipo_radar.model import train, tune

        df = pd.read_parquet(base / "ipos.parquet")
        indices = pd.read_parquet(base / "indices.parquet")
        log.info("run %s: %d IPO rows, %d index rows", RUN_ID, len(df), len(indices))
        best, trials = tune.search(df, n_trials=PROFILE["n_trials"], seed=PROFILE["seed"],
                                   time_budget_s=PROFILE["search_budget_s"])
        trials.to_csv(WORKING / "tune_trials.csv", index=False)
        log.info("search finished: %d trial rows, best params %s", len(trials), best)
        seeds = [PROFILE["seed"] + i for i in range(PROFILE["final_seeds"])]
        report = train.train(df, out_dir=WORKING, seed=PROFILE["seed"], quick=PROFILE["quick"], params=best,
                             seeds=seeds)
        bundle = WORKING / "bundle.joblib"
        if not bundle.is_file():
            raise RuntimeError("train() did not write bundle.joblib")
        folds = tune.fold_counts(df, n_folds=tune.N_FOLDS)
        folds.to_csv(WORKING / "fold_counts.csv", index=False)
        done = trials[trials["state"] == "COMPLETE"]
        payload.update(
            status="ok",
            best_params=best,
            best_score=float(done["score"].min()),
            completed_trials=int(len(done)),
            trial_rows=int(len(trials)),
            train_report=asdict(report) if is_dataclass(report) else {"repr": repr(report)},
            folds=folds.to_dict("records"),
            bundle_sha256=sha256_file(bundle),
            bundle_bytes=bundle.stat().st_size,
            rows=int(len(df)),
            elapsed_s=round(time.time() - started, 1),
        )
        write_metrics(payload)
        log.info("done: bundle %s", payload["bundle_sha256"][:12])
    except Exception as exc:
        log.error("kernel failed: %s", exc)
        log.error(traceback.format_exc())
        payload.update(status="error", error=f"{type(exc).__name__}: {exc}",
                       elapsed_s=round(time.time() - started, 1))
        write_metrics(payload)
        raise


if __name__ == "__main__":
    main()
'''


def kernel_source(run_id: str, profile: dict[str, Any]) -> str:
    """The kernel file for one run. repr() gives Python literals for the id and the profile."""
    if "__RUN_ID__" not in KERNEL_TEMPLATE or "__PROFILE__" not in KERNEL_TEMPLATE:
        raise RuntimeError("kernel template lost its placeholders")
    return KERNEL_TEMPLATE.replace("__RUN_ID__", repr(run_id)).replace("__PROFILE__", repr(profile))


def write_kernel(kernel_dir: Path, *, run_id: str, username: str, profile: dict[str, Any]) -> Path:
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / KERNEL_FILE).write_text(kernel_source(run_id, profile), encoding="utf-8")
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(kernel_metadata(username), indent=2) + "\n", encoding="utf-8")
    return kernel_dir


# ---- kaggle operations ---------------------------------------------------------------------------
# Words that mean the CLI did not do what was asked, even with exit code 0. `datasets create` on an existing
# dataset prints a notice and can exit 0 (seen 2026-10-15); a failed upload prints "Upload unsuccessful".
FAILURE_RE = re.compile(r"already exists|\berror\b|\bwarning\b|\bfail|unsuccessful|forbidden|denied|unauthorized",
                        re.IGNORECASE)
TABLE_HEADER_RE = re.compile(r"\bname\b.*\bsize\b", re.IGNORECASE)   # a successful listing prints this header
MISSING_RE = re.compile(r"not found|404|no files found", re.IGNORECASE)
LIST_PAGE = 200
MAX_PAGES = 50
NEXT_PAGE_RE = re.compile(r"Next Page Token\s*=\s*(\S+)")


def marker_name(run_id: str) -> str:
    """Top-level file that names the run. Its appearance in the dataset listing shows that this version is the one
    Kaggle serves now. A `ready` status alone is also true of the previous version while the new one processes."""
    return f"run-{run_id}.txt"


def cli_problem(res: Result) -> str | None:
    """Why a CLI call did not do its job, or None. Exit code 0 is not enough: the text must be clean too."""
    if res.code != 0:
        return res.last_line
    for line in res.text.splitlines():
        if FAILURE_RE.search(line):
            return line.strip()[:300]
    return None


def list_dataset_files(client: Kaggle, ref: str) -> str:
    """Listing text of every page. `datasets files` pages at 20 items, so each page token is followed to the end.

    A missing dataset gives an empty listing. Anything that is not a table and not a missing-dataset notice is an
    error: the CLI prints errors as plain text, sometimes with exit code 0.
    """
    pages, token = [], None
    for _ in range(MAX_PAGES):
        args = ["datasets", "files", ref, "--page-size", str(LIST_PAGE)]
        if token:
            args += ["--page-token", token]
        res = client.run(*args)
        if res.code == 0 and TABLE_HEADER_RE.search(res.text):
            pages.append(res.text)
            match = NEXT_PAGE_RE.search(res.text)
            if not match:
                return "\n".join(pages)
            token = match.group(1)
            continue
        if MISSING_RE.search(res.text or res.last_line):
            return "\n".join(pages)
        raise KaggleJobError(f"could not list dataset {ref}: {res.last_line}")
    raise KaggleJobError(f"dataset {ref} has more than {MAX_PAGES * LIST_PAGE} files; refusing to guess")


def dataset_exists(client: Kaggle, ref: str) -> bool:
    """True when the listing shows our manifest.json. Every dataset this module makes contains it."""
    return "manifest.json" in list_dataset_files(client, ref)


def ensure_dataset(client: Kaggle, upload_dir: Path, ref: str, message: str) -> str:
    """Exactly one of create or version, chosen by asking Kaggle whether the dataset exists first.

    Failure text in the CLI output counts as failure even when the exit code is 0. -r zip: the CLI skips subfolders
    with -r skip, so the package folder would not upload at all; zip uploads each top-level folder as one archive,
    which Kaggle extracts back into a folder. -t keeps the parquet files as they are, so the checksums in
    manifest.json still match what Kaggle stores.
    """
    if dataset_exists(client, ref):
        action = "new version"
        res = client.run("datasets", "version", "-p", str(upload_dir), "-m", message, "-r", "zip", "-t")
    else:
        action = "created"
        res = client.run("datasets", "create", "-p", str(upload_dir), "-r", "zip", "-t")
    problem = cli_problem(res)
    if problem:
        raise KaggleJobError(f"dataset upload failed ({action}): {problem}")
    return action


def wait_dataset_ready(client: Kaggle, ref: str, run_id: str, poll_s: float, timeout_min: float,
                       say: Callable[[str], None]) -> None:
    """Wait until this run's marker is listed AND the status says ready. Both are needed: `ready` is also true of
    the previous version while the new one is still processing, and the marker alone does not show the status."""
    deadline = time.monotonic() + timeout_min * 60
    delay = poll_s
    marker = marker_name(run_id)
    while True:
        status = client.run("datasets", "status", ref)
        status_line = status.last_line.lower()
        if status.code == 0 and "error" in status_line:
            raise KaggleJobError(f"dataset {ref} failed to process: {status.last_line}")
        ready = status.code == 0 and "ready" in status_line
        listed = marker in list_dataset_files(client, ref)
        if ready and listed:
            return
        if time.monotonic() >= deadline:
            raise KaggleJobError(
                f"dataset {ref} does not show {marker} with status ready after {timeout_min:g} min "
                f"(listed: {listed}, status: {status.last_line!r}). Kaggle is still processing this upload. "
                "Wait a few minutes, then run again.")
        say(f"dataset {ref}: waiting for {marker} (listed: {listed}, status: {status.last_line[:40]})")
        time.sleep(min(delay, POLL_MAX_S))
        delay = min(delay * 1.5, POLL_MAX_S)


def push_kernel(client: Kaggle, kernel_dir: Path, ref: str) -> str:
    """Refuse when a session for this kernel is still queued or running: a push starts a second session and
    nothing stops the first one."""
    state = kernel_state(client, ref)
    if state in ("RUNNING", "QUEUED"):
        raise KaggleJobError(f"{ref} is already {state} on Kaggle. Wait for it, or stop it with "
                             f"`kaggle kernels delete -y {ref}`, then run again")
    res = client.run("kernels", "push", "-p", str(kernel_dir))
    if res.code != 0:
        raise KaggleJobError(f"kernel push failed: {res.last_line}")
    return res.last_line


def wait_kernel(client: Kaggle, ref: str, poll_s: float, timeout_min: float,
                say: Callable[[str], None]) -> str:
    """Poll with back-off until a terminal state. Past the deadline the kernel is left running, on purpose."""
    deadline = time.monotonic() + timeout_min * 60
    delay = poll_s
    last = None
    while True:
        state = kernel_state(client, ref)
        if state != last:
            say(f"kernel {ref}: {state}")
            last = state
        if state in KERNEL_TERMINAL:
            return state
        if time.monotonic() >= deadline:
            raise KaggleJobError(f"kernel {ref} is still {state} after {timeout_min:g} min. It keeps running on "
                                 "Kaggle; collect it later with finish_run() on this run folder")
        time.sleep(min(delay, POLL_MAX_S))
        delay = min(delay * 1.5, POLL_MAX_S)


def download_output(client: Kaggle, ref: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = client.run("kernels", "output", ref, "-p", str(out_dir))
    if res.code != 0:
        raise KaggleJobError(f"could not download the kernel output: {res.last_line}")


# ---- verification and promotion -------------------------------------------------------------------
@dataclass
class Verification:
    ok: bool
    checks: list[str]
    errors: list[str]


def verify_outputs(out_dir: Path, *, run_id: str, ipos_path: Path, uploaded_ipos_sha256: str,
                   sample_rows: int = SAMPLE_ROWS) -> Verification:
    """Every check that must pass before a downloaded bundle may replace the working one.

    Checks: all output files exist; metrics.json says ok for this run; the bundle's checksum matches the one the
    kernel recorded; the local ipos.parquet is the file that was uploaded; the bundle loads with this venv's
    joblib as a model Bundle, and predicts finite numbers on real rows (bundle_predictions).
    """
    checks: list[str] = []
    errors: list[str] = []
    missing = [name for name in OUTPUT_FILES if not (out_dir / name).is_file()]
    if missing:
        return Verification(False, checks, [f"output files missing: {', '.join(missing)}"])
    try:
        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return Verification(False, checks, [f"metrics.json unreadable: {exc}"])
    if metrics.get("status") != "ok":
        errors.append(f"kernel reported status {metrics.get('status')!r}: {str(metrics.get('error', ''))[:300]}")
    if metrics.get("run_id") != run_id:
        errors.append(f"metrics.json is for run {metrics.get('run_id')!r}, expected {run_id!r}")
    bundle = out_dir / "bundle.joblib"
    digest = sha256_file(bundle)
    if digest != metrics.get("bundle_sha256"):
        errors.append("bundle.joblib checksum does not match metrics.json (corrupted or replaced)")
    else:
        checks.append(f"bundle sha256 {digest[:12]} matches metrics.json")
    if sha256_file(ipos_path) != uploaded_ipos_sha256:
        errors.append("local ipos.parquet changed since the upload; run the kaggle training again")
    else:
        checks.append("local ipos.parquet is the uploaded file")
    if errors:
        return Verification(False, checks, errors)
    try:
        trials = pd.read_csv(out_dir / "tune_trials.csv")
        if trials.empty or "score" not in trials.columns:
            errors.append("tune_trials.csv has no trials")
    except (OSError, ValueError) as exc:
        errors.append(f"tune_trials.csv unreadable: {exc}")
    try:
        import joblib

        loaded = joblib.load(bundle)
        sample = pd.read_parquet(ipos_path).head(sample_rows)
        pred = bundle_predictions(loaded, sample)
        if not pred:
            errors.append("the bundle produced no predictions for the sample rows")
        elif not all(math.isfinite(x) for x in pred):
            errors.append("the bundle produced non-finite predictions")
        else:
            checks.append(f"bundle loads and predicts {len(pred)} finite values on {len(sample)} real rows")
    except Exception as exc:  # the verifier's job is to refuse, with the reason; anything here means refuse
        errors.append(f"bundle check failed: {type(exc).__name__}: {str(exc)[:300]}")
    return Verification(not errors, checks, errors)


def bundle_predictions(loaded: Any, sample: pd.DataFrame) -> list[float]:
    """Open-gain median predictions of a loaded bundle, one per (row, stage) the bundle has a model for.

    Uses the Model agent's own pieces: the Bundle class, asof_expand, design and each stage's q50 model. Raises
    TypeError when the file is not a Bundle, so a wrong file can never pass as a model.
    """
    bundle_mod = importlib.import_module("ipo_radar.model.bundle")
    features = importlib.import_module("ipo_radar.model.features")
    if not isinstance(loaded, bundle_mod.Bundle):
        raise TypeError(f"the file holds a {type(loaded).__name__}, not a model Bundle")
    expanded = features.asof_expand(sample)
    values: list[float] = []
    for stage, models in loaded.stages.items():
        rows = expanded[expanded["as_of"] == stage]
        if rows.empty:
            continue
        values.extend(float(v) for v in models.q50.predict(features.design(rows, stage, loaded.vocab)))
    return values


def promote(bundle: Path, models_dir: Path, expected_sha256: str) -> None:
    """Make the verified bundle the working models/bundle.joblib. The old one becomes bundle.prev.joblib.

    Both swaps go through a staged copy and os.replace, so a crash never leaves a half-written bundle.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    current = models_dir / "bundle.joblib"
    previous = models_dir / "bundle.prev.joblib"
    staged = models_dir / ".bundle.staged.joblib"
    shutil.copyfile(bundle, staged)
    if sha256_file(staged) != expected_sha256:
        staged.unlink()
        raise KaggleJobError("copy of the bundle does not match its checksum; models/bundle.joblib was not changed")
    if current.is_file():
        backup = models_dir / ".bundle.prev.staged.joblib"
        shutil.copyfile(current, backup)
        os.replace(backup, previous)
    os.replace(staged, current)


def write_run_state(run_dir: Path, state: dict[str, Any]) -> None:
    (run_dir / RUN_STATE).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_run_state(run_dir: Path) -> dict[str, Any]:
    path = run_dir / RUN_STATE
    if not path.is_file():
        raise KaggleJobError(f"{run_dir} has no {RUN_STATE}; it is not a Kaggle run folder")
    return json.loads(path.read_text(encoding="utf-8"))
