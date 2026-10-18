"""`ipo train --where kaggle`: the heavy training on a free Kaggle CPU kernel, then verify, then promote.

Optional. Local training (`ipo train`) stays the default. The Kaggle kernel runs a large Optuna search over
LightGBM (walk-forward folds, many trials) and a multi-seed final fit. The table is a few thousand rows, so the
value is the breadth of the search, not speed: a local fit already takes seconds.

Flow: build models/kaggle-<run_id>/ (upload/ and kernel/), upload the private dataset, wait until Kaggle has
processed it, push the kernel (refused while a session of it is queued or running), poll with back-off, download
the output to output/, verify it, and only then replace models/bundle.joblib. The previous bundle is kept as
models/bundle.prev.joblib. A failed check leaves models/bundle.joblib untouched and the run folder for inspection.

A dry run builds the same folders and prints what would be sent. It makes no network call and needs no login.
Auth and secret handling: see kaggle_job.py. Errors are KaggleJobError with the token masked.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import ipo_radar
from ipo_radar import config
from ipo_radar.model import kaggle_job as job


def _say(message: str) -> None:
    print(message, flush=True)


def run_train(quick: bool = False, wait: bool = True, poll_s: float = 30, timeout_min: float = 90,
              dry_run: bool = False) -> Path:
    """Train on Kaggle and return the run folder models/kaggle-<run_id>/.

    quick: a few-minute smoke run instead of the full search. wait: block until the kernel ends, then verify and
    promote; with wait=False the kernel keeps running on Kaggle and finish_run(run_dir) collects it later.
    poll_s: first poll interval (back-off up to 5 min). timeout_min: local wait limit; the kernel's own search
    budget is shorter, so a run that finishes in time is always collected. dry_run: build and print, no network.
    Raises KaggleJobError when there is no login, an upload or push fails, the kernel fails, or verification fails.
    """
    if poll_s <= 0 or timeout_min <= 0:
        raise ValueError("poll_s and timeout_min must be positive")
    ipos, indices = config.IPOS_PARQUET, config.INDICES_PARQUET
    for path in (ipos, indices):
        if not path.is_file():
            raise job.KaggleJobError(f"{path} is missing: build the data first (`ipo data build`)")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = config.MODELS / f"kaggle-{run_id}"
    profile = job.run_profile(quick, timeout_min)

    if dry_run:
        username = os.environ.get(job.USER_ENV, "").strip() or job.CONFIG_TEMPLATE_NAME
        _build(run_dir, run_id, username, profile, ipos, indices)
        _print_plan(run_dir, run_id, username, profile)
        return run_dir

    client = job.connect()
    username = job.resolve_username(client)
    _build(run_dir, run_id, username, profile, ipos, indices)
    dataset, kernel = job.dataset_ref(username), job.kernel_ref(username)
    _say(f"uploading private dataset {dataset} (run {run_id})")
    _say(f"dataset: {job.ensure_dataset(client, run_dir / 'upload', f'run {run_id}')}")
    job.wait_dataset_ready(client, dataset, poll_s, timeout_min, _say)
    _say(f"pushing kernel {kernel}")
    _say(job.push_kernel(client, run_dir / "kernel", kernel))
    _update_state(run_dir, status="pushed")
    if not wait:
        _say(f"kernel is running on Kaggle; collect it later with finish_run('{run_dir}')")
        return run_dir
    return finish_run(run_dir, client=client, poll_s=poll_s, timeout_min=timeout_min)


def finish_run(run_dir: Path, *, client: job.Kaggle | None = None, poll_s: float = 30, timeout_min: float = 90,
               say: Callable[[str], None] = _say) -> Path:
    """Wait for a pushed kernel, download and verify its output, then promote the bundle. Returns run_dir."""
    state = job.read_run_state(run_dir)
    if state.get("status") == "promoted":
        say(f"{run_dir} was already promoted")
        return run_dir
    client = client or job.connect()
    ref = state["kernel_ref"]
    final = job.wait_kernel(client, ref, poll_s, timeout_min, say)
    out = run_dir / "output"
    job.download_output(client, ref, out)
    if final != "COMPLETE":
        _update_state(run_dir, status=f"kernel-{final.lower()}")
        raise job.KaggleJobError(f"kernel ended in {final}; its log is {out / 'log.txt'}. {_error_hint(out)}")

    verdict = job.verify_outputs(out, run_id=state["run_id"], ipos_path=config.IPOS_PARQUET,
                                 uploaded_ipos_sha256=state["uploaded_ipos_sha256"])
    for check in verdict.checks:
        say(f"check: {check}")
    if not verdict.ok:
        _update_state(run_dir, status="rejected")
        raise job.KaggleJobError("verification failed, models/bundle.joblib was NOT changed: "
                                 + "; ".join(verdict.errors))
    sha = json.loads((out / "metrics.json").read_text(encoding="utf-8"))["bundle_sha256"]
    job.promote(out / "bundle.joblib", config.MODELS, sha)
    _update_state(run_dir, status="promoted", bundle_sha256=sha)
    say(f"promoted bundle {sha[:12]} to {config.MODELS / 'bundle.joblib'}; the previous one is bundle.prev.joblib")
    return run_dir


def _build(run_dir: Path, run_id: str, username: str, profile: dict, ipos: Path, indices: Path) -> None:
    """Write the dataset folder and the kernel folder under a new run folder. Nothing here touches the network."""
    if run_dir.exists():
        raise job.KaggleJobError(f"{run_dir} already exists; wait a second and run again")
    run_dir.mkdir(parents=True)
    package_dir = Path(ipo_radar.__file__).resolve().parent
    manifest = job.write_upload(run_dir / "upload", run_id=run_id, username=username, ipos=ipos, indices=indices,
                                package_dir=package_dir)
    job.write_kernel(run_dir / "kernel", run_id=run_id, username=username, profile=profile)
    job.write_run_state(run_dir, {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_ref": job.dataset_ref(username),
        "kernel_ref": job.kernel_ref(username),
        "profile": profile,
        "uploaded_ipos_sha256": manifest["files"]["ipos.parquet"]["sha256"],
        "status": "built",
    })


def _update_state(run_dir: Path, **changes) -> None:
    state = job.read_run_state(run_dir)
    state.update(changes)
    job.write_run_state(run_dir, state)


def _error_hint(out: Path) -> str:
    """The kernel's own error line, if it wrote one. Printed as text only, never run."""
    try:
        error = json.loads((out / "metrics.json").read_text(encoding="utf-8")).get("error")
    except (OSError, json.JSONDecodeError, AttributeError):
        return "No metrics.json was written; the kernel stopped before the search finished."
    return f"Kernel error: {str(error)[:300]}" if error else "The kernel did not record an error."


def _print_plan(run_dir: Path, run_id: str, username: str, profile: dict) -> None:
    try:
        auth = job.resolve_auth().source
    except job.KaggleJobError:
        auth = "NONE (a real run stops here: run `kaggle auth login` or set KAGGLE_API_TOKEN)"
    budget_min = profile["search_budget_s"] / 60
    _say("dry run: nothing was sent, no network call was made")
    _say(f"  run folder:  {run_dir}")
    _say(f"  auth:        {auth}")
    hint = "" if username != job.CONFIG_TEMPLATE_NAME else f"  (set {job.USER_ENV} for a real run)"
    _say(f"  username:    {username}{hint}")
    _say(f"  dataset:     {job.dataset_ref(username)}  private, licence {job.DATASET_LICENCE}")
    manifest = json.loads((run_dir / "upload" / job.MANIFEST).read_text(encoding="utf-8"))
    _say(f"  dataset files ({len(manifest['files'])}, uploaded with -r zip so the package folder survives):")
    for name, meta in sorted(manifest["files"].items()):
        _say(f"               {name}  {meta['bytes']:>9} B  {meta['sha256'][:12]}")
    _say(f"  kernel:      {job.kernel_ref(username)}  script, CPU (default machine), internet off, GPU off")
    _say(f"  search:      {profile['n_trials']} trials, budget {budget_min:.0f} min, "
         f"{profile['final_seeds']} final seed(s), {'quick' if profile['quick'] else 'full'} profile")
    _say(f"  outputs:     {', '.join(job.OUTPUT_FILES)} (checked, then promoted to models/bundle.joblib)")
    _say(f"  run id:      {run_id}")
