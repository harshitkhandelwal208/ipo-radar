"""Shared fixtures for the Kaggle training tests: a schema-complete synthetic IPO table, a stand-in for the Model
agent's train(), and a fake kaggle CLI.

The train stand-in has train()'s arguments plus params (which the Kaggle kernel passes) and returns a real Bundle, so
verification exercises the real Bundle class and features.design. It is written into a copy of the package that the
test ships; the real features and tune modules run unchanged. The fake CLI treats a local folder as Kaggle: uploads
land in input/, a pushed kernel runs as a real subprocess with IPO_KAGGLE_INPUT and IPO_KAGGLE_WORKING set, and the
output is copied back on download.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import ipo_radar
from ipo_radar.model import kaggle_job as job
from ipo_radar.model.kaggle_job import Result

FEATURES_STUB = '''"""Test stand-in for ipo_radar.model.features."""
import pandas as pd

FEATURES = ["sub_total", "gmp_pct_last", "issue_size_m", "fresh_issue_pct"]


def asof_expand(df):
    return df.assign(as_of="pre_open")


def make_xy(df, target):
    X = df[[c for c in FEATURES if c in df.columns]].astype(float)
    y = df[target].astype(float)
    return X, y
'''

TRAIN_STUB = '''"""Test stand-in for ipo_radar.model.train. Same arguments as the real train() plus params, which the
Kaggle kernel passes. One LightGBM per stage, wrapped in the real Bundle class so verification checks the real type."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb

from ipo_radar.model import features, tune
from ipo_radar.model.bundle import Bundle, StageModels


@dataclass
class TrainReport:
    version: str
    n_rows: int
    params: dict
    stages: list
    seeds: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def train(df=None, out_dir=Path("models"), seed=7, quick=False, params=None, seeds=None):
    usable = df[df["listing_date"].notna()].reset_index(drop=True)
    vocab = features.fit_vocab(usable)
    expanded = features.asof_expand(usable)
    stages = {}
    for stage in features.STAGES:
        rows = expanded[(expanded["as_of"] == stage) & expanded["list_open_gain_pct"].notna()]
        if len(rows) < 30:
            continue
        X = features.design(rows, stage, vocab)
        z = features.to_model_scale(rows["list_open_gain_pct"].to_numpy(dtype=float))
        model = lgb.LGBMRegressor(objective="quantile", alpha=0.5, random_state=seed, **tune.lgbm_kwargs(params or {}))
        model.fit(X, z)
        stages[stage] = StageModels(stage=stage, features=list(X.columns), q10=model, q50=model, q90=model,
                                    close_q50=None, clf=None, iso=None, n_open=len(rows), n_close=0,
                                    n_by_market={}, core={})
    if not stages:
        raise ValueError("no stage had enough rows")
    now = datetime.now(timezone.utc)
    bundle = Bundle(version="stub", trained_at=now.isoformat(timespec="seconds"), n_rows=len(usable),
                    stages=stages, vocab=vocab, seed=seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / "bundle.joblib")
    return TrainReport(version="stub", n_rows=len(usable), params=dict(params or {}), stages=sorted(stages),
                       seeds=list(seeds or [seed]))
'''



def install_stubs_in_process(monkeypatch) -> None:
    """Put the train stand-in into sys.modules, for the local side of the tests (verification imports train).
    The real features and bundle modules stay in use."""
    name = "ipo_radar.model.train"
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)  # registered first: dataclasses look the module up
    exec(compile(TRAIN_STUB, name, "exec"), module.__dict__)


def make_ipos(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic IPO table with every schema column (missing = NaN), as features.design expects. Gains depend on
    GMP and subscription, with some missing values and noise. Markets cycle so every stage sees several."""
    from ipo_radar.schema import ALL_COLS

    rng = np.random.default_rng(seed)
    days = rng.integers(0, 5 * 365, n)
    gmp = rng.normal(20, 15, n)
    gmp[rng.random(n) < 0.2] = np.nan
    sub = rng.lognormal(0, 1, n)
    gain = 0.4 * np.nan_to_num(gmp) + 5 * np.log(sub) + rng.normal(0, 10, n)
    gain[rng.random(n) < 0.05] = np.nan
    listing = pd.Timestamp("2019-01-01") + pd.to_timedelta(days, unit="D")
    markets = np.array(["IN_MAIN", "IN_SME", "US", "HK"])[np.arange(n) % 4]
    df = pd.DataFrame({c: pd.Series([np.nan] * n, dtype=object) for c in ALL_COLS})
    df["ipo_id"] = [f"ipo-{i}" for i in range(n)]
    df["market"] = markets
    df["exchange"] = np.where(markets == "US", "NASDAQ", np.where(markets == "HK", "HKEX", "NSE"))
    df["sector"] = np.array(["tech", "bank", "pharma"])[np.arange(n) % 3]
    df["source"] = np.where(np.arange(n) % 5 == 0, "hf+px_join", "hf")
    df["listing_date"] = listing
    df["open_date"] = listing - pd.Timedelta(days=5)
    df["close_date"] = listing - pd.Timedelta(days=2)
    df["list_open_gain_pct"] = gain
    df["sub_total"] = sub
    df["sub_qib"] = sub * 0.5
    df["sub_retail"] = sub * 1.5
    df["gmp_pct_last"] = gmp
    df["issue_size_m"] = rng.lognormal(4, 1, n)
    df["issue_price"] = rng.uniform(50, 500, n)
    df["price_low"] = df["issue_price"] * 0.95
    df["price_high"] = df["issue_price"] * 1.05
    df["fresh_issue_pct"] = rng.uniform(0, 100, n)
    df["lot_size"] = 100.0
    return df


def write_inputs(folder: Path, n: int = 200, seed: int = 0) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    ipos = folder / "ipos.parquet"
    indices = folder / "indices.parquet"
    make_ipos(n, seed).to_parquet(ipos)
    pd.DataFrame({"date": pd.date_range("2019-01-01", periods=30, freq="D"), "symbol": "NIFTY50",
                  "close": np.linspace(100, 110, 30)}).to_parquet(indices)
    return ipos, indices


def package_copy(root: Path) -> Path:
    """A copy of the real package with the train stand-in written over train.py."""
    pkg = root / "src" / "ipo_radar"
    shutil.copytree(Path(ipo_radar.__file__).resolve().parent, pkg,
                    ignore=shutil.ignore_patterns("__pycache__"))
    (pkg / "model" / "train.py").write_text(TRAIN_STUB, encoding="utf-8")
    return pkg


def kaggle_service_transform(upload: Path, mount: Path) -> None:
    """What Kaggle does to an upload made by `kaggle datasets create -r zip -t` (observed 2026-10-15).

    The CLI zips each top-level folder with the same make_archive call it uses (entries relative to that folder),
    and the service extracts the zip into a folder of the same name. Top-level files are kept as they are. Every
    test that mounts a dataset goes through this, so a layout the service would not produce fails here, locally.
    """
    mount.mkdir(parents=True, exist_ok=True)
    for entry in sorted(upload.iterdir()):
        if entry.is_dir():
            with tempfile.TemporaryDirectory() as tmp:
                archive = shutil.make_archive(str(Path(tmp) / entry.name), "zip", str(entry))
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(mount / entry.name)
        else:
            shutil.copyfile(entry, mount / entry.name)


def kernel_env(input_root: Path, working: Path) -> dict[str, str]:
    """The kernel's environment on Kaggle: no token, only the two directories it needs."""
    env = {k: v for k, v in os.environ.items() if k != job.TOKEN_ENV}
    env.update(IPO_KAGGLE_INPUT=str(input_root), IPO_KAGGLE_WORKING=str(working))
    return env


class FakeKaggle:
    """Stands in for the kaggle CLI. Asserts the things Kaggle would reject, then acts on a local folder."""

    def __init__(self, root: Path, *, username: str = "tester", queue_polls: int = 1,
                 corrupt_mount: bool = False, tamper_output: bool = False, kernel_timeout_s: float = 600,
                 processing_polls: int = 1):
        self.root = root
        self.input_root = root / "input"
        self.working = root / "working"
        self.username = username
        self.queue_polls = queue_polls
        self.corrupt_mount = corrupt_mount
        self.tamper_output = tamper_output
        self.kernel_timeout_s = kernel_timeout_s
        self.calls: list[list[str]] = []
        self.kernel_exit: int | None = None
        self.kernel_log = ""
        self.pushes = 0
        # Like the real service: a new version is listed only after `processing_polls` listings have been read,
        # and `datasets status` says ready for the old version the whole time.
        self.processing_polls = processing_polls
        self.listed: list[str] = []
        self.pending: list[str] | None = None
        self.pending_polls = 0

    def run(self, *args: str) -> Result:
        args = list(args)
        self.calls.append(args)
        for item in args:
            assert "KGAT" not in item, "the token must never be passed on the command line"
        if args[:2] in (["datasets", "create"], ["datasets", "version"]):
            return self._upload(args)
        if args[:2] == ["datasets", "status"]:
            return Result(0, f'{args[2]} has status "ready"')
        if args[:2] == ["datasets", "files"]:
            return self._files(args)
        if args[:2] == ["kernels", "status"]:
            return self._kernel_status(args[2])
        if args[:2] == ["kernels", "push"]:
            return self._push(Path(args[args.index("-p") + 1]))
        if args[:2] == ["kernels", "output"]:
            return self._output(Path(args[args.index("-p") + 1]))
        if args[:2] == ["config", "view"]:
            return Result(0, f"Configuration values from /fake\n- username: {self.username}\n- auth_method: x")
        return Result(1, f"fake kaggle does not know {' '.join(args)}")

    def _upload(self, args: list[str]) -> Result:
        folder = Path(args[args.index("-p") + 1])
        assert "-t" in args and args[args.index("-r") + 1] == "zip", "uploads must keep tables and zip folders"
        assert not list(folder.rglob("*.zip")), "the package must ship as plain files, not a zip"
        meta = json.loads((folder / "dataset-metadata.json").read_text(encoding="utf-8"))
        assert meta["id"] == f"{self.username}/{job.DATASET_SLUG}"
        assert meta["licenses"] == [{"name": "other"}]
        mount = self.input_root / job.DATASET_SLUG
        if args[1] == "create" and mount.exists():
            # the real CLI's quirk: a notice and exit 0, nothing uploaded
            return Result(0, "Warning: dataset already exists; nothing was uploaded")
        shutil.rmtree(mount, ignore_errors=True)
        kaggle_service_transform(folder, mount)
        self.pending = sorted(p.relative_to(mount).as_posix() for p in mount.rglob("*") if p.is_file())
        self.pending_polls = self.processing_polls
        if self.corrupt_mount:
            with (mount / "ipos.parquet").open("ab") as handle:
                handle.write(b"corrupted in transit")
        return Result(0, "Dataset created" if args[1] == "create" else "Dataset version created")

    def _files(self, args: list[str]) -> Result:
        """`datasets files`: the old version is listed until the new one has processed, then the new one, paged."""
        if not (self.input_root / job.DATASET_SLUG).exists():
            return Result(0, "Dataset not found")
        if self.pending is not None and self.pending_polls > 0:
            self.pending_polls -= 1
        elif self.pending is not None:
            self.listed, self.pending = self.pending, None
        if not self.listed:
            return Result(0, "No files found")
        size = int(args[args.index("--page-size") + 1]) if "--page-size" in args else 20
        start = int(args[args.index("--page-token") + 1]) if "--page-token" in args else 0
        rows = self.listed[start:start + size]
        text = "name  size  creationDate\n" + "\n".join(f"{name}  1  2026-10-15" for name in rows)
        if start + size < len(self.listed):
            text += f"\nNext Page Token = {start + size}"
        return Result(0, text)

    def _push(self, folder: Path) -> Result:
        meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert set(meta) == {"id", "title", "code_file", "language", "kernel_type", "is_private", "enable_gpu",
                             "enable_tpu", "enable_internet", "dataset_sources", "competition_sources",
                             "kernel_sources", "model_sources"}, sorted(meta)
        assert meta["enable_internet"] is False and meta["enable_gpu"] is False
        assert meta["kernel_type"] == "script" and meta["is_private"] is True
        assert meta["dataset_sources"] == [f"{self.username}/{job.DATASET_SLUG}"]
        self.pushes += 1
        shutil.rmtree(self.working, ignore_errors=True)
        proc = subprocess.run([sys.executable, str(folder / meta["code_file"])], cwd=self.root,
                              env=kernel_env(self.input_root, self.working), capture_output=True, text=True,
                              timeout=self.kernel_timeout_s)
        self.kernel_exit = proc.returncode
        self.kernel_log = proc.stdout + proc.stderr
        return Result(0, "Kernel version 1 successfully pushed.")

    def _kernel_status(self, ref: str) -> Result:
        if self.kernel_exit is None:
            return Result(1, "404 Not Found")
        if self.queue_polls > 0:
            self.queue_polls -= 1
            return Result(0, f'{ref} has status "KernelWorkerStatus.RUNNING"')
        word = "COMPLETE" if self.kernel_exit == 0 else "ERROR"
        return Result(0, f'{ref} has status "KernelWorkerStatus.{word}"')

    def _output(self, out: Path) -> Result:
        out.mkdir(parents=True, exist_ok=True)
        if self.working.is_dir():
            for path in self.working.iterdir():
                shutil.copyfile(path, out / path.name)
        if self.tamper_output and (out / "bundle.joblib").is_file():
            with (out / "bundle.joblib").open("ab") as handle:
                handle.write(b"tampered")
        return Result(0, "Output downloaded")
