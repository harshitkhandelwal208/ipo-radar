"""The research report: a planted relation is recovered, cells under 20 rows are suppressed, NaN and zero codes are
handled, missing columns and an empty table do not crash, reruns are identical, and `ipo research` is wired through.
Synthetic frames only; the report default path is redirected so no test writes into docs/."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from ipo_radar import config
from ipo_radar import research as R
from ipo_radar.cli.app import app
from ipo_radar.schema import ALL_COLS

OPEN, CLOSE = R.OPEN, R.CLOSE


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch):
    monkeypatch.setattr(R, "BOOT", 200)


def make_frame(n: int = 600, seed: int = 3) -> pd.DataFrame:
    """Four markets. In every market the close gain is 8 * ln(total subscription) + noise (sd 5), so the planted
    rank correlation is about 0.85. Indian rows carry subscription and index context; IN_MAIN and IN_SME carry GMP."""
    rng = np.random.default_rng(seed)
    blocks = []
    for market in ("IN_MAIN", "IN_SME", "US", "HK"):
        india = market.startswith("IN_")
        total = np.exp(rng.normal(1.5, 1.0, n))
        close = 8.0 * np.log(total) + rng.normal(0, 5, n)
        opened = close + rng.normal(0, 5, n)
        d = {
            "ipo_id": [f"{market}-{i}" for i in range(n)],
            "market": market,
            "company": [f"{market} company {i}" for i in range(n)],
            "currency": "INR" if india else "USD",
            "listing_date": pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 6 * 365, n), unit="D"),
            "issue_size_m": np.exp(rng.normal(5, 1, n)),
            OPEN: opened,
            CLOSE: close,
            "source": "kaggle:ng3412+px_join" if market == "HK" else "synthetic",
            "n_missing_core": 0,
        }
        if india:
            d.update({
                "sub_total": total,
                "sub_qib": total * np.exp(rng.normal(0, 0.2, n)),
                "sub_nii": total * np.exp(rng.normal(0, 0.2, n)),
                "sub_retail": total * np.exp(rng.normal(0, 0.2, n)),
                "idx_ret_20d": rng.normal(0, 3, n),
                "idx_ret_5d": rng.normal(0, 2, n),
                "vix_level": rng.uniform(10, 30, n),
                "gmp_pct_last": opened + rng.normal(0, 10, n),
            })
        if market == "HK":
            d["sub_retail"] = total   # HK: public-offer multiple only
        blocks.append(pd.DataFrame(d))
    return pd.concat(blocks, ignore_index=True).reindex(columns=ALL_COLS)


def test_planted_relation_is_recovered_with_a_bootstrap_interval():
    df, _ = R._prepare(make_frame())
    g = df[df["market"] == "IN_MAIN"][["sub_total", CLOSE]].dropna()
    est, lo, hi = R._boot_rho(g["sub_total"].to_numpy(float), g[CLOSE].to_numpy(float), np.random.default_rng(1))
    assert est > 0.75 and lo < est < hi and lo > 0.6
    assert R._grade((est, lo, hi)) == "strong"


def test_hc3_regression_recovers_known_coefficients():
    rng = np.random.default_rng(5)
    x1, x2 = rng.normal(size=400), rng.normal(size=400)
    y = 1.0 + 2.0 * x1 - 3.0 * x2 + rng.normal(scale=0.5, size=400)
    res = R._regress(pd.DataFrame({"y": y, "x1": x1, "x2": x2}), "y", ["x1", "x2"])
    truth = [1.0, 2.0, -3.0]
    for j, t in enumerate(truth):
        assert res["lo"][j] < t < res["hi"][j], (j, res["beta"][j], res["lo"][j], res["hi"][j])
    assert res["n"] == 400 and 0.9 < res["r2"] < 1.0


def test_bucket_medians_follow_the_planted_relation():
    df, _ = R._prepare(make_frame())
    g = df[df["market"] == "IN_MAIN"]
    cut = pd.cut(g["sub_total"], bins=R.MULT_EDGES, labels=R.MULT_LABELS, right=False)
    rows = R._bucket_rows(g, cut, R.MULT_LABELS)
    medians = [float(r[5]) for r in rows if r[4] >= R.MIN_N]
    assert len(medians) >= 3
    assert all(a < b for a, b in zip(medians, medians[1:]))


def test_cells_under_twenty_rows_are_suppressed_and_show_their_n():
    assert R._pair(np.arange(19, dtype=float)) == ["n<20", "n<20"]
    assert R._pair(np.arange(20, dtype=float))[0] != "n<20"
    assert R._boot_rho(np.arange(19.0), np.arange(19.0), np.random.default_rng(0)) is None
    df, _ = R._prepare(make_frame())
    g = df[df["market"] == "IN_MAIN"]
    small = g.iloc[:5].assign(sub_total=[1.5, 2.5, 4.0, 15.0, 40.0])
    cut = pd.cut(small["sub_total"], bins=R.MULT_EDGES, labels=R.MULT_LABELS, right=False)
    row = [r for r in R._bucket_rows(small, cut, R.MULT_LABELS) if r[0] == "3-10x"][0]
    assert row[1] == 1 and row[2] == "n<20"


def test_nan_is_missing_and_never_becomes_zero():
    frame = make_frame()
    idx = frame.index[frame["market"] == "IN_MAIN"][:100]
    frame.loc[idx, "sub_qib"] = np.nan
    frame.loc[idx[:10], CLOSE] = np.nan
    df, meta = R._prepare(frame)
    assert df.loc[idx, "sub_qib"].isna().all()
    assert meta["zeroed"]["sub_qib"] == {}
    analysis = R._Analysis(df, meta, np.random.default_rng(0))
    g = df[df["market"] == "IN_MAIN"]
    n, _ = analysis.rho("IN_MAIN", g, "sub_qib", CLOSE)
    assert n == len(g[["sub_qib", CLOSE]].dropna())


def test_exact_zeros_are_missing_codes_and_are_counted():
    frame = make_frame()
    rows = frame.index[frame["market"] == "IN_MAIN"][:3]
    frame.loc[rows, CLOSE] = 0.0
    frame.loc[rows, "gmp_pct_last"] = 0.0
    frame.loc[rows, "sub_retail"] = 0.0
    df, meta = R._prepare(frame)
    assert df.loc[rows, CLOSE].isna().all() and df.loc[rows, "gmp"].isna().all()
    assert df.loc[rows, "sub_retail"].isna().all()
    assert meta["zeroed"][CLOSE] == {"IN_MAIN": 3}
    assert meta["zeroed"]["gmp_pct_last"] == {"IN_MAIN": 3}


def test_report_has_ten_summary_bullets_with_strength_words(tmp_path):
    out = R.run(out=tmp_path / "RESEARCH.md", frame=make_frame())
    text = out.read_text(encoding="utf-8")
    bullets = R.read_summary(out)
    assert len(bullets) == 10
    assert bullets[0].startswith("Listing day")
    assert "strong" in text and "not distinguishable from noise" in text
    assert "n<20" in text and "Schema columns missing" in text


def test_missing_whole_columns_do_not_crash(tmp_path):
    frame = make_frame()
    slim = frame[["ipo_id", "market", "company", "listing_date", OPEN, CLOSE, "source"]].copy()
    out = R.run(out=tmp_path / "slim.md", frame=slim)
    text = out.read_text(encoding="utf-8")
    assert "gmp_pct_last" in text and "sub_qib" in text
    assert len(R.read_summary(out)) == 10


def test_missing_listing_dates_and_market_do_not_crash(tmp_path):
    frame = make_frame().assign(listing_date=pd.NaT)
    out = R.run(out=tmp_path / "nodate.md", frame=frame)
    assert out.exists()


def test_empty_table_writes_a_report(tmp_path):
    out = R.run(out=tmp_path / "empty.md", frame=make_frame().iloc[0:0])
    assert "0 IPOs" in out.read_text(encoding="utf-8")


def test_rerun_on_the_same_table_is_identical(tmp_path):
    frame = make_frame()
    a = R.run(out=tmp_path / "a.md", frame=frame).read_text(encoding="utf-8")
    b = R.run(out=tmp_path / "b.md", frame=frame).read_text(encoding="utf-8")
    assert a == b


def test_default_output_resolves_at_call_time(tmp_path, monkeypatch):
    target = tmp_path / "redirected.md"
    monkeypatch.setattr(R, "OUT", target)
    assert R.run(frame=make_frame()) == target and target.exists()


def test_cli_missing_history_table_is_not_available(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IPOS_PARQUET", tmp_path / "absent.parquet")
    r = CliRunner().invoke(app, ["research"])
    assert r.exit_code == 2 and "ipo data build" in r.output


def test_cli_research_writes_and_prints_the_summary(tmp_path, monkeypatch):
    table = tmp_path / "ipos.parquet"
    make_frame().to_parquet(table)
    monkeypatch.setattr(config, "IPOS_PARQUET", table)
    monkeypatch.setattr(R, "OUT", tmp_path / "RESEARCH.md")
    r = CliRunner().invoke(app, ["--plain", "research"])
    assert r.exit_code == 0, r.output
    assert "wrote" in r.output and "Listing day" in r.output
    assert (tmp_path / "RESEARCH.md").exists()
