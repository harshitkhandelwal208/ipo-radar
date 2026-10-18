"""Evaluation tests: the time split, verdicts and metric arithmetic on known answers, skill flags, and the report."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ipo_radar import config
from ipo_radar.model import evaluate as ev
from ipo_radar.model import features as F
from ipo_radar.model import metrics
from ipo_radar.model import walkforward as W
from model_synth import make_ipos


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return make_ipos(n_per_year=80, years=range(2015, 2025), seed=31)


def test_each_block_trains_only_on_ipos_listed_before_it(data, monkeypatch):
    """Every fit_stage call of a test year sees training rows listed strictly before that year."""
    calls = []
    original = W.fit_stage

    def spy(rows, stage, vocab, spec, seeds, quick, params=None):
        calls.append(int(rows["listing_date"].dt.year.max()))
        return original(rows, stage, vocab, spec, seeds, quick, params)

    monkeypatch.setattr(W, "fit_stage", spy)
    base = data[data["listing_date"].notna()].reset_index(drop=True)
    records, _ = W.run_walk(base, ev.DEFAULT_DESIGN, [2022, 2023], [7], quick=True)
    fits = records.drop_duplicates(["year", "stage", "set"], keep="first")   # one fit per (year, stage, set)
    years = fits["year"].to_numpy()          # the walk fits in this same order
    assert len(calls) == len(years) > 0
    for year, last_train_year in zip(years, calls):
        assert last_train_year < year, (year, last_train_year)   # no training IPO listed in or after the test year


@pytest.mark.parametrize("interval, n, expected", [
    ((-2.0, -3.0, -1.0), 100, "beats"),
    ((2.0, 1.0, 3.0), 100, "worse than"),
    ((0.1, -0.5, 0.7), 100, "no clear difference"),
    ((-2.0, -3.0, -1.0), 10, "too few"),
    (None, 100, "too few"),
])
def test_verdicts_follow_the_interval_not_the_point_estimate(interval, n, expected):
    assert expected in metrics.verdict(interval, n)


def test_bootstrap_interval_covers_zero_for_noise_and_not_for_a_clear_gap():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, 400)
    _, lo, hi = metrics.ci(noise, rng)
    assert lo < 0 < hi
    _, _, hi2 = metrics.ci(noise - 5.0, rng)
    assert hi2 < 0


def _record_frame(n: int = 40, **over) -> pd.DataFrame:
    y = np.linspace(-10, 30, n)
    r = pd.DataFrame({
        "open_y": y, "close_y": y, "p10": y - 5, "p50": y, "p90": y + 5, "p_pos": np.where(y > 0, 0.9, 0.1),
        "med_open": np.zeros(n), "med_close": np.zeros(n), "close_p50": y, "lin_open": np.full(n, np.nan),
        "gmp_pct": np.full(n, np.nan), "base_pos": 0.5, "market": "IN_MAIN", "segment": "IN_MAIN", "stage": "close",
        "year": 2023,
    })
    for k, v in over.items():
        r[k] = v
    return r


def test_block_metrics_on_known_answers():
    out = metrics.block(_record_frame(), np.random.default_rng(1))
    assert out["mae"] == pytest.approx(0.0)
    assert out["coverage"] == pytest.approx(100.0)
    assert out["pinball"]["0.5"] == pytest.approx(0.0)
    assert out["auc"] == pytest.approx(1.0)
    assert out["brier"] < out["brier_base"]
    assert out["n_gmp"] == 0 and "mae_gmp" not in out


def test_gmp_baseline_is_only_scored_where_a_reading_exists():
    n = 30
    gmp = np.where(np.arange(n) % 2 == 0, np.arange(n, dtype=float), np.nan)
    r = _record_frame(n, gmp_pct=gmp, p50=np.zeros(n), p10=np.zeros(n), p90=np.zeros(n),
                      open_y=np.arange(n, dtype=float))
    out = metrics.block(r, np.random.default_rng(3))
    assert out["n_gmp"] == n // 2
    assert out["mae_gmp"] == pytest.approx(0.0)


def test_calibration_bins_cover_every_row_once():
    rng = np.random.default_rng(2)
    r = _record_frame(300, open_y=rng.normal(10, 30, 300), p_pos=rng.uniform(0, 1, 300))
    rows = metrics.calibration(r)
    assert len(rows) == 10 and sum(n for _, n, _, _ in rows) == 300


def test_skill_flags_need_the_interval_below_zero():
    rng = np.random.default_rng(4)
    n = 200
    y = rng.normal(10, 20, n)
    r = _record_frame(n, open_y=y, p50=y, med_open=np.full(n, 10.0), p_pos=np.full(n, 0.5), base_pos=0.5)
    r["segment"] = "IN_MAIN"
    flags = metrics.skill(r)["close"]["IN_MAIN"]
    assert flags["p50_beats_median"] is True
    assert flags["brier_beats_base"] is False          # identical probability to the base rate: no skill
    worse = metrics.skill(_record_frame(n, open_y=y, p50=np.full(n, 99.0), med_open=np.full(n, 10.0),
                                        p_pos=np.full(n, 0.5), base_pos=0.5, segment="IN_MAIN"))
    assert worse["close"]["IN_MAIN"]["p50_beats_median"] is False


def test_evaluate_writes_the_protocol_headline_and_experiment_sections(data, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS", tmp_path)   # keeps the real experiments.json out of the test
    out = tmp_path / "MODEL_REPORT.md"
    result = ev.evaluate(data, out_path=out, seed=7, quick=True)
    assert result.test_years == [2022, 2023, 2024]     # the synthetic table ends in 2024: no locked block
    text = out.read_text(encoding="utf-8")
    for section in ("## Protocol", "## Headline: locked blocks 2025-2026", "## What this model can and cannot tell you",
                    "## Per-stage skill flags", "## Experiments", "## GMP-bearing training rows", "## Data warnings",
                    "## Selection blocks 2022-2024"):
        assert section in text, section
    assert "no longer blind" in text
    assert "The experiment file is missing" in text    # no experiments.json in the temporary models folder
    assert set(result.verdicts) <= set(F.STAGES)


def test_gmp_row_counts_are_split_by_segment_and_year(data):
    counts = ev._gmp_counts(data)
    assert "IN_MAIN" in counts and "GLOBAL" in counts


def test_data_change_note_is_generated_with_and_without_the_old_snapshot(tmp_path):
    h_locked = {"pre_open": {"n": 40, "mae": 20.0, "mae_median": 18.0, "diff_median": [2.0, 1.0, 3.0],
                             "coverage": 90.0, "auc": 0.5}}
    missing = ev._data_change_section(h_locked, tmp_path / "absent.json")
    assert any("not available" in line for line in missing)
    snap = tmp_path / "locked_headline.json"
    snap.write_text(json.dumps({"stages": {"pre_open": {"n": 50, "mae": 18.6, "mae_median": 17.1,
                                                        "diff_median": [1.5, 0.9, 2.0], "coverage": 95.0,
                                                        "auc": 0.487}}}), encoding="utf-8")
    text = "\n".join(ev._data_change_section(h_locked, snap))
    assert "## Data change: placeholder zeros blanked" in text
    assert "50 to 40" in text and "18.6 to 20.0" in text and "no longer blind" in text
