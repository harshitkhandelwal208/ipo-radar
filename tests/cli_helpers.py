"""Hand-built live IPOs and fakes for the CLI tests. Dates are relative to today so alert rules behave the same
on any day, and every fetch time is 'now' so the freshness rules are predictable."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from ipo_radar.alerts.engine import IST
from ipo_radar.alerts.notify import Alert
from ipo_radar.schema import ALL_COLS, LiveIPO, Prediction, Sourced


def now() -> datetime:
    return datetime.now(IST)


def today() -> date:
    return now().date()


def tonbo() -> LiveIPO:
    """Open, day 1 of 5, OFS 100%, GMP from three aggregators, subscription from NSE."""
    t, n = today(), now()
    return LiveIPO(
        ipo_id="tonbo imaging", market="IN_MAIN", company="Tonbo Imaging India Limited", symbol="TONBO",
        status="open", open_date=t, close_date=t + timedelta(days=4), listing_date=t + timedelta(days=8),
        price_low=665.0, price_high=700.0, lot_size=21, issue_size_m=8520.0, fresh_issue_pct=0.0,
        sub={
            "sub_qib": Sourced(0.4, "nse", n, "Qualified Institutional"),
            "sub_nii": Sourced(1.2, "nse", n), "sub_retail": Sourced(0.9, "nse", n),
            "sub_total": Sourced(0.8, "nse", n),
        },
        sub_by_day={1: {"sub_qib": 0.1, "sub_nii": 0.6, "sub_retail": 0.5, "sub_total": 0.4}},
        gmp={
            "investorgain": Sourced(500.0, "investorgain", n), "ipopremium": Sourced(495.0, "ipopremium", n),
            "gmp": Sourced(500.0, "gmp", n, "median of 2 fresh reading(s)"),
            "gmp_pct": Sourced(71.43, "gmp", n, "gmp / price_high"),
        },
        sources=["nse", "chittorgarh", "gmp"],
    )


def zetwerk_upcoming() -> LiveIPO:
    t = today()
    return LiveIPO(ipo_id="zetwerk", market="IN_MAIN", company="Zetwerk Manufacturing Businesses Limited",
                   symbol="ZETWERK", status="upcoming", open_date=t + timedelta(days=2),
                   close_date=t + timedelta(days=4), price_low=143.0, price_high=151.0, lot_size=99,
                   issue_size_m=50120.0, fresh_issue_pct=56.0, sources=["chittorgarh"])


def hd_fire_sme_conflict() -> LiveIPO:
    """SME, nearly closing, QIB under 1x, two GMP sources disagreeing, a stale GMP reading."""
    t, n = today(), now()
    old = n - timedelta(hours=3)
    return LiveIPO(
        ipo_id="hd fire protect", market="IN_SME", company="HD Fire Protect Limited", symbol="HDFIRE",
        status="open", open_date=t - timedelta(days=2), close_date=t, price_low=258.0, price_high=271.0,
        lot_size=500, issue_size_m=150.0, fresh_issue_pct=100.0,
        sub={"sub_qib": Sourced(0.6, "nse", n), "sub_nii": Sourced(0.8, "nse", n),
             "sub_retail": Sourced(3.0, "nse", n), "sub_total": Sourced(2.1, "nse", n)},
        gmp={"investorgain": Sourced(90.0, "investorgain", n), "ipowatch": Sourced(150.0, "ipowatch", old),
             "gmp": Sourced(120.0, "gmp", n), "gmp_pct": Sourced(44.28, "gmp", n)},
        warnings=["GMP sources disagree: 90 (investorgain) vs 150 (ipowatch)"],
        sources=["nse"],
    )


def listed_old() -> LiveIPO:
    t = today()
    return LiveIPO(ipo_id="old co", market="IN_MAIN", company="Old Co Limited", status="listed",
                   open_date=t - timedelta(days=20), close_date=t - timedelta(days=18),
                   listing_date=t - timedelta(days=13), price_low=100.0, price_high=100.0,
                   listing={"open": 130.0})


def fake_ipos() -> list[LiveIPO]:
    return [tonbo(), zetwerk_upcoming(), hd_fire_sme_conflict(), listed_old()]


def fake_predict(ipo: LiveIPO) -> Prediction:
    """A prediction that has demonstrated skill: no '~' marker."""
    return Prediction(
        ipo_id=ipo.ipo_id, as_of="pre_open", open_gain_pct_p10=-4.0, open_gain_pct_p50=12.0,
        open_gain_pct_p90=38.0, p_positive_open=0.78, close_gain_pct_p50=9.0,
        drivers=[("gmp_pct", 18.2), ("sub_total", 3.1), ("ofs_pct", -2.4)],
        baseline_gmp_pct=71.4, model_version="test-1", reliable=True,
        caveats=["few SME examples in training"],
    )


def no_skill_predict(ipo: LiveIPO) -> Prediction:
    """A prediction that does not beat the historical median here: P50 gets the '~' marker."""
    p = fake_predict(ipo)
    p.reliable = False
    p.caveats = ["no demonstrated skill over the historical median at this stage", "few SME examples in training"]
    return p


def history_frame() -> pd.DataFrame:
    """Five past main-board IPOs and two SME ones, with the columns the views read."""
    rows = []
    for i, (gmp_last, gain, sub) in enumerate([(10, 5, 2.0), (30, 25, 8.0), (32, 40, 12.0), (60, 55, 30.0),
                                               (70, 80, 60.0), (90, -2, 1.1)]):
        rows.append({c: None for c in ALL_COLS} | {
            "ipo_id": f"past{i}", "market": "IN_MAIN", "company": f"Past Main {i}", "symbol": f"PM{i}",
            "open_date": pd.Timestamp(date(2025, 1, 1) + timedelta(days=30 * i)),
            "gmp_pct_last": float(gmp_last), "list_open_gain_pct": float(gain), "sub_total": sub,
            "sub_qib": sub, "sub_nii": sub, "sub_retail": sub, "issue_size_m": 5000.0,
        })
    for i, gain in enumerate([3.0, -4.0]):
        rows.append({c: None for c in ALL_COLS} | {
            "ipo_id": f"sme{i}", "market": "IN_SME", "company": f"Past SME {i}", "symbol": f"SM{i}",
            "open_date": pd.Timestamp(date(2025, 6, 1) + timedelta(days=i)),
            "gmp_pct_last": 20.0, "list_open_gain_pct": gain, "sub_total": 4.0, "sub_qib": 1.0,
        })
    return pd.DataFrame(rows)


class Recorder:
    """A notifier that records alerts instead of sending them."""

    name = "recorder"

    def __init__(self) -> None:
        self.sent: list[Alert] = []

    def send(self, alert: Alert) -> None:
        self.sent.append(alert)


class Broken(Recorder):
    name = "broken"

    def send(self, alert: Alert) -> None:
        from ipo_radar.errors import SourceError

        raise SourceError("broken: HTTP 503")
