"""Alert rules. `evaluate` compares the live data with the previous snapshot and returns the alerts that are new.

Pure given its inputs: the only read from `state` is "was this key already sent" and "what was the last
snapshot", so the same inputs always give the same alerts. Sending and marking are the daemon's job.

Missing data never alerts: a None GMP or subscription is skipped, and a missing previous snapshot means the
first observation is stored without any change alerts.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from datetime import date, datetime, time
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from ipo_radar.alerts.notify import Alert
from ipo_radar.schema import LiveIPO, Prediction, Sourced

IST = ZoneInfo("Asia/Kolkata")

P_SOON, P_OPEN, P_CLOSE, P_FLIP, P_MILESTONE, P_LISTING, P_STALE = 2, 3, 4, 4, 3, 4, 2
P_GMP_MOVE, P_ALLOT = 3, 3

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "open_soon_days": 2,
    "milestones": [1, 5, 10, 25, 50, 100],
    "gmp_move_min_inr": 10.0,       # a move must be at least max(this, gmp_move_min_pct of previous GMP)
    "gmp_move_min_pct": 20.0,
    "quiet_hours": "23:00-07:00",   # IST; empty string disables quiet hours
}
# subscription keys tracked for milestones, with the names shown to the user
SUB_LABELS = {"sub_total": "total", "sub_qib": "QIB", "sub_nii": "NII", "sub_retail": "retail"}
_SYMBOL = {"IN_MAIN": "₹", "IN_SME": "₹", "US": "$", "HK": "HK$"}
_STALE_RE = re.compile(r"unavailable|stale")


class StateView(Protocol):
    def last_snapshot(self, ipo_id: str) -> dict[str, Any] | None: ...
    def already_sent(self, key: str) -> bool: ...


@dataclass(frozen=True)
class _Cand:
    alert: Alert
    quiet_exempt: bool = False   # closes-today and listing-day alerts still make noise during quiet hours


def _num(v: float | None) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def snapshot_of(ipo: LiveIPO) -> dict[str, Any]:
    """What the next run compares against. Only numbers and the status string, so it serialises cleanly."""
    return {
        "status": ipo.status,
        "gmp": _num(ipo.gmp["gmp"].value) if "gmp" in ipo.gmp else None,
        "sub": {k: _num(ipo.sub[k].value) for k in SUB_LABELS if k in ipo.sub},
    }


def _aware(dt: datetime) -> datetime:
    """Naive timestamps are read as IST, the only timezone this tool reports in."""
    return dt.replace(tzinfo=IST) if dt.tzinfo is None else dt


def _thresholds(cfg: dict | None) -> dict[str, Any]:
    return {**DEFAULT_THRESHOLDS, **((cfg or {}).get("thresholds") or {})}


def _parse_quiet(spec: str) -> tuple[time, time] | None:
    if not spec:
        return None
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*", spec)
    if not m:
        raise ValueError(f"quiet_hours must look like '23:00-07:00', got {spec!r}")
    h1, m1, h2, m2 = (int(g) for g in m.groups())
    return time(h1, m1), time(h2, m2)


def _in_quiet(now_ist: datetime, window: tuple[time, time] | None) -> bool:
    if window is None:
        return False
    start, end = window
    t = now_ist.time()
    return start <= t < end if start <= end else (t >= start or t < end)  # window may cross midnight


def _money(v: float, market: str, digits: int = 0, sign: bool = False) -> str:
    sym = _SYMBOL.get(market, "")
    sign_s = "-" if v < 0 else ("+" if sign and v > 0 else "")
    return f"{sign_s}{sym}{abs(v):,.{digits}f}"


def _sub_parts(ipo: LiveIPO, keys: list[str]) -> list[str]:
    parts = []
    for key in keys:
        s = ipo.sub.get(key)
        if s is not None and s.value is not None:
            parts.append(f"{SUB_LABELS[key]} {s.value:.1f}x")
    return parts


def _gmp_now(ipo: LiveIPO) -> float | None:
    s = ipo.gmp.get("gmp")
    return _num(s.value) if s is not None else None


def _footer(ipo: LiveIPO, now: datetime) -> str:
    """Line 3 on the phone: where the newest number came from, and how old it is in minutes."""
    newest: Sourced | None = None
    for s in [*ipo.sub.values(), *ipo.gmp.values()]:
        if s.fetched_at is not None and (newest is None or _aware(s.fetched_at) > _aware(newest.fetched_at)):
            newest = s
    source = (newest.source if newest and newest.source else ", ".join(ipo.sources)) or "live feed"
    if newest is None:
        return f"{source} · age unknown"
    age = max(0, int((now - _aware(newest.fetched_at)).total_seconds() // 60))
    return f"{source} · {age} min old"


def _model_line(pred: Prediction | None) -> str | None:
    if pred is None:
        return None
    line = (f"model {pred.open_gain_pct_p50:+.0f}% open "
            f"(range {pred.open_gain_pct_p10:+.0f}% to {pred.open_gain_pct_p90:+.0f}%)")
    if not pred.reliable:
        line += " [inputs outside training data]"
    return line


def _band_line(ipo: LiveIPO) -> str | None:
    parts = []
    if ipo.price_low is not None and ipo.price_high is not None:
        if ipo.price_low == ipo.price_high:
            parts.append(f"price {_money(ipo.price_low, ipo.market)}")   # fixed-price issue
        else:
            parts.append(f"band {_money(ipo.price_low, ipo.market)}-{_money(ipo.price_high, ipo.market)}")
    elif ipo.price_high is not None or ipo.price_low is not None:
        parts.append(f"price {_money(ipo.price_high or ipo.price_low or 0, ipo.market)}")
    if ipo.lot_size:
        parts.append(f"lot {ipo.lot_size}")
    if ipo.issue_size_m is not None:
        if ipo.market in ("IN_MAIN", "IN_SME"):
            parts.append(f"size ₹{ipo.issue_size_m / 10:,.0f} cr")   # 1 crore = 10 million
        else:
            parts.append(f"size {ipo.issue_size_m:,.0f}m")
    return " · ".join(parts) or None


def _lines(*parts: str | None) -> str:
    return "\n".join(p for p in parts if p)


def _schedule_rules(ipo: LiveIPO, pred: Prediction | None, today: date, now: datetime,
                    th: dict[str, Any]) -> list[_Cand]:
    out: list[_Cand] = []
    name, rid = ipo.company, ipo.ipo_id
    model = _model_line(pred)
    footer = _footer(ipo, now)

    if ipo.open_date is not None:
        days = (ipo.open_date - today).days
        if days == 0:
            out.append(_Cand(Alert(f"{rid}:opens-today:{ipo.open_date}", f"{name} opens today",
                                   _lines(_band_line(ipo), model, footer), P_OPEN, ["calendar"])))
        elif 0 < days <= th["open_soon_days"]:
            out.append(_Cand(Alert(f"{rid}:opens-soon:{ipo.open_date}", f"{name} opens in {days} days",
                                   _lines(_band_line(ipo), model, footer), P_SOON, ["calendar"])))

    if ipo.close_date == today:
        parts = _sub_parts(ipo, list(SUB_LABELS))
        gmp = _gmp_now(ipo)
        if gmp is not None:
            parts.append(f"GMP {_money(gmp, ipo.market, sign=True)}")
        out.append(_Cand(Alert(f"{rid}:closes-today:{ipo.close_date}", f"Last chance: {name} closes today",
                               _lines(" · ".join(parts) or None, footer), P_CLOSE, ["hourglass"]),
                         quiet_exempt=True))

    if ipo.allotment_date == today:
        out.append(_Cand(Alert(f"{rid}:allotment:{ipo.allotment_date}", f"Allotment today: {name}",
                               _lines("Check your allotment status with the registrar", footer),
                               P_ALLOT, ["ticket"])))

    if ipo.listing_date is not None:
        days = (ipo.listing_date - today).days
        if days == 1:
            out.append(_Cand(Alert(f"{rid}:listing-tomorrow:{ipo.listing_date}", f"{name} lists tomorrow",
                                   _lines(model or "no model view", footer), P_LISTING, ["rocket"])))
        elif days == 0:
            out.append(_Cand(Alert(f"{rid}:listing-today:{ipo.listing_date}", f"{name} lists today",
                                   _lines(model or "no model view", footer), P_LISTING, ["rocket"]),
                             quiet_exempt=True))
    return out


def _listed_open_rule(ipo: LiveIPO, pred: Prediction | None, today: date, now: datetime) -> list[_Cand]:
    """Realised opening gain against the prediction. Needs the final issue price (`issue_price`, or a fixed-price
    issue where low == high); a band alone is not enough, so no gain is computed rather than guessed."""
    open_p = ipo.listing.get("open")
    if open_p is None or open_p <= 0 or ipo.listing_date is None or ipo.listing_date > today:
        return []
    issue = ipo.issue_price
    if issue is None and ipo.price_low is not None and ipo.price_low == ipo.price_high:
        issue = ipo.price_low
    if issue is None or issue <= 0:
        return []
    gain = (open_p / issue - 1) * 100
    body = [f"opened {_money(open_p, ipo.market, 2)} vs issue {_money(issue, ipo.market, 2)}"]
    if pred is not None:
        inside = pred.open_gain_pct_p10 <= gain <= pred.open_gain_pct_p90
        body.append(f"model said {pred.open_gain_pct_p50:+.0f}% "
                    f"(range {pred.open_gain_pct_p10:+.0f}% to {pred.open_gain_pct_p90:+.0f}%), "
                    f"{'inside' if inside else 'outside'} the range")
    body.append(_footer(ipo, now))
    alert = Alert(f"{ipo.ipo_id}:listed-open:{ipo.listing_date}", f"{ipo.company} opened {gain:+.1f}%",
                  _lines(*body), P_LISTING, ["rocket"])
    return [_Cand(alert)]


def _gmp_rules(ipo: LiveIPO, prev: dict[str, Any] | None, now: datetime, th: dict[str, Any]) -> list[_Cand]:
    cur = _gmp_now(ipo)
    prev_v = prev.get("gmp") if prev else None
    if cur is None or prev_v is None:
        return []
    footer = _footer(ipo, now)
    transition = f"{prev_v:.0f}>{cur:.0f}"
    if (prev_v > 0 and cur < 0) or (prev_v < 0 and cur > 0):
        word = "turned negative" if cur < 0 else "turned positive"
        return [_Cand(Alert(f"{ipo.ipo_id}:gmp-flip:{transition}", f"{ipo.company} GMP {word}",
                            _lines(f"GMP {_money(prev_v, ipo.market, sign=True)} to "
                                   f"{_money(cur, ipo.market, sign=True)}", footer),
                            P_FLIP, ["money_with_wings"]))]
    needed = max(th["gmp_move_min_inr"], th["gmp_move_min_pct"] / 100 * abs(prev_v))
    if abs(cur - prev_v) >= needed:
        return [_Cand(Alert(f"{ipo.ipo_id}:gmp-move:{transition}", f"{ipo.company} GMP moved",
                            _lines(f"GMP {_money(prev_v, ipo.market, sign=True)} to "
                                   f"{_money(cur, ipo.market, sign=True)}", footer),
                            P_GMP_MOVE, ["money_with_wings"]))]
    return []


def _milestone_rules(ipo: LiveIPO, prev: dict[str, Any] | None, now: datetime, th: dict[str, Any]) -> list[_Cand]:
    """One alert per category per poll: the highest threshold crossed upward since the previous snapshot."""
    if prev is None:
        return []
    prev_sub = prev.get("sub") or {}
    out = []
    for key, label in SUB_LABELS.items():
        s = ipo.sub.get(key)
        cur = _num(s.value) if s is not None else None
        before = prev_sub.get(key)
        if cur is None or before is None:
            continue
        crossed = [t for t in th["milestones"] if before < t <= cur]
        if not crossed:
            continue
        top = max(crossed)
        out.append(_Cand(Alert(
            f"{ipo.ipo_id}:milestone:{key}:{top:g}x",
            f"{ipo.company}: {label} crosses {top:g}x",
            _lines(f"{label} {cur:.1f}x (was {before:.1f}x)", _footer(ipo, now)),
            P_MILESTONE, ["chart_with_upwards_trend"],
        )))
    return out


def _source_name(warning: str) -> str:
    head = re.split(r"[:(]| unavailable| stale", warning, maxsplit=1)[0].strip()
    return head or "unknown source"


def _stale_rules(live: list[LiveIPO], now: datetime, today: date) -> list[_Cand]:
    """One 'data source down' alert per source per day, however many IPOs it affects."""
    groups: dict[str, list[str]] = {}
    for ipo in live:
        for w in ipo.warnings:
            if _STALE_RE.search(w):
                names = groups.setdefault(_source_name(w), [])
                if ipo.company not in names:
                    names.append(ipo.company)
    out = []
    for source, names in groups.items():
        slug = re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-") or "unknown"
        more = ", ..." if len(names) > 3 else ""
        out.append(_Cand(Alert(
            f"source-down:{slug}:{today}",
            f"Data source down: {source}",
            _lines(f"{len(names)} IPO(s) affected: {', '.join(names[:3])}{more}",
                   f"{source} · checked {now.astimezone(IST):%H:%M} IST"),
            P_STALE, ["warning"],
        )))
    return out


def evaluate(
    live: list[LiveIPO],
    preds: dict[str, Prediction],
    state: StateView,
    now: datetime | None = None,
    cfg: dict | None = None,
) -> list[Alert]:
    """Return the alerts that are new for this poll. `now` defaults to the current time in IST."""
    th = _thresholds(cfg)
    now = _aware(now) if now is not None else datetime.now(IST)
    now_ist = now.astimezone(IST)
    today = now_ist.date()
    window = _parse_quiet(th["quiet_hours"])
    quiet = _in_quiet(now_ist, window)

    cands: list[_Cand] = []
    for ipo in live:
        pred = preds.get(ipo.ipo_id)
        prev = state.last_snapshot(ipo.ipo_id)
        cands += _schedule_rules(ipo, pred, today, now_ist, th)
        cands += _listed_open_rule(ipo, pred, today, now_ist)
        cands += _gmp_rules(ipo, prev, now_ist, th)
        cands += _milestone_rules(ipo, prev, now_ist, th)
    cands += _stale_rules(live, now_ist, today)

    out: list[Alert] = []
    seen: set[str] = set()
    for c in cands:
        a = c.alert
        if a.key in seen or state.already_sent(a.key):
            continue
        seen.add(a.key)
        if quiet and not c.quiet_exempt:
            a = replace(a, priority=1)  # silent, but still delivered and recorded
        out.append(a)
    return out
