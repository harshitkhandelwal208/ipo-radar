"""Combine NSE, Chittorgarh and GMP into one list of LiveIPO, flagging every disagreement.

Rules: the exchange (NSE) wins for subscription and for any field both sources report. Chittorgarh fills
gaps. GMP is attached per aggregator and summarised. A failed source never crashes the run: it is a warning
on each IPO and its fields stay missing. If both NSE and Chittorgarh fail there is no data at all, so
SourceError is raised.

Matching: exact company_key, then symbol, then keys.same_company (a conservative alias rule that needs open
dates and markets to agree). GMP readings are matched the same way, using the open date and market the GMP
site gave.

Subscription warnings: one per IPO. A category is compared only when both sources report it, and only when
the difference is over 5% and over 0.05x. At most three categories are named. The warning is dropped when the
two samples were taken more than 10 minutes apart and the issue is still open (the figures are still moving).
Same-moment mismatches, and any mismatch after close, are kept.

GMP after listing is moot: a listed IPO gets no GMP block.
"""
from __future__ import annotations

import json
import logging
import os
import statistics
from dataclasses import fields, is_dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from ipo_radar import config
from ipo_radar.errors import SourceError
from ipo_radar.live import chittorgarh, nse, snapshots
from ipo_radar.live.keys import company_key, same_company
from ipo_radar.schema import SUB_COLS, LiveIPO, Sourced

_log = logging.getLogger(__name__)
SUB_TOLERANCE = 0.05             # relative difference between two sources on the same category
SUB_ABS_TOLERANCE = 0.05         # times subscribed; below this a difference is noise
SUB_NAMED_MAX = 3                # categories named in one warning
SAMPLE_GAP = timedelta(minutes=10)
SUB_LABEL = {"sub_qib": "QIB", "sub_nii": "NII", "sub_bnii": "bNII", "sub_snii": "sNII",
             "sub_retail": "retail", "sub_employee": "employee", "sub_shareholder": "shareholder",
             "sub_total": "total"}
GMP_ABS_TOLERANCE = 10.0         # rupees
GMP_REL_TOLERANCE = 0.25         # fraction of the larger reading
GMP_MAX_AGE = timedelta(hours=6)  # a GMP reading older than this is not used for the median
_DATE_FIELDS = ("open_date", "close_date", "allotment_date", "listing_date")
_SCALAR_FIELDS = ("price_low", "price_high", "issue_price", "lot_size", "issue_size_m", "fresh_issue_pct")


def _today() -> date:
    """Single place that reads the clock, so tests can patch it."""
    return date.today()


def _aware(stamp: datetime) -> datetime:
    return stamp if stamp.tzinfo else stamp.astimezone()


def derive_status(ipo: LiveIPO, today: date) -> str:
    """upcoming | open | closed | allotment | listing_today | listed | unknown, from the dates alone."""
    if ipo.open_date is None:
        return "unknown"
    if today < ipo.open_date:
        return "upcoming"
    if today <= (ipo.close_date or ipo.open_date):
        return "open"
    if ipo.listing_date is None or today < ipo.listing_date:
        if ipo.allotment_date is not None and today >= ipo.allotment_date:
            return "allotment"
        return "closed"
    if today == ipo.listing_date:
        return "listing_today"
    return "listed"


def _find(records: list[LiveIPO], keys: dict[str, LiveIPO], row: LiveIPO) -> LiveIPO | None:
    """The record a Chittorgarh row belongs to: exact key, then symbol, then the alias rule."""
    key = company_key(row.company)
    if key in keys:
        return keys[key]
    if row.symbol:
        for rec in records:
            if rec.symbol and rec.symbol == row.symbol:
                return rec
    for rec in records:
        if same_company(key, rec.ipo_id, market_a=row.market, market_b=rec.market,
                        open_a=row.open_date, open_b=rec.open_date):
            return rec
    return None


def _find_gmp(records: list[LiveIPO], keys: dict[str, LiveIPO], key: str, market: str,
              open_date: date | None) -> LiveIPO | None:
    """The record a GMP company key belongs to. Exact key first, then the alias rule with the site's date/market."""
    if key in keys:
        return keys[key]
    for rec in records:
        if same_company(key, rec.ipo_id, market_a=market, market_b=rec.market,
                        open_a=open_date, open_b=rec.open_date):
            return rec
    return None


def _differs(a: float | date | None, b: float | date | None) -> bool:
    return a is not None and b is not None and a != b


def _merge_chittorgarh(rec: LiveIPO, other: LiveIPO) -> None:
    """Fill the gaps in rec (NSE-sourced) from Chittorgarh. Scalar disagreements are warnings, not overwrites.
    Subscription is compared separately, once the status is known (see _subscription_warning)."""
    if not rec.market:
        rec.market = other.market
    if not rec.symbol:
        rec.symbol = other.symbol
    for attr in _DATE_FIELDS:
        mine, theirs = getattr(rec, attr), getattr(other, attr)
        if mine is None:
            setattr(rec, attr, theirs)
        elif _differs(mine, theirs):
            rec.warnings.append(f"{attr} disagree: nse {mine} vs chittorgarh {theirs}")
    for attr in _SCALAR_FIELDS:
        mine, theirs = getattr(rec, attr), getattr(other, attr)
        if mine is None:
            setattr(rec, attr, theirs)
        elif _differs(mine, theirs) and attr in ("price_low", "price_high", "issue_price"):
            rec.warnings.append(f"{attr} disagree: nse {mine} vs chittorgarh {theirs}")
    for key in SUB_COLS:
        if key in other.sub and key not in rec.sub:
            rec.sub[key] = other.sub[key]
    if not rec.sub_by_day and other.sub_by_day:
        rec.sub_by_day = other.sub_by_day
    rec.sources += [s for s in other.sources if s not in rec.sources]


def _fmt(value: float) -> str:
    """'1.2', '10', '0.05'; below 0.1 keep significant digits so '0.0004836' reads as '0.00048'."""
    if abs(value) >= 0.1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.2g}"


def _subscription_warning(rec: LiveIPO, cg_sub: dict[str, Sourced]) -> str | None:
    """One line per IPO: the categories where NSE and Chittorgarh both report and differ by over 5% and 0.05x."""
    diffs = []
    for key in SUB_COLS:
        a, b = rec.sub.get(key), cg_sub.get(key)
        if a is None or b is None or a.source != "nse" or b.source == "nse":
            continue
        if a.value is None or b.value is None or a.value == b.value:
            continue
        gap, base = abs(a.value - b.value), max(abs(a.value), abs(b.value))
        if gap > SUB_TOLERANCE * base and gap > SUB_ABS_TOLERANCE:
            diffs.append((key, a, b))
    if not diffs:
        return None
    spreads = [abs(_aware(a.fetched_at) - _aware(b.fetched_at))
               for _, a, b in diffs if a.fetched_at is not None and b.fetched_at is not None]
    sampled_apart = len(spreads) == len(diffs) and max(spreads) > SAMPLE_GAP
    if sampled_apart and rec.status == "open":
        return None  # taken minutes apart while bids still arrive: the gap is timing, not a disagreement
    parts = [f"{SUB_LABEL[k]} {_fmt(a.value)}x ({a.source}) vs {_fmt(b.value)}x ({b.source})"
             for k, a, b in diffs[:SUB_NAMED_MAX]]
    text = "subscription sources differ: " + ", ".join(parts)
    more = len(diffs) - SUB_NAMED_MAX
    if more > 0:
        text += f" and {more} more categor{'y' if more == 1 else 'ies'}"
    return text


def _gmp_meta(readings: list[Sourced]) -> tuple[str, date | None]:
    """The market and open day the GMP sites gave for this IPO (first one that gave each)."""
    market = next((getattr(r, "market", "") for r in readings if getattr(r, "market", "")), "")
    opened = next((getattr(r, "open_date", None) for r in readings if getattr(r, "open_date", None)), None)
    return market, opened


def _one_per_source(readings: list[Sourced]) -> list[Sourced]:
    """A site that lists one IPO under two names gives two rows. Keep its first reading, as one per site."""
    seen: dict[str, Sourced] = {}
    for reading in readings:
        if reading.source in seen:
            _log.info("GMP source %s listed one IPO twice; keeping the first reading", reading.source)
            continue
        seen[reading.source] = reading
    return list(seen.values())


def _attach_gmp(rec: LiveIPO, readings: list[Sourced], now: datetime) -> None:
    """One key per aggregator, plus 'gmp' (median of fresh readings) and 'gmp_pct' (% of the issue high)."""
    for reading in readings:
        rec.gmp[reading.source] = Sourced(reading.value, reading.source, reading.fetched_at, reading.note)
        if reading.value is not None and reading.fetched_at is not None:
            rec.gmp_history.append((reading.fetched_at, reading.value, reading.source))
    fresh = [r for r in readings if r.value is not None and r.fetched_at is not None
             and now - _aware(r.fetched_at) <= GMP_MAX_AGE]
    if not fresh:
        return
    values = [r.value for r in fresh]
    stamp = max(_aware(r.fetched_at) for r in fresh)  # type: ignore[arg-type]
    median = float(statistics.median(values))  # type: ignore[arg-type]
    rec.gmp["gmp"] = Sourced(median, "gmp", stamp, note=f"median of {len(fresh)} fresh reading(s)")
    if rec.price_high:
        rec.gmp["gmp_pct"] = Sourced(round(median / rec.price_high * 100, 2), "gmp", stamp,
                                     note="gmp / price_high")
    if len(fresh) >= 2:
        lo, hi = min(fresh, key=lambda r: r.value), max(fresh, key=lambda r: r.value)  # type: ignore[arg-type]
        spread = hi.value - lo.value  # type: ignore[operator]
        if spread > max(GMP_ABS_TOLERANCE, GMP_REL_TOLERANCE * abs(hi.value)):  # type: ignore[operator]
            rec.warnings.append(f"GMP sources disagree: {lo.value:g} ({lo.source}) vs "
                                f"{hi.value:g} ({hi.source})")


def merge_sources(nse_rows: list[LiveIPO], cg_rows: list[LiveIPO], gmp_map: dict[str, list[Sourced]] | None,
                  *, failures: dict[str, str], today: date, now: datetime | None = None) -> list[LiveIPO]:
    """Pure reconcile step. `failures` maps a source name to its error text; a failed source warns on every IPO."""
    now = now or datetime.now().astimezone()
    records: list[LiveIPO] = []
    keys: dict[str, LiveIPO] = {}
    cg_subs: dict[str, dict[str, Sourced]] = {}  # ipo_id -> Chittorgarh subscription, kept for the comparison
    for row in nse_rows:
        if row.ipo_id in keys:  # same company twice in the NSE feed (e.g. current and past)
            _merge_chittorgarh(keys[row.ipo_id], row)
            continue
        records.append(row)
        keys[row.ipo_id] = row
    for row in cg_rows:
        target = _find(records, keys, row)
        if target is None:
            records.append(row)
            keys[row.ipo_id] = row
        else:
            _merge_chittorgarh(target, row)
            cg_subs.setdefault(target.ipo_id, {}).update(row.sub)
    for rec in records:
        rec.status = derive_status(rec, today)
        if not rec.market:
            rec.warnings.append("market unknown: no source states the segment (main board or SME)")
        if rec.status == "closed" and rec.listing_date is None and rec.close_date and (today - rec.close_date).days > 3:
            rec.warnings.append("listing date not reported by any source; status stays 'closed'")
        if rec.ipo_id in cg_subs:
            warning = _subscription_warning(rec, cg_subs[rec.ipo_id])
            if warning:
                rec.warnings.append(warning)
    for source, error in failures.items():
        if source == "gmp":
            continue  # reported below, and only for IPOs that still have GMP
        for rec in records:
            rec.warnings.append(f"{source} unavailable: {error}")
    if "gmp" in failures:
        for rec in records:
            if rec.status != "listed":
                rec.warnings.append(f"GMP unavailable: {failures['gmp']}")
    if gmp_map is not None:
        # several GMP names can resolve to one IPO ('Sahajanand Medical' and 'Sahajanand Medical Technologies'):
        # pool their readings first, so the median and the per-source values come from one set
        pooled: dict[int, tuple[LiveIPO, list[Sourced]]] = {}
        for key, readings in gmp_map.items():
            market, opened = _gmp_meta(readings)
            target = _find_gmp(records, keys, key, market, opened)
            if target is None:
                _log.info("GMP for %r has no matching IPO in NSE or Chittorgarh data; not shown", key)
            elif target.status == "listed":
                continue  # GMP is moot after listing; the listing prices stay
            else:
                pooled.setdefault(id(target), (target, []))[1].extend(readings)
        for target, readings in pooled.values():
            _attach_gmp(target, _one_per_source(readings), now)
    records.sort(key=lambda r: (r.open_date or date.max, r.company))
    return records


def apply_own_snapshots(records: list[LiveIPO], figures: dict[str, dict[int, dict[str, float]]]) -> None:
    """Fill sub_by_day days the sites gave nothing for, from our own end-of-day rows (live/snapshots.py).

    A day a site supplied is never overwritten. Filled days are marked with the source 'own snapshot'.
    """
    for rec in records:
        own = figures.get(rec.ipo_id)
        if not own:
            continue
        added = [day for day in sorted(own) if day not in rec.sub_by_day]
        if not added:
            continue
        for day in added:
            rec.sub_by_day[day] = dict(own[day])
        marks = [m for m in rec.sub_by_day_source.split("; ") if m]
        if snapshots.OWN_SOURCE not in marks:
            marks.append(snapshots.OWN_SOURCE)
        rec.sub_by_day_source = "; ".join(marks)


def _run(name: str, fn) -> tuple[list[LiveIPO] | None, str | None]:
    try:
        return fn(), None
    except SourceError as exc:
        _log.warning("%s failed: %s", name, exc)
        return None, str(exc)


def _run_gmp() -> tuple[dict[str, list[Sourced]] | None, str | None]:
    """GMP is optional and lazily imported. Any failure becomes a warning, never a crash of the whole run."""
    try:
        from ipo_radar.live import gmp
    except ImportError as exc:
        return None, f"GMP module not available ({exc})"
    try:
        return gmp.fetch(), None
    except SourceError as exc:
        _log.warning("gmp failed: %s", exc)
        return None, str(exc)
    except Exception as exc:  # a bug in the GMP module must not take the exchange and Chittorgarh data down
        _log.exception("gmp crashed")
        return None, f"GMP module error ({type(exc).__name__}: {exc})"


def fetch_all(use_cache: bool = True, ttl_seconds: int = 120) -> list[LiveIPO]:
    """Every live source merged. Cached as JSON for ttl_seconds once NSE and Chittorgarh both succeeded."""
    if use_cache:
        cached = _read_cache(ttl_seconds)
        if cached is not None:
            return cached
    nse_rows, nse_err = _run("nse", nse.fetch)
    cg_rows, cg_err = _run("chittorgarh", chittorgarh.fetch)
    gmp_map, gmp_err = _run_gmp()
    if nse_rows is None and cg_rows is None:
        raise SourceError(f"no IPO data: nse: {nse_err}; chittorgarh: {cg_err}")
    failures = {k: v for k, v in (("nse", nse_err), ("chittorgarh", cg_err), ("gmp", gmp_err)) if v}
    now = datetime.now().astimezone()
    merged = merge_sources(nse_rows or [], cg_rows or [], gmp_map, failures=failures, today=_today(), now=now)
    snapshots.record(merged, day=_today(), taken_at=now)  # our own day-wise history, one row per value
    apply_own_snapshots(merged, snapshots.day_figures(merged))
    if nse_err is None and cg_err is None:  # GMP may be absent for now; the IPO list itself is complete
        _write_cache(merged)
    return merged


# ---- JSON cache -----------------------------------------------------------------------------------
def to_json(ipos: list[LiveIPO]) -> str:
    """Lossless for the schema types: dates and datetimes as ISO strings, int keys and tuples restored on read."""
    return json.dumps({"ipos": [_encode(i) for i in ipos]}, indent=1)


def from_json(text: str) -> list[LiveIPO]:
    return [_decode(d) for d in json.loads(text)["ipos"]]


def _encode(obj):
    if is_dataclass(obj):
        return {f.name: _encode(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    return obj


def _decode_sourced(d: dict) -> Sourced:
    return Sourced(d["value"], d["source"], datetime.fromisoformat(d["fetched_at"]) if d["fetched_at"] else None,
                   d["note"])


def _decode(d: dict) -> LiveIPO:
    out = dict(d)
    for attr in _DATE_FIELDS:
        if out.get(attr):
            out[attr] = date.fromisoformat(out[attr])
    out["sub"] = {k: _decode_sourced(v) for k, v in d["sub"].items()}
    out["gmp"] = {k: _decode_sourced(v) for k, v in d["gmp"].items()}
    out["sub_by_day"] = {int(k): v for k, v in d["sub_by_day"].items()}
    out["gmp_history"] = [(datetime.fromisoformat(w), v, s) for w, v, s in d["gmp_history"]]
    return LiveIPO(**out)


def _cache_path() -> Path:
    return config.LIVE_CACHE / "merged.json"


def _read_cache(ttl_seconds: int) -> list[LiveIPO] | None:
    path = _cache_path()
    try:
        if not path.exists() or (datetime.now().timestamp() - path.stat().st_mtime) > ttl_seconds:
            return None
        return from_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        _log.warning("ignoring unreadable live cache %s: %s", path, exc)
        return None


def _write_cache(ipos: list[LiveIPO]) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(to_json(ipos), encoding="utf-8")
    os.replace(tmp, path)
