"""Research report: which listing-time metrics go with listing-day gains and losses, and how reliably -> docs/RESEARCH.md.

Reads only data/processed/ipos.parquet. It never reads the live caches or the sqlite history. Everything here is
descriptive, or a rank correlation or regression on the observed table. Nothing is a forecast or a trading rule.

Conventions. NaN is missing. A 0.00% listing gain and a 0 subscription multiple are missing codes in this table
(section 1 of the report shows the evidence), so they are set to NaN and counted. Intervals are 95% percentile
bootstraps over IPOs, drawn from one generator with a fixed seed, so a rerun gives the same file. Cells with fewer
than MIN_N rows print 'n<20'. Nothing is winsorized: medians and rank correlations are the headline, and a mean is
always labelled as a mean.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ipo_radar import config
from ipo_radar.schema import ALL_COLS, SUB_COLS

SEED = 7                 # the seed of docs/MODEL_REPORT.md
BOOT = 2000              # bootstrap resamples per interval
MIN_N = 20               # cells with fewer rows print 'n<20'
Z95 = 1.96               # normal approximation for the HC3 intervals
RHO_STRONG = 0.3         # |rho| at which a CI-clear association is called 'strong' (a stated convention)
OUT = config.ROOT / "docs" / "RESEARCH.md"
MARKETS = ("IN_MAIN", "IN_SME", "US", "HK", "OTHER")
INDIA = ("IN_MAIN", "IN_SME")
OPEN, CLOSE = "list_open_gain_pct", "list_close_gain_pct"
CATEGORIES = [("sub_qib", "QIB"), ("sub_nii", "NII (HNI)"), ("sub_snii", "sNII"), ("sub_bnii", "bNII"),
              ("sub_retail", "retail"), ("sub_employee", "employee"), ("sub_shareholder", "shareholder"),
              ("sub_total", "total")]
MULT_EDGES = [0, 1, 3, 10, 30, 100, np.inf]
MULT_LABELS = ["<1x", "1-3x", "3-10x", "10-30x", "30-100x", ">100x"]
COVERAGE_FIELDS = [OPEN, CLOSE, "sub_qib", "sub_nii", "sub_snii", "sub_bnii", "sub_retail", "sub_employee",
                   "sub_shareholder", "sub_total", "gmp_pct_last", "idx_ret_20d", "vix_level", "issue_size_m",
                   "fresh_issue_pct", "ofs_pct", "price_high"]
GMP_YEARS = range(2019, 2025)
TEXT_COLS = {"market", "country", "exchange", "company", "symbol", "sector", "currency", "source", "source_url",
             "ipo_id", "open_date", "close_date", "listing_date"}


# ---- formatting ------------------------------------------------------------------------------------------
def _f(x: object, nd: int = 1, sign: bool = False) -> str:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def _pct(x: object) -> str:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    return f"{v * 100:.0f}%" if np.isfinite(v) else "n/a"


def _bar(share: float) -> str:
    k = int(round(share * 10))
    return f"{share * 100:.0f}% " + "█" * k + "░" * (10 - k)


def _table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c).replace("|", "/") for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _range_label(lo: float, hi: float) -> str:
    def show(x: float) -> str:
        return f"{x:,.0f}" if abs(x) >= 10 else f"{x:.2g}"
    if lo == -np.inf:
        return f"below {show(hi)}"
    if hi == np.inf:
        return f"{show(lo)} and above"
    return f"{show(lo)} to {show(hi)}"


# ---- statistics --------------------------------------------------------------------------------------------
def _rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def _chunks(total: int, size: int = 250):
    for start in range(0, total, size):
        yield min(size, total - start)


def _boot_rho(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float] | None:
    """Spearman rho with a 95% bootstrap interval over IPOs. None below MIN_N rows."""
    n = len(x)
    if n < MIN_N:
        return None
    draws = []
    for m in _chunks(BOOT):
        idx = rng.integers(0, n, size=(m, n))
        rx = rankdata(x[idx], axis=1)
        ry = rankdata(y[idx], axis=1)
        rx -= rx.mean(axis=1, keepdims=True)
        ry -= ry.mean(axis=1, keepdims=True)
        den = np.sqrt((rx * rx).sum(axis=1) * (ry * ry).sum(axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            draws.append((rx * ry).sum(axis=1) / den)
    r = np.concatenate(draws)
    r = r[np.isfinite(r)]
    est = _rho(x, y)
    if len(r) < BOOT // 2 or not np.isfinite(est):
        return None
    lo, hi = np.quantile(r, [0.025, 0.975])
    return est, float(lo), float(hi)


def _grade(res: tuple[float, float, float] | None) -> str:
    """The strength word. An interval that spans zero is noise, whatever the point estimate says."""
    if res is None:
        return "not tested (n<20)"
    est, lo, hi = res
    if lo <= 0 <= hi:
        return "not distinguishable from noise"
    return "strong" if abs(est) >= RHO_STRONG else "weak"


def _ci(res: tuple[float, float, float] | None) -> str:
    if res is None:
        return "n<20"
    est, lo, hi = res
    return f"{est:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def _direction(est: float | None) -> str:
    if est is None or not np.isfinite(est):
        return "no direction"
    return "higher goes with higher gains" if est > 0 else "higher goes with lower gains"


def _regress(d: pd.DataFrame, y: str, xs: list[str]) -> dict | None:
    """OLS with HC3 robust standard errors on the complete cases of `d` for these columns. None below MIN_N."""
    cc = d[[y, *xs]].dropna()
    n, k = len(cc), len(xs) + 1
    if n < MIN_N or n <= k:
        return None
    X = np.column_stack([np.ones(n), cc[xs].to_numpy(float)])
    yy = cc[y].to_numpy(float)
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ yy
    resid = yy - X @ beta
    h = np.clip(np.einsum("ij,jk,ik->i", X, xtx_inv, X), 0.0, 0.999999)
    meat = X.T @ (X * (resid ** 2 / (1 - h) ** 2)[:, None])
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    sst = float(((yy - yy.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / sst if sst > 0 else float("nan")
    return {"n": n, "names": ["const", *xs], "beta": beta, "lo": beta - Z95 * se, "hi": beta + Z95 * se, "r2": r2}


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    sst = float(((y - y.mean()) ** 2).sum())
    return 1 - float(resid @ resid) / sst if sst > 0 else float("nan")


def _pair(y: np.ndarray) -> list[str]:
    """Median and share above issue for one cell. Suppressed below MIN_N."""
    if len(y) < MIN_N:
        return ["n<20", "n<20"]
    return [_f(np.median(y), 1, True), _bar(float((y > 0).mean()))]


def _cut_quantiles(values: pd.Series) -> tuple[pd.Series, list[str]]:
    """Quartile buckets of one market's values. Edges come from the data, so the labels print them."""
    v = values.dropna().to_numpy(float)
    if len(v) < MIN_N:
        return pd.Series(np.nan, index=values.index, dtype="object"), []
    edges = sorted(set(np.quantile(v, [0.25, 0.5, 0.75]).tolist()))
    bins = [-np.inf, *edges, np.inf]
    labels = [_range_label(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
    return pd.cut(values, bins=bins, labels=labels, right=False), labels


class _Analysis:
    """Runs the sections on one prepared table. Sections append markdown to `body`, put the numbers the summary
    needs into `facts`, and count every interval they compute so the method section can state how many."""

    def __init__(self, df: pd.DataFrame, meta: dict, rng: np.random.Generator) -> None:
        self.df = df
        self.meta = meta
        self.rng = rng
        self.body: list[str] = []
        self.facts: dict = {}
        self.rho_cache: dict[tuple[str, str, str], tuple[int, tuple | None]] = {}
        self.n_ci = 0          # intervals computed on an association, coefficient or R-squared difference
        self.n_excl = 0        # of those, intervals that exclude zero

    # -- helpers --
    def mk(self, market: str) -> pd.DataFrame:
        return self.df[self.df["market"] == market]

    def tally(self, res: tuple | None) -> None:
        if res is not None:
            self.n_ci += 1
            if res[1] > 0 or res[2] < 0:
                self.n_excl += 1

    def rho(self, label: str, g: pd.DataFrame, x: str, y: str) -> tuple[int, tuple | None]:
        """Spearman rho of x with y on rows where both are known. Cached, so one test is counted once."""
        key = (label, x, y)
        if key not in self.rho_cache:
            pair = g[[x, y]].dropna()
            res = _boot_rho(pair[x].to_numpy(float), pair[y].to_numpy(float), self.rng) if len(pair) >= MIN_N else None
            self.tally(res)
            self.rho_cache[key] = (len(pair), res)
        return self.rho_cache[key]

    def head(self, text: str) -> None:
        self.body.append(f"\n## {text}\n")

    def sub(self, text: str) -> None:
        self.body.append(f"\n### {text}\n")

    def add(self, text: str = "") -> None:
        self.body.append(text)

    def table(self, headers: list[str], rows: list[list[object]]) -> None:
        self.add(_table(headers, rows))
        self.add()

    # -- 1 data --
    def section_data(self) -> None:
        df = self.df
        self.head("1. Data used")
        rows = []
        for m in MARKETS:
            g = self.mk(m)
            yrs = g["year"].dropna()
            span = f"{int(yrs.min())}-{int(yrs.max())}" if len(yrs) else "no rows"
            rows.append([m, len(g), span, int(g[OPEN].notna().sum()), int(g[CLOSE].notna().sum()),
                         int(g["gmp"].notna().sum()), int(g["sub_qib"].notna().sum()),
                         int(g["sub_retail"].notna().sum()), int(g["sub_total"].notna().sum()),
                         int(g["vix_level"].notna().sum()), int(g["px_join"].sum())])
        self.add("Rows per market. Counts are non-null values; NaN is missing. `+px_join` marks IPOs whose prices came "
                 "from a joined price file, a selected subset (docs/data_sources/global.md).\n")
        self.table(["market", "IPOs", "listing years", "open gain", "close gain", "GMP", "QIB", "retail", "total sub",
                    "VIX context", "+px_join"], rows)
        yrows = []
        for y in sorted(int(v) for v in df["year"].dropna().unique()):
            g = df[df["year"] == y]
            yrows.append([y] + [int((g["market"] == m).sum()) for m in MARKETS])
        self.add("Rows per listing year and market:\n")
        self.table(["listing year", *MARKETS], yrows)
        crow = []
        for c in COVERAGE_FIELDS:
            crow.append([c] + [_pct(self.mk(m)[c].notna().mean()) if len(self.mk(m)) else "n/a" for m in MARKETS])
        self.add("Non-null share of each field by market (denominator: that market's rows):\n")
        self.table(["field", *MARKETS], crow)
        for m in INDIA:
            g = self.mk(m)
            cy = []
            for y in sorted(int(v) for v in g["year"].dropna().unique()):
                s = g[g["year"] == y]
                cy.append([y, len(s)] + [_pct(s[c].notna().mean()) for c in
                                         ("sub_qib", "sub_nii", "sub_snii", "sub_retail", "sub_total", "gmp")])
            self.add(f"Category coverage by listing year, {m} (share of that year's rows):\n")
            self.table(["year", "n", "QIB", "NII", "sNII", "retail", "total", "GMP"], cy)

        sme = self.mk("IN_SME")
        self.facts["sme_qib_2023"] = self._share(sme[sme["year"] == 2023], "sub_qib")
        self.facts["sme_qib_late"] = self._share(sme[sme["year"] >= 2024], "sub_qib")
        gmp_rows = df[df["gmp"].notna()]
        self.facts["gmp_years"] = (int(gmp_rows["year"].min()), int(gmp_rows["year"].max())) if len(gmp_rows) else None
        self.facts["gmp_n"] = int(len(gmp_rows))
        us_years = sorted(int(v) for v in self.mk("US")["year"].dropna().unique())
        self.facts["us_gap_years"] = [y for y in range(us_years[0], us_years[-1] + 1) if y not in us_years] \
            if us_years else []
        ctx = df[df["vix_level"].notna()]
        self.facts["vix_last_year"] = int(ctx["year"].max()) if len(ctx) else None
        zero_total = sum(sum(z.values()) for z in self.meta["zeroed"].values())
        self.facts["zero_total"] = zero_total
        self.facts["us_px"] = (int(self.mk("US")["px_join"].sum()), len(self.mk("US")))
        self.facts["hk_px"] = (int(self.mk("HK")["px_join"].sum()), len(self.mk("HK")))

        absent = self.meta["absent"]
        self.add("**Field problems the reader needs to know about**\n")
        self.add(f"- Schema columns missing from this table: {', '.join(absent) if absent else 'none'}. A missing column "
                 "is read as all-NaN, so the sections that need it print 'n<20' or 'no rows'.")
        zrows = []
        for c in [OPEN, CLOSE, *SUB_COLS]:
            z = self.meta["zeroed"].get(c, {})
            if z:
                zrows.append([c] + [z.get(m, 0) for m in MARKETS] + [sum(z.values())])
        self.add("- Exact zeros were set to missing. A 0.00% listing gain or a 0x multiple is a missing code here: "
                 "open gains pile up at exactly 0 (US: 138 at 0.00%, 34 in (0, 1], 10 in [-1, 0)), and some rows show "
                 "a retail multiple of 0 next to a total of 8.6x. The build should write NaN; this report does not "
                 "depend on it. Counts set to missing:\n")
        if zrows:
            self.table(["field", *MARKETS, "total"], zrows)
        else:
            self.add("None were found.\n")
        self.add(f"- Survivorship: {self.facts['us_px'][0]} of {self.facts['us_px'][1]} US rows and "
                 f"{self.facts['hk_px'][0]} of {self.facts['hk_px'][1]} HK rows are joined price-file IPOs, a selected "
                 "subset. Every estimate describes IPOs that listed, so it is optimistic for the average issue.")
        self.add(f"- SME category subscription: QIB is present for {_pct(self.facts['sme_qib_2023'])} of 2023 SME "
                 f"listings and {_pct(self.facts['sme_qib_late'])} of 2024-2026 listings. SME category results rest on "
                 "the pre-2024 listings, mostly 2023.")
        g0, g1 = self.facts["gmp_years"] if self.facts["gmp_years"] else ("n/a", "n/a")
        self.add(f"- GMP: {self.facts['gmp_n']} India rows with a reading, listed {g0}-{g1}. One untimed "
                 "grey-market reading per IPO (no open-day timestamp), taken before listing. None for US, HK or OTHER.")
        self.add(f"- Market context (`idx_*`, `vix_level`) exists only for IN_MAIN and IN_SME. US, HK and OTHER have "
                 "no open date, so as-of context cannot be computed. VIX context ends with the "
                 f"{self.facts['vix_last_year']} listings.")
        self.add("- HK `sub_retail` is the public-offer multiple. HK has no QIB or NII field, and US has no "
                 "subscription field at all.")
        self.add("- Intervals treat IPOs as independent. IPOs that list on the same day share market conditions, so "
                 "the intervals are somewhat too narrow.\n")

    @staticmethod
    def _share(g: pd.DataFrame, col: str) -> float | None:
        return float(g[col].notna().mean()) if len(g) else None

    # -- 2 listing outcomes --
    def section_outcomes(self) -> None:
        self.head("2. Listing outcomes")
        self.add("Gain against the issue price, in percent. 'Above issue' means a gain above 0. 'Loss > 10%' means a "
                 "fall of more than 10%. Medians and percentiles; nothing is winsorized.\n")
        for col, name in ((OPEN, "open"), (CLOSE, "close")):
            rows, dists = [], {}
            for m in MARKETS:
                y = self.mk(m)[col].dropna().to_numpy(float)
                if len(y) < MIN_N:
                    rows.append([m, len(y), "n<20", "n<20", "n<20", "n<20", "n<20"])
                    continue
                d = {"n": len(y), "above": float((y > 0).mean()), "med": float(np.median(y)),
                     "p10": float(np.quantile(y, 0.1)), "p90": float(np.quantile(y, 0.9)),
                     "loss": float((y < -10).mean())}
                dists[m] = d
                rows.append([m, d["n"], _bar(d["above"]), _f(d["med"], 1, True), _f(d["p10"], 1, True),
                             _f(d["p90"], 1, True), _pct(d["loss"])])
            self.facts[f"dist_{name}"] = dists
            self.add(f"**Listing-day {name} gain by segment**\n")
            self.table(["segment", "n", "above issue", "median", "p10", "p90", "loss > 10%"], rows)
        for m in MARKETS:
            g = self.mk(m)
            rows = []
            for y in sorted(int(v) for v in g["year"].dropna().unique()):
                s = g[g["year"] == y]
                yo = s[OPEN].dropna().to_numpy(float)
                yc = s[CLOSE].dropna().to_numpy(float)
                loss = "n<20" if len(yc) < MIN_N else _pct((yc < -10).mean())
                rows.append([y, len(yo), *_pair(yo), len(yc), *_pair(yc), loss])
            if rows:
                self.add(f"**{m} by listing year**\n")
                self.table(["year", "n open", "open above issue", "open median", "n close", "close above issue",
                            "close median", "close loss > 10%"], rows)

    # -- 3 subscription --
    def section_subscription(self) -> None:
        self.head("3. Subscription")
        self.add("Spearman rank correlation of each subscription multiple with the listing-day gain, with a 95% bootstrap "
                 "interval over IPOs. Ranks are used because multiples are very skewed. The reading is on the close: "
                 "'not distinguishable from noise' when the interval spans zero; 'strong' when |rho| >= 0.3 and the "
                 "interval excludes zero; otherwise 'weak'.\n")
        rows = []
        targets = [(m, cat, lab) for m in INDIA for cat, lab in CATEGORIES]
        targets.append(("HK", "sub_retail", "public-offer (retail)"))
        for m, cat, lab in targets:
            g = self.mk(m)
            if g[cat].notna().sum() == 0:
                rows.append([m, lab, 0, "n<20", 0, "n<20", "no data"])
                continue
            no, ro = self.rho(m, g, cat, OPEN)
            nc, rc = self.rho(m, g, cat, CLOSE)
            rows.append([m, lab, no, _ci(ro), nc, _ci(rc), _grade(rc)])
            self.facts.setdefault("rho_total", {})[m] = (nc, rc) if cat == "sub_total" else None
            if cat == "sub_retail" and m == "HK":
                self.facts["hk_rho"] = (nc, rc)
        self.table(["segment", "multiple", "n open", "rho open [95% CI]", "n close", "rho close [95% CI]",
                    "reading (close)"], rows)
        self.facts["rho_total"] = {m: self.rho(m, self.mk(m), "sub_total", CLOSE) for m in INDIA}
        self.section_controls()
        self.section_quotas()
        self.section_divergence()

    def section_controls(self) -> None:
        self.sub("3.1 Which category carries the information once total is held fixed")
        self.add("Regressions of the gain on log multiples (natural log: ln 10 = 2.30 is a tenfold rise). Model A: total "
                 "only. Model B: QIB, NII and retail, no total. Model C: total plus QIB, NII and retail. All three use "
                 "one complete-case sample, so their R-squared values compare. Standard errors are HC3. The fit is OLS on "
                 "raw gains, so a few very large gains pull the coefficients. The rank correlations above are the robust "
                 "check.\n")
        coef_rows, r2_rows = [], []
        for m in INDIA:
            g = self.mk(m)
            for outcome, oname in ((CLOSE, "close"), (OPEN, "open")):
                xs = ["ln_total", "ln_qib", "ln_nii", "ln_retail"]
                cc = g[[outcome, *xs]].dropna()
                a = _regress(cc, outcome, ["ln_total"])
                b = _regress(cc, outcome, ["ln_qib", "ln_nii", "ln_retail"])
                c = _regress(cc, outcome, xs)
                r2_rows.append([m, oname, len(cc)] + [_f(r["r2"], 3) if r else "n<20" for r in (a, b, c)])
                if c is None:
                    continue
                self.facts.setdefault("reg", {})[(m, oname)] = {"c": c, "r2a": a["r2"] if a else None,
                                                               "r2b": b["r2"] if b else None}
                for j, name in enumerate(c["names"]):
                    if name == "const":
                        continue
                    lo, hi = float(c["lo"][j]), float(c["hi"][j])
                    self.tally((float(c["beta"][j]), lo, hi))
                    coef_rows.append([m, oname, name, _f(c["beta"][j], 2, True), f"[{lo:+.2f}, {hi:+.2f}]",
                                      "CI excludes 0" if (lo > 0 or hi < 0) else "CI spans 0"])
        self.table(["segment", "outcome", "n (complete cases)", "R2 model A (total)", "R2 model B (categories)",
                    "R2 model C (both)"], r2_rows)
        if coef_rows:
            self.add("Model C coefficients (pp of gain per unit of ln multiple):\n")
            self.table(["segment", "outcome", "variable", "coef", "95% CI (HC3)", "reading"], coef_rows)
        self.add("A coefficient is what that category adds once total is held fixed. Categories are parts of total, so "
                 "the coefficients describe the mix of demand at a given total, not a separate cause. The data are "
                 "observational.\n")

    def section_quotas(self) -> None:
        self.sub("3.2 Employee and shareholder quotas")
        g = self.mk("IN_MAIN")
        no, ro = self.rho("IN_MAIN", g, "sub_employee", OPEN)
        nc, rc = self.rho("IN_MAIN", g, "sub_employee", CLOSE)
        self.facts["emp"] = (nc, rc)
        self.facts["shareholder_n"] = int(g["sub_shareholder"].notna().sum())
        self.facts["sme_emp_n"] = int(self.mk("IN_SME")["sub_employee"].notna().sum())
        self.table(["category", "n open", "rho open [95% CI]", "n close", "rho close [95% CI]", "reading (close)"], [
            ["IN_MAIN employee", no, _ci(ro), nc, _ci(rc), _grade(rc)],
            ["IN_MAIN shareholder", self.facts["shareholder_n"], "n<20", "n<20", "n<20", "not tested"],
            ["IN_SME employee", self.facts["sme_emp_n"], "n<20", "n<20", "n<20", "not tested"],
        ])
        has = g["sub_employee"].notna()
        rows = []
        for label, sel in (("employee recorded", has), ("employee not recorded", ~has)):
            c = g.loc[sel, CLOSE].dropna().to_numpy(float)
            rows.append([label, len(c), *_pair(c)])
        self.table(["IN_MAIN", "n close", "close median", "close above issue"], rows)
        pre, post = g[g["year"] < 2021], g[g["year"] >= 2021]
        self.add(f"The recorded and unrecorded groups differ by era: employee multiples are recorded for "
                 f"{_pct(pre['sub_employee'].notna().mean())} of IN_MAIN listings before 2021 and "
                 f"{_pct(post['sub_employee'].notna().mean())} from 2021, so these rows mix quota design with the year. "
                 f"Shareholder quotas are recorded for {self.facts['shareholder_n']} IPOs, too few to test.\n")

    def section_divergence(self) -> None:
        self.sub("3.3 Retail against QIB: divergence flag")
        self.add("Each IPO falls in one group. The cut-offs (3x and 1x) were fixed before the outcomes were looked at. "
                 "Rows without both a QIB and a retail multiple are left out. These are descriptive medians, not "
                 "tests.\n")
        retail_hi, qib_lo = self._cond("sub_retail", ">=", 3), self._cond("sub_qib", "<", 1)
        qib_hi, retail_lo = self._cond("sub_qib", ">=", 3), self._cond("sub_retail", "<", 1)
        both_hi = retail_hi & (self.df["sub_qib"] >= 3)
        groups = [
            ("A: retail >= 3x and QIB < 1x (flag)", retail_hi & qib_lo),
            ("B: QIB >= 3x and retail < 1x", qib_hi & retail_lo),
            ("C: both >= 3x", both_hi),
        ]
        for m in INDIA:
            g = self.mk(m)
            known = g["sub_qib"].notna() & g["sub_retail"].notna()
            rows = []
            taken = pd.Series(False, index=g.index)
            for label, mask in groups:
                sel = mask.loc[g.index] & known.loc[g.index]
                taken |= sel
                yo = g.loc[sel, OPEN].dropna().to_numpy(float)
                yc = g.loc[sel, CLOSE].dropna().to_numpy(float)
                rows.append([label, len(yo), *_pair(yo), len(yc), *_pair(yc)])
                self.facts.setdefault("div", {})[(m, label[0])] = yc
            rest = known.loc[g.index] & ~taken
            yo = g.loc[rest, OPEN].dropna().to_numpy(float)
            yc = g.loc[rest, CLOSE].dropna().to_numpy(float)
            rows.append(["D: everything else with both known", len(yo), *_pair(yo), len(yc), *_pair(yc)])
            self.facts.setdefault("div", {})[(m, "D")] = yc
            self.add(f"**{m}** (rows with both QIB and retail: {int(known.sum())})\n")
            self.table(["group", "n open", "open median", "open above issue", "n close", "close median",
                        "close above issue"], rows)

    @staticmethod
    def _cond(col: str, op: str, value: float):
        def mask(df: pd.DataFrame) -> pd.Series:  # placeholder replaced below
            return pd.Series(False, index=df.index)
        return op, col, value

    # -- 4 GMP --
    def section_gmp(self) -> None:
        self.head("4. GMP")
        self.add("GMP is an unofficial grey-market percentage of the issue price, one untimed reading per IPO. The error "
                 "is GMP minus the open gain, so a positive bias means GMP was too high on average. 'MAE constant' is "
                 "the error of predicting the segment's median open gain (in-sample, all the segment's rows): the bar "
                 "that GMP has to clear.\n")
        rows = []
        for label, segs in (("IN_MAIN", ["IN_MAIN"]), ("IN_SME", ["IN_SME"]), ("India pooled", list(INDIA))):
            g = self.df[self.df["market"].isin(segs)]
            base = g[g["gmp"].notna() & g[OPEN].notna()]
            seg_open = g[OPEN].dropna().to_numpy(float)
            if len(base) < MIN_N or len(seg_open) < MIN_N:
                rows.append([label, len(base), "n<20", "n<20", "n<20", "n<20", "not tested"])
                continue
            err = base["gmp"].to_numpy(float) - base[OPEN].to_numpy(float)
            mae = float(np.abs(err).mean())
            const = float(np.abs(base[OPEN].to_numpy(float) - np.median(seg_open)).mean())
            _, res = self.rho(label, base, "gmp", OPEN)
            rows.append([label, len(base), _f(mae), _f(const), _f(err.mean(), 1, True), _ci(res), _grade(res)])
            self.facts.setdefault("gmp", {})[label] = {"n": len(base), "mae": mae, "const": const,
                                                       "bias": float(err.mean()), "res": res}
        self.table(["segment", "n", "MAE GMP (pp)", "MAE constant (pp)", "bias: mean error (pp)",
                    "rho GMP vs open [95% CI]", "reading"], rows)
        self.add("**Bucket table, India pooled.** Descriptive. Cells with fewer than 20 rows print 'n<20'.\n")
        ind = self.df[self.df["market"].isin(INDIA)]
        edges = [-np.inf, 0, 10, 30, 60, 100, np.inf]
        labels = ["below 0", "0 to 10", "10 to 30", "30 to 60", "60 to 100", "100 and above"]
        cut = pd.cut(ind["gmp"], bins=edges, labels=labels, right=False)
        brows = []
        for lab in labels:
            s = ind[cut == lab]
            yo = s[OPEN].dropna().to_numpy(float)
            err = (s["gmp"] - s[OPEN]).dropna().to_numpy(float)
            brows.append([lab, len(yo), *_pair(yo), len(err), _f(np.median(err), 1, True) if len(err) >= MIN_N else "n<20"])
        self.table(["GMP %", "n open", "open median", "open above issue", "n", "median error (pp)"], brows)
        self.add("**By listing year.** GMP readings exist only for 2020-2024 listings.\n")
        yrows = []
        for y in GMP_YEARS:
            row: list[object] = [y]
            for m in INDIA:
                g = self.mk(m)
                g = g[(g["year"] == y) & g["gmp"].notna() & g[OPEN].notna()]
                if len(g) < MIN_N:
                    row += [len(g), "n<20", "n<20"]
                    continue
                err = g["gmp"].to_numpy(float) - g[OPEN].to_numpy(float)
                _, res = self.rho(f"{m}-{y}", g, "gmp", OPEN)
                row += [len(g), _f(np.abs(err).mean()), _ci(res)]
            yrows.append(row)
        self.table(["year", "IN_MAIN n", "MAE (pp)", "rho [95% CI]", "IN_SME n", "MAE (pp)", "rho [95% CI]"], yrows)
        self.add("**Largest misses, India.** The ten IPOs with the biggest absolute open-gain error.\n")
        base = ind[ind["gmp"].notna() & ind[OPEN].notna()].assign(err=lambda d: d["gmp"] - d[OPEN])
        top = base.reindex(base["err"].abs().sort_values(ascending=False).index).head(10)
        trows = [[str(r["company"])[:34], pd.Timestamp(r["listing_date"]).date() if pd.notna(r["listing_date"]) else "n/a",
                  r["market"], _f(r["gmp"], 1, True), _f(r[OPEN], 1, True), _f(r["err"], 1, True)]
                 for _, r in top.iterrows()]
        self.table(["company", "listed", "segment", "GMP %", "open gain %", "error (pp)"], trows)
        self.section_gmp_nested()

    def section_gmp_nested(self) -> None:
        self.sub("4.1 GMP against subscription (nested)")
        self.add("Same rows for every model in a row: a GMP reading and the total subscription multiple. R-squared is the "
                 "share of the variance in the gain that a model explains. 'Adding' is the change in R-squared from "
                 "adding the variable to the other one, with a 95% bootstrap interval over IPOs.\n")
        rows = []
        self.facts["nested"] = {}
        for label, segs in (("IN_MAIN", ["IN_MAIN"]), ("IN_SME", ["IN_SME"]), ("India pooled", list(INDIA))):
            g = self.df[self.df["market"].isin(segs)]
            for outcome, oname in ((OPEN, "open"), (CLOSE, "close")):
                cc = g[[outcome, "gmp", "ln_total"]].dropna()
                n = len(cc)
                if n < MIN_N:
                    rows.append([label, oname, n, "n<20", "n<20", "n<20", "n<20", "n<20"])
                    continue
                y = cc[outcome].to_numpy(float)
                gm = cc["gmp"].to_numpy(float)
                lt = cc["ln_total"].to_numpy(float)
                one = np.ones(n)
                r_g = _r2(y, np.column_stack([one, gm]))
                r_t = _r2(y, np.column_stack([one, lt]))
                r_b = _r2(y, np.column_stack([one, gm, lt]))
                add_total, add_gmp = [], []
                for _ in range(BOOT):
                    i = self.rng.integers(0, n, n)
                    yi, gi, li = y[i], gm[i], lt[i]
                    one_i = np.ones(n)
                    both = _r2(yi, np.column_stack([one_i, gi, li]))
                    add_total.append(both - _r2(yi, np.column_stack([one_i, gi])))
                    add_gmp.append(both - _r2(yi, np.column_stack([one_i, li])))
                ci_t = np.nanquantile(add_total, [0.025, 0.975])
                ci_g = np.nanquantile(add_gmp, [0.025, 0.975])
                self.n_ci += 2
                self.n_excl += int(ci_t[0] > 0 or ci_t[1] < 0) + int(ci_g[0] > 0 or ci_g[1] < 0)
                rows.append([label, oname, n, _f(r_g, 3), _f(r_t, 3), _f(r_b, 3),
                             f"{r_b - r_t:+.3f} [{ci_g[0]:+.3f}, {ci_g[1]:+.3f}]",
                             f"{r_b - r_g:+.3f} [{ci_t[0]:+.3f}, {ci_t[1]:+.3f}]"])
                self.facts["nested"][(label, oname)] = {
                    "n": n, "r_g": r_g, "r_t": r_t, "r_b": r_b,
                    "add_gmp": (r_b - r_t, ci_g), "add_total": (r_b - r_g, ci_t)}
        self.table(["segment", "outcome", "n", "R2 GMP only", "R2 total only", "R2 both", "adding GMP to total",
                    "adding total to GMP"], rows)

    # -- 5 market conditions --
    def section_market(self) -> None:
        self.head("5. Market conditions")
        self.add("Index context is measured as of the day before the open: the benchmark's trailing 5- and 20-day "
                 "returns in percent, and the volatility index close. The table has it for India only (NIFTY 50 and "
                 "India VIX). US, HK and OTHER are n<20 by construction (no open date).\n")
        rows = []
        for m in INDIA:
            g = self.mk(m)
            for var, label in (("idx_ret_20d", "NIFTY 50, 20-day return"), ("idx_ret_5d", "NIFTY 50, 5-day return"),
                               ("vix_level", "India VIX level")):
                no, ro = self.rho(m, g, var, OPEN)
                nc, rc = self.rho(m, g, var, CLOSE)
                rows.append([m, label, nc, _ci(ro), _ci(rc), _grade(rc)])
                self.facts.setdefault("mkt", {})[(m, var)] = rc
        for m in ("US", "HK", "OTHER"):
            rows.append([m, "all context fields", 0, "n<20", "n<20", "no context in this table"])
        self.table(["segment", "condition", "n close", "rho open [95% CI]", "rho close [95% CI]", "reading (close)"], rows)
        for m in INDIA:
            g = self.mk(m)
            cut_r = pd.cut(g["idx_ret_20d"], bins=[-np.inf, -5, 0, 5, np.inf],
                           labels=["below -5%", "-5% to 0%", "0% to +5%", "+5% and above"], right=False)
            cut_v = pd.cut(g["vix_level"], bins=[0, 15, 20, 25, np.inf],
                           labels=["below 15", "15 to 20", "20 to 25", "25 and above"], right=False)
            for name, cut, labels in (("NIFTY 50 20-day return", cut_r, ["below -5%", "-5% to 0%", "0% to +5%", "+5% and above"]),
                                      ("India VIX", cut_v, ["below 15", "15 to 20", "20 to 25", "25 and above"])):
                self.add(f"**{m} by {name}**\n")
                self.table(["bucket", "n open", "open median", "open above issue", "n close", "close median",
                            "close above issue"], _bucket_rows(g, cut, labels))

    # -- 6 issue characteristics --
    def section_issue(self) -> None:
        self.head("6. Issue characteristics")
        self.add("Issue size is in millions of the listing currency (INR m for India, USD m for US, HKD m for HK). "
                 "Buckets are quartiles of each market's own distribution, so the edges are printed.\n")
        rows = []
        specs = [("issue_size_m", "issue size", ("IN_MAIN", "IN_SME", "US", "HK")),
                 ("price_high", "price band top (INR)", INDIA),
                 ("ofs_pct", "offer-for-sale share (%)", INDIA),
                 ("fresh_issue_pct", "fresh-issue share (%)", INDIA)]
        for var, label, segs in specs:
            for m in segs:
                g = self.mk(m)
                no, ro = self.rho(m, g, var, OPEN)
                nc, rc = self.rho(m, g, var, CLOSE)
                rows.append([label, m, nc, _ci(ro), _ci(rc), _grade(rc)])
                self.facts.setdefault("chars", {})[(var, m)] = rc
        self.table(["characteristic", "segment", "n close", "rho open [95% CI]", "rho close [95% CI]", "reading (close)"],
                   rows)
        for m in ("IN_MAIN", "IN_SME", "US", "HK"):
            g = self.mk(m)
            cut, labels = _cut_quantiles(g["issue_size_m"])
            if labels:
                self.add(f"**{m} by issue size quartile**\n")
                self.table(["size (m)", "n open", "open median", "open above issue", "n close", "close median",
                            "close above issue"], _bucket_rows(g, cut, labels))
        for m in INDIA:
            g = self.mk(m)
            cut, labels = _cut_quantiles(g["price_high"])
            if labels:
                self.add(f"**{m} by price band top (quartiles, INR)**\n")
                self.table(["price band top", "n open", "open median", "open above issue", "n close", "close median",
                            "close above issue"], _bucket_rows(g, cut, labels))
            cut = pd.cut(g["ofs_pct"], bins=[0, 25, 50, 75, 100.0001], labels=["below 25%", "25% to 50%", "50% to 75%",
                                                                              "75% and above"], right=False)
            self.add(f"**{m} by offer-for-sale share**\n")
            self.table(["OFS share", "n open", "open median", "open above issue", "n close", "close median",
                        "close above issue"], _bucket_rows(g, cut, ["below 25%", "25% to 50%", "50% to 75%",
                                                                    "75% and above"]))
            cut = pd.cut(g["fresh_issue_pct"], bins=[0, 50, 100, np.inf], labels=["below 50%", "50% to 99.9%", "100%"],
                         right=False)
            self.add(f"**{m} by fresh-issue share**\n")
            self.table(["fresh share", "n open", "open median", "open above issue", "n close", "close median",
                        "close above issue"], _bucket_rows(g, cut, ["below 50%", "50% to 99.9%", "100%"]))

    # -- 7 cross-market --
    def section_cross(self) -> None:
        self.head("7. Cross-market comparison")
        self.add("Does 'higher subscription, higher gain' hold outside India? Totals are compared in India. HK has only "
                 "the public-offer multiple, which is compared with India's retail multiple. Each cell's medians are "
                 "for the lowest multiple (under 3x) and the highest (10x and above).\n")
        specs = [("IN_MAIN", "sub_total", "total"), ("IN_SME", "sub_total", "total"),
                 ("IN_MAIN", "sub_retail", "retail"), ("IN_SME", "sub_retail", "retail"),
                 ("HK", "sub_retail", "public-offer (retail)")]
        rows = []
        for m, field, lab in specs:
            g = self.mk(m)
            no, ro = self.rho(m, g, field, OPEN)
            nc, rc = self.rho(m, g, field, CLOSE)
            low = g.loc[g[field] < 3, CLOSE].dropna().to_numpy(float)
            high = g.loc[g[field] >= 10, CLOSE].dropna().to_numpy(float)
            rows.append([m, lab, nc, _ci(rc), _cell_med(low), _cell_med(high), _grade(rc)])
            self.facts.setdefault("cross", {})[(m, field)] = rc
        rows.append(["US", "none (no subscription field)", 0, "n/a", "n/a", "n/a", "cannot be tested"])
        rows.append(["OTHER", "none (no subscription field)", 0, "n/a", "n/a", "n/a", "cannot be tested"])
        self.table(["segment", "multiple", "n close", "rho close [95% CI]", "close median, under 3x",
                    "close median, 10x and above", "reading"], rows)
        hk = self.facts["cross"].get(("HK", "sub_retail"))
        india_retail = self.facts["cross"].get(("IN_MAIN", "sub_retail"))
        self.facts["hk_vs_india"] = (hk, india_retail)
        self.add("HK is a selected subset: every HK row is a joined price-file IPO, and the HK sample lacks later years and "
                 "delisted names (docs/data_sources/global.md). The US has no subscription data, so the pattern cannot be "
                 "tested there. Section 2 gives its gains only.\n")

    # -- 8 practical --
    def section_practical(self) -> None:
        self.head("8. Practical reading for the live tool")
        self.add("Each row is one signal, read at the close of the issue, with what followed historically. Descriptive "
                 "rules, one signal at a time: nothing here combines signals or forecasts a specific IPO. Intervals are "
                 "95% bootstrap intervals for the share above issue.\n")
        rows = []
        rng_rows = []
        for m in INDIA:
            g = self.mk(m)
            t = g["sub_total"]
            rules = [("total subscription below 3x", t < 3), ("total subscription 3x to 10x", (t >= 3) & (t < 10)),
                     ("total subscription 10x and above", t >= 10)]
            for label, mask in rules:
                rng_rows.append((m, label, g[mask.fillna(False)]))
            a = self.mk(m)
            ret = (a["sub_retail"] >= 3) & (a["sub_qib"] < 1)
            rng_rows.append((m, "retail >= 3x and QIB < 1x", a[ret.fillna(False)]))
            vix = a["vix_level"] >= 20
            rng_rows.append((m, "India VIX 20 or higher", a[vix.fillna(False)]))
        india = self.df[self.df["market"].isin(INDIA)]
        g_pool = india["gmp"]
        for label, mask in (("GMP below 0%", g_pool < 0), ("GMP 0% to 30%", (g_pool >= 0) & (g_pool < 30)),
                            ("GMP 30% and above", g_pool >= 30)):
            rng_rows.append(("India", label, india[mask.fillna(False)]))
        for m, label, s in rng_rows:
            yc = s[CLOSE].dropna().to_numpy(float)
            yo = s[OPEN].dropna().to_numpy(float)
            if len(yc) < MIN_N:
                rows.append([label, m, len(yc), "n<20", "n<20", "n<20", "n<20"])
                continue
            share = _boot_stat((yc > 0).astype(float), np.mean, self.rng)
            p25, p75 = np.quantile(yc, [0.25, 0.75])
            rows.append([label, m, len(yc), f"{share[0] * 100:.0f}% [{share[1] * 100:.0f}, {share[2] * 100:.0f}]",
                         f"{np.median(yc):+.1f} ({p25:+.1f} to {p75:+.1f})",
                         _f(np.median(yo), 1, True) if len(yo) >= MIN_N else "n<20",
                         "descriptive"])
        self.table(["if at close you see", "segment", "n close", "close above issue [95% CI]",
                    "close median (p25 to p75)", "open median", "reading"], rows)
        self.sub("What the data cannot support")
        gy = self.facts["gmp_years"]
        bullets = [
            f"- GMP: no reading after {gy[1] if gy else 'n/a'}. Nothing can be checked for 2025-2026 listings, and "
            "GMP is untimed, so it cannot be compared with the open at any time of day.",
            f"- US: no listings in {', '.join(str(y) for y in self.facts['us_gap_years']) or 'none of the gap years'} "
            "in this table, so the US pattern is not known for those years.",
            "- No live US or HK source exists in the tool (`src/ipo_radar/live` covers India only). None of the "
            "rules above can be applied to a US or HK IPO at serving time.",
            "- No as-of index context for US, HK or OTHER: their rows have no open date.",
            "- HK subscription is the public-offer multiple only. US has no subscription field at all, and OTHER "
            "(Indonesia) has only close gains with no issue size.",
            "- sNII and bNII exist from 2022 on; the SME category fields are mostly pre-2024 (section 1).",
            f"- Survivorship: the joined rows ({self.facts['us_px'][0]} US, {self.facts['hk_px'][0]} HK) are a selected "
            "subset, so every rate here is optimistic for the average issue.",
            "- First-day outcomes only. Gains after the first day, withdrawn issues and IPOs that never listed are "
            "not in the table, so the rates describe listed IPOs.",
            "- Observational. A rule that held historically can change with rules on allotment, anchor books or "
            "market conditions no longer seen in this data.",
            "- Several estimates are reported with no correction for multiple testing (section 'Method'). A "
            "'CI excludes zero' result is one of many and should be read with that in mind.",
        ]
        self.add("\n".join(bullets))
        self.add()

    # -- appendix: bucket tables --
    def appendix(self) -> None:
        self.head("Appendix A. Subscription bucket tables (descriptive)")
        self.add("Each table bins one multiple into the buckets shown. Cells with fewer than 20 rows print 'n<20'. "
                 "The rates are shares of IPOs with the gain above issue; bars are the same share out of ten.\n")
        for m in INDIA + ("HK",):
            cats = [("sub_retail", "public-offer (retail)")] if m == "HK" else CATEGORIES
            g = self.mk(m)
            for cat, lab in cats:
                if g[cat].notna().sum() == 0:
                    continue
                cut = pd.cut(g[cat], bins=MULT_EDGES, labels=MULT_LABELS, right=False)
                self.add(f"**{m}: {lab}**\n")
                self.table(["bucket", "n open", "open median", "open above issue", "n close", "close median",
                            "close above issue"], _bucket_rows(g, cut, MULT_LABELS))

    # -- summary --
    def summary(self) -> list[str]:
        f = self.facts
        dist_o, dist_c = f.get("dist_open", {}), f.get("dist_close", {})

        def med(d: dict, m: str) -> str:
            return _f(d[m]["med"], 1, True) + f" (n {d[m]['n']})" if m in d else "n<20"

        def above(d: dict, m: str) -> str:
            return _pct(d[m]["above"]) if m in d else "n<20"

        rt = f["rho_total"]
        t_in = [rt.get(m, (0, None)) for m in INDIA]
        signs = {np.sign(r[1][0]) for r in t_in if r[1] is not None}
        direction = ("both segments point the same way: " + _direction(t_in[0][1][0] if t_in[0][1] else None)
                     if len(signs) == 1 else "the two segments disagree in sign")
        reg = f.get("reg", {})
        model = reg.get(("IN_MAIN", "close"))
        if model:
            c = model["c"]
            clear = [c["names"][j] for j in range(1, len(c["names"]))
                     if c["lo"][j] > 0 or c["hi"][j] < 0]
            holding = (f"holding total fixed, IN_MAIN close: " + (", ".join(clear) + " has a CI clear of zero"
                                                                     if clear else "no category coefficient has a CI clear of zero")
                       + f" (model C R2 {c['r2']:.2f} against {model['r2a']:.2f} for total alone, n {c['n']})")
        else:
            holding = "holding total fixed: n<20, not tested"
        div = f.get("div", {})
        a_med = _cell_med(div.get(("IN_MAIN", "A"), np.array([])))
        d_med = _cell_med(div.get(("IN_MAIN", "D"), np.array([])))
        emp_n, emp_r = f.get("emp", (0, None))
        gm = f.get("gmp", {}).get("India pooled")
        nest = f.get("nested", {}).get(("India pooled", "open"))
        mk = f.get("mkt", {})
        chars = f.get("chars", {})
        cross = f.get("cross", {})
        hk = f.get("hk_rho")
        hk_txt = (f"HK public-offer multiple rho {_ci(hk[1])} (n {hk[0]}), {_grade(hk[1])}"
                  if hk and hk[1] is not None else "HK public-offer multiple: n<20")
        india_sign = t_in[0][1][0] if t_in and t_in[0][1] else None
        hk_sign = hk[1][0] if hk and hk[1] is not None else None
        same = ("the same sign as India" if (hk_sign is not None and india_sign is not None
                                             and np.sign(hk_sign) == np.sign(india_sign))
                else "not the same sign as India" if hk_sign is not None and india_sign is not None
                else "no comparison possible")
        gmp_line = (f"GMP (India pooled, open, n {gm['n']}): MAE {gm['mae']:.1f} pp against {gm['const']:.1f} pp for "
                    f"predicting the segment median; bias {gm['bias']:+.1f} pp; rho {_ci(gm['res'])}, "
                    f"{_grade(gm['res'])}.") if gm else "GMP: n<20, not tested."
        nest_line = (f"GMP against subscription (India pooled, open, n {nest['n']}): R2 {nest['r_g']:.2f} from GMP, "
                     f"{nest['r_t']:.2f} from total, {nest['r_b']:.2f} from both. Adding GMP to total adds "
                     f"{nest['add_gmp'][0]:+.2f} [{nest['add_gmp'][1][0]:+.2f}, {nest['add_gmp'][1][1]:+.2f}]; adding "
                     f"total to GMP adds {nest['add_total'][0]:+.2f} [{nest['add_total'][1][0]:+.2f}, "
                     f"{nest['add_total'][1][1]:+.2f}]."
                     ) if nest else "GMP against subscription: n<20, not tested."
        gap = f["us_gap_years"]
        return [
            f"- Listing day, median open gain: IN_MAIN {med(dist_o, 'IN_MAIN')}, IN_SME {med(dist_o, 'IN_SME')}, "
            f"US {med(dist_o, 'US')}, HK {med(dist_o, 'HK')}. Above issue at the open: IN_MAIN {above(dist_o, 'IN_MAIN')}, "
            f"IN_SME {above(dist_o, 'IN_SME')}, US {above(dist_o, 'US')}, HK {above(dist_o, 'HK')}. Descriptive.",
            f"- Total subscription against the close gain: IN_MAIN rho {_ci(rt['IN_MAIN'][1])} (n {rt['IN_MAIN'][0]}, "
            f"{_grade(rt['IN_MAIN'][1])}); IN_SME rho {_ci(rt['IN_SME'][1])} (n {rt['IN_SME'][0]}, "
            f"{_grade(rt['IN_SME'][1])}). In both, {direction}.",
            f"- Categories once total is held fixed: {holding}.",
            f"- Retail-QIB flag (retail >= 3x, QIB < 1x), IN_MAIN close median {a_med} against {d_med} for the rest. "
            "Descriptive, not tested.",
            f"- Employee quota (IN_MAIN close): rho {_ci(emp_r)} (n {emp_n}), {_grade(emp_r)}. Shareholder quota: "
            f"n<20 for {f.get('shareholder_n', 0)} IPOs, not testable.",
            gmp_line,
            nest_line,
            f"- Market conditions (close): NIFTY 20-day return rho IN_MAIN {_ci(mk.get(('IN_MAIN', 'idx_ret_20d'), (0, None))[1] if False else None)}"
            f" (see section 5); the India VIX close: {_grade(mk.get(('IN_MAIN', 'vix_level')))} in IN_MAIN and "
            f"{_grade(mk.get(('IN_SME', 'vix_level')))} in IN_SME. US, HK and OTHER have no as-of context in the table.",
            f"- Issue size (close): IN_MAIN {_grade(chars.get(('issue_size_m', 'IN_MAIN')))}, IN_SME "
            f"{_grade(chars.get(('issue_size_m', 'IN_SME')))}, US {_grade(chars.get(('issue_size_m', 'US')))}. "
            f"Rho {_ci(chars.get(('issue_size_m', 'IN_MAIN')))} for IN_MAIN.",
            f"- Cross-market: {hk_txt}, {same}; US has no subscription field. Biggest caveats: survivorship (joined "
            f"rows: {f['us_px'][0]} US, {f['hk_px'][0]} HK), {f['zero_total']} zero values set to missing, SME category "
            f"data mostly pre-2024, GMP untimed and ending {f['gmp_years'][1] if f['gmp_years'] else 'n/a'}, "
            f"{self.n_ci} intervals without multiplicity correction.",
        ]

    # -- assembly --
    def render(self, digest: str, counts: dict[str, int]) -> str:
        head = [
            "# Research: what goes with listing-day gains and losses",
            "",
            f"Source: `data/processed/ipos.parquet`, {sum(counts.values())} IPOs "
            f"({', '.join(f'{m} {n}' for m, n in counts.items())}). Table digest `{digest}`, seed {SEED}. "
            "Written by `ipo research` (`src/ipo_radar/research.py`); a rerun on the same table gives the same file.",
            "",
            "## Summary",
            "",
            *self.summary(),
            "",
            "## Question and method",
            "",
            "Question: which of GMP, subscription (QIB, HNI with sNII and bNII, retail, employee, shareholder, total), "
            "issue size, offer-for-sale share, market conditions, segment and market relate to the listing-day gain and "
            "loss, and how reliably?",
            "",
            "- Gains are percent of the issue price on the first trading day (open and close).",
            "- NaN is missing and is never filled. Exact 0.00% gains and 0x multiples are set to missing (section 1).",
            f"- Intervals: 95% percentile bootstraps over IPOs, {BOOT} resamples, seed {SEED}. Rank correlations are "
            "Spearman. Regressions are OLS with HC3 standard errors and normal 95% intervals.",
            "- Strength words: 'not distinguishable from noise' when the interval spans zero; 'strong' when |rho| >= 0.3 "
            "and the interval excludes zero; otherwise 'weak'. The 0.3 threshold is a convention, not a test.",
            f"- Multiplicity: this report computes {self.n_ci} intervals on associations, coefficients and R-squared "
            f"differences. {self.n_excl} exclude zero. With no real association, roughly "
            f"{round(0.05 * self.n_ci)} would exclude zero by chance (a rough count, since the tests are not independent). "
            "No correction is applied. The bucket and year tables are descriptive: they are read, not tested, and "
            "no cell was chosen after looking at its outcome.",
            "",
        ]
        return "\n".join(head + self.body) + "\n\n" + "\n".join(self._appendix_text) + "\n"

    _appendix_text: list[str] = []


def _cell_med(y: np.ndarray) -> str:
    if len(y) < MIN_N:
        return "n<20"
    return f"{np.median(y):+.1f} (n {len(y)})"


def _bucket_rows(g: pd.DataFrame, cut: pd.Series, labels: list[str]) -> list[list[object]]:
    rows = []
    for lab in labels:
        s = g[cut == lab]
        yo = s[OPEN].dropna().to_numpy(float)
        yc = s[CLOSE].dropna().to_numpy(float)
        rows.append([lab, len(yo), *_pair(yo), len(yc), *_pair(yc)])
    return rows
