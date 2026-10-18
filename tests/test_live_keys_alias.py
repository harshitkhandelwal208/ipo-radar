"""Alias matching in keys.same_company: real name pairs from the sites that must join, and near misses that must not."""
from datetime import date

import pytest

from ipo_radar.live.gmp import _clean_name
from ipo_radar.live.keys import company_key, same_company

pytestmark = pytest.mark.usefixtures("live_isolation")
D15, D20, D21 = date(2026, 10, 15), date(2026, 10, 20), date(2026, 10, 21)


@pytest.mark.parametrize("a, b, kwargs", [
    # 'Sahajanand Medical' (ipowatch, ipocentral) vs 'Sahajanand Medical Technologies' (investorgain, ipopremium)
    ("sahajanand medical", "sahajanand medical technologies",
     dict(open_a=D15, open_b=D15, market_b="IN_MAIN")),
    # 'Zetwerk' (investorgain, ipowatch, ipocentral) vs 'Zetwerk Manufacturing Businesses' (NSE)
    ("zetwerk", "zetwerk manufacturing businesses", dict(open_a=D15, open_b=D15, market_a="IN_MAIN", market_b="IN_MAIN")),
    # 'Zetwerk Manufacturing' (ipopremium) vs NSE's full name
    ("zetwerk manufacturing", "zetwerk manufacturing businesses", dict(open_a=D15, open_b=D15)),
    # Chittorgarh's 'Businesses' vs 'Business' spelling
    ("zetwerk manufacturing business", "zetwerk manufacturing businesses", dict(open_a=D15, open_b=D15)),
])
def test_real_name_pairs_join_when_dates_agree(a, b, kwargs):
    assert same_company(a, b, **kwargs)


@pytest.mark.parametrize("a, b, kwargs, why", [
    ("tata motors", "tata motors finance", dict(open_a=D15, open_b=D20), "different open days"),
    ("tata motors", "tata motors finance", dict(), "no dates: names alone never join"),
    ("tata motors", "tata motors finance", dict(open_a=D15), "one side has no date"),
    ("hd fire protect", "hd fire", dict(open_a=date(2026, 10, 13), open_b=D15), "different open days"),
    ("jio", "jio platforms", dict(open_a=D21, open_b=D21), "one-word key shorter than 5 letters"),
    ("paras", "paras healthcare", dict(open_a=D21, open_b=date(2026, 10, 28)), "different open days"),
    ("sahajanand medical", "sahajanand medical technologies",
     dict(open_a=D15, open_b=D15, market_a="IN_SME", market_b="IN_MAIN"), "markets disagree"),
    ("acme industries", "acme universal safezone", dict(open_a=date(2026, 9, 30), open_b=date(2026, 9, 30)),
     "shared first word only"),
    ("acme india industries", "acme universal safezone 9", dict(open_a=date(2026, 9, 28), open_b=date(2026, 9, 30)),
     "different open days"),
])
def test_near_misses_do_not_join(a, b, kwargs, why):
    assert not same_company(a, b, **kwargs), why


def test_empty_keys_never_join():
    assert not same_company("", "zetwerk", open_a=D15, open_b=D15)
    assert not same_company("zetwerk", "", open_a=D15, open_b=D15)


def test_real_display_names_reduce_to_keys_that_the_rule_joins():
    # keys produced by company_key from the names the sites actually print
    nse_name = company_key("Zetwerk Manufacturing Businesses Limited")
    investorgain_name = company_key("Zetwerk")
    ipopremium_name = company_key(_clean_name("Zetwerk Manufacturing Ltd. (Mainboard)"))  # as gmp keys it
    assert same_company(investorgain_name, nse_name, open_a=D15, open_b=D15, market_a="IN_MAIN", market_b="IN_MAIN")
    assert same_company(ipopremium_name, nse_name, open_a=D15, open_b=D15)
