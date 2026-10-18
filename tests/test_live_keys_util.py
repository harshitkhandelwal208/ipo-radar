from datetime import date

import pytest

from ipo_radar.live._util import parse_date, parse_number, parse_price_band
from ipo_radar.live.keys import company_key


@pytest.mark.parametrize("name, key", [
    ("Tonbo Imaging India Limited", "tonbo imaging"),
    ("Tonbo Imaging India IPO", "tonbo imaging"),
    ("HD Fire Protect Ltd.", "hd fire protect"),
    ("Sahajanand Medical Technologies Limited", "sahajanand medical technologies"),
    ("R.K. Fashion Accessories Limited", "r k fashion accessories"),
    ("R.K.Fashion Accessories", "r k fashion accessories"),
    ("Nityas Gems & Jewellery Limited", "nityas gems and jewellery"),
    ("Zetwerk Manufacturing Businesses Pvt. Ltd.", "zetwerk manufacturing businesses"),
    ("   ", ""),
])
def test_company_key_collides_across_spellings(name, key):
    assert company_key(name) == key


def test_parse_price_band_forms():
    assert parse_price_band("Rs.665 to Rs.700") == (665.0, 700.0)
    assert parse_price_band("₹258 to ₹271") == (258.0, 271.0)
    assert parse_price_band("₹271 per share") == (271.0, 271.0)
    assert parse_price_band(None) == (None, None)
    assert parse_price_band("n/a") == (None, None)


def test_parse_number_and_date():
    assert parse_number("2,62,84,500shares(agg. up to ₹712Cr)") == 26284500.0
    assert parse_number("") is None
    assert parse_number("-") is None
    assert parse_date("15-Oct-2026") == date(2026, 10, 15)
    assert parse_date("30-SEP-2026") == date(2026, 9, 30)
    assert parse_date("Oct 21, 2026") == date(2026, 10, 21)
    assert parse_date("-") is None
    assert parse_date("31-Feb-2026") is None


def test_blank_or_unreadable_numbers_and_bands_are_none():
    assert parse_price_band("") == (None, None)
    assert parse_number("abc") is None
