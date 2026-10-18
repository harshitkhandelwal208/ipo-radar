"""How the CLI learns that the model's P50 is not the headline at a stage.

`predict()` sets the P50 to the segment's training median and adds NO_MODEL_CAVEAT when the release weight is 0.
schema.Prediction has no field for this, so the caveat text is the marker. It is copied here, not imported, so that
`ipo --help` does not load the model stack; tests/test_cli_pure.py checks it matches model/predict.py.
The 'worse than the historical median' flag from the bundle only supplies the reason shown to the user.
"""
from __future__ import annotations

import re

NO_MODEL_CAVEAT = "model estimate not used at this stage (release weight 0): the historical median is shown"
NO_MODEL_REASON = "the model's release weight at this stage is 0, so the segment median is shown"
WORSE_REASON = "worse than the historical median at this stage on the latest blocks"
_OWN_RANGE = re.compile(r"model's own range, not used at this stage: p10 (?P<p10>[+-]?\d+(?:\.\d+)?), "
                        r"p50 (?P<p50>[+-]?\d+(?:\.\d+)?), p90 (?P<p90>[+-]?\d+(?:\.\d+)?) pp")


def own_range(caveats: list[str]) -> tuple[float, float, float] | None:
    """The model's own P10, P50, P90 (pp), from the caveat predict() writes when the model is not used. None when
    absent or malformed."""
    for text in caveats:
        m = _OWN_RANGE.search(text)
        if m:
            return float(m["p10"]), float(m["p50"]), float(m["p90"])
    return None


def unused_stages(bundle: object) -> dict[str, str]:
    """stage -> the 'worse than the historical median' reason, for stages whose latest blocks say so."""
    recent = getattr(bundle, "recent", None) or {}
    return {stage: WORSE_REASON for stage, info in recent.items()
            if isinstance(info, dict) and info.get("worse") is True}
