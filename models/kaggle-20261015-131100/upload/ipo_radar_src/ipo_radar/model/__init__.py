"""Listing-gain model: features, training, walk-forward evaluation and serving.

`ModelNotTrained` subclasses FileNotFoundError on purpose: cli.backend already maps FileNotFoundError from
load_model to "run `ipo train`", so a missing model never crashes the dashboard.
"""


class ModelNotTrained(FileNotFoundError):
    """No trained bundle exists yet (or none covers the requested stage). The message says what to run."""
