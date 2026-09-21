"""Lazy adapters for immutable research baselines."""

import sys
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parent


def _load_hvp_adapters():
    sys.path.insert(0, str(BASELINE_ROOT))
    try:
        from hvp_baselines.adapters import sdpa_hvp_manual, sdpa_hvp_semi_manual
    finally:
        sys.path.remove(str(BASELINE_ROOT))
    return sdpa_hvp_manual, sdpa_hvp_semi_manual


def sdpa_hvp_manual(*args, **kwargs):
    manual, _ = _load_hvp_adapters()
    return manual(*args, **kwargs)


def sdpa_hvp_semi_manual(*args, **kwargs):
    _, semi_manual = _load_hvp_adapters()
    return semi_manual(*args, **kwargs)


__all__ = ["sdpa_hvp_manual", "sdpa_hvp_semi_manual"]
