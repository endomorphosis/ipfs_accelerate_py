"""Frozen pure validators extracted from qualified source; no runtime imports or I/O."""
from __future__ import annotations
import re
def _digest(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None
