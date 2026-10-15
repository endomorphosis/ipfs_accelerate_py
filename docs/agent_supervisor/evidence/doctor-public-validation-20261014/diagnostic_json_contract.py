"""Frozen pure validators extracted from qualified source; no runtime imports or I/O."""
from __future__ import annotations
import json
def canonical_json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError('invalid source-view value') from exc
