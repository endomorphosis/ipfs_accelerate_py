"""Bind an explicit requested Grok turn limit before native dispatch.

This validates argv policy, not the native CLI's tool/request accounting or a
token ceiling. Omitted router settings retain their existing legacy behavior.
"""
from __future__ import annotations

import re

MAX_GROK_TURNS = (1 << 32) - 1
MAX_ARGV_PARTS = 4096
MAX_ARG_BYTES = 65_536
MAX_ARGV_BYTES = 1024 * 1024
_DECIMAL = re.compile(r'[1-9][0-9]{0,9}\Z')


def validate_grok_max_turns(value):
    """Return one exact bounded integer; never coerce an explicit setting."""
    if type(value) is not int or not 1 <= value <= MAX_GROK_TURNS:
        raise ValueError('explicit grok_max_turns requires an integer from 1 through 4294967295')
    return value


def _turn_argument(value):
    if _DECIMAL.fullmatch(value) is None:
        raise ValueError('malformed explicit Grok maximum-turns argument')
    turns = int(value)
    if turns > MAX_GROK_TURNS:
        raise ValueError('malformed explicit Grok maximum-turns argument')
    return turns


def bind_explicit_grok_max_turns(argv, requested):
    """Copy argv and append or verify exactly one requested turn argument.

    Both native spellings are accepted without adding a second occurrence.
    Repeated options, conflicting values and end-of-options markers refuse
    locally. No files, provider environment, subprocesses or receipts are read.
    """
    requested = validate_grok_max_turns(requested)
    if type(argv) not in (list, tuple) or not 1 <= len(argv) <= MAX_ARGV_PARTS:
        raise ValueError('bounded structured Grok command argv required')
    result = list(argv)
    total = 0
    try:
        for part in result:
            if type(part) is not str or len(part) > MAX_ARG_BYTES or '\x00' in part:
                raise ValueError('bounded structured Grok command argv required')
            size = len(part.encode('utf-8'))
            total += size
            if size > MAX_ARG_BYTES or total > MAX_ARGV_BYTES:
                raise ValueError('bounded structured Grok command argv required')
    except UnicodeError:
        raise ValueError('bounded structured Grok command argv required') from None
    if not result[0]:
        raise ValueError('bounded structured Grok command argv required')
    if '--' in result:
        raise ValueError('explicit Grok maximum turns refuse end-of-options markers')
    observed = []
    index = 1
    while index < len(result):
        part = result[index]
        if part == '--max-turns':
            if index + 1 >= len(result):
                raise ValueError('malformed explicit Grok maximum-turns argument')
            observed.append(_turn_argument(result[index + 1]))
            index += 2
        elif part.startswith('--max-turns='):
            observed.append(_turn_argument(part.partition('=')[2]))
            index += 1
        else:
            index += 1
    if len(observed) > 1:
        raise ValueError('duplicate explicit Grok maximum-turns arguments')
    if observed and observed[0] != requested:
        raise ValueError('custom Grok command conflicts with explicit grok_max_turns')
    if not observed:
        result.extend(['--max-turns', str(requested)])
    return result
