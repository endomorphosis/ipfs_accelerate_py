#!/usr/bin/env python3
"""Stable module entry point for multi-supervisor implementation tracks.

The checked-in configured-board scheduler emits an explicit legacy task-source
selection, while this revision's implementation-supervisor parser predates
those four control-plane options.  This entry boundary validates and consumes
only the exact fail-closed legacy tuple.  The managed daemon already defaults
to ``legacy-markdown`` at this revision; no option is silently generalized.
"""

from pathlib import Path
import os
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    main,
)


_LEGACY_VALUE_OPTIONS = {
    "--task-source-kind": "legacy-markdown",
    "--authority-mode": "legacy_markdown",
    "--state-failover-policy": "fail_closed",
}
_LEGACY_FLAG = "--explicit-legacy-task-source"


def normalize_configured_task_source_args(argv: list[str]) -> list[str]:
    """Validate/consume the scheduler's exact legacy-source compatibility tuple."""

    filtered: list[str] = []
    observed: dict[str, str] = {}
    explicit_legacy = False
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == _LEGACY_FLAG:
            if explicit_legacy:
                raise ValueError(f"duplicate option: {_LEGACY_FLAG}")
            explicit_legacy = True
            index += 1
            continue
        matched = next(
            (
                option
                for option in _LEGACY_VALUE_OPTIONS
                if token == option or token.startswith(option + "=")
            ),
            None,
        )
        if matched is None:
            filtered.append(token)
            index += 1
            continue
        if matched in observed:
            raise ValueError(f"duplicate option: {matched}")
        if token == matched:
            if index + 1 >= len(argv):
                raise ValueError(f"missing value for {matched}")
            value = argv[index + 1]
            index += 2
        else:
            value = token.split("=", 1)[1]
            index += 1
        observed[matched] = value

    compatibility_requested = bool(observed or explicit_legacy)
    if compatibility_requested:
        missing = sorted(set(_LEGACY_VALUE_OPTIONS) - set(observed))
        if missing or not explicit_legacy:
            details = missing + ([] if explicit_legacy else [_LEGACY_FLAG])
            raise ValueError(
                "incomplete explicit legacy task-source tuple: "
                + ", ".join(details)
            )
        mismatches = {
            option: value
            for option, value in observed.items()
            if value != _LEGACY_VALUE_OPTIONS[option]
        }
        if mismatches:
            raise ValueError(
                "unsupported configured task-source tuple: "
                + ", ".join(
                    f"{option}={value!r}" for option, value in sorted(mismatches.items())
                )
            )
    return filtered


def seal_child_pythonpath(environment: dict[str, str] | None = None) -> str:
    """Bind ``python -P`` managed children to this exact checked-out package."""

    target = os.environ if environment is None else environment
    value = str(REPO_ROOT)
    target["PYTHONPATH"] = value
    return value


if __name__ == "__main__":
    try:
        normalized_argv = normalize_configured_task_source_args(sys.argv[1:])
    except ValueError as exc:
        print(f"implementation_supervisor_entry.py: error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    seal_child_pythonpath()
    raise SystemExit(main(normalized_argv))
