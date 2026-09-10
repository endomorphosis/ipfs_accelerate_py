"""Launch preflight must survive same-board sibling gitlink advances.

ASEH-032 landed one Kit commit on top of the R40 snapshot.  Exact-HEAD
equality then made exclusive-owner restart fail closed at launch_preflight
even though the gitlink matched nested HEAD and the snapshot stayed an
ancestor.  Datasets already used a descendant floor; Kit must too.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "scripts" / "validate_agent_supervisor_efficiency_state_hardening_board.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "aseh_board_validator_under_test",
        VALIDATOR,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_board_validator_source_uses_descendant_floor_for_kit_and_datasets() -> None:
    source = VALIDATOR.read_text(encoding="utf-8")
    assert "nested HEAD differs from exact R40 repair snapshot" not in source
    assert "nested HEAD is not a descendant of" in source
    assert source.count('merge-base",') >= 2
    assert "ASEH-032" in source


def test_live_checkout_kit_head_is_descendant_of_r40_snapshot() -> None:
    validator = _load_validator()
    kit_commit, _kit_tree = validator.BASES["ipfs_kit_py"]
    report = validator.validate(check_git=True)
    kit_errors = [
        error
        for error in report["errors"]
        if str(error).startswith("ipfs_kit_py:")
    ]
    assert kit_errors == [], kit_errors
    assert not any(
        "differs from exact R40 repair snapshot" in str(error)
        for error in report["errors"]
    )
    assert kit_commit == "ba5508d940fb5b23a6d0d9b2084f5195cd26a671"
