"""EAAEF-102: closed typed replanning triggers."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.external_replanning import (
    TRIGGERS,
    ReplanError,
    compile_trigger,
)


def test_closed_trigger_set() -> None:
    trigger = compile_trigger("failed_tests", plan_id="plan-1", evidence_id="sha256:" + "a" * 64)
    assert trigger.kind in TRIGGERS
    assert "no_progress" in TRIGGERS


def test_unknown_trigger_fails_closed() -> None:
    with pytest.raises(ReplanError, match="unknown"):
        compile_trigger("vibes", plan_id="plan-1", evidence_id="e")
