"""DOEP-111 Codex-primed baseline harness."""

from __future__ import annotations

import pytest

from benchmarks.agent_supervisor.doep.baseline import (
    BaselineHarnessError,
    run_codex_primed_baseline,
)


def test_frozen_baseline_does_not_invoke_codex_or_complete() -> None:
    result = run_codex_primed_baseline(
        [{"case_id": "c1", "prompt_cid": "bafy-prompt"}]
    )
    assert result["codex_invoked"] is False
    assert result["frozen"] is True
    assert result["completion_authority"] is False
    assert result["cases"] == ["c1"]


def test_harness_refuses_to_invoke_codex() -> None:
    with pytest.raises(BaselineHarnessError, match="must not auto-start"):
        run_codex_primed_baseline(
            [{"case_id": "c1", "prompt_cid": "bafy-prompt"}],
            invoke_codex=True,
        )
