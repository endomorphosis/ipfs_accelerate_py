"""Focused semantic binding admission checks."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.semantic_binding_admission import (
    BindingAdmissionError,
    admit_executable,
    project_task_record,
)


def _current() -> dict:
    return {
        "task_id": "LGSWF-023",
        "accepted_plan_cid": "sha256:" + ("aa" * 32),
        "semantic_state_root_cid": "sha256:" + ("bb" * 32),
        "completion_rule": "supervisor-acceptance",
        "binding_cid": "sha256:" + ("cc" * 32),
    }


def test_canonical_projection_is_ready() -> None:
    result = admit_executable(_current())
    assert result["ready"] is True
    assert result["projected"]["entity_id"] == "LGSWF-023"


def test_missing_stale_and_sentinel_are_explicit_non_ready() -> None:
    missing = project_task_record({"task_id": "LGSWF-023"})
    assert missing["ready"] is False
    assert any(reason.startswith("missing:") for reason in missing["reasons"])
    sentinel = project_task_record(
        {
            **_current(),
            "semantic_state_root_cid": "REBIND_REQUIRED_BY_LGSWF-005",
        }
    )
    assert sentinel["ready"] is False
    assert any(reason.startswith("sentinel:") for reason in sentinel["reasons"])
    stale = admit_executable({**_current(), "stale_binding": True})
    assert stale["ready"] is False
    assert "stale-binding" in stale["reasons"]


def test_markdown_and_history_mutation_rejected() -> None:
    with pytest.raises(BindingAdmissionError, match="Markdown"):
        project_task_record({**_current(), "treat_markdown_as_authority": True})
    with pytest.raises(BindingAdmissionError, match="immutable"):
        project_task_record({**_current(), "claimed_history_mutation": True})
