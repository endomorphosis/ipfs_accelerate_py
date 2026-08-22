"""EAAEF-170: release CI matrix is blocking and names required suites."""

from __future__ import annotations

from pathlib import Path

WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "external-agent-fabric.yml"
)

REQUIRED = (
    "test/api/test_external_agent_handoff_contracts.py",
    "test/api/test_external_agent_codex_adapter.py",
    "test/security/test_external_prompt_injection.py",
    "test/api/test_container_execution_contracts.py",
    "test/api/test_external_agent_control_schema.py",
)


def test_workflow_is_blocking_fail_closed() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "continue-on-error" not in text
    assert "set -euo pipefail" in text
    assert "pytest" in text
    for path in REQUIRED:
        assert path in text
