from __future__ import annotations

from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_task_router import (
    advise_board_task_kind,
    compose_task_kind,
    routing_questions,
)
from ipfs_accelerate_py.llm_allocation.intelligence_index import intelligence_floor_for_task


def test_routing_questions_are_closed_kinds() -> None:
    questions = routing_questions()
    assert questions["intent"].to_dict()["type"] == "choice"
    assert "implementation" in questions["intent"].to_dict()["criteria"]
    assert questions["is_coding_task"].to_dict()["type"] == "noul"


def test_compose_escalates_floor_and_keeps_board_on_low_confidence_downgrade() -> None:
    class _High:
        choices = {
            "intent": SimpleNamespace(choice="implementation", confidence=0.7),
        }

    kind, _conf, reasons = compose_task_kind(_High(), board_kind="legal")
    assert kind == "implementation"
    assert intelligence_floor_for_task(kind) > intelligence_floor_for_task("legal")
    assert "escalate_or_same_floor" in reasons

    class _LowDowngrade:
        choices = {
            "intent": SimpleNamespace(choice="inventory", confidence=0.4),
        }

    kind, _conf, reasons = compose_task_kind(_LowDowngrade(), board_kind="implementation")
    assert kind == "implementation"
    assert "keep_board_kind" in reasons


def test_advise_without_key_returns_board_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    item = {
        "task_id": "TASK-1",
        "metadata": {"kind": "legal"},
        "title": "review the license headers",
    }
    assert advise_board_task_kind(item) == "legal"


def test_allocate_supervisor_endpoint_without_key_still_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.allocation_compat import (
        allocate_supervisor_endpoint,
    )

    allocated = allocate_supervisor_endpoint(
        task={"task_id": "TASK-LEGAL", "metadata": {"kind": "legal"}, "title": "license"},
        available_providers=("openrouter", "grok_cli", "codex_cli"),
    )
    assert allocated["provider"]
    assert allocated["model_name"]
    assert "typesafe" not in str(allocated.get("provider") or "").casefold()
