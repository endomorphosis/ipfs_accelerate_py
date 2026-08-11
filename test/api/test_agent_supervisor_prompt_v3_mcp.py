"""ASE3-011 MCP prompt-lifecycle tool tests."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import (
    SupervisorObservation,
    SupervisorRun,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    PROMPT_LIFECYCLE_TOOLS,
    configure_prompt_lifecycle_supervisor,
    prompt_lifecycle_discovery_manifest,
    register_prompt_lifecycle_tools,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    prompt_entrypoints as pe,
)


class _RecordingManager:
    def __init__(self) -> None:
        self.tools: list[dict[str, Any]] = []

    def register_tool(self, **kwargs: Any) -> None:
        self.tools.append(kwargs)


class _FakeSupervisor:
    composition_cid = "cid:comp"

    def run(self, prompt: str) -> SupervisorRun:
        return SupervisorRun(
            run_id="run-mcp",
            run_revision=1,
            composition_cid=self.composition_cid,
            state="running",
            health="healthy",
            event_cursor="e1",
            effect_receipt_cids=("r1",),
        )

    def preview(self, prompt: str) -> SupervisorObservation:
        return SupervisorObservation(
            run_id="",
            state="preview",
            health="unknown",
            event_cursor="",
            composition_cid=self.composition_cid,
            summary="preview",
            values={"effect_applied": False, "prompt_cid": "cid:p"},
        )

    def status(self, run_id: str | None = None) -> SupervisorObservation:
        return SupervisorObservation(
            run_id=run_id or "run-mcp",
            state="running",
            health="healthy",
            event_cursor="e1",
            composition_cid=self.composition_cid,
            summary="ok",
        )

    def follow(self, run_id: str | None = None):
        yield self.status(run_id)

    def explain(self, run_id: str | None = None) -> SupervisorObservation:
        return self.status(run_id)

    def doctor(self, run_id: str | None = None) -> SupervisorObservation:
        return self.status(run_id)

    def steer(self, run_id: str, prompt: str) -> SupervisorObservation:
        return SupervisorObservation(
            run_id=run_id,
            state="running",
            health="healthy",
            event_cursor="e1",
            composition_cid=self.composition_cid,
            summary="steer",
            values={"effect_applied": False},
        )


def test_discovery_manifest_lists_prompt_tools() -> None:
    manifest = prompt_lifecycle_discovery_manifest()
    assert set(manifest["tools"]) == set(PROMPT_LIFECYCLE_TOOLS)
    assert manifest["cold_registration"] is True
    assert manifest["normal_input"] == "prompt"


def test_register_prompt_lifecycle_tools_is_cold() -> None:
    manager = _RecordingManager()
    register_prompt_lifecycle_tools(manager)
    names = {item["name"] for item in manager.tools}
    assert names == set(PROMPT_LIFECYCLE_TOOLS)
    for item in manager.tools:
        assert item["category"] == "agent_supervisor"
        assert "prompt-lifecycle" in item["tags"]
        assert item["input_schema"]["type"] == "object"


def test_run_tool_reaches_injected_supervisor() -> None:
    configure_prompt_lifecycle_supervisor(_FakeSupervisor())
    try:
        result = asyncio.run(pe.agent_supervisor_run(prompt="Improve gates"))
    finally:
        configure_prompt_lifecycle_supervisor(None)
    assert result["ok"] is True
    assert result["result"]["run_id"] == "run-mcp"
    assert result["composition_cid"] == "cid:comp"


def test_preview_does_not_leak_prompt_body() -> None:
    secret = "do-not-leak-this-prompt"
    configure_prompt_lifecycle_supervisor(_FakeSupervisor())
    try:
        result = asyncio.run(pe.agent_supervisor_preview(prompt=secret))
    finally:
        configure_prompt_lifecycle_supervisor(None)
    assert result["ok"] is True
    assert secret not in str(result)


def test_client_repository_path_denied_without_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_prompt_lifecycle_supervisor(None)
    monkeypatch.delenv(pe.REPOSITORY_ALLOWLIST_ENV, raising=False)
    result = asyncio.run(
        pe.agent_supervisor_run(
            prompt="x",
            repository="/tmp/not-allowlisted",
        )
    )
    assert result["ok"] is False
    assert result["error_code"] == "path_denied"


def test_allowlisted_repository_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv(pe.REPOSITORY_ALLOWLIST_ENV, str(tmp_path.resolve()))
    configure_prompt_lifecycle_supervisor(_FakeSupervisor())
    try:
        result = asyncio.run(
            pe.agent_supervisor_preview(
                prompt="Improve gates",
                repository=str(repo),
            )
        )
    finally:
        configure_prompt_lifecycle_supervisor(None)
        monkeypatch.delenv(pe.REPOSITORY_ALLOWLIST_ENV, raising=False)
    assert result["ok"] is True


def test_empty_prompt_invalid() -> None:
    configure_prompt_lifecycle_supervisor(_FakeSupervisor())
    try:
        result = asyncio.run(pe.agent_supervisor_run(prompt="  "))
    finally:
        configure_prompt_lifecycle_supervisor(None)
    assert result["ok"] is False
    assert result["error_code"] == "invalid"
