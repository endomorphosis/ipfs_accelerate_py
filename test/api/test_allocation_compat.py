"""Agent supervisor compatibility with the llm_allocation schema."""

from __future__ import annotations

import os

from ipfs_accelerate_py.agent_supervisor.todo_daemon.allocation_compat import (
    allocate_supervisor_endpoint,
    allocation_session_id_for_task,
    apply_resume_argv,
    ensure_allocation_schema,
    map_supervisor_provider,
    supervisor_cli_health,
    supervisor_generate_kwargs,
)
from ipfs_accelerate_py.llm_allocation import (
    AllocationStore,
    CallObservation,
    get_allocation_store,
    reset_allocation_store,
)
from ipfs_accelerate_py.llm_allocation.paths import CLI_PROVIDERS


def test_ensure_allocation_schema_creates_new_tables(tmp_path, monkeypatch) -> None:
    store = AllocationStore(tmp_path / "alloc.duckdb")
    reset_allocation_store(store)
    assert ensure_allocation_schema() is True
    conn = store._connect()
    tables = {
        str(row[0])
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    }
    conn.close()
    for name in (
        "allocation_sessions",
        "cli_sessions",
        "cli_session_aliases",
        "api_key_slots",
        "api_key_stats",
        "session_key_bindings",
        "call_observations",
    ):
        assert name in tables
    reset_allocation_store(None)


def test_supervisor_provider_map_and_generate_kwargs(tmp_path) -> None:
    store = AllocationStore(tmp_path / "alloc.duckdb")
    reset_allocation_store(store)
    assert map_supervisor_provider("grok") == "grok_cli"
    assert map_supervisor_provider("claude") == "claude_code"
    assert map_supervisor_provider("muse_code") == "muse_code"
    assert map_supervisor_provider("openrouter") == "openrouter"
    assert map_supervisor_provider("hf") == "hf_inference_api"
    assert map_supervisor_provider("codex") == "codex_cli"
    sid = allocation_session_id_for_task("TASK-9")
    assert sid == "asup-TASK-9"
    kwargs = supervisor_generate_kwargs(task_id="TASK-9", provider="grok")
    assert kwargs["allocation_session_id"] == sid
    assert kwargs["allocation_path"] == "cli"
    assert kwargs["provider"] == "grok_cli"
    store.record(
        CallObservation(
            provider="grok_cli",
            protocol="cli",
            path="cli",
            session_id=sid,
            success=True,
            extra_metadata={"session_id": "grok-native-1"},
        )
    )
    argv = apply_resume_argv(
        ["grok", "--prompt-file", "x.txt"],
        provider="grok",
        task_id="TASK-9",
    )
    assert "--resume" in argv
    assert "grok-native-1" in argv
    reset_allocation_store(None)


def test_supervisor_cli_health_aliases_supervisor_names() -> None:
    report = supervisor_cli_health()
    tools = report.get("tools") or {}
    for name in CLI_PROVIDERS:
        assert name in tools
    assert "grok" in tools
    assert tools["grok"]["router_provider"] == "grok_cli"


def test_api_backends_status_tracks_health_and_remaining_tokens(tmp_path, monkeypatch) -> None:
    from ipfs_accelerate_py.llm_allocation.api_status import api_backends_status
    from ipfs_accelerate_py.llm_allocation.paths import API_PROVIDERS

    store = AllocationStore(tmp_path / "alloc.duckdb")
    reset_allocation_store(store)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    store.record(
        CallObservation(
            provider="openrouter",
            protocol="http",
            path="api",
            success=True,
            remaining_tokens=80_000,
            total_tokens=1_200,
        )
    )
    store.record(
        CallObservation(
            provider="hf_inference_api",
            protocol="http",
            path="api",
            success=True,
            remaining_tokens=0,
            total_tokens=50,
        )
    )
    report = api_backends_status(store=store, environ=os.environ)
    for name in API_PROVIDERS:
        assert name in report["backends"]
    assert "openrouter" in report["ready"]
    assert report["backends"]["openrouter"]["tokens_ok"] is True
    assert report["backends"]["hf_inference_api"]["status"] == "exhausted"
    reset_allocation_store(None)


def test_allocate_prefers_healthy_api_with_remaining_tokens(tmp_path, monkeypatch) -> None:
    store = AllocationStore(tmp_path / "alloc.duckdb")
    reset_allocation_store(store)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    store.record(
        CallObservation(
            provider="openrouter",
            protocol="http",
            path="api",
            success=True,
            remaining_tokens=50_000,
        )
    )
    allocated = allocate_supervisor_endpoint(
        task={"metadata": {"kind": "implementation"}},
        available_providers=("openrouter", "grok_cli"),
    )
    assert allocated["provider"] == "openrouter"
    assert allocated["allocation_path"] == "api"
    assert allocated["model_name"]
    kwargs = supervisor_generate_kwargs(task_id="TASK-API-1")
    assert kwargs["allocation_session_id"] == "asup-TASK-API-1"
    reset_allocation_store(None)


def test_rank_prefers_api_backend_with_token_headroom(tmp_path) -> None:
    from ipfs_accelerate_py.llm_allocation import rank_provider_names

    store = AllocationStore(tmp_path / "alloc.duckdb")
    reset_allocation_store(store)
    store.upsert_health("openrouter", protocol="http", available=True, authenticated=True, ready=True)
    store.upsert_health("grok_cli", protocol="cli", available=True, authenticated=True, ready=True)
    store.record(
        CallObservation(
            provider="openrouter",
            protocol="http",
            path="api",
            success=True,
            remaining_tokens=90_000,
        )
    )
    store.record(
        CallObservation(
            provider="grok_cli",
            protocol="cli",
            path="cli",
            success=True,
            remaining_tokens=10,
        )
    )
    ranked = rank_provider_names(
        ["grok_cli", "openrouter"],
        store=store,
    )
    assert ranked[0] == "openrouter"
    reset_allocation_store(None)
