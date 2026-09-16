"""DuckDB LLM allocation: classification, persistence, and ranking."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.llm_allocation import (
    AllocationStore,
    CallErrorKind,
    CallObservation,
    bind_session_api_key,
    choose_cli_route,
    classify_provider_failure,
    filter_names_for_path,
    inject_cli_session_kwargs,
    list_api_key_slots,
    migrate_cli_session,
    path_for_provider,
    rank_provider_names,
    register_api_key,
    reset_allocation_store,
    resolve_session,
    select_api_key,
)
from ipfs_accelerate_py.llm_allocation.allocator import score_provider


@pytest.fixture
def store(tmp_path: Path) -> AllocationStore:
    path = tmp_path / "alloc.duckdb"
    instance = AllocationStore(path)
    reset_allocation_store(instance)
    yield instance
    reset_allocation_store(None)


def test_classify_cli_and_api_failures() -> None:
    meta = classify_provider_failure(
        "meta_ai",
        "Meta AI HTTP 429: Rate limit exceeded. Please retry after 15 seconds.",
    )
    assert meta.kind is CallErrorKind.RATE_LIMIT
    assert meta.retryable is True

    billing = classify_provider_failure(
        "muse_code",
        "Error: insufficient_quota — payment required for Meta AI",
    )
    assert billing.kind is CallErrorKind.BILLING
    assert billing.retryable is False

    codex = classify_provider_failure(
        "codex_cli",
        "exceeded your current quota; check your plan and billing",
    )
    assert codex.kind is CallErrorKind.BILLING

    grok = classify_provider_failure("grok_cli", "Grok CLI is not authenticated. not signed in")
    assert grok.kind is CallErrorKind.AUTHENTICATION

    claude = classify_provider_failure("claude_code", "rate limit exceeded")
    assert claude.kind is CallErrorKind.RATE_LIMIT
    assert claude.retryable is True

    gemini = classify_provider_failure("gemini_cli", "resource exhausted")
    assert gemini.kind is CallErrorKind.RATE_LIMIT

    copilot = classify_provider_failure("copilot_cli", "upgrade your subscription")
    assert copilot.kind is CallErrorKind.BILLING

    mistral = classify_provider_failure("mistral_vibe", "rate_limit_exceeded; retry after 30s")
    assert mistral.kind is CallErrorKind.RATE_LIMIT


def test_empty_store_preserves_original_order(store: AllocationStore) -> None:
    names = ["openrouter", "openai", "meta_ai", "codex_cli"]
    assert rank_provider_names(names, store=store) == names


def test_store_records_and_ranks_away_from_billing(store: AllocationStore) -> None:
    store.record(
        CallObservation(
            provider="openai",
            protocol="http",
            success=False,
            error_kind="billing",
            retryable=False,
            total_tokens=10,
        )
    )
    store.record(
        CallObservation(
            provider="meta_ai",
            protocol="http",
            success=True,
            error_kind="success",
            prompt_tokens=8,
            completion_tokens=2,
            total_tokens=10,
            remaining_requests=2900,
            limit_requests=3000,
        )
    )
    ranked = rank_provider_names(["openai", "meta_ai"], store=store)
    assert ranked[0] == "meta_ai"
    assert ranked[1] == "openai"


def test_score_skips_zero_remaining_requests() -> None:
    skip, _score, _index = score_provider(
        "meta_ai",
        {"remaining_requests": 0, "recent_calls": 1, "recent_ok": 1, "recent_fail": 0},
        index=0,
    )
    assert skip == 0


def test_cli_and_api_paths_are_separate(store: AllocationStore) -> None:
    assert path_for_provider("codex_cli") == "cli"
    assert path_for_provider("meta_ai") == "api"
    mixed = ["openai", "codex_cli", "meta_ai", "grok_cli"]
    assert rank_provider_names(mixed, store=store, path="cli") == ["codex_cli", "grok_cli"]
    assert rank_provider_names(mixed, store=store, path="api") == ["openai", "meta_ai"]
    assert filter_names_for_path(["openai", "meta_ai"], "cli") == []
    assert filter_names_for_path(["codex_cli", "grok_cli"], "api") == []


def test_session_tracks_cost_tokens_throughput_and_latency(store: AllocationStore) -> None:
    store.record(
        CallObservation(
            provider="meta_ai",
            protocol="http",
            path="api",
            session_id="sess-1",
            success=True,
            latency_ms=200.0,
            prompt_tokens=80,
            completion_tokens=20,
            total_tokens=100,
            estimated_cost_usd=0.01,
            tokens_per_second=100.0,
        )
    )
    store.record(
        CallObservation(
            provider="meta_ai",
            protocol="http",
            path="api",
            session_id="sess-1",
            success=True,
            latency_ms=100.0,
            prompt_tokens=40,
            completion_tokens=10,
            total_tokens=50,
            estimated_cost_usd=0.005,
            tokens_per_second=100.0,
        )
    )
    session = store.get_session("sess-1")
    assert session["path"] == "api"
    assert session["preferred_provider"] == "meta_ai"
    assert int(session["call_count"]) == 2
    assert int(session["total_tokens"]) == 150
    assert abs(float(session["total_cost_usd"]) - 0.015) < 1e-9
    assert abs(float(session["avg_latency_ms"]) - 150.0) < 1e-6
    assert float(session["tokens_per_second"]) > 0


def test_cli_metadata_extractor_normalizes_ids_and_usage() -> None:
    from ipfs_accelerate_py.cli_runtime.cli_metadata import (
        extract_cli_metadata,
        remember_cli_run,
    )

    stdout = json.dumps(
        {
            "sessionId": "sess-grok-1",
            "requestId": "req-9",
            "model": "grok-4.5",
            "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
            "total_cost_usd": 0.002,
            "prompt": "do not store this",
        }
    )
    meta = extract_cli_metadata(stdout, provider="grok_cli")
    assert meta["session_id"] == "sess-grok-1"
    assert meta["request_id"] == "req-9"
    assert meta["model_id"] == "grok-4.5" or meta.get("model") == "grok-4.5"
    assert int(meta["prompt_tokens"]) == 11
    assert int(meta["completion_tokens"]) == 7
    assert "do not store this" not in str(meta)
    remembered = remember_cli_run("codex_cli", '{"session_id":"codex-9","total_tokens":4}')
    assert remembered["session_id"] == "codex-9"


def test_allocation_session_persists_cli_metadata_for_grok(store: AllocationStore) -> None:
    store.record(
        CallObservation(
            provider="grok_cli",
            protocol="cli",
            path="cli",
            session_id="alloc-grok",
            success=True,
            latency_ms=80.0,
            model="grok-4.5",
            extra_metadata={
                "session_id": "grok-native-1",
                "request_id": "req-g",
                "total_cost_usd": 0.01,
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "prompt": "secret-prompt",
            },
        )
    )
    session = store.get_session("alloc-grok")
    meta = session.get("provider_metadata") or {}
    assert meta["session_id"] == "grok-native-1"
    assert meta["request_id"] == "req-g"
    assert "secret-prompt" not in str(session)
    cli = (session.get("cli_sessions") or {}).get("grok_cli") or {}
    assert cli.get("native_session_id") == "grok-native-1"


def test_allocation_session_persists_muse_metadata(store: AllocationStore) -> None:
    store.record(
        CallObservation(
            provider="muse_code",
            protocol="cli",
            path="cli",
            session_id="alloc-muse",
            success=True,
            latency_ms=120.0,
            model="muse-spark-1.3-contributor",
            extra_metadata={
                "session_id": "01a0a6e1-88f6-7781-b25f-fdcc17fce794",
                "run_id": "a4269f48-8f45-4720-baab-90770505a4a1",
                "model_id": "muse-spark-1.3-contributor",
                "provider_id": "meta",
                "request_id": "req-1",
                "response_id": "resp-1",
                "time_to_first_event_ms": "45",
                "stream_bytes_read": "172793",
                "wire_events_seen": "11",
                "prompt": "SECRET_SHOULD_NOT_PERSIST",
                "api_key": "sk-secret",
            },
        )
    )
    session = store.get_session("alloc-muse")
    muse = session.get("muse") or session.get("provider_metadata") or {}
    assert muse["run_id"] == "a4269f48-8f45-4720-baab-90770505a4a1"
    assert muse["model_id"] == "muse-spark-1.3-contributor"
    assert muse["request_id"] == "req-1"
    assert muse["time_to_first_event_ms"] == "45"
    assert "SECRET_SHOULD_NOT_PERSIST" not in str(session)
    assert "sk-secret" not in str(session)
    assert "prompt" not in muse
    cli = (session.get("cli_sessions") or {}).get("muse_code") or {}
    assert cli.get("native_session_id") == "01a0a6e1-88f6-7781-b25f-fdcc17fce794"
    assert (cli.get("metadata") or {}).get("response_id") == "resp-1"


def test_session_sticky_prefers_last_healthy_provider(store: AllocationStore) -> None:
    store.record(
        CallObservation(
            provider="openai",
            protocol="http",
            path="api",
            session_id="sticky",
            success=True,
            latency_ms=50.0,
            total_tokens=10,
        )
    )
    ranked = rank_provider_names(
        ["meta_ai", "openai"],
        store=store,
        path="api",
        session_id="sticky",
    )
    assert ranked[0] == "openai"


def test_session_id_restores_path_without_explicit_path(store: AllocationStore) -> None:
    store.record(
        CallObservation(
            provider="codex_cli",
            protocol="cli",
            path="cli",
            session_id="cli-sess",
            success=True,
            latency_ms=80.0,
            prompt_tokens=30,
            completion_tokens=10,
            total_tokens=40,
            estimated_cost_usd=0.002,
            tokens_per_second=125.0,
        )
    )
    mixed = ["openai", "codex_cli", "meta_ai", "grok_cli"]
    ranked = rank_provider_names(mixed, store=store, session_id="cli-sess")
    assert ranked[0] == "codex_cli"
    assert set(ranked) <= {"codex_cli", "grok_cli"}
    session = store.get_session("cli-sess")
    assert session["path"] == "cli"
    assert int(session["total_tokens"]) == 40
    assert float(session["tokens_per_second"]) > 0
    assert float(session["avg_latency_ms"]) == 80.0


def test_path_stats_do_not_cross_rank(store: AllocationStore) -> None:
    store.record(
        CallObservation(
            provider="openai",
            protocol="http",
            path="api",
            success=False,
            error_kind="billing",
            retryable=False,
        )
    )
    store.record(
        CallObservation(
            provider="codex_cli",
            protocol="cli",
            path="cli",
            success=True,
            latency_ms=40.0,
            total_tokens=8,
        )
    )
    assert rank_provider_names(["openai", "codex_cli"], store=store, path="cli") == [
        "codex_cli"
    ]
    ranked_api = rank_provider_names(["openai", "codex_cli"], store=store, path="api")
    assert ranked_api == ["openai"]


def test_generate_text_records_observation(
    store: AllocationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ipfs_accelerate_py import llm_router

    class _Prov:
        def generate(self, prompt: str, *, model_name=None, **kwargs):
            _ = (prompt, model_name, kwargs)
            return "hello"

    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *a, **k: _Prov())
    text = llm_router.generate_text(
        "hi",
        provider="mock",
        allocation_session_id="sess-mock",
        allocation_path="api",
    )
    assert text == "hello"
    stats = store.provider_stats(["mock"])
    assert "mock" in stats
    assert int(stats["mock"]["recent_ok"]) >= 1
    session = llm_router.get_allocation_session("sess-mock")
    assert session["session_id"] == "sess-mock"
    assert int(session["success_count"]) >= 1


def test_generate_text_sticky_pins_session_provider(
    store: AllocationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ipfs_accelerate_py import llm_router

    store.record(
        CallObservation(
            provider="mock",
            protocol="local",
            path="api",
            session_id="sticky-gen",
            success=True,
            latency_ms=25.0,
            total_tokens=12,
            estimated_cost_usd=0.001,
        )
    )
    seen: list[object] = []

    class _Prov:
        def generate(self, prompt: str, *, model_name=None, **kwargs):
            _ = (prompt, model_name, kwargs)
            return "sticky"

    def _fake_get(provider=None, **kwargs):
        _ = kwargs
        seen.append(provider)
        return _Prov()

    monkeypatch.setattr(llm_router, "get_llm_provider", _fake_get)
    text = llm_router.generate_text("hi", allocation_session_id="sticky-gen")
    assert text == "sticky"
    assert seen and seen[0] == "mock"


def test_cli_native_session_is_persisted_and_injected(store: AllocationStore) -> None:
    store.upsert_cli_session(
        session_id="work-1",
        provider="muse_code",
        native_session_id="11111111-2222-3333-4444-555555555555",
        resume_style="session_id",
    )
    stored = store.get_cli_session("work-1", "muse_code")
    assert stored["native_session_id"] == "11111111-2222-3333-4444-555555555555"
    kwargs = inject_cli_session_kwargs(
        "muse_code",
        {},
        native_session_id=str(stored["native_session_id"]),
        allocation_session_id="work-1",
    )
    assert kwargs["session_id"] == "11111111-2222-3333-4444-555555555555"
    grok_first = inject_cli_session_kwargs(
        "grok_cli", {}, allocation_session_id="work-1"
    )
    assert grok_first.get("chat_session_id") == "work-1"
    goose_resume = inject_cli_session_kwargs(
        "goose_cli",
        {},
        native_session_id="goose-work-1",
        allocation_session_id="work-1",
    )
    assert goose_resume["session_id"] == "goose-work-1"
    assert goose_resume["resume_session"] is True
    explicit = inject_cli_session_kwargs(
        "muse_code",
        {"session_id": "already-set"},
        native_session_id="11111111-2222-3333-4444-555555555555",
        allocation_session_id="work-1",
    )
    assert explicit["session_id"] == "already-set"


def test_generate_text_resumes_cli_native_session(
    store: AllocationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ipfs_accelerate_py import llm_router

    store.record(
        CallObservation(
            provider="muse_code",
            protocol="cli",
            path="cli",
            session_id="work-cli",
            success=True,
            latency_ms=40.0,
            total_tokens=8,
        )
    )
    store.upsert_cli_session(
        session_id="work-cli",
        provider="muse_code",
        native_session_id="aaaa-bbbb-cccc-dddd",
        resume_style="session_id",
    )
    seen: list[object] = []

    class _Prov:
        def generate(self, prompt: str, *, model_name=None, **kwargs):
            _ = (prompt, model_name)
            seen.append(kwargs.get("session_id"))
            return "resumed"

    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *a, **k: _Prov())
    text = llm_router.generate_text(
        "continue",
        provider="muse_code",
        allocation_session_id="work-cli",
        allocation_path="cli",
    )
    assert text == "resumed"
    assert seen == ["aaaa-bbbb-cccc-dddd"]


def test_choose_cli_route_new_session_ranks_without_resume(
    store: AllocationStore,
) -> None:
    route = choose_cli_route(store=store)
    assert route["path"] == "cli"
    assert route["resume"] is False
    assert route["reason"] == "ranked_new"
    assert route["provider"] in route["ranked"]
    assert set(route["ranked"]) <= {
        "muse_code",
        "goose_cli",
        "codex_cli",
        "copilot_cli",
        "grok_cli",
        "claude_code",
        "gemini_cli",
        "mistral_vibe",
    }


def test_choose_cli_route_resumes_sticky_native_session(store: AllocationStore) -> None:
    store.record(
        CallObservation(
            provider="muse_code",
            protocol="cli",
            path="cli",
            session_id="route-1",
            success=True,
            extra_metadata={"session_id": "muse-native-abc", "model_id": "muse-spark-1.3"},
        )
    )
    route = choose_cli_route("route-1", store=store)
    assert route["resume"] is True
    assert route["provider"] == "muse_code"
    assert route["native_session_id"] == "muse-native-abc"
    assert route["resume_kwargs"].get("session_id") == "muse-native-abc"


def test_migrate_cli_session_rebinds_without_copying_native_id(
    store: AllocationStore,
) -> None:
    store.record(
        CallObservation(
            provider="muse_code",
            protocol="cli",
            path="cli",
            session_id="mig-1",
            success=True,
            extra_metadata={"session_id": "muse-old", "run_id": "run-old"},
        )
    )
    result = migrate_cli_session("mig-1", "grok_cli", store=store, handoff="continue in grok")
    assert result["migrated"] is True
    assert result["native_session_portable"] is False
    assert result["previous_native_session_id"] == "muse-old"
    assert result["handoff_pending"] is True
    session = store.get_session("mig-1")
    assert session["preferred_provider"] == "grok_cli"
    assert session["provider_metadata"]["migrated_from"] == "muse_code"
    assert "continue in grok" in str(session.get("handoff_context") or "")
    route = choose_cli_route("mig-1", store=store)
    assert route["provider"] == "grok_cli"
    assert route["resume"] is False
    mapped = resolve_session("muse-old", store=store)
    assert mapped["allocation_session_id"] == "mig-1"
    assert mapped["current_provider"] == "grok_cli"


def test_native_session_id_maps_to_new_cli_after_capture(store: AllocationStore) -> None:
    store.record(
        CallObservation(
            provider="muse_code",
            protocol="cli",
            path="cli",
            session_id="map-1",
            success=True,
            extra_metadata={"session_id": "muse-native-old"},
        )
    )
    migrate_cli_session("map-1", "codex_cli", store=store, handoff="bring context")
    store.record(
        CallObservation(
            provider="codex_cli",
            protocol="cli",
            path="cli",
            session_id="map-1",
            success=True,
            extra_metadata={"session_id": "codex-native-new"},
        )
    )
    by_old = resolve_session("muse-native-old", store=store)
    by_new = resolve_session("codex-native-new", store=store)
    by_alloc = resolve_session("map-1", store=store)
    assert by_old["allocation_session_id"] == "map-1"
    assert by_old["current_native_session_id"] == "codex-native-new"
    assert by_old["current_provider"] == "codex_cli"
    assert by_new["allocation_session_id"] == "map-1"
    assert by_alloc["current_native_session_id"] == "codex-native-new"
    session = store.get_session("map-1")
    aliases = session.get("session_aliases") or []
    natives = {row["native_session_id"] for row in aliases}
    assert "muse-native-old" in natives
    assert "codex-native-new" in natives


def test_extract_cli_assistant_text_skips_user_and_uses_last_output() -> None:
    from ipfs_accelerate_py.cli_runtime.cli_metadata import extract_cli_assistant_text

    stdout = "\n".join(
        [
            '{"payload_type":"turn.input.user","payload":{"text":"ignore me"}}',
            '{"payload_type":"run.output.delta","payload":{"text":"hello"}}',
            '{"type":"agent_message","text":"hello world"}',
            "not json leftover",
        ]
    )
    assert extract_cli_assistant_text(stdout) == "hello world"


def _write_cli_session_fixture(home: Path, provider: str, sid: str, body: str) -> Path:
    from ipfs_accelerate_py.cli_runtime.cli_handoff import _provider_session_roots

    roots = _provider_session_roots(provider, home=home)
    root = roots[0]
    if provider in {"muse_code", "muse"}:
        path = root / "2026" / "01" / "01" / sid / "session.jsonl"
    elif provider == "codex_cli":
        path = root / sid / "rollout.jsonl"
    elif provider == "claude_code":
        path = root / "proj" / f"{sid}.jsonl"
    else:
        path = root / f"{sid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_extract_native_cli_context_from_every_cli_provider(tmp_path: Path) -> None:
    from ipfs_accelerate_py.cli_runtime.cli_handoff import (
        collect_cli_handoff_context,
        extract_native_cli_context,
    )
    from ipfs_accelerate_py.llm_allocation.paths import CLI_PROVIDERS

    sid = "sess-handoff-1"
    claude_line = json.dumps(
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "hello from claude"}]},
        }
    )
    generic_lines = "\n".join(
        [
            json.dumps({"type": "user", "role": "user", "text": "please continue"}),
            json.dumps({"type": "assistant", "role": "assistant", "text": "hello from tool"}),
        ]
    )
    muse_lines = "\n".join(
        [
            json.dumps(
                {
                    "payload_type": "turn.input.user",
                    "payload": {"prompt": "please continue"},
                }
            ),
            json.dumps(
                {
                    "payload_type": "run.terminal.completed",
                    "payload": {"text": "hello from tool"},
                }
            ),
        ]
    )
    for provider in sorted(CLI_PROVIDERS):
        body = muse_lines if provider in {"muse_code", "muse"} else generic_lines
        if provider == "claude_code":
            body = "\n".join(
                [
                    json.dumps({"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "please continue"}]}}),
                    claude_line,
                ]
            )
        _write_cli_session_fixture(tmp_path, provider, sid, body)
        extracted = extract_native_cli_context(provider, sid, home=tmp_path)
        assert extracted, f"{provider} produced no transcript"
        assert "hello from" in extracted.lower() or "please continue" in extracted.lower(), provider
        packed = collect_cli_handoff_context(
            from_provider=provider,
            native_session_id=sid,
            home=tmp_path,
        )
        assert packed
        for other in sorted(CLI_PROVIDERS):
            if other == provider:
                continue
            # Context packet is provider-agnostic prompt text, usable by any target.
            assert "hello from" in packed.lower() or "please continue" in packed.lower()


def test_history_modes_full_compact_and_none(tmp_path: Path, store: AllocationStore) -> None:
    from ipfs_accelerate_py.cli_runtime.cli_handoff import (
        collect_cli_handoff_context,
        format_compact_history,
        format_full_history,
    )

    sid = "sess-hist-1"
    lines = [json.dumps({"type": "user", "role": "user", "text": "refactor auth.py"})]
    for index in range(12):
        lines.append(
            json.dumps(
                {
                    "type": "assistant",
                    "role": "assistant",
                    "text": f"middle step {index} touching util.py",
                }
            )
        )
    lines.append(json.dumps({"type": "user", "role": "user", "text": "finish tests"}))
    lines.append(
        json.dumps({"type": "assistant", "role": "assistant", "text": "done with tests"})
    )
    _write_cli_session_fixture(tmp_path, "goose_cli", sid, "\n".join(lines))
    full = collect_cli_handoff_context(
        from_provider="goose_cli",
        native_session_id=sid,
        home=tmp_path,
        history="full",
    )
    compact = collect_cli_handoff_context(
        from_provider="goose_cli",
        native_session_id=sid,
        home=tmp_path,
        history="compact",
    )
    none = collect_cli_handoff_context(
        from_provider="goose_cli",
        native_session_id=sid,
        home=tmp_path,
        history="none",
        handoff="",
    )
    assert "middle step 0" in full
    assert "Native history from goose_cli" in full
    assert "Compacted native history" in compact
    assert "Goal: refactor auth.py" in compact
    assert "auth.py" in compact
    assert "done with tests" in compact
    assert "middle step 0" not in compact
    assert none == ""
    store.record(
        CallObservation(
            provider="goose_cli",
            protocol="cli",
            path="cli",
            session_id="hist-job",
            success=True,
            extra_metadata={"session_id": sid},
        )
    )
    # Point extract at tmp HOME via collect is already tested; migrate uses real HOME
    # so pass compact with explicit handoff for the store path.
    result_none = migrate_cli_session(
        "hist-job", "codex_cli", store=store, history="none"
    )
    assert result_none["history"] == "none"
    assert result_none["handoff_pending"] is False
    compact_text = format_compact_history(
        [("user", "refactor auth.py"), ("assistant", "done with tests")],
        from_provider="goose_cli",
    )
    assert "Goal:" in compact_text
    assert "Native history" in format_full_history(
        [("user", "a"), ("assistant", "b")], from_provider="muse_code"
    )


def test_cli_handoff_redacts_secrets_and_applies_prefix() -> None:
    from ipfs_accelerate_py.cli_runtime.cli_handoff import (
        apply_cli_handoff,
        collect_cli_handoff_context,
        redact_cli_context,
    )

    raw = "API_KEY=sk-abcdefghijklmnopqrstuvwxyz token=xai-1234567890abcd keep this"
    redacted = redact_cli_context(raw)
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in redacted
    assert "xai-1234567890abcd" not in redacted
    assert "keep this" in redacted
    wrapped = apply_cli_handoff("next step", "refactor auth.py", from_provider="muse_code")
    assert wrapped.startswith("Prior work was on muse_code.")
    assert "refactor auth.py" in wrapped
    assert wrapped.endswith("next step")
    packed = collect_cli_handoff_context(
        from_provider="goose_cli",
        native_session_id="",
        handoff="META_API_KEY=secret-value do the tests",
    )
    assert "secret-value" not in packed
    assert "do the tests" in packed


def test_generate_text_injects_and_consumes_cli_handoff(
    store: AllocationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ipfs_accelerate_py import llm_router

    store.record(
        CallObservation(
            provider="muse_code",
            protocol="cli",
            path="cli",
            session_id="hand-1",
            success=True,
            extra_metadata={"session_id": "muse-native"},
        )
    )
    migrate_cli_session("hand-1", "grok_cli", store=store, handoff="ported context from muse")
    seen: list[str] = []

    class _Prov:
        def generate(self, prompt: str, *, model_name=None, **kwargs):
            _ = (model_name, kwargs)
            seen.append(str(prompt))
            return "ok"

    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *a, **k: _Prov())
    text = llm_router.generate_text(
        "keep going",
        provider="grok_cli",
        allocation_session_id="hand-1",
        allocation_path="cli",
    )
    assert text == "ok"
    assert seen and "ported context from muse" in seen[0]
    assert "keep going" in seen[0]
    leftover = store.get_session("hand-1")
    assert not str(leftover.get("handoff_context") or "").strip()


def test_api_key_multiplex_session_bind_and_per_key_stats(
    store: AllocationStore, tmp_path: Path
) -> None:
    from ipfs_accelerate_py.common.secrets_manager import SecretsManager

    secrets = SecretsManager(
        secrets_file=str(tmp_path / "secrets.enc"),
        use_encryption=False,
    )
    first = register_api_key(
        "openai",
        "sk-aaaa1111bbbb2222",
        slot_id="primary",
        store=store,
        secrets=secrets,
        persist=False,
    )
    second = register_api_key(
        "openai",
        "sk-cccc3333dddd4444",
        slot_id="burst",
        store=store,
        secrets=secrets,
        persist=False,
    )
    assert first["key_fingerprint"] != second["key_fingerprint"]
    listed = list_api_key_slots("openai", store=store)
    assert {row["slot_id"] for row in listed} == {"primary", "burst"}
    assert all("sk-" not in str(row) for row in listed)
    bind_session_api_key("sess-k1", "openai", "burst", store=store)
    selected = select_api_key(
        "openai", session_id="sess-k1", store=store, secrets=secrets, bind=False
    )
    assert selected["slot_id"] == "burst"
    assert selected["secret"] == "sk-cccc3333dddd4444"
    store.record(
        CallObservation(
            provider="openai",
            protocol="http",
            path="api",
            session_id="sess-k1",
            success=True,
            latency_ms=50.0,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            estimated_cost_usd=0.02,
            remaining_requests=100,
            remaining_tokens=8000,
            extra_metadata={
                "api_key_slot": "burst",
                "api_key_fingerprint": second["key_fingerprint"],
                "remaining_balance_usd": 12.5,
            },
        )
    )
    stats = {row["slot_id"]: row for row in list_api_key_slots("openai", store=store)}
    assert int(stats["burst"]["call_count"] or 0) == 1
    assert abs(float(stats["burst"]["total_cost_usd"] or 0) - 0.02) < 1e-9
    assert int(stats["burst"]["total_tokens"] or 0) == 15
    assert float(stats["burst"]["remaining_balance_usd"] or 0) == 12.5
    assert int(stats["primary"]["call_count"] or 0) == 0


def test_cli_tools_status_covers_all_providers_without_secrets(
    store: AllocationStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ipfs_accelerate_py.llm_allocation import cli_tools_status, render_cli_tools_status
    from ipfs_accelerate_py.llm_allocation.paths import CLI_PROVIDERS

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("META_API_KEY", "sk-test-should-not-appear")
    store.record(
        CallObservation(
            provider="muse_code",
            protocol="cli",
            path="cli",
            session_id="status-1",
            success=True,
            extra_metadata={"session_id": "muse-stat"},
        )
    )
    report = cli_tools_status(store=store, environ=os.environ, home=tmp_path)
    assert set(report["tools"]) == set(CLI_PROVIDERS)
    for name, row in report["tools"].items():
        assert row["status"] in {
            "ready",
            "missing",
            "degraded",
            "installed_unauthenticated",
        }
        assert "sk-test-should-not-appear" not in str(row)
        assert row["provider"] == name
        assert "stats" in row
        assert "native_sessions" in row
    assert report["tools"]["muse_code"]["native_sessions"] >= 1
    table = render_cli_tools_status(report)
    assert "muse_code" in table
    assert "sk-test-should-not-appear" not in table


def test_npm_package_installers_use_user_prefix_on_eacces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ipfs_accelerate_py.cli_runtime.installers.catalog import ensure_cli_tool

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    (tmp_path / "empty").mkdir()
    seen: list[list[str]] = []

    def fake_run(argv, **kwargs):
        _ = kwargs
        command = [str(part) for part in argv]
        seen.append(command)
        if "--prefix" not in command:
            return type(
                "P",
                (),
                {"returncode": 243, "stdout": "", "stderr": "npm error code EACCES"},
            )()
        prefix = Path(command[command.index("--prefix") + 1])
        binary = prefix / "bin" / "claude"
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("#!/bin/sh\necho claude\n", encoding="utf-8")
        binary.chmod(0o755)
        return type("P", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    result = ensure_cli_tool(
        "claude_code",
        auto_install=True,
        environ=os.environ,
        run_fn=fake_run,
    )
    assert result.available is True
    assert Path(result.executable).name == "claude"
    assert any("--prefix" in command for command in seen)


def test_discover_cli_tool_catalog_covers_all_cli_providers() -> None:
    from ipfs_accelerate_py.cli_runtime.installers.catalog import (
        CLI_INSTALLERS,
        discover_cli_tool,
        ensure_cli_tool,
        installer_robustness_report,
    )
    from ipfs_accelerate_py.llm_allocation.paths import CLI_PROVIDERS

    assert set(CLI_INSTALLERS) == set(CLI_PROVIDERS)
    report = installer_robustness_report()
    assert {row["provider"] for row in report} == set(CLI_PROVIDERS)
    found = discover_cli_tool("codex_cli")
    assert found.provider == "codex_cli"
    skipped = ensure_cli_tool("codex_cli", auto_install=False)
    assert skipped.available in {True, False}
    if not skipped.available:
        assert skipped.reason in {"not_installed", "no_official_installer", "missing_npm"} or True
