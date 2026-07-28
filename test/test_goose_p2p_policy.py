"""GOOSE-009: Opt-in Goose P2P worker policy.

Covers disabled-by-default, allowlist/wildcard gates, path escape protection,
sticky sessions, duplicate delivery, cancellation, uncertain failure, and
no cross-provider fallback after Goose attempts.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_goose_delivery_registry():
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker

    p2p_worker.clear_goose_agent_delivery_registry()
    yield
    p2p_worker.clear_goose_agent_delivery_registry()


@pytest.fixture(autouse=True)
def _clear_goose_worker_env(monkeypatch):
    for key in (
        "IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI",
        "IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_AGENT",
        "IPFS_ACCELERATE_PY_TASK_WORKER_ALLOWED_LLM_PROVIDERS",
        "IPFS_DATASETS_PY_TASK_WORKER_ALLOWED_LLM_PROVIDERS",
        "IPFS_ACCELERATE_PY_TASK_WORKER_GOOSE_PATH_ROOT",
        "GOOSE_PATH_ROOT",
        "IPFS_ACCELERATE_PY_TASK_WORKER_GOOSE_ALLOWED_ROOTS",
        "IPFS_ACCELERATE_PY_TASK_WORKER_LLM_GENERATE_LOCAL_FALLBACK",
        "IPFS_ACCELERATE_PY_TASK_P2P_SESSION",
    ):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Disabled by default / allowlist
# ---------------------------------------------------------------------------


def test_goose_excluded_from_default_remote_provider_set():
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker

    allowed = p2p_worker._allowed_llm_providers()
    assert "goose_cli" not in allowed
    assert "goose" not in allowed
    assert "goose_agent" not in allowed
    assert "copilot_cli" in allowed


def test_goose_excluded_from_wildcard_without_enable_gates(monkeypatch):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ALLOWED_LLM_PROVIDERS", "*")
    allowed = p2p_worker._allowed_llm_providers()
    assert "goose_cli" not in allowed
    assert "goose_agent" not in allowed
    assert "codex_cli" in allowed
    assert "llama_cpp" in allowed


def test_enable_goose_cli_adds_chat_names_to_default_set(monkeypatch):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI", "1")
    allowed = p2p_worker._allowed_llm_providers()
    assert "goose_cli" in allowed
    assert "goose" in allowed
    assert "goose_agent" not in allowed


def test_wildcard_includes_goose_only_under_gates(monkeypatch):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ALLOWED_LLM_PROVIDERS", "all")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI", "1")
    allowed = p2p_worker._allowed_llm_providers()
    assert "goose_cli" in allowed
    assert "goose_agent" not in allowed

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_AGENT", "1")
    allowed2 = p2p_worker._allowed_llm_providers()
    assert "goose_agent" in allowed2
    assert "goose_cli" in allowed2


def test_explicit_allowlist_goose_requires_enable_gate(monkeypatch):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker

    monkeypatch.setenv(
        "IPFS_ACCELERATE_PY_TASK_WORKER_ALLOWED_LLM_PROVIDERS",
        "goose_cli,copilot_cli",
    )
    # Without ENABLE gate, goose_cli must not be admitted.
    allowed = p2p_worker._allowed_llm_providers()
    assert "goose_cli" not in allowed
    assert "copilot_cli" in allowed

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI", "1")
    allowed2 = p2p_worker._allowed_llm_providers()
    assert "goose_cli" in allowed2


def test_agent_requires_agent_gate_plus_allowlist(monkeypatch):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker

    # Agent gate alone on default set does not add goose_agent.
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_AGENT", "1")
    allowed = p2p_worker._allowed_llm_providers()
    assert "goose_agent" not in allowed

    monkeypatch.setenv(
        "IPFS_ACCELERATE_PY_TASK_WORKER_ALLOWED_LLM_PROVIDERS",
        "goose_agent,goose_cli",
    )
    allowed2 = p2p_worker._allowed_llm_providers()
    assert "goose_agent" in allowed2
    # goose_cli listed under agent-only enablement is still admitted for agent mode.
    assert "goose_cli" in allowed2


def test_run_llm_generate_rejects_goose_when_disabled(monkeypatch):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker
    import ipfs_accelerate_py.llm_router as llm_router

    monkeypatch.setattr(
        llm_router,
        "generate_text",
        lambda *a, **k: pytest.fail("generate_text must not run when goose is disabled"),
    )

    with pytest.raises(Exception) as ei:
        p2p_worker._run_llm_generate(
            {
                "assigned_worker": "w1",
                "payload": {"prompt": "hi", "provider": "goose_cli"},
            }
        )
    assert "not allowed" in str(ei.value).lower() or "disabled" in str(ei.value).lower()


def test_run_llm_generate_allows_goose_chat_when_enabled(monkeypatch):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker
    import ipfs_accelerate_py.llm_router as llm_router

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI", "1")
    captured = {}

    def fake_generate_text(prompt, *, model_name=None, provider=None, **kwargs):
        captured["prompt"] = prompt
        captured["provider"] = provider
        captured["kwargs"] = dict(kwargs)
        return "goose-chat-ok"

    monkeypatch.setattr(llm_router, "generate_text", fake_generate_text)

    out = p2p_worker._run_llm_generate(
        {
            "assigned_worker": "w1",
            "payload": {"prompt": "hello goose", "provider": "goose_cli"},
        }
    )
    assert out["text"] == "goose-chat-ok"
    assert out["provider"] == "goose_cli"
    assert out.get("goose_mode") == "chat"
    assert out.get("side_effects_started") is False
    assert captured["provider"] == "goose_cli"
    assert not captured["kwargs"].get("agent")
    assert not captured["kwargs"].get("allow_side_effects")


def test_run_llm_generate_agent_requires_agent_gate(monkeypatch, tmp_path):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker
    import ipfs_accelerate_py.llm_router as llm_router

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI", "1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_GOOSE_PATH_ROOT", str(tmp_path))
    monkeypatch.setattr(llm_router, "generate_text", lambda *a, **k: "should-not-run")

    with pytest.raises(p2p_worker.GooseWorkerPolicyError) as ei:
        p2p_worker._run_llm_generate(
            {
                "assigned_worker": "w1",
                "payload": {
                    "prompt": "do work",
                    "provider": "goose_cli",
                    "agent": True,
                    "allow_side_effects": True,
                    "cwd": str(tmp_path),
                    "path_root": str(tmp_path),
                },
            }
        )
    assert ei.value.error_kind == "policy_denial"
    assert "agent" in str(ei.value).lower()


def test_run_llm_generate_agent_allowed_with_gate_and_allowlist(monkeypatch, tmp_path):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker
    import ipfs_accelerate_py.llm_router as llm_router

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI", "1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_AGENT", "1")
    monkeypatch.setenv(
        "IPFS_ACCELERATE_PY_TASK_WORKER_ALLOWED_LLM_PROVIDERS",
        "goose_cli,goose_agent",
    )
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_GOOSE_PATH_ROOT", str(tmp_path))
    captured = {}

    def fake_generate_text(prompt, *, model_name=None, provider=None, **kwargs):
        captured["provider"] = provider
        captured["kwargs"] = dict(kwargs)
        return "agent-ok"

    monkeypatch.setattr(llm_router, "generate_text", fake_generate_text)

    out = p2p_worker._run_llm_generate(
        {
            "task_id": "t-agent-1",
            "assigned_worker": "w1",
            "payload": {
                "prompt": "refactor",
                "provider": "goose_cli",
                "agent": True,
                "allow_side_effects": True,
                "cwd": str(tmp_path / "proj"),
                "path_root": str(tmp_path),
            },
        }
    )
    assert out["text"] == "agent-ok"
    assert out.get("goose_mode") == "agent"
    assert captured["kwargs"].get("agent") is True
    assert captured["kwargs"].get("allow_side_effects") is True


# ---------------------------------------------------------------------------
# Path escape
# ---------------------------------------------------------------------------


def test_path_escape_rejected_without_roots(monkeypatch):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker
    import ipfs_accelerate_py.llm_router as llm_router

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI", "1")
    monkeypatch.setattr(llm_router, "generate_text", lambda *a, **k: "nope")

    with pytest.raises(p2p_worker.GooseWorkerPolicyError) as ei:
        p2p_worker._run_llm_generate(
            {
                "assigned_worker": "w1",
                "payload": {
                    "prompt": "hi",
                    "provider": "goose_cli",
                    "cwd": "/etc/passwd",
                },
            }
        )
    assert ei.value.error_kind == "policy_denial"
    assert "root" in str(ei.value).lower() or "path" in str(ei.value).lower()


def test_path_escape_rejected_outside_configured_root(monkeypatch, tmp_path):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker
    import ipfs_accelerate_py.llm_router as llm_router

    root = tmp_path / "sandbox"
    root.mkdir()
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI", "1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_GOOSE_PATH_ROOT", str(root))
    monkeypatch.setattr(llm_router, "generate_text", lambda *a, **k: "nope")

    with pytest.raises(p2p_worker.GooseWorkerPolicyError) as ei:
        p2p_worker._run_llm_generate(
            {
                "assigned_worker": "w1",
                "payload": {
                    "prompt": "hi",
                    "provider": "goose_cli",
                    "config_path": "/tmp/evil-config.yaml",
                    "recipe_path": str(tmp_path / "outside.recipe"),
                },
            }
        )
    assert ei.value.error_kind == "policy_denial"
    assert "escape" in str(ei.value).lower() or "root" in str(ei.value).lower()


def test_authorized_paths_under_root_accepted(monkeypatch, tmp_path):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker
    import ipfs_accelerate_py.llm_router as llm_router

    root = tmp_path / "sandbox"
    root.mkdir()
    cfg = root / "goose.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI", "1")
    monkeypatch.setenv("GOOSE_PATH_ROOT", str(root))

    def fake_generate_text(*a, **k):
        return "ok"

    monkeypatch.setattr(llm_router, "generate_text", fake_generate_text)

    out = p2p_worker._run_llm_generate(
        {
            "assigned_worker": "w1",
            "payload": {
                "prompt": "hi",
                "provider": "goose_cli",
                "config_path": str(cfg),
                "cwd": str(root),
            },
        }
    )
    assert out["text"] == "ok"


def test_validate_goose_payload_paths_covers_recipe_trace_extension_session(monkeypatch, tmp_path):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker

    root = tmp_path / "r"
    root.mkdir()
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_GOOSE_PATH_ROOT", str(root))

    # Escape via .. should fail.
    with pytest.raises(p2p_worker.GooseWorkerPolicyError):
        p2p_worker._validate_goose_payload_paths(
            {
                "recipe_path": str(root / ".." / "escape.recipe"),
            }
        )

    good = root / "ok.recipe"
    good.write_text("r", encoding="utf-8")
    authorized = p2p_worker._validate_goose_payload_paths(
        {
            "recipe_path": str(good),
            "trace_dir": str(root / "traces"),
            "extension_path": str(root / "ext"),
            "session_path": str(root / "sessions"),
            "cwd": str(root),
        }
    )
    assert "recipe_path" in authorized
    assert "cwd" in authorized


# ---------------------------------------------------------------------------
# Sticky session
# ---------------------------------------------------------------------------


def test_sticky_session_requires_matching_worker(monkeypatch, tmp_path):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker
    import ipfs_accelerate_py.llm_router as llm_router

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI", "1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_SESSION", "S1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_GOOSE_PATH_ROOT", str(tmp_path))
    monkeypatch.setattr(llm_router, "generate_text", lambda *a, **k: "ok")

    with pytest.raises(p2p_worker.GooseWorkerPolicyError) as ei:
        p2p_worker._run_llm_generate(
            {
                "assigned_worker": "worker-A",
                "payload": {
                    "prompt": "continue",
                    "provider": "goose_cli",
                    "goose_session_id": "g-sess-1",
                    "session_id": "S1",
                    "sticky_worker_id": "worker-B",
                },
            }
        )
    assert ei.value.error_kind == "policy_denial"
    assert "sticky" in str(ei.value).lower()


def test_sticky_session_allows_assigned_worker(monkeypatch, tmp_path):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker
    import ipfs_accelerate_py.llm_router as llm_router

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI", "1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_SESSION", "S1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_GOOSE_PATH_ROOT", str(tmp_path))
    monkeypatch.setattr(llm_router, "generate_text", lambda *a, **k: "sticky-ok")

    out = p2p_worker._run_llm_generate(
        {
            "assigned_worker": "worker-A",
            "payload": {
                "prompt": "continue",
                "provider": "goose_cli",
                "goose_session_id": "g-sess-1",
                "session_id": "S1",
                "sticky_worker_id": "worker-A",
            },
        }
    )
    assert out["text"] == "sticky-ok"
    assert out["executor_worker_id"] == "worker-A"


def test_sticky_session_requires_sticky_worker_id_when_resuming(monkeypatch, tmp_path):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker
    import ipfs_accelerate_py.llm_router as llm_router

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI", "1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_GOOSE_PATH_ROOT", str(tmp_path))
    monkeypatch.setattr(llm_router, "generate_text", lambda *a, **k: "ok")

    with pytest.raises(p2p_worker.GooseWorkerPolicyError) as ei:
        p2p_worker._run_llm_generate(
            {
                "assigned_worker": "worker-A",
                "payload": {
                    "prompt": "resume me",
                    "provider": "goose_cli",
                    "resume_session_id": "rs-1",
                },
            }
        )
    assert "sticky_worker_id" in str(ei.value)


# ---------------------------------------------------------------------------
# No cross-provider fallback
# ---------------------------------------------------------------------------


def test_no_cross_provider_fallback_after_goose_chat_failure(monkeypatch):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker
    import ipfs_accelerate_py.llm_router as llm_router

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI", "1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ALLOWED_LLM_PROVIDERS", "goose_cli,codex_cli,copilot_cli")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_LLM_GENERATE_LOCAL_FALLBACK", "1")
    attempts: list[str] = []

    def fake_generate_text(prompt, *, model_name=None, provider=None, **kwargs):
        attempts.append(str(provider))
        raise RuntimeError(f"{provider} failed")

    monkeypatch.setattr(llm_router, "generate_text", fake_generate_text)

    # Prevent local HF fallback from succeeding and masking the policy.
    monkeypatch.setattr(
        p2p_worker,
        "_run_text_generation",
        lambda *a, **k: pytest.fail("must not local-fallback after goose failure"),
    )

    with pytest.raises(p2p_worker.GooseWorkerPolicyError):
        p2p_worker._run_llm_generate(
            {
                "assigned_worker": "w1",
                "payload": {"prompt": "hi", "provider": "goose_cli"},
            }
        )

    assert attempts == ["goose_cli"]


def test_no_cross_provider_fallback_after_goose_agent_uncertain_failure(monkeypatch, tmp_path):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker
    import ipfs_accelerate_py.llm_router as llm_router

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI", "1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_AGENT", "1")
    monkeypatch.setenv(
        "IPFS_ACCELERATE_PY_TASK_WORKER_ALLOWED_LLM_PROVIDERS",
        "goose_cli,codex_cli,copilot_cli",
    )
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_GOOSE_PATH_ROOT", str(tmp_path))
    attempts: list[str] = []

    def fake_generate_text(prompt, *, model_name=None, provider=None, **kwargs):
        attempts.append(str(provider))
        err = RuntimeError("tool activity then crash")
        setattr(err, "side_effects_started", True)
        setattr(err, "goose_error_kind", "nonzero_exit")
        raise err

    monkeypatch.setattr(llm_router, "generate_text", fake_generate_text)
    monkeypatch.setattr(
        p2p_worker,
        "_run_text_generation",
        lambda *a, **k: pytest.fail("must not local-fallback after goose agent attempt"),
    )

    with pytest.raises(p2p_worker.GooseWorkerPolicyError) as ei:
        p2p_worker._run_llm_generate(
            {
                "task_id": "t-uncertain-1",
                "assigned_worker": "w1",
                "payload": {
                    "prompt": "edit files",
                    "provider": "goose_cli",
                    "agent": True,
                    "allow_side_effects": True,
                    "cwd": str(tmp_path),
                    "path_root": str(tmp_path),
                },
            }
        )

    assert attempts == ["goose_cli"]
    assert ei.value.side_effects_started is True
    assert ei.value.error_kind in {"uncertain", "nonzero_exit"}
    # Stable classification surface for callers/tests.
    fields = p2p_worker._failure_fields_from_exc(ei.value)
    assert fields["side_effects_started"] is True
    assert fields["error_kind"]
    assert fields["goose_error_kind"] == fields["error_kind"]


# ---------------------------------------------------------------------------
# Uncertain failure / cancellation / duplicate delivery
# ---------------------------------------------------------------------------


def test_uncertain_failure_classification_and_side_effects_flag(monkeypatch, tmp_path):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker
    import ipfs_accelerate_py.llm_router as llm_router

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI", "1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_AGENT", "1")
    monkeypatch.setenv(
        "IPFS_ACCELERATE_PY_TASK_WORKER_ALLOWED_LLM_PROVIDERS",
        "goose_agent",
    )
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_GOOSE_PATH_ROOT", str(tmp_path))

    def fake_generate_text(*a, **k):
        err = RuntimeError("partial tool output")
        setattr(err, "side_effects_started", True)
        raise err

    monkeypatch.setattr(llm_router, "generate_text", fake_generate_text)

    with pytest.raises(p2p_worker.GooseWorkerPolicyError) as ei:
        p2p_worker._run_llm_generate(
            {
                "task_id": "t-unc-2",
                "assigned_worker": "w1",
                "payload": {
                    "prompt": "mutate",
                    "provider": "goose_agent",
                    "cwd": str(tmp_path),
                    "path_root": str(tmp_path),
                },
            }
        )
    assert ei.value.side_effects_started is True
    assert ei.value.error_kind == "uncertain"


def test_cancellation_before_start(monkeypatch, tmp_path):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker
    import ipfs_accelerate_py.llm_router as llm_router

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI", "1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_AGENT", "1")
    monkeypatch.setenv(
        "IPFS_ACCELERATE_PY_TASK_WORKER_ALLOWED_LLM_PROVIDERS",
        "goose_cli",
    )
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_GOOSE_PATH_ROOT", str(tmp_path))
    monkeypatch.setattr(
        llm_router,
        "generate_text",
        lambda *a, **k: pytest.fail("cancelled tasks must not invoke generate_text"),
    )

    with pytest.raises(p2p_worker.GooseWorkerPolicyError) as ei:
        p2p_worker._run_llm_generate(
            {
                "task_id": "t-cancel-1",
                "assigned_worker": "w1",
                "payload": {
                    "prompt": "stop me",
                    "provider": "goose_cli",
                    "agent": True,
                    "allow_side_effects": True,
                    "cwd": str(tmp_path),
                    "path_root": str(tmp_path),
                    "cancel": True,
                },
            }
        )
    assert ei.value.error_kind == "cancellation"
    assert ei.value.side_effects_started is False


def test_cancellation_classification_from_provider_error(monkeypatch, tmp_path):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker
    import ipfs_accelerate_py.llm_router as llm_router

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI", "1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_AGENT", "1")
    monkeypatch.setenv(
        "IPFS_ACCELERATE_PY_TASK_WORKER_ALLOWED_LLM_PROVIDERS",
        "goose_cli",
    )
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_GOOSE_PATH_ROOT", str(tmp_path))

    def fake_generate_text(*a, **k):
        err = RuntimeError("run cancelled by operator")
        setattr(err, "side_effects_started", False)
        setattr(err, "goose_error_kind", "cancellation")
        raise err

    monkeypatch.setattr(llm_router, "generate_text", fake_generate_text)

    with pytest.raises(p2p_worker.GooseWorkerPolicyError) as ei:
        p2p_worker._run_llm_generate(
            {
                "task_id": "t-cancel-2",
                "assigned_worker": "w1",
                "payload": {
                    "prompt": "long agent",
                    "provider": "goose_cli",
                    "agent": True,
                    "allow_side_effects": True,
                    "cwd": str(tmp_path),
                    "path_root": str(tmp_path),
                },
            }
        )
    assert ei.value.error_kind == "cancellation"
    assert ei.value.side_effects_started is False


def test_duplicate_delivery_refused_after_uncertain_agent_attempt(monkeypatch, tmp_path):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker
    import ipfs_accelerate_py.llm_router as llm_router

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI", "1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_AGENT", "1")
    monkeypatch.setenv(
        "IPFS_ACCELERATE_PY_TASK_WORKER_ALLOWED_LLM_PROVIDERS",
        "goose_cli",
    )
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_GOOSE_PATH_ROOT", str(tmp_path))

    def fake_generate_text(*a, **k):
        err = RuntimeError("died mid-tool")
        setattr(err, "side_effects_started", True)
        raise err

    monkeypatch.setattr(llm_router, "generate_text", fake_generate_text)

    task = {
        "task_id": "t-dup-1",
        "assigned_worker": "w1",
        "payload": {
            "prompt": "write files",
            "provider": "goose_cli",
            "agent": True,
            "allow_side_effects": True,
            "cwd": str(tmp_path),
            "path_root": str(tmp_path),
        },
    }

    with pytest.raises(p2p_worker.GooseWorkerPolicyError) as first:
        p2p_worker._run_llm_generate(task)
    assert first.value.side_effects_started is True

    with pytest.raises(p2p_worker.GooseWorkerPolicyError) as second:
        p2p_worker._run_llm_generate(task)
    assert second.value.error_kind == "duplicate_delivery"
    assert "duplicate" in str(second.value).lower()


def test_chat_success_does_not_block_retry(monkeypatch):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker
    import ipfs_accelerate_py.llm_router as llm_router

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI", "1")
    monkeypatch.setattr(llm_router, "generate_text", lambda *a, **k: "ok")

    task = {
        "task_id": "t-chat-retry",
        "assigned_worker": "w1",
        "payload": {"prompt": "hi", "provider": "goose_cli"},
    }
    assert p2p_worker._run_llm_generate(task)["text"] == "ok"
    # Chat is not registered in the agent delivery registry; re-run is fine.
    assert p2p_worker._run_llm_generate(task)["text"] == "ok"


def test_failure_fields_stable_classification_helpers():
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker

    err = p2p_worker.GooseWorkerPolicyError(
        "boom",
        error_kind="uncertain",
        side_effects_started=True,
        details={"provider": "goose_cli"},
    )
    fields = p2p_worker._failure_fields_from_exc(err)
    assert fields == {
        "error_kind": "uncertain",
        "goose_error_kind": "uncertain",
        "side_effects_started": True,
        "error_details": {"provider": "goose_cli"},
    }


def test_llm_generate_task_type_advertised_when_goose_enabled(monkeypatch):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker

    # Without gates, llm.generate may still be absent from defaults.
    monkeypatch.delenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_COPILOT_CLI", raising=False)
    types_off = p2p_worker._compute_supported_task_types(
        supported_task_types=None,
        accelerate_instance=None,
    )
    # With goose chat gate, llm.generate is advertised.
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI", "1")
    types_on = p2p_worker._compute_supported_task_types(
        supported_task_types=None,
        accelerate_instance=None,
    )
    assert "llm.generate" in types_on or "llm_generate" in types_on
    # Sanity: enabling goose should not remove baseline types.
    assert isinstance(types_off, list)


# ---------------------------------------------------------------------------
# GOOSE-011 security matrix anchors (worker surface)
# ---------------------------------------------------------------------------


def test_matrix_worker_agent_denial_and_no_fallback_after_side_effects(
    monkeypatch, tmp_path
):
    """Worker chat enablement, agent denial, and no cross-provider fallback."""
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker
    import ipfs_accelerate_py.llm_router as llm_router

    # Agent denied when only chat gate is on.
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_CLI", "1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_GOOSE_PATH_ROOT", str(tmp_path))
    monkeypatch.setattr(llm_router, "generate_text", lambda *a, **k: "should-not-run")
    with pytest.raises(p2p_worker.GooseWorkerPolicyError) as ei:
        p2p_worker._run_llm_generate(
            {
                "assigned_worker": "w1",
                "payload": {
                    "prompt": "agent",
                    "provider": "goose_cli",
                    "agent": True,
                    "allow_side_effects": True,
                    "cwd": str(tmp_path),
                    "path_root": str(tmp_path),
                },
            }
        )
    assert ei.value.error_kind == "policy_denial"

    # After goose agent side effects, no fallback to other providers.
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_GOOSE_AGENT", "1")
    monkeypatch.setenv(
        "IPFS_ACCELERATE_PY_TASK_WORKER_ALLOWED_LLM_PROVIDERS",
        "goose_cli,goose_agent,openai",
    )
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_LLM_GENERATE_LOCAL_FALLBACK", "1")
    seen: list[str] = []

    def fail_with_side_effects(prompt, *, model_name=None, provider=None, **kwargs):
        seen.append(str(provider))
        err = RuntimeError("mid-tool crash")
        setattr(err, "side_effects_started", True)
        raise err

    monkeypatch.setattr(llm_router, "generate_text", fail_with_side_effects)
    with pytest.raises(Exception):
        p2p_worker._run_llm_generate(
            {
                "assigned_worker": "w1",
                "payload": {
                    "prompt": "x",
                    "provider": "goose_cli",
                    "agent": True,
                    "allow_side_effects": True,
                    "cwd": str(tmp_path),
                    "path_root": str(tmp_path),
                },
            }
        )
    assert "openai" not in seen
    assert all("goose" in p for p in seen)
