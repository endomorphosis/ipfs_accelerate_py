"""WPD-023: Residual provider invocation under sealed packet only.

Acceptance (from the sealed WPD board):

* Oversized or path-escaping packets rejected.
* Provider env excludes secrets.
* Packet CID logged.
* Evidence subset: prompt body bounds, forbidden full-task dump, path lease.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.residual_llm_packet import (
    DEFAULT_MAX_PACKET_BYTES,
    seal_residual_llm_packet,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.residual_provider_invocation import (
    DEFAULT_MAX_PROMPT_BYTES,
    EVENT_RESIDUAL_PROVIDER_INVOKED,
    EVENT_RESIDUAL_PROVIDER_PREPARED,
    RESIDUAL_PROVIDER_INVOCATION_EVIDENCE,
    RESIDUAL_PROVIDER_INVOCATION_INTERFACE,
    RESIDUAL_PROVIDER_INVOCATION_VERSION,
    PathLease,
    ResidualProviderBudgetError,
    ResidualProviderEnvError,
    ResidualProviderInvocation,
    ResidualProviderInvocationError,
    ResidualProviderPathError,
    ResidualProviderReason,
    assert_provider_env_excludes_secrets,
    build_provider_environment,
    build_residual_provider_invocation,
    invoke_residual_provider,
    is_secret_env_key,
    prepare_residual_provider_context,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _capsule(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema": "ipfs_accelerate_py/agent-supervisor/counterexample-context-capsule@1",
        "target_ids": ["symbol:target"],
        "counterexamples": [
            {
                "counterexample_id": "cex:wpd-023",
                "kind": "generic_failure",
                "summary": "focused residual repair required",
                "violated_property": "acceptance must hold",
            }
        ],
        "nodes": [],
        "edges": [],
        "usage": {
            "counterexamples": 1,
            "graph_nodes": 0,
            "graph_edges": 0,
            "encoded_bytes": 128,
            "omitted_counterexamples": 0,
        },
        "limits": {"max_bytes": 4096},
        "minimized": True,
        "redacted": True,
        "contains_private_material": False,
        "contains_raw_prover_output": False,
        "contains_source": False,
    }
    base.update(overrides)
    return base


def _seal(**overrides: object):
    base: dict[str, object] = {
        "task_id": "WPD-023",
        "repository_id": "repository:sha256:wpd-023",
        "tree_id": "tree:wpd-023",
        "write_paths": (
            "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/"
            "todo_daemon/residual_provider_invocation.py",
        ),
        "obligation_ids": ("obligation:residual-provider-seal",),
        "counterexample_capsule": _capsule(),
        "validation_commands": (
            "python3 -m pytest external/ipfs_accelerate/test/api/"
            "test_agent_supervisor_residual_provider_invocation.py -q",
        ),
        "policy_id": "policy:implementation-daemon",
        "policy_revision": "sha256:policy-wpd-023",
        "forest_id": "forest:wpd-023",
        "acceptance_ids": ("wpd/residual-provider-invocation@1",),
        "authority_roots": {
            "repository_id": "repository:sha256:wpd-023",
            "tree_id": "tree:wpd-023",
        },
    }
    base.update(overrides)
    return seal_residual_llm_packet(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Interface / discovery / cold source
# ---------------------------------------------------------------------------


def test_interface_and_evidence_identity_are_stable() -> None:
    assert RESIDUAL_PROVIDER_INVOCATION_INTERFACE == "ResidualProviderInvocation@1"
    assert RESIDUAL_PROVIDER_INVOCATION_VERSION == 1
    assert RESIDUAL_PROVIDER_INVOCATION_EVIDENCE == "wpd/residual-provider-invocation@1"
    discovery = ResidualProviderInvocation.discovery()
    assert discovery["interface"] == RESIDUAL_PROVIDER_INVOCATION_INTERFACE
    assert discovery["evidence_key"] == RESIDUAL_PROVIDER_INVOCATION_EVIDENCE
    assert discovery["adds_providers"] is False
    assert discovery["wraps_existing_provider_execution"] is True
    assert discovery["sealed_fields_only"] is True
    assert discovery["full_task_dump_forbidden"] is True
    assert discovery["secrets_excluded_from_env"] is True
    assert discovery["packet_cid_logged"] is True
    assert discovery["oversized_packets_rejected"] is True
    assert discovery["path_escaping_packets_rejected"] is True
    assert discovery["max_prompt_bytes"] == DEFAULT_MAX_PROMPT_BYTES
    assert DEFAULT_MAX_PROMPT_BYTES == DEFAULT_MAX_PACKET_BYTES


def test_cold_source_has_no_llm_client_imports() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        residual_provider_invocation as mod,
    )

    source = inspect.getsource(mod)
    for marker in (
        "openai",
        "anthropic",
        "litellm",
        "grok_cli",
        "import requests",
        "import httpx",
    ):
        assert marker not in source


def test_does_not_register_new_providers() -> None:
    discovery = ResidualProviderInvocation.discovery()
    assert discovery["adds_providers"] is False
    # Module must not invent provider factories — only wrap callables.
    source = inspect.getsource(
        __import__(
            "ipfs_accelerate_py.agent_supervisor.todo_daemon.residual_provider_invocation",
            fromlist=["*"],
        )
    )
    assert "register_provider" not in source
    assert "new_provider" not in source


# ---------------------------------------------------------------------------
# Happy path: sealed fields only + packet CID logged
# ---------------------------------------------------------------------------


def test_prepare_receives_only_sealed_residual_fields(caplog: pytest.LogCaptureFixture) -> None:
    packet = _seal()
    invoker = build_residual_provider_invocation()
    with caplog.at_level(logging.INFO, logger=invoker.log.name):
        context = invoker.prepare(
            packet,
            base_env={"PATH": "/usr/bin", "HOME": "/tmp/home", "OPENAI_API_KEY": "sk-secret"},
            attempt=2,
        )

    assert context.packet_cid == packet.packet_id
    assert context.task_id == "WPD-023"
    assert context.write_paths == packet.write_paths
    assert context.obligation_ids == packet.obligation_ids
    assert context.validation_commands == packet.validation_commands
    assert context.nomination_only is True
    assert context.semantic_authority is False
    assert context.write_authority is False
    assert context.completion_authority is False

    prompt = context.prompt_body
    assert packet.packet_id in prompt
    assert "obligation:residual-provider-seal" in prompt
    assert "write_paths" in prompt
    # Forbidden full-task dump material must not appear.
    for forbidden in (
        "full_task_body",
        "task_prose",
        "repository_dump",
        "source_body",
        "OPENAI_API_KEY",
        "sk-secret",
    ):
        assert forbidden not in prompt

    payload = context.to_dict()
    assert payload["contains_full_task_dump"] is False
    assert payload["contains_secrets_in_env"] is False
    assert payload["packet_cid"] == packet.packet_id
    assert "prompt_body" not in payload  # body-free diagnostics surface

    # Packet CID logged.
    assert any(packet.packet_id in record.message for record in caplog.records)
    assert any("residual_packet_cid=" in record.message for record in caplog.records)


def test_invoke_wraps_existing_provider_and_logs_packet_cid(
    caplog: pytest.LogCaptureFixture,
) -> None:
    packet = _seal()
    captured: dict[str, Any] = {}

    def existing_provider(
        *,
        prompt: str,
        env: dict[str, str],
        argv_bindings: dict[str, str],
        packet_cid: str,
        **_kwargs: Any,
    ) -> dict[str, str]:
        captured["prompt"] = prompt
        captured["env"] = env
        captured["argv_bindings"] = argv_bindings
        captured["packet_cid"] = packet_cid
        return {"status": "ok", "packet_cid": packet_cid}

    invoker = build_residual_provider_invocation()
    with caplog.at_level(logging.INFO, logger=invoker.log.name):
        receipt, result = invoker.invoke(
            packet,
            existing_provider,
            base_env={
                "PATH": "/usr/bin",
                "HOME": "/tmp/h",
                "XAI_API_KEY": "xai-secret",
                "PASSWORD": "hunter2",
            },
            path_lease=PathLease(
                permitted_write_paths=packet.write_paths,
                lease_id="lease:wpd-023",
            ),
        )

    assert receipt.invoked is True
    assert receipt.prepared is True
    assert receipt.provider_hook_count == 1
    assert receipt.packet_cid == packet.packet_id
    assert receipt.path_lease_id == "lease:wpd-023"
    assert receipt.reason_code == ResidualProviderReason.INVOKED.value
    assert packet.packet_id in receipt.log_records[0]
    assert result["status"] == "ok"
    assert captured["packet_cid"] == packet.packet_id
    assert packet.packet_id in captured["prompt"]
    assert "XAI_API_KEY" not in captured["env"]
    assert "PASSWORD" not in captured["env"]
    assert captured["env"]["IPFS_ACCELERATE_AGENT_RESIDUAL_PACKET_CID"] == packet.packet_id
    assert captured["argv_bindings"]["packet_cid"] == packet.packet_id
    assert any(packet.packet_id in record.message for record in caplog.records)

    event = receipt.to_event_payload(task_id="WPD-023", attempt=1)
    assert event["event"] == EVENT_RESIDUAL_PROVIDER_INVOKED
    assert event["packet_cid"] == packet.packet_id


def test_prepare_receipt_event_names_packet_cid() -> None:
    packet = _seal()
    context = prepare_residual_provider_context(
        packet,
        base_env={"PATH": "/usr/bin"},
    )
    invoker = build_residual_provider_invocation()
    receipt = invoker.prepare_receipt(context)
    assert receipt.prepared is True
    assert receipt.invoked is False
    assert receipt.packet_cid == packet.packet_id
    event = receipt.to_event_payload(task_id="WPD-023")
    assert event["event"] == EVENT_RESIDUAL_PROVIDER_PREPARED
    assert event["packet_cid"] == packet.packet_id


# ---------------------------------------------------------------------------
# Oversized packets / prompts rejected
# ---------------------------------------------------------------------------


def test_oversized_packet_rejected_at_coerce() -> None:
    huge_commands = tuple(f"python3 -m pytest test_{i}.py -q" for i in range(40))
    with pytest.raises(ResidualProviderBudgetError) as excinfo:
        invoker = build_residual_provider_invocation()
        # Build a mapping that ResidualLlmPacket itself rejects as over-budget.
        invoker.coerce_packet(
            {
                "task_id": "WPD-023",
                "repository_id": "repository:sha256:wpd-023",
                "tree_id": "tree:wpd-023",
                "write_paths": ["pkg/mod.py"],
                "obligation_ids": ["obligation:a"],
                "counterexample_capsule": _capsule(),
                "validation_commands": list(huge_commands),
                "limits": {
                    "max_bytes": 1024,
                    "max_tokens": 256,
                    "max_capsule_bytes": 1024,
                },
            }
        )
    assert excinfo.value.reason_code == ResidualProviderReason.OVERSIZED_PACKET.value


def test_oversized_prompt_rejected() -> None:
    packet = _seal()
    # Measure real sealed size, then tighten below it (still within constructor floor).
    baseline = build_residual_provider_invocation()
    _, byte_count, token_count = baseline.build_sealed_prompt(packet)
    assert byte_count > 256
    tight_bytes = max(256, byte_count // 2)
    tight_tokens = max(64, token_count // 2)
    invoker = ResidualProviderInvocation(
        max_prompt_bytes=tight_bytes,
        max_prompt_tokens=tight_tokens,
    )
    with pytest.raises(ResidualProviderBudgetError) as excinfo:
        invoker.build_sealed_prompt(packet)
    assert excinfo.value.reason_code == ResidualProviderReason.OVERSIZED_PROMPT.value


def test_module_helper_rejects_oversized_prompt() -> None:
    packet = _seal()
    baseline = build_residual_provider_invocation()
    _, byte_count, _ = baseline.build_sealed_prompt(packet)
    with pytest.raises(ResidualProviderBudgetError):
        prepare_residual_provider_context(
            packet,
            base_env={"PATH": "/usr/bin"},
            max_prompt_bytes=max(256, byte_count // 2),
        )


# ---------------------------------------------------------------------------
# Path-escaping packets rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_path",
    [
        "../secrets.env",
        "/etc/passwd",
        "pkg/../other.py",
        "pkg//mod.py",
        "pkg/",
        ".",
    ],
)
def test_path_escaping_packet_rejected(bad_path: str) -> None:
    invoker = build_residual_provider_invocation()
    with pytest.raises(ResidualProviderPathError) as excinfo:
        invoker.coerce_packet(
            {
                "task_id": "WPD-023",
                "repository_id": "repository:sha256:wpd-023",
                "tree_id": "tree:wpd-023",
                "write_paths": [bad_path],
                "obligation_ids": ["obligation:a"],
                "counterexample_capsule": _capsule(),
                "validation_commands": ["pytest -q"],
            }
        )
    assert excinfo.value.reason_code == ResidualProviderReason.PATH_ESCAPE.value


def test_path_lease_mismatch_rejected() -> None:
    packet = _seal()
    invoker = build_residual_provider_invocation()
    lease = PathLease(
        permitted_write_paths=("pkg/other.py",),
        lease_id="lease:narrow",
    )
    with pytest.raises(ResidualProviderPathError) as excinfo:
        invoker.prepare(packet, path_lease=lease, base_env={"PATH": "/usr/bin"})
    assert excinfo.value.reason_code == ResidualProviderReason.PATH_LEASE_MISMATCH.value


def test_require_path_lease_without_lease_rejected() -> None:
    packet = _seal()
    invoker = build_residual_provider_invocation(require_path_lease=True)
    with pytest.raises(ResidualProviderPathError) as excinfo:
        invoker.prepare(packet, base_env={"PATH": "/usr/bin"})
    assert excinfo.value.reason_code == ResidualProviderReason.PATH_LEASE_MISMATCH.value


def test_path_lease_covers_write_paths() -> None:
    packet = _seal()
    context = prepare_residual_provider_context(
        packet,
        path_lease=list(packet.write_paths),
        base_env={"PATH": "/usr/bin"},
    )
    assert context.write_paths == packet.write_paths
    assert all(path in packet.write_paths for path in context.write_paths)


def test_path_lease_rejects_escaping_permitted_path() -> None:
    with pytest.raises(ResidualProviderPathError):
        PathLease(permitted_write_paths=("../outside.py",))


# ---------------------------------------------------------------------------
# Provider env excludes secrets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "OPENAI_API_KEY",
        "XAI_API_KEY",
        "MODEL_API_KEY",
        "PASSWORD",
        "SECRET_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "GITHUB_TOKEN",
        "authorization",
        "my_private_key",
        "session_cookie",
    ],
)
def test_is_secret_env_key_detects_secret_shapes(key: str) -> None:
    assert is_secret_env_key(key) is True


def test_is_secret_env_key_allows_safe_keys() -> None:
    for key in ("PATH", "HOME", "LANG", "USER", "IPFS_ACCELERATE_AGENT_TASK_ID"):
        assert is_secret_env_key(key) is False


def test_build_provider_environment_excludes_secrets() -> None:
    env = build_provider_environment(
        {
            "PATH": "/usr/bin",
            "HOME": "/tmp/h",
            "LANG": "C",
            "OPENAI_API_KEY": "sk-live",
            "XAI_API_KEY": "xai-live",
            "PASSWORD": "hunter2",
            "AWS_SECRET_ACCESS_KEY": "aws-secret",
            "RANDOM_APP_SETTING": "drop-me",
            "IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR": "/tmp/ck",
        },
        packet_cid="cid:packet-1",
        task_id="WPD-023",
        attempt=3,
    )
    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == "/tmp/h"
    assert env["IPFS_ACCELERATE_AGENT_RESIDUAL_PACKET_CID"] == "cid:packet-1"
    assert env["IPFS_ACCELERATE_AGENT_TASK_ID"] == "WPD-023"
    assert env["IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ATTEMPT"] == "3"
    assert env["IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR"] == "/tmp/ck"
    for secret in (
        "OPENAI_API_KEY",
        "XAI_API_KEY",
        "PASSWORD",
        "AWS_SECRET_ACCESS_KEY",
        "RANDOM_APP_SETTING",
    ):
        assert secret not in env
    assert_provider_env_excludes_secrets(env)


def test_extra_env_secret_rejected() -> None:
    with pytest.raises(ResidualProviderEnvError) as excinfo:
        build_provider_environment(
            {"PATH": "/usr/bin"},
            extra={"OPENAI_API_KEY": "sk-injected"},
        )
    assert excinfo.value.reason_code == ResidualProviderReason.SECRET_IN_ENV.value


def test_assert_provider_env_excludes_secrets_fails_closed() -> None:
    with pytest.raises(ResidualProviderEnvError):
        assert_provider_env_excludes_secrets({"PATH": "/usr/bin", "API_KEY": "x"})


# ---------------------------------------------------------------------------
# Forbidden full-task dump
# ---------------------------------------------------------------------------


def test_full_task_dump_in_provider_kwargs_rejected() -> None:
    packet = _seal()

    def provider(**_kwargs: Any) -> None:
        return None

    with pytest.raises(ResidualProviderInvocationError) as excinfo:
        invoke_residual_provider(
            packet,
            provider,
            base_env={"PATH": "/usr/bin"},
            provider_kwargs={"full_task_body": "dump the entire backlog task here"},
        )
    assert (
        excinfo.value.reason_code
        == ResidualProviderReason.FULL_TASK_DUMP_FORBIDDEN.value
    )


def test_missing_packet_rejected() -> None:
    invoker = build_residual_provider_invocation()
    with pytest.raises(ResidualProviderInvocationError) as excinfo:
        invoker.coerce_packet(None)  # type: ignore[arg-type]
    assert excinfo.value.reason_code == ResidualProviderReason.PACKET_REQUIRED.value


def test_provider_not_configured_rejected() -> None:
    packet = _seal()
    invoker = build_residual_provider_invocation()
    with pytest.raises(ResidualProviderInvocationError) as excinfo:
        invoker.invoke(packet, None)  # type: ignore[arg-type]
    assert (
        excinfo.value.reason_code
        == ResidualProviderReason.PROVIDER_NOT_CONFIGURED.value
    )


# ---------------------------------------------------------------------------
# Prompt body bounds evidence
# ---------------------------------------------------------------------------


def test_prompt_body_within_default_bounds() -> None:
    packet = _seal()
    invoker = build_residual_provider_invocation()
    prompt, byte_count, tokens = invoker.build_sealed_prompt(packet)
    assert byte_count == len(prompt.encode("utf-8"))
    assert byte_count <= DEFAULT_MAX_PROMPT_BYTES
    assert tokens <= invoker.max_prompt_tokens
    assert byte_count > 0
    assert "sealed_fields_only" in prompt
    assert "full_task_dump_forbidden" in prompt


def test_context_identity_is_content_addressed() -> None:
    packet = _seal()
    first = prepare_residual_provider_context(packet, base_env={"PATH": "/usr/bin"})
    second = prepare_residual_provider_context(packet, base_env={"PATH": "/usr/bin"})
    assert first.content_id == second.content_id
    assert first.packet_cid == second.packet_cid == packet.packet_id


def test_round_trip_mapping_packet() -> None:
    packet = _seal()
    context = prepare_residual_provider_context(
        packet.to_dict(),
        base_env={"PATH": "/usr/bin", "HOME": "/tmp"},
    )
    assert context.packet_cid == packet.packet_id
    assert context.write_paths == packet.write_paths
