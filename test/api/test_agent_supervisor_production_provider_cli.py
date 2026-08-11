"""Focused production Grok-implementation/Codex-review CLI adapter tests."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor import (
    implementation_supervisor_command,
    plan_bundle_lanes,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    legacy_landed_provider_cli as native_cli,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.contract_packet_provider_router import (
    ImplementationProviderRouter,
    ProviderBounds,
    ProviderRequest,
    ProviderRole,
    ReviewPresence,
    RouteStatus,
    bind_applied_patch_to_review_chain,
    build_production_contract_packet,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
    parse_task_file,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    parse_args as parse_daemon_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    supervisor_config_from_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    parse_args as parse_supervisor_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.llm import (
    LLM_USAGE_MODE_ENFORCE,
    LlmChildResultEnvelope,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_provider_attestation import (
    DEFAULT_PRODUCTION_PROVIDER_REVIEW_KEY_NAME,
    ProductionProviderReviewAuthority,
    verify_production_provider_review_attestation,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_provider_cli import (
    PRODUCTION_CLI_EXECUTION_SCHEMA,
    PRODUCTION_CLI_POLICY_NAME,
    BoundProductionCLIProvider,
    ProductionCLIProviderPolicy,
    build_production_cli_provider_pair,
    production_cli_policy_readiness,
    production_landed_task_guard,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_reviewed_effect import (
    finalize_production_reviewed_effect,
)


def _child_receipt(config) -> LlmChildResultEnvelope:
    return LlmChildResultEnvelope(
        usage_mode=LLM_USAGE_MODE_ENFORCE,
        request_id=config.request_id,
        attempt=config.attempt,
        idempotency_key=config.idempotency_key,
        status="ok",
        effective_provider=str(config.provider or ""),
        text_chars=1,
        exit_code=0,
    )


def _request(role: ProviderRole) -> ProviderRequest:
    prompt = json.dumps(
        {
            "role": role.value,
            "task_id": "ASE-005",
            "provider_input": {"bounded": True},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ProviderRequest(
        role=role,
        packet_id="packet:ase-005",
        snapshot_id="git-commit:fixture",
        task_id="ASE-005",
        payload={"bounded": True},
        bounds=ProviderBounds(),
        prompt=prompt,
        prompt_tokens=32,
    )


def _native_request(role: ProviderRole) -> ProviderRequest:
    payload: dict[str, Any]
    if role is ProviderRole.GROK_IMPLEMENT:
        payload = {
            "contract_packet": {
                "scope": {"write_paths": ["module.py"]},
            }
        }
    else:
        payload = {"admitted_implementation_proposal": {"bounded": True}}
    prompt = json.dumps(
        {
            "role": role.value,
            "task_id": "ASE-005",
            "provider_input": payload,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return ProviderRequest(
        role=role,
        packet_id="packet:ase-005",
        snapshot_id="git-commit:fixture",
        task_id="ASE-005",
        payload=payload,
        bounds=ProviderBounds(),
        prompt=prompt,
        prompt_tokens=32,
    )


def test_policy_is_fixed_to_grok_implementation_and_independent_codex_review() -> None:
    policy = ProductionCLIProviderPolicy()
    payload = policy.to_dict()

    assert policy.name == PRODUCTION_CLI_POLICY_NAME
    assert payload["schema"].endswith("@2")
    assert policy.declared_roles == ("grok-implement", "codex-review")
    assert payload["implementation"]["provider"] == "grok_cli"
    assert payload["implementation"]["model"] == "grok-4.5"
    assert payload["review"]["provider"] == "codex_cli"
    assert payload["review"]["model"] == "gpt-5.6-terra"
    assert payload["review"]["reasoning_effort"] == "medium"
    assert payload["review"]["independent"] is True
    assert payload["implementation"]["fallback_provider"] == ""
    assert payload["implementation"]["failure_disposition"] == (
        "provider_review_pending"
    )
    assert payload["codex_implementation_fallback"] == {
        "enabled": False,
        "reason": "codex_cannot_implement_and_independently_self_review",
        "without_third_reviewer": "provider_review_pending",
    }
    assert payload["landed_task_recovery"] == {
        "blind_reimplementation_allowed": False,
        "review_only_requires_supervisor_observed_grok_provenance": True,
        "missing_typed_receipt": "provider_review_pending_no_reimplementation",
        "legacy_fallback_counts_as_review": False,
    }
    assert payload["task_metadata_mutated"] is False
    assert payload["policy_id"].startswith("sha256:")
    assert payload["completion_authoritative"] is False
    assert policy.provider_timeout_seconds == 300.0

    changed = ProductionCLIProviderPolicy(context_budget_tokens=2048)
    assert changed.policy_id != policy.policy_id


def test_exact_landed_binding_stays_pending_without_blind_reimplementation() -> None:
    guard = production_landed_task_guard(
        recovered_binding={
            "recovered": True,
            "implementation_commit": "a" * 40,
            "prior_merge_commit": "b" * 40,
        },
        workspace_clean=True,
        typed_provider_receipt_available=False,
    )

    assert guard["guarded"] is True
    assert guard["action"] == "provider_review_pending_no_reimplementation"
    assert guard["invoke_grok_implementation"] is False
    assert guard["invoke_codex_review"] is False
    assert guard["provider_review_pending"] is True
    assert guard["legacy_fallback_counts_as_review"] is False


def test_unlanded_task_routes_but_dirty_landed_task_stays_guarded() -> None:
    absent = production_landed_task_guard(
        recovered_binding={"recovered": False},
        workspace_clean=True,
    )
    dirty = production_landed_task_guard(
        recovered_binding={"recovered": True},
        workspace_clean=False,
    )

    assert absent["guarded"] is False
    assert dirty["guarded"] is True
    assert absent["invoke_grok_implementation"] is True
    assert dirty["invoke_grok_implementation"] is False
    assert dirty["reason"] == "landed_binding_workspace_not_clean"


def test_board_metadata_edit_keeps_task_cid_but_invalidates_source_digest(
    tmp_path: Path,
) -> None:
    board = tmp_path / "prompt-entrypoints.todo.md"
    base = """# Tasks

## ASE-005 Implement target resolution

- Status: ready
- Completion: manual
- Priority: P0
- Track: target
- Outputs: module.py
- Validation: python -m pytest test_module.py -q
- Acceptance: target resolution works
"""
    board.write_text(base, encoding="utf-8")
    before = parse_task_file(board, "## ASE-")[0]
    before_digest = hashlib.sha256(board.read_bytes()).hexdigest()

    board.write_text(
        base
        + "- Provider role: grok-implement, codex-review\n"
        + "- Context budget tokens: 4096\n",
        encoding="utf-8",
    )
    after = parse_task_file(board, "## ASE-")[0]
    after_digest = hashlib.sha256(board.read_bytes()).hexdigest()

    assert after.canonical_task_key == before.canonical_task_key
    assert after.canonical_task_cid == before.canonical_task_cid
    assert after_digest != before_digest


@pytest.mark.parametrize(
    "overrides",
    [
        {"context_budget_tokens": 0},
        {"provider_timeout_seconds": 0},
        {"provider_timeout_seconds": 601},
        {"max_new_tokens": 0},
        {"codex_provider": "grok_cli"},
        {"grok_model": "grok-4"},
        {"codex_model": "gpt-5.6-sol"},
        {"codex_reasoning_effort": "high"},
        {"name": "unreviewed-grok"},
    ],
)
def test_policy_rejects_unbounded_or_non_independent_configuration(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        ProductionCLIProviderPolicy(**overrides)


def test_adapter_binds_receipt_and_uses_empty_non_repository_cwd() -> None:
    observed: dict[str, Any] = {}

    def invoke(prompt, config):
        observed["prompt"] = prompt
        observed["repo_root"] = config.repo_root
        observed["children"] = tuple(config.repo_root.iterdir())
        observed["provider"] = config.provider
        observed["required"] = config.required_effective_providers
        observed["cross_provider_fallback"] = (
            config.allow_cross_provider_fallback
        )
        return (
            json.dumps(
                {
                    "proposal": {
                        "declared_paths": ["module.py"],
                        "files": [{"path": "module.py", "content": "ok\n"}],
                    }
                }
            ),
            _child_receipt(config),
        )

    policy = ProductionCLIProviderPolicy()
    provider = BoundProductionCLIProvider(
        policy=policy,
        role=ProviderRole.GROK_IMPLEMENT,
        provider_name=policy.grok_provider,
        model_name=policy.grok_model,
        invoker=invoke,
    )
    response = provider(_request(ProviderRole.GROK_IMPLEMENT))

    execution = response["supervisor_provider_execution"]
    assert execution["schema"] == PRODUCTION_CLI_EXECUTION_SCHEMA
    assert execution["policy_id"] == policy.policy_id
    assert execution["effective_provider"] == "grok_cli"
    assert execution["model_output_authored_receipt"] is False
    assert execution["repository_checkout_used_as_working_directory"] is False
    assert execution["operating_system_filesystem_confinement"] is False
    assert observed["provider"] == "grok_cli"
    assert observed["required"] == ("grok_cli",)
    assert observed["cross_provider_fallback"] is False
    assert observed["children"] == ()
    assert Path(observed["repo_root"]).exists() is False
    assert observed["prompt"] == _request(ProviderRole.GROK_IMPLEMENT).prompt.decode()


def test_native_pair_uses_request_bound_strict_schemas_and_exact_cli_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, dict[str, Any]] = {}
    requests = {
        role: _native_request(role)
        for role in (ProviderRole.GROK_IMPLEMENT, ProviderRole.CODEX_REVIEW)
    }
    monkeypatch.setattr(
        native_cli.shutil,
        "which",
        lambda name: f"/trusted/bin/{name}",
    )

    def run_cli(
        command: Any,
        *,
        cwd: Path,
        timeout_seconds: int,
        stdin_text: str | None = None,
    ) -> tuple[str, str]:
        cmd = list(command)
        provider = "grok_cli" if cmd[0].endswith("/grok") else "codex_cli"
        role = (
            ProviderRole.GROK_IMPLEMENT
            if provider == "grok_cli"
            else ProviderRole.CODEX_REVIEW
        )
        request = requests[role]
        if provider == "grok_cli":
            schema = json.loads(cmd[cmd.index("--json-schema") + 1])
            prompt_path = Path(cmd[cmd.index("--prompt-file") + 1])
            prompt_envelope = json.loads(prompt_path.read_text(encoding="utf-8"))
            assert prompt_envelope["type"] == "acp"
            assert len(prompt_envelope["content"]) == 1
            assert prompt_envelope["content"][0]["type"] == "text"
            prompt = prompt_envelope["content"][0]["text"]
            response = {
                "packet_id": request.packet_id,
                "snapshot_id": request.snapshot_id,
                "task_id": request.task_id,
                "proposal": {
                    "declared_paths": ["module.py"],
                    "files": [
                        {"path": "module.py", "content": "VALUE = 1\n"}
                    ],
                    "patch": "",
                },
            }
            output = json.dumps(
                {"text": response, "requestId": "exact-grok-request"}
            )
            assert stdin_text is None
        else:
            schema_path = Path(cmd[cmd.index("--output-schema") + 1])
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            prompt = str(stdin_text)
            response = {
                "packet_id": request.packet_id,
                "snapshot_id": request.snapshot_id,
                "task_id": request.task_id,
                "decision": "approve",
                "findings": [],
            }
            response_path = Path(cmd[cmd.index("--output-last-message") + 1])
            response_path.write_text(json.dumps(response), encoding="utf-8")
            output = '{"type":"turn.completed"}\n'
        observed[provider] = {
            "command": cmd,
            "cwd": cwd,
            "timeout_seconds": timeout_seconds,
            "prompt": prompt,
            "schema": schema,
        }
        return output, ""

    monkeypatch.setattr(native_cli, "_run_native_cli_process", run_cli)
    grok, codex = build_production_cli_provider_pair()
    grok_response = grok(requests[ProviderRole.GROK_IMPLEMENT])
    codex_response = codex(requests[ProviderRole.CODEX_REVIEW])

    for provider, expected_model, role in (
        ("grok_cli", "grok-4.5", ProviderRole.GROK_IMPLEMENT),
        ("codex_cli", "gpt-5.6-terra", ProviderRole.CODEX_REVIEW),
    ):
        record = observed[provider]
        command = record["command"]
        schema = record["schema"]
        request = requests[role]
        assert command[command.index("--model") + 1] == expected_model
        assert request.prompt.decode("utf-8") == record["prompt"]
        assert record["cwd"].exists() is False
        assert record["timeout_seconds"] == 119
        assert schema["additionalProperties"] is False
        assert schema["properties"]["packet_id"]["enum"] == [
            request.packet_id
        ]
        assert schema["properties"]["snapshot_id"]["enum"] == [
            request.snapshot_id
        ]
        assert schema["properties"]["task_id"]["enum"] == [request.task_id]
    grok_schema = observed["grok_cli"]["schema"]
    proposal_schema = grok_schema["properties"]["proposal"]
    assert "anyOf" in proposal_schema
    assert len(proposal_schema["anyOf"]) == 2
    assert all(
        branch.get("additionalProperties") is False
        for branch in proposal_schema["anyOf"]
    )
    assert proposal_schema["properties"]["declared_paths"]["items"][
        "enum"
    ] == ["module.py"]
    anyof_required = {
        frozenset(branch["required"]) for branch in proposal_schema["anyOf"]
    }
    assert anyof_required == {
        frozenset({"declared_paths", "files"}),
        frozenset({"declared_paths", "patch"}),
    }
    assert observed["grok_cli"]["command"][
        observed["grok_cli"]["command"].index("--tools") + 1
    ] == ""
    assert "--verbatim" in observed["grok_cli"]["command"]
    assert "--output-schema" in observed["codex_cli"]["command"]
    codex_command = observed["codex_cli"]["command"]
    assert codex_command[codex_command.index("-c") + 1] == (
        'model_reasoning_effort="medium"'
    )
    assert observed["codex_cli"]["command"][-1] == "-"
    assert observed["codex_cli"]["schema"]["properties"]["findings"][
        "maxItems"
    ] == 0

    for response, expected_provider, expected_model in (
        (grok_response, "grok_cli", "grok-4.5"),
        (codex_response, "codex_cli", "gpt-5.6-terra"),
    ):
        execution = response["supervisor_provider_execution"]
        assert execution["effective_provider"] == expected_provider
        assert execution["effective_model"] == expected_model
        assert execution["model_reasoning_effort"] == (
            "medium" if expected_provider == "codex_cli" else ""
        )
        assert execution["native_structured_output_enforced"] is True
        assert execution["cross_provider_fallback_allowed"] is False
        assert execution["supervisor_receipt_id"]
        assert execution["execution_result_id"]
        assert execution["native_output_schema_id"].startswith("sha256:")


def test_native_pair_rejects_response_with_wrong_request_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _native_request(ProviderRole.CODEX_REVIEW)
    monkeypatch.setattr(
        native_cli.shutil,
        "which",
        lambda name: f"/trusted/bin/{name}",
    )

    def mismatched(
        command: Any,
        *,
        cwd: Path,
        timeout_seconds: int,
        stdin_text: str | None = None,
    ) -> tuple[str, str]:
        del cwd, timeout_seconds, stdin_text
        cmd = list(command)
        response_path = Path(cmd[cmd.index("--output-last-message") + 1])
        response_path.write_text(
            json.dumps(
                {
                    "packet_id": "packet:wrong",
                    "snapshot_id": request.snapshot_id,
                    "task_id": request.task_id,
                    "decision": "approve",
                    "findings": [],
                }
            ),
            encoding="utf-8",
        )
        return "", ""

    monkeypatch.setattr(native_cli, "_run_native_cli_process", mismatched)
    _grok, codex = build_production_cli_provider_pair()

    with pytest.raises(RuntimeError, match="request binding"):
        codex(request)


def test_native_codex_rejects_approve_with_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _native_request(ProviderRole.CODEX_REVIEW)
    monkeypatch.setattr(
        native_cli.shutil,
        "which",
        lambda name: f"/trusted/bin/{name}",
    )

    def contradictory_review(
        command: Any,
        *,
        cwd: Path,
        timeout_seconds: int,
        stdin_text: str | None = None,
    ) -> tuple[str, str]:
        del cwd, timeout_seconds, stdin_text
        cmd = list(command)
        response_path = Path(cmd[cmd.index("--output-last-message") + 1])
        response_path.write_text(
            json.dumps(
                {
                    "packet_id": request.packet_id,
                    "snapshot_id": request.snapshot_id,
                    "task_id": request.task_id,
                    "decision": "approve",
                    "findings": ["the proposal is unsafe"],
                }
            ),
            encoding="utf-8",
        )
        return "", ""

    monkeypatch.setattr(
        native_cli,
        "_run_native_cli_process",
        contradictory_review,
    )
    _grok, codex = build_production_cli_provider_pair()

    with pytest.raises(RuntimeError, match="strict schema"):
        codex(request)


def test_custom_invoker_codex_rejects_approve_with_findings() -> None:
    policy = ProductionCLIProviderPolicy()

    def contradictory_review(_prompt, config):
        return (
            json.dumps(
                {
                    "decision": "approve",
                    "findings": ["the proposal is unsafe"],
                }
            ),
            _child_receipt(config),
        )

    reviewer = BoundProductionCLIProvider(
        policy=policy,
        role=ProviderRole.CODEX_REVIEW,
        provider_name=policy.codex_provider,
        model_name=policy.codex_model,
        invoker=contradictory_review,
    )

    with pytest.raises(RuntimeError, match="empty findings list"):
        reviewer(_request(ProviderRole.CODEX_REVIEW))


def test_native_codex_response_file_is_limited_before_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _native_request(ProviderRole.CODEX_REVIEW)
    request = replace(
        original,
        bounds=ProviderBounds(max_response_bytes=128),
    )
    monkeypatch.setattr(
        native_cli.shutil,
        "which",
        lambda name: f"/trusted/bin/{name}",
    )

    def oversized(
        command: Any,
        *,
        cwd: Path,
        timeout_seconds: int,
        stdin_text: str | None = None,
    ) -> tuple[str, str]:
        del cwd, timeout_seconds, stdin_text
        cmd = list(command)
        response_path = Path(cmd[cmd.index("--output-last-message") + 1])
        response_path.write_bytes(b"x" * 129)
        return "", ""

    monkeypatch.setattr(native_cli, "_run_native_cli_process", oversized)
    _grok, codex = build_production_cli_provider_pair()

    with pytest.raises(RuntimeError, match="exceeds byte bound"):
        codex(request)


def test_native_response_reader_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    response_path = tmp_path / "last-message.json"
    os.mkfifo(response_path, 0o600)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="private regular file"):
        native_cli._read_bounded_regular_utf8(response_path, max_bytes=128)
    assert time.monotonic() - started < 1.0


def test_native_process_capture_is_bounded_and_kills_descendants(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "child.pid"
    script = (
        "import os,pathlib,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
        "start_new_session=True);"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid));"
        "os.write(1,b'x'*(2*1024*1024));time.sleep(60)"
    )

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="capture bound"):
        native_cli._run_native_cli_process(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            timeout_seconds=10,
        )
    assert time.monotonic() - started < 10
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and Path(f"/proc/{child_pid}").exists():
        state = Path(f"/proc/{child_pid}/stat").read_text().split()[2]
        if state == "Z":
            break
        time.sleep(0.02)
    if Path(f"/proc/{child_pid}/stat").exists():
        assert Path(f"/proc/{child_pid}/stat").read_text().split()[2] == "Z"


def test_native_process_timeout_kills_detached_descendant(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "timeout-child.pid"
    script = (
        "import pathlib,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
        "start_new_session=True);"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid));"
        "time.sleep(60)"
    )

    with pytest.raises(RuntimeError, match="timed out"):
        native_cli._run_native_cli_process(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            timeout_seconds=1,
        )
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and Path(f"/proc/{child_pid}").exists():
        state = Path(f"/proc/{child_pid}/stat").read_text().split()[2]
        if state == "Z":
            break
        time.sleep(0.02)
    if Path(f"/proc/{child_pid}/stat").exists():
        assert Path(f"/proc/{child_pid}/stat").read_text().split()[2] == "Z"


@pytest.mark.parametrize(
    ("codex_status", "expected_ready"),
    [(0, True), (1, False)],
)
def test_policy_readiness_requires_bounded_codex_auth_check(
    monkeypatch: pytest.MonkeyPatch,
    codex_status: int,
    expected_ready: bool,
) -> None:
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon."
        "production_provider_cli.shutil.which",
        lambda name: f"/test-bin/{name}",
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.llm_router._grok_cli_auth_available",
        lambda: True,
    )

    def run_auth(command, **kwargs):
        observed["command"] = command
        observed["timeout"] = kwargs.get("timeout")
        observed["stdin"] = kwargs.get("stdin")
        return subprocess.CompletedProcess(command, codex_status, "", "")

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon."
        "production_provider_cli.subprocess.run",
        run_auth,
    )

    readiness = production_cli_policy_readiness()

    assert readiness["ready"] is expected_ready
    assert readiness["review"]["authenticated"] is expected_ready
    assert readiness["review"]["authentication_check"] == (
        "codex_login_status_ok"
        if expected_ready
        else "codex_login_status_failed"
    )
    assert observed["command"] == ["/test-bin/codex", "login", "status"]
    assert observed["timeout"] == 5.0
    assert observed["stdin"] is subprocess.DEVNULL


def test_adapter_rejects_role_or_effective_provider_mismatch() -> None:
    policy = ProductionCLIProviderPolicy()

    def wrong_provider(_prompt, config):
        receipt = _child_receipt(config)
        return (
            '{"decision":"approve"}',
            LlmChildResultEnvelope(
                usage_mode=receipt.usage_mode,
                request_id=receipt.request_id,
                attempt=receipt.attempt,
                idempotency_key=receipt.idempotency_key,
                status="ok",
                effective_provider="grok_cli",
                text_chars=1,
                exit_code=0,
            ),
        )

    reviewer = BoundProductionCLIProvider(
        policy=policy,
        role=ProviderRole.CODEX_REVIEW,
        provider_name=policy.codex_provider,
        model_name=policy.codex_model,
        invoker=wrong_provider,
    )
    with pytest.raises(RuntimeError, match="role mismatch"):
        reviewer(_request(ProviderRole.GROK_IMPLEMENT))
    with pytest.raises(RuntimeError, match="execution receipt is not bound"):
        reviewer(_request(ProviderRole.CODEX_REVIEW))


def test_adapter_rejects_non_json_or_missing_child_receipt() -> None:
    policy = ProductionCLIProviderPolicy()

    for output, receipt in (("```json\n{}\n```", None), ("{}", None)):
        provider = BoundProductionCLIProvider(
            policy=policy,
            role=ProviderRole.GROK_IMPLEMENT,
            provider_name=policy.grok_provider,
            model_name=policy.grok_model,
            invoker=lambda _prompt, _config, value=output, bound=receipt: (
                value,
                bound,
            ),
        )
        expected = "Expecting value" if output.startswith("```") else "execution receipt"
        with pytest.raises((ValueError, RuntimeError), match=expected):
            provider(_request(ProviderRole.GROK_IMPLEMENT))


def test_pair_routes_grok_then_independent_codex_with_bound_evidence() -> None:
    calls: list[dict[str, Any]] = []

    def invoke(prompt, config):
        envelope = json.loads(prompt)
        calls.append(
            {
                "provider": config.provider,
                "role": envelope["role"],
                "provider_input": envelope["provider_input"],
                "response_contract": envelope["response_contract"],
            }
        )
        if config.provider == "grok_cli":
            output = {
                "proposal": {
                    "declared_paths": ["module.py"],
                    "files": [{"path": "module.py", "content": "ok\n"}],
                }
            }
        else:
            output = {"decision": "approve", "findings": []}
        return json.dumps(output), _child_receipt(config)

    grok, codex = build_production_cli_provider_pair(invoker=invoke)
    assert grok is not codex
    packet = build_production_contract_packet(
        task_id="ASE-005",
        snapshot_id="git-commit:fixture",
        write_paths=["module.py"],
        validation_commands=["python -m pytest test_module.py -q"],
        acceptance_criteria="module behavior is validated",
    )
    router = ImplementationProviderRouter(
        grok_provider=grok,
        codex_provider=codex,
        admission_gate=lambda proposal: {
            "accepted": True,
            "reason_code": f"admitted:{proposal.role.value}",
        },
    )

    result = router.route(
        packet,
        current_snapshot_id="git-commit:fixture",
        apply=False,
    )

    assert result.status is RouteStatus.SUCCEEDED
    assert result.review_presence == ReviewPresence.INDEPENDENT.value
    assert result.provider_result_admitted is True
    assert [item["provider"] for item in calls] == ["grok_cli", "codex_cli"]
    assert calls[0]["role"] == ProviderRole.GROK_IMPLEMENT.value
    assert calls[1]["role"] == ProviderRole.CODEX_REVIEW.value
    assert "contract_packet" in calls[0]["provider_input"]
    assert "contract_packet" not in calls[1]["provider_input"]
    assert "admitted_implementation_proposal" in calls[1]["provider_input"]
    assert calls[0]["response_contract"]["format"] == (
        "canonical-json-object-only"
    )
    assert result.implementation_proposal is not None
    assert result.review_proposal is not None
    assert (
        result.implementation_proposal.payload["supervisor_provider_execution"][
            "effective_provider"
        ]
        == "grok_cli"
    )
    assert (
        result.review_proposal.payload["supervisor_provider_execution"][
            "effective_provider"
        ]
        == "codex_cli"
    )


def _applied_provider_route():
    def invoke(prompt, config):
        role = json.loads(prompt)["role"]
        output = (
            {
                "proposal": {
                    "declared_paths": ["module.py"],
                    "files": [{"path": "module.py", "content": "ok\n"}],
                }
            }
            if role == ProviderRole.GROK_IMPLEMENT.value
            else {"decision": "approve", "findings": []}
        )
        return json.dumps(output), _child_receipt(config)

    grok, codex = build_production_cli_provider_pair(invoker=invoke)
    packet = build_production_contract_packet(
        task_id="ASE-005",
        snapshot_id="git-commit:fixture",
        write_paths=["module.py"],
        validation_commands=["python -m pytest test_module.py -q"],
        acceptance_criteria="module behavior is validated",
    )
    router = ImplementationProviderRouter(
        grok_provider=grok,
        codex_provider=codex,
        admission_gate=lambda proposal: {
            "accepted": True,
            "reason_code": f"admitted:{proposal.role.value}",
        },
        writer=lambda _proposal, _lease: None,
    )
    result = router.route(
        packet,
        current_snapshot_id="git-commit:fixture",
        apply=True,
        writer_lease_id="lease:ase-005:1",
    )
    binding = bind_applied_patch_to_review_chain(result)
    assert result.provider_result_admitted is True
    assert binding is not None
    return result, binding


def _apply_real_reviewed_route(
    daemon: TodoImplementationDaemon,
    task: PortalTask,
    repo: Path,
):
    """Produce exact production route/effect/commit evidence in one real repo."""

    policy = daemon.production_provider_policy
    assert isinstance(policy, ProductionCLIProviderPolicy)

    def invoke(prompt, config):
        role = json.loads(prompt)["role"]
        output = (
            {
                "proposal": {
                    "declared_paths": list(task.outputs),
                    "files": [
                        {"path": task.outputs[0], "content": "VALUE = 'reviewed'\n"}
                    ],
                }
            }
            if role == ProviderRole.GROK_IMPLEMENT.value
            else {"decision": "approve", "findings": []}
        )
        return json.dumps(output), _child_receipt(config)

    grok, codex = build_production_cli_provider_pair(policy, invoker=invoke)
    daemon._production_grok_provider = grok
    daemon._production_codex_provider = codex
    baseline = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    route = daemon.run_production_model_assisted_route(
        task,
        attempt=1,
        workspace_path=repo,
        baseline_ref=baseline,
        apply=True,
    )
    result = route["route_result"]
    captured = route["reviewed_effect_binding"]
    assert captured is not None
    subprocess.run(
        ["git", "add", "--", *task.outputs],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "reviewed provider candidate"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    tree = "git-tree:" + subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    finalized = finalize_production_reviewed_effect(
        captured,
        repo_root=repo,
        task=task,
        task_identity=daemon._identity_for_task(task),
        implementation_commit=commit,
    )
    binding = bind_applied_patch_to_review_chain(
        result,
        implementation_commit=commit,
    )
    assert binding is not None
    return result, binding, finalized, commit, tree


def _real_reviewed_fixture(tmp_path: Path):
    repo = tmp_path / "reviewed-repo"
    repo.mkdir()
    for arguments in (
        ("init",),
        ("config", "user.name", "Reviewed Provider Test"),
        ("config", "user.email", "reviewed-provider@example.invalid"),
    ):
        subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        )
    todo_path = repo / "tasks.todo.md"
    todo_path.write_text("# Tasks\n", encoding="utf-8")
    (repo / ".gitignore").write_text(
        "bundle-state/\nworktrees/\nstate/\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", "."],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "baseline"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    state_path = tmp_path / "reviewed-state" / "task_state.json"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_path,
        strategy_path=state_path.parent / "strategy.json",
        events_path=state_path.parent / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ASE-",
        implement=True,
        production_provider_policy=PRODUCTION_CLI_POLICY_NAME,
        production_provider_review_authority_key_path=(
            tmp_path / "reviewed-state" / "review.ed25519"
        ),
    )
    task = PortalTask(
        task_id="ASE-005",
        title="Provider review",
        status="ready",
        completion="manual",
        priority="P0",
        track="provider",
        outputs=["module.py"],
        validation=["python -m py_compile module.py"],
        acceptance="review is independently verified",
        metadata={"Provider role": "grok-implement, codex-review"},
    )
    evidence = _apply_real_reviewed_route(daemon, task, repo)
    return daemon, task, repo, evidence


def test_ed25519_review_attestation_reconstructs_full_receipt_and_binding(
    tmp_path: Path,
) -> None:
    daemon, task, repo, evidence = _real_reviewed_fixture(tmp_path)
    result, binding, reviewed_effect, commit, tree = evidence
    key_path = tmp_path / "state" / ".provider-review.ed25519"
    authority = ProductionProviderReviewAuthority.load_or_create(key_path)
    attestation = authority.issue(
        provider_receipt=result.provider_receipt,
        review_chain_binding=binding,
        provider_policy_id=ProductionCLIProviderPolicy().policy_id,
        implementation_commit=commit,
        implementation_tree_id=tree,
        reviewed_effect_binding=reviewed_effect,
        repo_root=repo,
        task=task,
        task_identity=daemon._identity_for_task(task),
        issued_at_ms=1_800_000_000_000,
        nonce="fixed-test-nonce-00000001",
    )

    assert os.stat(key_path).st_mode & 0o777 == 0o600
    verification = verify_production_provider_review_attestation(
        attestation,
        trusted_public_keys={
            authority.issuer_key_id: authority.public_key_bytes,
        },
        provider_receipt=result.provider_receipt,
        review_chain_binding=binding,
        reviewed_effect_binding=reviewed_effect,
        repo_root=repo,
        task=task,
        task_identity=daemon._identity_for_task(task),
        expected_task_id="ASE-005",
        expected_snapshot_id=reviewed_effect.snapshot_id,
        expected_provider_policy_id=ProductionCLIProviderPolicy().policy_id,
        expected_implementation_commit=commit,
        expected_implementation_tree_id=tree,
    )

    assert verification.admitted is True
    assert verification.provider_receipt_cid == result.provider_receipt.receipt_id
    assert attestation.to_dict()["completion_authoritative"] is False
    assert "public" not in " ".join(attestation.to_dict()).casefold()


def test_review_authority_key_rejects_links_types_and_permissive_mode(
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "review.ed25519"
    ProductionProviderReviewAuthority.load_or_create(key_path)

    symlink_path = tmp_path / "review-link.ed25519"
    symlink_path.symlink_to(key_path)
    with pytest.raises(ValueError, match="unsafe|symlink"):
        ProductionProviderReviewAuthority.load_or_create(symlink_path)

    hardlink_path = tmp_path / "review-hardlink.ed25519"
    os.link(key_path, hardlink_path)
    with pytest.raises(ValueError, match="hard-linked"):
        ProductionProviderReviewAuthority.load_or_create(key_path)
    hardlink_path.unlink()

    os.chmod(key_path, 0o644)
    with pytest.raises(ValueError, match="permissions"):
        ProductionProviderReviewAuthority.load_or_create(key_path)

    fifo_path = tmp_path / "review.fifo"
    os.mkfifo(fifo_path, 0o600)
    with pytest.raises(ValueError, match="regular file"):
        ProductionProviderReviewAuthority.load_or_create(fifo_path)


def test_review_authority_creation_retries_partial_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_write = os.write
    write_calls = 0

    def partial_write(descriptor: int, value: bytes | memoryview) -> int:
        nonlocal write_calls
        write_calls += 1
        return real_write(descriptor, bytes(value[:1]))

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon."
        "production_provider_attestation.os.write",
        partial_write,
    )
    key_path = tmp_path / "review.ed25519"

    authority = ProductionProviderReviewAuthority.load_or_create(key_path)

    assert len(authority.public_key_bytes) == 32
    assert key_path.stat().st_size == 32
    assert write_calls == 32


def test_forged_fully_current_attestation_from_metadata_signer_is_rejected(
    tmp_path: Path,
) -> None:
    """A coherent attacker receipt/key cannot install its own trust root."""

    daemon, task, repo, evidence = _real_reviewed_fixture(tmp_path)
    result, binding, reviewed_effect, commit, tree = evidence
    trusted = ProductionProviderReviewAuthority.generate()
    attacker = ProductionProviderReviewAuthority.generate()
    forged = attacker.issue(
        provider_receipt=result.provider_receipt,
        review_chain_binding=binding,
        provider_policy_id=ProductionCLIProviderPolicy().policy_id,
        implementation_commit=commit,
        implementation_tree_id=tree,
        reviewed_effect_binding=reviewed_effect,
        repo_root=repo,
        task=task,
        task_identity=daemon._identity_for_task(task),
        issued_at_ms=1_800_000_000_000,
        nonce="forged-current-nonce-0001",
    )

    verification = verify_production_provider_review_attestation(
        forged,
        trusted_public_keys={trusted.issuer_key_id: trusted.public_key_bytes},
        provider_receipt=result.provider_receipt,
        review_chain_binding=binding,
        reviewed_effect_binding=reviewed_effect,
        repo_root=repo,
        task=task,
        task_identity=daemon._identity_for_task(task),
        expected_task_id="ASE-005",
        expected_snapshot_id=reviewed_effect.snapshot_id,
        expected_provider_policy_id=ProductionCLIProviderPolicy().policy_id,
        expected_implementation_commit=commit,
        expected_implementation_tree_id=tree,
    )

    assert verification.admitted is False
    assert "provider_review_issuer_untrusted" in verification.reason_codes


def test_attestation_rejects_review_binding_without_implementation_commit() -> None:
    result, binding = _applied_provider_route()
    authority = ProductionProviderReviewAuthority.generate()

    with pytest.raises(
        ValueError, match="provider_review_implementation_commit_missing"
    ):
        authority.issue(
            provider_receipt=result.provider_receipt,
            review_chain_binding=binding,
            provider_policy_id=ProductionCLIProviderPolicy().policy_id,
            implementation_commit="a" * 40,
            implementation_tree_id="git-tree:" + "b" * 40,
            issued_at_ms=1_800_000_000_000,
            nonce="missing-binding-commit-0001",
        )


def test_shared_cross_lane_authority_and_operator_policy_derive_provider_gate(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    for arguments in (
        ("init",),
        ("config", "user.name", "Provider Review Test"),
        ("config", "user.email", "provider-review@example.invalid"),
    ):
        subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        )
    todo_path = repo / "tasks.todo.md"
    todo_path.write_text("# Tasks\n", encoding="utf-8")
    (repo / ".gitignore").write_text(
        "bundle-state/\nworktrees/\nstate/\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", "."],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "baseline"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    shared_key_path = repo / "bundle-state" / "shared-review.ed25519"
    issuer_state_path = repo / "bundle-state" / "lane-a" / "task_state.json"
    issuer = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=issuer_state_path,
        strategy_path=issuer_state_path.parent / "strategy.json",
        events_path=issuer_state_path.parent / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ASE-",
        implement=True,
        production_provider_policy=PRODUCTION_CLI_POLICY_NAME,
        production_provider_review_authority_key_path=shared_key_path,
    )
    verifier_state_path = repo / "bundle-state" / "lane-b" / "task_state.json"
    verifier = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=verifier_state_path,
        strategy_path=verifier_state_path.parent / "strategy.json",
        events_path=verifier_state_path.parent / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ASE-",
        production_provider_policy=PRODUCTION_CLI_POLICY_NAME,
        production_provider_review_authority_key_path=shared_key_path,
    )
    task = PortalTask(
        task_id="ASE-005",
        title="Provider review",
        status="ready",
        completion="manual",
        priority="P0",
        track="provider",
        outputs=["module.py"],
        validation=["python -m pytest test_module.py -q"],
        acceptance="review is independently verified",
        metadata={
            "Provider role": "grok-implement, codex-review",
            "Context budget tokens": "4096",
        },
    )
    (
        result,
        binding,
        reviewed_effect,
        implementation_commit,
        implementation_tree_id,
    ) = _apply_real_reviewed_route(issuer, task, repo)
    authority = issuer._production_provider_review_authority
    assert isinstance(authority, ProductionProviderReviewAuthority)
    attestation = authority.issue(
        provider_receipt=result.provider_receipt,
        review_chain_binding=binding,
        provider_policy_id=ProductionCLIProviderPolicy().policy_id,
        implementation_commit=implementation_commit,
        implementation_tree_id=implementation_tree_id,
        reviewed_effect_binding=reviewed_effect,
        repo_root=repo,
        task=task,
        task_identity=issuer._identity_for_task(task),
        issued_at_ms=1_800_000_000_000,
        nonce="authoritative-gate-nonce-0001",
    )

    receipt = verifier.build_task_implementation_receipt(
        task,
        implementation_commit=implementation_commit,
        merge_commit=implementation_commit,
        repository_tree_id=implementation_tree_id,
        merged=True,
        gate_evidence={
            "provider_execution_receipt": result.provider_receipt.to_dict(),
            "admitted_review_chain_binding": binding.to_dict(),
            "provider_review_attestation": attestation.to_dict(),
            "production_reviewed_effect_binding": reviewed_effect.to_dict(),
            # Carrier-selected expectations are ignored. The verifier uses
            # its operator-configured policy and the signed receipt snapshot.
            "provider_review_expected_policy_id": "sha256:carrier-choice",
            "provider_review_expected_snapshot_id": "git-commit:carrier-choice",
            # This forged prebuilt gate must be ignored in favor of derivation.
            "provider_review": {"satisfied": True},
        },
        model_invocation_observed=True,
    )

    provider_gate = receipt.gate_evidence["provider_review"]
    assert provider_gate["satisfied"] is True
    assert provider_gate["review_receipt_id"] == attestation.attestation_id
    assert provider_gate["verification"] == (
        "ed25519_full_receipt_reconstruction"
    )
    assert os.stat(shared_key_path).st_mode & 0o777 == 0o600

    wrong_key_path = repo / "bundle-state" / "lane-c-review.ed25519"
    wrong_lane = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=repo / "bundle-state" / "lane-c" / "task_state.json",
        strategy_path=repo / "bundle-state" / "lane-c" / "strategy.json",
        events_path=repo / "bundle-state" / "lane-c" / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ASE-",
        implement=True,
        production_provider_policy=PRODUCTION_CLI_POLICY_NAME,
        production_provider_review_authority_key_path=wrong_key_path,
    )
    rejected = wrong_lane.build_task_implementation_receipt(
        task,
        implementation_commit=implementation_commit,
        merge_commit=implementation_commit,
        repository_tree_id=implementation_tree_id,
        merged=True,
        gate_evidence={
            "provider_execution_receipt": result.provider_receipt.to_dict(),
            "admitted_review_chain_binding": binding.to_dict(),
            "provider_review_attestation": attestation.to_dict(),
            "production_reviewed_effect_binding": reviewed_effect.to_dict(),
        },
        model_invocation_observed=True,
    )
    assert "provider_review" not in rejected.gate_evidence


def test_signed_attestation_cannot_choose_its_own_expected_policy(
    tmp_path: Path,
) -> None:
    """A trusted signing key does not make an unconfigured policy admissible."""

    daemon, task, repo, evidence = _real_reviewed_fixture(tmp_path)
    result, binding, reviewed_effect, commit, tree = evidence
    authority = daemon._production_provider_review_authority
    assert isinstance(authority, ProductionProviderReviewAuthority)
    attacker_policy_id = "sha256:" + "f" * 64
    attestation = authority.issue(
        provider_receipt=result.provider_receipt,
        review_chain_binding=binding,
        provider_policy_id=ProductionCLIProviderPolicy().policy_id,
        implementation_commit=commit,
        implementation_tree_id=tree,
        reviewed_effect_binding=reviewed_effect,
        repo_root=repo,
        task=task,
        task_identity=daemon._identity_for_task(task),
        issued_at_ms=1_800_000_000_000,
        nonce="policy-confusion-nonce-0001",
    )
    receipt = daemon.build_task_implementation_receipt(
        task,
        implementation_commit=commit,
        merge_commit=commit,
        repository_tree_id=tree,
        merged=True,
        gate_evidence={
            "provider_execution_receipt": result.provider_receipt.to_dict(),
            "admitted_review_chain_binding": binding.to_dict(),
            "provider_review_attestation": attestation.to_dict(),
            "production_reviewed_effect_binding": reviewed_effect.to_dict(),
            "provider_review_expected_policy_id": attacker_policy_id,
        },
        model_invocation_observed=True,
    )

    assert receipt.gate_evidence["provider_review"]["satisfied"] is True


def test_bundle_command_propagates_explicit_policy_without_task_metadata() -> None:
    shared_key_path = Path("bundle-state/shared-review.ed25519")
    command = implementation_supervisor_command(
        todo_path=Path("runtime.todo.md"),
        state_dir=Path("state"),
        worktree_root=Path("worktrees"),
        state_prefix="ase-provider",
        task_prefix="## ASE-",
        implement=True,
        daemon_interval=5,
        stale_seconds=30,
        check_interval=2,
        watchdog_startup_grace_seconds=None,
        max_restarts=1,
        implementation_timeout=300,
        production_provider_policy=PRODUCTION_CLI_POLICY_NAME,
        production_provider_context_budget_tokens=4096,
        production_provider_review_authority_key_path=shared_key_path,
    )

    assert command[command.index("--production-provider-policy") + 1] == (
        PRODUCTION_CLI_POLICY_NAME
    )
    assert command[
        command.index("--production-provider-context-budget-tokens") + 1
    ] == "4096"
    assert command[
        command.index("--production-provider-timeout-seconds") + 1
    ] == "300.0"
    assert command[
        command.index(
            "--production-provider-review-authority-key-path"
        )
        + 1
    ] == str(shared_key_path)


def test_bundle_lanes_default_to_one_bundle_root_review_authority(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    bundle_dir = repo / "bundles"
    bundle_dir.mkdir(parents=True)
    for task_id in ("ASE-005", "ASE-006"):
        (bundle_dir / f"{task_id}.todo.md").write_text(
            f"## {task_id} Provider task\n\n- Status: todo\n",
            encoding="utf-8",
        )
    index_path = bundle_dir / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "bundles": {
                    f"objective/provider/{task_id}": {
                        "shard_path": f"bundles/{task_id}.todo.md",
                        "parallel_lane": f"provider/{task_id}",
                        "tasks": [{"task_id": task_id}],
                    }
                    for task_id in ("ASE-005", "ASE-006")
                }
            }
        ),
        encoding="utf-8",
    )
    state_root = repo / "bundle-state"
    lanes = plan_bundle_lanes(
        bundle_index_path=index_path,
        repo_root=repo,
        state_root=state_root,
        worktree_root=repo / "worktrees",
        log_dir=repo / "logs",
        task_prefix="ASE-",
        implement=True,
        production_provider_policy=PRODUCTION_CLI_POLICY_NAME,
        production_provider_context_budget_tokens=4096,
        optimize_bundles=False,
    )
    expected = str(
        state_root / DEFAULT_PRODUCTION_PROVIDER_REVIEW_KEY_NAME
    )

    assert len(lanes) == 2
    assert {
        lane.command[
            lane.command.index(
                "--production-provider-review-authority-key-path"
            )
            + 1
        ]
        for lane in lanes
    } == {expected}


def test_daemon_cli_accepts_operator_policy_without_task_metadata(
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "bundle-state" / "shared-review.ed25519"
    args = parse_daemon_args(
        [
            "--todo-path",
            str(tmp_path / "runtime.todo.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--implement",
            "--production-provider-policy",
            PRODUCTION_CLI_POLICY_NAME,
            "--production-provider-context-budget-tokens",
            "3072",
            "--production-provider-timeout-seconds",
            "240",
            "--production-provider-review-authority-key-path",
            str(key_path),
        ]
    )

    assert args.production_provider_policy == PRODUCTION_CLI_POLICY_NAME
    assert args.production_provider_context_budget_tokens == 3072
    assert args.production_provider_timeout_seconds == 240.0
    assert args.production_provider_review_authority_key_path == key_path


def test_supervisor_policy_cli_normalizes_budget_and_fences_daemon_adoption(
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "bundle-state" / "shared-review.ed25519"
    args = parse_supervisor_args(
        [
            "--todo-path",
            str(tmp_path / "runtime.todo.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--implement",
            "--production-provider-policy",
            PRODUCTION_CLI_POLICY_NAME,
            "--production-provider-timeout-seconds",
            "240",
            "--production-provider-review-authority-key-path",
            str(key_path),
        ]
    )
    config = supervisor_config_from_args(args, repo_root=tmp_path)
    assert config.production_provider_policy == PRODUCTION_CLI_POLICY_NAME
    assert config.production_provider_context_budget_tokens == 4096
    assert config.production_provider_timeout_seconds == 240.0
    assert config.production_provider_review_authority_key_path == key_path

    supervisor = PortalImplementationSupervisor(config)
    command = supervisor._build_daemon_command()
    command_line = " ".join(command)
    assert "--production-provider-policy" in command
    assert supervisor._managed_daemon_matches_command_line(command_line) is True

    policy_index = command.index("--production-provider-policy")
    without_policy = [
        *command[:policy_index],
        *command[policy_index + 8 :],
    ]
    assert (
        supervisor._managed_daemon_matches_command_line(" ".join(without_policy))
        is False
    )
    wrong_timeout = list(command)
    timeout_index = wrong_timeout.index("--production-provider-timeout-seconds")
    wrong_timeout[timeout_index + 1] = "300.0"
    assert (
        supervisor._managed_daemon_matches_command_line(" ".join(wrong_timeout))
        is False
    )
    wrong_key = list(command)
    key_index = wrong_key.index(
        "--production-provider-review-authority-key-path"
    )
    wrong_key[key_index + 1] = str(tmp_path / "lane-local.ed25519")
    assert (
        supervisor._managed_daemon_matches_command_line(" ".join(wrong_key))
        is False
    )


def test_supervisor_rejects_budget_without_explicit_provider_policy(
    tmp_path: Path,
) -> None:
    args = parse_supervisor_args(
        [
            "--todo-path",
            str(tmp_path / "runtime.todo.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--production-provider-context-budget-tokens",
            "2048",
        ]
    )
    with pytest.raises(ValueError, match="require a production provider policy"):
        supervisor_config_from_args(args, repo_root=tmp_path)

    key_only = parse_supervisor_args(
        [
            "--todo-path",
            str(tmp_path / "runtime.todo.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--production-provider-review-authority-key-path",
            str(tmp_path / "review.ed25519"),
        ]
    )
    with pytest.raises(ValueError, match="require a production provider policy"):
        supervisor_config_from_args(key_only, repo_root=tmp_path)
