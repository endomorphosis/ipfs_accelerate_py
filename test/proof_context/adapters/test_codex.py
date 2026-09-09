"""PCCE-031: Codex adapter tests for the installed ``codex exec`` mechanism."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.proof_context.adapters.base import (
    ADAPTER_CONTRACT_CID,
    INTERFACE,
    CancellationToken,
    CodingAgentAdapter,
    execute_propose,
)
from ipfs_accelerate_py.proof_context.adapters.codex import (
    ADAPTER,
    CODEX_ADAPTER_CID,
    COMMAND_CONTRACT,
    LIVE_PERMIT_ENV,
    SUPPORTED_INTEGRATION_CLASS,
    SUPPORTED_INTEGRATION_MODULE,
    SUPPORTED_MECHANISM,
    CodexAdapter,
    InstalledCodexTransport,
    RecordedCodexTransport,
    UnavailableCodexTransport,
    bound_and_redact_log,
    build_codex_request,
    codex_adapter_cid,
    codex_adapter_descriptor,
    discover_supported_mechanism,
    extract_patch_paths,
    live_permit_granted,
    parse_structured_proposal,
    probe_supported_mechanism,
    redact_log_text,
)
from ipfs_accelerate_py.proof_context.adapters.models import (
    CODING_AGENT_INVOCATION_SCHEMA,
    CONTEXT_PACK_SCHEMA,
    MAX_LOG_BYTES,
    MODEL_ROUTE_DECISION_SCHEMA,
    PATCH_PROPOSAL_SCHEMA,
    TASK_SPECIFICATION_SCHEMA,
    ContextPack,
    ModelRouteDecision,
    TaskSpecification,
)
from ipfs_accelerate_py.proof_context.errors import (
    REDACTED,
    BoundaryViolationError,
    IdentityInconsistentError,
    MalformedError,
    ProofCancelledError,
    SimulatedPromotedError,
    UnavailableCapabilityError,
)

CID = "bafkreihxffr7ppivqrjwb3pafumcoh6mg7vyoglc6gpnmbtjgnq2pdajru"
CID_B = "bafkreihxffr7ppivqrjwb3pafumcoh6mg7vyoglc6gpnmbtjgnq2pdajrv"
CID_C = "bafkreihxffr7ppivqrjwb3pafumcoh6mg7vyoglc6gpnmbtjgnq2pdajrw"
CID_D = "bafkreihxffr7ppivqrjwb3pafumcoh6mg7vyoglc6gpnmbtjgnq2pdajrx"
OWNED = "src/demo/__init__.py"
PATCH = (
    "diff --git a/src/demo/__init__.py b/src/demo/__init__.py\n"
    "--- a/src/demo/__init__.py\n"
    "+++ b/src/demo/__init__.py\n"
    "@@ -1 +1 @@\n"
    "-VALUE = 1\n"
    "+VALUE = 2\n"
)


def _task(**overrides: Any) -> TaskSpecification:
    payload: dict[str, Any] = {
        "schema": TASK_SPECIFICATION_SCHEMA,
        "task_id": "PCCE-031",
        "objective_id": "PCCE-G300",
        "repository_state_cid": CID,
        "owned_paths": (OWNED,),
        "declared_files": (OWNED,),
        "route_cid": CID_B,
        "provenance": "live",
    }
    payload.update(overrides)
    return TaskSpecification.from_mapping(payload)


def _pack(**overrides: Any) -> ContextPack:
    payload: dict[str, Any] = {
        "schema": CONTEXT_PACK_SCHEMA,
        "pack_cid": CID_C,
        "repository_state_cid": CID,
        "sufficiency": "sufficient",
        "provenance": "live",
        "task_id": "PCCE-031",
        "capsule_cids": (CID_D,),
    }
    payload.update(overrides)
    return ContextPack.from_mapping(payload)


def _route(**overrides: Any) -> ModelRouteDecision:
    payload: dict[str, Any] = {
        "schema": MODEL_ROUTE_DECISION_SCHEMA,
        "decision_cid": CID_B,
        "task_id": "PCCE-031",
        "tier": "medium",
        "provider": "codex",
        "model": "gpt-5.6-terra",
        "revision": "r1",
        "repository_state_cid": CID,
        "provenance": "live",
    }
    payload.update(overrides)
    return ModelRouteDecision.from_mapping(payload)


def _recorded_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "task_id": "PCCE-031",
        "repository_state_cid": CID,
        "pack_cid": CID_C,
        "route_cid": CID_B,
        "declared_files": [OWNED],
        "patch": PATCH,
        "model": "gpt-5.6-terra",
        "revision": "r1",
        "token_count": 12,
        "cached_token_count": 4,
        "latency_ms": 9,
        "cost_micros": 3,
        "provenance": "replayed",
    }
    payload.update(overrides)
    return payload


def _adapter(payload: Mapping[str, Any] | None = None, **transport_kwargs: Any) -> CodexAdapter:
    response = dict(payload or _recorded_payload())
    return CodexAdapter(transport=RecordedCodexTransport(response, **transport_kwargs))


def test_cold_import_has_no_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))
    import ipfs_accelerate_py.proof_context.adapters.codex as codex

    after = set(tmp_path.rglob("*"))
    assert after == before
    assert codex.COMMAND_CONTRACT == "codex exec"
    assert codex.ADAPTER == ADAPTER


def test_supported_mechanism_is_codex_exec_and_is_not_substituted() -> None:
    probe = discover_supported_mechanism(which=lambda _name: None)
    assert probe.command_contract == COMMAND_CONTRACT == SUPPORTED_MECHANISM
    assert probe.module == SUPPORTED_INTEGRATION_MODULE
    assert SUPPORTED_INTEGRATION_CLASS == "OpenAICodexCLIIntegration"
    assert probe.substituted is False
    assert probe.module_available is True
    assert probe.builder_available is True
    assert probe.status == "unavailable"
    assert probe.available is False
    descriptor = codex_adapter_descriptor()
    assert descriptor["cid"] == CODEX_ADAPTER_CID == codex_adapter_cid()
    assert CODEX_ADAPTER_CID.startswith("b")
    assert len(CODEX_ADAPTER_CID) >= 59
    assert descriptor["command_contract"] == "codex exec"
    assert descriptor["silently_substitutes_mechanism"] is False
    assert descriptor["approval_authority"] is False
    assert descriptor["canonical_branch_authority"] is False
    assert descriptor["adapter_contract_cid"] == ADAPTER_CONTRACT_CID
    assert descriptor["interface"] == INTERFACE


def test_validation_environment_version_probe_is_truthful() -> None:
    probe = probe_supported_mechanism(version_probe=True, which=lambda _name: None)
    assert probe.command_contract == COMMAND_CONTRACT
    assert probe.available is False
    assert probe.status == "unavailable"
    assert probe.version is None
    assert probe.substituted is False


def test_version_probe_records_injected_runner_output() -> None:
    def runner(argv: Any) -> SimpleNamespace:
        assert argv[-1] == "--version"
        assert "openai" not in argv
        return SimpleNamespace(returncode=0, stdout="codex-cli 0.42.0\n", stderr="")

    probe = probe_supported_mechanism(
        version_probe=True,
        runner=runner,
        binary_path="/usr/bin/codex",
        which=lambda _name: "/usr/bin/codex",
    )
    assert probe.available is True
    assert probe.version == "codex-cli 0.42.0"
    assert probe.command_contract == "codex exec"


def test_unavailable_supported_integration_returns_unavailable() -> None:
    adapter = CodexAdapter(transport=UnavailableCodexTransport())
    with pytest.raises(UnavailableCapabilityError) as exc:
        adapter.propose(_task(), _pack(), _route())
    assert exc.value.status == "unavailable"
    assert exc.value.code == "unavailable_capability"
    assert COMMAND_CONTRACT in str(exc.value.details.get("capability", COMMAND_CONTRACT))


def test_missing_live_permit_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LIVE_PERMIT_ENV, raising=False)
    adapter = CodexAdapter()
    with pytest.raises(UnavailableCapabilityError) as exc:
        execute_propose(adapter, _task(), _pack(), _route())
    assert exc.value.status == "unavailable"


def test_mechanism_substitution_is_rejected() -> None:
    transport = RecordedCodexTransport(_recorded_payload(), command_contract="openai api")
    adapter = CodexAdapter(transport=transport)
    with pytest.raises(UnavailableCapabilityError) as exc:
        adapter.propose(_task(), _pack(), _route())
    assert exc.value.status == "unavailable"
    with pytest.raises(BoundaryViolationError):
        adapter.propose(_task(), _pack(), _route(provider="grok", model="grok-4.6"))


def test_identities_survive_request_and_response() -> None:
    transport = RecordedCodexTransport(_recorded_payload())
    adapter = CodexAdapter(transport=transport)
    task, pack, route = _task(), _pack(), _route()
    result = execute_propose(adapter, task, pack, route)
    request = dict(transport.requests[0])
    assert request["task_id"] == task.task_id == result.proposal.task_id
    assert request["repository_state_cid"] == task.repository_state_cid
    assert request["pack_cid"] == pack.pack_cid
    assert request["route_cid"] == route.decision_cid == result.invocation.route_cid
    assert result.proposal.repository_state_cid == task.repository_state_cid
    assert result.invocation.task_id == task.task_id
    assert result.invocation.provider == route.provider
    assert "credentials" not in request
    assert "api_key" not in request
    assert request["command_contract"] == COMMAND_CONTRACT
    assert set(request) == {
        "schema",
        "command_contract",
        "task_id",
        "objective_id",
        "repository_state_cid",
        "pack_cid",
        "route_cid",
        "owned_paths",
        "declared_files",
        "provider",
        "model",
        "revision",
        "tier",
        "task",
        "context_pack",
        "route",
        "instruction",
    }


def test_identity_drift_in_response_is_rejected() -> None:
    adapter = _adapter(_recorded_payload(task_id="PCCE-OTHER"))
    with pytest.raises(IdentityInconsistentError):
        adapter.propose(_task(), _pack(), _route())
    adapter = _adapter(_recorded_payload(pack_cid=CID_D))
    with pytest.raises(IdentityInconsistentError):
        adapter.propose(_task(), _pack(), _route())


def test_file_scope_is_constrained() -> None:
    adapter = _adapter(_recorded_payload(declared_files=["src/demo/secret.py"]))
    with pytest.raises(BoundaryViolationError):
        execute_propose(adapter, _task(), _pack(), _route())
    escaped = _recorded_payload(
        declared_files=[OWNED],
        patch="diff --git a/../secret.py b/../secret.py\n",
    )
    adapter = _adapter(escaped)
    with pytest.raises(BoundaryViolationError):
        adapter.propose(_task(), _pack(), _route())


def test_patch_path_must_be_declared() -> None:
    foreign = (
        "diff --git a/src/other.py b/src/other.py\n"
        "--- a/src/other.py\n"
        "+++ b/src/other.py\n"
        "@@ -1 +1 @@\n"
        "-A\n"
        "+B\n"
    )
    adapter = _adapter(_recorded_payload(patch=foreign))
    with pytest.raises(BoundaryViolationError):
        adapter.propose(_task(), _pack(), _route())
    assert OWNED in extract_patch_paths(PATCH)


def test_cancellation_stops_invocation() -> None:
    token = CancellationToken()
    token.cancel()
    adapter = _adapter()
    with pytest.raises(ProofCancelledError):
        execute_propose(adapter, _task(), _pack(), _route(), cancellation=token)
    live = CancellationToken()
    adapter.cancel(live)
    assert live.cancelled is True
    with pytest.raises(ProofCancelledError):
        adapter.propose(_task(), _pack(), _route(), cancellation=live)


def test_bounded_logs_are_redacted_and_capped() -> None:
    secret_log = "Authorization: Bearer sk-abcdefghijklmnop OPENAI_API_KEY=secret-value\n"
    oversized = secret_log + ("x" * (MAX_LOG_BYTES + 2048))
    adapter = _adapter(_recorded_payload(), log=oversized)
    result = adapter.propose(_task(), _pack(), _route())
    assert len(result.log_bytes) <= MAX_LOG_BYTES
    assert b"sk-abcdefghijklmnop" not in result.log_bytes
    assert b"secret-value" not in result.log_bytes
    assert REDACTED.encode("utf-8") in result.log_bytes
    assert len(bound_and_redact_log(oversized)) == MAX_LOG_BYTES
    assert REDACTED in redact_log_text(secret_log)


def test_model_revision_tokens_cache_latency_cost_response_provenance_recorded() -> None:
    result = execute_propose(_adapter(), _task(), _pack(), _route())
    invocation = result.invocation
    assert invocation.schema == CODING_AGENT_INVOCATION_SCHEMA
    assert invocation.model == "gpt-5.6-terra"
    assert invocation.revision == "r1"
    assert invocation.token_count == 12
    assert invocation.cached_token_count == 4
    assert invocation.latency_ms == 9
    assert invocation.cost_micros == 3
    assert invocation.provenance == "replayed"
    assert invocation.response_artifact_cid is None
    assert invocation.usage_is_explicit()
    assert result.proposal.schema == PATCH_PROPOSAL_SCHEMA
    assert result.proposal.accepted is False
    assert result.proposal.approved is False
    assert result.accepted is False
    assert result.approved is False
    assert result.patch_bytes.decode("utf-8") == PATCH
    mapping = result.to_mapping()
    assert mapping["approval_authority"] is False
    assert mapping["canonical_branch_authority"] is False


def test_replay_cannot_claim_live() -> None:
    adapter = _adapter(_recorded_payload(provenance="live"))
    with pytest.raises(SimulatedPromotedError):
        adapter.propose(_task(), _pack(), _route())


def test_codex_adapter_is_a_coding_agent_adapter() -> None:
    adapter = _adapter()
    assert isinstance(adapter, CodingAgentAdapter)


def test_insufficient_context_pack_cannot_propose() -> None:
    adapter = _adapter()
    with pytest.raises(BoundaryViolationError):
        execute_propose(adapter, _task(), _pack(sufficiency="insufficient"), _route())


def test_malformed_and_forbidden_responses_fail_closed() -> None:
    with pytest.raises(MalformedError):
        parse_structured_proposal("not json")
    with pytest.raises(BoundaryViolationError):
        parse_structured_proposal(json.dumps({"accepted": True, "declared_files": [OWNED], "patch": PATCH}))
    adapter = _adapter(_recorded_payload(accepted=True))
    with pytest.raises(BoundaryViolationError):
        adapter.propose(_task(), _pack(), _route())


def test_request_contains_only_admitted_records() -> None:
    request = dict(build_codex_request(_task(), _pack(), _route()))
    assert request["task"]["schema"] == TASK_SPECIFICATION_SCHEMA
    assert request["context_pack"]["schema"] == CONTEXT_PACK_SCHEMA
    assert request["route"]["schema"] == MODEL_ROUTE_DECISION_SCHEMA
    assert "hidden_prompt" not in request
    assert "gold_labels" not in request
    assert request["owned_paths"] == [OWNED]


def test_fenced_json_proposal_is_admitted() -> None:
    body = "```json\n" + json.dumps(_recorded_payload()) + "\n```"
    transport = RecordedCodexTransport({"text": body, "kind": "recorded"})
    result = CodexAdapter(transport=transport).propose(_task(), _pack(), _route())
    assert result.proposal.declared_files == (OWNED,)
    assert result.invocation.token_count == 12


def test_installed_transport_requires_explicit_permit() -> None:
    transport = InstalledCodexTransport(permit_live=False, binary_path="/usr/bin/codex")
    with pytest.raises(UnavailableCapabilityError) as exc:
        CodexAdapter(transport=transport).propose(_task(), _pack(), _route())
    assert exc.value.status == "unavailable"


def test_injected_live_runner_records_live_usage_and_response_identity() -> None:
    body = json.dumps(_recorded_payload(provenance="live"))

    def runner(argv: Any, stdin: Any = None, env: Any = None, timeout: Any = None) -> SimpleNamespace:
        assert "exec" in [str(part).lower() for part in argv]
        assert "completions.create" not in argv
        assert env is not None
        assert "OPENAI_API_KEY" not in env
        assert stdin
        return SimpleNamespace(returncode=0, stdout=body, stderr="")

    transport = InstalledCodexTransport(
        permit_live=True,
        binary_path="/usr/bin/codex",
        runner=runner,
        environ={"PATH": "/usr/bin", "HOME": "/tmp"},
    )
    result = execute_propose(
        CodexAdapter(transport=transport, permit_live=True),
        _task(),
        _pack(),
        _route(),
    )
    assert result.invocation.provenance == "live"
    assert result.proposal.provenance == "live"
    assert result.invocation.response_artifact_cid is not None
    assert result.invocation.response_artifact_cid.startswith("b")
    assert result.invocation.usage_is_explicit()
    assert result.invocation.model == "gpt-5.6-terra"
    assert result.invocation.revision == "r1"
    assert result.invocation.token_count == 12
    assert result.invocation.cached_token_count == 4
    assert result.invocation.cost_micros == 3
    assert result.accepted is False
    assert result.approved is False


def test_opt_in_live_integration_records_truthful_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(LIVE_PERMIT_ENV, raising=False)
    probe = probe_supported_mechanism(version_probe=True)
    assert probe.command_contract == COMMAND_CONTRACT
    assert live_permit_granted() is False
    adapter = CodexAdapter(permit_live=False)
    with pytest.raises(UnavailableCapabilityError) as exc:
        execute_propose(adapter, _task(), _pack(), _route())
    assert exc.value.status == "unavailable"
    if probe.available:
        assert probe.version
        assert probe.binary_path
    else:
        assert probe.status == "unavailable"
        assert probe.reason == "unavailable"
