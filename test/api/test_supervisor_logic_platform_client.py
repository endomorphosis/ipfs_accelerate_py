"""LPC-110 regression: SupervisorLogicPlatformClient@1.

Acceptance:

* Client supports handshake, catalog, formalization, slice/obligation/plan,
  capability discovery, typed invocation, reconstruction, verification,
  receipts, counterexamples, and cache freshness.
* Requests bind task/tree/policy/plan/budget/network/cancellation/deadline/
  correlation/evidence/authority.
* Caller cannot overclaim authority.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import time
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_capabilities import (
    ProofProviderOperation,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_provider import (
    CancellationToken,
)
from ipfs_accelerate_py.agent_supervisor.proof.logic_platform_client import (
    CLIENT_GOAL_ID,
    CLIENT_OPERATIONS,
    CLIENT_REQUEST_SCHEMA,
    CLIENT_RESULT_SCHEMA,
    CLIENT_SCHEMA_VERSION,
    CLIENT_TASK_ID,
    CacheFreshnessStatus,
    ClientOperationStatus,
    LogicPlatformClientAuthorityError,
    LogicPlatformClientBindingError,
    LogicPlatformClientError,
    LogicPlatformClientHandshakeError,
    LogicPlatformClientRequest,
    LogicPlatformClientResult,
    SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE,
    SUPERVISOR_LOGIC_PLATFORM_CLIENT_VERSION,
    SupervisorLogicPlatformClient,
    TYPED_PROVIDER_OPERATIONS,
    check_authority_overclaim,
    get_logic_platform_client,
)
from ipfs_accelerate_py.agent_supervisor.proof.logic_provider_contract import (
    SupervisorLogicProviderFacade,
)
from ipfs_datasets_py.logic.backends.provider import LogicProviderRequest
from ipfs_datasets_py.logic.families.canonical_catalog import (
    DEFAULT_CANONICAL_CATALOG_SNAPSHOT,
)
from ipfs_datasets_py.logic.platform.manifest import (
    HandshakeRequirements,
    handshake,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CLIENT_SOURCE = (
    REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "proof"
    / "logic_platform_client.py"
)
CLIENT_NOTE = (
    REPO_ROOT
    / "data"
    / "agent_supervisor"
    / "logic_platform_canonicalization"
    / "notes"
    / "supervisor_client.md"
)

REQUIRED_ACCEPTANCE_SURFACES = (
    "handshake",
    "catalog",
    "formalization",
    "slice",
    "obligation",
    "plan",
    "capability",
    "invocation",
    "reconstruction",
    "verification",
    "receipt",
    "counterexample",
    "cache freshness",
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _budget(*, network: bool = False) -> ResourceBudget:
    return ResourceBudget(
        wall_time_ms=5_000,
        cpu_time_ms=2_000,
        memory_bytes=32 * 1024 * 1024,
        disk_bytes=1_024,
        max_processes=2,
        max_premises=16,
        max_output_bytes=8_192,
        model_token_limit=256,
        provider_quota=1,
        network_allowed=network,
    )


def _bound_request(**overrides: Any) -> LogicPlatformClientRequest:
    values: dict[str, Any] = {
        "task_id": "LPC-110",
        "tree_id": "tree:lpc-110",
        "policy_id": "policy:implementation-daemon",
        "plan_id": "plan:lpc-110",
        "correlation_id": "corr:lpc-110-1",
        "evidence_kind": "candidate",
        "authority_ceiling": "candidate",
        "resource_budget": _budget(),
        "network_allowed": False,
        "deadline_unix_ms": int(time.time() * 1000) + 60_000,
        "request_id": "req-lpc-110-1",
    }
    values.update(overrides)
    return LogicPlatformClientRequest(**values)


class FixtureLogicProvider:
    provider_id = "fixture.logic.client"
    provider_version = "1.0.0"
    protocol_version = 1

    def __init__(self) -> None:
        self.requests: list[LogicProviderRequest] = []

    def _invoke(self, request: LogicProviderRequest) -> dict[str, object]:
        self.requests.append(request)
        return {
            "echo": dict(request.payload),
            "operation": request.operation.value,
            "provider_claimed_authority": "kernel",
        }

    capability = _invoke
    translate = _invoke
    prove = _invoke
    reconstruct = _invoke
    verify = _invoke
    attest = _invoke


def _facade() -> SupervisorLogicProviderFacade:
    return SupervisorLogicProviderFacade(
        provider_id="fixture.logic.client",
        provider_version="1.0.0",
        provider=FixtureLogicProvider(),
    )


def _client(
    *,
    handshaken: bool = True,
    provider: bool = True,
    require_handshake: bool = True,
) -> SupervisorLogicPlatformClient:
    client = SupervisorLogicPlatformClient(
        provider_facade=_facade() if provider else None,
        require_handshake=require_handshake,
    )
    if handshaken:
        result = client.handshake()
        assert result.compatible is True
    return client


# ---------------------------------------------------------------------------
# Module / note contracts
# ---------------------------------------------------------------------------


def test_client_module_has_no_top_level_datasets_import() -> None:
    source = CLIENT_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("ipfs_datasets_py"), alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("ipfs_datasets_py"), module


def test_client_interface_and_operations_are_stable() -> None:
    client = get_logic_platform_client(require_handshake=False)
    payload = client.to_dict()
    assert client.interface == SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE
    assert client.version == SUPERVISOR_LOGIC_PLATFORM_CLIENT_VERSION
    assert payload["schema_version"] == CLIENT_SCHEMA_VERSION
    assert payload["task_id"] == CLIENT_TASK_ID
    assert payload["goal_id"] == CLIENT_GOAL_ID
    assert tuple(payload["operations"]) == CLIENT_OPERATIONS
    for operation in (
        "handshake",
        "catalog",
        "formalize",
        "create_slice",
        "create_obligation",
        "create_plan",
        "discover_capabilities",
        "invoke",
        "reconstruct",
        "verify",
        "receipt",
        "counterexample",
        "cache_freshness",
    ):
        assert operation in CLIENT_OPERATIONS
    assert TYPED_PROVIDER_OPERATIONS == {
        "capability",
        "translate",
        "prove",
        "reconstruct",
        "verify",
        "attest",
    }


def test_supervisor_client_note_covers_acceptance_surfaces() -> None:
    assert CLIENT_NOTE.is_file(), f"missing note: {CLIENT_NOTE}"
    text = CLIENT_NOTE.read_text(encoding="utf-8")
    assert "SupervisorLogicPlatformClient@1" in text
    assert "LPC-110" in text
    assert "LPC-G110" in text
    lowered = text.casefold()
    for surface in REQUIRED_ACCEPTANCE_SURFACES:
        assert surface.casefold() in lowered, surface
    for binding in (
        "task_id",
        "tree_id",
        "policy_id",
        "plan_id",
        "resource_budget",
        "network_allowed",
        "cancellation",
        "deadline",
        "correlation",
        "evidence_kind",
        "authority_ceiling",
    ):
        assert binding in text
    assert "overclaim" in lowered


# ---------------------------------------------------------------------------
# Request binding / authority
# ---------------------------------------------------------------------------


def test_request_requires_all_binding_axes() -> None:
    request = _bound_request()
    payload = request.to_dict()
    assert payload["schema_version"] == CLIENT_REQUEST_SCHEMA
    for field in (
        "task_id",
        "tree_id",
        "policy_id",
        "plan_id",
        "resource_budget",
        "network_allowed",
        "cancellation",
        "deadline_unix_ms",
        "correlation_id",
        "evidence_kind",
        "authority_ceiling",
    ):
        assert field in payload
    assert payload["binding_digest"].startswith("sha256:")
    restored = LogicPlatformClientRequest.from_dict(payload)
    assert restored.task_id == request.task_id
    assert restored.binding_digest() == request.binding_digest()


def test_missing_binding_fields_fail_closed() -> None:
    with pytest.raises(LogicPlatformClientBindingError, match="task_id"):
        LogicPlatformClientRequest(
            task_id="",
            tree_id="tree:1",
            policy_id="policy:1",
            plan_id="plan:1",
            correlation_id="corr:1",
            evidence_kind="candidate",
            authority_ceiling="candidate",
        )


def test_caller_cannot_overclaim_authority() -> None:
    with pytest.raises(LogicPlatformClientAuthorityError, match="overclaims"):
        check_authority_overclaim(
            evidence_kind="candidate",
            authority_ceiling="kernel",
        )
    with pytest.raises(LogicPlatformClientAuthorityError, match="overclaims"):
        _bound_request(
            evidence_kind="candidate",
            authority_ceiling="reconstruction",
        )
    with pytest.raises(LogicPlatformClientAuthorityError, match="kernel"):
        check_authority_overclaim(
            evidence_kind="advisory",
            authority_ceiling="kernel",
        )
    # Allowed coordinate.
    check_authority_overclaim(
        evidence_kind="candidate",
        authority_ceiling="candidate",
    )


def test_network_allowed_cannot_exceed_budget() -> None:
    with pytest.raises(LogicPlatformClientBindingError, match="network_allowed"):
        _bound_request(
            resource_budget=_budget(network=False),
            network_allowed=True,
        )


def test_cancelled_and_expired_requests_fail_closed() -> None:
    client = _client(handshaken=True)
    token = CancellationToken()
    token.cancel()
    with pytest.raises(LogicPlatformClientError, match="cancelled"):
        client.catalog(_bound_request(cancellation=token, request_id="req-cancel"))
    with pytest.raises(LogicPlatformClientError, match="deadline"):
        client.catalog(
            _bound_request(
                deadline_unix_ms=1,
                request_id="req-expired",
            )
        )


# ---------------------------------------------------------------------------
# Handshake / catalog
# ---------------------------------------------------------------------------


def test_handshake_is_compatible_and_pins_client_adapter() -> None:
    client = SupervisorLogicPlatformClient(require_handshake=True)
    assert client.handshaken is False
    assert client.datasets_import_is_lazy() is True
    result = client.handshake()
    assert result.compatible is True
    assert client.handshaken is True
    assert SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE in (
        result.manifest.compatible_adapter_versions
    )
    # Direct platform handshake remains available for comparison.
    direct = handshake(
        HandshakeRequirements(
            required_adapter_versions=(SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE,)
        )
    )
    assert direct.compatible is True


def test_operations_require_handshake() -> None:
    client = SupervisorLogicPlatformClient(require_handshake=True)
    with pytest.raises(LogicPlatformClientHandshakeError, match="handshake"):
        client.catalog(_bound_request())


def test_catalog_returns_sealed_root_and_inventory() -> None:
    client = _client()
    result = client.catalog(_bound_request(request_id="req-catalog"))
    assert result.status is ClientOperationStatus.DECLARATIVE
    assert result.ok is True
    body = result.result or {}
    assert body["catalog_root"] == DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
    assert body["catalog_digest"] == DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_digest
    assert body["family_count"] >= 1
    assert "vocabulary" in body
    assert result.cache_freshness is CacheFreshnessStatus.CURRENT


# ---------------------------------------------------------------------------
# Formalization / slice / obligation / plan
# ---------------------------------------------------------------------------


def test_formalize_slice_obligation_plan_pipeline() -> None:
    client = _client()
    request = _bound_request(request_id="req-pipeline")

    formalized = client.formalize(
        request,
        document_id="document:lpc-110",
        source_digest=_digest("source-lpc-110"),
        expression_id="expression:lpc-110",
        expression_digest=_digest("expression-lpc-110"),
        statement="P implies P",
        domain="software",
        family="first_order",
    )
    assert formalized.status is ClientOperationStatus.SUCCEEDED, formalized.error
    artifact = (formalized.result or {})["artifact"]
    assert artifact["interface"] == "FormalizationArtifact@3"
    assert artifact["document_id"] == "document:lpc-110"

    sliced = client.create_slice(
        request,
        slice_id="slice:lpc-110",
        document_id="document:lpc-110",
        source_digest=_digest("source-lpc-110"),
        expression_id="expression:lpc-110",
        expression_digest=_digest("expression-lpc-110"),
        domain="software",
        family="first_order",
        property_id="theorem",
        view="source",
        notation="canonical_text",
    )
    assert sliced.status is ClientOperationStatus.SUCCEEDED, sliced.error
    slice_body = (sliced.result or {})["slice"]
    assert slice_body["interface"] == "DomainLogicSlice@2"
    assert slice_body["status"] == "admitted"

    obligation = client.create_obligation(
        request,
        slice_payload=slice_body,
        obligation_id="obligation:lpc-110",
        statement="P implies P",
        encoding="smt_lib",
        evidence_kind="candidate",
        bounds={
            "timeout_ms": 1_000,
            "max_steps": 1_024,
            "max_memory_bytes": 16 * 1024 * 1024,
            "max_output_bytes": 65_536,
        },
    )
    assert obligation.status is ClientOperationStatus.SUCCEEDED, obligation.error
    obl = (obligation.result or {})["obligation"]
    assert obl["interface"] == "LogicObligation@2"
    assert obl["authority_ceiling"] == "candidate"
    assert obl["slice_id"] == "slice:lpc-110"

    plan = client.create_plan(
        request,
        formal_goal_id="goal:lpc-110",
        graph_id="graph:lpc-110",
    )
    assert plan.status is ClientOperationStatus.SUCCEEDED, plan.error
    plan_body = (plan.result or {})["plan"]
    assert plan_body["interface"] == "GoalDirectedProofPlan@1"
    assert plan_body["proof_claimed"] is False
    assert plan_body["completion_claimed"] is False
    assert plan_body["tree_id"] == request.tree_id
    assert plan.authority_ceiling == "candidate"


# ---------------------------------------------------------------------------
# Capability / typed invocation / reconstruct / verify
# ---------------------------------------------------------------------------


def test_discover_capabilities_is_declarative() -> None:
    client = _client(provider=True)
    result = client.discover_capabilities(_bound_request(request_id="req-caps"))
    assert result.status is ClientOperationStatus.DECLARATIVE
    body = result.result or {}
    assert body["availability_is_not_proof"] is True
    assert set(body["provider_operations"]) == TYPED_PROVIDER_OPERATIONS
    assert body["provider_count"] >= 1


def test_typed_invocation_reconstruct_and_verify() -> None:
    client = _client(provider=True)
    request = _bound_request(request_id="req-invoke")
    for operation in (
        ProofProviderOperation.PROVE,
        "reconstruct",
        "verify",
        "capability",
        "translate",
        "attest",
    ):
        result = client.invoke(
            operation,
            request,
            payload={"obligation_id": "obligation:lpc-110"},
        )
        assert result.status is ClientOperationStatus.SUCCEEDED, (
            operation,
            result.error,
        )
        body = result.result or {}
        assert body["authority_upgraded"] is False
        assert body["provider_id"] == "fixture.logic.client"
        # Provider may claim kernel; client retains request ceiling.
        assert result.authority_ceiling == "candidate"

    # Convenience wrappers.
    reconstructed = client.reconstruct(
        request, payload={"artifact_id": "artifact:1"}
    )
    assert reconstructed.ok
    verified = client.verify(request, payload={"receipt_id": "receipt:1"})
    assert verified.ok


def test_invoke_without_facade_is_unavailable() -> None:
    client = _client(provider=False)
    result = client.invoke("prove", _bound_request(request_id="req-no-facade"))
    assert result.status is ClientOperationStatus.UNAVAILABLE
    assert "provider_facade" in (result.error or "")


def test_unknown_provider_operation_fails_closed() -> None:
    client = _client()
    with pytest.raises(LogicPlatformClientBindingError, match="unsupported"):
        client.invoke("install", _bound_request(request_id="req-bad-op"))


# ---------------------------------------------------------------------------
# Receipts / counterexamples / cache freshness
# ---------------------------------------------------------------------------


def test_receipt_projection_never_claims_proof() -> None:
    client = _client()
    result = client.receipt(
        _bound_request(request_id="req-receipt"),
        validation_result={
            "valid": True,
            "content_id": "content:1",
            "contract_id": "contract:1",
            "issues": [],
        },
    )
    assert result.status is ClientOperationStatus.SUCCEEDED, result.error
    body = result.result or {}
    assert body["proof_success"] is False
    assert body["authority"] == "none"
    assert result.authority_ceiling == "candidate"


def test_counterexample_is_never_a_proof() -> None:
    client = _client()
    result = client.counterexample(
        _bound_request(request_id="req-cex"),
        kind="smt_model",
        statement="P is not valid",
        witness={"assignment": {"P": False}},
    )
    assert result.status is ClientOperationStatus.SUCCEEDED, result.error
    body = result.result or {}
    assert body["is_proof"] is False
    assert body["kind"] == "smt_model"
    assert "counterexample" in body


def test_cache_freshness_current_stale_unknown_invalidated() -> None:
    client = _client()
    now = int(time.time() * 1000)
    current = client.cache_freshness(
        _bound_request(request_id="req-fresh-current"),
        stored_at_unix_ms=now - 1_000,
        ttl_ms=10_000,
        now_unix_ms=now,
    )
    assert current.status is ClientOperationStatus.SUCCEEDED
    assert current.cache_freshness is CacheFreshnessStatus.CURRENT
    assert (current.result or {})["is_fresh"] is True

    stale = client.cache_freshness(
        _bound_request(request_id="req-fresh-stale"),
        stored_at_unix_ms=now - 50_000,
        ttl_ms=10_000,
        now_unix_ms=now,
    )
    assert stale.cache_freshness is CacheFreshnessStatus.STALE
    assert (stale.result or {})["is_fresh"] is False

    unknown = client.cache_freshness(
        _bound_request(request_id="req-fresh-unknown"),
    )
    assert unknown.cache_freshness is CacheFreshnessStatus.UNKNOWN

    invalidated = client.cache_freshness(
        _bound_request(request_id="req-fresh-invalid"),
        invalidated=True,
        stored_at_unix_ms=now,
        ttl_ms=10_000,
    )
    assert invalidated.cache_freshness is CacheFreshnessStatus.INVALIDATED


# ---------------------------------------------------------------------------
# Result envelope
# ---------------------------------------------------------------------------


def test_client_result_rejects_authority_overclaim() -> None:
    with pytest.raises(LogicPlatformClientAuthorityError):
        LogicPlatformClientResult(
            operation="prove",
            status=ClientOperationStatus.SUCCEEDED,
            request_id="req-1",
            correlation_id="corr-1",
            evidence_kind="candidate",
            authority_ceiling="kernel",
            result={"ok": True},
        )


def test_client_result_round_trip_fields() -> None:
    result = LogicPlatformClientResult(
        operation="catalog",
        status=ClientOperationStatus.DECLARATIVE,
        request_id="req-1",
        correlation_id="corr-1",
        result={"count": 1},
        evidence_kind="candidate",
        authority_ceiling="candidate",
        cache_freshness=CacheFreshnessStatus.CURRENT,
        binding_digest="sha256:" + "ab" * 32,
        duration_ms=3,
    )
    payload = result.to_dict()
    assert payload["schema_version"] == CLIENT_RESULT_SCHEMA
    assert payload["interface"] == SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE
    assert payload["ok"] is True
    assert payload["cache_freshness"] == "current"


def test_reimport_client_module_stays_lazy_until_handshake() -> None:
    module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.proof.logic_platform_client"
    )
    client = module.SupervisorLogicPlatformClient(require_handshake=True)
    assert client.datasets_import_is_lazy() is True
    client.handshake()
    # After handshake, datasets has been loaded for the manifest path.
    assert client.handshaken is True
