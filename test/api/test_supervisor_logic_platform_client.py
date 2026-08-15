"""LPC-110 regression: SupervisorLogicPlatformClient@1.

The durable contract note is
``data/agent_supervisor/logic_platform_canonicalization/notes/supervisor_client.md``.

Acceptance:

* Client supports handshake, catalog, formalization, slice/obligation/plan,
  capability discovery, typed invocation, reconstruction, verification,
  receipts, counterexamples, and cache freshness.
* Requests bind task/tree/policy/plan/budget/network/cancellation/deadline/
  correlation/evidence/authority.
* Caller cannot overclaim authority.
* Module import never loads ``ipfs_datasets_py``.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import time
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.logic_platform_client import (
    CLIENT_GOAL_ID,
    CLIENT_SCHEMA_VERSION,
    CLIENT_TASK_ID,
    DEFAULT_REQUIRED_ADAPTER_VERSIONS,
    REQUIRED_CONTEXT_FIELDS,
    SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE,
    SUPERVISOR_LOGIC_PLATFORM_CLIENT_VERSION,
    CacheFreshnessReport,
    ClientCounterexampleView,
    ClientInvocationResult,
    ClientReceiptView,
    ClientRequestContext,
    LogicPlatformClientAuthorityError,
    LogicPlatformClientFreshnessError,
    LogicPlatformClientHandshakeError,
    SupervisorLogicPlatformClient,
    _clear_import_cache_for_tests,
    check_authority_overclaim,
    get_logic_platform_client,
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


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _context(**overrides: Any) -> ClientRequestContext:
    payload: dict[str, Any] = {
        "task_id": "task:lpc-110",
        "tree_id": "tree:lpc-110",
        "policy_id": "policy:lpc-110",
        "plan_id": "plan:lpc-110",
        "budget": {"wall_time_ms": 1000, "memory_bytes": 1024},
        "network_allowed": False,
        "evidence_kind": "candidate",
        "authority_ceiling": "advisory",
    }
    payload.update(overrides)
    return ClientRequestContext(**payload)


def _client(**kwargs: Any) -> SupervisorLogicPlatformClient:
    defaults: dict[str, Any] = {
        "require_handshake": True,
        "default_context": _context(),
    }
    defaults.update(kwargs)
    return SupervisorLogicPlatformClient(**defaults)


def _sha256_hex(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _minimal_slice_dict() -> dict[str, Any]:
    from ipfs_datasets_py.logic.families.namespaces import (
        family_id,
        notation_id,
        profile_id,
        property_id,
        view_id,
    )

    return {
        "slice_id": "slice:lpc-110-1",
        "domain": "software",
        "document_id": "doc:lpc-110",
        "source_digest": _sha256_hex("source"),
        "expression_id": "expr:lpc-110",
        "expression_digest": _sha256_hex("expression"),
        "family": family_id("first_order").to_dict(),
        "profile": profile_id("core").to_dict(),
        "property": property_id("theorem").to_dict(),
        "view": view_id("surface").to_dict(),
        "notation": notation_id("tptp").to_dict(),
        "status": "admitted",
    }


def _bounds_dict() -> dict[str, int]:
    return {
        "timeout_ms": 1000,
        "max_steps": 100,
        "max_memory_bytes": 1024 * 1024,
        "max_output_bytes": 64 * 1024,
    }


def _cache_key_fields(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "source": _digest("source"),
        "expression": _digest("expression"),
        "formalization": _digest("formalization"),
        "slice": _digest("slice"),
        "obligation": _digest("obligation"),
        "assumptions": _digest("assumptions"),
        "bounds": _digest("bounds"),
        "translation": _digest("translation"),
        "provider": "provider.z3",
        "environment": _digest("environment"),
        "policy": _digest("policy"),
        "schema": _digest("schema"),
        "checker": "checker.z3",
        "network_policy": _digest("network-deny"),
        "evidence_kind": "candidate",
        "authority_ceiling": "advisory",
    }
    fields.update(overrides)
    return fields


class _FakeProvider:
    provider_id = "provider.fixture"
    provider_version = "1.0.0"

    def __init__(self, response_factory: Any | None = None) -> None:
        self.response_factory = response_factory
        self.calls: list[Any] = []

    def invoke(self, request: Any) -> Any:
        self.calls.append(request)
        if self.response_factory is not None:
            return self.response_factory(request)
        from ipfs_datasets_py.logic.backends.response_v2 import ProviderResponseV2

        op = getattr(getattr(request, "operation", None), "value", None) or str(
            getattr(request, "operation", "capability")
        )
        return ProviderResponseV2.succeeded(
            request_id=str(getattr(request, "request_id", "req:fixture")),
            operation=op,
            provider_id=self.provider_id,
            provider_version=self.provider_version,
        )


# ---------------------------------------------------------------------------
# Note / interface surface
# ---------------------------------------------------------------------------


def test_client_note_exists_and_binds_task() -> None:
    assert CLIENT_NOTE.is_file()
    text = CLIENT_NOTE.read_text(encoding="utf-8")
    assert "LPC-110" in text
    assert "SupervisorLogicPlatformClient@1" in text
    assert "handshake" in text.lower()
    assert "cache freshness" in text.lower() or "cache_freshness" in text
    assert "authority" in text.lower()


def test_client_interface_constants_are_stable() -> None:
    client = _client(require_handshake=False)
    payload = client.to_dict()
    assert client.interface == SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE
    assert client.version == SUPERVISOR_LOGIC_PLATFORM_CLIENT_VERSION
    assert payload["schema_version"] == CLIENT_SCHEMA_VERSION
    assert payload["task_id"] == CLIENT_TASK_ID == "LPC-110"
    assert payload["goal_id"] == CLIENT_GOAL_ID == "LPC-G110"
    assert "SupervisorCanonicalLogicAdapter@1" in DEFAULT_REQUIRED_ADAPTER_VERSIONS
    assert SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE in DEFAULT_REQUIRED_ADAPTER_VERSIONS
    for field in (
        "task_id",
        "tree_id",
        "policy_id",
        "plan_id",
        "budget",
        "network_allowed",
        "cancellation",
        "deadline_unix_ms",
        "correlation_id",
        "evidence_kind",
        "authority_ceiling",
    ):
        assert field in REQUIRED_CONTEXT_FIELDS


def test_importing_client_module_does_not_import_datasets_package() -> None:
    source = CLIENT_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("ipfs_datasets_py"), alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("ipfs_datasets_py"), module

    _clear_import_cache_for_tests()
    module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.proof.logic_platform_client"
    )
    client = module.SupervisorLogicPlatformClient(require_handshake=False)
    assert client.datasets_import_is_lazy is True


# ---------------------------------------------------------------------------
# Context / authority
# ---------------------------------------------------------------------------


def test_request_context_binds_required_fields_and_mints_correlation() -> None:
    ctx = ClientRequestContext(
        task_id="task:a",
        tree_id="tree:a",
        policy_id="policy:a",
        evidence_kind="candidate",
        authority_ceiling="advisory",
    )
    assert ctx.correlation_id.startswith("corr:")
    payload = ctx.to_dict()
    assert payload["task_id"] == "task:a"
    assert payload["network_allowed"] is False
    assert isinstance(ctx.budget, MappingProxyType) or isinstance(ctx.budget, dict)


def test_authority_overclaim_rejects_kernel_with_candidate() -> None:
    with pytest.raises(LogicPlatformClientAuthorityError):
        check_authority_overclaim("kernel", "candidate")
    with pytest.raises(LogicPlatformClientAuthorityError):
        ClientRequestContext(
            task_id="task:a",
            tree_id="tree:a",
            policy_id="policy:a",
            evidence_kind="candidate",
            authority_ceiling="kernel",
        )
    with pytest.raises(LogicPlatformClientAuthorityError):
        check_authority_overclaim("reconstruction", "model")
    # Allowed pairings remain open.
    check_authority_overclaim("advisory", "candidate")
    check_authority_overclaim("kernel", "kernel")


# ---------------------------------------------------------------------------
# Handshake / catalog
# ---------------------------------------------------------------------------


def test_handshake_default_path_is_compatible_without_git() -> None:
    client = _client()
    result = client.handshake()
    assert result.compatible is True
    assert client.handshake_compatible is True
    assert client.catalog_root().startswith("b") or client.catalog_root()
    assert client.catalog_digest().startswith("sha256:")
    snapshot = client.catalog()
    assert snapshot.content_root == client.catalog_root()


def test_handshake_typed_incompatibility_does_not_raise() -> None:
    from ipfs_datasets_py.logic.platform.manifest import HandshakeRequirements

    client = _client()
    result = client.handshake(
        HandshakeRequirements(
            required_adapter_versions=("SupervisorLogicPlatformClient@99",)
        )
    )
    assert result.compatible is False
    assert client.handshake_compatible is False
    with pytest.raises(LogicPlatformClientHandshakeError):
        client.catalog()


def test_operations_blocked_until_compatible_handshake() -> None:
    client = _client(require_handshake=True)
    with pytest.raises(LogicPlatformClientHandshakeError):
        client.catalog()
    client.handshake()
    assert client.catalog() is not None


# ---------------------------------------------------------------------------
# Slice / obligation / plan / formalize
# ---------------------------------------------------------------------------


def test_create_slice_and_obligation_and_backend_request() -> None:
    client = _client()
    client.handshake()
    slice_ = client.create_slice(_minimal_slice_dict(), context=_context())
    assert getattr(slice_, "slice_id", None) == "slice:lpc-110-1"
    obligation = client.create_obligation(
        slice_=slice_,
        bounds=_bounds_dict(),
        evidence_kind="candidate",
        authority_ceiling="advisory",
        context=_context(),
    )
    assert getattr(obligation, "bounds", None) is not None
    request = client.create_backend_request(obligation=obligation)
    assert getattr(request, "request_id", None) or getattr(
        request, "content_digest", None
    )


def test_create_plan_is_proposal_only() -> None:
    from ipfs_datasets_py.logic.software_verification.tactician.contracts import (
        GoalDirectedProofPlan,
        PlanStatus,
    )

    client = _client()
    client.handshake()
    plan = GoalDirectedProofPlan(
        plan_id="plan:lpc-110",
        formal_goal_id="goal:lpc-110",
        graph_id="graph:lpc-110",
        tree_id="tree:lpc-110",
        candidates=(),
        status=PlanStatus.DRAFT,
    )
    admitted = client.create_plan(plan, context=_context())
    assert admitted.proof_claimed is False
    assert admitted.completion_claimed is False


def test_formalize_admits_artifact() -> None:
    from ipfs_datasets_py.logic.families.namespaces import (
        family_id,
        notation_id,
        profile_id,
        view_id,
    )
    from ipfs_datasets_py.logic.formalization.artifacts_v3 import (
        FormalizationArtifactStatus,
        FormalizationArtifactV3,
    )

    client = _client()
    client.handshake()
    artifact = FormalizationArtifactV3(
        artifact_id="artifact:lpc-110",
        sample_id="sample:lpc-110",
        domain="software",
        document_id="doc:lpc-110",
        source_digest=_sha256_hex("source"),
        expression_id="expr:lpc-110",
        expression_digest=_sha256_hex("expression"),
        family=family_id("first_order"),
        profile=profile_id("core"),
        view=view_id("surface"),
        notation=notation_id("tptp"),
        status=FormalizationArtifactStatus.OK,
    )
    admitted = client.formalize(artifact, context=_context())
    assert admitted is artifact
    assert getattr(admitted, "artifact_id", None) == "artifact:lpc-110"


# ---------------------------------------------------------------------------
# Capability / invoke / reconstruct / verify
# ---------------------------------------------------------------------------


def test_discover_capabilities_is_non_executable_and_untrusted() -> None:
    client = _client(provider=_FakeProvider())
    client.handshake()
    result = client.discover_capabilities(
        provider_id="provider.fixture",
        context=_context(),
    )
    assert isinstance(result, ClientInvocationResult)
    assert result.operation == "capability"
    assert result.evidence_authority == "advisory"
    receipt = client.project_receipt(result)
    assert isinstance(receipt, ClientReceiptView)
    assert receipt.authority == "advisory"
    assert receipt.simulated is False


def test_unbound_capability_succeeds_advisory_but_executable_requires_provider() -> None:
    client = _client(provider=None)
    client.handshake()
    result = client.discover_capabilities(context=_context())
    assert result.provider_id == "unbound"
    assert result.evidence_authority == "advisory"

    from ipfs_datasets_py.logic.backends.protocol_v2 import CapabilityRequestV2

    client.invoke(CapabilityRequestV2(provider_id="x"), context=_context())

    # Incomplete executable body fails closed at admission (no provider needed).
    with pytest.raises(Exception):
        client.invoke(
            {
                "operation": "prove",
                "bounds": _bounds_dict(),
            },
            context=_context(),
        )


def test_invoke_executable_requires_bounds_and_backend_request() -> None:
    client = _client(provider=_FakeProvider())
    client.handshake()
    with pytest.raises(Exception):
        client.invoke({"operation": "prove"}, context=_context())

    slice_ = client.create_slice(_minimal_slice_dict(), context=_context())
    obligation = client.create_obligation(
        slice_=slice_,
        bounds=_bounds_dict(),
        context=_context(),
    )
    backend = client.create_backend_request(obligation=obligation)
    result = client.invoke(
        {
            "operation": "prove",
            "bounds": _bounds_dict(),
            "backend_request": backend.to_dict(),
            "mode": "prove",
        },
        context=_context(),
    )
    assert result.operation == "prove"
    assert result.evidence_authority == "advisory"


def test_reconstruct_and_verify_dispatch() -> None:
    client = _client(provider=_FakeProvider())
    client.handshake()
    slice_ = client.create_slice(_minimal_slice_dict(), context=_context())
    obligation = client.create_obligation(
        slice_=slice_,
        bounds=_bounds_dict(),
        context=_context(),
    )
    backend = client.create_backend_request(obligation=obligation)
    reconstruct_body = {
        "bounds": _bounds_dict(),
        "backend_request": backend.to_dict(),
        "candidate_digest": _sha256_hex("candidate"),
    }
    reconstructed = client.reconstruct(reconstruct_body, context=_context())
    assert reconstructed.operation == "reconstruct"
    verify_body = {
        "bounds": _bounds_dict(),
        "backend_request": backend.to_dict(),
        "evidence_digest": _sha256_hex("evidence"),
    }
    verified = client.verify(verify_body, context=_context())
    assert verified.operation == "verify"


# ---------------------------------------------------------------------------
# Receipts / counterexamples
# ---------------------------------------------------------------------------


def test_project_receipt_marks_simulated() -> None:
    client = _client(provider=_FakeProvider())
    client.handshake()
    result = client.discover_capabilities(context=_context())
    receipt = client.project_receipt(result, simulated=True)
    assert receipt.simulated is True
    assert receipt.authority == "advisory"
    assert receipt.context.task_id == "task:lpc-110"


def test_project_counterexample_strips_private_material() -> None:
    client = _client(require_handshake=False)
    view = client.project_counterexample(
        {
            "request_id": "req:ce-1",
            "public_step": 3,
            "hidden_witness": "SECRET",
            "api_key": "k",
            "stdout": "dump",
            "source_code": "fn main() {}",
            "nested": {"private_inputs": [1, 2], "ok": True},
        }
    )
    assert isinstance(view, ClientCounterexampleView)
    assert view.redacted is True
    assert view.authority == "advisory"
    assert "hidden_witness" in view.stripped_keys
    assert "api_key" in view.stripped_keys
    assert "source_code" in view.stripped_keys
    assert "public_step" in view.public_fields
    assert "hidden_witness" not in view.public_fields
    assert "private_inputs" not in view.public_fields.get("nested", {})
    assert view.public_fields["nested"]["ok"] is True


# ---------------------------------------------------------------------------
# Cache freshness
# ---------------------------------------------------------------------------


def test_build_cache_key_and_freshness_fail_closed() -> None:
    client = _client(require_handshake=False)
    key = client.build_cache_key(**_cache_key_fields())
    assert key.key_id.startswith("canonical-proof-cache-key:")

    # Matching stored key without TTL is fresh.
    report = client.check_cache_freshness(request_key=key, stored_key=key)
    assert report.fresh is True
    assert report.reason == "fresh"

    # Cross-environment hits fail closed.
    other = client.build_cache_key(
        **_cache_key_fields(environment=_digest("other-env"))
    )
    mismatch = client.check_cache_freshness(request_key=key, stored_key=other)
    assert mismatch.fresh is False
    assert mismatch.reason in {
        "environment_mismatch",
        "cross_environment_hit",
        "stale_entry",
    }

    # Simulated evidence fails closed.
    simulated = client.check_cache_freshness(
        request_key=key,
        stored_key=key,
        stored_entry={"simulated": True},
    )
    assert simulated.fresh is False
    assert simulated.reason == "simulated_evidence"

    # TTL expiry fails closed.
    expired = client.check_cache_freshness(
        request_key=key,
        stored_key=key,
        stored_entry={"expires_at_unix_s": int(time.time()) - 10},
    )
    assert expired.fresh is False
    assert expired.reason == "ttl_expired"

    with pytest.raises(LogicPlatformClientFreshnessError):
        client.require_cache_fresh(
            request_key=key,
            stored_entry={"simulated_evidence": True},
        )

    # Candidate-as-kernel rejected at key build.
    with pytest.raises(Exception):
        client.build_cache_key(
            **_cache_key_fields(
                evidence_kind="candidate",
                authority_ceiling="authoritative",
            )
        )


def test_get_logic_platform_client_singleton() -> None:
    first = get_logic_platform_client(reset=True, require_handshake=False)
    second = get_logic_platform_client()
    assert first is second
    third = get_logic_platform_client(reset=True, require_handshake=False)
    assert third is not first


def test_vocabulary_projection_uses_adapter_not_hand_maps() -> None:
    client = _client(require_handshake=False)
    adapter = client.adapter()
    projection = adapter.project_analysis_family("tdfol")
    assert projection.canonical_id
    assert projection.domain == "analysis_family"
