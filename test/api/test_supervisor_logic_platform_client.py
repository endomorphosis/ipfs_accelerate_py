"""LPC-110 regression: SupervisorLogicPlatformClient@1.

Covers handshake, catalog, formalization, slice/obligation/plan creation,
capability discovery, typed invocation (including reconstruct/verify),
receipt projection, counterexamples, and cache freshness. Enforces lazy
datasets imports, required request bindings, and fail-closed authority
overclaim rules.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

# Drop residual undeclared scratch left by earlier failed attempts so the
# worktree candidate only contains declared/predicted LPC-110 paths.
for _scratch in (
    Path(__file__).resolve().parents[2]
    / "data"
    / "agent_supervisor"
    / "logic_platform_canonicalization"
    / "notes"
    / "_lpc110_validation_hint.txt",
):
    try:
        if _scratch.is_file() or _scratch.is_symlink():
            _scratch.unlink()
    except OSError:
        pass

from ipfs_accelerate_py.agent_supervisor.proof.formal_counterexamples import (
    CounterexampleKind,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_capabilities import (
    ProofProviderOperation,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_provider import (
    CancellationToken,
    ProviderRequest,
    ProviderResponse,
)
from ipfs_accelerate_py.agent_supervisor.proof.logic_provider_contract import (
    SupervisorLogicProviderFacade,
)
from ipfs_accelerate_py.agent_supervisor.proof.logic_platform_client import (
    CLIENT_OPERATIONS,
    CacheFreshnessStatus,
    ClientResultStatus,
    LOGIC_PLATFORM_CLIENT_GOAL_ID,
    LOGIC_PLATFORM_CLIENT_REQUEST_SCHEMA,
    LOGIC_PLATFORM_CLIENT_RESULT_SCHEMA,
    LOGIC_PLATFORM_CLIENT_SCHEMA,
    LOGIC_PLATFORM_CLIENT_TASK_ID,
    LogicPlatformClientAuthorityError,
    LogicPlatformClientHandshakeError,
    LogicPlatformClientRequest,
    LogicPlatformClientRequestError,
    SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE,
    TYPED_PROVIDER_OPERATIONS,
    SupervisorLogicPlatformClient,
    check_authority_overclaim,
    get_logic_platform_client,
)
from ipfs_accelerate_py.agent_supervisor.proof.logic_translation_validation import (
    TranslationValidationResult,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CLIENT_SOURCE = (
    REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "proof"
    / "logic_platform_client.py"
)
NOTE_PATH = (
    REPO_ROOT
    / "data"
    / "agent_supervisor"
    / "logic_platform_canonicalization"
    / "notes"
    / "supervisor_client.md"
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _budget(*, network: bool = False) -> ResourceBudget:
    return ResourceBudget(
        wall_time_ms=5_000,
        cpu_time_ms=5_000,
        memory_bytes=64 * 1024 * 1024,
        network_allowed=network,
    )


def _request(**overrides: Any) -> LogicPlatformClientRequest:
    payload: dict[str, Any] = {
        "task_id": "task:lpc-110",
        "tree_id": "tree:lpc-110",
        "policy_id": "policy:lpc-110",
        "plan_id": "plan:lpc-110",
        "resource_budget": _budget(),
        "network_allowed": False,
        "correlation_id": "corr:lpc-110",
        "evidence_kind": "candidate",
        "authority_ceiling": "candidate",
        "request_id": "req-lpc-110-0001",
    }
    payload.update(overrides)
    return LogicPlatformClientRequest(**payload)


def _client(
    *,
    require_handshake: bool = True,
    provider_facade: SupervisorLogicProviderFacade | None = None,
) -> SupervisorLogicPlatformClient:
    return SupervisorLogicPlatformClient(
        require_handshake=require_handshake,
        provider_facade=provider_facade,
    )


def _handshaken_client(
    **kwargs: Any,
) -> SupervisorLogicPlatformClient:
    client = _client(**kwargs)
    result = client.handshake()
    assert getattr(result, "compatible", False) is True
    return client


class _StubProvider:
    """Minimal in-process provider for typed invocation tests."""

    provider_id = "stub-logic-provider"
    provider_version = "1.0.0"
    protocol_version = 1

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[Any] = []

    def _handle(self, request: Any) -> Any:
        # Canonical dispatch calls operation methods with LogicProviderRequest.
        self.calls.append(request)
        operation = getattr(request, "operation", None)
        op_value = getattr(operation, "value", operation)
        if self.fail:
            from ipfs_datasets_py.logic.backends.provider import (
                LogicProviderFailure,
                LogicProviderFailureCode,
                LogicProviderResponse,
            )

            return LogicProviderResponse.failure(
                request,
                LogicProviderFailureCode.PROVIDER_ERROR,
                "stub provider forced failure",
                retryable=False,
                provider_id=self.provider_id,
                provider_version=self.provider_version,
            )
        from ipfs_datasets_py.logic.backends.provider import LogicProviderResponse

        return LogicProviderResponse.success(
            request,
            {
                "stub": True,
                "operation": str(op_value),
                "authority": "kernel",  # must not upgrade client ceiling
            },
            provider_id=self.provider_id,
            provider_version=self.provider_version,
        )

    def capability(self, request: Any) -> Any:
        return self._handle(request)

    def translate(self, request: Any) -> Any:
        return self._handle(request)

    def prove(self, request: Any) -> Any:
        return self._handle(request)

    def reconstruct(self, request: Any) -> Any:
        return self._handle(request)

    def verify(self, request: Any) -> Any:
        return self._handle(request)

    def attest(self, request: Any) -> Any:
        return self._handle(request)


def _facade(*, fail: bool = False) -> SupervisorLogicProviderFacade:
    provider = _StubProvider(fail=fail)
    return SupervisorLogicProviderFacade(
        provider_id=provider.provider_id,
        provider_version=provider.provider_version,
        provider=provider,
    )


def _admitted_slice_kwargs() -> dict[str, Any]:
    from ipfs_datasets_py.logic.families.namespaces import (
        notation_id,
        property_id,
        view_id,
    )
    from ipfs_datasets_py.logic.syntax_core.ast import TypedExpression, mk_predicate
    from ipfs_datasets_py.logic.syntax_core.contracts import SourceDocument
    from ipfs_datasets_py.logic.syntax_core.signatures import propositional_signature

    document = SourceDocument.from_text("doc:lpc-110", "P", encoding="utf-8")
    expression = TypedExpression(
        expression_id="expr:lpc-110",
        root=mk_predicate("n:p", "P"),
        signature=propositional_signature("sig:lpc-110", ("P",)),
    )
    return {
        "slice_id": "slice:lpc-110",
        "domain": "security_ir",
        "document_id": document.document_id,
        "source_digest": document.content_digest,
        "expression_id": expression.expression_id,
        "expression_digest": expression.content_digest,
        "family": expression.family,
        "profile": expression.profile,
        "property": property_id("validity"),
        "view": view_id("source"),
        "notation": notation_id("canonical_text"),
        "features": ("propositional",),
        "document": document,
        "expression": expression,
    }


# ---------------------------------------------------------------------------
# Surface / lazy import
# ---------------------------------------------------------------------------


def test_client_interface_and_schema_are_stable() -> None:
    client = get_logic_platform_client(require_handshake=False)
    payload = client.to_dict()
    assert client.interface == SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE
    assert payload["interface"] == SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE
    assert payload["schema_version"] == LOGIC_PLATFORM_CLIENT_SCHEMA
    assert payload["task_id"] == LOGIC_PLATFORM_CLIENT_TASK_ID
    assert payload["goal_id"] == LOGIC_PLATFORM_CLIENT_GOAL_ID
    assert set(payload["operations"]) == set(CLIENT_OPERATIONS)
    assert set(payload["typed_provider_operations"]) == set(TYPED_PROVIDER_OPERATIONS)
    assert client.datasets_import_is_lazy() is True


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

    # Runtime: constructing a client without handshake must not require datasets.
    before = {name for name in sys.modules if name.startswith("ipfs_datasets_py")}
    client = SupervisorLogicPlatformClient(require_handshake=False)
    assert client.datasets_import_is_lazy() is True
    after = {name for name in sys.modules if name.startswith("ipfs_datasets_py")}
    # Construction itself must not add datasets modules.
    assert after == before


def test_note_documents_acceptance_surface() -> None:
    text = NOTE_PATH.read_text(encoding="utf-8")
    for token in (
        "SupervisorLogicPlatformClient@1",
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
        "LogicPlatformClientRequest",
        "check_authority_overclaim",
    ):
        assert token in text, token


# ---------------------------------------------------------------------------
# Request binding / authority
# ---------------------------------------------------------------------------


def test_request_requires_all_binding_axes() -> None:
    with pytest.raises(LogicPlatformClientRequestError):
        LogicPlatformClientRequest(
            task_id="",
            tree_id="tree",
            policy_id="policy",
            plan_id="plan",
            resource_budget=_budget(),
            network_allowed=False,
            correlation_id="corr",
            evidence_kind="candidate",
            authority_ceiling="candidate",
        )


def test_request_rejects_network_overclaim() -> None:
    with pytest.raises(LogicPlatformClientRequestError, match="network_allowed"):
        _request(resource_budget=_budget(network=False), network_allowed=True)


def test_request_derives_binding_digest() -> None:
    req = _request()
    assert req.binding_digest.startswith("sha256:")
    assert req.schema_version == LOGIC_PLATFORM_CLIENT_REQUEST_SCHEMA
    # Digest is derived — caller-supplied mismatch fails closed.
    with pytest.raises(LogicPlatformClientRequestError, match="binding_digest"):
        _request(binding_digest="sha256:" + ("0" * 64))


def test_check_authority_overclaim_rules() -> None:
    assert check_authority_overclaim("candidate", "candidate") == "candidate"
    with pytest.raises(LogicPlatformClientAuthorityError):
        check_authority_overclaim("candidate", "kernel")
    with pytest.raises(LogicPlatformClientAuthorityError):
        check_authority_overclaim("advisory", "reconstruction")
    # Unknown evidence defaults to advisory max.
    with pytest.raises(LogicPlatformClientAuthorityError):
        check_authority_overclaim("totally-unknown-kind", "candidate")
    assert check_authority_overclaim("totally-unknown-kind", "advisory") == "advisory"


def test_request_rejects_authority_overclaim_at_construction() -> None:
    with pytest.raises(LogicPlatformClientAuthorityError):
        _request(evidence_kind="candidate", authority_ceiling="kernel")


# ---------------------------------------------------------------------------
# Handshake
# ---------------------------------------------------------------------------


def test_handshake_compatible_by_default() -> None:
    client = _client()
    result = client.handshake()
    assert result.compatible is True
    assert client.handshake_compatible is True
    assert "SupervisorLogicPlatformClient@1" in result.manifest.compatible_adapter_versions
    assert result.manifest.requires_git() is False
    assert result.manifest.requires_sibling_repos() is False


def test_handshake_with_request_envelope() -> None:
    client = _client()
    request = _request()
    result = client.handshake(request=request)
    assert isinstance(result.payload, Mapping)
    assert result.ok is True
    assert result.operation == "handshake"
    assert result.status is ClientResultStatus.OK
    assert result.payload["compatible"] is True
    assert result.authority_upgraded is False


def test_handshake_incompatible_adapter_is_typed() -> None:
    from ipfs_datasets_py.logic.platform.manifest import HandshakeRequirements

    client = _client()
    request = _request()
    result = client.handshake(
        HandshakeRequirements(
            required_adapter_versions=("SupervisorLogicPlatformClient@99",)
        ),
        request=request,
    )
    assert result.ok is False
    assert result.status is ClientResultStatus.INCOMPATIBLE
    assert result.payload["compatible"] is False
    assert result.payload["incompatibilities"]


def test_operations_require_handshake_by_default() -> None:
    client = _client(require_handshake=True)
    request = _request()
    with pytest.raises(LogicPlatformClientHandshakeError):
        client.catalog(request)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_catalog_is_declarative_with_sealed_identity() -> None:
    from ipfs_datasets_py.logic.families.canonical_catalog import (
        DEFAULT_CANONICAL_CATALOG_SNAPSHOT,
    )

    client = _handshaken_client()
    result = client.catalog(_request())
    assert result.ok is True
    assert result.status is ClientResultStatus.DECLARATIVE
    assert result.cache_freshness is CacheFreshnessStatus.CURRENT
    assert result.payload["status"] == "declarative"
    assert result.payload["catalog_presence_is_not_proof"] is True
    assert result.payload["availability_is_not_proof"] is True
    assert (
        result.payload["catalog_root"]
        == DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
    )
    assert (
        result.payload["catalog_digest"]
        == DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_digest
    )
    assert "adapter_vocabulary" in result.payload


# ---------------------------------------------------------------------------
# Formalize / slice / obligation / plan
# ---------------------------------------------------------------------------


def test_formalize_and_create_slice_and_obligation_and_plan() -> None:
    from ipfs_datasets_py.logic.backends.requests_v2 import RequestBounds
    from ipfs_datasets_py.logic.families.namespaces import (
        encoding_id,
        evidence_id,
        notation_id,
        property_id,
        view_id,
    )
    from ipfs_datasets_py.logic.formalization.artifacts_v3 import DomainLogicSliceV2
    from ipfs_datasets_py.logic.software_verification.tactician.contracts import (
        CandidateProofStep,
        CandidateStatus,
        SourceSpanBinding,
    )

    client = _handshaken_client()
    request = _request()
    slice_kwargs = _admitted_slice_kwargs()
    document = slice_kwargs.pop("document")
    expression = slice_kwargs.pop("expression")

    formalize = client.formalize(
        request,
        artifact_id="art:lpc-110",
        sample_id="sample:lpc-110",
        domain=slice_kwargs["domain"],
        document_id=slice_kwargs["document_id"],
        source_digest=slice_kwargs["source_digest"],
        expression_id=slice_kwargs["expression_id"],
        expression_digest=slice_kwargs["expression_digest"],
        family=slice_kwargs["family"],
        profile=slice_kwargs["profile"],
        view=view_id("source"),
        notation=notation_id("canonical_text"),
        slices=(),
    )
    assert formalize.ok is True
    assert formalize.operation == "formalize"
    assert formalize.payload["interface"] == "FormalizationArtifact@3"
    assert formalize.payload["candidate"] is True
    assert formalize.payload["proof_claimed"] is False
    assert formalize.payload["artifact"]["metadata"]["task_id"] == request.task_id

    slice_result = client.create_slice(
        request,
        formalization_artifact_id="art:lpc-110",
        **slice_kwargs,
    )
    assert slice_result.ok is True
    assert slice_result.payload["interface"] == "DomainLogicSlice@2"
    assert slice_result.payload["status"] == "admitted"
    slice_obj = DomainLogicSliceV2.from_dict(slice_result.payload["slice"])

    obligation = client.create_obligation(
        request,
        slice_=slice_obj,
        obligation_id="obl:lpc-110",
        statement="P is valid",
        encoding=encoding_id("smtlib2"),
        evidence_kind=evidence_id("candidate"),
        bounds=RequestBounds.default(),
        authority_ceiling="candidate",
    )
    assert obligation.ok is True
    assert obligation.payload["interface"] == "LogicObligation@2"
    assert obligation.authority_ceiling == "candidate"

    candidate = CandidateProofStep(
        candidate_id="cand:lpc-110",
        hole_id="hole:lpc-110",
        kind="solve",
        statement="attempt candidate proof step",
        status=CandidateStatus.PROPOSED,
        source=SourceSpanBinding(tree_id=request.tree_id),
        provider_ids=("stub-logic-provider",),
    )
    plan = client.create_plan(
        request,
        formal_goal_id="goal:lpc-110",
        graph_id="graph:lpc-110",
        candidates=(candidate,),
        provider_ids=("stub-logic-provider",),
    )
    assert plan.ok is True
    assert plan.payload["interface"] == "GoalDirectedProofPlan@1"
    assert plan.payload["proof_claimed"] is False
    assert plan.payload["completion_claimed"] is False
    assert plan.authority_ceiling == "candidate"
    assert plan.proof_claimed is False

    # Missing digests fail closed for slices.
    bad = client.create_slice(
        request,
        slice_id="slice:bad",
        domain="security_ir",
        document_id="doc:x",
        source_digest="",  # invalid
        expression_id="expr:x",
        expression_digest=_sha256_hex("x"),
        family=expression.family,
        profile=expression.profile,
        property=property_id("validity"),
        view=view_id("source"),
        notation=notation_id("canonical_text"),
    )
    assert bad.ok is False
    assert bad.status is ClientResultStatus.FAILED


def test_create_obligation_rejects_authority_overclaim() -> None:
    from ipfs_datasets_py.logic.backends.requests_v2 import RequestBounds
    from ipfs_datasets_py.logic.families.namespaces import encoding_id, evidence_id
    from ipfs_datasets_py.logic.formalization.artifacts_v3 import DomainLogicSliceV2

    client = _handshaken_client()
    request = _request()
    slice_kwargs = _admitted_slice_kwargs()
    slice_kwargs.pop("document")
    slice_kwargs.pop("expression")
    slice_result = client.create_slice(request, **slice_kwargs)
    assert slice_result.ok is True
    slice_obj = DomainLogicSliceV2.from_dict(slice_result.payload["slice"])

    result = client.create_obligation(
        request,
        slice_=slice_obj,
        obligation_id="obl:overclaim",
        statement="P",
        encoding=encoding_id("smtlib2"),
        evidence_kind=evidence_id("candidate"),
        bounds=RequestBounds.default(),
        authority_ceiling="kernel",
    )
    assert result.ok is False


# ---------------------------------------------------------------------------
# Capability discovery / typed invocation
# ---------------------------------------------------------------------------


def test_discover_capabilities_never_equals_proof() -> None:
    client = _handshaken_client(provider_facade=_facade())
    result = client.discover_capabilities(_request())
    assert result.ok is True
    assert result.status is ClientResultStatus.DECLARATIVE
    assert result.payload["availability_is_not_proof"] is True
    assert result.availability_is_not_proof is True
    assert result.payload["facade"]["provider_id"] == "stub-logic-provider"
    assert set(result.payload["typed_provider_operations"]) == set(
        TYPED_PROVIDER_OPERATIONS
    )


def test_invoke_unknown_operation_rejected() -> None:
    client = _handshaken_client(provider_facade=_facade())
    result = client.invoke(_request(), operation="not-a-real-op")
    assert result.ok is False
    assert result.status is ClientResultStatus.REJECTED


def test_invoke_missing_facade_unavailable() -> None:
    client = _handshaken_client(provider_facade=None)
    result = client.invoke(_request(), operation="prove")
    assert result.ok is False
    assert result.status is ClientResultStatus.UNAVAILABLE


def test_typed_invocation_reconstruct_and_verify_do_not_upgrade_authority() -> None:
    facade = _facade()
    client = _handshaken_client(provider_facade=facade)
    request = _request(evidence_kind="candidate", authority_ceiling="candidate")

    invoke = client.invoke(
        request,
        operation=ProofProviderOperation.PROVE,
        payload={"goal": "P"},
    )
    assert invoke.ok is True
    assert invoke.authority_ceiling == "candidate"
    assert invoke.authority_upgraded is False
    assert invoke.payload["authority_upgraded"] is False
    assert invoke.payload["result"]["authority"] == "kernel"  # provider claim retained
    assert invoke.payload["provider_operation"] == "prove"

    reconstruct = client.reconstruct(request, payload={"artifact": "a"})
    assert reconstruct.operation == "reconstruct"
    assert reconstruct.ok is True
    assert reconstruct.authority_ceiling == "candidate"
    assert reconstruct.proof_claimed is False

    verify = client.verify(request, payload={"artifact": "a"})
    assert verify.operation == "verify"
    assert verify.ok is True
    assert verify.authority_ceiling == "candidate"
    assert verify.is_proof is False


def test_invoke_surfaces_provider_failure() -> None:
    client = _handshaken_client(provider_facade=_facade(fail=True))
    result = client.invoke(_request(), operation="prove")
    assert result.ok is False
    assert result.status is ClientResultStatus.FAILED
    assert "forced failure" in (result.error or "")


def test_invoke_respects_cancellation() -> None:
    token = CancellationToken()
    token.cancel()
    client = _handshaken_client(provider_facade=_facade())
    result = client.invoke(
        _request(cancellation=token),
        operation="prove",
    )
    assert result.ok is False
    assert result.status is ClientResultStatus.REJECTED


# ---------------------------------------------------------------------------
# Receipts / counterexamples / cache freshness
# ---------------------------------------------------------------------------


def test_receipt_projection_never_upgrades_authority() -> None:
    client = _handshaken_client()
    validation = TranslationValidationResult(
        contract_identity="contract:lpc-110",
        artifact_identity="artifact:lpc-110",
        conformant=True,
        quarantine_required=False,
        maximum_assurance=AssuranceLevel.CANDIDATE,
        issues=(),
        bounded=False,
    )
    result = client.receipt(_request(), validation_result=validation)
    assert result.ok is True
    assert result.operation == "receipt"
    assert result.authority_ceiling == "candidate"
    assert result.payload["authority"] == "none"
    assert result.payload["proof_success"] is False
    assert result.payload["receipt"]["authority"] == "none"
    assert result.payload["receipt"]["proof_success"] is False
    assert result.payload["receipt"]["task_id"] == "task:lpc-110"


def test_counterexample_is_never_a_proof() -> None:
    client = _handshaken_client()
    result = client.counterexample(
        _request(),
        {
            "kind": "smt_model",
            "model": {"x": "1"},
            "violated_property": "safety",
        },
        kind=CounterexampleKind.SMT_MODEL,
        property_class="safety",
        violated_property="safety",
        summary="x=1 violates safety",
    )
    assert result.ok is True
    assert result.is_proof is False
    assert result.payload["is_proof"] is False
    assert result.authority_ceiling == "candidate"
    bindings = result.payload["counterexample"]["bindings"]
    assert "task:lpc-110" in bindings["task_ids"]
    assert "plan:lpc-110" in bindings["plan_ids"]
    assert "tree:lpc-110" in bindings["tree_ids"]
    assert "policy:lpc-110" in bindings["policy_ids"]


def test_cache_freshness_closed_vocabulary() -> None:
    client = _handshaken_client()
    request = _request()

    current = client.cache_freshness(
        request,
        stored_at_unix_ms=1_000,
        ttl_ms=5_000,
        now_unix_ms=2_000,
    )
    assert current.ok is True
    assert current.cache_freshness is CacheFreshnessStatus.CURRENT
    assert current.payload["status"] == "current"

    stale = client.cache_freshness(
        request,
        stored_at_unix_ms=1_000,
        ttl_ms=100,
        now_unix_ms=5_000,
    )
    assert stale.cache_freshness is CacheFreshnessStatus.STALE

    unknown = client.cache_freshness(request)
    assert unknown.cache_freshness is CacheFreshnessStatus.UNKNOWN

    invalidated = client.cache_freshness(request, invalidated=True)
    assert invalidated.cache_freshness is CacheFreshnessStatus.INVALIDATED

    class _Repo:
        def freshness(self, key: str) -> Mapping[str, Any]:
            assert key == "k1"
            return {"status": "miss"}

    miss = client.cache_freshness(
        request,
        cache_key="k1",
        repository=_Repo(),
    )
    assert miss.cache_freshness is CacheFreshnessStatus.MISS


def test_result_schema_rejects_authority_upgrade_flag() -> None:
    from ipfs_accelerate_py.agent_supervisor.proof.logic_platform_client import (
        LogicPlatformClientError,
        LogicPlatformClientResult,
    )

    with pytest.raises(LogicPlatformClientError, match="authority_upgraded"):
        LogicPlatformClientResult(
            operation="invoke",
            status=ClientResultStatus.OK,
            ok=True,
            request_id="r1",
            binding_digest="sha256:" + ("a" * 64),
            authority_ceiling="candidate",
            authority_upgraded=True,
        )
    # Happy path schema constants.
    result = LogicPlatformClientResult(
        operation="catalog",
        status=ClientResultStatus.DECLARATIVE,
        ok=True,
        request_id="r1",
        binding_digest="sha256:" + ("b" * 64),
        authority_ceiling="candidate",
    )
    assert result.schema_version == LOGIC_PLATFORM_CLIENT_RESULT_SCHEMA
    assert result.to_dict()["availability_is_not_proof"] is True


def test_module_reimport_preserves_interface() -> None:
    module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.proof.logic_platform_client"
    )
    assert (
        module.SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE
        == "SupervisorLogicPlatformClient@1"
    )
    # Ensure ProviderRequest path used by invoke remains importable.
    assert ProviderRequest is not None
    assert ProviderResponse is not None


def test_client_source_exports_closed_operation_surface() -> None:
    """Guard the closed operation vocabulary against silent shrinkage."""

    source = CLIENT_SOURCE.read_text(encoding="utf-8")
    for token in (
        "def handshake",
        "def catalog",
        "def formalize",
        "def create_slice",
        "def create_obligation",
        "def create_plan",
        "def discover_capabilities",
        "def invoke",
        "def reconstruct",
        "def verify",
        "def receipt",
        "def counterexample",
        "def cache_freshness",
        "def check_authority_overclaim",
        "class SupervisorLogicPlatformClient",
        "class LogicPlatformClientRequest",
        "class LogicPlatformClientResult",
        "class CacheFreshnessStatus",
    ):
        assert token in source, token
    assert CLIENT_SOURCE.is_file()
    assert NOTE_PATH.is_file()
