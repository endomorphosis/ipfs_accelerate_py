"""LPC-141 — Direct-versus-supervisor logic platform parity.

Representative operations must agree on request, obligation, provider request,
verdict, evidence, authority, boundedness, and receipt identities whether they
are constructed through the datasets APIs directly or mediated by the
supervisor client / provider facade / receipt admission boundary.

Hermetic: fixture providers only; no live provers, network, or PATH tools.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_capabilities import (
    ProofProviderOperation,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    EvidenceKind,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_provider import (
    ProviderRequest,
)
from ipfs_accelerate_py.agent_supervisor.proof.logic_platform_admission import (
    AdmissionContext,
    admit_receipt,
)
from ipfs_accelerate_py.agent_supervisor.proof.logic_platform_client import (
    ClientOperation,
    ClientRequestContext,
    SupervisorLogicPlatformClient,
)
from ipfs_accelerate_py.agent_supervisor.proof.logic_provider_contract import (
    SupervisorLogicProviderFacade,
    to_logic_provider_request,
)
from ipfs_datasets_py.logic.backends.artifacts_v2 import (
    admit_compiled_target,
    admit_parsed_result,
)
from ipfs_datasets_py.logic.backends.evidence_v2 import (
    ExecutionOutcome,
    ExecutionRecordKind,
    ProviderExecutionReceiptV2,
)
from ipfs_datasets_py.logic.backends.provider import (
    LogicProviderRequest,
    ProviderResourceBudget,
    dispatch_logic_provider_request,
)
from ipfs_datasets_py.logic.backends.requests_v2 import (
    BackendRequestV2,
    LogicObligationV2,
    RequestAuthorityCeiling,
    RequestBounds,
)
from ipfs_datasets_py.logic.backends.response_v2 import (
    DEFAULT_BOUNDEDNESS,
    DEFAULT_EVIDENCE_AUTHORITY,
    DEFAULT_EVIDENCE_KIND,
    DEFAULT_SEMANTIC_VERDICT,
    CacheProvenanceV2,
    ProviderResponseV2,
    ResponseArtifactRef,
    ResponseSourceRef,
)
from ipfs_datasets_py.logic.backends.protocol_v2 import ProtocolOperationV2
from ipfs_datasets_py.logic.families.namespaces import (
    encoding_id,
    evidence_id,
    notation_id,
    property_id,
    provider_id,
    view_id,
)
from ipfs_datasets_py.logic.formalization.artifacts_v3 import DomainLogicSliceV2
from ipfs_datasets_py.logic.ir_core.axes import (
    LogicBoundedness,
    LogicEvidenceAuthority,
    LogicEvidenceKind,
    LogicOperationStatus,
    LogicSemanticVerdict,
)
from ipfs_datasets_py.logic.ir_core.protocols import ResourceUsage
from ipfs_datasets_py.logic.syntax_core.ast import TypedExpression, mk_predicate
from ipfs_datasets_py.logic.syntax_core.contracts import (
    SourceDocument,
    SourceMap,
    SourceMapEntry,
    SourceRange,
)
from ipfs_datasets_py.logic.syntax_core.signatures import propositional_signature


REPO_ROOT = Path(__file__).resolve().parents[2]
PARITY_NOTE = (
    REPO_ROOT
    / "data"
    / "agent_supervisor"
    / "logic_platform_canonicalization"
    / "notes"
    / "direct_supervisor_parity.md"
)

# Eight acceptance identity dimensions (LPC-141).
PARITY_DIMENSIONS: tuple[str, ...] = (
    "request",
    "obligation",
    "provider_request",
    "verdict",
    "evidence",
    "authority",
    "boundedness",
    "receipt",
)

REPRESENTATIVE_OPERATIONS: tuple[str, ...] = (
    "prove",
    "verify",
    "reconstruct",
    "translate",
    "capability",
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _canonical_identity(payload: Mapping[str, Any]) -> str:
    """Stable identity over a JSON-compatible mapping (sorted keys)."""

    body = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ParityIdentityBundle:
    """Semantic identities that must agree across direct and supervisor paths."""

    request: str
    obligation: str
    provider_request: str
    verdict: str
    evidence: str
    authority: str
    boundedness: str
    receipt: str

    def as_dict(self) -> dict[str, str]:
        return {
            "request": self.request,
            "obligation": self.obligation,
            "provider_request": self.provider_request,
            "verdict": self.verdict,
            "evidence": self.evidence,
            "authority": self.authority,
            "boundedness": self.boundedness,
            "receipt": self.receipt,
        }

    def assert_agrees(self, other: "ParityIdentityBundle") -> None:
        left = self.as_dict()
        right = other.as_dict()
        for dimension in PARITY_DIMENSIONS:
            assert left[dimension] == right[dimension], (
                f"parity mismatch on {dimension}: "
                f"direct={left[dimension]!r} supervisor={right[dimension]!r}"
            )


@dataclass(frozen=True, slots=True)
class RepresentativeCase:
    """Compact recipe for one representative operation slice."""

    operation: str
    case_id: str
    statement: str
    domain: str
    document_text: str
    evidence_kind_token: str
    authority_ceiling: RequestAuthorityCeiling
    semantic_verdict: LogicSemanticVerdict
    evidence_authority: LogicEvidenceAuthority
    boundedness: LogicBoundedness
    logic_evidence_kind: LogicEvidenceKind


def _representative_cases() -> tuple[RepresentativeCase, ...]:
    """Compact generator of representative operations (no bulk golden dumps)."""

    return (
        RepresentativeCase(
            operation="prove",
            case_id="parity-prove",
            statement="prove P",
            domain="security_ir",
            document_text="P",
            evidence_kind_token="model",
            authority_ceiling=RequestAuthorityCeiling.SATISFIABILITY,
            semantic_verdict=LogicSemanticVerdict.UNKNOWN,
            evidence_authority=LogicEvidenceAuthority.ADVISORY,
            boundedness=LogicBoundedness.UNKNOWN,
            logic_evidence_kind=LogicEvidenceKind.CANDIDATE,
        ),
        RepresentativeCase(
            operation="verify",
            case_id="parity-verify",
            statement="verify reconstruction of P",
            domain="legal_ir",
            document_text="P /\\ Q",
            evidence_kind_token="proof",
            authority_ceiling=RequestAuthorityCeiling.RECONSTRUCTION,
            semantic_verdict=LogicSemanticVerdict.INCONCLUSIVE,
            evidence_authority=LogicEvidenceAuthority.ADVISORY,
            boundedness=LogicBoundedness.RESOURCE_BOUNDED,
            logic_evidence_kind=LogicEvidenceKind.CHECKED_PROOF,
        ),
        RepresentativeCase(
            operation="reconstruct",
            case_id="parity-reconstruct",
            statement="reconstruct candidate proof of P",
            domain="software_verification",
            document_text="invariant(P)",
            evidence_kind_token="proof",
            authority_ceiling=RequestAuthorityCeiling.RECONSTRUCTION,
            semantic_verdict=LogicSemanticVerdict.UNKNOWN,
            evidence_authority=LogicEvidenceAuthority.ADVISORY,
            boundedness=LogicBoundedness.FINITE_TRACE,
            logic_evidence_kind=LogicEvidenceKind.CHECKED_PROOF,
        ),
        RepresentativeCase(
            operation="translate",
            case_id="parity-translate",
            statement="translate P to smtlib2",
            domain="crypto_ir",
            document_text="encrypt(P)",
            evidence_kind_token="model",
            authority_ceiling=RequestAuthorityCeiling.BOUNDED,
            semantic_verdict=LogicSemanticVerdict.NOT_APPLICABLE,
            evidence_authority=LogicEvidenceAuthority.ADVISORY,
            boundedness=LogicBoundedness.UNKNOWN,
            logic_evidence_kind=LogicEvidenceKind.CANDIDATE,
        ),
        RepresentativeCase(
            operation="capability",
            case_id="parity-capability",
            statement="discover capability for P",
            domain="intent_ir",
            document_text="capability(P)",
            evidence_kind_token="advisory",
            authority_ceiling=RequestAuthorityCeiling.ADVISORY,
            semantic_verdict=LogicSemanticVerdict.NOT_APPLICABLE,
            evidence_authority=LogicEvidenceAuthority.ADVISORY,
            boundedness=LogicBoundedness.UNKNOWN,
            logic_evidence_kind=LogicEvidenceKind.CANDIDATE,
        ),
    )


def _document(case: RepresentativeCase) -> SourceDocument:
    return SourceDocument.from_text(
        f"doc:{case.case_id}", case.document_text, encoding="utf-8"
    )


def _expression(case: RepresentativeCase) -> TypedExpression:
    return TypedExpression(
        expression_id=f"expr:{case.case_id}",
        root=mk_predicate(f"n:{case.case_id}", "P"),
        signature=propositional_signature(f"sig:{case.case_id}", ("P",)),
    )


def _admitted_slice(case: RepresentativeCase) -> DomainLogicSliceV2:
    document = _document(case)
    expression = _expression(case)
    return DomainLogicSliceV2(
        slice_id=f"slice:{case.case_id}",
        domain=case.domain,
        document_id=document.document_id,
        source_digest=document.content_digest,
        expression_id=expression.expression_id,
        expression_digest=expression.content_digest,
        family=expression.family,
        profile=expression.profile,
        property=property_id("validity"),
        view=view_id("source"),
        notation=notation_id("canonical_text"),
        features=("propositional",),
    )


def _bounds() -> RequestBounds:
    return RequestBounds(
        timeout_ms=5_000,
        max_steps=10_000,
        max_memory_bytes=32 * 1024 * 1024,
        max_output_bytes=64 * 1024,
    )


def _obligation(case: RepresentativeCase) -> LogicObligationV2:
    return LogicObligationV2.from_slice(
        _admitted_slice(case),
        obligation_id=f"obl:{case.case_id}",
        statement=case.statement,
        encoding=encoding_id("smtlib2"),
        evidence_kind=evidence_id(case.evidence_kind_token),
        bounds=_bounds(),
        authority_ceiling=case.authority_ceiling,
    )


def _backend_request(case: RepresentativeCase) -> BackendRequestV2:
    return BackendRequestV2.from_obligation(
        _obligation(case),
        request_id=f"req:{case.case_id}",
        requested_provider=provider_id("z3"),
    )


def _provider_payload(case: RepresentativeCase, request: BackendRequestV2) -> dict[str, Any]:
    return {
        "obligation_id": request.obligation_id,
        "obligation_digest": request.obligation_digest,
        "request_digest": request.content_digest,
        "slice_id": request.slice_id,
        "slice_digest": request.slice_digest,
        "statement": case.statement,
        "operation": case.operation,
        "family": request.family.qualified
        if hasattr(request.family, "qualified")
        else str(request.family),
        "evidence_kind": case.evidence_kind_token,
        "authority_ceiling": case.authority_ceiling.value,
        "boundedness": case.boundedness.value,
        "semantic_verdict": case.semantic_verdict.value,
    }


def _resource_budget() -> ResourceBudget:
    return ResourceBudget(
        wall_time_ms=5_000,
        cpu_time_ms=2_000,
        memory_bytes=32 * 1024 * 1024,
        disk_bytes=1_024,
        max_processes=1,
        max_premises=8,
        max_output_bytes=64 * 1024,
        model_token_limit=0,
        provider_quota=1,
        network_allowed=False,
    )


def _direct_provider_request(
    case: RepresentativeCase, request: BackendRequestV2
) -> LogicProviderRequest:
    budget = _resource_budget()
    return LogicProviderRequest(
        operation=case.operation,
        request_id=f"preq:{case.case_id}",
        payload=_provider_payload(case, request),
        resource_budget=ProviderResourceBudget(
            wall_time_ms=budget.wall_time_ms,
            cpu_time_ms=budget.cpu_time_ms,
            memory_bytes=budget.memory_bytes,
            disk_bytes=budget.disk_bytes,
            max_processes=budget.max_processes,
            max_premises=budget.max_premises,
            max_output_bytes=budget.max_output_bytes,
            model_token_limit=budget.model_token_limit,
            provider_quota=budget.provider_quota,
            network_allowed=budget.network_allowed,
        ),
        network_allowed=False,
        deadline_unix_ms=4_102_444_800_000,
    )


def _supervisor_provider_request(
    case: RepresentativeCase, request: BackendRequestV2
) -> ProviderRequest:
    return ProviderRequest(
        operation=ProofProviderOperation(case.operation),
        request_id=f"preq:{case.case_id}",
        payload=_provider_payload(case, request),
        resource_budget=_resource_budget(),
        network_allowed=False,
        deadline_unix_ms=4_102_444_800_000,
    )


def _provider_request_identity(request: LogicProviderRequest) -> str:
    budget = request.resource_budget.to_dict()
    budget.pop("schema_version", None)
    return _canonical_identity(
        {
            "deadline_unix_ms": request.deadline_unix_ms,
            "network_allowed": request.network_allowed,
            "operation": request.operation.value,
            "payload": dict(request.payload),
            "request_id": request.request_id,
            "resource_budget": budget,
        }
    )


def _source_map(case: RepresentativeCase) -> SourceMap:
    document = _document(case)
    end = max(1, len(case.document_text))
    return SourceMap(
        map_id=f"map:{case.case_id}",
        document_id=document.document_id,
        entries=(
            SourceMapEntry(
                entry_id=f"map:entry:{case.case_id}",
                range=SourceRange(start=0, end=end),
                role="atom",
            ),
        ),
    )


def _execution_receipt(
    case: RepresentativeCase, request: BackendRequestV2
) -> ProviderExecutionReceiptV2:
    compiled = admit_compiled_target(
        request,
        artifact_id=f"compiled:{case.case_id}",
        compiler_id="smtlib2.emit",
        target_text=f"(assert {case.document_text})",
        source_map=_source_map(case),
    )
    parsed = admit_parsed_result(
        compiled,
        artifact_id=f"parsed:{case.case_id}",
        provider=provider_id("z3"),
        result_kind="satisfiability.model",
        output_text="sat\n((P true))",
        decoded_evidence_digest=_digest(f"decoded:{case.case_id}"),
    )
    return ProviderExecutionReceiptV2.from_parsed_target(
        parsed,
        receipt_id=f"exec:{case.case_id}",
        launch_id=f"launch:z3:{case.case_id}",
        tool_id="tool:z3:parity",
        bounds=request.bounds,
        record_kind=ExecutionRecordKind.HERMETIC_FIXTURE,
        execution_claimed=True,
        outcome=ExecutionOutcome.SUCCEEDED,
        exit_code=0,
        duration_ms=7,
        toolchain_id="toolchain:z3-parity",
    )


def _typed_response(
    case: RepresentativeCase, request: BackendRequestV2
) -> ProviderResponseV2:
    return ProviderResponseV2(
        request_id=request.request_id,
        operation=ProtocolOperationV2(case.operation)
        if case.operation in {op.value for op in ProtocolOperationV2}
        else ProtocolOperationV2.PROVE,
        provider_id="provider.z3.parity",
        provider_version="1.0.0",
        operation_status=LogicOperationStatus.SUCCEEDED,
        verdict=case.semantic_verdict,
        evidence_kind=case.logic_evidence_kind,
        evidence_authority=case.evidence_authority,
        boundedness=case.boundedness,
        assumptions=(f"asm:{case.case_id}",),
        sources=(
            ResponseSourceRef(
                document_id=request.document_id,
                source_digest=request.source_digest,
            ),
        ),
        artifacts=(
            ResponseArtifactRef(
                artifact_id=f"artifact:{case.case_id}",
                content_digest=_digest(f"artifact:{case.case_id}"),
                kind="witness",
            ),
        ),
        resources=ResourceUsage(
            elapsed_ms=7,
            steps=12,
            peak_memory_bytes=4096,
            output_bytes=128,
        ),
        cache_provenance=CacheProvenanceV2.miss(reason="parity-cold"),
        error=None,
        duration_ms=7,
    )


def _direct_identities(case: RepresentativeCase) -> ParityIdentityBundle:
    obligation = _obligation(case)
    request = BackendRequestV2.from_obligation(
        obligation,
        request_id=f"req:{case.case_id}",
        requested_provider=provider_id("z3"),
    )
    # Round-trip through dict as a wire/serialization boundary would.
    obligation_rt = LogicObligationV2.from_dict(obligation.to_dict())
    request_rt = BackendRequestV2.from_dict(request.to_dict())
    assert obligation_rt.content_digest == obligation.content_digest
    assert request_rt.content_digest == request.content_digest

    provider_request = _direct_provider_request(case, request)
    response = _typed_response(case, request)
    receipt = _execution_receipt(case, request)
    receipt.validate_against(request=request)

    return ParityIdentityBundle(
        request=request.content_digest,
        obligation=obligation.content_digest,
        provider_request=_provider_request_identity(provider_request),
        verdict=response.verdict.value,
        evidence=(
            f"{response.evidence_kind.value}|{response.sources[0].source_digest}"
        ),
        authority=response.evidence_authority.value,
        boundedness=response.boundedness.value,
        receipt=receipt.content_digest,
    )


class _ParityFixtureProvider:
    """In-process provider that echoes parity axes without claiming proof."""

    provider_id = "fixture.parity"
    provider_version = "1.0.0"
    protocol_version = 1

    def __init__(self, case: RepresentativeCase) -> None:
        self.case = case
        self.requests: list[LogicProviderRequest] = []

    def _invoke(self, request: LogicProviderRequest) -> dict[str, object]:
        self.requests.append(request)
        return {
            "echo": dict(request.payload),
            "operation": request.operation.value,
            "operation_status": "succeeded",
            "semantic_verdict": self.case.semantic_verdict.value,
            "evidence_kind": self.case.logic_evidence_kind.value,
            "evidence_authority": self.case.evidence_authority.value,
            "boundedness": self.case.boundedness.value,
            "authority_ceiling": AssuranceLevel.SOLVER_CHECKED.value,
            "provider_claimed_authority": "authoritative",
            "proof_success": False,
            "simulated": False,
        }

    capability = _invoke
    translate = _invoke
    prove = _invoke
    reconstruct = _invoke
    verify = _invoke
    attest = _invoke


def _client_context(case: RepresentativeCase) -> ClientRequestContext:
    return ClientRequestContext(
        task_id="LPC-141",
        repository_tree_id="tree:sha256:parity",
        policy_id="policy:implementation-daemon",
        policy_revision="sha256:" + ("22" * 32),
        resource_budget=_resource_budget(),
        network_allowed=False,
        deadline_unix_ms=int(time.time() * 1000) + 60_000,
        correlation_id=f"corr:{case.case_id}",
        authority_ceiling=AssuranceLevel.SOLVER_CHECKED.value,
        evidence_kind=EvidenceKind.SOLVER_RESULT.value,
        plan_id=f"plan:{case.case_id}",
    )


def _supervisor_identities(case: RepresentativeCase) -> ParityIdentityBundle:
    # Datasets still owns obligation/request minting; supervisor must not redefine.
    obligation = _obligation(case)
    request = BackendRequestV2.from_obligation(
        obligation,
        request_id=f"req:{case.case_id}",
        requested_provider=provider_id("z3"),
    )
    # Supervisor-mediated wire conversion of the provider request.
    supervisor_request = _supervisor_provider_request(case, request)
    converted = to_logic_provider_request(supervisor_request)
    assert converted.request_id == supervisor_request.request_id
    assert converted.operation.value == supervisor_request.operation.value
    assert converted.payload == supervisor_request.payload

    # Facade + client path for response axes.
    provider = _ParityFixtureProvider(case)
    facade = SupervisorLogicProviderFacade(
        provider_id=provider.provider_id,
        provider_version=provider.provider_version,
        provider=provider,
    )
    client = SupervisorLogicPlatformClient(
        provider_facade=facade,
        require_handshake=True,
    )
    handshake = client.handshake()
    assert handshake.ok, handshake.to_dict()
    ctx = _client_context(case)
    result = client.invoke(
        ctx,
        case.operation,
        {
            **_provider_payload(case, request),
            "semantic_verdict": case.semantic_verdict.value,
            "evidence_kind": case.logic_evidence_kind.value,
            "evidence_authority": case.evidence_authority.value,
            "boundedness": case.boundedness.value,
        },
        request_id=f"preq:{case.case_id}",
    )
    assert result.ok, result.to_dict()
    assert result.semantic_verdict == case.semantic_verdict.value
    assert result.payload is not None
    assert result.payload.get("boundedness") == case.boundedness.value
    assert result.payload.get("evidence_kind") == case.logic_evidence_kind.value
    assert result.payload.get("evidence_authority") == case.evidence_authority.value
    # Provider-claimed authority must not promote above context ceiling.
    assert "provider_claimed_authority" not in result.payload
    assert result.authority_ceiling == AssuranceLevel.SOLVER_CHECKED.value

    # Obligation projection through client remains structural (not a second authority).
    obligation_result = client.obligation(
        ctx,
        {
            "obligation_id": obligation.obligation_id,
            "statement": case.statement,
            "obligation_digest": obligation.content_digest,
        },
    )
    assert obligation_result.ok
    assert obligation_result.payload is not None
    assert obligation_result.payload["artifact_id"] == obligation.obligation_id
    assert obligation_result.semantic_verdict == "unknown"

    # Receipt identity: datasets receipt digest + supervisor projection + admission.
    receipt = _execution_receipt(case, request)
    receipt_envelope = {
        "content_id": receipt.content_digest
        if receipt.content_digest.startswith("sha256:")
        else f"sha256:{receipt.content_digest}",
        "receipt_id": receipt.content_digest
        if receipt.content_digest.startswith("sha256:")
        else f"sha256:{receipt.content_digest}",
        "obligation_id": obligation.obligation_id,
        "plan_id": f"plan:{case.case_id}",
        "repository_id": "repository:sha256:parity",
        "repository_tree_id": "tree:sha256:parity",
        "policy_id": "policy:implementation-daemon",
        "policy_revision": "sha256:" + ("22" * 32),
        "environment_id": "env:validation-hermetic",
        "source_id": f"source:sha256:{request.source_digest}"
        if not request.source_digest.startswith("sha256:")
        else f"source:{request.source_digest}",
        "operation": case.operation,
        "semantic_verdict": case.semantic_verdict.value,
        "evidence_kind": EvidenceKind.SOLVER_RESULT.value,
        "authority_ceiling": AssuranceLevel.SOLVER_CHECKED.value,
        "freshness": "current",
        "simulated": False,
        "reconstruction_passed": case.operation in {"verify", "reconstruct"},
        "kernel_checked": False,
        "translation": {
            "valid": True,
            "translation_class": "exact",
            "source_id": "source:ast",
            "target_id": "target:smtlib2",
        },
        "evidence": [
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/proof-evidence@1",
                "kind": EvidenceKind.SOLVER_RESULT.value,
                "authority": "solver",
                "verdict": "accepted",
                "artifact_id": f"artifact:{case.case_id}",
                "subject_id": obligation.obligation_id,
                "verifier_id": "solver:z3",
                "freshness": "current",
                "independent": True,
                "simulated": False,
                "metadata": {},
            }
        ],
        "policy_admitted": True,
        "network_allowed": False,
        "execution_receipt_digest": receipt.content_digest,
    }
    projected = client.receipts(ctx, {"receipts": [receipt_envelope]})
    assert projected.ok
    assert projected.payload is not None
    assert projected.payload["admitted"] is False
    projected_item = projected.payload["receipts"][0]
    assert projected_item["admitted"] is False
    assert projected_item["trusted"] is False
    assert projected_item["receipt"]["execution_receipt_digest"] == (
        receipt.content_digest
    )

    admission = admit_receipt(
        receipt_envelope,
        AdmissionContext(
            task_id="LPC-141",
            repository_tree_id="tree:sha256:parity",
            policy_id="policy:implementation-daemon",
            operation=case.operation,
            required_authority=AssuranceLevel.SOLVER_CHECKED.value,
            repository_id="repository:sha256:parity",
            environment_id="env:validation-hermetic",
            source_id=receipt_envelope["source_id"],
            policy_revision="sha256:" + ("22" * 32),
            plan_id=f"plan:{case.case_id}",
            obligation_id=obligation.obligation_id,
            require_reconstruction=case.operation in {"verify", "reconstruct"},
            require_kernel=False,
            network_allowed=False,
        ),
    )
    # Solver-checked floor may or may not admit depending on ten-point details;
    # identity parity only requires both paths share the same receipt digest.
    assert admission.checks  # admission ran
    assert projected_item["receipt"]["receipt_id"] == receipt_envelope["receipt_id"]

    return ParityIdentityBundle(
        request=request.content_digest,
        obligation=obligation.content_digest,
        provider_request=_provider_request_identity(converted),
        verdict=result.semantic_verdict,
        evidence=(
            f"{result.payload['evidence_kind']}|{request.source_digest}"
        ),
        authority=result.payload["evidence_authority"],
        boundedness=result.payload["boundedness"],
        receipt=receipt.content_digest,
    )


# ---------------------------------------------------------------------------
# Note / dimension inventory
# ---------------------------------------------------------------------------


def test_declared_parity_note_documents_acceptance_surface() -> None:
    assert PARITY_NOTE.is_file(), f"missing declared output note: {PARITY_NOTE}"
    text = PARITY_NOTE.read_text(encoding="utf-8")
    assert "LPC-141" in text
    assert "Direct-versus-supervisor" in text or "direct-versus-supervisor" in text
    for dimension in PARITY_DIMENSIONS:
        assert dimension in text.lower()
    assert "test_direct_vs_supervisor_logic_parity.py" in text
    assert "SupervisorLogicPlatformClient@1" in text
    for operation in REPRESENTATIVE_OPERATIONS:
        assert operation in text


def test_parity_dimension_inventory_is_closed() -> None:
    assert PARITY_DIMENSIONS == (
        "request",
        "obligation",
        "provider_request",
        "verdict",
        "evidence",
        "authority",
        "boundedness",
        "receipt",
    )


# ---------------------------------------------------------------------------
# Representative operation parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _representative_cases(), ids=lambda c: c.operation)
def test_direct_and_supervisor_agree_on_all_parity_identities(
    case: RepresentativeCase,
) -> None:
    direct = _direct_identities(case)
    supervisor = _supervisor_identities(case)
    direct.assert_agrees(supervisor)

    # Non-empty digests / closed axis tokens.
    assert len(direct.request) == 64
    assert len(direct.obligation) == 64
    assert direct.provider_request.startswith("sha256:")
    assert direct.verdict == case.semantic_verdict.value
    assert direct.authority == case.evidence_authority.value
    assert direct.boundedness == case.boundedness.value
    assert len(direct.receipt) == 64


def test_provider_request_conversion_is_lossless_for_representative_prove() -> None:
    case = _representative_cases()[0]
    request = _backend_request(case)
    direct = _direct_provider_request(case, request)
    supervisor = _supervisor_provider_request(case, request)
    converted = to_logic_provider_request(supervisor)

    assert _provider_request_identity(direct) == _provider_request_identity(converted)
    assert direct.request_id == converted.request_id
    assert direct.operation.value == converted.operation.value
    assert direct.payload == converted.payload
    assert direct.network_allowed is converted.network_allowed
    assert direct.deadline_unix_ms == converted.deadline_unix_ms

    direct_budget = direct.resource_budget.to_dict()
    converted_budget = converted.resource_budget.to_dict()
    direct_budget.pop("schema_version", None)
    converted_budget.pop("schema_version", None)
    assert direct_budget == converted_budget


def test_obligation_and_request_identities_stable_across_construction_paths() -> None:
    case = _representative_cases()[0]
    slice_item = _admitted_slice(case)
    from_slice_obligation = LogicObligationV2.from_slice(
        slice_item,
        obligation_id=f"obl:{case.case_id}",
        statement=case.statement,
        encoding=encoding_id("smtlib2"),
        evidence_kind=evidence_id(case.evidence_kind_token),
        bounds=_bounds(),
        authority_ceiling=case.authority_ceiling,
    )
    from_slice_request = BackendRequestV2.from_slice(
        slice_item,
        request_id=f"req:{case.case_id}",
        obligation_id=f"obl:{case.case_id}",
        statement=case.statement,
        encoding=encoding_id("smtlib2"),
        evidence_kind=evidence_id(case.evidence_kind_token),
        bounds=_bounds(),
        authority_ceiling=case.authority_ceiling,
        requested_provider=provider_id("z3"),
    )
    from_obligation_request = BackendRequestV2.from_obligation(
        from_slice_obligation,
        request_id=f"req:{case.case_id}",
        requested_provider=provider_id("z3"),
    )
    assert from_slice_request.obligation_digest == from_slice_obligation.content_digest
    assert from_obligation_request.content_digest == from_slice_request.content_digest
    assert from_obligation_request.obligation_id == from_slice_obligation.obligation_id


def test_response_axes_default_untrusted_and_do_not_imply_proof() -> None:
    case = _representative_cases()[0]
    request = _backend_request(case)
    response = ProviderResponseV2(
        request_id=request.request_id,
        operation=ProtocolOperationV2.PROVE,
        provider_id="provider.z3.parity",
        provider_version="1.0.0",
        operation_status=LogicOperationStatus.SUCCEEDED,
    )
    assert response.verdict is DEFAULT_SEMANTIC_VERDICT
    assert response.evidence_kind is DEFAULT_EVIDENCE_KIND
    assert response.evidence_authority is DEFAULT_EVIDENCE_AUTHORITY
    assert response.boundedness is DEFAULT_BOUNDEDNESS
    assert response.is_success is True
    assert response.is_trusted is False
    assert response.default_authority_applied is True

    # Supervisor path: success still never means proved.
    provider = _ParityFixtureProvider(
        RepresentativeCase(
            operation="prove",
            case_id="parity-success-not-proof",
            statement="prove P",
            domain="security_ir",
            document_text="P",
            evidence_kind_token="model",
            authority_ceiling=RequestAuthorityCeiling.SATISFIABILITY,
            semantic_verdict=LogicSemanticVerdict.UNKNOWN,
            evidence_authority=LogicEvidenceAuthority.ADVISORY,
            boundedness=LogicBoundedness.UNKNOWN,
            logic_evidence_kind=LogicEvidenceKind.CANDIDATE,
        )
    )
    client = SupervisorLogicPlatformClient(
        provider_facade=SupervisorLogicProviderFacade(
            provider_id=provider.provider_id,
            provider_version=provider.provider_version,
            provider=provider,
        ),
        require_handshake=True,
    )
    assert client.handshake().ok
    result = client.prove(
        _client_context(case),
        {"obligation_id": f"obl:{case.case_id}", "semantic_verdict": "unknown"},
        request_id=f"preq:{case.case_id}-snp",
    )
    assert result.ok
    assert result.operation_status == "succeeded"
    assert result.semantic_verdict == "unknown"
    assert result.payload is not None
    assert result.payload.get("proof_success") is False


def test_direct_dispatch_and_supervisor_facade_share_provider_request_identity() -> None:
    case = next(c for c in _representative_cases() if c.operation == "prove")
    request = _backend_request(case)
    provider = _ParityFixtureProvider(case)

    direct_request = _direct_provider_request(case, request)
    direct_response = dispatch_logic_provider_request(provider, direct_request)
    assert direct_response.ok
    assert direct_response.request_id == direct_request.request_id

    facade = SupervisorLogicProviderFacade(
        provider_id=provider.provider_id,
        provider_version=provider.provider_version,
        provider=provider,
    )
    supervisor_request = _supervisor_provider_request(case, request)
    facade_response = facade.invoke(supervisor_request)
    assert facade_response.ok
    assert facade_response.request_id == supervisor_request.request_id

    # Both paths delivered the same payload identity to the provider.
    assert len(provider.requests) == 2
    first, second = provider.requests
    assert first.payload == second.payload
    assert first.operation.value == second.operation.value
    assert first.request_id == second.request_id


def test_receipt_projection_never_auto_admits_and_preserves_identity() -> None:
    case = _representative_cases()[0]
    request = _backend_request(case)
    receipt = _execution_receipt(case, request)
    client = SupervisorLogicPlatformClient(require_handshake=True)
    assert client.handshake().ok
    ctx = _client_context(case)
    content_id = (
        receipt.content_digest
        if receipt.content_digest.startswith("sha256:")
        else f"sha256:{receipt.content_digest}"
    )
    envelope = {
        "content_id": content_id,
        "receipt_id": content_id,
        "obligation_id": request.obligation_id,
        "execution_receipt_digest": receipt.content_digest,
        "semantic_verdict": "unknown",
        "authority_ceiling": AssuranceLevel.KERNEL_VERIFIED.value,
        "simulated": True,
    }
    projected = client.receipts(ctx, {"receipts": [envelope]})
    assert projected.ok
    assert projected.payload is not None
    item = projected.payload["receipts"][0]
    assert item["admitted"] is False
    assert item["trusted"] is False
    assert item["simulated"] is True
    # Simulated kernel claim is reduced.
    assert item["authority_ceiling"] == AssuranceLevel.CANDIDATE.value
    assert item["receipt"]["execution_receipt_digest"] == receipt.content_digest
    assert item["receipt"]["receipt_id"] == content_id


def test_all_representative_operations_covered_by_generator() -> None:
    ops = {case.operation for case in _representative_cases()}
    assert ops == set(REPRESENTATIVE_OPERATIONS)


def test_client_operation_vocabulary_covers_representative_set() -> None:
    for operation in REPRESENTATIVE_OPERATIONS:
        assert operation in {item.value for item in ClientOperation}
