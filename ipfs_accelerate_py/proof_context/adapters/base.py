"""Provider-neutral CodingAgentAdapter contract (PCCE-030).

Concrete adapters consume an admitted TaskSpecification, ContextPack, and
ModelRouteDecision and return a schema-valid PatchProposal bound to a
CodingAgentInvocation. This module never invokes a provider, never approves a
patch, and never exposes canonical-branch authority. Importing it performs no
I/O, network, process, or filesystem mutation.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Protocol, runtime_checkable

from ipfs_accelerate_py.proof_context.adapters.models import (
    CODING_AGENT_INVOCATION_SCHEMA,
    CODING_AGENT_INVOCATION_SCHEMA_DIGEST,
    CONTEXT_PACK_SCHEMA,
    CONTEXT_PACK_SCHEMA_DIGEST,
    CONTRACT_SCHEMA_PREFIX,
    CONTRACT_VERSION,
    INVOCATION_USAGE_FIELDS,
    MAX_FILE_BYTES,
    MAX_LOG_BYTES,
    MAX_PATCH_BYTES,
    MAX_PROVIDER_OUTPUT_BYTES,
    MODEL_ROUTE_DECISION_SCHEMA,
    MODEL_ROUTE_DECISION_SCHEMA_DIGEST,
    PCCE_006_CONTENT_ID,
    PATCH_PROPOSAL_SCHEMA,
    PATCH_PROPOSAL_SCHEMA_DIGEST,
    PROVENANCES,
    SCHEMA,
    TASK_SPECIFICATION_SCHEMA,
    TASK_SPECIFICATION_SCHEMA_DIGEST,
    WIRE_SCHEMAS,
    CodingAgentInvocation,
    ContextPack,
    ModelRouteDecision,
    PatchProposal,
    TaskSpecification,
    admit_bounded_log,
    admit_bounded_patch,
    assert_declared_scope,
    wire_canonical_utf8,
)
from ipfs_accelerate_py.proof_context.compatibility import FROZEN_MATRIX
from ipfs_accelerate_py.proof_context.errors import (
    BoundaryViolationError,
    IdentityInconsistentError,
    MalformedError,
    ProofCancelledError,
    SimulatedPromotedError,
)

INTERFACE: Final[str] = "CodingAgentAdapter@0.1"
ADAPTER_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/coding-agent-adapter"
ADAPTER_RESULT_SCHEMA: Final[str] = (
    "ipfs-accelerate.proof-context.v0.1/coding-agent-adapter-result"
)
SIBLING_LAYOUT_REQUIRED: Final[bool] = False
PROVIDER_BOUND: Final[bool] = False
APPROVAL_AUTHORITY: Final[bool] = False
CANONICAL_BRANCH_AUTHORITY: Final[bool] = False
COMPATIBILITY_MATRIX_CONTENT_ID: Final[str] = FROZEN_MATRIX["content_id"]

PROTOCOL_METHODS: Final[tuple[str, ...]] = ("propose", "cancel")


class CancellationToken:
    """Cooperative cancellation flag. Cancelled invocations cannot claim live."""

    __slots__ = ("_cancelled",)

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def check(self) -> None:
        if self._cancelled:
            raise ProofCancelledError("adapter invocation cancelled")


def _truthy(value: Any) -> bool:
    return value is True or value == "true" or value == 1


def _same_identity(left: str | None, right: str | None, *, field: str) -> None:
    if left is None or right is None:
        return
    if left != right:
        raise IdentityInconsistentError(
            f"adapter identity field {field} drifted",
            details={"field": field},
        )


def bind_adapter_request(
    task: TaskSpecification,
    context_pack: ContextPack,
    route: ModelRouteDecision,
) -> None:
    """Require task, pack, and route identities to agree."""

    _same_identity(task.task_id, context_pack.task_id, field="task_id")
    _same_identity(task.task_id, route.task_id, field="task_id")
    _same_identity(
        task.repository_state_cid,
        context_pack.repository_state_cid,
        field="repository_state_cid",
    )
    _same_identity(
        task.repository_state_cid,
        route.repository_state_cid,
        field="repository_state_cid",
    )
    _same_identity(task.route_cid, route.decision_cid, field="route_cid")
    if context_pack.sufficiency != "sufficient":
        raise BoundaryViolationError(
            "adapters cannot propose from an insufficient ContextPack",
            details={"reason": "context_insufficient"},
        )


def _reject_self_approval(payload: Mapping[str, Any] | None) -> None:
    if payload is None:
        return
    if _truthy(payload.get("self_approved")) or _truthy(payload.get("adapter_approved")):
        raise BoundaryViolationError("an adapter cannot approve its own patch")
    if _truthy(payload.get("accepted")) or _truthy(payload.get("approved")):
        raise BoundaryViolationError("an adapter cannot approve its own patch")
    if _truthy(payload.get("published")):
        raise BoundaryViolationError("an adapter cannot publish or accept a patch")
    adapter_id = payload.get("adapter_id")
    approver_id = payload.get("approver_id")
    if (
        isinstance(adapter_id, str)
        and isinstance(approver_id, str)
        and adapter_id
        and adapter_id == approver_id
    ):
        raise BoundaryViolationError("an adapter cannot approve its own patch")


def require_explicit_usage(invocation: CodingAgentInvocation) -> None:
    missing = [name for name in INVOCATION_USAGE_FIELDS if getattr(invocation, name) is None]
    if missing:
        raise MalformedError(f"adapter invocation is missing explicit {missing[0]}")


def require_live_evidence(invocation: CodingAgentInvocation, proposal: PatchProposal) -> None:
    if invocation.provenance == "live" or proposal.provenance == "live":
        if invocation.provenance != "live" or proposal.provenance != "live":
            raise SimulatedPromotedError(
                "live and non-live provenance cannot be mixed on one adapter result"
            )
        if not invocation.has_live_evidence():
            raise SimulatedPromotedError(
                "live adapter results require a response artifact and explicit usage"
            )
        return
    if invocation.has_live_evidence() and invocation.provenance != "live":
        raise SimulatedPromotedError(
            "replayed or simulated results cannot claim live response identity"
        )


def admit_adapter_result(
    task: TaskSpecification,
    context_pack: ContextPack,
    route: ModelRouteDecision,
    result: AdapterResult,
    *,
    cancellation: CancellationToken | None = None,
) -> AdapterResult:
    """Admit a bounded, schema-valid, non-authoritative adapter result."""

    if cancellation is not None:
        cancellation.check()
    bind_adapter_request(task, context_pack, route)
    proposal = result.proposal
    invocation = result.invocation
    _same_identity(task.task_id, proposal.task_id, field="task_id")
    _same_identity(task.task_id, invocation.task_id, field="task_id")
    _same_identity(
        task.repository_state_cid,
        proposal.repository_state_cid,
        field="repository_state_cid",
    )
    _same_identity(
        task.repository_state_cid,
        invocation.repository_state_cid,
        field="repository_state_cid",
    )
    _same_identity(route.provider, invocation.provider, field="provider")
    _same_identity(route.model, invocation.model, field="model")
    _same_identity(route.revision, invocation.revision, field="revision")
    _same_identity(route.tier, invocation.tier, field="tier")
    _same_identity(route.decision_cid, invocation.route_cid, field="route_cid")
    _same_identity(invocation.invocation_cid, proposal.invocation_cid, field="invocation_cid")
    extra_allow = task.declared_files
    assert_declared_scope(proposal.declared_files, task.owned_paths, extra_allow)
    require_explicit_usage(invocation)
    require_live_evidence(invocation, proposal)
    if result.accepted or result.approved:
        raise BoundaryViolationError("an adapter cannot approve its own patch")
    if result.cancelled:
        raise ProofCancelledError("adapter invocation cancelled")
    object.__setattr__(result, "patch_bytes", admit_bounded_patch(result.patch_bytes))
    object.__setattr__(result, "log_bytes", admit_bounded_log(result.log_bytes))
    if len(result.patch_bytes) > MAX_FILE_BYTES and len(proposal.declared_files) == 1:
        raise BoundaryViolationError("single-file patch exceeds the frozen byte bound")
    if len(result.patch_bytes) + len(result.log_bytes) > MAX_PROVIDER_OUTPUT_BYTES:
        raise BoundaryViolationError("provider output exceeds the frozen byte bound")
    return result


@dataclass(frozen=True)
class AdapterResult:
    """Runtime bundle of exact wire records. Not itself a competing wire schema."""

    proposal: PatchProposal
    invocation: CodingAgentInvocation
    cancelled: bool = False
    patch_bytes: bytes = b""
    log_bytes: bytes = b""
    accepted: bool = field(init=False, default=False)
    approved: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted", False)
        object.__setattr__(self, "approved", False)
        object.__setattr__(self, "patch_bytes", admit_bounded_patch(self.patch_bytes))
        object.__setattr__(self, "log_bytes", admit_bounded_log(self.log_bytes))
        if self.cancelled and (
            self.proposal.provenance == "live" or self.invocation.provenance == "live"
        ):
            raise BoundaryViolationError("cancelled proposals cannot claim live provenance")

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": ADAPTER_RESULT_SCHEMA,
                "proposal": dict(self.proposal.to_mapping()),
                "invocation": dict(self.invocation.to_mapping()),
                "cancelled": self.cancelled,
                "accepted": False,
                "approved": False,
                "approval_authority": APPROVAL_AUTHORITY,
                "canonical_branch_authority": CANONICAL_BRANCH_AUTHORITY,
                "patch_bytes": len(self.patch_bytes),
                "log_bytes": len(self.log_bytes),
            }
        )


@runtime_checkable
class CodingAgentAdapter(Protocol):
    """Provider-neutral proposal protocol. Lifecycle remains the sole acceptor."""

    def propose(
        self,
        task: TaskSpecification,
        context_pack: ContextPack,
        route: ModelRouteDecision,
        cancellation: CancellationToken | None = None,
    ) -> AdapterResult: ...

    def cancel(self, cancellation: CancellationToken) -> None: ...


def cancel_adapter(adapter: CodingAgentAdapter, cancellation: CancellationToken) -> None:
    adapter.cancel(cancellation)
    cancellation.cancel()


def execute_propose(
    adapter: CodingAgentAdapter,
    task: TaskSpecification,
    context_pack: ContextPack,
    route: ModelRouteDecision,
    cancellation: CancellationToken | None = None,
) -> AdapterResult:
    """Run propose through the frozen contract. Does not invoke a provider itself."""

    if cancellation is not None:
        cancellation.check()
    bind_adapter_request(task, context_pack, route)
    raw = adapter.propose(task, context_pack, route, cancellation)
    if not isinstance(raw, AdapterResult):
        raise MalformedError("adapters must return AdapterResult")
    _reject_self_approval(raw.to_mapping())
    _reject_self_approval(raw.proposal.to_mapping())
    _reject_self_approval(raw.invocation.to_mapping())
    return admit_adapter_result(
        task,
        context_pack,
        route,
        raw,
        cancellation=cancellation,
    )


def _snapshot_callable(fn: Any) -> Mapping[str, Any]:
    signature = inspect.signature(fn)
    parameters: list[str] = []
    keyword_only: list[str] = []
    for name, parameter in signature.parameters.items():
        if name in {"self", "cls"}:
            continue
        parameters.append(name)
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            keyword_only.append(name)
    annotation = signature.return_annotation
    if annotation is inspect.Signature.empty:
        return_name = None
    elif isinstance(annotation, str):
        return_name = annotation
    else:
        return_name = getattr(annotation, "__name__", str(annotation))
    return MappingProxyType(
        {
            "parameters": tuple(parameters),
            "keyword_only": tuple(keyword_only),
            "return": return_name,
        }
    )


def protocol_signature() -> Mapping[str, Any]:
    """Stable CodingAgentAdapter method snapshot used as PCCE-030 evidence."""

    return MappingProxyType(
        {
            "interface": INTERFACE,
            "propose": dict(_snapshot_callable(CodingAgentAdapter.propose)),
            "cancel": dict(_snapshot_callable(CodingAgentAdapter.cancel)),
            "approval_authority": APPROVAL_AUTHORITY,
            "canonical_branch_authority": CANONICAL_BRANCH_AUTHORITY,
            "provider_bound": PROVIDER_BOUND,
        }
    )


def _mint_cid(value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(wire_canonical_utf8(value).encode("utf-8")).digest()
    raw = bytes([0x01, 0x55, 0x12, 0x20]) + digest
    return "b" + base64.b32encode(raw).decode("ascii").lower().rstrip("=")


_DESCRIPTOR_BODY: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "schema": ADAPTER_SCHEMA,
        "interface": INTERFACE,
        "contract_version": CONTRACT_VERSION,
        "contract_schema_prefix": CONTRACT_SCHEMA_PREFIX,
        "runtime_schema": SCHEMA,
        "wire_schemas": WIRE_SCHEMAS,
        "protocol_methods": PROTOCOL_METHODS,
        "provenances": PROVENANCES,
        "sibling_layout_required": SIBLING_LAYOUT_REQUIRED,
        "provider_bound": PROVIDER_BOUND,
        "approval_authority": APPROVAL_AUTHORITY,
        "canonical_branch_authority": CANONICAL_BRANCH_AUTHORITY,
        "max_patch_bytes": MAX_PATCH_BYTES,
        "max_provider_output_bytes": MAX_PROVIDER_OUTPUT_BYTES,
        "max_file_bytes": MAX_FILE_BYTES,
        "max_log_bytes": MAX_LOG_BYTES,
        "task_specification_schema": TASK_SPECIFICATION_SCHEMA,
        "coding_agent_invocation_schema": CODING_AGENT_INVOCATION_SCHEMA,
        "patch_proposal_schema": PATCH_PROPOSAL_SCHEMA,
        "context_pack_schema": CONTEXT_PACK_SCHEMA,
        "model_route_decision_schema": MODEL_ROUTE_DECISION_SCHEMA,
        "task_specification_schema_digest": TASK_SPECIFICATION_SCHEMA_DIGEST,
        "coding_agent_invocation_schema_digest": CODING_AGENT_INVOCATION_SCHEMA_DIGEST,
        "patch_proposal_schema_digest": PATCH_PROPOSAL_SCHEMA_DIGEST,
        "context_pack_schema_digest": CONTEXT_PACK_SCHEMA_DIGEST,
        "model_route_decision_schema_digest": MODEL_ROUTE_DECISION_SCHEMA_DIGEST,
        "pcce_006_content_id": PCCE_006_CONTENT_ID,
        "compatibility_matrix_content_id": COMPATIBILITY_MATRIX_CONTENT_ID,
        "protocol_signature": dict(protocol_signature()),
    }
)
ADAPTER_CONTRACT_CID: Final[str] = _mint_cid(_DESCRIPTOR_BODY)
ADAPTER_CONTRACT_DESCRIPTOR: Final[Mapping[str, Any]] = MappingProxyType(
    {**dict(_DESCRIPTOR_BODY), "cid": ADAPTER_CONTRACT_CID}
)


def adapter_contract_descriptor() -> Mapping[str, Any]:
    return ADAPTER_CONTRACT_DESCRIPTOR


def adapter_contract_cid() -> str:
    return ADAPTER_CONTRACT_CID


def frozen_adapter_contract() -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "interface": INTERFACE,
            "cid": ADAPTER_CONTRACT_CID,
            "wire_schemas": WIRE_SCHEMAS,
            "approval_authority": APPROVAL_AUTHORITY,
            "canonical_branch_authority": CANONICAL_BRANCH_AUTHORITY,
            "provider_bound": PROVIDER_BOUND,
            "pcce_006_content_id": PCCE_006_CONTENT_ID,
        }
    )


__all__ = [
    "ADAPTER_CONTRACT_CID",
    "ADAPTER_CONTRACT_DESCRIPTOR",
    "ADAPTER_RESULT_SCHEMA",
    "ADAPTER_SCHEMA",
    "APPROVAL_AUTHORITY",
    "AdapterResult",
    "CANONICAL_BRANCH_AUTHORITY",
    "COMPATIBILITY_MATRIX_CONTENT_ID",
    "CONTRACT_SCHEMA_PREFIX",
    "CONTRACT_VERSION",
    "CancellationToken",
    "CodingAgentAdapter",
    "INTERFACE",
    "PROTOCOL_METHODS",
    "PROVIDER_BOUND",
    "SCHEMA",
    "SIBLING_LAYOUT_REQUIRED",
    "adapter_contract_cid",
    "adapter_contract_descriptor",
    "admit_adapter_result",
    "bind_adapter_request",
    "cancel_adapter",
    "execute_propose",
    "frozen_adapter_contract",
    "protocol_signature",
    "require_explicit_usage",
    "require_live_evidence",
]
