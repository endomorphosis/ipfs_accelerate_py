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
import re
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
    PATCH_PROPOSAL_SCHEMA,
    PATCH_PROPOSAL_SCHEMA_DIGEST,
    PCCE_006_CONTENT_ID,
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
    admit_relative_path,
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
ADAPTER_RESULT_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/coding-agent-adapter-result"
SIBLING_LAYOUT_REQUIRED: Final[bool] = False
PROVIDER_BOUND: Final[bool] = False
APPROVAL_AUTHORITY: Final[bool] = False
CANONICAL_BRANCH_AUTHORITY: Final[bool] = False
COMPATIBILITY_MATRIX_CONTENT_ID: Final[str] = FROZEN_MATRIX["content_id"]

PROTOCOL_METHODS: Final[tuple[str, ...]] = ("propose", "cancel")
_HUNK_HEADER: Final[re.Pattern[str]] = re.compile(
    r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@(?: .*)?$"
)
_EXTENDED_PATH_HEADERS: Final[tuple[tuple[str, str, str], ...]] = (
    ("rename from ", "rename", "from"),
    ("rename to ", "rename", "to"),
    ("copy from ", "copy", "from"),
    ("copy to ", "copy", "to"),
)
_EXTENDED_PATH_STEMS: Final[tuple[str, ...]] = (
    "rename from",
    "rename to",
    "copy from",
    "copy to",
)


def _unquoted_patch_path(raw: str) -> str:
    """Admit one unquoted Git path under the repository-relative grammar."""

    if (
        '"' in raw
        or "\\" in raw
        or any(ord(character) < 32 or ord(character) == 127 for character in raw)
    ):
        raise MalformedError("adapter patch uses unsupported path syntax")
    return admit_relative_path(raw, field="patch_path")


def _diff_header_paths(line: str) -> tuple[str, str]:
    """Admit both repository paths from one unquoted ``diff --git`` header."""

    prefix = "diff --git a/"
    if not line.startswith(prefix):
        raise MalformedError("adapter patch has a malformed diff header")
    body = line[len(prefix) :]
    # An unquoted path containing the separator is ambiguous. Git's quoted
    # pathname grammar is deliberately outside this frozen, fail-closed parser.
    if body.count(" b/") != 1:
        raise MalformedError("adapter patch has an ambiguous diff header")
    old_raw, new_raw = body.split(" b/", 1)
    old_path = _unquoted_patch_path(old_raw)
    new_path = _unquoted_patch_path(new_raw)
    return old_path, new_path


def _file_marker_path(raw: str, *, side: str) -> str | None:
    """Admit one ``---``/``+++`` path without exposing it in diagnostics."""

    # A tab introduces the optional unified-diff timestamp. Literal tabs in a
    # Git pathname are represented by Git's quoted form, which is not admitted.
    value = raw.split("\t", 1)[0]
    if value == "/dev/null":
        return None
    prefix = "a/" if side == "old" else "b/"
    if not value.startswith(prefix):
        raise MalformedError("adapter patch has a malformed file marker")
    return _unquoted_patch_path(value[len(prefix) :])


def _extended_header_path(raw: str) -> str:
    if raw == "/dev/null":
        raise MalformedError("adapter patch extended headers cannot use a null path")
    return _unquoted_patch_path(raw)


def _binary_summary_path(raw: str, *, side: str) -> str | None:
    """Admit one path from Git's ``Binary files ... differ`` summary."""

    if raw == "/dev/null":
        return None
    prefix = "a/" if side == "old" else "b/"
    if not raw.startswith(prefix):
        raise MalformedError("adapter patch has a malformed binary summary")
    return _unquoted_patch_path(raw[len(prefix) :])


def _parse_unified_diff_paths(patch_bytes: Any) -> tuple[str, ...]:
    """Return every path named by a bounded, textual Git unified diff.

    This is a pure scope-binding parser, not an apply or correctness oracle.
    It admits header-only replay evidence, while tracking hunk lengths so
    marker-looking source lines cannot be confused with file markers.
    """

    patch = admit_bounded_patch(patch_bytes)
    if not patch:
        raise MalformedError("adapter patch must not be empty")
    text: str | None
    try:
        text = patch.decode("utf-8", "strict")
    except UnicodeDecodeError:
        text = None
    if text is None:
        # Raise outside the handler: UnicodeDecodeError.object retains the raw
        # provider bytes even when an exception is raised ``from None``.
        raise MalformedError("adapter patch must be valid UTF-8 text")
    if "\x00" in text:
        raise MalformedError("adapter patch must not contain NUL bytes")

    lines = text.splitlines()
    paths: list[str] = []
    seen: set[str] = set()
    current: tuple[str, str] | None = None
    file_markers_seen = False
    binary_summary_seen = False
    extended_kind: str | None = None
    extended_pending: str | None = None
    hunk_remaining: tuple[int, int] | None = None
    header_count = 0
    index = 0

    def record(path: str) -> None:
        if path not in seen:
            seen.add(path)
            paths.append(path)

    while index < len(lines):
        line = lines[index]
        if hunk_remaining is not None:
            old_remaining, new_remaining = hunk_remaining
            if old_remaining == 0 and new_remaining == 0:
                if line == r"\ No newline at end of file":
                    index += 1
                hunk_remaining = None
                continue
            if line == r"\ No newline at end of file":
                index += 1
                continue
            if line.startswith(" "):
                old_remaining -= 1
                new_remaining -= 1
            elif line.startswith("-"):
                old_remaining -= 1
            elif line.startswith("+"):
                new_remaining -= 1
            else:
                raise MalformedError("adapter patch has a malformed hunk body")
            if old_remaining < 0 or new_remaining < 0:
                raise MalformedError("adapter patch hunk exceeds its declared length")
            hunk_remaining = (old_remaining, new_remaining)
            index += 1
            continue

        if line.startswith("diff --git "):
            if extended_pending is not None:
                raise MalformedError("adapter patch has an unpaired extended path header")
            current = _diff_header_paths(line)
            record(current[0])
            record(current[1])
            header_count += 1
            file_markers_seen = False
            binary_summary_seen = False
            extended_kind = None
            extended_pending = None
            index += 1
            continue
        if line.startswith("diff --git"):
            raise MalformedError("adapter patch has a malformed diff header")
        if line.startswith(("diff --cc ", "diff --combined ")):
            raise MalformedError("adapter patch uses an unsupported combined diff header")

        if line.startswith("--- "):
            if current is None or file_markers_seen:
                raise MalformedError("adapter patch has a foreign file marker")
            if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
                raise MalformedError("adapter patch has an unpaired file marker")
            old_marker = _file_marker_path(line[4:], side="old")
            new_marker = _file_marker_path(lines[index + 1][4:], side="new")
            if old_marker is None and new_marker is None:
                raise MalformedError("adapter patch file markers cannot both be null")
            if (old_marker is not None and old_marker != current[0]) or (
                new_marker is not None and new_marker != current[1]
            ):
                raise BoundaryViolationError(
                    "adapter patch file markers disagree with its diff header",
                    details={"field": "declared_files", "reason": "scope"},
                )
            file_markers_seen = True
            index += 2
            continue
        if line.startswith("+++ "):
            raise MalformedError("adapter patch has an unpaired file marker")

        if line.startswith("Binary files "):
            if current is None or file_markers_seen or binary_summary_seen:
                raise MalformedError("adapter patch has a foreign binary summary")
            body = line[len("Binary files ") :]
            suffix = " differ"
            if not body.endswith(suffix) or body[: -len(suffix)].count(" and ") != 1:
                raise MalformedError("adapter patch has a malformed binary summary")
            old_raw, new_raw = body[: -len(suffix)].split(" and ", 1)
            old_summary = _binary_summary_path(old_raw, side="old")
            new_summary = _binary_summary_path(new_raw, side="new")
            if old_summary is None and new_summary is None:
                raise MalformedError("adapter patch binary paths cannot both be null")
            if (old_summary is not None and old_summary != current[0]) or (
                new_summary is not None and new_summary != current[1]
            ):
                raise BoundaryViolationError(
                    "adapter patch binary summary disagrees with its diff header",
                    details={"field": "declared_files", "reason": "scope"},
                )
            binary_summary_seen = True
            index += 1
            continue
        if line.startswith("Binary files"):
            raise MalformedError("adapter patch has a malformed binary summary")

        matched_extended = False
        for prefix, kind, direction in _EXTENDED_PATH_HEADERS:
            if not line.startswith(prefix):
                continue
            matched_extended = True
            if current is None:
                raise MalformedError("adapter patch has a foreign extended path header")
            path = _extended_header_path(line[len(prefix) :])
            if direction == "from":
                if extended_kind is not None or extended_pending is not None:
                    raise MalformedError("adapter patch repeats an extended path header")
                if path != current[0]:
                    raise BoundaryViolationError(
                        "adapter patch extended headers disagree with its diff header",
                        details={"field": "declared_files", "reason": "scope"},
                    )
                extended_pending = kind
            else:
                if extended_pending != kind or extended_kind is not None:
                    raise MalformedError("adapter patch has an unpaired extended path header")
                if path != current[1]:
                    raise BoundaryViolationError(
                        "adapter patch extended headers disagree with its diff header",
                        details={"field": "declared_files", "reason": "scope"},
                    )
                extended_pending = None
                extended_kind = kind
            index += 1
            break
        if matched_extended:
            continue
        if line.startswith(_EXTENDED_PATH_STEMS):
            raise MalformedError("adapter patch has a malformed extended path header")

        if line.startswith("@@"):
            if current is None:
                raise MalformedError("adapter patch has a hunk without a diff header")
            match = _HUNK_HEADER.fullmatch(line)
            if match is None:
                raise MalformedError("adapter patch has a malformed hunk header")
            old_raw, new_raw = match.groups()
            if any(raw is not None and len(raw) > 7 for raw in (old_raw, new_raw)):
                raise MalformedError("adapter patch has an invalid hunk length")
            old_count = int(old_raw) if old_raw is not None else 1
            new_count = int(new_raw) if new_raw is not None else 1
            hunk_remaining = (old_count, new_count)
            index += 1
            continue

        index += 1

    if hunk_remaining not in {None, (0, 0)}:
        raise MalformedError("adapter patch hunk is shorter than its declared length")
    if extended_pending is not None:
        raise MalformedError("adapter patch has an unpaired extended path header")
    if header_count == 0 or not paths:
        raise MalformedError("adapter patch must contain a diff header")
    return tuple(paths)


def _assert_patch_paths_match_declared(
    patch_bytes: Any,
    declared_files: tuple[str, ...],
) -> None:
    """Require exact equality between parsed patch paths and the wire claim."""

    if not declared_files:
        raise BoundaryViolationError(
            "adapter patch requires at least one declared file",
            details={"field": "declared_files", "reason": "scope"},
        )
    admitted_declared = tuple(
        admit_relative_path(path, field="declared_files") for path in declared_files
    )
    parsed = _parse_unified_diff_paths(patch_bytes)
    if set(parsed) != set(admitted_declared):
        raise BoundaryViolationError(
            "adapter patch paths do not match declared files",
            details={"field": "declared_files", "reason": "scope"},
        )


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
    _assert_patch_paths_match_declared(result.patch_bytes, proposal.declared_files)
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
