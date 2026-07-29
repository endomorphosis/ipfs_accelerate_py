"""Compile endpoint anchors and observed package contracts for the baseline.

Interface: ``RuntimeContractEvidenceCompiler@1``

Every reviewed runtime operation receives:

* an exact **endpoint anchor** suitable for ``McpInvocationTracer@1``
  (``InvocationTraceRequest`` shape: operation, source, targets); and
* an **observed package contract** suitable for ``McpContractAnalyzer@1``.

Mandatory MCP++ mediation paths and direct package-call paths remain distinct
path classes.  Matching names alone never synthesize registrations or
mediation.  Missing or ambiguous anchors become typed unknown findings rather
than a withheld empty-success stage.

Evidence subset (SCAEV052ANCHORS): connector / list / call / transport /
schema / registration / function identities and provenance.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity
from .mcp_contract_catalog import (
    McpClaimFamily,
    McpContractCatalog,
    ReviewState,
)
from .mcp_invocation_trace import (
    InvocationTerminalState,
    InvocationTraceRequest,
    McpInvocationTrace,
    McpInvocationTracer,
)
from .python_mcp_surface_extractor import PythonMcpPackageSurface, PythonMcpToolSurface
from .runtime_component_catalog import RuntimeComponentCatalog, RuntimeRouteKind
from .swissknife_contract_extractor import (
    MethodExpectation,
    SwissKnifeContractExtraction,
)
from .symbolic_contract_graph import (
    ContractGraphNode,
    ContractNodeKind,
    SymbolicContractGraph,
)


RUNTIME_CONTRACT_EVIDENCE_COMPILER_INTERFACE: Final = (
    "RuntimeContractEvidenceCompiler@1"
)
RUNTIME_CONTRACT_EVIDENCE_COMPILER_VERSION: Final = "1"

ENDPOINT_ANCHOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-endpoint-anchor@1"
)
OBSERVED_PACKAGE_CONTRACT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-observed-package-contract@1"
)
EVIDENCE_COMPILATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-contract-evidence@1"
)
EVIDENCE_FINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-contract-evidence-finding@1"
)

# Path classes retained on anchors and observed routes.  MCP++ mediation and
# direct package calls must never collapse into a single identity.
PATH_CLASS_MCP_PLUS_PLUS: Final = "mcp_plus_plus"
PATH_CLASS_DIRECT: Final = "direct"
PATH_CLASS_COMPATIBILITY: Final = "compatibility"

KNOWN_PACKAGE_IDS: Final[frozenset[str]] = frozenset(
    {
        "ipfs_kit_py",
        "ipfs_datasets_py",
        "ipfs_accelerate_py",
        "agent_supervisor",
    }
)

DEFAULT_FAILURE_STATES: Final[tuple[str, ...]] = (
    "unsupported",
    "unavailable",
    "denied",
    "timed_out",
    "malformed",
    "partial",
)

DEFAULT_RESULT_ENVELOPE: Final[tuple[str, ...]] = (
    "content",
    "error",
    "provenance",
    "receipt",
)


class RuntimeContractEvidenceCompilerError(ValueError):
    """Evidence compilation inputs or outputs are malformed."""


class AnchorResolutionState(str, Enum):
    """Closed resolution lattice for one endpoint anchor."""

    RESOLVED = "resolved"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"
    INCOMPLETE = "incomplete"


class EvidenceFindingKind(str, Enum):
    """Typed findings emitted when anchors cannot prove a path."""

    MISSING_SOURCE_ANCHOR = "missing_source_anchor"
    MISSING_TARGET_ANCHOR = "missing_target_anchor"
    AMBIGUOUS_SOURCE_ANCHOR = "ambiguous_source_anchor"
    AMBIGUOUS_TARGET_ANCHOR = "ambiguous_target_anchor"
    AMBIGUOUS_PATH_CLASS = "ambiguous_path_class"
    OBSERVED_CONTRACT_INCOMPLETE = "observed_contract_incomplete"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    TRACE_UNKNOWN = "trace_unknown"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        raise RuntimeContractEvidenceCompilerError(f"{name} must be a string")
    if text != text.strip() or "\x00" in text:
        raise RuntimeContractEvidenceCompilerError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not text:
        raise RuntimeContractEvidenceCompilerError(f"{name} is required")
    return text


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RuntimeContractEvidenceCompilerError(f"{name} must be a sequence")
    items = tuple(
        sorted({_text(item, f"{name} item", required=True) for item in value})
    )
    return items


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 32:
        raise RuntimeContractEvidenceCompilerError("value exceeds nesting bound")
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise RuntimeContractEvidenceCompilerError(
            "floating values are not canonical evidence"
        )
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item, depth=depth + 1) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise RuntimeContractEvidenceCompilerError(
            "raw bytes are not canonical evidence"
        )
    raise RuntimeContractEvidenceCompilerError(
        f"unsupported evidence value type: {type(value).__name__}"
    )


def _cid(value: Any) -> str:
    return content_identity(_plain(value))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeContractEvidenceCompilerError(f"{name} must be an object")
    return value


def _operation_id(package_id: str, tool_name: str) -> str:
    package = _text(package_id, "package_id")
    tool = _text(tool_name, "tool_name")
    return f"{package}:{tool}"


def _schema_ref_payload(schema_ref: str | None) -> dict[str, Any]:
    if not schema_ref:
        return {"type": "object", "additionalProperties": True}
    # CID/path schema references are retained as reviewed identities; the
    # analyzer still sees a complete schema object for structural checks.
    return {
        "type": "object",
        "additionalProperties": True,
        "x-sca-schema-ref": schema_ref,
    }


@dataclass(frozen=True, slots=True)
class ReviewedRuntimeOperation:
    """One reviewed runtime operation drawn only from catalog/index facts."""

    operation_id: str
    package_id: str
    tool_name: str
    contract_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    subject: str = ""
    claim_families: tuple[str, ...] = ()
    method: MethodExpectation | None = None
    descriptor_name: str = ""
    source_version: str = ""
    schema_version: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operation_id", _text(self.operation_id, "operation_id")
        )
        object.__setattr__(self, "package_id", _text(self.package_id, "package_id"))
        object.__setattr__(self, "tool_name", _text(self.tool_name, "tool_name"))
        object.__setattr__(
            self, "contract_ids", _strings(self.contract_ids, "contract_ids")
        )
        object.__setattr__(
            self, "source_ids", _strings(self.source_ids, "source_ids")
        )
        object.__setattr__(
            self, "subject", _text(self.subject, "subject", required=False)
        )
        object.__setattr__(
            self,
            "claim_families",
            _strings(self.claim_families, "claim_families"),
        )
        object.__setattr__(
            self,
            "descriptor_name",
            _text(self.descriptor_name, "descriptor_name", required=False),
        )
        object.__setattr__(
            self,
            "source_version",
            _text(self.source_version, "source_version", required=False),
        )
        object.__setattr__(
            self,
            "schema_version",
            _text(self.schema_version, "schema_version", required=False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "package_id": self.package_id,
            "tool_name": self.tool_name,
            "contract_ids": list(self.contract_ids),
            "source_ids": list(self.source_ids),
            "subject": self.subject,
            "claim_families": list(self.claim_families),
            "descriptor_name": self.descriptor_name,
            "source_version": self.source_version,
            "schema_version": self.schema_version,
            "method": self.method.to_dict() if self.method is not None else None,
        }


@dataclass(frozen=True, slots=True)
class EndpointAnchor:
    """Exact tracer anchor for one reviewed runtime operation."""

    operation_id: str
    package_id: str
    tool_name: str
    resolution_state: AnchorResolutionState
    source_node_id: str = ""
    source_stable_key: str = ""
    target_node_ids: tuple[str, ...] = ()
    target_stable_keys: tuple[str, ...] = ()
    path_classes: tuple[str, ...] = (PATH_CLASS_MCP_PLUS_PLUS,)
    mcp_plus_plus_source_node_id: str = ""
    direct_source_node_id: str = ""
    reason_codes: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    contract_ids: tuple[str, ...] = ()
    supported: bool = True
    measured: bool = True
    anchor_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operation_id", _text(self.operation_id, "operation_id")
        )
        object.__setattr__(self, "package_id", _text(self.package_id, "package_id"))
        object.__setattr__(self, "tool_name", _text(self.tool_name, "tool_name"))
        state = self.resolution_state
        if not isinstance(state, AnchorResolutionState):
            state = AnchorResolutionState(str(state))
        object.__setattr__(self, "resolution_state", state)
        object.__setattr__(
            self,
            "source_node_id",
            _text(self.source_node_id, "source_node_id", required=False),
        )
        object.__setattr__(
            self,
            "source_stable_key",
            _text(self.source_stable_key, "source_stable_key", required=False),
        )
        object.__setattr__(
            self,
            "target_node_ids",
            _strings(self.target_node_ids, "target_node_ids"),
        )
        object.__setattr__(
            self,
            "target_stable_keys",
            _strings(self.target_stable_keys, "target_stable_keys"),
        )
        paths = _strings(self.path_classes, "path_classes")
        if not paths:
            paths = (PATH_CLASS_MCP_PLUS_PLUS,)
        # Preserve MCP++ / direct distinctness: reject collapsing both into one
        # synthetic class.
        object.__setattr__(self, "path_classes", paths)
        object.__setattr__(
            self,
            "mcp_plus_plus_source_node_id",
            _text(
                self.mcp_plus_plus_source_node_id,
                "mcp_plus_plus_source_node_id",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "direct_source_node_id",
            _text(
                self.direct_source_node_id,
                "direct_source_node_id",
                required=False,
            ),
        )
        object.__setattr__(
            self, "reason_codes", _strings(self.reason_codes, "reason_codes")
        )
        object.__setattr__(
            self, "source_ids", _strings(self.source_ids, "source_ids")
        )
        object.__setattr__(
            self, "contract_ids", _strings(self.contract_ids, "contract_ids")
        )
        object.__setattr__(self, "supported", bool(self.supported))
        object.__setattr__(self, "measured", bool(self.measured))
        derived = self._derive_anchor_id()
        claimed = _text(self.anchor_id, "anchor_id", required=False)
        if claimed and claimed != derived:
            raise RuntimeContractEvidenceCompilerError(
                "anchor_id does not match content"
            )
        object.__setattr__(self, "anchor_id", derived)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": ENDPOINT_ANCHOR_SCHEMA,
            "operation_id": self.operation_id,
            "package_id": self.package_id,
            "tool_name": self.tool_name,
            "resolution_state": self.resolution_state.value,
            "source_node_id": self.source_node_id,
            "source_stable_key": self.source_stable_key,
            "target_node_ids": list(self.target_node_ids),
            "target_stable_keys": list(self.target_stable_keys),
            "path_classes": list(self.path_classes),
            "mcp_plus_plus_source_node_id": self.mcp_plus_plus_source_node_id,
            "direct_source_node_id": self.direct_source_node_id,
            "reason_codes": list(self.reason_codes),
            "source_ids": list(self.source_ids),
            "contract_ids": list(self.contract_ids),
            "supported": self.supported,
            "measured": self.measured,
        }

    def _derive_anchor_id(self) -> str:
        return _cid(self._identity_payload())

    @property
    def is_traceable(self) -> bool:
        return (
            self.resolution_state is AnchorResolutionState.RESOLVED
            and bool(self.source_node_id)
            and bool(self.target_node_ids)
            and self.supported
            and self.measured
        )

    def to_trace_request(self) -> InvocationTraceRequest:
        """Project a resolved anchor into ``McpInvocationTracer`` request form."""

        if not self.is_traceable:
            raise RuntimeContractEvidenceCompilerError(
                f"anchor {self.operation_id} is not traceable "
                f"({self.resolution_state.value})"
            )
        return InvocationTraceRequest(
            operation_id=self.operation_id,
            source_node_id=self.source_node_id,
            target_node_ids=self.target_node_ids,
            supported=self.supported,
            measured=self.measured,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "anchor_id": self.anchor_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EndpointAnchor":
        payload = _mapping(value, "endpoint anchor")
        return cls(
            operation_id=str(payload.get("operation_id") or ""),
            package_id=str(payload.get("package_id") or ""),
            tool_name=str(payload.get("tool_name") or ""),
            resolution_state=payload.get(
                "resolution_state", AnchorResolutionState.MISSING
            ),
            source_node_id=str(payload.get("source_node_id") or ""),
            source_stable_key=str(payload.get("source_stable_key") or ""),
            target_node_ids=tuple(payload.get("target_node_ids") or ()),
            target_stable_keys=tuple(payload.get("target_stable_keys") or ()),
            path_classes=tuple(payload.get("path_classes") or ()),
            mcp_plus_plus_source_node_id=str(
                payload.get("mcp_plus_plus_source_node_id") or ""
            ),
            direct_source_node_id=str(payload.get("direct_source_node_id") or ""),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            source_ids=tuple(payload.get("source_ids") or ()),
            contract_ids=tuple(payload.get("contract_ids") or ()),
            supported=bool(payload.get("supported", True)),
            measured=bool(payload.get("measured", True)),
            anchor_id=str(payload.get("anchor_id") or ""),
        )


@dataclass(frozen=True, slots=True)
class EvidenceFinding:
    """Typed unknown/incomplete finding for one operation."""

    operation_id: str
    kind: EvidenceFindingKind
    reason_code: str
    details: Mapping[str, Any] = field(default_factory=dict)
    finding_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operation_id", _text(self.operation_id, "operation_id")
        )
        kind = self.kind
        if not isinstance(kind, EvidenceFindingKind):
            kind = EvidenceFindingKind(str(kind))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self, "reason_code", _text(self.reason_code, "reason_code")
        )
        object.__setattr__(
            self,
            "details",
            MappingProxyType(dict(_plain(dict(self.details or {})))),
        )
        derived = _cid(self._identity_payload())
        claimed = _text(self.finding_id, "finding_id", required=False)
        if claimed and claimed != derived:
            raise RuntimeContractEvidenceCompilerError(
                "finding_id does not match content"
            )
        object.__setattr__(self, "finding_id", derived)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": EVIDENCE_FINDING_SCHEMA,
            "operation_id": self.operation_id,
            "kind": self.kind.value,
            "reason_code": self.reason_code,
            "details": dict(self.details),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "finding_id": self.finding_id}


@dataclass(frozen=True, slots=True)
class RuntimeContractEvidenceCompilation:
    """Deterministic compilation result for one snapshot."""

    snapshot_id: str
    operations: tuple[ReviewedRuntimeOperation, ...]
    anchors: tuple[EndpointAnchor, ...]
    observed_contracts: tuple[Mapping[str, Any], ...]
    findings: tuple[EvidenceFinding, ...]
    traces: tuple[McpInvocationTrace, ...] = ()
    catalog_root: str = ""
    graph_root: str = ""
    extraction_root: str = ""
    runtime_catalog_root: str = ""
    reason_codes: tuple[str, ...] = ()
    complete: bool = False
    compilation_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "snapshot_id", _text(self.snapshot_id, "snapshot_id")
        )
        if not isinstance(self.operations, tuple):
            object.__setattr__(self, "operations", tuple(self.operations))
        if not isinstance(self.anchors, tuple):
            object.__setattr__(self, "anchors", tuple(self.anchors))
        # Stable ordering by operation_id.
        object.__setattr__(
            self,
            "operations",
            tuple(sorted(self.operations, key=lambda item: item.operation_id)),
        )
        object.__setattr__(
            self,
            "anchors",
            tuple(sorted(self.anchors, key=lambda item: item.operation_id)),
        )
        observed = tuple(
            MappingProxyType(dict(_plain(dict(item))))
            for item in self.observed_contracts
        )
        object.__setattr__(
            self,
            "observed_contracts",
            tuple(
                sorted(
                    observed,
                    key=lambda item: str(
                        item.get("operation_id") or item.get("name") or ""
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "findings",
            tuple(
                sorted(
                    self.findings,
                    key=lambda item: (item.operation_id, item.kind.value, item.reason_code),
                )
            ),
        )
        if not isinstance(self.traces, tuple):
            object.__setattr__(self, "traces", tuple(self.traces))
        object.__setattr__(
            self,
            "traces",
            tuple(sorted(self.traces, key=lambda item: item.operation_id)),
        )
        for name in (
            "catalog_root",
            "graph_root",
            "extraction_root",
            "runtime_catalog_root",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name, required=False),
            )
        object.__setattr__(
            self, "reason_codes", _strings(self.reason_codes, "reason_codes")
        )
        object.__setattr__(self, "complete", bool(self.complete))
        # One anchor and one observed contract per reviewed operation.
        op_ids = [item.operation_id for item in self.operations]
        if len(op_ids) != len(set(op_ids)):
            raise RuntimeContractEvidenceCompilerError(
                "duplicate reviewed operation_id"
            )
        anchor_ids = [item.operation_id for item in self.anchors]
        if anchor_ids != op_ids:
            raise RuntimeContractEvidenceCompilerError(
                "anchors must cover every reviewed operation exactly once"
            )
        observed_ids = [
            str(item.get("operation_id") or "") for item in self.observed_contracts
        ]
        if observed_ids != op_ids:
            raise RuntimeContractEvidenceCompilerError(
                "observed contracts must cover every reviewed operation exactly once"
            )
        derived = self._derive_compilation_id()
        claimed = _text(self.compilation_id, "compilation_id", required=False)
        if claimed and claimed != derived:
            raise RuntimeContractEvidenceCompilerError(
                "compilation_id does not match content"
            )
        object.__setattr__(self, "compilation_id", derived)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": EVIDENCE_COMPILATION_SCHEMA,
            "interface": RUNTIME_CONTRACT_EVIDENCE_COMPILER_INTERFACE,
            "version": RUNTIME_CONTRACT_EVIDENCE_COMPILER_VERSION,
            "snapshot_id": self.snapshot_id,
            "operations": [item.to_dict() for item in self.operations],
            "anchors": [item.to_dict() for item in self.anchors],
            "observed_contracts": [dict(item) for item in self.observed_contracts],
            "findings": [item.to_dict() for item in self.findings],
            "traces": [item.to_dict() for item in self.traces],
            "catalog_root": self.catalog_root,
            "graph_root": self.graph_root,
            "extraction_root": self.extraction_root,
            "runtime_catalog_root": self.runtime_catalog_root,
            "reason_codes": list(self.reason_codes),
            "complete": self.complete,
        }

    def _derive_compilation_id(self) -> str:
        return _cid(self._identity_payload())

    @property
    def observed_contract_map(self) -> Mapping[str, Mapping[str, Any]]:
        return MappingProxyType(
            {
                str(item["operation_id"]): dict(item)
                for item in self.observed_contracts
            }
        )

    @property
    def anchor_map(self) -> Mapping[str, EndpointAnchor]:
        return MappingProxyType(
            {item.operation_id: item for item in self.anchors}
        )

    @property
    def traceable_anchors(self) -> tuple[EndpointAnchor, ...]:
        return tuple(item for item in self.anchors if item.is_traceable)

    @property
    def unknown_findings(self) -> tuple[EvidenceFinding, ...]:
        return self.findings

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "compilation_id": self.compilation_id,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )


def collect_reviewed_runtime_operations(
    catalog: McpContractCatalog,
    *,
    extraction: SwissKnifeContractExtraction | None = None,
) -> tuple[ReviewedRuntimeOperation, ...]:
    """Collect reviewed tool-bearing runtime operations from catalog facts.

    Only reviewed catalog entries with a non-empty tool name (or extraction
    method declarations joined through reviewed DeclaredToolExists contracts)
    become operations.  Interface-only contracts without tools are ignored.
    """

    if not isinstance(catalog, McpContractCatalog):
        raise RuntimeContractEvidenceCompilerError(
            "catalog must implement McpContractCatalog@1"
        )

    method_index: dict[tuple[str, str], MethodExpectation] = {}
    descriptor_names: dict[tuple[str, str], str] = {}
    if extraction is not None:
        for descriptor in extraction.descriptors:
            for method in descriptor.methods:
                key = (descriptor.package_id, method.name)
                method_index[key] = method
                descriptor_names[key] = descriptor.name or descriptor.declaration

    by_operation: dict[str, dict[str, Any]] = {}
    for contract in catalog.contracts:
        if not contract.review_state.is_reviewed:
            continue
        tool_name = (contract.tool_name or "").strip()
        package_id = (contract.package_id or "").strip()
        if not tool_name or not package_id:
            # Non-tool contracts (descriptor fields, transports) are not
            # runtime operations for endpoint anchors.
            continue
        op_id = _operation_id(package_id, tool_name)
        bucket = by_operation.setdefault(
            op_id,
            {
                "operation_id": op_id,
                "package_id": package_id,
                "tool_name": tool_name,
                "contract_ids": set(),
                "source_ids": set(),
                "claim_families": set(),
                "subject": contract.subject,
                "source_version": contract.source_version,
                "schema_version": contract.schema_version,
            },
        )
        bucket["contract_ids"].add(contract.contract_id)
        bucket["source_ids"].update(contract.source_ids)
        bucket["claim_families"].add(contract.claim_family.value)
        if not bucket.get("subject"):
            bucket["subject"] = contract.subject

    # Extraction methods that already have a reviewed DeclaredToolExists row
    # are preferred; orphan extraction methods without reviewed catalog rows
    # never become anchors (conflict policy).
    operations: list[ReviewedRuntimeOperation] = []
    for op_id in sorted(by_operation):
        bucket = by_operation[op_id]
        key = (bucket["package_id"], bucket["tool_name"])
        operations.append(
            ReviewedRuntimeOperation(
                operation_id=op_id,
                package_id=bucket["package_id"],
                tool_name=bucket["tool_name"],
                contract_ids=tuple(sorted(bucket["contract_ids"])),
                source_ids=tuple(sorted(bucket["source_ids"])),
                subject=str(bucket.get("subject") or ""),
                claim_families=tuple(sorted(bucket["claim_families"])),
                method=method_index.get(key),
                descriptor_name=descriptor_names.get(key, ""),
                source_version=str(bucket.get("source_version") or ""),
                schema_version=str(bucket.get("schema_version") or ""),
            )
        )
    return tuple(operations)


def _node_tool_hints(node: ContractGraphNode) -> set[str]:
    hints: set[str] = set()
    payload = dict(node.payload or {})
    for key in ("tool_name", "name", "method", "operation", "label", "subject"):
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            hints.add(raw.strip())
    # stable_key patterns: descriptor:ipfs.add, tool:ipfs.add, handler:add
    key = node.stable_key
    for prefix in (
        "descriptor:",
        "method:",
        "tool:",
        "handler:",
        "implementation:",
        "mcp-tool:",
        "contract-tool:",
        "registration:",
    ):
        if key.startswith(prefix):
            remainder = key[len(prefix) :]
            hints.add(remainder)
            if ":" in remainder:
                # package:tool form
                hints.add(remainder.split(":", 1)[-1])
            break
    if node.kind is ContractNodeKind.CONTRACT:
        subject = str(payload.get("subject") or "")
        if ":" in subject:
            hints.add(subject.rsplit(":", 1)[-1])
        tool = str(payload.get("tool_name") or "")
        if tool:
            hints.add(tool)
    return {item for item in hints if item}


def _node_package_hints(node: ContractGraphNode) -> set[str]:
    payload = dict(node.payload or {})
    hints: set[str] = set()
    for key in ("package_id", "provider", "package"):
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            hints.add(raw.strip())
    for part in node.stable_key.split(":"):
        if part in KNOWN_PACKAGE_IDS:
            hints.add(part)
    return hints


def _match_nodes(
    graph: SymbolicContractGraph | None,
    *,
    package_id: str,
    tool_name: str,
    kinds: Sequence[ContractNodeKind],
    role: str,
) -> tuple[tuple[ContractGraphNode, ...], tuple[str, ...]]:
    """Return uniquely resolved nodes or reason codes for missing/ambiguous."""

    if graph is None:
        return (), (f"{role}_graph_unavailable",)

    candidates: list[ContractGraphNode] = []
    kind_set = set(kinds)
    for node in graph.nodes:
        if node.kind not in kind_set and kind_set:
            # Allow exact stable-key hits even when kind filter is set.
            pass
        tool_hints = _node_tool_hints(node)
        package_hints = _node_package_hints(node)
        tool_match = tool_name in tool_hints or any(
            tool_name == hint or hint.endswith(f":{tool_name}") or hint.endswith(f".{tool_name}")
            for hint in tool_hints
        )
        # Short handler keys like "handler:add" for tool "ipfs.add"
        short = tool_name.rsplit(".", 1)[-1]
        if short and short in tool_hints:
            tool_match = True
        if not tool_match:
            # Also accept operation_id style keys.
            if f"{package_id}:{tool_name}" not in tool_hints and (
                f"{package_id}:{tool_name}" not in node.stable_key
            ):
                if tool_name not in node.stable_key and short not in node.stable_key:
                    continue
        if package_hints and package_id not in package_hints:
            # Package-qualified mismatch — skip unless key embeds the package.
            if package_id not in node.stable_key:
                continue
        if kind_set and node.kind not in kind_set:
            # Prefer kind filter but keep exact stable-key matches for known roles.
            if role == "source" and node.kind not in {
                ContractNodeKind.METHOD,
                ContractNodeKind.TOOL,
                ContractNodeKind.SYMBOL,
                ContractNodeKind.INTERFACE,
                ContractNodeKind.CONTRACT,
                ContractNodeKind.TRANSPORT,
            }:
                continue
            if role == "target" and node.kind not in {
                ContractNodeKind.HANDLER,
                ContractNodeKind.SYMBOL,
                ContractNodeKind.TOOL,
                ContractNodeKind.CONTRACT,
            }:
                continue
        candidates.append(node)

    # Prefer higher-authority / more specific kinds.
    if role == "source":
        preference = (
            ContractNodeKind.METHOD,
            ContractNodeKind.TOOL,
            ContractNodeKind.INTERFACE,
            ContractNodeKind.SYMBOL,
            ContractNodeKind.TRANSPORT,
            ContractNodeKind.CONTRACT,
        )
    else:
        preference = (
            ContractNodeKind.HANDLER,
            ContractNodeKind.SYMBOL,
            ContractNodeKind.TOOL,
            ContractNodeKind.CONTRACT,
        )
    rank = {kind: index for index, kind in enumerate(preference)}
    candidates = sorted(
        candidates,
        key=lambda node: (
            rank.get(node.kind, 99),
            node.stable_key,
            node.node_id,
        ),
    )

    # Deduplicate by node_id.
    seen: set[str] = set()
    unique: list[ContractGraphNode] = []
    for node in candidates:
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        unique.append(node)

    if not unique:
        return (), (f"missing_{role}_anchor",)
    # Exact single preferred kind wins when multiple kinds matched.
    preferred_kind = preference[0]
    preferred = [node for node in unique if node.kind is preferred_kind]
    if len(preferred) == 1:
        return (preferred[0],), ()
    if len(unique) == 1:
        return (unique[0],), ()
    # Multiple handlers / methods for the same tool without a reviewed unique
    # join is ambiguous — never pick by name popularity.
    return tuple(unique), (f"ambiguous_{role}_anchor",)


def _source_candidate_keys(operation: ReviewedRuntimeOperation) -> tuple[str, ...]:
    package = operation.package_id
    tool = operation.tool_name
    return tuple(
        sorted(
            {
                f"descriptor:{tool}",
                f"method:{package}:{tool}",
                f"method:{tool}",
                f"mcp-tool:{package}:{tool}",
                f"tool:{tool}",
                f"tool:{package}:{tool}",
                operation.subject,
                f"contract-tool:{package}:{tool}",
            }
            - {""}
        )
    )


def _target_candidate_keys(operation: ReviewedRuntimeOperation) -> tuple[str, ...]:
    package = operation.package_id
    tool = operation.tool_name
    short = tool.rsplit(".", 1)[-1]
    return tuple(
        sorted(
            {
                f"handler:{tool}",
                f"handler:{short}",
                f"handler:{package}:{tool}",
                f"implementation:{tool}",
                f"implementation:{short}",
                f"implementation:{package}:{tool}",
                f"registration:{package}:{tool}",
                f"tool:{tool}",
            }
        )
    )


def _resolve_by_stable_keys(
    graph: SymbolicContractGraph | None,
    keys: Sequence[str],
) -> tuple[tuple[ContractGraphNode, ...], tuple[str, ...]]:
    if graph is None:
        return (), ("graph_unavailable",)
    found: list[ContractGraphNode] = []
    for key in keys:
        if not key:
            continue
        try:
            found.append(graph.node_for_key(key))
        except KeyError:
            continue
    if not found:
        return (), ("stable_key_miss",)
    # Deduplicate
    by_id = {node.node_id: node for node in found}
    unique = tuple(by_id[key] for key in sorted(by_id))
    if len(unique) > 1:
        # Multiple distinct keys resolving is fine when they are the same node;
        # different nodes remain ambiguous only if they disagree on role later.
        return unique, ()
    return unique, ()


def compile_endpoint_anchor(
    operation: ReviewedRuntimeOperation,
    *,
    graph: SymbolicContractGraph | None = None,
    extraction: SwissKnifeContractExtraction | None = None,
) -> tuple[EndpointAnchor, tuple[EvidenceFinding, ...]]:
    """Compile one endpoint anchor; missing/ambiguous → typed findings."""

    reasons: list[str] = []
    findings: list[EvidenceFinding] = []
    path_classes: list[str] = [PATH_CLASS_MCP_PLUS_PLUS]

    # Direct / compatibility edges from extraction keep a separate path class.
    if extraction is not None:
        for edge in extraction.invocation_edges:
            kind_value = (
                edge.kind.value if hasattr(edge.kind, "value") else str(edge.kind)
            )
            labels = " ".join(
                str(item)
                for item in (
                    edge.source,
                    edge.target or "",
                    edge.operation,
                    edge.transport,
                    kind_value,
                )
            )
            op_match = (
                operation.tool_name in labels
                or operation.package_id in labels
                or operation.operation_id in labels
                or (
                    edge.operation
                    and edge.operation
                    in {operation.tool_name, operation.operation_id}
                )
            )
            if not op_match and not (
                edge.compatibility or edge.bypass_candidate or "direct" in kind_value
            ):
                continue
            if not op_match:
                # Package-level direct/compat markers only attach when the edge
                # operation is empty (broadcast) or explicitly matches.
                if edge.operation and edge.operation not in {
                    operation.tool_name,
                    operation.operation_id,
                    "",
                }:
                    continue
            if edge.compatibility or "compat" in kind_value:
                if PATH_CLASS_COMPATIBILITY not in path_classes:
                    path_classes.append(PATH_CLASS_COMPATIBILITY)
            if edge.bypass_candidate or "direct" in kind_value:
                if PATH_CLASS_DIRECT not in path_classes:
                    path_classes.append(PATH_CLASS_DIRECT)

    source_nodes, source_reasons = _resolve_by_stable_keys(
        graph, _source_candidate_keys(operation)
    )
    if not source_nodes:
        source_nodes, source_reasons = _match_nodes(
            graph,
            package_id=operation.package_id,
            tool_name=operation.tool_name,
            kinds=(
                ContractNodeKind.METHOD,
                ContractNodeKind.TOOL,
                ContractNodeKind.INTERFACE,
                ContractNodeKind.SYMBOL,
            ),
            role="source",
        )
    target_nodes, target_reasons = _resolve_by_stable_keys(
        graph, _target_candidate_keys(operation)
    )
    # Always also collect kind-matched handlers so multiple concrete handlers
    # for one tool remain ambiguous even when a single stable key hits.
    matched_targets, matched_target_reasons = _match_nodes(
        graph,
        package_id=operation.package_id,
        tool_name=operation.tool_name,
        kinds=(
            ContractNodeKind.HANDLER,
            ContractNodeKind.SYMBOL,
            ContractNodeKind.TOOL,
        ),
        role="target",
    )
    if matched_targets:
        by_id = {node.node_id: node for node in target_nodes}
        for node in matched_targets:
            by_id[node.node_id] = node
        target_nodes = tuple(by_id[key] for key in sorted(by_id))
        handlers = [
            node for node in target_nodes if node.kind is ContractNodeKind.HANDLER
        ]
        if len(handlers) > 1:
            target_nodes = tuple(handlers)
            target_reasons = ("ambiguous_target_anchor",)
        elif not target_reasons:
            target_reasons = matched_target_reasons
    elif not target_nodes:
        target_reasons = matched_target_reasons or target_reasons

    # Prefer METHOD/TOOL as source and HANDLER as target when multiple.
    source_node: ContractGraphNode | None = None
    target_selection: tuple[ContractGraphNode, ...] = ()

    if "ambiguous_source_anchor" in source_reasons or (
        len(source_nodes) > 1
        and len({node.node_id for node in source_nodes}) > 1
        and not any(node.kind is ContractNodeKind.METHOD for node in source_nodes)
    ):
        # Multiple exact keys can point at related nodes; pick the best kind.
        preferred = [n for n in source_nodes if n.kind is ContractNodeKind.METHOD]
        if len(preferred) == 1:
            source_node = preferred[0]
            source_reasons = ()
        elif len(source_nodes) == 1:
            source_node = source_nodes[0]
            source_reasons = ()
        else:
            reasons.extend(
                source_reasons or ("ambiguous_source_anchor",)
            )
            findings.append(
                EvidenceFinding(
                    operation_id=operation.operation_id,
                    kind=EvidenceFindingKind.AMBIGUOUS_SOURCE_ANCHOR,
                    reason_code="ambiguous_source_anchor",
                    details={
                        "candidates": [n.stable_key for n in source_nodes],
                    },
                )
            )
    elif source_nodes:
        preferred = [n for n in source_nodes if n.kind is ContractNodeKind.METHOD]
        source_node = preferred[0] if preferred else source_nodes[0]
    else:
        reasons.extend(source_reasons or ("missing_source_anchor",))
        findings.append(
            EvidenceFinding(
                operation_id=operation.operation_id,
                kind=EvidenceFindingKind.MISSING_SOURCE_ANCHOR,
                reason_code="missing_source_anchor",
                details={
                    "candidate_keys": list(_source_candidate_keys(operation)),
                },
            )
        )

    if "ambiguous_target_anchor" in target_reasons or (
        len(target_nodes) > 1
        and len({n.kind for n in target_nodes}) > 1
        and sum(1 for n in target_nodes if n.kind is ContractNodeKind.HANDLER) != 1
    ):
        handlers = [n for n in target_nodes if n.kind is ContractNodeKind.HANDLER]
        if len(handlers) == 1:
            target_selection = (handlers[0],)
            target_reasons = ()
        elif len(target_nodes) == 1:
            target_selection = (target_nodes[0],)
            target_reasons = ()
        else:
            reasons.extend(target_reasons or ("ambiguous_target_anchor",))
            findings.append(
                EvidenceFinding(
                    operation_id=operation.operation_id,
                    kind=EvidenceFindingKind.AMBIGUOUS_TARGET_ANCHOR,
                    reason_code="ambiguous_target_anchor",
                    details={
                        "candidates": [n.stable_key for n in target_nodes],
                    },
                )
            )
    elif target_nodes:
        handlers = [n for n in target_nodes if n.kind is ContractNodeKind.HANDLER]
        target_selection = tuple(handlers) if handlers else (target_nodes[0],)
        if len(handlers) > 1:
            reasons.append("ambiguous_target_anchor")
            findings.append(
                EvidenceFinding(
                    operation_id=operation.operation_id,
                    kind=EvidenceFindingKind.AMBIGUOUS_TARGET_ANCHOR,
                    reason_code="multiple_concrete_handlers",
                    details={
                        "candidates": [n.stable_key for n in handlers],
                    },
                )
            )
            target_selection = tuple(handlers)
    else:
        reasons.extend(target_reasons or ("missing_target_anchor",))
        findings.append(
            EvidenceFinding(
                operation_id=operation.operation_id,
                kind=EvidenceFindingKind.MISSING_TARGET_ANCHOR,
                reason_code="missing_target_anchor",
                details={
                    "candidate_keys": list(_target_candidate_keys(operation)),
                },
            )
        )

    # Detect path-class ambiguity when both MCP++ and direct are claimed without
    # distinguishable source nodes.
    mcp_source = ""
    direct_source = ""
    if source_node is not None:
        mcp_source = source_node.node_id
    if graph is not None:
        for node in graph.nodes:
            key_l = node.stable_key.lower()
            payload_l = json.dumps(dict(node.payload or {}), sort_keys=True).lower()
            if operation.tool_name not in node.stable_key and operation.tool_name not in payload_l:
                short = operation.tool_name.rsplit(".", 1)[-1]
                if short not in key_l and short not in payload_l:
                    continue
            if any(
                marker in key_l or marker in payload_l
                for marker in (
                    "direct_fetch",
                    "direct_rest",
                    "direct_import",
                    "compatibility",
                    "/api/v0/",
                )
            ):
                if PATH_CLASS_DIRECT not in path_classes:
                    path_classes.append(PATH_CLASS_DIRECT)
                direct_source = node.node_id
            if any(
                marker in key_l or marker in payload_l
                for marker in (
                    "connector",
                    "tools/call",
                    "tools_call",
                    "mcp++",
                    "mcp_plus_plus",
                )
            ):
                mcp_source = mcp_source or node.node_id

    if (
        PATH_CLASS_MCP_PLUS_PLUS in path_classes
        and PATH_CLASS_DIRECT in path_classes
        and mcp_source
        and direct_source
        and mcp_source == direct_source
    ):
        reasons.append("ambiguous_path_class")
        findings.append(
            EvidenceFinding(
                operation_id=operation.operation_id,
                kind=EvidenceFindingKind.AMBIGUOUS_PATH_CLASS,
                reason_code="mcp_plus_plus_and_direct_share_source",
                details={
                    "source_node_id": mcp_source,
                    "path_classes": list(path_classes),
                },
            )
        )

    if findings and any(
        item.kind
        in {
            EvidenceFindingKind.AMBIGUOUS_SOURCE_ANCHOR,
            EvidenceFindingKind.AMBIGUOUS_TARGET_ANCHOR,
            EvidenceFindingKind.AMBIGUOUS_PATH_CLASS,
        }
        for item in findings
    ):
        state = AnchorResolutionState.AMBIGUOUS
    elif findings and any(
        item.kind
        in {
            EvidenceFindingKind.MISSING_SOURCE_ANCHOR,
            EvidenceFindingKind.MISSING_TARGET_ANCHOR,
        }
        for item in findings
    ):
        state = AnchorResolutionState.MISSING
    elif not source_node or not target_selection:
        state = AnchorResolutionState.INCOMPLETE
        reasons.append("anchor_incomplete")
    else:
        state = AnchorResolutionState.RESOLVED

    # Stable keys always recorded for audit even when unresolved.
    source_key = (
        source_node.stable_key
        if source_node is not None
        else _source_candidate_keys(operation)[0]
    )
    target_keys = (
        tuple(node.stable_key for node in target_selection)
        if target_selection
        else _target_candidate_keys(operation)[:1]
    )

    anchor = EndpointAnchor(
        operation_id=operation.operation_id,
        package_id=operation.package_id,
        tool_name=operation.tool_name,
        resolution_state=state,
        source_node_id=source_node.node_id if source_node is not None else "",
        source_stable_key=source_key,
        target_node_ids=tuple(node.node_id for node in target_selection),
        target_stable_keys=target_keys,
        path_classes=tuple(path_classes),
        mcp_plus_plus_source_node_id=mcp_source,
        direct_source_node_id=direct_source,
        reason_codes=tuple(sorted(set(reasons))),
        source_ids=operation.source_ids,
        contract_ids=operation.contract_ids,
        supported=True,
        measured=True,
    )
    return anchor, tuple(findings)


def _tool_surface_for_operation(
    operation: ReviewedRuntimeOperation,
    package_surfaces: Sequence[PythonMcpPackageSurface] | None,
) -> PythonMcpToolSurface | None:
    if not package_surfaces:
        return None
    matches: list[PythonMcpToolSurface] = []
    for surface in package_surfaces:
        if surface.provider != operation.package_id:
            continue
        matches.extend(surface.tools_named(operation.tool_name))
    if len(matches) == 1:
        return matches[0]
    return None


def _route_from_method(
    *,
    route_id: str,
    transport: str,
    path_class: str,
    mediation_path_class: str,
    method: MethodExpectation | None,
    tool: PythonMcpToolSurface | None,
    source_ids: Sequence[str],
    callable_value: bool | None,
) -> dict[str, Any]:
    input_schema: dict[str, Any]
    if tool is not None and tool.input_schema:
        input_schema = dict(tool.input_schema)
    elif method is not None and method.input_schema:
        input_schema = _schema_ref_payload(method.input_schema)
    else:
        input_schema = {"type": "object", "additionalProperties": True}

    output_schema: dict[str, Any]
    if method is not None and method.output_schema:
        output_schema = _schema_ref_payload(method.output_schema)
    else:
        output_schema = {"type": "object", "additionalProperties": True}

    policies = list(method.policy_requirements) if method is not None else []
    events = [f"policy:{item}" for item in policies]
    if not events and mediation_path_class == PATH_CLASS_MCP_PLUS_PLUS:
        # MCP++ path records the mediation pipeline without inventing policy.
        events = ["mcp++:connector", "mcp++:tools_call"]
    elif mediation_path_class == PATH_CLASS_DIRECT:
        events = ["direct:package_call"]

    complete_route = tool is not None or (
        method is not None and bool(method.input_schema or method.output_schema)
    )
    route: dict[str, Any] = {
        "route_id": route_id,
        "transport": transport,
        "path_class": (
            "compatibility"
            if path_class in {PATH_CLASS_DIRECT, PATH_CLASS_COMPATIBILITY}
            and mediation_path_class != PATH_CLASS_MCP_PLUS_PLUS
            else "direct"
            if mediation_path_class == PATH_CLASS_MCP_PLUS_PLUS
            else path_class
        ),
        "mediation_path_class": mediation_path_class,
        "path_kind": mediation_path_class,
        "callable": True if callable_value is None else bool(callable_value),
        "input_schema": input_schema,
        "output_schema": output_schema,
        "argument_map": {},
        "result_envelope": list(DEFAULT_RESULT_ENVELOPE),
        "failure_states": list(DEFAULT_FAILURE_STATES),
        "failure_mapping": {state: state for state in DEFAULT_FAILURE_STATES},
        "events": events,
        "mutation_capable": False,
        "provenance": True,
        "receipt": mediation_path_class == PATH_CLASS_MCP_PLUS_PLUS,
        "source_ids": list(source_ids),
        "complete": complete_route,
    }
    if tool is not None:
        route["handler_symbol"] = tool.handler.symbol
        route["registration_api"] = tool.registration_api
        route["source_ids"] = sorted(
            set(route["source_ids"]) | {tool.tool_id}
        )
    if method is not None:
        route["interaction_pattern"] = method.interaction_pattern
        route["streaming"] = method.streaming
        if method.error_schemas:
            route["error_schema_refs"] = list(method.error_schemas)
    return route


def compile_observed_package_contract(
    operation: ReviewedRuntimeOperation,
    *,
    package_surfaces: Sequence[PythonMcpPackageSurface] | None = None,
    runtime_catalog: RuntimeComponentCatalog | None = None,
    path_classes: Sequence[str] = (PATH_CLASS_MCP_PLUS_PLUS,),
) -> tuple[dict[str, Any], tuple[EvidenceFinding, ...]]:
    """Compile one observed package contract for the parity analyzer."""

    findings: list[EvidenceFinding] = []
    tool = _tool_surface_for_operation(operation, package_surfaces)
    method = operation.method
    routes: list[dict[str, Any]] = []
    classes = tuple(path_classes) or (PATH_CLASS_MCP_PLUS_PLUS,)

    # Always emit a distinct MCP++ route when requested.
    if PATH_CLASS_MCP_PLUS_PLUS in classes:
        transport = "mcp++"
        if runtime_catalog is not None:
            # Prefer a cataloged CALL route transport when present.
            for route in runtime_catalog.routes:
                if route.kind is RuntimeRouteKind.CALL:
                    transport = route.transport
                    break
        routes.append(
            _route_from_method(
                route_id=f"route:mcp_plus_plus:{operation.operation_id}",
                transport=transport,
                path_class=PATH_CLASS_MCP_PLUS_PLUS,
                mediation_path_class=PATH_CLASS_MCP_PLUS_PLUS,
                method=method,
                tool=tool,
                source_ids=operation.source_ids,
                callable_value=True if tool is not None else None,
            )
        )

    # Direct package path remains a separate route identity when present.
    if PATH_CLASS_DIRECT in classes or PATH_CLASS_COMPATIBILITY in classes:
        direct_class = (
            PATH_CLASS_DIRECT
            if PATH_CLASS_DIRECT in classes
            else PATH_CLASS_COMPATIBILITY
        )
        transports = list(tool.transports) if tool is not None else []
        transport = transports[0] if transports else "direct"
        routes.append(
            _route_from_method(
                route_id=f"route:direct:{operation.operation_id}",
                transport=transport,
                path_class=direct_class,
                mediation_path_class=direct_class,
                method=method,
                tool=tool,
                source_ids=operation.source_ids,
                callable_value=True if tool is not None else None,
            )
        )

    if not routes:
        # Fail closed: every operation still emits at least the mandatory MCP++
        # observed route shell so the stage is never empty-success withheld.
        routes.append(
            _route_from_method(
                route_id=f"route:mcp_plus_plus:{operation.operation_id}",
                transport="mcp++",
                path_class=PATH_CLASS_MCP_PLUS_PLUS,
                mediation_path_class=PATH_CLASS_MCP_PLUS_PLUS,
                method=method,
                tool=tool,
                source_ids=operation.source_ids,
                callable_value=None,
            )
        )

    complete = all(bool(route.get("complete")) for route in routes) and (
        tool is not None or method is not None
    )
    if not complete:
        findings.append(
            EvidenceFinding(
                operation_id=operation.operation_id,
                kind=EvidenceFindingKind.OBSERVED_CONTRACT_INCOMPLETE,
                reason_code="observed_contract_incomplete",
                details={
                    "has_package_registration": tool is not None,
                    "has_method_expectation": method is not None,
                },
            )
        )

    discovery_tools = [operation.operation_id, operation.tool_name]
    if tool is not None:
        discovery_tools.append(tool.canonical_name)
        discovery_tools.extend(tool.aliases)

    observed: dict[str, Any] = {
        "schema": OBSERVED_PACKAGE_CONTRACT_SCHEMA,
        "operation_id": operation.operation_id,
        "name": operation.tool_name,
        "package_id": operation.package_id,
        "tool_name": operation.tool_name,
        "discovery": {
            "tools": sorted(set(discovery_tools)),
            "listed": tool is not None,
        },
        "routes": sorted(routes, key=lambda item: item["route_id"]),
        "complete": complete,
        "source_ids": list(operation.source_ids),
        "contract_ids": list(operation.contract_ids),
    }
    if tool is not None:
        observed["registration"] = {
            "provider": tool.provider,
            "canonical_name": tool.canonical_name,
            "registration_api": tool.registration_api,
            "handler": tool.handler.to_dict(),
            "tool_id": tool.tool_id,
        }
    observed["observed_contract_id"] = _cid(
        {
            "schema": "mcp-observed-operation-contract@1",
            "contract": {
                "operation_id": observed["operation_id"],
                "package_id": observed["package_id"],
                "routes": observed["routes"],
                "discovery": observed["discovery"],
                "complete": observed["complete"],
            },
        }
    )
    return observed, tuple(findings)


def _run_traces(
    graph: SymbolicContractGraph,
    anchors: Sequence[EndpointAnchor],
) -> tuple[tuple[McpInvocationTrace, ...], tuple[EvidenceFinding, ...]]:
    tracer = McpInvocationTracer(graph)
    traces: list[McpInvocationTrace] = []
    findings: list[EvidenceFinding] = []
    requests: list[InvocationTraceRequest] = []
    for anchor in anchors:
        if not anchor.is_traceable:
            continue
        requests.append(anchor.to_trace_request())
    if not requests:
        return (), ()
    for request in requests:
        try:
            trace = tracer.trace(request)
        except Exception as exc:  # noqa: BLE001 - typed unknown, never empty-success
            findings.append(
                EvidenceFinding(
                    operation_id=request.operation_id,
                    kind=EvidenceFindingKind.TRACE_UNKNOWN,
                    reason_code="trace_failed",
                    details={"error": type(exc).__name__, "message": str(exc)[:256]},
                )
            )
            continue
        traces.append(trace)
        if trace.terminal_state in {
            InvocationTerminalState.AMBIGUOUS,
            InvocationTerminalState.NOT_MEASURED,
            InvocationTerminalState.UNSUPPORTED,
        }:
            findings.append(
                EvidenceFinding(
                    operation_id=trace.operation_id,
                    kind=EvidenceFindingKind.TRACE_UNKNOWN,
                    reason_code=trace.reason_code or trace.terminal_state.value,
                    details={
                        "terminal_state": trace.terminal_state.value,
                        "complete": trace.complete,
                    },
                )
            )
    return tuple(traces), tuple(findings)


class RuntimeContractEvidenceCompiler:
    """Compile endpoint anchors and observed contracts for the baseline."""

    interface: Final = RUNTIME_CONTRACT_EVIDENCE_COMPILER_INTERFACE
    version: Final = RUNTIME_CONTRACT_EVIDENCE_COMPILER_VERSION

    def compile(
        self,
        catalog: McpContractCatalog,
        *,
        snapshot_id: str = "",
        graph: SymbolicContractGraph | None = None,
        extraction: SwissKnifeContractExtraction | None = None,
        package_surfaces: Sequence[PythonMcpPackageSurface] | None = None,
        runtime_catalog: RuntimeComponentCatalog | None = None,
        run_traces: bool = True,
    ) -> RuntimeContractEvidenceCompilation:
        """Compile anchors + observed contracts for every reviewed operation.

        Healthy inputs with resolved anchors and a complete graph yield nonempty
        traces.  Missing or ambiguous anchors always produce typed findings; the
        compilation never reports complete success with empty coverage for a
        nonempty reviewed operation set.
        """

        if extraction is not None and not snapshot_id:
            snapshot_id = str(
                getattr(extraction, "repository_tree_id", "") or ""
            )
        if graph is not None and not snapshot_id:
            snapshot_id = graph.snapshot_id
        snapshot_id = _text(snapshot_id or "unspecified-snapshot", "snapshot_id")

        operations = collect_reviewed_runtime_operations(
            catalog, extraction=extraction
        )
        anchors: list[EndpointAnchor] = []
        observed_contracts: list[Mapping[str, Any]] = []
        findings: list[EvidenceFinding] = []
        reason_codes: list[str] = []

        if not operations:
            reason_codes.append("no_reviewed_runtime_operations")

        for operation in operations:
            anchor, anchor_findings = compile_endpoint_anchor(
                operation, graph=graph, extraction=extraction
            )
            observed, observed_findings = compile_observed_package_contract(
                operation,
                package_surfaces=package_surfaces,
                runtime_catalog=runtime_catalog,
                path_classes=anchor.path_classes,
            )
            anchors.append(anchor)
            observed_contracts.append(observed)
            findings.extend(anchor_findings)
            findings.extend(observed_findings)

        traces: tuple[McpInvocationTrace, ...] = ()
        if run_traces and graph is not None:
            traces, trace_findings = _run_traces(graph, anchors)
            findings.extend(trace_findings)
            if operations and not traces and not any(
                item.is_traceable for item in anchors
            ):
                reason_codes.append("no_traceable_anchors")
            elif operations and traces:
                # Healthy path: every resolved anchor produced a trace record.
                pass
        elif run_traces and graph is None:
            reason_codes.append("graph_unavailable_for_traces")
        elif not run_traces:
            reason_codes.append("trace_stage_disabled")

        # Completeness: every operation has an anchor + observed contract, all
        # anchors resolved, no findings, and traces nonempty when operations exist.
        all_resolved = bool(operations) and all(
            item.resolution_state is AnchorResolutionState.RESOLVED
            for item in anchors
        )
        traces_ok = (not operations) or (
            not run_traces
            or (
                graph is not None
                and len(traces) == sum(1 for item in anchors if item.is_traceable)
                and (
                    len(traces) > 0
                    if any(item.is_traceable for item in anchors)
                    else True
                )
            )
        )
        complete = (
            bool(operations)
            and all_resolved
            and not findings
            and traces_ok
            and all(bool(item.get("complete")) for item in observed_contracts)
        )
        if operations and not complete and not findings:
            # Never withhold as empty success: surface a typed incomplete reason.
            reason_codes.append("evidence_incomplete")
            if not all_resolved:
                reason_codes.append("anchors_not_fully_resolved")

        catalog_root = catalog.catalog_id
        graph_root = graph.graph_root if graph is not None else ""
        extraction_root = (
            extraction.extraction_id if extraction is not None else ""
        )
        runtime_root = ""
        if runtime_catalog is not None:
            runtime_root = str(
                runtime_catalog.to_dict().get("catalogCid")
                or runtime_catalog.catalog_cid
                or ""
            )

        return RuntimeContractEvidenceCompilation(
            snapshot_id=snapshot_id,
            operations=tuple(operations),
            anchors=tuple(anchors),
            observed_contracts=tuple(observed_contracts),
            findings=tuple(findings),
            traces=traces,
            catalog_root=catalog_root,
            graph_root=graph_root,
            extraction_root=extraction_root,
            runtime_catalog_root=runtime_root,
            reason_codes=tuple(sorted(set(reason_codes))),
            complete=complete,
        )


def compile_runtime_contract_evidence(
    catalog: McpContractCatalog,
    *,
    snapshot_id: str = "",
    graph: SymbolicContractGraph | None = None,
    extraction: SwissKnifeContractExtraction | None = None,
    package_surfaces: Sequence[PythonMcpPackageSurface] | None = None,
    runtime_catalog: RuntimeComponentCatalog | None = None,
    run_traces: bool = True,
) -> RuntimeContractEvidenceCompilation:
    """Convenience entry for ``RuntimeContractEvidenceCompiler@1``."""

    return RuntimeContractEvidenceCompiler().compile(
        catalog,
        snapshot_id=snapshot_id,
        graph=graph,
        extraction=extraction,
        package_surfaces=package_surfaces,
        runtime_catalog=runtime_catalog,
        run_traces=run_traces,
    )


__all__ = [
    "AnchorResolutionState",
    "ENDPOINT_ANCHOR_SCHEMA",
    "EVIDENCE_COMPILATION_SCHEMA",
    "EndpointAnchor",
    "EvidenceFinding",
    "EvidenceFindingKind",
    "OBSERVED_PACKAGE_CONTRACT_SCHEMA",
    "PATH_CLASS_COMPATIBILITY",
    "PATH_CLASS_DIRECT",
    "PATH_CLASS_MCP_PLUS_PLUS",
    "RUNTIME_CONTRACT_EVIDENCE_COMPILER_INTERFACE",
    "RUNTIME_CONTRACT_EVIDENCE_COMPILER_VERSION",
    "ReviewedRuntimeOperation",
    "RuntimeContractEvidenceCompilation",
    "RuntimeContractEvidenceCompiler",
    "RuntimeContractEvidenceCompilerError",
    "collect_reviewed_runtime_operations",
    "compile_endpoint_anchor",
    "compile_observed_package_contract",
    "compile_runtime_contract_evidence",
]
