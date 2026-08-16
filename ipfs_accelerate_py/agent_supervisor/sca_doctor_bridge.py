"""SCA formal-first bridge: map contract findings to deterministic doctor outcomes.

This module is the SCA selection/repair surface for ENABLE-DOCTOR. It does not
mint kernel proofs and never calls a model. A finding either yields a typed
``transform_receipt`` (analytical repair) or an ``analytical_abstention``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Final

from .proof.formal_verification_contracts import canonical_json_bytes, content_identity

SCA_DOCTOR_BRIDGE_INTERFACE: Final[str] = "ScaDoctorBridge@1"
TRANSFORM_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/sca-doctor-transform-receipt@1"
)
ABSTENTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/sca-doctor-analytical-abstention@1"
)
DCR_DOCTOR_FINDING_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/dcr-doctor-finding@1"
_DCR051_RELATION_ORDER: Final[dict[str, int]] = {
    "expects_descriptor": 10,
    "binds_orb_idl": 20,
    "defines_method_schema": 30,
    "binds_mediator_route": 40,
    "routes_to_observed_dispatcher": 50,
    "dispatches_to_handler": 60,
    "performs_effect": 70,
    "emits_receipt_runtime_identity": 80,
}
_DCR051_MISMATCH_CLASSES: Final[frozenset[str]] = frozenset(
    {"protocol", "schema", "authority", "liveness", "identity", "mediation", "implementation"}
)
_DCR051_NONPASSING: Final[frozenset[str]] = frozenset({"missing", "failed"})

# Finding kinds that map to deterministic analytical transforms.
_TRANSFORMABLE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "parity_refuted",
        "refuted",
        "mismatch",
        "policy_before_effect",
        "direct_dispatch",
        "missing_guard",
        "package_surface_missing",
        "endpoint_anchor_missing",
        "invocation_trace_gap",
        "contract_violation",
        # Live SCA-180 baseline kinds: incomplete observation is a deterministic
        # instrumentation/surface gap, not free-form prose.
        "observed_contract_incomplete",
        "package_surface_incomplete",
        "mcp_surface_incomplete",
        # Multi-match anchors after registration repair still admit a
        # deterministic disambiguation transform (prefer mcp_server surfaces).
        "ambiguous_source_anchor",
        "ambiguous_target_anchor",
        "ambiguous_path_class",
    }
)


class DoctorBridgeError(ValueError):
    """Fail-closed doctor-bridge error."""


class DoctorDisposition(StrEnum):
    TRANSFORM_RECEIPT = "transform_receipt"
    ANALYTICAL_ABSTENTION = "analytical_abstention"


@dataclass(frozen=True)
class ScaContractFinding:
    """Minimal finding view used by the doctor bridge."""

    finding_id: str
    kind: str
    snapshot_id: str = ""
    contract_id: str = ""
    path: str = ""
    symbol: str = ""
    reason_code: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.finding_id or "").strip():
            raise DoctorBridgeError("finding_id is required")
        if not str(self.kind or "").strip():
            raise DoctorBridgeError("kind is required")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ScaContractFinding:
        if not isinstance(value, Mapping):
            raise DoctorBridgeError("finding must be a mapping")
        return cls(
            finding_id=str(
                value.get("finding_id") or value.get("id") or value.get("findingId") or ""
            ),
            kind=str(value.get("kind") or value.get("reason_code") or value.get("state") or ""),
            snapshot_id=str(value.get("snapshot_id") or value.get("snapshot_root") or ""),
            contract_id=str(value.get("contract_id") or value.get("operation_id") or ""),
            path=str(
                (value.get("path") or "")
                or (
                    (value.get("affected_paths") or [""])[0]
                    if isinstance(value.get("affected_paths"), (list, tuple))
                    and value.get("affected_paths")
                    else ""
                )
            ),
            symbol=str(
                value.get("symbol")
                or (
                    (value.get("affected_symbols") or [""])[0]
                    if isinstance(value.get("affected_symbols"), (list, tuple))
                    and value.get("affected_symbols")
                    else ""
                )
            ),
            reason_code=str(value.get("reason_code") or value.get("kind") or ""),
            evidence=dict(value.get("evidence") or {}),
        )


@dataclass(frozen=True)
class TransformReceipt:
    """Analytical transform receipt (zero model calls)."""

    schema: str
    disposition: str
    finding_id: str
    operator: str
    snapshot_id: str = ""
    contract_id: str = ""
    path: str = ""
    symbol: str = ""
    model_call_count: int = 0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalyticalAbstention:
    """Typed abstention when no deterministic transform is admitted."""

    schema: str
    disposition: str
    finding_id: str
    reason_code: str
    snapshot_id: str = ""
    model_call_count: int = 0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _operator_for_kind(kind: str) -> str:
    mapping = {
        "parity_refuted": "restore_parity_guard",
        "refuted": "restore_parity_guard",
        "mismatch": "align_contract_surface",
        "policy_before_effect": "insert_policy_gate",
        "direct_dispatch": "route_mediated_dispatch",
        "missing_guard": "insert_policy_gate",
        "package_surface_missing": "publish_package_surface",
        "endpoint_anchor_missing": "bind_endpoint_anchor",
        "invocation_trace_gap": "complete_invocation_trace",
        "contract_violation": "align_contract_surface",
        "observed_contract_incomplete": "complete_observed_contract_surface",
        "package_surface_incomplete": "complete_observed_contract_surface",
        "mcp_surface_incomplete": "complete_observed_contract_surface",
        "ambiguous_source_anchor": "disambiguate_source_anchor",
        "ambiguous_target_anchor": "disambiguate_target_anchor",
        "ambiguous_path_class": "disambiguate_path_class",
    }
    return mapping.get(kind.lower().replace("-", "_"), "analytical_transform")


def map_finding(
    finding: ScaContractFinding | Mapping[str, Any],
) -> TransformReceipt | AnalyticalAbstention:
    """Map one finding to transform_receipt or analytical_abstention."""

    if not isinstance(finding, ScaContractFinding):
        finding = ScaContractFinding.from_mapping(finding)

    kind = finding.kind.lower().replace("-", "_").replace(" ", "_")
    reason = finding.reason_code.lower().replace("-", "_")
    token = kind or reason

    if token in _TRANSFORMABLE_KINDS or any(key in token for key in _TRANSFORMABLE_KINDS):
        return TransformReceipt(
            schema=TRANSFORM_RECEIPT_SCHEMA,
            disposition=DoctorDisposition.TRANSFORM_RECEIPT.value,
            finding_id=finding.finding_id,
            operator=_operator_for_kind(token),
            snapshot_id=finding.snapshot_id,
            contract_id=finding.contract_id,
            path=finding.path,
            symbol=finding.symbol,
            model_call_count=0,
            notes="deterministic analytical transform; requires re-index/re-prove",
        )

    return AnalyticalAbstention(
        schema=ABSTENTION_SCHEMA,
        disposition=DoctorDisposition.ANALYTICAL_ABSTENTION.value,
        finding_id=finding.finding_id,
        reason_code=finding.reason_code or finding.kind or "no_deterministic_transform",
        snapshot_id=finding.snapshot_id,
        model_call_count=0,
        notes="no deterministic transform admitted; residual may need RPR packet",
    )


def map_findings(
    findings: list[ScaContractFinding | Mapping[str, Any]] | tuple,
) -> list[dict[str, Any]]:
    """Map many findings; returns serializable receipts/abstentions."""

    if not isinstance(findings, (list, tuple)):
        raise DoctorBridgeError("findings must be a list or tuple")
    return [map_finding(item).to_dict() for item in findings]


def diagnose_finding_with_ir(
    finding: ScaContractFinding | Mapping[str, Any],
    *,
    apply_ir: bool = True,
    ir_families: tuple[str, ...] | None = None,
    domain: str = "sca",
) -> dict[str, Any]:
    """Doctor path: map finding + apply logic to intermediate representations.

    Uses the **general** IR logic consumers (not SCA-only machinery). The SCA
    disposition vocabulary is layered on top; domain defaults to ``sca`` for
    this bridge but can be any supervisor domain.
    """
    if not isinstance(finding, ScaContractFinding):
        finding = ScaContractFinding.from_mapping(finding)
    disposition = map_finding(finding)
    out: dict[str, Any] = {
        "schema": "ipfs_accelerate_py/agent-supervisor/sca-doctor-ir-diagnosis@1",
        "interface": SCA_DOCTOR_BRIDGE_INTERFACE,
        "disposition": disposition.to_dict(),
        "model_call_count": 0,
        "domain": domain,
        "ir_logic_apply": {},
    }
    if not apply_ir:
        return out
    try:
        from .planning.ir_logic_consumers import diagnose_with_ir_logic
        from .proof.ir_logic_application import (
            DEFAULT_APPLY_FAMILIES,
            IrLogicApplyPolicy,
        )

        general = diagnose_with_ir_logic(
            {
                "finding_id": finding.finding_id,
                "kind": finding.kind,
                "contract_id": finding.contract_id,
                "path": finding.path,
                "symbol": finding.symbol,
                "reason_code": finding.reason_code,
                "snapshot_id": finding.snapshot_id,
            },
            disposition=disposition.to_dict(),
            policy=IrLogicApplyPolicy(
                families=tuple(ir_families or DEFAULT_APPLY_FAMILIES),
                evaluate_security=True,
                include_plan_admission=False,
            ),
            domain=domain,
        )
        out["ir_logic_apply"] = general.get("ir_logic") or {}
        out["notes"] = (
            "Doctor disposition is analytical-only; general IR apply attaches "
            "intermediate constraint/AST/KG/vector context without execution authority."
        )
    except Exception as exc:  # noqa: BLE001
        out["ir_logic_apply"] = {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return out


def diagnose_findings_with_ir(
    findings: list[ScaContractFinding | Mapping[str, Any]] | tuple,
    *,
    apply_ir: bool = True,
) -> list[dict[str, Any]]:
    if not isinstance(findings, (list, tuple)):
        raise DoctorBridgeError("findings must be a list or tuple")
    return [diagnose_finding_with_ir(item, apply_ir=apply_ir) for item in findings]


# ---------------------------------------------------------------------------
# DCR-051 earliest-edge diagnosis (strictly non-authoritative)
# ---------------------------------------------------------------------------


class DoctorFindingDisposition(StrEnum):
    FINDING = "finding"
    ABSTAINED = "abstained"
    DEFERRED = "deferred"
    REJECTED = "rejected"


@dataclass(frozen=True)
class DoctorSourceSlice:
    """Exact static source evidence for one observed DCR-021 edge."""

    edge_id: str
    root_owner: str
    relative_path: str
    source_bytes: bytes
    source_sha256: str
    span_start: int
    span_end: int
    span_sha256: str
    authority: str = "observed_source"

    def __post_init__(self) -> None:
        if (
            not all(
                isinstance(value, str) and value
                for value in (self.edge_id, self.root_owner, self.relative_path)
            )
            or self.relative_path.startswith("/")
            or ".." in self.relative_path.split("/")
            or self.root_owner in {"fixture", "generated", "expected"}
            or self.authority != "observed_source"
            or not self.source_bytes
            or not 0 <= self.span_start < self.span_end <= len(self.source_bytes)
        ):
            raise DoctorBridgeError("DCR-051 source slice is not exact observed source evidence")
        source_digest = "sha256:" + hashlib.sha256(self.source_bytes).hexdigest()
        span_digest = (
            "sha256:"
            + hashlib.sha256(self.source_bytes[self.span_start : self.span_end]).hexdigest()
        )
        if self.source_sha256 != source_digest or self.span_sha256 != span_digest:
            raise DoctorBridgeError("DCR-051 source slice digest or span is stale")

    @property
    def slice_cid(self) -> str:
        return content_identity(
            {
                "edge_id": self.edge_id,
                "root_owner": self.root_owner,
                "relative_path": self.relative_path,
                "source_sha256": self.source_sha256,
                "span_start": self.span_start,
                "span_end": self.span_end,
                "span_sha256": self.span_sha256,
                "authority": self.authority,
            }
        )


@dataclass(frozen=True)
class DoctorFinding:
    disposition: DoctorFindingDisposition
    reason_code: str = ""
    finding_id: str = ""
    edge_id: str = ""
    semantic_key: Mapping[str, str] = field(default_factory=dict)
    mismatch_class: str = ""
    forest_id: str = ""
    graph_cid: str = ""
    epoch_cid: str = ""
    findings_cid: str = ""
    evidence_cids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": DCR_DOCTOR_FINDING_SCHEMA,
            "interface": SCA_DOCTOR_BRIDGE_INTERFACE,
            "authoritative": False,
            "mutation_authorized": False,
            "planning_authorized": False,
            "completion_authorized": False,
            "disposition": self.disposition.value,
            "reason_code": self.reason_code,
            "finding_id": self.finding_id,
            "edge_id": self.edge_id,
            "semantic_key": dict(sorted(self.semantic_key.items())),
            "mismatch_class": self.mismatch_class,
            "forest_id": self.forest_id,
            "graph_cid": self.graph_cid,
            "epoch_cid": self.epoch_cid,
            "findings_cid": self.findings_cid,
            "evidence_cids": list(self.evidence_cids),
            "model_call_count": 0,
            "provider_call_count": 0,
            "network_call_count": 0,
        }
        return {**body, "doctor_finding_cid": content_identity(body)}


def _dcr051_abstain(disposition: DoctorFindingDisposition, reason: str) -> DoctorFinding:
    return DoctorFinding(disposition=disposition, reason_code=reason)


def _canonical_graph(graph: Any) -> Mapping[str, Any] | None:
    if not isinstance(graph, Mapping):
        return None
    value = dict(graph)
    graph_cid = value.pop("graph_cid", "")
    encoded = value.pop("canonical_bytes", "")
    if (
        value.get("schema") != "ipfs_accelerate_py/agent-supervisor/mcp-contract-graph@1"
        or value.get("interface") != "McpContractGraph@1"
        or value.get("authoritative") is not False
        or value.get("blockers") != []
        or not isinstance(encoded, str)
        or encoded != canonical_json_bytes(value).decode("utf-8")
        or graph_cid != content_identity(value)
    ):
        return None
    nodes, edges = value.get("nodes"), value.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return None
    node_ids = {item.get("id") for item in nodes if isinstance(item, Mapping)}
    if len(node_ids) != len(nodes) or not all(isinstance(item, str) and item for item in node_ids):
        return None
    if any(
        not isinstance(item, Mapping)
        or item.get("relation") not in _DCR051_RELATION_ORDER
        or item.get("source") not in node_ids
        or item.get("target") not in node_ids
        for item in edges
    ):
        return None
    return graph


def _canonical_mismatch_report(report: Any, graph_cid: str) -> Mapping[str, Any] | None:
    if not isinstance(report, Mapping):
        return None
    value = dict(report)
    findings_cid = value.pop("findings_cid", "")
    if (
        value.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/mcp-contract-mismatch-analysis@1"
        or value.get("interface") != "McpContractMismatchAnalysis@1"
        or value.get("authoritative") is not False
        or value.get("graph_cid") != graph_cid
        or value.get("dcr023_current_valid") is not True
        or value.get("production_readiness") != "nonpassing"
        or findings_cid != content_identity(value)
        or not isinstance(value.get("findings"), list)
        or not isinstance(value.get("semantic_roots"), Mapping)
        or not isinstance(value.get("snapshot_roots"), Mapping)
    ):
        return None
    return report


def diagnose_earliest_dcr_edge(
    *,
    composition_result: Any,
    composition_binding: Mapping[str, Any],
    graph: Mapping[str, Any],
    mismatch_report: Mapping[str, Any],
    source_slices: tuple[DoctorSourceSlice, ...],
    reconstruction: Any | None = None,
) -> DoctorFinding:
    """Select one earliest observed failure; never select an operator or repair.

    The composition is re-inspected, so a manually assembled DCR-050 result or
    synthetic identity record cannot cross this non-authoritative boundary.
    """

    from .control.default_doctor_factory import (
        DcrDoctorCompositionResult,
        inspect_dcr_doctor_composition,
    )

    current_composition = inspect_dcr_doctor_composition(composition_binding)
    if (
        not isinstance(composition_result, DcrDoctorCompositionResult)
        or composition_result.to_dict() != current_composition.to_dict()
        or current_composition.disposition != "integration_pending"
        or not current_composition.binding_complete
    ):
        return _dcr051_abstain(DoctorFindingDisposition.DEFERRED, "dcr050_binding_not_current")
    if "transitional_self_attested_bindings_non_live" in current_composition.reason_codes:
        return _dcr051_abstain(
            DoctorFindingDisposition.DEFERRED, "dcr050_transitional_binding_not_diagnostic_evidence"
        )
    canonical_graph = _canonical_graph(graph)
    if canonical_graph is None:
        return _dcr051_abstain(
            DoctorFindingDisposition.REJECTED, "dcr021_graph_not_canonical_blocker_free"
        )
    graph_cid = str(canonical_graph["graph_cid"])
    report = _canonical_mismatch_report(mismatch_report, graph_cid)
    if report is None:
        return _dcr051_abstain(DoctorFindingDisposition.DEFERRED, "dcr024_epoch_not_current")
    checkout_record = composition_binding.get("checkout_forest")
    checkout_binding = (
        checkout_record.get("binding") if isinstance(checkout_record, Mapping) else None
    )
    forest_id = checkout_binding.get("forest_id") if isinstance(checkout_binding, Mapping) else ""
    if (
        not isinstance(forest_id, str)
        or not forest_id
        or report.get("snapshot_roots", {}).get("forest_id") != forest_id
    ):
        return _dcr051_abstain(DoctorFindingDisposition.DEFERRED, "forest_root_not_current")
    edge_by_id = {
        str(edge.get("id")): edge for edge in canonical_graph["edges"] if isinstance(edge, Mapping)
    }
    candidates: list[tuple[int, str, Mapping[str, Any], Mapping[str, Any]]] = []
    for finding in report["findings"]:
        if not isinstance(finding, Mapping):
            return _dcr051_abstain(DoctorFindingDisposition.REJECTED, "dcr024_finding_not_typed")
        mismatch_class, status, edge_id = (
            finding.get("mismatch_class"),
            finding.get("status"),
            finding.get("edge_id"),
        )
        if (
            mismatch_class not in _DCR051_MISMATCH_CLASSES
            or status not in _DCR051_NONPASSING
            or not isinstance(edge_id, str)
            or edge_id not in edge_by_id
            or not isinstance(finding.get("semantic_key"), Mapping)
            or set(finding["semantic_key"])
            != {"package", "operation", "direction", "schema", "profile", "transport"}
            or not all(isinstance(item, str) and item for item in finding["semantic_key"].values())
        ):
            return _dcr051_abstain(
                DoctorFindingDisposition.ABSTAINED, "unsupported_or_expected_only_finding"
            )
        edge = edge_by_id[edge_id]
        candidates.append((_DCR051_RELATION_ORDER[str(edge["relation"])], edge_id, finding, edge))
    if not candidates:
        return _dcr051_abstain(
            DoctorFindingDisposition.ABSTAINED, "no_supported_nonpassing_finding"
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return _dcr051_abstain(DoctorFindingDisposition.ABSTAINED, "earliest_edge_ambiguous")
    _, edge_id, finding, edge = candidates[0]
    if not isinstance(source_slices, tuple) or len(source_slices) != 1:
        return _dcr051_abstain(
            DoctorFindingDisposition.ABSTAINED, "minimal_exact_source_slice_required"
        )
    source = source_slices[0]
    if not isinstance(source, DoctorSourceSlice) or source.edge_id != edge_id:
        return _dcr051_abstain(
            DoctorFindingDisposition.ABSTAINED, "source_slice_not_bound_to_earliest_edge"
        )
    if finding["mismatch_class"] in {"implementation", "mediation", "identity"}:
        from .proof.kernel_reconstruction import (
            KernelReconstructionDisposition,
            KernelReconstructionResult,
        )

        if (
            not isinstance(reconstruction, KernelReconstructionResult)
            or reconstruction.disposition is not KernelReconstructionDisposition.REFUTED
            or reconstruction.roots is None
            or reconstruction.roots.graph_cid != graph_cid
        ):
            return _dcr051_abstain(
                DoctorFindingDisposition.DEFERRED, "dcr033_counterexample_roots_required"
            )
    semantic_key = dict(sorted(finding["semantic_key"].items()))
    epoch_cid = str(report["findings_cid"])
    finding_id = content_identity(
        {
            "edge_id": edge_id,
            "semantic_key": semantic_key,
            "mismatch_class": finding["mismatch_class"],
            "forest_id": forest_id,
            "graph_cid": graph_cid,
            "epoch_cid": epoch_cid,
        }
    )
    evidence_cids = (source.slice_cid, epoch_cid, content_identity(dict(edge)))
    return DoctorFinding(
        disposition=DoctorFindingDisposition.FINDING,
        reason_code="earliest_topological_nonpassing_edge",
        finding_id=finding_id,
        edge_id=edge_id,
        semantic_key=semantic_key,
        mismatch_class=str(finding["mismatch_class"]),
        forest_id=forest_id,
        graph_cid=graph_cid,
        epoch_cid=epoch_cid,
        findings_cid=epoch_cid,
        evidence_cids=evidence_cids,
    )


# ---------------------------------------------------------------------------
# DCR-052 registry-only transform selection (no materialization)
# ---------------------------------------------------------------------------

DCR_DOCTOR_TRANSFORM_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-doctor-transform@1"
)
_DCR052_FORBIDDEN_FIELDS: Final[frozenset[str]] = frozenset(
    {"body", "code", "source", "source_bytes", "prompt", "command", "shell", "callable"}
)


class DoctorTransformDisposition(StrEnum):
    TRANSFORM = "transform"
    ABSTAINED = "abstained"
    DEFERRED = "deferred"
    REJECTED = "rejected"


@dataclass(frozen=True)
class DoctorTransform:
    disposition: DoctorTransformDisposition
    reason_code: str = ""
    transform_id: str = ""
    finding_id: str = ""
    operator_id: str = ""
    descriptor_id: str = ""
    registry_cid: str = ""
    policy_pin_cid: str = ""
    applicability_cid: str = ""
    proof_cid: str = ""
    impact_cid: str = ""
    roots_cid: str = ""

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": DCR_DOCTOR_TRANSFORM_SCHEMA,
            "interface": SCA_DOCTOR_BRIDGE_INTERFACE,
            "authoritative": False,
            "mutation_authorized": False,
            "planning_authorized": False,
            "completion_authorized": False,
            "disposition": self.disposition.value,
            "reason_code": self.reason_code,
            "transform_id": self.transform_id,
            "finding_id": self.finding_id,
            "operator_id": self.operator_id,
            "descriptor_id": self.descriptor_id,
            "registry_cid": self.registry_cid,
            "policy_pin_cid": self.policy_pin_cid,
            "applicability_cid": self.applicability_cid,
            "proof_cid": self.proof_cid,
            "impact_cid": self.impact_cid,
            "roots_cid": self.roots_cid,
            "model_call_count": 0,
            "provider_call_count": 0,
            "network_call_count": 0,
        }
        return {**body, "transform_cid": content_identity(body)}


def _dcr052_result(disposition: DoctorTransformDisposition, reason: str) -> DoctorTransform:
    return DoctorTransform(disposition=disposition, reason_code=reason)


def _dcr052_value_safe(value: Any) -> bool:
    if callable(value):
        return False
    if isinstance(value, Mapping):
        return not any(
            key.lower() in _DCR052_FORBIDDEN_FIELDS or not _dcr052_value_safe(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, frozenset, set)):
        return all(_dcr052_value_safe(item) for item in value)
    return True


def _closed_dcr052_mapping(value: Any, fields: frozenset[str], name: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or any(key.lower() in _DCR052_FORBIDDEN_FIELDS for key in value)
    ):
        raise DoctorBridgeError(f"{name}_shape_or_raw_body_invalid")
    if not _dcr052_value_safe(value):
        raise DoctorBridgeError(f"{name}_callable_invalid")
    return value


def select_deterministic_doctor_transform(
    *,
    finding: DoctorFinding,
    registry: Any,
    reviewed_registry_cid: str,
    policy_pin: Mapping[str, Any],
    applicability: Mapping[str, Any],
    proof: Mapping[str, Any],
    impact: Mapping[str, Any],
    roots: Mapping[str, Any],
) -> DoctorTransform:
    """Select exactly one policy-pinned DCR-040 descriptor, never a patch."""

    from .autonomous_repair.operators.registry import OperatorRegistry

    if (
        not isinstance(finding, DoctorFinding)
        or finding.disposition is not DoctorFindingDisposition.FINDING
        or not finding.finding_id
        or not finding.edge_id
        or len(finding.evidence_cids) != 3
        or not all(isinstance(item, str) and item for item in finding.evidence_cids)
    ):
        return _dcr052_result(DoctorTransformDisposition.DEFERRED, "dcr051_finding_not_current")
    expected_finding_id = content_identity(
        {
            "edge_id": finding.edge_id,
            "semantic_key": dict(sorted(finding.semantic_key.items())),
            "mismatch_class": finding.mismatch_class,
            "forest_id": finding.forest_id,
            "graph_cid": finding.graph_cid,
            "epoch_cid": finding.epoch_cid,
        }
    )
    if (
        finding.finding_id != expected_finding_id
        or finding.epoch_cid != finding.findings_cid
        or finding.reason_code != "earliest_topological_nonpassing_edge"
    ):
        return _dcr052_result(DoctorTransformDisposition.DEFERRED, "dcr051_finding_identity_stale")
    if not isinstance(registry, OperatorRegistry):
        return _dcr052_result(DoctorTransformDisposition.REJECTED, "typed_dcr040_registry_required")
    report = registry.report()
    if reviewed_registry_cid != report.get("registry_cid"):
        return _dcr052_result(DoctorTransformDisposition.DEFERRED, "dcr040_registry_cid_stale")
    try:
        pin = _closed_dcr052_mapping(
            policy_pin,
            frozenset(
                {
                    "issuer",
                    "policy_id",
                    "policy_digest",
                    "registry_cid",
                    "descriptor_id",
                    "finding_class",
                    "pin_cid",
                }
            ),
            "policy_pin",
        )
        if (
            pin["issuer"] != "reviewed_external_policy"
            or pin["registry_cid"] != reviewed_registry_cid
            or pin["finding_class"] != finding.mismatch_class
            or pin["pin_cid"]
            != content_identity({key: pin[key] for key in pin if key != "pin_cid"})
            or not all(isinstance(pin[key], str) and pin[key] for key in pin)
        ):
            return _dcr052_result(
                DoctorTransformDisposition.DEFERRED, "external_policy_pin_not_current"
            )
        applicable = _closed_dcr052_mapping(
            applicability,
            frozenset(
                {
                    "operator_ids",
                    "finding_id",
                    "edge_id",
                    "owner_root",
                    "relative_path",
                    "source_slice_cid",
                    "predicate_cids",
                    "applicability_cid",
                }
            ),
            "applicability",
        )
        if (
            applicable["finding_id"] != finding.finding_id
            or applicable["edge_id"] != finding.edge_id
            or not isinstance(applicable["operator_ids"], tuple)
            or not applicable["operator_ids"]
            or not all(isinstance(item, str) and item for item in applicable["operator_ids"])
            or not isinstance(applicable["predicate_cids"], Mapping)
            or applicable["applicability_cid"]
            != content_identity(
                {key: applicable[key] for key in applicable if key != "applicability_cid"}
            )
        ):
            return _dcr052_result(
                DoctorTransformDisposition.DEFERRED, "operator_applicability_not_current"
            )
        root_value = _closed_dcr052_mapping(
            roots,
            frozenset({"forest_id", "graph_cid", "epoch_cid", "findings_cid", "roots_cid"}),
            "roots",
        )
        if (
            root_value["forest_id"] != finding.forest_id
            or root_value["graph_cid"] != finding.graph_cid
            or root_value["epoch_cid"] != finding.epoch_cid
            or root_value["findings_cid"] != finding.findings_cid
            or root_value["roots_cid"]
            != content_identity({key: root_value[key] for key in root_value if key != "roots_cid"})
        ):
            return _dcr052_result(DoctorTransformDisposition.DEFERRED, "typed_roots_stale")
        impact_value = _closed_dcr052_mapping(
            impact,
            frozenset({"finding_id", "edge_id", "roots_cid", "impact_id", "impact_cid"}),
            "impact",
        )
        if (
            impact_value["finding_id"] != finding.finding_id
            or impact_value["edge_id"] != finding.edge_id
            or impact_value["roots_cid"] != root_value["roots_cid"]
            or impact_value["impact_cid"]
            != content_identity(
                {key: impact_value[key] for key in impact_value if key != "impact_cid"}
            )
        ):
            return _dcr052_result(DoctorTransformDisposition.DEFERRED, "impact_not_current")
        proof_value = _closed_dcr052_mapping(
            proof,
            frozenset({"finding_id", "applicability_cid", "impact_cid", "proof_id", "proof_cid"}),
            "proof",
        )
        if (
            proof_value["finding_id"] != finding.finding_id
            or proof_value["applicability_cid"] != applicable["applicability_cid"]
            or proof_value["impact_cid"] != impact_value["impact_cid"]
            or proof_value["proof_cid"]
            != content_identity(
                {key: proof_value[key] for key in proof_value if key != "proof_cid"}
            )
        ):
            return _dcr052_result(
                DoctorTransformDisposition.DEFERRED, "applicability_proof_not_current"
            )
    except DoctorBridgeError as exc:
        return _dcr052_result(DoctorTransformDisposition.REJECTED, str(exc))
    candidates = [
        descriptor
        for descriptor in registry.enumerate()
        if descriptor.operator_id in applicable["operator_ids"]
        and descriptor.descriptor_id == pin["descriptor_id"]
        and descriptor.owner_root == applicable["owner_root"]
        and applicable["relative_path"] in descriptor.write_scope
        and set(descriptor.before_predicates) == set(applicable["predicate_cids"])
    ]
    if not candidates:
        return _dcr052_result(
            DoctorTransformDisposition.ABSTAINED, "no_registered_applicable_operator"
        )
    if len(candidates) != 1:
        return _dcr052_result(DoctorTransformDisposition.ABSTAINED, "registered_operator_ambiguous")
    descriptor = candidates[0]
    transform_id = content_identity(
        {
            "finding_id": finding.finding_id,
            "operator_id": descriptor.operator_id,
            "descriptor_id": descriptor.descriptor_id,
            "policy_pin_cid": pin["pin_cid"],
            "applicability_cid": applicable["applicability_cid"],
            "proof_cid": proof_value["proof_cid"],
            "impact_cid": impact_value["impact_cid"],
            "roots_cid": root_value["roots_cid"],
        }
    )
    return DoctorTransform(
        disposition=DoctorTransformDisposition.TRANSFORM,
        reason_code="unique_policy_pinned_registered_operator",
        transform_id=transform_id,
        finding_id=finding.finding_id,
        operator_id=descriptor.operator_id,
        descriptor_id=descriptor.descriptor_id,
        registry_cid=reviewed_registry_cid,
        policy_pin_cid=pin["pin_cid"],
        applicability_cid=applicable["applicability_cid"],
        proof_cid=proof_value["proof_cid"],
        impact_cid=impact_value["impact_cid"],
        roots_cid=root_value["roots_cid"],
    )


__all__ = [
    "ABSTENTION_SCHEMA",
    "AnalyticalAbstention",
    "DoctorBridgeError",
    "DoctorFinding",
    "DoctorFindingDisposition",
    "DoctorTransform",
    "DoctorTransformDisposition",
    "DoctorSourceSlice",
    "DoctorDisposition",
    "SCA_DOCTOR_BRIDGE_INTERFACE",
    "ScaContractFinding",
    "TRANSFORM_RECEIPT_SCHEMA",
    "TransformReceipt",
    "diagnose_finding_with_ir",
    "diagnose_findings_with_ir",
    "diagnose_earliest_dcr_edge",
    "map_finding",
    "map_findings",
    "select_deterministic_doctor_transform",
]
