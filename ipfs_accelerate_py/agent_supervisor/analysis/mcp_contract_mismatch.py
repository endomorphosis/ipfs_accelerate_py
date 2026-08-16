"""DCR-024 pure, fail-closed MCP contract mismatch analysis.

It consumes graph/transcript *data* only.  Runtime conformance belongs to
DCR-023; without a current valid transcript this module reports an explicit
integration-pending liveness finding and never asserts production readiness.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, Final

from ..proof.formal_verification_contracts import canonical_json_bytes, content_identity
from .mcp_live_observer import (
    MCP_LIVE_OBSERVATION_EPOCH_SCHEMA,
    McpObservationEpoch,
    is_current_mcp_observation_epoch,
)

MCP_CONTRACT_MISMATCH_INTERFACE: Final = "McpContractMismatchAnalysis@1"
MCP_CONTRACT_MISMATCH_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-mismatch-analysis@1"
)
_RELATION_ORDER: Final[dict[str, int]] = {
    "expects_descriptor": 10,
    "binds_orb_idl": 20,
    "defines_method_schema": 30,
    "binds_mediator_route": 40,
    "routes_to_observed_dispatcher": 50,
    "dispatches_to_handler": 60,
    "performs_effect": 70,
    "emits_receipt_runtime_identity": 80,
}
_SEMANTIC_KEY_FIELDS: Final[tuple[str, ...]] = (
    "package",
    "operation",
    "direction",
    "schema",
    "profile",
    "transport",
)


class McpContractMismatchError(ValueError):
    """A graph, transcript, or root fixture is not deterministic evidence."""


class MismatchClass(Enum):
    PROTOCOL = "protocol"
    SCHEMA = "schema"
    AUTHORITY = "authority"
    LIVENESS = "liveness"
    IDENTITY = "identity"
    MEDIATION = "mediation"
    IMPLEMENTATION = "implementation"


class EvidenceStatus(Enum):
    PASSED = "passed"
    EXPECTED_ONLY = "expected_only"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"
    UNOBSERVED = "unobserved"
    FAILED = "failed"

    @property
    def passing(self) -> bool:
        return self is EvidenceStatus.PASSED


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise McpContractMismatchError(f"{field} must be a mapping")
    return value


def _records(value: Any, field: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise McpContractMismatchError(f"{field} must be a sequence of records")
    return [_mapping(item, field) for item in value]


def _roots(value: Any, field: str) -> dict[str, str]:
    raw = _mapping(value, field)
    if not raw:
        raise McpContractMismatchError(f"{field} must not be empty")
    normalized = {str(key): str(item) for key, item in raw.items()}
    if any(not key.strip() or not item.strip() for key, item in normalized.items()):
        raise McpContractMismatchError(f"{field} must contain non-empty exact roots")
    return dict(sorted(normalized.items()))


def _enum(enum: type[Enum], value: Any, field: str) -> Enum:
    try:
        return enum(value)
    except (TypeError, ValueError) as exc:
        raise McpContractMismatchError(f"{field} is not closed deterministic vocabulary") from exc


def _edge_index(graph: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for edge in _records(graph.get("edges", ()), "graph edges"):
        edge_id = str(edge.get("id") or "").strip()
        if not edge_id or edge_id in result:
            raise McpContractMismatchError("graph edge identities must be unique and non-empty")
        result[edge_id] = edge
    return result


def _candidate(
    *,
    mismatch_class: MismatchClass,
    status: EvidenceStatus,
    edge_id: str,
    edge_order: int,
    semantic_roots: Mapping[str, str],
    snapshot_roots: Mapping[str, str],
    detail: Mapping[str, Any],
    semantic_key: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidate = {
        "mismatch_class": mismatch_class.value,
        "status": status.value,
        "edge_id": edge_id,
        "edge_order": edge_order,
        "semantic_roots": dict(semantic_roots),
        "snapshot_roots": dict(snapshot_roots),
        "detail": dict(detail),
    }
    if semantic_key is not None:
        candidate["semantic_key"] = dict(semantic_key)
    return candidate


def _class_for_relation(relation: str) -> MismatchClass:
    if relation in {"expects_descriptor", "binds_orb_idl"}:
        return MismatchClass.PROTOCOL
    if relation == "defines_method_schema":
        return MismatchClass.SCHEMA
    if relation in {"binds_mediator_route", "routes_to_observed_dispatcher"}:
        return MismatchClass.MEDIATION
    if relation in {"dispatches_to_handler", "performs_effect"}:
        return MismatchClass.IMPLEMENTATION
    if relation == "emits_receipt_runtime_identity":
        return MismatchClass.IDENTITY
    return MismatchClass.PROTOCOL


def _current_dcr023(
    transcript: McpObservationEpoch | Mapping[str, Any] | None,
    *,
    graph_cid: str,
    semantic_roots: Mapping[str, str],
    snapshot_roots: Mapping[str, str],
) -> bool:
    return is_current_mcp_observation_epoch(
        transcript,
        graph_cid=graph_cid,
        semantic_roots=semantic_roots,
        snapshot_roots=snapshot_roots,
    )


def _transcript_mapping(
    transcript: McpObservationEpoch | Mapping[str, Any],
) -> Mapping[str, Any]:
    return transcript.to_dict() if isinstance(transcript, McpObservationEpoch) else transcript


def _complete_semantic_key(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping) or not all(
        isinstance(value.get(field), str) and value[field].strip() for field in _SEMANTIC_KEY_FIELDS
    ):
        return None
    return value


def analyze_mcp_contract_mismatches(
    *,
    graph: Mapping[str, Any],
    semantic_roots: Mapping[str, str],
    snapshot_roots: Mapping[str, str],
    transcript: McpObservationEpoch | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return earliest non-passing evidence per class and exact root binding."""

    graph_value = _mapping(graph, "graph")
    graph_cid = str(graph_value.get("graph_cid") or "").strip()
    if not graph_cid:
        raise McpContractMismatchError("graph must carry canonical graph_cid")
    semantic = _roots(semantic_roots, "semantic_roots")
    snapshot = _roots(snapshot_roots, "snapshot_roots")
    edges = _edge_index(graph_value)
    candidates: list[dict[str, Any]] = []
    for blocker in _records(graph_value.get("blockers", ()), "graph blockers"):
        kind = str(blocker.get("kind") or "").strip()
        status = EvidenceStatus.AMBIGUOUS if "ambiguous" in kind else EvidenceStatus.MISSING
        mismatch_class = (
            MismatchClass.AUTHORITY
            if kind == "authority_conflict"
            else MismatchClass.IMPLEMENTATION
        )
        candidates.append(
            _candidate(
                mismatch_class=mismatch_class,
                status=status,
                edge_id="graph:" + (kind or "blocker"),
                edge_order=0,
                semantic_roots=semantic,
                snapshot_roots=snapshot,
                detail=blocker,
            )
        )
    current_transcript = _current_dcr023(
        transcript, graph_cid=graph_cid, semantic_roots=semantic, snapshot_roots=snapshot
    )
    if not current_transcript:
        candidates.append(
            _candidate(
                mismatch_class=MismatchClass.LIVENESS,
                status=EvidenceStatus.UNOBSERVED,
                edge_id="dcr023:current-transcript",
                edge_order=0,
                semantic_roots=semantic,
                snapshot_roots=snapshot,
                detail={"reason": "dcr023_current_valid_transcript_required"},
            )
        )
    else:
        assert transcript is not None
        transcript_value = _transcript_mapping(transcript)
        for check in _records(transcript_value.get("checks", ()), "DCR-023 transcript checks"):
            mismatch_class = _enum(MismatchClass, check.get("mismatch_class"), "mismatch_class")
            status = _enum(EvidenceStatus, check.get("status"), "status")
            if status.passing:
                continue
            edge_id = str(check.get("edge_id") or "").strip()
            if edge_id not in edges:
                raise McpContractMismatchError("transcript check does not bind a graph edge")
            edge = edges[edge_id]
            relation = str(edge.get("relation") or "")
            candidates.append(
                _candidate(
                    mismatch_class=mismatch_class,
                    status=status,
                    edge_id=edge_id,
                    edge_order=_RELATION_ORDER.get(relation, 10_000),
                    semantic_roots=semantic,
                    snapshot_roots=snapshot,
                    detail={"check": check, "edge": edge},
                    semantic_key=_complete_semantic_key(check.get("semantic_key")),
                )
            )
    # Exact canonical duplicates alone collapse.  Same-looking entries with a
    # different root, class, edge, or detail remain independent evidence.
    unique: dict[bytes, dict[str, Any]] = {}
    for item in candidates:
        unique.setdefault(canonical_json_bytes(item), item)
    earliest: dict[tuple[bytes, bytes, str, bytes], dict[str, Any]] = {}
    for item in unique.values():
        semantic_key = _complete_semantic_key(item.get("semantic_key"))
        # Only an explicit complete semantic key may collapse different graph
        # edges.  Missing keys deliberately retain independent failures.
        semantic_group = (
            canonical_json_bytes(semantic_key)
            if isinstance(semantic_key, Mapping)
            else canonical_json_bytes(item)
        )
        key = (
            canonical_json_bytes(item["semantic_roots"]),
            canonical_json_bytes(item["snapshot_roots"]),
            item["mismatch_class"],
            semantic_group,
        )
        existing = earliest.get(key)
        if existing is None or (item["edge_order"], item["edge_id"]) < (
            existing["edge_order"],
            existing["edge_id"],
        ):
            earliest[key] = item
    findings = sorted(
        earliest.values(),
        key=lambda item: (item["mismatch_class"], item["edge_order"], item["edge_id"]),
    )
    status = (
        "integration_pending" if not current_transcript else ("nonpassing" if findings else "ready")
    )
    body = {
        "schema": MCP_CONTRACT_MISMATCH_SCHEMA,
        "interface": MCP_CONTRACT_MISMATCH_INTERFACE,
        "authoritative": False,
        "graph_cid": graph_cid,
        "semantic_roots": semantic,
        "snapshot_roots": snapshot,
        "dcr023_current_valid": current_transcript,
        "production_readiness": status,
        "findings": findings,
    }
    return {**body, "findings_cid": content_identity(body)}


__all__ = [
    "EvidenceStatus",
    "MCP_LIVE_OBSERVATION_EPOCH_SCHEMA",
    "MCP_CONTRACT_MISMATCH_INTERFACE",
    "MCP_CONTRACT_MISMATCH_SCHEMA",
    "MismatchClass",
    "McpContractMismatchError",
    "analyze_mcp_contract_mismatches",
]
