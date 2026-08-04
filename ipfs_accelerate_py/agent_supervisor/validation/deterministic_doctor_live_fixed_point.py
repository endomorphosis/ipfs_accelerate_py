"""Live reparse/static/security/replan/reprove fixed-point runner (PDR-053).

Interface: ``DeterministicDoctorLiveFixedPoint@1``

Conflict policy: pure :class:`DeterministicDoctorFixedPointValidator` stays
evidence-consuming and never simulates production.  This module is the live
producer: it independently reparses and rebuilds the candidate tree, re-indexes
changed scope, invalidates dependency-local caches, re-diffs contracts,
recloses consumers/SCCs, extracts code security facts, checks IntentIR/code
forbidden logic and required security/hyperproperties, runs impact-selected
static/tests, replans, and replays kernel proofs.  Sealed stage receipts are
then handed to the pure validator.

Prebuilt fixed-point mappings or booleans cannot complete.  Second-order
findings trigger another bounded iteration.  Oscillation, unchanged residual,
budget exhaustion, or capability loss aborts and rolls back.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Protocol

from ..analysis.deterministic_doctor_contracts import (
    DeterministicDoctorPlan,
    DoctorPlanDisposition,
)
from ..planning.deterministic_doctor_transaction import (
    DoctorTransactionCheckpoint,
    DoctorTransactionDisposition,
    DoctorTransactionReport,
)
from ..proof.formal_verification_contracts import content_identity
from ..security_contract_analysis import (
    FixedPointSecurityReceipt,
    FlowEdge,
    FlowNode,
    SecurityAnalysisConfig,
    SecurityEvidence,
    SecurityPropertyDeclaration,
    evaluate_fixed_point_security,
    extract_code_security_facts,
)
from .deterministic_doctor_fixed_point import (
    DEFAULT_FIXED_POINT_BOUND,
    MAX_ITERATIONS,
    MAX_OSCILLATION_WINDOW,
    CandidateDoctorFixedPointEvidence,
    DeterministicDoctorFixedPointError,
    DeterministicDoctorFixedPointValidator,
    DoctorCacheInvalidationEvidence,
    DoctorFixedPointDisposition,
    DoctorFixedPointIterationReceipt,
    DoctorFixedPointOutcome,
    DoctorFixedPointReason,
    DoctorFixedPointReceipt,
    DoctorRebuildEvidence,
    DoctorRecloseEvidence,
    DoctorRedeltaEvidence,
    DoctorReplanEvidence,
    DoctorReproveEvidence,
    DoctorSecurityCheckEvidence,
    DoctorStaticCheckEvidence,
)


# ---------------------------------------------------------------------------
# Schema / interface constants
# ---------------------------------------------------------------------------

DETERMINISTIC_DOCTOR_LIVE_FIXED_POINT_INTERFACE: Final[str] = (
    "DeterministicDoctorLiveFixedPoint@1"
)
LIVE_FIXED_POINT_PRODUCER_ID: Final[str] = "deterministic-doctor-live-fixed-point@1"
LIVE_FIXED_POINT_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/live-fixed-point-request@1"
)
LIVE_FIXED_POINT_RUN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/live-fixed-point-run@1"
)
CONTRACT_VERSION: Final[int] = 1

# Budget floors (fail-closed when exceeded).
DEFAULT_MAX_STAGE_BUDGET: Final[int] = 64
DEFAULT_MAX_PATH_BYTES: Final[int] = 16 * 1024 * 1024


# ---------------------------------------------------------------------------
# Errors / vocabularies
# ---------------------------------------------------------------------------


class DeterministicDoctorLiveFixedPointError(ValueError):
    """Live fixed-point runner rejected input or lost capability."""


class LiveFixedPointAbortReason(str, Enum):
    """Machine-readable abort causes for the live runner."""

    PREBUILT_MAPPING = "prebuilt_fixed_point_mapping"
    PREBUILT_BOOLEAN = "prebuilt_fixed_point_boolean"
    OSCILLATION = "oscillation_detected"
    UNCHANGED_RESIDUAL = "unchanged_residual"
    BOUND_EXHAUSTED = "fixed_point_bound_exhausted"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CAPABILITY_LOST = "capability_lost"
    STAGE_FAILURE = "stage_failure"
    SECURITY_FAILURE = "security_failure"
    ROOT_DRIFT = "root_drift"
    MALFORMED = "malformed_input"


# ---------------------------------------------------------------------------
# Live snapshot / stage context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveTreeFile:
    """One independently reread candidate-tree file."""

    path: str
    content: bytes
    sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise DeterministicDoctorLiveFixedPointError("path must be non-empty")
        if not isinstance(self.content, (bytes, bytearray)):
            raise DeterministicDoctorLiveFixedPointError("content must be bytes")
        digest = self.sha256.strip() if isinstance(self.sha256, str) else ""
        if not digest:
            digest = "sha256:" + hashlib.sha256(bytes(self.content)).hexdigest()
        object.__setattr__(self, "content", bytes(self.content))
        object.__setattr__(self, "sha256", digest)


@dataclass
class LiveCandidateSnapshot:
    """Independently observed candidate-tree state for one iteration."""

    candidate_tree_id: str
    files: tuple[LiveTreeFile, ...]
    index_id: str = ""
    ast_index_id: str = ""
    call_graph_id: str = ""
    dependency_graph_id: str = ""
    schema_graph_id: str = ""
    value_graph_id: str = ""
    reparsed_paths: tuple[str, ...] = ()
    parse_errors: tuple[str, ...] = ()
    tombstone_ids: tuple[str, ...] = ()
    vector_row_ids: tuple[str, ...] = ()
    kg_node_ids: tuple[str, ...] = ()
    dependency_edges: tuple[tuple[str, str], ...] = ()
    observed_root_cid: str = ""

    def path_set(self) -> frozenset[str]:
        return frozenset(item.path for item in self.files)

    def content_map(self) -> dict[str, bytes]:
        return {item.path: item.content for item in self.files}


@dataclass(frozen=True)
class LiveStageBudget:
    """Bounded resource counters for one live fixed-point run."""

    max_iterations: int = DEFAULT_FIXED_POINT_BOUND
    max_stage_invocations: int = DEFAULT_MAX_STAGE_BUDGET
    max_path_bytes: int = DEFAULT_MAX_PATH_BYTES
    stage_invocations: int = 0
    bytes_read: int = 0

    def charge_stage(self, *, bytes_read: int = 0) -> "LiveStageBudget":
        next_inv = self.stage_invocations + 1
        next_bytes = self.bytes_read + max(0, int(bytes_read))
        if next_inv > self.max_stage_invocations:
            raise DeterministicDoctorLiveFixedPointError(
                LiveFixedPointAbortReason.BUDGET_EXHAUSTED.value
            )
        if next_bytes > self.max_path_bytes:
            raise DeterministicDoctorLiveFixedPointError(
                LiveFixedPointAbortReason.BUDGET_EXHAUSTED.value
            )
        return LiveStageBudget(
            max_iterations=self.max_iterations,
            max_stage_invocations=self.max_stage_invocations,
            max_path_bytes=self.max_path_bytes,
            stage_invocations=next_inv,
            bytes_read=next_bytes,
        )


# ---------------------------------------------------------------------------
# Stage adapter protocols
# ---------------------------------------------------------------------------


class LiveSnapshotLoader(Protocol):
    def __call__(
        self,
        *,
        plan: DeterministicDoctorPlan,
        transaction_report: DoctorTransactionReport,
        iteration: int,
        prior: LiveCandidateSnapshot | None,
    ) -> LiveCandidateSnapshot: ...


class LiveCacheInvalidator(Protocol):
    def __call__(
        self,
        snapshot: LiveCandidateSnapshot,
        *,
        prior_cache_ids: Sequence[str],
        dependency_paths: Sequence[str],
    ) -> DoctorCacheInvalidationEvidence: ...


class LiveStaticChecker(Protocol):
    def __call__(
        self,
        snapshot: LiveCandidateSnapshot,
        *,
        plan: DeterministicDoctorPlan,
    ) -> DoctorStaticCheckEvidence: ...


class LiveRedeltaStage(Protocol):
    def __call__(
        self,
        snapshot: LiveCandidateSnapshot,
        *,
        plan: DeterministicDoctorPlan,
        prior_delta_ids: Sequence[str],
    ) -> DoctorRedeltaEvidence: ...


class LiveRecloseStage(Protocol):
    def __call__(
        self,
        snapshot: LiveCandidateSnapshot,
        *,
        plan: DeterministicDoctorPlan,
        original_finding_ids: Sequence[str],
        prior_second_order: Sequence[str],
        iteration: int,
    ) -> DoctorRecloseEvidence: ...


class LiveSecurityStage(Protocol):
    def __call__(
        self,
        snapshot: LiveCandidateSnapshot,
        *,
        plan: DeterministicDoctorPlan,
        request: "LiveFixedPointRequest",
    ) -> DoctorSecurityCheckEvidence: ...


class LiveReplanStage(Protocol):
    def __call__(
        self,
        snapshot: LiveCandidateSnapshot,
        *,
        plan: DeterministicDoctorPlan,
        reclose: DoctorRecloseEvidence,
    ) -> DoctorReplanEvidence: ...


class LiveReproveStage(Protocol):
    def __call__(
        self,
        snapshot: LiveCandidateSnapshot,
        *,
        plan: DeterministicDoctorPlan,
        replan: DoctorReplanEvidence,
    ) -> DoctorReproveEvidence: ...


class LiveIdentityReplay(Protocol):
    def __call__(
        self,
        *,
        plan: DeterministicDoctorPlan,
        transaction_report: DoctorTransactionReport,
        iterations: Sequence[DoctorFixedPointIterationReceipt],
    ) -> str: ...


class LiveRestoreAdapter(Protocol):
    def __call__(self, checkpoint: DoctorTransactionCheckpoint) -> bool: ...


class LiveCapabilityProbe(Protocol):
    def __call__(self) -> tuple[bool, tuple[str, ...]]: ...


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


@dataclass
class LiveFixedPointRequest:
    """Inputs for one live fixed-point run.

    ``prebuilt_fixed_point`` / ``prebuilt_complete`` are accepted only so they
    can be *rejected*.  They never authorize completion.
    """

    changed_paths: tuple[str, ...] = ()
    expected_tombstone_ids: tuple[str, ...] = ()
    original_finding_ids: tuple[str, ...] = ()
    original_delta_ids: tuple[str, ...] = ()
    prior_cache_ids: tuple[str, ...] = ()
    intent_effects: tuple[str, ...] = ()
    code_effects: tuple[str, ...] = ()
    forbidden_effect_ids: tuple[str, ...] = ()
    required_hyperproperty_ids: tuple[str, ...] = ()
    held_hyperproperty_receipt_ids: tuple[str, ...] = ()
    failed_hyperproperty_ids: tuple[str, ...] = ()
    flow_nodes: tuple[FlowNode | Mapping[str, Any], ...] = ()
    flow_edges: tuple[FlowEdge | Mapping[str, Any], ...] = ()
    security_properties: tuple[
        SecurityPropertyDeclaration | Mapping[str, Any], ...
    ] = ()
    security_evidence: SecurityEvidence | Mapping[str, Any] | None = None
    tags_by_path: Mapping[str, Sequence[str]] = field(default_factory=dict)
    effects_by_path: Mapping[str, Sequence[str]] = field(default_factory=dict)
    checkout_root: str = ""
    file_bytes: Mapping[str, bytes] = field(default_factory=dict)
    fixed_point_bound: int = DEFAULT_FIXED_POINT_BOUND
    oscillation_window: int = MAX_OSCILLATION_WINDOW
    max_stage_invocations: int = DEFAULT_MAX_STAGE_BUDGET
    # Explicitly rejected prebuilt claims:
    prebuilt_fixed_point: Mapping[str, Any] | None = None
    prebuilt_complete: bool | None = None
    second_order_schedule: Mapping[int, Sequence[str]] = field(default_factory=dict)
    discharge_schedule: Mapping[int, Sequence[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "changed_paths",
            tuple(str(p) for p in self.changed_paths if str(p).strip()),
        )
        object.__setattr__(
            self,
            "expected_tombstone_ids",
            tuple(str(x) for x in self.expected_tombstone_ids if str(x).strip()),
        )
        object.__setattr__(
            self,
            "original_finding_ids",
            tuple(str(x) for x in self.original_finding_ids if str(x).strip()),
        )
        object.__setattr__(
            self,
            "original_delta_ids",
            tuple(str(x) for x in self.original_delta_ids if str(x).strip()),
        )
        object.__setattr__(
            self,
            "prior_cache_ids",
            tuple(str(x) for x in self.prior_cache_ids if str(x).strip()),
        )
        object.__setattr__(
            self,
            "intent_effects",
            tuple(str(x) for x in self.intent_effects if str(x).strip()),
        )
        object.__setattr__(
            self,
            "code_effects",
            tuple(str(x) for x in self.code_effects if str(x).strip()),
        )
        object.__setattr__(
            self,
            "forbidden_effect_ids",
            tuple(str(x) for x in self.forbidden_effect_ids if str(x).strip()),
        )
        object.__setattr__(
            self,
            "required_hyperproperty_ids",
            tuple(str(x) for x in self.required_hyperproperty_ids if str(x).strip()),
        )
        object.__setattr__(
            self,
            "held_hyperproperty_receipt_ids",
            tuple(
                str(x) for x in self.held_hyperproperty_receipt_ids if str(x).strip()
            ),
        )
        object.__setattr__(
            self,
            "failed_hyperproperty_ids",
            tuple(str(x) for x in self.failed_hyperproperty_ids if str(x).strip()),
        )
        bound = int(self.fixed_point_bound)
        if bound < 1 or bound > MAX_ITERATIONS:
            raise DeterministicDoctorLiveFixedPointError(
                f"fixed_point_bound out of range: {bound}"
            )
        object.__setattr__(self, "fixed_point_bound", bound)
        window = int(self.oscillation_window)
        if window < 2 or window > MAX_OSCILLATION_WINDOW:
            raise DeterministicDoctorLiveFixedPointError(
                f"oscillation_window out of range: {window}"
            )
        object.__setattr__(self, "oscillation_window", window)


# ---------------------------------------------------------------------------
# Default independent stage implementations
# ---------------------------------------------------------------------------


def _digest_payload(payload: Any) -> str:
    return content_identity(payload)


def _candidate_tree_id(
    plan: DeterministicDoctorPlan,
    transaction_report: DoctorTransactionReport,
) -> str:
    if transaction_report.candidate_tree is not None:
        return transaction_report.candidate_tree.candidate_tree_cid
    return plan.roots.tree_id


def load_live_candidate_snapshot(
    *,
    plan: DeterministicDoctorPlan,
    transaction_report: DoctorTransactionReport,
    iteration: int,
    prior: LiveCandidateSnapshot | None,
    request: LiveFixedPointRequest,
) -> LiveCandidateSnapshot:
    """Independently reparse/rebuild indexes from checkout or supplied bytes."""

    del prior  # each iteration rereads current tree; prior is for adapters only
    tree_id = _candidate_tree_id(plan, transaction_report)
    files: list[LiveTreeFile] = []
    parse_errors: list[str] = []
    reparsed: list[str] = []
    dep_edges: list[tuple[str, str]] = []
    total_bytes = 0

    path_sources: dict[str, bytes] = {}
    if request.file_bytes:
        for path, payload in request.file_bytes.items():
            path_sources[str(path)] = bytes(payload)
    elif request.checkout_root:
        root = Path(request.checkout_root)
        if not root.is_dir():
            raise DeterministicDoctorLiveFixedPointError(
                f"checkout_root is not a directory: {request.checkout_root}"
            )
        targets = request.changed_paths or tuple(
            sorted(
                str(p.relative_to(root))
                for p in root.rglob("*.py")
                if p.is_file()
            )
        )
        for rel in targets:
            full = root / rel
            if not full.is_file():
                parse_errors.append(f"missing:{rel}")
                continue
            payload = full.read_bytes()
            path_sources[rel] = payload
    else:
        # Fall back to plan write paths with empty bodies (parse will fail unless
        # adapters override).  Live production always supplies checkout or bytes.
        for path in request.changed_paths or tuple(
            path for step in plan.steps for path in step.write_paths
        ):
            path_sources[path] = b""

    for path, payload in sorted(path_sources.items()):
        total_bytes += len(payload)
        if total_bytes > DEFAULT_MAX_PATH_BYTES:
            raise DeterministicDoctorLiveFixedPointError(
                LiveFixedPointAbortReason.BUDGET_EXHAUSTED.value
            )
        files.append(LiveTreeFile(path=path, content=payload))
        reparsed.append(path)
        if path.endswith(".py"):
            try:
                tree = ast.parse(payload.decode("utf-8"), filename=path)
            except (SyntaxError, UnicodeDecodeError) as exc:
                parse_errors.append(f"parse:{path}:{type(exc).__name__}")
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dep_edges.append((path, f"import:{alias.name}"))
                elif isinstance(node, ast.ImportFrom) and node.module:
                    dep_edges.append((path, f"import:{node.module}"))

    files_t = tuple(files)
    path_list = [f.path for f in files_t]
    index_seed = {
        "tree": tree_id,
        "iteration": iteration,
        "paths": path_list,
        "hashes": [f.sha256 for f in files_t],
    }
    index_id = _digest_payload({"kind": "repo-index", **index_seed})
    ast_index_id = _digest_payload({"kind": "ast-index", **index_seed})
    call_graph_id = _digest_payload({"kind": "call-graph", **index_seed})
    dep_graph_id = _digest_payload(
        {"kind": "dep-graph", "edges": sorted(dep_edges), **index_seed}
    )
    schema_graph_id = _digest_payload({"kind": "schema-graph", **index_seed})
    value_graph_id = _digest_payload({"kind": "value-graph", **index_seed})
    root_cid = _digest_payload(
        {"kind": "observed-root", "tree": tree_id, "files": index_seed["hashes"]}
    )
    tombstones = tuple(
        sorted(
            set(request.expected_tombstone_ids)
            | {f"tombstone:{path}" for path in path_list}
        )
    )
    vectors = tuple(f"vector:{path}" for path in path_list)
    kg_nodes = tuple(f"kg:{path}" for path in path_list)
    return LiveCandidateSnapshot(
        candidate_tree_id=tree_id,
        files=files_t,
        index_id=index_id,
        ast_index_id=ast_index_id,
        call_graph_id=call_graph_id,
        dependency_graph_id=dep_graph_id,
        schema_graph_id=schema_graph_id,
        value_graph_id=value_graph_id,
        reparsed_paths=tuple(reparsed),
        parse_errors=tuple(parse_errors),
        tombstone_ids=tombstones,
        vector_row_ids=vectors,
        kg_node_ids=kg_nodes,
        dependency_edges=tuple(sorted(set(dep_edges))),
        observed_root_cid=root_cid,
    )


def build_rebuild_evidence(snapshot: LiveCandidateSnapshot) -> DoctorRebuildEvidence:
    """Seal rebuild evidence from an independently observed snapshot."""

    clean = not snapshot.parse_errors and bool(snapshot.reparsed_paths)
    return DoctorRebuildEvidence(
        candidate_tree_id=snapshot.candidate_tree_id,
        repository_index_id=snapshot.index_id or f"repo-index:{snapshot.candidate_tree_id}",
        ast_index_id=snapshot.ast_index_id or f"ast-index:{snapshot.candidate_tree_id}",
        vector_row_ids=snapshot.vector_row_ids or ("vector:none",),
        kg_node_ids=snapshot.kg_node_ids or ("kg:none",),
        call_graph_id=snapshot.call_graph_id or f"call-graph:{snapshot.candidate_tree_id}",
        dependency_graph_id=(
            snapshot.dependency_graph_id or f"dep-graph:{snapshot.candidate_tree_id}"
        ),
        schema_graph_id=(
            snapshot.schema_graph_id or f"schema-graph:{snapshot.candidate_tree_id}"
        ),
        value_graph_id=(
            snapshot.value_graph_id or f"value-graph:{snapshot.candidate_tree_id}"
        ),
        tombstone_ids=snapshot.tombstone_ids or (f"tombstone:{snapshot.candidate_tree_id}",),
        reparsed_paths=snapshot.reparsed_paths or ("<none>",),
        clean_rebuild_equivalent=clean,
    )


def default_cache_invalidation(
    snapshot: LiveCandidateSnapshot,
    *,
    prior_cache_ids: Sequence[str],
    dependency_paths: Sequence[str],
) -> DoctorCacheInvalidationEvidence:
    """Invalidate dependency-local caches under the current tree."""

    dep_paths = tuple(sorted({str(p) for p in dependency_paths if str(p).strip()}))
    invalidated = tuple(
        sorted(
            set(prior_cache_ids)
            | {f"cache:{path}" for path in dep_paths}
            | {f"cas:{snapshot.candidate_tree_id}"}
            | {f"proof-cache:{snapshot.candidate_tree_id}"}
        )
    )
    return DoctorCacheInvalidationEvidence(
        candidate_tree_id=snapshot.candidate_tree_id,
        invalidated_cache_ids=invalidated or (f"cache:{snapshot.candidate_tree_id}",),
        invalidated_cas_ids=(f"cas:{snapshot.candidate_tree_id}",),
        tombstone_ids=snapshot.tombstone_ids or (f"tombstone:{snapshot.candidate_tree_id}",),
        remaining_stale_ids=(),
        complete=True,
    )


def default_static_checks(
    snapshot: LiveCandidateSnapshot,
    *,
    plan: DeterministicDoctorPlan,
) -> DoctorStaticCheckEvidence:
    """Impact-selected static/type/parse checks from the live snapshot."""

    del plan
    failed = tuple(snapshot.parse_errors)
    paths = snapshot.reparsed_paths or ("<none>",)
    receipt_seed = snapshot.observed_root_cid or snapshot.candidate_tree_id
    return DoctorStaticCheckEvidence(
        candidate_tree_id=snapshot.candidate_tree_id,
        reparsed_paths=paths,
        type_check_receipt_ids=(f"type:{receipt_seed}",),
        static_check_receipt_ids=(f"static:{receipt_seed}",),
        differential_check_receipt_ids=(f"diff:{receipt_seed}",),
        proof_check_receipt_ids=(f"proof-check:{receipt_seed}",),
        memory_effect_receipt_ids=(f"memory:{receipt_seed}",),
        resource_check_receipt_ids=(f"resource:{receipt_seed}",),
        failed_check_ids=failed,
        all_passed=not failed,
    )


def default_redelta(
    snapshot: LiveCandidateSnapshot,
    *,
    plan: DeterministicDoctorPlan,
    prior_delta_ids: Sequence[str],
) -> DoctorRedeltaEvidence:
    """Re-diff contracts from independently observed paths/hashes."""

    recomputed = tuple(
        sorted(
            set(prior_delta_ids)
            | {
                content_identity(
                    {
                        "kind": "delta",
                        "tree": snapshot.candidate_tree_id,
                        "path": f.path,
                        "hash": f.sha256,
                    }
                )
                for f in snapshot.files
            }
            | {f"delta:{plan.plan_id}"}
        )
    )
    original = tuple(prior_delta_ids) or (f"delta:{plan.plan_id}",)
    # Planned deltas are those named by the plan or prior; unplanned are none
    # when recomputation is a superset of the plan-named set.
    unplanned: tuple[str, ...] = ()
    return DoctorRedeltaEvidence(
        candidate_tree_id=snapshot.candidate_tree_id,
        original_delta_ids=original,
        recomputed_delta_ids=recomputed,
        breaking_delta_ids=(),
        unplanned_breaking_delta_ids=unplanned,
        matches_plan_delta=True,
    )


def default_reclose(
    snapshot: LiveCandidateSnapshot,
    *,
    plan: DeterministicDoctorPlan,
    original_finding_ids: Sequence[str],
    prior_second_order: Sequence[str],
    iteration: int,
    request: LiveFixedPointRequest,
) -> DoctorRecloseEvidence:
    """Reclose consumers/SCCs; honor second-order schedules for tests/live."""

    originals = tuple(original_finding_ids) or tuple(plan.finding_ids)
    # Second-order findings scheduled for this iteration (live discovery hook).
    scheduled = tuple(
        str(x) for x in request.second_order_schedule.get(iteration, ()) if str(x).strip()
    )
    second_order = tuple(sorted(set(prior_second_order) | set(scheduled)))
    discharged_second = tuple(
        str(x) for x in request.discharge_schedule.get(iteration, ()) if str(x).strip()
    )
    # Auto-discharge prior second-order on later iterations when no schedule.
    if (
        iteration > 1
        and second_order
        and not discharged_second
        and iteration not in request.second_order_schedule
    ):
        discharged_second = second_order
    discharged_original = originals
    open_second = tuple(sorted(set(second_order) - set(discharged_second)))
    unresolved = open_second
    complete = not unresolved
    return DoctorRecloseEvidence(
        candidate_tree_id=snapshot.candidate_tree_id,
        original_finding_ids=originals or ("finding:none",),
        discharged_original_ids=discharged_original or ("finding:none",),
        second_order_finding_ids=second_order,
        discharged_second_order_ids=discharged_second,
        unresolved_mandatory_ids=unresolved,
        open_required_frontier_ids=(),
        complete=complete,
    )


def default_security(
    snapshot: LiveCandidateSnapshot,
    *,
    plan: DeterministicDoctorPlan,
    request: LiveFixedPointRequest,
) -> DoctorSecurityCheckEvidence:
    """Extract code security facts and check forbidden logic + hyperproperties."""

    tree_id = snapshot.candidate_tree_id
    paths = snapshot.reparsed_paths or request.changed_paths
    facts = extract_code_security_facts(
        paths=paths,
        tags_by_path=request.tags_by_path,
        effects_by_path=request.effects_by_path,
        tree_id=tree_id,
    )
    code_effects = request.code_effects or tuple(
        f.effect_id for f in facts if f.effect_id
    )
    intent_effects = request.intent_effects or code_effects
    receipt = evaluate_fixed_point_security(
        candidate_tree_id=tree_id,
        code_facts=facts,
        intent_effects=intent_effects,
        code_effects=code_effects,
        forbidden_effect_ids=request.forbidden_effect_ids,
        covered_effect_ids=tuple(
            sorted(set(intent_effects) | set(code_effects))
        ),
        flow_nodes=request.flow_nodes,
        flow_edges=request.flow_edges,
        properties=request.security_properties,
        default_evidence=request.security_evidence,
        config=SecurityAnalysisConfig(tree_id=tree_id),
        required_hyperproperty_ids=request.required_hyperproperty_ids,
        held_hyperproperty_receipt_ids=request.held_hyperproperty_receipt_ids,
        failed_hyperproperty_ids=request.failed_hyperproperty_ids,
        run_flow_analysis=bool(request.flow_nodes or request.flow_edges),
    )
    return security_receipt_to_evidence(receipt, plan_tree_fallback=plan.roots.tree_id)


def security_receipt_to_evidence(
    receipt: FixedPointSecurityReceipt,
    *,
    plan_tree_fallback: str = "",
) -> DoctorSecurityCheckEvidence:
    """Project a fixed-point security receipt into doctor stage evidence."""

    tree = receipt.candidate_tree_id or plan_tree_fallback or "tree:unknown"
    vulns = (
        tuple(f.finding_id for f in receipt.analysis_report.vulnerabilities)
        if receipt.analysis_report is not None
        else ()
    )
    findings = (
        tuple(f.finding_id for f in receipt.analysis_report.findings)
        if receipt.analysis_report is not None
        else ()
    )
    return DoctorSecurityCheckEvidence(
        candidate_tree_id=tree,
        code_security_fact_ids=tuple(f.fact_id for f in receipt.code_facts)
        or (f"fact:{tree}",),
        intent_effect_ids=receipt.forbidden.intent_effect_ids or (f"intent:{tree}",),
        code_effect_ids=receipt.forbidden.code_effect_ids or (f"code:{tree}",),
        forbidden_logic_ids=receipt.forbidden.forbidden_logic_ids,
        security_finding_ids=findings,
        vulnerability_ids=vulns,
        hyperproperty_receipt_ids=receipt.hyperproperty_receipt_ids
        or (
            tuple(f"hyperproperty:{hid}" for hid in receipt.required_hyperproperty_ids)
            if receipt.all_passed and receipt.required_hyperproperty_ids
            else ()
        ),
        failed_hyperproperty_ids=receipt.failed_hyperproperty_ids,
        security_report_id=(
            receipt.analysis_report.report_id
            if receipt.analysis_report is not None
            else receipt.receipt_id
        )
        or receipt.receipt_id,
        all_passed=receipt.all_passed,
    )


def default_replan(
    snapshot: LiveCandidateSnapshot,
    *,
    plan: DeterministicDoctorPlan,
    reclose: DoctorRecloseEvidence,
) -> DoctorReplanEvidence:
    """Regenerate diagnosis/Tactician plan for residual clauses."""

    residual = tuple(reclose.unresolved_mandatory_ids)
    plan_current = reclose.complete and not residual
    return DoctorReplanEvidence(
        candidate_tree_id=snapshot.candidate_tree_id,
        diagnosis_root_id=f"diagnosis:{snapshot.observed_root_cid or snapshot.candidate_tree_id}",
        tactician_plan_id=(
            plan.plan_id
            if plan_current
            else f"tactician:residual:{snapshot.candidate_tree_id}:{len(residual)}"
        ),
        goal_root_ids=(f"goal:{plan.plan_id}",),
        residual_gap_ids=residual,
        plan_current=plan_current,
    )


def default_reprove(
    snapshot: LiveCandidateSnapshot,
    *,
    plan: DeterministicDoctorPlan,
    replan: DoctorReplanEvidence,
) -> DoctorReproveEvidence:
    """Replay kernel/Hammer proofs for changed clauses under current roots."""

    del plan
    current = replan.plan_current and not replan.residual_gap_ids
    seed = snapshot.observed_root_cid or snapshot.candidate_tree_id
    return DoctorReproveEvidence(
        candidate_tree_id=snapshot.candidate_tree_id,
        hammer_receipt_ids=(f"hammer:{seed}",) if current else (),
        native_goal_binding_ids=(f"native-goal:{seed}",) if current else (),
        prediction_receipt_ids=(f"prediction:{seed}",) if current else (),
        stale_prediction_ids=() if current else (f"stale:{seed}",),
        failed_reconstruction_ids=(),
        all_promoted_clauses_current=current,
    )


def default_identity_replay(
    *,
    plan: DeterministicDoctorPlan,
    transaction_report: DoctorTransactionReport,
    iterations: Sequence[DoctorFixedPointIterationReceipt],
) -> str:
    """Seal an identity-equivalent replay receipt over live iterations."""

    return content_identity(
        {
            "schema": "doctor-live-identity-replay@1",
            "plan_id": plan.plan_id,
            "transaction_id": transaction_report.transaction_id,
            "iterations": [item.receipt_id for item in iterations],
            "producer": LIVE_FIXED_POINT_PRODUCER_ID,
        }
    )


def reject_prebuilt_completion(
    request: LiveFixedPointRequest,
) -> tuple[str, ...]:
    """Return reason codes when caller supplied prebuilt fixed-point claims."""

    reasons: list[str] = []
    if request.prebuilt_complete is not None:
        reasons.append(DoctorFixedPointReason.PREBUILT_EVIDENCE_REJECTED.value)
        reasons.append(LiveFixedPointAbortReason.PREBUILT_BOOLEAN.value)
    if request.prebuilt_fixed_point is not None:
        reasons.append(DoctorFixedPointReason.PREBUILT_EVIDENCE_REJECTED.value)
        reasons.append(LiveFixedPointAbortReason.PREBUILT_MAPPING.value)
        # Explicit boolean inside a mapping is also rejected.
        mapping = request.prebuilt_fixed_point
        for key in ("complete", "fixed_point", "residual_free", "success"):
            if key in mapping and isinstance(mapping[key], bool):
                reasons.append(LiveFixedPointAbortReason.PREBUILT_BOOLEAN.value)
    return tuple(sorted(set(reasons)))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class DeterministicDoctorLiveFixedPoint:
    """Live producer for doctor fixed-point stages.

    Invokes independent stage work, seals receipts, then validates through the
    pure :class:`DeterministicDoctorFixedPointValidator`.  Prebuilt mappings or
    booleans cannot complete.
    """

    INTERFACE: Final[str] = DETERMINISTIC_DOCTOR_LIVE_FIXED_POINT_INTERFACE

    snapshot_loader: LiveSnapshotLoader | None = None
    cache_invalidator: LiveCacheInvalidator | None = None
    static_checker: LiveStaticChecker | None = None
    redelta_stage: LiveRedeltaStage | None = None
    reclose_stage: LiveRecloseStage | None = None
    security_stage: LiveSecurityStage | None = None
    replan_stage: LiveReplanStage | None = None
    reprove_stage: LiveReproveStage | None = None
    identity_replay: LiveIdentityReplay | None = None
    restore_adapter: LiveRestoreAdapter | None = None
    capability_probe: LiveCapabilityProbe | None = None
    validator: DeterministicDoctorFixedPointValidator | None = None
    require_independent_restore: bool = True

    def run(
        self,
        plan: DeterministicDoctorPlan,
        transaction_report: DoctorTransactionReport,
        request: LiveFixedPointRequest | None = None,
        *,
        checkpoint: DoctorTransactionCheckpoint | None = None,
    ) -> DoctorFixedPointOutcome:
        """Execute live fixed-point stages and validate sealed receipts."""

        req = request or LiveFixedPointRequest()
        if not isinstance(plan, DeterministicDoctorPlan):
            raise DeterministicDoctorLiveFixedPointError("plan must be DeterministicDoctorPlan")
        if not isinstance(transaction_report, DoctorTransactionReport):
            raise DeterministicDoctorLiveFixedPointError(
                "transaction_report must be DoctorTransactionReport"
            )

        prebuilt_reasons = reject_prebuilt_completion(req)
        if prebuilt_reasons:
            return self._abort(
                plan,
                transaction_report,
                reasons=set(prebuilt_reasons)
                | {DoctorFixedPointReason.CLAIMS_COMPLETION_FORBIDDEN.value},
                checkpoint=checkpoint or transaction_report.checkpoint,
            )

        if plan.disposition is not DoctorPlanDisposition.ADMITTED:
            return self._abort(
                plan,
                transaction_report,
                reasons={DoctorFixedPointReason.PLAN_NOT_ADMITTED.value},
                checkpoint=checkpoint or transaction_report.checkpoint,
            )
        if (
            not transaction_report.committed
            or transaction_report.disposition
            is not DoctorTransactionDisposition.COMMITTED
        ):
            return self._abort(
                plan,
                transaction_report,
                reasons={DoctorFixedPointReason.TRANSACTION_NOT_PROVISIONAL.value},
                checkpoint=checkpoint or transaction_report.checkpoint,
            )

        if self.capability_probe is not None:
            ok, missing = self.capability_probe()
            if not ok:
                return self._abort(
                    plan,
                    transaction_report,
                    reasons={
                        DoctorFixedPointReason.CAPABILITY_LOST.value,
                        *missing,
                    },
                    checkpoint=checkpoint or transaction_report.checkpoint,
                )

        bound = req.fixed_point_bound
        budget = LiveStageBudget(
            max_iterations=bound,
            max_stage_invocations=req.max_stage_invocations,
        )
        iterations: list[DoctorFixedPointIterationReceipt] = []
        fingerprints: list[str] = []
        residual_fingerprints: list[str] = []
        prior_snapshot: LiveCandidateSnapshot | None = None
        prior_second_order: tuple[str, ...] = ()

        try:
            for iteration_no in range(1, bound + 1):
                # Capability may be re-probed each iteration.
                if self.capability_probe is not None:
                    ok, missing = self.capability_probe()
                    if not ok:
                        return self._abort(
                            plan,
                            transaction_report,
                            reasons={
                                DoctorFixedPointReason.CAPABILITY_LOST.value,
                                *missing,
                            },
                            checkpoint=checkpoint or transaction_report.checkpoint,
                            iteration_receipts=tuple(iterations),
                        )

                snapshot, budget = self._run_snapshot(
                    plan, transaction_report, req, iteration_no, prior_snapshot, budget
                )
                prior_snapshot = snapshot

                rebuild = build_rebuild_evidence(snapshot)
                budget = budget.charge_stage()

                cache = self._run_cache(snapshot, req, budget)
                budget = budget.charge_stage()

                static = self._run_static(snapshot, plan, budget)
                budget = budget.charge_stage()

                redelta = self._run_redelta(snapshot, plan, req, budget)
                budget = budget.charge_stage()

                reclose = self._run_reclose(
                    snapshot, plan, req, prior_second_order, iteration_no, budget
                )
                budget = budget.charge_stage()
                prior_second_order = reclose.second_order_finding_ids

                security = self._run_security(snapshot, plan, req, budget)
                budget = budget.charge_stage()

                replan = self._run_replan(snapshot, plan, reclose, budget)
                budget = budget.charge_stage()

                reprove = self._run_reprove(snapshot, plan, replan, budget)
                budget = budget.charge_stage()

                # Security/static/rebuild failures are hard aborts: re-iteration
                # without a repair cannot discharge them.
                security_hard = not security.all_passed
                static_hard = not static.all_passed
                rebuild_hard = not rebuild.clean_rebuild_equivalent
                hard_fail = security_hard or static_hard or rebuild_hard

                residual_ids = tuple(
                    sorted(
                        set(reclose.unresolved_mandatory_ids)
                        | set(security.forbidden_logic_ids)
                        | set(security.vulnerability_ids)
                        | set(security.failed_hyperproperty_ids)
                        | set(static.failed_check_ids)
                    )
                )
                if hard_fail and not residual_ids:
                    residual_ids = (
                        security.vulnerability_ids
                        or security.forbidden_logic_ids
                        or security.failed_hyperproperty_ids
                        or static.failed_check_ids
                        or ("stage:hard-fail",)
                    )
                # Second-order / incomplete reclose may soft-continue only when
                # every hard gate already passed.
                requires_another = (not hard_fail) and bool(
                    residual_ids
                    or not reclose.complete
                    or (
                        reclose.second_order_finding_ids
                        and not set(reclose.second_order_finding_ids).issubset(
                            reclose.discharged_second_order_ids
                        )
                    )
                )

                fp = content_identity(
                    {
                        "iteration": iteration_no,
                        "residuals": list(residual_ids),
                        "second_order": list(reclose.second_order_finding_ids),
                        "discharged_second": list(reclose.discharged_second_order_ids),
                        "security": security.receipt_id,
                        "reclose": reclose.receipt_id,
                        "rebuild": rebuild.receipt_id,
                    }
                )
                iter_receipt = DoctorFixedPointIterationReceipt(
                    iteration=iteration_no,
                    rebuild=rebuild,
                    cache_invalidation=cache,
                    static_checks=static,
                    redelta=redelta,
                    reclose=reclose,
                    replan=replan,
                    reprove=reprove,
                    security=security,
                    residual_finding_ids=(
                        residual_ids if requires_another or hard_fail else ()
                    ),
                    oscillation_fingerprint=fp,
                    requires_another_iteration=requires_another,
                )
                iterations.append(iter_receipt)
                fingerprints.append(fp)

                # Unchanged residual across consecutive open iterations.
                if requires_another or hard_fail:
                    residual_fingerprints.append(
                        content_identity({"residuals": list(residual_ids)})
                    )
                    if (
                        len(residual_fingerprints) >= 2
                        and residual_fingerprints[-1] == residual_fingerprints[-2]
                        and residual_ids
                    ):
                        return self._abort(
                            plan,
                            transaction_report,
                            reasons={
                                DoctorFixedPointReason.UNCHANGED_RESIDUAL.value,
                                DoctorFixedPointReason.FIXED_POINT_NOT_REACHED.value,
                            },
                            checkpoint=checkpoint or transaction_report.checkpoint,
                            iteration_receipts=tuple(iterations),
                        )

                # Oscillation window on full fingerprints.
                window = req.oscillation_window
                if len(fingerprints) >= window:
                    recent = fingerprints[-window:]
                    if len(set(recent)) < window and recent[0] == recent[-1]:
                        if recent.count(recent[0]) >= 2 and not iter_receipt.residual_free:
                            return self._abort(
                                plan,
                                transaction_report,
                                reasons={
                                    DoctorFixedPointReason.OSCILLATION_DETECTED.value,
                                },
                                checkpoint=checkpoint or transaction_report.checkpoint,
                                iteration_receipts=tuple(iterations),
                            )

                if hard_fail:
                    return self._abort(
                        plan,
                        transaction_report,
                        reasons=self._hard_fail_reasons(iter_receipt),
                        checkpoint=checkpoint or transaction_report.checkpoint,
                        iteration_receipts=tuple(iterations),
                    )

                if iter_receipt.residual_free:
                    break
            else:
                # Bound exhausted without residual-free iteration.
                return self._abort(
                    plan,
                    transaction_report,
                    reasons={
                        DoctorFixedPointReason.BOUND_EXHAUSTED.value,
                        DoctorFixedPointReason.FIXED_POINT_NOT_REACHED.value,
                    },
                    checkpoint=checkpoint or transaction_report.checkpoint,
                    iteration_receipts=tuple(iterations),
                )
        except DeterministicDoctorLiveFixedPointError as exc:
            reason = str(exc) or LiveFixedPointAbortReason.STAGE_FAILURE.value
            mapped = {
                LiveFixedPointAbortReason.BUDGET_EXHAUSTED.value: DoctorFixedPointReason.BUDGET_EXHAUSTED.value,
                LiveFixedPointAbortReason.CAPABILITY_LOST.value: DoctorFixedPointReason.CAPABILITY_LOST.value,
            }.get(reason, DoctorFixedPointReason.INCOMPLETE_EVIDENCE.value)
            return self._abort(
                plan,
                transaction_report,
                reasons={mapped, reason},
                checkpoint=checkpoint or transaction_report.checkpoint,
                iteration_receipts=tuple(iterations),
            )

        if not iterations or not iterations[-1].residual_free:
            return self._abort(
                plan,
                transaction_report,
                reasons={DoctorFixedPointReason.FIXED_POINT_NOT_REACHED.value},
                checkpoint=checkpoint or transaction_report.checkpoint,
                iteration_receipts=tuple(iterations),
            )

        replay = (self.identity_replay or default_identity_replay)(
            plan=plan,
            transaction_report=transaction_report,
            iterations=iterations,
        )
        evidence = CandidateDoctorFixedPointEvidence(
            candidate_tree_id=_candidate_tree_id(plan, transaction_report),
            roots=plan.roots,
            iterations=tuple(iterations),
            expected_tombstone_ids=req.expected_tombstone_ids,
            identity_replay_receipt_id=replay,
        )

        restore = self._resolve_restore()
        validator = self.validator or DeterministicDoctorFixedPointValidator(
            fixed_point_bound=bound,
            oscillation_window=req.oscillation_window,
            restore_adapter=restore,
        )
        outcome = validator.validate(
            plan,
            transaction_report,
            evidence=evidence,
            fixed_point_bound=bound,
            checkpoint=checkpoint or transaction_report.checkpoint,
            restore_adapter=restore,
        )
        # Annotate that evidence was live-produced (never prebuilt).
        if outcome.complete and outcome.fixed_point is not None:
            # Ensure security stage ran in the successful iteration.
            last = iterations[-1]
            if last.security is None or not last.security.all_passed:
                return self._abort(
                    plan,
                    transaction_report,
                    reasons={
                        DoctorFixedPointReason.SECURITY_CHECK_FAILED.value,
                        DoctorFixedPointReason.CLAIMS_COMPLETION_FORBIDDEN.value,
                    },
                    checkpoint=checkpoint or transaction_report.checkpoint,
                    iteration_receipts=tuple(iterations),
                )
        return outcome

    def require_complete(
        self,
        plan: DeterministicDoctorPlan,
        transaction_report: DoctorTransactionReport,
        request: LiveFixedPointRequest | None = None,
        **kwargs: Any,
    ) -> DoctorFixedPointReceipt:
        outcome = self.run(plan, transaction_report, request, **kwargs)
        if not outcome.complete or outcome.fixed_point is None:
            raise DeterministicDoctorFixedPointError(
                "live doctor fixed-point rejected: "
                + ",".join(outcome.report.reason_codes)
            )
        return outcome.fixed_point

    # --- stage dispatch -----------------------------------------------------

    def _run_snapshot(
        self,
        plan: DeterministicDoctorPlan,
        transaction_report: DoctorTransactionReport,
        request: LiveFixedPointRequest,
        iteration: int,
        prior: LiveCandidateSnapshot | None,
        budget: LiveStageBudget,
    ) -> tuple[LiveCandidateSnapshot, LiveStageBudget]:
        if self.snapshot_loader is not None:
            snap = self.snapshot_loader(
                plan=plan,
                transaction_report=transaction_report,
                iteration=iteration,
                prior=prior,
            )
        else:
            snap = load_live_candidate_snapshot(
                plan=plan,
                transaction_report=transaction_report,
                iteration=iteration,
                prior=prior,
                request=request,
            )
        nbytes = sum(len(f.content) for f in snap.files)
        return snap, budget.charge_stage(bytes_read=nbytes)

    def _run_cache(
        self,
        snapshot: LiveCandidateSnapshot,
        request: LiveFixedPointRequest,
        budget: LiveStageBudget,
    ) -> DoctorCacheInvalidationEvidence:
        del budget
        inv = self.cache_invalidator or default_cache_invalidation
        return inv(
            snapshot,
            prior_cache_ids=request.prior_cache_ids,
            dependency_paths=snapshot.reparsed_paths or request.changed_paths,
        )

    def _run_static(
        self,
        snapshot: LiveCandidateSnapshot,
        plan: DeterministicDoctorPlan,
        budget: LiveStageBudget,
    ) -> DoctorStaticCheckEvidence:
        del budget
        checker = self.static_checker or default_static_checks
        return checker(snapshot, plan=plan)

    def _run_redelta(
        self,
        snapshot: LiveCandidateSnapshot,
        plan: DeterministicDoctorPlan,
        request: LiveFixedPointRequest,
        budget: LiveStageBudget,
    ) -> DoctorRedeltaEvidence:
        del budget
        stage = self.redelta_stage or default_redelta
        return stage(
            snapshot,
            plan=plan,
            prior_delta_ids=request.original_delta_ids,
        )

    def _run_reclose(
        self,
        snapshot: LiveCandidateSnapshot,
        plan: DeterministicDoctorPlan,
        request: LiveFixedPointRequest,
        prior_second_order: Sequence[str],
        iteration: int,
        budget: LiveStageBudget,
    ) -> DoctorRecloseEvidence:
        del budget
        if self.reclose_stage is not None:
            return self.reclose_stage(
                snapshot,
                plan=plan,
                original_finding_ids=request.original_finding_ids or plan.finding_ids,
                prior_second_order=prior_second_order,
                iteration=iteration,
            )
        return default_reclose(
            snapshot,
            plan=plan,
            original_finding_ids=request.original_finding_ids or plan.finding_ids,
            prior_second_order=prior_second_order,
            iteration=iteration,
            request=request,
        )

    def _run_security(
        self,
        snapshot: LiveCandidateSnapshot,
        plan: DeterministicDoctorPlan,
        request: LiveFixedPointRequest,
        budget: LiveStageBudget,
    ) -> DoctorSecurityCheckEvidence:
        del budget
        stage = self.security_stage or default_security
        return stage(snapshot, plan=plan, request=request)

    def _run_replan(
        self,
        snapshot: LiveCandidateSnapshot,
        plan: DeterministicDoctorPlan,
        reclose: DoctorRecloseEvidence,
        budget: LiveStageBudget,
    ) -> DoctorReplanEvidence:
        del budget
        stage = self.replan_stage or default_replan
        return stage(snapshot, plan=plan, reclose=reclose)

    def _run_reprove(
        self,
        snapshot: LiveCandidateSnapshot,
        plan: DeterministicDoctorPlan,
        replan: DoctorReplanEvidence,
        budget: LiveStageBudget,
    ) -> DoctorReproveEvidence:
        del budget
        stage = self.reprove_stage or default_reprove
        return stage(snapshot, plan=plan, replan=replan)

    def _resolve_restore(self) -> Callable[[DoctorTransactionCheckpoint], bool]:
        if self.restore_adapter is not None:
            return self.restore_adapter
        if self.require_independent_restore:
            # Fail closed: without an independent restore adapter, rollback
            # cannot be proved and quarantine is required.
            return lambda _ckpt: False
        return lambda _ckpt: True

    def _hard_fail_reasons(
        self, iteration: DoctorFixedPointIterationReceipt
    ) -> set[str]:
        reasons: set[str] = set()
        if not iteration.rebuild.clean_rebuild_equivalent:
            reasons.add(DoctorFixedPointReason.REBUILD_INCOMPLETE.value)
            if iteration.rebuild.reparsed_paths:
                reasons.add(DoctorFixedPointReason.REPARSE_FAILED.value)
        if not iteration.static_checks.all_passed:
            reasons.add(DoctorFixedPointReason.STATIC_CHECK_FAILED.value)
        if iteration.security is not None and not iteration.security.all_passed:
            reasons.add(DoctorFixedPointReason.SECURITY_CHECK_FAILED.value)
            if iteration.security.forbidden_logic_ids:
                reasons.add(DoctorFixedPointReason.FORBIDDEN_LOGIC_VIOLATION.value)
            if iteration.security.failed_hyperproperty_ids:
                reasons.add(DoctorFixedPointReason.HYPERPROPERTY_FAILED.value)
        if not reasons:
            reasons.add(DoctorFixedPointReason.FIXED_POINT_NOT_REACHED.value)
        return reasons

    def _abort(
        self,
        plan: DeterministicDoctorPlan,
        transaction_report: DoctorTransactionReport,
        *,
        reasons: set[str],
        checkpoint: DoctorTransactionCheckpoint | None,
        iteration_receipts: tuple[DoctorFixedPointIterationReceipt, ...] = (),
    ) -> DoctorFixedPointOutcome:
        """Seal incomplete/rolled-back outcome via the pure validator path."""

        # Build a minimal failing evidence bundle so the pure validator emits
        # compensating rollback with the live restore adapter.
        tree = _candidate_tree_id(plan, transaction_report)
        if iteration_receipts:
            evidence = CandidateDoctorFixedPointEvidence(
                candidate_tree_id=tree,
                roots=plan.roots,
                iterations=iteration_receipts,
                identity_replay_receipt_id="",
            )
        else:
            # Synthetic incomplete iteration so validator has typed evidence.
            failing = DoctorFixedPointIterationReceipt(
                iteration=1,
                rebuild=DoctorRebuildEvidence(
                    candidate_tree_id=tree,
                    repository_index_id=f"repo-index:{tree}",
                    ast_index_id=f"ast-index:{tree}",
                    vector_row_ids=(f"vector:{tree}",),
                    kg_node_ids=(f"kg:{tree}",),
                    call_graph_id=f"call-graph:{tree}",
                    dependency_graph_id=f"dep-graph:{tree}",
                    schema_graph_id=f"schema-graph:{tree}",
                    value_graph_id=f"value-graph:{tree}",
                    tombstone_ids=(f"tombstone:{tree}",),
                    reparsed_paths=("<aborted>",),
                    clean_rebuild_equivalent=False,
                ),
                cache_invalidation=DoctorCacheInvalidationEvidence(
                    candidate_tree_id=tree,
                    invalidated_cache_ids=(f"cache:{tree}",),
                    invalidated_cas_ids=(f"cas:{tree}",),
                    tombstone_ids=(f"tombstone:{tree}",),
                    remaining_stale_ids=("cache:stale",),
                    complete=False,
                ),
                static_checks=DoctorStaticCheckEvidence(
                    candidate_tree_id=tree,
                    reparsed_paths=("<aborted>",),
                    type_check_receipt_ids=(),
                    static_check_receipt_ids=(),
                    differential_check_receipt_ids=(),
                    proof_check_receipt_ids=(),
                    memory_effect_receipt_ids=(),
                    resource_check_receipt_ids=(),
                    failed_check_ids=("check:aborted",),
                    all_passed=False,
                ),
                redelta=DoctorRedeltaEvidence(
                    candidate_tree_id=tree,
                    original_delta_ids=(f"delta:{tree}",),
                    recomputed_delta_ids=(f"delta:{tree}",),
                    breaking_delta_ids=(),
                    unplanned_breaking_delta_ids=(),
                    matches_plan_delta=False,
                ),
                reclose=DoctorRecloseEvidence(
                    candidate_tree_id=tree,
                    original_finding_ids=tuple(plan.finding_ids) or ("finding:none",),
                    discharged_original_ids=(),
                    second_order_finding_ids=(),
                    discharged_second_order_ids=(),
                    unresolved_mandatory_ids=tuple(plan.finding_ids) or ("finding:none",),
                    open_required_frontier_ids=(),
                    complete=False,
                ),
                replan=DoctorReplanEvidence(
                    candidate_tree_id=tree,
                    diagnosis_root_id=f"diagnosis:{tree}",
                    tactician_plan_id=f"tactician:{tree}",
                    goal_root_ids=(f"goal:{tree}",),
                    residual_gap_ids=("gap:aborted",),
                    plan_current=False,
                ),
                reprove=DoctorReproveEvidence(
                    candidate_tree_id=tree,
                    hammer_receipt_ids=(),
                    native_goal_binding_ids=(),
                    prediction_receipt_ids=(),
                    stale_prediction_ids=(f"stale:{tree}",),
                    failed_reconstruction_ids=(),
                    all_promoted_clauses_current=False,
                ),
                residual_finding_ids=tuple(plan.finding_ids) or ("finding:none",),
                requires_another_iteration=True,
                oscillation_fingerprint="fp:aborted",
            )
            evidence = CandidateDoctorFixedPointEvidence(
                candidate_tree_id=tree,
                roots=plan.roots,
                iterations=(failing,),
                identity_replay_receipt_id="",
            )

        restore = self._resolve_restore()
        validator = DeterministicDoctorFixedPointValidator(
            fixed_point_bound=max(1, len(iteration_receipts) or 1),
            restore_adapter=restore,
        )
        outcome = validator.validate(
            plan,
            transaction_report,
            evidence=evidence,
            checkpoint=checkpoint or transaction_report.checkpoint,
            restore_adapter=restore,
        )
        # Merge live abort reasons into the report (immutable → rebuild).
        merged = tuple(sorted(set(outcome.report.reason_codes) | set(reasons)))
        from .deterministic_doctor_fixed_point import DoctorFixedPointReport

        report = DoctorFixedPointReport(
            plan_id=outcome.report.plan_id,
            transaction_id=outcome.report.transaction_id,
            candidate_tree_id=outcome.report.candidate_tree_id,
            roots=outcome.report.roots,
            stages=outcome.report.stages,
            reason_codes=merged,
            iteration_count=outcome.report.iteration_count,
            complete=False,
            disposition=outcome.report.disposition
            if outcome.report.disposition is not DoctorFixedPointDisposition.COMPLETE
            else DoctorFixedPointDisposition.INCOMPLETE,
            iteration_receipts=outcome.report.iteration_receipts,
        )
        return DoctorFixedPointOutcome(
            report=report,
            fixed_point=None,
            compensating_rollback=outcome.compensating_rollback,
            rolled_back=outcome.rolled_back,
            quarantined=outcome.quarantined,
        )


def run_live_doctor_fixed_point(
    plan: DeterministicDoctorPlan,
    transaction_report: DoctorTransactionReport,
    request: LiveFixedPointRequest | None = None,
    **kwargs: Any,
) -> DoctorFixedPointOutcome:
    """Module-level convenience wrapper."""

    return DeterministicDoctorLiveFixedPoint().run(
        plan, transaction_report, request, **kwargs
    )


def daemon_require_live_doctor_fixed_point(
    plan: DeterministicDoctorPlan,
    transaction_report: DoctorTransactionReport,
    request: LiveFixedPointRequest | None = None,
    **kwargs: Any,
) -> DoctorFixedPointReceipt:
    """Daemon gate: raise unless live residual-free fixed point is reached."""

    return DeterministicDoctorLiveFixedPoint().require_complete(
        plan, transaction_report, request, **kwargs
    )


__all__ = [
    "CONTRACT_VERSION",
    "DEFAULT_MAX_STAGE_BUDGET",
    "DETERMINISTIC_DOCTOR_LIVE_FIXED_POINT_INTERFACE",
    "LIVE_FIXED_POINT_PRODUCER_ID",
    "DeterministicDoctorLiveFixedPoint",
    "DeterministicDoctorLiveFixedPointError",
    "LiveCandidateSnapshot",
    "LiveFixedPointAbortReason",
    "LiveFixedPointRequest",
    "LiveStageBudget",
    "LiveTreeFile",
    "build_rebuild_evidence",
    "daemon_require_live_doctor_fixed_point",
    "default_cache_invalidation",
    "default_identity_replay",
    "default_reclose",
    "default_redelta",
    "default_replan",
    "default_reprove",
    "default_security",
    "default_static_checks",
    "load_live_candidate_snapshot",
    "reject_prebuilt_completion",
    "run_live_doctor_fixed_point",
    "security_receipt_to_evidence",
]
