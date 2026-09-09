"""Independent contract tests for SPAR-041 shadow_apply rollout gate."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring import rollout as spar040
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.rollout import (
    AUTHORITY,
    AUTHORITY_OWNER,
    BOOTSTRAP_MODE,
    CANDIDATE_DISPOSITIONS,
    CandidateComparison,
    CandidateDisposition,
    DECLARED_HARD_CONSTRAINTS,
    DECLARED_RECEIPT_FLOOR,
    DECLARED_ROLLOUT_MODES,
    DECLARED_TERMINAL_KINDS,
    DRY_RUN_IS_DETERMINISTIC,
    DRY_RUN_MUTATES,
    DUCKLAKE_IS_AUTHORITY,
    EXISTING_ADAPTER_AUTHORITIES,
    FALSE_UNSAFE_CANDIDATES_RETAINED,
    FORBIDDEN_ROLLOUT_NAMES,
    GOAL_ID,
    HARD_CONSTRAINTS,
    IDENTITY_EXCLUDED_FIELDS,
    MARKDOWN_IS_NOT_COMPLETION,
    MODE_CONTRACTS,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    ModeContract,
    NEGATIVE_EVIDENCE_RETAINED,
    NETWORK_DENIED,
    NETWORK_DENY,
    PROGRAM,
    PROGRESSIVE_MODES,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_SOURCE_REQUIRED,
    RECEIPT_FLOOR,
    ROLLOUT_BASELINE_SCHEMA,
    ROLLOUT_CONTRACT_VERSION,
    ROLLOUT_MODES,
    RolloutError,
    RolloutMode,
    SHADOW_PLAN_MODE,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    ShadowPlanCandidate,
    TEST_PASS_IS_NOT_COMPLETION,
    TerminalKind,
    TypedTerminal,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_MAY_CHANGE_MODE,
    WORKER_SELF_APPROVAL,
    compare_and_retain_candidates,
    rollout_cid_profile,
    sealed_rollout_baseline,
)


ROOT = Path(__file__).resolve().parents[3]
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = ("test/api/semantic_refactoring/test_shadow_apply.py",)
PROTECTED_PATHS = (
    ".gitignore",
    "benchmarks/agent_supervisor/semantic_refactoring/preregistration.json",
    "config/agent_supervisor_semantic_preserving_remodularization_scheduler.json",
    "config/semantic_preserving_autonomous_remodularization_dependencies.seal.json",
    "docs/architecture/SEMANTIC_PRESERVING_AUTONOMOUS_REMODULARIZATION_PLAN.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization.objectives.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization.todo.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/authority_matrix.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/benchmark_preregistration.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/dynamic_python_risk_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/identity_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/interface_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/overlap_gap_matrix.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/repository_baseline.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/rollout_baseline.json",
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/rollout.py",
    "scripts/materialize_semantic_preserving_remodularization_program.py",
    "scripts/ops/agent_supervisor/semantic_preserving_remodularization.py",
    "scripts/validate_semantic_preserving_remodularization_board.py",
    "scripts/validate_semantic_preserving_remodularization_dependencies.py",
    "test/api/semantic_refactoring/test_bootstrap_controls.py",
    "test/api/semantic_refactoring/test_shadow_plan.py",
)
CAPSULE_TYPES = (
    "FunctionSemanticCapsule",
    "MethodSemanticCapsule",
    "ClassSemanticCapsule",
    "TopLevelBlockCapsule",
    "ModuleSemanticCapsule",
    "PackageSemanticCapsule",
    "CallsiteSemanticCapsule",
    "StateOwnerCapsule",
    "RegistrationCapsule",
    "ResourceLifecycleCapsule",
    "RolloutStore",
)
TREE_ID = "fbc6fa1ddefb2f9ecb7b5c718d618e3b60aa3051"
OTHER_TREE = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
BASELINE_PATH = (
    ROOT
    / "docs"
    / "architecture"
    / "semantic_preserving_autonomous_remodularization_inventory"
    / "rollout_baseline.json"
)

TASK_ID: str = "SPAR-041"
PREDECESSOR_TASK_IDS: tuple[str, ...] = ("SPAR-040",)
ANALYZER_ID: str = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.shadow_apply@1"
)
SHADOW_APPLY_GATE_INTERFACE: str = "ShadowApplyGate@1"
SHADOW_APPLY_RECEIPT_INTERFACE: str = "ShadowApplyReceipt@1"
HYPOTHETICAL_TRANSITION_INTERFACE: str = "HypotheticalRefactorTransition@1"
SHADOW_APPLY_GATE_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/shadow-apply-gate@1"
)
SHADOW_APPLY_RECEIPT_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/shadow-apply-receipt@1"
)
HYPOTHETICAL_TRANSITION_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/hypothetical-refactor-transition@1"
)
SHADOW_APPLY_MODE: str = "shadow_apply"
ALLOWED_CURRENT_MODES: frozenset[str] = frozenset(
    {SHADOW_PLAN_MODE, SHADOW_APPLY_MODE}
)
SHADOW_APPLY_SOURCE_MUTATION: str = "disposable_worktree_only"
SHADOW_APPLY_MERGE: bool = False
SHADOW_APPLY_PROMOTES_ROOT: bool = False
SHADOW_APPLY_INFLUENCES_ROUTING: bool = False
SHADOW_APPLY_WRITES_REPOSITORY: bool = False
GATE_CAN_AUTHORIZE_TRANSITION: bool = False
GATE_CAN_AUTHORIZE_COMPLETION: bool = False
GATE_CAN_CREATE_AUTHORITY: bool = False
GATE_CAN_CHANGE_MODE: bool = False
GATE_WRITES_REPOSITORY: bool = False
GATE_IS_NOMINATION_ONLY: bool = True
FULL_VALIDATION_REQUIRED: bool = True
COMPLETE_ANALYSIS_REQUIRED: bool = True
COMPLETE_PLANNING_REQUIRED: bool = True
DISPOSABLE_WORKTREE_ONLY: bool = True
ISOLATED_WORKTREE_REQUIRED: bool = True
HYPOTHETICAL_TRANSITION_ONLY: bool = True
BOUNDED_PACKET_ONLY: bool = True

_APPLY_AUTHORITY_FLAGS: tuple[str, ...] = (
    "can_authorize_transition",
    "can_authorize_completion",
    "can_create_authority",
    "can_change_mode",
    "projection_is_authority",
    "writes_repository",
    "worker_self_approval",
    "worker_may_change_mode",
    "influences_routing",
    "merge",
    "root_promoted",
)


class ShadowApplyStatus(str, Enum):
    NOMINATED_SHADOW_APPLY = "nominated_shadow_apply"
    TYPED_TERMINAL = "typed_terminal"


DECLARED_GATE_STATUSES: frozenset[str] = frozenset(
    item.value for item in ShadowApplyStatus
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _candidate(
    label: str,
    disposition: str = "admitted",
    violations: tuple[str, ...] = (),
    score: int = 0,
) -> dict[str, Any]:
    return {
        "candidate_cid": _cid(f"candidate:{label}"),
        "disposition": disposition,
        "hard_constraint_violations": list(violations),
        "score": score,
    }


def shadow_apply_gate_descriptor() -> dict[str, Any]:
    return {
        "schema": SHADOW_APPLY_GATE_SCHEMA,
        "interface": SHADOW_APPLY_GATE_INTERFACE,
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "analyzer_id": ANALYZER_ID,
        "predecessor_task_ids": list(PREDECESSOR_TASK_IDS),
        "authority_owner": AUTHORITY_OWNER,
        "nomination_only": True,
        "writes_repository": False,
        "worker_may_change_mode": False,
        "source_mutation": SHADOW_APPLY_SOURCE_MUTATION,
        "merge": False,
        "influences_routing": False,
        "promotes_root": False,
        "nominated_mode": SHADOW_APPLY_MODE,
        "gate_task": TASK_ID,
        "network": NETWORK_DENY,
        "disposable_worktree_only": True,
        "hypothetical_transition_only": True,
        "full_validation_required": True,
        "false_unsafe_candidates_retained": True,
    }


def provider_free_exports() -> tuple[str, ...]:
    return (
        "ShadowApplyGate",
        "ShadowApplyReceipt",
        "activate_shadow_apply",
        "compare_and_retain_candidates",
        "dry_run_shadow_apply",
        "run_shadow_apply",
    )


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & spar040._FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise RolloutError(
            f"shadow_apply gate must not define competing types: {sorted(overlap)}"
        )
    spar040.assert_not_competing_capsule_family()


def _pop_apply_authority_flags(payload: dict[str, Any], name: str) -> None:
    claimed_mutation = payload.pop("source_mutation", SHADOW_APPLY_SOURCE_MUTATION)
    if claimed_mutation != SHADOW_APPLY_SOURCE_MUTATION:
        raise RolloutError(
            f"{name} source_mutation is disposable_worktree_only"
        )
    for flag in _APPLY_AUTHORITY_FLAGS:
        if flag not in payload:
            continue
        claimed = payload.pop(flag)
        if claimed is not False:
            raise RolloutError(f"{name} cannot claim {flag}")


def _require_disposable_isolated_worktree(payload: Mapping[str, Any]) -> tuple[str, str, str]:
    worktree_id = spar040._text(payload.get("worktree_id", ""), "worktree_id", empty=True)
    lease_id = spar040._text(payload.get("lease_id", ""), "lease_id", empty=True)
    fence_id = spar040._text(payload.get("fence_id", ""), "fence_id", empty=True)
    disposable = spar040._bool(
        payload.get("worktree_disposable", False), "worktree_disposable"
    )
    isolated = spar040._bool(
        payload.get("worktree_isolated", False), "worktree_isolated"
    )
    if not worktree_id or not lease_id or not fence_id or not disposable or not isolated:
        raise RolloutError("disposable isolated worktree required")
    return worktree_id, lease_id, fence_id


def _reject_worker_mode_change(payload: Mapping[str, Any]) -> str:
    if payload.get("worker_may_change_mode", False) is not False:
        raise RolloutError("workers cannot change rollout mode")
    requested = payload.get("requested_mode", "")
    if requested not in (None, "", SHADOW_APPLY_MODE):
        raise RolloutError("workers cannot change rollout mode")
    current = spar040._mode(payload.get("current_mode", BOOTSTRAP_MODE), "current_mode")
    if current not in ALLOWED_CURRENT_MODES:
        raise RolloutError("SPAR-041 cannot skip or regress sealed rollout gates")
    return current


def _reject_merge_root_routing(payload: Mapping[str, Any]) -> None:
    if payload.get("merge", False) is not False:
        raise RolloutError("shadow_apply cannot merge")
    if payload.get("merge_worktree", False) is not False:
        raise RolloutError("shadow_apply cannot merge")
    if payload.get("writes_repository", False) is not False:
        raise RolloutError("shadow_apply cannot write the repository")
    if payload.get("apply_to_repository", False) is not False:
        raise RolloutError("shadow_apply cannot write the repository")
    if payload.get("promote_root", False) is not False:
        raise RolloutError("shadow_apply cannot promote a root")
    if payload.get("routing_influence", False) is not False:
        raise RolloutError("shadow_apply cannot influence routing")
    if payload.get("influences_routing", False) is not False:
        raise RolloutError("shadow_apply cannot influence routing")
    proposed_route = payload.get("proposed_route_decision_cid", "")
    route = payload.get("route_decision_cid", "")
    if proposed_route not in (None, "", route):
        raise RolloutError("shadow_apply cannot influence routing")
    post = payload.get("post_world_root_cid", "")
    pre = payload.get("pre_world_root_cid", "")
    if post not in (None, "", pre):
        raise RolloutError("shadow_apply cannot mutate or promote the world root")
    resulting = payload.get(
        "resulting_root_generation", payload.get("expected_root_generation")
    )
    expected = payload.get("expected_root_generation")
    if resulting not in (None, expected):
        raise RolloutError("shadow_apply cannot promote root generation")


def _reject_unbounded_mutation(payload: Mapping[str, Any], *, mutate: bool) -> None:
    claimed = payload.get("source_mutation", False)
    if claimed is True:
        raise RolloutError("shadow_apply cannot mutate source outside a disposable worktree")
    if claimed not in (False, "", None, SHADOW_APPLY_SOURCE_MUTATION):
        raise RolloutError("shadow_apply cannot mutate source outside a disposable worktree")
    if mutate is True and claimed not in (False, "", None, SHADOW_APPLY_SOURCE_MUTATION):
        raise RolloutError("shadow_apply cannot mutate source outside a disposable worktree")
    if payload.get("packet_bounded", True) is not True:
        raise RolloutError("shadow_apply applies bounded packets only")
    if payload.get("unrestricted_diff", False) is not False:
        raise RolloutError("shadow_apply applies bounded packets only")


def _require_complete_analysis_planning_validation(payload: Mapping[str, Any]) -> None:
    if spar040._bool(payload.get("analysis_complete", False), "analysis_complete") is not True:
        raise RolloutError("shadow_apply requires complete analysis")
    if spar040._bool(payload.get("planning_complete", False), "planning_complete") is not True:
        raise RolloutError("shadow_apply requires complete planning")
    if spar040._bool(payload.get("validation_complete", False), "validation_complete") is not True:
        raise RolloutError("shadow_apply requires full validation")


def _hypothetical_transition_cid(payload: Mapping[str, Any]) -> str:
    return cid_for_dag_json(
        {
            "schema": HYPOTHETICAL_TRANSITION_SCHEMA,
            "interface": HYPOTHETICAL_TRANSITION_INTERFACE,
            "tree_id": payload["tree_id"],
            "transformation_packet_cid": payload["transformation_packet_cid"],
            "worktree_id": payload["worktree_id"],
            "hypothetical": True,
            "accepted": False,
            "merged": False,
            "root_promoted": False,
        }
    )


def _resolve_transition(
    payload: Mapping[str, Any],
    *,
    mutate: bool,
) -> tuple[str, bool]:
    claimed = payload.get("refactor_transition_cid", "")
    hypothetical = payload.get("hypothetical_transition", None)
    if payload.get("accepted_transition", False) is not False:
        raise RolloutError("shadow_apply cannot publish an accepted transition")
    if claimed in (None, ""):
        if not mutate:
            return "", False
        return _hypothetical_transition_cid(payload), True
    cid = spar040._cid(claimed, "refactor_transition_cid")
    if hypothetical is False:
        raise RolloutError("shadow_apply cannot publish an accepted transition")
    if hypothetical not in (None, True):
        raise RolloutError("shadow_apply publishes hypothetical transitions only")
    return cid, True


@dataclass(frozen=True, slots=True)
class ShadowApplyReceipt:
    """Nomination-only SPAR-041 shadow_apply rollout receipt."""

    tree_id: str
    status: str
    current_mode: str
    nominated_mode: str
    mode_contract: ModeContract
    comparison: CandidateComparison
    pre_world_root_cid: str
    post_world_root_cid: str
    program_graph_snapshot_cid: str
    partition_candidate_cid: str
    boundary_contract_set_cid: str
    transformation_packet_cid: str
    context_receipt_cid: str
    route_decision_cid: str
    validation_receipt_cids: Sequence[str] = ()
    refactor_transition_cid: str = ""
    expected_root_generation: int = 0
    resulting_root_generation: int = 0
    analyzer_id: str = ANALYZER_ID
    worktree_id: str = ""
    lease_id: str = ""
    fence_id: str = ""
    worktree_disposable: bool = True
    worktree_isolated: bool = True
    worktree_mutated: bool = False
    packet_applied: bool = False
    hypothetical_transition: bool = False
    negative_evidence_cids: Sequence[str] = ()
    terminal: TypedTerminal | None = None

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "status",
            "current_mode",
            "nominated_mode",
            "mode_contract",
            "comparison",
            "pre_world_root_cid",
            "post_world_root_cid",
            "program_graph_snapshot_cid",
            "partition_candidate_cid",
            "boundary_contract_set_cid",
            "transformation_packet_cid",
            "context_receipt_cid",
            "route_decision_cid",
            "validation_receipt_cids",
            "refactor_transition_cid",
            "expected_root_generation",
            "resulting_root_generation",
            "rollout_mode",
            "analyzer_id",
            "worktree_id",
            "lease_id",
            "fence_id",
            "worktree_disposable",
            "worktree_isolated",
            "worktree_mutated",
            "packet_applied",
            "hypothetical_transition",
            "negative_evidence_cids",
            "terminal",
            "receipt_cid",
            "nominated",
            "accepted",
            "gate_is_nomination_only",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "can_change_mode",
            "projection_is_authority",
            "writes_repository",
            "worker_self_approval",
            "worker_may_change_mode",
            "influences_routing",
            "source_mutation",
            "merge",
            "root_promoted",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", spar040._tree_id(self.tree_id))
        object.__setattr__(
            self, "status", spar040._enum(self.status, ShadowApplyStatus, "status")
        )
        object.__setattr__(
            self, "current_mode", spar040._mode(self.current_mode, "current_mode")
        )
        object.__setattr__(
            self, "nominated_mode", spar040._mode(self.nominated_mode, "nominated_mode")
        )
        if self.nominated_mode != SHADOW_APPLY_MODE:
            raise RolloutError("SPAR-041 can only nominate shadow_apply")
        if self.current_mode not in ALLOWED_CURRENT_MODES:
            raise RolloutError("SPAR-041 cannot skip or regress sealed rollout gates")
        contract = self.mode_contract
        if not isinstance(contract, ModeContract):
            contract = ModeContract.from_dict(contract)
        if contract.mode != SHADOW_APPLY_MODE:
            raise RolloutError("shadow_apply gate must bind the shadow_apply contract")
        object.__setattr__(self, "mode_contract", contract)
        comparison = self.comparison
        if not isinstance(comparison, CandidateComparison):
            comparison = CandidateComparison.from_dict(comparison)
        if comparison.tree_id != self.tree_id:
            raise RolloutError("receipt tree_id does not match comparison")
        object.__setattr__(self, "comparison", comparison)
        object.__setattr__(
            self,
            "pre_world_root_cid",
            spar040._cid(self.pre_world_root_cid, "pre_world_root_cid"),
        )
        object.__setattr__(
            self,
            "post_world_root_cid",
            spar040._cid(self.post_world_root_cid, "post_world_root_cid"),
        )
        if self.post_world_root_cid != self.pre_world_root_cid:
            raise RolloutError("shadow_apply cannot mutate or promote the world root")
        object.__setattr__(
            self,
            "program_graph_snapshot_cid",
            spar040._cid(self.program_graph_snapshot_cid, "program_graph_snapshot_cid"),
        )
        object.__setattr__(
            self,
            "partition_candidate_cid",
            spar040._cid(self.partition_candidate_cid, "partition_candidate_cid"),
        )
        object.__setattr__(
            self,
            "boundary_contract_set_cid",
            spar040._cid(self.boundary_contract_set_cid, "boundary_contract_set_cid"),
        )
        object.__setattr__(
            self,
            "transformation_packet_cid",
            spar040._cid(self.transformation_packet_cid, "transformation_packet_cid"),
        )
        object.__setattr__(
            self,
            "context_receipt_cid",
            spar040._cid(self.context_receipt_cid, "context_receipt_cid"),
        )
        object.__setattr__(
            self,
            "route_decision_cid",
            spar040._cid(self.route_decision_cid, "route_decision_cid"),
        )
        object.__setattr__(
            self,
            "validation_receipt_cids",
            spar040._cids(
                list(self.validation_receipt_cids), "validation_receipt_cids"
            ),
        )
        object.__setattr__(
            self,
            "refactor_transition_cid",
            spar040._optional_cid(
                self.refactor_transition_cid, "refactor_transition_cid"
            ),
        )
        object.__setattr__(
            self,
            "expected_root_generation",
            spar040._int(
                self.expected_root_generation,
                "expected_root_generation",
                maximum=spar040.MAX_ROOT_GENERATION,
            ),
        )
        object.__setattr__(
            self,
            "resulting_root_generation",
            spar040._int(
                self.resulting_root_generation,
                "resulting_root_generation",
                maximum=spar040.MAX_ROOT_GENERATION,
            ),
        )
        if self.resulting_root_generation != self.expected_root_generation:
            raise RolloutError("shadow_apply cannot promote root generation")
        object.__setattr__(
            self, "analyzer_id", spar040._text(self.analyzer_id, "analyzer_id")
        )
        if self.analyzer_id != ANALYZER_ID:
            raise RolloutError("analyzer_id must remain SPAR-041")
        object.__setattr__(
            self, "worktree_id", spar040._text(self.worktree_id, "worktree_id")
        )
        object.__setattr__(self, "lease_id", spar040._text(self.lease_id, "lease_id"))
        object.__setattr__(self, "fence_id", spar040._text(self.fence_id, "fence_id"))
        object.__setattr__(
            self,
            "worktree_disposable",
            spar040._bool(self.worktree_disposable, "worktree_disposable"),
        )
        object.__setattr__(
            self,
            "worktree_isolated",
            spar040._bool(self.worktree_isolated, "worktree_isolated"),
        )
        if not self.worktree_disposable or not self.worktree_isolated:
            raise RolloutError("disposable isolated worktree required")
        object.__setattr__(
            self,
            "worktree_mutated",
            spar040._bool(self.worktree_mutated, "worktree_mutated"),
        )
        object.__setattr__(
            self, "packet_applied", spar040._bool(self.packet_applied, "packet_applied")
        )
        object.__setattr__(
            self,
            "hypothetical_transition",
            spar040._bool(self.hypothetical_transition, "hypothetical_transition"),
        )
        if self.packet_applied and not self.worktree_mutated:
            raise RolloutError("applied packets mutate only a disposable worktree")
        if self.worktree_mutated and not self.packet_applied:
            raise RolloutError("worktree mutation requires a bounded packet apply")
        if self.refactor_transition_cid and not self.hypothetical_transition:
            raise RolloutError("shadow_apply publishes hypothetical transitions only")
        if self.hypothetical_transition and not self.refactor_transition_cid:
            raise RolloutError("hypothetical transition requires a transition identity")
        object.__setattr__(
            self,
            "negative_evidence_cids",
            spar040._cids(list(self.negative_evidence_cids), "negative_evidence_cids"),
        )
        expected_negative = tuple(comparison.retained_negative_cids)
        if tuple(self.negative_evidence_cids) != expected_negative:
            raise RolloutError("receipt must retain compared false and unsafe candidates")
        terminal = self.terminal
        if terminal not in (None,):
            if not isinstance(terminal, TypedTerminal):
                terminal = TypedTerminal.from_mapping(terminal)
        else:
            terminal = None
        object.__setattr__(self, "terminal", terminal)
        if self.status == ShadowApplyStatus.NOMINATED_SHADOW_APPLY.value:
            if self.terminal is not None:
                raise RolloutError("nominated shadow_apply cannot carry a typed terminal")
            if not self.validation_receipt_cids:
                raise RolloutError("shadow_apply requires full validation")
        if self.status == ShadowApplyStatus.TYPED_TERMINAL.value:
            if self.terminal is None:
                raise RolloutError("typed terminal status requires a typed terminal")
        if (
            self.status != ShadowApplyStatus.TYPED_TERMINAL.value
            and self.terminal is not None
        ):
            raise RolloutError("non-terminal status cannot carry a typed terminal")

    @property
    def nominated(self) -> bool:
        return self.status == ShadowApplyStatus.NOMINATED_SHADOW_APPLY.value

    @property
    def accepted(self) -> bool:
        return False

    @property
    def rollout_mode(self) -> str:
        return self.nominated_mode

    @property
    def root_promoted(self) -> bool:
        return False

    @property
    def merge(self) -> bool:
        return False

    @property
    def writes_repository(self) -> bool:
        return False

    @property
    def can_authorize_transition(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    @property
    def can_create_authority(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": SHADOW_APPLY_RECEIPT_SCHEMA,
            "interface": SHADOW_APPLY_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "status": self.status,
            "current_mode": self.current_mode,
            "nominated_mode": self.nominated_mode,
            "rollout_mode": self.nominated_mode,
            "mode_contract": self.mode_contract.to_dict(),
            "comparison": self.comparison.to_dict(),
            "pre_world_root_cid": self.pre_world_root_cid,
            "post_world_root_cid": self.post_world_root_cid,
            "program_graph_snapshot_cid": self.program_graph_snapshot_cid,
            "partition_candidate_cid": self.partition_candidate_cid,
            "boundary_contract_set_cid": self.boundary_contract_set_cid,
            "transformation_packet_cid": self.transformation_packet_cid,
            "context_receipt_cid": self.context_receipt_cid,
            "route_decision_cid": self.route_decision_cid,
            "validation_receipt_cids": list(self.validation_receipt_cids),
            "refactor_transition_cid": self.refactor_transition_cid,
            "expected_root_generation": self.expected_root_generation,
            "resulting_root_generation": self.resulting_root_generation,
            "analyzer_id": ANALYZER_ID,
            "worktree_id": self.worktree_id,
            "lease_id": self.lease_id,
            "fence_id": self.fence_id,
            "worktree_disposable": True,
            "worktree_isolated": True,
            "worktree_mutated": self.worktree_mutated,
            "packet_applied": self.packet_applied,
            "hypothetical_transition": self.hypothetical_transition,
            "negative_evidence_cids": list(self.negative_evidence_cids),
            "terminal": None if self.terminal is None else self.terminal.to_dict(),
            "nominated": self.nominated,
            "accepted": False,
            "gate_is_nomination_only": True,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "can_change_mode": False,
            "projection_is_authority": False,
            "writes_repository": False,
            "worker_self_approval": False,
            "worker_may_change_mode": False,
            "influences_routing": False,
            "source_mutation": SHADOW_APPLY_SOURCE_MUTATION,
            "merge": False,
            "root_promoted": False,
        }
        spar040._require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def receipt_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["receipt_cid"] = self.receipt_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShadowApplyReceipt":
        spar040._reject_excluded(data, cls.__name__)
        payload = spar040._closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != SHADOW_APPLY_RECEIPT_SCHEMA:
            raise RolloutError("unsupported ShadowApplyReceipt schema")
        if payload.pop("interface") != SHADOW_APPLY_RECEIPT_INTERFACE:
            raise RolloutError("unsupported ShadowApplyReceipt interface")
        if payload.pop("accepted") is not False:
            raise RolloutError("workers cannot self-approve a shadow_apply")
        if payload.pop("gate_is_nomination_only") is not True:
            raise RolloutError("gate must remain nomination_only")
        if payload.pop("rollout_mode") != SHADOW_APPLY_MODE:
            raise RolloutError("receipt rollout_mode must remain shadow_apply")
        nominated = payload.pop("nominated")
        _pop_apply_authority_flags(payload, "ShadowApplyReceipt")
        payload["mode_contract"] = ModeContract.from_dict(payload["mode_contract"])
        payload["comparison"] = CandidateComparison.from_dict(payload["comparison"])
        if payload["terminal"] is not None:
            payload["terminal"] = TypedTerminal.from_dict(payload["terminal"])
        result = cls(**payload)
        if nominated is not result.nominated:
            raise RolloutError("nominated flag does not match status")
        spar040._verify_cid(claimed, result.receipt_cid, "ShadowApplyReceipt receipt_cid")
        return result


def _receipt_from_payload(
    payload: Mapping[str, Any],
    *,
    status: str,
    current_mode: str,
    comparison: CandidateComparison,
    mutate: bool,
    terminal: TypedTerminal | None = None,
) -> ShadowApplyReceipt:
    pre = spar040._cid(payload.get("pre_world_root_cid"), "pre_world_root_cid")
    expected = spar040._int(
        payload.get("expected_root_generation", 0),
        "expected_root_generation",
        maximum=spar040.MAX_ROOT_GENERATION,
    )
    worktree_id, lease_id, fence_id = _require_disposable_isolated_worktree(payload)
    transition_cid = ""
    hypothetical = False
    packet_applied = False
    worktree_mutated = False
    if terminal is None:
        transition_cid, hypothetical = _resolve_transition(payload, mutate=mutate)
        packet_applied = mutate is True
        worktree_mutated = mutate is True
        if mutate is True and not hypothetical:
            raise RolloutError("shadow_apply publishes hypothetical transitions only")
    return ShadowApplyReceipt(
        tree_id=payload.get("tree_id"),
        status=status,
        current_mode=current_mode,
        nominated_mode=SHADOW_APPLY_MODE,
        mode_contract=ModeContract.for_mode(SHADOW_APPLY_MODE),
        comparison=comparison,
        pre_world_root_cid=pre,
        post_world_root_cid=pre,
        program_graph_snapshot_cid=payload.get("program_graph_snapshot_cid"),
        partition_candidate_cid=payload.get("partition_candidate_cid"),
        boundary_contract_set_cid=payload.get("boundary_contract_set_cid"),
        transformation_packet_cid=payload.get("transformation_packet_cid"),
        context_receipt_cid=payload.get("context_receipt_cid"),
        route_decision_cid=payload.get("route_decision_cid"),
        validation_receipt_cids=payload.get("validation_receipt_cids") or (),
        refactor_transition_cid=transition_cid,
        expected_root_generation=expected,
        resulting_root_generation=expected,
        worktree_id=worktree_id,
        lease_id=lease_id,
        fence_id=fence_id,
        worktree_disposable=True,
        worktree_isolated=True,
        worktree_mutated=worktree_mutated,
        packet_applied=packet_applied,
        hypothetical_transition=hypothetical,
        negative_evidence_cids=comparison.retained_negative_cids,
        terminal=terminal,
    )


def run_shadow_apply(
    evidence: Mapping[str, Any],
    *,
    mutate: bool = False,
) -> ShadowApplyReceipt:
    """Apply a bounded packet in a disposable worktree without merge or root promotion."""

    if mutate is not False and mutate is not True:
        raise RolloutError("mutate must be a boolean")
    payload = spar040._mapping(evidence, "shadow-apply evidence")
    spar040._reject_non_admitting(payload, "shadow-apply evidence")
    if payload.get("network", NETWORK_DENY) != NETWORK_DENY:
        raise RolloutError("network is denied")
    current_mode = _reject_worker_mode_change(payload)
    _reject_merge_root_routing(payload)
    _reject_unbounded_mutation(payload, mutate=mutate)
    _require_disposable_isolated_worktree(payload)
    terminal = spar040._parse_terminal(payload.get("terminal"))
    candidates = spar040._parse_candidates(payload.get("candidates"))
    tree_id = spar040._tree_id(payload.get("tree_id"))
    comparison = compare_and_retain_candidates(candidates, tree_id=tree_id)
    if payload.get("analysis_available", True) is not True:
        terminal = TypedTerminal(
            kind=TerminalKind.CAPABILITY_UNAVAILABLE.value,
            reason="required analysis capability is unavailable",
        )
    if payload.get("planning_available", True) is not True:
        terminal = TypedTerminal(
            kind=TerminalKind.CAPABILITY_UNAVAILABLE.value,
            reason="required planning capability is unavailable",
        )
    if payload.get("validation_available", True) is not True:
        terminal = TypedTerminal(
            kind=TerminalKind.CAPABILITY_UNAVAILABLE.value,
            reason="required validation capability is unavailable",
        )
    if terminal is not None:
        return _receipt_from_payload(
            payload,
            status=ShadowApplyStatus.TYPED_TERMINAL.value,
            current_mode=current_mode,
            comparison=comparison,
            mutate=False,
            terminal=terminal,
        )
    _require_complete_analysis_planning_validation(payload)
    return _receipt_from_payload(
        payload,
        status=ShadowApplyStatus.NOMINATED_SHADOW_APPLY.value,
        current_mode=current_mode,
        comparison=comparison,
        mutate=mutate,
    )


def activate_shadow_apply(evidence: Mapping[str, Any]) -> ShadowApplyReceipt:
    """Nominate SPAR-041 activation of shadow_apply from sealed shadow_plan."""

    return run_shadow_apply(evidence)


def dry_run_shadow_apply(evidence: Mapping[str, Any]) -> ShadowApplyReceipt:
    """Deterministic dry-run. Never mutates and never merges."""

    return run_shadow_apply(evidence, mutate=False)


def encode_canonical_receipt(receipt: ShadowApplyReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(payload: Mapping[str, Any]) -> ShadowApplyReceipt:
    return ShadowApplyReceipt.from_dict(payload)


class ShadowApplyGate:
    """SPAR-041 shadow_apply rollout gate. Nomination-only; disposable worktree only."""

    interface: ClassVar[str] = SHADOW_APPLY_GATE_INTERFACE
    schema: ClassVar[str] = SHADOW_APPLY_GATE_SCHEMA
    analyzer_id: ClassVar[str] = ANALYZER_ID

    def run(
        self,
        evidence: Mapping[str, Any],
        *,
        mutate: bool = False,
    ) -> ShadowApplyReceipt:
        return run_shadow_apply(evidence, mutate=mutate)

    def activate(self, evidence: Mapping[str, Any]) -> ShadowApplyReceipt:
        return activate_shadow_apply(evidence)

    def compare(
        self,
        candidates: Sequence[ShadowPlanCandidate] | Sequence[Mapping[str, Any]],
        *,
        tree_id: str,
    ) -> CandidateComparison:
        return compare_and_retain_candidates(candidates, tree_id=tree_id)

    def dry_run(self, evidence: Mapping[str, Any]) -> ShadowApplyReceipt:
        return dry_run_shadow_apply(evidence)


def _evidence(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "current_mode": SHADOW_PLAN_MODE,
        "analysis_complete": True,
        "planning_complete": True,
        "validation_complete": True,
        "source_mutation": SHADOW_APPLY_SOURCE_MUTATION,
        "merge": False,
        "routing_influence": False,
        "worker_may_change_mode": False,
        "pre_world_root_cid": _cid("world-root:pre"),
        "program_graph_snapshot_cid": _cid("graph"),
        "partition_candidate_cid": _cid("partition"),
        "boundary_contract_set_cid": _cid("boundary"),
        "transformation_packet_cid": _cid("packet"),
        "context_receipt_cid": _cid("context"),
        "route_decision_cid": _cid("route"),
        "validation_receipt_cids": [_cid("validation:full")],
        "refactor_transition_cid": "",
        "expected_root_generation": 3,
        "candidates": [
            _candidate("safe-high", score=4),
            _candidate("safe-low", score=1),
            _candidate("false", "false"),
            _candidate("unsafe", "unsafe", ("scc",)),
        ],
        "network": NETWORK_DENY,
        "worktree_id": _cid("worktree"),
        "lease_id": "lease-1",
        "fence_id": "fence-1",
        "worktree_disposable": True,
        "worktree_isolated": True,
        "packet_bounded": True,
    }
    fields.update(overrides)
    return fields


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-041"
    assert GOAL_ID == "SPAR-G073"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert PREDECESSOR_TASK_IDS == ("SPAR-040",)
    assert SHADOW_APPLY_GATE_INTERFACE == "ShadowApplyGate@1"
    assert SHADOW_APPLY_RECEIPT_INTERFACE == "ShadowApplyReceipt@1"
    assert ROLLOUT_CONTRACT_VERSION == "1"
    assert ROLLOUT_BASELINE_SCHEMA == "spar/rollout-baseline@1"
    assert ANALYZER_ID.endswith("shadow_apply@1")
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()
    assert ROLLOUT_MODES == (
        "bootstrap",
        "shadow_plan",
        "shadow_apply",
        "guarded",
        "required",
    )
    assert DECLARED_ROLLOUT_MODES == set(ROLLOUT_MODES)
    assert SHADOW_APPLY_MODE == "shadow_apply"
    assert SHADOW_PLAN_MODE == "shadow_plan"
    assert BOOTSTRAP_MODE == "bootstrap"
    assert ALLOWED_CURRENT_MODES == {"shadow_plan", "shadow_apply"}
    assert PROGRESSIVE_MODES == ("shadow_apply", "guarded", "required")
    assert DECLARED_GATE_STATUSES == {"nominated_shadow_apply", "typed_terminal"}
    assert DECLARED_TERMINAL_KINDS == {
        "unsupported",
        "human_review",
        "capability_unavailable",
    }
    assert CANDIDATE_DISPOSITIONS == ("admitted", "false", "unsafe")
    assert HARD_CONSTRAINTS == (
        "scc",
        "state",
        "consumer",
        "compatibility",
        "ordering",
        "resource",
        "frontier",
        "proof",
        "transaction",
    )
    assert DECLARED_HARD_CONSTRAINTS == set(HARD_CONSTRAINTS)
    assert RECEIPT_FLOOR == (
        "pre_world_root_cid",
        "program_graph_snapshot_cid",
        "partition_candidate_cid",
        "boundary_contract_set_cid",
        "transformation_packet_cid",
        "context_receipt_cid",
        "route_decision_cid",
        "validation_receipt_cids",
        "refactor_transition_cid",
        "post_world_root_cid",
        "expected_root_generation",
        "resulting_root_generation",
        "rollout_mode",
    )
    assert DECLARED_RECEIPT_FLOOR == set(RECEIPT_FLOOR)


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "operational refactoring authority"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert GATE_CAN_AUTHORIZE_COMPLETION is False
    assert GATE_CAN_AUTHORIZE_TRANSITION is False
    assert GATE_CAN_CREATE_AUTHORITY is False
    assert GATE_CAN_CHANGE_MODE is False
    assert GATE_WRITES_REPOSITORY is False
    assert SHADOW_APPLY_WRITES_REPOSITORY is False
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert WORKER_MAY_CHANGE_MODE is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert GATE_IS_NOMINATION_ONLY is True
    assert RAW_SOURCE_REQUIRED is True
    assert NETWORK_DENIED is True
    assert NETWORK_DENY == "deny"
    assert DRY_RUN_IS_DETERMINISTIC is True
    assert DRY_RUN_MUTATES is False
    assert NEGATIVE_EVIDENCE_RETAINED is True
    assert SHADOW_APPLY_SOURCE_MUTATION == "disposable_worktree_only"
    assert SHADOW_APPLY_MERGE is False
    assert SHADOW_APPLY_INFLUENCES_ROUTING is False
    assert SHADOW_APPLY_PROMOTES_ROOT is False
    assert FULL_VALIDATION_REQUIRED is True
    assert COMPLETE_ANALYSIS_REQUIRED is True
    assert COMPLETE_PLANNING_REQUIRED is True
    assert FALSE_UNSAFE_CANDIDATES_RETAINED is True
    assert DISPOSABLE_WORKTREE_ONLY is True
    assert ISOLATED_WORKTREE_REQUIRED is True
    assert HYPOTHETICAL_TRANSITION_ONLY is True
    assert BOUNDED_PACKET_ONLY is True
    profile = rollout_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]
    assert "spar_narrow" in EXISTING_ADAPTER_AUTHORITIES
    assert "RolloutStore" in FORBIDDEN_ROLLOUT_NAMES
    assert "promote_root" in FORBIDDEN_ROLLOUT_NAMES
    descriptor = shadow_apply_gate_descriptor()
    assert descriptor["nomination_only"] is True
    assert descriptor["writes_repository"] is False
    assert descriptor["worker_may_change_mode"] is False
    assert descriptor["influences_routing"] is False
    assert descriptor["promotes_root"] is False
    assert descriptor["merge"] is False
    assert descriptor["source_mutation"] == "disposable_worktree_only"
    assert descriptor["nominated_mode"] == "shadow_apply"
    assert descriptor["gate_task"] == "SPAR-041"
    assert descriptor["predecessor_task_ids"] == ["SPAR-040"]


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(TEST_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "ShadowApplyGate" in names
    assert "ShadowApplyReceipt" in names
    assert "RolloutStore" not in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "ShadowApplyGate" in exports
    assert "run_shadow_apply" in exports
    assert "activate_shadow_apply" in exports
    functions = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert "authorize_mode_change" in FORBIDDEN_ROLLOUT_NAMES
    for forbidden in (
        "authorize_mode_change",
        "worker_set_mode",
        "mutate_source",
        "influence_routing",
        "promote_root",
    ):
        assert forbidden in FORBIDDEN_ROLLOUT_NAMES
        assert forbidden not in functions
        assert forbidden not in names


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_mode_table_matches_sealed_rollout_baseline() -> None:
    sealed = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    encoded = sealed_rollout_baseline()
    assert sealed["schema"] == ROLLOUT_BASELINE_SCHEMA
    assert sealed["worker_may_change_mode"] is False
    assert encoded["worker_may_change_mode"] is False
    assert encoded["receipt_floor"] == sealed["receipt_floor"]
    assert list(RECEIPT_FLOOR) == sealed["receipt_floor"]
    for mode in ROLLOUT_MODES:
        assert dict(MODE_CONTRACTS[mode]) == sealed["modes"][mode]
        assert encoded["modes"][mode] == sealed["modes"][mode]
    assert MODE_CONTRACTS["shadow_apply"]["gate_task"] == "SPAR-041"
    assert MODE_CONTRACTS["shadow_apply"]["source_mutation"] == "disposable_worktree_only"
    assert MODE_CONTRACTS["shadow_apply"]["merge"] is False
    assert MODE_CONTRACTS["shadow_plan"]["gate_task"] == "SPAR-040"


def test_shadow_apply_nominates_without_merge_or_root_promotion() -> None:
    receipt = run_shadow_apply(_evidence())
    assert receipt.status == "nominated_shadow_apply"
    assert receipt.nominated is True
    assert receipt.accepted is False
    assert receipt.can_authorize_completion is False
    assert receipt.can_authorize_transition is False
    assert receipt.can_create_authority is False
    assert receipt.current_mode == SHADOW_PLAN_MODE
    assert receipt.nominated_mode == SHADOW_APPLY_MODE
    assert receipt.rollout_mode == SHADOW_APPLY_MODE
    assert receipt.mode_contract.gate_task == "SPAR-041"
    assert receipt.mode_contract.source_mutation == "disposable_worktree_only"
    assert receipt.mode_contract.merge is False
    assert receipt.post_world_root_cid == receipt.pre_world_root_cid
    assert receipt.resulting_root_generation == receipt.expected_root_generation
    assert receipt.root_promoted is False
    assert receipt.merge is False
    assert receipt.writes_repository is False
    assert receipt.packet_applied is False
    assert receipt.worktree_mutated is False
    encoded = encode_canonical_receipt(receipt)
    assert encoded["source_mutation"] == "disposable_worktree_only"
    assert encoded["merge"] is False
    assert encoded["influences_routing"] is False
    assert encoded["worker_may_change_mode"] is False
    assert encoded["writes_repository"] is False
    assert encoded["root_promoted"] is False
    restored = decode_canonical_receipt(encoded)
    assert restored == receipt
    assert restored.receipt_cid == receipt.receipt_cid
    for field in RECEIPT_FLOOR:
        assert field in encoded


def test_bounded_packet_applies_only_in_disposable_isolated_worktree() -> None:
    receipt = run_shadow_apply(_evidence(), mutate=True)
    assert receipt.status == "nominated_shadow_apply"
    assert receipt.packet_applied is True
    assert receipt.worktree_mutated is True
    assert receipt.worktree_disposable is True
    assert receipt.worktree_isolated is True
    assert receipt.hypothetical_transition is True
    assert receipt.refactor_transition_cid
    assert receipt.merge is False
    assert receipt.root_promoted is False
    assert receipt.writes_repository is False
    assert receipt.accepted is False
    assert receipt.post_world_root_cid == receipt.pre_world_root_cid
    assert receipt.resulting_root_generation == receipt.expected_root_generation


def test_hypothetical_transitions_are_published_without_merge() -> None:
    transition = _cid("transition:hypothetical")
    receipt = run_shadow_apply(
        _evidence(
            refactor_transition_cid=transition,
            hypothetical_transition=True,
        ),
        mutate=True,
    )
    assert receipt.refactor_transition_cid == transition
    assert receipt.hypothetical_transition is True
    assert receipt.merge is False
    assert receipt.root_promoted is False
    with pytest.raises(RolloutError, match="accepted transition"):
        run_shadow_apply(
            _evidence(
                refactor_transition_cid=transition,
                hypothetical_transition=False,
            ),
            mutate=True,
        )
    with pytest.raises(RolloutError, match="accepted transition"):
        run_shadow_apply(_evidence(accepted_transition=True), mutate=True)


def test_false_and_unsafe_candidates_are_compared_and_retained() -> None:
    receipt = activate_shadow_apply(_evidence())
    comparison = receipt.comparison
    admitted = comparison.ranked_admitted_cids
    assert admitted == (
        _cid("candidate:safe-high"),
        _cid("candidate:safe-low"),
    )
    assert _cid("candidate:false") in comparison.false_candidate_cids
    assert _cid("candidate:unsafe") in comparison.unsafe_candidate_cids
    assert set(comparison.retained_negative_cids) == {
        _cid("candidate:false"),
        _cid("candidate:unsafe"),
    }
    assert tuple(receipt.negative_evidence_cids) == tuple(
        comparison.retained_negative_cids
    )
    assert FALSE_UNSAFE_CANDIDATES_RETAINED is True


def test_soft_score_cannot_admit_an_unsafe_candidate() -> None:
    with pytest.raises(RolloutError, match="hard constraint"):
        run_shadow_apply(
            _evidence(
                candidates=[
                    _candidate("promoted", "admitted", ("scc",), score=99),
                ]
            )
        )
    with pytest.raises(RolloutError, match="unsafe candidates require"):
        ShadowPlanCandidate(
            candidate_cid=_cid("candidate:bare-unsafe"),
            disposition="unsafe",
        )


def test_idempotent_when_already_in_shadow_apply() -> None:
    receipt = run_shadow_apply(_evidence(current_mode=SHADOW_APPLY_MODE))
    assert receipt.status == "nominated_shadow_apply"
    assert receipt.current_mode == SHADOW_APPLY_MODE
    assert receipt.nominated_mode == SHADOW_APPLY_MODE


def test_worker_cannot_change_or_skip_rollout_mode() -> None:
    with pytest.raises(RolloutError, match="cannot change rollout"):
        run_shadow_apply(_evidence(worker_may_change_mode=True))
    with pytest.raises(RolloutError, match="cannot change rollout"):
        run_shadow_apply(_evidence(requested_mode="guarded"))
    with pytest.raises(RolloutError, match="cannot change rollout"):
        run_shadow_apply(_evidence(requested_mode="required"))
    with pytest.raises(RolloutError, match="cannot change rollout"):
        run_shadow_apply(_evidence(requested_mode="bootstrap"))
    with pytest.raises(RolloutError, match="cannot skip or regress"):
        run_shadow_apply(_evidence(current_mode=BOOTSTRAP_MODE))
    with pytest.raises(RolloutError, match="cannot skip or regress"):
        run_shadow_apply(_evidence(current_mode="guarded"))
    with pytest.raises(RolloutError, match="cannot skip or regress"):
        run_shadow_apply(_evidence(current_mode="required"))


def test_non_disposable_or_unfenced_worktree_fails_closed() -> None:
    with pytest.raises(RolloutError, match="disposable isolated worktree"):
        run_shadow_apply(_evidence(worktree_disposable=False), mutate=True)
    with pytest.raises(RolloutError, match="disposable isolated worktree"):
        run_shadow_apply(_evidence(worktree_isolated=False), mutate=True)
    with pytest.raises(RolloutError, match="disposable isolated worktree"):
        run_shadow_apply(_evidence(worktree_id=""), mutate=True)
    with pytest.raises(RolloutError, match="disposable isolated worktree"):
        run_shadow_apply(_evidence(lease_id=""), mutate=True)
    with pytest.raises(RolloutError, match="disposable isolated worktree"):
        run_shadow_apply(_evidence(fence_id=""), mutate=True)


def test_repository_write_merge_and_root_promotion_fail_closed() -> None:
    with pytest.raises(RolloutError, match="cannot mutate source"):
        run_shadow_apply(_evidence(source_mutation=True), mutate=True)
    with pytest.raises(RolloutError, match="cannot write the repository"):
        run_shadow_apply(_evidence(writes_repository=True), mutate=True)
    with pytest.raises(RolloutError, match="cannot write the repository"):
        run_shadow_apply(_evidence(apply_to_repository=True), mutate=True)
    with pytest.raises(RolloutError, match="cannot merge"):
        run_shadow_apply(_evidence(merge=True), mutate=True)
    with pytest.raises(RolloutError, match="cannot merge"):
        run_shadow_apply(_evidence(merge_worktree=True), mutate=True)
    with pytest.raises(RolloutError, match="cannot influence routing"):
        run_shadow_apply(_evidence(routing_influence=True))
    with pytest.raises(RolloutError, match="cannot influence routing"):
        run_shadow_apply(
            _evidence(proposed_route_decision_cid=_cid("route:other"))
        )
    with pytest.raises(RolloutError, match="cannot mutate or promote"):
        run_shadow_apply(_evidence(post_world_root_cid=_cid("world-root:post")))
    with pytest.raises(RolloutError, match="cannot promote root generation"):
        run_shadow_apply(_evidence(resulting_root_generation=4))
    with pytest.raises(RolloutError, match="cannot promote a root"):
        run_shadow_apply(_evidence(promote_root=True), mutate=True)
    with pytest.raises(RolloutError, match="bounded packets only"):
        run_shadow_apply(_evidence(packet_bounded=False), mutate=True)


def test_full_validation_is_required() -> None:
    with pytest.raises(RolloutError, match="complete analysis"):
        run_shadow_apply(_evidence(analysis_complete=False))
    with pytest.raises(RolloutError, match="complete planning"):
        run_shadow_apply(_evidence(planning_complete=False))
    with pytest.raises(RolloutError, match="full validation"):
        run_shadow_apply(_evidence(validation_complete=False))
    with pytest.raises(RolloutError, match="full validation"):
        run_shadow_apply(_evidence(validation_receipt_cids=[]))
    with pytest.raises(RolloutError, match="requires planned candidates"):
        run_shadow_apply(_evidence(candidates=[]))


def test_capability_unavailable_is_typed_terminal() -> None:
    receipt = run_shadow_apply(_evidence(analysis_available=False))
    assert receipt.status == "typed_terminal"
    assert receipt.accepted is False
    assert receipt.nominated is False
    assert receipt.terminal is not None
    assert receipt.terminal.kind == TerminalKind.CAPABILITY_UNAVAILABLE.value
    assert receipt.rollout_mode == SHADOW_APPLY_MODE
    planning = run_shadow_apply(_evidence(planning_available=False))
    assert planning.terminal.kind == TerminalKind.CAPABILITY_UNAVAILABLE.value
    validation = run_shadow_apply(_evidence(validation_available=False))
    assert validation.terminal.kind == TerminalKind.CAPABILITY_UNAVAILABLE.value
    assert validation.packet_applied is False


def test_explicit_human_review_terminal_stops_without_completion() -> None:
    receipt = run_shadow_apply(
        _evidence(
            terminal={
                "kind": TerminalKind.HUMAN_REVIEW.value,
                "reason": "unsafe partition split requires review",
            }
        )
    )
    assert receipt.status == "typed_terminal"
    assert receipt.terminal.kind == TerminalKind.HUMAN_REVIEW.value
    assert receipt.accepted is False
    assert receipt.nominated is False


def test_dry_run_is_deterministic_and_never_mutates() -> None:
    payload = _evidence()
    first = dry_run_shadow_apply(payload)
    second = ShadowApplyGate().dry_run(payload)
    assert first.receipt_cid == second.receipt_cid
    assert first.packet_applied is False
    assert first.worktree_mutated is False
    assert DRY_RUN_MUTATES is False
    applied = run_shadow_apply(payload, mutate=True)
    assert applied.receipt_cid != first.receipt_cid
    assert applied.packet_applied is True


def test_identity_excludes_observational_fields() -> None:
    receipt = run_shadow_apply(_evidence())
    encoded = receipt.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(encoded))
    encoded["timestamp"] = "now"
    with pytest.raises(RolloutError, match="observational"):
        ShadowApplyReceipt.from_dict(encoded)
    with pytest.raises(RolloutError, match="observational"):
        run_shadow_apply(_evidence(model_output="guess"))


def test_vector_and_model_evidence_cannot_admit_apply() -> None:
    with pytest.raises(RolloutError, match="cannot admit"):
        run_shadow_apply(_evidence(vector_candidate={"score": 1}))
    with pytest.raises(RolloutError, match="cannot admit"):
        run_shadow_apply(_evidence(model_hypothesis={"ok": True}))
    with pytest.raises(RolloutError, match="cannot admit"):
        run_shadow_apply(_evidence(heuristic=True))


def test_network_is_denied() -> None:
    with pytest.raises(RolloutError, match="network is denied"):
        run_shadow_apply(_evidence(network="allow"))


def test_tree_mismatch_fails_closed() -> None:
    receipt = run_shadow_apply(_evidence())
    payload = receipt.to_dict()
    payload["tree_id"] = OTHER_TREE
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(RolloutError, match="tree_id"):
        ShadowApplyReceipt.from_dict(payload)
    with pytest.raises(RolloutError, match="tree_id"):
        run_shadow_apply(_evidence(tree_id="not-a-tree"))


def test_worker_cannot_self_approve_receipt() -> None:
    receipt = run_shadow_apply(_evidence())
    payload = receipt.to_dict()
    payload["accepted"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(RolloutError, match="self-approve"):
        ShadowApplyReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["can_authorize_completion"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(RolloutError, match="can_authorize_completion"):
        ShadowApplyReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["worker_may_change_mode"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(RolloutError, match="cannot claim worker_may_change_mode"):
        ShadowApplyReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["source_mutation"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(RolloutError, match="source_mutation is disposable_worktree_only"):
        ShadowApplyReceipt.from_dict(payload)


def test_mode_contract_cannot_be_rewritten_by_workers() -> None:
    with pytest.raises(RolloutError, match="source_mutation does not match"):
        ModeContract(
            mode="shadow_apply",
            source_mutation=True,
            merge=False,
            gate_task="SPAR-041",
        )
    with pytest.raises(RolloutError, match="merge does not match"):
        ModeContract(
            mode="shadow_apply",
            source_mutation="disposable_worktree_only",
            merge=True,
            gate_task="SPAR-041",
        )
    with pytest.raises(RolloutError, match="gate_task does not match"):
        ModeContract(
            mode="shadow_apply",
            source_mutation="disposable_worktree_only",
            merge=False,
            gate_task="SPAR-040",
        )
    contract = ModeContract.for_mode("shadow_apply")
    restored = ModeContract.from_dict(contract.to_dict())
    assert restored == contract
    payload = contract.to_dict()
    payload["worker_may_change_mode"] = True
    payload["contract_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "contract_cid"}
    )
    with pytest.raises(RolloutError, match="cannot change rollout"):
        ModeContract.from_dict(payload)


def test_gate_methods_match_module_functions() -> None:
    gate = ShadowApplyGate()
    payload = _evidence()
    assert gate.interface == SHADOW_APPLY_GATE_INTERFACE
    assert gate.run(payload).receipt_cid == run_shadow_apply(payload).receipt_cid
    assert gate.activate(payload).receipt_cid == activate_shadow_apply(payload).receipt_cid
    applied = gate.run(payload, mutate=True)
    assert applied.packet_applied is True
    comparison = gate.compare(payload["candidates"], tree_id=TREE_ID)
    assert comparison.comparison_cid == compare_and_retain_candidates(
        payload["candidates"], tree_id=TREE_ID
    ).comparison_cid
    terminal = TypedTerminal(
        kind=TerminalKind.UNSUPPORTED.value,
        reason="required dynamic frontier remains unresolved",
    )
    assert TypedTerminal.from_dict(terminal.to_dict()) == terminal
    assert RolloutMode.SHADOW_APPLY.value == "shadow_apply"
    assert CandidateDisposition.UNSAFE.value == "unsafe"
    assert ShadowApplyStatus.NOMINATED_SHADOW_APPLY.value == "nominated_shadow_apply"
