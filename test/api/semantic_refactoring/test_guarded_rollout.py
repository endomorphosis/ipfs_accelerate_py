"""Independent contract tests for SPAR-042 guarded rollout qualification."""

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
WRITE_SCOPE = ("test/api/semantic_refactoring/test_guarded_rollout.py",)
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
    "test/api/semantic_refactoring/test_shadow_apply.py",
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

TASK_ID: str = "SPAR-042"
PREDECESSOR_TASK_IDS: tuple[str, ...] = ("SPAR-041",)
ANALYZER_ID: str = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.guarded_rollout@1"
)
GUARDED_ROLLOUT_GATE_INTERFACE: str = "GuardedRolloutGate@1"
GUARDED_ROLLOUT_RECEIPT_INTERFACE: str = "GuardedRolloutReceipt@1"
GUARDED_WAVE_QUALIFICATION_INTERFACE: str = "GuardedWaveQualification@1"
NOMINATED_TRANSITION_INTERFACE: str = "NominatedRefactorTransition@1"
GUARDED_ROLLOUT_GATE_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/guarded-rollout-gate@1"
)
GUARDED_ROLLOUT_RECEIPT_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/guarded-rollout-receipt@1"
)
GUARDED_WAVE_QUALIFICATION_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/guarded-wave-qualification@1"
)
NOMINATED_TRANSITION_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/nominated-refactor-transition@1"
)
SHADOW_APPLY_MODE: str = "shadow_apply"
GUARDED_MODE: str = "guarded"
ALLOWED_CURRENT_MODES: frozenset[str] = frozenset({SHADOW_APPLY_MODE, GUARDED_MODE})
GUARDED_SOURCE_MUTATION: str = "Tier A and qualified Tier B"
GUARDED_MERGE: str = "current authority gates"
GATE_CAN_AUTHORIZE_TRANSITION: bool = False
GATE_CAN_AUTHORIZE_COMPLETION: bool = False
GATE_CAN_CREATE_AUTHORITY: bool = False
GATE_CAN_CHANGE_MODE: bool = False
GATE_WRITES_REPOSITORY: bool = False
GATE_IS_NOMINATION_ONLY: bool = True
FULL_VALIDATION_REQUIRED: bool = True
COMPLETE_ANALYSIS_REQUIRED: bool = True
COMPLETE_PLANNING_REQUIRED: bool = True
ISOLATED_WORKTREE_REQUIRED: bool = True
BOUNDED_PACKET_ONLY: bool = True
VECTOR_MODEL_AUTHORITY_REJECTED: bool = True
APPROVAL_REQUIRED_ABOVE_CEILING: bool = True
DECLARED_AUTONOMY_TIERS: tuple[str, ...] = ("A", "B", "C", "D", "E")
CEILING_TIERS: frozenset[str] = frozenset({"A", "B"})

_GUARDED_AUTHORITY_FLAGS: tuple[str, ...] = (
    "can_authorize_transition",
    "can_authorize_completion",
    "can_create_authority",
    "can_change_mode",
    "projection_is_authority",
    "writes_repository",
    "worker_self_approval",
    "worker_may_change_mode",
    "influences_routing",
    "root_promoted",
)


class GuardedStatus(str, Enum):
    NOMINATED_GUARDED = "nominated_guarded"
    TYPED_TERMINAL = "typed_terminal"


class WaveAutonomyTier(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


DECLARED_GATE_STATUSES: frozenset[str] = frozenset(
    item.value for item in GuardedStatus
)
DECLARED_WAVE_AUTONOMY_TIERS: frozenset[str] = frozenset(
    item.value for item in WaveAutonomyTier
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


def guarded_rollout_gate_descriptor() -> dict[str, Any]:
    return {
        "schema": GUARDED_ROLLOUT_GATE_SCHEMA,
        "interface": GUARDED_ROLLOUT_GATE_INTERFACE,
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "analyzer_id": ANALYZER_ID,
        "predecessor_task_ids": list(PREDECESSOR_TASK_IDS),
        "authority_owner": AUTHORITY_OWNER,
        "nomination_only": True,
        "writes_repository": False,
        "worker_may_change_mode": False,
        "source_mutation": GUARDED_SOURCE_MUTATION,
        "merge": GUARDED_MERGE,
        "influences_routing": False,
        "promotes_root": False,
        "nominated_mode": GUARDED_MODE,
        "gate_task": TASK_ID,
        "network": NETWORK_DENY,
        "isolated_worktree_required": True,
        "full_validation_required": True,
        "false_unsafe_candidates_retained": True,
        "vector_model_authority_rejected": True,
        "approval_required_above_ceiling": True,
        "ceiling": GUARDED_SOURCE_MUTATION,
    }


def provider_free_exports() -> tuple[str, ...]:
    return (
        "GuardedRolloutGate",
        "GuardedRolloutReceipt",
        "GuardedWaveQualification",
        "activate_guarded_rollout",
        "compare_and_retain_candidates",
        "dry_run_guarded_rollout",
        "run_guarded_rollout",
    )


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & spar040._FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise RolloutError(
            f"guarded rollout gate must not define competing types: {sorted(overlap)}"
        )
    spar040.assert_not_competing_capsule_family()


def _pop_guarded_authority_flags(payload: dict[str, Any], name: str) -> None:
    claimed_mutation = payload.pop("source_mutation", GUARDED_SOURCE_MUTATION)
    if claimed_mutation != GUARDED_SOURCE_MUTATION:
        raise RolloutError(
            f"{name} source_mutation is Tier A and qualified Tier B"
        )
    claimed_merge = payload.pop("merge", GUARDED_MERGE)
    if claimed_merge != GUARDED_MERGE:
        raise RolloutError(f"{name} merge is current authority gates")
    for flag in _GUARDED_AUTHORITY_FLAGS:
        if flag not in payload:
            continue
        claimed = payload.pop(flag)
        if claimed is not False:
            raise RolloutError(f"{name} cannot claim {flag}")


def _require_isolated_fenced_worktree(payload: Mapping[str, Any]) -> tuple[str, str, str]:
    worktree_id = spar040._text(payload.get("worktree_id", ""), "worktree_id", empty=True)
    lease_id = spar040._text(payload.get("lease_id", ""), "lease_id", empty=True)
    fence_id = spar040._text(payload.get("fence_id", ""), "fence_id", empty=True)
    isolated = spar040._bool(
        payload.get("worktree_isolated", False), "worktree_isolated"
    )
    if not worktree_id or not lease_id or not fence_id or not isolated:
        raise RolloutError("isolated fenced worktree required")
    return worktree_id, lease_id, fence_id


def _reject_worker_mode_change(payload: Mapping[str, Any]) -> str:
    if payload.get("worker_may_change_mode", False) is not False:
        raise RolloutError("workers cannot change rollout mode")
    requested = payload.get("requested_mode", "")
    if requested not in (None, "", GUARDED_MODE):
        raise RolloutError("workers cannot change rollout mode")
    current = spar040._mode(payload.get("current_mode", BOOTSTRAP_MODE), "current_mode")
    if current not in ALLOWED_CURRENT_MODES:
        raise RolloutError("SPAR-042 cannot skip or regress sealed rollout gates")
    return current


def _reject_self_merge_root_routing(payload: Mapping[str, Any]) -> None:
    claimed_merge = payload.get("merge", GUARDED_MERGE)
    if claimed_merge is True:
        raise RolloutError("guarded cannot self-merge; merge is current authority gates")
    if claimed_merge not in (False, "", None, GUARDED_MERGE):
        raise RolloutError("guarded cannot self-merge; merge is current authority gates")
    if payload.get("merge_worktree", False) is not False:
        raise RolloutError("guarded cannot self-merge; merge is current authority gates")
    if payload.get("self_merge", False) is not False:
        raise RolloutError("guarded cannot self-merge; merge is current authority gates")
    if payload.get("writes_repository", False) is not False:
        raise RolloutError("guarded cannot write the repository")
    if payload.get("apply_to_repository", False) is not False:
        raise RolloutError("guarded cannot write the repository")
    if payload.get("promote_root", False) is not False:
        raise RolloutError("guarded cannot promote a root")
    if payload.get("routing_influence", False) is not False:
        raise RolloutError("guarded cannot influence routing")
    if payload.get("influences_routing", False) is not False:
        raise RolloutError("guarded cannot influence routing")
    proposed_route = payload.get("proposed_route_decision_cid", "")
    route = payload.get("route_decision_cid", "")
    if proposed_route not in (None, "", route):
        raise RolloutError("guarded cannot influence routing")
    post = payload.get("post_world_root_cid", "")
    pre = payload.get("pre_world_root_cid", "")
    if post not in (None, "", pre):
        raise RolloutError("guarded cannot mutate or promote the world root")
    resulting = payload.get(
        "resulting_root_generation", payload.get("expected_root_generation")
    )
    expected = payload.get("expected_root_generation")
    if resulting not in (None, expected):
        raise RolloutError("guarded cannot promote root generation")


def _reject_unbounded_mutation(payload: Mapping[str, Any]) -> None:
    claimed = payload.get("source_mutation", GUARDED_SOURCE_MUTATION)
    if claimed is True:
        raise RolloutError(
            "guarded cannot mutate source outside Tier A and qualified Tier B"
        )
    if claimed not in (False, "", None, GUARDED_SOURCE_MUTATION):
        raise RolloutError(
            "guarded cannot mutate source outside Tier A and qualified Tier B"
        )
    if payload.get("packet_bounded", True) is not True:
        raise RolloutError("guarded applies bounded packets only")
    if payload.get("unrestricted_diff", False) is not False:
        raise RolloutError("guarded applies bounded packets only")


def _require_complete_analysis_planning_validation(payload: Mapping[str, Any]) -> None:
    if spar040._bool(payload.get("analysis_complete", False), "analysis_complete") is not True:
        raise RolloutError("guarded requires complete analysis")
    if spar040._bool(payload.get("planning_complete", False), "planning_complete") is not True:
        raise RolloutError("guarded requires complete planning")
    if spar040._bool(payload.get("validation_complete", False), "validation_complete") is not True:
        raise RolloutError("guarded requires full validation")


def _reject_worker_approval(payload: Mapping[str, Any]) -> None:
    if payload.get("approved", False) is not False:
        raise RolloutError("workers cannot self-approve a guarded wave")
    if payload.get("current_authority_approval", False) is not False:
        raise RolloutError("workers cannot self-approve a guarded wave")
    if payload.get("worker_self_approval", False) is not False:
        raise RolloutError("workers cannot self-approve a guarded wave")


def _nominated_transition_cid(payload: Mapping[str, Any]) -> str:
    return cid_for_dag_json(
        {
            "schema": NOMINATED_TRANSITION_SCHEMA,
            "interface": NOMINATED_TRANSITION_INTERFACE,
            "tree_id": payload["tree_id"],
            "transformation_packet_cid": payload["transformation_packet_cid"],
            "worktree_id": payload["worktree_id"],
            "nominated": True,
            "accepted": False,
            "merged": False,
            "root_promoted": False,
            "merge": GUARDED_MERGE,
        }
    )


def _resolve_transition(
    payload: Mapping[str, Any],
    *,
    mutate: bool,
) -> tuple[str, bool]:
    claimed = payload.get("refactor_transition_cid", "")
    nominated = payload.get("nominated_transition", None)
    if payload.get("accepted_transition", False) is not False:
        raise RolloutError("guarded cannot publish an accepted transition")
    if claimed in (None, ""):
        if not mutate:
            return "", False
        return _nominated_transition_cid(payload), True
    cid = spar040._cid(claimed, "refactor_transition_cid")
    if nominated is False:
        raise RolloutError("guarded cannot publish an accepted transition")
    if nominated not in (None, True):
        raise RolloutError("guarded publishes nominated transitions only")
    return cid, True


@dataclass(frozen=True, slots=True)
class GuardedWaveQualification:
    """Qualification of one wave against the guarded autonomy ceiling."""

    autonomy_tier: str
    selected: bool = False
    qualified: bool = False
    requires_approval: bool = False

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "autonomy_tier",
            "selected",
            "qualified",
            "requires_approval",
            "ceiling",
            "vector_authority",
            "model_authority",
            "qualification_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "autonomy_tier",
            spar040._enum(self.autonomy_tier, WaveAutonomyTier, "autonomy_tier"),
        )
        object.__setattr__(self, "selected", spar040._bool(self.selected, "selected"))
        object.__setattr__(self, "qualified", spar040._bool(self.qualified, "qualified"))
        computed = _computed_requires_approval(
            autonomy_tier=self.autonomy_tier,
            selected=self.selected,
            qualified=self.qualified,
        )
        claimed = spar040._bool(self.requires_approval, "requires_approval")
        if claimed != computed:
            raise RolloutError("requires_approval does not match guarded ceiling")
        object.__setattr__(self, "requires_approval", computed)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": GUARDED_WAVE_QUALIFICATION_SCHEMA,
            "interface": GUARDED_WAVE_QUALIFICATION_INTERFACE,
            "autonomy_tier": self.autonomy_tier,
            "selected": self.selected,
            "qualified": self.qualified,
            "requires_approval": self.requires_approval,
            "ceiling": GUARDED_SOURCE_MUTATION,
            "vector_authority": False,
            "model_authority": False,
        }
        spar040._require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def qualification_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["qualification_cid"] = self.qualification_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GuardedWaveQualification":
        spar040._reject_excluded(data, cls.__name__)
        payload = spar040._closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("qualification_cid")
        if payload.pop("schema") != GUARDED_WAVE_QUALIFICATION_SCHEMA:
            raise RolloutError("unsupported GuardedWaveQualification schema")
        if payload.pop("interface") != GUARDED_WAVE_QUALIFICATION_INTERFACE:
            raise RolloutError("unsupported GuardedWaveQualification interface")
        if payload.pop("ceiling") != GUARDED_SOURCE_MUTATION:
            raise RolloutError("qualification ceiling is Tier A and qualified Tier B")
        if payload.pop("vector_authority") is not False:
            raise RolloutError("vector similarity is not qualification authority")
        if payload.pop("model_authority") is not False:
            raise RolloutError("model output is not qualification authority")
        result = cls(**payload)
        spar040._verify_cid(
            claimed, result.qualification_cid, "GuardedWaveQualification qualification_cid"
        )
        return result

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | "GuardedWaveQualification"
    ) -> "GuardedWaveQualification":
        if isinstance(data, GuardedWaveQualification):
            return data
        if "qualification_cid" in data:
            return cls.from_dict(data)
        payload = spar040._mapping(data, "qualification")
        tier = payload.get("autonomy_tier", WaveAutonomyTier.A.value)
        selected = payload.get("selected", False)
        qualified = payload.get("qualified")
        if qualified is None:
            qualified = tier == WaveAutonomyTier.A.value
        computed = _computed_requires_approval(
            autonomy_tier=spar040._enum(tier, WaveAutonomyTier, "autonomy_tier"),
            selected=spar040._bool(selected, "selected"),
            qualified=spar040._bool(qualified, "qualified"),
        )
        return cls(
            autonomy_tier=tier,
            selected=selected,
            qualified=qualified,
            requires_approval=payload.get("requires_approval", computed),
        )


def _computed_requires_approval(
    *,
    autonomy_tier: str,
    selected: bool,
    qualified: bool,
) -> bool:
    if autonomy_tier == WaveAutonomyTier.A.value:
        if qualified is not True:
            raise RolloutError("Tier A waves are qualified")
        return False
    if autonomy_tier == WaveAutonomyTier.B.value:
        if qualified is True and selected is not True:
            raise RolloutError("Tier B qualification requires selection")
        return not (selected is True and qualified is True)
    if qualified is True or selected is True:
        raise RolloutError("waves above the guarded ceiling cannot be qualified")
    return True


def _qualification_from_payload(payload: Mapping[str, Any]) -> GuardedWaveQualification:
    if "qualification" in payload and payload.get("qualification") not in (None, "", {}):
        return GuardedWaveQualification.from_mapping(payload["qualification"])
    return GuardedWaveQualification.from_mapping(
        {
            "autonomy_tier": payload.get("autonomy_tier", WaveAutonomyTier.A.value),
            "selected": payload.get("selected", False),
            "qualified": payload.get("qualified", payload.get("autonomy_tier", "A") == "A"),
        }
    )


@dataclass(frozen=True, slots=True)
class GuardedRolloutReceipt:
    """Nomination-only SPAR-042 guarded rollout receipt."""

    tree_id: str
    status: str
    current_mode: str
    nominated_mode: str
    mode_contract: ModeContract
    comparison: CandidateComparison
    qualification: GuardedWaveQualification
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
    worktree_isolated: bool = True
    worktree_mutated: bool = False
    packet_applied: bool = False
    nominated_transition: bool = False
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
            "qualification",
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
            "worktree_isolated",
            "worktree_mutated",
            "packet_applied",
            "nominated_transition",
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
            self, "status", spar040._enum(self.status, GuardedStatus, "status")
        )
        object.__setattr__(
            self, "current_mode", spar040._mode(self.current_mode, "current_mode")
        )
        object.__setattr__(
            self, "nominated_mode", spar040._mode(self.nominated_mode, "nominated_mode")
        )
        if self.nominated_mode != GUARDED_MODE:
            raise RolloutError("SPAR-042 can only nominate guarded")
        if self.current_mode not in ALLOWED_CURRENT_MODES:
            raise RolloutError("SPAR-042 cannot skip or regress sealed rollout gates")
        contract = self.mode_contract
        if not isinstance(contract, ModeContract):
            contract = ModeContract.from_dict(contract)
        if contract.mode != GUARDED_MODE:
            raise RolloutError("guarded gate must bind the guarded contract")
        object.__setattr__(self, "mode_contract", contract)
        comparison = self.comparison
        if not isinstance(comparison, CandidateComparison):
            comparison = CandidateComparison.from_dict(comparison)
        if comparison.tree_id != self.tree_id:
            raise RolloutError("receipt tree_id does not match comparison")
        object.__setattr__(self, "comparison", comparison)
        qualification = self.qualification
        if not isinstance(qualification, GuardedWaveQualification):
            qualification = GuardedWaveQualification.from_dict(qualification)
        object.__setattr__(self, "qualification", qualification)
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
            raise RolloutError("guarded cannot mutate or promote the world root")
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
            raise RolloutError("guarded cannot promote root generation")
        object.__setattr__(
            self, "analyzer_id", spar040._text(self.analyzer_id, "analyzer_id")
        )
        if self.analyzer_id != ANALYZER_ID:
            raise RolloutError("analyzer_id must remain SPAR-042")
        object.__setattr__(
            self, "worktree_id", spar040._text(self.worktree_id, "worktree_id")
        )
        object.__setattr__(self, "lease_id", spar040._text(self.lease_id, "lease_id"))
        object.__setattr__(self, "fence_id", spar040._text(self.fence_id, "fence_id"))
        object.__setattr__(
            self,
            "worktree_isolated",
            spar040._bool(self.worktree_isolated, "worktree_isolated"),
        )
        if not self.worktree_isolated:
            raise RolloutError("isolated fenced worktree required")
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
            "nominated_transition",
            spar040._bool(self.nominated_transition, "nominated_transition"),
        )
        if self.packet_applied and not self.worktree_mutated:
            raise RolloutError("applied packets mutate only a qualified guarded wave")
        if self.worktree_mutated and not self.packet_applied:
            raise RolloutError("worktree mutation requires a bounded packet apply")
        if self.refactor_transition_cid and not self.nominated_transition:
            raise RolloutError("guarded publishes nominated transitions only")
        if self.nominated_transition and not self.refactor_transition_cid:
            raise RolloutError("nominated transition requires a transition identity")
        if self.packet_applied and self.qualification.requires_approval:
            raise RolloutError("waves above the guarded ceiling require approval")
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
        if self.status == GuardedStatus.NOMINATED_GUARDED.value:
            if self.terminal is not None:
                raise RolloutError("nominated guarded cannot carry a typed terminal")
            if not self.validation_receipt_cids:
                raise RolloutError("guarded requires full validation")
            if self.qualification.requires_approval:
                raise RolloutError(
                    "waves above the guarded ceiling require current-authority approval"
                )
        if self.status == GuardedStatus.TYPED_TERMINAL.value:
            if self.terminal is None:
                raise RolloutError("typed terminal status requires a typed terminal")
        if (
            self.status != GuardedStatus.TYPED_TERMINAL.value
            and self.terminal is not None
        ):
            raise RolloutError("non-terminal status cannot carry a typed terminal")

    @property
    def nominated(self) -> bool:
        return self.status == GuardedStatus.NOMINATED_GUARDED.value

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
    def merge(self) -> str:
        return GUARDED_MERGE

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
            "schema": GUARDED_ROLLOUT_RECEIPT_SCHEMA,
            "interface": GUARDED_ROLLOUT_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "status": self.status,
            "current_mode": self.current_mode,
            "nominated_mode": self.nominated_mode,
            "rollout_mode": self.nominated_mode,
            "mode_contract": self.mode_contract.to_dict(),
            "comparison": self.comparison.to_dict(),
            "qualification": self.qualification.to_dict(),
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
            "worktree_isolated": True,
            "worktree_mutated": self.worktree_mutated,
            "packet_applied": self.packet_applied,
            "nominated_transition": self.nominated_transition,
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
            "source_mutation": GUARDED_SOURCE_MUTATION,
            "merge": GUARDED_MERGE,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "GuardedRolloutReceipt":
        spar040._reject_excluded(data, cls.__name__)
        payload = spar040._closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != GUARDED_ROLLOUT_RECEIPT_SCHEMA:
            raise RolloutError("unsupported GuardedRolloutReceipt schema")
        if payload.pop("interface") != GUARDED_ROLLOUT_RECEIPT_INTERFACE:
            raise RolloutError("unsupported GuardedRolloutReceipt interface")
        if payload.pop("accepted") is not False:
            raise RolloutError("workers cannot self-approve a guarded wave")
        if payload.pop("gate_is_nomination_only") is not True:
            raise RolloutError("gate must remain nomination_only")
        if payload.pop("rollout_mode") != GUARDED_MODE:
            raise RolloutError("receipt rollout_mode must remain guarded")
        nominated = payload.pop("nominated")
        _pop_guarded_authority_flags(payload, "GuardedRolloutReceipt")
        payload["mode_contract"] = ModeContract.from_dict(payload["mode_contract"])
        payload["comparison"] = CandidateComparison.from_dict(payload["comparison"])
        payload["qualification"] = GuardedWaveQualification.from_dict(
            payload["qualification"]
        )
        if payload["terminal"] is not None:
            payload["terminal"] = TypedTerminal.from_dict(payload["terminal"])
        result = cls(**payload)
        if nominated is not result.nominated:
            raise RolloutError("nominated flag does not match status")
        spar040._verify_cid(claimed, result.receipt_cid, "GuardedRolloutReceipt receipt_cid")
        return result


def _receipt_from_payload(
    payload: Mapping[str, Any],
    *,
    status: str,
    current_mode: str,
    comparison: CandidateComparison,
    qualification: GuardedWaveQualification,
    mutate: bool,
    terminal: TypedTerminal | None = None,
) -> GuardedRolloutReceipt:
    pre = spar040._cid(payload.get("pre_world_root_cid"), "pre_world_root_cid")
    expected = spar040._int(
        payload.get("expected_root_generation", 0),
        "expected_root_generation",
        maximum=spar040.MAX_ROOT_GENERATION,
    )
    worktree_id, lease_id, fence_id = _require_isolated_fenced_worktree(payload)
    transition_cid = ""
    nominated_transition = False
    packet_applied = False
    worktree_mutated = False
    if terminal is None:
        transition_cid, nominated_transition = _resolve_transition(
            payload, mutate=mutate
        )
        packet_applied = mutate is True
        worktree_mutated = mutate is True
        if mutate is True and not nominated_transition:
            raise RolloutError("guarded publishes nominated transitions only")
    return GuardedRolloutReceipt(
        tree_id=payload.get("tree_id"),
        status=status,
        current_mode=current_mode,
        nominated_mode=GUARDED_MODE,
        mode_contract=ModeContract.for_mode(GUARDED_MODE),
        comparison=comparison,
        qualification=qualification,
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
        worktree_isolated=True,
        worktree_mutated=worktree_mutated,
        packet_applied=packet_applied,
        nominated_transition=nominated_transition,
        negative_evidence_cids=comparison.retained_negative_cids,
        terminal=terminal,
    )


def run_guarded_rollout(
    evidence: Mapping[str, Any],
    *,
    mutate: bool = False,
) -> GuardedRolloutReceipt:
    """Qualify Tier A / selected Tier B waves for current merge gates."""

    if mutate is not False and mutate is not True:
        raise RolloutError("mutate must be a boolean")
    payload = spar040._mapping(evidence, "guarded evidence")
    spar040._reject_non_admitting(payload, "guarded evidence")
    if payload.get("network", NETWORK_DENY) != NETWORK_DENY:
        raise RolloutError("network is denied")
    current_mode = _reject_worker_mode_change(payload)
    _reject_self_merge_root_routing(payload)
    _reject_unbounded_mutation(payload)
    _reject_worker_approval(payload)
    _require_isolated_fenced_worktree(payload)
    terminal = spar040._parse_terminal(payload.get("terminal"))
    candidates = spar040._parse_candidates(payload.get("candidates"))
    tree_id = spar040._tree_id(payload.get("tree_id"))
    comparison = compare_and_retain_candidates(candidates, tree_id=tree_id)
    qualification = _qualification_from_payload(payload)
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
    if terminal is None and qualification.requires_approval:
        terminal = TypedTerminal(
            kind=TerminalKind.HUMAN_REVIEW.value,
            reason="waves above the guarded ceiling require current-authority approval",
        )
    if terminal is not None:
        return _receipt_from_payload(
            payload,
            status=GuardedStatus.TYPED_TERMINAL.value,
            current_mode=current_mode,
            comparison=comparison,
            qualification=qualification,
            mutate=False,
            terminal=terminal,
        )
    _require_complete_analysis_planning_validation(payload)
    return _receipt_from_payload(
        payload,
        status=GuardedStatus.NOMINATED_GUARDED.value,
        current_mode=current_mode,
        comparison=comparison,
        qualification=qualification,
        mutate=mutate,
    )


def activate_guarded_rollout(evidence: Mapping[str, Any]) -> GuardedRolloutReceipt:
    """Nominate SPAR-042 activation of guarded from sealed shadow_apply."""

    return run_guarded_rollout(evidence)


def dry_run_guarded_rollout(evidence: Mapping[str, Any]) -> GuardedRolloutReceipt:
    """Deterministic dry-run. Never mutates and never self-merges."""

    return run_guarded_rollout(evidence, mutate=False)


def encode_canonical_receipt(receipt: GuardedRolloutReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(payload: Mapping[str, Any]) -> GuardedRolloutReceipt:
    return GuardedRolloutReceipt.from_dict(payload)


class GuardedRolloutGate:
    """SPAR-042 guarded rollout gate. Nomination-only; current authority merges."""

    interface: ClassVar[str] = GUARDED_ROLLOUT_GATE_INTERFACE
    schema: ClassVar[str] = GUARDED_ROLLOUT_GATE_SCHEMA
    analyzer_id: ClassVar[str] = ANALYZER_ID

    def run(
        self,
        evidence: Mapping[str, Any],
        *,
        mutate: bool = False,
    ) -> GuardedRolloutReceipt:
        return run_guarded_rollout(evidence, mutate=mutate)

    def activate(self, evidence: Mapping[str, Any]) -> GuardedRolloutReceipt:
        return activate_guarded_rollout(evidence)

    def compare(
        self,
        candidates: Sequence[ShadowPlanCandidate] | Sequence[Mapping[str, Any]],
        *,
        tree_id: str,
    ) -> CandidateComparison:
        return compare_and_retain_candidates(candidates, tree_id=tree_id)

    def dry_run(self, evidence: Mapping[str, Any]) -> GuardedRolloutReceipt:
        return dry_run_guarded_rollout(evidence)


def _evidence(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "current_mode": SHADOW_APPLY_MODE,
        "analysis_complete": True,
        "planning_complete": True,
        "validation_complete": True,
        "source_mutation": GUARDED_SOURCE_MUTATION,
        "merge": GUARDED_MERGE,
        "routing_influence": False,
        "worker_may_change_mode": False,
        "autonomy_tier": WaveAutonomyTier.A.value,
        "selected": False,
        "qualified": True,
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
        "worktree_isolated": True,
        "packet_bounded": True,
    }
    fields.update(overrides)
    return fields


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-042"
    assert GOAL_ID == "SPAR-G073"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert PREDECESSOR_TASK_IDS == ("SPAR-041",)
    assert GUARDED_ROLLOUT_GATE_INTERFACE == "GuardedRolloutGate@1"
    assert GUARDED_ROLLOUT_RECEIPT_INTERFACE == "GuardedRolloutReceipt@1"
    assert GUARDED_WAVE_QUALIFICATION_INTERFACE == "GuardedWaveQualification@1"
    assert ROLLOUT_CONTRACT_VERSION == "1"
    assert ROLLOUT_BASELINE_SCHEMA == "spar/rollout-baseline@1"
    assert ANALYZER_ID.endswith("guarded_rollout@1")
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
    assert GUARDED_MODE == "guarded"
    assert SHADOW_APPLY_MODE == "shadow_apply"
    assert SHADOW_PLAN_MODE == "shadow_plan"
    assert BOOTSTRAP_MODE == "bootstrap"
    assert ALLOWED_CURRENT_MODES == {"shadow_apply", "guarded"}
    assert PROGRESSIVE_MODES == ("shadow_apply", "guarded", "required")
    assert DECLARED_GATE_STATUSES == {"nominated_guarded", "typed_terminal"}
    assert DECLARED_TERMINAL_KINDS == {
        "unsupported",
        "human_review",
        "capability_unavailable",
    }
    assert DECLARED_WAVE_AUTONOMY_TIERS == set(DECLARED_AUTONOMY_TIERS)
    assert CEILING_TIERS == {"A", "B"}
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
    assert GUARDED_SOURCE_MUTATION == "Tier A and qualified Tier B"
    assert GUARDED_MERGE == "current authority gates"
    assert VECTOR_MODEL_AUTHORITY_REJECTED is True
    assert APPROVAL_REQUIRED_ABOVE_CEILING is True
    assert FULL_VALIDATION_REQUIRED is True
    assert COMPLETE_ANALYSIS_REQUIRED is True
    assert COMPLETE_PLANNING_REQUIRED is True
    assert FALSE_UNSAFE_CANDIDATES_RETAINED is True
    assert ISOLATED_WORKTREE_REQUIRED is True
    assert BOUNDED_PACKET_ONLY is True
    profile = rollout_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]
    assert "spar_narrow" in EXISTING_ADAPTER_AUTHORITIES
    assert "RolloutStore" in FORBIDDEN_ROLLOUT_NAMES
    assert "promote_root" in FORBIDDEN_ROLLOUT_NAMES
    descriptor = guarded_rollout_gate_descriptor()
    assert descriptor["nomination_only"] is True
    assert descriptor["writes_repository"] is False
    assert descriptor["worker_may_change_mode"] is False
    assert descriptor["influences_routing"] is False
    assert descriptor["promotes_root"] is False
    assert descriptor["merge"] == "current authority gates"
    assert descriptor["source_mutation"] == "Tier A and qualified Tier B"
    assert descriptor["nominated_mode"] == "guarded"
    assert descriptor["gate_task"] == "SPAR-042"
    assert descriptor["predecessor_task_ids"] == ["SPAR-041"]
    assert descriptor["vector_model_authority_rejected"] is True
    assert descriptor["approval_required_above_ceiling"] is True


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(TEST_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "GuardedRolloutGate" in names
    assert "GuardedRolloutReceipt" in names
    assert "GuardedWaveQualification" in names
    assert "RolloutStore" not in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "GuardedRolloutGate" in exports
    assert "GuardedWaveQualification" in exports
    assert "run_guarded_rollout" in exports
    assert "activate_guarded_rollout" in exports
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
    assert MODE_CONTRACTS["guarded"]["gate_task"] == "SPAR-042"
    assert MODE_CONTRACTS["guarded"]["source_mutation"] == "Tier A and qualified Tier B"
    assert MODE_CONTRACTS["guarded"]["merge"] == "current authority gates"
    assert MODE_CONTRACTS["shadow_apply"]["gate_task"] == "SPAR-041"
    assert MODE_CONTRACTS["required"]["gate_task"] == "SPAR-043"


def test_guarded_nominates_qualified_tier_a_without_self_merge() -> None:
    receipt = run_guarded_rollout(_evidence())
    assert receipt.status == "nominated_guarded"
    assert receipt.nominated is True
    assert receipt.accepted is False
    assert receipt.can_authorize_completion is False
    assert receipt.can_authorize_transition is False
    assert receipt.can_create_authority is False
    assert receipt.current_mode == SHADOW_APPLY_MODE
    assert receipt.nominated_mode == GUARDED_MODE
    assert receipt.rollout_mode == GUARDED_MODE
    assert receipt.mode_contract.gate_task == "SPAR-042"
    assert receipt.mode_contract.source_mutation == "Tier A and qualified Tier B"
    assert receipt.mode_contract.merge == "current authority gates"
    assert receipt.merge == "current authority gates"
    assert receipt.qualification.autonomy_tier == "A"
    assert receipt.qualification.qualified is True
    assert receipt.qualification.requires_approval is False
    assert receipt.post_world_root_cid == receipt.pre_world_root_cid
    assert receipt.resulting_root_generation == receipt.expected_root_generation
    assert receipt.root_promoted is False
    assert receipt.writes_repository is False
    assert receipt.packet_applied is False
    assert receipt.worktree_mutated is False
    encoded = encode_canonical_receipt(receipt)
    assert encoded["source_mutation"] == "Tier A and qualified Tier B"
    assert encoded["merge"] == "current authority gates"
    assert encoded["influences_routing"] is False
    assert encoded["worker_may_change_mode"] is False
    assert encoded["writes_repository"] is False
    assert encoded["root_promoted"] is False
    restored = decode_canonical_receipt(encoded)
    assert restored == receipt
    assert restored.receipt_cid == receipt.receipt_cid
    for field in RECEIPT_FLOOR:
        assert field in encoded


def test_selected_qualified_tier_b_is_permitted() -> None:
    receipt = run_guarded_rollout(
        _evidence(autonomy_tier="B", selected=True, qualified=True),
        mutate=True,
    )
    assert receipt.status == "nominated_guarded"
    assert receipt.qualification.autonomy_tier == "B"
    assert receipt.qualification.selected is True
    assert receipt.qualification.qualified is True
    assert receipt.qualification.requires_approval is False
    assert receipt.packet_applied is True
    assert receipt.worktree_mutated is True
    assert receipt.nominated_transition is True
    assert receipt.refactor_transition_cid
    assert receipt.merge == "current authority gates"
    assert receipt.root_promoted is False
    assert receipt.writes_repository is False
    assert receipt.accepted is False
    assert receipt.post_world_root_cid == receipt.pre_world_root_cid


def test_unqualified_or_unselected_tier_b_requires_approval() -> None:
    unselected = run_guarded_rollout(
        _evidence(autonomy_tier="B", selected=False, qualified=False)
    )
    assert unselected.status == "typed_terminal"
    assert unselected.nominated is False
    assert unselected.accepted is False
    assert unselected.terminal is not None
    assert unselected.terminal.kind == TerminalKind.HUMAN_REVIEW.value
    assert unselected.qualification.requires_approval is True
    assert unselected.packet_applied is False
    selected_unqualified = run_guarded_rollout(
        _evidence(autonomy_tier="B", selected=True, qualified=False)
    )
    assert selected_unqualified.status == "typed_terminal"
    assert selected_unqualified.terminal.kind == TerminalKind.HUMAN_REVIEW.value
    with pytest.raises(RolloutError, match="Tier B qualification requires selection"):
        run_guarded_rollout(
            _evidence(autonomy_tier="B", selected=False, qualified=True)
        )


def test_waves_above_ceiling_require_approval() -> None:
    for tier in ("C", "D", "E"):
        receipt = run_guarded_rollout(
            _evidence(autonomy_tier=tier, selected=False, qualified=False)
        )
        assert receipt.status == "typed_terminal"
        assert receipt.accepted is False
        assert receipt.nominated is False
        assert receipt.terminal.kind == TerminalKind.HUMAN_REVIEW.value
        assert receipt.qualification.requires_approval is True
        assert receipt.packet_applied is False
        with pytest.raises(RolloutError, match="cannot be qualified"):
            run_guarded_rollout(
                _evidence(autonomy_tier=tier, selected=True, qualified=True)
            )


def test_nominated_transitions_are_not_accepted_merges() -> None:
    transition = _cid("transition:nominated")
    receipt = run_guarded_rollout(
        _evidence(
            refactor_transition_cid=transition,
            nominated_transition=True,
        ),
        mutate=True,
    )
    assert receipt.refactor_transition_cid == transition
    assert receipt.nominated_transition is True
    assert receipt.merge == "current authority gates"
    assert receipt.root_promoted is False
    assert receipt.accepted is False
    with pytest.raises(RolloutError, match="accepted transition"):
        run_guarded_rollout(
            _evidence(
                refactor_transition_cid=transition,
                nominated_transition=False,
            ),
            mutate=True,
        )
    with pytest.raises(RolloutError, match="accepted transition"):
        run_guarded_rollout(_evidence(accepted_transition=True), mutate=True)


def test_false_and_unsafe_candidates_are_compared_and_retained() -> None:
    receipt = activate_guarded_rollout(_evidence())
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
        run_guarded_rollout(
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


def test_idempotent_when_already_in_guarded() -> None:
    receipt = run_guarded_rollout(_evidence(current_mode=GUARDED_MODE))
    assert receipt.status == "nominated_guarded"
    assert receipt.current_mode == GUARDED_MODE
    assert receipt.nominated_mode == GUARDED_MODE


def test_worker_cannot_change_or_skip_rollout_mode() -> None:
    with pytest.raises(RolloutError, match="cannot change rollout"):
        run_guarded_rollout(_evidence(worker_may_change_mode=True))
    with pytest.raises(RolloutError, match="cannot change rollout"):
        run_guarded_rollout(_evidence(requested_mode="required"))
    with pytest.raises(RolloutError, match="cannot change rollout"):
        run_guarded_rollout(_evidence(requested_mode="shadow_apply"))
    with pytest.raises(RolloutError, match="cannot change rollout"):
        run_guarded_rollout(_evidence(requested_mode="bootstrap"))
    with pytest.raises(RolloutError, match="cannot skip or regress"):
        run_guarded_rollout(_evidence(current_mode=BOOTSTRAP_MODE))
    with pytest.raises(RolloutError, match="cannot skip or regress"):
        run_guarded_rollout(_evidence(current_mode=SHADOW_PLAN_MODE))
    with pytest.raises(RolloutError, match="cannot skip or regress"):
        run_guarded_rollout(_evidence(current_mode="required"))


def test_non_isolated_or_unfenced_worktree_fails_closed() -> None:
    with pytest.raises(RolloutError, match="isolated fenced worktree"):
        run_guarded_rollout(_evidence(worktree_isolated=False), mutate=True)
    with pytest.raises(RolloutError, match="isolated fenced worktree"):
        run_guarded_rollout(_evidence(worktree_id=""), mutate=True)
    with pytest.raises(RolloutError, match="isolated fenced worktree"):
        run_guarded_rollout(_evidence(lease_id=""), mutate=True)
    with pytest.raises(RolloutError, match="isolated fenced worktree"):
        run_guarded_rollout(_evidence(fence_id=""), mutate=True)


def test_repository_write_self_merge_and_root_promotion_fail_closed() -> None:
    with pytest.raises(RolloutError, match="cannot mutate source"):
        run_guarded_rollout(_evidence(source_mutation=True), mutate=True)
    with pytest.raises(RolloutError, match="cannot mutate source"):
        run_guarded_rollout(
            _evidence(source_mutation="disposable_worktree_only"), mutate=True
        )
    with pytest.raises(RolloutError, match="cannot write the repository"):
        run_guarded_rollout(_evidence(writes_repository=True), mutate=True)
    with pytest.raises(RolloutError, match="cannot write the repository"):
        run_guarded_rollout(_evidence(apply_to_repository=True), mutate=True)
    with pytest.raises(RolloutError, match="cannot self-merge"):
        run_guarded_rollout(_evidence(merge=True), mutate=True)
    with pytest.raises(RolloutError, match="cannot self-merge"):
        run_guarded_rollout(_evidence(merge_worktree=True), mutate=True)
    with pytest.raises(RolloutError, match="cannot influence routing"):
        run_guarded_rollout(_evidence(routing_influence=True))
    with pytest.raises(RolloutError, match="cannot influence routing"):
        run_guarded_rollout(
            _evidence(proposed_route_decision_cid=_cid("route:other"))
        )
    with pytest.raises(RolloutError, match="cannot mutate or promote"):
        run_guarded_rollout(_evidence(post_world_root_cid=_cid("world-root:post")))
    with pytest.raises(RolloutError, match="cannot promote root generation"):
        run_guarded_rollout(_evidence(resulting_root_generation=4))
    with pytest.raises(RolloutError, match="cannot promote a root"):
        run_guarded_rollout(_evidence(promote_root=True), mutate=True)
    with pytest.raises(RolloutError, match="bounded packets only"):
        run_guarded_rollout(_evidence(packet_bounded=False), mutate=True)


def test_full_validation_is_required() -> None:
    with pytest.raises(RolloutError, match="complete analysis"):
        run_guarded_rollout(_evidence(analysis_complete=False))
    with pytest.raises(RolloutError, match="complete planning"):
        run_guarded_rollout(_evidence(planning_complete=False))
    with pytest.raises(RolloutError, match="full validation"):
        run_guarded_rollout(_evidence(validation_complete=False))
    with pytest.raises(RolloutError, match="full validation"):
        run_guarded_rollout(_evidence(validation_receipt_cids=[]))
    with pytest.raises(RolloutError, match="requires planned candidates"):
        run_guarded_rollout(_evidence(candidates=[]))


def test_capability_unavailable_is_typed_terminal() -> None:
    receipt = run_guarded_rollout(_evidence(analysis_available=False))
    assert receipt.status == "typed_terminal"
    assert receipt.accepted is False
    assert receipt.nominated is False
    assert receipt.terminal is not None
    assert receipt.terminal.kind == TerminalKind.CAPABILITY_UNAVAILABLE.value
    assert receipt.rollout_mode == GUARDED_MODE
    planning = run_guarded_rollout(_evidence(planning_available=False))
    assert planning.terminal.kind == TerminalKind.CAPABILITY_UNAVAILABLE.value
    validation = run_guarded_rollout(_evidence(validation_available=False))
    assert validation.terminal.kind == TerminalKind.CAPABILITY_UNAVAILABLE.value
    assert validation.packet_applied is False


def test_explicit_human_review_terminal_stops_without_completion() -> None:
    receipt = run_guarded_rollout(
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
    first = dry_run_guarded_rollout(payload)
    second = GuardedRolloutGate().dry_run(payload)
    assert first.receipt_cid == second.receipt_cid
    assert first.packet_applied is False
    assert first.worktree_mutated is False
    assert DRY_RUN_MUTATES is False
    applied = run_guarded_rollout(payload, mutate=True)
    assert applied.receipt_cid != first.receipt_cid
    assert applied.packet_applied is True


def test_identity_excludes_observational_fields() -> None:
    receipt = run_guarded_rollout(_evidence())
    encoded = receipt.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(encoded))
    encoded["timestamp"] = "now"
    with pytest.raises(RolloutError, match="observational"):
        GuardedRolloutReceipt.from_dict(encoded)
    with pytest.raises(RolloutError, match="observational"):
        run_guarded_rollout(_evidence(model_output="guess"))


def test_vector_and_model_evidence_cannot_qualify() -> None:
    with pytest.raises(RolloutError, match="cannot admit"):
        run_guarded_rollout(_evidence(vector_candidate={"score": 1}))
    with pytest.raises(RolloutError, match="cannot admit"):
        run_guarded_rollout(_evidence(model_hypothesis={"ok": True}))
    with pytest.raises(RolloutError, match="cannot admit"):
        run_guarded_rollout(_evidence(heuristic=True))
    qualification = GuardedWaveQualification(
        autonomy_tier="A",
        selected=False,
        qualified=True,
        requires_approval=False,
    )
    payload = qualification.to_dict()
    payload["vector_authority"] = True
    payload["qualification_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "qualification_cid"}
    )
    with pytest.raises(RolloutError, match="vector similarity"):
        GuardedWaveQualification.from_dict(payload)
    payload = qualification.to_dict()
    payload["model_authority"] = True
    payload["qualification_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "qualification_cid"}
    )
    with pytest.raises(RolloutError, match="model output"):
        GuardedWaveQualification.from_dict(payload)


def test_network_is_denied() -> None:
    with pytest.raises(RolloutError, match="network is denied"):
        run_guarded_rollout(_evidence(network="allow"))


def test_tree_mismatch_fails_closed() -> None:
    receipt = run_guarded_rollout(_evidence())
    payload = receipt.to_dict()
    payload["tree_id"] = OTHER_TREE
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(RolloutError, match="tree_id"):
        GuardedRolloutReceipt.from_dict(payload)
    with pytest.raises(RolloutError, match="tree_id"):
        run_guarded_rollout(_evidence(tree_id="not-a-tree"))


def test_worker_cannot_self_approve_receipt() -> None:
    receipt = run_guarded_rollout(_evidence())
    payload = receipt.to_dict()
    payload["accepted"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(RolloutError, match="self-approve"):
        GuardedRolloutReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["can_authorize_completion"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(RolloutError, match="can_authorize_completion"):
        GuardedRolloutReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["worker_may_change_mode"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(RolloutError, match="cannot claim worker_may_change_mode"):
        GuardedRolloutReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["source_mutation"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(RolloutError, match="source_mutation is Tier A and qualified Tier B"):
        GuardedRolloutReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["merge"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(RolloutError, match="merge is current authority gates"):
        GuardedRolloutReceipt.from_dict(payload)
    with pytest.raises(RolloutError, match="self-approve"):
        run_guarded_rollout(_evidence(approved=True))
    with pytest.raises(RolloutError, match="self-approve"):
        run_guarded_rollout(_evidence(current_authority_approval=True))


def test_mode_contract_cannot_be_rewritten_by_workers() -> None:
    with pytest.raises(RolloutError, match="source_mutation does not match"):
        ModeContract(
            mode="guarded",
            source_mutation=True,
            merge="current authority gates",
            gate_task="SPAR-042",
        )
    with pytest.raises(RolloutError, match="merge does not match"):
        ModeContract(
            mode="guarded",
            source_mutation="Tier A and qualified Tier B",
            merge=True,
            gate_task="SPAR-042",
        )
    with pytest.raises(RolloutError, match="gate_task does not match"):
        ModeContract(
            mode="guarded",
            source_mutation="Tier A and qualified Tier B",
            merge="current authority gates",
            gate_task="SPAR-041",
        )
    contract = ModeContract.for_mode("guarded")
    restored = ModeContract.from_dict(contract.to_dict())
    assert restored == contract
    assert contract.source_mutation == "Tier A and qualified Tier B"
    assert contract.merge == "current authority gates"
    payload = contract.to_dict()
    payload["worker_may_change_mode"] = True
    payload["contract_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "contract_cid"}
    )
    with pytest.raises(RolloutError, match="cannot change rollout"):
        ModeContract.from_dict(payload)


def test_gate_methods_match_module_functions() -> None:
    gate = GuardedRolloutGate()
    payload = _evidence()
    assert gate.interface == GUARDED_ROLLOUT_GATE_INTERFACE
    assert gate.run(payload).receipt_cid == run_guarded_rollout(payload).receipt_cid
    assert (
        gate.activate(payload).receipt_cid
        == activate_guarded_rollout(payload).receipt_cid
    )
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
    assert RolloutMode.GUARDED.value == "guarded"
    assert CandidateDisposition.UNSAFE.value == "unsafe"
    assert GuardedStatus.NOMINATED_GUARDED.value == "nominated_guarded"
    qualification = GuardedWaveQualification.from_dict(
        GuardedWaveQualification(
            autonomy_tier="B",
            selected=True,
            qualified=True,
            requires_approval=False,
        ).to_dict()
    )
    assert qualification.requires_approval is False
