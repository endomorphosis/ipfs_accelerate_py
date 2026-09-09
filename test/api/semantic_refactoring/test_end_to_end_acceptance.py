"""Independent contract tests for SPAR-046 end-to-end acceptance matrix."""

from __future__ import annotations

import ast
import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.codemod import (
    MemberLocator,
    TargetKind,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.extraction_wave import (
    ExtractionWaveError,
    dry_run_extraction_wave,
    execute_extraction_wave,
    rollback_extraction_wave,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.facade_planner import (
    CompatibilityDisposition,
    CompatibilityKind,
    plan_compatibility_facades,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.fixed_point import (
    REBUILD_SLICE_ORDER,
    dry_run_fixed_point,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.opportunity_detector import (
    compile_durable_goals,
    detect_monolith_opportunities,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.partition_generators import (
    GeneratorKind,
    ProgramPartitionCandidate,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.proof_adapter import (
    ArtifactKind,
    ProofAdapterError,
    reconstruct_proof,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.refactor_memory import (
    ReuseDecisionKind,
    RevokeReason,
    TransitionOutcome,
    compile_refactor_transition,
    compile_reuse_key,
    decide_exact_reuse,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.service import (
    ControlRequest,
    ControlStatus,
    SemanticRefactoringService,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.task_compiler import (
    TaskKind,
    compile_backlog,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.transformation_packet import (
    compile_refactor_transformation_packet,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.translation_validation import (
    DEFAULT_REQUIRED_DIMENSIONS,
    ValidationStatus,
    admitted_evidence_classes,
    compile_translation_validation_request,
    dry_run_translation_validation,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.world_root_adapter import (
    ACCELERATOR_TASK_OWNER,
    AdapterStatus,
    NETWORK_DENY as WORLD_ROOT_NETWORK_DENY,
    integrate_world_root,
    recover_world_root,
)

from test.api.semantic_refactoring.test_benchmark_corpus import (
    BenchmarkCorpusError,
    generate_controlled_monolith,
    load_benchmark_corpus_manifest,
    load_preregistration,
    load_split_policy,
    seal_fixture_splits,
)


ROOT = Path(__file__).resolve().parents[3]
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "test/api/semantic_refactoring/test_end_to_end_acceptance.py",
    "benchmarks/agent_supervisor/semantic_refactoring/acceptance_matrix.json",
)
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
    "scripts/materialize_semantic_preserving_remodularization_program.py",
    "scripts/ops/agent_supervisor/semantic_preserving_remodularization.py",
    "scripts/validate_semantic_preserving_remodularization_board.py",
    "scripts/validate_semantic_preserving_remodularization_dependencies.py",
    "test/api/semantic_refactoring/test_bootstrap_controls.py",
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
)
TREE_ID = "fbc6fa1ddefb2f9ecb7b5c718d618e3b60aa3051"
OTHER_TREE = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
MATRIX_RELATIVE = "benchmarks/agent_supervisor/semantic_refactoring/acceptance_matrix.json"
PREREGISTRATION_RELATIVE = (
    "benchmarks/agent_supervisor/semantic_refactoring/preregistration.json"
)
CORPUS_MANIFEST_RELATIVE = (
    "benchmarks/agent_supervisor/semantic_refactoring/corpus_manifest.json"
)
WRITE_PATHS = ("pkg/mod.py", "pkg/extracted.py")
VALIDATION = ("python3 -m pytest -q tests/test_mod.py",)
TOOLCHAIN_ID = "toolchain:hammer@1"
PROCEDURE_VERSION = "procedure:v1"
SOURCE_WITH_COMMENT = '''"""mod doc"""
from x import y

# keep this comment
def leaf():
    # inner comment
    return 1

class Other:
    pass
'''

TASK_ID: str = "SPAR-046"
GOAL_ID: str = "SPAR-G081"
PROGRAM: str = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: str = "acceptance-matrix"
AUTHORITY_OWNER: str = "ipfs_accelerate_py"
ANALYZER_ID: str = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.end_to_end_acceptance@1"
)
PREDECESSOR_TASK_IDS: tuple[str, ...] = ("SPAR-044", "SPAR-045")
SPAR_ACCEPTANCE_MATRIX_INTERFACE: str = "SparEndToEndAcceptanceMatrix@1"
SPAR_ACCEPTANCE_RECEIPT_INTERFACE: str = "SparEndToEndAcceptanceReceipt@1"
SPAR_ACCEPTANCE_SCENARIO_INTERFACE: str = "SparAcceptanceScenarioResult@1"
SPAR_ACCEPTANCE_MATRIX_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/spar-acceptance-matrix@1"
)
SPAR_ACCEPTANCE_RECEIPT_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/spar-acceptance-receipt@1"
)
SPAR_ACCEPTANCE_SCENARIO_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/spar-acceptance-scenario-result@1"
)
ACCEPTANCE_CONTRACT_VERSION: str = "1"
NETWORK_DENY: str = "deny"
CAN_AUTHORIZE_COMPLETION: bool = False
CAN_AUTHORIZE_TRANSITION: bool = False
CAN_CREATE_AUTHORITY: bool = False
GATE_WRITES_REPOSITORY: bool = False
GATE_IS_NOMINATION_ONLY: bool = True
VECTOR_SIMILARITY_IS_AUTHORITY: bool = False
PROJECTION_CLUSTERING_IS_AUTHORITY: bool = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: bool = True
TEST_PASS_IS_NOT_COMPLETION: bool = True
MARKDOWN_IS_NOT_COMPLETION: bool = True
WORKER_SELF_APPROVAL: bool = False
DUCKLAKE_IS_AUTHORITY: bool = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: bool = True
NETWORK_DENIED: bool = True
RAW_SOURCE_REQUIRED: bool = True
DRY_RUN_IS_DETERMINISTIC: bool = True
DRY_RUN_MUTATES: bool = False

REQUIRED_STAGES: tuple[str, ...] = (
    "inventory",
    "extraction",
    "facade",
    "validation",
    "transition",
    "restart",
    "reuse",
    "replan",
    "rollback",
    "corruption",
    "staleness",
    "unsafe_splits",
    "failed_proofs",
    "root_conflicts",
)
DECLARED_STATUSES: frozenset[str] = frozenset(
    {"nominated", "rejected", "typed_terminal"}
)
IDENTITY_EXCLUDED_FIELDS: frozenset[str] = frozenset(
    {
        "timestamp",
        "timestamps",
        "process_id",
        "pid",
        "local_path",
        "local_paths",
        "checkout_path",
        "model_output",
        "model",
        "provider",
        "prompt",
        "lease",
        "fence",
        "generation",
        "receipt",
        "acceptance",
        "wall_clock",
        "clock",
        "source",
        "source_text",
        "source_body",
        "file_contents",
        "repository_dump",
    }
)
_AUTHORITY_FLAGS: tuple[str, ...] = (
    "can_authorize_transition",
    "can_authorize_completion",
    "can_create_authority",
    "writes_repository",
    "worker_self_approval",
    "projection_is_authority",
)


class EndToEndAcceptanceError(RuntimeError):
    """Fail-closed SPAR-046 acceptance-matrix contract violation."""


def provider_free_exports() -> tuple[str, ...]:
    return (
        "SparAcceptanceMatrix",
        "SparAcceptanceScenarioResult",
        "SparEndToEndAcceptanceReceipt",
        "load_acceptance_matrix",
        "nominate_end_to_end_acceptance",
        "run_acceptance_scenario",
        "run_end_to_end_acceptance_matrix",
    )


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & set(CAPSULE_TYPES)
    if overlap:
        raise EndToEndAcceptanceError(
            f"acceptance matrix must not define competing types: {sorted(overlap)}"
        )


def acceptance_cid_profile() -> dict[str, str]:
    return {
        "codec": "dag-json",
        "hash": "sha2-256",
        "rule": "content-addressed over sealed fields; not universal meaning",
    }


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _load_json(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EndToEndAcceptanceError(f"{relative} must be a JSON object")
    return payload


def load_acceptance_matrix() -> dict[str, Any]:
    payload = _load_json(MATRIX_RELATIVE)
    if payload.get("schema") != SPAR_ACCEPTANCE_MATRIX_SCHEMA:
        raise EndToEndAcceptanceError("unsupported acceptance matrix schema")
    if payload.get("interface") != SPAR_ACCEPTANCE_MATRIX_INTERFACE:
        raise EndToEndAcceptanceError("unsupported acceptance matrix interface")
    if payload.get("task_id") != TASK_ID:
        raise EndToEndAcceptanceError("acceptance matrix task_id must remain SPAR-046")
    if payload.get("nomination_only") is not True:
        raise EndToEndAcceptanceError("acceptance matrix must remain nomination_only")
    for flag in _AUTHORITY_FLAGS:
        if payload.get(flag) is True:
            raise EndToEndAcceptanceError(f"acceptance matrix cannot claim {flag}")
    return payload


@dataclass(frozen=True, slots=True)
class SparAcceptanceMatrix:
    """Sealed SPAR-046 acceptance matrix bound to SPAR-044/045 predecessors."""

    payload: Mapping[str, Any]
    matrix_cid: str

    def identity_payload(self) -> dict[str, Any]:
        return {
            "interface": SPAR_ACCEPTANCE_MATRIX_INTERFACE,
            "matrix_cid": self.matrix_cid,
            "payload": dict(self.payload),
            "schema": SPAR_ACCEPTANCE_MATRIX_SCHEMA,
            "task_id": TASK_ID,
        }


@dataclass(frozen=True, slots=True)
class SparAcceptanceScenarioResult:
    """One SPAR-046 scenario observation. Nomination-only."""

    scenario_id: str
    stage: str
    status: str
    evidence_cid: str
    expected_status: str
    expected_terminal: str = ""
    terminal: str = ""
    safety_floor: str = ""
    nominated: bool = True
    accepted: bool = False
    promotes: bool = False
    writes_repository: bool = False
    interface: str = SPAR_ACCEPTANCE_SCENARIO_INTERFACE
    schema: str = SPAR_ACCEPTANCE_SCENARIO_SCHEMA

    def identity_payload(self) -> dict[str, Any]:
        return {
            "accepted": False,
            "evidence_cid": self.evidence_cid,
            "expected_status": self.expected_status,
            "expected_terminal": self.expected_terminal,
            "interface": self.interface,
            "nominated": True,
            "promotes": False,
            "safety_floor": self.safety_floor,
            "scenario_id": self.scenario_id,
            "schema": self.schema,
            "stage": self.stage,
            "status": self.status,
            "terminal": self.terminal,
            "writes_repository": False,
        }

    @property
    def result_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())


@dataclass(frozen=True, slots=True)
class SparEndToEndAcceptanceReceipt:
    """Nomination-only SPAR-046 acceptance-matrix receipt."""

    tree_id: str
    matrix_cid: str
    scenario_results: tuple[SparAcceptanceScenarioResult, ...]
    stages: tuple[str, ...]
    nominated: bool = True
    accepted: bool = False
    nomination_only: bool = True
    can_authorize_completion: bool = False
    can_authorize_transition: bool = False
    can_create_authority: bool = False
    writes_repository: bool = False
    worker_self_approval: bool = False
    projection_is_authority: bool = False
    network: str = NETWORK_DENY
    analyzer_id: str = ANALYZER_ID
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID
    interface: str = SPAR_ACCEPTANCE_RECEIPT_INTERFACE
    schema: str = SPAR_ACCEPTANCE_RECEIPT_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "accepted",
            "analyzer_id",
            "can_authorize_completion",
            "can_authorize_transition",
            "can_create_authority",
            "goal_id",
            "interface",
            "matrix_cid",
            "network",
            "nominated",
            "nomination_only",
            "projection_is_authority",
            "receipt_cid",
            "scenario_results",
            "schema",
            "stages",
            "task_id",
            "tree_id",
            "worker_self_approval",
            "writes_repository",
        }
    )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "accepted": False,
            "analyzer_id": self.analyzer_id,
            "can_authorize_completion": False,
            "can_authorize_transition": False,
            "can_create_authority": False,
            "goal_id": self.goal_id,
            "interface": self.interface,
            "matrix_cid": self.matrix_cid,
            "network": NETWORK_DENY,
            "nominated": True,
            "nomination_only": True,
            "projection_is_authority": False,
            "scenario_results": [item.identity_payload() for item in self.scenario_results],
            "schema": self.schema,
            "stages": list(self.stages),
            "task_id": self.task_id,
            "tree_id": self.tree_id,
            "worker_self_approval": False,
            "writes_repository": False,
        }

    @property
    def receipt_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["receipt_cid"] = self.receipt_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SparEndToEndAcceptanceReceipt":
        excluded = set(data) & IDENTITY_EXCLUDED_FIELDS
        if excluded:
            raise EndToEndAcceptanceError(
                f"observational fields are excluded from identity: {sorted(excluded)}"
            )
        unknown = set(data) - cls._FIELDS
        if unknown:
            raise EndToEndAcceptanceError(
                f"unsupported SparEndToEndAcceptanceReceipt fields: {sorted(unknown)}"
            )
        payload = dict(data)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != SPAR_ACCEPTANCE_RECEIPT_SCHEMA:
            raise EndToEndAcceptanceError("unsupported acceptance receipt schema")
        if payload.pop("interface") != SPAR_ACCEPTANCE_RECEIPT_INTERFACE:
            raise EndToEndAcceptanceError("unsupported acceptance receipt interface")
        if payload.pop("accepted") is not False:
            raise EndToEndAcceptanceError("workers cannot self-approve SPAR-046")
        if payload.pop("nomination_only") is not True:
            raise EndToEndAcceptanceError("receipt must remain nomination_only")
        if payload.pop("nominated") is not True:
            raise EndToEndAcceptanceError("receipt must remain nominated")
        for flag in _AUTHORITY_FLAGS:
            if payload.pop(flag) is not False:
                raise EndToEndAcceptanceError(f"receipt cannot claim {flag}")
        if payload.pop("network") != NETWORK_DENY:
            raise EndToEndAcceptanceError("receipt network must remain deny")
        results = tuple(
            SparAcceptanceScenarioResult(
                scenario_id=str(item["scenario_id"]),
                stage=str(item["stage"]),
                status=str(item["status"]),
                evidence_cid=str(item["evidence_cid"]),
                expected_status=str(item["expected_status"]),
                expected_terminal=str(item.get("expected_terminal") or ""),
                terminal=str(item.get("terminal") or ""),
                safety_floor=str(item.get("safety_floor") or ""),
            )
            for item in payload["scenario_results"]
        )
        result = cls(
            tree_id=str(payload["tree_id"]),
            matrix_cid=str(payload["matrix_cid"]),
            scenario_results=results,
            stages=tuple(str(item) for item in payload["stages"]),
            analyzer_id=str(payload["analyzer_id"]),
            task_id=str(payload["task_id"]),
            goal_id=str(payload["goal_id"]),
        )
        if claimed != result.receipt_cid:
            raise EndToEndAcceptanceError(
                "SparEndToEndAcceptanceReceipt receipt_cid mismatch"
            )
        return result


def _candidate(**overrides: Any) -> ProgramPartitionCandidate:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "generator_kind": GeneratorKind.SCC,
        "member_ids": ("node:leaf",),
        "admitted": True,
        "consumer_ids": ("pkg.cli",),
    }
    fields.update(overrides)
    return ProgramPartitionCandidate(**fields)


def _contracts(partition_cid: str) -> dict[str, Any]:
    return {
        "tree_id": TREE_ID,
        "source_cid": _cid("source"),
        "partition_cid": partition_cid,
        "contract_set_cid": _cid("contracts"),
        "contracts": [
            {
                "edge_id": "edge:import",
                "source_id": "pkg.cli",
                "target_id": "node:leaf",
                "kind": "import",
                "disposition": "admitted",
                "complete": True,
                "required": True,
                "allowed_effects": ["bounded_source_edit", "isolated_validation"],
                "forbidden_effects": ["network"],
            }
        ],
    }


def _target_api(candidate: ProgramPartitionCandidate) -> dict[str, Any]:
    return {
        "tree_id": TREE_ID,
        "plan_cid": _cid("target-api"),
        "modules": [
            {
                "module_id": candidate.candidate_cid,
                "member_ids": list(candidate.member_ids),
                "public_exports": [
                    {
                        "member_id": candidate.member_ids[0],
                        "consumer_ids": ["pkg.cli"],
                    }
                ],
            }
        ],
    }


def _facade(candidate: ProgramPartitionCandidate) -> dict[str, Any]:
    return {
        "tree_id": TREE_ID,
        "plan_cid": _cid("facade"),
        "consumer_plans": [
            {
                "obligation_id": "obl:import",
                "consumer_id": "pkg.cli",
                "subject_id": "symbol:pkg.mod.Record",
                "subject_module": "pkg.mod",
                "kind": "import_path",
                "disposition": CompatibilityDisposition.PRESERVE.value,
                "migration_kind": "reexport",
                "required": True,
                "target_module_id": candidate.candidate_cid,
            }
        ],
        "subject_facades": [
            {
                "subject_id": "symbol:pkg.mod.Record",
                "subject_module": "pkg.mod",
                "facade_required": False,
                "consumer_ids": ["pkg.cli"],
                "undispositioned_consumer_ids": [],
                "migration_kinds": ["reexport"],
                "target_module_id": candidate.candidate_cid,
                "can_retire_facade": False,
            }
        ],
    }


def _packet() -> Any:
    candidate = _candidate()
    return compile_refactor_transformation_packet(
        boundary_contracts=_contracts(candidate.candidate_cid),
        target_api_plan=_target_api(candidate),
        facade_plan=_facade(candidate),
        candidates=(candidate,),
        preimage={
            "repository_id": PROGRAM,
            "environment_cid": _cid("env"),
            "graph_cid": _cid("graph"),
            "source_cids": [_cid("source")],
        },
        write_paths=WRITE_PATHS,
        lease_id=_cid("lease"),
        fence_id=_cid("fence"),
        epoch_id=_cid("epoch"),
        validation_commands=VALIDATION,
        repository_id=PROGRAM,
    )


def _wave_kwargs() -> dict[str, Any]:
    candidate = _candidate()
    return {
        "raw_sources": {"pkg/mod.py": SOURCE_WITH_COMMENT, "pkg/extracted.py": ""},
        "locators": (
            MemberLocator(
                member_id="node:leaf",
                path="pkg/mod.py",
                symbol="leaf",
                kind=TargetKind.FUNCTION,
            ),
        ),
        "destination_paths": {candidate.candidate_cid: "pkg/extracted.py"},
    }


def _reuse_key(**overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "state_cid": _cid("state"),
        "partition_cid": _cid("partition"),
        "policy_cid": _cid("policy"),
        "environment_cid": _cid("env"),
        "toolchain_id": TOOLCHAIN_ID,
        "obligation_root_cid": _cid("obligations"),
        "validation_cid": _cid("validation"),
        "procedure_version": PROCEDURE_VERSION,
    }
    fields.update(overrides)
    return compile_reuse_key(**fields)


def _transition(**overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "reuse_key": _reuse_key(),
        "outcome": TransitionOutcome.ACCEPTED.value,
        "wave_receipt_cid": _cid("wave"),
        "validation_result_cid": _cid("tv-result"),
        "proof_receipt_cid": _cid("proof"),
        "packet_cid": _cid("packet"),
        "raw_source_cids": [_cid("source")],
        "write_paths": list(WRITE_PATHS),
        "validation_commands": list(VALIDATION),
        "evidence_class": "transition",
    }
    fields.update(overrides)
    return compile_refactor_transition(**fields)


def _passing_evidence() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for dimension in DEFAULT_REQUIRED_DIMENSIONS:
        evidence_class = sorted(admitted_evidence_classes(dimension))[0]
        items.append(
            {
                "dimension": dimension,
                "evidence_class": evidence_class,
                "evidence_cid": _cid(f"{dimension}:{evidence_class}"),
                "status": "pass",
                "full_suite_executed": dimension == "tests",
            }
        )
    return items


def _world_root_evidence(**overrides: Any) -> dict[str, Any]:
    root = _cid("world-root:pre")
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "pre_world_root_cid": root,
        "current_world_root_cid": root,
        "expected_root_generation": 3,
        "current_root_generation": 3,
        "packet_cids": [_cid("packet")],
        "projection_cids": [_cid("projection")],
        "receipt_cids": [_cid("receipt")],
        "transition_cids": [_cid("transition")],
        "task_owner": ACCELERATOR_TASK_OWNER,
        "gitlinks": [],
        "nested_write_paths": [],
        "kit_vfs_available": True,
        "crash": False,
        "network": WORLD_ROOT_NETWORK_DENY,
        "worktree_id": _cid("worktree"),
        "lease_id": "lease-1",
        "fence_id": "fence-1",
    }
    fields.update(overrides)
    return fields


def _rebuild_slices(label: str) -> dict[str, str]:
    return {name: _cid(f"{label}:{name}") for name in REBUILD_SLICE_ORDER}


def _module(
    module_id: str,
    *,
    loc: int = 10,
    member_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    return {
        "module_id": module_id,
        "loc": loc,
        "member_ids": list(member_ids or (module_id.replace("mod:", "node:"),)),
        "public_export_ids": [],
    }


def _scope(module_id: str, *, evidence_label: str = "ev:old") -> dict[str, Any]:
    suffix = module_id.split(":", 1)[-1]
    return {
        "module_id": module_id,
        "write_paths": [f"pkg/{suffix}.py"],
        "member_ids": [module_id.replace("mod:", "node:")],
        "evidence_cids": [_cid(evidence_label)],
        "validation_commands": [
            "python3 -m pytest -q test/api/semantic_refactoring/test_end_to_end_acceptance.py"
        ],
    }


def _observed(
    *,
    status: str,
    evidence_cid: str,
    terminal: str = "",
    nominated: bool | None = None,
) -> dict[str, Any]:
    if status not in DECLARED_STATUSES:
        raise EndToEndAcceptanceError(f"unsupported scenario status {status!r}")
    return {
        "status": status,
        "evidence_cid": evidence_cid,
        "terminal": terminal,
        "nominated": nominated if nominated is not None else status == "nominated",
        "accepted": False,
        "promotes": False,
        "writes_repository": False,
    }


def _run_inventory(_scenario: Mapping[str, Any]) -> dict[str, Any]:
    fixture = generate_controlled_monolith("small")
    detection = detect_monolith_opportunities(
        {
            "tree_id": TREE_ID,
            "modules": [
                _module(f"mod:{item.domain}", loc=item.loc, member_ids=(f"node:{item.domain}",))
                for item in fixture.modules
            ],
        }
    )
    diagnosed = SemanticRefactoringService().execute(
        ControlRequest(operation="spar.diagnose")
    )
    if diagnosed.status != ControlStatus.DIAGNOSED.value:
        raise EndToEndAcceptanceError("SPAR-044 diagnose must remain non-authoritative")
    if diagnosed.payload.get("authoritative") is not False:
        raise EndToEndAcceptanceError("diagnostics cannot be authoritative")
    if diagnosed.payload.get("can_authorize_completion") is not False:
        raise EndToEndAcceptanceError("control surface cannot complete SPAR-046")
    return _observed(
        status="nominated",
        evidence_cid=cid_for_dag_json(
            {
                "control_status": diagnosed.status,
                "detection_cid": detection.receipt_cid,
                "fixture_cid": fixture.fixture_cid,
                "loc": fixture.loc,
            }
        ),
    )


def _run_extraction(_scenario: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dry_run_extraction_wave(_packet(), **_wave_kwargs())
    if receipt.mutated is not False:
        raise EndToEndAcceptanceError("extraction dry-run cannot mutate")
    if receipt.writes_repository is not False:
        raise EndToEndAcceptanceError("extraction cannot write the repository")
    if receipt.advance_accepted_roots is not False:
        raise EndToEndAcceptanceError("extraction cannot advance accepted roots")
    return _observed(status="nominated", evidence_cid=receipt.receipt_cid)


def _run_facade(_scenario: Mapping[str, Any]) -> dict[str, Any]:
    plan = plan_compatibility_facades(
        {
            "tree_id": TREE_ID,
            "obligations": [
                {
                    "obligation_id": "obl:import_path:record",
                    "consumer_id": "pkg.cli",
                    "subject_id": "symbol:pkg.mod.Record",
                    "subject_module": "pkg.mod",
                    "kind": CompatibilityKind.IMPORT_PATH.value,
                    "disposition": CompatibilityDisposition.PRESERVE.value,
                    "required": True,
                }
            ],
        },
        candidates=(_candidate(),),
    )
    if plan.plan_is_nomination_only is not True:
        raise EndToEndAcceptanceError("façade plan must remain nomination_only")
    if plan.can_authorize_completion is not False:
        raise EndToEndAcceptanceError("façade plan cannot complete")
    if plan.can_retire_facade is not False:
        raise EndToEndAcceptanceError("façade plan cannot retire façades")
    return _observed(status="nominated", evidence_cid=plan.plan_cid)


def _run_validation(_scenario: Mapping[str, Any]) -> dict[str, Any]:
    request = compile_translation_validation_request(
        packet={
            "tree_id": TREE_ID,
            "packet_cid": _cid("packet"),
            "preimage": {
                "source_cids": [_cid("source")],
                "environment_cid": _cid("env"),
                "graph_cid": _cid("graph"),
            },
            "write_paths": list(WRITE_PATHS),
            "validation_commands": list(VALIDATION),
        },
        wave={
            "tree_id": TREE_ID,
            "receipt_cid": _cid("wave"),
            "packet_cids": [_cid("packet")],
            "write_paths": list(WRITE_PATHS),
            "after_source_cids": [_cid("candidate")],
            "status": "applied",
            "mutated": False,
            "writes_repository": False,
            "advance_accepted_roots": False,
            "executor_is_nomination_only": True,
        },
        selection={
            "tree_id": TREE_ID,
            "validation_selection_cid": _cid("selection"),
            "packet_cid": _cid("packet"),
            "effective_fallback": "none",
            "full_suite_required": False,
            "raw_source_cids": [_cid("source")],
            "write_paths": list(WRITE_PATHS),
            "validation_commands": list(VALIDATION),
            "selected_pytest_node_ids": ["tests/test_mod.py::test_a"],
            "selected_proof_ids": ["proof:mod"],
        },
        dimension_evidence=_passing_evidence(),
        candidate_source_cids=[_cid("candidate")],
    )
    result = dry_run_translation_validation(request)
    if result.status != ValidationStatus.VALIDATED.value:
        raise EndToEndAcceptanceError("translation validation did not pass")
    if result.mutated is not False:
        raise EndToEndAcceptanceError("validation dry-run cannot mutate")
    if result.can_authorize_completion is not False:
        raise EndToEndAcceptanceError("validation cannot complete")
    if result.claims_general_python_equivalence is not False:
        raise EndToEndAcceptanceError("validation cannot claim general Python equivalence")
    return _observed(status="nominated", evidence_cid=result.result_cid)


def _run_transition(_scenario: Mapping[str, Any]) -> dict[str, Any]:
    transition = _transition()
    if transition.can_authorize_completion is not False:
        raise EndToEndAcceptanceError("transition cannot complete")
    if transition.can_authorize_transition is not False:
        raise EndToEndAcceptanceError("SPAR-033 cannot authorize a transition")
    return _observed(status="nominated", evidence_cid=transition.transition_cid)


def _run_restart(_scenario: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dry_run_fixed_point(
        {
            "tree_id": TREE_ID,
            "current_roots": _rebuild_slices("prior"),
            "rebuild": _rebuild_slices("epoch-1"),
            "wave": {
                "wave_cid": _cid("wave"),
                "status": "applied",
                "packet_cids": [_cid("packet")],
            },
            "validation": {"validation_cid": _cid("validation"), "status": "validated"},
            "merge_receipts": [
                {
                    "merge_cid": _cid("merge"),
                    "wave_cid": _cid("wave"),
                    "tree_id": TREE_ID,
                    "disposition": "accepted",
                    "validation_cid": _cid("validation"),
                }
            ],
            "remaining_mandatory_task_cids": [_cid("task:next")],
            "pending_merge_cids": [],
            "iteration": 1,
        }
    )
    payload = receipt.identity_payload()
    if receipt.accepted is not False:
        raise EndToEndAcceptanceError("fixed-point controller cannot accept")
    if payload.get("writes_repository") is not False:
        raise EndToEndAcceptanceError("fixed-point controller cannot write")
    if payload.get("can_authorize_completion") is not False:
        raise EndToEndAcceptanceError("fixed-point controller cannot complete")
    if receipt.status not in {"continue", "nominated_fixed_point"}:
        raise EndToEndAcceptanceError("restart must continue or nominate a fixed point")
    return _observed(status="nominated", evidence_cid=receipt.receipt_cid)


def _run_reuse_exact(_scenario: Mapping[str, Any]) -> dict[str, Any]:
    episode = _transition()
    decision = decide_exact_reuse(_reuse_key(), (episode,))
    if decision.decision != ReuseDecisionKind.REUSE.value:
        raise EndToEndAcceptanceError("exact accepted episode must reuse")
    if decision.can_authorize_completion is not False:
        raise EndToEndAcceptanceError("reuse cannot complete")
    return _observed(status="nominated", evidence_cid=decision.decision_cid)


def _run_reuse_stale(_scenario: Mapping[str, Any]) -> dict[str, Any]:
    stored = _transition()
    decision = decide_exact_reuse(_reuse_key(tree_id=OTHER_TREE), (stored,))
    if decision.decision != ReuseDecisionKind.REVOKE.value:
        raise EndToEndAcceptanceError("stale tree must revoke reuse")
    if RevokeReason.STALE_TREE.value not in decision.reasons:
        raise EndToEndAcceptanceError("stale reuse must name stale_tree")
    return _observed(
        status="rejected",
        evidence_cid=decision.decision_cid,
        terminal="stale_tree",
        nominated=False,
    )


def _run_replan(_scenario: Mapping[str, Any]) -> dict[str, Any]:
    goals = compile_durable_goals(
        detect_monolith_opportunities(
            {
                "tree_id": TREE_ID,
                "modules": [_module("mod:core", loc=520, member_ids=("node:core",))],
            }
        )
    )
    first = compile_backlog(
        {
            "tree_id": TREE_ID,
            "goals": goals,
            "scopes": [_scope("mod:core", evidence_label="ev:old")],
        }
    )
    extraction = next(item for item in first.tasks if item.kind == TaskKind.EXTRACTION.value)
    repaired = compile_backlog(
        {
            "tree_id": TREE_ID,
            "goals": goals,
            "scopes": [_scope("mod:core", evidence_label="ev:new")],
            "prior_failures": [
                {
                    "module_id": "mod:core",
                    "kind": TaskKind.EXTRACTION.value,
                    "write_paths": list(extraction.write_paths),
                    "evidence_cids": list(extraction.evidence_cids),
                }
            ],
        }
    )
    kinds = {item.kind for item in repaired.tasks}
    if TaskKind.REPAIR.value not in kinds:
        raise EndToEndAcceptanceError("new evidence must admit a repair replan")
    if repaired.can_authorize_completion is not False:
        raise EndToEndAcceptanceError("replan cannot complete")
    return _observed(status="nominated", evidence_cid=repaired.receipt_cid)


def _run_rollback(_scenario: Mapping[str, Any]) -> dict[str, Any]:
    receipt = rollback_extraction_wave(_packet())
    if receipt.advance_accepted_roots is not False:
        raise EndToEndAcceptanceError("rollback cannot advance accepted roots")
    if receipt.writes_repository is not False:
        raise EndToEndAcceptanceError("rollback cannot write the repository")
    if receipt.mutated is not False:
        raise EndToEndAcceptanceError("rollback cannot mutate")
    return _observed(status="nominated", evidence_cid=receipt.receipt_cid)


def _run_corruption(_scenario: Mapping[str, Any]) -> dict[str, Any]:
    committed = _cid("world-root:committed")
    receipt = recover_world_root(
        _world_root_evidence(
            crash=True,
            current_world_root_cid=_cid("world-root:dirty"),
            last_committed_world_root_cid=committed,
            last_committed_root_generation=3,
            pending_outbox_cids=[_cid("pending-outbox")],
        )
    )
    if receipt.status != AdapterStatus.RECOVERED.value:
        raise EndToEndAcceptanceError("crash must recover the last committed root")
    if receipt.accepted is not False:
        raise EndToEndAcceptanceError("recovery cannot accept")
    if receipt.overwrite_prevented is not True:
        raise EndToEndAcceptanceError("recovery must prevent overwrite")
    if receipt.world_root.post_world_root_cid != committed:
        raise EndToEndAcceptanceError("recovery must restore the committed root")
    return _observed(status="nominated", evidence_cid=receipt.receipt_cid)


def _run_staleness(_scenario: Mapping[str, Any]) -> dict[str, Any]:
    packet = _packet()
    with pytest.raises(ExtractionWaveError, match="before hashes do not verify"):
        execute_extraction_wave(
            packet,
            claimed_before_hashes={packet.packet_cid: (_cid("stale"),)},
            **_wave_kwargs(),
        )
    return _observed(
        status="rejected",
        evidence_cid=_cid("stale_before_hashes"),
        terminal="stale_before_hashes",
        nominated=False,
    )


def _run_unsafe_splits(_scenario: Mapping[str, Any]) -> dict[str, Any]:
    fixture = generate_controlled_monolith("small")
    policy = dict(load_split_policy())
    policy["unsafe_intra_domain_split"] = True
    with pytest.raises(BenchmarkCorpusError, match="unsafe intra-domain split"):
        seal_fixture_splits(fixture, policy=policy)
    return _observed(
        status="rejected",
        evidence_cid=_cid("unsafe_intra_domain_split"),
        terminal="unsafe_intra_domain_split",
        nominated=False,
    )


def _run_failed_proofs(_scenario: Mapping[str, Any]) -> dict[str, Any]:
    with pytest.raises(ProofAdapterError, match="proof_candidate cannot admit proofs"):
        reconstruct_proof(
            {
                "obligation_id": "obl:bound",
                "kind": ArtifactKind.PROOF_CANDIDATE.value,
                "artifact_cid": _cid("candidate"),
                "reconstruction_cid": _cid("candidate"),
                "kernel_checked": False,
                "tree_id": TREE_ID,
                "corpus_revision": "corpus:v1",
                "environment_id": "env:test",
                "toolchain_id": TOOLCHAIN_ID,
            },
            tree_id=TREE_ID,
            corpus_revision="corpus:v1",
            environment_id="env:test",
            toolchain_id=TOOLCHAIN_ID,
        )
    return _observed(
        status="rejected",
        evidence_cid=_cid("proof_candidate_cannot_admit"),
        terminal="proof_candidate_cannot_admit",
        nominated=False,
    )


def _run_root_conflicts(_scenario: Mapping[str, Any]) -> dict[str, Any]:
    current = _cid("world-root:current")
    receipt = integrate_world_root(
        _world_root_evidence(
            pre_world_root_cid=_cid("world-root:other"),
            current_world_root_cid=current,
            expected_root_generation=3,
            current_root_generation=3,
        )
    )
    if receipt.status != AdapterStatus.REJECTED_ROOT_CONFLICT.value:
        raise EndToEndAcceptanceError("root conflict must be rejected")
    if receipt.accepted is not False:
        raise EndToEndAcceptanceError("root conflict cannot accept")
    if receipt.overwrite_prevented is not True:
        raise EndToEndAcceptanceError("root conflict must not overwrite")
    if receipt.world_root.post_world_root_cid != current:
        raise EndToEndAcceptanceError("conflicting root must remain unchanged")
    return _observed(
        status="rejected",
        evidence_cid=receipt.receipt_cid,
        terminal="rejected_root_conflict",
        nominated=False,
    )


_DISPATCH: dict[str, Any] = {
    "inventory.small_corpus": _run_inventory,
    "extraction.dry_run_wave": _run_extraction,
    "facade.preserve_import": _run_facade,
    "validation.translation_pass": _run_validation,
    "transition.nominate_memory": _run_transition,
    "restart.fixed_point_continue": _run_restart,
    "reuse.exact_accepted": _run_reuse_exact,
    "reuse.stale_tree": _run_reuse_stale,
    "replan.new_evidence_repair": _run_replan,
    "rollback.restore_preimages": _run_rollback,
    "corruption.crash_recovery": _run_corruption,
    "staleness.stale_before_hashes": _run_staleness,
    "unsafe_splits.intra_domain": _run_unsafe_splits,
    "failed_proofs.candidate_cannot_admit": _run_failed_proofs,
    "root_conflicts.no_overwrite": _run_root_conflicts,
}


def run_acceptance_scenario(scenario: Mapping[str, Any]) -> SparAcceptanceScenarioResult:
    """Exercise one sealed SPAR-046 scenario against current adapters."""

    scenario_id = str(scenario.get("scenario_id") or "")
    stage = str(scenario.get("stage") or "")
    expected_status = str(scenario.get("expected_status") or "")
    expected_terminal = str(scenario.get("expected_terminal") or "")
    if scenario_id not in _DISPATCH:
        raise EndToEndAcceptanceError(f"unknown acceptance scenario {scenario_id!r}")
    if stage not in REQUIRED_STAGES:
        raise EndToEndAcceptanceError(f"unknown acceptance stage {stage!r}")
    if expected_status not in DECLARED_STATUSES:
        raise EndToEndAcceptanceError(f"unknown expected status {expected_status!r}")
    if scenario.get("promotes") is True:
        raise EndToEndAcceptanceError("acceptance scenarios cannot promote")
    if scenario.get("writes_repository") is True:
        raise EndToEndAcceptanceError("acceptance scenarios cannot write the repository")
    observed = _DISPATCH[scenario_id](scenario)
    if observed["status"] != expected_status:
        raise EndToEndAcceptanceError(
            f"{scenario_id} status {observed['status']!r} != {expected_status!r}"
        )
    if expected_terminal and observed.get("terminal") != expected_terminal:
        raise EndToEndAcceptanceError(
            f"{scenario_id} terminal {observed.get('terminal')!r} != {expected_terminal!r}"
        )
    if observed["accepted"] or observed["promotes"] or observed["writes_repository"]:
        raise EndToEndAcceptanceError(f"{scenario_id} cannot promote or write")
    return SparAcceptanceScenarioResult(
        scenario_id=scenario_id,
        stage=stage,
        status=str(observed["status"]),
        evidence_cid=str(observed["evidence_cid"]),
        expected_status=expected_status,
        expected_terminal=expected_terminal,
        terminal=str(observed.get("terminal") or ""),
        safety_floor=str(scenario.get("safety_floor") or ""),
        nominated=True,
    )


def run_end_to_end_acceptance_matrix(
    matrix: Mapping[str, Any] | None = None,
) -> SparEndToEndAcceptanceReceipt:
    """Run the sealed SPAR-046 matrix. Current authority remains separate."""

    loaded = dict(matrix) if matrix is not None else load_acceptance_matrix()
    if tuple(loaded.get("required_stages") or ()) != REQUIRED_STAGES:
        raise EndToEndAcceptanceError("required stages must remain sealed")
    if tuple(loaded.get("predecessor_task_ids") or ()) != PREDECESSOR_TASK_IDS:
        raise EndToEndAcceptanceError("predecessors must remain SPAR-044 and SPAR-045")
    preregistration = load_preregistration()
    if dict(loaded.get("zero_safety_floors") or {}) != dict(
        preregistration["zero_safety_floors"]
    ):
        raise EndToEndAcceptanceError("safety floors must match preregistration")
    corpus = load_benchmark_corpus_manifest()
    if dict(corpus["zero_safety_floors"]) != dict(loaded["zero_safety_floors"]):
        raise EndToEndAcceptanceError("corpus floors must match the acceptance matrix")
    covered = {str(item["stage"]) for item in loaded["scenarios"]}
    missing = [stage for stage in REQUIRED_STAGES if stage not in covered]
    if missing:
        raise EndToEndAcceptanceError(f"acceptance matrix missing stages: {missing}")
    results = tuple(run_acceptance_scenario(item) for item in loaded["scenarios"])
    identity = {
        "payload": loaded,
        "task_id": TASK_ID,
        "tree_id": TREE_ID,
    }
    return SparEndToEndAcceptanceReceipt(
        tree_id=TREE_ID,
        matrix_cid=cid_for_dag_json(identity),
        scenario_results=results,
        stages=REQUIRED_STAGES,
    )


def nominate_end_to_end_acceptance() -> SparEndToEndAcceptanceReceipt:
    """Nominate the SPAR-046 matrix. Independent validation remains separate."""

    return run_end_to_end_acceptance_matrix()


@functools.lru_cache(maxsize=1)
def _cached_nomination() -> SparEndToEndAcceptanceReceipt:
    return nominate_end_to_end_acceptance()


def dry_run_end_to_end_acceptance() -> SparEndToEndAcceptanceReceipt:
    return _cached_nomination()


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-046"
    assert GOAL_ID == "SPAR-G081"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert PREDECESSOR_TASK_IDS == ("SPAR-044", "SPAR-045")
    assert SPAR_ACCEPTANCE_MATRIX_INTERFACE == "SparEndToEndAcceptanceMatrix@1"
    assert SPAR_ACCEPTANCE_RECEIPT_INTERFACE == "SparEndToEndAcceptanceReceipt@1"
    assert SPAR_ACCEPTANCE_SCENARIO_INTERFACE == "SparAcceptanceScenarioResult@1"
    assert ACCEPTANCE_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("end_to_end_acceptance@1")
    assert TEST_PATH.is_file()
    assert (ROOT / MATRIX_RELATIVE).is_file()
    assert WRITE_SCOPE == (
        "test/api/semantic_refactoring/test_end_to_end_acceptance.py",
        MATRIX_RELATIVE,
    )
    assert (ROOT / PREREGISTRATION_RELATIVE).is_file()
    assert (ROOT / CORPUS_MANIFEST_RELATIVE).is_file()
    assert (
        ROOT / "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/service.py"
    ).is_file()
    assert (
        ROOT / "test/api/semantic_refactoring/test_control_surface.py"
    ).is_file()
    assert (
        ROOT / "test/api/semantic_refactoring/test_benchmark_corpus.py"
    ).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "acceptance-matrix"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert CAN_AUTHORIZE_COMPLETION is False
    assert CAN_AUTHORIZE_TRANSITION is False
    assert CAN_CREATE_AUTHORITY is False
    assert GATE_WRITES_REPOSITORY is False
    assert GATE_IS_NOMINATION_ONLY is True
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert NETWORK_DENIED is True
    assert NETWORK_DENY == "deny"
    assert RAW_SOURCE_REQUIRED is True
    assert DRY_RUN_IS_DETERMINISTIC is True
    assert DRY_RUN_MUTATES is False
    profile = acceptance_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]
    matrix = load_acceptance_matrix()
    assert matrix["nomination_only"] is True
    assert matrix["can_authorize_completion"] is False
    assert matrix["can_authorize_transition"] is False
    assert matrix["can_create_authority"] is False
    assert matrix["worker_self_approval"] is False
    assert matrix["model_output_is_proposal_only"] is True
    assert matrix["test_pass_is_not_completion"] is True
    assert matrix["writes_repository"] is False
    assert matrix["projection_is_authority"] is False


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(TEST_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "SparAcceptanceMatrix" in names
    assert "SparAcceptanceScenarioResult" in names
    assert "SparEndToEndAcceptanceReceipt" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "SparEndToEndAcceptanceReceipt" in exports
    assert "nominate_end_to_end_acceptance" in exports
    functions = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    for forbidden in (
        "authorize_completion",
        "promote_root",
        "dump_repository",
        "admit_third_party_source",
        "overwrite_root_conflict",
    ):
        assert forbidden not in functions
        assert forbidden not in names


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_matrix_covers_required_stages_and_floors() -> None:
    matrix = load_acceptance_matrix()
    preregistration = load_preregistration()
    corpus = load_benchmark_corpus_manifest()
    assert tuple(matrix["required_stages"]) == REQUIRED_STAGES
    assert tuple(matrix["predecessor_task_ids"]) == PREDECESSOR_TASK_IDS
    assert [item["stage"] for item in matrix["scenarios"]]
    covered = {item["stage"] for item in matrix["scenarios"]}
    assert covered == set(REQUIRED_STAGES)
    assert matrix["zero_safety_floors"] == preregistration["zero_safety_floors"]
    assert corpus["zero_safety_floors"] == matrix["zero_safety_floors"]
    assert all(value == 0 for value in matrix["zero_safety_floors"].values())
    assert matrix["preregistration_path"] == PREREGISTRATION_RELATIVE
    assert matrix["corpus_manifest_path"] == CORPUS_MANIFEST_RELATIVE
    assert matrix["tree_id"] == TREE_ID
    for item in matrix["scenarios"]:
        assert item["promotes"] is False
        assert item["writes_repository"] is False
        assert item["expected_status"] in DECLARED_STATUSES
        assert item["stage"] in REQUIRED_STAGES


def test_predecessors_remain_readable_without_dump() -> None:
    matrix = load_acceptance_matrix()
    encoded = json.dumps(matrix)
    assert "def leaf" not in encoded
    assert "class ShadowPlanGate" not in encoded
    assert "repository_dump" not in encoded
    assert "source_body" not in encoded
    corpus = load_benchmark_corpus_manifest()
    assert corpus["task_id"] == "SPAR-045"
    assert corpus["whole_repository_dump"] is False if "whole_repository_dump" in corpus else True
    admitted = corpus["profiles"][-1]
    assert admitted["name"] == "real_current_tree"
    assert admitted.get("whole_repository_dump") is False


def test_each_required_stage_is_exercised() -> None:
    receipt = dry_run_end_to_end_acceptance()
    stages = tuple(item.stage for item in receipt.scenario_results)
    assert set(stages) == set(REQUIRED_STAGES)
    assert receipt.stages == REQUIRED_STAGES
    by_id = {item.scenario_id: item for item in receipt.scenario_results}
    assert by_id["inventory.small_corpus"].status == "nominated"
    assert by_id["extraction.dry_run_wave"].status == "nominated"
    assert by_id["facade.preserve_import"].status == "nominated"
    assert by_id["validation.translation_pass"].status == "nominated"
    assert by_id["transition.nominate_memory"].status == "nominated"
    assert by_id["restart.fixed_point_continue"].status == "nominated"
    assert by_id["reuse.exact_accepted"].status == "nominated"
    assert by_id["replan.new_evidence_repair"].status == "nominated"
    assert by_id["rollback.restore_preimages"].status == "nominated"
    assert by_id["corruption.crash_recovery"].status == "nominated"


def test_fail_closed_scenarios_do_not_promote() -> None:
    receipt = dry_run_end_to_end_acceptance()
    rejected = {
        item.scenario_id: item
        for item in receipt.scenario_results
        if item.status == "rejected"
    }
    assert rejected["reuse.stale_tree"].terminal == "stale_tree"
    assert rejected["staleness.stale_before_hashes"].terminal == "stale_before_hashes"
    assert rejected["unsafe_splits.intra_domain"].terminal == "unsafe_intra_domain_split"
    assert rejected["failed_proofs.candidate_cannot_admit"].terminal == (
        "proof_candidate_cannot_admit"
    )
    assert rejected["root_conflicts.no_overwrite"].terminal == "rejected_root_conflict"
    for item in receipt.scenario_results:
        assert item.accepted is False
        assert item.promotes is False
        assert item.writes_repository is False
        payload = item.identity_payload()
        assert payload["accepted"] is False
        assert payload["promotes"] is False
        assert payload["writes_repository"] is False


def test_unsafe_split_and_root_conflict_keep_zero_floors() -> None:
    matrix = load_acceptance_matrix()
    floors = dict(matrix["zero_safety_floors"])
    receipt = dry_run_end_to_end_acceptance()
    by_id = {item.scenario_id: item for item in receipt.scenario_results}
    assert by_id["unsafe_splits.intra_domain"].safety_floor == "unauthorized_mutations"
    assert by_id["root_conflicts.no_overwrite"].safety_floor == "root_conflict_overwrite"
    assert by_id["failed_proofs.candidate_cannot_admit"].safety_floor == (
        "test_or_proof_weakening"
    )
    assert by_id["reuse.stale_tree"].safety_floor == "stale_authoritative_reuse"
    assert by_id["corruption.crash_recovery"].safety_floor == "rollback_failure"
    assert floors["unauthorized_mutations"] == 0
    assert floors["root_conflict_overwrite"] == 0
    assert floors["test_or_proof_weakening"] == 0
    assert floors["stale_authoritative_reuse"] == 0
    assert floors["rollback_failure"] == 0
    assert floors["false_task_completions"] == 0


def test_nomination_receipt_cannot_complete_or_self_approve() -> None:
    receipt = dry_run_end_to_end_acceptance()
    assert receipt.nominated is True
    assert receipt.accepted is False
    assert receipt.nomination_only is True
    assert receipt.can_authorize_completion is False
    assert receipt.can_authorize_transition is False
    assert receipt.can_create_authority is False
    assert receipt.writes_repository is False
    assert receipt.worker_self_approval is False
    assert receipt.network == NETWORK_DENY
    assert receipt.task_id == TASK_ID
    assert receipt.goal_id == GOAL_ID
    assert receipt.tree_id == TREE_ID
    encoded = receipt.to_dict()
    restored = SparEndToEndAcceptanceReceipt.from_dict(encoded)
    assert restored.receipt_cid == receipt.receipt_cid
    assert restored == receipt
    dry = dry_run_end_to_end_acceptance()
    assert dry.receipt_cid == receipt.receipt_cid
    with pytest.raises(EndToEndAcceptanceError, match="self-approve"):
        SparEndToEndAcceptanceReceipt.from_dict({**encoded, "accepted": True})
    with pytest.raises(EndToEndAcceptanceError, match="can_authorize_completion"):
        SparEndToEndAcceptanceReceipt.from_dict(
            {**encoded, "can_authorize_completion": True}
        )


def test_dry_run_is_deterministic_and_does_not_mutate() -> None:
    first = run_end_to_end_acceptance_matrix()
    second = run_end_to_end_acceptance_matrix()
    assert first.receipt_cid == second.receipt_cid
    assert [item.result_cid for item in first.scenario_results] == [
        item.result_cid for item in second.scenario_results
    ]
    assert first.writes_repository is False
    assert DRY_RUN_MUTATES is False


def test_identity_excludes_observational_fields() -> None:
    receipt = dry_run_end_to_end_acceptance()
    encoded = receipt.to_dict()
    for field in IDENTITY_EXCLUDED_FIELDS:
        assert field not in encoded
        with pytest.raises(EndToEndAcceptanceError, match="observational"):
            SparEndToEndAcceptanceReceipt.from_dict({**encoded, field: "now"})


def test_vector_or_model_cannot_admit_matrix() -> None:
    matrix = dict(load_acceptance_matrix())
    matrix["vector_score"] = 0.99
    with pytest.raises(EndToEndAcceptanceError, match="unknown acceptance scenario|stages"):
        run_end_to_end_acceptance_matrix(
            {
                **matrix,
                "scenarios": [
                    {
                        "scenario_id": "vector.similarity_admit",
                        "stage": "reuse",
                        "expected_status": "nominated",
                        "expected_terminal": None,
                        "promotes": False,
                        "writes_repository": False,
                    }
                ],
            }
        )
    with pytest.raises(EndToEndAcceptanceError, match="cannot promote"):
        run_acceptance_scenario(
            {
                "scenario_id": "reuse.exact_accepted",
                "stage": "reuse",
                "expected_status": "nominated",
                "expected_terminal": None,
                "promotes": True,
                "writes_repository": False,
            }
        )


def test_missing_stage_and_floor_drift_fail_closed() -> None:
    matrix = dict(load_acceptance_matrix())
    truncated = [item for item in matrix["scenarios"] if item["stage"] != "rollback"]
    with pytest.raises(EndToEndAcceptanceError, match="missing stages"):
        run_end_to_end_acceptance_matrix({**matrix, "scenarios": truncated})
    drifted = dict(matrix["zero_safety_floors"])
    drifted["false_task_completions"] = 1
    with pytest.raises(EndToEndAcceptanceError, match="safety floors"):
        run_end_to_end_acceptance_matrix({**matrix, "zero_safety_floors": drifted})
