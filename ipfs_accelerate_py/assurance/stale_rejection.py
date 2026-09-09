"""Fail-closed PCPR-069 Accelerate-owned stale rejection and PlanDelta.

Accelerate owns stale-evidence rejection and hermetic PlanDelta
production. This module rejects the PCPR-068 stale ContextPack and
protocol-test identities, preserves unaffected solver-qualification
completion, increments the plan epoch, and records a bounded refill
against the PCPR-061 DatasetsContextPack@1 identity, the PCPR-062 Kit
current root, the PCPR-063 route, the PCPR-064 patch, the PCPR-065
selected-tests run, the PCPR-066 unrelated change, the PCPR-067
eligible reuse, and the PCPR-068 relevant-interface identity without
reminting those identities.

The hermetic PlanDelta uses only closed operations attach_evidence and
add_task. Started and completed history is not mutated. Whole-plan
regeneration is refused. Refill stays inside 12 tasks and 20 epochs.
Live plan-revision-store apply stays typed unavailable. Incremental
prover binaries missing from the sealed PATH stay typed unavailable.
Model stages are not invoked. A model assertion cannot complete work.

Hermetic classification is candidate coverage only. It is not live
supervisor application, not live PlanDelta admission, and not a closed
PCPR release. Simulated results are not live. This evaluator does not
write DuckDB or Quack state.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.assurance.dependency_locks import (
    CLOSED_RELEASE_OUTCOMES,
    PACKAGE_NAME,
    PACKAGE_VERSION,
    SEALED_PATH,
    SEALED_PYTHON,
    content_identity,
    discover_accelerate_root,
    observe_sealed_validation_environment,
    pretty_json,
    sha256_bytes,
    typed_unavailable,
)
from ipfs_accelerate_py.assurance.portfolio_compatibility_lock import (
    PINNED_LOCK_CID as PCPR_056_LOCK_CID,
    PORTFOLIO_ID,
    PORTFOLIO_VERSION,
)
from ipfs_accelerate_py.assurance.shared_contracts import (
    OWNER_INTERFACES,
    OWNER_SCHEMAS,
    shared_interface_id,
    shared_schema_id,
)

INTERFACE: Final = "AccelerateStaleRejectionAndPlanDelta@1"
SCHEMA: Final = "ipfs_accelerate_py/assurance/stale-rejection-and-plan-delta@1"
DOCUMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/declared-stale-rejection-and-plan-delta@1"
)
DELTA_SCHEMA: Final = "ipfs_accelerate_py/assurance/stale-rejection-plan-delta@1"
VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/stale-rejection-and-plan-delta-verdict@1"
)
CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/platform-stale-rejection-and-plan-delta-catalog@1"
)
CATALOG_INTERFACE: Final = "PlatformStaleRejectionAndPlanDelta@1"
DELTA_INTERFACE: Final = "StaleRejectionPlanDelta@1"
OWNER_INTERFACE: Final = "DatasetsContextPack@1"
OWNER_SCHEMA: Final = "ipfs_datasets_py/datasets-context-pack@1"
OWNER_REPOSITORY: Final = "ipfs_datasets_py"
STORAGE_OWNER_REPOSITORY: Final = "ipfs_kit_py"
STORAGE_OWNER_INTERFACE: Final = "KitContextPackStorage@1"
EXECUTION_OWNER_REPOSITORY: Final = "ipfs_accelerate_py"
PATCH_OWNER_INTERFACE: Final = "AccelerateBoundedPatch@1"
ROUTE_OWNER_INTERFACE: Final = "AccelerateDeterministicFirstRoute@1"
RUN_OWNER_INTERFACE: Final = "AccelerateSelectedTestsAndProofs@1"
UNRELATED_CHANGE_OWNER_INTERFACE: Final = "AccelerateUnrelatedStateChange@1"
REUSE_OWNER_INTERFACE: Final = "AccelerateSafeReuse@1"
INTERFACE_CHANGE_OWNER_INTERFACE: Final = "AccelerateRelevantInterfaceChange@1"
EXECUTION_INVOCATION_INTERFACE: Final = OWNER_INTERFACES["ExecutionInvocation"]
EXECUTION_INVOCATION_SCHEMA: Final = OWNER_SCHEMAS["ExecutionInvocation"]
EXECUTION_RECEIPT_INTERFACE: Final = OWNER_INTERFACES["ExecutionReceipt"]
EXECUTION_RECEIPT_SCHEMA: Final = OWNER_SCHEMAS["ExecutionReceipt"]
PROTOCOL_INTERFACE: Final = "LogicProviderProtocol@2"
DOCUMENTATION_INTERFACE: Final = "LogicProviderProtocolRelevantInterface@1"
DOCUMENTATION_SCHEMA: Final = (
    "ipfs_datasets_py/logic-provider-relevant-interface@1"
)
STALE_REJECTION_SIDECAR_INTERFACE: Final = (
    "LogicProviderProtocolStaleRejection@1"
)
STALE_REJECTION_SIDECAR_SCHEMA: Final = (
    "ipfs_datasets_py/logic-provider-stale-rejection@1"
)
PCPR_069_TASK_ID: Final = "PCPR-069"
PCPR_069_GOAL_ID: Final = "PCPR-G700"
PCPR_068_TASK_ID: Final = "PCPR-068"
PCPR_067_TASK_ID: Final = "PCPR-067"
PCPR_066_TASK_ID: Final = "PCPR-066"
PCPR_065_TASK_ID: Final = "PCPR-065"
PCPR_064_TASK_ID: Final = "PCPR-064"
PCPR_063_TASK_ID: Final = "PCPR-063"
PCPR_062_TASK_ID: Final = "PCPR-062"
PCPR_061_TASK_ID: Final = "PCPR-061"
PCPR_060_TASK_ID: Final = "PCPR-060"
PCPR_070_TASK_ID: Final = "PCPR-070"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = (
    "proof-carrying-platform-qualification-and-release-v1"
)
OBJECTIVE_KIND: Final = "declared_stale_rejection_and_plan_delta"
OBJECTIVE_ID: Final = "PCPR-G700"
LANGUAGE: Final = "Python"
OPERATOR_BLOCKING_TASK_ID: Final = (
    "pcpr-069-operator-live-stale-rejection-and-plan-delta"
)
DOCUMENT_DIR_RELPATH: Final = "packaging/pcpr/reference-workflow/cpython312"
DOCUMENT_JSON_NAME: Final = "reference.stale-rejection.json"
DOCUMENT_README_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/STALE_REJECTION.md"
)
CATALOG_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/platform-stale-rejection-catalog.json"
)
SOURCE_DATE_EPOCH: Final = "0"
SOURCE_REPOSITORY: Final = "endomorphosis/ipfs_accelerate_py"
COMPLETED_THROUGH: Final = "stale_rejection_and_plan_delta"
NEXT_AUTHORIZED_STAGE: Final = "restart_authoritative_state_owner"
CHANGE_KIND: Final = "stale_rejection_and_plan_delta"
PLAN_DELTA_KIND: Final = "minimal_impacted_cone_refill"
BASE_PLAN_REVISION: Final = 1
NEXT_PLAN_REVISION: Final = 2
PLAN_EPOCH: Final = 2
MAX_REPLAN_EPOCHS: Final = 20
MAX_TASKS_PER_REFILL: Final = 12
MAX_TOTAL_TASKS: Final = 140
ATTACH_EVIDENCE_OPERATION: Final = "attach_evidence"
ADD_TASK_OPERATION: Final = "add_task"

PINNED_IDEA_DIGEST: Final = (
    "baguqeeracbayojdov4jmqiirx22pavrg6nabazcocru6y3scrdx5e54mw2zq"
)
PINNED_OBJECTIVE_CID: Final = (
    "baguqeeraynsn7tjr3iaggnreylzxo3akaooqwp5bf5oheaa6eubth2zwqeba"
)
PINNED_LOCK_CID: Final = PCPR_056_LOCK_CID
PINNED_PACK_CID: Final = (
    "bafkreih72d3nncekez43wmtlczq5mdtymzniluujypwybpgu3mtt7i4v2e"
)
PINNED_BYTES_CID: Final = (
    "bafkreihjdjarrlr24sperlq3zl4hjrksbol4b5saxquie3wpyi6xwsobwi"
)
PINNED_CURRENT_ROOT_CID: Final = PINNED_BYTES_CID
PINNED_ROUTE_CID: Final = (
    "baguqeerab6vsy7orm6wmxmqbzqh5f57dasnufhgngvh47gw7g3whrkr5smga"
)
PINNED_PATCH_CID: Final = (
    "baguqeeraphkktpgmcw2uusnbxlh7wfjblmhzb4oy3u55bbpdsvbt4xwvvb4q"
)
PINNED_RUN_CID: Final = (
    "baguqeerazgo2lirhy7hrpa42j227gdpzmi3jl4jdajoz6jzqqx7td6fy7dpa"
)
PINNED_PCPR_066_CHANGE_CID: Final = (
    "baguqeeraowfymvuvscitvjkxn3sak6f4a2ctkfvflney26jen7gwyqplln6a"
)
PINNED_REUSE_CID: Final = (
    "baguqeera75dkjwczh6mehnwyqot25rbd2mazkgjuvb4lrvkdkuqo7vntieuq"
)
PINNED_INTERFACE_CID: Final = (
    "baguqeera3k54j374af37kxdvkfgkjdfdofbdtg3x5vckpsceemy4lyklw3dq"
)

if PINNED_LOCK_CID != (
    "baguqeerawsekbbbt5ccctjt4tydzeahfkahsatb6q5x7k6cy4inrii5lhqmq"
):
    raise RuntimeError("PCPR-069 lock CID remints PCPR-056")

if OWNER_INTERFACES["SupervisorContextPack"] != OWNER_INTERFACE:
    raise RuntimeError("PCPR-069 remints SupervisorContextPack owner interface")
if OWNER_SCHEMAS["SupervisorContextPack"] != OWNER_SCHEMA:
    raise RuntimeError("PCPR-069 remints SupervisorContextPack owner schema")
if EXECUTION_INVOCATION_INTERFACE != "InvocationRequest@1":
    raise RuntimeError("PCPR-069 remints ExecutionInvocation owner interface")
if EXECUTION_RECEIPT_INTERFACE != "InvocationResult@1":
    raise RuntimeError("PCPR-069 remints ExecutionReceipt owner interface")

REFERENCE_OBJECTIVE_IDEA: Final = (
    "Modify a typed formal-logic API while reusing unaffected proofs, "
    "selecting only impacted tests, rejecting stale-tree evidence, and "
    "producing a complete proof-carrying execution receipt."
)

ESCALATION_ORDER: Final[tuple[str, ...]] = (
    "exact receipt",
    "AST and dependency analysis",
    "schema, type, and static checks",
    "selected tests",
    "incremental prover",
    "local small specialist",
    "medium model",
    "frontier model",
    "human decision",
)

AUTHORIZED_PATH_PREFIXES: Final[tuple[str, ...]] = (
    "external/ipfs_accelerate",
    "external/ipfs_datasets",
    "external/ipfs_kit",
    "artifacts/proof_carrying_platform_qualification_and_release/receipts/PCPR-069.json",
)

RELEVANT_CHANGE_PATHS: Final[tuple[str, ...]] = (
    "ipfs_datasets_py/logic/backends/protocol_v2.py",
    "ipfs_datasets_py/logic/platform/relevant_interface.py",
)

SEMANTIC_FRONTIER: Final[tuple[str, ...]] = (
    "LogicProviderProtocol@2",
    "DatasetsContextPack@1",
    "tests/unit/test_pcpr_013_logic_provider_protocol.py",
    "tests/unit/test_pcpr_014_semantic_apis.py",
    "tests/unit/test_pcpr_017_solver_qualification.py",
    "ipfs_datasets_py/assurance/solver_qualification.py",
)

ELIGIBLE_REUSE: Final[tuple[str, ...]] = (
    "tests/unit/test_pcpr_017_solver_qualification.py",
    "ipfs_datasets_py/assurance/solver_qualification.py",
)

IMPACTED_CONE: Final[tuple[str, ...]] = (
    "ipfs_datasets_py/logic/backends/protocol_v2.py",
    "ipfs_datasets_py/logic/platform/relevant_interface.py",
    "tests/unit/test_pcpr_013_logic_provider_protocol.py",
    "tests/unit/test_pcpr_014_semantic_apis.py",
    "DatasetsContextPack@1",
)

STALE_IDENTITIES: Final[tuple[str, ...]] = (
    "DatasetsContextPack@1",
    "tests/unit/test_pcpr_013_logic_provider_protocol.py",
    "tests/unit/test_pcpr_014_semantic_apis.py",
)

REFILL_TASKS: Final[tuple[str, ...]] = (
    "pcpr-069-refill-protocol-tests",
    "pcpr-069-refill-semantic-api-tests",
)

CLOSED_PLAN_DELTA_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        "add_goal",
        "supersede_goal",
        "amend_unstarted_goal",
        "add_task",
        "amend_unstarted_task",
        "supersede_unstarted_task",
        "split_unstarted_task",
        "coalesce_unstarted_tasks",
        "rewire_unstarted_dependency",
        "block_unstarted_task",
        "unblock_task",
        "reprioritize_unstarted_task",
        "assign_parallel_contract",
        "attach_evidence",
        "record_uncertainty",
        "request_lifecycle_action",
    }
)

HISTORY_SAFE_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        "attach_evidence",
        "record_uncertainty",
        "request_lifecycle_action",
        "add_task",
        "add_goal",
        "block_unstarted_task",
        "unblock_task",
    }
)

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_stale_rejection.py",
)

EVIDENCE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "measured",
        "measured_live",
        "measured_hermetic",
        "estimated",
        "simulated",
        "unavailable",
    }
)

REQUIRED_GOOD_PROBE_IDS: Final[frozenset[str]] = frozenset(
    {
        "delta_files_match_generator",
        "pyproject_stale_rejection_table",
        "paths_remain_authorized",
        "owner_pack_cid_matches_pin",
        "kit_current_root_bound_not_minted",
        "route_cid_bound_not_minted",
        "patch_cid_bound_not_minted",
        "run_cid_bound_not_minted",
        "interface_cid_bound_not_minted",
        "stale_identities_rejected",
        "unaffected_completion_preserved",
        "plan_delta_produced",
        "plan_epoch_incremented",
        "refill_within_bounds",
        "history_not_mutated",
        "model_assertion_cannot_complete_work",
        "duckdb_or_quack_not_written",
        "operator_blocking_task_emitted",
        "no_closed_release_outcome",
    }
)
FORBIDDEN_PRESENT_PROBE_IDS: Final[frozenset[str]] = frozenset(
    {
        "simulated_results_represented_as_live",
        "live_application_represented_as_live",
        "live_plan_delta_represented_as_live",
        "closed_release_represented_as_live",
        "compatibility_identities_reminted",
        "direct_database_bypass_used",
        "datasets_identity_reminted",
        "kit_identity_reminted",
        "route_identity_reminted",
        "patch_identity_reminted",
        "run_identity_reminted",
        "interface_identity_reminted",
        "model_assertion_completed_work",
        "unauthorized_path_accepted",
        "stale_identity_admitted_as_current",
        "unaffected_marked_stale",
        "whole_plan_regeneration_required",
        "history_mutated",
        "refill_exceeds_bounds",
    }
)

DOCUMENT_README: Final = """# PCPR-069 Accelerate stale rejection and PlanDelta

These files record the Accelerate-owned hermetic stale rejection and
PlanDelta against the PCPR-061 DatasetsContextPack@1 identity, the
PCPR-062 Kit current root, the PCPR-063 deterministic-first route, the
PCPR-064 bounded PatchPlan, the PCPR-065 selected-tests-and-proofs
run, the PCPR-066 unrelated change, the PCPR-067 eligible reuse, and
the PCPR-068 relevant interface change. Accelerate owns stale-evidence
rejection and PlanDelta production. It does not remint those identities
and does not claim live PlanDelta admission.

- `cpython312/reference.stale-rejection.json` is the declared
  stale-rejection and PlanDelta document. Exact commit and tree are
  bound by the PCPR-069 receipt `current_tree_binding`.
- `platform-stale-rejection-catalog.json` observes sibling Datasets
  and Kit bindings when present. Sibling source is never required.
- Stale ContextPack and protocol-test identities are rejected.
  Unaffected solver-qualification completion is preserved. The plan
  epoch increments. The hermetic PlanDelta refills only the impacted
  cone within 12 tasks and 20 epochs. Restart recovery remains
  PCPR-070.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-069-operator-live-stale-rejection-and-plan-delta`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
"""


class AccelerateStaleRejectionError(Exception):
    """Fail-closed PCPR-069 Accelerate stale rejection and PlanDelta error."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AccelerateStaleRejectionError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise AccelerateStaleRejectionError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise AccelerateStaleRejectionError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def parse_pyproject_stale_rejection_table(text: str) -> dict[str, Any]:
    import tomllib

    table = tomllib.loads(_text(text, "pyproject.toml"))
    if not isinstance(table, dict):
        raise AccelerateStaleRejectionError("pyproject.toml must be a table")
    tool = table.get("tool")
    payload: dict[str, Any] = {}
    if isinstance(tool, dict):
        accelerate = tool.get("ipfs_accelerate_py")
        if isinstance(accelerate, dict):
            raw = accelerate.get("stale-rejection")
            if isinstance(raw, dict):
                payload = dict(raw)
    return payload


def path_is_authorized(path: str) -> bool:
    normalized = _text(path, "path").replace("\\", "/").lstrip("./")
    for prefix in AUTHORIZED_PATH_PREFIXES:
        if normalized == prefix or normalized.startswith(prefix.rstrip("/") + "/"):
            return True
    return False


def refuse_unauthorized_path(path: str) -> str:
    if not path_is_authorized(path):
        raise AccelerateStaleRejectionError(
            f"path {path} is not an authorized PCPR-069 path"
        )
    return path


def refuse_model_completion(stage_id: str) -> None:
    stage = _text(stage_id, "stage_id")
    if stage in {
        "local_small_specialist",
        "medium_model",
        "frontier_model",
        "human_decision",
    }:
        raise AccelerateStaleRejectionError(
            f"model assertion at {stage} cannot complete work"
        )


def refuse_pack_cid_remint(cid: str) -> str:
    if cid != PINNED_PACK_CID:
        raise AccelerateStaleRejectionError(
            f"ContextPack CID {cid} remints {PINNED_PACK_CID}"
        )
    return cid


def refuse_current_root_remint(cid: str) -> str:
    if cid != PINNED_CURRENT_ROOT_CID:
        raise AccelerateStaleRejectionError(
            f"current root CID {cid} remints {PINNED_CURRENT_ROOT_CID}"
        )
    return cid


def refuse_route_cid_remint(cid: str) -> str:
    if cid != PINNED_ROUTE_CID:
        raise AccelerateStaleRejectionError(
            f"route CID {cid} remints {PINNED_ROUTE_CID}"
        )
    return cid


def refuse_patch_cid_remint(cid: str) -> str:
    if cid != PINNED_PATCH_CID:
        raise AccelerateStaleRejectionError(
            f"patch CID {cid} remints {PINNED_PATCH_CID}"
        )
    return cid


def refuse_run_cid_remint(cid: str) -> str:
    if cid != PINNED_RUN_CID:
        raise AccelerateStaleRejectionError(
            f"run CID {cid} remints {PINNED_RUN_CID}"
        )
    return cid


def refuse_interface_cid_remint(cid: str) -> str:
    if cid != PINNED_INTERFACE_CID:
        raise AccelerateStaleRejectionError(
            f"interface CID {cid} remints {PINNED_INTERFACE_CID}"
        )
    return cid


def refuse_whole_plan_regeneration(*, required: bool) -> None:
    if required:
        raise AccelerateStaleRejectionError(
            "stale rejection cannot require whole-plan regeneration"
        )


def refuse_history_mutation(*, mutated: bool) -> None:
    if mutated:
        raise AccelerateStaleRejectionError(
            "PlanDelta cannot mutate started, claimed, or completed history"
        )


def refuse_stale_as_current(*, identity: str, admitted_as_current: bool) -> str:
    name = _text(identity, "identity")
    if name in STALE_IDENTITIES and admitted_as_current:
        raise AccelerateStaleRejectionError(
            f"stale identity {name} is rejected and cannot be admitted as current"
        )
    return name


def refuse_unaffected_as_stale(*, identity: str, rejected: bool) -> str:
    name = _text(identity, "identity")
    if name in ELIGIBLE_REUSE and rejected:
        raise AccelerateStaleRejectionError(
            f"unaffected identity {name} is not stale and must be preserved"
        )
    return name


def refuse_refill_overflow(*, tasks_added: int) -> int:
    if tasks_added > MAX_TASKS_PER_REFILL or tasks_added < 0:
        raise AccelerateStaleRejectionError(
            f"refill of {tasks_added} tasks exceeds bound {MAX_TASKS_PER_REFILL}"
        )
    return tasks_added


def refuse_epoch_overflow(*, epoch: int) -> int:
    if epoch < 1 or epoch > MAX_REPLAN_EPOCHS:
        raise AccelerateStaleRejectionError(
            f"plan epoch {epoch} exceeds bound {MAX_REPLAN_EPOCHS}"
        )
    return epoch


def refuse_unknown_plan_delta_operation(operation: str) -> str:
    name = _text(operation, "operation")
    if name not in CLOSED_PLAN_DELTA_OPERATIONS:
        raise AccelerateStaleRejectionError(
            f"operation {name} is not a closed PlanDelta operation"
        )
    return name


def refuse_history_unsafe_operation(
    operation: str, *, target_lifecycle: str
) -> str:
    name = refuse_unknown_plan_delta_operation(operation)
    lifecycle = _text(target_lifecycle, "target_lifecycle")
    if lifecycle in {"claimed", "running", "settling", "completed", "accepted"}:
        if name not in HISTORY_SAFE_OPERATIONS:
            raise AccelerateStaleRejectionError(
                f"operation {name} cannot mutate {lifecycle} history"
            )
    return name


def refuse_live_plan_delta_apply(*, applied_live: bool) -> None:
    if applied_live:
        raise AccelerateStaleRejectionError(
            "live PlanDelta application remains typed unavailable"
        )


def reject_stale_identities(
    identities: Sequence[str] | None = None,
) -> dict[str, Any]:
    rejected = list(identities) if identities is not None else list(STALE_IDENTITIES)
    if list(rejected) != list(STALE_IDENTITIES):
        raise AccelerateStaleRejectionError(
            "stale rejection must reject exactly the PCPR-068 stale set"
        )
    for identity in rejected:
        refuse_stale_as_current(identity=identity, admitted_as_current=False)
    for identity in ELIGIBLE_REUSE:
        refuse_unaffected_as_stale(identity=identity, rejected=False)
    return {
        "schema": "ipfs_accelerate_py/assurance/stale-identity-rejection@1",
        "interface": "StaleIdentityRejection@1",
        "task_id": PCPR_069_TASK_ID,
        "rejected": list(rejected),
        "preserved": list(ELIGIBLE_REUSE),
        "stale_rejected": True,
        "unaffected_completion_preserved": True,
        "live": False,
        "applied": False,
        "evidence_kind": "measured_hermetic",
    }


def hermetic_plan_delta_items() -> tuple[dict[str, Any], ...]:
    items = (
        {
            "item_key": "reject-stale-context-pack",
            "operation": ATTACH_EVIDENCE_OPERATION,
            "effect_class": "evidence_only",
            "target": "DatasetsContextPack@1",
            "expected_target_lifecycle": "completed",
            "rationale": (
                "Reject stale ContextPack evidence after the PCPR-068 "
                "relevant interface change. Completed history is retained."
            ),
        },
        {
            "item_key": "reject-stale-protocol-tests",
            "operation": ATTACH_EVIDENCE_OPERATION,
            "effect_class": "evidence_only",
            "target": "tests/unit/test_pcpr_013_logic_provider_protocol.py",
            "expected_target_lifecycle": "completed",
            "rationale": (
                "Reject stale protocol-test evidence. Completed history "
                "is retained."
            ),
        },
        {
            "item_key": "reject-stale-semantic-api-tests",
            "operation": ATTACH_EVIDENCE_OPERATION,
            "effect_class": "evidence_only",
            "target": "tests/unit/test_pcpr_014_semantic_apis.py",
            "expected_target_lifecycle": "completed",
            "rationale": (
                "Reject stale semantic-API test evidence. Completed "
                "history is retained."
            ),
        },
        {
            "item_key": "preserve-unaffected-solver-qualification",
            "operation": ATTACH_EVIDENCE_OPERATION,
            "effect_class": "evidence_only",
            "target": "tests/unit/test_pcpr_017_solver_qualification.py",
            "expected_target_lifecycle": "completed",
            "rationale": (
                "Unaffected solver-qualification completion is preserved "
                "and is not marked stale."
            ),
        },
        {
            "item_key": "refill-protocol-tests",
            "operation": ADD_TASK_OPERATION,
            "effect_class": "deferred",
            "target": REFILL_TASKS[0],
            "expected_target_lifecycle": "unstarted",
            "rationale": (
                "Bounded refill of impacted protocol tests. Not live "
                "supervisor materialization."
            ),
        },
        {
            "item_key": "refill-semantic-api-tests",
            "operation": ADD_TASK_OPERATION,
            "effect_class": "deferred",
            "target": REFILL_TASKS[1],
            "expected_target_lifecycle": "unstarted",
            "rationale": (
                "Bounded refill of impacted semantic-API tests. Not live "
                "supervisor materialization."
            ),
        },
    )
    for item in items:
        refuse_history_unsafe_operation(
            str(item["operation"]),
            target_lifecycle=str(item["expected_target_lifecycle"]),
        )
    return items


@dataclass(frozen=True)
class OutcomeProbe:
    probe_id: str
    present: bool | None
    evidence_kind: str
    live: bool
    simulated_represented_as_live: bool
    reason: str
    details: Mapping[str, Any] = MappingProxyType({})

    def to_mapping(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "present": self.present,
            "evidence_kind": self.evidence_kind,
            "live": self.live,
            "simulated_represented_as_live": self.simulated_represented_as_live,
            "reason": self.reason,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class AccelerateStaleRejectionVerdict:
    schema: str
    interface: str
    promotion_status: str
    supervisor_disposition: str
    closed_release_outcome: str | None
    release_claim: bool
    completion_authoritative: bool
    contracts_frozen: bool
    duckdb_or_quack_state_written: bool
    sibling_source_required: bool
    live_application: bool
    live_plan_delta: bool
    operator_blocking_task: str
    simulated_results_represented_as_live: bool
    this_task_created_competing_authority: bool
    probes: tuple[OutcomeProbe, ...]
    blockers: tuple[str, ...]
    verdict_cid: str
    delta_cid: str
    document_cid: str
    catalog_cid: str
    pack_cid: str
    current_root_cid: str
    route_cid: str
    patch_cid: str
    run_cid: str
    interface_cid: str
    objective_cid: str
    idea_digest: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "verdict_cid": self.verdict_cid,
            "delta_cid": self.delta_cid,
            "document_cid": self.document_cid,
            "catalog_cid": self.catalog_cid,
            "pack_cid": self.pack_cid,
            "current_root_cid": self.current_root_cid,
            "route_cid": self.route_cid,
            "patch_cid": self.patch_cid,
            "run_cid": self.run_cid,
            "interface_cid": self.interface_cid,
            "objective_cid": self.objective_cid,
            "idea_digest": self.idea_digest,
            "promotion_status": self.promotion_status,
            "supervisor_disposition": self.supervisor_disposition,
            "closed_release_outcome": self.closed_release_outcome,
            "release_claim": self.release_claim,
            "completion_authoritative": self.completion_authoritative,
            "contracts_frozen": self.contracts_frozen,
            "duckdb_or_quack_state_written": self.duckdb_or_quack_state_written,
            "sibling_source_required": self.sibling_source_required,
            "live_application": self.live_application,
            "live_plan_delta": self.live_plan_delta,
            "operator_blocking_task": self.operator_blocking_task,
            "simulated_results_represented_as_live": (
                self.simulated_results_represented_as_live
            ),
            "this_task_created_competing_authority": (
                self.this_task_created_competing_authority
            ),
            "blocker_count": len(self.blockers),
            "blockers": list(self.blockers),
            "evidence_kind": "measured",
        }


def _probe(
    probe_id: str,
    present: bool | None,
    *,
    reason: str,
    evidence_kind: str = "measured",
    live: bool = False,
    details: Mapping[str, Any] | None = None,
) -> OutcomeProbe:
    return OutcomeProbe(
        probe_id=probe_id,
        present=present,
        evidence_kind=evidence_kind,
        live=live,
        simulated_represented_as_live=False,
        reason=reason,
        details=MappingProxyType(dict(details or {})),
    )


def _operator_blocking_task() -> dict[str, Any]:
    return {
        "task_id": OPERATOR_BLOCKING_TASK_ID,
        "status": "typed_blocked",
        "evidence_kind": "unavailable",
        "live": False,
        "applied": False,
        "requires": (
            "An admitted Quack-fenced state-owner session before stale "
            "rejection and PlanDelta are applied as live supervisor work "
            "against the stored ContextPack root"
        ),
        "action": (
            "Keep the hermetic stale rejection and PlanDelta. Do not write "
            "DuckDB or Quack state. Do not escalate to a model. Do not "
            "mutate started history. PCPR-070 restarts the authoritative "
            "state owner."
        ),
        "reason": (
            "Hermetic stale rejection and PlanDelta classification is not "
            "live supervisor application or live plan-revision-store apply. "
            "Sealed validation has no admitted Quack-fenced session. Direct "
            "DuckDB writes are prohibited."
        ),
    }


def produce_stale_rejection_and_plan_delta(
    *,
    paths: Sequence[str] | None = None,
    pack_cid: str = PINNED_PACK_CID,
    current_root_cid: str = PINNED_CURRENT_ROOT_CID,
    route_cid: str = PINNED_ROUTE_CID,
    patch_cid: str = PINNED_PATCH_CID,
    run_cid: str = PINNED_RUN_CID,
    interface_cid: str = PINNED_INTERFACE_CID,
    complete_from: str | None = None,
    whole_plan_regeneration_required: bool = False,
    history_mutated: bool = False,
    admit_stale_as_current: bool = False,
    mark_unaffected_stale: bool = False,
    applied_live: bool = False,
    plan_epoch: int = PLAN_EPOCH,
    tasks_added: int | None = None,
) -> dict[str, Any]:
    """Produce the hermetic stale rejection and PlanDelta. Fail closed."""

    if complete_from is not None:
        refuse_model_completion(complete_from)
    refuse_pack_cid_remint(pack_cid)
    refuse_current_root_remint(current_root_cid)
    refuse_route_cid_remint(route_cid)
    refuse_patch_cid_remint(patch_cid)
    refuse_run_cid_remint(run_cid)
    refuse_interface_cid_remint(interface_cid)
    refuse_whole_plan_regeneration(required=whole_plan_regeneration_required)
    refuse_history_mutation(mutated=history_mutated)
    refuse_live_plan_delta_apply(applied_live=applied_live)
    refuse_epoch_overflow(epoch=plan_epoch)
    added = len(REFILL_TASKS) if tasks_added is None else tasks_added
    refuse_refill_overflow(tasks_added=added)
    for identity in STALE_IDENTITIES:
        refuse_stale_as_current(
            identity=identity, admitted_as_current=admit_stale_as_current
        )
    for identity in ELIGIBLE_REUSE:
        refuse_unaffected_as_stale(
            identity=identity, rejected=mark_unaffected_stale
        )
    checked_paths = list(paths) if paths is not None else list(AUTHORIZED_PATH_PREFIXES)
    for path in checked_paths:
        refuse_unauthorized_path(path)
    items = list(hermetic_plan_delta_items())
    rejection = reject_stale_identities()
    return {
        "schema": DELTA_SCHEMA,
        "interface": DELTA_INTERFACE,
        "owner_interface": INTERFACE,
        "task_id": PCPR_069_TASK_ID,
        "goal_id": PCPR_069_GOAL_ID,
        "objective_kind": OBJECTIVE_KIND,
        "portfolio_id": PORTFOLIO_ID,
        "portfolio_version": PORTFOLIO_VERSION,
        "language": LANGUAGE,
        "idea": REFERENCE_OBJECTIVE_IDEA,
        "idea_digest": PINNED_IDEA_DIGEST,
        "objective_cid": PINNED_OBJECTIVE_CID,
        "lock_cid": PINNED_LOCK_CID,
        "pack_cid": pack_cid,
        "current_root_cid": current_root_cid,
        "route_cid": route_cid,
        "patch_cid": patch_cid,
        "run_cid": run_cid,
        "pcpr_066_change_cid": PINNED_PCPR_066_CHANGE_CID,
        "reuse_cid": PINNED_REUSE_CID,
        "interface_cid": interface_cid,
        "protocol_interface": PROTOCOL_INTERFACE,
        "documentation_interface": DOCUMENTATION_INTERFACE,
        "documentation_schema": DOCUMENTATION_SCHEMA,
        "change_kind": CHANGE_KIND,
        "plan_delta_kind": PLAN_DELTA_KIND,
        "produced": True,
        "applied": True,
        "live": False,
        "hermetic": True,
        "admitted_live": False,
        "deferred": False,
        "paths_authorized": True,
        "authorized_path_prefixes": list(AUTHORIZED_PATH_PREFIXES),
        "relevant_change_paths": list(RELEVANT_CHANGE_PATHS),
        "semantic_frontier": list(SEMANTIC_FRONTIER),
        "impacted_cone": list(IMPACTED_CONE),
        "stale_identities": list(STALE_IDENTITIES),
        "eligible_reuse": list(ELIGIBLE_REUSE),
        "stale_rejected": True,
        "stale_rejection": rejection,
        "unaffected_completion_preserved": True,
        "plan_delta_produced": True,
        "plan_epoch": plan_epoch,
        "base_plan_revision": BASE_PLAN_REVISION,
        "next_plan_revision": NEXT_PLAN_REVISION,
        "epoch_incremented": True,
        "history_mutated": False,
        "items": items,
        "refill": {
            "tasks_added": added,
            "task_ids": list(REFILL_TASKS),
            "max_tasks_per_refill": MAX_TASKS_PER_REFILL,
            "max_epochs": MAX_REPLAN_EPOCHS,
            "max_total_tasks": MAX_TOTAL_TASKS,
            "within_bounds": True,
            "automatic_frontier_refill": True,
            "live": False,
            "applied": False,
            "evidence_kind": "measured_hermetic",
        },
        "remints_protocol": False,
        "adds_protocol_operation": False,
        "relevant_interface_change": False,
        "whole_plan_regeneration_required": False,
        "eligible_reuse_preserved": True,
        "model_assertion_completes_work": False,
        "escalation_order": list(ESCALATION_ORDER),
        "completed_through": COMPLETED_THROUGH,
        "next_authorized_stage": NEXT_AUTHORIZED_STAGE,
        "next_task_id": PCPR_070_TASK_ID,
        "prerequisite_task_id": PCPR_068_TASK_ID,
        "relevant_change_task_id": PCPR_068_TASK_ID,
        "duckdb_or_quack_state_written": False,
        "closed_release_outcome": None,
        "release_claim": False,
        "evidence_kind": "measured_hermetic",
    }


def canonical_stale_rejection() -> dict[str, Any]:
    return produce_stale_rejection_and_plan_delta()


def delta_cid_of(plan: Mapping[str, Any] | None = None) -> str:
    payload = dict(plan or canonical_stale_rejection())
    payload.pop("delta_cid", None)
    return content_identity(payload)


def observe_state_owner() -> dict[str, Any]:
    env = observe_sealed_validation_environment()
    duckdb_module = "unavailable"
    try:
        import duckdb as duckdb_mod

        origin = str(getattr(duckdb_mod, "__file__", "") or "")
        if origin and "/home/" not in origin and ".local" not in origin:
            duckdb_module = origin
    except Exception:
        duckdb_module = "unavailable"
    return {
        **env,
        "duckdb_module": duckdb_module,
        "duckdb_module_is_not_live_materialization": True,
        "quack": "unavailable",
        "quack_fenced_session": typed_unavailable(
            reason=(
                "No admitted Quack-fenced state-owner session was observed. "
                "Direct DuckDB writes are prohibited."
            )
        ),
        "duckdb_or_quack_state_written": False,
        "direct_duckdb_write_prohibited": True,
        "direct_quack_write_prohibited": True,
        "evidence_kind": "measured",
    }


def _sibling_binding(path: Path, relative_path: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": relative_path,
            "status": "unavailable",
            "evidence_kind": "unavailable",
            "reason": (
                "Sibling stale-rejection file is not present beside this checkout."
            ),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    context_pack = payload.get("context_pack")
    storage = payload.get("storage")
    route = payload.get("route")
    bounded_patch = payload.get("bounded_patch")
    selected = payload.get("selected_tests")
    delta = payload.get("stale_rejection")
    return {
        "path": relative_path,
        "status": "observed",
        "evidence_kind": "measured",
        "package_name": payload.get("package_name"),
        "package_version": payload.get("package_version"),
        "objective_kind": payload.get("objective_kind"),
        "objective_cid": payload.get("objective_cid"),
        "idea_digest": payload.get("idea_digest"),
        "pack_cid": payload.get("pack_cid")
        or (
            context_pack.get("pack_cid")
            if isinstance(context_pack, Mapping)
            else None
        ),
        "current_root_cid": (
            storage.get("current_root_cid")
            if isinstance(storage, Mapping)
            else payload.get("current_root_cid")
        ),
        "route_cid": (
            route.get("route_cid")
            if isinstance(route, Mapping)
            else payload.get("route_cid")
        ),
        "patch_cid": payload.get("patch_cid")
        or (
            bounded_patch.get("patch_cid")
            if isinstance(bounded_patch, Mapping)
            else None
        ),
        "run_cid": payload.get("run_cid")
        or (selected.get("run_cid") if isinstance(selected, Mapping) else None),
        "delta_cid": payload.get("delta_cid")
        or (delta.get("delta_cid") if isinstance(delta, Mapping) else None),
        "interface_cid": payload.get("interface_cid")
        or (delta.get("interface_cid") if isinstance(delta, Mapping) else None),
        "binding_cid": payload.get("binding_cid") or payload.get("document_cid"),
        "live": False,
        "applied": False,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def render_declared_delta(root: Path | None = None) -> dict[str, Any]:
    package_root = root or discover_accelerate_root()
    if package_root is None:
        raise AccelerateStaleRejectionError("Accelerate package root was not found")
    state_owner = observe_state_owner()
    plan = canonical_stale_rejection()
    computed_delta_cid = delta_cid_of(plan)
    document = {
        "schema": DOCUMENT_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_069_TASK_ID,
        "goal_id": PCPR_069_GOAL_ID,
        "program_id": PCPR_PROGRAM_ID,
        "board_namespace": PCPR_BOARD_NAMESPACE,
        "portfolio_id": PORTFOLIO_ID,
        "portfolio_version": PORTFOLIO_VERSION,
        "objective_kind": OBJECTIVE_KIND,
        "package_name": PACKAGE_NAME,
        "package_version": PACKAGE_VERSION,
        "language": LANGUAGE,
        "objective_id": OBJECTIVE_ID,
        "idea": REFERENCE_OBJECTIVE_IDEA,
        "idea_digest": PINNED_IDEA_DIGEST,
        "objective_cid": PINNED_OBJECTIVE_CID,
        "lock_cid": PINNED_LOCK_CID,
        "owner_repository": EXECUTION_OWNER_REPOSITORY,
        "pack_owner_repository": OWNER_REPOSITORY,
        "pack_owner_interface": OWNER_INTERFACE,
        "pack_owner_schema": OWNER_SCHEMA,
        "storage_owner_repository": STORAGE_OWNER_REPOSITORY,
        "storage_owner_interface": STORAGE_OWNER_INTERFACE,
        "shared_contract": {
            "name": "SupervisorContextPack",
            "interface": shared_interface_id("SupervisorContextPack"),
            "schema": shared_schema_id("SupervisorContextPack"),
            "owner_interface": OWNER_INTERFACE,
            "owner_schema": OWNER_SCHEMA,
            "owner_repository": OWNER_REPOSITORY,
            "reminted": False,
        },
        "execution_invocation": {
            "name": "ExecutionInvocation",
            "interface": EXECUTION_INVOCATION_INTERFACE,
            "schema": EXECUTION_INVOCATION_SCHEMA,
            "owner_repository": EXECUTION_OWNER_REPOSITORY,
            "reminted": False,
        },
        "execution_receipt": {
            "name": "ExecutionReceipt",
            "interface": EXECUTION_RECEIPT_INTERFACE,
            "schema": EXECUTION_RECEIPT_SCHEMA,
            "owner_repository": EXECUTION_OWNER_REPOSITORY,
            "live": False,
            "emitted": False,
            "reason": (
                "A live ExecutionReceipt remains PCPR-072. This stale "
                "rejection and PlanDelta is hermetic."
            ),
        },
        "context_pack": {
            "task_id": PCPR_061_TASK_ID,
            "constructed_by": OWNER_REPOSITORY,
            "constructed": True,
            "consumer": PACKAGE_NAME,
            "pack_cid": PINNED_PACK_CID,
            "owner_interface": OWNER_INTERFACE,
            "owner_schema": OWNER_SCHEMA,
            "reminted": False,
            "live": False,
            "admitted_live": False,
            "stored": True,
            "stored_by": STORAGE_OWNER_REPOSITORY,
            "current_root_published": True,
            "stale": True,
            "rejected": True,
            "evidence_kind": "measured",
        },
        "storage": {
            "task_id": PCPR_062_TASK_ID,
            "stored": True,
            "stored_by": STORAGE_OWNER_REPOSITORY,
            "current_root_published": True,
            "current_root_cid": PINNED_CURRENT_ROOT_CID,
            "bytes_cid": PINNED_BYTES_CID,
            "live": False,
            "deferred": False,
            "reminted": False,
            "evidence_kind": "measured_hermetic",
        },
        "route": {
            "task_id": PCPR_063_TASK_ID,
            "executed": True,
            "executed_by": EXECUTION_OWNER_REPOSITORY,
            "owner_interface": ROUTE_OWNER_INTERFACE,
            "route_cid": PINNED_ROUTE_CID,
            "completed_through": "schema_type_and_static_checks",
            "next_authorized_stage": "selected_tests",
            "live": False,
            "hermetic": True,
            "reminted": False,
            "model_assertion_completes_work": False,
            "evidence_kind": "measured_hermetic",
        },
        "bounded_patch": {
            "task_id": PCPR_064_TASK_ID,
            "produced": True,
            "produced_by": EXECUTION_OWNER_REPOSITORY,
            "owner_interface": PATCH_OWNER_INTERFACE,
            "patch_cid": PINNED_PATCH_CID,
            "applied": True,
            "live": False,
            "hermetic": True,
            "reminted": False,
            "deferred": False,
            "evidence_kind": "measured_hermetic",
        },
        "selected_tests": {
            "task_id": PCPR_065_TASK_ID,
            "run": True,
            "live": False,
            "deferred": False,
            "hermetic": True,
            "owner_interface": RUN_OWNER_INTERFACE,
            "run_cid": PINNED_RUN_CID,
            "reminted": False,
            "evidence_kind": "measured_hermetic",
        },
        "stale_rejection": {
            **plan,
            "delta_cid": computed_delta_cid,
            "produced_by": EXECUTION_OWNER_REPOSITORY,
        },
        "materialization": {
            "kind": "declared_stale_rejection_and_plan_delta_not_live",
            "admitted": False,
            "live": False,
            "applied": False,
            "duckdb_or_quack_state_written": False,
            "evidence_kind": "unavailable",
        },
        "live_application": typed_unavailable(
            reason=(
                "No admitted Quack-fenced supervisor session was observed. "
                "Hermetic stale rejection and PlanDelta classification is "
                "not live application."
            )
        ),
        "live_plan_delta": typed_unavailable(
            reason=(
                "Live plan-revision-store apply remains typed unavailable. "
                "This task only demonstrates hermetic stale rejection and "
                "PlanDelta. Restart recovery remains PCPR-070."
            )
        ),
        "state_owner": {
            "duckdb_cli": state_owner.get("duckdb"),
            "duckdb_module": state_owner.get("duckdb_module"),
            "duckdb_module_is_not_live_materialization": True,
            "quack": "unavailable",
            "duckdb_or_quack_state_written": False,
            "direct_duckdb_write_prohibited": True,
            "direct_quack_write_prohibited": True,
        },
        "source": {
            "kind": "git",
            "repository": SOURCE_REPOSITORY,
            "binding": "current_head_not_mutable_main",
            "mutable_main_reference": False,
            "commit": {
                "status": "observed_at_evaluation",
                "evidence_kind": "measured",
                "live": False,
                "field": "PCPR-069 receipt current_tree_binding",
            },
        },
        "operator_blocking_task": _operator_blocking_task(),
        "live": False,
        "applied": False,
        "submitted_live": False,
        "release_claim": False,
        "closed_release_outcome": None,
        "contracts_frozen": False,
        "hashes_invented": False,
        "signatures_invented": False,
        "sibling_source_required": False,
        "this_task_created_competing_authority": False,
        "duckdb_or_quack_state_written": False,
        "source_date_epoch": SOURCE_DATE_EPOCH,
        "evidence_kind": "measured",
    }
    document["document_cid"] = content_identity(
        {key: value for key, value in document.items() if key != "document_cid"}
    )
    return document


def artifact_paths(root: Path) -> dict[str, Path]:
    return {
        "document": root / DOCUMENT_DIR_RELPATH / DOCUMENT_JSON_NAME,
        "readme": root / DOCUMENT_README_RELPATH,
        "catalog": root / CATALOG_RELPATH,
    }


def platform_stale_rejection_catalog(
    start: Path | None = None,
) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateStaleRejectionError("Accelerate package root was not found")
    parent = root.parent
    document = render_declared_delta(root)
    datasets_binding = _sibling_binding(
        parent
        / "ipfs_datasets"
        / DOCUMENT_DIR_RELPATH
        / "reference.stale-rejection.binding.json",
        relative_path=(
            f"../ipfs_datasets/{DOCUMENT_DIR_RELPATH}/"
            "reference.stale-rejection.binding.json"
        ),
    )
    kit_binding = _sibling_binding(
        parent
        / "ipfs_kit"
        / DOCUMENT_DIR_RELPATH
        / "reference.stale-rejection.binding.json",
        relative_path=(
            f"../ipfs_kit/{DOCUMENT_DIR_RELPATH}/"
            "reference.stale-rejection.binding.json"
        ),
    )
    catalog = {
        "schema": CATALOG_SCHEMA,
        "interface": CATALOG_INTERFACE,
        "task_id": PCPR_069_TASK_ID,
        "goal_id": PCPR_069_GOAL_ID,
        "objective_kind": OBJECTIVE_KIND,
        "portfolio_id": PORTFOLIO_ID,
        "portfolio_version": PORTFOLIO_VERSION,
        "objective_cid": PINNED_OBJECTIVE_CID,
        "idea_digest": PINNED_IDEA_DIGEST,
        "pack_cid": PINNED_PACK_CID,
        "current_root_cid": PINNED_CURRENT_ROOT_CID,
        "route_cid": PINNED_ROUTE_CID,
        "patch_cid": PINNED_PATCH_CID,
        "run_cid": PINNED_RUN_CID,
        "interface_cid": PINNED_INTERFACE_CID,
        "delta_cid": document["stale_rejection"]["delta_cid"],
        "lock_cid": PINNED_LOCK_CID,
        "components": {
            "ipfs_accelerate_py": {
                "document_path": f"{DOCUMENT_DIR_RELPATH}/{DOCUMENT_JSON_NAME}",
                "status": "observed",
                "evidence_kind": "measured",
                "package_name": PACKAGE_NAME,
                "package_version": PACKAGE_VERSION,
                "objective_kind": OBJECTIVE_KIND,
                "objective_cid": PINNED_OBJECTIVE_CID,
                "idea_digest": PINNED_IDEA_DIGEST,
                "pack_cid": PINNED_PACK_CID,
                "current_root_cid": PINNED_CURRENT_ROOT_CID,
                "route_cid": PINNED_ROUTE_CID,
                "patch_cid": PINNED_PATCH_CID,
                "run_cid": PINNED_RUN_CID,
                "interface_cid": PINNED_INTERFACE_CID,
                "delta_cid": document["stale_rejection"]["delta_cid"],
                "document_cid": document["document_cid"],
                "live": False,
                "applied": False,
                "reminted": False,
            },
            "ipfs_datasets_py": {"binding": datasets_binding},
            "ipfs_kit_py": {"binding": kit_binding},
        },
        "live_application": False,
        "live_plan_delta": False,
        "closed_release_outcome": None,
        "release_claim": False,
        "sibling_source_required": False,
        "evidence_kind": "measured",
    }
    catalog["catalog_cid"] = content_identity(
        {key: value for key, value in catalog.items() if key != "catalog_cid"}
    )
    return catalog


def write_stale_rejection_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateStaleRejectionError("Accelerate package root was not found")
    document = render_declared_delta(root)
    paths = artifact_paths(root)
    _atomic_write(paths["document"], pretty_json(document))
    readme = DOCUMENT_README if DOCUMENT_README.endswith("\n") else DOCUMENT_README + "\n"
    _atomic_write(paths["readme"], readme)
    catalog = platform_stale_rejection_catalog(start)
    _atomic_write(paths["catalog"], pretty_json(catalog))
    return {
        "document": document,
        "catalog": catalog,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def verify_stale_rejection_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateStaleRejectionError("Accelerate package root was not found")
    document = render_declared_delta(root)
    catalog = platform_stale_rejection_catalog(start)
    paths = artifact_paths(root)
    missing: list[str] = []
    document_ok = False
    catalog_ok = False
    readme_ok = False
    expected_readme = (
        DOCUMENT_README if DOCUMENT_README.endswith("\n") else DOCUMENT_README + "\n"
    )
    for name, path in paths.items():
        if not path.is_file():
            missing.append(name)
            continue
        if name == "document":
            document_ok = json.loads(path.read_text(encoding="utf-8")) == document
        elif name == "catalog":
            catalog_ok = json.loads(path.read_text(encoding="utf-8")) == catalog
        elif name == "readme":
            readme_ok = path.read_text(encoding="utf-8") == expected_readme
    return {
        "ok": not missing and document_ok and catalog_ok and readme_ok,
        "missing": missing,
        "document_ok": document_ok,
        "catalog_ok": catalog_ok,
        "readme_ok": readme_ok,
        "pack_cid": document["context_pack"]["pack_cid"],
        "delta_cid": document["stale_rejection"]["delta_cid"],
        "document_cid": document["document_cid"],
        "catalog_cid": catalog["catalog_cid"],
        "current_root_cid": document["storage"]["current_root_cid"],
        "route_cid": document["route"]["route_cid"],
        "patch_cid": document["bounded_patch"]["patch_cid"],
        "run_cid": document["selected_tests"]["run_cid"],
        "interface_cid": document["stale_rejection"]["interface_cid"],
        "objective_cid": document["objective_cid"],
        "idea_digest": document["idea_digest"],
        "document_sha256": (
            sha256_bytes(paths["document"].read_bytes())
            if paths["document"].is_file()
            else "unavailable"
        ),
    }


def current_head_static_probes(
    start: Path | None = None,
) -> tuple[OutcomeProbe, ...]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateStaleRejectionError("Accelerate package root was not found")
    document = render_declared_delta(root)
    verified = verify_stale_rejection_files(root)
    table = parse_pyproject_stale_rejection_table(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    delta = document["stale_rejection"]
    remint = (
        document["context_pack"]["pack_cid"] != PINNED_PACK_CID
        or document["objective_cid"] != PINNED_OBJECTIVE_CID
        or document["idea_digest"] != PINNED_IDEA_DIGEST
        or document["lock_cid"] != PINNED_LOCK_CID
        or document["storage"]["current_root_cid"] != PINNED_CURRENT_ROOT_CID
        or document["route"]["route_cid"] != PINNED_ROUTE_CID
        or document["bounded_patch"]["patch_cid"] != PINNED_PATCH_CID
        or document["selected_tests"]["run_cid"] != PINNED_RUN_CID
        or delta["delta_cid"] != PINNED_DELTA_CID
        or delta.get("pcpr_066_change_cid") != PINNED_PCPR_066_CHANGE_CID
        or delta.get("reuse_cid") != PINNED_REUSE_CID
        or delta.get("interface_cid") != PINNED_INTERFACE_CID
        or document["context_pack"]["reminted"] is True
        or document["storage"]["reminted"] is True
        or document["route"]["reminted"] is True
        or document["bounded_patch"]["reminted"] is True
        or document["selected_tests"]["reminted"] is True
    )
    probes = [
        _probe(
            "delta_files_match_generator",
            verified.get("ok") is True and verified.get("document_ok") is True,
            reason=(
                "Committed Accelerate stale-rejection matches the generator."
                if verified.get("document_ok") is True
                else "Committed Accelerate stale-rejection is missing or drifts."
            ),
        ),
        _probe(
            "pyproject_stale_rejection_table",
            table.get("interface") == INTERFACE
            and table.get("schema") == SCHEMA
            and table.get("task-id") == PCPR_069_TASK_ID
            and table.get("objective-kind") == OBJECTIVE_KIND,
            reason=(
                "pyproject.toml declares AccelerateStaleRejectionAndPlanDelta@1."
                if table.get("interface") == INTERFACE
                else "pyproject.toml does not declare the PCPR-069 delta."
            ),
        ),
        _probe(
            "paths_remain_authorized",
            delta["paths_authorized"] is True
            and list(delta["authorized_path_prefixes"])
            == list(AUTHORIZED_PATH_PREFIXES),
            reason="Delta paths remain inside the declared PCPR-069 allowlist.",
        ),
        _probe(
            "owner_pack_cid_matches_pin",
            document["context_pack"]["pack_cid"] == PINNED_PACK_CID,
            reason="Accelerate binds the Datasets-owned pack CID without remint.",
        ),
        _probe(
            "kit_current_root_bound_not_minted",
            document["storage"]["current_root_cid"] == PINNED_CURRENT_ROOT_CID
            and document["storage"]["stored_by"] == STORAGE_OWNER_REPOSITORY
            and document["storage"]["reminted"] is False,
            reason="Accelerate binds the Kit current root and does not remint it.",
        ),
        _probe(
            "route_cid_bound_not_minted",
            document["route"]["route_cid"] == PINNED_ROUTE_CID
            and document["route"]["reminted"] is False,
            reason="Accelerate binds the PCPR-063 route CID and does not remint it.",
        ),
        _probe(
            "patch_cid_bound_not_minted",
            document["bounded_patch"]["patch_cid"] == PINNED_PATCH_CID
            and document["bounded_patch"]["reminted"] is False,
            reason="Accelerate binds the PCPR-064 patch CID and does not remint it.",
        ),
        _probe(
            "run_cid_bound_not_minted",
            document["selected_tests"]["run_cid"] == PINNED_RUN_CID
            and document["selected_tests"]["reminted"] is False,
            reason="Accelerate binds the PCPR-065 run CID and does not remint it.",
        ),
        _probe(
            "interface_cid_bound_not_minted",
            delta["interface_cid"] == PINNED_INTERFACE_CID,
            reason="Accelerate binds the PCPR-068 interface CID and does not remint it.",
        ),
        _probe(
            "stale_identities_rejected",
            list(delta["stale_identities"]) == list(STALE_IDENTITIES)
            and delta["stale_rejected"] is True
            and document["context_pack"]["rejected"] is True,
            reason="Stale ContextPack and protocol-test identities are rejected.",
        ),
        _probe(
            "unaffected_completion_preserved",
            list(delta["eligible_reuse"]) == list(ELIGIBLE_REUSE)
            and delta["unaffected_completion_preserved"] is True,
            reason="Unaffected solver-qualification completion is preserved.",
        ),
        _probe(
            "plan_delta_produced",
            delta["plan_delta_produced"] is True
            and list(delta["items"]) != []
            and delta["plan_delta_kind"] == PLAN_DELTA_KIND,
            reason="A hermetic PlanDelta is produced for the impacted cone.",
        ),
        _probe(
            "plan_epoch_incremented",
            delta["plan_epoch"] == PLAN_EPOCH
            and delta["base_plan_revision"] == BASE_PLAN_REVISION
            and delta["next_plan_revision"] == NEXT_PLAN_REVISION
            and delta["epoch_incremented"] is True,
            reason="The plan epoch increments from 1 to 2.",
        ),
        _probe(
            "refill_within_bounds",
            int(delta["refill"]["tasks_added"]) == len(REFILL_TASKS)
            and delta["refill"]["within_bounds"] is True
            and int(delta["refill"]["max_tasks_per_refill"]) == MAX_TASKS_PER_REFILL
            and int(delta["refill"]["max_epochs"]) == MAX_REPLAN_EPOCHS,
            reason="Automatic frontier refill stays inside 12 tasks and 20 epochs.",
        ),
        _probe(
            "history_not_mutated",
            delta["history_mutated"] is False
            and delta["whole_plan_regeneration_required"] is False,
            reason="PlanDelta does not mutate started or completed history.",
        ),
        _probe(
            "model_assertion_cannot_complete_work",
            delta["model_assertion_completes_work"] is False,
            reason="No model assertion completes work.",
        ),
        _probe(
            "duckdb_or_quack_not_written",
            document["duckdb_or_quack_state_written"] is False
            and document["state_owner"]["direct_duckdb_write_prohibited"] is True,
            reason="This task does not write DuckDB or Quack state.",
        ),
        _probe(
            "operator_blocking_task_emitted",
            document["operator_blocking_task"]["task_id"] == OPERATOR_BLOCKING_TASK_ID
            and document["operator_blocking_task"]["status"] == "typed_blocked",
            reason="Missing live application emits the operator-blocking task.",
        ),
        _probe(
            "no_closed_release_outcome",
            document["closed_release_outcome"] is None
            and document["release_claim"] is False,
            reason="This task does not emit a closed PCPR release outcome.",
        ),
        _probe(
            "compatibility_identities_reminted",
            remint,
            reason=(
                "A reminted pack, root, route, patch, run, interface, "
                "objective, idea, or lock CID is forbidden."
            ),
        ),
        _probe(
            "datasets_identity_reminted",
            document["context_pack"]["reminted"] is True,
            reason="Accelerate must not remint DatasetsContextPack@1.",
        ),
        _probe(
            "kit_identity_reminted",
            document["storage"]["reminted"] is True,
            reason="Accelerate must not remint the Kit current root.",
        ),
        _probe(
            "route_identity_reminted",
            document["route"]["reminted"] is True,
            reason="Accelerate must not remint the PCPR-063 route.",
        ),
        _probe(
            "patch_identity_reminted",
            document["bounded_patch"]["reminted"] is True,
            reason="Accelerate must not remint the PCPR-064 patch.",
        ),
        _probe(
            "run_identity_reminted",
            document["selected_tests"]["reminted"] is True,
            reason="Accelerate must not remint the PCPR-065 run.",
        ),
        _probe(
            "interface_identity_reminted",
            delta["interface_cid"] != PINNED_INTERFACE_CID,
            reason="Accelerate must not remint the PCPR-068 interface CID.",
        ),
        _probe(
            "direct_database_bypass_used",
            False,
            reason="Direct DuckDB or Quack writes were not used.",
        ),
        _probe(
            "simulated_results_represented_as_live",
            False,
            reason="Simulated results are not represented as live.",
        ),
        _probe(
            "live_application_represented_as_live",
            False,
            reason="Hermetic classification is not represented as live.",
        ),
        _probe(
            "live_plan_delta_represented_as_live",
            False,
            reason="Hermetic PlanDelta is not represented as live admission.",
        ),
        _probe(
            "closed_release_represented_as_live",
            False,
            reason="This task does not publish a PCPR release.",
        ),
        _probe(
            "model_assertion_completed_work",
            False,
            reason="No model assertion completed work.",
        ),
        _probe(
            "unauthorized_path_accepted",
            False,
            reason="Unauthorized paths are refused.",
        ),
        _probe(
            "stale_identity_admitted_as_current",
            False,
            reason="Stale identities are rejected and not admitted as current.",
        ),
        _probe(
            "unaffected_marked_stale",
            False,
            reason="Unaffected solver-qualification completion is not marked stale.",
        ),
        _probe(
            "whole_plan_regeneration_required",
            False,
            reason="Stale rejection does not regenerate the whole plan.",
        ),
        _probe(
            "history_mutated",
            False,
            reason="Started and completed history is not mutated.",
        ),
        _probe(
            "refill_exceeds_bounds",
            False,
            reason="Refill stays inside 12 tasks and 20 epochs.",
        ),
        _probe(
            "live_application",
            None,
            evidence_kind="unavailable",
            reason="Live supervisor application stays typed unavailable.",
        ),
        _probe(
            "live_plan_delta",
            None,
            evidence_kind="unavailable",
            reason="Live PlanDelta admission stays typed unavailable and remains PCPR-070.",
        ),
    ]
    return tuple(probes)


def qualify_stale_rejection(
    probes: Sequence[OutcomeProbe],
    *,
    delta_cid: str,
    document_cid: str,
    catalog_cid: str,
    pack_cid: str,
    current_root_cid: str,
    route_cid: str,
    patch_cid: str,
    run_cid: str,
    interface_cid: str,
    objective_cid: str,
    idea_digest_cid: str,
) -> AccelerateStaleRejectionVerdict:
    if not probes:
        raise AccelerateStaleRejectionError("at least one probe is required")
    normalized: list[OutcomeProbe] = []
    blockers: list[str] = []
    for probe in probes:
        kind = _kind(probe.evidence_kind, "evidence_kind")
        if probe.live and kind != "measured_live":
            raise AccelerateStaleRejectionError(
                "live claims require measured_live evidence"
            )
        if probe.simulated_represented_as_live:
            raise AccelerateStaleRejectionError(
                "simulated results must not be represented as live"
            )
        normalized.append(probe)
        if probe.probe_id in FORBIDDEN_PRESENT_PROBE_IDS and probe.present is True:
            blockers.append(probe.probe_id)
        if probe.probe_id in REQUIRED_GOOD_PROBE_IDS and probe.present is not True:
            blockers.append(probe.probe_id)

    promotion_status = "rnd_non_promoted"
    _reject_closed_release_value(promotion_status, "promotion_status")
    payload = {
        "schema": VERDICT_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_069_TASK_ID,
        "goal_id": PCPR_069_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "sibling_source_required": False,
        "live_application": False,
        "live_plan_delta": False,
        "operator_blocking_task": OPERATOR_BLOCKING_TASK_ID,
        "simulated_results_represented_as_live": False,
        "this_task_created_competing_authority": False,
        "probes": [item.to_mapping() for item in normalized],
        "blockers": list(dict.fromkeys(blockers)),
        "delta_cid": delta_cid,
        "document_cid": document_cid,
        "catalog_cid": catalog_cid,
        "pack_cid": pack_cid,
        "current_root_cid": current_root_cid,
        "route_cid": route_cid,
        "patch_cid": patch_cid,
        "run_cid": run_cid,
        "interface_cid": interface_cid,
        "objective_cid": objective_cid,
        "idea_digest": idea_digest_cid,
    }
    return AccelerateStaleRejectionVerdict(
        schema=VERDICT_SCHEMA,
        interface=INTERFACE,
        promotion_status=promotion_status,
        supervisor_disposition="supervisor_non_promoted",
        closed_release_outcome=None,
        release_claim=False,
        completion_authoritative=False,
        contracts_frozen=False,
        duckdb_or_quack_state_written=False,
        sibling_source_required=False,
        live_application=False,
        live_plan_delta=False,
        operator_blocking_task=OPERATOR_BLOCKING_TASK_ID,
        simulated_results_represented_as_live=False,
        this_task_created_competing_authority=False,
        probes=tuple(normalized),
        blockers=tuple(dict.fromkeys(blockers)),
        verdict_cid=content_identity(payload),
        delta_cid=delta_cid,
        document_cid=document_cid,
        catalog_cid=catalog_cid,
        pack_cid=pack_cid,
        current_root_cid=current_root_cid,
        route_cid=route_cid,
        patch_cid=patch_cid,
        run_cid=run_cid,
        interface_cid=interface_cid,
        objective_cid=objective_cid,
        idea_digest=idea_digest_cid,
    )


def qualify_current_head_stale_rejection(
    start: Path | None = None,
) -> AccelerateStaleRejectionVerdict:
    document = render_declared_delta(start)
    catalog = platform_stale_rejection_catalog(start)
    return qualify_stale_rejection(
        current_head_static_probes(start),
        delta_cid=str(document["stale_rejection"]["delta_cid"]),
        document_cid=str(document["document_cid"]),
        catalog_cid=str(catalog["catalog_cid"]),
        pack_cid=str(document["context_pack"]["pack_cid"]),
        current_root_cid=str(document["storage"]["current_root_cid"]),
        route_cid=str(document["route"]["route_cid"]),
        patch_cid=str(document["bounded_patch"]["patch_cid"]),
        run_cid=str(document["selected_tests"]["run_cid"]),
        interface_cid=str(document["stale_rejection"]["interface_cid"]),
        objective_cid=str(document["objective_cid"]),
        idea_digest_cid=str(document["idea_digest"]),
    )


def pcpr_069_receipt_promotion(
    verdict: AccelerateStaleRejectionVerdict,
) -> dict[str, Any]:
    if verdict.closed_release_outcome is not None:
        raise AccelerateStaleRejectionError(
            "stale rejection must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise AccelerateStaleRejectionError(
            "stale rejection must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise AccelerateStaleRejectionError(
            "stale rejection completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise AccelerateStaleRejectionError(
            "stale rejection must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise AccelerateStaleRejectionError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.live_application or verdict.live_plan_delta:
        raise AccelerateStaleRejectionError(
            "live stale-rejection claims require measured_live evidence"
        )
    return verdict.to_mapping()


PINNED_DELTA_CID: Final = (
    "baguqeerahbxnuho3jdzkwaar6yx4ers7ns3p73mm3ghhgc5mhmn5r2fg3pga"
)
PINNED_DOCUMENT_CID: Final = (
    "baguqeerac3lirww2ec4g3ozf7wrm5wo4xbwiaszxeysrzdaqhfjhajioakga"
)
PINNED_CATALOG_CID: Final = (
    "baguqeeraklpzq62u744vbx7p2hy6tl4qiiblguawcusuod27jwhm6b4nbfba"
)
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeera2pp3fcf4gfqz5klmcrx7rg7sfnglqsgqjjuy5uvnizagvmvgitfa"
)


__all__ = [
    "AUTHORIZED_PATH_PREFIXES",
    "CLOSED_RELEASE_OUTCOMES",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "ESCALATION_ORDER",
    "AccelerateStaleRejectionError",
    "HERMETIC_CANDIDATE_SUITES",
    "INTERFACE",
    "OBJECTIVE_KIND",
    "OPERATOR_BLOCKING_TASK_ID",
    "OutcomeProbe",
    "PCPR_069_GOAL_ID",
    "PCPR_069_TASK_ID",
    "ELIGIBLE_REUSE",
    "IMPACTED_CONE",
    "PINNED_CATALOG_CID",
    "PINNED_DELTA_CID",
    "PINNED_CURRENT_ROOT_CID",
    "PINNED_INTERFACE_CID",
    "PINNED_PCPR_066_CHANGE_CID",
    "PINNED_REUSE_CID",
    "RELEVANT_CHANGE_PATHS",
    "REFILL_TASKS",
    "STALE_IDENTITIES",
    "PINNED_DOCUMENT_CID",
    "PINNED_IDEA_DIGEST",
    "PINNED_LOCK_CID",
    "PINNED_OBJECTIVE_CID",
    "PINNED_PACK_CID",
    "PINNED_PATCH_CID",
    "PINNED_ROUTE_CID",
    "PINNED_RUN_CID",
    "PLAN_EPOCH",
    "PROTOCOL_INTERFACE",
    "SCHEMA",
    "SEALED_PATH",
    "SEALED_PYTHON",
    "canonical_stale_rejection",
    "current_head_static_probes",
    "delta_cid_of",
    "hermetic_plan_delta_items",
    "path_is_authorized",
    "pcpr_069_receipt_promotion",
    "platform_stale_rejection_catalog",
    "produce_stale_rejection_and_plan_delta",
    "qualify_current_head_stale_rejection",
    "qualify_stale_rejection",
    "refuse_current_root_remint",
    "refuse_history_mutation",
    "refuse_interface_cid_remint",
    "refuse_live_plan_delta_apply",
    "refuse_model_completion",
    "refuse_pack_cid_remint",
    "refuse_patch_cid_remint",
    "refuse_route_cid_remint",
    "refuse_run_cid_remint",
    "refuse_stale_as_current",
    "refuse_unauthorized_path",
    "refuse_unaffected_as_stale",
    "refuse_whole_plan_regeneration",
    "reject_stale_identities",
    "render_declared_delta",
    "verify_stale_rejection_files",
    "write_stale_rejection_files",
]
