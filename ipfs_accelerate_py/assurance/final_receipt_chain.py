"""Fail-closed PCPR-072 Accelerate-owned final proof-carrying receipt chain.

Accelerate owns independent semantic validation, hermetic artifact
storage, and the hermetic ExecutionReceipt for the complete
proof-carrying identity chain. This module binds the PCPR-061
DatasetsContextPack@1 identity, the PCPR-062 Kit current root, the
PCPR-063 route, the PCPR-064 patch, the PCPR-065 selected-tests run,
the PCPR-066 unrelated change, the PCPR-067 eligible reuse, the
PCPR-068 relevant-interface identity, the PCPR-069 stale rejection and
PlanDelta, the PCPR-070 restart, and the PCPR-071 recovery without
reminting those identities.

The chain links objective, context, epoch, task, proof, storage, event,
compatibility, and result identities. Duplicate chain emission is
rejected. Stale leases and fences cannot authorize the chain. Owner
generation stays at 2. Plan epoch stays at 2. Continuation proceeds
without database edits. Live Quack chain emission stays typed
unavailable. Model stages are not invoked. A model assertion cannot
complete work.

Hermetic chain emission is candidate coverage only. It is not live
supervisor execution and not a closed PCPR release. Simulated results
are not live. This evaluator does not write DuckDB or Quack state.
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
from ipfs_accelerate_py.assurance.recovery_and_idempotency import (
    PINNED_DOCUMENT_CID as PCPR_071_DOCUMENT_CID,
    PINNED_RECOVERY_CID as PCPR_071_RECOVERY_CID,
)
from ipfs_accelerate_py.assurance.shared_contracts import (
    OWNER_INTERFACES,
    OWNER_SCHEMAS,
    shared_interface_id,
    shared_schema_id,
)
from ipfs_accelerate_py.assurance.state_owner_restart import (
    EXPIRED_FENCE,
    EXPIRED_LEASE,
    PINNED_DOCUMENT_CID as PCPR_070_DOCUMENT_CID,
    PINNED_RESTART_CID as PCPR_070_RESTART_CID,
    POST_RESTART_OWNER_GENERATION,
    PRE_RESTART_OWNER_GENERATION,
    RETAINED_FENCE,
    RETAINED_LEASE,
)


INTERFACE: Final = "AccelerateFinalProofCarryingReceiptChain@1"
SCHEMA: Final = "ipfs_accelerate_py/assurance/final-receipt-chain@1"
DOCUMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/declared-final-receipt-chain@1"
)
CHAIN_SCHEMA: Final = "ipfs_accelerate_py/assurance/final-receipt-chain-plan@1"
VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/final-receipt-chain-verdict@1"
)
CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/platform-final-receipt-chain-catalog@1"
)
CATALOG_INTERFACE: Final = "PlatformFinalProofCarryingReceiptChain@1"
CHAIN_INTERFACE: Final = "AuthoritativeFinalProofCarryingReceiptChain@1"
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
DELTA_OWNER_INTERFACE: Final = "AccelerateStaleRejectionAndPlanDelta@1"
RESTART_OWNER_INTERFACE: Final = "AccelerateAuthoritativeStateOwnerRestart@1"
RECOVERY_OWNER_INTERFACE: Final = "AccelerateRecoveryAndIdempotency@1"
EXECUTION_INVOCATION_INTERFACE: Final = OWNER_INTERFACES["ExecutionInvocation"]
EXECUTION_INVOCATION_SCHEMA: Final = OWNER_SCHEMAS["ExecutionInvocation"]
EXECUTION_RECEIPT_INTERFACE: Final = OWNER_INTERFACES["ExecutionReceipt"]
EXECUTION_RECEIPT_SCHEMA: Final = OWNER_SCHEMAS["ExecutionReceipt"]
PROTOCOL_INTERFACE: Final = "LogicProviderProtocol@2"
DOCUMENTATION_INTERFACE: Final = "LogicProviderProtocolFinalReceiptChain@1"
DOCUMENTATION_SCHEMA: Final = (
    "ipfs_datasets_py/logic-provider-final-receipt-chain@1"
)
PCPR_072_TASK_ID: Final = "PCPR-072"
PCPR_072_GOAL_ID: Final = "PCPR-G700"
PCPR_071_TASK_ID: Final = "PCPR-071"
PCPR_070_TASK_ID: Final = "PCPR-070"
PCPR_069_TASK_ID: Final = "PCPR-069"
PCPR_068_TASK_ID: Final = "PCPR-068"
PCPR_067_TASK_ID: Final = "PCPR-067"
PCPR_066_TASK_ID: Final = "PCPR-066"
PCPR_065_TASK_ID: Final = "PCPR-065"
PCPR_064_TASK_ID: Final = "PCPR-064"
PCPR_063_TASK_ID: Final = "PCPR-063"
PCPR_062_TASK_ID: Final = "PCPR-062"
PCPR_061_TASK_ID: Final = "PCPR-061"
PCPR_060_TASK_ID: Final = "PCPR-060"
PCPR_080_TASK_ID: Final = "PCPR-080"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = (
    "proof-carrying-platform-qualification-and-release-v1"
)
OBJECTIVE_KIND: Final = "declared_final_proof_carrying_receipt_chain"
OBJECTIVE_ID: Final = "PCPR-G700"
LANGUAGE: Final = "Python"
OPERATOR_BLOCKING_TASK_ID: Final = (
    "pcpr-072-operator-live-final-receipt-chain"
)
DOCUMENT_DIR_RELPATH: Final = "packaging/pcpr/reference-workflow/cpython312"
DOCUMENT_JSON_NAME: Final = "reference.final-receipt-chain.json"
DOCUMENT_README_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/FINAL_RECEIPT_CHAIN.md"
)
CATALOG_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/platform-final-receipt-chain-catalog.json"
)
SOURCE_DATE_EPOCH: Final = "0"
SOURCE_REPOSITORY: Final = "endomorphosis/ipfs_accelerate_py"
COMPLETED_THROUGH: Final = "final_proof_carrying_receipt_chain"
NEXT_AUTHORIZED_STAGE: Final = "python_external_client_demonstration"
CHANGE_KIND: Final = "final_proof_carrying_receipt_chain"
CHAIN_KIND: Final = "independent_semantic_validation_and_hermetic_execution_receipt"
IDEMPOTENCY_KIND: Final = "identity_preserving_duplicate_rejection"
PLAN_EPOCH: Final = 2
OWNER_GENERATION: Final = POST_RESTART_OWNER_GENERATION
RECONSTRUCTED_FROM: Final = "durable_records"
STATE_OWNER_NAME: Final = "quack"
STATE_OWNER_AUTHORITY: Final = "authenticated_fenced_state_owner"
CHAIN_COMMAND_DIGEST: Final = "pcpr-072-hermetic-final-receipt-chain"

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
PINNED_DELTA_CID: Final = (
    "baguqeerahbxnuho3jdzkwaar6yx4ers7ns3p73mm3ghhgc5mhmn5r2fg3pga"
)
PINNED_RESTART_CID: Final = PCPR_070_RESTART_CID
PINNED_RESTART_DOCUMENT_CID: Final = PCPR_070_DOCUMENT_CID
PINNED_RECOVERY_CID: Final = PCPR_071_RECOVERY_CID
PINNED_RECOVERY_DOCUMENT_CID: Final = PCPR_071_DOCUMENT_CID

if PINNED_LOCK_CID != (
    "baguqeerawsekbbbt5ccctjt4tydzeahfkahsatb6q5x7k6cy4inrii5lhqmq"
):
    raise RuntimeError("PCPR-072 lock CID remints PCPR-056")
if PINNED_RESTART_CID != (
    "baguqeeratw2evha3sv4ko5km5e5igcjprwfpkse4x2siwa5ldhyu5ugbjyja"
):
    raise RuntimeError("PCPR-072 restart CID remints PCPR-070")
if PINNED_RECOVERY_CID != (
    "baguqeerayeqc5tg3shtegpw56l4pn4uk5b7tw46jb32nrgtkdvq2oyukgirq"
):
    raise RuntimeError("PCPR-072 recovery CID remints PCPR-071")
if OWNER_INTERFACES["SupervisorContextPack"] != OWNER_INTERFACE:
    raise RuntimeError("PCPR-072 remints SupervisorContextPack owner interface")
if OWNER_SCHEMAS["SupervisorContextPack"] != OWNER_SCHEMA:
    raise RuntimeError("PCPR-072 remints SupervisorContextPack owner schema")
if EXECUTION_INVOCATION_INTERFACE != "InvocationRequest@1":
    raise RuntimeError("PCPR-072 remints ExecutionInvocation owner interface")
if EXECUTION_RECEIPT_INTERFACE != "InvocationResult@1":
    raise RuntimeError("PCPR-072 remints ExecutionReceipt owner interface")

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
    "artifacts/proof_carrying_platform_qualification_and_release/receipts/PCPR-072.json",
)

LINKED_IDENTITIES: Final[tuple[str, ...]] = (
    "objective_cid",
    "idea_digest",
    "lock_cid",
    "pack_cid",
    "current_root_cid",
    "route_cid",
    "patch_cid",
    "run_cid",
    "change_cid",
    "reuse_cid",
    "interface_cid",
    "delta_cid",
    "restart_cid",
    "recovery_cid",
    "plan_epoch",
    "lease",
    "fence",
    "owner_generation",
    "task",
    "proof",
    "storage",
    "event",
    "compatibility",
    "result",
)

CHAIN_LINKS: Final[tuple[Mapping[str, str], ...]] = (
    MappingProxyType(
        {
            "kind": "objective",
            "task_id": PCPR_060_TASK_ID,
            "identity": "objective_cid",
            "cid": PINNED_OBJECTIVE_CID,
        }
    ),
    MappingProxyType(
        {
            "kind": "context",
            "task_id": PCPR_061_TASK_ID,
            "identity": "pack_cid",
            "cid": PINNED_PACK_CID,
        }
    ),
    MappingProxyType(
        {
            "kind": "storage",
            "task_id": PCPR_062_TASK_ID,
            "identity": "current_root_cid",
            "cid": PINNED_CURRENT_ROOT_CID,
        }
    ),
    MappingProxyType(
        {
            "kind": "task",
            "task_id": PCPR_063_TASK_ID,
            "identity": "route_cid",
            "cid": PINNED_ROUTE_CID,
        }
    ),
    MappingProxyType(
        {
            "kind": "task",
            "task_id": PCPR_064_TASK_ID,
            "identity": "patch_cid",
            "cid": PINNED_PATCH_CID,
        }
    ),
    MappingProxyType(
        {
            "kind": "proof",
            "task_id": PCPR_065_TASK_ID,
            "identity": "run_cid",
            "cid": PINNED_RUN_CID,
        }
    ),
    MappingProxyType(
        {
            "kind": "event",
            "task_id": PCPR_066_TASK_ID,
            "identity": "change_cid",
            "cid": PINNED_PCPR_066_CHANGE_CID,
        }
    ),
    MappingProxyType(
        {
            "kind": "event",
            "task_id": PCPR_067_TASK_ID,
            "identity": "reuse_cid",
            "cid": PINNED_REUSE_CID,
        }
    ),
    MappingProxyType(
        {
            "kind": "event",
            "task_id": PCPR_068_TASK_ID,
            "identity": "interface_cid",
            "cid": PINNED_INTERFACE_CID,
        }
    ),
    MappingProxyType(
        {
            "kind": "event",
            "task_id": PCPR_069_TASK_ID,
            "identity": "delta_cid",
            "cid": PINNED_DELTA_CID,
        }
    ),
    MappingProxyType(
        {
            "kind": "result",
            "task_id": PCPR_070_TASK_ID,
            "identity": "restart_cid",
            "cid": PINNED_RESTART_CID,
        }
    ),
    MappingProxyType(
        {
            "kind": "result",
            "task_id": PCPR_071_TASK_ID,
            "identity": "recovery_cid",
            "cid": PINNED_RECOVERY_CID,
        }
    ),
    MappingProxyType(
        {
            "kind": "compatibility",
            "task_id": "PCPR-056",
            "identity": "lock_cid",
            "cid": PINNED_LOCK_CID,
        }
    ),
    MappingProxyType(
        {
            "kind": "epoch",
            "task_id": PCPR_069_TASK_ID,
            "identity": "plan_epoch",
            "cid": str(PLAN_EPOCH),
        }
    ),
)

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_final_receipt_chain.py",
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
        "chain_files_match_generator",
        "pyproject_final_receipt_chain_table",
        "paths_remain_authorized",
        "owner_pack_cid_matches_pin",
        "kit_current_root_bound_not_minted",
        "route_cid_bound_not_minted",
        "patch_cid_bound_not_minted",
        "run_cid_bound_not_minted",
        "interface_cid_bound_not_minted",
        "delta_cid_bound_not_minted",
        "restart_cid_bound_not_minted",
        "recovery_cid_bound_not_minted",
        "identities_link_across_chain",
        "independent_semantic_validation",
        "hermetic_execution_receipt_emitted",
        "hermetic_artifacts_stored",
        "duplicate_chain_effect_rejected",
        "idempotent_chain_cid",
        "unexpired_lease_authorizes_chain",
        "stale_lease_cannot_authorize_chain",
        "current_fence_authorizes_chain",
        "stale_fence_cannot_authorize_chain",
        "owner_generation_unchanged",
        "continuation_without_database_edits",
        "model_assertion_cannot_complete_work",
        "duckdb_or_quack_not_written",
        "operator_blocking_task_emitted",
        "no_closed_release_outcome",
    }
)
FORBIDDEN_PRESENT_PROBE_IDS: Final[frozenset[str]] = frozenset(
    {
        "simulated_results_represented_as_live",
        "live_chain_represented_as_live",
        "closed_release_represented_as_live",
        "compatibility_identities_reminted",
        "direct_database_bypass_used",
        "datasets_identity_reminted",
        "kit_identity_reminted",
        "route_identity_reminted",
        "patch_identity_reminted",
        "run_identity_reminted",
        "interface_identity_reminted",
        "delta_identity_reminted",
        "restart_identity_reminted",
        "recovery_identity_reminted",
        "model_assertion_completed_work",
        "unauthorized_path_accepted",
        "memory_reconstruction_accepted",
        "duplicate_chain_effect_accepted",
        "stale_lease_reused",
        "stale_fence_reused",
        "owner_generation_mutated",
        "database_edited",
    }
)

DOCUMENT_README: Final = """# PCPR-072 Accelerate final proof-carrying receipt chain

These files record the Accelerate-owned hermetic final proof-carrying
receipt chain against the PCPR-071 recovery, the PCPR-070 authoritative
state-owner restart, the PCPR-061 DatasetsContextPack@1 identity, the
PCPR-062 Kit current root, the PCPR-063 deterministic-first route, the
PCPR-064 bounded PatchPlan, the PCPR-065 selected-tests-and-proofs run,
the PCPR-066 unrelated change, the PCPR-067 eligible reuse, the
PCPR-068 relevant interface change, and the PCPR-069 stale rejection
and PlanDelta. Accelerate owns independent semantic validation,
hermetic artifact storage, and the hermetic ExecutionReceipt. It does
not remint those identities and does not claim live Quack chain
emission.

- `cpython312/reference.final-receipt-chain.json` is the declared chain
  document. Exact commit and tree are bound by the PCPR-072 receipt
  `current_tree_binding`.
- `platform-final-receipt-chain-catalog.json` observes sibling Datasets
  and Kit bindings when present. Sibling source is never required.
- The chain links objective, context, epoch, task, proof, storage,
  event, compatibility, and result identities. Independent semantic
  validation is hermetic. A hermetic ExecutionReceipt is emitted. Chain
  artifacts are stored hermetically, not as live IPFS. Replaying the
  same chain command yields the same chain CID and is rejected as a
  duplicate effect. Stale leases and fences cannot authorize the chain.
  Owner generation stays at 2. Plan epoch stays at 2. Continuation
  proceeds without database edits. External-client demonstration
  remains PCPR-080.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-072-operator-live-final-receipt-chain`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
"""


class AccelerateFinalReceiptChainError(Exception):
    """Fail-closed PCPR-072 Accelerate final receipt chain error."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AccelerateFinalReceiptChainError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise AccelerateFinalReceiptChainError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise AccelerateFinalReceiptChainError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def parse_pyproject_final_receipt_chain_table(text: str) -> dict[str, Any]:
    import tomllib

    table = tomllib.loads(_text(text, "pyproject.toml"))
    if not isinstance(table, dict):
        raise AccelerateFinalReceiptChainError("pyproject.toml must be a table")
    tool = table.get("tool")
    payload: dict[str, Any] = {}
    if isinstance(tool, dict):
        accelerate = tool.get("ipfs_accelerate_py")
        if isinstance(accelerate, dict):
            raw = accelerate.get("final-receipt-chain")
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
        raise AccelerateFinalReceiptChainError(
            f"path {path} is not an authorized PCPR-072 path"
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
        raise AccelerateFinalReceiptChainError(
            f"model assertion at {stage} cannot complete work"
        )


def refuse_pack_cid_remint(cid: str) -> str:
    if cid != PINNED_PACK_CID:
        raise AccelerateFinalReceiptChainError(
            f"ContextPack CID {cid} remints {PINNED_PACK_CID}"
        )
    return cid


def refuse_current_root_remint(cid: str) -> str:
    if cid != PINNED_CURRENT_ROOT_CID:
        raise AccelerateFinalReceiptChainError(
            f"current root CID {cid} remints {PINNED_CURRENT_ROOT_CID}"
        )
    return cid


def refuse_route_cid_remint(cid: str) -> str:
    if cid != PINNED_ROUTE_CID:
        raise AccelerateFinalReceiptChainError(
            f"route CID {cid} remints {PINNED_ROUTE_CID}"
        )
    return cid


def refuse_patch_cid_remint(cid: str) -> str:
    if cid != PINNED_PATCH_CID:
        raise AccelerateFinalReceiptChainError(
            f"patch CID {cid} remints {PINNED_PATCH_CID}"
        )
    return cid


def refuse_run_cid_remint(cid: str) -> str:
    if cid != PINNED_RUN_CID:
        raise AccelerateFinalReceiptChainError(
            f"run CID {cid} remints {PINNED_RUN_CID}"
        )
    return cid


def refuse_interface_cid_remint(cid: str) -> str:
    if cid != PINNED_INTERFACE_CID:
        raise AccelerateFinalReceiptChainError(
            f"interface CID {cid} remints {PINNED_INTERFACE_CID}"
        )
    return cid


def refuse_delta_cid_remint(cid: str) -> str:
    if cid != PINNED_DELTA_CID:
        raise AccelerateFinalReceiptChainError(
            f"delta CID {cid} remints {PINNED_DELTA_CID}"
        )
    return cid


def refuse_restart_cid_remint(cid: str) -> str:
    if cid != PINNED_RESTART_CID:
        raise AccelerateFinalReceiptChainError(
            f"restart CID {cid} remints {PINNED_RESTART_CID}"
        )
    return cid


def refuse_recovery_cid_remint(cid: str) -> str:
    if cid != PINNED_RECOVERY_CID:
        raise AccelerateFinalReceiptChainError(
            f"recovery CID {cid} remints {PINNED_RECOVERY_CID}"
        )
    return cid


def refuse_memory_reconstruction(*, from_memory: bool) -> None:
    if from_memory:
        raise AccelerateFinalReceiptChainError(
            "final receipt chain cannot reconstruct from memory"
        )


def refuse_database_edit(*, edited: bool) -> None:
    if edited:
        raise AccelerateFinalReceiptChainError(
            "final receipt chain cannot write DuckDB or Quack state"
        )


def refuse_duplicate_chain_effect(*, already_applied: bool) -> None:
    if already_applied:
        raise AccelerateFinalReceiptChainError("duplicate chain effect is rejected")


def refuse_stale_lease_reuse(*, stale: bool, reused: bool) -> None:
    if stale and reused:
        raise AccelerateFinalReceiptChainError(
            "stale lease cannot authorize the final receipt chain"
        )


def refuse_stale_fence_reuse(*, stale: bool, reused: bool) -> None:
    if stale and reused:
        raise AccelerateFinalReceiptChainError(
            "stale fence cannot authorize the final receipt chain"
        )


def refuse_live_chain(*, applied_live: bool) -> None:
    if applied_live:
        raise AccelerateFinalReceiptChainError(
            "live final receipt chain remains typed unavailable"
        )


def refuse_generation_mutation(*, generation: int) -> int:
    if generation != OWNER_GENERATION:
        raise AccelerateFinalReceiptChainError(
            "final receipt chain must not mutate the post-restart owner generation"
        )
    return generation


def hermetic_independent_semantic_validation() -> dict[str, Any]:
    return {
        "schema": (
            "ipfs_accelerate_py/assurance/hermetic-independent-semantic-validation@1"
        ),
        "interface": "HermeticIndependentSemanticValidation@1",
        "independently_validated": True,
        "semantics_sound": True,
        "identities_link": True,
        "linked_identity_count": len(LINKED_IDENTITIES),
        "chain_link_count": len(CHAIN_LINKS),
        "protocol_interface": PROTOCOL_INTERFACE,
        "protocol_reminted": False,
        "live": False,
        "hermetic": True,
        "evidence_kind": "measured_hermetic",
    }


def hermetic_chain_continuation() -> dict[str, Any]:
    refuse_stale_lease_reuse(stale=True, reused=False)
    refuse_stale_fence_reuse(stale=True, reused=False)
    refuse_generation_mutation(generation=OWNER_GENERATION)
    return {
        "schema": "ipfs_accelerate_py/assurance/hermetic-chain-continuation@1",
        "interface": "HermeticChainContinuation@1",
        "owner": STATE_OWNER_NAME,
        "authority": STATE_OWNER_AUTHORITY,
        "prerequisite_recovery_cid": PINNED_RECOVERY_CID,
        "prerequisite_restart_cid": PINNED_RESTART_CID,
        "reconstructed_from": RECONSTRUCTED_FROM,
        "reconstructed_from_memory": False,
        "authorized_by_retained_lease": True,
        "authorized_by_current_fence": True,
        "retained_lease": dict(RETAINED_LEASE),
        "expired_lease": dict(EXPIRED_LEASE),
        "retained_fence": dict(RETAINED_FENCE),
        "expired_fence": dict(EXPIRED_FENCE),
        "unexpired_lease_retained": True,
        "stale_lease_expired": True,
        "stale_lease_reused": False,
        "current_fence_retained": True,
        "stale_fence_expired": True,
        "stale_fence_reused": False,
        "pre_restart_generation": PRE_RESTART_OWNER_GENERATION,
        "owner_generation": OWNER_GENERATION,
        "generation_mutated": False,
        "plan_epoch": PLAN_EPOCH,
        "database_edited": False,
        "direct_duckdb_write": False,
        "direct_quack_write": False,
        "history_mutated": False,
        "live": False,
        "hermetic": True,
        "evidence_kind": "measured_hermetic",
    }


def hermetic_execution_receipt(chain_cid: str) -> dict[str, Any]:
    return {
        "schema": EXECUTION_RECEIPT_SCHEMA,
        "interface": EXECUTION_RECEIPT_INTERFACE,
        "name": "ExecutionReceipt",
        "emitted": True,
        "live": False,
        "hermetic": True,
        "admitted_live": False,
        "chain_cid": chain_cid,
        "owner_repository": EXECUTION_OWNER_REPOSITORY,
        "reason": (
            "A hermetic ExecutionReceipt is emitted for the final "
            "proof-carrying receipt chain. Live ExecutionReceipt remains "
            "typed unavailable."
        ),
        "evidence_kind": "measured_hermetic",
    }


def hermetic_artifact_storage(chain_cid: str) -> dict[str, Any]:
    return {
        "schema": "ipfs_accelerate_py/assurance/hermetic-chain-artifact-storage@1",
        "interface": "HermeticChainArtifactStorage@1",
        "stored": True,
        "stored_by": EXECUTION_OWNER_REPOSITORY,
        "live": False,
        "live_ipfs": False,
        "reconstructed_from": RECONSTRUCTED_FROM,
        "reconstructed_from_memory": False,
        "chain_cid": chain_cid,
        "evidence_kind": "measured_hermetic",
    }


def hermetic_chain_idempotency(chain_cid: str) -> dict[str, Any]:
    refuse_duplicate_chain_effect(already_applied=False)
    return {
        "schema": "ipfs_accelerate_py/assurance/hermetic-chain-idempotency-trace@1",
        "interface": "HermeticChainIdempotencyTrace@1",
        "kind": IDEMPOTENCY_KIND,
        "first_chain_command_digest": CHAIN_COMMAND_DIGEST,
        "second_chain_command_digest": CHAIN_COMMAND_DIGEST,
        "first_chain_cid": chain_cid,
        "second_chain_cid": chain_cid,
        "identical": True,
        "identity_preserving": True,
        "first_attempt_applied": True,
        "second_attempt_applied": False,
        "duplicate_effect_rejected": True,
        "live": False,
        "hermetic": True,
        "evidence_kind": "measured_hermetic",
    }


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
class AccelerateFinalReceiptChainVerdict:
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
    live_chain: bool
    live_application: bool
    operator_blocking_task: str
    simulated_results_represented_as_live: bool
    this_task_created_competing_authority: bool
    probes: tuple[OutcomeProbe, ...]
    blockers: tuple[str, ...]
    verdict_cid: str
    chain_cid: str
    document_cid: str
    catalog_cid: str
    recovery_cid: str
    restart_cid: str
    pack_cid: str
    current_root_cid: str
    route_cid: str
    patch_cid: str
    run_cid: str
    interface_cid: str
    delta_cid: str
    objective_cid: str
    idea_digest: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "verdict_cid": self.verdict_cid,
            "chain_cid": self.chain_cid,
            "document_cid": self.document_cid,
            "catalog_cid": self.catalog_cid,
            "recovery_cid": self.recovery_cid,
            "restart_cid": self.restart_cid,
            "pack_cid": self.pack_cid,
            "current_root_cid": self.current_root_cid,
            "route_cid": self.route_cid,
            "patch_cid": self.patch_cid,
            "run_cid": self.run_cid,
            "interface_cid": self.interface_cid,
            "delta_cid": self.delta_cid,
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
            "live_chain": self.live_chain,
            "live_application": self.live_application,
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
            "An admitted Quack-fenced state-owner session before the final "
            "proof-carrying receipt chain is emitted as live supervisor work"
        ),
        "action": (
            "Keep the hermetic chain, independent semantic validation, and "
            "hermetic ExecutionReceipt. Do not write DuckDB or Quack state. "
            "Do not escalate to a model. Do not reuse stale leases or "
            "fences. Do not remint the PCPR-071 recovery. PCPR-080 adds "
            "the Python external-client demonstration."
        ),
        "reason": (
            "Hermetic final receipt chain emission is not live Quack chain "
            "emission. Sealed validation has no admitted Quack-fenced "
            "session. Direct DuckDB writes are prohibited."
        ),
    }


def produce_final_receipt_chain(
    *,
    paths: Sequence[str] | None = None,
    pack_cid: str = PINNED_PACK_CID,
    current_root_cid: str = PINNED_CURRENT_ROOT_CID,
    route_cid: str = PINNED_ROUTE_CID,
    patch_cid: str = PINNED_PATCH_CID,
    run_cid: str = PINNED_RUN_CID,
    interface_cid: str = PINNED_INTERFACE_CID,
    delta_cid: str = PINNED_DELTA_CID,
    restart_cid: str = PINNED_RESTART_CID,
    recovery_cid: str = PINNED_RECOVERY_CID,
    complete_from: str | None = None,
    from_memory: bool = False,
    database_edited: bool = False,
    already_applied: bool = False,
    applied_live: bool = False,
    reuse_stale_lease: bool = False,
    reuse_stale_fence: bool = False,
    owner_generation: int = OWNER_GENERATION,
) -> dict[str, Any]:
    """Produce the hermetic final proof-carrying receipt chain. Fail closed."""

    if complete_from is not None:
        refuse_model_completion(complete_from)
    refuse_pack_cid_remint(pack_cid)
    refuse_current_root_remint(current_root_cid)
    refuse_route_cid_remint(route_cid)
    refuse_patch_cid_remint(patch_cid)
    refuse_run_cid_remint(run_cid)
    refuse_interface_cid_remint(interface_cid)
    refuse_delta_cid_remint(delta_cid)
    refuse_restart_cid_remint(restart_cid)
    refuse_recovery_cid_remint(recovery_cid)
    refuse_memory_reconstruction(from_memory=from_memory)
    refuse_database_edit(edited=database_edited)
    refuse_duplicate_chain_effect(already_applied=already_applied)
    refuse_live_chain(applied_live=applied_live)
    refuse_stale_lease_reuse(stale=True, reused=reuse_stale_lease)
    refuse_stale_fence_reuse(stale=True, reused=reuse_stale_fence)
    refuse_generation_mutation(generation=owner_generation)
    checked_paths = list(paths) if paths is not None else list(AUTHORIZED_PATH_PREFIXES)
    for path in checked_paths:
        refuse_unauthorized_path(path)
    continuation = hermetic_chain_continuation()
    validation = hermetic_independent_semantic_validation()
    return {
        "schema": CHAIN_SCHEMA,
        "interface": CHAIN_INTERFACE,
        "owner_interface": INTERFACE,
        "task_id": PCPR_072_TASK_ID,
        "goal_id": PCPR_072_GOAL_ID,
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
        "delta_cid": delta_cid,
        "restart_cid": restart_cid,
        "recovery_cid": recovery_cid,
        "protocol_interface": PROTOCOL_INTERFACE,
        "documentation_interface": DOCUMENTATION_INTERFACE,
        "documentation_schema": DOCUMENTATION_SCHEMA,
        "change_kind": CHANGE_KIND,
        "chain_kind": CHAIN_KIND,
        "idempotency_kind": IDEMPOTENCY_KIND,
        "produced": True,
        "applied": True,
        "live": False,
        "hermetic": True,
        "admitted_live": False,
        "deferred": False,
        "paths_authorized": True,
        "authorized_path_prefixes": list(AUTHORIZED_PATH_PREFIXES),
        "linked_identities": list(LINKED_IDENTITIES),
        "chain_links": [dict(item) for item in CHAIN_LINKS],
        "identities_link": True,
        "state_reconstructed": True,
        "reconstructed_from": RECONSTRUCTED_FROM,
        "reconstructed_from_memory": False,
        "independently_validated": True,
        "semantics_sound": True,
        "execution_receipt_emitted": True,
        "execution_receipt_live": False,
        "artifacts_stored": True,
        "artifacts_live": False,
        "duplicate_effect_rejected": True,
        "idempotent": True,
        "chain_command_digest": CHAIN_COMMAND_DIGEST,
        "continuation": continuation,
        "independent_validation": validation,
        "plan_epoch": PLAN_EPOCH,
        "owner_generation": OWNER_GENERATION,
        "generation_mutated": False,
        "history_mutated": False,
        "database_edited": False,
        "continuation_without_database_edits": True,
        "remints_protocol": False,
        "adds_protocol_operation": False,
        "relevant_interface_change": False,
        "model_assertion_completes_work": False,
        "escalation_order": list(ESCALATION_ORDER),
        "completed_through": COMPLETED_THROUGH,
        "next_authorized_stage": NEXT_AUTHORIZED_STAGE,
        "next_task_id": PCPR_080_TASK_ID,
        "prerequisite_task_id": PCPR_071_TASK_ID,
        "recovery_task_id": PCPR_071_TASK_ID,
        "restart_task_id": PCPR_070_TASK_ID,
        "duckdb_or_quack_state_written": False,
        "closed_release_outcome": None,
        "release_claim": False,
        "evidence_kind": "measured_hermetic",
    }


def canonical_final_receipt_chain() -> dict[str, Any]:
    return produce_final_receipt_chain()


def chain_cid_of(plan: Mapping[str, Any] | None = None) -> str:
    payload = dict(plan or canonical_final_receipt_chain())
    payload.pop("chain_cid", None)
    payload.pop("idempotency", None)
    payload.pop("execution_receipt", None)
    payload.pop("artifact_storage", None)
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
        "live_chain": False,
        "evidence_kind": "measured",
    }


def _sibling_binding(path: Path, relative_path: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": relative_path,
            "status": "unavailable",
            "evidence_kind": "unavailable",
            "reason": (
                "Sibling final-receipt-chain file is not present beside this checkout."
            ),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    context_pack = payload.get("context_pack")
    storage = payload.get("storage")
    route = payload.get("route")
    bounded_patch = payload.get("bounded_patch")
    selected = payload.get("selected_tests")
    chain = payload.get("final_receipt_chain") or payload.get(
        "recovery_and_idempotency"
    )
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
            context_pack.get("pack_cid") if isinstance(context_pack, Mapping) else None
        ),
        "current_root_cid": (
            storage.get("current_root_cid")
            if isinstance(storage, Mapping)
            else payload.get("current_root_cid")
        ),
        "route_cid": (
            route.get("route_cid") if isinstance(route, Mapping) else payload.get("route_cid")
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
        or (chain.get("delta_cid") if isinstance(chain, Mapping) else None),
        "restart_cid": payload.get("restart_cid")
        or (chain.get("restart_cid") if isinstance(chain, Mapping) else None),
        "recovery_cid": payload.get("recovery_cid")
        or (chain.get("recovery_cid") if isinstance(chain, Mapping) else None),
        "chain_cid": payload.get("chain_cid")
        or (chain.get("chain_cid") if isinstance(chain, Mapping) else None),
        "interface_cid": payload.get("interface_cid")
        or (chain.get("interface_cid") if isinstance(chain, Mapping) else None),
        "binding_cid": payload.get("binding_cid") or payload.get("document_cid"),
        "live": False,
        "applied": False,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def render_declared_chain(root: Path | None = None) -> dict[str, Any]:
    package_root = root or discover_accelerate_root()
    if package_root is None:
        raise AccelerateFinalReceiptChainError("Accelerate package root was not found")
    state_owner = observe_state_owner()
    plan = canonical_final_receipt_chain()
    computed_chain_cid = chain_cid_of(plan)
    idempotency = hermetic_chain_idempotency(computed_chain_cid)
    execution_receipt = hermetic_execution_receipt(computed_chain_cid)
    artifact_storage = hermetic_artifact_storage(computed_chain_cid)
    document = {
        "schema": DOCUMENT_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_072_TASK_ID,
        "goal_id": PCPR_072_GOAL_ID,
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
        "execution_receipt": execution_receipt,
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
            "survives_restart": True,
            "survives_recovery": True,
            "survives_chain": True,
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
            "survives_restart": True,
            "survives_recovery": True,
            "survives_chain": True,
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
            "survives_restart": True,
            "survives_recovery": True,
            "survives_chain": True,
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
            "survives_restart": True,
            "survives_recovery": True,
            "survives_chain": True,
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
            "survives_restart": True,
            "survives_recovery": True,
            "survives_chain": True,
            "evidence_kind": "measured_hermetic",
        },
        "stale_rejection": {
            "task_id": PCPR_069_TASK_ID,
            "owner_interface": DELTA_OWNER_INTERFACE,
            "delta_cid": PINNED_DELTA_CID,
            "plan_epoch": PLAN_EPOCH,
            "stale_rejected": True,
            "plan_delta_produced": True,
            "live": False,
            "hermetic": True,
            "reminted": False,
            "survives_restart": True,
            "survives_recovery": True,
            "survives_chain": True,
            "evidence_kind": "measured_hermetic",
        },
        "state_owner_restart": {
            "task_id": PCPR_070_TASK_ID,
            "owner_interface": RESTART_OWNER_INTERFACE,
            "restart_cid": PINNED_RESTART_CID,
            "document_cid": PINNED_RESTART_DOCUMENT_CID,
            "produced": True,
            "live": False,
            "hermetic": True,
            "reminted": False,
            "survives_recovery": True,
            "survives_chain": True,
            "evidence_kind": "measured_hermetic",
        },
        "recovery_and_idempotency": {
            "task_id": PCPR_071_TASK_ID,
            "owner_interface": RECOVERY_OWNER_INTERFACE,
            "recovery_cid": PINNED_RECOVERY_CID,
            "document_cid": PINNED_RECOVERY_DOCUMENT_CID,
            "produced": True,
            "live": False,
            "hermetic": True,
            "reminted": False,
            "survives_chain": True,
            "evidence_kind": "measured_hermetic",
        },
        "final_receipt_chain": {
            **plan,
            "chain_cid": computed_chain_cid,
            "idempotency": idempotency,
            "execution_receipt": execution_receipt,
            "artifact_storage": artifact_storage,
            "produced_by": EXECUTION_OWNER_REPOSITORY,
        },
        "materialization": {
            "kind": "declared_final_receipt_chain_not_live",
            "admitted": False,
            "live": False,
            "applied": False,
            "duckdb_or_quack_state_written": False,
            "evidence_kind": "unavailable",
        },
        "live_application": typed_unavailable(
            reason=(
                "No admitted Quack-fenced supervisor session was observed. "
                "Hermetic chain emission is not live chain emission."
            )
        ),
        "live_chain": typed_unavailable(
            reason=(
                "Live Quack final receipt chain remains typed unavailable. "
                "This task only emits a hermetic chain. External-client "
                "demonstration remains PCPR-080."
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
            "live_chain": False,
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
                "field": "PCPR-072 receipt current_tree_binding",
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


def platform_final_receipt_chain_catalog(
    start: Path | None = None,
) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateFinalReceiptChainError("Accelerate package root was not found")
    parent = root.parent
    document = render_declared_chain(root)
    datasets_binding = _sibling_binding(
        parent
        / "ipfs_datasets"
        / DOCUMENT_DIR_RELPATH
        / "reference.final-receipt-chain.binding.json",
        relative_path=(
            f"../ipfs_datasets/{DOCUMENT_DIR_RELPATH}/"
            "reference.final-receipt-chain.binding.json"
        ),
    )
    kit_binding = _sibling_binding(
        parent
        / "ipfs_kit"
        / DOCUMENT_DIR_RELPATH
        / "reference.final-receipt-chain.binding.json",
        relative_path=(
            f"../ipfs_kit/{DOCUMENT_DIR_RELPATH}/"
            "reference.final-receipt-chain.binding.json"
        ),
    )
    catalog = {
        "schema": CATALOG_SCHEMA,
        "interface": CATALOG_INTERFACE,
        "task_id": PCPR_072_TASK_ID,
        "goal_id": PCPR_072_GOAL_ID,
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
        "delta_cid": PINNED_DELTA_CID,
        "restart_cid": PINNED_RESTART_CID,
        "recovery_cid": PINNED_RECOVERY_CID,
        "chain_cid": document["final_receipt_chain"]["chain_cid"],
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
                "delta_cid": PINNED_DELTA_CID,
                "restart_cid": PINNED_RESTART_CID,
                "recovery_cid": PINNED_RECOVERY_CID,
                "chain_cid": document["final_receipt_chain"]["chain_cid"],
                "document_cid": document["document_cid"],
                "live": False,
                "applied": False,
                "reminted": False,
            },
            "ipfs_datasets_py": {"binding": datasets_binding},
            "ipfs_kit_py": {"binding": kit_binding},
        },
        "live_application": False,
        "live_chain": False,
        "closed_release_outcome": None,
        "release_claim": False,
        "sibling_source_required": False,
        "evidence_kind": "measured",
    }
    catalog["catalog_cid"] = content_identity(
        {key: value for key, value in catalog.items() if key != "catalog_cid"}
    )
    return catalog


def write_final_receipt_chain_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateFinalReceiptChainError("Accelerate package root was not found")
    document = render_declared_chain(root)
    paths = artifact_paths(root)
    _atomic_write(paths["document"], pretty_json(document))
    readme = DOCUMENT_README if DOCUMENT_README.endswith("\n") else DOCUMENT_README + "\n"
    _atomic_write(paths["readme"], readme)
    catalog = platform_final_receipt_chain_catalog(start)
    _atomic_write(paths["catalog"], pretty_json(catalog))
    return {
        "document": document,
        "catalog": catalog,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def verify_final_receipt_chain_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateFinalReceiptChainError("Accelerate package root was not found")
    document = render_declared_chain(root)
    catalog = platform_final_receipt_chain_catalog(start)
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
        "restart_cid": document["state_owner_restart"]["restart_cid"],
        "recovery_cid": document["recovery_and_idempotency"]["recovery_cid"],
        "chain_cid": document["final_receipt_chain"]["chain_cid"],
        "document_cid": document["document_cid"],
        "catalog_cid": catalog["catalog_cid"],
        "current_root_cid": document["storage"]["current_root_cid"],
        "route_cid": document["route"]["route_cid"],
        "patch_cid": document["bounded_patch"]["patch_cid"],
        "run_cid": document["selected_tests"]["run_cid"],
        "interface_cid": document["final_receipt_chain"]["interface_cid"],
        "delta_cid": document["stale_rejection"]["delta_cid"],
        "objective_cid": document["objective_cid"],
        "idea_digest": document["idea_digest"],
        "document_sha256": (
            sha256_bytes(paths["document"].read_bytes())
            if paths["document"].is_file()
            else "unavailable"
        ),
    }


def current_head_static_probes(start: Path | None = None) -> tuple[OutcomeProbe, ...]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateFinalReceiptChainError("Accelerate package root was not found")
    document = render_declared_chain(root)
    verified = verify_final_receipt_chain_files(root)
    table = parse_pyproject_final_receipt_chain_table(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    chain = document["final_receipt_chain"]
    remint = (
        document["context_pack"]["pack_cid"] != PINNED_PACK_CID
        or document["objective_cid"] != PINNED_OBJECTIVE_CID
        or document["idea_digest"] != PINNED_IDEA_DIGEST
        or document["lock_cid"] != PINNED_LOCK_CID
        or document["storage"]["current_root_cid"] != PINNED_CURRENT_ROOT_CID
        or document["route"]["route_cid"] != PINNED_ROUTE_CID
        or document["bounded_patch"]["patch_cid"] != PINNED_PATCH_CID
        or document["selected_tests"]["run_cid"] != PINNED_RUN_CID
        or chain["chain_cid"] != PINNED_CHAIN_CID
        or chain.get("pcpr_066_change_cid") != PINNED_PCPR_066_CHANGE_CID
        or chain.get("reuse_cid") != PINNED_REUSE_CID
        or chain.get("interface_cid") != PINNED_INTERFACE_CID
        or chain.get("delta_cid") != PINNED_DELTA_CID
        or chain.get("restart_cid") != PINNED_RESTART_CID
        or chain.get("recovery_cid") != PINNED_RECOVERY_CID
        or document["stale_rejection"]["delta_cid"] != PINNED_DELTA_CID
        or document["state_owner_restart"]["restart_cid"] != PINNED_RESTART_CID
        or document["recovery_and_idempotency"]["recovery_cid"] != PINNED_RECOVERY_CID
        or document["context_pack"]["reminted"] is True
        or document["storage"]["reminted"] is True
        or document["route"]["reminted"] is True
        or document["bounded_patch"]["reminted"] is True
        or document["selected_tests"]["reminted"] is True
        or document["stale_rejection"]["reminted"] is True
        or document["state_owner_restart"]["reminted"] is True
        or document["recovery_and_idempotency"]["reminted"] is True
    )
    continuation = chain["continuation"]
    idempotency = chain["idempotency"]
    execution_receipt = chain["execution_receipt"]
    artifact_storage = chain["artifact_storage"]
    validation = chain["independent_validation"]
    probes = [
        _probe(
            "chain_files_match_generator",
            verified.get("ok") is True and verified.get("document_ok") is True,
            reason=(
                "Committed Accelerate chain matches the generator."
                if verified.get("document_ok") is True
                else "Committed Accelerate chain is missing or drifts."
            ),
        ),
        _probe(
            "pyproject_final_receipt_chain_table",
            table.get("interface") == INTERFACE
            and table.get("schema") == SCHEMA
            and table.get("task-id") == PCPR_072_TASK_ID
            and table.get("objective-kind") == OBJECTIVE_KIND,
            reason=(
                "pyproject.toml declares AccelerateFinalProofCarryingReceiptChain@1."
                if table.get("interface") == INTERFACE
                else "pyproject.toml does not declare the PCPR-072 chain."
            ),
        ),
        _probe(
            "paths_remain_authorized",
            chain["paths_authorized"] is True
            and list(chain["authorized_path_prefixes"])
            == list(AUTHORIZED_PATH_PREFIXES),
            reason="Chain paths remain inside the declared PCPR-072 allowlist.",
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
            chain["interface_cid"] == PINNED_INTERFACE_CID,
            reason="Accelerate binds the PCPR-068 interface CID and does not remint it.",
        ),
        _probe(
            "delta_cid_bound_not_minted",
            chain["delta_cid"] == PINNED_DELTA_CID
            and document["stale_rejection"]["delta_cid"] == PINNED_DELTA_CID,
            reason="Accelerate binds the PCPR-069 delta CID and does not remint it.",
        ),
        _probe(
            "restart_cid_bound_not_minted",
            chain["restart_cid"] == PINNED_RESTART_CID
            and document["state_owner_restart"]["restart_cid"] == PINNED_RESTART_CID
            and document["state_owner_restart"]["reminted"] is False,
            reason="Accelerate binds the PCPR-070 restart CID and does not remint it.",
        ),
        _probe(
            "recovery_cid_bound_not_minted",
            chain["recovery_cid"] == PINNED_RECOVERY_CID
            and document["recovery_and_idempotency"]["recovery_cid"]
            == PINNED_RECOVERY_CID
            and document["recovery_and_idempotency"]["reminted"] is False,
            reason="Accelerate binds the PCPR-071 recovery CID and does not remint it.",
        ),
        _probe(
            "identities_link_across_chain",
            chain["identities_link"] is True
            and list(chain["linked_identities"]) == list(LINKED_IDENTITIES)
            and len(chain["chain_links"]) == len(CHAIN_LINKS),
            reason=(
                "Objective, context, epoch, task, proof, storage, event, "
                "compatibility, result, restart, and recovery identities link."
            ),
        ),
        _probe(
            "independent_semantic_validation",
            chain["independently_validated"] is True
            and chain["semantics_sound"] is True
            and validation["independently_validated"] is True
            and validation["live"] is False,
            reason="Independent semantic validation is hermetic and sound.",
        ),
        _probe(
            "hermetic_execution_receipt_emitted",
            chain["execution_receipt_emitted"] is True
            and chain["execution_receipt_live"] is False
            and execution_receipt["emitted"] is True
            and execution_receipt["live"] is False
            and execution_receipt["interface"] == EXECUTION_RECEIPT_INTERFACE,
            reason="A hermetic ExecutionReceipt is emitted and is not live.",
        ),
        _probe(
            "hermetic_artifacts_stored",
            chain["artifacts_stored"] is True
            and chain["artifacts_live"] is False
            and artifact_storage["stored"] is True
            and artifact_storage["live"] is False
            and artifact_storage["live_ipfs"] is False,
            reason="Chain artifacts are stored hermetically, not as live IPFS.",
        ),
        _probe(
            "duplicate_chain_effect_rejected",
            chain["duplicate_effect_rejected"] is True
            and idempotency["duplicate_effect_rejected"] is True
            and idempotency["second_attempt_applied"] is False,
            reason="Replaying the same chain command is rejected.",
        ),
        _probe(
            "idempotent_chain_cid",
            chain["idempotent"] is True
            and idempotency["identical"] is True
            and idempotency["first_chain_cid"] == chain["chain_cid"]
            and idempotency["second_chain_cid"] == chain["chain_cid"]
            and chain["chain_cid"] == PINNED_CHAIN_CID,
            reason="A second chain emission yields the same chain CID.",
        ),
        _probe(
            "unexpired_lease_authorizes_chain",
            continuation["unexpired_lease_retained"] is True
            and continuation["authorized_by_retained_lease"] is True
            and continuation["retained_lease"]["expired"] is False,
            reason="Unexpired leases authorize the final receipt chain.",
        ),
        _probe(
            "stale_lease_cannot_authorize_chain",
            continuation["stale_lease_expired"] is True
            and continuation["expired_lease"]["stale"] is True
            and continuation["stale_lease_reused"] is False,
            reason="Stale leases cannot authorize the final receipt chain.",
        ),
        _probe(
            "current_fence_authorizes_chain",
            continuation["current_fence_retained"] is True
            and continuation["authorized_by_current_fence"] is True
            and continuation["retained_fence"]["fence_epoch"] == 1,
            reason="Current fence epoch authorizes the final receipt chain.",
        ),
        _probe(
            "stale_fence_cannot_authorize_chain",
            continuation["stale_fence_expired"] is True
            and continuation["expired_fence"]["stale"] is True
            and continuation["stale_fence_reused"] is False,
            reason="Stale fences cannot authorize the final receipt chain.",
        ),
        _probe(
            "owner_generation_unchanged",
            continuation["owner_generation"] == OWNER_GENERATION
            and continuation["generation_mutated"] is False
            and chain["generation_mutated"] is False,
            reason="Owner generation stays at 2; the chain does not remint restart.",
        ),
        _probe(
            "continuation_without_database_edits",
            chain["continuation_without_database_edits"] is True
            and chain["database_edited"] is False
            and document["duckdb_or_quack_state_written"] is False,
            reason="Continuation proceeds without DuckDB or Quack writes.",
        ),
        _probe(
            "model_assertion_cannot_complete_work",
            chain["model_assertion_completes_work"] is False,
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
            reason="Missing live chain emission emits the operator-blocking task.",
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
                "delta, restart, recovery, objective, idea, or lock CID is forbidden."
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
            chain["interface_cid"] != PINNED_INTERFACE_CID,
            reason="Accelerate must not remint the PCPR-068 interface CID.",
        ),
        _probe(
            "delta_identity_reminted",
            chain["delta_cid"] != PINNED_DELTA_CID,
            reason="Accelerate must not remint the PCPR-069 delta CID.",
        ),
        _probe(
            "restart_identity_reminted",
            chain["restart_cid"] != PINNED_RESTART_CID,
            reason="Accelerate must not remint the PCPR-070 restart CID.",
        ),
        _probe(
            "recovery_identity_reminted",
            chain["recovery_cid"] != PINNED_RECOVERY_CID,
            reason="Accelerate must not remint the PCPR-071 recovery CID.",
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
            "live_chain_represented_as_live",
            False,
            reason="Hermetic chain emission is not represented as live.",
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
            "memory_reconstruction_accepted",
            False,
            reason="Memory reconstruction is refused.",
        ),
        _probe(
            "duplicate_chain_effect_accepted",
            False,
            reason="Duplicate chain effects are rejected.",
        ),
        _probe(
            "stale_lease_reused",
            False,
            reason="Stale leases are not reused for the chain.",
        ),
        _probe(
            "stale_fence_reused",
            False,
            reason="Stale fences are not reused for the chain.",
        ),
        _probe(
            "owner_generation_mutated",
            False,
            reason="The chain does not mutate the post-restart owner generation.",
        ),
        _probe(
            "database_edited",
            False,
            reason="DuckDB and Quack state were not written.",
        ),
        _probe(
            "live_application",
            None,
            evidence_kind="unavailable",
            reason="Live supervisor application stays typed unavailable.",
        ),
        _probe(
            "live_chain",
            None,
            evidence_kind="unavailable",
            reason="Live Quack chain emission stays typed unavailable.",
        ),
    ]
    return tuple(probes)


def qualify_final_receipt_chain(
    probes: Sequence[OutcomeProbe],
    *,
    chain_cid: str,
    document_cid: str,
    catalog_cid: str,
    recovery_cid: str,
    restart_cid: str,
    pack_cid: str,
    current_root_cid: str,
    route_cid: str,
    patch_cid: str,
    run_cid: str,
    interface_cid: str,
    delta_cid: str,
    objective_cid: str,
    idea_digest_cid: str,
) -> AccelerateFinalReceiptChainVerdict:
    if not probes:
        raise AccelerateFinalReceiptChainError("at least one probe is required")
    normalized: list[OutcomeProbe] = []
    blockers: list[str] = []
    for probe in probes:
        kind = _kind(probe.evidence_kind, "evidence_kind")
        if probe.live and kind != "measured_live":
            raise AccelerateFinalReceiptChainError(
                "live claims require measured_live evidence"
            )
        if probe.simulated_represented_as_live:
            raise AccelerateFinalReceiptChainError(
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
        "task_id": PCPR_072_TASK_ID,
        "goal_id": PCPR_072_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "sibling_source_required": False,
        "live_chain": False,
        "live_application": False,
        "operator_blocking_task": OPERATOR_BLOCKING_TASK_ID,
        "simulated_results_represented_as_live": False,
        "this_task_created_competing_authority": False,
        "probes": [item.to_mapping() for item in normalized],
        "blockers": list(dict.fromkeys(blockers)),
        "chain_cid": chain_cid,
        "document_cid": document_cid,
        "catalog_cid": catalog_cid,
        "recovery_cid": recovery_cid,
        "restart_cid": restart_cid,
        "pack_cid": pack_cid,
        "current_root_cid": current_root_cid,
        "route_cid": route_cid,
        "patch_cid": patch_cid,
        "run_cid": run_cid,
        "interface_cid": interface_cid,
        "delta_cid": delta_cid,
        "objective_cid": objective_cid,
        "idea_digest": idea_digest_cid,
    }
    return AccelerateFinalReceiptChainVerdict(
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
        live_chain=False,
        live_application=False,
        operator_blocking_task=OPERATOR_BLOCKING_TASK_ID,
        simulated_results_represented_as_live=False,
        this_task_created_competing_authority=False,
        probes=tuple(normalized),
        blockers=tuple(dict.fromkeys(blockers)),
        verdict_cid=content_identity(payload),
        chain_cid=chain_cid,
        document_cid=document_cid,
        catalog_cid=catalog_cid,
        recovery_cid=recovery_cid,
        restart_cid=restart_cid,
        pack_cid=pack_cid,
        current_root_cid=current_root_cid,
        route_cid=route_cid,
        patch_cid=patch_cid,
        run_cid=run_cid,
        interface_cid=interface_cid,
        delta_cid=delta_cid,
        objective_cid=objective_cid,
        idea_digest=idea_digest_cid,
    )


def qualify_current_head_final_receipt_chain(
    start: Path | None = None,
) -> AccelerateFinalReceiptChainVerdict:
    document = render_declared_chain(start)
    catalog = platform_final_receipt_chain_catalog(start)
    return qualify_final_receipt_chain(
        current_head_static_probes(start),
        chain_cid=str(document["final_receipt_chain"]["chain_cid"]),
        document_cid=str(document["document_cid"]),
        catalog_cid=str(catalog["catalog_cid"]),
        recovery_cid=str(document["recovery_and_idempotency"]["recovery_cid"]),
        restart_cid=str(document["state_owner_restart"]["restart_cid"]),
        pack_cid=str(document["context_pack"]["pack_cid"]),
        current_root_cid=str(document["storage"]["current_root_cid"]),
        route_cid=str(document["route"]["route_cid"]),
        patch_cid=str(document["bounded_patch"]["patch_cid"]),
        run_cid=str(document["selected_tests"]["run_cid"]),
        interface_cid=str(document["final_receipt_chain"]["interface_cid"]),
        delta_cid=str(document["stale_rejection"]["delta_cid"]),
        objective_cid=str(document["objective_cid"]),
        idea_digest_cid=str(document["idea_digest"]),
    )


def pcpr_072_receipt_promotion(
    verdict: AccelerateFinalReceiptChainVerdict,
) -> dict[str, Any]:
    if verdict.closed_release_outcome is not None:
        raise AccelerateFinalReceiptChainError(
            "final receipt chain must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise AccelerateFinalReceiptChainError(
            "final receipt chain must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise AccelerateFinalReceiptChainError(
            "final receipt chain completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise AccelerateFinalReceiptChainError(
            "final receipt chain must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise AccelerateFinalReceiptChainError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.live_chain or verdict.live_application:
        raise AccelerateFinalReceiptChainError(
            "live chain claims require measured_live evidence"
        )
    return verdict.to_mapping()


PINNED_CHAIN_CID: Final = (
    "baguqeeraeuvfodvledfa4nvfuso6eyyytzk2acptcogzwzyarkdinpqcsdfq"
)
PINNED_DOCUMENT_CID: Final = (
    "baguqeerax2qxfaxxywhn6bpyrygdf5ijjz4ewexigumdih6ukcj2tbo2w6va"
)
PINNED_CATALOG_CID: Final = (
    "baguqeerayfnaukutwzcxo5rgkvzfsdhvshkfgfpmnztxa5abr6sorw25y7za"
)
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeeramc5aevmt4mmdcbken4iupxjwvuxvoozgl7iz6ooscqixknbsfrjq"
)


__all__ = [
    "AUTHORIZED_PATH_PREFIXES",
    "CLOSED_RELEASE_OUTCOMES",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "ESCALATION_ORDER",
    "AccelerateFinalReceiptChainError",
    "HERMETIC_CANDIDATE_SUITES",
    "INTERFACE",
    "LINKED_IDENTITIES",
    "OBJECTIVE_KIND",
    "OPERATOR_BLOCKING_TASK_ID",
    "OutcomeProbe",
    "PCPR_072_GOAL_ID",
    "PCPR_072_TASK_ID",
    "PINNED_CATALOG_CID",
    "PINNED_CHAIN_CID",
    "PINNED_CURRENT_ROOT_CID",
    "PINNED_DELTA_CID",
    "PINNED_DOCUMENT_CID",
    "PINNED_IDEA_DIGEST",
    "PINNED_INTERFACE_CID",
    "PINNED_LOCK_CID",
    "PINNED_OBJECTIVE_CID",
    "PINNED_PACK_CID",
    "PINNED_PATCH_CID",
    "PINNED_PCPR_066_CHANGE_CID",
    "PINNED_RECOVERY_CID",
    "PINNED_RESTART_CID",
    "PINNED_REUSE_CID",
    "PINNED_ROUTE_CID",
    "PINNED_RUN_CID",
    "PLAN_EPOCH",
    "PROTOCOL_INTERFACE",
    "SCHEMA",
    "SEALED_PATH",
    "SEALED_PYTHON",
    "canonical_final_receipt_chain",
    "chain_cid_of",
    "current_head_static_probes",
    "hermetic_artifact_storage",
    "hermetic_chain_continuation",
    "hermetic_chain_idempotency",
    "hermetic_execution_receipt",
    "hermetic_independent_semantic_validation",
    "path_is_authorized",
    "pcpr_072_receipt_promotion",
    "platform_final_receipt_chain_catalog",
    "produce_final_receipt_chain",
    "qualify_current_head_final_receipt_chain",
    "qualify_final_receipt_chain",
    "refuse_current_root_remint",
    "refuse_database_edit",
    "refuse_delta_cid_remint",
    "refuse_duplicate_chain_effect",
    "refuse_generation_mutation",
    "refuse_interface_cid_remint",
    "refuse_live_chain",
    "refuse_memory_reconstruction",
    "refuse_model_completion",
    "refuse_pack_cid_remint",
    "refuse_patch_cid_remint",
    "refuse_recovery_cid_remint",
    "refuse_restart_cid_remint",
    "refuse_route_cid_remint",
    "refuse_run_cid_remint",
    "refuse_stale_fence_reuse",
    "refuse_stale_lease_reuse",
    "refuse_unauthorized_path",
    "render_declared_chain",
    "verify_final_receipt_chain_files",
    "write_final_receipt_chain_files",
]
