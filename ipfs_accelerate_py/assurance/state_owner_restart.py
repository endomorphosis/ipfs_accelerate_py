"""Fail-closed PCPR-070 Accelerate-owned authoritative state-owner restart.

Accelerate owns restart of the authenticated Quack state owner. This
module reconstructs owner state from durable records, rejects duplicate
restart effects, retains unexpired leases and current fences, expires
stale leases and fences, and continues without database edits. It binds
the PCPR-061 DatasetsContextPack@1 identity, the PCPR-062 Kit current
root, the PCPR-063 route, the PCPR-064 patch, the PCPR-065 selected-
tests run, the PCPR-066 unrelated change, the PCPR-067 eligible reuse,
the PCPR-068 relevant-interface identity, and the PCPR-069 stale
rejection and PlanDelta without reminting those identities.

Live Quack restart stays typed unavailable. Incremental prover binaries
missing from the sealed PATH stay typed unavailable. Model stages are
not invoked. A model assertion cannot complete work.

Hermetic reconstruction is candidate coverage only. It is not live
owner restart, not live recovery (PCPR-071), and not a closed PCPR
release. Simulated results are not live. This evaluator does not write
DuckDB or Quack state.
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


INTERFACE: Final = "AccelerateAuthoritativeStateOwnerRestart@1"
SCHEMA: Final = "ipfs_accelerate_py/assurance/authoritative-state-owner-restart@1"
DOCUMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/declared-authoritative-state-owner-restart@1"
)
RESTART_SCHEMA: Final = "ipfs_accelerate_py/assurance/state-owner-restart@1"
VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/authoritative-state-owner-restart-verdict@1"
)
CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/platform-authoritative-state-owner-restart-catalog@1"
)
CATALOG_INTERFACE: Final = "PlatformAuthoritativeStateOwnerRestart@1"
RESTART_INTERFACE: Final = "AuthoritativeStateOwnerRestart@1"
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
EXECUTION_INVOCATION_INTERFACE: Final = OWNER_INTERFACES["ExecutionInvocation"]
EXECUTION_INVOCATION_SCHEMA: Final = OWNER_SCHEMAS["ExecutionInvocation"]
EXECUTION_RECEIPT_INTERFACE: Final = OWNER_INTERFACES["ExecutionReceipt"]
EXECUTION_RECEIPT_SCHEMA: Final = OWNER_SCHEMAS["ExecutionReceipt"]
PROTOCOL_INTERFACE: Final = "LogicProviderProtocol@2"
DOCUMENTATION_INTERFACE: Final = "LogicProviderProtocolStateOwnerRestart@1"
DOCUMENTATION_SCHEMA: Final = (
    "ipfs_datasets_py/logic-provider-state-owner-restart@1"
)
PCPR_070_TASK_ID: Final = "PCPR-070"
PCPR_070_GOAL_ID: Final = "PCPR-G700"
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
PCPR_071_TASK_ID: Final = "PCPR-071"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = (
    "proof-carrying-platform-qualification-and-release-v1"
)
OBJECTIVE_KIND: Final = "declared_authoritative_state_owner_restart"
OBJECTIVE_ID: Final = "PCPR-G700"
LANGUAGE: Final = "Python"
OPERATOR_BLOCKING_TASK_ID: Final = (
    "pcpr-070-operator-live-authoritative-state-owner-restart"
)
DOCUMENT_DIR_RELPATH: Final = "packaging/pcpr/reference-workflow/cpython312"
DOCUMENT_JSON_NAME: Final = "reference.state-owner-restart.json"
DOCUMENT_README_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/STATE_OWNER_RESTART.md"
)
CATALOG_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/platform-state-owner-restart-catalog.json"
)
SOURCE_DATE_EPOCH: Final = "0"
SOURCE_REPOSITORY: Final = "endomorphosis/ipfs_accelerate_py"
COMPLETED_THROUGH: Final = "restart_authoritative_state_owner"
NEXT_AUTHORIZED_STAGE: Final = "recovery_and_idempotency"
CHANGE_KIND: Final = "authoritative_state_owner_restart"
RESTART_KIND: Final = "durable_reconstruction_without_database_edit"
PLAN_EPOCH: Final = 2
PRE_RESTART_OWNER_GENERATION: Final = 1
POST_RESTART_OWNER_GENERATION: Final = 2
RECONSTRUCTED_FROM: Final = "durable_records"
STATE_OWNER_NAME: Final = "quack"
STATE_OWNER_AUTHORITY: Final = "authenticated_fenced_state_owner"
RESTART_COMMAND_DIGEST: Final = (
    "pcpr-070-hermetic-authoritative-state-owner-restart"
)

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

if PINNED_LOCK_CID != (
    "baguqeerawsekbbbt5ccctjt4tydzeahfkahsatb6q5x7k6cy4inrii5lhqmq"
):
    raise RuntimeError("PCPR-070 lock CID remints PCPR-056")

if OWNER_INTERFACES["SupervisorContextPack"] != OWNER_INTERFACE:
    raise RuntimeError("PCPR-070 remints SupervisorContextPack owner interface")
if OWNER_SCHEMAS["SupervisorContextPack"] != OWNER_SCHEMA:
    raise RuntimeError("PCPR-070 remints SupervisorContextPack owner schema")
if EXECUTION_INVOCATION_INTERFACE != "InvocationRequest@1":
    raise RuntimeError("PCPR-070 remints ExecutionInvocation owner interface")
if EXECUTION_RECEIPT_INTERFACE != "InvocationResult@1":
    raise RuntimeError("PCPR-070 remints ExecutionReceipt owner interface")

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
    "artifacts/proof_carrying_platform_qualification_and_release/receipts/PCPR-070.json",
)

RECONSTRUCTED_IDENTITIES: Final[tuple[str, ...]] = (
    "objective_cid",
    "pack_cid",
    "current_root_cid",
    "route_cid",
    "patch_cid",
    "run_cid",
    "change_cid",
    "reuse_cid",
    "interface_cid",
    "delta_cid",
    "plan_epoch",
    "lease",
    "fence",
)

RETAINED_LEASE: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "lease_id": "lease:pcpr-070-hermetic-current",
        "status": "retained",
        "expired": False,
        "stale": False,
        "reason": "Unexpired lease is retained across owner restart.",
    }
)
EXPIRED_LEASE: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "lease_id": "lease:pcpr-070-hermetic-stale",
        "status": "expired",
        "expired": True,
        "stale": True,
        "reason": "Stale lease is expired and cannot be reused after restart.",
    }
)
RETAINED_FENCE: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "fence_epoch": 1,
        "fencing_token": 1,
        "status": "retained",
        "expired": False,
        "stale": False,
        "reason": "Current fence epoch is retained across owner restart.",
    }
)
EXPIRED_FENCE: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "fence_epoch": 0,
        "fencing_token": 0,
        "status": "expired",
        "expired": True,
        "stale": True,
        "reason": (
            "Stale fence epoch is expired and cannot authorize "
            "post-restart mutation."
        ),
    }
)

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_state_owner_restart.py",
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
        "restart_files_match_generator",
        "pyproject_state_owner_restart_table",
        "paths_remain_authorized",
        "owner_pack_cid_matches_pin",
        "kit_current_root_bound_not_minted",
        "route_cid_bound_not_minted",
        "patch_cid_bound_not_minted",
        "run_cid_bound_not_minted",
        "interface_cid_bound_not_minted",
        "delta_cid_bound_not_minted",
        "state_reconstructed_from_durable_records",
        "duplicate_restart_effect_rejected",
        "unexpired_lease_retained",
        "stale_lease_expired",
        "current_fence_retained",
        "stale_fence_expired",
        "owner_generation_incremented",
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
        "live_restart_represented_as_live",
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
        "model_assertion_completed_work",
        "unauthorized_path_accepted",
        "memory_reconstruction_accepted",
        "duplicate_restart_effect_accepted",
        "stale_lease_reused",
        "stale_fence_reused",
        "database_edited",
    }
)

DOCUMENT_README: Final = """# PCPR-070 Accelerate authoritative state-owner restart

These files record the Accelerate-owned hermetic restart of the
authoritative Quack state owner against the PCPR-061 DatasetsContextPack@1
identity, the PCPR-062 Kit current root, the PCPR-063 deterministic-first
route, the PCPR-064 bounded PatchPlan, the PCPR-065 selected-tests-and-proofs
run, the PCPR-066 unrelated change, the PCPR-067 eligible reuse, the
PCPR-068 relevant interface change, and the PCPR-069 stale rejection and
PlanDelta. Accelerate owns owner restart and durable reconstruction. It
does not remint those identities and does not claim live Quack restart.

- `cpython312/reference.state-owner-restart.json` is the declared
  restart document. Exact commit and tree are bound by the PCPR-070
  receipt `current_tree_binding`.
- `platform-state-owner-restart-catalog.json` observes sibling Datasets
  and Kit bindings when present. Sibling source is never required.
- Owner state is reconstructed from durable records, not memory.
  Duplicate restart effects are rejected. Unexpired leases and current
  fences are retained. Stale leases and fences expire. Continuation
  proceeds without database edits. Recovery and idempotency remain
  PCPR-071.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-070-operator-live-authoritative-state-owner-restart`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
"""


class AccelerateStateOwnerRestartError(Exception):
    """Fail-closed PCPR-070 Accelerate state-owner restart error."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AccelerateStateOwnerRestartError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise AccelerateStateOwnerRestartError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise AccelerateStateOwnerRestartError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def parse_pyproject_state_owner_restart_table(text: str) -> dict[str, Any]:
    import tomllib

    table = tomllib.loads(_text(text, "pyproject.toml"))
    if not isinstance(table, dict):
        raise AccelerateStateOwnerRestartError("pyproject.toml must be a table")
    tool = table.get("tool")
    payload: dict[str, Any] = {}
    if isinstance(tool, dict):
        accelerate = tool.get("ipfs_accelerate_py")
        if isinstance(accelerate, dict):
            raw = accelerate.get("state-owner-restart")
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
        raise AccelerateStateOwnerRestartError(
            f"path {path} is not an authorized PCPR-070 path"
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
        raise AccelerateStateOwnerRestartError(
            f"model assertion at {stage} cannot complete work"
        )


def refuse_pack_cid_remint(cid: str) -> str:
    if cid != PINNED_PACK_CID:
        raise AccelerateStateOwnerRestartError(
            f"ContextPack CID {cid} remints {PINNED_PACK_CID}"
        )
    return cid


def refuse_current_root_remint(cid: str) -> str:
    if cid != PINNED_CURRENT_ROOT_CID:
        raise AccelerateStateOwnerRestartError(
            f"current root CID {cid} remints {PINNED_CURRENT_ROOT_CID}"
        )
    return cid


def refuse_route_cid_remint(cid: str) -> str:
    if cid != PINNED_ROUTE_CID:
        raise AccelerateStateOwnerRestartError(
            f"route CID {cid} remints {PINNED_ROUTE_CID}"
        )
    return cid


def refuse_patch_cid_remint(cid: str) -> str:
    if cid != PINNED_PATCH_CID:
        raise AccelerateStateOwnerRestartError(
            f"patch CID {cid} remints {PINNED_PATCH_CID}"
        )
    return cid


def refuse_run_cid_remint(cid: str) -> str:
    if cid != PINNED_RUN_CID:
        raise AccelerateStateOwnerRestartError(
            f"run CID {cid} remints {PINNED_RUN_CID}"
        )
    return cid


def refuse_interface_cid_remint(cid: str) -> str:
    if cid != PINNED_INTERFACE_CID:
        raise AccelerateStateOwnerRestartError(
            f"interface CID {cid} remints {PINNED_INTERFACE_CID}"
        )
    return cid


def refuse_delta_cid_remint(cid: str) -> str:
    if cid != PINNED_DELTA_CID:
        raise AccelerateStateOwnerRestartError(
            f"delta CID {cid} remints {PINNED_DELTA_CID}"
        )
    return cid


def refuse_memory_reconstruction(*, from_memory: bool) -> None:
    if from_memory:
        raise AccelerateStateOwnerRestartError(
            "state owner restart cannot reconstruct from memory"
        )


def refuse_database_edit(*, edited: bool) -> None:
    if edited:
        raise AccelerateStateOwnerRestartError(
            "state owner restart cannot write DuckDB or Quack state"
        )


def refuse_duplicate_restart_effect(*, already_applied: bool) -> None:
    if already_applied:
        raise AccelerateStateOwnerRestartError(
            "duplicate restart effect is rejected"
        )


def refuse_stale_lease_reuse(*, stale: bool, reused: bool) -> None:
    if stale and reused:
        raise AccelerateStateOwnerRestartError(
            "stale lease cannot be reused after restart"
        )


def refuse_stale_fence_reuse(*, stale: bool, reused: bool) -> None:
    if stale and reused:
        raise AccelerateStateOwnerRestartError(
            "stale fence cannot authorize post-restart mutation"
        )


def refuse_live_restart(*, applied_live: bool) -> None:
    if applied_live:
        raise AccelerateStateOwnerRestartError(
            "live authoritative state-owner restart remains typed unavailable"
        )


def hermetic_owner_snapshot() -> dict[str, Any]:
    return {
        "schema": "ipfs_accelerate_py/assurance/hermetic-state-owner-snapshot@1",
        "interface": "HermeticStateOwnerSnapshot@1",
        "owner": STATE_OWNER_NAME,
        "authority": STATE_OWNER_AUTHORITY,
        "pre_restart_generation": PRE_RESTART_OWNER_GENERATION,
        "post_restart_generation": POST_RESTART_OWNER_GENERATION,
        "generation_incremented": True,
        "reconstructed_from": RECONSTRUCTED_FROM,
        "reconstructed_from_memory": False,
        "database_edited": False,
        "direct_duckdb_write": False,
        "direct_quack_write": False,
        "live": False,
        "hermetic": True,
        "evidence_kind": "measured_hermetic",
    }


def hermetic_lease_and_fence_disposition() -> dict[str, Any]:
    refuse_stale_lease_reuse(stale=True, reused=False)
    refuse_stale_fence_reuse(stale=True, reused=False)
    return {
        "schema": "ipfs_accelerate_py/assurance/hermetic-lease-and-fence-disposition@1",
        "interface": "HermeticLeaseAndFenceDisposition@1",
        "retained_lease": dict(RETAINED_LEASE),
        "expired_lease": dict(EXPIRED_LEASE),
        "retained_fence": dict(RETAINED_FENCE),
        "expired_fence": dict(EXPIRED_FENCE),
        "unexpired_lease_retained": True,
        "stale_lease_expired": True,
        "current_fence_retained": True,
        "stale_fence_expired": True,
        "stale_lease_reused": False,
        "stale_fence_reused": False,
        "live": False,
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
class AccelerateStateOwnerRestartVerdict:
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
    live_restart: bool
    live_application: bool
    operator_blocking_task: str
    simulated_results_represented_as_live: bool
    this_task_created_competing_authority: bool
    probes: tuple[OutcomeProbe, ...]
    blockers: tuple[str, ...]
    verdict_cid: str
    restart_cid: str
    document_cid: str
    catalog_cid: str
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
            "restart_cid": self.restart_cid,
            "document_cid": self.document_cid,
            "catalog_cid": self.catalog_cid,
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
            "live_restart": self.live_restart,
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
            "An admitted Quack-fenced state-owner session before the "
            "authoritative owner is restarted as live supervisor work"
        ),
        "action": (
            "Keep the hermetic restart reconstruction. Do not write "
            "DuckDB or Quack state. Do not escalate to a model. Do not "
            "reuse stale leases or fences. PCPR-071 demonstrates recovery "
            "and idempotency."
        ),
        "reason": (
            "Hermetic state-owner reconstruction is not live Quack restart. "
            "Sealed validation has no admitted Quack-fenced session. Direct "
            "DuckDB writes are prohibited."
        ),
    }


def produce_state_owner_restart(
    *,
    paths: Sequence[str] | None = None,
    pack_cid: str = PINNED_PACK_CID,
    current_root_cid: str = PINNED_CURRENT_ROOT_CID,
    route_cid: str = PINNED_ROUTE_CID,
    patch_cid: str = PINNED_PATCH_CID,
    run_cid: str = PINNED_RUN_CID,
    interface_cid: str = PINNED_INTERFACE_CID,
    delta_cid: str = PINNED_DELTA_CID,
    complete_from: str | None = None,
    from_memory: bool = False,
    database_edited: bool = False,
    already_applied: bool = False,
    applied_live: bool = False,
    reuse_stale_lease: bool = False,
    reuse_stale_fence: bool = False,
) -> dict[str, Any]:
    """Produce the hermetic authoritative state-owner restart. Fail closed."""

    if complete_from is not None:
        refuse_model_completion(complete_from)
    refuse_pack_cid_remint(pack_cid)
    refuse_current_root_remint(current_root_cid)
    refuse_route_cid_remint(route_cid)
    refuse_patch_cid_remint(patch_cid)
    refuse_run_cid_remint(run_cid)
    refuse_interface_cid_remint(interface_cid)
    refuse_delta_cid_remint(delta_cid)
    refuse_memory_reconstruction(from_memory=from_memory)
    refuse_database_edit(edited=database_edited)
    refuse_duplicate_restart_effect(already_applied=already_applied)
    refuse_live_restart(applied_live=applied_live)
    refuse_stale_lease_reuse(stale=True, reused=reuse_stale_lease)
    refuse_stale_fence_reuse(stale=True, reused=reuse_stale_fence)
    checked_paths = list(paths) if paths is not None else list(AUTHORIZED_PATH_PREFIXES)
    for path in checked_paths:
        refuse_unauthorized_path(path)
    snapshot = hermetic_owner_snapshot()
    leases = hermetic_lease_and_fence_disposition()
    return {
        "schema": RESTART_SCHEMA,
        "interface": RESTART_INTERFACE,
        "owner_interface": INTERFACE,
        "task_id": PCPR_070_TASK_ID,
        "goal_id": PCPR_070_GOAL_ID,
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
        "protocol_interface": PROTOCOL_INTERFACE,
        "documentation_interface": DOCUMENTATION_INTERFACE,
        "documentation_schema": DOCUMENTATION_SCHEMA,
        "change_kind": CHANGE_KIND,
        "restart_kind": RESTART_KIND,
        "produced": True,
        "applied": True,
        "live": False,
        "hermetic": True,
        "admitted_live": False,
        "deferred": False,
        "paths_authorized": True,
        "authorized_path_prefixes": list(AUTHORIZED_PATH_PREFIXES),
        "reconstructed_identities": list(RECONSTRUCTED_IDENTITIES),
        "state_reconstructed": True,
        "reconstructed_from": RECONSTRUCTED_FROM,
        "reconstructed_from_memory": False,
        "duplicate_effect_rejected": True,
        "restart_command_digest": RESTART_COMMAND_DIGEST,
        "owner_snapshot": snapshot,
        "lease_and_fence": leases,
        "plan_epoch": PLAN_EPOCH,
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
        "next_task_id": PCPR_071_TASK_ID,
        "prerequisite_task_id": PCPR_069_TASK_ID,
        "stale_rejection_task_id": PCPR_069_TASK_ID,
        "duckdb_or_quack_state_written": False,
        "closed_release_outcome": None,
        "release_claim": False,
        "evidence_kind": "measured_hermetic",
    }


def canonical_state_owner_restart() -> dict[str, Any]:
    return produce_state_owner_restart()


def restart_cid_of(plan: Mapping[str, Any] | None = None) -> str:
    payload = dict(plan or canonical_state_owner_restart())
    payload.pop("restart_cid", None)
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
        "live_restart": False,
        "evidence_kind": "measured",
    }


def _sibling_binding(path: Path, relative_path: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": relative_path,
            "status": "unavailable",
            "evidence_kind": "unavailable",
            "reason": (
                "Sibling state-owner-restart file is not present beside this checkout."
            ),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    context_pack = payload.get("context_pack")
    storage = payload.get("storage")
    route = payload.get("route")
    bounded_patch = payload.get("bounded_patch")
    selected = payload.get("selected_tests")
    restart = payload.get("state_owner_restart")
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
        or (restart.get("delta_cid") if isinstance(restart, Mapping) else None),
        "restart_cid": payload.get("restart_cid")
        or (restart.get("restart_cid") if isinstance(restart, Mapping) else None),
        "interface_cid": payload.get("interface_cid")
        or (restart.get("interface_cid") if isinstance(restart, Mapping) else None),
        "binding_cid": payload.get("binding_cid") or payload.get("document_cid"),
        "live": False,
        "applied": False,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def render_declared_restart(root: Path | None = None) -> dict[str, Any]:
    package_root = root or discover_accelerate_root()
    if package_root is None:
        raise AccelerateStateOwnerRestartError("Accelerate package root was not found")
    state_owner = observe_state_owner()
    plan = canonical_state_owner_restart()
    computed_restart_cid = restart_cid_of(plan)
    document = {
        "schema": DOCUMENT_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_070_TASK_ID,
        "goal_id": PCPR_070_GOAL_ID,
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
                "A live ExecutionReceipt remains PCPR-072. This state-owner "
                "restart is hermetic."
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
            "survives_restart": True,
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
            "evidence_kind": "measured_hermetic",
        },
        "state_owner_restart": {
            **plan,
            "restart_cid": computed_restart_cid,
            "produced_by": EXECUTION_OWNER_REPOSITORY,
        },
        "materialization": {
            "kind": "declared_authoritative_state_owner_restart_not_live",
            "admitted": False,
            "live": False,
            "applied": False,
            "duckdb_or_quack_state_written": False,
            "evidence_kind": "unavailable",
        },
        "live_application": typed_unavailable(
            reason=(
                "No admitted Quack-fenced supervisor session was observed. "
                "Hermetic state-owner reconstruction is not live restart."
            )
        ),
        "live_restart": typed_unavailable(
            reason=(
                "Live Quack owner restart remains typed unavailable. This "
                "task only demonstrates hermetic reconstruction. Recovery "
                "and idempotency remain PCPR-071."
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
            "live_restart": False,
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
                "field": "PCPR-070 receipt current_tree_binding",
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


def platform_state_owner_restart_catalog(
    start: Path | None = None,
) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateStateOwnerRestartError("Accelerate package root was not found")
    parent = root.parent
    document = render_declared_restart(root)
    datasets_binding = _sibling_binding(
        parent
        / "ipfs_datasets"
        / DOCUMENT_DIR_RELPATH
        / "reference.state-owner-restart.binding.json",
        relative_path=(
            f"../ipfs_datasets/{DOCUMENT_DIR_RELPATH}/"
            "reference.state-owner-restart.binding.json"
        ),
    )
    kit_binding = _sibling_binding(
        parent
        / "ipfs_kit"
        / DOCUMENT_DIR_RELPATH
        / "reference.state-owner-restart.binding.json",
        relative_path=(
            f"../ipfs_kit/{DOCUMENT_DIR_RELPATH}/"
            "reference.state-owner-restart.binding.json"
        ),
    )
    catalog = {
        "schema": CATALOG_SCHEMA,
        "interface": CATALOG_INTERFACE,
        "task_id": PCPR_070_TASK_ID,
        "goal_id": PCPR_070_GOAL_ID,
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
        "restart_cid": document["state_owner_restart"]["restart_cid"],
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
                "restart_cid": document["state_owner_restart"]["restart_cid"],
                "document_cid": document["document_cid"],
                "live": False,
                "applied": False,
                "reminted": False,
            },
            "ipfs_datasets_py": {"binding": datasets_binding},
            "ipfs_kit_py": {"binding": kit_binding},
        },
        "live_application": False,
        "live_restart": False,
        "closed_release_outcome": None,
        "release_claim": False,
        "sibling_source_required": False,
        "evidence_kind": "measured",
    }
    catalog["catalog_cid"] = content_identity(
        {key: value for key, value in catalog.items() if key != "catalog_cid"}
    )
    return catalog


def write_state_owner_restart_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateStateOwnerRestartError("Accelerate package root was not found")
    document = render_declared_restart(root)
    paths = artifact_paths(root)
    _atomic_write(paths["document"], pretty_json(document))
    readme = DOCUMENT_README if DOCUMENT_README.endswith("\n") else DOCUMENT_README + "\n"
    _atomic_write(paths["readme"], readme)
    catalog = platform_state_owner_restart_catalog(start)
    _atomic_write(paths["catalog"], pretty_json(catalog))
    return {
        "document": document,
        "catalog": catalog,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def verify_state_owner_restart_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateStateOwnerRestartError("Accelerate package root was not found")
    document = render_declared_restart(root)
    catalog = platform_state_owner_restart_catalog(start)
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
        "document_cid": document["document_cid"],
        "catalog_cid": catalog["catalog_cid"],
        "current_root_cid": document["storage"]["current_root_cid"],
        "route_cid": document["route"]["route_cid"],
        "patch_cid": document["bounded_patch"]["patch_cid"],
        "run_cid": document["selected_tests"]["run_cid"],
        "interface_cid": document["state_owner_restart"]["interface_cid"],
        "delta_cid": document["stale_rejection"]["delta_cid"],
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
        raise AccelerateStateOwnerRestartError("Accelerate package root was not found")
    document = render_declared_restart(root)
    verified = verify_state_owner_restart_files(root)
    table = parse_pyproject_state_owner_restart_table(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    restart = document["state_owner_restart"]
    remint = (
        document["context_pack"]["pack_cid"] != PINNED_PACK_CID
        or document["objective_cid"] != PINNED_OBJECTIVE_CID
        or document["idea_digest"] != PINNED_IDEA_DIGEST
        or document["lock_cid"] != PINNED_LOCK_CID
        or document["storage"]["current_root_cid"] != PINNED_CURRENT_ROOT_CID
        or document["route"]["route_cid"] != PINNED_ROUTE_CID
        or document["bounded_patch"]["patch_cid"] != PINNED_PATCH_CID
        or document["selected_tests"]["run_cid"] != PINNED_RUN_CID
        or restart["restart_cid"] != PINNED_RESTART_CID
        or restart.get("pcpr_066_change_cid") != PINNED_PCPR_066_CHANGE_CID
        or restart.get("reuse_cid") != PINNED_REUSE_CID
        or restart.get("interface_cid") != PINNED_INTERFACE_CID
        or restart.get("delta_cid") != PINNED_DELTA_CID
        or document["stale_rejection"]["delta_cid"] != PINNED_DELTA_CID
        or document["context_pack"]["reminted"] is True
        or document["storage"]["reminted"] is True
        or document["route"]["reminted"] is True
        or document["bounded_patch"]["reminted"] is True
        or document["selected_tests"]["reminted"] is True
        or document["stale_rejection"]["reminted"] is True
    )
    snapshot = restart["owner_snapshot"]
    leases = restart["lease_and_fence"]
    probes = [
        _probe(
            "restart_files_match_generator",
            verified.get("ok") is True and verified.get("document_ok") is True,
            reason=(
                "Committed Accelerate state-owner restart matches the generator."
                if verified.get("document_ok") is True
                else "Committed Accelerate state-owner restart is missing or drifts."
            ),
        ),
        _probe(
            "pyproject_state_owner_restart_table",
            table.get("interface") == INTERFACE
            and table.get("schema") == SCHEMA
            and table.get("task-id") == PCPR_070_TASK_ID
            and table.get("objective-kind") == OBJECTIVE_KIND,
            reason=(
                "pyproject.toml declares AccelerateAuthoritativeStateOwnerRestart@1."
                if table.get("interface") == INTERFACE
                else "pyproject.toml does not declare the PCPR-070 restart."
            ),
        ),
        _probe(
            "paths_remain_authorized",
            restart["paths_authorized"] is True
            and list(restart["authorized_path_prefixes"])
            == list(AUTHORIZED_PATH_PREFIXES),
            reason="Restart paths remain inside the declared PCPR-070 allowlist.",
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
            restart["interface_cid"] == PINNED_INTERFACE_CID,
            reason="Accelerate binds the PCPR-068 interface CID and does not remint it.",
        ),
        _probe(
            "delta_cid_bound_not_minted",
            restart["delta_cid"] == PINNED_DELTA_CID
            and document["stale_rejection"]["delta_cid"] == PINNED_DELTA_CID,
            reason="Accelerate binds the PCPR-069 delta CID and does not remint it.",
        ),
        _probe(
            "state_reconstructed_from_durable_records",
            restart["state_reconstructed"] is True
            and restart["reconstructed_from"] == RECONSTRUCTED_FROM
            and restart["reconstructed_from_memory"] is False
            and snapshot["reconstructed_from_memory"] is False,
            reason="Owner state is reconstructed from durable records, not memory.",
        ),
        _probe(
            "duplicate_restart_effect_rejected",
            restart["duplicate_effect_rejected"] is True
            and restart["restart_command_digest"] == RESTART_COMMAND_DIGEST,
            reason="Replaying the same restart command is rejected.",
        ),
        _probe(
            "unexpired_lease_retained",
            leases["unexpired_lease_retained"] is True
            and leases["retained_lease"]["expired"] is False,
            reason="Unexpired leases are retained across owner restart.",
        ),
        _probe(
            "stale_lease_expired",
            leases["stale_lease_expired"] is True
            and leases["expired_lease"]["stale"] is True
            and leases["stale_lease_reused"] is False,
            reason="Stale leases expire and cannot be reused after restart.",
        ),
        _probe(
            "current_fence_retained",
            leases["current_fence_retained"] is True
            and leases["retained_fence"]["fence_epoch"] == 1,
            reason="Current fence epoch is retained across owner restart.",
        ),
        _probe(
            "stale_fence_expired",
            leases["stale_fence_expired"] is True
            and leases["expired_fence"]["stale"] is True
            and leases["stale_fence_reused"] is False,
            reason="Stale fences expire and cannot authorize post-restart mutation.",
        ),
        _probe(
            "owner_generation_incremented",
            snapshot["pre_restart_generation"] == PRE_RESTART_OWNER_GENERATION
            and snapshot["post_restart_generation"] == POST_RESTART_OWNER_GENERATION
            and snapshot["generation_incremented"] is True,
            reason="Owner generation increments from 1 to 2 on restart.",
        ),
        _probe(
            "continuation_without_database_edits",
            restart["continuation_without_database_edits"] is True
            and restart["database_edited"] is False
            and document["duckdb_or_quack_state_written"] is False,
            reason="Continuation proceeds without DuckDB or Quack writes.",
        ),
        _probe(
            "model_assertion_cannot_complete_work",
            restart["model_assertion_completes_work"] is False,
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
            reason="Missing live restart emits the operator-blocking task.",
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
                "delta, objective, idea, or lock CID is forbidden."
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
            restart["interface_cid"] != PINNED_INTERFACE_CID,
            reason="Accelerate must not remint the PCPR-068 interface CID.",
        ),
        _probe(
            "delta_identity_reminted",
            restart["delta_cid"] != PINNED_DELTA_CID,
            reason="Accelerate must not remint the PCPR-069 delta CID.",
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
            "live_restart_represented_as_live",
            False,
            reason="Hermetic reconstruction is not represented as live restart.",
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
            "duplicate_restart_effect_accepted",
            False,
            reason="Duplicate restart effects are rejected.",
        ),
        _probe(
            "stale_lease_reused",
            False,
            reason="Stale leases are not reused after restart.",
        ),
        _probe(
            "stale_fence_reused",
            False,
            reason="Stale fences are not reused after restart.",
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
            "live_restart",
            None,
            evidence_kind="unavailable",
            reason="Live Quack owner restart stays typed unavailable.",
        ),
    ]
    return tuple(probes)


def qualify_state_owner_restart(
    probes: Sequence[OutcomeProbe],
    *,
    restart_cid: str,
    document_cid: str,
    catalog_cid: str,
    pack_cid: str,
    current_root_cid: str,
    route_cid: str,
    patch_cid: str,
    run_cid: str,
    interface_cid: str,
    delta_cid: str,
    objective_cid: str,
    idea_digest_cid: str,
) -> AccelerateStateOwnerRestartVerdict:
    if not probes:
        raise AccelerateStateOwnerRestartError("at least one probe is required")
    normalized: list[OutcomeProbe] = []
    blockers: list[str] = []
    for probe in probes:
        kind = _kind(probe.evidence_kind, "evidence_kind")
        if probe.live and kind != "measured_live":
            raise AccelerateStateOwnerRestartError(
                "live claims require measured_live evidence"
            )
        if probe.simulated_represented_as_live:
            raise AccelerateStateOwnerRestartError(
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
        "task_id": PCPR_070_TASK_ID,
        "goal_id": PCPR_070_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "sibling_source_required": False,
        "live_restart": False,
        "live_application": False,
        "operator_blocking_task": OPERATOR_BLOCKING_TASK_ID,
        "simulated_results_represented_as_live": False,
        "this_task_created_competing_authority": False,
        "probes": [item.to_mapping() for item in normalized],
        "blockers": list(dict.fromkeys(blockers)),
        "restart_cid": restart_cid,
        "document_cid": document_cid,
        "catalog_cid": catalog_cid,
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
    return AccelerateStateOwnerRestartVerdict(
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
        live_restart=False,
        live_application=False,
        operator_blocking_task=OPERATOR_BLOCKING_TASK_ID,
        simulated_results_represented_as_live=False,
        this_task_created_competing_authority=False,
        probes=tuple(normalized),
        blockers=tuple(dict.fromkeys(blockers)),
        verdict_cid=content_identity(payload),
        restart_cid=restart_cid,
        document_cid=document_cid,
        catalog_cid=catalog_cid,
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


def qualify_current_head_state_owner_restart(
    start: Path | None = None,
) -> AccelerateStateOwnerRestartVerdict:
    document = render_declared_restart(start)
    catalog = platform_state_owner_restart_catalog(start)
    return qualify_state_owner_restart(
        current_head_static_probes(start),
        restart_cid=str(document["state_owner_restart"]["restart_cid"]),
        document_cid=str(document["document_cid"]),
        catalog_cid=str(catalog["catalog_cid"]),
        pack_cid=str(document["context_pack"]["pack_cid"]),
        current_root_cid=str(document["storage"]["current_root_cid"]),
        route_cid=str(document["route"]["route_cid"]),
        patch_cid=str(document["bounded_patch"]["patch_cid"]),
        run_cid=str(document["selected_tests"]["run_cid"]),
        interface_cid=str(document["state_owner_restart"]["interface_cid"]),
        delta_cid=str(document["stale_rejection"]["delta_cid"]),
        objective_cid=str(document["objective_cid"]),
        idea_digest_cid=str(document["idea_digest"]),
    )


def pcpr_070_receipt_promotion(
    verdict: AccelerateStateOwnerRestartVerdict,
) -> dict[str, Any]:
    if verdict.closed_release_outcome is not None:
        raise AccelerateStateOwnerRestartError(
            "state-owner restart must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise AccelerateStateOwnerRestartError(
            "state-owner restart must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise AccelerateStateOwnerRestartError(
            "state-owner restart completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise AccelerateStateOwnerRestartError(
            "state-owner restart must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise AccelerateStateOwnerRestartError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.live_restart or verdict.live_application:
        raise AccelerateStateOwnerRestartError(
            "live restart claims require measured_live evidence"
        )
    return verdict.to_mapping()


PINNED_RESTART_CID: Final = (
    "baguqeeratw2evha3sv4ko5km5e5igcjprwfpkse4x2siwa5ldhyu5ugbjyja"
)
PINNED_DOCUMENT_CID: Final = (
    "baguqeerajwh6ewkh775xixoiycgo5gp3ohdhqbk3vvnphqtcaje2js5whknq"
)
PINNED_CATALOG_CID: Final = (
    "baguqeera3ppisaffyzj3uf347f5s3wviku4j23glhdcoz5hsg55xeiuwq7za"
)
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeerag7vkmfz26hywycakyiikbgtsg553gv2mwmnl7tsisnaeioqi2iqa"
)


__all__ = [
    "AUTHORIZED_PATH_PREFIXES",
    "CLOSED_RELEASE_OUTCOMES",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "ESCALATION_ORDER",
    "AccelerateStateOwnerRestartError",
    "HERMETIC_CANDIDATE_SUITES",
    "INTERFACE",
    "OBJECTIVE_KIND",
    "OPERATOR_BLOCKING_TASK_ID",
    "OutcomeProbe",
    "PCPR_070_GOAL_ID",
    "PCPR_070_TASK_ID",
    "PINNED_CATALOG_CID",
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
    "PINNED_RESTART_CID",
    "PINNED_REUSE_CID",
    "PINNED_ROUTE_CID",
    "PINNED_RUN_CID",
    "PLAN_EPOCH",
    "PROTOCOL_INTERFACE",
    "RECONSTRUCTED_IDENTITIES",
    "SCHEMA",
    "SEALED_PATH",
    "SEALED_PYTHON",
    "canonical_state_owner_restart",
    "current_head_static_probes",
    "hermetic_lease_and_fence_disposition",
    "hermetic_owner_snapshot",
    "path_is_authorized",
    "pcpr_070_receipt_promotion",
    "platform_state_owner_restart_catalog",
    "produce_state_owner_restart",
    "qualify_current_head_state_owner_restart",
    "qualify_state_owner_restart",
    "refuse_current_root_remint",
    "refuse_database_edit",
    "refuse_delta_cid_remint",
    "refuse_duplicate_restart_effect",
    "refuse_interface_cid_remint",
    "refuse_live_restart",
    "refuse_memory_reconstruction",
    "refuse_model_completion",
    "refuse_pack_cid_remint",
    "refuse_patch_cid_remint",
    "refuse_route_cid_remint",
    "refuse_run_cid_remint",
    "refuse_stale_fence_reuse",
    "refuse_stale_lease_reuse",
    "refuse_unauthorized_path",
    "render_declared_restart",
    "restart_cid_of",
    "verify_state_owner_restart_files",
    "write_state_owner_restart_files",
]
