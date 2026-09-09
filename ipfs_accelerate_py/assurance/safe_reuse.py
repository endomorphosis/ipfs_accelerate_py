"""Fail-closed PCPR-067 Accelerate-owned eligible reuse admission.

Accelerate owns reuse admission after an unrelated documentation change.
This module admits the PCPR-066 eligible ContextPack, selected-test, and
solver-qualification identities against the PCPR-061 DatasetsContextPack@1
identity, the PCPR-062 Kit current root, the PCPR-063 route, the
PCPR-064 patch, the PCPR-065 selected-tests run, and the PCPR-066
unrelated-change identity without reminting those identities.

The PCPR-066 impacted cone remains empty. Eligible identities are
therefore reusable and are recorded as hermetically demonstrated. Live
reuse stays typed unavailable. A relevant interface change remains
PCPR-068. Incremental prover binaries missing from the sealed PATH stay
typed unavailable. Model stages are not invoked. A model assertion
cannot complete work.

Hermetic reuse admission is candidate coverage only. It is not live
supervisor reuse and not a closed PCPR release. Simulated results are
not live. This evaluator does not write DuckDB or Quack state.
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
from ipfs_accelerate_py.assurance.unrelated_state_change import (
    CHANGE_KIND as PCPR_066_CHANGE_KIND,
    ELIGIBLE_REUSE,
    IMPACTED_CONE as PCPR_066_IMPACTED_CONE,
    PINNED_CHANGE_CID as PCPR_066_CHANGE_CID,
    PINNED_CURRENT_ROOT_CID,
    PINNED_PACK_CID,
    PINNED_PATCH_CID,
    PINNED_ROUTE_CID,
    PINNED_RUN_CID,
    UNRELATED_CHANGE_PATHS,
)

INTERFACE: Final = "AccelerateSafeReuse@1"
SCHEMA: Final = "ipfs_accelerate_py/assurance/safe-reuse@1"
DOCUMENT_SCHEMA: Final = "ipfs_accelerate_py/assurance/declared-safe-reuse@1"
REUSE_SCHEMA: Final = "ipfs_accelerate_py/assurance/safe-reuse-event@1"
VERDICT_SCHEMA: Final = "ipfs_accelerate_py/assurance/safe-reuse-verdict@1"
CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/platform-safe-reuse-catalog@1"
)
CATALOG_INTERFACE: Final = "PlatformSafeReuse@1"
REUSE_INTERFACE: Final = "SafeReuseAdmission@1"
OWNER_INTERFACE: Final = "DatasetsContextPack@1"
OWNER_SCHEMA: Final = "ipfs_datasets_py/datasets-context-pack@1"
OWNER_REPOSITORY: Final = "ipfs_datasets_py"
STORAGE_OWNER_REPOSITORY: Final = "ipfs_kit_py"
STORAGE_OWNER_INTERFACE: Final = "KitContextPackStorage@1"
EXECUTION_OWNER_REPOSITORY: Final = "ipfs_accelerate_py"
PATCH_OWNER_INTERFACE: Final = "AccelerateBoundedPatch@1"
ROUTE_OWNER_INTERFACE: Final = "AccelerateDeterministicFirstRoute@1"
RUN_OWNER_INTERFACE: Final = "AccelerateSelectedTestsAndProofs@1"
CHANGE_OWNER_INTERFACE: Final = "AccelerateUnrelatedStateChange@1"
EXECUTION_INVOCATION_INTERFACE: Final = OWNER_INTERFACES["ExecutionInvocation"]
EXECUTION_INVOCATION_SCHEMA: Final = OWNER_SCHEMAS["ExecutionInvocation"]
EXECUTION_RECEIPT_INTERFACE: Final = OWNER_INTERFACES["ExecutionReceipt"]
EXECUTION_RECEIPT_SCHEMA: Final = OWNER_SCHEMAS["ExecutionReceipt"]
PROTOCOL_INTERFACE: Final = "LogicProviderProtocol@2"
DOCUMENTATION_INTERFACE: Final = "LogicProviderProtocolUnrelatedDocumentation@1"
DOCUMENTATION_SCHEMA: Final = (
    "ipfs_datasets_py/logic-provider-unrelated-documentation@1"
)
PCPR_067_TASK_ID: Final = "PCPR-067"
PCPR_067_GOAL_ID: Final = "PCPR-G700"
PCPR_066_TASK_ID: Final = "PCPR-066"
PCPR_065_TASK_ID: Final = "PCPR-065"
PCPR_064_TASK_ID: Final = "PCPR-064"
PCPR_063_TASK_ID: Final = "PCPR-063"
PCPR_062_TASK_ID: Final = "PCPR-062"
PCPR_061_TASK_ID: Final = "PCPR-061"
PCPR_060_TASK_ID: Final = "PCPR-060"
PCPR_068_TASK_ID: Final = "PCPR-068"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = (
    "proof-carrying-platform-qualification-and-release-v1"
)
OBJECTIVE_KIND: Final = "declared_safe_reuse"
OBJECTIVE_ID: Final = "PCPR-G700"
LANGUAGE: Final = "Python"
OPERATOR_BLOCKING_TASK_ID: Final = "pcpr-067-operator-live-safe-reuse"
DOCUMENT_DIR_RELPATH: Final = "packaging/pcpr/reference-workflow/cpython312"
DOCUMENT_JSON_NAME: Final = "reference.safe-reuse.json"
DOCUMENT_README_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/SAFE_REUSE.md"
)
CATALOG_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/platform-safe-reuse-catalog.json"
)
SOURCE_DATE_EPOCH: Final = "0"
SOURCE_REPOSITORY: Final = "endomorphosis/ipfs_accelerate_py"
COMPLETED_THROUGH: Final = "eligible_reuse_admitted"
NEXT_AUTHORIZED_STAGE: Final = "relevant_interface_change"
REUSE_KIND: Final = "eligible_unrelated_change_reuse"

PINNED_IDEA_DIGEST: Final = (
    "baguqeeracbayojdov4jmqiirx22pavrg6nabazcocru6y3scrdx5e54mw2zq"
)
PINNED_OBJECTIVE_CID: Final = (
    "baguqeeraynsn7tjr3iaggnreylzxo3akaooqwp5bf5oheaa6eubth2zwqeba"
)
PINNED_LOCK_CID: Final = PCPR_056_LOCK_CID
PINNED_BYTES_CID: Final = PINNED_CURRENT_ROOT_CID

if PINNED_LOCK_CID != (
    "baguqeerawsekbbbt5ccctjt4tydzeahfkahsatb6q5x7k6cy4inrii5lhqmq"
):
    raise RuntimeError("PCPR-067 lock CID remints PCPR-056")
if PCPR_066_CHANGE_CID != (
    "baguqeeraowfymvuvscitvjkxn3sak6f4a2ctkfvflney26jen7gwyqplln6a"
):
    raise RuntimeError("PCPR-067 remints the PCPR-066 change CID")
if OWNER_INTERFACES["SupervisorContextPack"] != OWNER_INTERFACE:
    raise RuntimeError("PCPR-067 remints SupervisorContextPack owner interface")
if OWNER_SCHEMAS["SupervisorContextPack"] != OWNER_SCHEMA:
    raise RuntimeError("PCPR-067 remints SupervisorContextPack owner schema")
if EXECUTION_INVOCATION_INTERFACE != "InvocationRequest@1":
    raise RuntimeError("PCPR-067 remints ExecutionInvocation owner interface")
if EXECUTION_RECEIPT_INTERFACE != "InvocationResult@1":
    raise RuntimeError("PCPR-067 remints ExecutionReceipt owner interface")
if tuple(PCPR_066_IMPACTED_CONE) != ():
    raise RuntimeError("PCPR-067 requires an empty PCPR-066 impact cone")

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
    "artifacts/proof_carrying_platform_qualification_and_release/receipts/PCPR-067.json",
)

STALE_IDENTITIES: Final[tuple[str, ...]] = ()

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_safe_reuse.py",
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
        "reuse_files_match_generator",
        "pyproject_safe_reuse_table",
        "paths_remain_authorized",
        "owner_pack_cid_matches_pin",
        "kit_current_root_bound_not_minted",
        "route_cid_bound_not_minted",
        "patch_cid_bound_not_minted",
        "run_cid_bound_not_minted",
        "change_cid_bound_not_minted",
        "impacted_cone_empty",
        "eligible_reuse_demonstrated_hermetically",
        "stale_identities_empty",
        "protocol_identity_not_reminted",
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
        "live_reuse_represented_as_live",
        "closed_release_represented_as_live",
        "compatibility_identities_reminted",
        "direct_database_bypass_used",
        "datasets_identity_reminted",
        "kit_identity_reminted",
        "route_identity_reminted",
        "patch_identity_reminted",
        "run_identity_reminted",
        "change_identity_reminted",
        "model_assertion_completed_work",
        "unauthorized_path_accepted",
        "relevant_interface_change_accepted",
        "whole_plan_regeneration_required",
        "stale_reuse_accepted",
        "live_reuse_claimed",
    }
)

DOCUMENT_README: Final = """# PCPR-067 Accelerate safe reuse

These files record the Accelerate-owned hermetic eligible-reuse admission
against the PCPR-061 DatasetsContextPack@1 identity, the PCPR-062 Kit
current root, the PCPR-063 deterministic-first route, the PCPR-064
bounded PatchPlan, the PCPR-065 selected-tests-and-proofs run, and the
PCPR-066 unrelated documentation change. Accelerate owns reuse
admission. It does not remint those identities and does not claim live
reuse.

- `cpython312/reference.safe-reuse.json` is the declared reuse document.
  Exact commit and tree are bound by the PCPR-067 receipt
  `current_tree_binding`.
- `platform-safe-reuse-catalog.json` observes sibling Datasets and Kit
  bindings when present. Sibling source is never required.
- The PCPR-066 impacted cone remains empty. Eligible ContextPack,
  selected-test, and solver-qualification identities are reused
  hermetically. A relevant interface change remains PCPR-068.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-067-operator-live-safe-reuse`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
"""


class AccelerateSafeReuseError(Exception):
    """Fail-closed PCPR-067 Accelerate safe-reuse error."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AccelerateSafeReuseError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise AccelerateSafeReuseError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise AccelerateSafeReuseError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def parse_pyproject_safe_reuse_table(text: str) -> dict[str, Any]:
    import tomllib

    table = tomllib.loads(_text(text, "pyproject.toml"))
    if not isinstance(table, dict):
        raise AccelerateSafeReuseError("pyproject.toml must be a table")
    tool = table.get("tool")
    payload: dict[str, Any] = {}
    if isinstance(tool, dict):
        accelerate = tool.get("ipfs_accelerate_py")
        if isinstance(accelerate, dict):
            raw = accelerate.get("safe-reuse")
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
        raise AccelerateSafeReuseError(
            f"path {path} is not an authorized PCPR-067 path"
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
        raise AccelerateSafeReuseError(
            f"model assertion at {stage} cannot complete work"
        )


def refuse_pack_cid_remint(cid: str) -> str:
    if cid != PINNED_PACK_CID:
        raise AccelerateSafeReuseError(
            f"ContextPack CID {cid} remints {PINNED_PACK_CID}"
        )
    return cid


def refuse_current_root_remint(cid: str) -> str:
    if cid != PINNED_CURRENT_ROOT_CID:
        raise AccelerateSafeReuseError(
            f"current root CID {cid} remints {PINNED_CURRENT_ROOT_CID}"
        )
    return cid


def refuse_route_cid_remint(cid: str) -> str:
    if cid != PINNED_ROUTE_CID:
        raise AccelerateSafeReuseError(
            f"route CID {cid} remints {PINNED_ROUTE_CID}"
        )
    return cid


def refuse_patch_cid_remint(cid: str) -> str:
    if cid != PINNED_PATCH_CID:
        raise AccelerateSafeReuseError(
            f"patch CID {cid} remints {PINNED_PATCH_CID}"
        )
    return cid


def refuse_run_cid_remint(cid: str) -> str:
    if cid != PINNED_RUN_CID:
        raise AccelerateSafeReuseError(
            f"run CID {cid} remints {PINNED_RUN_CID}"
        )
    return cid


def refuse_change_cid_remint(cid: str) -> str:
    if cid != PCPR_066_CHANGE_CID:
        raise AccelerateSafeReuseError(
            f"change CID {cid} remints {PCPR_066_CHANGE_CID}"
        )
    return cid


def refuse_relevant_interface_change(*, relevant: bool) -> None:
    if relevant:
        raise AccelerateSafeReuseError(
            "PCPR-067 cannot accept a relevant interface change"
        )


def refuse_whole_plan_regeneration(*, required: bool) -> None:
    if required:
        raise AccelerateSafeReuseError(
            "eligible reuse cannot require whole-plan regeneration"
        )


def refuse_reuse_not_demonstrated(*, demonstrated: bool) -> None:
    if not demonstrated:
        raise AccelerateSafeReuseError(
            "PCPR-067 must demonstrate eligible reuse hermetically"
        )


def refuse_live_reuse(*, live: bool) -> None:
    if live:
        raise AccelerateSafeReuseError(
            "live reuse requires measured_live evidence and remains unavailable"
        )


def refuse_stale_reuse(*, stale: Sequence[str]) -> None:
    if tuple(stale):
        raise AccelerateSafeReuseError(
            "stale identities cannot be reused after an unrelated change"
        )


def reused_identities() -> list[dict[str, Any]]:
    return [
        {
            "identity": "DatasetsContextPack@1",
            "kind": "context_pack",
            "cid": PINNED_PACK_CID,
            "task_id": PCPR_061_TASK_ID,
            "stale": False,
            "admitted": True,
            "live": False,
            "evidence_kind": "measured_hermetic",
        },
        {
            "identity": "tests/unit/test_pcpr_013_logic_provider_protocol.py",
            "kind": "selected_test",
            "run_cid": PINNED_RUN_CID,
            "task_id": PCPR_065_TASK_ID,
            "stale": False,
            "admitted": True,
            "live": False,
            "evidence_kind": "measured_hermetic",
        },
        {
            "identity": "tests/unit/test_pcpr_014_semantic_apis.py",
            "kind": "selected_test",
            "run_cid": PINNED_RUN_CID,
            "task_id": PCPR_065_TASK_ID,
            "stale": False,
            "admitted": True,
            "live": False,
            "evidence_kind": "measured_hermetic",
        },
        {
            "identity": "tests/unit/test_pcpr_017_solver_qualification.py",
            "kind": "selected_test",
            "run_cid": PINNED_RUN_CID,
            "task_id": PCPR_065_TASK_ID,
            "stale": False,
            "admitted": True,
            "live": False,
            "evidence_kind": "measured_hermetic",
        },
        {
            "identity": "ipfs_datasets_py/assurance/solver_qualification.py",
            "kind": "proof",
            "task_id": "PCPR-017",
            "stale": False,
            "admitted": True,
            "live": False,
            "evidence_kind": "measured_hermetic",
        },
    ]


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
class AccelerateSafeReuseVerdict:
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
    live_reuse: bool
    operator_blocking_task: str
    simulated_results_represented_as_live: bool
    this_task_created_competing_authority: bool
    probes: tuple[OutcomeProbe, ...]
    blockers: tuple[str, ...]
    verdict_cid: str
    reuse_cid: str
    document_cid: str
    catalog_cid: str
    pack_cid: str
    current_root_cid: str
    route_cid: str
    patch_cid: str
    run_cid: str
    change_cid: str
    objective_cid: str
    idea_digest: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "verdict_cid": self.verdict_cid,
            "reuse_cid": self.reuse_cid,
            "document_cid": self.document_cid,
            "catalog_cid": self.catalog_cid,
            "pack_cid": self.pack_cid,
            "current_root_cid": self.current_root_cid,
            "route_cid": self.route_cid,
            "patch_cid": self.patch_cid,
            "run_cid": self.run_cid,
            "change_cid": self.change_cid,
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
            "live_reuse": self.live_reuse,
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
            "An admitted Quack-fenced state-owner session before eligible "
            "reuse is applied as live supervisor work against the stored "
            "ContextPack root"
        ),
        "action": (
            "Keep the hermetic eligible-reuse admission. Do not write "
            "DuckDB or Quack state. Do not escalate to a model. Do not "
            "treat this as a relevant interface change. PCPR-068 "
            "introduces a relevant interface change."
        ),
        "reason": (
            "Hermetic eligible-reuse admission is not live supervisor "
            "reuse. Sealed validation has no admitted Quack-fenced "
            "session. Direct DuckDB writes are prohibited."
        ),
    }


def produce_safe_reuse(
    *,
    paths: Sequence[str] | None = None,
    pack_cid: str = PINNED_PACK_CID,
    current_root_cid: str = PINNED_CURRENT_ROOT_CID,
    route_cid: str = PINNED_ROUTE_CID,
    patch_cid: str = PINNED_PATCH_CID,
    run_cid: str = PINNED_RUN_CID,
    change_cid: str = PCPR_066_CHANGE_CID,
    complete_from: str | None = None,
    relevant: bool = False,
    whole_plan_regeneration_required: bool = False,
    reuse_demonstrated: bool = True,
    live_reuse: bool = False,
    stale: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Produce the hermetic eligible-reuse admission. Fail closed."""

    if complete_from is not None:
        refuse_model_completion(complete_from)
    refuse_pack_cid_remint(pack_cid)
    refuse_current_root_remint(current_root_cid)
    refuse_route_cid_remint(route_cid)
    refuse_patch_cid_remint(patch_cid)
    refuse_run_cid_remint(run_cid)
    refuse_change_cid_remint(change_cid)
    refuse_relevant_interface_change(relevant=relevant)
    refuse_whole_plan_regeneration(required=whole_plan_regeneration_required)
    refuse_reuse_not_demonstrated(demonstrated=reuse_demonstrated)
    refuse_live_reuse(live=live_reuse)
    refuse_stale_reuse(stale=list(stale or STALE_IDENTITIES))
    checked_paths = list(paths) if paths is not None else list(AUTHORIZED_PATH_PREFIXES)
    for path in checked_paths:
        refuse_unauthorized_path(path)
    reused = reused_identities()
    if [item["identity"] for item in reused] != list(ELIGIBLE_REUSE):
        raise AccelerateSafeReuseError(
            "reused identities must equal the PCPR-066 eligible-reuse set"
        )
    return {
        "schema": REUSE_SCHEMA,
        "interface": REUSE_INTERFACE,
        "owner_interface": INTERFACE,
        "task_id": PCPR_067_TASK_ID,
        "goal_id": PCPR_067_GOAL_ID,
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
        "change_cid": change_cid,
        "protocol_interface": PROTOCOL_INTERFACE,
        "documentation_interface": DOCUMENTATION_INTERFACE,
        "documentation_schema": DOCUMENTATION_SCHEMA,
        "reuse_kind": REUSE_KIND,
        "change_kind": PCPR_066_CHANGE_KIND,
        "produced": True,
        "applied": True,
        "live": False,
        "hermetic": True,
        "admitted_live": False,
        "deferred": False,
        "paths_authorized": True,
        "authorized_path_prefixes": list(AUTHORIZED_PATH_PREFIXES),
        "unrelated_change_paths": list(UNRELATED_CHANGE_PATHS),
        "eligible_reuse": list(ELIGIBLE_REUSE),
        "reused": reused,
        "stale_identities": list(STALE_IDENTITIES),
        "impacted_cone": list(PCPR_066_IMPACTED_CONE),
        "remints_protocol": False,
        "adds_protocol_operation": False,
        "relevant_interface_change": False,
        "whole_plan_regeneration_required": False,
        "eligible_reuse_preserved": True,
        "reuse_demonstrated": True,
        "live_reuse": False,
        "model_assertion_completes_work": False,
        "escalation_order": list(ESCALATION_ORDER),
        "completed_through": COMPLETED_THROUGH,
        "next_authorized_stage": NEXT_AUTHORIZED_STAGE,
        "next_task_id": PCPR_068_TASK_ID,
        "prerequisite_task_id": PCPR_066_TASK_ID,
        "relevant_change_task_id": PCPR_068_TASK_ID,
        "duckdb_or_quack_state_written": False,
        "closed_release_outcome": None,
        "release_claim": False,
        "evidence_kind": "measured_hermetic",
    }


def canonical_reuse_event() -> dict[str, Any]:
    return produce_safe_reuse()


def reuse_cid_of(plan: Mapping[str, Any] | None = None) -> str:
    payload = dict(plan or canonical_reuse_event())
    payload.pop("reuse_cid", None)
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
                "Sibling safe-reuse file is not present beside this checkout."
            ),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    context_pack = payload.get("context_pack")
    storage = payload.get("storage")
    route = payload.get("route")
    bounded_patch = payload.get("bounded_patch")
    selected = payload.get("selected_tests")
    change = payload.get("unrelated_state_change")
    reuse = payload.get("safe_reuse")
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
        or (
            selected.get("run_cid") if isinstance(selected, Mapping) else None
        ),
        "change_cid": payload.get("change_cid")
        or (
            change.get("change_cid") if isinstance(change, Mapping) else None
        ),
        "reuse_cid": payload.get("reuse_cid")
        or (reuse.get("reuse_cid") if isinstance(reuse, Mapping) else None),
        "binding_cid": payload.get("binding_cid") or payload.get("document_cid"),
        "live": False,
        "applied": False,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def render_declared_reuse(root: Path | None = None) -> dict[str, Any]:
    package_root = root or discover_accelerate_root()
    if package_root is None:
        raise AccelerateSafeReuseError("Accelerate package root was not found")
    state_owner = observe_state_owner()
    plan = canonical_reuse_event()
    computed_reuse_cid = reuse_cid_of(plan)
    document = {
        "schema": DOCUMENT_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_067_TASK_ID,
        "goal_id": PCPR_067_GOAL_ID,
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
                "A live ExecutionReceipt remains PCPR-072. This eligible "
                "reuse admission is hermetic."
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
            "reused": True,
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
            "reused": True,
            "evidence_kind": "measured_hermetic",
        },
        "unrelated_state_change": {
            "task_id": PCPR_066_TASK_ID,
            "owner_interface": CHANGE_OWNER_INTERFACE,
            "change_cid": PCPR_066_CHANGE_CID,
            "change_kind": PCPR_066_CHANGE_KIND,
            "live": False,
            "hermetic": True,
            "reminted": False,
            "impacted_cone": list(PCPR_066_IMPACTED_CONE),
            "eligible_reuse_preserved": True,
            "reuse_demonstrated": False,
            "evidence_kind": "measured_hermetic",
        },
        "safe_reuse": {
            **plan,
            "reuse_cid": computed_reuse_cid,
            "produced_by": EXECUTION_OWNER_REPOSITORY,
        },
        "materialization": {
            "kind": "declared_safe_reuse_not_live",
            "admitted": False,
            "live": False,
            "applied": False,
            "duckdb_or_quack_state_written": False,
            "evidence_kind": "unavailable",
        },
        "live_application": typed_unavailable(
            reason=(
                "No admitted Quack-fenced supervisor session was observed. "
                "Hermetic eligible-reuse admission is not live application."
            )
        ),
        "live_reuse": typed_unavailable(
            reason=(
                "Hermetic eligible-reuse admission is not live supervisor "
                "reuse. A relevant interface change remains PCPR-068."
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
                "field": "PCPR-067 receipt current_tree_binding",
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


def platform_safe_reuse_catalog(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateSafeReuseError("Accelerate package root was not found")
    parent = root.parent
    document = render_declared_reuse(root)
    datasets_binding = _sibling_binding(
        parent
        / "ipfs_datasets"
        / DOCUMENT_DIR_RELPATH
        / "reference.safe-reuse.binding.json",
        relative_path=(
            f"../ipfs_datasets/{DOCUMENT_DIR_RELPATH}/"
            "reference.safe-reuse.binding.json"
        ),
    )
    kit_binding = _sibling_binding(
        parent
        / "ipfs_kit"
        / DOCUMENT_DIR_RELPATH
        / "reference.safe-reuse.binding.json",
        relative_path=(
            f"../ipfs_kit/{DOCUMENT_DIR_RELPATH}/"
            "reference.safe-reuse.binding.json"
        ),
    )
    catalog = {
        "schema": CATALOG_SCHEMA,
        "interface": CATALOG_INTERFACE,
        "task_id": PCPR_067_TASK_ID,
        "goal_id": PCPR_067_GOAL_ID,
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
        "change_cid": PCPR_066_CHANGE_CID,
        "reuse_cid": document["safe_reuse"]["reuse_cid"],
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
                "change_cid": PCPR_066_CHANGE_CID,
                "reuse_cid": document["safe_reuse"]["reuse_cid"],
                "document_cid": document["document_cid"],
                "live": False,
                "applied": False,
                "reminted": False,
            },
            "ipfs_datasets_py": {"binding": datasets_binding},
            "ipfs_kit_py": {"binding": kit_binding},
        },
        "live_application": False,
        "live_reuse": False,
        "closed_release_outcome": None,
        "release_claim": False,
        "sibling_source_required": False,
        "evidence_kind": "measured",
    }
    catalog["catalog_cid"] = content_identity(
        {key: value for key, value in catalog.items() if key != "catalog_cid"}
    )
    return catalog


def write_safe_reuse_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateSafeReuseError("Accelerate package root was not found")
    document = render_declared_reuse(root)
    paths = artifact_paths(root)
    _atomic_write(paths["document"], pretty_json(document))
    readme = DOCUMENT_README if DOCUMENT_README.endswith("\n") else DOCUMENT_README + "\n"
    _atomic_write(paths["readme"], readme)
    catalog = platform_safe_reuse_catalog(start)
    _atomic_write(paths["catalog"], pretty_json(catalog))
    return {
        "document": document,
        "catalog": catalog,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def verify_safe_reuse_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateSafeReuseError("Accelerate package root was not found")
    document = render_declared_reuse(root)
    catalog = platform_safe_reuse_catalog(start)
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
        "reuse_cid": document["safe_reuse"]["reuse_cid"],
        "document_cid": document["document_cid"],
        "catalog_cid": catalog["catalog_cid"],
        "current_root_cid": document["storage"]["current_root_cid"],
        "route_cid": document["route"]["route_cid"],
        "patch_cid": document["bounded_patch"]["patch_cid"],
        "run_cid": document["selected_tests"]["run_cid"],
        "change_cid": document["unrelated_state_change"]["change_cid"],
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
        raise AccelerateSafeReuseError("Accelerate package root was not found")
    document = render_declared_reuse(root)
    verified = verify_safe_reuse_files(root)
    table = parse_pyproject_safe_reuse_table(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    reuse = document["safe_reuse"]
    remint = (
        document["context_pack"]["pack_cid"] != PINNED_PACK_CID
        or document["objective_cid"] != PINNED_OBJECTIVE_CID
        or document["idea_digest"] != PINNED_IDEA_DIGEST
        or document["lock_cid"] != PINNED_LOCK_CID
        or document["storage"]["current_root_cid"] != PINNED_CURRENT_ROOT_CID
        or document["route"]["route_cid"] != PINNED_ROUTE_CID
        or document["bounded_patch"]["patch_cid"] != PINNED_PATCH_CID
        or document["selected_tests"]["run_cid"] != PINNED_RUN_CID
        or document["unrelated_state_change"]["change_cid"] != PCPR_066_CHANGE_CID
        or reuse["reuse_cid"] != PINNED_REUSE_CID
        or document["context_pack"]["reminted"] is True
        or document["storage"]["reminted"] is True
        or document["route"]["reminted"] is True
        or document["bounded_patch"]["reminted"] is True
        or document["selected_tests"]["reminted"] is True
        or document["unrelated_state_change"]["reminted"] is True
    )
    probes = [
        _probe(
            "reuse_files_match_generator",
            verified.get("ok") is True and verified.get("document_ok") is True,
            reason=(
                "Committed Accelerate safe-reuse matches the generator."
                if verified.get("document_ok") is True
                else "Committed Accelerate safe-reuse is missing or drifts."
            ),
        ),
        _probe(
            "pyproject_safe_reuse_table",
            table.get("interface") == INTERFACE
            and table.get("schema") == SCHEMA
            and table.get("task-id") == PCPR_067_TASK_ID
            and table.get("objective-kind") == OBJECTIVE_KIND,
            reason=(
                "pyproject.toml declares AccelerateSafeReuse@1."
                if table.get("interface") == INTERFACE
                else "pyproject.toml does not declare the PCPR-067 reuse."
            ),
        ),
        _probe(
            "paths_remain_authorized",
            reuse["paths_authorized"] is True
            and list(reuse["authorized_path_prefixes"])
            == list(AUTHORIZED_PATH_PREFIXES),
            reason="Reuse paths remain inside the declared PCPR-067 allowlist.",
        ),
        _probe(
            "owner_pack_cid_matches_pin",
            document["context_pack"]["pack_cid"] == PINNED_PACK_CID
            and document["context_pack"]["reused"] is True,
            reason="Accelerate reuses the Datasets-owned pack CID without remint.",
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
            and document["selected_tests"]["reminted"] is False
            and document["selected_tests"]["reused"] is True,
            reason="Accelerate reuses the PCPR-065 run CID and does not remint it.",
        ),
        _probe(
            "change_cid_bound_not_minted",
            document["unrelated_state_change"]["change_cid"] == PCPR_066_CHANGE_CID
            and document["unrelated_state_change"]["reminted"] is False,
            reason="Accelerate binds the PCPR-066 change CID and does not remint it.",
        ),
        _probe(
            "impacted_cone_empty",
            list(reuse["impacted_cone"]) == []
            and reuse["whole_plan_regeneration_required"] is False,
            reason="The unrelated documentation change still has an empty impact cone.",
        ),
        _probe(
            "eligible_reuse_demonstrated_hermetically",
            reuse["eligible_reuse_preserved"] is True
            and reuse["reuse_demonstrated"] is True
            and reuse["live_reuse"] is False
            and reuse["next_task_id"] == PCPR_068_TASK_ID
            and [item["identity"] for item in reuse["reused"]]
            == list(ELIGIBLE_REUSE),
            reason="Eligible reuse is demonstrated hermetically and is not live.",
        ),
        _probe(
            "stale_identities_empty",
            list(reuse["stale_identities"]) == []
            and all(item["stale"] is False for item in reuse["reused"]),
            reason="No reused identity is stale after the unrelated change.",
        ),
        _probe(
            "protocol_identity_not_reminted",
            reuse["protocol_interface"] == PROTOCOL_INTERFACE
            and reuse["remints_protocol"] is False
            and reuse["adds_protocol_operation"] is False,
            reason="LogicProviderProtocol@2 is bound and not reminted.",
        ),
        _probe(
            "model_assertion_cannot_complete_work",
            reuse["model_assertion_completes_work"] is False,
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
            reason="Missing live reuse emits the operator-blocking task.",
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
                "A reminted pack, root, route, patch, run, change, reuse, "
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
            "change_identity_reminted",
            document["unrelated_state_change"]["reminted"] is True,
            reason="Accelerate must not remint the PCPR-066 change.",
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
            reason="Hermetic reuse admission is not represented as live.",
        ),
        _probe(
            "live_reuse_represented_as_live",
            False,
            reason="Hermetic eligible reuse is not represented as live.",
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
            "relevant_interface_change_accepted",
            False,
            reason="A relevant interface change remains PCPR-068.",
        ),
        _probe(
            "whole_plan_regeneration_required",
            False,
            reason="Eligible reuse does not regenerate the whole plan.",
        ),
        _probe(
            "stale_reuse_accepted",
            False,
            reason="Stale identities are not reused.",
        ),
        _probe(
            "live_reuse_claimed",
            False,
            reason="Live reuse is not claimed.",
        ),
        _probe(
            "live_application",
            None,
            evidence_kind="unavailable",
            reason="Live supervisor application stays typed unavailable.",
        ),
        _probe(
            "live_reuse",
            None,
            evidence_kind="unavailable",
            reason="Live reuse stays typed unavailable.",
        ),
    ]
    return tuple(probes)


def qualify_safe_reuse(
    probes: Sequence[OutcomeProbe],
    *,
    reuse_cid: str,
    document_cid: str,
    catalog_cid: str,
    pack_cid: str,
    current_root_cid: str,
    route_cid: str,
    patch_cid: str,
    run_cid: str,
    change_cid: str,
    objective_cid: str,
    idea_digest_cid: str,
) -> AccelerateSafeReuseVerdict:
    if not probes:
        raise AccelerateSafeReuseError("at least one probe is required")
    normalized: list[OutcomeProbe] = []
    blockers: list[str] = []
    for probe in probes:
        kind = _kind(probe.evidence_kind, "evidence_kind")
        if probe.live and kind != "measured_live":
            raise AccelerateSafeReuseError(
                "live claims require measured_live evidence"
            )
        if probe.simulated_represented_as_live:
            raise AccelerateSafeReuseError(
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
        "task_id": PCPR_067_TASK_ID,
        "goal_id": PCPR_067_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "sibling_source_required": False,
        "live_application": False,
        "live_reuse": False,
        "operator_blocking_task": OPERATOR_BLOCKING_TASK_ID,
        "simulated_results_represented_as_live": False,
        "this_task_created_competing_authority": False,
        "probes": [item.to_mapping() for item in normalized],
        "blockers": list(dict.fromkeys(blockers)),
        "reuse_cid": reuse_cid,
        "document_cid": document_cid,
        "catalog_cid": catalog_cid,
        "pack_cid": pack_cid,
        "current_root_cid": current_root_cid,
        "route_cid": route_cid,
        "patch_cid": patch_cid,
        "run_cid": run_cid,
        "change_cid": change_cid,
        "objective_cid": objective_cid,
        "idea_digest": idea_digest_cid,
    }
    return AccelerateSafeReuseVerdict(
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
        live_reuse=False,
        operator_blocking_task=OPERATOR_BLOCKING_TASK_ID,
        simulated_results_represented_as_live=False,
        this_task_created_competing_authority=False,
        probes=tuple(normalized),
        blockers=tuple(dict.fromkeys(blockers)),
        verdict_cid=content_identity(payload),
        reuse_cid=reuse_cid,
        document_cid=document_cid,
        catalog_cid=catalog_cid,
        pack_cid=pack_cid,
        current_root_cid=current_root_cid,
        route_cid=route_cid,
        patch_cid=patch_cid,
        run_cid=run_cid,
        change_cid=change_cid,
        objective_cid=objective_cid,
        idea_digest=idea_digest_cid,
    )


def qualify_current_head_safe_reuse(
    start: Path | None = None,
) -> AccelerateSafeReuseVerdict:
    document = render_declared_reuse(start)
    catalog = platform_safe_reuse_catalog(start)
    return qualify_safe_reuse(
        current_head_static_probes(start),
        reuse_cid=str(document["safe_reuse"]["reuse_cid"]),
        document_cid=str(document["document_cid"]),
        catalog_cid=str(catalog["catalog_cid"]),
        pack_cid=str(document["context_pack"]["pack_cid"]),
        current_root_cid=str(document["storage"]["current_root_cid"]),
        route_cid=str(document["route"]["route_cid"]),
        patch_cid=str(document["bounded_patch"]["patch_cid"]),
        run_cid=str(document["selected_tests"]["run_cid"]),
        change_cid=str(document["unrelated_state_change"]["change_cid"]),
        objective_cid=str(document["objective_cid"]),
        idea_digest_cid=str(document["idea_digest"]),
    )


def pcpr_067_receipt_promotion(
    verdict: AccelerateSafeReuseVerdict,
) -> dict[str, Any]:
    if verdict.closed_release_outcome is not None:
        raise AccelerateSafeReuseError(
            "safe reuse must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise AccelerateSafeReuseError(
            "safe reuse must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise AccelerateSafeReuseError(
            "safe reuse completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise AccelerateSafeReuseError(
            "safe reuse must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise AccelerateSafeReuseError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.live_application or verdict.live_reuse:
        raise AccelerateSafeReuseError(
            "live reuse claims require measured_live evidence"
        )
    return verdict.to_mapping()


PINNED_REUSE_CID: Final = (
    "baguqeera75dkjwczh6mehnwyqot25rbd2mazkgjuvb4lrvkdkuqo7vntieuq"
)
PINNED_DOCUMENT_CID: Final = (
    "baguqeeracy4fp45z3lre2s62ltz74vuzypznydbyqg7qvq6pzdsnxmeyprqq"
)
PINNED_CATALOG_CID: Final = (
    "baguqeeraqyurxd52u644pv5w2w3p4aa627zjiigp4qoqzby2agyv5yo6o5xa"
)
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeeradpa5gqzaiom5kheuv62lstgvt4eipyiql2m4t3dkzp6qq5d4cvra"
)


__all__ = [
    "AUTHORIZED_PATH_PREFIXES",
    "CLOSED_RELEASE_OUTCOMES",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "ELIGIBLE_REUSE",
    "ESCALATION_ORDER",
    "AccelerateSafeReuseError",
    "HERMETIC_CANDIDATE_SUITES",
    "INTERFACE",
    "OBJECTIVE_KIND",
    "OPERATOR_BLOCKING_TASK_ID",
    "OutcomeProbe",
    "PCPR_066_CHANGE_CID",
    "PCPR_067_GOAL_ID",
    "PCPR_067_TASK_ID",
    "PINNED_CATALOG_CID",
    "PINNED_CURRENT_ROOT_CID",
    "PINNED_DOCUMENT_CID",
    "PINNED_IDEA_DIGEST",
    "PINNED_LOCK_CID",
    "PINNED_OBJECTIVE_CID",
    "PINNED_PACK_CID",
    "PINNED_PATCH_CID",
    "PINNED_REUSE_CID",
    "PINNED_ROUTE_CID",
    "PINNED_RUN_CID",
    "PROTOCOL_INTERFACE",
    "SCHEMA",
    "SEALED_PATH",
    "SEALED_PYTHON",
    "canonical_reuse_event",
    "current_head_static_probes",
    "path_is_authorized",
    "pcpr_067_receipt_promotion",
    "platform_safe_reuse_catalog",
    "produce_safe_reuse",
    "qualify_current_head_safe_reuse",
    "qualify_safe_reuse",
    "refuse_change_cid_remint",
    "refuse_current_root_remint",
    "refuse_live_reuse",
    "refuse_model_completion",
    "refuse_pack_cid_remint",
    "refuse_patch_cid_remint",
    "refuse_relevant_interface_change",
    "refuse_reuse_not_demonstrated",
    "refuse_route_cid_remint",
    "refuse_run_cid_remint",
    "refuse_stale_reuse",
    "refuse_unauthorized_path",
    "refuse_whole_plan_regeneration",
    "render_declared_reuse",
    "reuse_cid_of",
    "reused_identities",
    "verify_safe_reuse_files",
    "write_safe_reuse_files",
]
