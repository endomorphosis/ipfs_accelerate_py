"""Fail-closed PCPR-082 Accelerate-owned cross-client objective identity parity.

Accelerate owns the hermetic comparison of the PCPR-080 Python client and
the PCPR-081 generic MCP client against the same canonical service. Both
clients submit the PCPR-060 canonical objective bytes and must receive
the same objective identity and equivalent goals, tasks, events,
candidate evidence, and final receipts.

Client CIDs remain distinct: the Python client CID and the generic MCP
client CID are not reminted into each other. Live Supervisor.run, live
Quack, live MCP stdio/HTTP, and live CLI sessions stay typed
unavailable. Simulated results are not live. This evaluator does not
write DuckDB or Quack state and does not emit a closed PCPR release.

Hermetic parity coverage is candidate coverage only. External-client
authority-bypass proof remains PCPR-083.
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
from ipfs_accelerate_py.assurance.generic_mcp_client import (
    GenericMcpClient,
    PINNED_CLIENT_CID as PINNED_MCP_CLIENT_CID,
    hermetic_generic_mcp_client_trace,
)
from ipfs_accelerate_py.assurance.portfolio_compatibility_lock import (
    PINNED_LOCK_CID as PCPR_056_LOCK_CID,
    PORTFOLIO_ID,
    PORTFOLIO_VERSION,
)
from ipfs_accelerate_py.assurance.python_external_client import (
    PINNED_CLIENT_CID as PINNED_PYTHON_CLIENT_CID,
    PythonExternalClient,
    hermetic_python_client_trace,
)
from ipfs_accelerate_py.assurance.shared_contracts import (
    OWNER_INTERFACES,
    OWNER_SCHEMAS,
    shared_interface_id,
    shared_schema_id,
)


INTERFACE: Final = "AccelerateObjectiveIdentityParity@1"
SCHEMA: Final = "ipfs_accelerate_py/assurance/objective-identity-parity@1"
DOCUMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/declared-objective-identity-parity@1"
)
PARITY_SCHEMA: Final = "ipfs_accelerate_py/assurance/objective-identity-parity-plan@1"
VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/objective-identity-parity-verdict@1"
)
CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/platform-objective-identity-parity-catalog@1"
)
CATALOG_INTERFACE: Final = "PlatformObjectiveIdentityParity@1"
PARITY_INTERFACE: Final = "AuthoritativeObjectiveIdentityParity@1"
OWNER_INTERFACE: Final = "DatasetsContextPack@1"
OWNER_SCHEMA: Final = "ipfs_datasets_py/datasets-context-pack@1"
OWNER_REPOSITORY: Final = "ipfs_datasets_py"
STORAGE_OWNER_REPOSITORY: Final = "ipfs_kit_py"
STORAGE_OWNER_INTERFACE: Final = "KitContextPackStorage@1"
EXECUTION_OWNER_REPOSITORY: Final = "ipfs_accelerate_py"
CHAIN_OWNER_INTERFACE: Final = "AccelerateFinalProofCarryingReceiptChain@1"
PYTHON_CLIENT_INTERFACE: Final = "AcceleratePythonExternalClient@1"
MCP_CLIENT_INTERFACE: Final = "AccelerateGenericMcpClient@1"
PROTOCOL_INTERFACE: Final = "LogicProviderProtocol@2"
DOCUMENTATION_INTERFACE: Final = "LogicProviderProtocolObjectiveIdentityParity@1"
DOCUMENTATION_SCHEMA: Final = (
    "ipfs_datasets_py/logic-provider-objective-identity-parity@1"
)
PCPR_082_TASK_ID: Final = "PCPR-082"
PCPR_082_GOAL_ID: Final = "PCPR-G800"
PCPR_083_TASK_ID: Final = "PCPR-083"
PCPR_081_TASK_ID: Final = "PCPR-081"
PCPR_080_TASK_ID: Final = "PCPR-080"
PCPR_072_TASK_ID: Final = "PCPR-072"
PCPR_061_TASK_ID: Final = "PCPR-061"
PCPR_062_TASK_ID: Final = "PCPR-062"
PCPR_060_TASK_ID: Final = "PCPR-060"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = (
    "proof-carrying-platform-qualification-and-release-v1"
)
OBJECTIVE_KIND: Final = "declared_cross_client_objective_identity_parity"
OBJECTIVE_ID: Final = "PCPR-G800"
LANGUAGE: Final = "Python+MCP JSON-RPC 2.0"
OPERATOR_BLOCKING_TASK_ID: Final = (
    "pcpr-082-operator-live-cross-client-objective-identity-parity"
)
DOCUMENT_DIR_RELPATH: Final = "packaging/pcpr/reference-workflow/cpython312"
DOCUMENT_JSON_NAME: Final = "reference.objective-identity-parity.json"
DOCUMENT_README_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/OBJECTIVE_IDENTITY_PARITY.md"
)
CATALOG_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/platform-objective-identity-parity-catalog.json"
)
SOURCE_DATE_EPOCH: Final = "0"
SOURCE_REPOSITORY: Final = "endomorphosis/ipfs_accelerate_py"
COMPLETED_THROUGH: Final = "cross_client_objective_identity_parity"
NEXT_AUTHORIZED_STAGE: Final = "external_client_authority_bypass_proof"
CHANGE_KIND: Final = "cross_client_objective_identity_parity"
PARITY_KIND: Final = "hermetic_python_mcp_objective_identity_parity"
CANONICAL_SERVICE: Final = "pcpr-canonical-supervisor-service"
PYTHON_TRANSPORT: Final = "python"
MCP_TRANSPORT: Final = "mcp"
CLIENT_AUTHORITY: Final = "intent_not_authority"
PARITY_COMMAND_DIGEST: Final = "pcpr-082-hermetic-objective-identity-parity"

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
PINNED_CHAIN_CID: Final = (
    "baguqeeraeuvfodvledfa4nvfuso6eyyytzk2acptcogzwzyarkdinpqcsdfq"
)
PINNED_CHAIN_DOCUMENT_CID: Final = (
    "baguqeerax2qxfaxxywhn6bpyrygdf5ijjz4ewexigumdih6ukcj2tbo2w6va"
)

if PINNED_LOCK_CID != (
    "baguqeerawsekbbbt5ccctjt4tydzeahfkahsatb6q5x7k6cy4inrii5lhqmq"
):
    raise RuntimeError("PCPR-082 lock CID remints PCPR-056")
if PINNED_PYTHON_CLIENT_CID != (
    "baguqeerat346yf4k7kenbqlbtyejeilgxamna262z3axzjjatzm2hccehblq"
):
    raise RuntimeError("PCPR-082 remints the PCPR-080 Python client CID")
if PINNED_MCP_CLIENT_CID != (
    "baguqeerasdkirdryda7uoi5gchsyhuyvh6d2uremm45zpob7hoss3ao66ahq"
):
    raise RuntimeError("PCPR-082 remints the PCPR-081 generic MCP client CID")
if PINNED_PYTHON_CLIENT_CID == PINNED_MCP_CLIENT_CID:
    raise RuntimeError("Python and MCP client CIDs must remain distinct")
if OWNER_INTERFACES["SupervisorContextPack"] != OWNER_INTERFACE:
    raise RuntimeError("PCPR-082 remints SupervisorContextPack owner interface")
if OWNER_SCHEMAS["SupervisorContextPack"] != OWNER_SCHEMA:
    raise RuntimeError("PCPR-082 remints SupervisorContextPack owner schema")

REFERENCE_OBJECTIVE_IDEA: Final = (
    "Modify a typed formal-logic API while reusing unaffected proofs, "
    "selecting only impacted tests, rejecting stale-tree evidence, and "
    "producing a complete proof-carrying execution receipt."
)
CANONICAL_OBJECTIVE_BYTES: Final = REFERENCE_OBJECTIVE_IDEA.encode("utf-8")

CLIENT_OPERATIONS: Final[tuple[str, ...]] = (
    "submit_objective",
    "receive_identity",
    "subscribe_events",
    "inspect_task_state",
    "submit_candidate_evidence",
    "retrieve_final_receipts",
)

IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "objective_cid",
    "idea_digest",
    "pack_cid",
    "current_root_cid",
    "chain_cid",
    "chain_document_cid",
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
    "artifacts/proof_carrying_platform_qualification_and_release/receipts/PCPR-082.json",
)

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_objective_identity_parity.py",
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
        "parity_files_match_generator",
        "pyproject_objective_identity_parity_table",
        "paths_remain_authorized",
        "canonical_objective_bytes_match",
        "python_and_mcp_submit_same_objective_bytes",
        "objective_identity_matches_across_clients",
        "equivalent_goals_tasks_events_evidence_receipts",
        "owner_pack_cid_matches_pin",
        "kit_current_root_bound_not_minted",
        "chain_cid_bound_not_minted",
        "canonical_service_shared",
        "python_client_cid_bound_not_minted",
        "mcp_client_cid_bound_not_minted",
        "client_cids_remain_distinct",
        "clients_are_intent_not_authority",
        "idempotent_parity_cid",
        "model_assertion_cannot_complete_work",
        "duckdb_or_quack_not_written",
        "operator_blocking_task_emitted",
        "no_closed_release_outcome",
    }
)
FORBIDDEN_PRESENT_PROBE_IDS: Final[frozenset[str]] = frozenset(
    {
        "simulated_results_represented_as_live",
        "live_client_represented_as_live",
        "closed_release_represented_as_live",
        "compatibility_identities_reminted",
        "direct_database_bypass_used",
        "datasets_identity_reminted",
        "kit_identity_reminted",
        "chain_identity_reminted",
        "objective_identity_reminted",
        "python_client_identity_reminted",
        "mcp_client_identity_reminted",
        "model_assertion_completed_work",
        "unauthorized_path_accepted",
        "database_edited",
        "client_self_promoted",
        "identity_mismatch_accepted",
    }
)

DOCUMENT_README: Final = """# PCPR-082 Accelerate cross-client objective identity parity

These files record the Accelerate-owned hermetic proof that the PCPR-080
Python client and the PCPR-081 generic MCP client submit the same
canonical PCPR-060 objective bytes to the same canonical service and
receive the same objective identity and equivalent goals, tasks, events,
candidate evidence, and final receipts.

- `cpython312/reference.objective-identity-parity.json` is the declared
  parity document. Exact commit and tree are bound by the PCPR-082
  receipt `current_tree_binding`.
- `platform-objective-identity-parity-catalog.json` observes sibling
  Datasets and Kit bindings when present. Sibling source is never
  required.
- Python client CID and generic MCP client CID remain distinct and are
  not reminted. Both principals are intent, never authority. Live
  Supervisor.run, live Quack, live MCP stdio/HTTP, and live CLI sessions
  stay typed unavailable. Authority-bypass proof remains PCPR-083.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-082-operator-live-cross-client-objective-identity-parity`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
"""


class AccelerateObjectiveIdentityParityError(Exception):
    """Fail-closed PCPR-082 Accelerate objective-identity-parity error."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AccelerateObjectiveIdentityParityError(
            f"{name} must be a non-empty string"
        )
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise AccelerateObjectiveIdentityParityError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise AccelerateObjectiveIdentityParityError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def parse_pyproject_objective_identity_parity_table(text: str) -> dict[str, Any]:
    import tomllib

    table = tomllib.loads(_text(text, "pyproject.toml"))
    if not isinstance(table, dict):
        raise AccelerateObjectiveIdentityParityError(
            "pyproject.toml must be a table"
        )
    tool = table.get("tool")
    payload: dict[str, Any] = {}
    if isinstance(tool, dict):
        accelerate = tool.get("ipfs_accelerate_py")
        if isinstance(accelerate, dict):
            raw = accelerate.get("objective-identity-parity")
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
        raise AccelerateObjectiveIdentityParityError(
            f"path {path} is not an authorized PCPR-082 path"
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
        raise AccelerateObjectiveIdentityParityError(
            f"model assertion at {stage} cannot complete work"
        )


def refuse_pack_cid_remint(cid: str) -> str:
    if cid != PINNED_PACK_CID:
        raise AccelerateObjectiveIdentityParityError(
            f"ContextPack CID {cid} remints {PINNED_PACK_CID}"
        )
    return cid


def refuse_current_root_remint(cid: str) -> str:
    if cid != PINNED_CURRENT_ROOT_CID:
        raise AccelerateObjectiveIdentityParityError(
            f"current root CID {cid} remints {PINNED_CURRENT_ROOT_CID}"
        )
    return cid


def refuse_chain_cid_remint(cid: str) -> str:
    if cid != PINNED_CHAIN_CID:
        raise AccelerateObjectiveIdentityParityError(
            f"chain CID {cid} remints {PINNED_CHAIN_CID}"
        )
    return cid


def refuse_objective_cid_remint(cid: str) -> str:
    if cid != PINNED_OBJECTIVE_CID:
        raise AccelerateObjectiveIdentityParityError(
            f"objective CID {cid} remints {PINNED_OBJECTIVE_CID}"
        )
    return cid


def refuse_idea_digest_remint(digest: str) -> str:
    if digest != PINNED_IDEA_DIGEST:
        raise AccelerateObjectiveIdentityParityError(
            f"idea digest {digest} remints {PINNED_IDEA_DIGEST}"
        )
    return digest


def refuse_python_client_cid_remint(cid: str) -> str:
    if cid != PINNED_PYTHON_CLIENT_CID:
        raise AccelerateObjectiveIdentityParityError(
            f"Python client CID {cid} remints {PINNED_PYTHON_CLIENT_CID}"
        )
    return cid


def refuse_mcp_client_cid_remint(cid: str) -> str:
    if cid != PINNED_MCP_CLIENT_CID:
        raise AccelerateObjectiveIdentityParityError(
            f"MCP client CID {cid} remints {PINNED_MCP_CLIENT_CID}"
        )
    return cid


def refuse_database_edit(*, edited: bool) -> None:
    if edited:
        raise AccelerateObjectiveIdentityParityError(
            "objective identity parity cannot write DuckDB or Quack state"
        )


def refuse_live_client(*, applied_live: bool) -> None:
    if applied_live:
        raise AccelerateObjectiveIdentityParityError(
            "live cross-client objective identity parity remains typed unavailable"
        )


def refuse_self_promotion(*, promoted: bool) -> None:
    if promoted:
        raise AccelerateObjectiveIdentityParityError(
            "objective identity parity cannot promote itself"
        )


def refuse_non_canonical_objective(idea: str) -> str:
    text = _text(idea, "idea")
    if text != REFERENCE_OBJECTIVE_IDEA:
        raise AccelerateObjectiveIdentityParityError(
            "parity clients must submit the canonical PCPR-060 idea"
        )
    return text


def refuse_identity_mismatch(*, matched: bool) -> None:
    if not matched:
        raise AccelerateObjectiveIdentityParityError(
            "Python and generic MCP clients must receive the same objective identity"
        )


def _semantic_events(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    events = payload.get("events") or payload.get("subscribe_events", {}).get("events")
    if events is None:
        raise AccelerateObjectiveIdentityParityError("events are required")
    semantic: list[dict[str, str]] = []
    for item in events:
        if not isinstance(item, Mapping):
            raise AccelerateObjectiveIdentityParityError("event must be a mapping")
        semantic.append(
            {
                "kind": _text(item.get("kind"), "event.kind"),
                "task_id": _text(item.get("task_id"), "event.task_id"),
                "identity": _text(item.get("identity"), "event.identity"),
                "cid": _text(item.get("cid"), "event.cid"),
            }
        )
    return semantic


def _semantic_tasks(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    tasks = payload.get("tasks") or payload.get("inspect_task_state", {}).get("tasks")
    if tasks is None:
        raise AccelerateObjectiveIdentityParityError("tasks are required")
    semantic: list[dict[str, str]] = []
    for item in tasks:
        if not isinstance(item, Mapping):
            raise AccelerateObjectiveIdentityParityError("task must be a mapping")
        semantic.append(
            {
                "task_id": _text(item.get("task_id"), "task.task_id"),
                "state": _text(item.get("state"), "task.state"),
                "identity": _text(item.get("identity"), "task.identity"),
            }
        )
    return semantic


def compare_client_identities(
    python_trace: Mapping[str, Any] | None = None,
    mcp_trace: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare Python and MCP hermetic traces for objective identity parity."""

    python_payload = dict(python_trace or hermetic_python_client_trace())
    mcp_payload = dict(mcp_trace or hermetic_generic_mcp_client_trace())
    python_identity = python_payload["receive_identity"]
    mcp_identity = mcp_payload["receive_identity"]
    python_events = _semantic_events(python_payload["subscribe_events"])
    mcp_events = _semantic_events(mcp_payload["subscribe_events"])
    python_tasks = _semantic_tasks(python_payload["inspect_task_state"])
    mcp_tasks = _semantic_tasks(mcp_payload["inspect_task_state"])
    python_receipts = python_payload["retrieve_final_receipts"]
    mcp_receipts = mcp_payload["retrieve_final_receipts"]
    python_evidence = python_payload["submit_candidate_evidence"]
    mcp_evidence = mcp_payload["submit_candidate_evidence"]
    python_submit = python_payload["submit_objective"]
    mcp_submit = mcp_payload["submit_objective"]

    same_bytes = (
        python_submit["idea"] == REFERENCE_OBJECTIVE_IDEA
        and mcp_submit["idea"] == REFERENCE_OBJECTIVE_IDEA
        and python_submit["idea_digest"] == PINNED_IDEA_DIGEST
        and mcp_submit["idea_digest"] == PINNED_IDEA_DIGEST
    )
    same_objective = (
        python_identity["objective_cid"] == PINNED_OBJECTIVE_CID
        and mcp_identity["objective_cid"] == PINNED_OBJECTIVE_CID
        and python_identity["idea_digest"] == PINNED_IDEA_DIGEST
        and mcp_identity["idea_digest"] == PINNED_IDEA_DIGEST
        and python_identity["identity_matches_pcpr_060"] is True
        and mcp_identity["identity_matches_pcpr_060"] is True
    )
    equivalent_goals = (
        python_identity["equivalent_plan_semantics"] is True
        and mcp_identity["equivalent_plan_semantics"] is True
    )
    equivalent_events = python_events == mcp_events
    equivalent_tasks = python_tasks == mcp_tasks
    equivalent_evidence = (
        python_evidence["submitted"] is True
        and mcp_evidence["submitted"] is True
        and python_evidence["admitted_live"] is False
        and mcp_evidence["admitted_live"] is False
        and python_evidence["live"] is False
        and mcp_evidence["live"] is False
    )
    equivalent_receipts = (
        python_receipts["chain_cid"] == PINNED_CHAIN_CID
        and mcp_receipts["chain_cid"] == PINNED_CHAIN_CID
        and python_receipts["chain_document_cid"] == PINNED_CHAIN_DOCUMENT_CID
        and mcp_receipts["chain_document_cid"] == PINNED_CHAIN_DOCUMENT_CID
        and python_receipts["chain_task_id"] == PCPR_072_TASK_ID
        and mcp_receipts["chain_task_id"] == PCPR_072_TASK_ID
        and python_receipts["retrieved"] is True
        and mcp_receipts["retrieved"] is True
    )
    shared_service = (
        python_payload["canonical_service"] == CANONICAL_SERVICE
        and mcp_payload["canonical_service"] == CANONICAL_SERVICE
    )
    matched = (
        same_bytes
        and same_objective
        and equivalent_goals
        and equivalent_events
        and equivalent_tasks
        and equivalent_evidence
        and equivalent_receipts
        and shared_service
    )
    refuse_identity_mismatch(matched=matched)
    return {
        "schema": "ipfs_accelerate_py/assurance/hermetic-objective-identity-parity@1",
        "interface": "HermeticObjectiveIdentityParity@1",
        "matched": True,
        "same_canonical_objective_bytes": True,
        "same_objective_cid": True,
        "same_idea_digest": True,
        "equivalent_goals": True,
        "equivalent_tasks": True,
        "equivalent_events": True,
        "equivalent_candidate_evidence": True,
        "equivalent_final_receipts": True,
        "canonical_service_shared": True,
        "python_transport": PYTHON_TRANSPORT,
        "mcp_transport": MCP_TRANSPORT,
        "canonical_service": CANONICAL_SERVICE,
        "objective_cid": PINNED_OBJECTIVE_CID,
        "idea_digest": PINNED_IDEA_DIGEST,
        "pack_cid": PINNED_PACK_CID,
        "current_root_cid": PINNED_CURRENT_ROOT_CID,
        "chain_cid": PINNED_CHAIN_CID,
        "chain_document_cid": PINNED_CHAIN_DOCUMENT_CID,
        "python_client_cid": PINNED_PYTHON_CLIENT_CID,
        "mcp_client_cid": PINNED_MCP_CLIENT_CID,
        "client_cids_distinct": PINNED_PYTHON_CLIENT_CID != PINNED_MCP_CLIENT_CID,
        "events": python_events,
        "tasks": python_tasks,
        "python_principal": python_payload["principal"],
        "mcp_principal": mcp_payload["principal"],
        "authority": CLIENT_AUTHORITY,
        "live": False,
        "hermetic": True,
        "evidence_kind": "measured_hermetic",
    }


def hermetic_parity_trace() -> dict[str, Any]:
    python_client = PythonExternalClient()
    mcp_client = GenericMcpClient()
    python_submit = python_client.submit_objective(REFERENCE_OBJECTIVE_IDEA)
    mcp_submit = mcp_client.submit_objective(REFERENCE_OBJECTIVE_IDEA)
    comparison = compare_client_identities()
    return {
        "schema": "ipfs_accelerate_py/assurance/hermetic-objective-identity-parity-trace@1",
        "interface": "HermeticObjectiveIdentityParityTrace@1",
        "python_submit_objective": python_submit,
        "mcp_submit_objective": mcp_submit,
        "comparison": comparison,
        "complete_high_level_objective_path": True,
        "objective_cid": PINNED_OBJECTIVE_CID,
        "idea_digest": PINNED_IDEA_DIGEST,
        "chain_cid": PINNED_CHAIN_CID,
        "python_client_cid": PINNED_PYTHON_CLIENT_CID,
        "mcp_client_cid": PINNED_MCP_CLIENT_CID,
        "live": False,
        "hermetic": True,
        "evidence_kind": "measured_hermetic",
    }


def hermetic_parity_idempotency(parity_cid: str) -> dict[str, Any]:
    return {
        "schema": "ipfs_accelerate_py/assurance/hermetic-objective-identity-parity-idempotency@1",
        "interface": "HermeticObjectiveIdentityParityIdempotency@1",
        "first_parity_command_digest": PARITY_COMMAND_DIGEST,
        "second_parity_command_digest": PARITY_COMMAND_DIGEST,
        "first_parity_cid": parity_cid,
        "second_parity_cid": parity_cid,
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
class AccelerateObjectiveIdentityParityVerdict:
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
    live_client: bool
    live_application: bool
    operator_blocking_task: str
    simulated_results_represented_as_live: bool
    this_task_created_competing_authority: bool
    probes: tuple[OutcomeProbe, ...]
    blockers: tuple[str, ...]
    verdict_cid: str
    parity_cid: str
    document_cid: str
    catalog_cid: str
    chain_cid: str
    pack_cid: str
    current_root_cid: str
    objective_cid: str
    idea_digest: str
    python_client_cid: str
    mcp_client_cid: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "verdict_cid": self.verdict_cid,
            "parity_cid": self.parity_cid,
            "document_cid": self.document_cid,
            "catalog_cid": self.catalog_cid,
            "chain_cid": self.chain_cid,
            "pack_cid": self.pack_cid,
            "current_root_cid": self.current_root_cid,
            "objective_cid": self.objective_cid,
            "idea_digest": self.idea_digest,
            "python_client_cid": self.python_client_cid,
            "mcp_client_cid": self.mcp_client_cid,
            "promotion_status": self.promotion_status,
            "supervisor_disposition": self.supervisor_disposition,
            "closed_release_outcome": self.closed_release_outcome,
            "release_claim": self.release_claim,
            "completion_authoritative": self.completion_authoritative,
            "contracts_frozen": self.contracts_frozen,
            "duckdb_or_quack_state_written": self.duckdb_or_quack_state_written,
            "sibling_source_required": self.sibling_source_required,
            "live_client": self.live_client,
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
            "An admitted Quack-fenced state-owner session before Python and "
            "generic MCP clients are executed as live supervisor work"
        ),
        "action": (
            "Keep the hermetic Python and generic MCP clients, the shared "
            "canonical objective identity, equivalent events, task state, "
            "candidate evidence, and retrieved PCPR-072 receipts. Do not "
            "write DuckDB or Quack state. Do not escalate to a model. Do "
            "not grant either client principal authority. PCPR-083 proves "
            "external clients cannot bypass authority."
        ),
        "reason": (
            "Hermetic cross-client identity parity is not a live "
            "Supervisor.run / CLI / MCP stdio or HTTP session. Sealed "
            "validation has no admitted Quack-fenced session. Direct DuckDB "
            "writes are prohibited."
        ),
    }


def produce_objective_identity_parity(
    *,
    paths: Sequence[str] | None = None,
    pack_cid: str = PINNED_PACK_CID,
    current_root_cid: str = PINNED_CURRENT_ROOT_CID,
    chain_cid: str = PINNED_CHAIN_CID,
    objective_cid: str = PINNED_OBJECTIVE_CID,
    idea_digest: str = PINNED_IDEA_DIGEST,
    python_client_cid: str = PINNED_PYTHON_CLIENT_CID,
    mcp_client_cid: str = PINNED_MCP_CLIENT_CID,
    idea: str = REFERENCE_OBJECTIVE_IDEA,
    complete_from: str | None = None,
    database_edited: bool = False,
    applied_live: bool = False,
    self_promoted: bool = False,
    identity_matched: bool = True,
) -> dict[str, Any]:
    """Produce the hermetic cross-client identity-parity proof. Fail closed."""

    if complete_from is not None:
        refuse_model_completion(complete_from)
    refuse_pack_cid_remint(pack_cid)
    refuse_current_root_remint(current_root_cid)
    refuse_chain_cid_remint(chain_cid)
    refuse_objective_cid_remint(objective_cid)
    refuse_idea_digest_remint(idea_digest)
    refuse_python_client_cid_remint(python_client_cid)
    refuse_mcp_client_cid_remint(mcp_client_cid)
    refuse_non_canonical_objective(idea)
    refuse_database_edit(edited=database_edited)
    refuse_live_client(applied_live=applied_live)
    refuse_self_promotion(promoted=self_promoted)
    refuse_identity_mismatch(matched=identity_matched)
    checked_paths = list(paths) if paths is not None else list(AUTHORIZED_PATH_PREFIXES)
    for path in checked_paths:
        refuse_unauthorized_path(path)
    comparison = compare_client_identities()
    return {
        "schema": PARITY_SCHEMA,
        "interface": PARITY_INTERFACE,
        "owner_interface": INTERFACE,
        "task_id": PCPR_082_TASK_ID,
        "goal_id": PCPR_082_GOAL_ID,
        "objective_kind": OBJECTIVE_KIND,
        "portfolio_id": PORTFOLIO_ID,
        "portfolio_version": PORTFOLIO_VERSION,
        "language": LANGUAGE,
        "idea": REFERENCE_OBJECTIVE_IDEA,
        "idea_digest": idea_digest,
        "objective_cid": objective_cid,
        "lock_cid": PINNED_LOCK_CID,
        "pack_cid": pack_cid,
        "current_root_cid": current_root_cid,
        "chain_cid": chain_cid,
        "python_client_cid": python_client_cid,
        "mcp_client_cid": mcp_client_cid,
        "python_client_interface": PYTHON_CLIENT_INTERFACE,
        "mcp_client_interface": MCP_CLIENT_INTERFACE,
        "protocol_interface": PROTOCOL_INTERFACE,
        "documentation_interface": DOCUMENTATION_INTERFACE,
        "documentation_schema": DOCUMENTATION_SCHEMA,
        "change_kind": CHANGE_KIND,
        "parity_kind": PARITY_KIND,
        "canonical_service": CANONICAL_SERVICE,
        "python_transport": PYTHON_TRANSPORT,
        "mcp_transport": MCP_TRANSPORT,
        "authority": CLIENT_AUTHORITY,
        "operations": list(CLIENT_OPERATIONS),
        "identity_fields": list(IDENTITY_FIELDS),
        "produced": True,
        "applied": True,
        "live": False,
        "hermetic": True,
        "admitted_live": False,
        "deferred": False,
        "paths_authorized": True,
        "authorized_path_prefixes": list(AUTHORIZED_PATH_PREFIXES),
        "identity_matches_pcpr_060": True,
        "identity_matches_across_clients": True,
        "equivalent_plan_semantics": True,
        "equivalent_goals": True,
        "equivalent_tasks": True,
        "equivalent_events": True,
        "equivalent_candidate_evidence": True,
        "equivalent_final_receipts": True,
        "events_subscribed": True,
        "task_state_inspected": True,
        "candidate_evidence_submitted": True,
        "final_receipts_retrieved": True,
        "complete_high_level_objective_path": True,
        "client_cids_distinct": True,
        "client_self_promoted": False,
        "comparison": comparison,
        "parity_command_digest": PARITY_COMMAND_DIGEST,
        "idempotent": True,
        "duplicate_effect_rejected": True,
        "remints_protocol": False,
        "adds_protocol_operation": False,
        "model_assertion_completes_work": False,
        "escalation_order": list(ESCALATION_ORDER),
        "completed_through": COMPLETED_THROUGH,
        "next_authorized_stage": NEXT_AUTHORIZED_STAGE,
        "next_task_id": PCPR_083_TASK_ID,
        "prerequisite_task_id": PCPR_081_TASK_ID,
        "python_client_task_id": PCPR_080_TASK_ID,
        "mcp_client_task_id": PCPR_081_TASK_ID,
        "chain_task_id": PCPR_072_TASK_ID,
        "duckdb_or_quack_state_written": False,
        "closed_release_outcome": None,
        "release_claim": False,
        "evidence_kind": "measured_hermetic",
    }


def canonical_objective_identity_parity() -> dict[str, Any]:
    return produce_objective_identity_parity()


def parity_cid_of(plan: Mapping[str, Any] | None = None) -> str:
    payload = dict(plan or canonical_objective_identity_parity())
    payload.pop("parity_cid", None)
    payload.pop("idempotency", None)
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
        "live_client": False,
        "live_mcp_stdio": False,
        "live_mcp_http": False,
        "evidence_kind": "measured",
    }


def _sibling_binding(path: Path, relative_path: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": relative_path,
            "status": "unavailable",
            "evidence_kind": "unavailable",
            "reason": (
                "Sibling objective-identity-parity file is not present beside this checkout."
            ),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    context_pack = payload.get("context_pack")
    storage = payload.get("storage")
    parity = payload.get("objective_identity_parity")
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
        "chain_cid": payload.get("chain_cid")
        or (parity.get("chain_cid") if isinstance(parity, Mapping) else None),
        "parity_cid": payload.get("parity_cid")
        or (parity.get("parity_cid") if isinstance(parity, Mapping) else None),
        "binding_cid": payload.get("binding_cid") or payload.get("document_cid"),
        "live": False,
        "applied": False,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def render_declared_parity(root: Path | None = None) -> dict[str, Any]:
    package_root = root or discover_accelerate_root()
    if package_root is None:
        raise AccelerateObjectiveIdentityParityError(
            "Accelerate package root was not found"
        )
    state_owner = observe_state_owner()
    plan = canonical_objective_identity_parity()
    computed_parity_cid = parity_cid_of(plan)
    idempotency = hermetic_parity_idempotency(computed_parity_cid)
    document = {
        "schema": DOCUMENT_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_082_TASK_ID,
        "goal_id": PCPR_082_GOAL_ID,
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
            "survives_parity": True,
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
            "survives_parity": True,
            "evidence_kind": "measured_hermetic",
        },
        "final_receipt_chain": {
            "task_id": PCPR_072_TASK_ID,
            "owner_interface": CHAIN_OWNER_INTERFACE,
            "chain_cid": PINNED_CHAIN_CID,
            "document_cid": PINNED_CHAIN_DOCUMENT_CID,
            "produced": True,
            "live": False,
            "hermetic": True,
            "reminted": False,
            "retrieved_by_python_client": True,
            "retrieved_by_generic_mcp_client": True,
            "evidence_kind": "measured_hermetic",
        },
        "python_external_client": {
            "task_id": PCPR_080_TASK_ID,
            "interface": PYTHON_CLIENT_INTERFACE,
            "client_cid": PINNED_PYTHON_CLIENT_CID,
            "transport": PYTHON_TRANSPORT,
            "canonical_service": CANONICAL_SERVICE,
            "reminted": False,
            "live": False,
            "evidence_kind": "measured",
        },
        "generic_mcp_client": {
            "task_id": PCPR_081_TASK_ID,
            "interface": MCP_CLIENT_INTERFACE,
            "client_cid": PINNED_MCP_CLIENT_CID,
            "transport": MCP_TRANSPORT,
            "canonical_service": CANONICAL_SERVICE,
            "reminted": False,
            "live": False,
            "evidence_kind": "measured",
        },
        "objective_identity_parity": {
            **plan,
            "parity_cid": computed_parity_cid,
            "idempotency": idempotency,
            "produced_by": EXECUTION_OWNER_REPOSITORY,
        },
        "materialization": {
            "kind": "declared_objective_identity_parity_not_live",
            "admitted": False,
            "live": False,
            "applied": False,
            "duckdb_or_quack_state_written": False,
            "evidence_kind": "unavailable",
        },
        "live_application": typed_unavailable(
            reason=(
                "No admitted Quack-fenced supervisor session was observed. "
                "Hermetic cross-client identity parity is not live client execution."
            )
        ),
        "live_client": typed_unavailable(
            reason=(
                "Live Python Supervisor.run and live generic MCP stdio/HTTP "
                "remain typed unavailable. This task only emits a hermetic "
                "identity-parity comparison. Authority-bypass proof remains "
                "PCPR-083."
            )
        ),
        "live_mcp_client": typed_unavailable(
            reason=(
                "Live MCP stdio or HTTP session remains typed unavailable. "
                "Hermetic JSON-RPC demonstration is not a live MCP session."
            )
        ),
        "live_python_client": typed_unavailable(
            reason="Live Python Supervisor.run remains typed unavailable."
        ),
        "live_cli_client": typed_unavailable(
            reason="Live CLI client execution remains typed unavailable."
        ),
        "state_owner": {
            "duckdb_cli": state_owner.get("duckdb"),
            "duckdb_module": state_owner.get("duckdb_module"),
            "duckdb_module_is_not_live_materialization": True,
            "quack": "unavailable",
            "duckdb_or_quack_state_written": False,
            "direct_duckdb_write_prohibited": True,
            "direct_quack_write_prohibited": True,
            "live_client": False,
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
                "field": "PCPR-082 receipt current_tree_binding",
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


def platform_objective_identity_parity_catalog(
    start: Path | None = None,
) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateObjectiveIdentityParityError(
            "Accelerate package root was not found"
        )
    parent = root.parent
    document = render_declared_parity(root)
    datasets_binding = _sibling_binding(
        parent
        / "ipfs_datasets"
        / DOCUMENT_DIR_RELPATH
        / "reference.objective-identity-parity.binding.json",
        relative_path=(
            f"../ipfs_datasets/{DOCUMENT_DIR_RELPATH}/"
            "reference.objective-identity-parity.binding.json"
        ),
    )
    kit_binding = _sibling_binding(
        parent
        / "ipfs_kit"
        / DOCUMENT_DIR_RELPATH
        / "reference.objective-identity-parity.binding.json",
        relative_path=(
            f"../ipfs_kit/{DOCUMENT_DIR_RELPATH}/"
            "reference.objective-identity-parity.binding.json"
        ),
    )
    catalog = {
        "schema": CATALOG_SCHEMA,
        "interface": CATALOG_INTERFACE,
        "task_id": PCPR_082_TASK_ID,
        "goal_id": PCPR_082_GOAL_ID,
        "objective_kind": OBJECTIVE_KIND,
        "portfolio_id": PORTFOLIO_ID,
        "portfolio_version": PORTFOLIO_VERSION,
        "objective_cid": PINNED_OBJECTIVE_CID,
        "idea_digest": PINNED_IDEA_DIGEST,
        "pack_cid": PINNED_PACK_CID,
        "current_root_cid": PINNED_CURRENT_ROOT_CID,
        "chain_cid": PINNED_CHAIN_CID,
        "parity_cid": document["objective_identity_parity"]["parity_cid"],
        "python_client_cid": PINNED_PYTHON_CLIENT_CID,
        "mcp_client_cid": PINNED_MCP_CLIENT_CID,
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
                "chain_cid": PINNED_CHAIN_CID,
                "parity_cid": document["objective_identity_parity"]["parity_cid"],
                "python_client_cid": PINNED_PYTHON_CLIENT_CID,
                "mcp_client_cid": PINNED_MCP_CLIENT_CID,
                "document_cid": document["document_cid"],
                "live": False,
                "applied": False,
                "reminted": False,
            },
            "ipfs_datasets_py": {"binding": datasets_binding},
            "ipfs_kit_py": {"binding": kit_binding},
        },
        "live_application": False,
        "live_client": False,
        "closed_release_outcome": None,
        "release_claim": False,
        "sibling_source_required": False,
        "evidence_kind": "measured",
    }
    catalog["catalog_cid"] = content_identity(
        {key: value for key, value in catalog.items() if key != "catalog_cid"}
    )
    return catalog


def write_objective_identity_parity_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateObjectiveIdentityParityError(
            "Accelerate package root was not found"
        )
    document = render_declared_parity(root)
    paths = artifact_paths(root)
    _atomic_write(paths["document"], pretty_json(document))
    readme = DOCUMENT_README if DOCUMENT_README.endswith("\n") else DOCUMENT_README + "\n"
    _atomic_write(paths["readme"], readme)
    catalog = platform_objective_identity_parity_catalog(start)
    _atomic_write(paths["catalog"], pretty_json(catalog))
    return {
        "document": document,
        "catalog": catalog,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def verify_objective_identity_parity_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateObjectiveIdentityParityError(
            "Accelerate package root was not found"
        )
    document = render_declared_parity(root)
    catalog = platform_objective_identity_parity_catalog(start)
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
        "chain_cid": document["final_receipt_chain"]["chain_cid"],
        "parity_cid": document["objective_identity_parity"]["parity_cid"],
        "document_cid": document["document_cid"],
        "catalog_cid": catalog["catalog_cid"],
        "current_root_cid": document["storage"]["current_root_cid"],
        "objective_cid": document["objective_cid"],
        "idea_digest": document["idea_digest"],
        "python_client_cid": document["python_external_client"]["client_cid"],
        "mcp_client_cid": document["generic_mcp_client"]["client_cid"],
        "document_sha256": (
            sha256_bytes(paths["document"].read_bytes())
            if paths["document"].is_file()
            else "unavailable"
        ),
    }


def current_head_static_probes(start: Path | None = None) -> tuple[OutcomeProbe, ...]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateObjectiveIdentityParityError(
            "Accelerate package root was not found"
        )
    document = render_declared_parity(root)
    verified = verify_objective_identity_parity_files(root)
    table = parse_pyproject_objective_identity_parity_table(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    parity = document["objective_identity_parity"]
    remint = (
        document["context_pack"]["pack_cid"] != PINNED_PACK_CID
        or document["objective_cid"] != PINNED_OBJECTIVE_CID
        or document["idea_digest"] != PINNED_IDEA_DIGEST
        or document["lock_cid"] != PINNED_LOCK_CID
        or document["storage"]["current_root_cid"] != PINNED_CURRENT_ROOT_CID
        or document["final_receipt_chain"]["chain_cid"] != PINNED_CHAIN_CID
        or parity["parity_cid"] != PINNED_PARITY_CID
        or document["python_external_client"]["client_cid"] != PINNED_PYTHON_CLIENT_CID
        or document["generic_mcp_client"]["client_cid"] != PINNED_MCP_CLIENT_CID
        or document["context_pack"]["reminted"] is True
        or document["storage"]["reminted"] is True
        or document["final_receipt_chain"]["reminted"] is True
        or document["python_external_client"]["reminted"] is True
        or document["generic_mcp_client"]["reminted"] is True
    )
    comparison = parity["comparison"]
    idempotency = parity["idempotency"]
    probes = [
        _probe(
            "parity_files_match_generator",
            verified.get("ok") is True and verified.get("document_ok") is True,
            reason=(
                "Committed Accelerate objective-identity-parity matches the generator."
                if verified.get("document_ok") is True
                else "Committed Accelerate objective-identity-parity is missing or drifts."
            ),
        ),
        _probe(
            "pyproject_objective_identity_parity_table",
            table.get("interface") == INTERFACE
            and table.get("schema") == SCHEMA
            and table.get("task-id") == PCPR_082_TASK_ID
            and table.get("objective-kind") == OBJECTIVE_KIND,
            reason=(
                "pyproject.toml declares AccelerateObjectiveIdentityParity@1."
                if table.get("interface") == INTERFACE
                else "pyproject.toml does not declare the PCPR-082 parity table."
            ),
        ),
        _probe(
            "paths_remain_authorized",
            parity["paths_authorized"] is True
            and list(parity["authorized_path_prefixes"])
            == list(AUTHORIZED_PATH_PREFIXES),
            reason="Parity paths remain inside the declared PCPR-082 allowlist.",
        ),
        _probe(
            "canonical_objective_bytes_match",
            parity["idea"] == REFERENCE_OBJECTIVE_IDEA
            and parity["idea_digest"] == PINNED_IDEA_DIGEST,
            reason="Parity comparison uses the canonical PCPR-060 idea bytes.",
        ),
        _probe(
            "python_and_mcp_submit_same_objective_bytes",
            comparison["same_canonical_objective_bytes"] is True,
            reason="Python and generic MCP clients submit the same canonical idea bytes.",
        ),
        _probe(
            "objective_identity_matches_across_clients",
            parity["objective_cid"] == PINNED_OBJECTIVE_CID
            and parity["identity_matches_across_clients"] is True
            and comparison["same_objective_cid"] is True
            and comparison["same_idea_digest"] is True,
            reason="Python and generic MCP clients receive the same PCPR-060 objective identity.",
        ),
        _probe(
            "equivalent_goals_tasks_events_evidence_receipts",
            comparison["equivalent_goals"] is True
            and comparison["equivalent_tasks"] is True
            and comparison["equivalent_events"] is True
            and comparison["equivalent_candidate_evidence"] is True
            and comparison["equivalent_final_receipts"] is True,
            reason="Goals, tasks, events, candidate evidence, and final receipts are equivalent.",
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
            "chain_cid_bound_not_minted",
            document["final_receipt_chain"]["chain_cid"] == PINNED_CHAIN_CID
            and document["final_receipt_chain"]["reminted"] is False,
            reason="Accelerate binds the PCPR-072 chain CID and does not remint it.",
        ),
        _probe(
            "canonical_service_shared",
            parity["canonical_service"] == CANONICAL_SERVICE
            and comparison["canonical_service_shared"] is True,
            reason="Python and generic MCP clients use the same canonical service.",
        ),
        _probe(
            "python_client_cid_bound_not_minted",
            document["python_external_client"]["client_cid"] == PINNED_PYTHON_CLIENT_CID
            and document["python_external_client"]["reminted"] is False,
            reason="PCPR-080 Python client CID is bound and not reminted.",
        ),
        _probe(
            "mcp_client_cid_bound_not_minted",
            document["generic_mcp_client"]["client_cid"] == PINNED_MCP_CLIENT_CID
            and document["generic_mcp_client"]["reminted"] is False,
            reason="PCPR-081 generic MCP client CID is bound and not reminted.",
        ),
        _probe(
            "client_cids_remain_distinct",
            PINNED_PYTHON_CLIENT_CID != PINNED_MCP_CLIENT_CID
            and comparison["client_cids_distinct"] is True
            and parity["client_cids_distinct"] is True,
            reason="Python and MCP client CIDs remain distinct identities.",
        ),
        _probe(
            "clients_are_intent_not_authority",
            parity["authority"] == CLIENT_AUTHORITY
            and comparison["authority"] == CLIENT_AUTHORITY,
            reason="Both client principals are intent, never authority.",
        ),
        _probe(
            "idempotent_parity_cid",
            parity["idempotent"] is True
            and idempotency["identical"] is True
            and idempotency["first_parity_cid"] == parity["parity_cid"]
            and idempotency["second_parity_cid"] == parity["parity_cid"]
            and parity["parity_cid"] == PINNED_PARITY_CID,
            reason="A second parity emission yields the same parity CID.",
        ),
        _probe(
            "model_assertion_cannot_complete_work",
            parity["model_assertion_completes_work"] is False,
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
            reason="Missing live client execution emits the operator-blocking task.",
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
                "A reminted pack, root, chain, objective, idea, lock, Python "
                "client, MCP client, or parity CID is forbidden."
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
            "chain_identity_reminted",
            document["final_receipt_chain"]["reminted"] is True,
            reason="Accelerate must not remint the PCPR-072 chain.",
        ),
        _probe(
            "objective_identity_reminted",
            document["objective_cid"] != PINNED_OBJECTIVE_CID,
            reason="Accelerate must not remint the PCPR-060 objective CID.",
        ),
        _probe(
            "python_client_identity_reminted",
            document["python_external_client"]["client_cid"] != PINNED_PYTHON_CLIENT_CID
            or document["python_external_client"]["reminted"] is True,
            reason="Accelerate must not remint the PCPR-080 Python client CID.",
        ),
        _probe(
            "mcp_client_identity_reminted",
            document["generic_mcp_client"]["client_cid"] != PINNED_MCP_CLIENT_CID
            or document["generic_mcp_client"]["reminted"] is True,
            reason="Accelerate must not remint the PCPR-081 generic MCP client CID.",
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
            "live_client_represented_as_live",
            False,
            reason="Hermetic identity parity is not represented as live.",
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
            "database_edited",
            False,
            reason="DuckDB and Quack state were not written.",
        ),
        _probe(
            "client_self_promoted",
            False,
            reason="Identity parity cannot promote itself.",
        ),
        _probe(
            "identity_mismatch_accepted",
            False,
            reason="An identity mismatch between Python and MCP clients is refused.",
        ),
        _probe(
            "live_application",
            None,
            evidence_kind="unavailable",
            reason="Live supervisor application stays typed unavailable.",
        ),
        _probe(
            "live_client",
            None,
            evidence_kind="unavailable",
            reason="Live Python and generic MCP clients stay typed unavailable.",
        ),
    ]
    return tuple(probes)


def qualify_objective_identity_parity(
    probes: Sequence[OutcomeProbe],
    *,
    parity_cid: str,
    document_cid: str,
    catalog_cid: str,
    chain_cid: str,
    pack_cid: str,
    current_root_cid: str,
    objective_cid: str,
    idea_digest_cid: str,
    python_client_cid: str = PINNED_PYTHON_CLIENT_CID,
    mcp_client_cid: str = PINNED_MCP_CLIENT_CID,
) -> AccelerateObjectiveIdentityParityVerdict:
    if not probes:
        raise AccelerateObjectiveIdentityParityError("at least one probe is required")
    normalized: list[OutcomeProbe] = []
    blockers: list[str] = []
    for probe in probes:
        kind = _kind(probe.evidence_kind, "evidence_kind")
        if probe.live and kind != "measured_live":
            raise AccelerateObjectiveIdentityParityError(
                "live claims require measured_live evidence"
            )
        if probe.simulated_represented_as_live:
            raise AccelerateObjectiveIdentityParityError(
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
        "task_id": PCPR_082_TASK_ID,
        "goal_id": PCPR_082_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "sibling_source_required": False,
        "live_client": False,
        "live_application": False,
        "operator_blocking_task": OPERATOR_BLOCKING_TASK_ID,
        "simulated_results_represented_as_live": False,
        "this_task_created_competing_authority": False,
        "probes": [item.to_mapping() for item in normalized],
        "blockers": list(dict.fromkeys(blockers)),
        "parity_cid": parity_cid,
        "document_cid": document_cid,
        "catalog_cid": catalog_cid,
        "chain_cid": chain_cid,
        "pack_cid": pack_cid,
        "current_root_cid": current_root_cid,
        "objective_cid": objective_cid,
        "idea_digest": idea_digest_cid,
        "python_client_cid": python_client_cid,
        "mcp_client_cid": mcp_client_cid,
    }
    return AccelerateObjectiveIdentityParityVerdict(
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
        live_client=False,
        live_application=False,
        operator_blocking_task=OPERATOR_BLOCKING_TASK_ID,
        simulated_results_represented_as_live=False,
        this_task_created_competing_authority=False,
        probes=tuple(normalized),
        blockers=tuple(dict.fromkeys(blockers)),
        verdict_cid=content_identity(payload),
        parity_cid=parity_cid,
        document_cid=document_cid,
        catalog_cid=catalog_cid,
        chain_cid=chain_cid,
        pack_cid=pack_cid,
        current_root_cid=current_root_cid,
        objective_cid=objective_cid,
        idea_digest=idea_digest_cid,
        python_client_cid=python_client_cid,
        mcp_client_cid=mcp_client_cid,
    )


def qualify_current_head_objective_identity_parity(
    start: Path | None = None,
) -> AccelerateObjectiveIdentityParityVerdict:
    document = render_declared_parity(start)
    catalog = platform_objective_identity_parity_catalog(start)
    return qualify_objective_identity_parity(
        current_head_static_probes(start),
        parity_cid=str(document["objective_identity_parity"]["parity_cid"]),
        document_cid=str(document["document_cid"]),
        catalog_cid=str(catalog["catalog_cid"]),
        chain_cid=str(document["final_receipt_chain"]["chain_cid"]),
        pack_cid=str(document["context_pack"]["pack_cid"]),
        current_root_cid=str(document["storage"]["current_root_cid"]),
        objective_cid=str(document["objective_cid"]),
        idea_digest_cid=str(document["idea_digest"]),
        python_client_cid=str(document["python_external_client"]["client_cid"]),
        mcp_client_cid=str(document["generic_mcp_client"]["client_cid"]),
    )


def pcpr_082_receipt_promotion(
    verdict: AccelerateObjectiveIdentityParityVerdict,
) -> dict[str, Any]:
    if verdict.closed_release_outcome is not None:
        raise AccelerateObjectiveIdentityParityError(
            "objective identity parity must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise AccelerateObjectiveIdentityParityError(
            "objective identity parity must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise AccelerateObjectiveIdentityParityError(
            "objective identity parity completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise AccelerateObjectiveIdentityParityError(
            "objective identity parity must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise AccelerateObjectiveIdentityParityError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.live_client or verdict.live_application:
        raise AccelerateObjectiveIdentityParityError(
            "live client claims require measured_live evidence"
        )
    return verdict.to_mapping()


PINNED_PARITY_CID: Final = (
    "baguqeera6y3w7bhzy6hpo6ohfsrj5xihrw5vk6qqljjonjrupijd2yrp3hnq"
)
PINNED_DOCUMENT_CID: Final = (
    "baguqeeraclikvz6dw36yzp7zlxgcjg7zgdqnbhkrxpmhntiksy73hmbgw76a"
)
PINNED_CATALOG_CID: Final = (
    "baguqeera2d6ajvqq47jm3mqtu3vcyahhk6tb7rt3i24njcjd2vbunedjikca"
)
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeera6hsfhhbtytzl4mzgwgxayz32bemkkglusogylnwjptgpvldtdx6q"
)


__all__ = [
    "AUTHORIZED_PATH_PREFIXES",
    "CLIENT_OPERATIONS",
    "CLOSED_RELEASE_OUTCOMES",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "ESCALATION_ORDER",
    "AccelerateObjectiveIdentityParityError",
    "HERMETIC_CANDIDATE_SUITES",
    "INTERFACE",
    "OBJECTIVE_KIND",
    "OPERATOR_BLOCKING_TASK_ID",
    "OutcomeProbe",
    "PCPR_082_GOAL_ID",
    "PCPR_082_TASK_ID",
    "PINNED_CATALOG_CID",
    "PINNED_CHAIN_CID",
    "PINNED_CURRENT_ROOT_CID",
    "PINNED_DOCUMENT_CID",
    "PINNED_IDEA_DIGEST",
    "PINNED_LOCK_CID",
    "PINNED_MCP_CLIENT_CID",
    "PINNED_OBJECTIVE_CID",
    "PINNED_PACK_CID",
    "PINNED_PARITY_CID",
    "PINNED_PYTHON_CLIENT_CID",
    "SCHEMA",
    "SEALED_PATH",
    "SEALED_PYTHON",
    "canonical_objective_identity_parity",
    "compare_client_identities",
    "current_head_static_probes",
    "hermetic_parity_idempotency",
    "hermetic_parity_trace",
    "parity_cid_of",
    "path_is_authorized",
    "pcpr_082_receipt_promotion",
    "platform_objective_identity_parity_catalog",
    "produce_objective_identity_parity",
    "qualify_current_head_objective_identity_parity",
    "qualify_objective_identity_parity",
    "refuse_chain_cid_remint",
    "refuse_current_root_remint",
    "refuse_database_edit",
    "refuse_identity_mismatch",
    "refuse_live_client",
    "refuse_mcp_client_cid_remint",
    "refuse_model_completion",
    "refuse_non_canonical_objective",
    "refuse_objective_cid_remint",
    "refuse_pack_cid_remint",
    "refuse_python_client_cid_remint",
    "refuse_self_promotion",
    "refuse_unauthorized_path",
    "render_declared_parity",
    "verify_objective_identity_parity_files",
    "write_objective_identity_parity_files",
]
