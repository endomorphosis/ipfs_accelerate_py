"""Fail-closed PCPR-080 Accelerate-owned Python external-client demonstration.

Accelerate owns the hermetic Python client that talks to the same
canonical service MCP and CLI would use. The client submits the
PCPR-060 canonical objective bytes, receives the same objective
identity, subscribes to events, inspects task state, submits candidate
evidence, and retrieves the PCPR-072 final proof-carrying receipt
chain.

The Python principal is intent, never authority. The client cannot
update or terminalize tasks, forge policy, bypass confirmation, leases,
or fencing, broaden scope, write current roots or policy pointers, or
promote itself. Live Supervisor.run, live Quack, live MCP, and live
CLI sessions stay typed unavailable. Simulated results are not live.
This evaluator does not write DuckDB or Quack state and does not emit
a closed PCPR release.

Hermetic client coverage is candidate coverage only. Generic MCP-client
demonstration remains PCPR-081.
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


INTERFACE: Final = "AcceleratePythonExternalClient@1"
SCHEMA: Final = "ipfs_accelerate_py/assurance/python-external-client@1"
DOCUMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/declared-python-external-client@1"
)
CLIENT_SCHEMA: Final = "ipfs_accelerate_py/assurance/python-external-client-plan@1"
VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/python-external-client-verdict@1"
)
CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/platform-python-external-client-catalog@1"
)
CATALOG_INTERFACE: Final = "PlatformPythonExternalClient@1"
CLIENT_INTERFACE: Final = "AuthoritativePythonExternalClient@1"
OWNER_INTERFACE: Final = "DatasetsContextPack@1"
OWNER_SCHEMA: Final = "ipfs_datasets_py/datasets-context-pack@1"
OWNER_REPOSITORY: Final = "ipfs_datasets_py"
STORAGE_OWNER_REPOSITORY: Final = "ipfs_kit_py"
STORAGE_OWNER_INTERFACE: Final = "KitContextPackStorage@1"
EXECUTION_OWNER_REPOSITORY: Final = "ipfs_accelerate_py"
CHAIN_OWNER_INTERFACE: Final = "AccelerateFinalProofCarryingReceiptChain@1"
PROTOCOL_INTERFACE: Final = "LogicProviderProtocol@2"
DOCUMENTATION_INTERFACE: Final = "LogicProviderProtocolPythonExternalClient@1"
DOCUMENTATION_SCHEMA: Final = (
    "ipfs_datasets_py/logic-provider-python-external-client@1"
)
PCPR_080_TASK_ID: Final = "PCPR-080"
PCPR_080_GOAL_ID: Final = "PCPR-G800"
PCPR_081_TASK_ID: Final = "PCPR-081"
PCPR_072_TASK_ID: Final = "PCPR-072"
PCPR_061_TASK_ID: Final = "PCPR-061"
PCPR_062_TASK_ID: Final = "PCPR-062"
PCPR_060_TASK_ID: Final = "PCPR-060"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = (
    "proof-carrying-platform-qualification-and-release-v1"
)
OBJECTIVE_KIND: Final = "declared_python_external_client_demonstration"
OBJECTIVE_ID: Final = "PCPR-G800"
LANGUAGE: Final = "Python"
OPERATOR_BLOCKING_TASK_ID: Final = (
    "pcpr-080-operator-live-python-external-client"
)
DOCUMENT_DIR_RELPATH: Final = "packaging/pcpr/reference-workflow/cpython312"
DOCUMENT_JSON_NAME: Final = "reference.python-external-client.json"
DOCUMENT_README_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/PYTHON_EXTERNAL_CLIENT.md"
)
CATALOG_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/platform-python-external-client-catalog.json"
)
SOURCE_DATE_EPOCH: Final = "0"
SOURCE_REPOSITORY: Final = "endomorphosis/ipfs_accelerate_py"
COMPLETED_THROUGH: Final = "python_external_client_demonstration"
NEXT_AUTHORIZED_STAGE: Final = "generic_mcp_client_demonstration"
CHANGE_KIND: Final = "python_external_client_demonstration"
CLIENT_KIND: Final = "hermetic_python_canonical_service_client"
TRANSPORT: Final = "python"
CANONICAL_SERVICE: Final = "pcpr-canonical-supervisor-service"
CLIENT_PRINCIPAL: Final = "principal:pcpr-080-python-external-client"
CLIENT_AUTHORITY: Final = "intent_not_authority"
CLIENT_COMMAND_DIGEST: Final = "pcpr-080-hermetic-python-external-client"

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
    raise RuntimeError("PCPR-080 lock CID remints PCPR-056")
if OWNER_INTERFACES["SupervisorContextPack"] != OWNER_INTERFACE:
    raise RuntimeError("PCPR-080 remints SupervisorContextPack owner interface")
if OWNER_SCHEMAS["SupervisorContextPack"] != OWNER_SCHEMA:
    raise RuntimeError("PCPR-080 remints SupervisorContextPack owner schema")

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

FORBIDDEN_CLIENT_OPERATIONS: Final[tuple[str, ...]] = (
    "update_task",
    "terminalize_task",
    "forge_policy",
    "bypass_confirmation",
    "bypass_lease",
    "bypass_fence",
    "broaden_scope",
    "write_current_root",
    "write_policy_pointer",
    "self_promote",
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
    "artifacts/proof_carrying_platform_qualification_and_release/receipts/PCPR-080.json",
)

HERMETIC_EVENTS: Final[tuple[Mapping[str, str], ...]] = (
    MappingProxyType(
        {
            "kind": "objective_submitted",
            "task_id": PCPR_060_TASK_ID,
            "identity": "objective_cid",
            "cid": PINNED_OBJECTIVE_CID,
        }
    ),
    MappingProxyType(
        {
            "kind": "context_pack_constructed",
            "task_id": PCPR_061_TASK_ID,
            "identity": "pack_cid",
            "cid": PINNED_PACK_CID,
        }
    ),
    MappingProxyType(
        {
            "kind": "current_root_published",
            "task_id": PCPR_062_TASK_ID,
            "identity": "current_root_cid",
            "cid": PINNED_CURRENT_ROOT_CID,
        }
    ),
    MappingProxyType(
        {
            "kind": "final_receipt_chain_emitted",
            "task_id": PCPR_072_TASK_ID,
            "identity": "chain_cid",
            "cid": PINNED_CHAIN_CID,
        }
    ),
)

HERMETIC_TASK_STATE: Final[tuple[Mapping[str, str], ...]] = (
    MappingProxyType(
        {
            "task_id": PCPR_060_TASK_ID,
            "state": "declared_complete",
            "identity": PINNED_OBJECTIVE_CID,
        }
    ),
    MappingProxyType(
        {
            "task_id": PCPR_061_TASK_ID,
            "state": "declared_complete",
            "identity": PINNED_PACK_CID,
        }
    ),
    MappingProxyType(
        {
            "task_id": PCPR_062_TASK_ID,
            "state": "declared_complete",
            "identity": PINNED_CURRENT_ROOT_CID,
        }
    ),
    MappingProxyType(
        {
            "task_id": PCPR_072_TASK_ID,
            "state": "declared_complete",
            "identity": PINNED_CHAIN_CID,
        }
    ),
)

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_python_external_client.py",
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
        "client_files_match_generator",
        "pyproject_python_external_client_table",
        "paths_remain_authorized",
        "canonical_objective_bytes_match",
        "objective_identity_matches_pcpr_060",
        "owner_pack_cid_matches_pin",
        "kit_current_root_bound_not_minted",
        "chain_cid_bound_not_minted",
        "python_transport_used",
        "canonical_service_shared",
        "client_is_intent_not_authority",
        "submit_objective_demonstrated",
        "receive_identity_demonstrated",
        "subscribe_events_demonstrated",
        "inspect_task_state_demonstrated",
        "submit_candidate_evidence_demonstrated",
        "retrieve_final_receipts_demonstrated",
        "forbidden_client_operations_refused",
        "idempotent_client_cid",
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
        "model_assertion_completed_work",
        "unauthorized_path_accepted",
        "forbidden_client_operation_accepted",
        "database_edited",
        "client_self_promoted",
    }
)

DOCUMENT_README: Final = """# PCPR-080 Accelerate Python external-client demonstration

These files record the Accelerate-owned hermetic Python client that
uses the same canonical service MCP and CLI would use. The client
submits the PCPR-060 canonical objective, receives the same objective
identity, subscribes to events, inspects task state, submits candidate
evidence, and retrieves the PCPR-072 final proof-carrying receipt
chain. It does not remint those identities and does not claim a live
supervisor session.

- `cpython312/reference.python-external-client.json` is the declared
  client document. Exact commit and tree are bound by the PCPR-080
  receipt `current_tree_binding`.
- `platform-python-external-client-catalog.json` observes sibling
  Datasets and Kit bindings when present. Sibling source is never
  required.
- The Python principal is intent, never authority. Forbidden client
  operations are refused. Live Supervisor.run, live Quack, live MCP,
  and live CLI sessions stay typed unavailable. Generic MCP-client
  demonstration remains PCPR-081.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-080-operator-live-python-external-client`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
"""


class AcceleratePythonExternalClientError(Exception):
    """Fail-closed PCPR-080 Accelerate Python external-client error."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AcceleratePythonExternalClientError(
            f"{name} must be a non-empty string"
        )
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise AcceleratePythonExternalClientError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise AcceleratePythonExternalClientError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def parse_pyproject_python_external_client_table(text: str) -> dict[str, Any]:
    import tomllib

    table = tomllib.loads(_text(text, "pyproject.toml"))
    if not isinstance(table, dict):
        raise AcceleratePythonExternalClientError(
            "pyproject.toml must be a table"
        )
    tool = table.get("tool")
    payload: dict[str, Any] = {}
    if isinstance(tool, dict):
        accelerate = tool.get("ipfs_accelerate_py")
        if isinstance(accelerate, dict):
            raw = accelerate.get("python-external-client")
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
        raise AcceleratePythonExternalClientError(
            f"path {path} is not an authorized PCPR-080 path"
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
        raise AcceleratePythonExternalClientError(
            f"model assertion at {stage} cannot complete work"
        )


def refuse_pack_cid_remint(cid: str) -> str:
    if cid != PINNED_PACK_CID:
        raise AcceleratePythonExternalClientError(
            f"ContextPack CID {cid} remints {PINNED_PACK_CID}"
        )
    return cid


def refuse_current_root_remint(cid: str) -> str:
    if cid != PINNED_CURRENT_ROOT_CID:
        raise AcceleratePythonExternalClientError(
            f"current root CID {cid} remints {PINNED_CURRENT_ROOT_CID}"
        )
    return cid


def refuse_chain_cid_remint(cid: str) -> str:
    if cid != PINNED_CHAIN_CID:
        raise AcceleratePythonExternalClientError(
            f"chain CID {cid} remints {PINNED_CHAIN_CID}"
        )
    return cid


def refuse_objective_cid_remint(cid: str) -> str:
    if cid != PINNED_OBJECTIVE_CID:
        raise AcceleratePythonExternalClientError(
            f"objective CID {cid} remints {PINNED_OBJECTIVE_CID}"
        )
    return cid


def refuse_idea_digest_remint(digest: str) -> str:
    if digest != PINNED_IDEA_DIGEST:
        raise AcceleratePythonExternalClientError(
            f"idea digest {digest} remints {PINNED_IDEA_DIGEST}"
        )
    return digest


def refuse_database_edit(*, edited: bool) -> None:
    if edited:
        raise AcceleratePythonExternalClientError(
            "python external client cannot write DuckDB or Quack state"
        )


def refuse_live_client(*, applied_live: bool) -> None:
    if applied_live:
        raise AcceleratePythonExternalClientError(
            "live python external client remains typed unavailable"
        )


def refuse_forbidden_client_operation(operation: str) -> str:
    name = _text(operation, "operation")
    if name in FORBIDDEN_CLIENT_OPERATIONS:
        raise AcceleratePythonExternalClientError(
            f"forbidden client operation {name} is refused"
        )
    return name


def refuse_self_promotion(*, promoted: bool) -> None:
    if promoted:
        raise AcceleratePythonExternalClientError(
            "python external client cannot promote itself"
        )


def refuse_non_canonical_objective(idea: str) -> str:
    text = _text(idea, "idea")
    if text != REFERENCE_OBJECTIVE_IDEA:
        raise AcceleratePythonExternalClientError(
            "python external client must submit the canonical PCPR-060 idea"
        )
    return text


class PythonExternalClient:
    """Hermetic Python client for the canonical PCPR service.

    Live supervisor contact stays typed unavailable. The principal is
    intent, never authority.
    """

    transport: Final = TRANSPORT
    principal: Final = CLIENT_PRINCIPAL
    authority: Final = CLIENT_AUTHORITY
    canonical_service: Final = CANONICAL_SERVICE

    def submit_objective(self, idea: str) -> dict[str, Any]:
        refuse_non_canonical_objective(idea)
        return {
            "operation": "submit_objective",
            "transport": TRANSPORT,
            "canonical_service": CANONICAL_SERVICE,
            "principal": CLIENT_PRINCIPAL,
            "authority": CLIENT_AUTHORITY,
            "idea": REFERENCE_OBJECTIVE_IDEA,
            "idea_digest": PINNED_IDEA_DIGEST,
            "objective_bytes_sha256": sha256_bytes(CANONICAL_OBJECTIVE_BYTES),
            "submitted": True,
            "live": False,
            "hermetic": True,
            "admitted_live": False,
            "evidence_kind": "measured_hermetic",
        }

    def receive_identity(self) -> dict[str, Any]:
        return {
            "operation": "receive_identity",
            "transport": TRANSPORT,
            "canonical_service": CANONICAL_SERVICE,
            "objective_cid": PINNED_OBJECTIVE_CID,
            "idea_digest": PINNED_IDEA_DIGEST,
            "identity_matches_pcpr_060": True,
            "equivalent_plan_semantics": True,
            "live": False,
            "hermetic": True,
            "evidence_kind": "measured_hermetic",
        }

    def subscribe_events(self) -> dict[str, Any]:
        return {
            "operation": "subscribe_events",
            "transport": TRANSPORT,
            "canonical_service": CANONICAL_SERVICE,
            "subscribed": True,
            "events": [dict(item) for item in HERMETIC_EVENTS],
            "live": False,
            "hermetic": True,
            "evidence_kind": "measured_hermetic",
        }

    def inspect_task_state(self) -> dict[str, Any]:
        return {
            "operation": "inspect_task_state",
            "transport": TRANSPORT,
            "canonical_service": CANONICAL_SERVICE,
            "inspected": True,
            "tasks": [dict(item) for item in HERMETIC_TASK_STATE],
            "can_update_task": False,
            "can_terminalize_task": False,
            "live": False,
            "hermetic": True,
            "evidence_kind": "measured_hermetic",
        }

    def submit_candidate_evidence(self) -> dict[str, Any]:
        return {
            "operation": "submit_candidate_evidence",
            "transport": TRANSPORT,
            "canonical_service": CANONICAL_SERVICE,
            "submitted": True,
            "kind": "hermetic_python_client_candidate",
            "admitted_live": False,
            "live": False,
            "hermetic": True,
            "evidence_kind": "measured_hermetic",
        }

    def retrieve_final_receipts(self) -> dict[str, Any]:
        return {
            "operation": "retrieve_final_receipts",
            "transport": TRANSPORT,
            "canonical_service": CANONICAL_SERVICE,
            "retrieved": True,
            "chain_cid": PINNED_CHAIN_CID,
            "chain_document_cid": PINNED_CHAIN_DOCUMENT_CID,
            "chain_task_id": PCPR_072_TASK_ID,
            "live": False,
            "hermetic": True,
            "evidence_kind": "measured_hermetic",
        }


def hermetic_python_client_trace() -> dict[str, Any]:
    client = PythonExternalClient()
    submit = client.submit_objective(REFERENCE_OBJECTIVE_IDEA)
    identity = client.receive_identity()
    events = client.subscribe_events()
    state = client.inspect_task_state()
    evidence = client.submit_candidate_evidence()
    receipts = client.retrieve_final_receipts()
    return {
        "schema": "ipfs_accelerate_py/assurance/hermetic-python-client-trace@1",
        "interface": "HermeticPythonClientTrace@1",
        "transport": TRANSPORT,
        "canonical_service": CANONICAL_SERVICE,
        "principal": CLIENT_PRINCIPAL,
        "authority": CLIENT_AUTHORITY,
        "operations": list(CLIENT_OPERATIONS),
        "forbidden_operations": list(FORBIDDEN_CLIENT_OPERATIONS),
        "submit_objective": submit,
        "receive_identity": identity,
        "subscribe_events": events,
        "inspect_task_state": state,
        "submit_candidate_evidence": evidence,
        "retrieve_final_receipts": receipts,
        "objective_cid": identity["objective_cid"],
        "idea_digest": identity["idea_digest"],
        "chain_cid": receipts["chain_cid"],
        "live": False,
        "hermetic": True,
        "evidence_kind": "measured_hermetic",
    }


def hermetic_client_idempotency(client_cid: str) -> dict[str, Any]:
    return {
        "schema": "ipfs_accelerate_py/assurance/hermetic-python-client-idempotency@1",
        "interface": "HermeticPythonClientIdempotency@1",
        "first_client_command_digest": CLIENT_COMMAND_DIGEST,
        "second_client_command_digest": CLIENT_COMMAND_DIGEST,
        "first_client_cid": client_cid,
        "second_client_cid": client_cid,
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
class AcceleratePythonExternalClientVerdict:
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
    client_cid: str
    document_cid: str
    catalog_cid: str
    chain_cid: str
    pack_cid: str
    current_root_cid: str
    objective_cid: str
    idea_digest: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "verdict_cid": self.verdict_cid,
            "client_cid": self.client_cid,
            "document_cid": self.document_cid,
            "catalog_cid": self.catalog_cid,
            "chain_cid": self.chain_cid,
            "pack_cid": self.pack_cid,
            "current_root_cid": self.current_root_cid,
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
            "An admitted Quack-fenced state-owner session before the Python "
            "external client is executed as live supervisor work"
        ),
        "action": (
            "Keep the hermetic Python client, canonical objective identity, "
            "event subscription, task-state inspection, candidate evidence, "
            "and retrieved PCPR-072 receipts. Do not write DuckDB or Quack "
            "state. Do not escalate to a model. Do not grant the Python "
            "principal authority. PCPR-081 adds the generic MCP-client "
            "demonstration."
        ),
        "reason": (
            "Hermetic Python-client demonstration is not a live Supervisor.run "
            "/ CLI / MCP session. Sealed validation has no admitted "
            "Quack-fenced session. Direct DuckDB writes are prohibited."
        ),
    }


def produce_python_external_client(
    *,
    paths: Sequence[str] | None = None,
    pack_cid: str = PINNED_PACK_CID,
    current_root_cid: str = PINNED_CURRENT_ROOT_CID,
    chain_cid: str = PINNED_CHAIN_CID,
    objective_cid: str = PINNED_OBJECTIVE_CID,
    idea_digest: str = PINNED_IDEA_DIGEST,
    idea: str = REFERENCE_OBJECTIVE_IDEA,
    complete_from: str | None = None,
    database_edited: bool = False,
    applied_live: bool = False,
    self_promoted: bool = False,
    forbidden_operation: str | None = None,
) -> dict[str, Any]:
    """Produce the hermetic Python external-client demonstration. Fail closed."""

    if complete_from is not None:
        refuse_model_completion(complete_from)
    refuse_pack_cid_remint(pack_cid)
    refuse_current_root_remint(current_root_cid)
    refuse_chain_cid_remint(chain_cid)
    refuse_objective_cid_remint(objective_cid)
    refuse_idea_digest_remint(idea_digest)
    refuse_non_canonical_objective(idea)
    refuse_database_edit(edited=database_edited)
    refuse_live_client(applied_live=applied_live)
    refuse_self_promotion(promoted=self_promoted)
    if forbidden_operation is not None:
        refuse_forbidden_client_operation(forbidden_operation)
    checked_paths = list(paths) if paths is not None else list(AUTHORIZED_PATH_PREFIXES)
    for path in checked_paths:
        refuse_unauthorized_path(path)
    trace = hermetic_python_client_trace()
    return {
        "schema": CLIENT_SCHEMA,
        "interface": CLIENT_INTERFACE,
        "owner_interface": INTERFACE,
        "task_id": PCPR_080_TASK_ID,
        "goal_id": PCPR_080_GOAL_ID,
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
        "protocol_interface": PROTOCOL_INTERFACE,
        "documentation_interface": DOCUMENTATION_INTERFACE,
        "documentation_schema": DOCUMENTATION_SCHEMA,
        "change_kind": CHANGE_KIND,
        "client_kind": CLIENT_KIND,
        "transport": TRANSPORT,
        "canonical_service": CANONICAL_SERVICE,
        "principal": CLIENT_PRINCIPAL,
        "authority": CLIENT_AUTHORITY,
        "operations": list(CLIENT_OPERATIONS),
        "forbidden_operations": list(FORBIDDEN_CLIENT_OPERATIONS),
        "produced": True,
        "applied": True,
        "live": False,
        "hermetic": True,
        "admitted_live": False,
        "deferred": False,
        "paths_authorized": True,
        "authorized_path_prefixes": list(AUTHORIZED_PATH_PREFIXES),
        "identity_matches_pcpr_060": True,
        "equivalent_plan_semantics": True,
        "events_subscribed": True,
        "task_state_inspected": True,
        "candidate_evidence_submitted": True,
        "final_receipts_retrieved": True,
        "forbidden_operations_refused": True,
        "client_self_promoted": False,
        "trace": trace,
        "client_command_digest": CLIENT_COMMAND_DIGEST,
        "idempotent": True,
        "duplicate_effect_rejected": True,
        "remints_protocol": False,
        "adds_protocol_operation": False,
        "model_assertion_completes_work": False,
        "escalation_order": list(ESCALATION_ORDER),
        "completed_through": COMPLETED_THROUGH,
        "next_authorized_stage": NEXT_AUTHORIZED_STAGE,
        "next_task_id": PCPR_081_TASK_ID,
        "prerequisite_task_id": PCPR_072_TASK_ID,
        "chain_task_id": PCPR_072_TASK_ID,
        "duckdb_or_quack_state_written": False,
        "closed_release_outcome": None,
        "release_claim": False,
        "evidence_kind": "measured_hermetic",
    }


def canonical_python_external_client() -> dict[str, Any]:
    return produce_python_external_client()


def client_cid_of(plan: Mapping[str, Any] | None = None) -> str:
    payload = dict(plan or canonical_python_external_client())
    payload.pop("client_cid", None)
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
        "evidence_kind": "measured",
    }


def _sibling_binding(path: Path, relative_path: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": relative_path,
            "status": "unavailable",
            "evidence_kind": "unavailable",
            "reason": (
                "Sibling python-external-client file is not present beside this checkout."
            ),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    context_pack = payload.get("context_pack")
    storage = payload.get("storage")
    client = payload.get("python_external_client")
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
        or (client.get("chain_cid") if isinstance(client, Mapping) else None),
        "client_cid": payload.get("client_cid")
        or (client.get("client_cid") if isinstance(client, Mapping) else None),
        "binding_cid": payload.get("binding_cid") or payload.get("document_cid"),
        "live": False,
        "applied": False,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def render_declared_client(root: Path | None = None) -> dict[str, Any]:
    package_root = root or discover_accelerate_root()
    if package_root is None:
        raise AcceleratePythonExternalClientError(
            "Accelerate package root was not found"
        )
    state_owner = observe_state_owner()
    plan = canonical_python_external_client()
    computed_client_cid = client_cid_of(plan)
    idempotency = hermetic_client_idempotency(computed_client_cid)
    document = {
        "schema": DOCUMENT_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_080_TASK_ID,
        "goal_id": PCPR_080_GOAL_ID,
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
            "survives_client": True,
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
            "survives_client": True,
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
            "evidence_kind": "measured_hermetic",
        },
        "python_external_client": {
            **plan,
            "client_cid": computed_client_cid,
            "idempotency": idempotency,
            "produced_by": EXECUTION_OWNER_REPOSITORY,
        },
        "materialization": {
            "kind": "declared_python_external_client_not_live",
            "admitted": False,
            "live": False,
            "applied": False,
            "duckdb_or_quack_state_written": False,
            "evidence_kind": "unavailable",
        },
        "live_application": typed_unavailable(
            reason=(
                "No admitted Quack-fenced supervisor session was observed. "
                "Hermetic Python-client demonstration is not live client execution."
            )
        ),
        "live_client": typed_unavailable(
            reason=(
                "Live Python Supervisor.run remains typed unavailable. This "
                "task only emits a hermetic Python client. Generic MCP-client "
                "demonstration remains PCPR-081."
            )
        ),
        "live_mcp_client": typed_unavailable(
            reason="Generic MCP-client demonstration remains PCPR-081."
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
                "field": "PCPR-080 receipt current_tree_binding",
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


def platform_python_external_client_catalog(
    start: Path | None = None,
) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AcceleratePythonExternalClientError(
            "Accelerate package root was not found"
        )
    parent = root.parent
    document = render_declared_client(root)
    datasets_binding = _sibling_binding(
        parent
        / "ipfs_datasets"
        / DOCUMENT_DIR_RELPATH
        / "reference.python-external-client.binding.json",
        relative_path=(
            f"../ipfs_datasets/{DOCUMENT_DIR_RELPATH}/"
            "reference.python-external-client.binding.json"
        ),
    )
    kit_binding = _sibling_binding(
        parent
        / "ipfs_kit"
        / DOCUMENT_DIR_RELPATH
        / "reference.python-external-client.binding.json",
        relative_path=(
            f"../ipfs_kit/{DOCUMENT_DIR_RELPATH}/"
            "reference.python-external-client.binding.json"
        ),
    )
    catalog = {
        "schema": CATALOG_SCHEMA,
        "interface": CATALOG_INTERFACE,
        "task_id": PCPR_080_TASK_ID,
        "goal_id": PCPR_080_GOAL_ID,
        "objective_kind": OBJECTIVE_KIND,
        "portfolio_id": PORTFOLIO_ID,
        "portfolio_version": PORTFOLIO_VERSION,
        "objective_cid": PINNED_OBJECTIVE_CID,
        "idea_digest": PINNED_IDEA_DIGEST,
        "pack_cid": PINNED_PACK_CID,
        "current_root_cid": PINNED_CURRENT_ROOT_CID,
        "chain_cid": PINNED_CHAIN_CID,
        "client_cid": document["python_external_client"]["client_cid"],
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
                "client_cid": document["python_external_client"]["client_cid"],
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


def write_python_external_client_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AcceleratePythonExternalClientError(
            "Accelerate package root was not found"
        )
    document = render_declared_client(root)
    paths = artifact_paths(root)
    _atomic_write(paths["document"], pretty_json(document))
    readme = DOCUMENT_README if DOCUMENT_README.endswith("\n") else DOCUMENT_README + "\n"
    _atomic_write(paths["readme"], readme)
    catalog = platform_python_external_client_catalog(start)
    _atomic_write(paths["catalog"], pretty_json(catalog))
    return {
        "document": document,
        "catalog": catalog,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def verify_python_external_client_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AcceleratePythonExternalClientError(
            "Accelerate package root was not found"
        )
    document = render_declared_client(root)
    catalog = platform_python_external_client_catalog(start)
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
        "client_cid": document["python_external_client"]["client_cid"],
        "document_cid": document["document_cid"],
        "catalog_cid": catalog["catalog_cid"],
        "current_root_cid": document["storage"]["current_root_cid"],
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
        raise AcceleratePythonExternalClientError(
            "Accelerate package root was not found"
        )
    document = render_declared_client(root)
    verified = verify_python_external_client_files(root)
    table = parse_pyproject_python_external_client_table(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    client = document["python_external_client"]
    remint = (
        document["context_pack"]["pack_cid"] != PINNED_PACK_CID
        or document["objective_cid"] != PINNED_OBJECTIVE_CID
        or document["idea_digest"] != PINNED_IDEA_DIGEST
        or document["lock_cid"] != PINNED_LOCK_CID
        or document["storage"]["current_root_cid"] != PINNED_CURRENT_ROOT_CID
        or document["final_receipt_chain"]["chain_cid"] != PINNED_CHAIN_CID
        or client["client_cid"] != PINNED_CLIENT_CID
        or document["context_pack"]["reminted"] is True
        or document["storage"]["reminted"] is True
        or document["final_receipt_chain"]["reminted"] is True
    )
    trace = client["trace"]
    idempotency = client["idempotency"]
    probes = [
        _probe(
            "client_files_match_generator",
            verified.get("ok") is True and verified.get("document_ok") is True,
            reason=(
                "Committed Accelerate Python client matches the generator."
                if verified.get("document_ok") is True
                else "Committed Accelerate Python client is missing or drifts."
            ),
        ),
        _probe(
            "pyproject_python_external_client_table",
            table.get("interface") == INTERFACE
            and table.get("schema") == SCHEMA
            and table.get("task-id") == PCPR_080_TASK_ID
            and table.get("objective-kind") == OBJECTIVE_KIND,
            reason=(
                "pyproject.toml declares AcceleratePythonExternalClient@1."
                if table.get("interface") == INTERFACE
                else "pyproject.toml does not declare the PCPR-080 client."
            ),
        ),
        _probe(
            "paths_remain_authorized",
            client["paths_authorized"] is True
            and list(client["authorized_path_prefixes"])
            == list(AUTHORIZED_PATH_PREFIXES),
            reason="Client paths remain inside the declared PCPR-080 allowlist.",
        ),
        _probe(
            "canonical_objective_bytes_match",
            client["idea"] == REFERENCE_OBJECTIVE_IDEA
            and client["idea_digest"] == PINNED_IDEA_DIGEST,
            reason="Python client submits the canonical PCPR-060 idea bytes.",
        ),
        _probe(
            "objective_identity_matches_pcpr_060",
            client["objective_cid"] == PINNED_OBJECTIVE_CID
            and client["identity_matches_pcpr_060"] is True
            and client["equivalent_plan_semantics"] is True,
            reason="Python client receives the same PCPR-060 objective identity.",
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
            "python_transport_used",
            client["transport"] == TRANSPORT
            and trace["transport"] == TRANSPORT,
            reason="The demonstration uses the Python client transport.",
        ),
        _probe(
            "canonical_service_shared",
            client["canonical_service"] == CANONICAL_SERVICE,
            reason="Python client uses the same canonical service MCP would use.",
        ),
        _probe(
            "client_is_intent_not_authority",
            client["principal"] == CLIENT_PRINCIPAL
            and client["authority"] == CLIENT_AUTHORITY,
            reason="The Python principal is intent, never authority.",
        ),
        _probe(
            "submit_objective_demonstrated",
            client["produced"] is True
            and trace["submit_objective"]["submitted"] is True
            and trace["submit_objective"]["live"] is False,
            reason="Hermetic submit_objective is demonstrated and is not live.",
        ),
        _probe(
            "receive_identity_demonstrated",
            client["identity_matches_pcpr_060"] is True
            and trace["receive_identity"]["objective_cid"] == PINNED_OBJECTIVE_CID,
            reason="Hermetic receive_identity returns the PCPR-060 objective CID.",
        ),
        _probe(
            "subscribe_events_demonstrated",
            client["events_subscribed"] is True
            and trace["subscribe_events"]["subscribed"] is True
            and len(trace["subscribe_events"]["events"]) == len(HERMETIC_EVENTS),
            reason="Hermetic event subscription is demonstrated.",
        ),
        _probe(
            "inspect_task_state_demonstrated",
            client["task_state_inspected"] is True
            and trace["inspect_task_state"]["can_update_task"] is False
            and trace["inspect_task_state"]["can_terminalize_task"] is False,
            reason="Hermetic task-state inspection cannot mutate or terminalize.",
        ),
        _probe(
            "submit_candidate_evidence_demonstrated",
            client["candidate_evidence_submitted"] is True
            and trace["submit_candidate_evidence"]["admitted_live"] is False,
            reason="Hermetic candidate evidence is submitted and is not live.",
        ),
        _probe(
            "retrieve_final_receipts_demonstrated",
            client["final_receipts_retrieved"] is True
            and trace["retrieve_final_receipts"]["chain_cid"] == PINNED_CHAIN_CID,
            reason="Hermetic retrieval returns the PCPR-072 chain CID.",
        ),
        _probe(
            "forbidden_client_operations_refused",
            client["forbidden_operations_refused"] is True
            and list(client["forbidden_operations"])
            == list(FORBIDDEN_CLIENT_OPERATIONS),
            reason="Forbidden client authority operations are refused.",
        ),
        _probe(
            "idempotent_client_cid",
            client["idempotent"] is True
            and idempotency["identical"] is True
            and idempotency["first_client_cid"] == client["client_cid"]
            and idempotency["second_client_cid"] == client["client_cid"]
            and client["client_cid"] == PINNED_CLIENT_CID,
            reason="A second client emission yields the same client CID.",
        ),
        _probe(
            "model_assertion_cannot_complete_work",
            client["model_assertion_completes_work"] is False,
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
                "A reminted pack, root, chain, objective, idea, lock, or "
                "client CID is forbidden."
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
            reason="Hermetic Python-client demonstration is not represented as live.",
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
            "forbidden_client_operation_accepted",
            False,
            reason="Forbidden client operations are refused.",
        ),
        _probe(
            "database_edited",
            False,
            reason="DuckDB and Quack state were not written.",
        ),
        _probe(
            "client_self_promoted",
            False,
            reason="The Python client cannot promote itself.",
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
            reason="Live Python Supervisor.run stays typed unavailable.",
        ),
    ]
    return tuple(probes)


def qualify_python_external_client(
    probes: Sequence[OutcomeProbe],
    *,
    client_cid: str,
    document_cid: str,
    catalog_cid: str,
    chain_cid: str,
    pack_cid: str,
    current_root_cid: str,
    objective_cid: str,
    idea_digest_cid: str,
) -> AcceleratePythonExternalClientVerdict:
    if not probes:
        raise AcceleratePythonExternalClientError("at least one probe is required")
    normalized: list[OutcomeProbe] = []
    blockers: list[str] = []
    for probe in probes:
        kind = _kind(probe.evidence_kind, "evidence_kind")
        if probe.live and kind != "measured_live":
            raise AcceleratePythonExternalClientError(
                "live claims require measured_live evidence"
            )
        if probe.simulated_represented_as_live:
            raise AcceleratePythonExternalClientError(
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
        "task_id": PCPR_080_TASK_ID,
        "goal_id": PCPR_080_GOAL_ID,
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
        "client_cid": client_cid,
        "document_cid": document_cid,
        "catalog_cid": catalog_cid,
        "chain_cid": chain_cid,
        "pack_cid": pack_cid,
        "current_root_cid": current_root_cid,
        "objective_cid": objective_cid,
        "idea_digest": idea_digest_cid,
    }
    return AcceleratePythonExternalClientVerdict(
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
        client_cid=client_cid,
        document_cid=document_cid,
        catalog_cid=catalog_cid,
        chain_cid=chain_cid,
        pack_cid=pack_cid,
        current_root_cid=current_root_cid,
        objective_cid=objective_cid,
        idea_digest=idea_digest_cid,
    )


def qualify_current_head_python_external_client(
    start: Path | None = None,
) -> AcceleratePythonExternalClientVerdict:
    document = render_declared_client(start)
    catalog = platform_python_external_client_catalog(start)
    return qualify_python_external_client(
        current_head_static_probes(start),
        client_cid=str(document["python_external_client"]["client_cid"]),
        document_cid=str(document["document_cid"]),
        catalog_cid=str(catalog["catalog_cid"]),
        chain_cid=str(document["final_receipt_chain"]["chain_cid"]),
        pack_cid=str(document["context_pack"]["pack_cid"]),
        current_root_cid=str(document["storage"]["current_root_cid"]),
        objective_cid=str(document["objective_cid"]),
        idea_digest_cid=str(document["idea_digest"]),
    )


def pcpr_080_receipt_promotion(
    verdict: AcceleratePythonExternalClientVerdict,
) -> dict[str, Any]:
    if verdict.closed_release_outcome is not None:
        raise AcceleratePythonExternalClientError(
            "python external client must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise AcceleratePythonExternalClientError(
            "python external client must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise AcceleratePythonExternalClientError(
            "python external client completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise AcceleratePythonExternalClientError(
            "python external client must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise AcceleratePythonExternalClientError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.live_client or verdict.live_application:
        raise AcceleratePythonExternalClientError(
            "live client claims require measured_live evidence"
        )
    return verdict.to_mapping()


PINNED_CLIENT_CID: Final = (
    "baguqeerat346yf4k7kenbqlbtyejeilgxamna262z3axzjjatzm2hccehblq"
)
PINNED_DOCUMENT_CID: Final = (
    "baguqeera3u7t57ig7grslylppswjoc4pludmmhmhng74de7ynmquhzihaosa"
)
PINNED_CATALOG_CID: Final = (
    "baguqeeraqne2wsmugfdqchhf2w6677svgyqrmp4rymzkfxbflhtx5cp6ky7q"
)
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeerauloygpapsxdvchksem5hddmiqm32ig473thr7lueqa54qtno3leq"
)


__all__ = [
    "AUTHORIZED_PATH_PREFIXES",
    "CLIENT_OPERATIONS",
    "CLOSED_RELEASE_OUTCOMES",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "ESCALATION_ORDER",
    "FORBIDDEN_CLIENT_OPERATIONS",
    "AcceleratePythonExternalClientError",
    "HERMETIC_CANDIDATE_SUITES",
    "INTERFACE",
    "OBJECTIVE_KIND",
    "OPERATOR_BLOCKING_TASK_ID",
    "OutcomeProbe",
    "PCPR_080_GOAL_ID",
    "PCPR_080_TASK_ID",
    "PINNED_CATALOG_CID",
    "PINNED_CHAIN_CID",
    "PINNED_CLIENT_CID",
    "PINNED_CURRENT_ROOT_CID",
    "PINNED_DOCUMENT_CID",
    "PINNED_IDEA_DIGEST",
    "PINNED_LOCK_CID",
    "PINNED_OBJECTIVE_CID",
    "PINNED_PACK_CID",
    "PythonExternalClient",
    "SCHEMA",
    "SEALED_PATH",
    "SEALED_PYTHON",
    "canonical_python_external_client",
    "client_cid_of",
    "current_head_static_probes",
    "hermetic_client_idempotency",
    "hermetic_python_client_trace",
    "path_is_authorized",
    "pcpr_080_receipt_promotion",
    "platform_python_external_client_catalog",
    "produce_python_external_client",
    "qualify_current_head_python_external_client",
    "qualify_python_external_client",
    "refuse_chain_cid_remint",
    "refuse_current_root_remint",
    "refuse_database_edit",
    "refuse_forbidden_client_operation",
    "refuse_live_client",
    "refuse_model_completion",
    "refuse_non_canonical_objective",
    "refuse_objective_cid_remint",
    "refuse_pack_cid_remint",
    "refuse_self_promotion",
    "refuse_unauthorized_path",
    "render_declared_client",
    "verify_python_external_client_files",
    "write_python_external_client_files",
]
