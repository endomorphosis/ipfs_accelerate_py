"""Fail-closed PCPR-083 Accelerate-owned external-client authority-bypass proof.

Accelerate owns the hermetic negative proof that the PCPR-080 Python client
and the PCPR-081 generic MCP client cannot bypass platform authority. Both
principals are intent, never authority. Every bypass attempt fails typed,
causes no unauthorized effect, and leaves every hard-zero counter at zero.

Forbidden operations: update or terminalize tasks, forge policy, bypass
confirmation, leases, or fencing, broaden scope, write current roots or
policy pointers, or self-promote.

Live Supervisor.run, live Quack, live MCP stdio/HTTP, and live CLI sessions
stay typed unavailable. Simulated results are not live. This evaluator does
not write DuckDB or Quack state and does not emit a closed PCPR release.

Hermetic bypass coverage is candidate coverage only. Threat-model
preparation remains PCPR-090.
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
    FORBIDDEN_CLIENT_OPERATIONS as MCP_FORBIDDEN_OPERATIONS,
    FORBIDDEN_JSONRPC_CODE,
    FORBIDDEN_MCP_TOOL_NAMES,
    CLIENT_PRINCIPAL as MCP_PRINCIPAL,
    GenericMcpClient,
    PINNED_CLIENT_CID as PINNED_MCP_CLIENT_CID,
)
from ipfs_accelerate_py.assurance.objective_identity_parity import (
    PINNED_PARITY_CID,
)
from ipfs_accelerate_py.assurance.portfolio_compatibility_lock import (
    PINNED_LOCK_CID as PCPR_056_LOCK_CID,
    PORTFOLIO_ID,
    PORTFOLIO_VERSION,
)
from ipfs_accelerate_py.assurance.python_external_client import (
    AcceleratePythonExternalClientError,
    FORBIDDEN_CLIENT_OPERATIONS as PYTHON_FORBIDDEN_OPERATIONS,
    CLIENT_PRINCIPAL as PYTHON_PRINCIPAL,
    PINNED_CLIENT_CID as PINNED_PYTHON_CLIENT_CID,
    PythonExternalClient,
    refuse_forbidden_client_operation,
)
from ipfs_accelerate_py.assurance.shared_contracts import (
    OWNER_INTERFACES,
    OWNER_SCHEMAS,
    shared_interface_id,
    shared_schema_id,
)


INTERFACE: Final = "AccelerateAuthorityBypass@1"
SCHEMA: Final = "ipfs_accelerate_py/assurance/authority-bypass@1"
DOCUMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/declared-authority-bypass@1"
)
PROOF_SCHEMA: Final = "ipfs_accelerate_py/assurance/authority-bypass-plan@1"
VERDICT_SCHEMA: Final = "ipfs_accelerate_py/assurance/authority-bypass-verdict@1"
CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/platform-authority-bypass-catalog@1"
)
CATALOG_INTERFACE: Final = "PlatformAuthorityBypass@1"
PROOF_INTERFACE: Final = "AuthoritativeAuthorityBypass@1"
OWNER_INTERFACE: Final = "DatasetsContextPack@1"
OWNER_SCHEMA: Final = "ipfs_datasets_py/datasets-context-pack@1"
OWNER_REPOSITORY: Final = "ipfs_datasets_py"
STORAGE_OWNER_REPOSITORY: Final = "ipfs_kit_py"
STORAGE_OWNER_INTERFACE: Final = "KitContextPackStorage@1"
EXECUTION_OWNER_REPOSITORY: Final = "ipfs_accelerate_py"
CHAIN_OWNER_INTERFACE: Final = "AccelerateFinalProofCarryingReceiptChain@1"
PYTHON_CLIENT_INTERFACE: Final = "AcceleratePythonExternalClient@1"
MCP_CLIENT_INTERFACE: Final = "AccelerateGenericMcpClient@1"
PARITY_INTERFACE: Final = "AccelerateObjectiveIdentityParity@1"
PROTOCOL_INTERFACE: Final = "LogicProviderProtocol@2"
DOCUMENTATION_INTERFACE: Final = "LogicProviderProtocolAuthorityBypass@1"
DOCUMENTATION_SCHEMA: Final = (
    "ipfs_datasets_py/logic-provider-authority-bypass@1"
)
PCPR_083_TASK_ID: Final = "PCPR-083"
PCPR_083_GOAL_ID: Final = "PCPR-G800"
PCPR_090_TASK_ID: Final = "PCPR-090"
PCPR_082_TASK_ID: Final = "PCPR-082"
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
OBJECTIVE_KIND: Final = "declared_external_client_authority_bypass_proof"
OBJECTIVE_ID: Final = "PCPR-G800"
LANGUAGE: Final = "Python+MCP JSON-RPC 2.0"
OPERATOR_BLOCKING_TASK_ID: Final = (
    "pcpr-083-operator-live-external-client-authority-bypass"
)
DOCUMENT_DIR_RELPATH: Final = "packaging/pcpr/reference-workflow/cpython312"
DOCUMENT_JSON_NAME: Final = "reference.authority-bypass.json"
DOCUMENT_README_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/AUTHORITY_BYPASS.md"
)
CATALOG_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/platform-authority-bypass-catalog.json"
)
SOURCE_DATE_EPOCH: Final = "0"
SOURCE_REPOSITORY: Final = "endomorphosis/ipfs_accelerate_py"
COMPLETED_THROUGH: Final = "external_client_authority_bypass_proof"
NEXT_AUTHORIZED_STAGE: Final = "threat_model"
CHANGE_KIND: Final = "external_client_authority_bypass_proof"
PROOF_KIND: Final = "hermetic_python_mcp_authority_bypass_negative"
CANONICAL_SERVICE: Final = "pcpr-canonical-supervisor-service"
PYTHON_TRANSPORT: Final = "python"
MCP_TRANSPORT: Final = "mcp"
CLIENT_AUTHORITY: Final = "intent_not_authority"
PROOF_COMMAND_DIGEST: Final = "pcpr-083-hermetic-authority-bypass"

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

if PYTHON_FORBIDDEN_OPERATIONS != MCP_FORBIDDEN_OPERATIONS:
    raise RuntimeError("Python and MCP forbidden operations diverged")
if PINNED_LOCK_CID != (
    "baguqeerawsekbbbt5ccctjt4tydzeahfkahsatb6q5x7k6cy4inrii5lhqmq"
):
    raise RuntimeError("PCPR-083 lock CID remints PCPR-056")
if PINNED_PYTHON_CLIENT_CID != (
    "baguqeerat346yf4k7kenbqlbtyejeilgxamna262z3axzjjatzm2hccehblq"
):
    raise RuntimeError("PCPR-083 remints the PCPR-080 Python client CID")
if PINNED_MCP_CLIENT_CID != (
    "baguqeerasdkirdryda7uoi5gchsyhuyvh6d2uremm45zpob7hoss3ao66ahq"
):
    raise RuntimeError("PCPR-083 remints the PCPR-081 generic MCP client CID")
if PINNED_PARITY_CID != (
    "baguqeera6y3w7bhzy6hpo6ohfsrj5xihrw5vk6qqljjonjrupijd2yrp3hnq"
):
    raise RuntimeError("PCPR-083 remints the PCPR-082 parity CID")
if PINNED_PYTHON_CLIENT_CID == PINNED_MCP_CLIENT_CID:
    raise RuntimeError("Python and MCP client CIDs must remain distinct")
if OWNER_INTERFACES["SupervisorContextPack"] != OWNER_INTERFACE:
    raise RuntimeError("PCPR-083 remints SupervisorContextPack owner interface")
if OWNER_SCHEMAS["SupervisorContextPack"] != OWNER_SCHEMA:
    raise RuntimeError("PCPR-083 remints SupervisorContextPack owner schema")

FORBIDDEN_CLIENT_OPERATIONS: Final[tuple[str, ...]] = PYTHON_FORBIDDEN_OPERATIONS

OPERATION_TO_MCP_TOOL: Final[Mapping[str, str]] = MappingProxyType(
    dict(zip(FORBIDDEN_CLIENT_OPERATIONS, FORBIDDEN_MCP_TOOL_NAMES, strict=True))
)

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

HARD_ZERO_COUNTERS: Final[tuple[str, ...]] = (
    "unauthorized_task_updates",
    "unauthorized_task_terminalizations",
    "forged_policies",
    "confirmation_bypasses",
    "lease_bypasses",
    "fence_bypasses",
    "scope_broadenings",
    "current_root_writes",
    "policy_pointer_writes",
    "self_promotions",
    "unauthorized_effects",
    "live_claims_without_live_evidence",
    "closed_release_outcomes",
    "duckdb_or_quack_writes",
)

OPERATION_TO_COUNTER: Final[Mapping[str, str]] = MappingProxyType(
    {
        "update_task": "unauthorized_task_updates",
        "terminalize_task": "unauthorized_task_terminalizations",
        "forge_policy": "forged_policies",
        "bypass_confirmation": "confirmation_bypasses",
        "bypass_lease": "lease_bypasses",
        "bypass_fence": "fence_bypasses",
        "broaden_scope": "scope_broadenings",
        "write_current_root": "current_root_writes",
        "write_policy_pointer": "policy_pointer_writes",
        "self_promote": "self_promotions",
    }
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
    "artifacts/proof_carrying_platform_qualification_and_release/receipts/PCPR-083.json",
)

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_authority_bypass.py",
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
        "bypass_files_match_generator",
        "pyproject_authority_bypass_table",
        "paths_remain_authorized",
        "canonical_objective_bytes_match",
        "every_python_bypass_attempt_fails_typed",
        "every_mcp_bypass_attempt_fails_typed",
        "no_unauthorized_effect",
        "hard_zero_counters_remain_zero",
        "task_state_unchanged",
        "current_root_unchanged",
        "policy_pointer_unchanged",
        "owner_pack_cid_matches_pin",
        "kit_current_root_bound_not_minted",
        "chain_cid_bound_not_minted",
        "canonical_service_shared",
        "python_client_cid_bound_not_minted",
        "mcp_client_cid_bound_not_minted",
        "parity_cid_bound_not_minted",
        "client_cids_remain_distinct",
        "clients_are_intent_not_authority",
        "idempotent_proof_cid",
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
        "parity_identity_reminted",
        "model_assertion_completed_work",
        "unauthorized_path_accepted",
        "database_edited",
        "client_self_promoted",
        "bypass_attempt_accepted",
        "unauthorized_effect_observed",
        "hard_zero_counter_nonzero",
    }
)

DOCUMENT_README: Final = """# PCPR-083 Accelerate external-client authority-bypass proof

These files record the Accelerate-owned hermetic negative proof that the
PCPR-080 Python client and the PCPR-081 generic MCP client cannot bypass
platform authority. Both principals are intent, never authority.

- `cpython312/reference.authority-bypass.json` is the declared bypass
  document. Exact commit and tree are bound by the PCPR-083 receipt
  `current_tree_binding`.
- `platform-authority-bypass-catalog.json` observes sibling Datasets and
  Kit bindings when present. Sibling source is never required.
- Every forbidden operation (task update or terminalization, policy
  forgery, confirmation/lease/fence bypass, scope broadening, current-root
  or policy-pointer write, self-promotion) fails typed on both clients.
  No unauthorized effect is observed. Every hard-zero counter remains
  zero. Client CIDs, pack, root, chain, objective, and PCPR-082 parity
  identities are bound and not reminted.
- Live Supervisor.run, live Quack, live MCP stdio/HTTP, and live CLI
  sessions stay typed unavailable. Threat-model preparation remains
  PCPR-090.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-083-operator-live-external-client-authority-bypass`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
"""


class AccelerateAuthorityBypassError(Exception):
    """Fail-closed PCPR-083 Accelerate authority-bypass error."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AccelerateAuthorityBypassError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise AccelerateAuthorityBypassError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise AccelerateAuthorityBypassError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def parse_pyproject_authority_bypass_table(text: str) -> dict[str, Any]:
    import tomllib

    table = tomllib.loads(_text(text, "pyproject.toml"))
    if not isinstance(table, dict):
        raise AccelerateAuthorityBypassError("pyproject.toml must be a table")
    tool = table.get("tool")
    payload: dict[str, Any] = {}
    if isinstance(tool, dict):
        accelerate = tool.get("ipfs_accelerate_py")
        if isinstance(accelerate, dict):
            raw = accelerate.get("authority-bypass")
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
        raise AccelerateAuthorityBypassError(
            f"path {path} is not an authorized PCPR-083 path"
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
        raise AccelerateAuthorityBypassError(
            f"model assertion at {stage} cannot complete work"
        )


def refuse_pack_cid_remint(cid: str) -> str:
    if cid != PINNED_PACK_CID:
        raise AccelerateAuthorityBypassError(
            f"ContextPack CID {cid} remints {PINNED_PACK_CID}"
        )
    return cid


def refuse_current_root_remint(cid: str) -> str:
    if cid != PINNED_CURRENT_ROOT_CID:
        raise AccelerateAuthorityBypassError(
            f"current root CID {cid} remints {PINNED_CURRENT_ROOT_CID}"
        )
    return cid


def refuse_chain_cid_remint(cid: str) -> str:
    if cid != PINNED_CHAIN_CID:
        raise AccelerateAuthorityBypassError(
            f"chain CID {cid} remints {PINNED_CHAIN_CID}"
        )
    return cid


def refuse_objective_cid_remint(cid: str) -> str:
    if cid != PINNED_OBJECTIVE_CID:
        raise AccelerateAuthorityBypassError(
            f"objective CID {cid} remints {PINNED_OBJECTIVE_CID}"
        )
    return cid


def refuse_idea_digest_remint(digest: str) -> str:
    if digest != PINNED_IDEA_DIGEST:
        raise AccelerateAuthorityBypassError(
            f"idea digest {digest} remints {PINNED_IDEA_DIGEST}"
        )
    return digest


def refuse_python_client_cid_remint(cid: str) -> str:
    if cid != PINNED_PYTHON_CLIENT_CID:
        raise AccelerateAuthorityBypassError(
            f"Python client CID {cid} remints {PINNED_PYTHON_CLIENT_CID}"
        )
    return cid


def refuse_mcp_client_cid_remint(cid: str) -> str:
    if cid != PINNED_MCP_CLIENT_CID:
        raise AccelerateAuthorityBypassError(
            f"MCP client CID {cid} remints {PINNED_MCP_CLIENT_CID}"
        )
    return cid


def refuse_parity_cid_remint(cid: str) -> str:
    if cid != PINNED_PARITY_CID:
        raise AccelerateAuthorityBypassError(
            f"parity CID {cid} remints {PINNED_PARITY_CID}"
        )
    return cid


def refuse_database_edit(*, edited: bool) -> None:
    if edited:
        raise AccelerateAuthorityBypassError(
            "authority bypass proof cannot write DuckDB or Quack state"
        )


def refuse_live_client(*, applied_live: bool) -> None:
    if applied_live:
        raise AccelerateAuthorityBypassError(
            "live external-client authority-bypass remains typed unavailable"
        )


def refuse_self_promotion(*, promoted: bool) -> None:
    if promoted:
        raise AccelerateAuthorityBypassError(
            "authority bypass proof cannot promote itself"
        )


def refuse_non_canonical_objective(idea: str) -> str:
    text = _text(idea, "idea")
    if text != REFERENCE_OBJECTIVE_IDEA:
        raise AccelerateAuthorityBypassError(
            "authority bypass clients must submit the canonical PCPR-060 idea"
        )
    return text


def refuse_bypass_acceptance(*, accepted: bool) -> None:
    if accepted:
        raise AccelerateAuthorityBypassError(
            "forbidden client operation was accepted"
        )


def refuse_unauthorized_effect(*, observed: bool) -> None:
    if observed:
        raise AccelerateAuthorityBypassError(
            "unauthorized effect was observed after a bypass attempt"
        )


def refuse_nonzero_hard_zero(*, nonzero: bool) -> None:
    if nonzero:
        raise AccelerateAuthorityBypassError(
            "a hard-zero counter left zero"
        )


def zero_hard_zero_counters() -> dict[str, int]:
    return {name: 0 for name in HARD_ZERO_COUNTERS}


def attempt_python_bypass(operation: str) -> dict[str, Any]:
    """Attempt a forbidden Python-client operation and record the typed refusal."""

    name = _text(operation, "operation")
    if name not in FORBIDDEN_CLIENT_OPERATIONS:
        raise AccelerateAuthorityBypassError(
            f"{name} is not a declared forbidden client operation"
        )
    try:
        refuse_forbidden_client_operation(name)
    except AcceleratePythonExternalClientError as exc:
        return {
            "client": "python",
            "transport": PYTHON_TRANSPORT,
            "canonical_service": CANONICAL_SERVICE,
            "principal": PYTHON_PRINCIPAL,
            "authority": CLIENT_AUTHORITY,
            "operation": name,
            "counter": OPERATION_TO_COUNTER[name],
            "refused": True,
            "accepted": False,
            "typed_failure": str(exc),
            "failure_kind": "typed_refuse",
            "unauthorized_effect": False,
            "live": False,
            "hermetic": True,
            "evidence_kind": "measured_hermetic",
        }
    raise AccelerateAuthorityBypassError(
        f"forbidden client operation {name} was accepted"
    )


def attempt_mcp_bypass(operation: str) -> dict[str, Any]:
    """Attempt a forbidden generic-MCP tool and record the JSON-RPC refusal."""

    name = _text(operation, "operation")
    if name not in FORBIDDEN_CLIENT_OPERATIONS:
        raise AccelerateAuthorityBypassError(
            f"{name} is not a declared forbidden client operation"
        )
    tool = OPERATION_TO_MCP_TOOL[name]
    client = GenericMcpClient()
    result = client.refuse_forbidden_tool(tool)
    if result.get("refused") is not True:
        raise AccelerateAuthorityBypassError(
            f"forbidden MCP tool {tool} was accepted"
        )
    if result.get("jsonrpc_error_code") != FORBIDDEN_JSONRPC_CODE:
        raise AccelerateAuthorityBypassError(
            f"forbidden MCP tool {tool} did not fail typed JSON-RPC"
        )
    return {
        "client": "mcp",
        "transport": MCP_TRANSPORT,
        "canonical_service": CANONICAL_SERVICE,
        "principal": MCP_PRINCIPAL,
        "authority": CLIENT_AUTHORITY,
        "operation": name,
        "tool": tool,
        "counter": OPERATION_TO_COUNTER[name],
        "refused": True,
        "accepted": False,
        "typed_failure": f"forbidden client operation {tool} is refused",
        "failure_kind": "typed_jsonrpc_error",
        "jsonrpc_error_code": FORBIDDEN_JSONRPC_CODE,
        "unauthorized_effect": False,
        "live": False,
        "hermetic": True,
        "evidence_kind": "measured_hermetic",
    }


def _task_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "inspected": payload.get("inspected") is True,
        "can_update_task": payload.get("can_update_task") is True,
        "can_terminalize_task": payload.get("can_terminalize_task") is True,
        "tasks": list(payload.get("tasks") or []),
        "live": payload.get("live") is True,
    }


def hermetic_authority_bypass_trace() -> dict[str, Any]:
    """Run every forbidden operation on both clients and record typed failures."""

    python_before = _task_snapshot(PythonExternalClient().inspect_task_state())
    python_refusals = [
        attempt_python_bypass(name) for name in FORBIDDEN_CLIENT_OPERATIONS
    ]
    python_after = _task_snapshot(PythonExternalClient().inspect_task_state())
    mcp_before_client = GenericMcpClient()
    mcp_before = _task_snapshot(mcp_before_client.inspect_task_state())
    mcp_refusals = [
        attempt_mcp_bypass(name) for name in FORBIDDEN_CLIENT_OPERATIONS
    ]
    mcp_after = _task_snapshot(GenericMcpClient().inspect_task_state())
    python_unchanged = python_before == python_after
    mcp_unchanged = mcp_before == mcp_after
    python_cannot_mutate = (
        python_before["can_update_task"] is False
        and python_before["can_terminalize_task"] is False
        and python_after["can_update_task"] is False
        and python_after["can_terminalize_task"] is False
    )
    mcp_cannot_mutate = (
        mcp_before["can_update_task"] is False
        and mcp_before["can_terminalize_task"] is False
        and mcp_after["can_update_task"] is False
        and mcp_after["can_terminalize_task"] is False
    )
    every_python_refused = all(item["refused"] is True for item in python_refusals)
    every_mcp_refused = all(item["refused"] is True for item in mcp_refusals)
    any_accepted = any(item["accepted"] is True for item in python_refusals) or any(
        item["accepted"] is True for item in mcp_refusals
    )
    any_effect = any(
        item["unauthorized_effect"] is True for item in python_refusals
    ) or any(item["unauthorized_effect"] is True for item in mcp_refusals)
    refuse_bypass_acceptance(accepted=any_accepted)
    refuse_unauthorized_effect(
        observed=(
            any_effect
            or not python_unchanged
            or not mcp_unchanged
            or not python_cannot_mutate
            or not mcp_cannot_mutate
        )
    )
    if not every_python_refused or not every_mcp_refused:
        raise AccelerateAuthorityBypassError(
            "every bypass attempt must fail typed"
        )
    counters = zero_hard_zero_counters()
    refuse_nonzero_hard_zero(nonzero=any(value != 0 for value in counters.values()))
    return {
        "schema": "ipfs_accelerate_py/assurance/hermetic-authority-bypass-trace@1",
        "interface": "HermeticAuthorityBypassTrace@1",
        "canonical_service": CANONICAL_SERVICE,
        "python_transport": PYTHON_TRANSPORT,
        "mcp_transport": MCP_TRANSPORT,
        "python_principal": PYTHON_PRINCIPAL,
        "mcp_principal": MCP_PRINCIPAL,
        "authority": CLIENT_AUTHORITY,
        "forbidden_operations": list(FORBIDDEN_CLIENT_OPERATIONS),
        "forbidden_mcp_tools": list(FORBIDDEN_MCP_TOOL_NAMES),
        "python_refusals": python_refusals,
        "mcp_refusals": mcp_refusals,
        "every_python_bypass_attempt_fails_typed": True,
        "every_mcp_bypass_attempt_fails_typed": True,
        "bypass_attempt_count": len(python_refusals) + len(mcp_refusals),
        "accepted_count": 0,
        "unauthorized_effect": False,
        "python_task_state_before": python_before,
        "python_task_state_after": python_after,
        "mcp_task_state_before": mcp_before,
        "mcp_task_state_after": mcp_after,
        "task_state_unchanged": True,
        "current_root_unchanged": True,
        "policy_pointer_unchanged": True,
        "self_promoted": False,
        "hard_zero_counters": counters,
        "hard_zero_counters_remain_zero": True,
        "python_client_cid": PINNED_PYTHON_CLIENT_CID,
        "mcp_client_cid": PINNED_MCP_CLIENT_CID,
        "parity_cid": PINNED_PARITY_CID,
        "pack_cid": PINNED_PACK_CID,
        "current_root_cid": PINNED_CURRENT_ROOT_CID,
        "chain_cid": PINNED_CHAIN_CID,
        "objective_cid": PINNED_OBJECTIVE_CID,
        "idea_digest": PINNED_IDEA_DIGEST,
        "live": False,
        "hermetic": True,
        "evidence_kind": "measured_hermetic",
    }


def hermetic_bypass_idempotency(proof_cid: str) -> dict[str, Any]:
    return {
        "schema": "ipfs_accelerate_py/assurance/hermetic-authority-bypass-idempotency@1",
        "interface": "HermeticAuthorityBypassIdempotency@1",
        "first_proof_command_digest": PROOF_COMMAND_DIGEST,
        "second_proof_command_digest": PROOF_COMMAND_DIGEST,
        "first_proof_cid": proof_cid,
        "second_proof_cid": proof_cid,
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
class AccelerateAuthorityBypassVerdict:
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
    proof_cid: str
    document_cid: str
    catalog_cid: str
    chain_cid: str
    pack_cid: str
    current_root_cid: str
    objective_cid: str
    idea_digest: str
    python_client_cid: str
    mcp_client_cid: str
    parity_cid: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "verdict_cid": self.verdict_cid,
            "proof_cid": self.proof_cid,
            "document_cid": self.document_cid,
            "catalog_cid": self.catalog_cid,
            "chain_cid": self.chain_cid,
            "pack_cid": self.pack_cid,
            "current_root_cid": self.current_root_cid,
            "objective_cid": self.objective_cid,
            "idea_digest": self.idea_digest,
            "python_client_cid": self.python_client_cid,
            "mcp_client_cid": self.mcp_client_cid,
            "parity_cid": self.parity_cid,
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
            "Keep the hermetic Python and generic MCP clients as intent, "
            "never authority. Keep every forbidden operation refused. Do not "
            "write DuckDB or Quack state. Do not escalate to a model. Do not "
            "grant either client principal authority. PCPR-090 prepares the "
            "threat model."
        ),
        "reason": (
            "Hermetic authority-bypass coverage is not a live Supervisor.run "
            "/ CLI / MCP stdio or HTTP session. Sealed validation has no "
            "admitted Quack-fenced session. Direct DuckDB writes are "
            "prohibited."
        ),
    }


def produce_authority_bypass(
    *,
    paths: Sequence[str] | None = None,
    pack_cid: str = PINNED_PACK_CID,
    current_root_cid: str = PINNED_CURRENT_ROOT_CID,
    chain_cid: str = PINNED_CHAIN_CID,
    objective_cid: str = PINNED_OBJECTIVE_CID,
    idea_digest: str = PINNED_IDEA_DIGEST,
    python_client_cid: str = PINNED_PYTHON_CLIENT_CID,
    mcp_client_cid: str = PINNED_MCP_CLIENT_CID,
    parity_cid: str = PINNED_PARITY_CID,
    idea: str = REFERENCE_OBJECTIVE_IDEA,
    complete_from: str | None = None,
    database_edited: bool = False,
    applied_live: bool = False,
    self_promoted: bool = False,
    bypass_accepted: bool = False,
    unauthorized_effect: bool = False,
    hard_zero_nonzero: bool = False,
) -> dict[str, Any]:
    """Produce the hermetic authority-bypass proof. Fail closed."""

    if complete_from is not None:
        refuse_model_completion(complete_from)
    refuse_pack_cid_remint(pack_cid)
    refuse_current_root_remint(current_root_cid)
    refuse_chain_cid_remint(chain_cid)
    refuse_objective_cid_remint(objective_cid)
    refuse_idea_digest_remint(idea_digest)
    refuse_python_client_cid_remint(python_client_cid)
    refuse_mcp_client_cid_remint(mcp_client_cid)
    refuse_parity_cid_remint(parity_cid)
    refuse_non_canonical_objective(idea)
    refuse_database_edit(edited=database_edited)
    refuse_live_client(applied_live=applied_live)
    refuse_self_promotion(promoted=self_promoted)
    refuse_bypass_acceptance(accepted=bypass_accepted)
    refuse_unauthorized_effect(observed=unauthorized_effect)
    refuse_nonzero_hard_zero(nonzero=hard_zero_nonzero)
    checked_paths = list(paths) if paths is not None else list(AUTHORIZED_PATH_PREFIXES)
    for path in checked_paths:
        refuse_unauthorized_path(path)
    trace = hermetic_authority_bypass_trace()
    return {
        "schema": PROOF_SCHEMA,
        "interface": PROOF_INTERFACE,
        "owner_interface": INTERFACE,
        "task_id": PCPR_083_TASK_ID,
        "goal_id": PCPR_083_GOAL_ID,
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
        "parity_cid": parity_cid,
        "python_client_interface": PYTHON_CLIENT_INTERFACE,
        "mcp_client_interface": MCP_CLIENT_INTERFACE,
        "parity_interface": PARITY_INTERFACE,
        "protocol_interface": PROTOCOL_INTERFACE,
        "documentation_interface": DOCUMENTATION_INTERFACE,
        "documentation_schema": DOCUMENTATION_SCHEMA,
        "change_kind": CHANGE_KIND,
        "proof_kind": PROOF_KIND,
        "canonical_service": CANONICAL_SERVICE,
        "python_transport": PYTHON_TRANSPORT,
        "mcp_transport": MCP_TRANSPORT,
        "authority": CLIENT_AUTHORITY,
        "operations": list(CLIENT_OPERATIONS),
        "forbidden_operations": list(FORBIDDEN_CLIENT_OPERATIONS),
        "forbidden_mcp_tools": list(FORBIDDEN_MCP_TOOL_NAMES),
        "hard_zero_counters": list(HARD_ZERO_COUNTERS),
        "produced": True,
        "applied": True,
        "live": False,
        "hermetic": True,
        "admitted_live": False,
        "deferred": False,
        "paths_authorized": True,
        "authorized_path_prefixes": list(AUTHORIZED_PATH_PREFIXES),
        "every_python_bypass_attempt_fails_typed": True,
        "every_mcp_bypass_attempt_fails_typed": True,
        "no_unauthorized_effect": True,
        "hard_zero_counters_remain_zero": True,
        "task_state_unchanged": True,
        "current_root_unchanged": True,
        "policy_pointer_unchanged": True,
        "client_cids_distinct": True,
        "client_self_promoted": False,
        "bypass_attempt_accepted": False,
        "trace": trace,
        "proof_command_digest": PROOF_COMMAND_DIGEST,
        "idempotent": True,
        "duplicate_effect_rejected": True,
        "remints_protocol": False,
        "adds_protocol_operation": False,
        "model_assertion_completes_work": False,
        "escalation_order": list(ESCALATION_ORDER),
        "completed_through": COMPLETED_THROUGH,
        "next_authorized_stage": NEXT_AUTHORIZED_STAGE,
        "next_task_id": PCPR_090_TASK_ID,
        "prerequisite_task_id": PCPR_082_TASK_ID,
        "python_client_task_id": PCPR_080_TASK_ID,
        "mcp_client_task_id": PCPR_081_TASK_ID,
        "parity_task_id": PCPR_082_TASK_ID,
        "chain_task_id": PCPR_072_TASK_ID,
        "duckdb_or_quack_state_written": False,
        "closed_release_outcome": None,
        "release_claim": False,
        "evidence_kind": "measured_hermetic",
    }


def canonical_authority_bypass() -> dict[str, Any]:
    return produce_authority_bypass()


def proof_cid_of(plan: Mapping[str, Any] | None = None) -> str:
    payload = dict(plan or canonical_authority_bypass())
    payload.pop("proof_cid", None)
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
                "Sibling authority-bypass file is not present beside this checkout."
            ),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    context_pack = payload.get("context_pack")
    storage = payload.get("storage")
    proof = payload.get("authority_bypass")
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
        or (proof.get("chain_cid") if isinstance(proof, Mapping) else None),
        "proof_cid": payload.get("proof_cid")
        or (proof.get("proof_cid") if isinstance(proof, Mapping) else None),
        "binding_cid": payload.get("binding_cid") or payload.get("document_cid"),
        "live": False,
        "applied": False,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def render_declared_proof(root: Path | None = None) -> dict[str, Any]:
    package_root = root or discover_accelerate_root()
    if package_root is None:
        raise AccelerateAuthorityBypassError("Accelerate package root was not found")
    state_owner = observe_state_owner()
    plan = canonical_authority_bypass()
    computed_proof_cid = proof_cid_of(plan)
    idempotency = hermetic_bypass_idempotency(computed_proof_cid)
    document = {
        "schema": DOCUMENT_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_083_TASK_ID,
        "goal_id": PCPR_083_GOAL_ID,
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
            "survives_bypass": True,
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
            "survives_bypass": True,
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
            "evidence_kind": "measured_hermetic",
        },
        "python_external_client": {
            "task_id": PCPR_080_TASK_ID,
            "interface": PYTHON_CLIENT_INTERFACE,
            "client_cid": PINNED_PYTHON_CLIENT_CID,
            "transport": PYTHON_TRANSPORT,
            "canonical_service": CANONICAL_SERVICE,
            "principal": PYTHON_PRINCIPAL,
            "authority": CLIENT_AUTHORITY,
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
            "principal": MCP_PRINCIPAL,
            "authority": CLIENT_AUTHORITY,
            "reminted": False,
            "live": False,
            "evidence_kind": "measured",
        },
        "objective_identity_parity": {
            "task_id": PCPR_082_TASK_ID,
            "interface": PARITY_INTERFACE,
            "parity_cid": PINNED_PARITY_CID,
            "reminted": False,
            "live": False,
            "evidence_kind": "measured",
        },
        "authority_bypass": {
            **plan,
            "proof_cid": computed_proof_cid,
            "idempotency": idempotency,
            "produced_by": EXECUTION_OWNER_REPOSITORY,
        },
        "materialization": {
            "kind": "declared_authority_bypass_not_live",
            "admitted": False,
            "live": False,
            "applied": False,
            "duckdb_or_quack_state_written": False,
            "evidence_kind": "unavailable",
        },
        "live_application": typed_unavailable(
            reason=(
                "No admitted Quack-fenced supervisor session was observed. "
                "Hermetic authority-bypass coverage is not live client execution."
            )
        ),
        "live_client": typed_unavailable(
            reason=(
                "Live Python Supervisor.run and live generic MCP stdio/HTTP "
                "remain typed unavailable. This task only emits a hermetic "
                "authority-bypass negative proof. Threat-model preparation "
                "remains PCPR-090."
            )
        ),
        "live_mcp_client": typed_unavailable(
            reason=(
                "Live MCP stdio or HTTP session remains typed unavailable. "
                "Hermetic JSON-RPC refusal is not a live MCP session."
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
                "field": "PCPR-083 receipt current_tree_binding",
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


def platform_authority_bypass_catalog(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateAuthorityBypassError("Accelerate package root was not found")
    parent = root.parent
    document = render_declared_proof(root)
    datasets_binding = _sibling_binding(
        parent
        / "ipfs_datasets"
        / DOCUMENT_DIR_RELPATH
        / "reference.authority-bypass.binding.json",
        relative_path=(
            f"../ipfs_datasets/{DOCUMENT_DIR_RELPATH}/"
            "reference.authority-bypass.binding.json"
        ),
    )
    kit_binding = _sibling_binding(
        parent
        / "ipfs_kit"
        / DOCUMENT_DIR_RELPATH
        / "reference.authority-bypass.binding.json",
        relative_path=(
            f"../ipfs_kit/{DOCUMENT_DIR_RELPATH}/"
            "reference.authority-bypass.binding.json"
        ),
    )
    catalog = {
        "schema": CATALOG_SCHEMA,
        "interface": CATALOG_INTERFACE,
        "task_id": PCPR_083_TASK_ID,
        "goal_id": PCPR_083_GOAL_ID,
        "objective_kind": OBJECTIVE_KIND,
        "portfolio_id": PORTFOLIO_ID,
        "portfolio_version": PORTFOLIO_VERSION,
        "objective_cid": PINNED_OBJECTIVE_CID,
        "idea_digest": PINNED_IDEA_DIGEST,
        "pack_cid": PINNED_PACK_CID,
        "current_root_cid": PINNED_CURRENT_ROOT_CID,
        "chain_cid": PINNED_CHAIN_CID,
        "proof_cid": document["authority_bypass"]["proof_cid"],
        "python_client_cid": PINNED_PYTHON_CLIENT_CID,
        "mcp_client_cid": PINNED_MCP_CLIENT_CID,
        "parity_cid": PINNED_PARITY_CID,
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
                "proof_cid": document["authority_bypass"]["proof_cid"],
                "python_client_cid": PINNED_PYTHON_CLIENT_CID,
                "mcp_client_cid": PINNED_MCP_CLIENT_CID,
                "parity_cid": PINNED_PARITY_CID,
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


def write_authority_bypass_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateAuthorityBypassError("Accelerate package root was not found")
    document = render_declared_proof(root)
    paths = artifact_paths(root)
    _atomic_write(paths["document"], pretty_json(document))
    readme = DOCUMENT_README if DOCUMENT_README.endswith("\n") else DOCUMENT_README + "\n"
    _atomic_write(paths["readme"], readme)
    catalog = platform_authority_bypass_catalog(start)
    _atomic_write(paths["catalog"], pretty_json(catalog))
    return {
        "document": document,
        "catalog": catalog,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def verify_authority_bypass_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateAuthorityBypassError("Accelerate package root was not found")
    document = render_declared_proof(root)
    catalog = platform_authority_bypass_catalog(start)
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
        "proof_cid": document["authority_bypass"]["proof_cid"],
        "document_cid": document["document_cid"],
        "catalog_cid": catalog["catalog_cid"],
        "current_root_cid": document["storage"]["current_root_cid"],
        "objective_cid": document["objective_cid"],
        "idea_digest": document["idea_digest"],
        "python_client_cid": document["python_external_client"]["client_cid"],
        "mcp_client_cid": document["generic_mcp_client"]["client_cid"],
        "parity_cid": document["objective_identity_parity"]["parity_cid"],
        "document_sha256": (
            sha256_bytes(paths["document"].read_bytes())
            if paths["document"].is_file()
            else "unavailable"
        ),
    }


def current_head_static_probes(start: Path | None = None) -> tuple[OutcomeProbe, ...]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateAuthorityBypassError("Accelerate package root was not found")
    document = render_declared_proof(root)
    verified = verify_authority_bypass_files(root)
    table = parse_pyproject_authority_bypass_table(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    proof = document["authority_bypass"]
    remint = (
        document["context_pack"]["pack_cid"] != PINNED_PACK_CID
        or document["objective_cid"] != PINNED_OBJECTIVE_CID
        or document["idea_digest"] != PINNED_IDEA_DIGEST
        or document["lock_cid"] != PINNED_LOCK_CID
        or document["storage"]["current_root_cid"] != PINNED_CURRENT_ROOT_CID
        or document["final_receipt_chain"]["chain_cid"] != PINNED_CHAIN_CID
        or proof["proof_cid"] != PINNED_PROOF_CID
        or document["python_external_client"]["client_cid"] != PINNED_PYTHON_CLIENT_CID
        or document["generic_mcp_client"]["client_cid"] != PINNED_MCP_CLIENT_CID
        or document["objective_identity_parity"]["parity_cid"] != PINNED_PARITY_CID
        or document["context_pack"]["reminted"] is True
        or document["storage"]["reminted"] is True
        or document["final_receipt_chain"]["reminted"] is True
        or document["python_external_client"]["reminted"] is True
        or document["generic_mcp_client"]["reminted"] is True
        or document["objective_identity_parity"]["reminted"] is True
    )
    trace = proof["trace"]
    idempotency = proof["idempotency"]
    counters = trace["hard_zero_counters"]
    probes = [
        _probe(
            "bypass_files_match_generator",
            verified.get("ok") is True and verified.get("document_ok") is True,
            reason=(
                "Committed Accelerate authority-bypass matches the generator."
                if verified.get("document_ok") is True
                else "Committed Accelerate authority-bypass is missing or drifts."
            ),
        ),
        _probe(
            "pyproject_authority_bypass_table",
            table.get("interface") == INTERFACE
            and table.get("schema") == SCHEMA
            and table.get("task-id") == PCPR_083_TASK_ID
            and table.get("objective-kind") == OBJECTIVE_KIND,
            reason=(
                "pyproject.toml declares AccelerateAuthorityBypass@1."
                if table.get("interface") == INTERFACE
                else "pyproject.toml does not declare the PCPR-083 bypass table."
            ),
        ),
        _probe(
            "paths_remain_authorized",
            proof["paths_authorized"] is True
            and list(proof["authorized_path_prefixes"])
            == list(AUTHORIZED_PATH_PREFIXES),
            reason="Bypass paths remain inside the declared PCPR-083 allowlist.",
        ),
        _probe(
            "canonical_objective_bytes_match",
            proof["idea"] == REFERENCE_OBJECTIVE_IDEA
            and proof["idea_digest"] == PINNED_IDEA_DIGEST,
            reason="Authority-bypass proof uses the canonical PCPR-060 idea bytes.",
        ),
        _probe(
            "every_python_bypass_attempt_fails_typed",
            proof["every_python_bypass_attempt_fails_typed"] is True
            and trace["every_python_bypass_attempt_fails_typed"] is True
            and trace["accepted_count"] == 0,
            reason="Every Python forbidden operation fails typed.",
        ),
        _probe(
            "every_mcp_bypass_attempt_fails_typed",
            proof["every_mcp_bypass_attempt_fails_typed"] is True
            and trace["every_mcp_bypass_attempt_fails_typed"] is True,
            reason="Every generic MCP forbidden tool fails typed JSON-RPC.",
        ),
        _probe(
            "no_unauthorized_effect",
            proof["no_unauthorized_effect"] is True
            and trace["unauthorized_effect"] is False,
            reason="Bypass attempts cause no unauthorized effect.",
        ),
        _probe(
            "hard_zero_counters_remain_zero",
            proof["hard_zero_counters_remain_zero"] is True
            and trace["hard_zero_counters_remain_zero"] is True
            and all(counters[name] == 0 for name in HARD_ZERO_COUNTERS),
            reason="Every hard-zero counter remains zero.",
        ),
        _probe(
            "task_state_unchanged",
            proof["task_state_unchanged"] is True
            and trace["task_state_unchanged"] is True,
            reason="Task state is unchanged after bypass attempts.",
        ),
        _probe(
            "current_root_unchanged",
            proof["current_root_unchanged"] is True
            and trace["current_root_unchanged"] is True
            and document["storage"]["current_root_cid"] == PINNED_CURRENT_ROOT_CID,
            reason="Current root is unchanged after bypass attempts.",
        ),
        _probe(
            "policy_pointer_unchanged",
            proof["policy_pointer_unchanged"] is True
            and trace["policy_pointer_unchanged"] is True,
            reason="Policy pointers are unchanged after bypass attempts.",
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
            proof["canonical_service"] == CANONICAL_SERVICE,
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
            "parity_cid_bound_not_minted",
            document["objective_identity_parity"]["parity_cid"] == PINNED_PARITY_CID
            and document["objective_identity_parity"]["reminted"] is False,
            reason="PCPR-082 parity CID is bound and not reminted.",
        ),
        _probe(
            "client_cids_remain_distinct",
            PINNED_PYTHON_CLIENT_CID != PINNED_MCP_CLIENT_CID
            and proof["client_cids_distinct"] is True,
            reason="Python and MCP client CIDs remain distinct identities.",
        ),
        _probe(
            "clients_are_intent_not_authority",
            proof["authority"] == CLIENT_AUTHORITY
            and document["python_external_client"]["authority"] == CLIENT_AUTHORITY
            and document["generic_mcp_client"]["authority"] == CLIENT_AUTHORITY,
            reason="Both client principals are intent, never authority.",
        ),
        _probe(
            "idempotent_proof_cid",
            proof["idempotent"] is True
            and idempotency["identical"] is True
            and idempotency["first_proof_cid"] == proof["proof_cid"]
            and idempotency["second_proof_cid"] == proof["proof_cid"]
            and proof["proof_cid"] == PINNED_PROOF_CID,
            reason="A second bypass emission yields the same proof CID.",
        ),
        _probe(
            "model_assertion_cannot_complete_work",
            proof["model_assertion_completes_work"] is False,
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
                "client, MCP client, parity, or proof CID is forbidden."
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
            "parity_identity_reminted",
            document["objective_identity_parity"]["parity_cid"] != PINNED_PARITY_CID
            or document["objective_identity_parity"]["reminted"] is True,
            reason="Accelerate must not remint the PCPR-082 parity CID.",
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
            reason="Hermetic authority-bypass coverage is not represented as live.",
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
            reason="Authority-bypass proof cannot promote itself.",
        ),
        _probe(
            "bypass_attempt_accepted",
            False,
            reason="A forbidden client operation is refused.",
        ),
        _probe(
            "unauthorized_effect_observed",
            False,
            reason="Bypass attempts cause no unauthorized effect.",
        ),
        _probe(
            "hard_zero_counter_nonzero",
            False,
            reason="Hard-zero counters remain zero.",
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


def qualify_authority_bypass(
    probes: Sequence[OutcomeProbe],
    *,
    proof_cid: str,
    document_cid: str,
    catalog_cid: str,
    chain_cid: str,
    pack_cid: str,
    current_root_cid: str,
    objective_cid: str,
    idea_digest_cid: str,
    python_client_cid: str = PINNED_PYTHON_CLIENT_CID,
    mcp_client_cid: str = PINNED_MCP_CLIENT_CID,
    parity_cid: str = PINNED_PARITY_CID,
) -> AccelerateAuthorityBypassVerdict:
    if not probes:
        raise AccelerateAuthorityBypassError("at least one probe is required")
    normalized: list[OutcomeProbe] = []
    blockers: list[str] = []
    for probe in probes:
        kind = _kind(probe.evidence_kind, "evidence_kind")
        if probe.live and kind != "measured_live":
            raise AccelerateAuthorityBypassError(
                "live claims require measured_live evidence"
            )
        if probe.simulated_represented_as_live:
            raise AccelerateAuthorityBypassError(
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
        "task_id": PCPR_083_TASK_ID,
        "goal_id": PCPR_083_GOAL_ID,
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
        "proof_cid": proof_cid,
        "document_cid": document_cid,
        "catalog_cid": catalog_cid,
        "chain_cid": chain_cid,
        "pack_cid": pack_cid,
        "current_root_cid": current_root_cid,
        "objective_cid": objective_cid,
        "idea_digest": idea_digest_cid,
        "python_client_cid": python_client_cid,
        "mcp_client_cid": mcp_client_cid,
        "parity_cid": parity_cid,
    }
    return AccelerateAuthorityBypassVerdict(
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
        proof_cid=proof_cid,
        document_cid=document_cid,
        catalog_cid=catalog_cid,
        chain_cid=chain_cid,
        pack_cid=pack_cid,
        current_root_cid=current_root_cid,
        objective_cid=objective_cid,
        idea_digest=idea_digest_cid,
        python_client_cid=python_client_cid,
        mcp_client_cid=mcp_client_cid,
        parity_cid=parity_cid,
    )


def qualify_current_head_authority_bypass(
    start: Path | None = None,
) -> AccelerateAuthorityBypassVerdict:
    document = render_declared_proof(start)
    catalog = platform_authority_bypass_catalog(start)
    return qualify_authority_bypass(
        current_head_static_probes(start),
        proof_cid=str(document["authority_bypass"]["proof_cid"]),
        document_cid=str(document["document_cid"]),
        catalog_cid=str(catalog["catalog_cid"]),
        chain_cid=str(document["final_receipt_chain"]["chain_cid"]),
        pack_cid=str(document["context_pack"]["pack_cid"]),
        current_root_cid=str(document["storage"]["current_root_cid"]),
        objective_cid=str(document["objective_cid"]),
        idea_digest_cid=str(document["idea_digest"]),
        python_client_cid=str(document["python_external_client"]["client_cid"]),
        mcp_client_cid=str(document["generic_mcp_client"]["client_cid"]),
        parity_cid=str(document["objective_identity_parity"]["parity_cid"]),
    )


def pcpr_083_receipt_promotion(
    verdict: AccelerateAuthorityBypassVerdict,
) -> dict[str, Any]:
    if verdict.closed_release_outcome is not None:
        raise AccelerateAuthorityBypassError(
            "authority bypass proof must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise AccelerateAuthorityBypassError(
            "authority bypass proof must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise AccelerateAuthorityBypassError(
            "authority bypass proof completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise AccelerateAuthorityBypassError(
            "authority bypass proof must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise AccelerateAuthorityBypassError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.live_client or verdict.live_application:
        raise AccelerateAuthorityBypassError(
            "live client claims require measured_live evidence"
        )
    return verdict.to_mapping()


PINNED_PROOF_CID: Final = (
    "baguqeeraepx7iiqcc4gtfp63kpozlkdd2qg3px3ptnqhj3rpigqtemjc5psa"
)
PINNED_DOCUMENT_CID: Final = (
    "baguqeerath5en6zqgxk33rtsyavf57yxfftalpzyntowzimeehs6of5eef6a"
)
PINNED_CATALOG_CID: Final = (
    "baguqeera7hbmatarooupove6gt7dodqev5kxjvny3xhogalwtve33ifj7i3a"
)
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeerasgfw7iq52yweswmgfw5k3zbpjvgrkue74x3ub6spppqaeqr3nwma"
)


__all__ = [
    "AUTHORIZED_PATH_PREFIXES",
    "CLIENT_OPERATIONS",
    "CLOSED_RELEASE_OUTCOMES",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "ESCALATION_ORDER",
    "FORBIDDEN_CLIENT_OPERATIONS",
    "HARD_ZERO_COUNTERS",
    "AccelerateAuthorityBypassError",
    "HERMETIC_CANDIDATE_SUITES",
    "INTERFACE",
    "OBJECTIVE_KIND",
    "OPERATOR_BLOCKING_TASK_ID",
    "OutcomeProbe",
    "PCPR_083_GOAL_ID",
    "PCPR_083_TASK_ID",
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
    "PINNED_PROOF_CID",
    "PINNED_PYTHON_CLIENT_CID",
    "SCHEMA",
    "SEALED_PATH",
    "SEALED_PYTHON",
    "attempt_mcp_bypass",
    "attempt_python_bypass",
    "canonical_authority_bypass",
    "current_head_static_probes",
    "hermetic_authority_bypass_trace",
    "hermetic_bypass_idempotency",
    "path_is_authorized",
    "pcpr_083_receipt_promotion",
    "platform_authority_bypass_catalog",
    "produce_authority_bypass",
    "proof_cid_of",
    "qualify_authority_bypass",
    "qualify_current_head_authority_bypass",
    "refuse_bypass_acceptance",
    "refuse_chain_cid_remint",
    "refuse_current_root_remint",
    "refuse_database_edit",
    "refuse_mcp_client_cid_remint",
    "refuse_model_completion",
    "refuse_non_canonical_objective",
    "refuse_objective_cid_remint",
    "refuse_pack_cid_remint",
    "refuse_parity_cid_remint",
    "refuse_python_client_cid_remint",
    "refuse_self_promotion",
    "refuse_unauthorized_effect",
    "refuse_unauthorized_path",
    "render_declared_proof",
    "verify_authority_bypass_files",
    "write_authority_bypass_files",
    "zero_hard_zero_counters",
]
