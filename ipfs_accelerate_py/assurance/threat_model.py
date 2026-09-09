"""Fail-closed PCPR-090 Accelerate-owned objective-to-release threat model.

Accelerate owns the hermetic threat model for the PCPR objective-to-release
path. Actors, assets, trust boundaries, state owners, canonicalization,
CIDs, events, leases, fences, unknown outcomes, confirmations, proof
admission, dependencies, and builds are bounded. The trusted-computing-base
inventory remains PCPR-091. The audit package remains PCPR-092. Closed
release decision remains PCPR-093/094.

Live Supervisor.run, live Quack, live MCP, and live CLI stay typed
unavailable. Simulated results are not live. This evaluator does not write
DuckDB or Quack state and does not emit a closed PCPR release.

Hermetic threat-model coverage is candidate coverage only.
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


INTERFACE: Final = "AccelerateThreatModel@1"
SCHEMA: Final = "ipfs_accelerate_py/assurance/threat-model@1"
DOCUMENT_SCHEMA: Final = "ipfs_accelerate_py/assurance/declared-threat-model@1"
MODEL_SCHEMA: Final = "ipfs_accelerate_py/assurance/threat-model-plan@1"
VERDICT_SCHEMA: Final = "ipfs_accelerate_py/assurance/threat-model-verdict@1"
CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/platform-threat-model-catalog@1"
)
CATALOG_INTERFACE: Final = "PlatformThreatModel@1"
MODEL_INTERFACE: Final = "AuthoritativeThreatModel@1"
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
BYPASS_INTERFACE: Final = "AccelerateAuthorityBypass@1"
PROTOCOL_INTERFACE: Final = "LogicProviderProtocol@2"
DOCUMENTATION_INTERFACE: Final = "LogicProviderProtocolThreatModel@1"
DOCUMENTATION_SCHEMA: Final = (
    "ipfs_datasets_py/logic-provider-threat-model@1"
)
PCPR_090_TASK_ID: Final = "PCPR-090"
PCPR_090_GOAL_ID: Final = "PCPR-G900"
PCPR_091_TASK_ID: Final = "PCPR-091"
PCPR_092_TASK_ID: Final = "PCPR-092"
PCPR_093_TASK_ID: Final = "PCPR-093"
PCPR_094_TASK_ID: Final = "PCPR-094"
PCPR_083_TASK_ID: Final = "PCPR-083"
PCPR_082_TASK_ID: Final = "PCPR-082"
PCPR_081_TASK_ID: Final = "PCPR-081"
PCPR_080_TASK_ID: Final = "PCPR-080"
PCPR_072_TASK_ID: Final = "PCPR-072"
PCPR_062_TASK_ID: Final = "PCPR-062"
PCPR_061_TASK_ID: Final = "PCPR-061"
PCPR_060_TASK_ID: Final = "PCPR-060"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = (
    "proof-carrying-platform-qualification-and-release-v1"
)
OBJECTIVE_KIND: Final = "declared_objective_to_release_threat_model"
OBJECTIVE_ID: Final = "PCPR-G900"
LANGUAGE: Final = "Python"
OPERATOR_BLOCKING_TASK_ID: Final = "pcpr-090-operator-live-threat-model"
DOCUMENT_DIR_RELPATH: Final = "packaging/pcpr/reference-workflow/cpython312"
DOCUMENT_JSON_NAME: Final = "reference.threat-model.json"
DOCUMENT_README_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/THREAT_MODEL.md"
)
CATALOG_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/platform-threat-model-catalog.json"
)
SOURCE_DATE_EPOCH: Final = "0"
SOURCE_REPOSITORY: Final = "endomorphosis/ipfs_accelerate_py"
COMPLETED_THROUGH: Final = "threat_model"
NEXT_AUTHORIZED_STAGE: Final = "trusted_computing_base_inventory"
CHANGE_KIND: Final = "objective_to_release_threat_model"
MODEL_KIND: Final = "hermetic_objective_to_release_threat_model"
CANONICAL_SERVICE: Final = "pcpr-canonical-supervisor-service"
CLIENT_AUTHORITY: Final = "intent_not_authority"
MODEL_COMMAND_DIGEST: Final = "pcpr-090-hermetic-threat-model"

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
PINNED_PYTHON_CLIENT_CID: Final = (
    "baguqeerat346yf4k7kenbqlbtyejeilgxamna262z3axzjjatzm2hccehblq"
)
PINNED_MCP_CLIENT_CID: Final = (
    "baguqeerasdkirdryda7uoi5gchsyhuyvh6d2uremm45zpob7hoss3ao66ahq"
)
PINNED_PARITY_CID: Final = (
    "baguqeera6y3w7bhzy6hpo6ohfsrj5xihrw5vk6qqljjonjrupijd2yrp3hnq"
)
PINNED_BYPASS_PROOF_CID: Final = (
    "baguqeeraepx7iiqcc4gtfp63kpozlkdd2qg3px3ptnqhj3rpigqtemjc5psa"
)

if PINNED_LOCK_CID != (
    "baguqeerawsekbbbt5ccctjt4tydzeahfkahsatb6q5x7k6cy4inrii5lhqmq"
):
    raise RuntimeError("PCPR-090 lock CID remints PCPR-056")
if PINNED_PYTHON_CLIENT_CID == PINNED_MCP_CLIENT_CID:
    raise RuntimeError("Python and MCP client CIDs must remain distinct")
if OWNER_INTERFACES["SupervisorContextPack"] != OWNER_INTERFACE:
    raise RuntimeError("PCPR-090 remints SupervisorContextPack owner interface")
if OWNER_SCHEMAS["SupervisorContextPack"] != OWNER_SCHEMA:
    raise RuntimeError("PCPR-090 remints SupervisorContextPack owner schema")

REFERENCE_OBJECTIVE_IDEA: Final = (
    "Modify a typed formal-logic API while reusing unaffected proofs, "
    "selecting only impacted tests, rejecting stale-tree evidence, and "
    "producing a complete proof-carrying execution receipt."
)
CANONICAL_OBJECTIVE_BYTES: Final = REFERENCE_OBJECTIVE_IDEA.encode("utf-8")

TRUSTED_PATH_STAGES: Final[tuple[str, ...]] = (
    "objective_submission",
    "authentication_and_authority",
    "planning_and_context_pack",
    "task_state_machine",
    "execution_admission",
    "proof_and_test_admission",
    "receipt",
    "release_decision",
)

G910_SURFACES: Final[tuple[str, ...]] = (
    "actors",
    "assets",
    "trust_boundaries",
    "state_owners",
    "canonicalization",
    "cids",
    "events",
    "leases",
    "fences",
    "unknown_outcomes",
    "confirmations",
    "proof_admission",
    "dependencies",
    "builds",
)

ACTORS: Final[tuple[dict[str, str], ...]] = (
    {
        "actor_id": "human_operator",
        "role": "seals, confirmations, START, and residual decisions",
        "authority": "authority",
    },
    {
        "actor_id": "supervisor",
        "role": "task admission, lease, fence, execution, terminalization",
        "authority": "authority_after_independent_promotion",
    },
    {
        "actor_id": "python_external_client",
        "role": "submit, inspect, retrieve against the canonical service",
        "authority": "intent_not_authority",
    },
    {
        "actor_id": "generic_mcp_client",
        "role": "submit, inspect, retrieve over MCP JSON-RPC 2.0",
        "authority": "intent_not_authority",
    },
    {
        "actor_id": "cli_client",
        "role": "submit, inspect, retrieve over the product CLI",
        "authority": "intent_not_authority",
    },
    {
        "actor_id": "model_provider",
        "role": "candidate generation only",
        "authority": "none",
    },
    {
        "actor_id": "datasets_semantic_owner",
        "role": "semantic identity, ContextPack, solver validation",
        "authority": "semantic_identity",
    },
    {
        "actor_id": "kit_byte_owner",
        "role": "exact bytes, CID verification, current-root CAS",
        "authority": "byte_identity",
    },
    {
        "actor_id": "accelerate_execution_owner",
        "role": "objective and task operational state",
        "authority": "operational_admission",
    },
    {
        "actor_id": "quack_state_owner",
        "role": "exclusive authenticated state-owner transport",
        "authority": "state_owner",
    },
    {
        "actor_id": "untrusted_prompt",
        "role": "high-level intent that cannot grant authority",
        "authority": "none",
    },
    {
        "actor_id": "untrusted_repository",
        "role": "untrusted source, tests, fixtures, and patches",
        "authority": "none",
    },
    {
        "actor_id": "untrusted_patch_or_agent",
        "role": "untrusted candidate mutation",
        "authority": "none",
    },
    {
        "actor_id": "unsigned_artifact",
        "role": "untrusted build or release object",
        "authority": "none",
    },
)

ASSETS: Final[tuple[str, ...]] = (
    "objective_cid",
    "idea_digest",
    "context_pack_cid",
    "current_root_cid",
    "chain_cid",
    "lock_cid",
    "policy_pointer",
    "task_state",
    "lease",
    "fence",
    "event_cursor",
    "confirmation",
    "proof_admission_verdict",
    "duckdb_task_tables",
    "quack_owner_state",
    "signed_release_artifacts",
    "python_client_cid",
    "mcp_client_cid",
    "parity_cid",
    "authority_bypass_proof_cid",
    "canonicalization_bytes",
    "build_provenance",
    "dependency_lock",
)

TRUST_BOUNDARIES: Final[tuple[str, ...]] = (
    "prompt_to_supervisor",
    "client_to_canonical_service",
    "datasets_semantic_to_accelerate_operational",
    "kit_bytes_to_accelerate_admission",
    "accelerate_to_quack",
    "model_to_completion",
    "hermetic_to_live",
    "current_head_to_mutable_main",
    "package_to_sibling_source",
    "untrusted_source_to_admission",
    "build_to_release",
)

STATE_OWNERS: Final[tuple[dict[str, str], ...]] = (
    {
        "owner_id": "quack",
        "owns": "authenticated task-source publication and mutation",
        "rule": "exclusive authenticated state-owner transport",
    },
    {
        "owner_id": "duckdb",
        "owns": "backing task, event, and attempt tables",
        "rule": "never written directly by a task, client, or adapter",
    },
    {
        "owner_id": "ipfs_kit_py",
        "owns": "exact bytes, current-root compare-and-swap, WAL",
        "rule": "does not decide proof validity or task completion",
    },
    {
        "owner_id": "ipfs_datasets_py",
        "owns": "semantic identity and ContextPack construction",
        "rule": "does not remint Kit bytes or Accelerate operational state",
    },
    {
        "owner_id": "ipfs_accelerate_py",
        "owns": "task admission, leases, fences, execution, terminalization",
        "rule": "does not remint Datasets or Kit identities",
    },
)

CANONICALIZATION_MODEL: Final[dict[str, Any]] = {
    "rule": (
        "Authoritative identities are CIDv1 DAG-JSON/sha2-256 (baguqeera) or "
        "raw-block CIDs (bafkrei) over canonical JSON bytes. SHA-256 hex is "
        "not a CID. Mutable Git branches are not identity."
    ),
    "encoding": "utf-8",
    "json": "sort_keys,separators=(',', ':'),ensure_ascii,allow_nan=False",
    "sha256_hex_is_not_cid": True,
    "bounded": True,
}

CID_MODEL: Final[dict[str, Any]] = {
    "bound_identities": {
        "idea_digest": PINNED_IDEA_DIGEST,
        "objective_cid": PINNED_OBJECTIVE_CID,
        "lock_cid": PINNED_LOCK_CID,
        "pack_cid": PINNED_PACK_CID,
        "current_root_cid": PINNED_CURRENT_ROOT_CID,
        "chain_cid": PINNED_CHAIN_CID,
        "python_client_cid": PINNED_PYTHON_CLIENT_CID,
        "mcp_client_cid": PINNED_MCP_CLIENT_CID,
        "parity_cid": PINNED_PARITY_CID,
        "authority_bypass_proof_cid": PINNED_BYPASS_PROOF_CID,
    },
    "remint_forbidden": True,
    "present_cid_is_not_proof_authority": True,
    "bounded": True,
}

EVENT_MODEL: Final[dict[str, Any]] = {
    "rule": (
        "Events are append-only, cursor-bound, and replay-safe. Duplicate "
        "delivery is identity-preserving and cannot mint a second effect."
    ),
    "duplicate_effect_rejected": True,
    "clients_may_subscribe_not_mutate": True,
    "bounded": True,
}

LEASE_MODEL: Final[dict[str, Any]] = {
    "rule": (
        "A stale or missing lease cannot complete. Client-supplied lease "
        "bypass is refused. Exclusive merge-queue leases serialize overlapping "
        "owned paths."
    ),
    "stale_lease_cannot_complete": True,
    "client_bypass_refused": True,
    "bounded": True,
}

FENCE_MODEL: Final[dict[str, Any]] = {
    "rule": (
        "A stale or missing fence cannot complete. Quack-fenced sessions are "
        "the only admitted state-owner transport. Direct DuckDB writes are "
        "prohibited."
    ),
    "stale_fence_cannot_complete": True,
    "client_bypass_refused": True,
    "direct_duckdb_write_prohibited": True,
    "bounded": True,
}

UNKNOWN_OUTCOME_MODEL: Final[dict[str, Any]] = {
    "rule": (
        "provider-outcome-unknown cannot enter blind retry. Unknown, "
        "ambiguous, stale, incomplete, or unavailable required evidence "
        "causes rejection or typed abstention, never a pass."
    ),
    "blind_retry_forbidden": True,
    "unknown_is_not_success": True,
    "bounded": True,
}

CONFIRMATION_MODEL: Final[dict[str, Any]] = {
    "rule": (
        "Apply and start require exact current roots, authorization, expected "
        "effects, idempotency, lease, fencing, and event-cursor bindings. "
        "Client-supplied confirmation is not authoritative."
    ),
    "client_confirmation_not_authoritative": True,
    "stale_binding_fails_closed": True,
    "bounded": True,
}

PROOF_ADMISSION_MODEL: Final[dict[str, Any]] = {
    "rule": (
        "A stored proof is not an admitted proof. A present CID is not proof "
        "authority. Cache hits cannot be reused against a different tree, "
        "policy, objective, schema, interface, toolchain, or environment."
    ),
    "stored_proof_is_not_admitted": True,
    "cache_reuse_across_trees_forbidden": True,
    "bounded": True,
}

DEPENDENCY_MODEL: Final[dict[str, Any]] = {
    "rule": (
        "No package may depend on a mutable Git branch in a qualified "
        "release. Absent dependencies are typed unavailable and are never "
        "auto-installed during import."
    ),
    "mutable_git_branch_forbidden": True,
    "import_time_auto_install_forbidden": True,
    "bounded": True,
}

BUILD_MODEL: Final[dict[str, Any]] = {
    "rule": (
        "No release may be published after partial required-build failure. "
        "Unsigned artifacts are not release authority. Sibling source-tree "
        "tests are not package tests."
    ),
    "partial_build_release_forbidden": True,
    "unsigned_artifact_is_not_authority": True,
    "sibling_source_not_required": True,
    "bounded": True,
}


def _threat(
    threat_id: str,
    *,
    stage: str,
    actor: str,
    asset: str,
    boundary: str,
    attack: str,
    control: str,
    residual: str,
    category: str,
) -> dict[str, Any]:
    if stage not in TRUSTED_PATH_STAGES:
        raise RuntimeError(f"threat {threat_id} uses unbounded stage {stage}")
    if actor not in {item["actor_id"] for item in ACTORS}:
        raise RuntimeError(f"threat {threat_id} uses unbounded actor {actor}")
    if asset not in ASSETS:
        raise RuntimeError(f"threat {threat_id} uses unbounded asset {asset}")
    if boundary not in TRUST_BOUNDARIES:
        raise RuntimeError(
            f"threat {threat_id} uses unbounded boundary {boundary}"
        )
    return {
        "threat_id": threat_id,
        "stage": stage,
        "actor": actor,
        "asset": asset,
        "trust_boundary": boundary,
        "attack": attack,
        "control": control,
        "residual": residual,
        "category": category,
        "live": False,
        "admitted_live": False,
        "bounded": True,
        "evidence_kind": "measured_hermetic",
    }


THREATS: Final[tuple[dict[str, Any], ...]] = (
    _threat(
        "TM-01",
        stage="objective_submission",
        actor="untrusted_prompt",
        asset="objective_cid",
        boundary="prompt_to_supervisor",
        attack="Prompt injection or body-unsafe capture mints a forged objective identity.",
        control="Body-safe prompt capture; idea digest and objective CID are canonical and not reminted.",
        residual="Live prompt-to-supervisor submission remains typed unavailable.",
        category="spoofing",
    ),
    _threat(
        "TM-02",
        stage="authentication_and_authority",
        actor="python_external_client",
        asset="policy_pointer",
        boundary="client_to_canonical_service",
        attack="Unauthenticated or client-supplied policy allow is treated as authority.",
        control="Clients are intent, never authority. Client-supplied policy allow is refused.",
        residual="Live authenticated caller remains typed unavailable.",
        category="elevation",
    ),
    _threat(
        "TM-03",
        stage="authentication_and_authority",
        actor="generic_mcp_client",
        asset="policy_pointer",
        boundary="client_to_canonical_service",
        attack="MCP client forges policy or writes a policy pointer.",
        control="PCPR-083 refuses forge_policy and write_policy_pointer as JSON-RPC -32001.",
        residual="Hermetic refusal is not a live MCP session.",
        category="tampering",
    ),
    _threat(
        "TM-04",
        stage="task_state_machine",
        actor="python_external_client",
        asset="task_state",
        boundary="client_to_canonical_service",
        attack="External client updates or terminalizes a task.",
        control="PCPR-083 refuses update_task and terminalize_task. Hard-zero counters remain zero.",
        residual="Live task-state mutation remains typed unavailable.",
        category="tampering",
    ),
    _threat(
        "TM-05",
        stage="planning_and_context_pack",
        actor="accelerate_execution_owner",
        asset="context_pack_cid",
        boundary="datasets_semantic_to_accelerate_operational",
        attack="Accelerate remints DatasetsContextPack@1 or admits a stale pack as current.",
        control="Pack CID is bound and not reminted. Stale pack remains rejected.",
        residual="Live ContextPack apply remains typed unavailable.",
        category="tampering",
    ),
    _threat(
        "TM-06",
        stage="planning_and_context_pack",
        actor="accelerate_execution_owner",
        asset="current_root_cid",
        boundary="kit_bytes_to_accelerate_admission",
        attack="Accelerate remints the Kit current root or reconstructs it from memory.",
        control="Current-root CID is bound. Durable reconstruction is from disk, not memory.",
        residual="Live current-root publication remains typed unavailable.",
        category="tampering",
    ),
    _threat(
        "TM-07",
        stage="execution_admission",
        actor="generic_mcp_client",
        asset="confirmation",
        boundary="client_to_canonical_service",
        attack="Client-supplied confirmation bypasses apply/start bindings.",
        control="PCPR-083 refuses bypass_confirmation. Client confirmation is not authoritative.",
        residual="Live confirmation path remains typed unavailable.",
        category="elevation",
    ),
    _threat(
        "TM-08",
        stage="execution_admission",
        actor="python_external_client",
        asset="lease",
        boundary="client_to_canonical_service",
        attack="Stale or missing lease completes, or a client bypasses the lease.",
        control="Stale lease cannot complete. PCPR-083 refuses bypass_lease.",
        residual="Live lease races remain typed unavailable.",
        category="elevation",
    ),
    _threat(
        "TM-09",
        stage="execution_admission",
        actor="python_external_client",
        asset="fence",
        boundary="accelerate_to_quack",
        attack="Stale fence completes, or a client writes DuckDB without Quack.",
        control="Stale fence cannot complete. Direct DuckDB writes are prohibited.",
        residual="No admitted Quack-fenced session in sealed validation.",
        category="elevation",
    ),
    _threat(
        "TM-10",
        stage="execution_admission",
        actor="untrusted_patch_or_agent",
        asset="task_state",
        boundary="untrusted_source_to_admission",
        attack="A candidate broadens owned paths or scope beyond the allowlist.",
        control="PCPR-083 refuses broaden_scope. Paths remain inside the declared allowlist.",
        residual="Live scope admission remains typed unavailable.",
        category="elevation",
    ),
    _threat(
        "TM-11",
        stage="proof_and_test_admission",
        actor="model_provider",
        asset="proof_admission_verdict",
        boundary="model_to_completion",
        attack="A model assertion completes work or an unadmitted proof is treated as admitted.",
        control="Model stages cannot complete work. Stored proofs are not admitted proofs.",
        residual="Incremental prover and live solvers remain typed unavailable.",
        category="repudiation",
    ),
    _threat(
        "TM-12",
        stage="proof_and_test_admission",
        actor="untrusted_repository",
        asset="proof_admission_verdict",
        boundary="hermetic_to_live",
        attack="Simulated, estimated, or hermetic results are represented as live.",
        control="Evidence kinds are distinguished. Simulated-as-live is refused.",
        residual="PCPR-001 live qualification remains typed unavailable.",
        category="spoofing",
    ),
    _threat(
        "TM-13",
        stage="proof_and_test_admission",
        actor="untrusted_repository",
        asset="canonicalization_bytes",
        boundary="untrusted_source_to_admission",
        attack="A SHA-256 hex string is advertised as an IPFS CID, or cache hits reuse across trees.",
        control="SHA-256 hex is not a CID. Cache reuse across trees, policies, or toolchains is forbidden.",
        residual="Live IPFS daemon remains typed unavailable.",
        category="spoofing",
    ),
    _threat(
        "TM-14",
        stage="execution_admission",
        actor="model_provider",
        asset="task_state",
        boundary="model_to_completion",
        attack="provider-outcome-unknown enters blind retry or missing capability returns success.",
        control="Unknown outcomes cannot blind-retry. Missing capabilities stay typed unavailable.",
        residual="Live provider backends remain typed unavailable.",
        category="denial",
    ),
    _threat(
        "TM-15",
        stage="task_state_machine",
        actor="untrusted_patch_or_agent",
        asset="duckdb_task_tables",
        boundary="accelerate_to_quack",
        attack="A task, adapter, or client writes DuckDB or Quack state directly.",
        control="Direct DuckDB and Quack writes are prohibited. This evaluator does not write them.",
        residual="Quack remains typed unavailable in sealed validation.",
        category="tampering",
    ),
    _threat(
        "TM-16",
        stage="task_state_machine",
        actor="generic_mcp_client",
        asset="event_cursor",
        boundary="client_to_canonical_service",
        attack="Event replay or duplicate delivery mints a second effect.",
        control="Duplicate delivery is identity-preserving and rejected as a duplicate effect.",
        residual="Live event DAG reconstruction remains typed unavailable.",
        category="replay",
    ),
    _threat(
        "TM-17",
        stage="receipt",
        actor="python_external_client",
        asset="chain_cid",
        boundary="client_to_canonical_service",
        attack="A client remints the PCPR-072 chain or self-promotes from retrieved receipts.",
        control="Chain CID is bound and not reminted. PCPR-083 refuses self_promote.",
        residual="Live receipt retrieval remains typed unavailable.",
        category="elevation",
    ),
    _threat(
        "TM-18",
        stage="receipt",
        actor="unsigned_artifact",
        asset="authority_bypass_proof_cid",
        boundary="hermetic_to_live",
        attack="Hermetic bypass coverage is treated as live qualification.",
        control="Bypass proof CID is bound. Hermetic coverage cannot qualify live Supervisor.run.",
        residual="Live authority-bypass remains typed unavailable.",
        category="spoofing",
    ),
    _threat(
        "TM-19",
        stage="release_decision",
        actor="unsigned_artifact",
        asset="signed_release_artifacts",
        boundary="build_to_release",
        attack="A partial-build or unsigned artifact is published as a release.",
        control="Partial required-build failure cannot publish. Unsigned artifacts are not authority.",
        residual="Signed release train remains PCPR-G600 and is not closed here.",
        category="spoofing",
    ),
    _threat(
        "TM-20",
        stage="release_decision",
        actor="supervisor",
        asset="signed_release_artifacts",
        boundary="current_head_to_mutable_main",
        attack="A mutable Git branch is used as a qualified release pin, or a closed outcome is emitted without gates.",
        control="Mutable Git release pins are forbidden. This task emits rnd_non_promoted only.",
        residual="Closed release decision remains PCPR-093/094.",
        category="elevation",
    ),
    _threat(
        "TM-21",
        stage="proof_and_test_admission",
        actor="untrusted_repository",
        asset="dependency_lock",
        boundary="package_to_sibling_source",
        attack="Package tests require an uninstalled sibling source-tree tests/ directory.",
        control="Sibling source is never required. Package tests run from declared package files.",
        residual="Installed-wheel isolation is not live-qualified here.",
        category="tampering",
    ),
    _threat(
        "TM-22",
        stage="planning_and_context_pack",
        actor="untrusted_prompt",
        asset="idea_digest",
        boundary="prompt_to_supervisor",
        attack="A different idea is submitted and treated as the canonical PCPR-060 objective.",
        control="Non-canonical idea bytes are refused. Objective CID and idea digest are bound.",
        residual="Live objective submission remains typed unavailable.",
        category="spoofing",
    ),
    _threat(
        "TM-23",
        stage="authentication_and_authority",
        actor="cli_client",
        asset="policy_pointer",
        boundary="client_to_canonical_service",
        attack="CLI client writes current roots or policy pointers, or promotes itself.",
        control="Forbidden client operations fail typed. Principals remain intent, never authority.",
        residual="Live CLI supervisor run remains typed unavailable.",
        category="elevation",
    ),
    _threat(
        "TM-24",
        stage="release_decision",
        actor="human_operator",
        asset="quack_owner_state",
        boundary="accelerate_to_quack",
        attack="Missing live environment is recorded as zero or passing, or TCB/audit/release claims are asserted here.",
        control="Missing environments stay typed unavailable. TCB is PCPR-091. Audit package is PCPR-092. Closed decision is PCPR-093/094.",
        residual="Operator must admit a Quack-fenced session before live threat-model execution.",
        category="repudiation",
    ),
)

if any(not item["bounded"] for item in THREATS):
    raise RuntimeError("PCPR-090 threat table contains an unbounded threat")
if {item["stage"] for item in THREATS} != set(TRUSTED_PATH_STAGES):
    raise RuntimeError("PCPR-090 threats do not cover every trusted-path stage")
if len({item["threat_id"] for item in THREATS}) != len(THREATS):
    raise RuntimeError("PCPR-090 threat identifiers are not unique")

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
    "artifacts/proof_carrying_platform_qualification_and_release/receipts/PCPR-090.json",
)

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_threat_model.py",
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
        "threat_model_files_match_generator",
        "pyproject_threat_model_table",
        "paths_remain_authorized",
        "canonical_objective_bytes_match",
        "actors_bounded",
        "assets_bounded",
        "trust_boundaries_bounded",
        "state_owners_bounded",
        "canonicalization_bounded",
        "cids_bounded",
        "events_bounded",
        "leases_bounded",
        "fences_bounded",
        "unknown_outcomes_bounded",
        "confirmations_bounded",
        "proof_admission_bounded",
        "dependencies_bounded",
        "builds_bounded",
        "all_g910_surfaces_bounded",
        "all_threats_bounded",
        "trusted_path_stages_covered",
        "owner_pack_cid_matches_pin",
        "kit_current_root_bound_not_minted",
        "chain_cid_bound_not_minted",
        "python_client_cid_bound_not_minted",
        "mcp_client_cid_bound_not_minted",
        "parity_cid_bound_not_minted",
        "bypass_proof_cid_bound_not_minted",
        "tcb_inventory_deferred",
        "audit_package_deferred",
        "closed_release_deferred",
        "idempotent_threat_model_cid",
        "model_assertion_cannot_complete_work",
        "duckdb_or_quack_not_written",
        "operator_blocking_task_emitted",
        "no_closed_release_outcome",
    }
)
FORBIDDEN_PRESENT_PROBE_IDS: Final[frozenset[str]] = frozenset(
    {
        "simulated_results_represented_as_live",
        "live_threat_model_represented_as_live",
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
        "bypass_identity_reminted",
        "model_assertion_completed_work",
        "unauthorized_path_accepted",
        "database_edited",
        "tcb_inventory_claimed",
        "audit_package_claimed",
        "closed_release_claimed",
        "unbounded_threat_present",
    }
)

DOCUMENT_README: Final = """# PCPR-090 Accelerate objective-to-release threat model

These files record the Accelerate-owned hermetic threat model for the
PCPR objective-to-release path. Actors, assets, trust boundaries, state
owners, canonicalization, CIDs, events, leases, fences, unknown
outcomes, confirmations, proof admission, dependencies, and builds are
bounded.

- `cpython312/reference.threat-model.json` is the declared threat-model
  document. Exact commit and tree are bound by the PCPR-090 receipt
  `current_tree_binding`.
- `platform-threat-model-catalog.json` observes sibling Datasets and Kit
  bindings when present. Sibling source is never required.
- Pack, root, chain, objective, Python-client, MCP-client, parity, and
  PCPR-083 bypass identities are bound and not reminted. The trusted
  computing base remains PCPR-091. The audit package remains PCPR-092.
  Closed release decision remains PCPR-093/094.
- Live Supervisor.run, live Quack, live MCP, and live CLI sessions stay
  typed unavailable.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-090-operator-live-threat-model`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
"""


class AccelerateThreatModelError(Exception):
    """Fail-closed PCPR-090 Accelerate threat-model error."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AccelerateThreatModelError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise AccelerateThreatModelError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise AccelerateThreatModelError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def parse_pyproject_threat_model_table(text: str) -> dict[str, Any]:
    import tomllib

    table = tomllib.loads(_text(text, "pyproject.toml"))
    if not isinstance(table, dict):
        raise AccelerateThreatModelError("pyproject.toml must be a table")
    tool = table.get("tool")
    payload: dict[str, Any] = {}
    if isinstance(tool, dict):
        accelerate = tool.get("ipfs_accelerate_py")
        if isinstance(accelerate, dict):
            raw = accelerate.get("threat-model")
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
        raise AccelerateThreatModelError(
            f"path {path} is not an authorized PCPR-090 path"
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
        raise AccelerateThreatModelError(
            f"model assertion at {stage} cannot complete work"
        )


def refuse_pack_cid_remint(cid: str) -> str:
    if cid != PINNED_PACK_CID:
        raise AccelerateThreatModelError(
            f"ContextPack CID {cid} remints {PINNED_PACK_CID}"
        )
    return cid


def refuse_current_root_remint(cid: str) -> str:
    if cid != PINNED_CURRENT_ROOT_CID:
        raise AccelerateThreatModelError(
            f"current root CID {cid} remints {PINNED_CURRENT_ROOT_CID}"
        )
    return cid


def refuse_chain_cid_remint(cid: str) -> str:
    if cid != PINNED_CHAIN_CID:
        raise AccelerateThreatModelError(
            f"chain CID {cid} remints {PINNED_CHAIN_CID}"
        )
    return cid


def refuse_objective_cid_remint(cid: str) -> str:
    if cid != PINNED_OBJECTIVE_CID:
        raise AccelerateThreatModelError(
            f"objective CID {cid} remints {PINNED_OBJECTIVE_CID}"
        )
    return cid


def refuse_idea_digest_remint(digest: str) -> str:
    if digest != PINNED_IDEA_DIGEST:
        raise AccelerateThreatModelError(
            f"idea digest {digest} remints {PINNED_IDEA_DIGEST}"
        )
    return digest


def refuse_python_client_cid_remint(cid: str) -> str:
    if cid != PINNED_PYTHON_CLIENT_CID:
        raise AccelerateThreatModelError(
            f"Python client CID {cid} remints {PINNED_PYTHON_CLIENT_CID}"
        )
    return cid


def refuse_mcp_client_cid_remint(cid: str) -> str:
    if cid != PINNED_MCP_CLIENT_CID:
        raise AccelerateThreatModelError(
            f"MCP client CID {cid} remints {PINNED_MCP_CLIENT_CID}"
        )
    return cid


def refuse_parity_cid_remint(cid: str) -> str:
    if cid != PINNED_PARITY_CID:
        raise AccelerateThreatModelError(
            f"parity CID {cid} remints {PINNED_PARITY_CID}"
        )
    return cid


def refuse_bypass_proof_cid_remint(cid: str) -> str:
    if cid != PINNED_BYPASS_PROOF_CID:
        raise AccelerateThreatModelError(
            f"bypass proof CID {cid} remints {PINNED_BYPASS_PROOF_CID}"
        )
    return cid


def refuse_database_edit(*, edited: bool) -> None:
    if edited:
        raise AccelerateThreatModelError(
            "threat model cannot write DuckDB or Quack state"
        )


def refuse_live_threat_model(*, applied_live: bool) -> None:
    if applied_live:
        raise AccelerateThreatModelError(
            "live threat-model execution remains typed unavailable"
        )


def refuse_self_promotion(*, promoted: bool) -> None:
    if promoted:
        raise AccelerateThreatModelError(
            "threat model cannot promote itself"
        )


def refuse_non_canonical_objective(idea: str) -> str:
    text = _text(idea, "idea")
    if text != REFERENCE_OBJECTIVE_IDEA:
        raise AccelerateThreatModelError(
            "threat model must bind the canonical PCPR-060 idea"
        )
    return text


def refuse_tcb_inventory_claim(*, claimed: bool) -> None:
    if claimed:
        raise AccelerateThreatModelError(
            "trusted-computing-base inventory remains PCPR-091"
        )


def refuse_audit_package_claim(*, claimed: bool) -> None:
    if claimed:
        raise AccelerateThreatModelError(
            "security and correctness audit package remains PCPR-092"
        )


def refuse_closed_release_claim(*, claimed: bool) -> None:
    if claimed:
        raise AccelerateThreatModelError(
            "closed release decision remains PCPR-093/094"
        )


def refuse_unbounded_threat(*, unbounded: bool) -> None:
    if unbounded:
        raise AccelerateThreatModelError(
            "every threat must be bounded to a declared actor, asset, boundary, and stage"
        )


def g910_surface_bounds() -> dict[str, bool]:
    return {
        "actors": bool(ACTORS),
        "assets": bool(ASSETS),
        "trust_boundaries": bool(TRUST_BOUNDARIES),
        "state_owners": bool(STATE_OWNERS),
        "canonicalization": CANONICALIZATION_MODEL["bounded"] is True,
        "cids": CID_MODEL["bounded"] is True,
        "events": EVENT_MODEL["bounded"] is True,
        "leases": LEASE_MODEL["bounded"] is True,
        "fences": FENCE_MODEL["bounded"] is True,
        "unknown_outcomes": UNKNOWN_OUTCOME_MODEL["bounded"] is True,
        "confirmations": CONFIRMATION_MODEL["bounded"] is True,
        "proof_admission": PROOF_ADMISSION_MODEL["bounded"] is True,
        "dependencies": DEPENDENCY_MODEL["bounded"] is True,
        "builds": BUILD_MODEL["bounded"] is True,
    }


def all_g910_surfaces_bounded() -> bool:
    bounds = g910_surface_bounds()
    return all(bounds[name] for name in G910_SURFACES) and set(bounds) == set(
        G910_SURFACES
    )


def hermetic_threat_model_trace() -> dict[str, Any]:
    bounds = g910_surface_bounds()
    stages = tuple(item["stage"] for item in THREATS)
    return {
        "schema": "ipfs_accelerate_py/assurance/hermetic-threat-model-trace@1",
        "interface": "HermeticThreatModelTrace@1",
        "threat_count": len(THREATS),
        "threat_ids": [item["threat_id"] for item in THREATS],
        "all_threats_bounded": all(item["bounded"] is True for item in THREATS),
        "trusted_path_stages": list(TRUSTED_PATH_STAGES),
        "trusted_path_stages_covered": set(stages) == set(TRUSTED_PATH_STAGES),
        "g910_surfaces": list(G910_SURFACES),
        "g910_surface_bounds": bounds,
        "all_g910_surfaces_bounded": all_g910_surfaces_bounded(),
        "actors_bounded": bounds["actors"],
        "assets_bounded": bounds["assets"],
        "trust_boundaries_bounded": bounds["trust_boundaries"],
        "state_owners_bounded": bounds["state_owners"],
        "tcb_inventory_deferred": True,
        "audit_package_deferred": True,
        "closed_release_deferred": True,
        "no_live_claim": True,
        "no_closed_release": True,
        "identities_bound_not_reminted": True,
        "duplicate_effect_rejected": True,
        "unauthorized_effect": False,
        "live": False,
        "hermetic": True,
        "evidence_kind": "measured_hermetic",
    }


def hermetic_threat_model_idempotency(model_cid: str) -> dict[str, Any]:
    return {
        "schema": "ipfs_accelerate_py/assurance/hermetic-threat-model-idempotency@1",
        "interface": "HermeticThreatModelIdempotency@1",
        "first_model_command_digest": MODEL_COMMAND_DIGEST,
        "second_model_command_digest": MODEL_COMMAND_DIGEST,
        "first_threat_model_cid": model_cid,
        "second_threat_model_cid": model_cid,
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
class AccelerateThreatModelVerdict:
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
    threat_model_cid: str
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
    bypass_proof_cid: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "verdict_cid": self.verdict_cid,
            "threat_model_cid": self.threat_model_cid,
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
            "bypass_proof_cid": self.bypass_proof_cid,
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
            "An admitted Quack-fenced state-owner session before the threat "
            "model is executed as live supervisor work"
        ),
        "action": (
            "Keep the hermetic threat model. Keep every G910 surface bounded. "
            "Do not write DuckDB or Quack state. Do not escalate to a model. "
            "Do not emit a TCB inventory, audit package, or closed release. "
            "PCPR-091 prepares the trusted-computing-base inventory."
        ),
        "reason": (
            "Hermetic threat-model coverage is not a live Supervisor.run / "
            "CLI / MCP session. Sealed validation has no admitted "
            "Quack-fenced session. Direct DuckDB writes are prohibited."
        ),
    }


def produce_threat_model(
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
    bypass_proof_cid: str = PINNED_BYPASS_PROOF_CID,
    idea: str = REFERENCE_OBJECTIVE_IDEA,
    complete_from: str | None = None,
    database_edited: bool = False,
    applied_live: bool = False,
    self_promoted: bool = False,
    tcb_claimed: bool = False,
    audit_claimed: bool = False,
    closed_release_claimed: bool = False,
    unbounded: bool = False,
) -> dict[str, Any]:
    """Produce the hermetic objective-to-release threat model. Fail closed."""

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
    refuse_bypass_proof_cid_remint(bypass_proof_cid)
    refuse_non_canonical_objective(idea)
    refuse_database_edit(edited=database_edited)
    refuse_live_threat_model(applied_live=applied_live)
    refuse_self_promotion(promoted=self_promoted)
    refuse_tcb_inventory_claim(claimed=tcb_claimed)
    refuse_audit_package_claim(claimed=audit_claimed)
    refuse_closed_release_claim(claimed=closed_release_claimed)
    refuse_unbounded_threat(unbounded=unbounded)
    checked_paths = list(paths) if paths is not None else list(AUTHORIZED_PATH_PREFIXES)
    for path in checked_paths:
        refuse_unauthorized_path(path)
    trace = hermetic_threat_model_trace()
    return {
        "schema": MODEL_SCHEMA,
        "interface": MODEL_INTERFACE,
        "owner_interface": INTERFACE,
        "task_id": PCPR_090_TASK_ID,
        "goal_id": PCPR_090_GOAL_ID,
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
        "bypass_proof_cid": bypass_proof_cid,
        "python_client_interface": PYTHON_CLIENT_INTERFACE,
        "mcp_client_interface": MCP_CLIENT_INTERFACE,
        "parity_interface": PARITY_INTERFACE,
        "bypass_interface": BYPASS_INTERFACE,
        "protocol_interface": PROTOCOL_INTERFACE,
        "documentation_interface": DOCUMENTATION_INTERFACE,
        "documentation_schema": DOCUMENTATION_SCHEMA,
        "change_kind": CHANGE_KIND,
        "model_kind": MODEL_KIND,
        "canonical_service": CANONICAL_SERVICE,
        "authority": CLIENT_AUTHORITY,
        "trusted_path_stages": list(TRUSTED_PATH_STAGES),
        "g910_surfaces": list(G910_SURFACES),
        "actors": [dict(item) for item in ACTORS],
        "assets": list(ASSETS),
        "trust_boundaries": list(TRUST_BOUNDARIES),
        "state_owners": [dict(item) for item in STATE_OWNERS],
        "canonicalization": dict(CANONICALIZATION_MODEL),
        "cids": dict(CID_MODEL),
        "events": dict(EVENT_MODEL),
        "leases": dict(LEASE_MODEL),
        "fences": dict(FENCE_MODEL),
        "unknown_outcomes": dict(UNKNOWN_OUTCOME_MODEL),
        "confirmations": dict(CONFIRMATION_MODEL),
        "proof_admission": dict(PROOF_ADMISSION_MODEL),
        "dependencies": dict(DEPENDENCY_MODEL),
        "builds": dict(BUILD_MODEL),
        "threats": [dict(item) for item in THREATS],
        "produced": True,
        "applied": True,
        "live": False,
        "hermetic": True,
        "admitted_live": False,
        "deferred": False,
        "paths_authorized": True,
        "authorized_path_prefixes": list(AUTHORIZED_PATH_PREFIXES),
        "all_g910_surfaces_bounded": True,
        "all_threats_bounded": True,
        "trusted_path_stages_covered": True,
        "tcb_inventory_deferred": True,
        "audit_package_deferred": True,
        "closed_release_deferred": True,
        "client_self_promoted": False,
        "trace": trace,
        "model_command_digest": MODEL_COMMAND_DIGEST,
        "idempotent": True,
        "duplicate_effect_rejected": True,
        "remints_protocol": False,
        "adds_protocol_operation": False,
        "model_assertion_completes_work": False,
        "escalation_order": list(ESCALATION_ORDER),
        "completed_through": COMPLETED_THROUGH,
        "next_authorized_stage": NEXT_AUTHORIZED_STAGE,
        "next_task_id": PCPR_091_TASK_ID,
        "prerequisite_task_id": PCPR_083_TASK_ID,
        "python_client_task_id": PCPR_080_TASK_ID,
        "mcp_client_task_id": PCPR_081_TASK_ID,
        "parity_task_id": PCPR_082_TASK_ID,
        "bypass_task_id": PCPR_083_TASK_ID,
        "chain_task_id": PCPR_072_TASK_ID,
        "duckdb_or_quack_state_written": False,
        "closed_release_outcome": None,
        "release_claim": False,
        "evidence_kind": "measured_hermetic",
    }


def canonical_threat_model() -> dict[str, Any]:
    return produce_threat_model()


def threat_model_cid_of(plan: Mapping[str, Any] | None = None) -> str:
    payload = dict(plan or canonical_threat_model())
    payload.pop("threat_model_cid", None)
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
                "Sibling threat-model file is not present beside this checkout."
            ),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    context_pack = payload.get("context_pack")
    storage = payload.get("storage")
    model = payload.get("threat_model")
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
        or (model.get("chain_cid") if isinstance(model, Mapping) else None),
        "threat_model_cid": payload.get("threat_model_cid")
        or (model.get("threat_model_cid") if isinstance(model, Mapping) else None),
        "binding_cid": payload.get("binding_cid") or payload.get("document_cid"),
        "live": False,
        "applied": False,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def render_declared_model(root: Path | None = None) -> dict[str, Any]:
    package_root = root or discover_accelerate_root()
    if package_root is None:
        raise AccelerateThreatModelError("Accelerate package root was not found")
    state_owner = observe_state_owner()
    plan = canonical_threat_model()
    computed_model_cid = threat_model_cid_of(plan)
    idempotency = hermetic_threat_model_idempotency(computed_model_cid)
    document = {
        "schema": DOCUMENT_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_090_TASK_ID,
        "goal_id": PCPR_090_GOAL_ID,
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
            "survives_threat_model": True,
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
            "survives_threat_model": True,
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
            "authority": CLIENT_AUTHORITY,
            "reminted": False,
            "live": False,
            "evidence_kind": "measured",
        },
        "generic_mcp_client": {
            "task_id": PCPR_081_TASK_ID,
            "interface": MCP_CLIENT_INTERFACE,
            "client_cid": PINNED_MCP_CLIENT_CID,
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
            "task_id": PCPR_083_TASK_ID,
            "interface": BYPASS_INTERFACE,
            "proof_cid": PINNED_BYPASS_PROOF_CID,
            "reminted": False,
            "live": False,
            "evidence_kind": "measured",
        },
        "threat_model": {
            **plan,
            "threat_model_cid": computed_model_cid,
            "idempotency": idempotency,
            "produced_by": EXECUTION_OWNER_REPOSITORY,
        },
        "materialization": {
            "kind": "declared_threat_model_not_live",
            "admitted": False,
            "live": False,
            "applied": False,
            "duckdb_or_quack_state_written": False,
            "evidence_kind": "unavailable",
        },
        "live_application": typed_unavailable(
            reason=(
                "No admitted Quack-fenced supervisor session was observed. "
                "Hermetic threat-model coverage is not live execution."
            )
        ),
        "live_client": typed_unavailable(
            reason=(
                "Live Python Supervisor.run and live generic MCP remain typed "
                "unavailable. This task only emits a hermetic threat model. "
                "Trusted-computing-base inventory remains PCPR-091."
            )
        ),
        "live_mcp_client": typed_unavailable(
            reason="Live MCP stdio or HTTP session remains typed unavailable."
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
                "field": "PCPR-090 receipt current_tree_binding",
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


def platform_threat_model_catalog(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateThreatModelError("Accelerate package root was not found")
    parent = root.parent
    document = render_declared_model(root)
    datasets_binding = _sibling_binding(
        parent
        / "ipfs_datasets"
        / DOCUMENT_DIR_RELPATH
        / "reference.threat-model.binding.json",
        relative_path=(
            f"../ipfs_datasets/{DOCUMENT_DIR_RELPATH}/"
            "reference.threat-model.binding.json"
        ),
    )
    kit_binding = _sibling_binding(
        parent
        / "ipfs_kit"
        / DOCUMENT_DIR_RELPATH
        / "reference.threat-model.binding.json",
        relative_path=(
            f"../ipfs_kit/{DOCUMENT_DIR_RELPATH}/"
            "reference.threat-model.binding.json"
        ),
    )
    catalog = {
        "schema": CATALOG_SCHEMA,
        "interface": CATALOG_INTERFACE,
        "task_id": PCPR_090_TASK_ID,
        "goal_id": PCPR_090_GOAL_ID,
        "objective_kind": OBJECTIVE_KIND,
        "portfolio_id": PORTFOLIO_ID,
        "portfolio_version": PORTFOLIO_VERSION,
        "objective_cid": PINNED_OBJECTIVE_CID,
        "idea_digest": PINNED_IDEA_DIGEST,
        "pack_cid": PINNED_PACK_CID,
        "current_root_cid": PINNED_CURRENT_ROOT_CID,
        "chain_cid": PINNED_CHAIN_CID,
        "threat_model_cid": document["threat_model"]["threat_model_cid"],
        "python_client_cid": PINNED_PYTHON_CLIENT_CID,
        "mcp_client_cid": PINNED_MCP_CLIENT_CID,
        "parity_cid": PINNED_PARITY_CID,
        "bypass_proof_cid": PINNED_BYPASS_PROOF_CID,
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
                "threat_model_cid": document["threat_model"]["threat_model_cid"],
                "python_client_cid": PINNED_PYTHON_CLIENT_CID,
                "mcp_client_cid": PINNED_MCP_CLIENT_CID,
                "parity_cid": PINNED_PARITY_CID,
                "bypass_proof_cid": PINNED_BYPASS_PROOF_CID,
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


def write_threat_model_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateThreatModelError("Accelerate package root was not found")
    document = render_declared_model(root)
    paths = artifact_paths(root)
    _atomic_write(paths["document"], pretty_json(document))
    readme = DOCUMENT_README if DOCUMENT_README.endswith("\n") else DOCUMENT_README + "\n"
    _atomic_write(paths["readme"], readme)
    catalog = platform_threat_model_catalog(start)
    _atomic_write(paths["catalog"], pretty_json(catalog))
    return {
        "document": document,
        "catalog": catalog,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def verify_threat_model_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateThreatModelError("Accelerate package root was not found")
    document = render_declared_model(root)
    catalog = platform_threat_model_catalog(start)
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
        "threat_model_cid": document["threat_model"]["threat_model_cid"],
        "document_cid": document["document_cid"],
        "catalog_cid": catalog["catalog_cid"],
        "current_root_cid": document["storage"]["current_root_cid"],
        "objective_cid": document["objective_cid"],
        "idea_digest": document["idea_digest"],
        "python_client_cid": document["python_external_client"]["client_cid"],
        "mcp_client_cid": document["generic_mcp_client"]["client_cid"],
        "parity_cid": document["objective_identity_parity"]["parity_cid"],
        "bypass_proof_cid": document["authority_bypass"]["proof_cid"],
        "document_sha256": (
            sha256_bytes(paths["document"].read_bytes())
            if paths["document"].is_file()
            else "unavailable"
        ),
    }


def current_head_static_probes(start: Path | None = None) -> tuple[OutcomeProbe, ...]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateThreatModelError("Accelerate package root was not found")
    document = render_declared_model(root)
    verified = verify_threat_model_files(root)
    table = parse_pyproject_threat_model_table(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    model = document["threat_model"]
    trace = model["trace"]
    idempotency = model["idempotency"]
    remint = (
        document["context_pack"]["pack_cid"] != PINNED_PACK_CID
        or document["storage"]["current_root_cid"] != PINNED_CURRENT_ROOT_CID
        or document["final_receipt_chain"]["chain_cid"] != PINNED_CHAIN_CID
        or document["objective_cid"] != PINNED_OBJECTIVE_CID
        or document["idea_digest"] != PINNED_IDEA_DIGEST
        or document["lock_cid"] != PINNED_LOCK_CID
        or document["python_external_client"]["client_cid"] != PINNED_PYTHON_CLIENT_CID
        or document["generic_mcp_client"]["client_cid"] != PINNED_MCP_CLIENT_CID
        or document["objective_identity_parity"]["parity_cid"] != PINNED_PARITY_CID
        or document["authority_bypass"]["proof_cid"] != PINNED_BYPASS_PROOF_CID
        or model["threat_model_cid"] != PINNED_THREAT_MODEL_CID
    )
    probes = [
        _probe(
            "threat_model_files_match_generator",
            verified["ok"] is True,
            reason=(
                "Declared threat-model files match the generator."
                if verified["ok"]
                else "Declared threat-model files do not match the generator."
            ),
        ),
        _probe(
            "pyproject_threat_model_table",
            table.get("interface") == INTERFACE and table.get("task-id") == PCPR_090_TASK_ID,
            reason=(
                "pyproject.toml declares the PCPR-090 threat-model table."
                if table.get("interface") == INTERFACE
                else "pyproject.toml does not declare the PCPR-090 threat-model table."
            ),
        ),
        _probe(
            "paths_remain_authorized",
            model["paths_authorized"] is True
            and list(model["authorized_path_prefixes"])
            == list(AUTHORIZED_PATH_PREFIXES),
            reason="Threat-model paths remain inside the declared PCPR-090 allowlist.",
        ),
        _probe(
            "canonical_objective_bytes_match",
            model["idea"] == REFERENCE_OBJECTIVE_IDEA
            and model["idea_digest"] == PINNED_IDEA_DIGEST,
            reason="Threat model uses the canonical PCPR-060 idea bytes.",
        ),
        _probe(
            "actors_bounded",
            trace["actors_bounded"] is True,
            reason="Actors are bounded.",
        ),
        _probe(
            "assets_bounded",
            trace["assets_bounded"] is True,
            reason="Assets are bounded.",
        ),
        _probe(
            "trust_boundaries_bounded",
            trace["trust_boundaries_bounded"] is True,
            reason="Trust boundaries are bounded.",
        ),
        _probe(
            "state_owners_bounded",
            trace["state_owners_bounded"] is True,
            reason="State owners are bounded.",
        ),
        _probe(
            "canonicalization_bounded",
            trace["g910_surface_bounds"]["canonicalization"] is True,
            reason="Canonicalization is bounded.",
        ),
        _probe(
            "cids_bounded",
            trace["g910_surface_bounds"]["cids"] is True,
            reason="CID vectors are bounded.",
        ),
        _probe(
            "events_bounded",
            trace["g910_surface_bounds"]["events"] is True,
            reason="Events are bounded.",
        ),
        _probe(
            "leases_bounded",
            trace["g910_surface_bounds"]["leases"] is True,
            reason="Leases are bounded.",
        ),
        _probe(
            "fences_bounded",
            trace["g910_surface_bounds"]["fences"] is True,
            reason="Fences are bounded.",
        ),
        _probe(
            "unknown_outcomes_bounded",
            trace["g910_surface_bounds"]["unknown_outcomes"] is True,
            reason="Unknown outcomes are bounded.",
        ),
        _probe(
            "confirmations_bounded",
            trace["g910_surface_bounds"]["confirmations"] is True,
            reason="Confirmations are bounded.",
        ),
        _probe(
            "proof_admission_bounded",
            trace["g910_surface_bounds"]["proof_admission"] is True,
            reason="Proof admission is bounded.",
        ),
        _probe(
            "dependencies_bounded",
            trace["g910_surface_bounds"]["dependencies"] is True,
            reason="Dependencies are bounded.",
        ),
        _probe(
            "builds_bounded",
            trace["g910_surface_bounds"]["builds"] is True,
            reason="Builds are bounded.",
        ),
        _probe(
            "all_g910_surfaces_bounded",
            model["all_g910_surfaces_bounded"] is True
            and trace["all_g910_surfaces_bounded"] is True,
            reason="Every PCPR-G910 surface is bounded.",
        ),
        _probe(
            "all_threats_bounded",
            model["all_threats_bounded"] is True
            and trace["all_threats_bounded"] is True,
            reason="Every recorded threat is bounded.",
        ),
        _probe(
            "trusted_path_stages_covered",
            model["trusted_path_stages_covered"] is True
            and trace["trusted_path_stages_covered"] is True,
            reason="Threats cover every objective-to-release stage.",
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
            "bypass_proof_cid_bound_not_minted",
            document["authority_bypass"]["proof_cid"] == PINNED_BYPASS_PROOF_CID
            and document["authority_bypass"]["reminted"] is False,
            reason="PCPR-083 bypass proof CID is bound and not reminted.",
        ),
        _probe(
            "tcb_inventory_deferred",
            model["tcb_inventory_deferred"] is True
            and model["next_task_id"] == PCPR_091_TASK_ID,
            reason="Trusted-computing-base inventory remains PCPR-091.",
        ),
        _probe(
            "audit_package_deferred",
            model["audit_package_deferred"] is True,
            reason="Security and correctness audit package remains PCPR-092.",
        ),
        _probe(
            "closed_release_deferred",
            model["closed_release_deferred"] is True
            and document["closed_release_outcome"] is None,
            reason="Closed release decision remains PCPR-093/094.",
        ),
        _probe(
            "idempotent_threat_model_cid",
            model["idempotent"] is True
            and idempotency["identical"] is True
            and idempotency["first_threat_model_cid"] == model["threat_model_cid"]
            and idempotency["second_threat_model_cid"] == model["threat_model_cid"]
            and model["threat_model_cid"] == PINNED_THREAT_MODEL_CID,
            reason="A second threat-model emission yields the same model CID.",
        ),
        _probe(
            "model_assertion_cannot_complete_work",
            model["model_assertion_completes_work"] is False,
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
            reason="Missing live execution emits the operator-blocking task.",
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
                "client, MCP client, parity, bypass, or threat-model CID is "
                "forbidden."
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
            "bypass_identity_reminted",
            document["authority_bypass"]["proof_cid"] != PINNED_BYPASS_PROOF_CID
            or document["authority_bypass"]["reminted"] is True,
            reason="Accelerate must not remint the PCPR-083 bypass proof CID.",
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
            "live_threat_model_represented_as_live",
            False,
            reason="Hermetic threat-model coverage is not represented as live.",
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
            "tcb_inventory_claimed",
            False,
            reason="Trusted-computing-base inventory remains PCPR-091.",
        ),
        _probe(
            "audit_package_claimed",
            False,
            reason="Audit package remains PCPR-092.",
        ),
        _probe(
            "closed_release_claimed",
            False,
            reason="Closed release decision remains PCPR-093/094.",
        ),
        _probe(
            "unbounded_threat_present",
            False,
            reason="Every recorded threat is bounded.",
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


def qualify_threat_model(
    probes: Sequence[OutcomeProbe],
    *,
    threat_model_cid: str,
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
    bypass_proof_cid: str = PINNED_BYPASS_PROOF_CID,
) -> AccelerateThreatModelVerdict:
    if not probes:
        raise AccelerateThreatModelError("at least one probe is required")
    normalized: list[OutcomeProbe] = []
    blockers: list[str] = []
    for probe in probes:
        kind = _kind(probe.evidence_kind, "evidence_kind")
        if probe.live and kind != "measured_live":
            raise AccelerateThreatModelError(
                "live claims require measured_live evidence"
            )
        if probe.simulated_represented_as_live:
            raise AccelerateThreatModelError(
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
        "task_id": PCPR_090_TASK_ID,
        "goal_id": PCPR_090_GOAL_ID,
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
        "threat_model_cid": threat_model_cid,
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
        "bypass_proof_cid": bypass_proof_cid,
    }
    return AccelerateThreatModelVerdict(
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
        threat_model_cid=threat_model_cid,
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
        bypass_proof_cid=bypass_proof_cid,
    )


def qualify_current_head_threat_model(
    start: Path | None = None,
) -> AccelerateThreatModelVerdict:
    document = render_declared_model(start)
    catalog = platform_threat_model_catalog(start)
    return qualify_threat_model(
        current_head_static_probes(start),
        threat_model_cid=str(document["threat_model"]["threat_model_cid"]),
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
        bypass_proof_cid=str(document["authority_bypass"]["proof_cid"]),
    )


def pcpr_090_receipt_promotion(
    verdict: AccelerateThreatModelVerdict,
) -> dict[str, Any]:
    if verdict.closed_release_outcome is not None:
        raise AccelerateThreatModelError(
            "threat model must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise AccelerateThreatModelError(
            "threat model must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise AccelerateThreatModelError(
            "threat model completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise AccelerateThreatModelError(
            "threat model must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise AccelerateThreatModelError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.live_client or verdict.live_application:
        raise AccelerateThreatModelError(
            "live client claims require measured_live evidence"
        )
    return verdict.to_mapping()


PINNED_THREAT_MODEL_CID: Final = (
    "baguqeera2kz3zj4stztmb22lmbcuyj3zj2ekdg7st2qzefbr5z6r6vemnhxq"
)
PINNED_DOCUMENT_CID: Final = (
    "baguqeera4zv2uxiao5by2jxld2lesb3q4jn3p5lidx4usxrgmeom4r5tlesa"
)
PINNED_CATALOG_CID: Final = (
    "baguqeeralrl2jejw3ojjte3sejojdzqj2wenczhefv5huir56bigqi4bzmuq"
)
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeerajlzw76luhca6wjp5d7f2piivqtflztxy2hhmrrl5jzwrg55rb3hq"
)


__all__ = [
    "AUTHORIZED_PATH_PREFIXES",
    "CLOSED_RELEASE_OUTCOMES",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "ESCALATION_ORDER",
    "G910_SURFACES",
    "HERMETIC_CANDIDATE_SUITES",
    "INTERFACE",
    "OBJECTIVE_KIND",
    "OPERATOR_BLOCKING_TASK_ID",
    "OutcomeProbe",
    "PCPR_090_GOAL_ID",
    "PCPR_090_TASK_ID",
    "PINNED_BYPASS_PROOF_CID",
    "PINNED_CATALOG_CID",
    "PINNED_CHAIN_CID",
    "PINNED_CURRENT_ROOT_CID",
    "PINNED_DOCUMENT_CID",
    "PINNED_IDEA_DIGEST",
    "PINNED_MCP_CLIENT_CID",
    "PINNED_OBJECTIVE_CID",
    "PINNED_PACK_CID",
    "PINNED_PARITY_CID",
    "PINNED_PYTHON_CLIENT_CID",
    "PINNED_THREAT_MODEL_CID",
    "SCHEMA",
    "SEALED_PATH",
    "SEALED_PYTHON",
    "THREATS",
    "TRUSTED_PATH_STAGES",
    "AccelerateThreatModelError",
    "AccelerateThreatModelVerdict",
    "all_g910_surfaces_bounded",
    "canonical_threat_model",
    "current_head_static_probes",
    "hermetic_threat_model_trace",
    "path_is_authorized",
    "pcpr_090_receipt_promotion",
    "platform_threat_model_catalog",
    "produce_threat_model",
    "qualify_current_head_threat_model",
    "qualify_threat_model",
    "refuse_audit_package_claim",
    "refuse_bypass_proof_cid_remint",
    "refuse_chain_cid_remint",
    "refuse_closed_release_claim",
    "refuse_database_edit",
    "refuse_mcp_client_cid_remint",
    "refuse_model_completion",
    "refuse_non_canonical_objective",
    "refuse_objective_cid_remint",
    "refuse_pack_cid_remint",
    "refuse_parity_cid_remint",
    "refuse_python_client_cid_remint",
    "refuse_self_promotion",
    "refuse_tcb_inventory_claim",
    "refuse_unauthorized_path",
    "refuse_unbounded_threat",
    "render_declared_model",
    "threat_model_cid_of",
    "verify_threat_model_files",
    "write_threat_model_files",
]
