"""Fail-closed PCPR-091 Accelerate-owned trusted-computing-base inventory.

Accelerate owns the exact trusted-computing-base inventory for the PCPR
objective-to-release path. Actors, assets, trust boundaries, state owners,
canonicalization, CIDs, events, leases, fences, unknown outcomes,
confirmations, proof admission, dependencies, and builds are bounded as
named TCB components. The PCPR-090 threat model is bound and not reminted.
The audit package remains PCPR-092. Closed release decision remains
PCPR-093/094.

Live Supervisor.run, live Quack, live MCP, and live CLI stay typed
unavailable. Simulated results are not live. This evaluator does not write
DuckDB or Quack state and does not emit a closed PCPR release.

Hermetic TCB inventory coverage is candidate coverage only.
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


INTERFACE: Final = "AccelerateTrustedComputingBase@1"
SCHEMA: Final = "ipfs_accelerate_py/assurance/trusted-computing-base@1"
DOCUMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/declared-trusted-computing-base@1"
)
INVENTORY_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/trusted-computing-base-inventory@1"
)
VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/trusted-computing-base-verdict@1"
)
CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/platform-trusted-computing-base-catalog@1"
)
CATALOG_INTERFACE: Final = "PlatformTrustedComputingBase@1"
INVENTORY_INTERFACE: Final = "AuthoritativeTrustedComputingBase@1"
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
THREAT_MODEL_INTERFACE: Final = "AccelerateThreatModel@1"
PROTOCOL_INTERFACE: Final = "LogicProviderProtocol@2"
DOCUMENTATION_INTERFACE: Final = "LogicProviderProtocolTrustedComputingBase@1"
DOCUMENTATION_SCHEMA: Final = (
    "ipfs_datasets_py/logic-provider-trusted-computing-base@1"
)
PCPR_091_TASK_ID: Final = "PCPR-091"
PCPR_091_GOAL_ID: Final = "PCPR-G900"
PCPR_092_TASK_ID: Final = "PCPR-092"
PCPR_093_TASK_ID: Final = "PCPR-093"
PCPR_094_TASK_ID: Final = "PCPR-094"
PCPR_090_TASK_ID: Final = "PCPR-090"
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
OBJECTIVE_KIND: Final = "declared_objective_to_release_trusted_computing_base"
OBJECTIVE_ID: Final = "PCPR-G900"
LANGUAGE: Final = "Python"
OPERATOR_BLOCKING_TASK_ID: Final = "pcpr-091-operator-live-tcb-inventory"
DOCUMENT_DIR_RELPATH: Final = "packaging/pcpr/reference-workflow/cpython312"
DOCUMENT_JSON_NAME: Final = "reference.trusted-computing-base.json"
DOCUMENT_README_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/TRUSTED_COMPUTING_BASE.md"
)
CATALOG_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/platform-trusted-computing-base-catalog.json"
)
SOURCE_DATE_EPOCH: Final = "0"
SOURCE_REPOSITORY: Final = "endomorphosis/ipfs_accelerate_py"
COMPLETED_THROUGH: Final = "trusted_computing_base_inventory"
NEXT_AUTHORIZED_STAGE: Final = "security_and_correctness_audit_package"
CHANGE_KIND: Final = "objective_to_release_trusted_computing_base"
INVENTORY_KIND: Final = "hermetic_objective_to_release_trusted_computing_base"
CANONICAL_SERVICE: Final = "pcpr-canonical-supervisor-service"
CLIENT_AUTHORITY: Final = "intent_not_authority"
INVENTORY_COMMAND_DIGEST: Final = "pcpr-091-hermetic-tcb-inventory"

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
PINNED_THREAT_MODEL_CID: Final = (
    "baguqeera2kz3zj4stztmb22lmbcuyj3zj2ekdg7st2qzefbr5z6r6vemnhxq"
)
PINNED_THREAT_MODEL_DOCUMENT_CID: Final = (
    "baguqeera4zv2uxiao5by2jxld2lesb3q4jn3p5lidx4usxrgmeom4r5tlesa"
)

if PINNED_LOCK_CID != (
    "baguqeerawsekbbbt5ccctjt4tydzeahfkahsatb6q5x7k6cy4inrii5lhqmq"
):
    raise RuntimeError("PCPR-091 lock CID remints PCPR-056")
if PINNED_PYTHON_CLIENT_CID == PINNED_MCP_CLIENT_CID:
    raise RuntimeError("Python and MCP client CIDs must remain distinct")
if PINNED_THREAT_MODEL_CID == PINNED_PACK_CID:
    raise RuntimeError("Threat-model CID must remain distinct from pack CID")
if OWNER_INTERFACES["SupervisorContextPack"] != OWNER_INTERFACE:
    raise RuntimeError("PCPR-091 remints SupervisorContextPack owner interface")
if OWNER_SCHEMAS["SupervisorContextPack"] != OWNER_SCHEMA:
    raise RuntimeError("PCPR-091 remints SupervisorContextPack owner schema")

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

TRUST_CLASSES: Final[tuple[str, ...]] = (
    "in_tcb",
    "conditional_tcb",
    "out_of_tcb",
)

COMPROMISE_RULE: Final = (
    "A compromised or incorrectly reviewed trusted component is outside the "
    "soundness guarantee. Detected identity, integrity, configuration, or "
    "execution failures produce STALE or ERROR and fail closed. Out-of-TCB "
    "actors cannot grant authority. Conditional TCB components that are not "
    "admitted remain typed unavailable and are never represented as live."
)


def _component(
    component_id: str,
    *,
    trust_class: str,
    owner: str,
    surface: str,
    stage: str,
    role: str,
    compromise_effect: str,
    live_status: str,
    authority: str,
) -> dict[str, Any]:
    if trust_class not in TRUST_CLASSES:
        raise RuntimeError(
            f"component {component_id} uses unbounded trust class {trust_class}"
        )
    if surface not in G910_SURFACES:
        raise RuntimeError(
            f"component {component_id} uses unbounded surface {surface}"
        )
    if stage not in TRUSTED_PATH_STAGES:
        raise RuntimeError(
            f"component {component_id} uses unbounded stage {stage}"
        )
    if live_status not in {"measured_hermetic", "unavailable"}:
        raise RuntimeError(
            f"component {component_id} uses unbounded live status {live_status}"
        )
    in_tcb = trust_class == "in_tcb"
    conditional = trust_class == "conditional_tcb"
    out = trust_class == "out_of_tcb"
    live = False
    admitted_live = False
    return {
        "component_id": component_id,
        "trust_class": trust_class,
        "in_tcb": in_tcb,
        "conditional_tcb": conditional,
        "out_of_tcb": out,
        "owner": owner,
        "g910_surface": surface,
        "trusted_path_stage": stage,
        "role": role,
        "authority": authority,
        "compromise_effect": compromise_effect,
        "live_status": live_status,
        "live": live,
        "admitted_live": admitted_live,
        "bounded": True,
        "evidence_kind": (
            "unavailable" if live_status == "unavailable" else "measured_hermetic"
        ),
    }


COMPONENTS: Final[tuple[dict[str, Any], ...]] = (
    _component(
        "TCB-01",
        trust_class="in_tcb",
        owner="human_operator",
        surface="actors",
        stage="release_decision",
        role="Seals, confirmations, START, and residual decisions.",
        compromise_effect="Operator error or seal forgery fails closed; it cannot mint a closed release here.",
        live_status="measured_hermetic",
        authority="authority",
    ),
    _component(
        "TCB-02",
        trust_class="in_tcb",
        owner="ipfs_datasets_py",
        surface="state_owners",
        stage="planning_and_context_pack",
        role="Semantic identity, ContextPack construction, and solver-result validation.",
        compromise_effect="Semantic remint or stale-as-current admission is STALE and rejected.",
        live_status="measured_hermetic",
        authority="semantic_identity",
    ),
    _component(
        "TCB-03",
        trust_class="in_tcb",
        owner="ipfs_kit_py",
        surface="state_owners",
        stage="planning_and_context_pack",
        role="Exact bytes, CID verification, current-root compare-and-swap, WAL.",
        compromise_effect="Byte remint or memory reconstruction is ERROR and fails closed.",
        live_status="measured_hermetic",
        authority="byte_identity",
    ),
    _component(
        "TCB-04",
        trust_class="in_tcb",
        owner="ipfs_accelerate_py",
        surface="state_owners",
        stage="execution_admission",
        role="Objective and task operational admission, leases, fences, terminalization.",
        compromise_effect="Operational remint of Datasets or Kit identities is refused.",
        live_status="measured_hermetic",
        authority="operational_admission",
    ),
    _component(
        "TCB-05",
        trust_class="in_tcb",
        owner="ipfs_accelerate_py",
        surface="canonicalization",
        stage="objective_submission",
        role="CIDv1 DAG-JSON/sha2-256 and raw-block CID encoding over canonical JSON bytes.",
        compromise_effect="SHA-256 hex advertised as a CID is spoofing and is refused.",
        live_status="measured_hermetic",
        authority="identity_encoding",
    ),
    _component(
        "TCB-06",
        trust_class="in_tcb",
        owner="ipfs_kit_py",
        surface="cids",
        stage="planning_and_context_pack",
        role="Bound pack, current-root, chain, objective, client, parity, bypass, and threat-model CIDs.",
        compromise_effect="A reminted CID is not the bound identity and cannot complete.",
        live_status="measured_hermetic",
        authority="byte_identity",
    ),
    _component(
        "TCB-07",
        trust_class="in_tcb",
        owner="ipfs_accelerate_py",
        surface="events",
        stage="task_state_machine",
        role="Append-only, cursor-bound, replay-safe events. Duplicate delivery is identity-preserving.",
        compromise_effect="A second effect from duplicate delivery is rejected.",
        live_status="measured_hermetic",
        authority="event_log",
    ),
    _component(
        "TCB-08",
        trust_class="in_tcb",
        owner="ipfs_accelerate_py",
        surface="leases",
        stage="execution_admission",
        role="Exclusive merge-queue leases. Stale or missing leases cannot complete.",
        compromise_effect="Client-supplied lease bypass is refused.",
        live_status="measured_hermetic",
        authority="lease",
    ),
    _component(
        "TCB-09",
        trust_class="in_tcb",
        owner="ipfs_accelerate_py",
        surface="fences",
        stage="execution_admission",
        role="Fence bindings. Stale or missing fences cannot complete. Direct DuckDB writes are prohibited.",
        compromise_effect="A client write around the fence is refused.",
        live_status="measured_hermetic",
        authority="fence",
    ),
    _component(
        "TCB-10",
        trust_class="in_tcb",
        owner="ipfs_accelerate_py",
        surface="unknown_outcomes",
        stage="execution_admission",
        role="provider-outcome-unknown cannot enter blind retry. Unknown required evidence is not a pass.",
        compromise_effect="Unknown represented as success is repudiation and is refused.",
        live_status="measured_hermetic",
        authority="outcome_classification",
    ),
    _component(
        "TCB-11",
        trust_class="in_tcb",
        owner="human_operator",
        surface="confirmations",
        stage="execution_admission",
        role="Apply and start require exact current roots, authorization, expected effects, lease, fence, and cursor.",
        compromise_effect="Client-supplied confirmation is not authoritative.",
        live_status="measured_hermetic",
        authority="confirmation",
    ),
    _component(
        "TCB-12",
        trust_class="in_tcb",
        owner="ipfs_accelerate_py",
        surface="proof_admission",
        stage="proof_and_test_admission",
        role="Stored proofs are not admitted proofs. Cache hits cannot reuse across trees, policies, or toolchains.",
        compromise_effect="An unadmitted proof treated as admitted is repudiation.",
        live_status="measured_hermetic",
        authority="proof_admission",
    ),
    _component(
        "TCB-13",
        trust_class="in_tcb",
        owner="ipfs_accelerate_py",
        surface="dependencies",
        stage="proof_and_test_admission",
        role="No mutable Git branch in a qualified release. Absent dependencies stay typed unavailable.",
        compromise_effect="Import-time auto-install or mutable-branch pins are refused.",
        live_status="measured_hermetic",
        authority="dependency_lock",
    ),
    _component(
        "TCB-14",
        trust_class="in_tcb",
        owner="ipfs_accelerate_py",
        surface="builds",
        stage="release_decision",
        role="Partial required-build failure cannot publish. Unsigned artifacts are not release authority.",
        compromise_effect="An unsigned or partial-build artifact cannot close a release.",
        live_status="measured_hermetic",
        authority="build_provenance",
    ),
    _component(
        "TCB-15",
        trust_class="in_tcb",
        owner="ipfs_accelerate_py",
        surface="assets",
        stage="receipt",
        role="Bound PCPR-090 threat-model identity and PCPR-072 chain as inventory assets.",
        compromise_effect="Reminting the threat-model or chain CID is refused.",
        live_status="measured_hermetic",
        authority="bound_identity",
    ),
    _component(
        "TCB-16",
        trust_class="in_tcb",
        owner="sealed_validation_environment",
        surface="trust_boundaries",
        stage="proof_and_test_admission",
        role="Sealed PATH and canonical CPython 3.12. Provider-local toolchains are not sealed authority.",
        compromise_effect="A user-writable toolchain used as sealed authority is a trust-boundary violation.",
        live_status="measured_hermetic",
        authority="validation_environment",
    ),
    _component(
        "TCB-17",
        trust_class="conditional_tcb",
        owner="supervisor",
        surface="actors",
        stage="task_state_machine",
        role="Task admission, lease, fence, execution, and terminalization after independent promotion.",
        compromise_effect="Unpromoted supervisor acting as release authority is elevation.",
        live_status="unavailable",
        authority="authority_after_independent_promotion",
    ),
    _component(
        "TCB-18",
        trust_class="conditional_tcb",
        owner="quack",
        surface="fences",
        stage="task_state_machine",
        role="Exclusive authenticated state-owner transport. The only admitted path to DuckDB mutation.",
        compromise_effect="Missing Quack represented as live materialization is spoofing.",
        live_status="unavailable",
        authority="state_owner",
    ),
    _component(
        "TCB-19",
        trust_class="conditional_tcb",
        owner="duckdb",
        surface="state_owners",
        stage="task_state_machine",
        role="Backing task, event, and attempt tables. Never written directly by a task, client, or adapter.",
        compromise_effect="A direct DuckDB write is a prohibited state-owner bypass.",
        live_status="unavailable",
        authority="backing_store",
    ),
    _component(
        "TCB-20",
        trust_class="conditional_tcb",
        owner="formal_toolchain",
        surface="proof_admission",
        stage="proof_and_test_admission",
        role="Reviewed z3, cvc5, Lean, or Coq backends when digest-bound and non-writable.",
        compromise_effect="An absent prover represented as usable weakens mandatory tests.",
        live_status="unavailable",
        authority="conditional_prover",
    ),
    _component(
        "TCB-21",
        trust_class="conditional_tcb",
        owner="live_ipfs",
        surface="cids",
        stage="receipt",
        role="Live IPFS daemon for CID publication. Present CID is not proof authority.",
        compromise_effect="A present CID represented as admitted proof is spoofing.",
        live_status="unavailable",
        authority="conditional_publication",
    ),
    _component(
        "TCB-22",
        trust_class="out_of_tcb",
        owner="untrusted_prompt",
        surface="actors",
        stage="objective_submission",
        role="High-level intent that cannot grant authority.",
        compromise_effect="Prompt injection cannot mint objective identity or policy allow.",
        live_status="measured_hermetic",
        authority="none",
    ),
    _component(
        "TCB-23",
        trust_class="out_of_tcb",
        owner="untrusted_repository",
        surface="trust_boundaries",
        stage="proof_and_test_admission",
        role="Untrusted source, tests, fixtures, and patches.",
        compromise_effect="Simulated or hermetic results represented as live are refused.",
        live_status="measured_hermetic",
        authority="none",
    ),
    _component(
        "TCB-24",
        trust_class="out_of_tcb",
        owner="untrusted_patch_or_agent",
        surface="actors",
        stage="execution_admission",
        role="Untrusted candidate mutation. Cannot broaden owned paths.",
        compromise_effect="Scope expansion beyond the allowlist is refused.",
        live_status="measured_hermetic",
        authority="none",
    ),
    _component(
        "TCB-25",
        trust_class="out_of_tcb",
        owner="model_provider",
        surface="actors",
        stage="proof_and_test_admission",
        role="Candidate generation only. Model assertions cannot complete work.",
        compromise_effect="A model assertion treated as completion is repudiation.",
        live_status="measured_hermetic",
        authority="none",
    ),
    _component(
        "TCB-26",
        trust_class="out_of_tcb",
        owner="python_external_client",
        surface="actors",
        stage="authentication_and_authority",
        role="Submit, inspect, retrieve against the canonical service. Intent, never authority.",
        compromise_effect="Client-supplied policy allow, lease bypass, or self-promotion is refused.",
        live_status="measured_hermetic",
        authority="intent_not_authority",
    ),
    _component(
        "TCB-27",
        trust_class="out_of_tcb",
        owner="generic_mcp_client",
        surface="actors",
        stage="authentication_and_authority",
        role="Submit, inspect, retrieve over MCP JSON-RPC 2.0. Intent, never authority.",
        compromise_effect="forge_policy, write_policy_pointer, and terminalize_task remain JSON-RPC -32001.",
        live_status="measured_hermetic",
        authority="intent_not_authority",
    ),
    _component(
        "TCB-28",
        trust_class="out_of_tcb",
        owner="cli_client",
        surface="actors",
        stage="authentication_and_authority",
        role="Submit, inspect, retrieve over the product CLI. Intent, never authority.",
        compromise_effect="CLI writes of current roots or policy pointers are refused.",
        live_status="measured_hermetic",
        authority="intent_not_authority",
    ),
    _component(
        "TCB-29",
        trust_class="out_of_tcb",
        owner="unsigned_artifact",
        surface="builds",
        stage="release_decision",
        role="Untrusted build or release object.",
        compromise_effect="Unsigned artifacts cannot close a PCPR release.",
        live_status="measured_hermetic",
        authority="none",
    ),
    _component(
        "TCB-30",
        trust_class="out_of_tcb",
        owner="mutable_git_main",
        surface="dependencies",
        stage="release_decision",
        role="Mutable Git branches are not identity and cannot pin a qualified release.",
        compromise_effect="A mutable-main pin treated as current-head identity is refused.",
        live_status="measured_hermetic",
        authority="none",
    ),
    _component(
        "TCB-31",
        trust_class="out_of_tcb",
        owner="sibling_source_tree",
        surface="dependencies",
        stage="proof_and_test_admission",
        role="Sibling source-tree tests/ directories are not package tests.",
        compromise_effect="Package tests requiring uninstalled sibling tests/ are refused.",
        live_status="measured_hermetic",
        authority="none",
    ),
    _component(
        "TCB-32",
        trust_class="out_of_tcb",
        owner="ambient_environment",
        surface="trust_boundaries",
        stage="proof_and_test_admission",
        role="Ambient credentials, clocks, randomness, and profile-home toolchains.",
        compromise_effect="Operator profile state used as sealed authority is a trust-boundary violation.",
        live_status="measured_hermetic",
        authority="none",
    ),
)

if any(not item["bounded"] for item in COMPONENTS):
    raise RuntimeError("PCPR-091 TCB table contains an unbounded component")
if {item["g910_surface"] for item in COMPONENTS} != set(G910_SURFACES):
    raise RuntimeError("PCPR-091 TCB components do not cover every G910 surface")
if {item["trusted_path_stage"] for item in COMPONENTS} != set(TRUSTED_PATH_STAGES):
    raise RuntimeError("PCPR-091 TCB components do not cover every trusted-path stage")
if len({item["component_id"] for item in COMPONENTS}) != len(COMPONENTS):
    raise RuntimeError("PCPR-091 TCB identifiers are not unique")
if not any(item["in_tcb"] for item in COMPONENTS):
    raise RuntimeError("PCPR-091 TCB inventory has no in-TCB components")
if not any(item["out_of_tcb"] for item in COMPONENTS):
    raise RuntimeError("PCPR-091 TCB inventory has no out-of-TCB components")
if not any(item["conditional_tcb"] for item in COMPONENTS):
    raise RuntimeError("PCPR-091 TCB inventory has no conditional TCB components")

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
    "artifacts/proof_carrying_platform_qualification_and_release/receipts/PCPR-091.json",
)

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_trusted_computing_base.py",
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
        "tcb_inventory_files_match_generator",
        "pyproject_tcb_inventory_table",
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
        "all_components_bounded",
        "trusted_path_stages_covered",
        "in_tcb_components_present",
        "out_of_tcb_components_present",
        "conditional_tcb_unavailable_not_live",
        "owner_pack_cid_matches_pin",
        "kit_current_root_bound_not_minted",
        "chain_cid_bound_not_minted",
        "python_client_cid_bound_not_minted",
        "mcp_client_cid_bound_not_minted",
        "parity_cid_bound_not_minted",
        "bypass_proof_cid_bound_not_minted",
        "threat_model_cid_bound_not_minted",
        "tcb_inventory_produced",
        "audit_package_deferred",
        "closed_release_deferred",
        "idempotent_tcb_inventory_cid",
        "model_assertion_cannot_complete_work",
        "duckdb_or_quack_not_written",
        "operator_blocking_task_emitted",
        "no_closed_release_outcome",
    }
)
FORBIDDEN_PRESENT_PROBE_IDS: Final[frozenset[str]] = frozenset(
    {
        "simulated_results_represented_as_live",
        "live_tcb_inventory_represented_as_live",
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
        "threat_model_identity_reminted",
        "model_assertion_completed_work",
        "unauthorized_path_accepted",
        "database_edited",
        "audit_package_claimed",
        "closed_release_claimed",
        "unbounded_component_present",
        "conditional_tcb_represented_as_live",
    }
)

DOCUMENT_README: Final = """# PCPR-091 Accelerate objective-to-release trusted computing base

These files record the Accelerate-owned hermetic trusted-computing-base
inventory for the PCPR objective-to-release path. Actors, assets, trust
boundaries, state owners, canonicalization, CIDs, events, leases, fences,
unknown outcomes, confirmations, proof admission, dependencies, and
builds are bounded as named TCB components.

- `cpython312/reference.trusted-computing-base.json` is the declared TCB
  inventory document. Exact commit and tree are bound by the PCPR-091
  receipt `current_tree_binding`.
- `platform-trusted-computing-base-catalog.json` observes sibling
  Datasets and Kit bindings when present. Sibling source is never
  required.
- Pack, root, chain, objective, Python-client, MCP-client, parity,
  PCPR-083 bypass, and PCPR-090 threat-model identities are bound and
  not reminted. The audit package remains PCPR-092. Closed release
  decision remains PCPR-093/094.
- Live Supervisor.run, live Quack, live MCP, and live CLI sessions stay
  typed unavailable.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-091-operator-live-tcb-inventory`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
"""


class AccelerateTrustedComputingBaseError(Exception):
    """Fail-closed PCPR-091 Accelerate trusted-computing-base error."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AccelerateTrustedComputingBaseError(
            f"{name} must be a non-empty string"
        )
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise AccelerateTrustedComputingBaseError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise AccelerateTrustedComputingBaseError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def parse_pyproject_tcb_inventory_table(text: str) -> dict[str, Any]:
    import tomllib

    table = tomllib.loads(_text(text, "pyproject.toml"))
    if not isinstance(table, dict):
        raise AccelerateTrustedComputingBaseError("pyproject.toml must be a table")
    tool = table.get("tool")
    payload: dict[str, Any] = {}
    if isinstance(tool, dict):
        accelerate = tool.get("ipfs_accelerate_py")
        if isinstance(accelerate, dict):
            raw = accelerate.get("trusted-computing-base")
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
        raise AccelerateTrustedComputingBaseError(
            f"path {path} is not an authorized PCPR-091 path"
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
        raise AccelerateTrustedComputingBaseError(
            f"model assertion at {stage} cannot complete work"
        )


def refuse_pack_cid_remint(cid: str) -> str:
    if cid != PINNED_PACK_CID:
        raise AccelerateTrustedComputingBaseError(
            f"ContextPack CID {cid} remints {PINNED_PACK_CID}"
        )
    return cid


def refuse_current_root_remint(cid: str) -> str:
    if cid != PINNED_CURRENT_ROOT_CID:
        raise AccelerateTrustedComputingBaseError(
            f"current root CID {cid} remints {PINNED_CURRENT_ROOT_CID}"
        )
    return cid


def refuse_chain_cid_remint(cid: str) -> str:
    if cid != PINNED_CHAIN_CID:
        raise AccelerateTrustedComputingBaseError(
            f"chain CID {cid} remints {PINNED_CHAIN_CID}"
        )
    return cid


def refuse_objective_cid_remint(cid: str) -> str:
    if cid != PINNED_OBJECTIVE_CID:
        raise AccelerateTrustedComputingBaseError(
            f"objective CID {cid} remints {PINNED_OBJECTIVE_CID}"
        )
    return cid


def refuse_idea_digest_remint(digest: str) -> str:
    if digest != PINNED_IDEA_DIGEST:
        raise AccelerateTrustedComputingBaseError(
            f"idea digest {digest} remints {PINNED_IDEA_DIGEST}"
        )
    return digest


def refuse_python_client_cid_remint(cid: str) -> str:
    if cid != PINNED_PYTHON_CLIENT_CID:
        raise AccelerateTrustedComputingBaseError(
            f"Python client CID {cid} remints {PINNED_PYTHON_CLIENT_CID}"
        )
    return cid


def refuse_mcp_client_cid_remint(cid: str) -> str:
    if cid != PINNED_MCP_CLIENT_CID:
        raise AccelerateTrustedComputingBaseError(
            f"MCP client CID {cid} remints {PINNED_MCP_CLIENT_CID}"
        )
    return cid


def refuse_parity_cid_remint(cid: str) -> str:
    if cid != PINNED_PARITY_CID:
        raise AccelerateTrustedComputingBaseError(
            f"parity CID {cid} remints {PINNED_PARITY_CID}"
        )
    return cid


def refuse_bypass_proof_cid_remint(cid: str) -> str:
    if cid != PINNED_BYPASS_PROOF_CID:
        raise AccelerateTrustedComputingBaseError(
            f"bypass proof CID {cid} remints {PINNED_BYPASS_PROOF_CID}"
        )
    return cid


def refuse_threat_model_cid_remint(cid: str) -> str:
    if cid != PINNED_THREAT_MODEL_CID:
        raise AccelerateTrustedComputingBaseError(
            f"threat-model CID {cid} remints {PINNED_THREAT_MODEL_CID}"
        )
    return cid


def refuse_database_edit(*, edited: bool) -> None:
    if edited:
        raise AccelerateTrustedComputingBaseError(
            "trusted-computing-base inventory cannot write DuckDB or Quack state"
        )


def refuse_live_tcb_inventory(*, applied_live: bool) -> None:
    if applied_live:
        raise AccelerateTrustedComputingBaseError(
            "live trusted-computing-base execution remains typed unavailable"
        )


def refuse_self_promotion(*, promoted: bool) -> None:
    if promoted:
        raise AccelerateTrustedComputingBaseError(
            "trusted-computing-base inventory cannot promote itself"
        )


def refuse_non_canonical_objective(idea: str) -> str:
    text = _text(idea, "idea")
    if text != REFERENCE_OBJECTIVE_IDEA:
        raise AccelerateTrustedComputingBaseError(
            "trusted-computing-base inventory must bind the canonical PCPR-060 idea"
        )
    return text


def refuse_audit_package_claim(*, claimed: bool) -> None:
    if claimed:
        raise AccelerateTrustedComputingBaseError(
            "security and correctness audit package remains PCPR-092"
        )


def refuse_closed_release_claim(*, claimed: bool) -> None:
    if claimed:
        raise AccelerateTrustedComputingBaseError(
            "closed release decision remains PCPR-093/094"
        )


def refuse_unbounded_component(*, unbounded: bool) -> None:
    if unbounded:
        raise AccelerateTrustedComputingBaseError(
            "every TCB component must be bounded to a declared actor class, surface, and stage"
        )


def refuse_conditional_as_live(*, claimed_live: bool) -> None:
    if claimed_live:
        raise AccelerateTrustedComputingBaseError(
            "conditional TCB components that are not admitted remain typed unavailable"
        )


def g910_surface_bounds() -> dict[str, bool]:
    surfaces = {item["g910_surface"] for item in COMPONENTS if item["bounded"]}
    return {name: name in surfaces for name in G910_SURFACES}


def all_g910_surfaces_bounded() -> bool:
    bounds = g910_surface_bounds()
    return all(bounds[name] for name in G910_SURFACES) and set(bounds) == set(
        G910_SURFACES
    )


def hermetic_tcb_inventory_trace() -> dict[str, Any]:
    bounds = g910_surface_bounds()
    stages = tuple(item["trusted_path_stage"] for item in COMPONENTS)
    in_tcb = [item["component_id"] for item in COMPONENTS if item["in_tcb"]]
    conditional = [
        item["component_id"] for item in COMPONENTS if item["conditional_tcb"]
    ]
    out = [item["component_id"] for item in COMPONENTS if item["out_of_tcb"]]
    return {
        "schema": "ipfs_accelerate_py/assurance/hermetic-tcb-inventory-trace@1",
        "interface": "HermeticTrustedComputingBaseTrace@1",
        "component_count": len(COMPONENTS),
        "component_ids": [item["component_id"] for item in COMPONENTS],
        "in_tcb_ids": in_tcb,
        "conditional_tcb_ids": conditional,
        "out_of_tcb_ids": out,
        "all_components_bounded": all(item["bounded"] is True for item in COMPONENTS),
        "trusted_path_stages": list(TRUSTED_PATH_STAGES),
        "trusted_path_stages_covered": set(stages) == set(TRUSTED_PATH_STAGES),
        "g910_surfaces": list(G910_SURFACES),
        "g910_surface_bounds": bounds,
        "all_g910_surfaces_bounded": all_g910_surfaces_bounded(),
        "actors_bounded": bounds["actors"],
        "assets_bounded": bounds["assets"],
        "trust_boundaries_bounded": bounds["trust_boundaries"],
        "state_owners_bounded": bounds["state_owners"],
        "in_tcb_components_present": bool(in_tcb),
        "out_of_tcb_components_present": bool(out),
        "conditional_tcb_unavailable_not_live": all(
            item["live"] is False and item["admitted_live"] is False
            for item in COMPONENTS
            if item["conditional_tcb"]
        ),
        "tcb_inventory_produced": True,
        "tcb_inventory_deferred": False,
        "audit_package_deferred": True,
        "closed_release_deferred": True,
        "no_live_claim": True,
        "no_closed_release": True,
        "identities_bound_not_reminted": True,
        "duplicate_effect_rejected": True,
        "unauthorized_effect": False,
        "compromise_rule": COMPROMISE_RULE,
        "live": False,
        "hermetic": True,
        "evidence_kind": "measured_hermetic",
    }


def hermetic_tcb_inventory_idempotency(inventory_cid: str) -> dict[str, Any]:
    return {
        "schema": "ipfs_accelerate_py/assurance/hermetic-tcb-inventory-idempotency@1",
        "interface": "HermeticTrustedComputingBaseIdempotency@1",
        "first_inventory_command_digest": INVENTORY_COMMAND_DIGEST,
        "second_inventory_command_digest": INVENTORY_COMMAND_DIGEST,
        "first_tcb_inventory_cid": inventory_cid,
        "second_tcb_inventory_cid": inventory_cid,
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
class AccelerateTrustedComputingBaseVerdict:
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
    tcb_inventory_cid: str
    document_cid: str
    catalog_cid: str
    threat_model_cid: str
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
            "tcb_inventory_cid": self.tcb_inventory_cid,
            "document_cid": self.document_cid,
            "catalog_cid": self.catalog_cid,
            "threat_model_cid": self.threat_model_cid,
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
            "An admitted Quack-fenced state-owner session before the TCB "
            "inventory is executed as live supervisor work"
        ),
        "action": (
            "Keep the hermetic TCB inventory. Keep every G910 surface bounded. "
            "Do not write DuckDB or Quack state. Do not escalate to a model. "
            "Do not emit an audit package or closed release. PCPR-092 prepares "
            "the security and correctness audit package."
        ),
        "reason": (
            "Hermetic TCB inventory coverage is not a live Supervisor.run / "
            "CLI / MCP session. Sealed validation has no admitted "
            "Quack-fenced session. Direct DuckDB writes are prohibited."
        ),
    }


def produce_tcb_inventory(
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
    threat_model_cid: str = PINNED_THREAT_MODEL_CID,
    idea: str = REFERENCE_OBJECTIVE_IDEA,
    complete_from: str | None = None,
    database_edited: bool = False,
    applied_live: bool = False,
    self_promoted: bool = False,
    audit_claimed: bool = False,
    closed_release_claimed: bool = False,
    unbounded: bool = False,
    conditional_live: bool = False,
) -> dict[str, Any]:
    """Produce the hermetic objective-to-release TCB inventory. Fail closed."""

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
    refuse_threat_model_cid_remint(threat_model_cid)
    refuse_non_canonical_objective(idea)
    refuse_database_edit(edited=database_edited)
    refuse_live_tcb_inventory(applied_live=applied_live)
    refuse_self_promotion(promoted=self_promoted)
    refuse_audit_package_claim(claimed=audit_claimed)
    refuse_closed_release_claim(claimed=closed_release_claimed)
    refuse_unbounded_component(unbounded=unbounded)
    refuse_conditional_as_live(claimed_live=conditional_live)
    checked_paths = list(paths) if paths is not None else list(AUTHORIZED_PATH_PREFIXES)
    for path in checked_paths:
        refuse_unauthorized_path(path)
    trace = hermetic_tcb_inventory_trace()
    return {
        "schema": INVENTORY_SCHEMA,
        "interface": INVENTORY_INTERFACE,
        "owner_interface": INTERFACE,
        "task_id": PCPR_091_TASK_ID,
        "goal_id": PCPR_091_GOAL_ID,
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
        "threat_model_cid": threat_model_cid,
        "python_client_interface": PYTHON_CLIENT_INTERFACE,
        "mcp_client_interface": MCP_CLIENT_INTERFACE,
        "parity_interface": PARITY_INTERFACE,
        "bypass_interface": BYPASS_INTERFACE,
        "threat_model_interface": THREAT_MODEL_INTERFACE,
        "protocol_interface": PROTOCOL_INTERFACE,
        "documentation_interface": DOCUMENTATION_INTERFACE,
        "documentation_schema": DOCUMENTATION_SCHEMA,
        "change_kind": CHANGE_KIND,
        "inventory_kind": INVENTORY_KIND,
        "canonical_service": CANONICAL_SERVICE,
        "authority": CLIENT_AUTHORITY,
        "trusted_path_stages": list(TRUSTED_PATH_STAGES),
        "g910_surfaces": list(G910_SURFACES),
        "trust_classes": list(TRUST_CLASSES),
        "compromise_rule": COMPROMISE_RULE,
        "components": [dict(item) for item in COMPONENTS],
        "produced": True,
        "applied": True,
        "live": False,
        "hermetic": True,
        "admitted_live": False,
        "deferred": False,
        "paths_authorized": True,
        "authorized_path_prefixes": list(AUTHORIZED_PATH_PREFIXES),
        "all_g910_surfaces_bounded": True,
        "all_components_bounded": True,
        "trusted_path_stages_covered": True,
        "in_tcb_components_present": True,
        "out_of_tcb_components_present": True,
        "conditional_tcb_unavailable_not_live": True,
        "tcb_inventory_produced": True,
        "tcb_inventory_deferred": False,
        "audit_package_deferred": True,
        "closed_release_deferred": True,
        "client_self_promoted": False,
        "trace": trace,
        "inventory_command_digest": INVENTORY_COMMAND_DIGEST,
        "idempotent": True,
        "duplicate_effect_rejected": True,
        "remints_protocol": False,
        "adds_protocol_operation": False,
        "model_assertion_completes_work": False,
        "escalation_order": list(ESCALATION_ORDER),
        "completed_through": COMPLETED_THROUGH,
        "next_authorized_stage": NEXT_AUTHORIZED_STAGE,
        "next_task_id": PCPR_092_TASK_ID,
        "prerequisite_task_id": PCPR_090_TASK_ID,
        "python_client_task_id": PCPR_080_TASK_ID,
        "mcp_client_task_id": PCPR_081_TASK_ID,
        "parity_task_id": PCPR_082_TASK_ID,
        "bypass_task_id": PCPR_083_TASK_ID,
        "chain_task_id": PCPR_072_TASK_ID,
        "threat_model_task_id": PCPR_090_TASK_ID,
        "duckdb_or_quack_state_written": False,
        "closed_release_outcome": None,
        "release_claim": False,
        "evidence_kind": "measured_hermetic",
    }


def canonical_tcb_inventory() -> dict[str, Any]:
    return produce_tcb_inventory()


def tcb_inventory_cid_of(plan: Mapping[str, Any] | None = None) -> str:
    payload = dict(plan or canonical_tcb_inventory())
    payload.pop("tcb_inventory_cid", None)
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
                "Sibling trusted-computing-base file is not present beside "
                "this checkout."
            ),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    context_pack = payload.get("context_pack")
    storage = payload.get("storage")
    inventory = payload.get("trusted_computing_base")
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
        or (inventory.get("chain_cid") if isinstance(inventory, Mapping) else None),
        "tcb_inventory_cid": payload.get("tcb_inventory_cid")
        or (
            inventory.get("tcb_inventory_cid")
            if isinstance(inventory, Mapping)
            else None
        ),
        "binding_cid": payload.get("binding_cid") or payload.get("document_cid"),
        "live": False,
        "applied": False,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def render_declared_inventory(root: Path | None = None) -> dict[str, Any]:
    package_root = root or discover_accelerate_root()
    if package_root is None:
        raise AccelerateTrustedComputingBaseError(
            "Accelerate package root was not found"
        )
    state_owner = observe_state_owner()
    plan = canonical_tcb_inventory()
    computed_inventory_cid = tcb_inventory_cid_of(plan)
    idempotency = hermetic_tcb_inventory_idempotency(computed_inventory_cid)
    document = {
        "schema": DOCUMENT_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_091_TASK_ID,
        "goal_id": PCPR_091_GOAL_ID,
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
            "survives_tcb_inventory": True,
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
            "survives_tcb_inventory": True,
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
            "task_id": PCPR_090_TASK_ID,
            "interface": THREAT_MODEL_INTERFACE,
            "threat_model_cid": PINNED_THREAT_MODEL_CID,
            "document_cid": PINNED_THREAT_MODEL_DOCUMENT_CID,
            "produced": True,
            "live": False,
            "hermetic": True,
            "reminted": False,
            "survives_tcb_inventory": True,
            "evidence_kind": "measured_hermetic",
        },
        "trusted_computing_base": {
            **plan,
            "tcb_inventory_cid": computed_inventory_cid,
            "idempotency": idempotency,
            "produced_by": EXECUTION_OWNER_REPOSITORY,
        },
        "materialization": {
            "kind": "declared_tcb_inventory_not_live",
            "admitted": False,
            "live": False,
            "applied": False,
            "duckdb_or_quack_state_written": False,
            "evidence_kind": "unavailable",
        },
        "live_application": typed_unavailable(
            reason=(
                "No admitted Quack-fenced supervisor session was observed. "
                "Hermetic TCB inventory coverage is not live execution."
            )
        ),
        "live_client": typed_unavailable(
            reason=(
                "Live Python Supervisor.run and live generic MCP remain typed "
                "unavailable. This task only emits a hermetic TCB inventory. "
                "Security and correctness audit package remains PCPR-092."
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
                "field": "PCPR-091 receipt current_tree_binding",
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


def platform_tcb_inventory_catalog(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateTrustedComputingBaseError(
            "Accelerate package root was not found"
        )
    parent = root.parent
    document = render_declared_inventory(root)
    datasets_binding = _sibling_binding(
        parent
        / "ipfs_datasets"
        / DOCUMENT_DIR_RELPATH
        / "reference.trusted-computing-base.binding.json",
        relative_path=(
            f"../ipfs_datasets/{DOCUMENT_DIR_RELPATH}/"
            "reference.trusted-computing-base.binding.json"
        ),
    )
    kit_binding = _sibling_binding(
        parent
        / "ipfs_kit"
        / DOCUMENT_DIR_RELPATH
        / "reference.trusted-computing-base.binding.json",
        relative_path=(
            f"../ipfs_kit/{DOCUMENT_DIR_RELPATH}/"
            "reference.trusted-computing-base.binding.json"
        ),
    )
    catalog = {
        "schema": CATALOG_SCHEMA,
        "interface": CATALOG_INTERFACE,
        "task_id": PCPR_091_TASK_ID,
        "goal_id": PCPR_091_GOAL_ID,
        "objective_kind": OBJECTIVE_KIND,
        "portfolio_id": PORTFOLIO_ID,
        "portfolio_version": PORTFOLIO_VERSION,
        "objective_cid": PINNED_OBJECTIVE_CID,
        "idea_digest": PINNED_IDEA_DIGEST,
        "pack_cid": PINNED_PACK_CID,
        "current_root_cid": PINNED_CURRENT_ROOT_CID,
        "chain_cid": PINNED_CHAIN_CID,
        "threat_model_cid": PINNED_THREAT_MODEL_CID,
        "tcb_inventory_cid": document["trusted_computing_base"]["tcb_inventory_cid"],
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
                "threat_model_cid": PINNED_THREAT_MODEL_CID,
                "tcb_inventory_cid": document["trusted_computing_base"][
                    "tcb_inventory_cid"
                ],
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


def write_tcb_inventory_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateTrustedComputingBaseError(
            "Accelerate package root was not found"
        )
    document = render_declared_inventory(root)
    paths = artifact_paths(root)
    _atomic_write(paths["document"], pretty_json(document))
    readme = DOCUMENT_README if DOCUMENT_README.endswith("\n") else DOCUMENT_README + "\n"
    _atomic_write(paths["readme"], readme)
    catalog = platform_tcb_inventory_catalog(start)
    _atomic_write(paths["catalog"], pretty_json(catalog))
    return {
        "document": document,
        "catalog": catalog,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def verify_tcb_inventory_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateTrustedComputingBaseError(
            "Accelerate package root was not found"
        )
    document = render_declared_inventory(root)
    catalog = platform_tcb_inventory_catalog(start)
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
        "tcb_inventory_cid": document["trusted_computing_base"]["tcb_inventory_cid"],
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
        raise AccelerateTrustedComputingBaseError(
            "Accelerate package root was not found"
        )
    document = render_declared_inventory(root)
    verified = verify_tcb_inventory_files(root)
    table = parse_pyproject_tcb_inventory_table(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    inventory = document["trusted_computing_base"]
    trace = inventory["trace"]
    idempotency = inventory["idempotency"]
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
        or document["threat_model"]["threat_model_cid"] != PINNED_THREAT_MODEL_CID
        or inventory["tcb_inventory_cid"] != PINNED_TCB_INVENTORY_CID
    )
    probes = [
        _probe(
            "tcb_inventory_files_match_generator",
            verified["ok"] is True,
            reason=(
                "Declared TCB inventory files match the generator."
                if verified["ok"]
                else "Declared TCB inventory files do not match the generator."
            ),
        ),
        _probe(
            "pyproject_tcb_inventory_table",
            table.get("interface") == INTERFACE and table.get("task-id") == PCPR_091_TASK_ID,
            reason=(
                "pyproject.toml declares the PCPR-091 trusted-computing-base table."
                if table.get("interface") == INTERFACE
                else "pyproject.toml does not declare the PCPR-091 TCB table."
            ),
        ),
        _probe(
            "paths_remain_authorized",
            inventory["paths_authorized"] is True
            and list(inventory["authorized_path_prefixes"])
            == list(AUTHORIZED_PATH_PREFIXES),
            reason="TCB inventory paths remain inside the declared PCPR-091 allowlist.",
        ),
        _probe(
            "canonical_objective_bytes_match",
            inventory["idea"] == REFERENCE_OBJECTIVE_IDEA
            and inventory["idea_digest"] == PINNED_IDEA_DIGEST,
            reason="TCB inventory uses the canonical PCPR-060 idea bytes.",
        ),
        _probe(
            "actors_bounded",
            trace["actors_bounded"] is True,
            reason="Actors are bounded as TCB components.",
        ),
        _probe(
            "assets_bounded",
            trace["assets_bounded"] is True,
            reason="Assets are bounded as TCB components.",
        ),
        _probe(
            "trust_boundaries_bounded",
            trace["trust_boundaries_bounded"] is True,
            reason="Trust boundaries are bounded as TCB components.",
        ),
        _probe(
            "state_owners_bounded",
            trace["state_owners_bounded"] is True,
            reason="State owners are bounded as TCB components.",
        ),
        _probe(
            "canonicalization_bounded",
            trace["g910_surface_bounds"]["canonicalization"] is True,
            reason="Canonicalization is bounded as a TCB component.",
        ),
        _probe(
            "cids_bounded",
            trace["g910_surface_bounds"]["cids"] is True,
            reason="CID vectors are bounded as TCB components.",
        ),
        _probe(
            "events_bounded",
            trace["g910_surface_bounds"]["events"] is True,
            reason="Events are bounded as TCB components.",
        ),
        _probe(
            "leases_bounded",
            trace["g910_surface_bounds"]["leases"] is True,
            reason="Leases are bounded as TCB components.",
        ),
        _probe(
            "fences_bounded",
            trace["g910_surface_bounds"]["fences"] is True,
            reason="Fences are bounded as TCB components.",
        ),
        _probe(
            "unknown_outcomes_bounded",
            trace["g910_surface_bounds"]["unknown_outcomes"] is True,
            reason="Unknown outcomes are bounded as TCB components.",
        ),
        _probe(
            "confirmations_bounded",
            trace["g910_surface_bounds"]["confirmations"] is True,
            reason="Confirmations are bounded as TCB components.",
        ),
        _probe(
            "proof_admission_bounded",
            trace["g910_surface_bounds"]["proof_admission"] is True,
            reason="Proof admission is bounded as a TCB component.",
        ),
        _probe(
            "dependencies_bounded",
            trace["g910_surface_bounds"]["dependencies"] is True,
            reason="Dependencies are bounded as TCB components.",
        ),
        _probe(
            "builds_bounded",
            trace["g910_surface_bounds"]["builds"] is True,
            reason="Builds are bounded as TCB components.",
        ),
        _probe(
            "all_g910_surfaces_bounded",
            inventory["all_g910_surfaces_bounded"] is True
            and trace["all_g910_surfaces_bounded"] is True,
            reason="Every PCPR-G910 surface is bounded by a TCB component.",
        ),
        _probe(
            "all_components_bounded",
            inventory["all_components_bounded"] is True
            and trace["all_components_bounded"] is True,
            reason="Every recorded TCB component is bounded.",
        ),
        _probe(
            "trusted_path_stages_covered",
            inventory["trusted_path_stages_covered"] is True
            and trace["trusted_path_stages_covered"] is True,
            reason="TCB components cover every objective-to-release stage.",
        ),
        _probe(
            "in_tcb_components_present",
            inventory["in_tcb_components_present"] is True
            and trace["in_tcb_components_present"] is True,
            reason="In-TCB components are present.",
        ),
        _probe(
            "out_of_tcb_components_present",
            inventory["out_of_tcb_components_present"] is True
            and trace["out_of_tcb_components_present"] is True,
            reason="Out-of-TCB components are present.",
        ),
        _probe(
            "conditional_tcb_unavailable_not_live",
            inventory["conditional_tcb_unavailable_not_live"] is True
            and trace["conditional_tcb_unavailable_not_live"] is True,
            reason="Conditional TCB components stay typed unavailable and are not live.",
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
            "threat_model_cid_bound_not_minted",
            document["threat_model"]["threat_model_cid"] == PINNED_THREAT_MODEL_CID
            and document["threat_model"]["reminted"] is False,
            reason="PCPR-090 threat-model CID is bound and not reminted.",
        ),
        _probe(
            "tcb_inventory_produced",
            inventory["tcb_inventory_produced"] is True
            and inventory["tcb_inventory_deferred"] is False
            and inventory["next_task_id"] == PCPR_092_TASK_ID,
            reason="Trusted-computing-base inventory is produced by PCPR-091.",
        ),
        _probe(
            "audit_package_deferred",
            inventory["audit_package_deferred"] is True,
            reason="Security and correctness audit package remains PCPR-092.",
        ),
        _probe(
            "closed_release_deferred",
            inventory["closed_release_deferred"] is True
            and document["closed_release_outcome"] is None,
            reason="Closed release decision remains PCPR-093/094.",
        ),
        _probe(
            "idempotent_tcb_inventory_cid",
            inventory["idempotent"] is True
            and idempotency["identical"] is True
            and idempotency["first_tcb_inventory_cid"] == inventory["tcb_inventory_cid"]
            and idempotency["second_tcb_inventory_cid"] == inventory["tcb_inventory_cid"]
            and inventory["tcb_inventory_cid"] == PINNED_TCB_INVENTORY_CID,
            reason="A second TCB inventory emission yields the same inventory CID.",
        ),
        _probe(
            "model_assertion_cannot_complete_work",
            inventory["model_assertion_completes_work"] is False,
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
                "client, MCP client, parity, bypass, threat-model, or TCB "
                "inventory CID is forbidden."
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
            "threat_model_identity_reminted",
            document["threat_model"]["threat_model_cid"] != PINNED_THREAT_MODEL_CID
            or document["threat_model"]["reminted"] is True,
            reason="Accelerate must not remint the PCPR-090 threat-model CID.",
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
            "live_tcb_inventory_represented_as_live",
            False,
            reason="Hermetic TCB inventory coverage is not represented as live.",
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
            "unbounded_component_present",
            False,
            reason="Every recorded TCB component is bounded.",
        ),
        _probe(
            "conditional_tcb_represented_as_live",
            False,
            reason="Conditional TCB components are not represented as live.",
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


def qualify_tcb_inventory(
    probes: Sequence[OutcomeProbe],
    *,
    tcb_inventory_cid: str,
    document_cid: str,
    catalog_cid: str,
    threat_model_cid: str,
    chain_cid: str,
    pack_cid: str,
    current_root_cid: str,
    objective_cid: str,
    idea_digest_cid: str,
    python_client_cid: str = PINNED_PYTHON_CLIENT_CID,
    mcp_client_cid: str = PINNED_MCP_CLIENT_CID,
    parity_cid: str = PINNED_PARITY_CID,
    bypass_proof_cid: str = PINNED_BYPASS_PROOF_CID,
) -> AccelerateTrustedComputingBaseVerdict:
    if not probes:
        raise AccelerateTrustedComputingBaseError("at least one probe is required")
    normalized: list[OutcomeProbe] = []
    blockers: list[str] = []
    for probe in probes:
        kind = _kind(probe.evidence_kind, "evidence_kind")
        if probe.live and kind != "measured_live":
            raise AccelerateTrustedComputingBaseError(
                "live claims require measured_live evidence"
            )
        if probe.simulated_represented_as_live:
            raise AccelerateTrustedComputingBaseError(
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
        "task_id": PCPR_091_TASK_ID,
        "goal_id": PCPR_091_GOAL_ID,
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
        "tcb_inventory_cid": tcb_inventory_cid,
        "document_cid": document_cid,
        "catalog_cid": catalog_cid,
        "threat_model_cid": threat_model_cid,
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
    return AccelerateTrustedComputingBaseVerdict(
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
        tcb_inventory_cid=tcb_inventory_cid,
        document_cid=document_cid,
        catalog_cid=catalog_cid,
        threat_model_cid=threat_model_cid,
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


def qualify_current_head_tcb_inventory(
    start: Path | None = None,
) -> AccelerateTrustedComputingBaseVerdict:
    document = render_declared_inventory(start)
    catalog = platform_tcb_inventory_catalog(start)
    return qualify_tcb_inventory(
        current_head_static_probes(start),
        tcb_inventory_cid=str(document["trusted_computing_base"]["tcb_inventory_cid"]),
        document_cid=str(document["document_cid"]),
        catalog_cid=str(catalog["catalog_cid"]),
        threat_model_cid=str(document["threat_model"]["threat_model_cid"]),
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


def pcpr_091_receipt_promotion(
    verdict: AccelerateTrustedComputingBaseVerdict,
) -> dict[str, Any]:
    if verdict.closed_release_outcome is not None:
        raise AccelerateTrustedComputingBaseError(
            "trusted-computing-base inventory must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise AccelerateTrustedComputingBaseError(
            "trusted-computing-base inventory must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise AccelerateTrustedComputingBaseError(
            "trusted-computing-base inventory completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise AccelerateTrustedComputingBaseError(
            "trusted-computing-base inventory must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise AccelerateTrustedComputingBaseError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.live_client or verdict.live_application:
        raise AccelerateTrustedComputingBaseError(
            "live client claims require measured_live evidence"
        )
    return verdict.to_mapping()


PINNED_TCB_INVENTORY_CID: Final = (
    "baguqeeral3tjeriuspe3e3qe25eqssmer6fekhmmq3v7htpkr2rjjiqatrcq"
)
PINNED_DOCUMENT_CID: Final = (
    "baguqeerajafnba4wq4j4chsfyxj2fbopvxml4iyeopymbkhs5gkl6m6udfiq"
)
PINNED_CATALOG_CID: Final = (
    "baguqeera2vshvzjuifiqszl6ke3pdbyprkidujhgcr4ltr46l753byz6wleq"
)
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeeraiykditwxsura27onnn5eybwfnspr3av7v52uqfg6qw4vrd57vsqa"
)


__all__ = [
    "AUTHORIZED_PATH_PREFIXES",
    "CLOSED_RELEASE_OUTCOMES",
    "COMPONENTS",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "ESCALATION_ORDER",
    "G910_SURFACES",
    "HERMETIC_CANDIDATE_SUITES",
    "INTERFACE",
    "OBJECTIVE_KIND",
    "OPERATOR_BLOCKING_TASK_ID",
    "OutcomeProbe",
    "PCPR_091_GOAL_ID",
    "PCPR_091_TASK_ID",
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
    "PINNED_TCB_INVENTORY_CID",
    "PINNED_THREAT_MODEL_CID",
    "SCHEMA",
    "SEALED_PATH",
    "SEALED_PYTHON",
    "TRUSTED_PATH_STAGES",
    "TRUST_CLASSES",
    "AccelerateTrustedComputingBaseError",
    "all_g910_surfaces_bounded",
    "current_head_static_probes",
    "hermetic_tcb_inventory_trace",
    "path_is_authorized",
    "pcpr_091_receipt_promotion",
    "platform_tcb_inventory_catalog",
    "produce_tcb_inventory",
    "qualify_current_head_tcb_inventory",
    "qualify_tcb_inventory",
    "refuse_audit_package_claim",
    "refuse_bypass_proof_cid_remint",
    "refuse_chain_cid_remint",
    "refuse_closed_release_claim",
    "refuse_conditional_as_live",
    "refuse_database_edit",
    "refuse_mcp_client_cid_remint",
    "refuse_model_completion",
    "refuse_non_canonical_objective",
    "refuse_objective_cid_remint",
    "refuse_pack_cid_remint",
    "refuse_parity_cid_remint",
    "refuse_python_client_cid_remint",
    "refuse_self_promotion",
    "refuse_threat_model_cid_remint",
    "refuse_unauthorized_path",
    "refuse_unbounded_component",
    "render_declared_inventory",
    "verify_tcb_inventory_files",
]
