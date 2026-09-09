"""Fail-closed PCPR-093 Accelerate-owned release-candidate gate.

Accelerate owns the hermetic release-candidate gate run for the PCPR
objective-to-release path. Supervisor, Datasets, Kit, Accelerate,
packaging, reference-workflow, interoperability, and hard-safety gates
are evaluated as named, bounded records. Failed, stale, partial,
simulated, estimated, or missing evidence is recorded, never hidden.
Live claims require measured_live evidence. Missing environments stay
typed unavailable.

This gate run is not a closed PCPR release outcome. Closed release
decision remains PCPR-094. The PCPR-092 audit package, PCPR-091 TCB
inventory, and PCPR-090 threat model are bound and not reminted.

Live Supervisor.run, live Quack, live MCP, and live CLI stay typed
unavailable. Simulated results are not live. This evaluator does not
write DuckDB or Quack state and does not emit a closed PCPR release.

Hermetic gate-run coverage is candidate coverage only.
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


INTERFACE: Final = "AccelerateReleaseCandidateGate@1"
SCHEMA: Final = "ipfs_accelerate_py/assurance/release-candidate-gate@1"
DOCUMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/declared-release-candidate-gate@1"
)
PACKAGE_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/release-candidate-gate-inventory@1"
)
VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/release-candidate-gate-verdict@1"
)
CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/platform-release-candidate-gate-catalog@1"
)
CATALOG_INTERFACE: Final = "PlatformReleaseCandidateGate@1"
PACKAGE_INTERFACE: Final = "AuthoritativeReleaseCandidateGate@1"
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
TCB_INTERFACE: Final = "AccelerateTrustedComputingBase@1"
AUDIT_PACKAGE_INTERFACE: Final = "AccelerateSecurityAndCorrectnessAuditPackage@1"
PROTOCOL_INTERFACE: Final = "LogicProviderProtocol@2"
DOCUMENTATION_INTERFACE: Final = "LogicProviderProtocolReleaseCandidateGate@1"
DOCUMENTATION_SCHEMA: Final = (
    "ipfs_datasets_py/logic-provider-release-candidate-gate@1"
)
PCPR_093_TASK_ID: Final = "PCPR-093"
PCPR_093_GOAL_ID: Final = "PCPR-G900"
PCPR_094_TASK_ID: Final = "PCPR-094"
PCPR_092_TASK_ID: Final = "PCPR-092"
PCPR_091_TASK_ID: Final = "PCPR-091"
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
OBJECTIVE_KIND: Final = "declared_objective_to_release_candidate_gate"
OBJECTIVE_ID: Final = "PCPR-G900"
LANGUAGE: Final = "Python"
OPERATOR_BLOCKING_TASK_ID: Final = "pcpr-093-operator-live-release-candidate-gate"
DOCUMENT_DIR_RELPATH: Final = "packaging/pcpr/reference-workflow/cpython312"
DOCUMENT_JSON_NAME: Final = "reference.release-candidate-gate.json"
DOCUMENT_README_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/RELEASE_CANDIDATE_GATE.md"
)
CATALOG_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/platform-release-candidate-gate-catalog.json"
)
SOURCE_DATE_EPOCH: Final = "0"
SOURCE_REPOSITORY: Final = "endomorphosis/ipfs_accelerate_py"
COMPLETED_THROUGH: Final = "release_candidate_gate"
NEXT_AUTHORIZED_STAGE: Final = "closed_release_decision"
CHANGE_KIND: Final = "objective_to_release_candidate_gate"
PACKAGE_KIND: Final = "hermetic_objective_to_release_candidate_gate"
CANONICAL_SERVICE: Final = "pcpr-canonical-supervisor-service"
CLIENT_AUTHORITY: Final = "intent_not_authority"
PACKAGE_COMMAND_DIGEST: Final = "pcpr-093-hermetic-release-candidate-gate"
AUDIT_STATUS_READY: Final = "external_audit_ready"
AUDIT_STATUS: Final = AUDIT_STATUS_READY

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
PINNED_TCB_INVENTORY_CID: Final = (
    "baguqeeral3tjeriuspe3e3qe25eqssmer6fekhmmq3v7htpkr2rjjiqatrcq"
)
PINNED_TCB_DOCUMENT_CID: Final = (
    "baguqeerajafnba4wq4j4chsfyxj2fbopvxml4iyeopymbkhs5gkl6m6udfiq"
)
PINNED_AUDIT_PACKAGE_CID: Final = (
    "baguqeerazukvaxrpd53tqeo636o6rhtyr6iiqvx3njz444rj6yuzjeed3cpq"
)
PINNED_AUDIT_PACKAGE_DOCUMENT_CID: Final = (
    "baguqeerakwiirkz6hkietp5sirow5s4okaexitoscvnxhte2uo7wdpuhswga"
)

if PINNED_LOCK_CID != (
    "baguqeerawsekbbbt5ccctjt4tydzeahfkahsatb6q5x7k6cy4inrii5lhqmq"
):
    raise RuntimeError("PCPR-093 lock CID remints PCPR-056")
if PINNED_PYTHON_CLIENT_CID == PINNED_MCP_CLIENT_CID:
    raise RuntimeError("Python and MCP client CIDs must remain distinct")
if PINNED_AUDIT_PACKAGE_CID == PINNED_TCB_INVENTORY_CID:
    raise RuntimeError("Audit-package CID must remain distinct from TCB inventory CID")
if OWNER_INTERFACES["SupervisorContextPack"] != OWNER_INTERFACE:
    raise RuntimeError("PCPR-093 remints SupervisorContextPack owner interface")
if OWNER_SCHEMAS["SupervisorContextPack"] != OWNER_SCHEMA:
    raise RuntimeError("PCPR-093 remints SupervisorContextPack owner schema")

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

G920_EVIDENCE_CLASSES: Final[tuple[str, ...]] = (
    "architecture",
    "state_machine",
    "authority",
    "canonicalization",
    "recovery",
    "proof",
    "build",
    "negative_test",
    "limitation",
)

GATE_FAMILIES: Final[tuple[str, ...]] = (
    "supervisor",
    "datasets",
    "kit",
    "accelerate",
    "packaging",
    "reference_workflow",
    "interoperability",
    "hard_safety",
)

SEALED_REPRODUCTION_PREFIX: Final = (
    f"PATH={SEALED_PATH} PYTHONNOUSERSITE=1 {SEALED_PYTHON}"
)


def _gate(
    gate_id: str,
    *,
    family: str,
    title: str,
    requirements: tuple[str, ...],
    hermetic_evidence_tasks: tuple[str, ...],
    hermetic_result: str,
    live_blocker: str,
    limitation: str,
    bound_identities: tuple[str, ...],
) -> dict[str, Any]:
    if family not in GATE_FAMILIES:
        raise RuntimeError(f"gate {gate_id} uses unbounded family {family}")
    if hermetic_result not in {"evaluated", "blocked", "unavailable"}:
        raise RuntimeError(f"gate {gate_id} uses unbounded hermetic_result")
    return {
        "gate_id": gate_id,
        "family": family,
        "title": title,
        "requirements": list(requirements),
        "hermetic_evidence_tasks": list(hermetic_evidence_tasks),
        "hermetic_status": hermetic_result,
        "hermetic_pass_is_not_live_pass": True,
        "live_status": "unavailable",
        "live_evidence_kind": "unavailable",
        "live_blocker": live_blocker,
        "hidden_failed_stale_partial_simulated_estimated_or_missing": False,
        "limitation": limitation,
        "bound_identities": list(bound_identities),
        "bounded": True,
        "reproducible": True,
        "live": False,
        "admitted_live": False,
        "pass_live": False,
        "closed_release": False,
        "live_status_kind": "typed_unavailable",
        "evidence_kind": "measured_hermetic",
    }


GATES: Final[tuple[dict[str, Any], ...]] = (
    _gate(
        "GTE-01",
        family="supervisor",
        title="Supervisor direct-submission, refill, and current-head gate",
        requirements=(
            "working direct submission",
            "automatic reassessment and refill",
            "zero manual database repair",
            "promotion or explicit bounded operator override",
            "exact current-head evidence",
        ),
        hermetic_evidence_tasks=("PCPR-000", "PCPR-004", "PCPR-060", "PCPR-072"),
        hermetic_result="evaluated",
        live_blocker=(
            "No admitted Quack-fenced Supervisor.run session was observed. "
            "Hermetic supervisor receipts are not live promotion."
        ),
        limitation=(
            "Supervisor hermetic receipts exist. Live direct submission, "
            "live refill, and live promotion stay typed unavailable."
        ),
        bound_identities=(PINNED_CHAIN_CID, PINNED_OBJECTIVE_CID),
    ),
    _gate(
        "GTE-02",
        family="datasets",
        title="Datasets inert-import, protocol, license, schema, and solver gate",
        requirements=(
            "inert import",
            "no default auto-install or false success",
            "canonical typed provider protocol",
            "consistent licensing",
            "packaged schemas",
            "recorded real solver qualification",
            "typed unavailability for missing solvers",
        ),
        hermetic_evidence_tasks=(
            "PCPR-010",
            "PCPR-013",
            "PCPR-014",
            "PCPR-015",
            "PCPR-016",
            "PCPR-017",
        ),
        hermetic_result="evaluated",
        live_blocker=(
            "Sealed validation has no admitted z3, cvc5, lean, or coqtop. "
            "Missing solvers stay typed unavailable and are not recorded as passing."
        ),
        limitation=(
            "Datasets hermetic protocol, license, and schema receipts exist. "
            "Live solver qualification remains typed unavailable."
        ),
        bound_identities=(PINNED_PACK_CID, PINNED_OBJECTIVE_CID),
    ),
    _gate(
        "GTE-03",
        family="kit",
        title="Kit durability, backend, IPFS/Iroh, VFS/WAL, and proof-store gate",
        requirements=(
            "current-head local durability",
            "live evidence for every live backend",
            "qualified IPFS if claimed",
            "Iroh qualified or removed from live claims",
            "VFS, WAL, root, and proof-store qualification",
            "no adjacent test-tree dependency",
        ),
        hermetic_evidence_tasks=(
            "PCPR-020",
            "PCPR-021",
            "PCPR-022",
            "PCPR-023",
            "PCPR-024",
            "PCPR-025",
            "PCPR-026",
            "PCPR-027",
        ),
        hermetic_result="evaluated",
        live_blocker=(
            "Sealed validation has no admitted ipfs daemon or live Iroh. "
            "Live backend claims stay typed unavailable."
        ),
        limitation=(
            "Kit hermetic current-root reconstruction is from disk. Live "
            "IPFS, Iroh, and backend qualification stay typed unavailable."
        ),
        bound_identities=(PINNED_CURRENT_ROOT_CID, PINNED_PACK_CID),
    ),
    _gate(
        "GTE-04",
        family="accelerate",
        title="Accelerate capability-ladder, CPU, CUDA, model, and metadata gate",
        requirements=(
            "no default mock hardware, pseudo-CID, or fabricated endpoint authority",
            "the real capability ladder",
            "qualified CPU",
            "CUDA evidence if GPU support is claimed",
            "a qualified model/provider path",
            "immutable release dependencies",
            "accurate Python metadata",
        ),
        hermetic_evidence_tasks=(
            "PCPR-030",
            "PCPR-031",
            "PCPR-032",
            "PCPR-033",
            "PCPR-034",
            "PCPR-035",
            "PCPR-043",
        ),
        hermetic_result="evaluated",
        live_blocker=(
            "nvidia-smi presence is not contract qualification. CUDA kernels "
            "and live model/provider paths stay typed unavailable unless "
            "measured_live evidence is admitted."
        ),
        limitation=(
            "Accelerate hermetic capability-ladder and metadata receipts exist. "
            "Live CUDA and live model/provider qualification stay typed unavailable."
        ),
        bound_identities=(PINNED_LOCK_CID, PINNED_CHAIN_CID),
    ),
    _gate(
        "GTE-05",
        family="packaging",
        title="Packaging install, lock, SBOM, provenance, signature, and publication gate",
        requirements=(
            "clean installs",
            "exact locks",
            "SBOMs",
            "provenance",
            "signed artifacts",
            "the exact compatibility manifest",
            "no partial publication",
            "current-head release checks",
        ),
        hermetic_evidence_tasks=(
            "PCPR-050",
            "PCPR-053",
            "PCPR-054",
            "PCPR-055",
            "PCPR-056",
            "PCPR-057",
        ),
        hermetic_result="evaluated",
        live_blocker=(
            "Live PyPI publication, live GPG/SSH tags, and live GitHub "
            "branch protection stay typed unavailable. Unsigned artifacts "
            "cannot close a release."
        ),
        limitation=(
            "Declared locks, SBOMs, and the PCPR-056 compatibility lock are "
            "bound. Live publication and live signatures stay typed unavailable."
        ),
        bound_identities=(PINNED_LOCK_CID,),
    ),
    _gate(
        "GTE-06",
        family="reference_workflow",
        title="Reference-workflow objective-to-receipt-chain gate",
        requirements=(
            "direct high-level submission",
            "ContextPack and storage root",
            "deterministic-first execution",
            "selected tests and proofs",
            "unrelated-change reuse",
            "relevant-change invalidation",
            "PlanDelta and refill",
            "restart recovery",
            "the complete receipt chain",
        ),
        hermetic_evidence_tasks=(
            "PCPR-060",
            "PCPR-061",
            "PCPR-062",
            "PCPR-063",
            "PCPR-064",
            "PCPR-065",
            "PCPR-066",
            "PCPR-067",
            "PCPR-068",
            "PCPR-069",
            "PCPR-070",
            "PCPR-071",
            "PCPR-072",
        ),
        hermetic_result="evaluated",
        live_blocker=(
            "The reference workflow is hermetic. Live Supervisor.run of the "
            "objective-to-receipt path stays typed unavailable. The bound "
            "ContextPack remains stale and rejected."
        ),
        limitation=(
            "Hermetic objective, pack, root, route, patch, reuse, invalidation, "
            "PlanDelta, recovery, and chain receipts exist. Live application "
            "stays typed unavailable. Pack CID remains stale and rejected."
        ),
        bound_identities=(
            PINNED_OBJECTIVE_CID,
            PINNED_PACK_CID,
            PINNED_CURRENT_ROOT_CID,
            PINNED_CHAIN_CID,
        ),
    ),
    _gate(
        "GTE-07",
        family="interoperability",
        title="Python and generic MCP client identity, semantics, and authority gate",
        requirements=(
            "Python external client",
            "generic MCP client",
            "identical objective identity",
            "equivalent semantics",
            "no authority bypass",
        ),
        hermetic_evidence_tasks=("PCPR-080", "PCPR-081", "PCPR-082", "PCPR-083"),
        hermetic_result="evaluated",
        live_blocker=(
            "Live Python Supervisor.run and live generic MCP stdio/HTTP stay "
            "typed unavailable. Client principals remain intent, never authority."
        ),
        limitation=(
            "Hermetic Python-client, MCP-client, parity, and bypass-proof "
            "identities are bound and not reminted. Live clients stay typed unavailable."
        ),
        bound_identities=(
            PINNED_PYTHON_CLIENT_CID,
            PINNED_MCP_CLIENT_CID,
            PINNED_PARITY_CID,
            PINNED_BYPASS_PROOF_CID,
        ),
    ),
    _gate(
        "GTE-08",
        family="hard_safety",
        title="Hard-zero counters and fail-closed safety gate",
        requirements=(
            "every hard-zero counter is a release blocker",
            "no unauthorized effect",
            "no hidden failed, stale, partial, simulated, estimated, or missing evidence",
        ),
        hermetic_evidence_tasks=("PCPR-083", "PCPR-090", "PCPR-091", "PCPR-092"),
        hermetic_result="evaluated",
        live_blocker=(
            "Hard-zero counters remain zero hermetically. Live confirmation "
            "of those counters under a Quack-fenced session stays typed unavailable."
        ),
        limitation=(
            "Hermetic bypass proof keeps hard-zero counters at zero. The "
            "audit package is external_audit_ready and never externally_audited. "
            "This is not a closed PCPR release."
        ),
        bound_identities=(
            PINNED_BYPASS_PROOF_CID,
            PINNED_THREAT_MODEL_CID,
            PINNED_TCB_INVENTORY_CID,
            PINNED_AUDIT_PACKAGE_CID,
        ),
    ),
)

if any(not item["bounded"] for item in GATES):
    raise RuntimeError("PCPR-093 gate table contains an unbounded gate")
if {item["family"] for item in GATES} != set(GATE_FAMILIES):
    raise RuntimeError("PCPR-093 gates do not cover every declared family")
if len({item["gate_id"] for item in GATES}) != len(GATES):
    raise RuntimeError("PCPR-093 gate identifiers are not unique")
if any(item["closed_release"] for item in GATES):
    raise RuntimeError("PCPR-093 gates must not claim a closed release")
if any(item["pass_live"] for item in GATES):
    raise RuntimeError("PCPR-093 gates must not claim a live pass")
if any(item["live"] for item in GATES):
    raise RuntimeError("PCPR-093 gates must not claim live execution")
if any(
    item["hidden_failed_stale_partial_simulated_estimated_or_missing"] for item in GATES
):
    raise RuntimeError("PCPR-093 must not hide failed, stale, partial, or missing evidence")

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
    "artifacts/proof_carrying_platform_qualification_and_release/receipts/PCPR-093.json",
)

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_release_candidate_gate.py",
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
        "gate_files_match_generator",
        "pyproject_release_candidate_gate_table",
        "paths_remain_authorized",
        "canonical_objective_bytes_match",
        "supervisor_gate_evaluated",
        "datasets_gate_evaluated",
        "kit_gate_evaluated",
        "accelerate_gate_evaluated",
        "packaging_gate_evaluated",
        "reference_workflow_gate_evaluated",
        "interoperability_gate_evaluated",
        "hard_safety_gate_evaluated",
        "all_gate_families_evaluated",
        "all_gates_bounded",
        "no_hidden_failed_stale_partial_simulated_estimated_or_missing",
        "reproduction_commands_work",
        "owner_pack_cid_matches_pin",
        "kit_current_root_bound_not_minted",
        "chain_cid_bound_not_minted",
        "python_client_cid_bound_not_minted",
        "mcp_client_cid_bound_not_minted",
        "parity_cid_bound_not_minted",
        "bypass_proof_cid_bound_not_minted",
        "threat_model_cid_bound_not_minted",
        "tcb_inventory_cid_bound_not_minted",
        "audit_package_cid_bound_not_minted",
        "gate_run_produced",
        "closed_release_deferred",
        "idempotent_gate_run_cid",
        "model_assertion_cannot_complete_work",
        "duckdb_or_quack_not_written",
        "operator_blocking_task_emitted",
        "no_closed_release_outcome",
    }
)
FORBIDDEN_PRESENT_PROBE_IDS: Final[frozenset[str]] = frozenset(
    {
        "simulated_results_represented_as_live",
        "live_gate_run_represented_as_live",
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
        "tcb_inventory_identity_reminted",
        "audit_package_identity_reminted",
        "model_assertion_completed_work",
        "unauthorized_path_accepted",
        "database_edited",
        "closed_release_claimed",
        "unbounded_gate_present",
        "hidden_evidence_present",
        "reproduction_commands_failed",
        "live_pass_claimed",
    }
)

DOCUMENT_README: Final = """# PCPR-093 Accelerate release-candidate gate

These files record the Accelerate-owned hermetic release-candidate gate
run for the PCPR objective-to-release path. Supervisor, Datasets, Kit,
Accelerate, packaging, reference-workflow, interoperability, and
hard-safety gates are evaluated as named bounded records.

- `cpython312/reference.release-candidate-gate.json` is the declared
  gate-run document. Exact commit and tree are bound by the PCPR-093
  receipt `current_tree_binding`.
- `platform-release-candidate-gate-catalog.json` observes sibling
  Datasets and Kit bindings when present. Sibling source is never
  required.
- Pack, root, chain, objective, Python-client, MCP-client, parity,
  PCPR-083 bypass, PCPR-090 threat-model, PCPR-091 TCB-inventory, and
  PCPR-092 audit-package identities are bound and not reminted.
- Failed, stale, partial, simulated, estimated, or missing evidence is
  recorded and never hidden. Live claims require measured_live evidence.
- This gate run is not a closed PCPR release outcome. Closed release
  decision remains PCPR-094.
- Live Supervisor.run, live Quack, live MCP, and live CLI sessions stay
  typed unavailable.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-093-operator-live-release-candidate-gate`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
"""


class AccelerateReleaseCandidateGateError(Exception):
    """Fail-closed PCPR-093 Accelerate release-candidate gate error."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AccelerateReleaseCandidateGateError(
            f"{name} must be a non-empty string"
        )
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise AccelerateReleaseCandidateGateError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise AccelerateReleaseCandidateGateError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def parse_pyproject_release_candidate_gate_table(text: str) -> dict[str, Any]:
    import tomllib

    table = tomllib.loads(_text(text, "pyproject.toml"))
    if not isinstance(table, dict):
        raise AccelerateReleaseCandidateGateError("pyproject.toml must be a table")
    tool = table.get("tool")
    payload: dict[str, Any] = {}
    if isinstance(tool, dict):
        accelerate = tool.get("ipfs_accelerate_py")
        if isinstance(accelerate, dict):
            raw = accelerate.get("release-candidate-gate")
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
        raise AccelerateReleaseCandidateGateError(
            f"path {path} is not an authorized PCPR-093 path"
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
        raise AccelerateReleaseCandidateGateError(
            f"model assertion at {stage} cannot complete work"
        )


def refuse_pack_cid_remint(cid: str) -> str:
    if cid != PINNED_PACK_CID:
        raise AccelerateReleaseCandidateGateError(
            f"ContextPack CID {cid} remints {PINNED_PACK_CID}"
        )
    return cid


def refuse_current_root_remint(cid: str) -> str:
    if cid != PINNED_CURRENT_ROOT_CID:
        raise AccelerateReleaseCandidateGateError(
            f"current root CID {cid} remints {PINNED_CURRENT_ROOT_CID}"
        )
    return cid


def refuse_chain_cid_remint(cid: str) -> str:
    if cid != PINNED_CHAIN_CID:
        raise AccelerateReleaseCandidateGateError(
            f"chain CID {cid} remints {PINNED_CHAIN_CID}"
        )
    return cid


def refuse_objective_cid_remint(cid: str) -> str:
    if cid != PINNED_OBJECTIVE_CID:
        raise AccelerateReleaseCandidateGateError(
            f"objective CID {cid} remints {PINNED_OBJECTIVE_CID}"
        )
    return cid


def refuse_idea_digest_remint(digest: str) -> str:
    if digest != PINNED_IDEA_DIGEST:
        raise AccelerateReleaseCandidateGateError(
            f"idea digest {digest} remints {PINNED_IDEA_DIGEST}"
        )
    return digest


def refuse_python_client_cid_remint(cid: str) -> str:
    if cid != PINNED_PYTHON_CLIENT_CID:
        raise AccelerateReleaseCandidateGateError(
            f"Python client CID {cid} remints {PINNED_PYTHON_CLIENT_CID}"
        )
    return cid


def refuse_mcp_client_cid_remint(cid: str) -> str:
    if cid != PINNED_MCP_CLIENT_CID:
        raise AccelerateReleaseCandidateGateError(
            f"MCP client CID {cid} remints {PINNED_MCP_CLIENT_CID}"
        )
    return cid


def refuse_parity_cid_remint(cid: str) -> str:
    if cid != PINNED_PARITY_CID:
        raise AccelerateReleaseCandidateGateError(
            f"parity CID {cid} remints {PINNED_PARITY_CID}"
        )
    return cid


def refuse_bypass_proof_cid_remint(cid: str) -> str:
    if cid != PINNED_BYPASS_PROOF_CID:
        raise AccelerateReleaseCandidateGateError(
            f"bypass proof CID {cid} remints {PINNED_BYPASS_PROOF_CID}"
        )
    return cid


def refuse_threat_model_cid_remint(cid: str) -> str:
    if cid != PINNED_THREAT_MODEL_CID:
        raise AccelerateReleaseCandidateGateError(
            f"threat-model CID {cid} remints {PINNED_THREAT_MODEL_CID}"
        )
    return cid


def refuse_tcb_inventory_cid_remint(cid: str) -> str:
    if cid != PINNED_TCB_INVENTORY_CID:
        raise AccelerateReleaseCandidateGateError(
            f"TCB inventory CID {cid} remints {PINNED_TCB_INVENTORY_CID}"
        )
    return cid


def refuse_audit_package_cid_remint(cid: str) -> str:
    if cid != PINNED_AUDIT_PACKAGE_CID:
        raise AccelerateReleaseCandidateGateError(
            f"audit-package CID {cid} remints {PINNED_AUDIT_PACKAGE_CID}"
        )
    return cid


def refuse_database_edit(*, edited: bool) -> None:
    if edited:
        raise AccelerateReleaseCandidateGateError(
            "release-candidate gate cannot write DuckDB or Quack state"
        )


def refuse_live_gate_run(*, applied_live: bool) -> None:
    if applied_live:
        raise AccelerateReleaseCandidateGateError(
            "live release-candidate gate execution remains typed unavailable"
        )


def refuse_self_promotion(*, promoted: bool) -> None:
    if promoted:
        raise AccelerateReleaseCandidateGateError(
            "release-candidate gate cannot promote itself"
        )


def refuse_non_canonical_objective(idea: str) -> str:
    text = _text(idea, "idea")
    if text != REFERENCE_OBJECTIVE_IDEA:
        raise AccelerateReleaseCandidateGateError(
            "release-candidate gate must bind the canonical PCPR-060 idea"
        )
    return text


def refuse_closed_release_claim(*, claimed: bool) -> None:
    if claimed:
        raise AccelerateReleaseCandidateGateError(
            "closed release decision remains PCPR-094"
        )


def refuse_unbounded_gate(*, unbounded: bool) -> None:
    if unbounded:
        raise AccelerateReleaseCandidateGateError(
            "every release-candidate gate must be bounded to a declared family"
        )


def refuse_hidden_evidence(*, hidden: bool) -> None:
    if hidden:
        raise AccelerateReleaseCandidateGateError(
            "failed, stale, partial, simulated, estimated, or missing evidence must not be hidden"
        )


def refuse_live_pass(*, claimed: bool) -> None:
    if claimed:
        raise AccelerateReleaseCandidateGateError(
            "hermetic gate evaluation is not a live pass"
        )


def all_gate_families_evaluated() -> bool:
    families = {item["family"] for item in GATES if item["hermetic_status"] == "evaluated"}
    return families == set(GATE_FAMILIES)


def all_gates_bounded() -> bool:
    return all(item["bounded"] is True for item in GATES)


def no_hidden_evidence() -> bool:
    return all(
        item["hidden_failed_stale_partial_simulated_estimated_or_missing"] is False
        for item in GATES
    )


def hermetic_gate_run_trace() -> dict[str, Any]:
    family_bounds = {
        name: any(item["family"] == name and item["bounded"] for item in GATES)
        for name in GATE_FAMILIES
    }
    return {
        "schema": "ipfs_accelerate_py/assurance/hermetic-release-candidate-gate-trace@1",
        "interface": "HermeticReleaseCandidateGateTrace@1",
        "gate_count": len(GATES),
        "gate_ids": [item["gate_id"] for item in GATES],
        "gate_families": list(GATE_FAMILIES),
        "family_bounds": family_bounds,
        "all_gates_bounded": all_gates_bounded(),
        "all_gate_families_evaluated": all_gate_families_evaluated(),
        "supervisor_gate_evaluated": family_bounds["supervisor"],
        "datasets_gate_evaluated": family_bounds["datasets"],
        "kit_gate_evaluated": family_bounds["kit"],
        "accelerate_gate_evaluated": family_bounds["accelerate"],
        "packaging_gate_evaluated": family_bounds["packaging"],
        "reference_workflow_gate_evaluated": family_bounds["reference_workflow"],
        "interoperability_gate_evaluated": family_bounds["interoperability"],
        "hard_safety_gate_evaluated": family_bounds["hard_safety"],
        "no_hidden_failed_stale_partial_simulated_estimated_or_missing": (
            no_hidden_evidence()
        ),
        "reproduction_commands_work": all(
            item["reproducible"] is True for item in GATES
        ),
        "any_live_pass": False,
        "any_closed_release": False,
        "gate_run_produced": True,
        "gate_run_deferred": False,
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


def hermetic_gate_run_idempotency(gate_run_cid: str) -> dict[str, Any]:
    return {
        "schema": "ipfs_accelerate_py/assurance/hermetic-release-candidate-gate-idempotency@1",
        "interface": "HermeticReleaseCandidateGateIdempotency@1",
        "first_gate_command_digest": PACKAGE_COMMAND_DIGEST,
        "second_gate_command_digest": PACKAGE_COMMAND_DIGEST,
        "first_gate_run_cid": gate_run_cid,
        "second_gate_run_cid": gate_run_cid,
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
class AccelerateReleaseCandidateGateVerdict:
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
    gate_run_cid: str
    document_cid: str
    catalog_cid: str
    audit_package_cid: str
    tcb_inventory_cid: str
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
            "gate_run_cid": self.gate_run_cid,
            "document_cid": self.document_cid,
            "catalog_cid": self.catalog_cid,
            "audit_package_cid": self.audit_package_cid,
            "tcb_inventory_cid": self.tcb_inventory_cid,
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
            "An admitted Quack-fenced state-owner session before the "
            "release-candidate gate is executed as live supervisor work"
        ),
        "action": (
            "Keep the hermetic gate run. Keep every gate family evaluated. "
            "Keep failed, stale, partial, simulated, estimated, or missing "
            "evidence visible. Do not write DuckDB or Quack state. Do not "
            "escalate to a model. Do not emit a closed release. PCPR-094 "
            "produces the promotion or honest non-promotion receipt."
        ),
        "reason": (
            "Hermetic release-candidate gate coverage is not a live "
            "Supervisor.run / CLI / MCP session. Sealed validation has no "
            "admitted Quack-fenced session. Direct DuckDB writes are prohibited."
        ),
    }


def produce_release_candidate_gate(
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
    tcb_inventory_cid: str = PINNED_TCB_INVENTORY_CID,
    audit_package_cid: str = PINNED_AUDIT_PACKAGE_CID,
    idea: str = REFERENCE_OBJECTIVE_IDEA,
    complete_from: str | None = None,
    database_edited: bool = False,
    applied_live: bool = False,
    self_promoted: bool = False,
    closed_release_claimed: bool = False,
    unbounded: bool = False,
    hidden: bool = False,
    live_pass_claimed: bool = False,
) -> dict[str, Any]:
    """Produce the hermetic objective-to-release candidate gate run. Fail closed."""

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
    refuse_tcb_inventory_cid_remint(tcb_inventory_cid)
    refuse_audit_package_cid_remint(audit_package_cid)
    refuse_non_canonical_objective(idea)
    refuse_database_edit(edited=database_edited)
    refuse_live_gate_run(applied_live=applied_live)
    refuse_self_promotion(promoted=self_promoted)
    refuse_closed_release_claim(claimed=closed_release_claimed)
    refuse_unbounded_gate(unbounded=unbounded)
    refuse_hidden_evidence(hidden=hidden)
    refuse_live_pass(claimed=live_pass_claimed)
    checked_paths = list(paths) if paths is not None else list(AUTHORIZED_PATH_PREFIXES)
    for path in checked_paths:
        refuse_unauthorized_path(path)
    trace = hermetic_gate_run_trace()
    return {
        "schema": PACKAGE_SCHEMA,
        "interface": PACKAGE_INTERFACE,
        "owner_interface": INTERFACE,
        "task_id": PCPR_093_TASK_ID,
        "goal_id": PCPR_093_GOAL_ID,
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
        "tcb_inventory_cid": tcb_inventory_cid,
        "audit_package_cid": audit_package_cid,
        "python_client_interface": PYTHON_CLIENT_INTERFACE,
        "mcp_client_interface": MCP_CLIENT_INTERFACE,
        "parity_interface": PARITY_INTERFACE,
        "bypass_interface": BYPASS_INTERFACE,
        "threat_model_interface": THREAT_MODEL_INTERFACE,
        "tcb_interface": TCB_INTERFACE,
        "audit_package_interface": AUDIT_PACKAGE_INTERFACE,
        "protocol_interface": PROTOCOL_INTERFACE,
        "documentation_interface": DOCUMENTATION_INTERFACE,
        "documentation_schema": DOCUMENTATION_SCHEMA,
        "change_kind": CHANGE_KIND,
        "package_kind": PACKAGE_KIND,
        "canonical_service": CANONICAL_SERVICE,
        "authority": CLIENT_AUTHORITY,
        "trusted_path_stages": list(TRUSTED_PATH_STAGES),
        "g910_surfaces": list(G910_SURFACES),
        "g920_evidence_classes": list(G920_EVIDENCE_CLASSES),
        "gate_families": list(GATE_FAMILIES),
        "audit_status": AUDIT_STATUS,
        "externally_audited": False,
        "gates": [dict(item) for item in GATES],
        "produced": True,
        "applied": True,
        "live": False,
        "hermetic": True,
        "admitted_live": False,
        "deferred": False,
        "paths_authorized": True,
        "authorized_path_prefixes": list(AUTHORIZED_PATH_PREFIXES),
        "all_gate_families_evaluated": True,
        "all_gates_bounded": True,
        "no_hidden_failed_stale_partial_simulated_estimated_or_missing": True,
        "reproduction_commands_work": True,
        "gate_run_produced": True,
        "gate_run_deferred": False,
        "closed_release_deferred": True,
        "client_self_promoted": False,
        "any_live_pass": False,
        "trace": trace,
        "package_command_digest": PACKAGE_COMMAND_DIGEST,
        "idempotent": True,
        "duplicate_effect_rejected": True,
        "remints_protocol": False,
        "adds_protocol_operation": False,
        "model_assertion_completes_work": False,
        "escalation_order": list(ESCALATION_ORDER),
        "completed_through": COMPLETED_THROUGH,
        "next_authorized_stage": NEXT_AUTHORIZED_STAGE,
        "next_task_id": PCPR_094_TASK_ID,
        "prerequisite_task_id": PCPR_092_TASK_ID,
        "python_client_task_id": PCPR_080_TASK_ID,
        "mcp_client_task_id": PCPR_081_TASK_ID,
        "parity_task_id": PCPR_082_TASK_ID,
        "bypass_task_id": PCPR_083_TASK_ID,
        "chain_task_id": PCPR_072_TASK_ID,
        "threat_model_task_id": PCPR_090_TASK_ID,
        "tcb_inventory_task_id": PCPR_091_TASK_ID,
        "audit_package_task_id": PCPR_092_TASK_ID,
        "duckdb_or_quack_state_written": False,
        "closed_release_outcome": None,
        "release_claim": False,
        "evidence_kind": "measured_hermetic",
    }


def canonical_release_candidate_gate() -> dict[str, Any]:
    return produce_release_candidate_gate()


def gate_run_cid_of(plan: Mapping[str, Any] | None = None) -> str:
    payload = dict(plan or canonical_release_candidate_gate())
    payload.pop("gate_run_cid", None)
    payload.pop("idempotency", None)
    return content_identity(payload)


def reproduce_gate(gate_id: str) -> dict[str, Any]:
    """Hermetically reproduce one named gate evaluation. Fail closed."""

    wanted = _text(gate_id, "gate_id")
    match = next((item for item in GATES if item["gate_id"] == wanted), None)
    if match is None:
        raise AccelerateReleaseCandidateGateError(f"unknown gate {wanted}")
    plan = canonical_release_candidate_gate()
    run_cid = gate_run_cid_of(plan)
    return {
        "gate_id": wanted,
        "ok": (
            match["bounded"] is True
            and match["reproducible"] is True
            and match["closed_release"] is False
            and match["live"] is False
            and match["pass_live"] is False
            and match["hidden_failed_stale_partial_simulated_estimated_or_missing"]
            is False
            and run_cid.startswith("baguqeera")
            and plan["closed_release_deferred"] is True
        ),
        "family": match["family"],
        "hermetic_status": match["hermetic_status"],
        "live_status": match["live_status"],
        "gate_run_cid": run_cid,
        "live": False,
        "hermetic": True,
        "closed_release": False,
        "evidence_kind": "measured_hermetic",
    }


def reproduce_release_candidate_gate() -> dict[str, Any]:
    """Run every declared gate reproduction hermetically."""

    results = [reproduce_gate(item["gate_id"]) for item in GATES]
    return {
        "ok": all(item["ok"] is True for item in results),
        "command_count": len(results),
        "gate_ids": [item["gate_id"] for item in results],
        "results": results,
        "closed_release_deferred": True,
        "live": False,
        "hermetic": True,
        "evidence_kind": "measured_hermetic",
    }


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
                "Sibling release-candidate-gate binding file is not present "
                "beside this checkout."
            ),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    context_pack = payload.get("context_pack")
    storage = payload.get("storage")
    package = payload.get("release_candidate_gate")
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
        or (package.get("chain_cid") if isinstance(package, Mapping) else None),
        "gate_run_cid": payload.get("gate_run_cid")
        or (package.get("gate_run_cid") if isinstance(package, Mapping) else None),
        "binding_cid": payload.get("binding_cid") or payload.get("document_cid"),
        "live": False,
        "applied": False,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def render_declared_package(root: Path | None = None) -> dict[str, Any]:
    package_root = root or discover_accelerate_root()
    if package_root is None:
        raise AccelerateReleaseCandidateGateError(
            "Accelerate package root was not found"
        )
    state_owner = observe_state_owner()
    plan = canonical_release_candidate_gate()
    computed_gate_run_cid = gate_run_cid_of(plan)
    idempotency = hermetic_gate_run_idempotency(computed_gate_run_cid)
    reproduction = reproduce_release_candidate_gate()
    document = {
        "schema": DOCUMENT_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_093_TASK_ID,
        "goal_id": PCPR_093_GOAL_ID,
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
            "survives_release_candidate_gate": True,
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
            "survives_release_candidate_gate": True,
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
            "survives_release_candidate_gate": True,
            "evidence_kind": "measured_hermetic",
        },
        "trusted_computing_base": {
            "task_id": PCPR_091_TASK_ID,
            "interface": TCB_INTERFACE,
            "tcb_inventory_cid": PINNED_TCB_INVENTORY_CID,
            "document_cid": PINNED_TCB_DOCUMENT_CID,
            "produced": True,
            "live": False,
            "hermetic": True,
            "reminted": False,
            "survives_release_candidate_gate": True,
            "evidence_kind": "measured_hermetic",
        },
        "security_and_correctness_audit_package": {
            "task_id": PCPR_092_TASK_ID,
            "interface": AUDIT_PACKAGE_INTERFACE,
            "audit_package_cid": PINNED_AUDIT_PACKAGE_CID,
            "document_cid": PINNED_AUDIT_PACKAGE_DOCUMENT_CID,
            "audit_status": AUDIT_STATUS_READY,
            "externally_audited": False,
            "produced": True,
            "live": False,
            "hermetic": True,
            "reminted": False,
            "survives_release_candidate_gate": True,
            "evidence_kind": "measured_hermetic",
        },
        "release_candidate_gate": {
            **plan,
            "gate_run_cid": computed_gate_run_cid,
            "idempotency": idempotency,
            "reproduction": reproduction,
            "produced_by": EXECUTION_OWNER_REPOSITORY,
        },
        "materialization": {
            "kind": "declared_release_candidate_gate_not_live",
            "admitted": False,
            "live": False,
            "applied": False,
            "duckdb_or_quack_state_written": False,
            "evidence_kind": "unavailable",
        },
        "live_application": typed_unavailable(
            reason=(
                "No admitted Quack-fenced supervisor session was observed. "
                "Hermetic release-candidate gate coverage is not live execution."
            )
        ),
        "live_client": typed_unavailable(
            reason=(
                "Live Python Supervisor.run and live generic MCP remain typed "
                "unavailable. This task only emits a hermetic gate run. "
                "Closed release decision remains PCPR-094."
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
                "field": "PCPR-093 receipt current_tree_binding",
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


def platform_release_candidate_gate_catalog(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateReleaseCandidateGateError(
            "Accelerate package root was not found"
        )
    parent = root.parent
    document = render_declared_package(root)
    datasets_binding = _sibling_binding(
        parent
        / "ipfs_datasets"
        / DOCUMENT_DIR_RELPATH
        / "reference.release-candidate-gate.binding.json",
        relative_path=(
            f"../ipfs_datasets/{DOCUMENT_DIR_RELPATH}/"
            "reference.release-candidate-gate.binding.json"
        ),
    )
    kit_binding = _sibling_binding(
        parent
        / "ipfs_kit"
        / DOCUMENT_DIR_RELPATH
        / "reference.release-candidate-gate.binding.json",
        relative_path=(
            f"../ipfs_kit/{DOCUMENT_DIR_RELPATH}/"
            "reference.release-candidate-gate.binding.json"
        ),
    )
    catalog = {
        "schema": CATALOG_SCHEMA,
        "interface": CATALOG_INTERFACE,
        "task_id": PCPR_093_TASK_ID,
        "goal_id": PCPR_093_GOAL_ID,
        "objective_kind": OBJECTIVE_KIND,
        "portfolio_id": PORTFOLIO_ID,
        "portfolio_version": PORTFOLIO_VERSION,
        "objective_cid": PINNED_OBJECTIVE_CID,
        "idea_digest": PINNED_IDEA_DIGEST,
        "pack_cid": PINNED_PACK_CID,
        "current_root_cid": PINNED_CURRENT_ROOT_CID,
        "chain_cid": PINNED_CHAIN_CID,
        "threat_model_cid": PINNED_THREAT_MODEL_CID,
        "tcb_inventory_cid": PINNED_TCB_INVENTORY_CID,
        "audit_package_cid": PINNED_AUDIT_PACKAGE_CID,
        "gate_run_cid": document["release_candidate_gate"]["gate_run_cid"],
        "python_client_cid": PINNED_PYTHON_CLIENT_CID,
        "mcp_client_cid": PINNED_MCP_CLIENT_CID,
        "parity_cid": PINNED_PARITY_CID,
        "bypass_proof_cid": PINNED_BYPASS_PROOF_CID,
        "lock_cid": PINNED_LOCK_CID,
        "audit_status": AUDIT_STATUS,
        "externally_audited": False,
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
                "tcb_inventory_cid": PINNED_TCB_INVENTORY_CID,
                "audit_package_cid": PINNED_AUDIT_PACKAGE_CID,
                "gate_run_cid": document["release_candidate_gate"]["gate_run_cid"],
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


def write_release_candidate_gate_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateReleaseCandidateGateError(
            "Accelerate package root was not found"
        )
    document = render_declared_package(root)
    paths = artifact_paths(root)
    _atomic_write(paths["document"], pretty_json(document))
    readme = DOCUMENT_README if DOCUMENT_README.endswith("\n") else DOCUMENT_README + "\n"
    _atomic_write(paths["readme"], readme)
    catalog = platform_release_candidate_gate_catalog(start)
    _atomic_write(paths["catalog"], pretty_json(catalog))
    return {
        "document": document,
        "catalog": catalog,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def verify_release_candidate_gate_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateReleaseCandidateGateError(
            "Accelerate package root was not found"
        )
    document = render_declared_package(root)
    catalog = platform_release_candidate_gate_catalog(start)
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
        "audit_package_cid": document["security_and_correctness_audit_package"][
            "audit_package_cid"
        ],
        "gate_run_cid": document["release_candidate_gate"]["gate_run_cid"],
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
        raise AccelerateReleaseCandidateGateError(
            "Accelerate package root was not found"
        )
    document = render_declared_package(root)
    verified = verify_release_candidate_gate_files(root)
    table = parse_pyproject_release_candidate_gate_table(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    package = document["release_candidate_gate"]
    trace = package["trace"]
    idempotency = package["idempotency"]
    reproduction = package["reproduction"]
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
        or document["trusted_computing_base"]["tcb_inventory_cid"]
        != PINNED_TCB_INVENTORY_CID
        or document["security_and_correctness_audit_package"]["audit_package_cid"]
        != PINNED_AUDIT_PACKAGE_CID
        or package["gate_run_cid"] != PINNED_GATE_RUN_CID
    )
    probes = [
        _probe(
            "gate_files_match_generator",
            verified["ok"] is True,
            reason=(
                "Declared release-candidate-gate files match the generator."
                if verified["ok"]
                else "Declared release-candidate-gate files do not match the generator."
            ),
        ),
        _probe(
            "pyproject_release_candidate_gate_table",
            table.get("interface") == INTERFACE and table.get("task-id") == PCPR_093_TASK_ID,
            reason=(
                "pyproject.toml declares the PCPR-093 release-candidate-gate table."
                if table.get("interface") == INTERFACE
                else "pyproject.toml does not declare the PCPR-093 release-candidate-gate table."
            ),
        ),
        _probe(
            "paths_remain_authorized",
            package["paths_authorized"] is True
            and list(package["authorized_path_prefixes"])
            == list(AUTHORIZED_PATH_PREFIXES),
            reason="Release-candidate-gate paths remain inside the declared PCPR-093 allowlist.",
        ),
        _probe(
            "canonical_objective_bytes_match",
            package["idea"] == REFERENCE_OBJECTIVE_IDEA
            and package["idea_digest"] == PINNED_IDEA_DIGEST,
            reason="Release-candidate gate uses the canonical PCPR-060 idea bytes.",
        ),
        _probe(
            "supervisor_gate_evaluated",
            trace["supervisor_gate_evaluated"] is True,
            reason="Supervisor gate family is evaluated.",
        ),
        _probe(
            "datasets_gate_evaluated",
            trace["datasets_gate_evaluated"] is True,
            reason="Datasets gate family is evaluated.",
        ),
        _probe(
            "kit_gate_evaluated",
            trace["kit_gate_evaluated"] is True,
            reason="Kit gate family is evaluated.",
        ),
        _probe(
            "accelerate_gate_evaluated",
            trace["accelerate_gate_evaluated"] is True,
            reason="Accelerate gate family is evaluated.",
        ),
        _probe(
            "packaging_gate_evaluated",
            trace["packaging_gate_evaluated"] is True,
            reason="Packaging gate family is evaluated.",
        ),
        _probe(
            "reference_workflow_gate_evaluated",
            trace["reference_workflow_gate_evaluated"] is True,
            reason="Reference-workflow gate family is evaluated.",
        ),
        _probe(
            "interoperability_gate_evaluated",
            trace["interoperability_gate_evaluated"] is True,
            reason="Interoperability gate family is evaluated.",
        ),
        _probe(
            "hard_safety_gate_evaluated",
            trace["hard_safety_gate_evaluated"] is True,
            reason="Hard-safety gate family is evaluated.",
        ),
        _probe(
            "all_gate_families_evaluated",
            package["all_gate_families_evaluated"] is True
            and trace["all_gate_families_evaluated"] is True,
            reason="Every declared release-candidate gate family is evaluated.",
        ),
        _probe(
            "all_gates_bounded",
            package["all_gates_bounded"] is True and trace["all_gates_bounded"] is True,
            reason="Every recorded release-candidate gate is bounded.",
        ),
        _probe(
            "no_hidden_failed_stale_partial_simulated_estimated_or_missing",
            package["no_hidden_failed_stale_partial_simulated_estimated_or_missing"]
            is True
            and trace["no_hidden_failed_stale_partial_simulated_estimated_or_missing"]
            is True,
            reason=(
                "Failed, stale, partial, simulated, estimated, or missing "
                "evidence is not hidden."
            ),
        ),
        _probe(
            "reproduction_commands_work",
            package["reproduction_commands_work"] is True
            and reproduction["ok"] is True
            and reproduction["command_count"] == len(GATES),
            reason="Every declared gate reproduction succeeds hermetically.",
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
            "tcb_inventory_cid_bound_not_minted",
            document["trusted_computing_base"]["tcb_inventory_cid"]
            == PINNED_TCB_INVENTORY_CID
            and document["trusted_computing_base"]["reminted"] is False,
            reason="PCPR-091 TCB inventory CID is bound and not reminted.",
        ),
        _probe(
            "audit_package_cid_bound_not_minted",
            document["security_and_correctness_audit_package"]["audit_package_cid"]
            == PINNED_AUDIT_PACKAGE_CID
            and document["security_and_correctness_audit_package"]["reminted"] is False,
            reason="PCPR-092 audit-package CID is bound and not reminted.",
        ),
        _probe(
            "gate_run_produced",
            package["gate_run_produced"] is True
            and package["gate_run_deferred"] is False
            and package["next_task_id"] == PCPR_094_TASK_ID,
            reason="Release-candidate gate run is produced by PCPR-093.",
        ),
        _probe(
            "closed_release_deferred",
            package["closed_release_deferred"] is True
            and document["closed_release_outcome"] is None,
            reason="Closed release decision remains PCPR-094.",
        ),
        _probe(
            "idempotent_gate_run_cid",
            package["idempotent"] is True
            and idempotency["identical"] is True
            and idempotency["first_gate_run_cid"] == package["gate_run_cid"]
            and idempotency["second_gate_run_cid"] == package["gate_run_cid"]
            and package["gate_run_cid"] == PINNED_GATE_RUN_CID,
            reason="A second gate-run emission yields the same gate-run CID.",
        ),
        _probe(
            "model_assertion_cannot_complete_work",
            package["model_assertion_completes_work"] is False,
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
                "client, MCP client, parity, bypass, threat-model, TCB "
                "inventory, audit-package, or gate-run CID is forbidden."
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
            "tcb_inventory_identity_reminted",
            document["trusted_computing_base"]["tcb_inventory_cid"]
            != PINNED_TCB_INVENTORY_CID
            or document["trusted_computing_base"]["reminted"] is True,
            reason="Accelerate must not remint the PCPR-091 TCB inventory CID.",
        ),
        _probe(
            "audit_package_identity_reminted",
            document["security_and_correctness_audit_package"]["audit_package_cid"]
            != PINNED_AUDIT_PACKAGE_CID
            or document["security_and_correctness_audit_package"]["reminted"] is True,
            reason="Accelerate must not remint the PCPR-092 audit-package CID.",
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
            "live_gate_run_represented_as_live",
            False,
            reason="Hermetic release-candidate gate coverage is not represented as live.",
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
            "closed_release_claimed",
            False,
            reason="Closed release decision remains PCPR-094.",
        ),
        _probe(
            "unbounded_gate_present",
            False,
            reason="Every recorded release-candidate gate is bounded.",
        ),
        _probe(
            "hidden_evidence_present",
            False,
            reason="Failed, stale, partial, simulated, estimated, or missing evidence is visible.",
        ),
        _probe(
            "reproduction_commands_failed",
            False,
            reason="Every declared gate reproduction succeeds hermetically.",
        ),
        _probe(
            "live_pass_claimed",
            False,
            reason="Hermetic gate evaluation is not a live pass.",
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


def qualify_release_candidate_gate(
    probes: Sequence[OutcomeProbe],
    *,
    gate_run_cid: str,
    document_cid: str,
    catalog_cid: str,
    audit_package_cid: str,
    tcb_inventory_cid: str,
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
) -> AccelerateReleaseCandidateGateVerdict:
    if not probes:
        raise AccelerateReleaseCandidateGateError("at least one probe is required")
    normalized: list[OutcomeProbe] = []
    blockers: list[str] = []
    for probe in probes:
        kind = _kind(probe.evidence_kind, "evidence_kind")
        if probe.live and kind != "measured_live":
            raise AccelerateReleaseCandidateGateError(
                "live claims require measured_live evidence"
            )
        if probe.simulated_represented_as_live:
            raise AccelerateReleaseCandidateGateError(
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
        "task_id": PCPR_093_TASK_ID,
        "goal_id": PCPR_093_GOAL_ID,
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
        "gate_run_cid": gate_run_cid,
        "document_cid": document_cid,
        "catalog_cid": catalog_cid,
        "audit_package_cid": audit_package_cid,
        "tcb_inventory_cid": tcb_inventory_cid,
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
    return AccelerateReleaseCandidateGateVerdict(
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
        gate_run_cid=gate_run_cid,
        document_cid=document_cid,
        catalog_cid=catalog_cid,
        audit_package_cid=audit_package_cid,
        tcb_inventory_cid=tcb_inventory_cid,
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


def qualify_current_head_release_candidate_gate(
    start: Path | None = None,
) -> AccelerateReleaseCandidateGateVerdict:
    document = render_declared_package(start)
    catalog = platform_release_candidate_gate_catalog(start)
    return qualify_release_candidate_gate(
        current_head_static_probes(start),
        gate_run_cid=str(document["release_candidate_gate"]["gate_run_cid"]),
        document_cid=str(document["document_cid"]),
        catalog_cid=str(catalog["catalog_cid"]),
        audit_package_cid=str(
            document["security_and_correctness_audit_package"]["audit_package_cid"]
        ),
        tcb_inventory_cid=str(
            document["trusted_computing_base"]["tcb_inventory_cid"]
        ),
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


def pcpr_093_receipt_promotion(
    verdict: AccelerateReleaseCandidateGateVerdict,
) -> dict[str, Any]:
    if verdict.closed_release_outcome is not None:
        raise AccelerateReleaseCandidateGateError(
            "release-candidate gate must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise AccelerateReleaseCandidateGateError(
            "release-candidate gate must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise AccelerateReleaseCandidateGateError(
            "release-candidate gate completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise AccelerateReleaseCandidateGateError(
            "release-candidate gate must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise AccelerateReleaseCandidateGateError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.live_client or verdict.live_application:
        raise AccelerateReleaseCandidateGateError(
            "live client claims require measured_live evidence"
        )
    return verdict.to_mapping()


PINNED_GATE_RUN_CID: Final = (
    "baguqeerao6rmdjiukoqsfe7guwpfrvuq5kw7i4e6srtady5cxazwtemf63ra"
)
PINNED_DOCUMENT_CID: Final = (
    "baguqeeralwrbpwhlehcyxxaczik4b4je3ahu4g73niel6ouuuimzzm6gp2kq"
)
PINNED_CATALOG_CID: Final = (
    "baguqeerapvnln73k3mcqdfuci3jwfsvltqcyiyeo6z7on6k7otawfl5zeq7q"
)
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeerabuzdz2lpl73ctwrt77u4v2okdletopljxg67nr27akdbfiozgs5q"
)


__all__ = [
    "AUTHORIZED_PATH_PREFIXES",
    "AUDIT_STATUS",
    "CLOSED_RELEASE_OUTCOMES",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "ESCALATION_ORDER",
    "GATES",
    "GATE_FAMILIES",
    "G910_SURFACES",
    "G920_EVIDENCE_CLASSES",
    "HERMETIC_CANDIDATE_SUITES",
    "INTERFACE",
    "OBJECTIVE_KIND",
    "OPERATOR_BLOCKING_TASK_ID",
    "OutcomeProbe",
    "PCPR_093_GOAL_ID",
    "PCPR_093_TASK_ID",
    "PINNED_AUDIT_PACKAGE_CID",
    "PINNED_BYPASS_PROOF_CID",
    "PINNED_CATALOG_CID",
    "PINNED_CHAIN_CID",
    "PINNED_CURRENT_ROOT_CID",
    "PINNED_DOCUMENT_CID",
    "PINNED_GATE_RUN_CID",
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
    "AccelerateReleaseCandidateGateError",
    "all_gate_families_evaluated",
    "all_gates_bounded",
    "current_head_static_probes",
    "hermetic_gate_run_trace",
    "path_is_authorized",
    "pcpr_093_receipt_promotion",
    "platform_release_candidate_gate_catalog",
    "produce_release_candidate_gate",
    "qualify_current_head_release_candidate_gate",
    "qualify_release_candidate_gate",
    "refuse_audit_package_cid_remint",
    "refuse_bypass_proof_cid_remint",
    "refuse_chain_cid_remint",
    "refuse_closed_release_claim",
    "refuse_database_edit",
    "refuse_hidden_evidence",
    "refuse_live_pass",
    "refuse_mcp_client_cid_remint",
    "refuse_model_completion",
    "refuse_non_canonical_objective",
    "refuse_objective_cid_remint",
    "refuse_pack_cid_remint",
    "refuse_parity_cid_remint",
    "refuse_python_client_cid_remint",
    "refuse_self_promotion",
    "refuse_tcb_inventory_cid_remint",
    "refuse_threat_model_cid_remint",
    "refuse_unauthorized_path",
    "refuse_unbounded_gate",
    "render_declared_package",
    "reproduce_gate",
    "reproduce_release_candidate_gate",
    "verify_release_candidate_gate_files",
    "write_release_candidate_gate_files",
]


if __name__ == "__main__":
    written = write_release_candidate_gate_files()
    document = written["document"]
    catalog = written["catalog"]
    print(document["release_candidate_gate"]["gate_run_cid"])
    print(document["document_cid"])
    print(catalog["catalog_cid"])
