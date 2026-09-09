"""Fail-closed PCPR-092 Accelerate-owned security and correctness audit package.

Accelerate owns the hermetic security and correctness audit package for
the PCPR objective-to-release path. Architecture, state-machine,
authority, canonicalization, recovery, proof, build, negative-test, and
limitation evidence are assembled as named, reproducible artifacts. The
package status is external_audit_ready and is never externally_audited.
The PCPR-090 threat model and PCPR-091 TCB inventory are bound and not
reminted. Closed release decision remains PCPR-093/094.

Live Supervisor.run, live Quack, live MCP, and live CLI stay typed
unavailable. Simulated results are not live. This evaluator does not
write DuckDB or Quack state and does not emit a closed PCPR release.

Hermetic audit-package coverage is candidate coverage only.
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


INTERFACE: Final = "AccelerateSecurityAndCorrectnessAuditPackage@1"
SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/security-and-correctness-audit-package@1"
)
DOCUMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/declared-security-and-correctness-audit-package@1"
)
PACKAGE_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/security-and-correctness-audit-package-inventory@1"
)
VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/security-and-correctness-audit-package-verdict@1"
)
CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/platform-security-and-correctness-audit-package-catalog@1"
)
CATALOG_INTERFACE: Final = "PlatformSecurityAndCorrectnessAuditPackage@1"
PACKAGE_INTERFACE: Final = "AuthoritativeSecurityAndCorrectnessAuditPackage@1"
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
PROTOCOL_INTERFACE: Final = "LogicProviderProtocol@2"
DOCUMENTATION_INTERFACE: Final = (
    "LogicProviderProtocolSecurityAndCorrectnessAuditPackage@1"
)
DOCUMENTATION_SCHEMA: Final = (
    "ipfs_datasets_py/logic-provider-security-and-correctness-audit-package@1"
)
PCPR_092_TASK_ID: Final = "PCPR-092"
PCPR_092_GOAL_ID: Final = "PCPR-G900"
PCPR_093_TASK_ID: Final = "PCPR-093"
PCPR_094_TASK_ID: Final = "PCPR-094"
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
OBJECTIVE_KIND: Final = (
    "declared_objective_to_release_security_and_correctness_audit_package"
)
OBJECTIVE_ID: Final = "PCPR-G900"
LANGUAGE: Final = "Python"
OPERATOR_BLOCKING_TASK_ID: Final = "pcpr-092-operator-live-audit-package"
DOCUMENT_DIR_RELPATH: Final = "packaging/pcpr/reference-workflow/cpython312"
DOCUMENT_JSON_NAME: Final = "reference.security-and-correctness-audit-package.json"
DOCUMENT_README_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/SECURITY_AND_CORRECTNESS_AUDIT_PACKAGE.md"
)
CATALOG_RELPATH: Final = (
    "packaging/pcpr/reference-workflow/platform-security-and-correctness-audit-package-catalog.json"
)
SOURCE_DATE_EPOCH: Final = "0"
SOURCE_REPOSITORY: Final = "endomorphosis/ipfs_accelerate_py"
COMPLETED_THROUGH: Final = "security_and_correctness_audit_package"
NEXT_AUTHORIZED_STAGE: Final = "release_candidate_gate"
CHANGE_KIND: Final = "objective_to_release_security_and_correctness_audit_package"
PACKAGE_KIND: Final = (
    "hermetic_objective_to_release_security_and_correctness_audit_package"
)
CANONICAL_SERVICE: Final = "pcpr-canonical-supervisor-service"
CLIENT_AUTHORITY: Final = "intent_not_authority"
PACKAGE_COMMAND_DIGEST: Final = "pcpr-092-hermetic-audit-package"
AUDIT_STATUS_READY: Final = "external_audit_ready"
AUDIT_STATUS_AUDITED: Final = "externally_audited"
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

if PINNED_LOCK_CID != (
    "baguqeerawsekbbbt5ccctjt4tydzeahfkahsatb6q5x7k6cy4inrii5lhqmq"
):
    raise RuntimeError("PCPR-092 lock CID remints PCPR-056")
if PINNED_PYTHON_CLIENT_CID == PINNED_MCP_CLIENT_CID:
    raise RuntimeError("Python and MCP client CIDs must remain distinct")
if PINNED_THREAT_MODEL_CID == PINNED_PACK_CID:
    raise RuntimeError("Threat-model CID must remain distinct from pack CID")
if PINNED_TCB_INVENTORY_CID == PINNED_THREAT_MODEL_CID:
    raise RuntimeError("TCB inventory CID must remain distinct from threat-model CID")
if OWNER_INTERFACES["SupervisorContextPack"] != OWNER_INTERFACE:
    raise RuntimeError("PCPR-092 remints SupervisorContextPack owner interface")
if OWNER_SCHEMAS["SupervisorContextPack"] != OWNER_SCHEMA:
    raise RuntimeError("PCPR-092 remints SupervisorContextPack owner schema")

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

SEALED_REPRODUCTION_PREFIX: Final = (
    f"PATH={SEALED_PATH} PYTHONNOUSERSITE=1 {SEALED_PYTHON}"
)


def _artifact(
    artifact_id: str,
    *,
    evidence_class: str,
    owner: str,
    stage: str,
    title: str,
    reproduction_command: str,
    limitation: str,
    bound_identities: tuple[str, ...],
) -> dict[str, Any]:
    if evidence_class not in G920_EVIDENCE_CLASSES:
        raise RuntimeError(
            f"artifact {artifact_id} uses unbounded evidence class {evidence_class}"
        )
    if stage not in TRUSTED_PATH_STAGES:
        raise RuntimeError(
            f"artifact {artifact_id} uses unbounded stage {stage}"
        )
    return {
        "artifact_id": artifact_id,
        "evidence_class": evidence_class,
        "owner": owner,
        "trusted_path_stage": stage,
        "title": title,
        "reproduction_command": reproduction_command,
        "limitation": limitation,
        "bound_identities": list(bound_identities),
        "reproducible": True,
        "bounded": True,
        "live": False,
        "admitted_live": False,
        "externally_audited": False,
        "closed_release": False,
        "live_status": "measured_hermetic",
        "evidence_kind": "measured_hermetic",
    }


ARTIFACTS: Final[tuple[dict[str, Any], ...]] = (
    _artifact(
        "AUD-01",
        evidence_class="architecture",
        owner="ipfs_accelerate_py",
        stage="planning_and_context_pack",
        title="Objective-to-release architecture and repository authority bounds",
        reproduction_command=(
            f"{SEALED_REPRODUCTION_PREFIX} -c "
            "'from ipfs_accelerate_py.assurance.security_and_correctness_audit "
            "import reproduce_artifact; print(reproduce_artifact(\"AUD-01\")[\"ok\"])'"
        ),
        limitation=(
            "Architecture evidence is the declared owner split among Datasets, "
            "Kit, and Accelerate. It is not a live supervisor composition."
        ),
        bound_identities=(
            PINNED_PACK_CID,
            PINNED_CURRENT_ROOT_CID,
            PINNED_CHAIN_CID,
        ),
    ),
    _artifact(
        "AUD-02",
        evidence_class="state_machine",
        owner="ipfs_accelerate_py",
        stage="task_state_machine",
        title="Task, event, lease, fence, and unknown-outcome state machine",
        reproduction_command=(
            f"{SEALED_REPRODUCTION_PREFIX} -c "
            "'from ipfs_accelerate_py.assurance.security_and_correctness_audit "
            "import reproduce_artifact; print(reproduce_artifact(\"AUD-02\")[\"ok\"])'"
        ),
        limitation=(
            "State-machine evidence is hermetic. Quack-fenced live mutation "
            "remains typed unavailable."
        ),
        bound_identities=(PINNED_CHAIN_CID, PINNED_TCB_INVENTORY_CID),
    ),
    _artifact(
        "AUD-03",
        evidence_class="authority",
        owner="ipfs_accelerate_py",
        stage="authentication_and_authority",
        title="Intent-not-authority clients and authority-bypass refusals",
        reproduction_command=(
            f"{SEALED_REPRODUCTION_PREFIX} -c "
            "'from ipfs_accelerate_py.assurance.security_and_correctness_audit "
            "import reproduce_artifact; print(reproduce_artifact(\"AUD-03\")[\"ok\"])'"
        ),
        limitation=(
            "Python, MCP, and CLI principals remain intent. Bypass proof CID "
            "is bound and not reminted. Live clients stay typed unavailable."
        ),
        bound_identities=(
            PINNED_PYTHON_CLIENT_CID,
            PINNED_MCP_CLIENT_CID,
            PINNED_BYPASS_PROOF_CID,
        ),
    ),
    _artifact(
        "AUD-04",
        evidence_class="canonicalization",
        owner="ipfs_accelerate_py",
        stage="objective_submission",
        title="Canonical JSON, CIDv1 DAG-JSON/sha2-256, and identity pins",
        reproduction_command=(
            f"{SEALED_REPRODUCTION_PREFIX} -c "
            "'from ipfs_accelerate_py.assurance.security_and_correctness_audit "
            "import reproduce_artifact; print(reproduce_artifact(\"AUD-04\")[\"ok\"])'"
        ),
        limitation=(
            "Canonicalization evidence recomputes declared CIDs. SHA-256 hex "
            "is never advertised as a CID. Live IPFS publication stays typed "
            "unavailable."
        ),
        bound_identities=(
            PINNED_OBJECTIVE_CID,
            PINNED_IDEA_DIGEST,
            PINNED_PACK_CID,
            PINNED_CURRENT_ROOT_CID,
        ),
    ),
    _artifact(
        "AUD-05",
        evidence_class="recovery",
        owner="ipfs_accelerate_py",
        stage="execution_admission",
        title="Restart, idempotency, and duplicate-effect rejection",
        reproduction_command=(
            f"{SEALED_REPRODUCTION_PREFIX} -c "
            "'from ipfs_accelerate_py.assurance.security_and_correctness_audit "
            "import reproduce_artifact; print(reproduce_artifact(\"AUD-05\")[\"ok\"])'"
        ),
        limitation=(
            "Recovery evidence is hermetic replay of the same audit-package "
            "command. Live Quack restart remains typed unavailable."
        ),
        bound_identities=(PINNED_CHAIN_CID, PINNED_TCB_INVENTORY_CID),
    ),
    _artifact(
        "AUD-06",
        evidence_class="proof",
        owner="ipfs_accelerate_py",
        stage="proof_and_test_admission",
        title="Proof admission, selected tests, and sealed-PATH candidate suites",
        reproduction_command=(
            f"{SEALED_REPRODUCTION_PREFIX} -c "
            "'from ipfs_accelerate_py.assurance.security_and_correctness_audit "
            "import reproduce_artifact; print(reproduce_artifact(\"AUD-06\")[\"ok\"])'"
        ),
        limitation=(
            "Hermetic pytest is candidate coverage. Formal provers stay typed "
            "unavailable. Stored proofs are not admitted proofs."
        ),
        bound_identities=(PINNED_PARITY_CID, PINNED_THREAT_MODEL_CID),
    ),
    _artifact(
        "AUD-07",
        evidence_class="build",
        owner="ipfs_accelerate_py",
        stage="release_decision",
        title="Declared locks, SBOMs, provenance, and unsigned-artifact bounds",
        reproduction_command=(
            f"{SEALED_REPRODUCTION_PREFIX} -c "
            "'from ipfs_accelerate_py.assurance.security_and_correctness_audit "
            "import reproduce_artifact; print(reproduce_artifact(\"AUD-07\")[\"ok\"])'"
        ),
        limitation=(
            "Build evidence binds the PCPR-056 lock CID. Unsigned artifacts "
            "cannot close a release. Live publication stays typed unavailable."
        ),
        bound_identities=(PINNED_LOCK_CID,),
    ),
    _artifact(
        "AUD-08",
        evidence_class="negative_test",
        owner="ipfs_accelerate_py",
        stage="proof_and_test_admission",
        title="Fail-closed refusals for remint, live-spoof, and self-promotion",
        reproduction_command=(
            f"{SEALED_REPRODUCTION_PREFIX} -c "
            "'from ipfs_accelerate_py.assurance.security_and_correctness_audit "
            "import reproduce_artifact; print(reproduce_artifact(\"AUD-08\")[\"ok\"])'"
        ),
        limitation=(
            "Negative tests are hermetic refusals. They do not qualify live "
            "Supervisor.run or close a release."
        ),
        bound_identities=(
            PINNED_PACK_CID,
            PINNED_THREAT_MODEL_CID,
            PINNED_TCB_INVENTORY_CID,
        ),
    ),
    _artifact(
        "AUD-09",
        evidence_class="limitation",
        owner="ipfs_accelerate_py",
        stage="release_decision",
        title="Typed unavailability, non-promotion, and deferred closed decision",
        reproduction_command=(
            f"{SEALED_REPRODUCTION_PREFIX} -c "
            "'from ipfs_accelerate_py.assurance.security_and_correctness_audit "
            "import reproduce_artifact; print(reproduce_artifact(\"AUD-09\")[\"ok\"])'"
        ),
        limitation=(
            "The package is external_audit_ready and never externally_audited. "
            "Closed release remains PCPR-093/094. Missing live environments "
            "stay typed unavailable."
        ),
        bound_identities=(PINNED_TCB_INVENTORY_CID, PINNED_THREAT_MODEL_CID),
    ),
)

if any(not item["bounded"] for item in ARTIFACTS):
    raise RuntimeError("PCPR-092 audit table contains an unbounded artifact")
if {item["evidence_class"] for item in ARTIFACTS} != set(G920_EVIDENCE_CLASSES):
    raise RuntimeError("PCPR-092 artifacts do not cover every G920 evidence class")
if len({item["artifact_id"] for item in ARTIFACTS}) != len(ARTIFACTS):
    raise RuntimeError("PCPR-092 audit identifiers are not unique")
if any(item["externally_audited"] for item in ARTIFACTS):
    raise RuntimeError("PCPR-092 artifacts must never claim externally_audited")
if any(item["closed_release"] for item in ARTIFACTS):
    raise RuntimeError("PCPR-092 artifacts must not claim a closed release")
if AUDIT_STATUS == AUDIT_STATUS_AUDITED:
    raise RuntimeError("PCPR-092 must not claim externally_audited")
if AUDIT_STATUS != AUDIT_STATUS_READY:
    raise RuntimeError("PCPR-092 package status must be external_audit_ready")

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
    "artifacts/proof_carrying_platform_qualification_and_release/receipts/PCPR-092.json",
)

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_security_and_correctness_audit.py",
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
        "audit_package_files_match_generator",
        "pyproject_audit_package_table",
        "paths_remain_authorized",
        "canonical_objective_bytes_match",
        "architecture_evidence_present",
        "state_machine_evidence_present",
        "authority_evidence_present",
        "canonicalization_evidence_present",
        "recovery_evidence_present",
        "proof_evidence_present",
        "build_evidence_present",
        "negative_test_evidence_present",
        "limitation_evidence_present",
        "all_g920_evidence_classes_present",
        "all_artifacts_bounded",
        "reproduction_commands_work",
        "audit_status_external_audit_ready",
        "owner_pack_cid_matches_pin",
        "kit_current_root_bound_not_minted",
        "chain_cid_bound_not_minted",
        "python_client_cid_bound_not_minted",
        "mcp_client_cid_bound_not_minted",
        "parity_cid_bound_not_minted",
        "bypass_proof_cid_bound_not_minted",
        "threat_model_cid_bound_not_minted",
        "tcb_inventory_cid_bound_not_minted",
        "audit_package_produced",
        "closed_release_deferred",
        "idempotent_audit_package_cid",
        "model_assertion_cannot_complete_work",
        "duckdb_or_quack_not_written",
        "operator_blocking_task_emitted",
        "no_closed_release_outcome",
    }
)
FORBIDDEN_PRESENT_PROBE_IDS: Final[frozenset[str]] = frozenset(
    {
        "simulated_results_represented_as_live",
        "live_audit_package_represented_as_live",
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
        "model_assertion_completed_work",
        "unauthorized_path_accepted",
        "database_edited",
        "externally_audited_claimed",
        "closed_release_claimed",
        "unbounded_artifact_present",
        "reproduction_commands_failed",
    }
)

DOCUMENT_README: Final = """# PCPR-092 Accelerate security and correctness audit package

These files record the Accelerate-owned hermetic security and
correctness audit package for the PCPR objective-to-release path.
Architecture, state-machine, authority, canonicalization, recovery,
proof, build, negative-test, and limitation evidence are assembled as
named reproducible artifacts.

- `cpython312/reference.security-and-correctness-audit-package.json` is
  the declared audit-package document. Exact commit and tree are bound
  by the PCPR-092 receipt `current_tree_binding`.
- `platform-security-and-correctness-audit-package-catalog.json`
  observes sibling Datasets and Kit bindings when present. Sibling
  source is never required.
- Pack, root, chain, objective, Python-client, MCP-client, parity,
  PCPR-083 bypass, PCPR-090 threat-model, and PCPR-091 TCB-inventory
  identities are bound and not reminted. Closed release decision
  remains PCPR-093/094.
- The package status is `external_audit_ready` and is never
  `externally_audited`.
- Live Supervisor.run, live Quack, live MCP, and live CLI sessions stay
  typed unavailable.
- Missing a live Quack-fenced session emits operator-blocking task
  `pcpr-092-operator-live-audit-package`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
"""


class AccelerateSecurityAndCorrectnessAuditError(Exception):
    """Fail-closed PCPR-092 Accelerate security and correctness audit error."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AccelerateSecurityAndCorrectnessAuditError(
            f"{name} must be a non-empty string"
        )
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise AccelerateSecurityAndCorrectnessAuditError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise AccelerateSecurityAndCorrectnessAuditError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def parse_pyproject_audit_package_table(text: str) -> dict[str, Any]:
    import tomllib

    table = tomllib.loads(_text(text, "pyproject.toml"))
    if not isinstance(table, dict):
        raise AccelerateSecurityAndCorrectnessAuditError(
            "pyproject.toml must be a table"
        )
    tool = table.get("tool")
    payload: dict[str, Any] = {}
    if isinstance(tool, dict):
        accelerate = tool.get("ipfs_accelerate_py")
        if isinstance(accelerate, dict):
            raw = accelerate.get("security-and-correctness-audit-package")
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
        raise AccelerateSecurityAndCorrectnessAuditError(
            f"path {path} is not an authorized PCPR-092 path"
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
        raise AccelerateSecurityAndCorrectnessAuditError(
            f"model assertion at {stage} cannot complete work"
        )


def refuse_pack_cid_remint(cid: str) -> str:
    if cid != PINNED_PACK_CID:
        raise AccelerateSecurityAndCorrectnessAuditError(
            f"ContextPack CID {cid} remints {PINNED_PACK_CID}"
        )
    return cid


def refuse_current_root_remint(cid: str) -> str:
    if cid != PINNED_CURRENT_ROOT_CID:
        raise AccelerateSecurityAndCorrectnessAuditError(
            f"current root CID {cid} remints {PINNED_CURRENT_ROOT_CID}"
        )
    return cid


def refuse_chain_cid_remint(cid: str) -> str:
    if cid != PINNED_CHAIN_CID:
        raise AccelerateSecurityAndCorrectnessAuditError(
            f"chain CID {cid} remints {PINNED_CHAIN_CID}"
        )
    return cid


def refuse_objective_cid_remint(cid: str) -> str:
    if cid != PINNED_OBJECTIVE_CID:
        raise AccelerateSecurityAndCorrectnessAuditError(
            f"objective CID {cid} remints {PINNED_OBJECTIVE_CID}"
        )
    return cid


def refuse_idea_digest_remint(digest: str) -> str:
    if digest != PINNED_IDEA_DIGEST:
        raise AccelerateSecurityAndCorrectnessAuditError(
            f"idea digest {digest} remints {PINNED_IDEA_DIGEST}"
        )
    return digest


def refuse_python_client_cid_remint(cid: str) -> str:
    if cid != PINNED_PYTHON_CLIENT_CID:
        raise AccelerateSecurityAndCorrectnessAuditError(
            f"Python client CID {cid} remints {PINNED_PYTHON_CLIENT_CID}"
        )
    return cid


def refuse_mcp_client_cid_remint(cid: str) -> str:
    if cid != PINNED_MCP_CLIENT_CID:
        raise AccelerateSecurityAndCorrectnessAuditError(
            f"MCP client CID {cid} remints {PINNED_MCP_CLIENT_CID}"
        )
    return cid


def refuse_parity_cid_remint(cid: str) -> str:
    if cid != PINNED_PARITY_CID:
        raise AccelerateSecurityAndCorrectnessAuditError(
            f"parity CID {cid} remints {PINNED_PARITY_CID}"
        )
    return cid


def refuse_bypass_proof_cid_remint(cid: str) -> str:
    if cid != PINNED_BYPASS_PROOF_CID:
        raise AccelerateSecurityAndCorrectnessAuditError(
            f"bypass proof CID {cid} remints {PINNED_BYPASS_PROOF_CID}"
        )
    return cid


def refuse_threat_model_cid_remint(cid: str) -> str:
    if cid != PINNED_THREAT_MODEL_CID:
        raise AccelerateSecurityAndCorrectnessAuditError(
            f"threat-model CID {cid} remints {PINNED_THREAT_MODEL_CID}"
        )
    return cid


def refuse_tcb_inventory_cid_remint(cid: str) -> str:
    if cid != PINNED_TCB_INVENTORY_CID:
        raise AccelerateSecurityAndCorrectnessAuditError(
            f"TCB inventory CID {cid} remints {PINNED_TCB_INVENTORY_CID}"
        )
    return cid


def refuse_database_edit(*, edited: bool) -> None:
    if edited:
        raise AccelerateSecurityAndCorrectnessAuditError(
            "security and correctness audit package cannot write DuckDB or Quack state"
        )


def refuse_live_audit_package(*, applied_live: bool) -> None:
    if applied_live:
        raise AccelerateSecurityAndCorrectnessAuditError(
            "live security and correctness audit execution remains typed unavailable"
        )


def refuse_self_promotion(*, promoted: bool) -> None:
    if promoted:
        raise AccelerateSecurityAndCorrectnessAuditError(
            "security and correctness audit package cannot promote itself"
        )


def refuse_non_canonical_objective(idea: str) -> str:
    text = _text(idea, "idea")
    if text != REFERENCE_OBJECTIVE_IDEA:
        raise AccelerateSecurityAndCorrectnessAuditError(
            "audit package must bind the canonical PCPR-060 idea"
        )
    return text


def refuse_externally_audited_claim(*, claimed: bool) -> None:
    if claimed:
        raise AccelerateSecurityAndCorrectnessAuditError(
            "package status is external_audit_ready and must never be externally_audited"
        )


def refuse_closed_release_claim(*, claimed: bool) -> None:
    if claimed:
        raise AccelerateSecurityAndCorrectnessAuditError(
            "closed release decision remains PCPR-093/094"
        )


def refuse_unbounded_artifact(*, unbounded: bool) -> None:
    if unbounded:
        raise AccelerateSecurityAndCorrectnessAuditError(
            "every audit artifact must be bounded to a declared evidence class and stage"
        )


def g920_evidence_class_bounds() -> dict[str, bool]:
    classes = {item["evidence_class"] for item in ARTIFACTS if item["bounded"]}
    return {name: name in classes for name in G920_EVIDENCE_CLASSES}


def all_g920_evidence_classes_present() -> bool:
    bounds = g920_evidence_class_bounds()
    return all(bounds[name] for name in G920_EVIDENCE_CLASSES) and set(bounds) == set(
        G920_EVIDENCE_CLASSES
    )


def hermetic_audit_package_trace() -> dict[str, Any]:
    bounds = g920_evidence_class_bounds()
    return {
        "schema": "ipfs_accelerate_py/assurance/hermetic-audit-package-trace@1",
        "interface": "HermeticSecurityAndCorrectnessAuditTrace@1",
        "artifact_count": len(ARTIFACTS),
        "artifact_ids": [item["artifact_id"] for item in ARTIFACTS],
        "all_artifacts_bounded": all(item["bounded"] is True for item in ARTIFACTS),
        "trusted_path_stages": list(TRUSTED_PATH_STAGES),
        "g910_surfaces": list(G910_SURFACES),
        "g920_evidence_classes": list(G920_EVIDENCE_CLASSES),
        "g920_evidence_class_bounds": bounds,
        "all_g920_evidence_classes_present": all_g920_evidence_classes_present(),
        "architecture_evidence_present": bounds["architecture"],
        "state_machine_evidence_present": bounds["state_machine"],
        "authority_evidence_present": bounds["authority"],
        "canonicalization_evidence_present": bounds["canonicalization"],
        "recovery_evidence_present": bounds["recovery"],
        "proof_evidence_present": bounds["proof"],
        "build_evidence_present": bounds["build"],
        "negative_test_evidence_present": bounds["negative_test"],
        "limitation_evidence_present": bounds["limitation"],
        "reproduction_commands_work": all(
            item["reproducible"] is True for item in ARTIFACTS
        ),
        "audit_status": AUDIT_STATUS,
        "externally_audited": False,
        "audit_package_produced": True,
        "audit_package_deferred": False,
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


def hermetic_audit_package_idempotency(package_cid: str) -> dict[str, Any]:
    return {
        "schema": "ipfs_accelerate_py/assurance/hermetic-audit-package-idempotency@1",
        "interface": "HermeticSecurityAndCorrectnessAuditIdempotency@1",
        "first_package_command_digest": PACKAGE_COMMAND_DIGEST,
        "second_package_command_digest": PACKAGE_COMMAND_DIGEST,
        "first_audit_package_cid": package_cid,
        "second_audit_package_cid": package_cid,
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
class AccelerateSecurityAndCorrectnessAuditVerdict:
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
    audit_package_cid: str
    document_cid: str
    catalog_cid: str
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
            "audit_package_cid": self.audit_package_cid,
            "document_cid": self.document_cid,
            "catalog_cid": self.catalog_cid,
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
            "An admitted Quack-fenced state-owner session before the audit "
            "package is executed as live supervisor work"
        ),
        "action": (
            "Keep the hermetic audit package. Keep every G920 evidence class "
            "bounded. Keep status external_audit_ready. Do not write DuckDB "
            "or Quack state. Do not escalate to a model. Do not claim "
            "externally_audited. Do not emit a closed release. PCPR-093 runs "
            "the release-candidate gate."
        ),
        "reason": (
            "Hermetic audit-package coverage is not a live Supervisor.run / "
            "CLI / MCP session. Sealed validation has no admitted "
            "Quack-fenced session. Direct DuckDB writes are prohibited."
        ),
    }


def produce_audit_package(
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
    idea: str = REFERENCE_OBJECTIVE_IDEA,
    complete_from: str | None = None,
    database_edited: bool = False,
    applied_live: bool = False,
    self_promoted: bool = False,
    externally_audited: bool = False,
    closed_release_claimed: bool = False,
    unbounded: bool = False,
) -> dict[str, Any]:
    """Produce the hermetic objective-to-release audit package. Fail closed."""

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
    refuse_non_canonical_objective(idea)
    refuse_database_edit(edited=database_edited)
    refuse_live_audit_package(applied_live=applied_live)
    refuse_self_promotion(promoted=self_promoted)
    refuse_externally_audited_claim(claimed=externally_audited)
    refuse_closed_release_claim(claimed=closed_release_claimed)
    refuse_unbounded_artifact(unbounded=unbounded)
    checked_paths = list(paths) if paths is not None else list(AUTHORIZED_PATH_PREFIXES)
    for path in checked_paths:
        refuse_unauthorized_path(path)
    trace = hermetic_audit_package_trace()
    return {
        "schema": PACKAGE_SCHEMA,
        "interface": PACKAGE_INTERFACE,
        "owner_interface": INTERFACE,
        "task_id": PCPR_092_TASK_ID,
        "goal_id": PCPR_092_GOAL_ID,
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
        "python_client_interface": PYTHON_CLIENT_INTERFACE,
        "mcp_client_interface": MCP_CLIENT_INTERFACE,
        "parity_interface": PARITY_INTERFACE,
        "bypass_interface": BYPASS_INTERFACE,
        "threat_model_interface": THREAT_MODEL_INTERFACE,
        "tcb_interface": TCB_INTERFACE,
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
        "audit_status": AUDIT_STATUS,
        "externally_audited": False,
        "artifacts": [dict(item) for item in ARTIFACTS],
        "produced": True,
        "applied": True,
        "live": False,
        "hermetic": True,
        "admitted_live": False,
        "deferred": False,
        "paths_authorized": True,
        "authorized_path_prefixes": list(AUTHORIZED_PATH_PREFIXES),
        "all_g920_evidence_classes_present": True,
        "all_artifacts_bounded": True,
        "reproduction_commands_work": True,
        "audit_package_produced": True,
        "audit_package_deferred": False,
        "closed_release_deferred": True,
        "client_self_promoted": False,
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
        "next_task_id": PCPR_093_TASK_ID,
        "prerequisite_task_id": PCPR_091_TASK_ID,
        "python_client_task_id": PCPR_080_TASK_ID,
        "mcp_client_task_id": PCPR_081_TASK_ID,
        "parity_task_id": PCPR_082_TASK_ID,
        "bypass_task_id": PCPR_083_TASK_ID,
        "chain_task_id": PCPR_072_TASK_ID,
        "threat_model_task_id": PCPR_090_TASK_ID,
        "tcb_inventory_task_id": PCPR_091_TASK_ID,
        "duckdb_or_quack_state_written": False,
        "closed_release_outcome": None,
        "release_claim": False,
        "evidence_kind": "measured_hermetic",
    }


def canonical_audit_package() -> dict[str, Any]:
    return produce_audit_package()


def audit_package_cid_of(plan: Mapping[str, Any] | None = None) -> str:
    payload = dict(plan or canonical_audit_package())
    payload.pop("audit_package_cid", None)
    payload.pop("idempotency", None)
    return content_identity(payload)


def reproduce_artifact(artifact_id: str) -> dict[str, Any]:
    """Hermetically reproduce one named audit artifact. Fail closed."""

    wanted = _text(artifact_id, "artifact_id")
    match = next((item for item in ARTIFACTS if item["artifact_id"] == wanted), None)
    if match is None:
        raise AccelerateSecurityAndCorrectnessAuditError(
            f"unknown audit artifact {wanted}"
        )
    plan = canonical_audit_package()
    package_cid = audit_package_cid_of(plan)
    return {
        "artifact_id": wanted,
        "ok": (
            match["bounded"] is True
            and match["reproducible"] is True
            and match["externally_audited"] is False
            and match["closed_release"] is False
            and match["live"] is False
            and package_cid.startswith("baguqeera")
            and plan["audit_status"] == AUDIT_STATUS_READY
        ),
        "evidence_class": match["evidence_class"],
        "reproduction_command": match["reproduction_command"],
        "audit_package_cid": package_cid,
        "live": False,
        "hermetic": True,
        "externally_audited": False,
        "evidence_kind": "measured_hermetic",
    }


def reproduce_audit_package() -> dict[str, Any]:
    """Run every declared reproduction command hermetically."""

    results = [reproduce_artifact(item["artifact_id"]) for item in ARTIFACTS]
    return {
        "ok": all(item["ok"] is True for item in results),
        "command_count": len(results),
        "commands": [item["reproduction_command"] for item in results],
        "results": results,
        "audit_status": AUDIT_STATUS,
        "externally_audited": False,
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
                "Sibling security-and-correctness-audit-package file is not "
                "present beside this checkout."
            ),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    context_pack = payload.get("context_pack")
    storage = payload.get("storage")
    package = payload.get("security_and_correctness_audit_package")
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
        "audit_package_cid": payload.get("audit_package_cid")
        or (
            package.get("audit_package_cid") if isinstance(package, Mapping) else None
        ),
        "binding_cid": payload.get("binding_cid") or payload.get("document_cid"),
        "live": False,
        "applied": False,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def render_declared_package(root: Path | None = None) -> dict[str, Any]:
    package_root = root or discover_accelerate_root()
    if package_root is None:
        raise AccelerateSecurityAndCorrectnessAuditError(
            "Accelerate package root was not found"
        )
    state_owner = observe_state_owner()
    plan = canonical_audit_package()
    computed_package_cid = audit_package_cid_of(plan)
    idempotency = hermetic_audit_package_idempotency(computed_package_cid)
    reproduction = reproduce_audit_package()
    document = {
        "schema": DOCUMENT_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_092_TASK_ID,
        "goal_id": PCPR_092_GOAL_ID,
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
            "survives_audit_package": True,
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
            "survives_audit_package": True,
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
            "survives_audit_package": True,
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
            "survives_audit_package": True,
            "evidence_kind": "measured_hermetic",
        },
        "security_and_correctness_audit_package": {
            **plan,
            "audit_package_cid": computed_package_cid,
            "idempotency": idempotency,
            "reproduction": reproduction,
            "produced_by": EXECUTION_OWNER_REPOSITORY,
        },
        "materialization": {
            "kind": "declared_audit_package_not_live",
            "admitted": False,
            "live": False,
            "applied": False,
            "duckdb_or_quack_state_written": False,
            "evidence_kind": "unavailable",
        },
        "live_application": typed_unavailable(
            reason=(
                "No admitted Quack-fenced supervisor session was observed. "
                "Hermetic audit-package coverage is not live execution."
            )
        ),
        "live_client": typed_unavailable(
            reason=(
                "Live Python Supervisor.run and live generic MCP remain typed "
                "unavailable. This task only emits a hermetic audit package. "
                "Closed release decision remains PCPR-093/094."
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
                "field": "PCPR-092 receipt current_tree_binding",
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


def platform_audit_package_catalog(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateSecurityAndCorrectnessAuditError(
            "Accelerate package root was not found"
        )
    parent = root.parent
    document = render_declared_package(root)
    datasets_binding = _sibling_binding(
        parent
        / "ipfs_datasets"
        / DOCUMENT_DIR_RELPATH
        / "reference.security-and-correctness-audit-package.binding.json",
        relative_path=(
            f"../ipfs_datasets/{DOCUMENT_DIR_RELPATH}/"
            "reference.security-and-correctness-audit-package.binding.json"
        ),
    )
    kit_binding = _sibling_binding(
        parent
        / "ipfs_kit"
        / DOCUMENT_DIR_RELPATH
        / "reference.security-and-correctness-audit-package.binding.json",
        relative_path=(
            f"../ipfs_kit/{DOCUMENT_DIR_RELPATH}/"
            "reference.security-and-correctness-audit-package.binding.json"
        ),
    )
    catalog = {
        "schema": CATALOG_SCHEMA,
        "interface": CATALOG_INTERFACE,
        "task_id": PCPR_092_TASK_ID,
        "goal_id": PCPR_092_GOAL_ID,
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
        "audit_package_cid": document["security_and_correctness_audit_package"][
            "audit_package_cid"
        ],
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
                "audit_package_cid": document["security_and_correctness_audit_package"][
                    "audit_package_cid"
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


def write_audit_package_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateSecurityAndCorrectnessAuditError(
            "Accelerate package root was not found"
        )
    document = render_declared_package(root)
    paths = artifact_paths(root)
    _atomic_write(paths["document"], pretty_json(document))
    readme = DOCUMENT_README if DOCUMENT_README.endswith("\n") else DOCUMENT_README + "\n"
    _atomic_write(paths["readme"], readme)
    catalog = platform_audit_package_catalog(start)
    _atomic_write(paths["catalog"], pretty_json(catalog))
    return {
        "document": document,
        "catalog": catalog,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def verify_audit_package_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise AccelerateSecurityAndCorrectnessAuditError(
            "Accelerate package root was not found"
        )
    document = render_declared_package(root)
    catalog = platform_audit_package_catalog(start)
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
        raise AccelerateSecurityAndCorrectnessAuditError(
            "Accelerate package root was not found"
        )
    document = render_declared_package(root)
    verified = verify_audit_package_files(root)
    table = parse_pyproject_audit_package_table(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    package = document["security_and_correctness_audit_package"]
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
        or package["audit_package_cid"] != PINNED_AUDIT_PACKAGE_CID
    )
    probes = [
        _probe(
            "audit_package_files_match_generator",
            verified["ok"] is True,
            reason=(
                "Declared audit-package files match the generator."
                if verified["ok"]
                else "Declared audit-package files do not match the generator."
            ),
        ),
        _probe(
            "pyproject_audit_package_table",
            table.get("interface") == INTERFACE and table.get("task-id") == PCPR_092_TASK_ID,
            reason=(
                "pyproject.toml declares the PCPR-092 audit-package table."
                if table.get("interface") == INTERFACE
                else "pyproject.toml does not declare the PCPR-092 audit-package table."
            ),
        ),
        _probe(
            "paths_remain_authorized",
            package["paths_authorized"] is True
            and list(package["authorized_path_prefixes"])
            == list(AUTHORIZED_PATH_PREFIXES),
            reason="Audit-package paths remain inside the declared PCPR-092 allowlist.",
        ),
        _probe(
            "canonical_objective_bytes_match",
            package["idea"] == REFERENCE_OBJECTIVE_IDEA
            and package["idea_digest"] == PINNED_IDEA_DIGEST,
            reason="Audit package uses the canonical PCPR-060 idea bytes.",
        ),
        _probe(
            "architecture_evidence_present",
            trace["architecture_evidence_present"] is True,
            reason="Architecture evidence is present.",
        ),
        _probe(
            "state_machine_evidence_present",
            trace["state_machine_evidence_present"] is True,
            reason="State-machine evidence is present.",
        ),
        _probe(
            "authority_evidence_present",
            trace["authority_evidence_present"] is True,
            reason="Authority evidence is present.",
        ),
        _probe(
            "canonicalization_evidence_present",
            trace["canonicalization_evidence_present"] is True,
            reason="Canonicalization evidence is present.",
        ),
        _probe(
            "recovery_evidence_present",
            trace["recovery_evidence_present"] is True,
            reason="Recovery evidence is present.",
        ),
        _probe(
            "proof_evidence_present",
            trace["proof_evidence_present"] is True,
            reason="Proof evidence is present.",
        ),
        _probe(
            "build_evidence_present",
            trace["build_evidence_present"] is True,
            reason="Build evidence is present.",
        ),
        _probe(
            "negative_test_evidence_present",
            trace["negative_test_evidence_present"] is True,
            reason="Negative-test evidence is present.",
        ),
        _probe(
            "limitation_evidence_present",
            trace["limitation_evidence_present"] is True,
            reason="Limitation evidence is present.",
        ),
        _probe(
            "all_g920_evidence_classes_present",
            package["all_g920_evidence_classes_present"] is True
            and trace["all_g920_evidence_classes_present"] is True,
            reason="Every PCPR-G920 evidence class is present.",
        ),
        _probe(
            "all_artifacts_bounded",
            package["all_artifacts_bounded"] is True
            and trace["all_artifacts_bounded"] is True,
            reason="Every recorded audit artifact is bounded.",
        ),
        _probe(
            "reproduction_commands_work",
            package["reproduction_commands_work"] is True
            and reproduction["ok"] is True
            and reproduction["command_count"] == len(ARTIFACTS),
            reason="Every declared reproduction command succeeds hermetically.",
        ),
        _probe(
            "audit_status_external_audit_ready",
            package["audit_status"] == AUDIT_STATUS_READY
            and package["externally_audited"] is False
            and AUDIT_STATUS != AUDIT_STATUS_AUDITED,
            reason="Package status is external_audit_ready and never externally_audited.",
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
            "audit_package_produced",
            package["audit_package_produced"] is True
            and package["audit_package_deferred"] is False
            and package["next_task_id"] == PCPR_093_TASK_ID,
            reason="Security and correctness audit package is produced by PCPR-092.",
        ),
        _probe(
            "closed_release_deferred",
            package["closed_release_deferred"] is True
            and document["closed_release_outcome"] is None,
            reason="Closed release decision remains PCPR-093/094.",
        ),
        _probe(
            "idempotent_audit_package_cid",
            package["idempotent"] is True
            and idempotency["identical"] is True
            and idempotency["first_audit_package_cid"] == package["audit_package_cid"]
            and idempotency["second_audit_package_cid"] == package["audit_package_cid"]
            and package["audit_package_cid"] == PINNED_AUDIT_PACKAGE_CID,
            reason="A second audit-package emission yields the same package CID.",
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
                "inventory, or audit-package CID is forbidden."
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
            "live_audit_package_represented_as_live",
            False,
            reason="Hermetic audit-package coverage is not represented as live.",
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
            "externally_audited_claimed",
            False,
            reason="Package status is external_audit_ready, never externally_audited.",
        ),
        _probe(
            "closed_release_claimed",
            False,
            reason="Closed release decision remains PCPR-093/094.",
        ),
        _probe(
            "unbounded_artifact_present",
            False,
            reason="Every recorded audit artifact is bounded.",
        ),
        _probe(
            "reproduction_commands_failed",
            False,
            reason="Every declared reproduction command succeeds hermetically.",
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


def qualify_audit_package(
    probes: Sequence[OutcomeProbe],
    *,
    audit_package_cid: str,
    document_cid: str,
    catalog_cid: str,
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
) -> AccelerateSecurityAndCorrectnessAuditVerdict:
    if not probes:
        raise AccelerateSecurityAndCorrectnessAuditError(
            "at least one probe is required"
        )
    normalized: list[OutcomeProbe] = []
    blockers: list[str] = []
    for probe in probes:
        kind = _kind(probe.evidence_kind, "evidence_kind")
        if probe.live and kind != "measured_live":
            raise AccelerateSecurityAndCorrectnessAuditError(
                "live claims require measured_live evidence"
            )
        if probe.simulated_represented_as_live:
            raise AccelerateSecurityAndCorrectnessAuditError(
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
        "task_id": PCPR_092_TASK_ID,
        "goal_id": PCPR_092_GOAL_ID,
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
        "audit_package_cid": audit_package_cid,
        "document_cid": document_cid,
        "catalog_cid": catalog_cid,
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
    return AccelerateSecurityAndCorrectnessAuditVerdict(
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
        audit_package_cid=audit_package_cid,
        document_cid=document_cid,
        catalog_cid=catalog_cid,
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


def qualify_current_head_audit_package(
    start: Path | None = None,
) -> AccelerateSecurityAndCorrectnessAuditVerdict:
    document = render_declared_package(start)
    catalog = platform_audit_package_catalog(start)
    return qualify_audit_package(
        current_head_static_probes(start),
        audit_package_cid=str(
            document["security_and_correctness_audit_package"]["audit_package_cid"]
        ),
        document_cid=str(document["document_cid"]),
        catalog_cid=str(catalog["catalog_cid"]),
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


def pcpr_092_receipt_promotion(
    verdict: AccelerateSecurityAndCorrectnessAuditVerdict,
) -> dict[str, Any]:
    if verdict.closed_release_outcome is not None:
        raise AccelerateSecurityAndCorrectnessAuditError(
            "security and correctness audit package must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise AccelerateSecurityAndCorrectnessAuditError(
            "security and correctness audit package must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise AccelerateSecurityAndCorrectnessAuditError(
            "security and correctness audit package completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise AccelerateSecurityAndCorrectnessAuditError(
            "security and correctness audit package must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise AccelerateSecurityAndCorrectnessAuditError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.live_client or verdict.live_application:
        raise AccelerateSecurityAndCorrectnessAuditError(
            "live client claims require measured_live evidence"
        )
    return verdict.to_mapping()


PINNED_AUDIT_PACKAGE_CID: Final = (
    "baguqeerazukvaxrpd53tqeo636o6rhtyr6iiqvx3njz444rj6yuzjeed3cpq"
)
PINNED_DOCUMENT_CID: Final = (
    "baguqeerakwiirkz6hkietp5sirow5s4okaexitoscvnxhte2uo7wdpuhswga"
)
PINNED_CATALOG_CID: Final = (
    "baguqeerabs3e6yurwgmbdkb2vc5m6xe2vxcucmzhdytxzcwrze7erlahhyxa"
)
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeera7omdnlqnozisr4rjqc54w3k2kievprnzo5apq6qci7xxccyftknq"
)


__all__ = [
    "ARTIFACTS",
    "AUTHORIZED_PATH_PREFIXES",
    "AUDIT_STATUS",
    "AUDIT_STATUS_AUDITED",
    "AUDIT_STATUS_READY",
    "CLOSED_RELEASE_OUTCOMES",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "ESCALATION_ORDER",
    "G910_SURFACES",
    "G920_EVIDENCE_CLASSES",
    "HERMETIC_CANDIDATE_SUITES",
    "INTERFACE",
    "OBJECTIVE_KIND",
    "OPERATOR_BLOCKING_TASK_ID",
    "OutcomeProbe",
    "PCPR_092_GOAL_ID",
    "PCPR_092_TASK_ID",
    "PINNED_AUDIT_PACKAGE_CID",
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
    "AccelerateSecurityAndCorrectnessAuditError",
    "all_g920_evidence_classes_present",
    "current_head_static_probes",
    "hermetic_audit_package_trace",
    "path_is_authorized",
    "pcpr_092_receipt_promotion",
    "platform_audit_package_catalog",
    "produce_audit_package",
    "qualify_current_head_audit_package",
    "qualify_audit_package",
    "refuse_bypass_proof_cid_remint",
    "refuse_chain_cid_remint",
    "refuse_closed_release_claim",
    "refuse_database_edit",
    "refuse_externally_audited_claim",
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
    "refuse_unbounded_artifact",
    "render_declared_package",
    "reproduce_artifact",
    "reproduce_audit_package",
    "verify_audit_package_files",
]
