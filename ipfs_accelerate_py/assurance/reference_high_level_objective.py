"""Fail-closed PCPR-060 Accelerate reference high-level objective.

Submit the Phase-6 reference idea through the repaired PCPR-004 direct
interface as a declared SupervisorObjectiveIntent. Authenticate the
caller, validate authority, and materialize the declared objective,
goals, tasks, assumptions, guarantees, acceptance conditions, and
budgets.

Live DuckDB or Quack materialization, live Supervisor.run / CLI / MCP
submission, and a separate authorized START stay typed unavailable.
This module never writes DuckDB or Quack state, never starts the
supervisor, never constructs a ContextPack (PCPR-061), never stores a
root (PCPR-062), and never emits a closed PCPR release outcome.

PCPR-001 live qualification and the PCPR-002 freeze remain
rnd_non_promoted. Simulated results are not live.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.assurance.branch_and_release_gates import (
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID as PCPR_057_VERDICT_CID,
    PINNED_BRANCH_POLICY_CID as PCPR_057_BRANCH_POLICY_CID,
    PINNED_RELEASE_GATE_CID as PCPR_057_RELEASE_GATE_CID,
)
from ipfs_accelerate_py.assurance.canonical_byte_cid_vectors import (
    PINNED_CONTRACT_VECTORS,
)
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
    admit_shared_contract,
    shared_interface_id,
    shared_schema_id,
)

INTERFACE: Final = "AccelerateReferenceHighLevelObjective@1"
SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/reference-high-level-objective@1"
)
OBJECTIVE_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/declared-reference-objective@1"
)
VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/reference-high-level-objective-verdict@1"
)
CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/assurance/platform-reference-objective-catalog@1"
)
CATALOG_INTERFACE: Final = "PlatformReferenceHighLevelObjective@1"
PCPR_060_TASK_ID: Final = "PCPR-060"
PCPR_060_GOAL_ID: Final = "PCPR-G700"
PCPR_061_TASK_ID: Final = "PCPR-061"
PCPR_062_TASK_ID: Final = "PCPR-062"
PCPR_072_TASK_ID: Final = "PCPR-072"
PCPR_057_TASK_ID: Final = "PCPR-057"
PCPR_056_TASK_ID: Final = "PCPR-056"
PCPR_040_TASK_ID: Final = "PCPR-040"
PCPR_004_TASK_ID: Final = "PCPR-004"
PCPR_002_TASK_ID: Final = "PCPR-002"
PCPR_001_TASK_ID: Final = "PCPR-001"
PCPR_093_TASK_ID: Final = "PCPR-093"
PCPR_094_TASK_ID: Final = "PCPR-094"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = (
    "proof-carrying-platform-qualification-and-release-v1"
)
EVIDENCE_ID: Final = "pcpr/reference-high-level-objective@1"
OBJECTIVE_KIND: Final = "declared_reference_high_level_objective"
OBJECTIVE_ID: Final = "PCPR-G700"
LANGUAGE: Final = "Python"
DECLARED_CALLER: Final = "principal:pcpr-060-reference-submitter"
REQUESTED_INTERFACE: Final = "supervisor.objectives.submit"
REPAIRED_DIRECT_INTERFACE_TASK: Final = PCPR_004_TASK_ID
SOURCE_REPOSITORY: Final = "endomorphosis/ipfs_accelerate_py"
OBJECTIVE_DIR_RELPATH: Final = "packaging/pcpr/reference-workflow/cpython312"
OBJECTIVE_JSON_NAME: Final = "reference.objective.json"
OBJECTIVE_README_RELPATH: Final = "packaging/pcpr/reference-workflow/README.md"
CATALOG_RELPATH: Final = "packaging/pcpr/reference-workflow/platform-catalog.json"
SOURCE_DATE_EPOCH: Final = "0"
OPERATOR_BLOCKING_TASK_ID: Final = (
    "pcpr-060-operator-live-objective-materialization"
)
PCPR_041_OBJECTIVE_INTENT_VECTOR_CID: Final = PINNED_CONTRACT_VECTORS[
    "SupervisorObjectiveIntent"
]["cid"]

if PCPR_056_LOCK_CID != (
    "baguqeerawsekbbbt5ccctjt4tydzeahfkahsatb6q5x7k6cy4inrii5lhqmq"
):
    raise RuntimeError("PCPR-060 lock CID remints PCPR-056")
if PCPR_057_BRANCH_POLICY_CID != (
    "baguqeeram3uegwfx6c4i76eyj5nuzlgsqw6yl5vb5c6drkuyacn5l77ksnna"
):
    raise RuntimeError("PCPR-060 branch-policy CID remints PCPR-057")
if PCPR_057_RELEASE_GATE_CID != (
    "baguqeerajahrusqwd33fqbuw46zgqcsw4xsuswdqq6td7c3nyddeeg5qofca"
):
    raise RuntimeError("PCPR-060 release-gate CID remints PCPR-057")
if PCPR_041_OBJECTIVE_INTENT_VECTOR_CID != (
    "baguqeerak5dpgbfqb3y3lzidrzhctptq4czihlmihlrozcysqrwtk57luzya"
):
    raise RuntimeError("PCPR-060 remints the PCPR-041 SupervisorObjectiveIntent vector")

REFERENCE_OBJECTIVE_IDEA: Final = (
    "Modify a typed formal-logic API while reusing unaffected proofs, "
    "selecting only impacted tests, rejecting stale-tree evidence, and "
    "producing a complete proof-carrying execution receipt."
)

DIRECT_INTERFACE_PATHS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "supervisor_run": (
            "ipfs_accelerate_py/agent_supervisor/entrypoints/facade.py"
        ),
        "cli_supervisor_run": (
            "ipfs_accelerate_py/agent_supervisor/entrypoints/cli.py"
        ),
        "intent_service": (
            "ipfs_accelerate_py/agent_supervisor/entrypoints/intent_service.py"
        ),
        "prompt_workflow": (
            "ipfs_accelerate_py/agent_supervisor/prompt/prompt_workflow.py"
        ),
        "plan_supervisor_service": (
            "ipfs_accelerate_py/agent_supervisor/prompt/plan_supervisor_service.py"
        ),
        "plan_revision_store": (
            "ipfs_accelerate_py/agent_supervisor/task_sources/plan_revision_store.py"
        ),
        "mcp_agent_supervisor_run": (
            "ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/"
            "prompt_entrypoints.py"
        ),
    }
)

REFERENCE_WORKFLOW_TASKS: Final[tuple[Mapping[str, str], ...]] = (
    MappingProxyType(
        {
            "task_id": "PCPR-060",
            "title": "Submit reference high-level objective",
            "role": "this_task",
        }
    ),
    MappingProxyType(
        {
            "task_id": "PCPR-061",
            "title": "Build semantic ContextPack",
            "role": "successor",
        }
    ),
    MappingProxyType(
        {
            "task_id": "PCPR-062",
            "title": "Persist and publish current ContextPack root",
            "role": "successor",
        }
    ),
    MappingProxyType(
        {
            "task_id": "PCPR-063",
            "title": "Execute deterministic-first route",
            "role": "successor",
        }
    ),
    MappingProxyType(
        {
            "task_id": "PCPR-064",
            "title": "Produce bounded patch",
            "role": "successor",
        }
    ),
    MappingProxyType(
        {
            "task_id": "PCPR-065",
            "title": "Run selected tests and proofs",
            "role": "successor",
        }
    ),
    MappingProxyType(
        {
            "task_id": "PCPR-066",
            "title": "Introduce unrelated state change",
            "role": "successor",
        }
    ),
    MappingProxyType(
        {
            "task_id": "PCPR-067",
            "title": "Demonstrate safe reuse",
            "role": "successor",
        }
    ),
    MappingProxyType(
        {
            "task_id": "PCPR-068",
            "title": "Introduce relevant interface change",
            "role": "successor",
        }
    ),
    MappingProxyType(
        {
            "task_id": "PCPR-069",
            "title": "Demonstrate stale rejection and PlanDelta",
            "role": "successor",
        }
    ),
    MappingProxyType(
        {
            "task_id": "PCPR-070",
            "title": "Restart authoritative state owner",
            "role": "successor",
        }
    ),
    MappingProxyType(
        {
            "task_id": "PCPR-071",
            "title": "Demonstrate recovery and idempotency",
            "role": "successor",
        }
    ),
    MappingProxyType(
        {
            "task_id": "PCPR-072",
            "title": "Produce final proof-carrying receipt chain",
            "role": "successor",
        }
    ),
)

REFERENCE_GOALS: Final[tuple[Mapping[str, str], ...]] = (
    MappingProxyType(
        {
            "goal_id": "PCPR-G700",
            "title": "Reference proof-carrying workflow and recovery",
        }
    ),
    MappingProxyType(
        {
            "goal_id": "PCPR-G710",
            "title": "Objective, ContextPack, and durable root",
        }
    ),
    MappingProxyType(
        {
            "goal_id": "PCPR-G720",
            "title": "Deterministic route, patch, tests, and proofs",
        }
    ),
    MappingProxyType(
        {
            "goal_id": "PCPR-G730",
            "title": "Reuse, invalidation, PlanDelta, and refill",
        }
    ),
    MappingProxyType(
        {
            "goal_id": "PCPR-G740",
            "title": "Restart recovery and final receipt chain",
        }
    ),
)

ASSUMPTIONS: Final[tuple[str, ...]] = (
    "The repaired PCPR-004 Supervisor.run / CLI / MCP path is the only submission interface.",
    "PCPR-001 live qualification remains rnd_non_promoted.",
    "PCPR-002 did not freeze public supervisor contracts.",
    "PCPR-057 branch and release gates are declared, not live GitHub protection.",
    "Direct DuckDB or Quack writes are prohibited.",
    "ContextPack construction is PCPR-061.",
    "Durable storage and current-root CAS are PCPR-062.",
    "Closed release decision remains PCPR-093/094.",
)

GUARANTEES: Final[tuple[str, ...]] = (
    "Live claims require live evidence.",
    "Simulated results are not represented as live.",
    "Missing environments stay typed unavailable.",
    "DuckDB and Quack state are not written.",
    "No closed PCPR release outcome is emitted.",
    "No competing supervisor, planner, task database, or event database is created.",
    "START remains a separate authorized control operation and is not performed.",
)

ACCEPTANCE_CONDITIONS: Final[tuple[str, ...]] = (
    "The named PCPR-060 receipt exists.",
    "promotion_status is rnd_non_promoted or an honest typed unavailable/blocked status.",
    "The receipt does not claim a closed release outcome.",
)

BUDGETS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "maximum_initial_tasks": 80,
        "maximum_total_tasks_without_operator_scope_expansion": 140,
        "maximum_replan_epochs": 20,
        "maximum_automatically_generated_tasks_per_refill": 12,
        "final_validation_reserve_percent": 30,
        "escalation_order": (
            "exact receipt",
            "AST and dependency analysis",
            "schema, type, and static checks",
            "selected tests",
            "incremental prover",
            "local small specialist",
            "medium model",
            "frontier model",
            "human decision",
        ),
    }
)

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_reference_high_level_objective.py",
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
        "objective_files_match_generator",
        "pyproject_reference_high_level_objective_table",
        "supervisor_objective_intent_admitted",
        "idea_digest_matches_pin",
        "direct_interface_source_present",
        "live_materialization_is_unavailable",
        "start_not_performed",
        "duckdb_or_quack_not_written",
        "context_pack_deferred_to_pcpr_061",
        "storage_deferred_to_pcpr_062",
        "pcpr_057_identities_not_reminted",
        "pcpr_041_intent_vector_not_reminted",
        "operator_blocking_task_emitted",
        "no_closed_release_outcome",
    }
)
FORBIDDEN_PRESENT_PROBE_IDS: Final[frozenset[str]] = frozenset(
    {
        "simulated_results_represented_as_live",
        "live_materialization_represented_as_live",
        "live_start_represented_as_live",
        "closed_release_represented_as_live",
        "direct_database_bypass_used",
        "competing_authority_created",
        "compatibility_identities_reminted",
    }
)

OBJECTIVE_README: Final = """# PCPR-060 declared reference high-level objective

These files are the Accelerate *declared* Phase-6 reference objective
for proof-carrying-platform-0.1.0. They submit the exact idea:

    Modify a typed formal-logic API while reusing unaffected proofs,
    selecting only impacted tests, rejecting stale-tree evidence, and
    producing a complete proof-carrying execution receipt.

through the repaired PCPR-004 direct-interface contracts as a
SupervisorObjectiveIntent. They are not a live Supervisor.run, CLI, or
MCP submission, not a live DuckDB or Quack materialization, not a
START, not a ContextPack (PCPR-061), not a stored root (PCPR-062), not
a freeze (PCPR-002), and not a closed PCPR release (PCPR-093/094).

- `cpython312/reference.objective.json` binds the idea digest, declared
  caller, authority checks, goals, tasks, assumptions, guarantees,
  acceptance conditions, and budgets. Exact commit and tree are bound
  by the PCPR-060 receipt `current_tree_binding` because nested
  admission rewrites HEAD. `origin/main` is not the release identity.
- `platform-catalog.json` binds Datasets and Kit objective-binding
  documents when those sibling trees are present. Sibling source is
  never required.
- Missing a live Quack-fenced state-owner session emits explicit
  operator-blocking task `pcpr-060-operator-live-objective-materialization`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
Provider-local `~/.local` tools are not sealed-environment authority.
"""


class ReferenceHighLevelObjectiveError(Exception):
    """Fail-closed PCPR-060 contract error."""


class ReferenceHighLevelObjectiveAdmissionError(ReferenceHighLevelObjectiveError):
    """Raised when a reference-objective input is rejected."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReferenceHighLevelObjectiveError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise ReferenceHighLevelObjectiveError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise ReferenceHighLevelObjectiveError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def idea_digest() -> str:
    """CIDv1 of the exact reference idea. Not the PCPR-041 contract vector."""

    return content_identity(
        {
            "idea": REFERENCE_OBJECTIVE_IDEA,
            "language": LANGUAGE,
            "objective_id": OBJECTIVE_ID,
            "program_id": PCPR_PROGRAM_ID,
            "goal_id": PCPR_060_GOAL_ID,
            "task_id": PCPR_060_TASK_ID,
        }
    )


def admit_supervisor_objective_intent() -> dict[str, Any]:
    """Admit SupervisorObjectiveIntent@1 for the reference idea."""

    return admit_shared_contract(
        "SupervisorObjectiveIntent",
        {
            "schema": shared_schema_id("SupervisorObjectiveIntent"),
            "interface": shared_interface_id("SupervisorObjectiveIntent"),
            "objective_id": OBJECTIVE_ID,
            "language": LANGUAGE,
            "idea_digest": idea_digest(),
        },
    )


def parse_pyproject_objective_table(text: str) -> dict[str, Any]:
    import tomllib

    table = tomllib.loads(_text(text, "pyproject.toml"))
    if not isinstance(table, dict):
        raise ReferenceHighLevelObjectiveError("pyproject.toml must be a table")
    tool = table.get("tool")
    payload: dict[str, Any] = {}
    if isinstance(tool, dict):
        accelerate = tool.get("ipfs_accelerate_py")
        if isinstance(accelerate, dict):
            raw = accelerate.get("reference-high-level-objective")
            if isinstance(raw, dict):
                payload = dict(raw)
    return payload


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
class ReferenceHighLevelObjectiveVerdict:
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
    live_objective_submission: bool
    live_objective_submission_evidence_kind: str
    live_materialization: bool
    live_materialization_evidence_kind: str
    live_start: bool
    live_authenticated_caller: bool
    operator_blocking_task: str
    simulated_results_represented_as_live: bool
    this_task_created_competing_authority: bool
    probes: tuple[OutcomeProbe, ...]
    blockers: tuple[str, ...]
    verdict_cid: str
    objective_cid: str
    idea_digest: str
    intent_cid: str
    catalog_cid: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "verdict_cid": self.verdict_cid,
            "objective_cid": self.objective_cid,
            "idea_digest": self.idea_digest,
            "intent_cid": self.intent_cid,
            "catalog_cid": self.catalog_cid,
            "promotion_status": self.promotion_status,
            "supervisor_disposition": self.supervisor_disposition,
            "closed_release_outcome": self.closed_release_outcome,
            "release_claim": self.release_claim,
            "completion_authoritative": self.completion_authoritative,
            "contracts_frozen": self.contracts_frozen,
            "duckdb_or_quack_state_written": self.duckdb_or_quack_state_written,
            "sibling_source_required": self.sibling_source_required,
            "live_objective_submission": self.live_objective_submission,
            "live_objective_submission_evidence_kind": (
                self.live_objective_submission_evidence_kind
            ),
            "live_materialization": self.live_materialization,
            "live_materialization_evidence_kind": (
                self.live_materialization_evidence_kind
            ),
            "live_start": self.live_start,
            "live_authenticated_caller": self.live_authenticated_caller,
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
            "An admitted Quack-fenced state-owner session and an authenticated "
            "Supervisor.run / CLI / MCP caller on the repaired PCPR-004 path"
        ),
        "action": (
            "Submit the exact reference idea through supervisor.objectives.submit "
            "with current-root, lease, fence, idempotency, and expected-effect "
            "bindings. Apply through PlanRevisionStore and DatabaseTaskSource. "
            "Do not write DuckDB or Quack state directly. Keep START separate."
        ),
        "reason": (
            "Sealed validation has no admitted Quack-fenced state-owner session. "
            "Direct DuckDB writes are prohibited. A live authenticated "
            "Supervisor.run, CLI supervisor run, or MCP agent_supervisor_run "
            "was not performed. The declared intent is not live materialization."
        ),
    }


def _source_binding() -> dict[str, Any]:
    return {
        "kind": "git",
        "repository": SOURCE_REPOSITORY,
        "binding": "current_head_not_mutable_main",
        "mutable_main_reference": False,
        "commit": {
            "status": "observed_at_evaluation",
            "evidence_kind": "measured",
            "live": False,
            "field": "PCPR-060 receipt current_tree_binding",
            "reason": (
                "Exact commit and tree are bound by the task receipt "
                "because nested admission rewrites HEAD. This document "
                "does not pin origin/main."
            ),
        },
    }


def observe_state_owner() -> dict[str, Any]:
    """Measure sealed-PATH state-owner tools. Never write DuckDB or Quack."""

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
        "duckdb_cli": env.get("duckdb", "unavailable"),
        "quack": "unavailable",
        "quack_fenced_session": typed_unavailable(
            reason=(
                "No admitted Quack-fenced state-owner session was observed. "
                "Direct DuckDB writes are prohibited, so library presence is "
                "not live objective materialization."
            )
        ),
        "live_authenticated_caller": typed_unavailable(
            reason=(
                "No live authenticated Supervisor.run / CLI / MCP caller "
                "session was admitted. The declared principal is intent, "
                "never authority."
            )
        ),
        "duckdb_or_quack_state_written": False,
        "direct_duckdb_write_prohibited": True,
        "direct_quack_write_prohibited": True,
        "evidence_kind": "measured",
    }


def observe_direct_interface_source(root: Path) -> dict[str, Any]:
    """Measure PCPR-004 repaired-path source files. This is not a live run."""

    observed: dict[str, Any] = {}
    missing: list[str] = []
    for name, relpath in DIRECT_INTERFACE_PATHS.items():
        path = root / relpath
        present = path.is_file()
        observed[name] = {
            "path": relpath,
            "present": present,
            "sha256": sha256_bytes(path.read_bytes()) if present else "unavailable",
            "live": False,
            "evidence_kind": "measured" if present else "unavailable",
        }
        if not present:
            missing.append(name)
    return {
        "all_present": not missing,
        "missing": missing,
        "surfaces": observed,
        "live_submission": False,
        "evidence_kind": "measured",
    }


def render_supervisor_objective_intent() -> dict[str, Any]:
    admitted = admit_supervisor_objective_intent()
    digest = idea_digest()
    envelope = {
        **admitted,
        "task_id": PCPR_060_TASK_ID,
        "goal_id": PCPR_060_GOAL_ID,
        "program_id": PCPR_PROGRAM_ID,
        "idea": REFERENCE_OBJECTIVE_IDEA,
        "pcpr_041_contract_vector_cid": PCPR_041_OBJECTIVE_INTENT_VECTOR_CID,
        "contract_vector_is_not_this_idea": True,
        "live": False,
        "authority": "intent_not_authority",
        "evidence_kind": "measured",
    }
    if envelope["idea_digest"] != digest:
        raise ReferenceHighLevelObjectiveError("idea_digest drifted")
    envelope["intent_cid"] = content_identity(
        {key: value for key, value in envelope.items() if key != "intent_cid"}
    )
    return envelope


def render_declared_reference_objective(
    root: Path | None = None,
) -> dict[str, Any]:
    package_root = root or discover_accelerate_root()
    if package_root is None:
        raise ReferenceHighLevelObjectiveError("Accelerate package root was not found")
    intent = render_supervisor_objective_intent()
    state_owner = observe_state_owner()
    interfaces = observe_direct_interface_source(package_root)
    document = {
        "schema": OBJECTIVE_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_060_TASK_ID,
        "goal_id": PCPR_060_GOAL_ID,
        "program_id": PCPR_PROGRAM_ID,
        "board_namespace": PCPR_BOARD_NAMESPACE,
        "evidence_id": EVIDENCE_ID,
        "portfolio_id": PORTFOLIO_ID,
        "portfolio_version": PORTFOLIO_VERSION,
        "objective_kind": OBJECTIVE_KIND,
        "package_name": PACKAGE_NAME,
        "package_version": PACKAGE_VERSION,
        "language": LANGUAGE,
        "objective_id": OBJECTIVE_ID,
        "idea": REFERENCE_OBJECTIVE_IDEA,
        "idea_digest": intent["idea_digest"],
        "intent": {
            "schema": intent["schema"],
            "interface": intent["interface"],
            "objective_id": intent["objective_id"],
            "language": intent["language"],
            "idea_digest": intent["idea_digest"],
            "intent_cid": intent["intent_cid"],
            "pcpr_041_contract_vector_cid": PCPR_041_OBJECTIVE_INTENT_VECTOR_CID,
            "live": False,
        },
        "caller": {
            "principal": DECLARED_CALLER,
            "authenticated": False,
            "live": False,
            "authority": "intent_not_authority",
            "evidence_kind": "unavailable",
            "reason": (
                "The declared principal is captured as intent. No live "
                "authenticated caller session was admitted."
            ),
        },
        "authority": {
            "requested_interface": REQUESTED_INTERFACE,
            "repaired_direct_interface_task": REPAIRED_DIRECT_INTERFACE_TASK,
            "python_cli_mcp_parity_required": True,
            "complete_launch_plan_caller_injection_required": False,
            "start_is_separate_authorized_control_operation": True,
            "freeze_task": PCPR_002_TASK_ID,
            "contracts_frozen": False,
            "direct_interface_source": interfaces,
            "live": False,
            "evidence_kind": "measured",
        },
        "goals": [dict(item) for item in REFERENCE_GOALS],
        "tasks": [dict(item) for item in REFERENCE_WORKFLOW_TASKS],
        "assumptions": list(ASSUMPTIONS),
        "guarantees": list(GUARANTEES),
        "acceptance_conditions": list(ACCEPTANCE_CONDITIONS),
        "budgets": {
            "maximum_initial_tasks": BUDGETS["maximum_initial_tasks"],
            "maximum_total_tasks_without_operator_scope_expansion": (
                BUDGETS["maximum_total_tasks_without_operator_scope_expansion"]
            ),
            "maximum_replan_epochs": BUDGETS["maximum_replan_epochs"],
            "maximum_automatically_generated_tasks_per_refill": (
                BUDGETS["maximum_automatically_generated_tasks_per_refill"]
            ),
            "final_validation_reserve_percent": (
                BUDGETS["final_validation_reserve_percent"]
            ),
            "escalation_order": list(BUDGETS["escalation_order"]),
        },
        "materialization": {
            "kind": "declared_not_live",
            "admitted": False,
            "live": False,
            "applied": False,
            "plan_revision_store_applied": False,
            "database_task_source_projected": False,
            "duckdb_or_quack_state_written": False,
            "evidence_kind": "unavailable",
            "reason": (
                "Live materialization requires an admitted Quack-fenced "
                "state-owner session. Direct DuckDB writes are prohibited. "
                "This document is declared intent, not live authority."
            ),
        },
        "start": {
            "performed": False,
            "live": False,
            "separate_authorized_control_operation": True,
            "evidence_kind": "measured",
            "reason": "START is separate from materialize and was not performed.",
        },
        "context_pack": {
            "task_id": PCPR_061_TASK_ID,
            "constructed": False,
            "live": False,
            "deferred": True,
        },
        "storage": {
            "task_id": PCPR_062_TASK_ID,
            "stored": False,
            "current_root_published": False,
            "live": False,
            "deferred": True,
        },
        "prerequisites": {
            "pcpr_001_task_id": PCPR_001_TASK_ID,
            "pcpr_001_promotion_status": "rnd_non_promoted",
            "pcpr_002_task_id": PCPR_002_TASK_ID,
            "pcpr_002_contracts_frozen": False,
            "pcpr_004_task_id": PCPR_004_TASK_ID,
            "pcpr_056_task_id": PCPR_056_TASK_ID,
            "pcpr_056_lock_cid": PCPR_056_LOCK_CID,
            "pcpr_057_task_id": PCPR_057_TASK_ID,
            "pcpr_057_verdict_cid": PCPR_057_VERDICT_CID,
            "pcpr_057_branch_policy_cid": PCPR_057_BRANCH_POLICY_CID,
            "pcpr_057_release_gate_cid": PCPR_057_RELEASE_GATE_CID,
            "closed_decision_tasks": [PCPR_093_TASK_ID, PCPR_094_TASK_ID],
        },
        "source": _source_binding(),
        "state_owner": {
            "duckdb_cli": state_owner.get("duckdb_cli"),
            "duckdb_module": state_owner.get("duckdb_module"),
            "duckdb_module_is_not_live_materialization": True,
            "quack": "unavailable",
            "duckdb_or_quack_state_written": False,
            "direct_duckdb_write_prohibited": True,
            "direct_quack_write_prohibited": True,
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
        "source_date_epoch": SOURCE_DATE_EPOCH,
        "evidence_kind": "measured",
    }
    document["objective_cid"] = content_identity(
        {key: value for key, value in document.items() if key != "objective_cid"}
    )
    return document


def refuse_objective_remint(cid: str) -> str:
    if cid != PINNED_OBJECTIVE_CID:
        raise ReferenceHighLevelObjectiveError(
            f"reference objective CID {cid} remints {PINNED_OBJECTIVE_CID}"
        )
    return cid


def refuse_idea_digest_remint(cid: str) -> str:
    if cid != PINNED_IDEA_DIGEST:
        raise ReferenceHighLevelObjectiveError(
            f"idea digest {cid} remints {PINNED_IDEA_DIGEST}"
        )
    return cid


def artifact_paths(root: Path) -> dict[str, Path]:
    return {
        "objective": root / OBJECTIVE_DIR_RELPATH / OBJECTIVE_JSON_NAME,
        "readme": root / OBJECTIVE_README_RELPATH,
        "catalog": root / CATALOG_RELPATH,
    }


def _sibling_binding(path: Path, relative_path: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": relative_path,
            "status": "unavailable",
            "evidence_kind": "unavailable",
            "reason": "Sibling objective-binding file is not present beside this checkout.",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": relative_path,
        "status": "observed",
        "evidence_kind": "measured",
        "package_name": payload.get("package_name"),
        "package_version": payload.get("package_version"),
        "objective_kind": payload.get("objective_kind"),
        "objective_cid": payload.get("objective_cid"),
        "idea_digest": payload.get("idea_digest"),
        "binding_cid": payload.get("binding_cid"),
        "live": False,
        "applied": False,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def platform_objective_catalog(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise ReferenceHighLevelObjectiveError("Accelerate package root was not found")
    parent = root.parent
    objective = render_declared_reference_objective(root)
    datasets_binding = _sibling_binding(
        parent / "ipfs_datasets" / OBJECTIVE_DIR_RELPATH / "reference.binding.json",
        relative_path=(
            f"../ipfs_datasets/{OBJECTIVE_DIR_RELPATH}/reference.binding.json"
        ),
    )
    kit_binding = _sibling_binding(
        parent / "ipfs_kit" / OBJECTIVE_DIR_RELPATH / "reference.binding.json",
        relative_path=f"../ipfs_kit/{OBJECTIVE_DIR_RELPATH}/reference.binding.json",
    )
    catalog = {
        "schema": CATALOG_SCHEMA,
        "interface": CATALOG_INTERFACE,
        "task_id": PCPR_060_TASK_ID,
        "goal_id": PCPR_060_GOAL_ID,
        "objective_kind": OBJECTIVE_KIND,
        "portfolio_id": PORTFOLIO_ID,
        "portfolio_version": PORTFOLIO_VERSION,
        "objective_cid": objective["objective_cid"],
        "idea_digest": objective["idea_digest"],
        "intent_cid": objective["intent"]["intent_cid"],
        "lock_cid": PCPR_056_LOCK_CID,
        "components": {
            "ipfs_accelerate_py": {
                "objective_path": f"{OBJECTIVE_DIR_RELPATH}/{OBJECTIVE_JSON_NAME}",
                "status": "observed",
                "evidence_kind": "measured",
                "package_name": PACKAGE_NAME,
                "package_version": PACKAGE_VERSION,
                "objective_kind": OBJECTIVE_KIND,
                "objective_cid": objective["objective_cid"],
                "idea_digest": objective["idea_digest"],
                "live": False,
                "applied": False,
            },
            "ipfs_datasets_py": {"binding": datasets_binding},
            "ipfs_kit_py": {"binding": kit_binding},
        },
        "live_objective_submission": False,
        "live_materialization": False,
        "live_start": False,
        "closed_release_outcome": None,
        "release_claim": False,
        "sibling_source_required": False,
        "evidence_kind": "measured",
    }
    catalog["catalog_cid"] = content_identity(
        {key: value for key, value in catalog.items() if key != "catalog_cid"}
    )
    return catalog


def write_reference_objective_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise ReferenceHighLevelObjectiveError("Accelerate package root was not found")
    objective = render_declared_reference_objective(root)
    paths = artifact_paths(root)
    _atomic_write(paths["objective"], pretty_json(objective))
    readme = (
        OBJECTIVE_README if OBJECTIVE_README.endswith("\n") else OBJECTIVE_README + "\n"
    )
    _atomic_write(paths["readme"], readme)
    catalog = platform_objective_catalog(start)
    _atomic_write(paths["catalog"], pretty_json(catalog))
    return {
        "objective": objective,
        "catalog": catalog,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def verify_reference_objective_files(start: Path | None = None) -> dict[str, Any]:
    root = discover_accelerate_root(start)
    if root is None:
        raise ReferenceHighLevelObjectiveError("Accelerate package root was not found")
    objective = render_declared_reference_objective(root)
    paths = artifact_paths(root)
    missing: list[str] = []
    objective_ok = False
    readme_ok = False
    expected_readme = (
        OBJECTIVE_README if OBJECTIVE_README.endswith("\n") else OBJECTIVE_README + "\n"
    )
    for name, path in paths.items():
        if name == "catalog":
            continue
        if not path.is_file():
            missing.append(name)
            continue
        if name == "objective":
            objective_ok = json.loads(path.read_text(encoding="utf-8")) == objective
        elif name == "readme":
            readme_ok = path.read_text(encoding="utf-8") == expected_readme
    return {
        "ok": not missing and objective_ok and readme_ok,
        "missing": missing,
        "objective_ok": objective_ok,
        "readme_ok": readme_ok,
        "objective_cid": objective["objective_cid"],
        "idea_digest": objective["idea_digest"],
        "objective_sha256": (
            sha256_bytes(paths["objective"].read_bytes())
            if paths["objective"].is_file()
            else "unavailable"
        ),
    }


def current_head_static_probes(
    start: Path | None = None,
) -> tuple[OutcomeProbe, ...]:
    root = discover_accelerate_root(start)
    if root is None:
        raise ReferenceHighLevelObjectiveError("Accelerate package root was not found")
    objective = render_declared_reference_objective(root)
    verified = verify_reference_objective_files(root)
    table = parse_pyproject_objective_table(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    interfaces = observe_direct_interface_source(root)
    remint = (
        objective["objective_cid"] != PINNED_OBJECTIVE_CID
        or objective["idea_digest"] != PINNED_IDEA_DIGEST
        or objective["prerequisites"]["pcpr_056_lock_cid"] != PCPR_056_LOCK_CID
        or objective["prerequisites"]["pcpr_057_branch_policy_cid"]
        != PCPR_057_BRANCH_POLICY_CID
        or objective["intent"]["pcpr_041_contract_vector_cid"]
        != PCPR_041_OBJECTIVE_INTENT_VECTOR_CID
    )
    probes = [
        _probe(
            "objective_files_match_generator",
            verified.get("ok") is True and verified.get("objective_ok") is True,
            reason=(
                "Committed reference objective matches the generator."
                if verified.get("objective_ok") is True
                else "Committed reference objective is missing or drifts."
            ),
            details=verified,
        ),
        _probe(
            "pyproject_reference_high_level_objective_table",
            table.get("interface") == INTERFACE
            and table.get("schema") == SCHEMA
            and table.get("task-id") == PCPR_060_TASK_ID
            and table.get("objective-kind") == OBJECTIVE_KIND,
            reason=(
                "pyproject.toml declares AccelerateReferenceHighLevelObjective@1."
                if table.get("interface") == INTERFACE
                else "pyproject.toml does not declare AccelerateReferenceHighLevelObjective@1."
            ),
            details={"table": table},
        ),
        _probe(
            "supervisor_objective_intent_admitted",
            objective["intent"]["interface"] == "SupervisorObjectiveIntent@1"
            and objective["intent"]["schema"]
            == "pcpr/shared-contracts/supervisor-objective-intent@1"
            and objective["intent"]["live"] is False,
            reason="SupervisorObjectiveIntent@1 is admitted for the reference idea.",
        ),
        _probe(
            "idea_digest_matches_pin",
            objective["idea_digest"] == PINNED_IDEA_DIGEST,
            reason="The reference idea digest is pinned and not reminted.",
        ),
        _probe(
            "direct_interface_source_present",
            interfaces.get("all_present") is True,
            reason="PCPR-004 repaired-path source files are present.",
            details={"missing": interfaces.get("missing")},
        ),
        _probe(
            "live_materialization_is_unavailable",
            objective["materialization"]["live"] is False
            and objective["materialization"]["admitted"] is False
            and objective["materialization"]["duckdb_or_quack_state_written"] is False,
            reason="Live DuckDB/Quack materialization stays typed unavailable.",
        ),
        _probe(
            "start_not_performed",
            objective["start"]["performed"] is False
            and objective["start"]["separate_authorized_control_operation"] is True,
            reason="START is a separate authorized control and was not performed.",
        ),
        _probe(
            "duckdb_or_quack_not_written",
            objective["state_owner"]["duckdb_or_quack_state_written"] is False
            and objective["state_owner"]["direct_duckdb_write_prohibited"] is True,
            reason="This task does not write DuckDB or Quack state.",
        ),
        _probe(
            "context_pack_deferred_to_pcpr_061",
            objective["context_pack"]["task_id"] == PCPR_061_TASK_ID
            and objective["context_pack"]["constructed"] is False,
            reason="ContextPack construction remains PCPR-061.",
        ),
        _probe(
            "storage_deferred_to_pcpr_062",
            objective["storage"]["task_id"] == PCPR_062_TASK_ID
            and objective["storage"]["stored"] is False,
            reason="Durable storage and current-root CAS remain PCPR-062.",
        ),
        _probe(
            "pcpr_057_identities_not_reminted",
            objective["prerequisites"]["pcpr_057_branch_policy_cid"]
            == PCPR_057_BRANCH_POLICY_CID
            and objective["prerequisites"]["pcpr_057_release_gate_cid"]
            == PCPR_057_RELEASE_GATE_CID
            and objective["prerequisites"]["pcpr_056_lock_cid"] == PCPR_056_LOCK_CID,
            reason="PCPR-056 lock and PCPR-057 gate identities are bound, not reminted.",
        ),
        _probe(
            "pcpr_041_intent_vector_not_reminted",
            objective["intent"]["pcpr_041_contract_vector_cid"]
            == PCPR_041_OBJECTIVE_INTENT_VECTOR_CID
            and objective["intent"]["idea_digest"] != PCPR_041_OBJECTIVE_INTENT_VECTOR_CID,
            reason=(
                "The PCPR-041 SupervisorObjectiveIntent vector CID is bound. "
                "The reference idea digest is a distinct identity."
            ),
        ),
        _probe(
            "operator_blocking_task_emitted",
            objective["operator_blocking_task"]["task_id"] == OPERATOR_BLOCKING_TASK_ID
            and objective["operator_blocking_task"]["status"] == "typed_blocked",
            reason=(
                "Missing live Quack-fenced materialization emits explicit "
                "operator-blocking task pcpr-060-operator-live-objective-materialization."
            ),
        ),
        _probe(
            "no_closed_release_outcome",
            objective["closed_release_outcome"] is None
            and objective["release_claim"] is False,
            reason="This task does not emit a closed PCPR release outcome.",
        ),
        _probe(
            "compatibility_identities_reminted",
            remint,
            reason="A reminted objective, idea, lock, gate, or contract-vector CID is forbidden.",
            details={
                "objective_cid": objective["objective_cid"],
                "idea_digest": objective["idea_digest"],
            },
        ),
        _probe(
            "simulated_results_represented_as_live",
            False,
            reason="Simulated results are not represented as live.",
        ),
        _probe(
            "live_materialization_represented_as_live",
            False,
            reason="No live DuckDB/Quack materialization is represented as live.",
        ),
        _probe(
            "live_start_represented_as_live",
            False,
            reason="START was not performed and is not represented as live.",
        ),
        _probe(
            "closed_release_represented_as_live",
            False,
            reason="This task does not publish a PCPR release.",
        ),
        _probe(
            "direct_database_bypass_used",
            False,
            reason="Direct DuckDB or Quack writes were not used.",
        ),
        _probe(
            "competing_authority_created",
            False,
            reason="This task did not create a competing supervisor, planner, or database.",
        ),
        _probe(
            "live_objective_submission",
            None,
            evidence_kind="unavailable",
            reason=(
                "A live authenticated Supervisor.run / CLI / MCP submission "
                "was not performed. Declared intent is not live submission."
            ),
        ),
        _probe(
            "live_materialization",
            None,
            evidence_kind="unavailable",
            reason=(
                "No admitted Quack-fenced state-owner session was observed. "
                "Live materialization stays typed unavailable."
            ),
        ),
        _probe(
            "live_supervisor_qualification",
            None,
            evidence_kind="unavailable",
            reason="PCPR-001 live qualification remains typed unavailable.",
        ),
    ]
    return tuple(probes)


def qualify_reference_high_level_objective(
    probes: Sequence[OutcomeProbe],
    *,
    objective_cid: str,
    idea_digest_cid: str,
    intent_cid: str,
    catalog_cid: str,
) -> ReferenceHighLevelObjectiveVerdict:
    if not probes:
        raise ReferenceHighLevelObjectiveError("at least one probe is required")
    normalized: list[OutcomeProbe] = []
    blockers: list[str] = []
    for probe in probes:
        kind = _kind(probe.evidence_kind, "evidence_kind")
        if probe.live and kind != "measured_live":
            raise ReferenceHighLevelObjectiveError(
                "live claims require measured_live evidence"
            )
        if probe.simulated_represented_as_live:
            raise ReferenceHighLevelObjectiveError(
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
        "task_id": PCPR_060_TASK_ID,
        "goal_id": PCPR_060_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "sibling_source_required": False,
        "live_objective_submission": False,
        "live_objective_submission_evidence_kind": "unavailable",
        "live_materialization": False,
        "live_materialization_evidence_kind": "unavailable",
        "live_start": False,
        "live_authenticated_caller": False,
        "operator_blocking_task": OPERATOR_BLOCKING_TASK_ID,
        "simulated_results_represented_as_live": False,
        "this_task_created_competing_authority": False,
        "probes": [item.to_mapping() for item in normalized],
        "blockers": list(dict.fromkeys(blockers)),
        "objective_cid": objective_cid,
        "idea_digest": idea_digest_cid,
        "intent_cid": intent_cid,
        "catalog_cid": catalog_cid,
    }
    return ReferenceHighLevelObjectiveVerdict(
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
        live_objective_submission=False,
        live_objective_submission_evidence_kind="unavailable",
        live_materialization=False,
        live_materialization_evidence_kind="unavailable",
        live_start=False,
        live_authenticated_caller=False,
        operator_blocking_task=OPERATOR_BLOCKING_TASK_ID,
        simulated_results_represented_as_live=False,
        this_task_created_competing_authority=False,
        probes=tuple(normalized),
        blockers=tuple(dict.fromkeys(blockers)),
        verdict_cid=content_identity(payload),
        objective_cid=objective_cid,
        idea_digest=idea_digest_cid,
        intent_cid=intent_cid,
        catalog_cid=catalog_cid,
    )


def qualify_current_head_reference_high_level_objective(
    start: Path | None = None,
) -> ReferenceHighLevelObjectiveVerdict:
    root = discover_accelerate_root(start)
    objective = render_declared_reference_objective(root)
    catalog = platform_objective_catalog(start)
    return qualify_reference_high_level_objective(
        current_head_static_probes(start),
        objective_cid=str(objective["objective_cid"]),
        idea_digest_cid=str(objective["idea_digest"]),
        intent_cid=str(objective["intent"]["intent_cid"]),
        catalog_cid=str(catalog["catalog_cid"]),
    )


def pcpr_060_receipt_promotion(
    verdict: ReferenceHighLevelObjectiveVerdict,
) -> dict[str, Any]:
    if verdict.closed_release_outcome is not None:
        raise ReferenceHighLevelObjectiveError(
            "reference objective must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise ReferenceHighLevelObjectiveError(
            "reference objective must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise ReferenceHighLevelObjectiveError(
            "reference-objective completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise ReferenceHighLevelObjectiveError(
            "reference objective must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise ReferenceHighLevelObjectiveError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.live_objective_submission or verdict.live_materialization:
        raise ReferenceHighLevelObjectiveError(
            "live objective submission requires measured_live evidence"
        )
    if verdict.live_start:
        raise ReferenceHighLevelObjectiveError("START was not performed")
    return verdict.to_mapping()


# Pinned after the encoder is measured. Drift is a remint.
PINNED_IDEA_DIGEST: Final = (
    "baguqeeracbayojdov4jmqiirx22pavrg6nabazcocru6y3scrdx5e54mw2zq"
)
PINNED_OBJECTIVE_CID: Final = (
    "baguqeeraynsn7tjr3iaggnreylzxo3akaooqwp5bf5oheaa6eubth2zwqeba"
)
PINNED_INTENT_CID: Final = (
    "baguqeeraa4tf7chcljsvpbljkswijpyq44aks2dq7zzjhos376f6rkjed3ra"
)
PINNED_CATALOG_CID: Final = (
    "baguqeera2nvt3yvomxdm2s2uacs5d2oi4sh4vvzalvxkl26d53cbjl4fy6pa"
)
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeerac76u7uspwafsoc5oflvssxg6vdwbijmuahpytcxdmel6wror7i4q"
)


__all__ = [
    "OBJECTIVE_KIND",
    "CLOSED_RELEASE_OUTCOMES",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "HERMETIC_CANDIDATE_SUITES",
    "INTERFACE",
    "OPERATOR_BLOCKING_TASK_ID",
    "OutcomeProbe",
    "PCPR_060_GOAL_ID",
    "PCPR_060_TASK_ID",
    "PINNED_CATALOG_CID",
    "PINNED_IDEA_DIGEST",
    "PINNED_INTENT_CID",
    "PINNED_OBJECTIVE_CID",
    "REFERENCE_OBJECTIVE_IDEA",
    "ReferenceHighLevelObjectiveError",
    "ReferenceHighLevelObjectiveVerdict",
    "SCHEMA",
    "SEALED_PATH",
    "SEALED_PYTHON",
    "admit_supervisor_objective_intent",
    "current_head_static_probes",
    "idea_digest",
    "pcpr_060_receipt_promotion",
    "platform_objective_catalog",
    "qualify_current_head_reference_high_level_objective",
    "qualify_reference_high_level_objective",
    "refuse_idea_digest_remint",
    "refuse_objective_remint",
    "render_declared_reference_objective",
    "render_supervisor_objective_intent",
    "verify_reference_objective_files",
    "write_reference_objective_files",
]
