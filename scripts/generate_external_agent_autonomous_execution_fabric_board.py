#!/usr/bin/env python3
"""Generate the canonical ExternalAgentAutonomousExecutionFabric board.

The compact task declarations below are expanded into both a canonical JSON
board and a human-readable Markdown projection.  Every expanded task carries
the complete supervisor work-packet fields required by the campaign.  The
source and compatibility manifests are inputs, so the board cannot be
regenerated against an unreviewed checkout or a mutable branch name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / "docs/architecture/external_agent_autonomous_execution_fabric"
SOURCE_MANIFEST = CAMPAIGN_DIR / "source_reconciliation_manifest.json"
STACK_MANIFEST = CAMPAIGN_DIR / "stack_compatibility_manifest.json"
JSON_BOARD = CAMPAIGN_DIR / "task_board.json"
MARKDOWN_BOARD = CAMPAIGN_DIR / "TASK_BOARD.md"
OBJECTIVES = CAMPAIGN_DIR / "OBJECTIVES.md"

BOARD_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-autonomous-execution-fabric-board@1"
)
TASK_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-autonomous-execution-fabric-task@1"
)
GOAL_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-autonomous-execution-fabric-goal@1"
)
BOARD_NAMESPACE = "external-agent-autonomous-execution-fabric-v1"
PLAN_REVISION = "EAAEF-PLAN-R1"
CONTROL_SCHEMA = "datasets-authoritative-operational-v1"
ROOT_GOAL = "EAAEF-G000"
OVERLAP_CONTRACT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-owned-path-overlap-contract@1"
)
OVERLAP_STRATEGY = "serialized_forward_extension"
OVERLAP_MERGE_LANE = "single_admitted_merge_lane"
CONTROL_ARTIFACT_OWNERSHIP_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-control-artifact-ownership@1"
)
CONTROL_ARTIFACT_OWNERSHIP: tuple[dict[str, Any], ...] = (
    {
        "schema": CONTROL_ARTIFACT_OWNERSHIP_SCHEMA,
        "path": "docs/architecture/external_agent_autonomous_execution_fabric/OBJECTIVES.md",
        "ownership_class": "generator_owned_projection",
        "owner": "scripts/generate_external_agent_autonomous_execution_fabric_board.py",
        "mutation_policy": "regenerate_from_reviewed_inputs_only",
        "worker_mutation_admitted": False,
    },
    {
        "schema": CONTROL_ARTIFACT_OWNERSHIP_SCHEMA,
        "path": "docs/architecture/external_agent_autonomous_execution_fabric/PLAN.md",
        "ownership_class": "reviewed_source_owned_control_document",
        "owner": "campaign_source_review",
        "mutation_policy": "reviewed_source_change_only",
        "worker_mutation_admitted": False,
    },
    {
        "schema": CONTROL_ARTIFACT_OWNERSHIP_SCHEMA,
        "path": "docs/architecture/external_agent_autonomous_execution_fabric/TASK_BOARD.md",
        "ownership_class": "generator_owned_projection",
        "owner": "scripts/generate_external_agent_autonomous_execution_fabric_board.py",
        "mutation_policy": "regenerate_from_reviewed_inputs_only",
        "worker_mutation_admitted": False,
    },
    {
        "schema": CONTROL_ARTIFACT_OWNERSHIP_SCHEMA,
        "path": "docs/architecture/external_agent_autonomous_execution_fabric/task_board.json",
        "ownership_class": "generator_owned_canonical_board",
        "owner": "scripts/generate_external_agent_autonomous_execution_fabric_board.py",
        "mutation_policy": "regenerate_from_reviewed_inputs_only",
        "worker_mutation_admitted": False,
    },
    {
        "schema": CONTROL_ARTIFACT_OWNERSHIP_SCHEMA,
        "path": "docs/architecture/external_agent_autonomous_execution_fabric/stack_compatibility_manifest.json",
        "ownership_class": "reviewed_source_owned_board_input",
        "owner": "campaign_source_review",
        "mutation_policy": "superseding_reviewed_revision_only",
        "worker_mutation_admitted": False,
    },
    {
        "schema": CONTROL_ARTIFACT_OWNERSHIP_SCHEMA,
        "path": "docs/architecture/external_agent_autonomous_execution_fabric/source_reconciliation_manifest.json",
        "ownership_class": "reviewed_source_owned_board_input",
        "owner": "campaign_source_review",
        "mutation_policy": "reviewed_source_change_only",
        "worker_mutation_admitted": False,
    },
    {
        "schema": CONTROL_ARTIFACT_OWNERSHIP_SCHEMA,
        "path": "docs/architecture/external_agent_autonomous_execution_fabric/reconciliation_report.md",
        "ownership_class": "reviewed_source_owned_human_projection",
        "owner": "campaign_source_review",
        "mutation_policy": "reviewed_source_change_only",
        "worker_mutation_admitted": False,
    },
    {
        "schema": CONTROL_ARTIFACT_OWNERSHIP_SCHEMA,
        "path": "docs/architecture/external_agent_autonomous_execution_fabric/bootstrap_materialization_attempts.json",
        "ownership_class": "reviewed_source_owned_evidence_ledger",
        "owner": "campaign_source_review",
        "mutation_policy": "reviewed_append_only_attempt_evidence",
        "worker_mutation_admitted": False,
    },
)


def _cid(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


# goal id, epic, title, predecessor goal, contract
EPICS: tuple[tuple[str, str, str, str, str], ...] = (
    ("EAAEF-G010", "A", "Unmerged-work reconciliation and release baseline", "", "All relevant refs and dirty overlays are classified, reviewed integration roots are immutable, and StackCompatibilityManifest@1 binds the exact cross-package stack."),
    ("EAAEF-G020", "B", "External agent-session handoff protocol", "EAAEF-G010", "Raw client exports and a bounded normalized event stream are durably preserved with separate identities, provenance, trust labels, privacy and retention."),
    ("EAAEF-G030", "C", "Complete Git repository transfer", "EAAEF-G020", "Every admitted repository state is reconstructed in quarantine and verified across Git objects, dirty overlays, submodules, LFS, modes, symlinks and transfer bounds."),
    ("EAAEF-G040", "D", "Caller identity, capability and disclosure policy", "EAAEF-G030", "Effect-bound authenticated authority is distinct from prompts, CIDs, transport identity and imported history; disclosure and approvals bind exact inputs."),
    ("EAAEF-G050", "E", "Project onboarding and codebase classification", "EAAEF-G040", "Every ordinary Git repository receives a typed assessment; autonomous mutation is admitted only through a qualified ProjectAdapter and known validation profile."),
    ("EAAEF-G060", "F", "OCI container execution fabric", "EAAEF-G050", "Workers execute only leased tasks in isolated containers with bounded resources, no Docker socket, default-deny network and restart-safe checkpoints; the engine is rootless where supported, otherwise an independently approved rootful-host-daemon/nonroot-worker fallback is required."),
    ("EAAEF-G070", "G", "Handoff context and federated retrieval", "EAAEF-G060", "Repository truth, imported claims, receipts, legal corpora and hypotheses remain distinct while AST, capsules, BM25, vector, GraphRAG and knowledge graphs compose through one provenance-preserving retrieval plan."),
    ("EAAEF-G080", "H", "Logic-governed goal and task compilation", "EAAEF-G070", "The existing logic platform admits only covered, acyclic, bounded, feasible goal/task plans with explicit conflicts, proof obligations and completion contracts."),
    ("EAAEF-G090", "I", "Conflict-free multi-agent parallel execution", "EAAEF-G080", "The existing semantic work fabric selects fenced conflict-free frontiers; multiple attempts are allowed but one logical result alone may be accepted."),
    ("EAAEF-G100", "J", "Production DuckDB, Quack and DuckLake plane", "EAAEF-G090", "DuckDB plus one fenced authenticated Quack owner form the sole mutable coordination plane; DuckLake and immutable artifacts provide non-authoritative history, analytics, lineage and recovery."),
    ("EAAEF-G110", "K", "Closed-loop execution and adaptive replanning", "EAAEF-G100", "Every accepted result refreshes source and semantic state, invalidates stale evidence, revises the immutable plan and converges to a bounded fixed point."),
    ("EAAEF-G120", "L", "Python, CLI, MCP and MCP++ surfaces", "EAAEF-G110", "All transports expose one semantic operation set and canonical identity, support detach/reconnect/cursors, and use only existing MCP++ profiles where a shared contract is required."),
    ("EAAEF-G130", "M", "Security hardening", "EAAEF-G120", "Hostile repositories and histories cannot widen policy, escape containers, forge verification, expose secrets or acquire mutation authority."),
    ("EAAEF-G140", "N", "Observability and accounting", "EAAEF-G130", "Typed, privacy-safe events and resource/cost metrics make every run observable, explainable, steerable and auditable without publishing sensitive bodies."),
    ("EAAEF-G150", "O", "End-to-end and fault qualification", "EAAEF-G140", "Real client, supervisor, worker, Quack, DuckDB, DuckLake, network and crash fixtures demonstrate safe recovery and evidence-backed terminal outcomes."),
    ("EAAEF-G160", "P", "Performance and parallelism benchmark", "EAAEF-G150", "Configurations A through D are measured honestly; missed targets remain reported and historical or simulated results never count as current qualification."),
    ("EAAEF-G170", "Q", "Packaging and external deployment", "EAAEF-G160", "Clean wheels and digest-pinned OCI images install without sibling checkouts or editable installs and ship locks, SBOMs, migrations, backup, restore and rollback."),
    ("EAAEF-G180", "R", "Blocking CI and qualification release", "EAAEF-G170", "Every required lane is blocking and current; the release emits a narrow evidence-backed qualification level and explicit go/no-go recommendation."),
)


def _paths(*items: str) -> tuple[str, ...]:
    return items


REPOSITORY_EXECUTION_PREFIXES = {
    "ipfs_accelerate_py": "",
    "ipfs_datasets_py": "ipfs_datasets_py",
    "ipfs_kit_py": "ipfs_kit_py",
    "Mcp-Plus-Plus": "ipfs_accelerate_py/mcplusplus",
}


def _execution_paths(repository: str, paths: Iterable[str]) -> list[str]:
    """Project repository-local ownership into the supervisor superproject."""

    prefix = REPOSITORY_EXECUTION_PREFIXES[repository]
    if not prefix:
        return _as_list(paths)
    return [f"{prefix}/{path}" for path in paths]


def _integration_conflict_keys(repository: str) -> list[str]:
    prefix = REPOSITORY_EXECUTION_PREFIXES[repository]
    return [] if not prefix else [f"serialized-superproject-gitlink:{prefix}"]


def _structured_execution_validation(
    repository: str, validation: str
) -> list[dict[str, Any]]:
    """Render repo-local commands as bounded argv plus an explicit root-relative cwd."""

    working_directory = REPOSITORY_EXECUTION_PREFIXES[repository] or "."
    return [
        {
            "working_directory": working_directory,
            "argv": shlex.split(command.strip()),
        }
        for command in validation.split(";")
        if command.strip()
    ]


# id, goal, title, repository, owned files, explicit dependencies, objective,
# focused validation.  The generator supplies every remaining required field.
TASK_ROWS: tuple[
    tuple[str, str, str, str, tuple[str, ...], tuple[str, ...], str, str], ...
] = (
    (
        "000",
        "EAAEF-G010",
        "Admit the fail-closed bootstrap runtime",
        "ipfs_accelerate_py",
        _paths(
            "ipfs_accelerate_py/agent_implementation_route.py",
            "ipfs_accelerate_py/llm_router.py",
            "ipfs_accelerate_py/agent_supervisor/analysis/contract_mismatch_analyzer.py",
            "ipfs_accelerate_py/agent_supervisor/analysis/contract_vulnerability_rules.py",
            "ipfs_accelerate_py/agent_supervisor/analysis/mcp_contract_catalog.py",
            "ipfs_accelerate_py/agent_supervisor/analysis/mcp_invocation_trace.py",
            "ipfs_accelerate_py/agent_supervisor/analysis/parser_failure_triage.py",
            "ipfs_accelerate_py/agent_supervisor/analysis/polyglot_ast_health.py",
            "ipfs_accelerate_py/agent_supervisor/analysis/polyglot_ast_provider.py",
            "ipfs_accelerate_py/agent_supervisor/analysis/python_mcp_surface_extractor.py",
            "ipfs_accelerate_py/agent_supervisor/analysis/runtime_component_catalog.py",
            "ipfs_accelerate_py/agent_supervisor/analysis/runtime_contract_evidence_compiler.py",
            "ipfs_accelerate_py/agent_supervisor/analysis/swissknife_contract_extractor.py",
            "ipfs_accelerate_py/agent_supervisor/control/plan_execution_store.py",
            "ipfs_accelerate_py/agent_supervisor/control/profile_authority.py",
            "ipfs_accelerate_py/agent_supervisor/entrypoints/inference_runtime.py",
            "ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_logic_provider.py",
            "ipfs_accelerate_py/agent_supervisor/merge/merge_train.py",
            "ipfs_accelerate_py/agent_supervisor/merge/worktree_lifecycle.py",
            "ipfs_accelerate_py/agent_supervisor/objectives/backlog_refinery.py",
            "ipfs_accelerate_py/agent_supervisor/objectives/objective_graph.py",
            "ipfs_accelerate_py/agent_supervisor/proof/mcp_contract_proof_cache.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/connect_allowlist_proxy.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/eaaef_bootstrap_gateway.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/external_agent_control_plane_promotion.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/plan_r2_remote_owner.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/worker_container_execution_profile.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/worker_network.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/worker_network_dispatch.py",
            "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_repository.py",
            "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_transactions.py",
            "ipfs_accelerate_py/agent_supervisor/task_sources/eaaef_borrowed_transaction.py",
            "ipfs_accelerate_py/agent_supervisor/task_sources/eaaef_operational_schema.py",
            "ipfs_accelerate_py/agent_supervisor/task_sources/external_agent_state_repository.py",
            "ipfs_accelerate_py/agent_supervisor/task_sources/persistent_task_queue.py",
            "ipfs_accelerate_py/agent_supervisor/task_sources/quack_command_authorization.py",
            "ipfs_accelerate_py/agent_supervisor/task_sources/quack_command_fabric.py",
            "ipfs_accelerate_py/agent_supervisor/task_sources/eaaef_bootstrap_daemon_gateway.py",
            "ipfs_accelerate_py/agent_supervisor/task_sources/quack_daemon_gateway.py",
            "ipfs_accelerate_py/agent_supervisor/task_sources/quack_state_client.py",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/external_agent_container_dispatcher.py",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor_runner.py",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/worktrees.py",
            "ipfs_accelerate_py/agent_supervisor/validation/agent_native_dependency_admission.py",
            "ipfs_accelerate_py/agent_supervisor/validation/eaaef_bootstrap_gateway_launch.py",
            "ipfs_accelerate_py/agent_supervisor/validation/eaaef_lane_gateway_admission.py",
            "ipfs_accelerate_py/agent_supervisor/validation/external_agent_fabric_bootstrap.py",
            "ipfs_accelerate_py/agent_supervisor/validation/external_agent_bootstrap_admission.py",
            "ipfs_accelerate_py/agent_supervisor/validation/external_agent_configured_board_capsule.py",
            "ipfs_accelerate_py/agent_supervisor/validation/plan_r2_remote_owner_admission.py",
            "ipfs_accelerate_py/agent_supervisor/validation/proof_cached_test_validation.py",
            "ipfs_accelerate_py/agent_supervisor/validation/validation_runtime.py",
            "ipfs_accelerate_py/testing/proof_reuse/default_identity_services.py",
            "ipfs_accelerate_py/testing/proof_reuse/item_identity.py",
            "containers/external-agent/bootstrap-reconciliation.Containerfile",
            "containers/external-agent/implementation-worker.Containerfile",
            "containers/external-agent/implementation-worker-minimal.Containerfile",
            "config/external_agent_autonomous_execution_fabric_bootstrap.json",
            "config/external_agent_autonomous_execution_fabric_scheduler.json",
            "scripts/generate_external_agent_autonomous_execution_fabric_board.py",
            "scripts/extract_typescript_ast.mjs",
            "scripts/launch_external_agent_autonomous_execution_fabric_materializer.py",
            "scripts/materialize_external_agent_autonomous_execution_fabric_control_plane.py",
            "scripts/qualify_external_agent_bootstrap_container.py",
            "scripts/qualify_external_agent_implementation_worker_image.py",
            "scripts/qualify_external_agent_implementation_worker_minimal_image.py",
            "scripts/validate_external_agent_autonomous_execution_fabric_board.py",
            "test/api/test_agent_supervisor_configured_board_scheduler.py",
            "test/api/test_agent_supervisor_contract_mismatch_analyzer.py",
            "test/api/test_agent_supervisor_contract_vulnerability_rules.py",
            "test/api/test_agent_supervisor_database_implementation_daemon.py",
            "test/api/test_agent_supervisor_grok_quota_terra_gate.py",
            "test/api/test_agent_supervisor_implementation_daemon_runner.py",
            "test/api/test_agent_supervisor_implementation_supervisor_authority_forwarding.py",
            "test/api/test_agent_supervisor_implementation_supervisor_runner.py",
            "test/api/test_agent_supervisor_incremental_runtime.py",
            "test/api/test_agent_supervisor_inference_runtime.py",
            "test/api/test_agent_supervisor_ipfs_datasets_logic_provider.py",
            "test/api/test_agent_supervisor_mcp_contract_catalog.py",
            "test/api/test_agent_supervisor_mcp_contract_proof_cache.py",
            "test/api/test_agent_supervisor_mcp_invocation_trace.py",
            "test/api/test_agent_supervisor_native_dependency_admission.py",
            "test/api/test_agent_supervisor_parser_failure_triage.py",
            "test/api/test_agent_supervisor_polyglot_ast_health.py",
            "test/api/test_agent_supervisor_polyglot_ast_provider.py",
            "test/api/test_agent_supervisor_prompt_v3_resolution_hardening.py",
            "test/api/test_agent_supervisor_proof_merge_gate.py",
            "test/api/test_agent_supervisor_proof_cached_test_validation.py",
            "test/api/test_agent_supervisor_python_mcp_surface_extractor.py",
            "test/api/test_agent_supervisor_quack_command_fabric.py",
            "test/api/test_agent_supervisor_quack_daemon_gateway.py",
            "test/api/test_agent_supervisor_quack_state_client.py",
            "test/api/test_agent_supervisor_quack_state_server.py",
            "test/api/test_agent_supervisor_runtime_component_catalog.py",
            "test/api/test_agent_supervisor_runtime_contract_evidence_compiler.py",
            "test/api/test_agent_supervisor_router_owned_provider_decision.py",
            "test/api/test_agent_supervisor_swissknife_contract_extractor.py",
            "test/api/test_agent_supervisor_todo_daemon_port.py",
            "test/api/test_agent_supervisor_validation_scheduler.py",
            "test/api/test_agent_supervisor_worktree_lifecycle.py",
            "test/api/test_eaaef_bootstrap_daemon_gateway.py",
            "test/api/test_eaaef_bootstrap_gateway_launch.py",
            "test/api/test_eaaef_bootstrap_runtime_gateway.py",
            "test/api/test_eaaef_borrowed_transaction.py",
            "test/api/test_eaaef_lane_gateway_runtime.py",
            "test/api/test_eaaef_operational_schema.py",
            "test/api/test_eaaef_quack_command_fabric.py",
            "test/api/test_eaaef_supervisor_daemon_birth_wiring.py",
            "test/api/test_external_agent_autonomous_execution_fabric_board.py",
            "test/api/test_external_agent_autonomous_execution_fabric_materializer.py",
            "test/api/test_external_agent_bootstrap_admission.py",
            "test/api/test_external_agent_configured_board_capsule.py",
            "test/api/test_external_agent_configured_board_runner_gate.py",
            "test/api/test_external_agent_control_plane_promotion.py",
            "test/api/test_external_agent_container_worker_dispatch.py",
            "test/api/test_external_agent_fabric_bootstrap_preflight.py",
            "test/api/test_external_agent_fabric_container_qualification.py",
            "test/api/test_external_agent_fabric_provider_authorization.py",
            "test/api/test_external_agent_implementation_worker_image_qualification.py",
            "test/api/test_external_agent_implementation_worker_minimal_image_qualification.py",
            "test/api/test_external_agent_state_repository.py",
            "test/api/test_external_agent_worker_authority_propagation.py",
            "test/api/test_external_agent_worker_network.py",
            "test/api/test_llm_router_agent_implementation_route.py",
            "test/api/test_llm_router_agent_supervisor_fallback_route.py",
            "test/api/test_llm_router_exact_provider_fallback.py",
            "test/api/test_proof_reuse_default_identity_services.py",
            "test/api/test_pytest_proof_reuse_item_identity.py",
            "test/api/test_plan_r2_remote_owner.py",
        ),
        (),
        "Independently bind signed EAAEF provider authorization, an exact task-capable worker image and SBOM, effect-bound per-attempt network approval, an explicitly identified rootless engine or independently approved rootful-host-daemon/nonroot-worker fallback, bounded internal proxy egress, the immutable materialization receipt, and an exact DuckDB 1.5.5/Quack 1.5.5 command-ingress qualification. The Quack owner must verify signed principal/authority/lease/deadline/fence envelopes before its private DuckDB mutation; a bare StateCommand or shared token is never authority. Reviewed source implements the native-dependency admission, V2 lane/verifier/merge chain, exact-envelope journals, lazy Quack/dispatcher factories, per-birth supervisor wiring and the distinct three-operation process-remote Plan-R2 owner seam, but those source seams confer no live authority. Before EAAEF-001 may run, the manual host gate must qualify the exact 31-operation task/claim/lease/provider/effect/validation/completion bootstrap vocabulary used by EAAEF-001 through EAAEF-009; the existing task.get/task.ready handlers and unrelated task.list read are explicitly insufficient. The capability excludes task materialization, host merge admission and the three Plan-R2 operations. Host merge remains independently reviewed, while EAAEF-009 uses a separately promoted prepare/apply/observe Plan-R2 gateway after EAAEF-008. Quack-mode daemons must use that complete bootstrap gateway and may not open local execution or coordination sidecars. Missing actual independently signed native, lane, Quack, dispatcher and Plan-R2 artifacts, deployed signed endpoints, a qualified extension, or admitted container/provider/network authority must emit a typed no-go and start no supervisor.",
        "python3 -m pytest -q test/api/test_agent_supervisor_configured_board_scheduler.py test/api/test_agent_supervisor_contract_mismatch_analyzer.py test/api/test_agent_supervisor_contract_vulnerability_rules.py test/api/test_agent_supervisor_database_implementation_daemon.py test/api/test_agent_supervisor_grok_quota_terra_gate.py test/api/test_agent_supervisor_implementation_daemon_runner.py test/api/test_agent_supervisor_implementation_supervisor_authority_forwarding.py test/api/test_agent_supervisor_implementation_supervisor_runner.py test/api/test_agent_supervisor_incremental_runtime.py test/api/test_agent_supervisor_inference_runtime.py test/api/test_agent_supervisor_ipfs_datasets_logic_provider.py test/api/test_agent_supervisor_mcp_contract_catalog.py test/api/test_agent_supervisor_mcp_contract_proof_cache.py test/api/test_agent_supervisor_mcp_invocation_trace.py test/api/test_agent_supervisor_native_dependency_admission.py test/api/test_agent_supervisor_parser_failure_triage.py test/api/test_agent_supervisor_polyglot_ast_health.py test/api/test_agent_supervisor_polyglot_ast_provider.py test/api/test_agent_supervisor_prompt_v3_resolution_hardening.py test/api/test_agent_supervisor_proof_merge_gate.py test/api/test_agent_supervisor_proof_cached_test_validation.py test/api/test_agent_supervisor_python_mcp_surface_extractor.py test/api/test_agent_supervisor_quack_command_fabric.py test/api/test_agent_supervisor_quack_daemon_gateway.py test/api/test_agent_supervisor_quack_state_client.py test/api/test_agent_supervisor_quack_state_server.py test/api/test_agent_supervisor_runtime_component_catalog.py test/api/test_agent_supervisor_runtime_contract_evidence_compiler.py test/api/test_agent_supervisor_router_owned_provider_decision.py test/api/test_agent_supervisor_swissknife_contract_extractor.py test/api/test_agent_supervisor_todo_daemon_port.py test/api/test_agent_supervisor_validation_scheduler.py test/api/test_agent_supervisor_worktree_lifecycle.py test/api/test_eaaef_bootstrap_daemon_gateway.py test/api/test_eaaef_bootstrap_gateway_launch.py test/api/test_eaaef_bootstrap_runtime_gateway.py test/api/test_eaaef_borrowed_transaction.py test/api/test_eaaef_lane_gateway_runtime.py test/api/test_eaaef_operational_schema.py test/api/test_eaaef_quack_command_fabric.py test/api/test_eaaef_supervisor_daemon_birth_wiring.py test/api/test_external_agent_fabric_bootstrap_preflight.py test/api/test_external_agent_fabric_container_qualification.py test/api/test_external_agent_fabric_provider_authorization.py test/api/test_external_agent_implementation_worker_image_qualification.py test/api/test_external_agent_implementation_worker_minimal_image_qualification.py test/api/test_external_agent_state_repository.py test/api/test_external_agent_worker_network.py test/api/test_external_agent_worker_authority_propagation.py test/api/test_external_agent_container_worker_dispatch.py test/api/test_external_agent_bootstrap_admission.py test/api/test_external_agent_control_plane_promotion.py test/api/test_external_agent_configured_board_capsule.py test/api/test_external_agent_configured_board_runner_gate.py test/api/test_external_agent_autonomous_execution_fabric_materializer.py test/api/test_llm_router_agent_implementation_route.py test/api/test_llm_router_agent_supervisor_fallback_route.py test/api/test_llm_router_exact_provider_fallback.py test/api/test_plan_r2_remote_owner.py test/api/test_proof_reuse_default_identity_services.py test/api/test_pytest_proof_reuse_item_identity.py test/api/test_external_agent_autonomous_execution_fabric_board.py",
    ),
    ("001", "EAAEF-G010", "Verify the complete source-reconciliation manifest", "ipfs_accelerate_py", _paths("docs/architecture/external_agent_autonomous_execution_fabric/receipts/source_reconciliation_verification.json", "test/api/test_external_agent_source_reconciliation.py"), ("EAAEF-000",), "Reproduce the frozen manifest and human report's branch, worktree, changed-path, schema/API, test, dependency, supersession, dirty-overlay and conflict classifications for all four repositories without mutating either board input or any preserved ref.", "python3 -m pytest -q test/api/test_external_agent_source_reconciliation.py; python3 scripts/validate_external_agent_autonomous_execution_fabric_board.py --source-only"),
    ("002", "EAAEF-G010", "Reconcile accelerator residual lineages and source hygiene", "ipfs_accelerate_py", _paths(".gitignore", "dashboard.pid", "data/model_manager.duckdb.wal", "state/p2p_gpt2_2peer/peer1_queue.duckdb.wal", "state/p2p_gpt2_2peer/peer2_queue.duckdb.wal", "state/smoketest_logs/driver.out", "state/tls/mcpplusplus.crt", "state/tls/mcpplusplus.key", "test/kitchen_sink_models.db.wal", "scripts/systemd/generate_self_signed_cert.py", "docs/architecture/external_agent_autonomous_execution_fabric/reconciliation/accelerator.json", "test/api/test_external_agent_source_hygiene.py"), ("EAAEF-000",), "Forward-audit DCR Git authority/replay, self-hosting recovery and task-contract residuals; verify that committed runtime and private-key-shaped artifacts were forward-removed while their provenance remains in history, require fail-closed 0600 permissions for replacement private keys, and port only behavior missing from the reviewed baseline.", "python3 -m pytest -q test/api/test_external_agent_source_hygiene.py test/api/test_agent_supervisor_checkout_lock.py"),
    ("003", "EAAEF-G010", "Reconcile datasets UI IR and proof-reuse residuals", "ipfs_datasets_py", _paths("docs/architecture/external_agent_fabric_reconciliation.json", "tests/integration/test_external_agent_reconciliation.py"), ("EAAEF-000",), "Verify the provenance-preserving UI/UX-IR merge, retain current LPC API semantics, classify proof-reuse and semantic-contract residuals, and reject wholesale stale snapshots.", "python -m pytest -q tests/integration/test_external_agent_reconciliation.py"),
    ("004", "EAAEF-G010", "Freeze the ipfs_kit_py reusable authority surface", "ipfs_kit_py", _paths("docs/external_agent_fabric_kit_contracts.md", "tests/test_external_agent_fabric_kit_contracts.py"), ("EAAEF-000",), "Bind existing artifact, semantic-root, proof-sealer and MCP++ adapter surfaces while excluding the in-process Profile-G coordinator from production authority.", "python -m pytest -q tests/test_external_agent_fabric_kit_contracts.py"),
    ("005", "EAAEF-G010", "Clarify existing MCP++ state-backend roles", "Mcp-Plus-Plus", _paths("docs/architecture/decisions/0004-state-modes.md", "docs/architecture/decisions/0005-durable-executor.md", "docs/architecture/durable-execution.md", "docs/architecture/state-model.md", "docs/spec/state-ref.md"), ("EAAEF-000",), "Preserve existing wire schemas while stating that DuckDB is transactional storage, Quack is the sole fenced multi-reader/multi-writer owner boundary, and DuckLake is non-authoritative history.", "python -m pytest -q tests/test_state_ref.py tests/test_durable_executor.py"),
    ("006", "EAAEF-G010", "Propose reviewed integration roots and stack compatibility", "ipfs_accelerate_py", _paths("docs/architecture/external_agent_autonomous_execution_fabric/proposals/stack_compatibility_manifest.r2.json", "docs/architecture/external_agent_autonomous_execution_fabric/receipts/stack_compatibility_verification.json", "test/api/test_external_agent_stack_compatibility.py"), ("EAAEF-001", "EAAEF-002", "EAAEF-003", "EAAEF-004", "EAAEF-005"), "Bind post-reconciliation commits, trees, schemas, package/protocol versions, the admitted bootstrap OCI identity and compatible ranges in a proposal plus verification receipt; never overwrite the R1 compatibility input in place.", "python3 -m pytest -q test/api/test_external_agent_stack_compatibility.py"),
    ("007", "EAAEF-G010", "Build the canonical multi-repository semantic root", "ipfs_datasets_py", _paths("ipfs_datasets_py/analysis/external_agent_source_state.py", "tests/unit/analysis/test_external_agent_source_state.py"), ("EAAEF-006",), "Build and independently verify the post-reconciliation AST, semantic-state and provenance root for the exact four-repository forest, preserving a content-addressed delta and invalidation receipt.", "python -m pytest -q tests/unit/analysis/test_external_agent_source_state.py"),
    ("008", "EAAEF-G010", "Renew Quack ownership and the accepted bootstrap capsule", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/runtime/external_agent_control_plane_promotion.py", "ipfs_accelerate_py/agent_supervisor/validation/external_agent_configured_board_capsule.py", "test/api/test_external_agent_configured_board_capsule.py", "test/api/test_external_agent_configured_board_runner_gate.py", "test/api/test_external_agent_control_plane_promotion.py"), ("EAAEF-007",), "Reverify the bootstrap DuckDB/Quack owner and issue or reject a fresh immutable configured-board capsule against the reconciled source forest and semantic root. The capsule binds one private file-opening owner, authenticated signed-command ingress, read-only projections, the current epoch/fence, exact conflict-free frontier and no direct-file fallback; it cannot authorize Plan R2 before its distinct independently reviewed Promotion@2 transition capability and receipt exist.", "python3 -m pytest -q test/api/test_external_agent_configured_board_capsule.py test/api/test_external_agent_configured_board_runner_gate.py test/api/test_external_agent_control_plane_promotion.py"),
    ("009", "EAAEF-G010", "Admit Plan R2 and transition the next population", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/planning/external_agent_plan_r2.py", "ipfs_accelerate_py/agent_supervisor/runtime/plan_r2_remote_owner.py", "ipfs_accelerate_py/agent_supervisor/task_sources/external_agent_state_repository.py", "ipfs_accelerate_py/agent_supervisor/validation/plan_r2_remote_owner_admission.py", "test/api/test_external_agent_plan_r2.py", "test/api/test_external_agent_state_repository.py", "test/api/test_plan_r2_remote_owner.py", "docs/architecture/external_agent_autonomous_execution_fabric/plan_revisions/README.md"), ("EAAEF-007", "EAAEF-008", "EAAEF-000"), "Consume the verified semantic root and promotion receipt through the distinct process-remote three-operation Plan-R2 owner seam, create an immutable Plan R2 with bounded add/supersede repairs, replace every future-task sentinel, CAS the active plan revision, materialize only the B frontier, and emit an admission or typed no-go receipt without editing completed R1 tasks. Source implementation is not live admission; the independently signed remote-owner capability, qualified wire-channel factory and supervisor repository wiring remain external requirements.", "python3 -m pytest -q test/api/test_external_agent_plan_r2.py test/api/test_external_agent_state_repository.py test/api/test_plan_r2_remote_owner.py"),

    ("010", "EAAEF-G020", "Define the transport-neutral handoff contract family", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/handoff/contracts.py", "test/api/test_external_agent_handoff_contracts.py"), (), "Implement content-addressed ExternalAgentHandoffRequest, Session, event, checkpoint, context, normalization and admission schemas with strict versioning and bounds.", "python3 -m pytest -q test/api/test_external_agent_handoff_contracts.py"),
    ("011", "EAAEF-G020", "Preserve encrypted raw exports and normalized projections", "ipfs_kit_py", _paths("ipfs_kit_py/external_agent_handoff/storage.py", "tests/test_external_agent_handoff_storage.py"), ("EAAEF-010",), "Store exact exported bytes through managed encrypted references and emit a separate ordered normalized projection without transcript bodies in public receipts.", "python -m pytest -q tests/test_external_agent_handoff_storage.py"),
    ("012", "EAAEF-G020", "Implement the Codex export adapter", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/handoff/adapters/codex.py", "test/api/test_external_agent_codex_adapter.py"), ("EAAEF-010",), "Normalize legitimately exportable Codex messages, tool calls/results, patches and explicit reasoning summaries without requesting hidden chain-of-thought or trusting success claims.", "python3 -m pytest -q test/api/test_external_agent_codex_adapter.py"),
    ("013", "EAAEF-G020", "Implement the Claude Code export adapter", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/handoff/adapters/claude_code.py", "test/api/test_external_agent_claude_adapter.py"), ("EAAEF-010",), "Detect supported Claude Code export versions, preserve branches and residual fields, and reject ambiguous/truncated authority claims.", "python3 -m pytest -q test/api/test_external_agent_claude_adapter.py"),
    ("014", "EAAEF-G020", "Implement Gemini CLI and generic JSON/MCP adapters", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/handoff/adapters/gemini_cli.py", "ipfs_accelerate_py/agent_supervisor/handoff/adapters/generic.py", "test/api/test_external_agent_generic_adapters.py"), ("EAAEF-010",), "Normalize supported Gemini CLI, generic MCP and documented JSON/JSONL exports while never executing imported calls and retaining bounded unknown fields.", "python3 -m pytest -q test/api/test_external_agent_generic_adapters.py"),
    ("015", "EAAEF-G020", "Qualify handoff trust, identity, privacy and retention", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/handoff/admission.py", "test/api/test_external_agent_handoff_admission.py"), ("EAAEF-011", "EAAEF-012", "EAAEF-013", "EAAEF-014"), "Keep raw session, normalized stream, objective, context, repository and patch identities distinct; only reverified or admitted receipts may satisfy gates.", "python3 -m pytest -q test/api/test_external_agent_handoff_admission.py"),

    ("020", "EAAEF-G030", "Define complete repository handoff and overlay schemas", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/repository_handoff/contracts.py", "test/api/test_repository_handoff_contracts.py"), (), "Account for exact HEAD/refs/index/worktree/untracked/submodules/nested repos/LFS/sparse checkout/hooks/attributes/modes/origin and shallow boundaries.", "python3 -m pytest -q test/api/test_repository_handoff_contracts.py"),
    ("021", "EAAEF-G030", "Implement bounded repository transfer modes", "ipfs_kit_py", _paths("ipfs_kit_py/repository_transfer/bundle.py", "tests/test_repository_transfer_bundle.py"), ("EAAEF-020",), "Support managed aliases, Git bundles, manifested source bundles, approved remote aliases and uploaded object sets without accepting arbitrary remote host paths.", "python -m pytest -q tests/test_repository_transfer_bundle.py"),
    ("022", "EAAEF-G030", "Quarantine and verify reconstructed repositories", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/repository_handoff/quarantine.py", "test/api/test_repository_handoff_quarantine.py"), ("EAAEF-020", "EAAEF-021"), "Verify object integrity, reconstructed tree/overlay identity, URL policy, symlink safety, disabled hooks and size/object bounds before onboarding.", "python3 -m pytest -q test/api/test_repository_handoff_quarantine.py"),
    ("023", "EAAEF-G030", "Reconcile imported history with repository truth", "ipfs_datasets_py", _paths("ipfs_datasets_py/analysis/agent_history_reconciliation.py", "tests/unit/analysis/test_agent_history_reconciliation.py"), ("EAAEF-020", "EAAEF-022"), "Compare referenced commits/files/patches/tests to reconstructed truth and classify stale, present, missing and history-only work with provenance.", "python -m pytest -q tests/unit/analysis/test_agent_history_reconciliation.py"),
    ("024", "EAAEF-G030", "Qualify end-to-end repository transfer receipts", "ipfs_accelerate_py", _paths("test/integration/test_external_agent_repository_transfer.py", "docs/architecture/external_agent_autonomous_execution_fabric/receipts/repository_transfer.json"), ("EAAEF-021", "EAAEF-022", "EAAEF-023"), "Prove every transfer mode reconstructs the declared state or returns a typed refusal without mutating a user checkout.", "python3 -m pytest -q test/integration/test_external_agent_repository_transfer.py"),

    ("030", "EAAEF-G040", "Implement principals and effect-bound capability decisions", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/authority/external_principal.py", "test/api/test_external_agent_principal.py"), (), "Bind principal, repository, run, exact effects, expiry, autonomy/resource ceilings, disclosure/provider policy and bounded nonce; never infer authority from history or a CID.", "python3 -m pytest -q test/api/test_external_agent_principal.py"),
    ("031", "EAAEF-G040", "Enforce source-disclosure policy before model calls", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/authority/source_disclosure.py", "test/api/test_source_disclosure_policy.py"), ("EAAEF-030",), "Apply confidentiality, exclusions, secret scanning, provider allowlists, local-only rules, byte limits and exact ContextPack identity before disclosure.", "python3 -m pytest -q test/api/test_source_disclosure_policy.py"),
    ("032", "EAAEF-G040", "Implement exact-action approval and denial boundaries", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/authority/approvals.py", "test/api/test_external_agent_approvals.py"), ("EAAEF-030",), "Require authenticated input-bound approval for install/network/secret/disclosure/merge/push/destructive/publication effects and preserve denials.", "python3 -m pytest -q test/api/test_external_agent_approvals.py"),
    ("033", "EAAEF-G040", "Bind delegated authority to existing MCP++ profiles", "Mcp-Plus-Plus", _paths("docs/spec/external-agent-delegation.md", "tests/test_external_agent_delegation.py"), ("EAAEF-030",), "Map the external principal to existing interface/artifact/delegation/runtime/event/fencing profiles without inventing a new profile or granting backend authority.", "python -m pytest -q tests/test_external_agent_delegation.py"),
    ("034", "EAAEF-G040", "Qualify authority, disclosure and approval invariants", "ipfs_accelerate_py", _paths("test/security/test_external_agent_authority_boundaries.py", "docs/architecture/external_agent_autonomous_execution_fabric/receipts/authority.json"), ("EAAEF-031", "EAAEF-032", "EAAEF-033"), "Demonstrate that prompts, payments, commits, run IDs and transport authentication cannot widen mutation, disclosure, secret, proof-key or merge authority.", "python3 -m pytest -q test/security/test_external_agent_authority_boundaries.py"),

    ("040", "EAAEF-G050", "Implement the safe generic ProjectAdapter", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/project_adapters/base.py", "test/api/test_project_adapter_generic.py"), (), "Perform bounded read-only language/build/test/static inventory and return typed support outcomes without fabricating mutation commands.", "python3 -m pytest -q test/api/test_project_adapter_generic.py"),
    ("041", "EAAEF-G050", "Qualify the Python ProjectAdapter", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/project_adapters/python.py", "test/api/test_project_adapter_python.py"), ("EAAEF-040",), "Compile locked Python toolchain, test and static-analysis profiles into structured argv and qualify the strongest currently supported mutation path.", "python3 -m pytest -q test/api/test_project_adapter_python.py"),
    ("042", "EAAEF-G050", "Discover and admit bounded validation commands", "ipfs_datasets_py", _paths("ipfs_datasets_py/analysis/project_validation_candidates.py", "tests/unit/analysis/test_project_validation_candidates.py"), ("EAAEF-040",), "Extract README/comment/package/CI/history commands only as untrusted candidates, then require adapter allowlists and execution policy.", "python -m pytest -q tests/unit/analysis/test_project_validation_candidates.py"),
    ("043", "EAAEF-G050", "Produce the typed project support matrix", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/project_adapters/assessment.py", "docs/architecture/external_agent_autonomous_execution_fabric/project_adapter_support.md", "test/api/test_repository_capability_assessment.py"), ("EAAEF-040", "EAAEF-041", "EAAEF-042"), "Distinguish preview-only, unsupported language/build, unsafe repository, insufficient validation, human configuration and mutation-not-admitted outcomes.", "python3 -m pytest -q test/api/test_repository_capability_assessment.py"),
    ("044", "EAAEF-G050", "Qualify onboarding against representative repositories", "ipfs_accelerate_py", _paths("test/integration/test_external_agent_project_onboarding.py", "docs/architecture/external_agent_autonomous_execution_fabric/receipts/project_onboarding.json"), ("EAAEF-043",), "Show safe classification for supported, unsupported and malicious fixtures and admit mutation only for the qualified Python profile.", "python3 -m pytest -q test/integration/test_external_agent_project_onboarding.py"),

    ("050", "EAAEF-G060", "Define container execution and worker-lease contracts", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/containers/contracts.py", "test/api/test_container_execution_contracts.py"), (), "Bind image, worktree, task, authority, resources, policy, artifact manifest, checkpoint and receipt while keeping host acceptance authority out of workers.", "python3 -m pytest -q test/api/test_container_execution_contracts.py"),
    ("051", "EAAEF-G060", "Implement default-deny OCI launching with rootless preference", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/containers/oci_runner.py", "test/security/test_container_default_deny.py"), ("EAAEF-050",), "Launch nonroot, read-only-base, capability-dropped, no-new-privileges workers with PID/CPU/RAM/GPU/disk/time bounds, no Docker socket and network deny by default; use a rootless engine where supported and require independent policy admission for any rootful-host-daemon fallback.", "python3 -m pytest -q test/security/test_container_default_deny.py"),
    ("052", "EAAEF-G060", "Build digest-pinned toolchain and prover images", "ipfs_accelerate_py", _paths("containers/external-agent/supervisor.Containerfile", "containers/external-agent/python-worker.Containerfile", "containers/external-agent/prover.Containerfile", "test/containers/test_external_agent_images.py"), ("EAAEF-050",), "Produce image digests, SBOMs, architectures, toolchain/verifier versions and supported adapter versions without mutable tags as authority.", "python3 -m pytest -q test/containers/test_external_agent_images.py"),
    ("053", "EAAEF-G060", "Implement tenant-safe dependency and analysis caches", "ipfs_kit_py", _paths("ipfs_kit_py/execution_cache/profile.py", "tests/test_execution_cache_profile.py"), ("EAAEF-050",), "Key caches by lock, toolchain, architecture, environment and network policy; never share untrusted writable cache authority across tenants.", "python -m pytest -q tests/test_execution_cache_profile.py"),
    ("054", "EAAEF-G060", "Implement fenced container checkpoints and restart", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/containers/checkpoint.py", "ipfs_accelerate_py/agent_supervisor/merge/worktree_lifecycle.py", "test/api/test_agent_supervisor_worktree_lifecycle.py", "test/api/test_container_checkpoint_restart.py"), ("EAAEF-050", "EAAEF-051"), "Preserve attempt/worktree/semantic delta/stages/tests/proofs/model calls/artifacts/resources/obligations/effects, recover only provably dead same-lane owners during controlled restart, and require a later fence on restart.", "python3 -m pytest -q test/api/test_agent_supervisor_worktree_lifecycle.py test/api/test_container_checkpoint_restart.py"),
    ("055", "EAAEF-G060", "Qualify container isolation and cleanup", "ipfs_accelerate_py", _paths("test/security/test_external_agent_container_isolation.py", "docs/architecture/external_agent_autonomous_execution_fabric/receipts/container.json"), ("EAAEF-051", "EAAEF-052", "EAAEF-053", "EAAEF-054"), "Verify no host source/credential/socket/device/cross-volume access, bounded resources, default-deny network, checkpoint recovery and terminal cleanup on a real admitted runtime.", "python3 -m pytest -q test/security/test_external_agent_container_isolation.py"),

    ("060", "EAAEF-G070", "Define federated retrieval request, plan and result", "ipfs_datasets_py", _paths("ipfs_datasets_py/retrieval/agent_work_contracts.py", "tests/unit/retrieval/test_agent_work_contracts.py"), (), "Specify objectives, symbols, evidence classes, source domains, per-engine budgets, graph/AST depth, proof policy, bytes, trust, recency and effective dates.", "python -m pytest -q tests/unit/retrieval/test_agent_work_contracts.py"),
    ("061", "EAAEF-G070", "Enforce separate retrieval corpora and provenance", "ipfs_datasets_py", _paths("ipfs_datasets_py/retrieval/agent_work_corpora.py", "tests/unit/retrieval/test_agent_work_corpora.py"), ("EAAEF-060",), "Keep repository truth, imported claims, verified receipts, requirements, external docs, legal/policy data and model hypotheses separate and trust-filtered.", "python -m pytest -q tests/unit/retrieval/test_agent_work_corpora.py"),
    ("062", "EAAEF-G070", "Compose existing AST, capsules and hybrid graph retrieval", "ipfs_datasets_py", _paths("ipfs_datasets_py/retrieval/agent_work_federation.py", "tests/integration/test_agent_work_federation.py"), ("EAAEF-060", "EAAEF-061"), "Federate existing AST/symbol/semantic/capsule/BM25/vector/sparse-GraphRAG/KG/legal/proof/counterexample indexes without creating a duplicate index system.", "python -m pytest -q tests/integration/test_agent_work_federation.py"),
    ("063", "EAAEF-G070", "Construct bounded trust-aware ContextPacks", "ipfs_kit_py", _paths("ipfs_kit_py/context_pack/external_agent.py", "tests/test_external_agent_context_pack.py"), ("EAAEF-060", "EAAEF-062"), "Distinguish edit-critical raw source, verified/conservative/heuristic capsules, conversation/legal context, proofs, counterexamples and assumptions; never replace opaque critical code with heuristics.", "python -m pytest -q tests/test_external_agent_context_pack.py"),
    ("064", "EAAEF-G070", "Qualify federated retrieval relevance and trust", "ipfs_datasets_py", _paths("tests/integration/test_external_agent_retrieval.py", "docs/architecture/external_agent_retrieval_report.md"), ("EAAEF-062", "EAAEF-063"), "Prove every item carries source CID/revision/trust/mode/score/path/span/capsule/freshness/reason and untrusted similarity cannot override source truth.", "python -m pytest -q tests/integration/test_external_agent_retrieval.py"),

    ("070", "EAAEF-G080", "Compile handoff objectives into typed goal contracts", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/planning/external_goal_contract.py", "test/api/test_external_goal_contract.py"), (), "Declare desired and prohibited outcomes, scope, budgets, authority ceiling, verification/proof/review requirements and completion evidence.", "python3 -m pytest -q test/api/test_external_goal_contract.py"),
    ("071", "EAAEF-G080", "Generate and validate bounded task decompositions", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/planning/external_work_plan.py", "test/api/test_external_work_plan.py"), ("EAAEF-070",), "Use existing FormalWorkPlan machinery to validate coverage, acyclicity, contradictions, scope, resources, merge/proof feasibility and duplicate semantics.", "python3 -m pytest -q test/api/test_external_work_plan.py"),
    ("072", "EAAEF-G080", "Compile planning proof obligations with existing logic", "ipfs_datasets_py", _paths("ipfs_datasets_py/logic/external_work_plan_obligations.py", "tests/unit/logic/test_external_work_plan_obligations.py"), ("EAAEF-070", "EAAEF-071"), "Prove child-to-parent coverage, safe parallel effects, validation-before-acceptance, immutable criteria and no self-granted authority using admitted provers only.", "python -m pytest -q tests/unit/logic/test_external_work_plan_obligations.py"),
    ("073", "EAAEF-G080", "Score alternatives and deterministically admit a plan", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/planning/plan_admission.py", "test/api/test_external_plan_admission.py"), ("EAAEF-071", "EAAEF-072"), "Score critical path, safe width, model/proof cost, resources, merge risk, uncertainty, prior success and cache locality; a deterministic logic gate chooses.", "python3 -m pytest -q test/api/test_external_plan_admission.py"),
    ("074", "EAAEF-G080", "Qualify goal coverage and plan admission", "ipfs_accelerate_py", _paths("test/integration/test_external_agent_plan_compilation.py", "docs/architecture/external_agent_autonomous_execution_fabric/receipts/planning.json"), ("EAAEF-073",), "Reject omitted postconditions, cycles, conflicts, impossible resources and weakened criteria; accept only a content-addressed FormalWorkPlan with current proofs.", "python3 -m pytest -q test/integration/test_external_agent_plan_compilation.py"),

    ("080", "EAAEF-G090", "Compose the multi-authority dependency graph", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/planning/external_composite_graph.py", "test/api/test_external_composite_graph.py"), (), "Compose goal/task/AST/data/schema/contract/proof/validation/scope/effect/merge/resource edges without conflating meanings or authority.", "python3 -m pytest -q test/api/test_external_composite_graph.py"),
    ("081", "EAAEF-G090", "Derive conservative semantic conflict sets", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/planning/external_conflict_graph.py", "test/api/test_external_conflict_graph.py"), ("EAAEF-080",), "Serialize overlapping symbols/files/interfaces/schemas/authorities/resources/effects unless an explicit merge contract proves compatibility; unknown scope conflicts.", "python3 -m pytest -q test/api/test_external_conflict_graph.py"),
    ("082", "EAAEF-G090", "Select resource-aware conflict-free frontiers", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/planning/external_frontier.py", "test/api/test_external_frontier.py"), ("EAAEF-080", "EAAEF-081"), "Maximize useful ready antichains under dependencies, leases, quotas, containers, worktrees, merge capacity, confidence and proofs with deterministic receipts.", "python3 -m pytest -q test/api/test_external_frontier.py"),
    ("083", "EAAEF-G090", "Prevent duplicate logical acceptance", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/runtime/external_logical_claim.py", "test/api/test_external_logical_claim.py"), ("EAAEF-080",), "Bind task, plan revision, base tree, semantic root, task-spec CID and idempotency key so many attempts can run but one result alone is accepted.", "python3 -m pytest -q test/api/test_external_logical_claim.py"),
    ("084", "EAAEF-G090", "Emit bounded subagent work packets", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/todo_daemon/external_agent_container_dispatcher.py", "ipfs_accelerate_py/agent_supervisor/todo_daemon/external_work_packet.py", "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py", "test/api/test_external_agent_container_worker_dispatch.py", "test/api/test_external_work_packet.py"), ("EAAEF-082", "EAAEF-083"), "Bind exact goal/task/repo/semantic/context/scope/effects/container/resources/model/contracts/tests/proofs/completion/lease/fence/checkpoint; reserve provider/effect authority before container launch, accept only content-addressed patch/evidence proposals, and require separate independent host merge admission; workers cannot self-approve.", "python3 -m pytest -q test/api/test_external_agent_container_worker_dispatch.py test/api/test_external_work_packet.py"),
    ("085", "EAAEF-G090", "Qualify parallel frontier safety", "ipfs_accelerate_py", _paths("test/integration/test_external_agent_parallel_frontier.py", "docs/architecture/external_agent_autonomous_execution_fabric/receipts/parallel_frontier.json"), ("EAAEF-082", "EAAEF-083", "EAAEF-084"), "Run conflicting and compatible tasks and prove safe concurrency, one accepted logical result, stale-fence rejection and resource enforcement.", "python3 -m pytest -q test/integration/test_external_agent_parallel_frontier.py"),

    ("090", "EAAEF-G100", "Version the complete mutable control-plane schema", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/task_sources/external_agent_control_schema.py", "test/api/test_external_agent_control_schema.py"), (), "Normalize repositories, handoffs, sessions, runs, goal/plan/task revisions, conflicts, processes, containers, claims, leases, reservations, approvals, events, checkpoints, validation/proofs, merge, artifacts, migrations and cursors.", "python3 -m pytest -q test/api/test_external_agent_control_schema.py"),
    ("091", "EAAEF-G100", "Complete authenticated typed Quack command envelopes", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_contracts.py", "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_transactions.py", "ipfs_accelerate_py/agent_supervisor/task_sources/quack_command_authorization.py", "ipfs_accelerate_py/agent_supervisor/task_sources/quack_command_fabric.py", "ipfs_accelerate_py/agent_supervisor/task_sources/quack_state_client.py", "test/api/test_agent_supervisor_quack_command_fabric.py", "test/api/test_agent_supervisor_quack_state_client.py"), ("EAAEF-090",), "Extend the bootstrap command ingress into the complete canonical operation set. Require request, authenticated principal, independent effect-bound authority, shard, epoch, live lease, fence, idempotency, deadline, expected version/CAS, typed arguments, correlation, cancellation and retry class on every bounded envelope; exact Quack SQL templates are transport only and never authority.", "python3 -m pytest -q test/api/test_agent_supervisor_quack_command_fabric.py test/api/test_agent_supervisor_quack_state_client.py"),
    ("092", "EAAEF-G100", "Route all mutable repositories through the Quack owner", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/task_sources/external_agent_state_repository.py", "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_repository.py", "ipfs_accelerate_py/agent_supervisor/task_sources/quack_daemon_gateway.py", "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py", "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py", "test/api/test_external_agent_state_repository.py", "test/api/test_agent_supervisor_quack_daemon_gateway.py", "test/api/test_agent_supervisor_implementation_supervisor_authority_forwarding.py"), ("EAAEF-090", "EAAEF-091"), "Consolidate handoff/run/goal/plan/task/attempt/provider/effect/validation/proof/merge operations behind one signed-command gateway whose sole local owner applies transactions to its private DuckDB. Remote clients receive Quack append/read capabilities only and never open or ATTACH the operational database; implementation daemons receive closed task, coordination and execution proxies rather than private sidecars.", "python3 -m pytest -q test/api/test_external_agent_state_repository.py test/api/test_agent_supervisor_quack_daemon_gateway.py test/api/test_agent_supervisor_implementation_supervisor_authority_forwarding.py"),
    ("093", "EAAEF-G100", "Qualify one fenced DuckDB/Quack owner and failover", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/runtime/external_quack_owner.py", "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py", "test/api/test_agent_supervisor_quack_state_server.py", "test/api/test_external_quack_owner_failover.py"), ("EAAEF-090", "EAAEF-091", "EAAEF-092"), "Make DuckDB plus Quack the joint orchestrator: Quack supplies bounded authenticated multi-reader/multi-writer transport while exactly one local owner validates signed envelopes and serializes private DuckDB transactions, advances epoch on failover, rejects stale owners and never exposes an operational table for remote UPDATE or arbitrary SQL.", "python3 -m pytest -q test/api/test_agent_supervisor_quack_state_server.py test/api/test_external_quack_owner_failover.py"),
    ("094", "EAAEF-G100", "Project immutable history into DuckLake", "ipfs_datasets_py", _paths("ipfs_datasets_py/ducklake/external_agent_history.py", "tests/integration/test_external_agent_ducklake_history.py"), ("EAAEF-090", "EAAEF-092"), "Publish immutable epochs, task/event/audit history, snapshots, lineage, benchmarks and recovery manifests from an authoritative DuckDB outbox cursor; DuckLake never grants current authority.", "python -m pytest -q tests/integration/test_external_agent_ducklake_history.py"),
    ("095", "EAAEF-G100", "Publish immutable Parquet, IPLD, CAR and IPFS artifacts", "ipfs_kit_py", _paths("ipfs_kit_py/external_agent_history/publication.py", "tests/test_external_agent_history_publication.py"), ("EAAEF-094",), "Content-address committed events/snapshots with privacy-safe manifests and optional CAR/IPFS publication; replication lag cannot grant or revoke authority.", "python -m pytest -q tests/test_external_agent_history_publication.py"),
    ("096", "EAAEF-G100", "Implement backup, restore and ambiguity recovery", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/runtime/external_control_recovery.py", "test/api/test_external_control_recovery.py"), ("EAAEF-093", "EAAEF-094"), "Recover owner crash/restart, duplicate/ambiguous transactions, network partitions, DuckLake outage/delay, corrupted snapshots and backups without accepting stale writes.", "python3 -m pytest -q test/api/test_external_control_recovery.py"),
    ("097", "EAAEF-G100", "Qualify DuckDB, Quack and DuckLake authority separation", "ipfs_accelerate_py", _paths("test/integration/test_external_agent_control_plane.py", "docs/architecture/external_agent_autonomous_execution_fabric/receipts/control_plane.json"), ("EAAEF-093", "EAAEF-094", "EAAEF-095", "EAAEF-096"), "Prove concurrent clients use the sole Quack owner, stale fences fail, retries are idempotent, and DuckLake loss/lag never changes claims, leases, fences or merge authority.", "python3 -m pytest -q test/integration/test_external_agent_control_plane.py"),

    ("100", "EAAEF-G110", "Merge accepted results through the admitted path", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/runtime/external_merge_loop.py", "test/api/test_external_merge_loop.py"), (), "Verify patch/receipts, merge with exact authority, recompute the canonical repository state and settle the merge queue before downstream acceptance.", "python3 -m pytest -q test/api/test_external_merge_loop.py"),
    ("101", "EAAEF-G110", "Refresh semantic indexes incrementally", "ipfs_datasets_py", _paths("ipfs_datasets_py/analysis/external_agent_incremental_refresh.py", "tests/integration/test_external_agent_incremental_refresh.py"), ("EAAEF-100",), "Incrementally refresh AST, semantic state, capsules, BM25, vectors, GraphRAG, KG, tests and proofs and emit precise invalidations/reuse receipts.", "python -m pytest -q tests/integration/test_external_agent_incremental_refresh.py"),
    ("102", "EAAEF-G110", "Compile typed bounded replanning triggers", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/planning/external_replanning.py", "test/api/test_external_replanning.py"), ("EAAEF-100", "EAAEF-101"), "React to changed assumptions, invalidation, failed tests/proofs, counterexamples, stale history, conflicts, outages, resources, task sizing, evidence gaps and no-progress.", "python3 -m pytest -q test/api/test_external_replanning.py"),
    ("103", "EAAEF-G110", "Apply immutable plan-revision operations", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/planning/external_plan_revision.py", "test/api/test_external_plan_revision.py"), ("EAAEF-102",), "Add, supersede, split, coalesce, rewire, reprioritize, block, unblock, cancel future work and add proof/repair/review tasks without editing claimed or accepted history.", "python3 -m pytest -q test/api/test_external_plan_revision.py"),
    ("104", "EAAEF-G110", "Prove fixed-point completion and typed termination", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/runtime/external_fixed_point.py", "test/integration/test_external_agent_fixed_point.py"), ("EAAEF-101", "EAAEF-103"), "Terminate only when goal/subgoals, current tests/proofs, invalidations, assurance, merge queue, source/semantic roots, claims and terminal seal agree—or emit a typed non-success state.", "python3 -m pytest -q test/integration/test_external_agent_fixed_point.py"),

    ("110", "EAAEF-G120", "Expose the canonical Python handoff API", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/api/__init__.py", "ipfs_accelerate_py/agent_supervisor/api/external_handoff.py", "test/api/test_external_handoff_python_api.py"), (), "Provide handoff, preview, attach, status, follow, steer, pause/resume, approve/reject, cancel, explain, doctor, report and export operations with canonical identities.", "python3 -m pytest -q test/api/test_external_handoff_python_api.py"),
    ("111", "EAAEF-G120", "Expose the canonical supervisor CLI", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/cli/supervisor_handoff.py", "ipfs_accelerate_py/cli_entry.py", "test/cli/test_supervisor_handoff_cli.py"), ("EAAEF-110",), "Implement and register structured argv for handoff, status, follow, attach, steer, pause, resume, approve, reject, cancel, explain, doctor, report and export-result; use large-input references, detach and typed exit states without shell execution.", "python3 -m pytest -q test/cli/test_supervisor_handoff_cli.py"),
    ("112", "EAAEF-G120", "Expose canonical MCP tools", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/external_handoff.py", "ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/__init__.py", "ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/native_agent_supervisor_tools.py", "test/mcp/test_agent_supervisor_handoff_tools.py"), ("EAAEF-110",), "Register agent_supervisor_handoff, preview_handoff, attach, status, follow, steer, pause, resume, approve, reject, cancel, explain, doctor and report using connector/file references and the same canonical requests, receipts and authority checks.", "python3 -m pytest -q test/mcp/test_agent_supervisor_handoff_tools.py"),
    ("113", "EAAEF-G120", "Bind MCP++ transport to existing profiles", "ipfs_kit_py", _paths("ipfs_kit_py/mcp_server/mcplusplus/external_agent_handoff.py", "tests/test_mcplusplus_external_agent_handoff.py"), ("EAAEF-110",), "Use admitted interface/artifact/delegation/runtime/event/fencing profiles and DurableExecutor only where configured; create no new profile and grant no storage authority.", "python -m pytest -q tests/test_mcplusplus_external_agent_handoff.py"),
    ("114", "EAAEF-G120", "Implement detach, reconnect and continuation cursors", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/api/external_run_handle.py", "test/api/test_external_run_reconnect.py"), ("EAAEF-110",), "Return durable run identity, survive client and host-supervisor restart, resume events from an exact cursor and enforce authority on steering/cancellation.", "python3 -m pytest -q test/api/test_external_run_reconnect.py"),
    ("115", "EAAEF-G120", "Qualify Python, CLI, MCP and MCP++ parity", "ipfs_accelerate_py", _paths("test/integration/test_external_handoff_transport_parity.py", "docs/architecture/external_agent_autonomous_execution_fabric/receipts/transport_parity.json"), ("EAAEF-111", "EAAEF-112", "EAAEF-113", "EAAEF-114"), "Prove equivalent semantic inputs produce the same canonical identities and lifecycle behavior across transports, with transport-specific envelopes only.", "python3 -m pytest -q test/integration/test_external_handoff_transport_parity.py"),

    ("120", "EAAEF-G130", "Enforce instruction and trust-domain separation", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/security/untrusted_context.py", "test/security/test_external_prompt_injection.py"), (), "Prevent source comments, outputs, history and attachments from changing policy, authority, secrets, proof keys, tests, Quack ownership or promotion criteria.", "python3 -m pytest -q test/security/test_external_prompt_injection.py"),
    ("121", "EAAEF-G130", "Harden malicious repository admission and execution", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/security/repository_policy.py", "test/security/test_malicious_repository.py"), ("EAAEF-120",), "Defend against hooks, symlinks, submodules, bombs, huge files, hostile tests, installs, network, sockets and cross-run data access.", "python3 -m pytest -q test/security/test_malicious_repository.py"),
    ("122", "EAAEF-G130", "Detect imported session poisoning and forged receipts", "ipfs_datasets_py", _paths("ipfs_datasets_py/security/external_session_poisoning.py", "tests/security/test_external_session_poisoning.py"), ("EAAEF-120",), "Detect fabricated tests/tools, stale files, cross-repo patches, replayed receipts, false approvals, secrets and policy-manipulation attempts.", "python -m pytest -q tests/security/test_external_session_poisoning.py"),
    ("123", "EAAEF-G130", "Qualify container escape resistance", "ipfs_accelerate_py", _paths("test/security/test_external_container_escape.py", "containers/external-agent/seccomp.json"), ("EAAEF-121",), "Test Docker socket, host PID, privileged syscall/device/mount/symlink/cgroup escape and cross-container volume access against the real profile.", "python3 -m pytest -q test/security/test_external_container_escape.py"),
    ("124", "EAAEF-G130", "Broker short-lived opaque secrets", "ipfs_kit_py", _paths("ipfs_kit_py/secret_broker/external_worker.py", "tests/test_external_worker_secret_broker.py"), ("EAAEF-120",), "Resolve opaque handles only for the current leased task/policy, mount ephemeral files, redact events and revoke at checkpoint/terminal boundaries.", "python -m pytest -q tests/test_external_worker_secret_broker.py"),
    ("125", "EAAEF-G130", "Run the integrated adversarial security gate", "ipfs_accelerate_py", _paths("test/security/test_external_agent_adversarial_gate.py", "docs/architecture/external_agent_autonomous_execution_fabric/receipts/security.json"), ("EAAEF-121", "EAAEF-122", "EAAEF-123", "EAAEF-124"), "Require all hostile history/repository/container/secret/effect cases to fail closed with no accepted mutation or leaked authority.", "python3 -m pytest -q test/security/test_external_agent_adversarial_gate.py"),

    ("130", "EAAEF-G140", "Emit typed lifecycle and assurance events", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/observability/external_events.py", "test/api/test_external_agent_events.py"), (), "Emit typed events from handoff through terminal state with exact run/task/attempt/fence/artifact identities and continuation ordering.", "python3 -m pytest -q test/api/test_external_agent_events.py"),
    ("131", "EAAEF-G140", "Account for resources, latency, reuse and cost", "ipfs_accelerate_py", _paths("ipfs_accelerate_py/agent_supervisor/observability/external_metrics.py", "test/api/test_external_agent_metrics.py"), ("EAAEF-130",), "Track bytes, index/retrieval/context/model/proof/test/supervisor/worker/resources/conflicts/duplicates/Quack/DuckDB/DuckLake/merge and per-result costs without treating estimates as observations.", "python3 -m pytest -q test/api/test_external_agent_metrics.py"),
    ("132", "EAAEF-G140", "Publish privacy-safe analytical telemetry", "ipfs_datasets_py", _paths("ipfs_datasets_py/ducklake/external_agent_telemetry.py", "tests/integration/test_external_agent_telemetry.py"), ("EAAEF-130", "EAAEF-131"), "Project digest/reference/count-only telemetry into DuckLake while raw prompts, source bodies, secrets and credentials remain excluded.", "python -m pytest -q tests/integration/test_external_agent_telemetry.py"),
    ("133", "EAAEF-G140", "Qualify observability, steering and accounting", "ipfs_accelerate_py", _paths("test/integration/test_external_agent_observability.py", "docs/architecture/external_agent_autonomous_execution_fabric/receipts/observability.json"), ("EAAEF-130", "EAAEF-131", "EAAEF-132"), "Demonstrate cursor continuity, explainability, pause/resume/cancel, bounded metrics cardinality, privacy and terminal accounting under restart.", "python3 -m pytest -q test/integration/test_external_agent_observability.py"),

    ("140", "EAAEF-G150", "Build external-client and hostile-input fixtures", "ipfs_accelerate_py", _paths("test/fixtures/external_agent_handoff/manifest.json", "test/api/test_external_agent_fixture_manifest.py"), (), "Cover Codex, Claude, Gemini, generic MCP, visible/truncated/branched history, failures/forgeries, dirty/submodule/LFS/unsupported/malicious/large repositories and budgets.", "python3 -m pytest -q test/api/test_external_agent_fixture_manifest.py"),
    ("141", "EAAEF-G150", "Run three-supervisor eight-worker qualification", "ipfs_accelerate_py", _paths("test/integration/test_external_agent_multi_supervisor.py", "docs/architecture/external_agent_autonomous_execution_fabric/receipts/multi_supervisor.json"), ("EAAEF-140",), "Use independent analysis/implementation/verification roles, eight real worker containers, competing resources and conflicting/nonconflicting tasks; verify concurrency, serialization, fencing and one acceptance.", "python3 -m pytest -q test/integration/test_external_agent_multi_supervisor.py"),
    ("142", "EAAEF-G150", "Qualify client disconnect and reconnect", "ipfs_accelerate_py", _paths("test/integration/test_external_client_disconnect.py", "docs/architecture/external_agent_autonomous_execution_fabric/receipts/disconnect.json"), ("EAAEF-140",), "For every supported client submit, receive run ID, terminate, continue, restart host supervisor, reattach, resume cursor, steer and reach a typed terminal state.", "python3 -m pytest -q test/integration/test_external_client_disconnect.py"),
    ("143", "EAAEF-G150", "Qualify real container lifecycle", "ipfs_accelerate_py", _paths("test/integration/test_external_container_lifecycle.py", "docs/architecture/external_agent_autonomous_execution_fabric/receipts/container_lifecycle.json"), ("EAAEF-140",), "Verify clean image startup, no host mutation/credential/socket exposure, resource/network bounds, checkpoint recovery and terminal cleanup.", "python3 -m pytest -q test/integration/test_external_container_lifecycle.py"),
    ("144", "EAAEF-G150", "Qualify crashes, partitions and stale authority", "ipfs_accelerate_py", _paths("test/integration/test_external_agent_fault_matrix.py", "docs/architecture/external_agent_autonomous_execution_fabric/receipts/fault_matrix.json"), ("EAAEF-140",), "Exercise provider/prover outage, Quack owner/supervisor/worker crash, partition, DuckLake outage, duplicates, conflict, stale roots/plans, exhaustion and no-progress.", "python3 -m pytest -q test/integration/test_external_agent_fault_matrix.py"),
    ("145", "EAAEF-G150", "Seal the end-to-end fault qualification", "ipfs_accelerate_py", _paths("test/integration/test_external_agent_end_to_end.py", "docs/architecture/external_agent_autonomous_execution_fabric/receipts/end_to_end.json"), ("EAAEF-141", "EAAEF-142", "EAAEF-143", "EAAEF-144"), "Require evidence rather than worker assertions for goal completion and produce exact current receipts for all client, container, control and recovery paths.", "python3 -m pytest -q test/integration/test_external_agent_end_to_end.py"),

    ("150", "EAAEF-G160", "Build the reproducible benchmark harness", "ipfs_accelerate_py", _paths("benchmarks/external_agent_fabric/harness.py", "test/benchmarks/test_external_agent_harness.py"), (), "Run the same task/repository/authority budget across configurations A-D and preserve exact input/image/model/provider/prover identities.", "python3 -m pytest -q test/benchmarks/test_external_agent_harness.py"),
    ("151", "EAAEF-G160", "Measure all four execution configurations", "ipfs_accelerate_py", _paths("benchmarks/external_agent_fabric/run_matrix.py", "test/benchmarks/test_external_agent_matrix.py"), ("EAAEF-150",), "Measure completion, accepted patches, wall/first-useful time, parallel/worker/resource efficiency, tokens/proofs/index reuse/duplicates/conflicts/retries/refill/cost.", "python3 -m pytest -q test/benchmarks/test_external_agent_matrix.py"),
    ("152", "EAAEF-G160", "Publish honest performance and parallelism results", "ipfs_datasets_py", _paths("docs/benchmarks/external_agent_fabric_results.md", "tests/benchmarks/test_external_agent_results_schema.py"), ("EAAEF-151",), "Report actual targets and misses; never represent simulated/historical providers or unavailable checks as live evidence.", "python -m pytest -q tests/benchmarks/test_external_agent_results_schema.py"),
    ("153", "EAAEF-G160", "Qualify benchmark evidence and target claims", "ipfs_accelerate_py", _paths("test/benchmarks/test_external_agent_release_benchmark.py", "docs/architecture/external_agent_autonomous_execution_fabric/receipts/benchmark.json"), ("EAAEF-152",), "Validate zero duplicate/stale/overlap acceptance and report measured efficiency, utilization, reuse and coordination overhead without converting goals into claims.", "python3 -m pytest -q test/benchmarks/test_external_agent_release_benchmark.py"),

    ("160", "EAAEF-G170", "Build clean cross-package wheels and schema bundles", "ipfs_accelerate_py", _paths("scripts/release/build_external_agent_stack.py", "test/packaging/test_external_agent_wheels.py"), (), "Build pinned wheels for the three primary packages plus MCP++ only when required, with schemas/migrations and no sibling checkout/editable-install dependence.", "python3 -m pytest -q test/packaging/test_external_agent_wheels.py"),
    ("161", "EAAEF-G170", "Build signed digest-pinned OCI images and SBOMs", "ipfs_accelerate_py", _paths("scripts/release/build_external_agent_images.py", "test/packaging/test_external_agent_images.py"), ("EAAEF-160",), "Produce versioned supervisor/worker/prover images, SBOMs, locks, architectures and signatures from immutable sources.", "python3 -m pytest -q test/packaging/test_external_agent_images.py"),
    ("162", "EAAEF-G170", "Package supported deployment profiles", "ipfs_accelerate_py", _paths("deploy/external-agent/local-supervised.yaml", "deploy/external-agent/detached-single-host.yaml", "deploy/external-agent/multi-container-single-host.yaml", "test/packaging/test_external_agent_deployments.py"), ("EAAEF-160", "EAAEF-161"), "Qualify local supervised, detached single-host and multi-container single-host; keep remote multi-host disabled until authenticated Quack failover/partition/security gates pass.", "python3 -m pytest -q test/packaging/test_external_agent_deployments.py"),
    ("163", "EAAEF-G170", "Document and test migration, backup, restore and rollback", "ipfs_accelerate_py", _paths("docs/deployment/EXTERNAL_AGENT_FABRIC.md", "test/packaging/test_external_agent_upgrade_rollback.py"), ("EAAEF-160", "EAAEF-162"), "Ship example configuration, schema migration, Quack owner backup/restore, DuckLake recovery, upgrades and rollback with exact compatibility checks.", "python3 -m pytest -q test/packaging/test_external_agent_upgrade_rollback.py"),
    ("164", "EAAEF-G170", "Qualify clean external installation", "ipfs_accelerate_py", _paths("test/packaging/test_external_agent_clean_install.py", "docs/architecture/external_agent_autonomous_execution_fabric/receipts/packaging.json"), ("EAAEF-161", "EAAEF-162", "EAAEF-163"), "Install only released artifacts into a clean environment and run the supported deployment smoke without mutable branches, sibling repos or editable paths.", "python3 -m pytest -q test/packaging/test_external_agent_clean_install.py"),

    ("170", "EAAEF-G180", "Install the blocking release CI matrix", "ipfs_accelerate_py", _paths(".github/workflows/external-agent-fabric.yml", "test/ci/test_external_agent_ci_matrix.py"), (), "Cover reconciliation, clean install, schemas/adapters/bundles/containers/projects/retrieval/logic/conflicts/Quack/DuckDB/DuckLake/multi-supervisor/reconnect/crash/security/parity/e2e/benchmark.", "python3 -m pytest -q test/ci/test_external_agent_ci_matrix.py"),
    ("171", "EAAEF-G180", "Enforce zero-skip fail-closed qualification receipts", "ipfs_accelerate_py", _paths("scripts/validate_external_agent_release.py", "test/ci/test_external_agent_release_validator.py"), ("EAAEF-170",), "Reject continue-on-error, shell success masking, skipped/xfailed required tests, unavailable-as-passed, simulations-as-live and stale historical counts; bind collected/passed populations.", "python3 -m pytest -q test/ci/test_external_agent_release_validator.py"),
    ("172", "EAAEF-G180", "Run the clean external handoff smoke", "ipfs_accelerate_py", _paths("test/release/test_external_agent_handoff_smoke.py", "docs/architecture/external_agent_autonomous_execution_fabric/receipts/release_smoke.json"), ("EAAEF-170", "EAAEF-171"), "From clean packages, submit a supported exported session and exact repository, disconnect, execute in containers, reattach, verify, and reach a typed terminal outcome.", "python3 -m pytest -q test/release/test_external_agent_handoff_smoke.py"),
    ("173", "EAAEF-G180", "Perform independent security and authority review", "ipfs_accelerate_py", _paths("docs/security/EXTERNAL_AGENT_FABRIC_REVIEW.md", "test/release/test_external_agent_security_release.py"), ("EAAEF-171",), "Audit trust, file/path, import/startup, environment, credential, Quack-owner, DuckDB, DuckLake, container and receipt boundaries; unresolved critical findings force no-go.", "python3 -m pytest -q test/release/test_external_agent_security_release.py"),
    ("174", "EAAEF-G180", "Emit the complete qualification report", "ipfs_accelerate_py", _paths("docs/architecture/external_agent_autonomous_execution_fabric/QUALIFICATION_REPORT.md", "docs/architecture/external_agent_autonomous_execution_fabric/qualification_report.json"), ("EAAEF-172", "EAAEF-173"), "Report exact revisions/manifests/schemas/adapters/transfer/authority/containers/support/retrieval/logic/parallel/control/DuckLake/fault/security/performance/packaging/CI results and level.", "python3 scripts/validate_external_agent_release.py --report-only"),
    ("175", "EAAEF-G180", "Seal the terminal source, semantic and authority state", "ipfs_accelerate_py", _paths("scripts/seal_external_agent_release.py", "test/release/test_external_agent_terminal_seal.py"), ("EAAEF-174",), "Require current source/semantic roots, tests/proofs, empty claims and merge queue, immutable DuckLake cursor evidence and a content-addressed terminal report; a worker cannot self-seal.", "python3 -m pytest -q test/release/test_external_agent_terminal_seal.py"),
    ("176", "EAAEF-G180", "Issue the narrow go or no-go recommendation", "ipfs_accelerate_py", _paths("docs/architecture/external_agent_autonomous_execution_fabric/FINAL_RECOMMENDATION.md", "docs/architecture/external_agent_autonomous_execution_fabric/final_recommendation.json"), ("EAAEF-175",), "Assign at most supervised_external_pilot only when real external clients, isolated containers, resumability and evidence gates pass; unsupported codebases remain preview-only or human-configured.", "python3 scripts/validate_external_agent_release.py --terminal"),
)


def _as_list(values: Iterable[str]) -> list[str]:
    return [str(value) for value in values]


def _epic_map() -> dict[str, dict[str, str]]:
    return {
        goal_id: {
            "epic": epic,
            "title": title,
            "predecessor": predecessor,
            "contract": contract,
        }
        for goal_id, epic, title, predecessor, contract in EPICS
    }


def _resource_request(task_id: str, repository: str) -> dict[str, Any]:
    initial = int(task_id.split("-")[-1]) < 10
    request = {
        "cpu_millicores": 2000 if initial else 4000,
        "ram_mib": 4096 if initial else 8192,
        "gpu_count": 0,
        "disk_mib": 8192 if initial else 16384,
        "network": "deny",
        "supervisor_processes": 1,
        "worktree_slots": 1,
        "container_slots": 1,
        "merge_slots": 1,
        "provider_concurrency": 1,
        "model_input_token_ceiling": 48000,
        "model_output_token_ceiling": 12000,
        "prover_concurrency": 1 if repository == "ipfs_datasets_py" else 0,
        "timeout_seconds": 7200,
    }
    if task_id == "EAAEF-000":
        request.update(
            {
                "cpu_millicores": 1000,
                "ram_mib": 2048,
                "disk_mib": 4096,
                "supervisor_processes": 0,
                "container_slots": 1,
                "merge_slots": 0,
                "provider_concurrency": 0,
                "model_input_token_ceiling": 0,
                "model_output_token_ceiling": 0,
                "prover_concurrency": 0,
                "timeout_seconds": 1800,
            }
        )
    heavy = {
        "EAAEF-052": (8000, 16384, 65536, 3, 1, 14400),
        "EAAEF-055": (8000, 16384, 32768, 4, 1, 14400),
        "EAAEF-093": (12000, 24576, 65536, 4, 2, 14400),
        "EAAEF-097": (12000, 24576, 65536, 4, 2, 14400),
        "EAAEF-124": (12000, 24576, 65536, 4, 1, 14400),
        "EAAEF-141": (24000, 49152, 131072, 8, 3, 28800),
        "EAAEF-142": (8000, 16384, 32768, 2, 2, 14400),
        "EAAEF-143": (12000, 24576, 65536, 4, 1, 14400),
        "EAAEF-144": (24000, 49152, 131072, 8, 3, 28800),
        "EAAEF-151": (24000, 49152, 131072, 8, 3, 43200),
        "EAAEF-161": (12000, 24576, 131072, 3, 1, 21600),
        "EAAEF-164": (12000, 24576, 65536, 4, 1, 21600),
    }
    if task_id in heavy:
        cpu, ram, disk, containers, supervisors, timeout = heavy[task_id]
        request.update(
            {
                "cpu_millicores": cpu,
                "ram_mib": ram,
                "disk_mib": disk,
                "container_slots": containers,
                "supervisor_processes": supervisors,
                "timeout_seconds": timeout,
            }
        )
    return request


def _external_effect_scope(task_id: str) -> list[str]:
    if task_id == "EAAEF-000":
        return [
            "host-controlled read-only verification of signed bootstrap evidence",
            "offline network-none OCI image build and non-provider qualification probes in one bounded diagnostic container slot",
            "reviewed task-owned policy/test writes and create-once receipt publication",
            "no implementation supervisor, provider invocation, external network, secret, merge, push or mutable control-plane effect before admission",
        ]
    effects = [
        "isolated task worktree writes",
        "bounded container subprocesses",
        "merge-queue proposal only",
        "no push, publication, secret, dependency-install or network effect without exact approval",
    ]
    overrides = {
        "EAAEF-052": "OCI build and SBOM generation from admitted cached inputs; base-image network pull requires an exact approval",
        "EAAEF-093": "Quack owner crash/restart, later-epoch takeover and stale-owner rejection are confined to a disposable test shard",
        "EAAEF-095": "optional CAR/IPFS publication only through an exact authenticated publication approval",
        "EAAEF-097": "DuckLake outage/lag/corruption and DuckDB/Quack partition exercises use disposable replicas and never the live authority shard",
        "EAAEF-124": "container-escape probes target only leased adversarial fixtures and isolated namespaces; no host exploit or unrelated device access",
        "EAAEF-141": "three supervisor processes and at least eight isolated worker containers within the declared reservation",
        "EAAEF-142": "client and supervisor process termination/restart is limited to the leased disconnect fixture",
        "EAAEF-143": "container lifecycle, checkpoint and cleanup effects are limited to isolated qualification worktrees and volumes",
        "EAAEF-144": "fault injection limited to leased campaign containers, Quack test shard and synthetic network namespace",
        "EAAEF-151": "four benchmark configurations with identical admitted provider, prover and resource-budget identities",
        "EAAEF-161": "release-image build, signing and SBOM effects; registry publication is separately approved",
        "EAAEF-164": "clean wheel and OCI installation exercises run only in disposable deployment profiles; no production deployment",
    }
    if task_id in overrides:
        effects.append(overrides[task_id])
    return effects


def _goal_governance_fields(*, epic: str, contract: str) -> dict[str, Any]:
    """Return the explicit H1 contract fields required on every goal."""

    return {
        "desired_postconditions": [contract],
        "prohibited_outcomes": [
            "universal autonomous-mutation support claim",
            "duplicate accepted work or overlapping accepted effects",
            "stale-fence acceptance",
            "authority, disclosure, resource or mutation scope wider than the admitted parent policy",
            "worker or model self-acceptance",
            "stale source, semantic, plan, lease, fence, test or proof evidence",
            "unverified imported history satisfying completion",
            "DuckLake or a replica granting current coordination authority",
        ],
        "scope": {
            "epic": epic,
            "repositories": [
                "ipfs_accelerate_py",
                "ipfs_datasets_py",
                "ipfs_kit_py",
                "Mcp-Plus-Plus only for an existing shared protocol contract",
            ],
            "mutation": "only task-owned files/effects in admitted isolated worktrees and containers",
        },
        "resource_budget": {
            "policy": "bounded by the sum of admitted child-task reservations and the parent run ceiling",
            "network": "deny unless an exact effect-bound approval names the action and inputs",
            "unbounded_refill": False,
        },
        "authority_ceiling": [
            "no protected-branch push or automatic production deployment",
            "merge or reviewed-patch delivery only through independent admission",
            "no worker, model, prompt, repository file, CID or run ID may widen authority",
        ],
        "verification_requirements": [
            "current pre-change, focused and affected-integration receipts",
            "zero required skip, xfail, xpass or failure",
            "independent verifier acceptance against exact source and plan roots",
        ],
        "proof_requirements": [
            "content identities and provenance for inputs, outputs and receipts",
            "dependency coverage plus read/write/effect conflict admission",
            "current proof obligations or a typed independently reviewed not-applicable decision",
        ],
        "human_review_requirements": [
            "authenticated review for merge, push, new dependency/network/secret access, wider disclosure, destructive cleanup or publication",
            "explicit review for unresolved critical security or compatibility findings",
        ],
        "completion_evidence": [
            "all required child task outcomes and independent acceptance receipts",
            "current source/semantic/plan roots with no blocking invalidation",
            "settled merge queue and no live mutating claims",
            "content-addressed terminal report or typed no-go decision",
        ],
    }


def _build() -> tuple[dict[str, Any], str, str]:
    source = _load_object(SOURCE_MANIFEST)
    stack = _load_object(STACK_MANIFEST)
    roots = stack.get("integration_roots")
    if not isinstance(roots, dict) or set(roots) != {
        "ipfs_accelerate_py",
        "ipfs_datasets_py",
        "ipfs_kit_py",
        "Mcp-Plus-Plus",
    }:
        raise ValueError("stack manifest must bind all four exact integration roots")
    source_forest_root = str(source.get("source_forest_root") or "")
    if not source_forest_root.startswith("sha256:"):
        raise ValueError("source manifest lacks source_forest_root")

    epics = _epic_map()
    previous_gate: dict[str, str] = {
        "EAAEF-G010": "",
        "EAAEF-G020": "EAAEF-009",
        "EAAEF-G030": "EAAEF-015",
        "EAAEF-G040": "EAAEF-024",
        "EAAEF-G050": "EAAEF-034",
        "EAAEF-G060": "EAAEF-044",
        "EAAEF-G070": "EAAEF-055",
        "EAAEF-G080": "EAAEF-064",
        "EAAEF-G090": "EAAEF-074",
        "EAAEF-G100": "EAAEF-085",
        "EAAEF-G110": "EAAEF-097",
        "EAAEF-G120": "EAAEF-104",
        "EAAEF-G130": "EAAEF-115",
        "EAAEF-G140": "EAAEF-125",
        "EAAEF-G150": "EAAEF-133",
        "EAAEF-G160": "EAAEF-145",
        "EAAEF-G170": "EAAEF-153",
        "EAAEF-G180": "EAAEF-164",
    }
    seen: set[str] = set()
    last_owner_by_path: dict[tuple[str, str], str] = {}
    tasks: list[dict[str, Any]] = []
    all_revisions = {
        name: {
            "commit": str(value.get("commit") or ""),
            "tree": str(value.get("tree") or ""),
            "integration_branch": str(value.get("integration_branch") or ""),
        }
        for name, value in roots.items()
        if isinstance(value, dict)
    }
    for number, goal_id, title, repository, paths, explicit_deps, objective, validation in TASK_ROWS:
        task_id = f"EAAEF-{number}"
        if task_id in seen:
            raise ValueError(f"duplicate task id: {task_id}")
        seen.add(task_id)
        if goal_id not in epics:
            raise ValueError(f"unknown goal for {task_id}: {goal_id}")
        deps = list(explicit_deps)
        predecessor = previous_gate[goal_id]
        if predecessor and predecessor not in deps:
            deps.insert(0, predecessor)
        path_values = _as_list(paths)
        if len(path_values) != len(set(path_values)):
            raise ValueError(f"{task_id} declares a duplicate owned path")
        overlap_merge_contracts: list[dict[str, str]] = []
        for path in path_values:
            prior_owner = last_owner_by_path.get((repository, path))
            if prior_owner is None:
                continue
            if prior_owner not in deps:
                deps.append(prior_owner)
            overlap_merge_contracts.append(
                {
                    "schema": OVERLAP_CONTRACT_SCHEMA,
                    "repository": repository,
                    "path": path,
                    "predecessor_task_id": prior_owner,
                    "successor_task_id": task_id,
                    "dependency_type": "direct",
                    "strategy": OVERLAP_STRATEGY,
                    "merge_lane": OVERLAP_MERGE_LANE,
                }
            )
        numeric = int(number)
        initial_population = numeric < 10
        semantic_root = (
            source_forest_root
            if initial_population
            else "REBIND_REQUIRED_BY_EAAEF-009"
        )
        execution_paths = _execution_paths(repository, paths)
        current_status = "todo" if initial_population else "blocked"
        population_state = (
            "materialized_bootstrap"
            if initial_population
            else "template_only_awaiting_plan_r2"
        )
        task = {
            "schema": TASK_SCHEMA,
            "stable_task_id": task_id,
            "parent_goal_id": ROOT_GOAL,
            "subgoal_id": goal_id,
            "epic": epics[goal_id]["epic"],
            "title": title,
            "objective": objective,
            "owning_repository": repository,
            "owned_files": _as_list(paths),
            "execution_owned_files": execution_paths,
            "integration_conflict_keys": _integration_conflict_keys(repository),
            "source_revisions": all_revisions,
            "source_semantic_state_root": semantic_root,
            "source_control_plane_schema_version": CONTROL_SCHEMA,
            "dependencies": deps,
            "overlap_merge_contracts": overlap_merge_contracts,
            "read_scope": [
                "exact files named by the current ContextPack",
                "source-reconciliation and compatibility manifests",
                "declared dependency receipts",
            ],
            "write_scope": _as_list(paths),
            "external_effect_scope": _external_effect_scope(task_id),
            "required_capsules": [
                "current task-owned symbol capsules when available",
                "raw source for opaque or edit-critical code",
            ],
            "context_artifacts": [
                "SourceReconciliationManifest@1",
                "StackCompatibilityManifest@1",
                "current FormalWorkPlan task specification",
                "current trust-filtered ContextPack",
            ],
            "resource_request": _resource_request(task_id, repository),
            "container_profile": (
                "ContainerExecutionProfile@1:host-controlled-bootstrap-admission"
                if task_id == "EAAEF-000"
                else (
                    "ContainerExecutionProfile@1:isolated-git-reconciliation"
                    if initial_population
                    else "ContainerExecutionProfile@1:qualified-project-worker"
                )
            ),
            "model_route": (
                "host-controlled deterministic admission; no model or provider invocation"
                if task_id == "EAAEF-000"
                else (
                    "exact bootstrap provider/model/container invocation bound by the admitted bootstrap-runtime receipt; unresolved identity is a no-go"
                    if initial_population
                    else "claim-time exact provider, model revision and container invocation bound by the admitted policy; no model is acceptance authority"
                )
            ),
            "provider_policy": (
                "no provider invocation; independently verify the signed EAAEF provider authorization that later tasks may consume"
                if task_id == "EAAEF-000"
                else "configured allowlist after source-disclosure admission; local-only when classification requires; imported history cannot select a provider"
            ),
            "test_requirements": {
                "pre_change": "run the admitted existing affected baseline before mutation; record a typed absent-new-test observation rather than fabricating a command",
                "focused": [validation],
                "affected_integration": "select current affected suites from the source/provenance graph and ProjectAdapter; record exact structured argv",
                "required_result": "zero required skips, xfails, xpasses or failures",
            },
            "proof_requirements": [
                "content identity for inputs, outputs and receipts",
                "dependency and write/effect conflict admission",
                "current fence and source/semantic roots",
                "independent verifier acceptance",
            ],
            "completion_contract": (
                f"{objective} The focused command must collect and pass its declared population; "
                "the task result must bind exact source, semantic, plan, worktree, container, tests, "
                "proofs and effects; an independent supervisor verifier, never the worker, accepts it."
            ),
            "lease_and_fencing_requirements": {
                "logical_claim_key": "task_id + plan_revision + repository_base_tree + semantic_state_root + task_spec_cid + idempotency_key",
                "task_claim": "one current claim with monotonically later fencing token on restart",
                "worktree_lease": "exclusive isolated worktree",
                "write_effect_leases": "exclusive for every overlapping file/symbol/schema/external effect",
                "coordination": "initial population materialization permits one offline embedded writer; every live request uses the EAAEF-000-admitted signed-command Quack transport and sole private DuckDB owner; EAAEF-008 renews that admission against the reconciled semantic root",
            },
            "idempotency_key": _cid(
                {
                    "task": task_id,
                    "plan": PLAN_REVISION,
                    "source_forest_root": source_forest_root,
                    "paths": list(paths),
                }
            ),
            "rollback_or_compensation": "Preserve failed attempts and receipts; do not overwrite another worktree or authority; abandon the isolated worktree or submit an explicit inverse patch; advance fences before retry; never force-push or delete preserved refs.",
            "required_evidence": [
                "task claim and fencing receipt",
                "container and ContextPack identities",
                "before/after Git tree and patch identity",
                "focused and affected-integration test receipts with collected/passed/skipped/failed counts",
                "proof/contract receipts or typed not-applicable decision",
                "independent review and merge-queue receipt",
                "resource use and partial-effect record",
            ],
            "terminal_status": "not_terminal",
            "allowed_terminal_statuses": [
                "completed",
                "cancelled",
                "failed",
                "quarantined",
            ],
            "outcome": "pending",
            "allowed_outcomes": [
                "accepted",
                "preview_only",
                "unsupported_language",
                "unsupported_build_system",
                "unsafe_repository",
                "insufficient_validation_profile",
                "human_configuration_required",
                "mutation_not_admitted",
                "blocked_external_dependency",
                "budget_exhausted",
                "cancelled",
                "failed",
            ],
            "outcome_status_mapping": {
                "accepted": "completed",
                "preview_only": "completed",
                "unsupported_language": "completed",
                "unsupported_build_system": "completed",
                "unsafe_repository": "quarantined",
                "insufficient_validation_profile": "completed",
                "human_configuration_required": "quarantined",
                "mutation_not_admitted": "completed",
                "blocked_external_dependency": "quarantined",
                "budget_exhausted": "quarantined",
                "cancelled": "cancelled",
                "failed": "failed",
            },
            "final_artifact_identities": [
                {
                    "role": "task_result_receipt",
                    "schema": "ExternalAgentTaskResultReceipt@1",
                    "identity": "BOUND_AT_TERMINAL_TASK_REVISION",
                },
                {
                    "role": "accepted_git_or_reviewed_patch",
                    "schema": "GitTreeOrPatchIdentity@1",
                    "identity": "BOUND_AT_TERMINAL_TASK_REVISION",
                },
                {
                    "role": "verification_bundle",
                    "schema": "VerificationBundle@1",
                    "identity": "BOUND_AT_TERMINAL_TASK_REVISION",
                },
            ],
            "status": current_status,
            "completion_mode": "manual" if task_id == "EAAEF-000" else "auto",
            "is_schedulable": initial_population,
            "initial_population": initial_population,
            "population_state": population_state,
            "blocked_reason": "" if initial_population else "awaiting_EAAEF-009_plan_revision",
            "priority": "P0" if numeric < 100 else "P1",
            "track": epics[goal_id]["title"].lower().replace(" ", "-")[:48],
            "plan_revision": PLAN_REVISION,
            "board_namespace": BOARD_NAMESPACE,
            "permitted_effects": [
                "read declared inputs",
                "write only owned files in the leased worktree",
                "run admitted structured argv inside the leased container",
                "submit a merge proposal and receipts",
            ],
            "prohibited_effects": [
                "worker self-approval",
                "direct protected-branch mutation or push",
                "unapproved network, dependency, secret or publication access",
                "direct remote DuckDB file access",
                "DuckLake-derived claim, lease, fence or merge authority",
                "hidden chain-of-thought collection or representation",
            ],
            "outputs": _as_list(paths),
            "execution_outputs": execution_paths,
            "validation": validation,
            "execution_validation": _structured_execution_validation(
                repository, validation
            ),
            "acceptance": (
                "Only an authenticated independent operator and security reviewer may sign the immutable bootstrap admission or typed no-go receipt; the task cannot self-admit and no supervisor starts before acceptance."
                if task_id == "EAAEF-000"
                else "Only the configured independent supervisor completion policy may accept current evidence for this exact task identity."
            ),
            "conflict_and_merge_contract": (
                "No independently executing task may mutate an identical repository-local or projected execution file. "
                "Repeated path ownership is admitted only through the exact overlap_merge_contracts chain: the later owner directly depends on the immediate prior owner, uses serialized_forward_extension, and enters the single_admitted_merge_lane. "
                "Broader symbol/schema/effect overlap requires a declared conflict edge and serialized merge task; unknown scope serializes. "
                + (
                    f"Promotion of {REPOSITORY_EXECUTION_PREFIXES[repository]} is a serialized superproject-gitlink merge effect and never occurs concurrently with another promotion of that gitlink."
                    if REPOSITORY_EXECUTION_PREFIXES[repository]
                    else "Accelerator central registries and merge-queue effects are changed only by their declared owner and the single admitted merge lane."
                )
            ),
        }
        if task_id == "EAAEF-000":
            task["completion_contract"] = (
                f"{objective} The deterministic preflight tests must pass, and an "
                "authenticated independent operator plus security reviewer must bind every "
                "required digest, signature, policy and Quack decision in an immutable "
                "admission or typed no-go receipt. No supervisor, worker or model may accept "
                "this task or start before that receipt verifies."
            )
            task["lease_and_fencing_requirements"] = {
                "logical_claim_key": "EAAEF-000 + plan_revision + source_forest_root + task_spec_cid + one-use operator nonce",
                "task_claim": "manual host admission request authenticated independently of the supervisor",
                "worktree_lease": "exclusive reviewed bootstrap worktree",
                "write_effect_leases": "no mutable control-plane effect; immutable receipt publication is create-once",
                "coordination": "no Quack or DuckDB authority is assumed; the receipt records the independently verified Quack admission/no-go decision",
            }
            task["required_evidence"] = [
                "authenticated one-use operator admission request",
                "signed EAAEF-scoped provider authorization",
                "independently signed task-capable OCI worker image and SBOM digests with at least five admitted worker slots",
                "effect-bound signed per-attempt internal-network/proxy authorizations for five collision-free lanes",
                "explicit rootless-engine or independently approved rootful-host-daemon/nonroot-worker mode, no-socket, bounded allowlisted proxy egress and bounded-resource policy",
                "immutable materialization identity and receipt",
                "exact DuckDB 1.5.5 and locked Quack 1.5.5 command-ingress qualification proving signed envelope authorization, one private file owner, multi-client append/read and no operational-table exposure",
                "independent operator and security-review signatures",
            ]
            task["permitted_effects"] = [
                "read and verify the declared immutable bootstrap evidence",
                "write only task-owned bootstrap policy/test artifacts in the reviewed worktree",
                "publish one create-once admission or typed no-go receipt",
            ]
            task["prohibited_effects"] = [
                "starting a supervisor, worker or model before admission",
                "self-approval by the task, worker, model or prospective supervisor",
                "unsigned or bare StateCommand mutation, shared direct DuckDB access, or operational-table exposure through Quack",
                "unapproved network, secret, dependency, merge, push or publication effects",
            ]
        task["task_spec_cid"] = _cid(task)
        tasks.append(task)
        for path in path_values:
            last_owner_by_path[(repository, path)] = task_id

    task_ids = {task["stable_task_id"] for task in tasks}
    for task in tasks:
        missing = sorted(set(task["dependencies"]) - task_ids)
        if missing:
            raise ValueError(f"{task['stable_task_id']} has missing dependencies: {missing}")

    goals: list[dict[str, Any]] = [
        {
            "schema": GOAL_SCHEMA,
            "goal_id": ROOT_GOAL,
            "parent_goal_id": "",
            "epic": "ROOT",
            "title": "Implement and qualify ExternalAgentAutonomousExecutionFabric",
            "dependencies": [],
            "completion_contract": "All mandatory epic goals reach accepted terminal evidence against one source/semantic/plan generation; no blocking invalidation, mutable claim or merge remains; the terminal report and seal verify.",
            **_goal_governance_fields(
                epic="ROOT",
                contract="All required A-R postconditions are independently accepted against one current source, semantic and plan generation, and the terminal seal verifies.",
            ),
        }
    ]
    for goal_id, epic, title, predecessor, contract in EPICS:
        goals.append(
            {
                "schema": GOAL_SCHEMA,
                "goal_id": goal_id,
                "parent_goal_id": ROOT_GOAL,
                "epic": epic,
                "title": title,
                "dependencies": [predecessor] if predecessor else [],
                "completion_contract": contract,
                **_goal_governance_fields(epic=epic, contract=contract),
                "task_ids": [
                    task["stable_task_id"]
                    for task in tasks
                    if task["subgoal_id"] == goal_id
                ],
            }
        )

    board: dict[str, Any] = {
        "schema": BOARD_SCHEMA,
        "board_namespace": BOARD_NAMESPACE,
        "plan_revision": PLAN_REVISION,
        "parent_objective": "ExternalAgentAutonomousExecutionFabric",
        "source_reconciliation_manifest_cid": _cid(source),
        "stack_compatibility_manifest_cid": _cid(stack),
        "source_forest_root": source_forest_root,
        "control_plane": {
            "bootstrap": "one embedded DuckDB writer for the initial reconciliation population",
            "continuous": "Quack provides bounded authenticated append/read transport; one fenced local owner independently verifies signed effect envelopes and alone applies transactional mutations to its private DuckDB",
            "history": "DuckLake plus immutable Parquet/IPLD/CAR/IPFS projections; never current coordination authority",
        },
        "control_artifact_ownership": [
            dict(item) for item in CONTROL_ARTIFACT_OWNERSHIP
        ],
        "implementation_order": [epic for _goal, epic, _title, _pred, _contract in EPICS],
        "initial_population_task_ids": [
            task["stable_task_id"] for task in tasks if task["initial_population"]
        ],
        "future_population_rule": "EAAEF-009 must replace every semantic-root sentinel in an immutable Plan R2 before materializing any task numbered 010 or later.",
        "goals": goals,
        "tasks": tasks,
    }
    board["board_cid"] = _cid(board)
    return board, _render_board(board), _render_objectives(board)


def _md(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _render_board(board: dict[str, Any]) -> str:
    lines = [
        "# ExternalAgentAutonomousExecutionFabric task board",
        "",
        f"Board namespace: `{board['board_namespace']}`. Plan revision: `{board['plan_revision']}`.",
        f"Canonical board identity: `{board['board_cid']}`.",
        "",
        "DuckDB is the private transactional mutable database. Quack is the mandatory bounded multi-reader/multi-writer command and projection transport; one fenced local owner verifies every signed effect envelope and is the only process that opens the operational DuckDB. DuckLake is downstream immutable history and analytics only.",
        "",
        "Only the A tasks are in the bootstrap population. EAAEF-009 must bind a current datasets-built semantic root and admit Plan R2 before later tasks are materialized.",
        "",
        "## Generator/source-owned control artifacts",
        "",
        "These campaign-control artifacts are not worker task outputs and may be changed only by their declared generator or reviewed source owner:",
        "",
        *[
            f"- `{item['path']}`: `{item['ownership_class']}`; mutation policy `{item['mutation_policy']}`."
            for item in board["control_artifact_ownership"]
        ],
        "",
        "## Parallel waves",
        "",
        "Within each epic, tasks whose dependencies and read/write/effect conflicts permit may run in parallel. Epic integration gates serialize A through R; exact conflicts and resource leases further constrain each frontier.",
        "",
    ]
    labels = (
        ("stable_task_id", "Stable task ID"),
        ("status", "Status"),
        ("blocked_reason", "Blocked reason"),
        ("completion_mode", "Completion"),
        ("is_schedulable", "Is schedulable"),
        ("initial_population", "Initial population"),
        ("population_state", "Population state"),
        ("priority", "Priority"),
        ("track", "Track"),
        ("parent_goal_id", "Parent goal ID"),
        ("subgoal_id", "Subgoal ID"),
        ("owning_repository", "Owning repository"),
        ("owned_files", "Owned files"),
        ("execution_owned_files", "Execution owned files"),
        ("integration_conflict_keys", "Integration conflict keys"),
        ("source_revisions", "Source revisions"),
        ("source_semantic_state_root", "Source semantic-state root"),
        ("source_control_plane_schema_version", "Source control-plane schema version"),
        ("objective", "Objective"),
        ("dependencies", "Depends on"),
        ("overlap_merge_contracts", "Owned-path overlap merge contracts"),
        ("read_scope", "Read scope"),
        ("write_scope", "Write scope"),
        ("external_effect_scope", "External-effect scope"),
        ("required_capsules", "Required capsules"),
        ("context_artifacts", "Context artifacts"),
        ("resource_request", "Resource request"),
        ("container_profile", "Container profile"),
        ("model_route", "Model route"),
        ("provider_policy", "Provider policy"),
        ("test_requirements", "Test requirements"),
        ("proof_requirements", "Proof requirements"),
        ("completion_contract", "Completion contract"),
        ("lease_and_fencing_requirements", "Lease and fencing requirements"),
        ("idempotency_key", "Idempotency key"),
        ("rollback_or_compensation", "Rollback or compensation"),
        ("required_evidence", "Required evidence"),
        ("terminal_status", "Terminal status"),
        ("allowed_terminal_statuses", "Allowed terminal statuses"),
        ("outcome", "Outcome"),
        ("allowed_outcomes", "Allowed outcomes"),
        ("outcome_status_mapping", "Outcome status mapping"),
        ("final_artifact_identities", "Final artifact identities"),
        ("permitted_effects", "Permitted effects"),
        ("prohibited_effects", "Prohibited effects"),
        ("outputs", "Outputs"),
        ("execution_outputs", "Execution outputs"),
        ("validation", "Validation"),
        ("execution_validation", "Execution validation"),
        ("acceptance", "Acceptance"),
        ("board_namespace", "Board namespace"),
        ("plan_revision", "Plan revision"),
        ("conflict_and_merge_contract", "Conflict and merge contract"),
        ("task_spec_cid", "Task specification CID"),
    )
    for task in board["tasks"]:
        lines.extend([f"## {task['stable_task_id']} {task['title']}", ""])
        for key, label in labels:
            if key in {"dependencies", "outputs"}:
                rendered = ", ".join(str(item) for item in task[key])
            else:
                rendered = _md(task[key])
            lines.append(f"- {label}: {rendered}".rstrip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_objectives(board: dict[str, Any]) -> str:
    lines = ["# ExternalAgentAutonomousExecutionFabric goals", ""]
    for goal in board["goals"]:
        heading = (
            f"## {goal['goal_id']} {goal['title']}"
            if goal["epic"] == "ROOT"
            else f"## {goal['goal_id']} {goal['epic']} — {goal['title']}"
        )
        lines.extend(
            [
                heading,
                "",
                "- Status: active",
                f"- Parent: {goal['parent_goal_id']}".rstrip(),
                f"- Depends on: {', '.join(goal['dependencies'])}".rstrip(),
                f"- Completion contract: {goal['completion_contract']}",
                f"- Desired postconditions: {_md(goal['desired_postconditions'])}",
                f"- Prohibited outcomes: {_md(goal['prohibited_outcomes'])}",
                f"- Scope: {_md(goal['scope'])}",
                f"- Resource budget: {_md(goal['resource_budget'])}",
                f"- Authority ceiling: {_md(goal['authority_ceiling'])}",
                f"- Verification requirements: {_md(goal['verification_requirements'])}",
                f"- Proof requirements: {_md(goal['proof_requirements'])}",
                f"- Human review requirements: {_md(goal['human_review_requirements'])}",
                f"- Completion evidence: {_md(goal['completion_evidence'])}",
                f"- Gap tasks: {', '.join(goal.get('task_ids') or ())}".rstrip(),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic EAAEF JSON and Markdown projections."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare projections without writing files",
    )
    args = parser.parse_args(argv)
    board, markdown, objectives = _build()
    if args.check:
        mismatches: list[str] = []
        expected = {
            JSON_BOARD: json.dumps(board, indent=2, sort_keys=True) + "\n",
            MARKDOWN_BOARD: markdown,
            OBJECTIVES: objectives,
        }
        for path, content in expected.items():
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                mismatches.append(str(path.relative_to(ROOT)))
        print(
            json.dumps(
                {
                    "schema": BOARD_SCHEMA,
                    "valid": not mismatches,
                    "mode": "check",
                    "board_cid": board["board_cid"],
                    "mismatches": mismatches,
                },
                sort_keys=True,
            )
        )
        return 0 if not mismatches else 1
    CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    JSON_BOARD.write_text(
        json.dumps(board, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN_BOARD.write_text(markdown, encoding="utf-8")
    OBJECTIVES.write_text(objectives, encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": BOARD_SCHEMA,
                "valid": True,
                "board_cid": board["board_cid"],
                "goal_count": len(board["goals"]),
                "task_count": len(board["tasks"]),
                "initial_population_count": len(board["initial_population_task_ids"]),
                "outputs": [
                    str(path.relative_to(ROOT))
                    for path in (JSON_BOARD, MARKDOWN_BOARD, OBJECTIVES)
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
