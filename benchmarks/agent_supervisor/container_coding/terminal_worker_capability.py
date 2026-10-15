"""Bind supervisor archives to reviewed worker and signed scope owners.

This is an archive compatibility check, not execution or completion authority.
The source pins cover the selected scope chain, not a complete import closure.
Older archives must be rebuilt, including the original timeout profiles.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .benchmark_resource_profile import coding_timeout_seconds, FORMALIZATION_SOURCE384_PROFILE

KEY = "router_worker_capability"
SCHEMA = "terminal-router-worker-capability@4"
WORKER_CHILD_CUSTODY_SCHEMA = "terminal-worker-child-custody@1"
SOURCE_PATHS = (
    # The owner must retain the observation result before container removal;
    # an installer from a newer checkout cannot silently use an older driver.
    "benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py",
    "benchmarks/agent_supervisor/container_coding/terminal_output_observation.py",
    "benchmarks/agent_supervisor/container_coding/terminal_source_view_policy.py",
    "benchmarks/agent_supervisor/container_coding/terminal_source_view_observation.py",
    "benchmarks/agent_supervisor/container_coding/terminal_numerical_retrieval_observation.py",
    "benchmarks/agent_supervisor/container_coding/terminal_owner_worker_attempt_observation.py",
    "benchmarks/agent_supervisor/container_coding/terminal_doctor_dispatch.py",
    "benchmarks/agent_supervisor/container_coding/terminal_context_audit.py",
    "benchmarks/agent_supervisor/container_coding/terminal_context_evidence.py",
    "benchmarks/agent_supervisor/container_coding/terminal_decision_usage_projection.py",
    "benchmarks/agent_supervisor/container_coding/semantic_decision_input_audit.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_decision_usage.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/cumulative_usage_reconciliation.py",
    "benchmarks/agent_supervisor/container_coding/terminal_world_packet_audit.py",
    "benchmarks/agent_supervisor/container_coding/terminal_world_nomination_audit.py",
    "benchmarks/agent_supervisor/container_coding/container_worker_deployment.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/owner_worker_attempt.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_world_packet.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_world_packet_replay.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_world_invocation_replay.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_world_artifact_store.py",
    "ipfs_accelerate_py/agent_supervisor/prompt/delta_task_packet.py",
    "ipfs_accelerate_py/agent_supervisor/procedure_compiler/world_model.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/representation_displacement.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/source_formalization_context.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/numerical_obligation_broker.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/numerical_obligation_client.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/worker_source_client.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/worker_source_broker.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/worker_source_request_observation.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/worker_source_views.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/source_view_policy.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/worker_candidate_artifacts.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/worker_candidate_selection.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/worker_candidate_context.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/worker_candidate_context_client.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/reversible_source_view.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_router_translation.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_context_runtime.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/context_selection_observation.py",
    "ipfs_accelerate_py/agent_supervisor/validation/validation_runtime.py",
    "ipfs_accelerate_py/llm_router.py",
    "ipfs_accelerate_py/cli_runtime/grok_stream_capture.py",
    "ipfs_accelerate_py/cli_runtime/grok_stream_usage.py",
    "ipfs_accelerate_py/cli_runtime/cli_metadata.py",
    "ipfs_accelerate_py/cli_runtime/grok_native_usage.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/benchmark_budget_policy.py",
    # Keep complete signed task custody and the selected context consumers in
    # the same reviewed archive as its worker. Feature labels alone cannot
    # establish that the canonical declaration or replay owner is present.
    "ipfs_accelerate_py/agent_supervisor/runtime/terminal_task_profile.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/terminal_semantic_scope.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/terminal_source_partition.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/local_planning_admission.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/router_public_instruction.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_instruction_view.py",
    "benchmarks/agent_supervisor/container_coding/terminal_task_profile.py",
    "benchmarks/agent_supervisor/container_coding/terminal_indexed_preparation.py",
    "benchmarks/agent_supervisor/container_coding/terminal_initial_context.py",
    "benchmarks/agent_supervisor/container_coding/terminal_program_population.py",
    "benchmarks/agent_supervisor/container_coding/terminal_context_rebind.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/task_context_bundle.py",
    "ipfs_accelerate_py/agent_supervisor/context/context_compiler.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/native_cli_subreaper.py",
)


def _current_capability():
    from .container_worker_deployment import WORKER_ENTRY
    from ipfs_accelerate_py.agent_supervisor.runtime.terminal_task_profile import (
        SCOPED_SCHEMA, SEMANTIC_SCOPE_SCHEMA,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import (
        MAX_IMPLEMENTATION_TIMEOUT_SECONDS, implementation_timeout_limit,
    )

    root = Path(__file__).resolve().parents[3]
    return {
        "schema": SCHEMA,
        "worker_child_custody_schema": WORKER_CHILD_CUSTODY_SCHEMA,
        "terminal_task_profile_schema": SCOPED_SCHEMA,
        "terminal_semantic_scope_schema": SEMANTIC_SCOPE_SCHEMA,
        "max_timeout_seconds": MAX_IMPLEMENTATION_TIMEOUT_SECONDS,
        "budget_profile_max_timeout_seconds": {
            FORMALIZATION_SOURCE384_PROFILE: implementation_timeout_limit(FORMALIZATION_SOURCE384_PROFILE)},
        "worker_entry_sha256": hashlib.sha256(WORKER_ENTRY.encode()).hexdigest(),
        "source_sha256": {
            "source/" + name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in SOURCE_PATHS
        },
    }


def _matching_inventory(inventory, capability):
    if type(inventory) is not list:
        return False
    for name, digest in capability["source_sha256"].items():
        rows = [row for row in inventory if type(row) is dict and row.get("path") == name]
        if len(rows) != 1 or rows[0].get("sha256") != digest:
            return False
    return True


def worker_capability_for_inventory(inventory):
    """Advertise only the exact executing installer's reviewed source bytes.

    Partial fixtures and archives of other source versions get no new capability.
    They cannot be used for current supervisor preparation or deployment.
    """
    expected_names = {"source/" + name for name in SOURCE_PATHS}
    if type(inventory) is not list or not expected_names.issubset(
        row.get("path") for row in inventory if type(row) is dict
    ):
        return None
    capability = _current_capability()
    return capability if _matching_inventory(inventory, capability) else None


def require_worker_capability(manifest, resource_profile):
    """Require matching child custody support before planning or deployment.

    No legacy profile bypass is safe: every current worker imports the helper.
    The exact source join also refuses a changed helper under a copied feature
    declaration; this declaration never grants process cleanup authority.
    """
    selected = coding_timeout_seconds(resource_profile)
    if type(manifest) is not dict:
        raise ValueError("rebuild runtime archive: matching worker archive capability required")
    expected = _current_capability()
    capability = manifest.get(KEY)
    if (type(capability) is not dict or capability != expected
            or type(capability.get("max_timeout_seconds")) is not int
            or selected > capability["budget_profile_max_timeout_seconds"].get(
                resource_profile, capability["max_timeout_seconds"])
            or not _matching_inventory(manifest.get("files"), expected)):
        raise ValueError("rebuild runtime archive: matching worker archive capability required")
    return dict(capability)
