"""Opt-in capsule references with exact live expansion; no proof authority."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .terminal_source_view_policy import DEFAULT_SOURCE_VIEW_PROFILE, validate_source_view_profile
from .terminal_semantic_transport_policy import DEFAULT_SEMANTIC_TRANSPORT_SCHEMA

DEFAULT_SEMANTIC_WORKING_SET = "legacy"
SEMANTIC_WORKING_SETS = (DEFAULT_SEMANTIC_WORKING_SET, "capsule-references@1")
SOURCE_PATHS = (
    "benchmarks/agent_supervisor/container_coding/terminal_semantic_working_set_policy.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_benchmark.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_harbor_agent.py",
    "benchmarks/agent_supervisor/container_coding/benchmark_controls.py",
    "benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py",
    "benchmarks/agent_supervisor/container_coding/terminal_doctor_dispatch.py",
    "benchmarks/agent_supervisor/container_coding/container_worker_deployment.py",
    "benchmarks/agent_supervisor/container_coding/terminal_context_audit.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_working_set.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/worker_semantic_evidence.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/worker_semantic_evidence_client.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/worker_source_broker.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/worker_source_client.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/worker_source_edit_client.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/worker_source_request_observation.py",
    "benchmarks/agent_supervisor/container_coding/terminal_source_view_observation.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/worker_source_views.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_router_translation.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py",
)


def _mode(value):
    if type(value) is not str or value not in SEMANTIC_WORKING_SETS:
        raise ValueError("explicit supported semantic working set required")
    return value


def validate_semantic_working_set(value, *, arm=None, source_view_profile=DEFAULT_SOURCE_VIEW_PROFILE,
                                  transport_schema=DEFAULT_SEMANTIC_TRANSPORT_SCHEMA,
                                  metadata_view="legacy"):
    mode = _mode(value)
    if mode != DEFAULT_SEMANTIC_WORKING_SET:
        if arm is not None and arm != "full":
            raise ValueError("semantic working set requires the full indexed arm")
        if validate_source_view_profile(source_view_profile, arm=arm) == DEFAULT_SOURCE_VIEW_PROFILE:
            raise ValueError("semantic working set requires an enabled source view profile")
        if transport_schema != DEFAULT_SEMANTIC_TRANSPORT_SCHEMA or metadata_view != "legacy":
            raise ValueError("semantic working set requires the original transport and legacy metadata view")
    return mode


def semantic_working_set_selection(value):
    return {} if _mode(value) == DEFAULT_SEMANTIC_WORKING_SET else {"semantic_working_set": value}


def semantic_working_set_observation(value):
    return {} if _mode(value) == DEFAULT_SEMANTIC_WORKING_SET else {
        "semantic_working_set": value, "semantic_working_set_proof_authority": False,
        "semantic_working_set_token_savings_claimed": False,
    }


def require_semantic_working_set_archive(manifest, value):
    """Bind every selected producer before deployment or model dispatch."""
    if _mode(value) == DEFAULT_SEMANTIC_WORKING_SET:
        return
    rows = manifest.get("files") if type(manifest) is dict else None
    if type(rows) is not list or len(rows) > 250_000:
        raise ValueError("semantic working set requires current runtime source inventory")
    checkout = Path(__file__).resolve().parents[3]
    for relative in SOURCE_PATHS:
        matches = [row for row in rows if type(row) is dict and row.get("path") == "source/" + relative]
        path = checkout / relative
        if (len(matches) != 1 or not path.is_file() or path.is_symlink()
                or path.stat().st_size > 2 * 1024 * 1024
                or matches[0].get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
                or "bytes" in matches[0] and (type(matches[0]["bytes"]) is not int
                    or matches[0]["bytes"] != path.stat().st_size)):
            raise ValueError("semantic working set requires a newly qualified runtime archive")
