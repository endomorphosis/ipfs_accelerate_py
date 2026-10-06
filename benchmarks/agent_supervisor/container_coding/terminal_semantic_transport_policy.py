"""Explicit coding-context transport selection for immutable Terminal Bench runs."""
from __future__ import annotations

import hashlib
from pathlib import Path


DEFAULT_SEMANTIC_TRANSPORT_SCHEMA = "supervisor-semantic-router-input@1"
CONTROLLER_DICTIONARY_TRANSPORT_SCHEMA = "supervisor-semantic-router-input@2"
SEMANTIC_TRANSPORT_SCHEMAS = (
    DEFAULT_SEMANTIC_TRANSPORT_SCHEMA, CONTROLLER_DICTIONARY_TRANSPORT_SCHEMA,
)


def validate_semantic_transport_schema(value: str, *, arm: str | None = None) -> str:
    """No environment selection or implicit migration of historical runs."""
    if type(value) is not str or value not in SEMANTIC_TRANSPORT_SCHEMAS:
        raise ValueError("explicit supported semantic transport schema required")
    if value != DEFAULT_SEMANTIC_TRANSPORT_SCHEMA and arm is not None and arm != "full":
        raise ValueError("controller dictionary transport requires the full indexed arm")
    return value


def semantic_transport_selection(value: str) -> dict:
    """Omit the legacy default so existing config and receipt shapes remain stable."""
    value = validate_semantic_transport_schema(value)
    return {} if value == DEFAULT_SEMANTIC_TRANSPORT_SCHEMA else {"semantic_transport_schema": value}


def require_semantic_transport_archive(manifest: dict, value: str) -> None:
    """Reject an old or incompatible archive before any provider invocation.

    The opt-in transport is qualified against the current source implementation.
    Preparation separately verifies the immutable archive and manifest digests.
    Legacy transport does not acquire this new source requirement.
    """
    if validate_semantic_transport_schema(value) == DEFAULT_SEMANTIC_TRANSPORT_SCHEMA:
        return
    checkout = Path(__file__).resolve().parents[3]
    paths = (
        "benchmarks/agent_supervisor/container_coding/terminal_semantic_transport_policy.py",
        "benchmarks/agent_supervisor/container_coding/full_supervisor_benchmark.py",
        "benchmarks/agent_supervisor/container_coding/full_supervisor_harbor_agent.py",
        "benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py",
        "benchmarks/agent_supervisor/container_coding/terminal_doctor_dispatch.py",
        "benchmarks/agent_supervisor/container_coding/container_worker_deployment.py",
        "benchmarks/agent_supervisor/container_coding/benchmark_controls.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/semantic_router_translation.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py",
    )
    rows = manifest.get("files")
    if type(rows) is not list or len(rows) > 250_000:
        raise ValueError("controller dictionary transport requires current runtime source inventory")
    for relative in paths:
        matches = [row for row in rows if type(row) is dict and row.get("path") == "source/" + relative]
        path = checkout / relative
        if (len(matches) != 1 or not path.is_file() or path.is_symlink() or path.stat().st_size > 2 * 1024 * 1024
                or matches[0].get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()):
            raise ValueError("controller dictionary transport requires a newly qualified runtime archive")
