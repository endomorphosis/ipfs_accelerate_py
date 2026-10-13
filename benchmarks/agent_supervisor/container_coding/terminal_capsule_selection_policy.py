"""Explicit source-bound capsule ranking for immutable supervisor trials."""
from __future__ import annotations

import hashlib
from pathlib import Path


DEFAULT_CAPSULE_SELECTION = "lexical-query-symbol-name@1"
RETRIEVAL_CAPSULE_SELECTION = "verified-retrieval-symbol-rank@1"
CAPSULE_SELECTIONS = (DEFAULT_CAPSULE_SELECTION, RETRIEVAL_CAPSULE_SELECTION)
SOURCE_PATHS = (
    "benchmarks/agent_supervisor/container_coding/terminal_capsule_selection_policy.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_benchmark.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_harbor_agent.py",
    "benchmarks/agent_supervisor/container_coding/benchmark_controls.py",
    "benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py",
    "benchmarks/agent_supervisor/container_coding/terminal_indexed_preparation.py",
    "benchmarks/agent_supervisor/container_coding/terminal_initial_context.py",
    "benchmarks/agent_supervisor/container_coding/terminal_context_audit.py",
    "benchmarks/agent_supervisor/container_coding/terminal_program_population.py",
    "benchmarks/agent_supervisor/container_coding/terminal_retrieval_selection.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_capsule_selection.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_context_runtime.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/context_selection_observation.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_router_translation.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/code_retrieval_context.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/published_task_context.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/published_retrieval.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/published_learned_retrieval.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/task_context_bundle.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/terminal_source_partition.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py",
    "ipfs_accelerate_py/agent_supervisor/entrypoints/admitted_benchmark_runtime.py",
    "ipfs_accelerate_py/agent_supervisor/analysis/code_symbol_vector_index.py",
    "ipfs_accelerate_py/agent_supervisor/analysis/program_ast_adapters.py",
    "ipfs_accelerate_py/agent_supervisor/semantic_state/datasets_adapter.py",
)


def validate_capsule_selection(value: str, *, arm: str | None = None) -> str:
    if type(value) is not str or value not in CAPSULE_SELECTIONS:
        raise ValueError("explicit supported capsule selection required")
    if value != DEFAULT_CAPSULE_SELECTION and arm is not None and arm != "full":
        raise ValueError("retrieval-ranked capsules require the full indexed arm")
    return value


def capsule_selection_options(value: str) -> dict:
    value = validate_capsule_selection(value)
    return {} if value == DEFAULT_CAPSULE_SELECTION else {"capsule_selection": value}


def capsule_selection_observation(value: str) -> dict:
    value = validate_capsule_selection(value)
    return {} if value == DEFAULT_CAPSULE_SELECTION else {"capsule_selection": value,
        "capsule_selection_token_savings_claimed": False, "capsule_selection_authority": False}


def require_capsule_selection_archive(manifest: dict, value: str) -> None:
    """Check opt-in producers against current source before setup or providers."""
    if validate_capsule_selection(value) == DEFAULT_CAPSULE_SELECTION:
        return
    rows = manifest.get("files") if type(manifest) is dict else None
    if type(rows) is not list or len(rows) > 250_000:
        raise ValueError("ranked capsules require current runtime source inventory")
    checkout = Path(__file__).resolve().parents[3]
    for relative in SOURCE_PATHS:
        matches = [row for row in rows if type(row) is dict and row.get("path") == "source/" + relative]
        path = checkout / relative
        if (len(matches) != 1 or not path.is_file() or path.is_symlink()
                or path.stat().st_size > 2 * 1024 * 1024
                or matches[0].get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
                or "bytes" in matches[0] and (type(matches[0]["bytes"]) is not int
                    or matches[0]["bytes"] != path.stat().st_size)):
            raise ValueError("ranked capsules require a newly qualified runtime archive")
