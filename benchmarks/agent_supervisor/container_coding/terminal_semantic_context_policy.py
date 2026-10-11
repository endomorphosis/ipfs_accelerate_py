"""Frozen capsule-selection controls for matched Terminal Bench experiments."""
from __future__ import annotations

import hashlib
from pathlib import Path

DEFAULT_SEMANTIC_CONTEXT_PROFILE = "legacy-bounded@1"
SEMANTIC_CONTEXT_PROFILES = (DEFAULT_SEMANTIC_CONTEXT_PROFILE,
    "indexed-default@1", "indexed-compact@1")
SOURCE_PATHS = (
    "benchmarks/agent_supervisor/container_coding/terminal_semantic_context_policy.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_benchmark.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_harbor_agent.py",
    "benchmarks/agent_supervisor/container_coding/benchmark_controls.py",
    "benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py",
    "benchmarks/agent_supervisor/container_coding/terminal_initial_context.py",
    "benchmarks/agent_supervisor/container_coding/terminal_indexed_preparation.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_context_runtime.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/supervised_task_context.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_candidate_index.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_capsule_selection.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_capsule_expansion.py",
)


def validate_semantic_context_profile(value, *, arm=None):
    if type(value) is not str or value not in SEMANTIC_CONTEXT_PROFILES:
        raise ValueError("explicit supported semantic context profile required")
    if value != DEFAULT_SEMANTIC_CONTEXT_PROFILE and arm not in (None, "full"):
        raise ValueError("semantic context profile requires the full indexed arm")
    return value


def semantic_context_selection(value):
    validate_semantic_context_profile(value)
    return {} if value == DEFAULT_SEMANTIC_CONTEXT_PROFILE else {"semantic_context_profile": value}


def semantic_context_options(value):
    validate_semantic_context_profile(value)
    return dict(worker_selector=("lexical-query-symbol-name@1" if value == DEFAULT_SEMANTIC_CONTEXT_PROFILE
                                  else "duckdb-query-symbol-name@1"),
        worker_capsule_limit=2 if value == "indexed-compact@1" else 8,
        worker_max_bytes=32768)


def semantic_context_objective(value, query):
    """Keep indexed objectives short while retaining the exact instruction elsewhere.

    The complete public text remains mandatory raw source and the ranking query.
    This descriptive label binds that text; it never substitutes for its body.
    Legacy profiles preserve their original whitespace-normalized objective.
    """
    validate_semantic_context_profile(value)
    if type(query) is not str or not query.strip():
        raise ValueError("semantic context requires its public instruction text")
    if value == DEFAULT_SEMANTIC_CONTEXT_PROFILE:
        return " ".join(query.split())
    return ("Public task in .supervisor-instruction.md; SHA-256 "
            + hashlib.sha256(query.encode("utf-8")).hexdigest() + ".")


def require_semantic_context_archive(manifest, value):
    if validate_semantic_context_profile(value) == DEFAULT_SEMANTIC_CONTEXT_PROFILE:
        return
    rows = manifest.get("files") if type(manifest) is dict else None
    if type(rows) is not list or len(rows) > 250_000:
        raise ValueError("semantic context profile requires current runtime source inventory")
    checkout = Path(__file__).resolve().parents[3]
    for relative in SOURCE_PATHS:
        matches = [row for row in rows if type(row) is dict and row.get("path") == "source/" + relative]
        path = checkout / relative
        if (len(matches) != 1 or path.is_symlink() or not path.is_file()
                or path.stat().st_size > 2 * 1024 * 1024
                or matches[0].get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
                or "bytes" in matches[0] and (type(matches[0]["bytes"]) is not int
                    or matches[0]["bytes"] != path.stat().st_size)):
            raise ValueError("semantic context profile requires a newly qualified runtime archive")
