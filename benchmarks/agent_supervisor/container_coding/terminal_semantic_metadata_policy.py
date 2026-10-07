"""Explicit experimental metadata view selection for frozen coding trials."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .terminal_semantic_transport_policy import DEFAULT_SEMANTIC_TRANSPORT_SCHEMA

DEFAULT_SEMANTIC_METADATA_VIEW = "legacy"
COMMON_BINDINGS_METADATA_VIEW = "common-bindings@1"
SEMANTIC_METADATA_VIEWS = (DEFAULT_SEMANTIC_METADATA_VIEW, COMMON_BINDINGS_METADATA_VIEW)
SOURCE_PATHS = (
    "benchmarks/agent_supervisor/container_coding/terminal_semantic_metadata_policy.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_benchmark.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_harbor_agent.py",
    "benchmarks/agent_supervisor/container_coding/benchmark_controls.py",
    "benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py",
    "benchmarks/agent_supervisor/container_coding/terminal_doctor_dispatch.py",
    "benchmarks/agent_supervisor/container_coding/container_worker_deployment.py",
    "benchmarks/agent_supervisor/container_coding/terminal_context_audit.py",
    "benchmarks/agent_supervisor/container_coding/terminal_coding_reply_policy.py",
    "benchmarks/agent_supervisor/container_coding/terminal_semantic_transport_policy.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_metadata_view.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_metadata_catalog.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/supervisor_meta_index.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/semantic_router_translation.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/coding_reply_contract.py",
    "ipfs_accelerate_py/agent_supervisor/context/context_contracts.py",
    "ipfs_accelerate_py/cli_runtime/cli_metadata.py",
    "ipfs_accelerate_py/llm_router.py",
)


def validate_semantic_metadata_view(value: str, *, arm: str | None = None,
                                   transport_schema: str = DEFAULT_SEMANTIC_TRANSPORT_SCHEMA,
                                   provider: str | None = None) -> str:
    if type(value) is not str or value not in SEMANTIC_METADATA_VIEWS:
        raise ValueError("explicit supported semantic metadata view required")
    if value != DEFAULT_SEMANTIC_METADATA_VIEW:
        if arm is not None and arm != "full":
            raise ValueError("metadata view requires the full indexed arm")
        if transport_schema != DEFAULT_SEMANTIC_TRANSPORT_SCHEMA:
            raise ValueError("metadata view requires the original semantic transport @1")
        if provider is not None and provider not in {"codex_cli", "grok_cli"}:
            raise ValueError("metadata view requires a signed model-router coding route")
    return value


def semantic_metadata_selection(value: str) -> dict:
    value = validate_semantic_metadata_view(value)
    return {} if value == DEFAULT_SEMANTIC_METADATA_VIEW else {"semantic_metadata_view": value}


def semantic_metadata_observation(value: str) -> dict:
    """An explicit experiment declaration is neither measured savings nor authority."""
    return {} if validate_semantic_metadata_view(value) == DEFAULT_SEMANTIC_METADATA_VIEW else {
        "semantic_metadata_view": value, "semantic_metadata_token_savings_claimed": False,
        "semantic_metadata_authority": False,
    }


def require_semantic_metadata_archive(manifest: dict, value: str) -> None:
    if validate_semantic_metadata_view(value) == DEFAULT_SEMANTIC_METADATA_VIEW:
        return
    rows = manifest.get("files") if type(manifest) is dict else None
    if type(rows) is not list or len(rows) > 250_000:
        raise ValueError("metadata view requires current runtime source inventory")
    checkout = Path(__file__).resolve().parents[3]
    for relative in SOURCE_PATHS:
        matches = [row for row in rows if type(row) is dict and row.get("path") == "source/" + relative]
        path = checkout / relative
        if (len(matches) != 1 or not path.is_file() or path.is_symlink()
                or path.stat().st_size > 2 * 1024 * 1024
                or matches[0].get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
                or "bytes" in matches[0] and (type(matches[0]["bytes"]) is not int
                    or matches[0]["bytes"] != path.stat().st_size)):
            raise ValueError("metadata view requires a newly qualified runtime archive")
