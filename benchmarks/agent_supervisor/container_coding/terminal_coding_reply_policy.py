"""Explicit, archive-bound coding reply selection for Terminal Bench."""
from __future__ import annotations

import hashlib
from pathlib import Path


DEFAULT_CODING_REPLY_MODE = "legacy"
ORDINARY_CODING_REPLY_MODE = "ordinary-completion@1"
CODING_REPLY_MODES = (DEFAULT_CODING_REPLY_MODE, ORDINARY_CODING_REPLY_MODE)


def validate_coding_reply_mode(value: str, *, arm: str | None = None,
                               provider: str | None = None) -> str:
    if type(value) is not str or value not in CODING_REPLY_MODES:
        raise ValueError("explicit supported coding reply mode required")
    if value != DEFAULT_CODING_REPLY_MODE:
        if arm is not None and arm != "full":
            raise ValueError("ordinary completion requires the full indexed arm")
        if provider is not None and provider != "codex_cli":
            raise ValueError("ordinary completion requires the Codex router route")
    return value


def coding_reply_selection(value: str) -> dict:
    value = validate_coding_reply_mode(value)
    return {} if value == DEFAULT_CODING_REPLY_MODE else {"coding_reply_mode": value}


def require_coding_reply_archive(manifest: dict, value: str) -> None:
    if validate_coding_reply_mode(value) == DEFAULT_CODING_REPLY_MODE:
        return
    checkout = Path(__file__).resolve().parents[3]
    paths = (
        "benchmarks/agent_supervisor/container_coding/terminal_coding_reply_policy.py",
        "benchmarks/agent_supervisor/container_coding/full_supervisor_benchmark.py",
        "benchmarks/agent_supervisor/container_coding/full_supervisor_harbor_agent.py",
        "benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py",
        "benchmarks/agent_supervisor/container_coding/terminal_doctor_dispatch.py",
        "benchmarks/agent_supervisor/container_coding/container_worker_deployment.py",
        "benchmarks/agent_supervisor/container_coding/terminal_context_audit.py",
        "benchmarks/agent_supervisor/container_coding/benchmark_controls.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/coding_reply_contract.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/semantic_router_translation.py",
        "ipfs_accelerate_py/cli_runtime/cli_metadata.py",
        "ipfs_accelerate_py/llm_router.py",
    )
    rows = manifest.get("files")
    if type(rows) is not list or len(rows) > 250_000:
        raise ValueError("coding reply mode requires current runtime source inventory")
    for relative in paths:
        matches = [row for row in rows if type(row) is dict
                   and row.get("path") == "source/" + relative]
        path = checkout / relative
        if (len(matches) != 1 or not path.is_file() or path.is_symlink()
                or path.stat().st_size > 2 * 1024 * 1024
                or matches[0].get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()):
            raise ValueError("coding reply mode requires a newly qualified runtime archive")
