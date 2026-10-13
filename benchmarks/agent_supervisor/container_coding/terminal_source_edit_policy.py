"""Explicit mapped source edits; selection is neither proof nor new write authority."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .terminal_source_view_policy import DEFAULT_SOURCE_VIEW_PROFILE, validate_source_view_profile

DEFAULT_SOURCE_EDIT_MODE = "disabled"
MAPPED_SOURCE_EDIT_MODE = "mapped-span@1"
SOURCE_EDIT_MODES = (DEFAULT_SOURCE_EDIT_MODE, MAPPED_SOURCE_EDIT_MODE)
SOURCE_PATHS = (
    "benchmarks/agent_supervisor/container_coding/terminal_source_edit_policy.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_benchmark.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_harbor_agent.py",
    "benchmarks/agent_supervisor/container_coding/benchmark_controls.py",
    "benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py",
    "benchmarks/agent_supervisor/container_coding/terminal_indexed_preparation.py",
    "benchmarks/agent_supervisor/container_coding/terminal_doctor_dispatch.py",
    "benchmarks/agent_supervisor/container_coding/container_worker_deployment.py",
    "benchmarks/agent_supervisor/container_coding/terminal_source_view_policy.py",
    "benchmarks/agent_supervisor/container_coding/terminal_source_view_observation.py",
    "benchmarks/agent_supervisor/container_coding/terminal_context_audit.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/worker_source_edit.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/worker_source_edit_client.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/worker_source_broker.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/worker_source_client.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/worker_source_views.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/reversible_source_view.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/source_view_edits.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/source_view_policy.py",
)


def _mode(value):
    if type(value) is not str or value not in SOURCE_EDIT_MODES:
        raise ValueError("explicit supported source edit mode required")
    return value


def validate_source_edit_mode(value, *, arm=None, source_view_profile=DEFAULT_SOURCE_VIEW_PROFILE):
    mode = _mode(value)
    if mode != DEFAULT_SOURCE_EDIT_MODE:
        if arm is not None and arm != "full":
            raise ValueError("mapped source edits require the full indexed arm")
        profile = validate_source_view_profile(source_view_profile, arm=arm)
        if profile == DEFAULT_SOURCE_VIEW_PROFILE:
            raise ValueError("mapped source edits require an enabled source view profile")
    return mode


def source_edit_selection(value):
    return {} if _mode(value) == DEFAULT_SOURCE_EDIT_MODE else {"source_edit_mode": value}


def source_edit_observation(value):
    return {} if _mode(value) == DEFAULT_SOURCE_EDIT_MODE else {
        "source_edit_mode": value, "source_edit_authority": False,
        "source_edit_token_savings_claimed": False,
    }


def require_source_edit_archive(manifest, value):
    """Refuse stale selected producers before deployment or provider dispatch."""
    if _mode(value) == DEFAULT_SOURCE_EDIT_MODE:
        return
    rows = manifest.get("files") if type(manifest) is dict else None
    if type(rows) is not list or len(rows) > 250_000:
        raise ValueError("mapped source edits require current runtime source inventory")
    checkout = Path(__file__).resolve().parents[3]
    for relative in SOURCE_PATHS:
        matches = [row for row in rows if type(row) is dict and row.get("path") == "source/" + relative]
        path = checkout / relative
        if (len(matches) != 1 or not path.is_file() or path.is_symlink()
                or path.stat().st_size > 2 * 1024 * 1024
                or matches[0].get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
                or "bytes" in matches[0] and (type(matches[0]["bytes"]) is not int
                    or matches[0]["bytes"] != path.stat().st_size)):
            raise ValueError("mapped source edits require a newly qualified runtime archive")
