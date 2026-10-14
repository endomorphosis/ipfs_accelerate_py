"""Explicit, frozen selection of the declared single-task planning route."""
from __future__ import annotations

import hashlib
from pathlib import Path

DEFAULT_PLANNING_MODE = "model"
DECLARED_PLANNING_MODE = "declared-contract@1"
PLANNING_MODES = (DEFAULT_PLANNING_MODE, DECLARED_PLANNING_MODE)
SOURCE_PATHS = (
    "benchmarks/agent_supervisor/container_coding/terminal_planning_policy.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_benchmark.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_harbor_agent.py",
    "benchmarks/agent_supervisor/container_coding/benchmark_controls.py",
    "benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py",
    "benchmarks/agent_supervisor/container_coding/terminal_indexed_preparation.py",
    "benchmarks/agent_supervisor/container_coding/terminal_numerical_observation.py",
    "benchmarks/agent_supervisor/container_coding/terminal_doctor_dispatch.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/declared_contract_planning.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/local_planning_admission.py",
    "ipfs_accelerate_py/agent_supervisor/prompt/prompt_workflow.py",
    "ipfs_accelerate_py/agent_supervisor/planning/formal_plan_compiler.py",
    "ipfs_accelerate_py/agent_supervisor/planning/formal_plan_validator.py",
)


def validate_planning_mode(value, *, intent_requirement_contract=None, task_profile=None):
    if type(value) is not str or value not in PLANNING_MODES:
        raise ValueError("explicit supported planning mode required")
    if value == DECLARED_PLANNING_MODE and intent_requirement_contract is not None:
        raise ValueError("declared planning mode cannot replace an explicit Intent requirement contract")
    if value == DECLARED_PLANNING_MODE and isinstance(task_profile, dict):
        from .terminal_task_profile import MULTITASK_SCHEMA
        if task_profile.get("schema") == MULTITASK_SCHEMA:
            raise ValueError("declared planning mode requires a single task")
    return value


def planning_selection(value):
    return {} if validate_planning_mode(value) == DEFAULT_PLANNING_MODE else {"planning_mode": value}


def planning_observation(value):
    """Selected configuration alone establishes neither execution nor savings."""
    return {} if validate_planning_mode(value) == DEFAULT_PLANNING_MODE else {
        "planning_mode": value, "planning_mode_authority": False,
        "planning_mode_token_savings_claimed": False,
    }


def require_planning_archive(manifest, value):
    if validate_planning_mode(value) == DEFAULT_PLANNING_MODE:
        return
    rows = manifest.get("files") if type(manifest) is dict else None
    if type(rows) is not list or len(rows) > 250_000:
        raise ValueError("declared planning requires current runtime source inventory")
    checkout = Path(__file__).resolve().parents[3]
    for relative in SOURCE_PATHS:
        matches = [row for row in rows if type(row) is dict and row.get("path") == "source/" + relative]
        path = checkout / relative
        if (len(matches) != 1 or not path.is_file() or path.is_symlink()
                or path.stat().st_size > 2 * 1024 * 1024
                or matches[0].get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
                or "bytes" in matches[0] and (type(matches[0]["bytes"]) is not int
                    or matches[0]["bytes"] != path.stat().st_size)):
            raise ValueError("declared planning requires a newly qualified runtime archive")
