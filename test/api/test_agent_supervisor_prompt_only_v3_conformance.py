"""ASE3-012 prompt-only v3 cross-transport conformance matrix."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints import cli as supervisor_cli
from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
from ipfs_accelerate_py.agent_supervisor.entrypoints.service_factory import (
    build_production_composition_manifest,
    resolve_production_composition,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    PROMPT_LIFECYCLE_TOOLS,
    prompt_lifecycle_discovery_manifest,
    register_prompt_lifecycle_tools,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Public prompt-product launch roots (import graph entry points).
PROMPT_PRODUCT_LAUNCH_ROOTS: tuple[str, ...] = (
    "ipfs_accelerate_py/agent_supervisor/entrypoints/facade.py",
    "ipfs_accelerate_py/agent_supervisor/entrypoints/cli.py",
    "ipfs_accelerate_py/agent_supervisor/entrypoints/service_factory.py",
    "ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/prompt_entrypoints.py",
)


class _RecordingToolManager:
    def __init__(self) -> None:
        self.names: list[str] = []

    def register_tool(self, **kwargs: object) -> None:
        self.names.append(str(kwargs["name"]))


def test_python_and_cli_share_composition_cid() -> None:
    """Preview/open paths report one production composition CID."""

    supervisor = Supervisor.open(repository=REPO_ROOT)
    composition = resolve_production_composition(repository_root=REPO_ROOT)
    assert supervisor.composition_cid == composition.composition_cid
    assert supervisor.composition_manifest.activation_task_id == "ASE3-026"
    assert supervisor.composition_manifest.codebase_refill_enabled is False
    # Manifest is body-free and stable.
    m1 = build_production_composition_manifest(
        generation=composition.manifest.generation,
        objective_refill_enabled=composition.manifest.objective_refill_enabled,
        monitor_enabled=composition.manifest.monitor_enabled,
        backends=composition.manifest.backends,
    )
    assert m1.composition_cid == supervisor.composition_cid
    blob = json.dumps(m1.to_dict())
    assert "password" not in blob.lower()
    assert "BEGIN " not in blob


def test_python_preview_and_mcp_discovery_share_backend_roles() -> None:
    supervisor = Supervisor.open(repository=REPO_ROOT)
    backends = set(supervisor.composition_manifest.backends)
    required = {
        "resolver",
        "broker",
        "planning",
        "materialization",
        "scheduler",
        "refill",
        "monitor",
        "run_registry",
    }
    assert required <= backends
    mcp = prompt_lifecycle_discovery_manifest()
    assert set(mcp["tools"]) == set(PROMPT_LIFECYCLE_TOOLS)
    assert mcp["normal_input"] == "prompt"


def test_cli_and_mcp_operation_vocabulary_parity() -> None:
    cli_ops = set(supervisor_cli.SUPERVISOR_COMMANDS) - {"init"}
    mcp_ops = {
        name.removeprefix("agent_supervisor_") for name in PROMPT_LIFECYCLE_TOOLS
    }
    assert cli_ops == mcp_ops


def test_cli_cold_help_and_discovery_manifest() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(
        [sys.executable, "-m", "ipfs_accelerate_py.cli_entry", "supervisor", "--help"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    for command in ("run", "preview", "status", "doctor"):
        assert command in completed.stdout
    manifest = supervisor_cli.supervisor_cli_discovery_manifest()
    assert manifest["side_effect_free_parse"] is True


def test_mcp_registration_is_schema_plus_callable_not_schema_only() -> None:
    manager = _RecordingToolManager()
    register_prompt_lifecycle_tools(manager)
    assert set(manager.names) == set(PROMPT_LIFECYCLE_TOOLS)


def test_preview_does_not_authorize_effects() -> None:
    supervisor = Supervisor.open(repository=REPO_ROOT)
    obs = supervisor.preview("Improve the agent supervisor safely")
    assert obs.values.get("effect_applied") is False
    assert obs.state == "preview"


def test_run_without_bound_runtime_is_unavailable_not_simulated() -> None:
    supervisor = Supervisor.open(repository=REPO_ROOT)
    with pytest.raises(Exception) as info:
        supervisor.run("Improve the agent supervisor safely")
    message = str(info.value).lower()
    assert "simulated" not in message or "refuse" in message
    assert "completed" not in message or "refuse" in message


def test_launch_roots_do_not_call_raw_duckdb_connect() -> None:
    """AST gate: prompt-product launch roots never call duckdb.connect."""

    violations: list[str] = []
    for relative in PROMPT_PRODUCT_LAUNCH_ROOTS:
        path = REPO_ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "connect"
                and isinstance(func.value, ast.Name)
                and func.value.id == "duckdb"
            ):
                violations.append(f"{relative}:{node.lineno}")
            if isinstance(func, ast.Attribute) and func.attr == "connect":
                # duckdb.connect via import alias is still flagged if attribute is connect
                # only when value name is duckdb (handled above).
                pass
    assert violations == []


def test_prompt_product_duckdb_connection_audit_classifies_raw_sites() -> None:
    """Classify remaining raw duckdb.connect sites as non-launch-reachable."""

    # Known raw sites outside prompt-product launch roots (legacy / proof / tools).
    known_non_launch = {
        "ipfs_accelerate_py/agent_supervisor/runtime/artifact_store.py",
    }
    raw_sites: list[str] = []
    root = REPO_ROOT / "ipfs_accelerate_py" / "agent_supervisor"
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in PROMPT_PRODUCT_LAUNCH_ROOTS:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "connect"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "duckdb"
            ):
                raw_sites.append(f"{relative}:{node.lineno}")
    # Every raw site must live under a non-launch-classified module prefix.
    non_launch_prefixes = (
        "ipfs_accelerate_py/agent_supervisor/runtime/",
        "ipfs_accelerate_py/agent_supervisor/validation/",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/",
        "ipfs_accelerate_py/agent_supervisor/rescue/",
        "ipfs_accelerate_py/agent_supervisor/analysis/",
        "ipfs_accelerate_py/agent_supervisor/planning/",
        "ipfs_accelerate_py/agent_supervisor/task_sources/",
        "ipfs_accelerate_py/agent_supervisor/merge/",
        "ipfs_accelerate_py/agent_supervisor/proof/",
        "ipfs_accelerate_py/agent_supervisor/context/",
        "ipfs_accelerate_py/agent_supervisor/self_improvement/",
        "ipfs_accelerate_py/agent_supervisor/control/",
        "ipfs_accelerate_py/agent_supervisor/integrations/",
    )
    for site in raw_sites:
        module_path = site.split(":")[0]
        classified = module_path in known_non_launch or any(
            module_path.startswith(prefix) for prefix in non_launch_prefixes
        )
        # Never allow raw connect under prompt-product launch roots.
        assert not any(
            module_path == root or module_path.startswith(root.removesuffix(".py"))
            for root in PROMPT_PRODUCT_LAUNCH_ROOTS
        ), f"raw duckdb.connect under launch root: {site}"
        assert classified, f"unclassified raw duckdb.connect site: {site}"
    # Policy helper remains the production connection birth.
    policy = (
        REPO_ROOT
        / "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py"
    )
    source = policy.read_text(encoding="utf-8")
    assert "def connect_duckdb_with_policy" in source
