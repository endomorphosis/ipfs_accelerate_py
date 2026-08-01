"""Tests for the thin IPFS Kit VFS symbolic assurance ops facade (LPR-027)."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = (
    REPO_ROOT
    / "scripts"
    / "ops"
    / "agent_supervisor"
    / "ipfs_kit_vfs_symbolic_assurance.py"
)
INTEGRATION = (
    REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "integrations"
    / "ipfs_kit_vfs_assurance.py"
)

_FORBIDDEN_WRAPPER_LOGIC = re.compile(
    r"\b(?:scan_|inventory_repository|evaluate_adversarial|ProgramGraph|"
    r"build_repository_forest|duckdb|sqlite3|socket\.|urllib|"
    r"requests\.|openai|anthropic|neo4j)\b"
)


def _run_cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    command_env = {
        **dict(**{k: v for k, v in __import__("os").environ.items()}),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "IPFS_ACCEL_SKIP_CORE": "1",
        "PYTHONPATH": str(REPO_ROOT)
        + (
            ":" + __import__("os").environ["PYTHONPATH"]
            if __import__("os").environ.get("PYTHONPATH")
            else ""
        ),
    }
    if env:
        command_env.update(env)
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=command_env,
        check=False,
    )


def test_wrapper_contains_only_argument_config_bootstrap_delegation_code():
    source = CLI.read_text(encoding="utf-8")
    tree = ast.parse(source)
    # No class definitions that implement domain engines.
    assert not [n for n in tree.body if isinstance(n, ast.ClassDef)]
    forbidden = _FORBIDDEN_WRAPPER_LOGIC.findall(source)
    assert forbidden == [], f"wrapper embeds engine/provider logic: {forbidden}"
    # Must mention argparse, config bootstrap, and delegation only.
    assert "argparse" in source
    assert "load_assurance_config" in source or "dispatch" in source
    assert "build_parser" in source
    assert "main" in source


def test_help_starts_no_process_opens_no_db_accesses_no_network_or_storage():
    script = f"""
import json, sys, os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
forbidden_roots = (
    'torch', 'transformers', 'openai', 'anthropic', 'neo4j', 'duckdb',
    'psycopg2', 'sqlalchemy', 'requests', 'httpx', 'aiohttp',
)
before = {{name for name in sys.modules if name.split('.')[0] in forbidden_roots}}
# Import wrapper module by path without executing main.
import importlib.util
spec = importlib.util.spec_from_file_location(
    'ipfs_kit_vfs_symbolic_assurance_cli',
    {str(CLI)!r},
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
# --help via parser only
parser = mod.build_parser()
help_text = parser.format_help()
after = {{name for name in sys.modules if name.split('.')[0] in forbidden_roots}}
print(json.dumps({{
    'added': sorted(after - before),
    'has_inventory': 'inventory' in help_text,
    'has_rollout': 'rollout' in help_text,
    'has_verify': 'verify' in help_text,
    'optional_providers': False,
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={
            **dict(__import__("os").environ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(REPO_ROOT),
        },
    )
    payload = json.loads(completed.stdout)
    assert payload["added"] == []
    assert payload["has_inventory"]
    assert payload["has_rollout"]
    assert payload["has_verify"]


def test_cli_help_exit_code_is_success():
    result = _run_cli("--help")
    assert result.returncode == 0
    assert "inventory" in result.stdout
    assert "rollout" in result.stdout
    assert "verify" in result.stdout


def test_cli_missing_command_is_usage_error():
    result = _run_cli()
    assert result.returncode == 2


def test_cli_rollout_shadow_default_and_mutation_disabled():
    result = _run_cli("rollout", "--mode", "shadow")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["automatic_mutation_enabled"] is False
    assert payload["decision"]["effective_mode"] == "shadow"
    assert payload["adversarial_e2e_gate"]["schema"] == "vfs/adversarial-e2e-gate@1"
    assert payload["status"]["schema"] == "vfs/symbolic-bounded-status@1"
    assert payload["findings"]["schema"] == "vfs/symbolic-bounded-findings@1"
    assert payload["receipts"]["schema"] == "vfs/symbolic-bounded-receipts@1"


def test_cli_rollout_assist_and_verify_exit_semantics():
    assist = _run_cli("rollout", "--mode", "assist")
    assert assist.returncode == 0, assist.stderr
    payload = json.loads(assist.stdout)
    assert payload["decision"]["effective_mode"] == "assist"
    assert payload["decision"]["automatic_mutation_enabled"] is False

    verify = _run_cli("verify")
    assert verify.returncode == 0, verify.stderr
    verified = json.loads(verify.stdout)
    assert verified["verified"] is True
    assert verified["automatic_mutation_enabled"] is False


def test_cli_contracts_projects_mappings():
    result = _run_cli("contracts")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "read" in payload["operations"]
    assert "path-normalized" in payload["invariants"]
    assert "not-found" in payload["error_codes"]
    assert payload["authority_flags"]["inventory_authorizes_repair"] is False


def test_cli_pilot_benchmark_parity_differential_bootstrap():
    for command in ("pilot", "benchmark", "parity", "differential"):
        result = _run_cli(command)
        assert result.returncode == 0, f"{command}: {result.stderr}"
        payload = json.loads(result.stdout)
        assert payload["adapter"] == command
        assert payload["automatic_mutation_enabled"] is False


def test_cold_import_of_integration_loads_no_optional_providers():
    script = """
import json, sys, os
os.environ["IPFS_ACCEL_SKIP_CORE"] = "1"
forbidden = ("torch", "transformers", "openai", "anthropic", "neo4j", "duckdb")
before = {name for name in sys.modules if name.split(".")[0] in forbidden}
from ipfs_accelerate_py.agent_supervisor.integrations import ipfs_kit_vfs_assurance as m
after = {name for name in sys.modules if name.split(".")[0] in forbidden}
print(json.dumps({
    "added": sorted(after - before),
    "optional": list(m.optional_providers_loaded()),
    "closed": sorted(m.CLOSED_ADAPTERS),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={
            **dict(__import__("os").environ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "IPFS_ACCEL_SKIP_CORE": "1",
            "PYTHONPATH": str(REPO_ROOT),
        },
    )
    payload = json.loads(completed.stdout)
    assert payload["added"] == []
    assert payload["optional"] == []
    assert set(payload["closed"]) == {
        "inventory",
        "contracts",
        "differential",
        "parity",
        "benchmark",
        "pilot",
        "rollout",
        "verify",
    }


def test_cli_automatic_desired_mode_cannot_enable_mutation():
    result = _run_cli("rollout", "--mode", "automatic")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["decision"]["effective_mode"] == "shadow"
    assert payload["decision"]["automatic_mutation_enabled"] is False
    assert payload["automatic_mutation_enabled"] is False
