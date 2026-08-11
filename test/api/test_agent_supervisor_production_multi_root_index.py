"""Production CLI multi-root provider indexing (SCA-603 / SCAEV043MULTIROOT).

Proves ``scripts/index_repository_contracts.py`` scans the superproject with
the three default provider package roots, publishes an independent
``provider-index.json`` ledger, emits symbol counts when providers are
healthy, and fails authority (exit 4) when a required provider is missing —
with zero model/provider/LLM calls.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.repository_snapshot import (
    SCOPE_POLICY_SCHEMA,
)

import importlib.util


def _load_multi_root_helpers():
    helper_path = (
        Path(__file__).resolve().parent
        / "test_agent_supervisor_multi_root_repository_index.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sca_multi_root_helpers", helper_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HELPERS = _load_multi_root_helpers()
_build_superproject = _HELPERS._build_superproject
_policy_for_fixture = _HELPERS._policy_for_fixture
_git = _HELPERS._git

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "index_repository_contracts.py"
)
_ACCELERATE_ROOT = Path(__file__).resolve().parents[2]


def _nest_primary_as_git_worktree(superproject: Path) -> Path:
    """Ensure policy primaryRoot (swissknife) is its own Git worktree root.

    The production CLI inventories the primary tree via
    ``rev-parse --show-toplevel`` equality; monorepo SwissKnife is nested.
    """

    primary = superproject / "swissknife"
    if not (primary / ".git").exists():
        _git(primary, "init", "-q")
        _git(primary, "config", "user.name", "SCA Production Multi-Root")
        _git(primary, "config", "user.email", "sca-prod-multi-root@example.invalid")
        _git(primary, "add", ".")
        _git(primary, "commit", "-qm", "primary swissknife fixture")
    return primary


def _run_cli(*args: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ACCELERATE_ROOT) + (
        (":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else ""
    )
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
        env=env,
    )


def _production_scope_policy() -> dict[str, object]:
    policy = dict(_policy_for_fixture())
    policy["schema"] = SCOPE_POLICY_SCHEMA
    policy["scopeId"] = "test-sca-production-multi-root-v1"
    # Production default is the three provider packages (Mcp-Plus-Plus optional).
    policy["providerScopes"] = [
        "external/ipfs_accelerate",
        "external/ipfs_kit",
        "external/ipfs_datasets",
    ]
    return policy


def test_production_cli_indexes_primary_and_three_provider_roots(
    tmp_path: Path,
) -> None:
    superproject, _commits = _build_superproject(tmp_path)
    _nest_primary_as_git_worktree(superproject)
    scope = tmp_path / "scope.json"
    # Monorepo shape: --repo-root is the superproject Git root; primaryRoot
    # swissknife is a nested worktree; providerScopes live under external/.
    scope.write_text(json.dumps(_production_scope_policy()), encoding="utf-8")
    output = tmp_path / "baseline"

    completed = _run_cli(
        "--repo-root",
        str(superproject),
        "--scope-config",
        str(scope),
        "--output-root",
        str(output),
        "--skip-extraction",
        "--include-provider-indexes",
        "--max-paths",
        "200",
    )
    assert completed.returncode in (0, 3, 4), (
        completed.returncode,
        completed.stdout[-2000:],
        completed.stderr[-2000:],
    )
    assert completed.stdout.strip(), completed.stderr[-2000:]
    summary = json.loads(completed.stdout.strip().splitlines()[-1])
    assert summary["llm_call_count"] == 0
    assert summary["provider_call_count"] == 0
    assert summary["model_call_count"] == 0

    provider_index = output / "provider-index.json"
    assert provider_index.is_file(), completed.stderr[-2000:]
    payload = json.loads(provider_index.read_text(encoding="utf-8"))
    assert payload["evidence_id"] == "SCAEV043MULTIROOT"
    assert payload["bodies_in_cas"] is True
    packages = {item["package"] for item in payload["providers"]}
    assert packages == {
        "ipfs_accelerate_py",
        "ipfs_kit_py",
        "ipfs_datasets_py",
    }
    for item in payload["providers"]:
        assert item["opaque_gitlink"] is False
        assert item["indexed"] is True
        assert item.get("head_commit_id") or item.get("snapshot_id")
        if item.get("symbol_extraction", {}).get("complete"):
            assert item["symbol_count"] >= 1

    multi = summary.get("multi_root_providers") or {}
    assert multi.get("included") is True
    assert multi.get("provider_count") == 3
    assert multi.get("llm_call_count") == 0
    assert set(multi.get("expected_packages") or ()) == packages


def test_production_cli_fails_authority_on_missing_provider(
    tmp_path: Path,
) -> None:
    superproject, _commits = _build_superproject(tmp_path)
    _nest_primary_as_git_worktree(superproject)
    shutil.rmtree(superproject / "external" / "ipfs_kit")
    scope = tmp_path / "scope.json"
    scope.write_text(json.dumps(_production_scope_policy()), encoding="utf-8")
    output = tmp_path / "baseline"

    completed = _run_cli(
        "--repo-root",
        str(superproject),
        "--scope-config",
        str(scope),
        "--output-root",
        str(output),
        "--skip-extraction",
        "--include-provider-indexes",
        "--require-provider-authority",
        "--max-paths",
        "200",
    )
    assert completed.returncode == 4, (
        completed.returncode,
        completed.stdout[-1500:],
        completed.stderr[-1500:],
    )
    summary = json.loads(completed.stdout.strip().splitlines()[-1])
    multi = summary["multi_root_providers"]
    assert multi["exhaustive_parity_allowed"] is False
    assert multi["has_blocking_contradictions"] or not multi["all_providers_indexed"]
    assert summary["llm_call_count"] == 0


def test_skip_provider_indexes_disables_multi_root(tmp_path: Path) -> None:
    superproject, _commits = _build_superproject(tmp_path)
    _nest_primary_as_git_worktree(superproject)
    scope = tmp_path / "scope.json"
    scope.write_text(json.dumps(_production_scope_policy()), encoding="utf-8")
    output = tmp_path / "baseline"
    completed = _run_cli(
        "--repo-root",
        str(superproject),
        "--scope-config",
        str(scope),
        "--output-root",
        str(output),
        "--skip-extraction",
        "--skip-provider-indexes",
        "--max-paths",
        "50",
    )
    assert completed.returncode in (0, 3), (
        completed.returncode,
        completed.stderr[-1500:],
    )
    summary = json.loads(completed.stdout.strip().splitlines()[-1])
    assert summary.get("multi_root_providers", {}).get("included") is False
    assert not (output / "provider-index.json").exists()
