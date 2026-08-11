"""Evidence tests for ASREF-G010, ASREF-G090, and ASREF-G100.

These tests close the parent ASREF-G000 missing-evidence set filed by the
objective gap scanner for ASREF-008:

- ASREF-G010 — bootstrap inventory and frozen move map
- ASREF-G090 — public API package README, entry points, cutover surface
- ASREF-G100 — multi-lane autonomous supervisor execution with Grok
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.asref_layout_evidence import (
    ASREF_DEFAULT_IMPLEMENTATION_PROVIDER,
    ASREF_DOMAIN_PACKAGES,
    ASREF_G000,
    ASREF_G010,
    ASREF_G090,
    ASREF_G100,
    ASREF_MERGE_BRANCH,
    ASREF_PARENT_MISSING_EVIDENCE_TERMS,
    ASREF_PROTECTED_PATHS,
    ASREF_RETIRED_FLAT_ENTRY_STEMS,
    asref_g100_launch_recipe,
    assert_asref_layout_evidence,
    collect_asref_layout_evidence,
    import_inventory_path,
    load_move_map,
    resolve_repo_root,
)


REPO_ROOT = resolve_repo_root()


def test_asref_evidence_goal_ids_are_stable() -> None:
    """Exact goal id strings must remain scannable evidence terms."""

    assert ASREF_PARENT_MISSING_EVIDENCE_TERMS == (
        ASREF_G010,
        ASREF_G090,
        ASREF_G100,
    )
    assert ASREF_G000 == "ASREF-G000"
    assert ASREF_G010 == "ASREF-G010"
    assert ASREF_G090 == "ASREF-G090"
    assert ASREF_G100 == "ASREF-G100"


def test_asref_g010_move_map_and_inventory() -> None:
    """ASREF-G010: frozen move map + import inventory exist and are coherent."""

    payload = load_move_map(REPO_ROOT)
    assert payload.get("branch_target") == ASREF_MERGE_BRANCH
    modules = payload.get("modules") or []
    assert isinstance(modules, (list, dict))
    assert len(modules) >= 50
    package_counts = payload.get("package_counts") or {}
    assert isinstance(package_counts, dict)
    assert package_counts

    inventory = import_inventory_path(REPO_ROOT)
    text = inventory.read_text(encoding="utf-8")
    assert ASREF_G010 in text
    assert "Dynamic import" in text or "dynamic import" in text.lower()
    assert "move_map" in text or "Modules mapped" in text

    plan = REPO_ROOT / "docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md"
    assert plan.is_file()


def test_asref_g090_public_api_and_entry_points() -> None:
    """ASREF-G090: package map README, domain packages, package entry points."""

    from ipfs_accelerate_py import agent_supervisor as package

    readme = REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/README.md"
    assert readme.is_file()
    readme_text = readme.read_text(encoding="utf-8")
    assert ASREF_G090 in readme_text
    assert "Package map" in readme_text or "package map" in readme_text.lower()

    domain = tuple(package.AGENT_SUPERVISOR_DOMAIN_PACKAGES)
    for name in ASREF_DOMAIN_PACKAGES:
        assert name in domain

    owners = dict(package.AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE)
    assert owners["objective_daemon"] == "objectives"
    assert owners["backlog_refinery"] == "objectives"
    assert owners["merge_resolver"] == "merge"

    package_root = REPO_ROOT / "ipfs_accelerate_py/agent_supervisor"
    for name in (
        "core",
        "control",
        "task_sources",
        "objectives",
        "planning",
        "validation",
        "merge",
        "rescue",
        "runtime",
        "self_improvement",
    ):
        assert (package_root / name / "__init__.py").is_file()
        assert (package_root / name / "README.md").is_file()

    for config_name in ("pyproject.toml", "setup.py"):
        text = (REPO_ROOT / config_name).read_text(encoding="utf-8")
        for stem in ASREF_RETIRED_FLAT_ENTRY_STEMS:
            assert not re.search(
                rf"agent_supervisor\.{stem}\b",
                text,
            ), f"{config_name} still references flat {stem}"
        assert (
            "agent_supervisor.objectives.objective_daemon" in text
            or "agent_supervisor.objectives.backlog_refinery" in text
        )
        assert "agent_supervisor.merge.merge_resolver" in text

    assert (REPO_ROOT / "docs/NESTED_PACKAGES.md").is_file()


def test_asref_g100_launch_recipe_and_protected_paths() -> None:
    """ASREF-G100: multi-lane Grok launch recipe + protected architecture paths."""

    recipe = asref_g100_launch_recipe(
        lanes=4,
        provider="grok",
        dry_run=True,
        enable_objective_refill=True,
    )
    assert recipe["goal_id"] == ASREF_G100
    assert recipe["implementation_provider"] == "grok"
    assert recipe["merge_branch"] == ASREF_MERGE_BRANCH
    assert recipe["root_goal_id"] == ASREF_G000
    assert set(recipe["protected_paths"]) == set(ASREF_PROTECTED_PATHS)
    assert "--implementation-provider" in recipe["argv"]
    assert "grok" in recipe["argv"]
    assert "--dry-run" in recipe["argv"]
    assert ASREF_DEFAULT_IMPLEMENTATION_PROVIDER == "auto"

    launch_script = REPO_ROOT / "scripts/ops/asref_module_refactor_supervisor.py"
    multi = REPO_ROOT / "scripts/ops/agent_supervisor/asref_multi_lane_launch.py"
    entry = REPO_ROOT / (
        "scripts/ops/agent_supervisor/implementation_supervisor_entry.py"
    )
    assert launch_script.is_file()
    assert multi.is_file()
    assert entry.is_file()

    launch_text = launch_script.read_text(encoding="utf-8")
    multi_text = multi.read_text(encoding="utf-8")
    assert ASREF_G100 in launch_text
    assert ASREF_G100 in multi_text
    for protected in ASREF_PROTECTED_PATHS:
        assert protected in launch_text
    assert "ImplementationSupervisorTrackConfig" in launch_text
    assert "grok" in launch_text.lower()
    assert "grok" in multi_text.lower()


def test_collect_asref_layout_evidence_passes() -> None:
    """Full structural gate for ASREF-G010 / ASREF-G090 / ASREF-G100."""

    report = collect_asref_layout_evidence(REPO_ROOT)
    failed = [c for c in report.checks if not c.ok]
    assert not report.errors, report.errors
    assert not failed, [
        f"{c.goal_id}/{c.check_id}: {c.detail}" for c in failed
    ]
    assert report.goal_ok(ASREF_G010)
    assert report.goal_ok(ASREF_G090)
    assert report.goal_ok(ASREF_G100)
    assert_asref_layout_evidence(REPO_ROOT)


def test_asref_layout_evidence_cli_exits_zero() -> None:
    """Module CLI prints JSON and exits 0 when evidence is complete."""

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ipfs_accelerate_py.agent_supervisor.asref_layout_evidence",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["goal_status"][ASREF_G010] is True
    assert payload["goal_status"][ASREF_G090] is True
    assert payload["goal_status"][ASREF_G100] is True


def test_asref_g100_recipe_cli() -> None:
    """ASREF-G100 multi-lane recipe subcommand returns structured JSON."""

    proc = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts/ops/agent_supervisor/asref_multi_lane_launch.py"
            ),
            "recipe",
            "--lanes",
            "2",
            "--implementation-provider",
            "grok",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # Banner + recipe are both JSON objects printed sequentially.
    assert ASREF_G100 in proc.stdout
    assert "grok" in proc.stdout
    assert "protected_paths" in proc.stdout


def test_asref_module_refactor_supervisor_preflight_runs() -> None:
    """Supervisor preflight includes ASREF-G100 layout evidence fields."""

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/ops/asref_module_refactor_supervisor.py"),
            "preflight",
            "--lanes",
            "2",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    # Preflight may exit 2 when HEAD is not the merge branch or the board
    # has no ready tasks; the JSON payload must still be valid evidence.
    assert proc.stdout.strip(), proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["schema"] == "ipfs_accelerate_py.asref.preflight.v1"
    assert payload["asref_autonomous_goal_id"] == ASREF_G100
    assert ASREF_G010 in payload["asref_layout_evidence_goals"]
    assert ASREF_G090 in payload["asref_layout_evidence_goals"]
    assert ASREF_G100 in payload["asref_layout_evidence_goals"]
    assert payload["asref_goal_status"][ASREF_G010] is True
    assert payload["asref_goal_status"][ASREF_G090] is True
    assert payload["asref_goal_status"][ASREF_G100] is True
    assert payload["inventory_present"] is True
    assert set(payload["protected_paths"]) == set(ASREF_PROTECTED_PATHS)
    assert payload["default_implementation_provider"] == "auto"


def test_semantic_layout_export_aliases() -> None:
    """Semantic layout names are canonical; board-prefix spellings are aliases."""

    from ipfs_accelerate_py import agent_supervisor as package

    assert package.AGENT_SUPERVISOR_CORE_PACKAGES == ("core",)
    assert package.AGENT_SUPERVISOR_G020_PACKAGES is package.AGENT_SUPERVISOR_CORE_PACKAGES
    assert (
        package.AGENT_SUPERVISOR_EVIDENCE_CLUSTER_G020_G050
        is package.AGENT_SUPERVISOR_FOUNDATION_LAYOUT_GOAL_IDS
    )
    assert (
        package.AGENT_SUPERVISOR_LANDED_MODULE_OWNERS
        is package.AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE
    )
    assert (
        package.AGENT_SUPERVISOR_PUBLIC_API_EXPORTS
        is package.AGENT_SUPERVISOR_V2_STABLE_EXPORTS
    )
    assert package.AGENT_SUPERVISOR_DOMAIN_LAYOUT_CUTOVER_GOAL_ID == "ASREF-G090"
    assert "core" in package.AGENT_SUPERVISOR_DOMAIN_PACKAGES
