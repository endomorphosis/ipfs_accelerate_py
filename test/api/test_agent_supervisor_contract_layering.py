"""ASE3-029 contract layering and zero-upward-import DAG gates."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "ipfs_accelerate_py" / "agent_supervisor"
LOWER_DOMAINS = ("runtime", "todo_daemon")
FORBIDDEN_ENTRYPOINT_PREFIXES = (
    "ipfs_accelerate_py.agent_supervisor.entrypoints",
    "agent_supervisor.entrypoints",
)


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if path.is_file() and "/__pycache__/" not in path.as_posix()
    )


def _module_name(path: Path) -> str:
    relative = path.relative_to(REPO_ROOT).with_suffix("")
    return ".".join(relative.parts)


def _import_targets(tree: ast.AST, *, module_name: str) -> set[str]:
    package_parts = module_name.split(".")[:-1]
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package_parts[: len(package_parts) - node.level + 1]
                if node.module:
                    absolute = ".".join([*base, *node.module.split(".")])
                else:
                    absolute = ".".join(base)
            else:
                absolute = node.module or ""
            if absolute:
                targets.add(absolute)
            for alias in node.names:
                if alias.name == "*":
                    continue
                if absolute:
                    targets.add(f"{absolute}.{alias.name}")
    return targets


def test_runtime_and_todo_daemon_have_zero_entrypoint_imports() -> None:
    violations: list[str] = []
    for domain in LOWER_DOMAINS:
        domain_root = PACKAGE_ROOT / domain
        assert domain_root.is_dir(), domain
        for path in _iter_python_files(domain_root):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            module_name = _module_name(path)
            for target in _import_targets(tree, module_name=module_name):
                if any(
                    target == prefix or target.startswith(prefix + ".")
                    for prefix in FORBIDDEN_ENTRYPOINT_PREFIXES
                ):
                    violations.append(f"{module_name} -> {target}")
                # relative ..entrypoints
                if ".entrypoints" in target or target.endswith("entrypoints"):
                    if "agent_supervisor.entrypoints" in target or target.startswith(
                        "entrypoints"
                    ):
                        violations.append(f"{module_name} -> {target}")
    # Also catch relative imports that resolve to entrypoints via AST level
    for domain in LOWER_DOMAINS:
        for path in _iter_python_files(PACKAGE_ROOT / domain):
            source = path.read_text(encoding="utf-8")
            if "entrypoints" not in source:
                continue
            tree = ast.parse(source, filename=str(path))
            module_name = _module_name(path)
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.level == 0:
                    continue
                # Resolve relative
                parts = module_name.split(".")[:-1]
                if node.level > len(parts):
                    continue
                base = parts[: len(parts) - node.level + 1]
                module = node.module or ""
                absolute = ".".join([*base, *module.split(".")]) if module else ".".join(base)
                if "entrypoints" in absolute.split("."):
                    violations.append(f"{module_name} -> {absolute}")

    assert violations == [], "upward entrypoint imports remain:\n" + "\n".join(
        violations
    )


def test_neutral_contracts_package_imports_are_side_effect_free() -> None:
    forbidden_substrings = (
        "agent_supervisor.entrypoints",
        "agent_supervisor.runtime",
        "agent_supervisor.todo_daemon",
        "agent_supervisor.control.",
    )
    # control is effectful; contracts must not import it
    for path in _iter_python_files(PACKAGE_ROOT / "contracts"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        module_name = _module_name(path)
        for target in _import_targets(tree, module_name=module_name):
            for bad in forbidden_substrings:
                assert bad not in target, f"{module_name} imports {target}"


def test_compatibility_reexports_preserve_object_identity() -> None:
    from ipfs_accelerate_py.agent_supervisor.contracts.execution import (
        InvocationBudget,
    )
    from ipfs_accelerate_py.agent_supervisor.control.plan_execution_store import (
        ConfiguredBoardExecutionSlices,
        ProductionParallelPlanAdapter,
    )
    from ipfs_accelerate_py.agent_supervisor.control.provider_attempt_store import (
        DurableProviderAttemptCAS,
        ProviderAttemptReservation,
    )
    from ipfs_accelerate_py.agent_supervisor.entrypoints import contracts as entry_contracts
    from ipfs_accelerate_py.agent_supervisor.entrypoints import execution_plan as entry_plan
    from ipfs_accelerate_py.agent_supervisor.entrypoints import (
        provider_attempt_store as entry_pas,
    )

    assert entry_contracts.InvocationBudget is InvocationBudget
    assert entry_plan.ConfiguredBoardExecutionSlices is ConfiguredBoardExecutionSlices
    assert entry_plan.ProductionParallelPlanAdapter is ProductionParallelPlanAdapter
    assert entry_pas.DurableProviderAttemptCAS is DurableProviderAttemptCAS
    assert entry_pas.ProviderAttemptReservation is ProviderAttemptReservation


def test_lower_control_services_are_importable() -> None:
    for name in (
        "ipfs_accelerate_py.agent_supervisor.control.plan_execution_store",
        "ipfs_accelerate_py.agent_supervisor.control.provider_attempt_store",
        "ipfs_accelerate_py.agent_supervisor.control.profile_authority",
        "ipfs_accelerate_py.agent_supervisor.contracts.authority",
        "ipfs_accelerate_py.agent_supervisor.contracts.provider_capacity",
    ):
        importlib.import_module(name)
