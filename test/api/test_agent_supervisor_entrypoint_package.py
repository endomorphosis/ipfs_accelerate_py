"""ASE-004 highest-layer entrypoint package and cold-import gates."""

from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor import entrypoints
from ipfs_accelerate_py.agent_supervisor.entrypoints import contracts

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_SUPERVISOR_ROOT = (
    REPO_ROOT / "ipfs_accelerate_py" / "agent_supervisor"
)
PACKAGE_MAP = (
    REPO_ROOT / "docs" / "architecture" / "agent_supervisor" / "PACKAGE_MAP.md"
)


def test_reviewed_contract_exports_preserve_exact_object_identity() -> None:
    assert entrypoints.ENTRYPOINT_PACKAGE_NAME == (
        "ipfs_accelerate_py.agent_supervisor.entrypoints"
    )
    assert entrypoints.ENTRYPOINT_CONTRACT_EXPORTS == tuple(contracts.__all__)
    assert entrypoints.ENTRYPOINT_LAZY_FACADE_EXPORTS == (
        "Supervisor",
        "SupervisorRun",
        "SupervisorObservation",
        "SupervisorError",
        "SupervisorConfigurationError",
        "SupervisorAmbiguityError",
        "SupervisorUnavailableError",
        "ProductionServiceCompositionManifest",
        "ProductionServiceComposition",
        "resolve_production_composition",
        "build_production_composition_manifest",
        "ServiceCompositionError",
        "ActivationNotReadyError",
        "ConfigurationUnavailableError",
    )

    metadata = {
        "ENTRYPOINT_PACKAGE_NAME",
        "ENTRYPOINT_CONTRACT_EXPORTS",
        "ENTRYPOINT_LAZY_FACADE_EXPORTS",
        "ENTRYPOINT_LOWER_DOMAIN_PACKAGES",
    }
    assert set(entrypoints.__all__) == (
        metadata | set(contracts.__all__) | set(entrypoints.ENTRYPOINT_LAZY_FACADE_EXPORTS)
    )
    for name in contracts.__all__:
        assert getattr(entrypoints, name) is getattr(contracts, name)

    # Reloading the composition package does not manufacture replacement
    # classes/enums or eagerly resolve a facade.
    request_type = entrypoints.SupervisorInvocationRequest
    reloaded = importlib.reload(entrypoints)
    assert reloaded.SupervisorInvocationRequest is request_type
    assert not hasattr(reloaded, "SupervisorControlService")
    assert not hasattr(reloaded, "PromptSupervisorService")
    assert not hasattr(reloaded, "PortalImplementationSupervisor")
    # Lazy facade symbols are listed but not eagerly bound as live class objects
    # until first attribute access.
    assert not isinstance(vars(reloaded).get("Supervisor"), type)

def test_import_is_cold_provider_free_and_effect_free(tmp_path: Path) -> None:
    script = r"""
import json
import os
import pathlib
import sqlite3
import subprocess
import sys
import threading

import ipfs_accelerate_py.agent_supervisor

before_modules = set(sys.modules)
before_files = sorted(str(item.relative_to(pathlib.Path.cwd()))
                      for item in pathlib.Path.cwd().rglob("*"))

def forbidden(*args, **kwargs):
    raise AssertionError("entrypoints cold import attempted an external effect")

original_scandir = os.scandir
original_walk = os.walk
original_glob = pathlib.Path.glob
original_rglob = pathlib.Path.rglob
original_iterdir = pathlib.Path.iterdir
subprocess.Popen = forbidden
subprocess.run = forbidden
threading.Thread.start = forbidden
sqlite3.connect = forbidden
os.scandir = forbidden
os.walk = forbidden
pathlib.Path.glob = forbidden
pathlib.Path.rglob = forbidden
pathlib.Path.iterdir = forbidden
if "duckdb" in sys.modules:
    sys.modules["duckdb"].connect = forbidden

from ipfs_accelerate_py.agent_supervisor import entrypoints

after_modules = set(sys.modules)
added = sorted(after_modules - before_modules)
forbidden_prefixes = (
    "ipfs_accelerate_py.agent_supervisor.todo_daemon",
    "ipfs_accelerate_py.agent_supervisor.integrations",
    "ipfs_accelerate_py.agent_supervisor.runtime",
    "ipfs_accelerate_py.agent_supervisor.prompt",
    "ipfs_accelerate_py.agent_supervisor.control.control_plane",
    "ipfs_accelerate_py.agent_supervisor.objectives",
    "duckdb",
    "torch",
    "transformers",
    "openai",
    "neo4j",
)
added_forbidden = [
    name for name in added
    if any(name == prefix or name.startswith(prefix + ".")
           for prefix in forbidden_prefixes)
]
os.scandir = original_scandir
os.walk = original_walk
pathlib.Path.glob = original_glob
pathlib.Path.rglob = original_rglob
pathlib.Path.iterdir = original_iterdir
after_files = sorted(str(item.relative_to(pathlib.Path.cwd()))
                     for item in pathlib.Path.cwd().rglob("*"))
print(json.dumps({
    "added": added,
    "added_forbidden": added_forbidden,
    "before_files": before_files,
    "after_files": after_files,
    "lazy_facades": list(entrypoints.ENTRYPOINT_LAZY_FACADE_EXPORTS),
    "has_service": hasattr(entrypoints, "SupervisorControlService"),
}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), environment.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["added_forbidden"] == []
    assert payload["before_files"] == payload["after_files"] == []
    assert payload["lazy_facades"] == [
        "Supervisor",
        "SupervisorRun",
        "SupervisorObservation",
        "SupervisorError",
        "SupervisorConfigurationError",
        "SupervisorAmbiguityError",
        "SupervisorUnavailableError",
        "ProductionServiceCompositionManifest",
        "ProductionServiceComposition",
        "resolve_production_composition",
        "build_production_composition_manifest",
        "ServiceCompositionError",
        "ActivationNotReadyError",
        "ConfigurationUnavailableError",
    ]
    assert payload["has_service"] is False
    assert {
        "ipfs_accelerate_py.agent_supervisor.entrypoints",
        "ipfs_accelerate_py.agent_supervisor.entrypoints.contracts",
    } <= set(payload["added"])
    # Cold import must not eagerly load facade/service_factory modules.
    assert "ipfs_accelerate_py.agent_supervisor.entrypoints.facade" not in payload["added"]
    assert (
        "ipfs_accelerate_py.agent_supervisor.entrypoints.service_factory"
        not in payload["added"]
    )

def _node_imports_entrypoints(node: ast.AST) -> bool:
    target = "ipfs_accelerate_py.agent_supervisor.entrypoints"
    if isinstance(node, ast.Import):
        return any(
            alias.name == target or alias.name.startswith(target + ".")
            for alias in node.names
        )
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if module == target or module.startswith(target + "."):
            return True
        if node.level and (
            module == "entrypoints" or module.startswith("entrypoints.")
        ):
            return True
        if node.level and not module:
            return any(alias.name == "entrypoints" for alias in node.names)
    if isinstance(node, ast.Call) and node.args:
        function = node.func
        function_name = (
            function.id
            if isinstance(function, ast.Name)
            else function.attr
            if isinstance(function, ast.Attribute)
            else ""
        )
        first = node.args[0]
        if (
            function_name in {"__import__", "import_module"}
            and isinstance(first, ast.Constant)
            and isinstance(first.value, str)
        ):
            return (
                first.value == target
                or first.value.startswith(target + ".")
                or first.value == ".entrypoints"
                or first.value.startswith(".entrypoints.")
            )
    return False


def test_no_lower_domain_package_imports_upward_to_entrypoints() -> None:
    violations: list[str] = []
    expected = set(entrypoints.ENTRYPOINT_LOWER_DOMAIN_PACKAGES)
    present = {
        path.name
        for path in AGENT_SUPERVISOR_ROOT.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }
    assert expected <= present

    for package in entrypoints.ENTRYPOINT_LOWER_DOMAIN_PACKAGES:
        package_root = AGENT_SUPERVISOR_ROOT / package
        for path in sorted(package_root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if _node_imports_entrypoints(node):
                    violations.append(
                        f"{path.relative_to(REPO_ROOT)}:{getattr(node, 'lineno', 0)}"
                    )
    assert violations == []


def test_entrypoint_initializer_imports_contracts_only() -> None:
    path = AGENT_SUPERVISOR_ROOT / "entrypoints" / "__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))

    assert imported == {"__future__", "typing", "."}
    # Lazy facade resolution is allowed via __getattr__, not eager imports.


def test_package_map_documents_highest_layer_and_storage_boundary() -> None:
    package_map = PACKAGE_MAP.read_text(encoding="utf-8")
    readme = (
        AGENT_SUPERVISOR_ROOT / "entrypoints" / "README.md"
    ).read_text(encoding="utf-8")

    assert "entrypoints (prompt-first composition facade)" in package_map
    assert "No upward entrypoint import" in package_map
    assert "`entrypoints/`" in package_map
    assert "ENTRYPOINT_CONTRACT_EXPORTS" in package_map
    assert "DuckDB remains" in package_map
    assert "Parquet/IPLD/CAR/IPFS" in package_map

    assert "highest package" in readme
    assert "no lower package may import `entrypoints`" in readme
    assert "Cold-import contract" in readme
    assert "one elected owner writes each DuckDB" in readme
    assert "IPFS availability never grants" in readme
