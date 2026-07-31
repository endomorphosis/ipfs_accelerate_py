"""Root-layout cutover guard for VFS assurance generalization (LPR-028).

Asserts recursively that the agent_supervisor package root contains no
``vfs_*.py`` implementation or compatibility stub, that no production import
references ``agent_supervisor.vfs_*``, and that generic engines plus the thin
ops wrapper obey placement and purity constraints.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_SUPERVISOR_ROOT = REPO_ROOT / "ipfs_accelerate_py" / "agent_supervisor"
OPS_CLI = (
    REPO_ROOT
    / "scripts"
    / "ops"
    / "agent_supervisor"
    / "ipfs_kit_vfs_symbolic_assurance.py"
)
INTEGRATION = (
    AGENT_SUPERVISOR_ROOT / "integrations" / "ipfs_kit_vfs_assurance.py"
)

GENERIC_ENGINE_MODULES: tuple[Path, ...] = (
    AGENT_SUPERVISOR_ROOT / "analysis" / "repository_surface_inventory.py",
    AGENT_SUPERVISOR_ROOT / "analysis" / "program_contract_profile.py",
    AGENT_SUPERVISOR_ROOT / "analysis" / "interface_contract_parity.py",
    AGENT_SUPERVISOR_ROOT / "validation" / "differential_contract_harness.py",
    AGENT_SUPERVISOR_ROOT / "validation" / "symbolic_efficiency_benchmark.py",
    AGENT_SUPERVISOR_ROOT / "runtime" / "symbolic_assurance_pilot.py",
    AGENT_SUPERVISOR_ROOT / "control" / "symbolic_assurance_rollout.py",
)

# Domain literals and board/fixed-checkout branches must not live in generic
# engines. Profile/integration modules may carry VFS vocabulary.
_FORBIDDEN_GENERIC = re.compile(
    r"(?i)\b(?:"
    r"fsspec|"
    r"swissknife|swiss[_-]?knife|"
    r"ipfs_kit(?!_vfs_assurance)|"
    r"(?<![a-z_])ipfs(?!_accelerate_py|_datasets_py|_kit_vfs_assurance)|"
    r"(?<![a-z_])vfs(?!_seeded|_surface|_manager)|"
    r"board[_-]?id|"
    r"board[_-]?namespace\s*=\s*[\"'](?:vfs|ipfs)"
    r")\b"
)

_IMPORT_VFS_ROOT = re.compile(
    r"(?m)^\s*(?:"
    r"from\s+ipfs_accelerate_py\.agent_supervisor\s+import\s+.*\bvfs_|"
    r"from\s+ipfs_accelerate_py\.agent_supervisor\.vfs_|"
    r"import\s+ipfs_accelerate_py\.agent_supervisor\.vfs_"
    r")"
)

_STRING_IMPORT_VFS = re.compile(
    r"""['"]ipfs_accelerate_py\.agent_supervisor\.vfs_[A-Za-z0-9_]+['"]"""
)

_FORBIDDEN_WRAPPER_LOGIC = re.compile(
    r"\b(?:scan_|inventory_repository|evaluate_adversarial|ProgramGraph|"
    r"build_repository_forest|duckdb|sqlite3|socket\.|urllib|"
    r"requests\.|openai|anthropic|neo4j)\b"
)

# Scan production Python under the package and scripts/ops entry points.
# Docs, architecture maps, and source-lock fixtures may still name historical
# root paths as evidence coordinates.
_SCAN_GLOBS: tuple[str, ...] = (
    "ipfs_accelerate_py/agent_supervisor/**/*.py",
    "scripts/ops/agent_supervisor/**/*.py",
    "test/api/test_agent_supervisor_*.py",
    "test/api/test_vfs_*.py",
    "test/api/test_ipfs_kit_vfs_*.py",
)

_EXCLUDE_NAME_PARTS = (
    "/__pycache__/",
    "/.git/",
)


@dataclass(frozen=True)
class VfsRootLayoutGuard:
    """Receipt for the no-root-vfs placement gate (LPR-028)."""

    schema: str = "ipfs_accelerate_py/agent-supervisor/vfs-root-layout-guard@1"
    root_vfs_modules: tuple[str, ...] = ()
    forbidden_import_hits: tuple[str, ...] = ()
    generic_domain_literal_hits: tuple[str, ...] = ()
    ops_wrapper_logic_hits: tuple[str, ...] = ()
    missing_generic_engines: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return not (
            self.root_vfs_modules
            or self.forbidden_import_hits
            or self.generic_domain_literal_hits
            or self.ops_wrapper_logic_hits
            or self.missing_generic_engines
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "passed": self.passed,
            "root_vfs_modules": list(self.root_vfs_modules),
            "forbidden_import_hits": list(self.forbidden_import_hits),
            "generic_domain_literal_hits": list(self.generic_domain_literal_hits),
            "ops_wrapper_logic_hits": list(self.ops_wrapper_logic_hits),
            "missing_generic_engines": list(self.missing_generic_engines),
            "notes": list(self.notes),
        }


def _is_excluded(path: Path) -> bool:
    posix = path.as_posix()
    return any(part in posix for part in _EXCLUDE_NAME_PARTS)


def _iter_scan_files() -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in _SCAN_GLOBS:
        for path in REPO_ROOT.glob(pattern):
            if not path.is_file() or _is_excluded(path):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def _root_vfs_modules() -> tuple[str, ...]:
    if not AGENT_SUPERVISOR_ROOT.is_dir():
        return ("missing:agent_supervisor_root",)
    hits = sorted(
        path.name
        for path in AGENT_SUPERVISOR_ROOT.glob("vfs_*.py")
        if path.is_file()
    )
    return tuple(hits)


def _forbidden_import_hits() -> tuple[str, ...]:
    hits: list[str] = []
    for path in _iter_scan_files():
        # This guard file and the generalization equivalence suite may mention
        # the forbidden pattern as a string literal under test.
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in {
            "test/api/test_agent_supervisor_vfs_root_layout_guard.py",
            "test/api/test_agent_supervisor_vfs_generalization_equivalence.py",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in _IMPORT_VFS_ROOT.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            hits.append(f"{rel}:{line_no}:import")
        for match in _STRING_IMPORT_VFS.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            hits.append(f"{rel}:{line_no}:string-import:{match.group(0)}")
    return tuple(sorted(hits))


def _generic_domain_literal_hits() -> tuple[str, ...]:
    hits: list[str] = []
    for path in GENERIC_ENGINE_MODULES:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(REPO_ROOT).as_posix()
        # Allow comments that say "no VFS/IPFS/..." as negative constraints.
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            # Docstrings at module top often restate the prohibition.
            if "must not" in line.lower() or "no vfs" in line.lower() or "without" in line.lower():
                if "vfs" in line.lower() or "ipfs" in line.lower():
                    continue
            if _FORBIDDEN_GENERIC.search(line):
                hits.append(f"{rel}:{line_no}:{line.strip()[:120]}")
    return tuple(hits)


def _ops_wrapper_logic_hits() -> tuple[str, ...]:
    if not OPS_CLI.is_file():
        return ("missing:ops_cli",)
    source = OPS_CLI.read_text(encoding="utf-8")
    tree = ast.parse(source)
    hits: list[str] = []
    if any(isinstance(node, ast.ClassDef) for node in tree.body):
        hits.append("ops_cli:class_definition")
    for match in _FORBIDDEN_WRAPPER_LOGIC.finditer(source):
        line_no = source.count("\n", 0, match.start()) + 1
        hits.append(f"ops_cli:{line_no}:{match.group(0)}")
    if "argparse" not in source or "main" not in source:
        hits.append("ops_cli:missing_bootstrap")
    if "dispatch" not in source and "load_assurance_config" not in source:
        hits.append("ops_cli:missing_delegation")
    return tuple(hits)


def _missing_generic_engines() -> tuple[str, ...]:
    missing = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in GENERIC_ENGINE_MODULES
        if not path.is_file()
    ]
    if not INTEGRATION.is_file():
        missing.append(INTEGRATION.relative_to(REPO_ROOT).as_posix())
    return tuple(missing)


def evaluate_vfs_root_layout_guard() -> VfsRootLayoutGuard:
    return VfsRootLayoutGuard(
        root_vfs_modules=_root_vfs_modules(),
        forbidden_import_hits=_forbidden_import_hits(),
        generic_domain_literal_hits=_generic_domain_literal_hits(),
        ops_wrapper_logic_hits=_ops_wrapper_logic_hits(),
        missing_generic_engines=_missing_generic_engines(),
        notes=(
            "Root vfs_*.py implementations and compatibility shims are forbidden.",
            "Generic engines accept profiles; domain vocabulary lives in integrations/config.",
            "Ops facade only parses args, bootstraps config, and delegates.",
        ),
    )


def test_agent_supervisor_root_has_no_vfs_star_py_modules() -> None:
    hits = _root_vfs_modules()
    assert hits == (), f"root vfs_*.py modules present: {hits}"


def test_no_production_imports_reference_agent_supervisor_vfs_star() -> None:
    hits = _forbidden_import_hits()
    assert hits == (), f"forbidden vfs_* imports remain:\n" + "\n".join(hits)


def test_generic_engines_reject_domain_and_board_branches() -> None:
    for path in GENERIC_ENGINE_MODULES:
        assert path.is_file(), f"missing generic engine: {path}"
    hits = _generic_domain_literal_hits()
    assert hits == (), f"generic domain literals:\n" + "\n".join(hits)


def test_ops_wrapper_contains_no_substantive_engine_logic() -> None:
    hits = _ops_wrapper_logic_hits()
    assert hits == (), f"ops wrapper embeds logic: {hits}"


def test_integration_adapter_is_lazy_and_not_at_package_root() -> None:
    assert INTEGRATION.is_file()
    assert not (AGENT_SUPERVISOR_ROOT / "ipfs_kit_vfs_assurance.py").exists()
    assert not (AGENT_SUPERVISOR_ROOT / "vfs_symbolic_rollout.py").exists()
    source = INTEGRATION.read_text(encoding="utf-8")
    assert "lazy" in source.lower()
    assert "CLOSED_ADAPTERS" in source or "closed" in source.lower()


def test_cold_import_of_agent_supervisor_package_is_side_effect_free() -> None:
    script = """
import json, sys, os
os.environ["IPFS_ACCEL_SKIP_CORE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
forbidden_roots = ("torch", "transformers", "openai", "anthropic", "neo4j", "duckdb")
before = {name for name in sys.modules if name.split(".")[0] in forbidden_roots}
import ipfs_accelerate_py.agent_supervisor as package
after = {name for name in sys.modules if name.split(".")[0] in forbidden_roots}
root = __import__("pathlib").Path(package.__file__).resolve().parent
vfs_root = sorted(p.name for p in root.glob("vfs_*.py"))
print(json.dumps({"added": sorted(after - before), "vfs_root": vfs_root}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={
            **dict(__import__("os").environ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "IPFS_ACCEL_SKIP_CORE": "1",
            "PYTHONPATH": str(REPO_ROOT)
            + (
                ":" + __import__("os").environ["PYTHONPATH"]
                if __import__("os").environ.get("PYTHONPATH")
                else ""
            ),
        },
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["added"] == []
    assert payload["vfs_root"] == []


def test_vfs_root_layout_guard_receipt_passes() -> None:
    first = evaluate_vfs_root_layout_guard()
    second = evaluate_vfs_root_layout_guard()
    assert isinstance(first, VfsRootLayoutGuard)
    assert first.passed, first.to_dict()
    assert first.schema.endswith("vfs-root-layout-guard@1")
    assert first.to_dict() == second.to_dict()
    assert first.root_vfs_modules == ()
    assert first.forbidden_import_hits == ()
    assert first.generic_domain_literal_hits == ()
    assert first.ops_wrapper_logic_hits == ()
    assert first.missing_generic_engines == ()
    assert len(GENERIC_ENGINE_MODULES) == 7
