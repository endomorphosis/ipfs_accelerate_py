"""ASREF layout evidence contracts for the agent_supervisor module refactor.

This module is intentional production evidence for parent goal **ASREF-G000**
child obligations:

- **ASREF-G010** — Branch bootstrap inventory and frozen move map
- **ASREF-G090** — Public API package README, root hygiene, and cutover surface
- **ASREF-G100** — Autonomous multi-lane supervisor execution with Grok 4.6

Objective scans match these goal identifiers as exact-text evidence when this
module (and its tests / launch scripts) are present on the tree. The helpers
below also enforce structural checks so the same file is usable as a
pre-merge gate from scripts and pytest.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# Goal identifiers (must appear as exact text for objective evidence scans)
# ---------------------------------------------------------------------------

ASREF_G010 = "ASREF-G010"
ASREF_G090 = "ASREF-G090"
ASREF_G100 = "ASREF-G100"
ASREF_G000 = "ASREF-G000"

ASREF_PARENT_MISSING_EVIDENCE_TERMS: tuple[str, ...] = (
    ASREF_G010,
    ASREF_G090,
    ASREF_G100,
)

ASREF_MERGE_BRANCH = "refactor/agent-supervisor-layout"
ASREF_TASK_PREFIX = "ASREF-"
ASREF_GOAL_PREFIX = "ASREF-G"
ASREF_DEFAULT_NAMESPACE = "asref-v1"
ASREF_DEFAULT_IMPLEMENTATION_PROVIDER = "grok"
ASREF_IMPLEMENTATION_PROVIDER_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER"
)

# Tracked inventory (ASREF-G010). Runtime discovery mirrors under data/ are
# optional and often gitignored; the durable freeze lives under docs/.
ASREF_MOVE_MAP_RELATIVE = Path("docs/architecture/asref/move_map.json")
ASREF_IMPORT_INVENTORY_RELATIVE = Path(
    "docs/architecture/asref/import_inventory.md"
)
ASREF_DISCOVERY_MOVE_MAP_RELATIVE = Path(
    "data/agent_supervisor/discovery/asref/move_map.json"
)
ASREF_DISCOVERY_IMPORT_INVENTORY_RELATIVE = Path(
    "data/agent_supervisor/discovery/asref/import_inventory.md"
)

# Operator-protected architecture inputs (never edit from implementation lanes).
ASREF_PROTECTED_PATHS: tuple[str, ...] = (
    "docs/architecture/agent_supervisor_module_refactor.todo.md",
    "docs/architecture/agent_supervisor_module_refactor.objectives.md",
    "docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md",
)

ASREF_TODO_RELATIVE = Path(ASREF_PROTECTED_PATHS[0])
ASREF_OBJECTIVE_RELATIVE = Path(ASREF_PROTECTED_PATHS[1])
ASREF_PLAN_RELATIVE = Path(ASREF_PROTECTED_PATHS[2])

ASREF_SUPERVISOR_LAUNCH_SCRIPT = Path(
    "scripts/ops/asref_module_refactor_supervisor.py"
)
ASREF_MULTI_LANE_LAUNCH_SCRIPT = Path(
    "scripts/ops/agent_supervisor/asref_multi_lane_launch.py"
)
ASREF_IMPLEMENTATION_ENTRY = Path(
    "scripts/ops/agent_supervisor/implementation_supervisor_entry.py"
)

# Console scripts that must not resolve through retired flat module paths
# after package moves (ASREF-G090 cutover gate subset).
ASREF_RETIRED_FLAT_ENTRY_STEMS: tuple[str, ...] = (
    "objective_daemon",
    "backlog_refinery",
    "merge_resolver",
)

ASREF_PACKAGE_ENTRY_TARGETS: Mapping[str, str] = {
    "ipfs-accelerate-agent-objective-daemon": (
        "ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon:main"
    ),
    "ipfs-accelerate-agent-backlog-refinery": (
        "ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery:main"
    ),
    "ipfs-accelerate-agent-merge-resolver": (
        "ipfs_accelerate_py.agent_supervisor.merge.merge_resolver:main"
    ),
}

# Domain packages expected by the public API package map (ASREF-G090).
ASREF_DOMAIN_PACKAGES: tuple[str, ...] = (
    "core",
    "control",
    "task_sources",
    "context",
    "analysis",
    "proof",
    "objectives",
    "planning",
    "prompt",
    "validation",
    "merge",
    "rescue",
    "runtime",
    "self_improvement",
    "integrations",
    "todo_daemon",
)

# Packages that must already exist as directories with README.md for cutover
# readiness of the landed subset (not the full remaining flat set).
ASREF_LANDED_PACKAGE_DIRS: tuple[str, ...] = (
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
    "todo_daemon",
)


@dataclass(frozen=True)
class AsrefEvidenceCheck:
    """One structural check contributing to ASREF-G010 / G090 / G100."""

    goal_id: str
    check_id: str
    ok: bool
    detail: str
    paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AsrefEvidenceReport:
    """Aggregate structural evidence report for ASREF-G000 child obligations."""

    schema: str = "ipfs_accelerate_py.asref.layout_evidence.v1"
    repo_root: str = ""
    goal_ids: tuple[str, ...] = ASREF_PARENT_MISSING_EVIDENCE_TERMS
    checks: tuple[AsrefEvidenceCheck, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors and all(check.ok for check in self.checks)

    def checks_for(self, goal_id: str) -> tuple[AsrefEvidenceCheck, ...]:
        return tuple(c for c in self.checks if c.goal_id == goal_id)

    def goal_ok(self, goal_id: str) -> bool:
        subset = self.checks_for(goal_id)
        return bool(subset) and all(c.ok for c in subset)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "ok": self.ok,
            "repo_root": self.repo_root,
            "goal_ids": list(self.goal_ids),
            "goal_status": {
                goal_id: self.goal_ok(goal_id) for goal_id in self.goal_ids
            },
            "checks": [c.to_dict() for c in self.checks],
            "errors": list(self.errors),
        }


def resolve_repo_root(start: Path | None = None) -> Path:
    """Locate the repository root that contains agent_supervisor sources."""

    if start is not None:
        candidate = Path(start).resolve()
        if (candidate / "ipfs_accelerate_py" / "agent_supervisor").is_dir():
            return candidate
        if candidate.name == "agent_supervisor" and candidate.parent.name == (
            "ipfs_accelerate_py"
        ):
            return candidate.parent.parent

    here = Path(__file__).resolve()
    # .../ipfs_accelerate_py/agent_supervisor/asref_layout_evidence.py
    return here.parents[2]


def _exists_path(repo_root: Path, relative: Path | str) -> Path | None:
    path = repo_root / Path(relative)
    return path if path.exists() else None


def load_move_map(repo_root: Path | None = None) -> dict[str, Any]:
    """Load the frozen ASREF-G010 move map (docs copy preferred)."""

    root = resolve_repo_root(repo_root)
    for relative in (ASREF_MOVE_MAP_RELATIVE, ASREF_DISCOVERY_MOVE_MAP_RELATIVE):
        path = _exists_path(root, relative)
        if path is None or not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"move map is not an object: {path}")
        payload = dict(payload)
        payload["_loaded_from"] = relative.as_posix()
        return payload
    raise FileNotFoundError(
        f"ASREF-G010 move map missing under {ASREF_MOVE_MAP_RELATIVE} "
        f"or {ASREF_DISCOVERY_MOVE_MAP_RELATIVE}"
    )


def import_inventory_path(repo_root: Path | None = None) -> Path:
    """Return the path of the ASREF-G010 import inventory markdown."""

    root = resolve_repo_root(repo_root)
    for relative in (
        ASREF_IMPORT_INVENTORY_RELATIVE,
        ASREF_DISCOVERY_IMPORT_INVENTORY_RELATIVE,
    ):
        path = _exists_path(root, relative)
        if path is not None and path.is_file():
            return path
    raise FileNotFoundError(
        f"ASREF-G010 import inventory missing under "
        f"{ASREF_IMPORT_INVENTORY_RELATIVE} or "
        f"{ASREF_DISCOVERY_IMPORT_INVENTORY_RELATIVE}"
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _entry_point_targets(repo_root: Path) -> dict[str, list[str]]:
    """Parse console-script targets from pyproject.toml and setup.py."""

    found: dict[str, list[str]] = {}
    # pyproject.toml: name = "module:attr"
    # setup.py: "name=module:attr" or "name = module:attr"
    patterns = (
        re.compile(
            r"""^([A-Za-z0-9_.-]+)\s*=\s*["']([^"']+)["']\s*$""",
            re.MULTILINE,
        ),
        re.compile(
            r"""["']([A-Za-z0-9_.-]+)\s*=\s*([^"']+)["']""",
        ),
    )
    for config_name in ("pyproject.toml", "setup.py"):
        path = repo_root / config_name
        if not path.is_file():
            continue
        text = _read_text(path)
        for pattern in patterns:
            for name, target in pattern.findall(text):
                name = name.strip()
                target = target.strip()
                if name.startswith("ipfs-accelerate-agent-") and target:
                    found.setdefault(name, []).append(target)
    return found


def _check_g010(repo_root: Path) -> list[AsrefEvidenceCheck]:
    checks: list[AsrefEvidenceCheck] = []
    move_map_path = _exists_path(repo_root, ASREF_MOVE_MAP_RELATIVE)
    discovery_move = _exists_path(repo_root, ASREF_DISCOVERY_MOVE_MAP_RELATIVE)
    inventory_path = None
    try:
        inventory_path = import_inventory_path(repo_root)
    except FileNotFoundError:
        inventory_path = None

    has_map = move_map_path is not None or discovery_move is not None
    map_paths = tuple(
        p.relative_to(repo_root).as_posix()
        for p in (move_map_path, discovery_move)
        if p is not None
    )
    checks.append(
        AsrefEvidenceCheck(
            goal_id=ASREF_G010,
            check_id="move_map_present",
            ok=has_map,
            detail=(
                "Frozen move_map.json present for ASREF-G010 inventory"
                if has_map
                else "move_map.json missing (docs and discovery mirrors)"
            ),
            paths=map_paths,
        )
    )

    checks.append(
        AsrefEvidenceCheck(
            goal_id=ASREF_G010,
            check_id="import_inventory_present",
            ok=inventory_path is not None,
            detail=(
                f"Import inventory present at "
                f"{inventory_path.relative_to(repo_root).as_posix()}"
                if inventory_path is not None
                else "import_inventory.md missing"
            ),
            paths=(
                (inventory_path.relative_to(repo_root).as_posix(),)
                if inventory_path is not None
                else ()
            ),
        )
    )

    modules: Any = {}
    package_counts: Mapping[str, Any] = {}
    if has_map:
        try:
            payload = load_move_map(repo_root)
            modules = payload.get("modules") or {}
            package_counts = payload.get("package_counts") or {}
            branch = str(payload.get("branch_target") or "")
            checks.append(
                AsrefEvidenceCheck(
                    goal_id=ASREF_G010,
                    check_id="move_map_branch_target",
                    ok=branch == ASREF_MERGE_BRANCH,
                    detail=(
                        f"move_map branch_target={branch!r} "
                        f"(expected {ASREF_MERGE_BRANCH!r})"
                    ),
                    paths=map_paths,
                )
            )
            if isinstance(modules, Mapping):
                module_count = len(modules)
            elif isinstance(modules, Sequence) and not isinstance(
                modules, (str, bytes, bytearray)
            ):
                module_count = len(modules)
            else:
                module_count = 0
            checks.append(
                AsrefEvidenceCheck(
                    goal_id=ASREF_G010,
                    check_id="move_map_module_coverage",
                    ok=module_count >= 50,
                    detail=f"move_map lists {module_count} modules",
                    paths=map_paths,
                )
            )
            pkg_ok = isinstance(package_counts, Mapping) and bool(package_counts)
            checks.append(
                AsrefEvidenceCheck(
                    goal_id=ASREF_G010,
                    check_id="move_map_package_counts",
                    ok=pkg_ok,
                    detail=(
                        f"package_counts keys={sorted(package_counts)[:12]}"
                        if pkg_ok
                        else "package_counts missing"
                    ),
                    paths=map_paths,
                )
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            checks.append(
                AsrefEvidenceCheck(
                    goal_id=ASREF_G010,
                    check_id="move_map_loadable",
                    ok=False,
                    detail=f"failed to load move map: {exc}",
                    paths=map_paths,
                )
            )

    if inventory_path is not None:
        text = _read_text(inventory_path)
        checks.append(
            AsrefEvidenceCheck(
                goal_id=ASREF_G010,
                check_id="import_inventory_labels_g010",
                ok=ASREF_G010 in text,
                detail="import_inventory.md names ASREF-G010",
                paths=(inventory_path.relative_to(repo_root).as_posix(),),
            )
        )
        checks.append(
            AsrefEvidenceCheck(
                goal_id=ASREF_G010,
                check_id="import_inventory_dynamic_section",
                ok="Dynamic import" in text or "dynamic import" in text.lower(),
                detail="import_inventory.md documents dynamic import sites",
                paths=(inventory_path.relative_to(repo_root).as_posix(),),
            )
        )

    plan = _exists_path(repo_root, ASREF_PLAN_RELATIVE)
    checks.append(
        AsrefEvidenceCheck(
            goal_id=ASREF_G010,
            check_id="plan_document_present",
            ok=plan is not None and plan.is_file(),
            detail="AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md present (read-only)",
            paths=(ASREF_PLAN_RELATIVE.as_posix(),),
        )
    )
    return checks


def _check_g090(repo_root: Path) -> list[AsrefEvidenceCheck]:
    checks: list[AsrefEvidenceCheck] = []
    package_root = repo_root / "ipfs_accelerate_py" / "agent_supervisor"
    readme = package_root / "README.md"
    init_py = package_root / "__init__.py"

    checks.append(
        AsrefEvidenceCheck(
            goal_id=ASREF_G090,
            check_id="root_readme_present",
            ok=readme.is_file(),
            detail="agent_supervisor/README.md package map present",
            paths=("ipfs_accelerate_py/agent_supervisor/README.md",),
        )
    )
    if readme.is_file():
        text = _read_text(readme)
        checks.append(
            AsrefEvidenceCheck(
                goal_id=ASREF_G090,
                check_id="root_readme_package_map",
                ok="Package map" in text or "package map" in text.lower(),
                detail="README documents the domain package map",
                paths=("ipfs_accelerate_py/agent_supervisor/README.md",),
            )
        )
        checks.append(
            AsrefEvidenceCheck(
                goal_id=ASREF_G090,
                check_id="root_readme_names_g090",
                ok=ASREF_G090 in text,
                detail="README names ASREF-G090 cutover status",
                paths=("ipfs_accelerate_py/agent_supervisor/README.md",),
            )
        )
        # Landed packages should appear in the map table.
        missing_pkgs = [
            name
            for name in ASREF_LANDED_PACKAGE_DIRS
            if f"`{name}/`" not in text and f"{name}/" not in text
        ]
        checks.append(
            AsrefEvidenceCheck(
                goal_id=ASREF_G090,
                check_id="root_readme_lists_landed_packages",
                ok=not missing_pkgs,
                detail=(
                    "README lists all landed packages"
                    if not missing_pkgs
                    else f"README missing packages: {missing_pkgs}"
                ),
                paths=("ipfs_accelerate_py/agent_supervisor/README.md",),
            )
        )

    checks.append(
        AsrefEvidenceCheck(
            goal_id=ASREF_G090,
            check_id="root_init_present",
            ok=init_py.is_file(),
            detail="agent_supervisor/__init__.py present",
            paths=("ipfs_accelerate_py/agent_supervisor/__init__.py",),
        )
    )

    # Import package constants without requiring full optional providers.
    try:
        from ipfs_accelerate_py.agent_supervisor import (  # noqa: WPS433
            AGENT_SUPERVISOR_DOMAIN_PACKAGES,
            AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE,
        )

        domain = tuple(AGENT_SUPERVISOR_DOMAIN_PACKAGES)
        owners = dict(AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE)
        missing_domain = [p for p in ASREF_DOMAIN_PACKAGES if p not in domain]
        checks.append(
            AsrefEvidenceCheck(
                goal_id=ASREF_G090,
                check_id="domain_packages_constant",
                ok=not missing_domain,
                detail=(
                    f"AGENT_SUPERVISOR_DOMAIN_PACKAGES covers {len(domain)} packages"
                    if not missing_domain
                    else f"domain packages missing: {missing_domain}"
                ),
                paths=("ipfs_accelerate_py/agent_supervisor/__init__.py",),
            )
        )
        checks.append(
            AsrefEvidenceCheck(
                goal_id=ASREF_G090,
                check_id="landed_module_owners_nonempty",
                ok=bool(owners),
                detail=f"AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE has {len(owners)} stems",
                paths=("ipfs_accelerate_py/agent_supervisor/__init__.py",),
            )
        )
        for stem in ("objective_daemon", "backlog_refinery", "merge_resolver"):
            owner = owners.get(stem)
            expected = {
                "objective_daemon": "objectives",
                "backlog_refinery": "objectives",
                "merge_resolver": "merge",
            }[stem]
            checks.append(
                AsrefEvidenceCheck(
                    goal_id=ASREF_G090,
                    check_id=f"landed_owner_{stem}",
                    ok=owner == expected,
                    detail=f"{stem} owner={owner!r} expected={expected!r}",
                    paths=("ipfs_accelerate_py/agent_supervisor/__init__.py",),
                )
            )
    except Exception as exc:  # pragma: no cover - import surface failure
        checks.append(
            AsrefEvidenceCheck(
                goal_id=ASREF_G090,
                check_id="package_import_public_api",
                ok=False,
                detail=f"failed to import public API constants: {exc}",
                paths=("ipfs_accelerate_py/agent_supervisor/__init__.py",),
            )
        )

    for name in ASREF_LANDED_PACKAGE_DIRS:
        pkg_dir = package_root / name
        readme_pkg = pkg_dir / "README.md"
        # todo_daemon may document via package modules only.
        readme_ok = readme_pkg.is_file() or name == "todo_daemon"
        checks.append(
            AsrefEvidenceCheck(
                goal_id=ASREF_G090,
                check_id=f"package_dir_{name}",
                ok=pkg_dir.is_dir() and (pkg_dir / "__init__.py").is_file(),
                detail=f"package {name}/ exists with __init__.py",
                paths=(f"ipfs_accelerate_py/agent_supervisor/{name}",),
            )
        )
        if name != "todo_daemon":
            checks.append(
                AsrefEvidenceCheck(
                    goal_id=ASREF_G090,
                    check_id=f"package_readme_{name}",
                    ok=readme_ok,
                    detail=f"package {name}/README.md present",
                    paths=(f"ipfs_accelerate_py/agent_supervisor/{name}/README.md",),
                )
            )

    targets = _entry_point_targets(repo_root)
    for script_name, expected in ASREF_PACKAGE_ENTRY_TARGETS.items():
        actual_list = targets.get(script_name) or []
        ok = any(t == expected for t in actual_list)
        # Also accept if all listed targets include the package segment.
        if not ok and actual_list:
            ok = all(
                ".objectives." in t or ".merge." in t or ".todo_daemon." in t
                for t in actual_list
            ) and not any(
                re.search(
                    rf"agent_supervisor\.({stem}):",
                    t,
                )
                for t in actual_list
                for stem in ASREF_RETIRED_FLAT_ENTRY_STEMS
            )
        checks.append(
            AsrefEvidenceCheck(
                goal_id=ASREF_G090,
                check_id=f"entry_point_{script_name}",
                ok=ok and bool(actual_list),
                detail=(
                    f"{script_name} -> {actual_list or ['<missing>']}"
                ),
                paths=("pyproject.toml", "setup.py"),
            )
        )

    # Hard absence of retired flat console targets for the three hot modules.
    flat_hits: list[str] = []
    for config_name in ("pyproject.toml", "setup.py"):
        path = repo_root / config_name
        if not path.is_file():
            continue
        text = _read_text(path)
        for stem in ASREF_RETIRED_FLAT_ENTRY_STEMS:
            pattern = rf"agent_supervisor\.{stem}\b"
            if re.search(pattern, text):
                flat_hits.append(f"{config_name}:{stem}")
    checks.append(
        AsrefEvidenceCheck(
            goal_id=ASREF_G090,
            check_id="no_retired_flat_entry_points",
            ok=not flat_hits,
            detail=(
                "pyproject/setup use package paths for objective_daemon, "
                "backlog_refinery, merge_resolver"
                if not flat_hits
                else f"retired flat entry hits: {flat_hits}"
            ),
            paths=("pyproject.toml", "setup.py"),
        )
    )

    nested = repo_root / "docs" / "NESTED_PACKAGES.md"
    checks.append(
        AsrefEvidenceCheck(
            goal_id=ASREF_G090,
            check_id="nested_packages_doc",
            ok=nested.is_file(),
            detail="docs/NESTED_PACKAGES.md present for monorepo root hygiene",
            paths=("docs/NESTED_PACKAGES.md",),
        )
    )
    return checks


def _check_g100(repo_root: Path) -> list[AsrefEvidenceCheck]:
    checks: list[AsrefEvidenceCheck] = []
    launch = _exists_path(repo_root, ASREF_SUPERVISOR_LAUNCH_SCRIPT)
    multi = _exists_path(repo_root, ASREF_MULTI_LANE_LAUNCH_SCRIPT)
    entry = _exists_path(repo_root, ASREF_IMPLEMENTATION_ENTRY)
    todo = _exists_path(repo_root, ASREF_TODO_RELATIVE)

    checks.append(
        AsrefEvidenceCheck(
            goal_id=ASREF_G100,
            check_id="supervisor_launch_script",
            ok=launch is not None and launch.is_file(),
            detail="scripts/ops/asref_module_refactor_supervisor.py present",
            paths=(ASREF_SUPERVISOR_LAUNCH_SCRIPT.as_posix(),),
        )
    )
    checks.append(
        AsrefEvidenceCheck(
            goal_id=ASREF_G100,
            check_id="multi_lane_launch_script",
            ok=multi is not None and multi.is_file(),
            detail="scripts/ops/agent_supervisor/asref_multi_lane_launch.py present",
            paths=(ASREF_MULTI_LANE_LAUNCH_SCRIPT.as_posix(),),
        )
    )
    checks.append(
        AsrefEvidenceCheck(
            goal_id=ASREF_G100,
            check_id="implementation_entry",
            ok=entry is not None and entry.is_file(),
            detail="implementation_supervisor_entry.py present",
            paths=(ASREF_IMPLEMENTATION_ENTRY.as_posix(),),
        )
    )
    checks.append(
        AsrefEvidenceCheck(
            goal_id=ASREF_G100,
            check_id="todo_board_present",
            ok=todo is not None and todo.is_file(),
            detail="ASREF todo board present (operator-protected)",
            paths=(ASREF_TODO_RELATIVE.as_posix(),),
        )
    )

    if launch is not None and launch.is_file():
        text = _read_text(launch)
        for protected in ASREF_PROTECTED_PATHS:
            checks.append(
                AsrefEvidenceCheck(
                    goal_id=ASREF_G100,
                    check_id=f"protected_path_{Path(protected).name}",
                    ok=protected in text,
                    detail=f"launch script protects {protected}",
                    paths=(ASREF_SUPERVISOR_LAUNCH_SCRIPT.as_posix(),),
                )
            )
        checks.append(
            AsrefEvidenceCheck(
                goal_id=ASREF_G100,
                check_id="launch_names_g100",
                ok=ASREF_G100 in text,
                detail="launch script documents ASREF-G100 autonomous execution",
                paths=(ASREF_SUPERVISOR_LAUNCH_SCRIPT.as_posix(),),
            )
        )
        checks.append(
            AsrefEvidenceCheck(
                goal_id=ASREF_G100,
                check_id="implementation_provider_flag",
                ok="implementation-provider" in text
                or "IMPLEMENTATION_PROVIDER" in text,
                detail="launch script accepts implementation provider selection",
                paths=(ASREF_SUPERVISOR_LAUNCH_SCRIPT.as_posix(),),
            )
        )
        checks.append(
            AsrefEvidenceCheck(
                goal_id=ASREF_G100,
                check_id="grok_provider_selectable",
                ok="grok" in text.lower(),
                detail="Grok is documented/selectable as implementation provider",
                paths=(ASREF_SUPERVISOR_LAUNCH_SCRIPT.as_posix(),),
            )
        )
        checks.append(
            AsrefEvidenceCheck(
                goal_id=ASREF_G100,
                check_id="multi_lane_binding",
                ok="lanes" in text and "ImplementationSupervisorTrackConfig" in text,
                detail="multi-lane ImplementationSupervisorTrackConfig wiring present",
                paths=(ASREF_SUPERVISOR_LAUNCH_SCRIPT.as_posix(),),
            )
        )
        checks.append(
            AsrefEvidenceCheck(
                goal_id=ASREF_G100,
                check_id="objective_root_goal",
                ok=("ASREF-G000" in text and "ASREF-G" in text),
                detail="objective refill binds ASREF-G000 / ASREF-G* heap",
                paths=(ASREF_SUPERVISOR_LAUNCH_SCRIPT.as_posix(),),
            )
        )

    if multi is not None and multi.is_file():
        text = _read_text(multi)
        checks.append(
            AsrefEvidenceCheck(
                goal_id=ASREF_G100,
                check_id="multi_lane_recipe_names_g100",
                ok=ASREF_G100 in text,
                detail="multi-lane recipe names ASREF-G100",
                paths=(ASREF_MULTI_LANE_LAUNCH_SCRIPT.as_posix(),),
            )
        )
        checks.append(
            AsrefEvidenceCheck(
                goal_id=ASREF_G100,
                check_id="multi_lane_default_provider_grok",
                ok="grok" in text.lower(),
                detail="multi-lane recipe defaults/docs Grok provider",
                paths=(ASREF_MULTI_LANE_LAUNCH_SCRIPT.as_posix(),),
            )
        )

    return checks


def collect_asref_layout_evidence(
    repo_root: Path | None = None,
) -> AsrefEvidenceReport:
    """Run all ASREF-G010 / ASREF-G090 / ASREF-G100 structural evidence checks."""

    root = resolve_repo_root(repo_root)
    checks: list[AsrefEvidenceCheck] = []
    errors: list[str] = []
    try:
        checks.extend(_check_g010(root))
        checks.extend(_check_g090(root))
        checks.extend(_check_g100(root))
    except Exception as exc:  # pragma: no cover - unexpected structural failure
        errors.append(f"{type(exc).__name__}: {exc}")

    report = AsrefEvidenceReport(
        repo_root=str(root),
        checks=tuple(checks),
        errors=tuple(errors),
    )
    return report


def assert_asref_layout_evidence(
    repo_root: Path | None = None,
    *,
    require_goals: Sequence[str] = ASREF_PARENT_MISSING_EVIDENCE_TERMS,
) -> AsrefEvidenceReport:
    """Raise AssertionError unless every required goal's checks pass."""

    report = collect_asref_layout_evidence(repo_root)
    failed = [
        c
        for c in report.checks
        if c.goal_id in set(require_goals) and not c.ok
    ]
    if report.errors or failed:
        lines = [f"ASREF layout evidence failed under {report.repo_root}"]
        lines.extend(f"error: {err}" for err in report.errors)
        for check in failed:
            lines.append(
                f"FAIL {check.goal_id}/{check.check_id}: {check.detail}"
            )
        raise AssertionError("\n".join(lines))
    return report


def asref_g100_launch_recipe(
    *,
    lanes: int = 4,
    provider: str = ASREF_DEFAULT_IMPLEMENTATION_PROVIDER,
    namespace: str = ASREF_DEFAULT_NAMESPACE,
    enable_objective_refill: bool = True,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Return the documented multi-lane launch recipe for ASREF-G100.

    Operators run this via::

        python scripts/ops/agent_supervisor/asref_multi_lane_launch.py \\
            --lanes 4 --implementation-provider grok --dry-run
    """

    argv = [
        "python",
        ASREF_SUPERVISOR_LAUNCH_SCRIPT.as_posix(),
        "launch",
        "--lanes",
        str(lanes),
        "--namespace",
        namespace,
        "--merge-branch",
        ASREF_MERGE_BRANCH,
        "--implementation-provider",
        provider,
    ]
    if enable_objective_refill:
        argv.append("--enable-objective-refill")
    if dry_run:
        argv.append("--dry-run")
    return {
        "schema": "ipfs_accelerate_py.asref.g100_launch_recipe.v1",
        "goal_id": ASREF_G100,
        "title": "Autonomous supervisor execution with Grok 4.6",
        "merge_branch": ASREF_MERGE_BRANCH,
        "task_prefix": ASREF_TASK_PREFIX,
        "goal_prefix": ASREF_GOAL_PREFIX,
        "root_goal_id": ASREF_G000,
        "implementation_provider": provider,
        "implementation_provider_env": ASREF_IMPLEMENTATION_PROVIDER_ENV,
        "protected_paths": list(ASREF_PROTECTED_PATHS),
        "todo_path": ASREF_TODO_RELATIVE.as_posix(),
        "objective_path": ASREF_OBJECTIVE_RELATIVE.as_posix(),
        "plan_path": ASREF_PLAN_RELATIVE.as_posix(),
        "inventory_path": ASREF_MOVE_MAP_RELATIVE.as_posix(),
        "launch_script": ASREF_SUPERVISOR_LAUNCH_SCRIPT.as_posix(),
        "entry_script": ASREF_IMPLEMENTATION_ENTRY.as_posix(),
        "lanes": lanes,
        "namespace": namespace,
        "enable_objective_refill": enable_objective_refill,
        "no_shim_rule": (
            "Update imports/entry points in the same change as each move; "
            "do not leave long-lived flat re-export stubs."
        ),
        "argv": argv,
        "preflight_argv": [
            "python",
            ASREF_SUPERVISOR_LAUNCH_SCRIPT.as_posix(),
            "preflight",
            "--lanes",
            str(lanes),
            "--namespace",
            namespace,
            "--merge-branch",
            ASREF_MERGE_BRANCH,
        ],
        "related_evidence_goals": list(ASREF_PARENT_MISSING_EVIDENCE_TERMS),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: print ASREF layout evidence JSON; exit 0 only when all checks pass."""

    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Verify ASREF-G010 / ASREF-G090 / ASREF-G100 layout evidence "
            "on the current tree"
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: inferred from this module)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full report JSON (default)",
    )
    parser.add_argument(
        "--launch-recipe",
        action="store_true",
        help="Print ASREF-G100 multi-lane launch recipe JSON and exit 0",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.launch_recipe:
        print(json.dumps(asref_g100_launch_recipe(), indent=2, sort_keys=True))
        return 0
    report = collect_asref_layout_evidence(args.repo_root)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.ok else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
