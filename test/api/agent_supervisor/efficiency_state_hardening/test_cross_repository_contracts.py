"""ASEH-062: current-head isolated package qualification across repositories.

Fresh independent installations of the three current-head package trees must
pass canonical-byte/CID, unknown-field, bound, stale-identity, missing-
dependency, and authority-boundary vectors. Sibling source-tree tests are
never imported. Qualification does not promote packages or policy pointers.
"""

from __future__ import annotations

import ast
import copy
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    content_identity,
)

# Isolated installs must not run Datasets/Kit auto-installers or sibling
# source-tree side effects. Set these before any optional package import.
os.environ["IPFS_DATASETS_PY_MINIMAL_IMPORTS"] = "1"
os.environ["IPFS_DATASETS_AUTO_INSTALL"] = "0"
os.environ["IPFS_AUTO_INSTALL"] = "0"
os.environ["IPFS_KIT_AUTO_INSTALL_DEPS"] = "0"
os.environ["IPFS_DATASETS_ENSURE_INSTALLER"] = "0"
os.environ["IPFS_ACCEL_SKIP_CORE"] = "1"
os.environ["IPFS_DATASETS_PY_BENCHMARK"] = "1"


ROOT = Path(__file__).resolve().parents[4]
INVENTORY = ROOT / "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory"
QUALIFICATION_PATH = INVENTORY / "cross_repository_qualification.json"
REQUIREMENTS_PATH = (
    ROOT / "docs/architecture/agent_supervisor_efficiency_state_hardening.requirements.json"
)
SEALED_BASELINE_PATH = INVENTORY / "sealed_baseline.json"

SCHEMA = "ipfs_accelerate_py/agent-supervisor/aseh-cross-repository-qualification@1"
PROGRAM_ID = "agent-supervisor-efficiency-and-state-hardening-v1"
TASK_ID = "ASEH-062"
OBJECTIVE_ID = "ASEH-G070"
PLAN_REVISION = "ASEH-PLAN-R1"
CANONICAL_PYTHON = "/usr/bin/python3.12"
CANONICAL_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
JSON_MAX_INT = 9_007_199_254_740_991
TREE_OID = "16ef68abe8a35a3033dfaf1ed4e8d6132600df8f"
STALE_TREE = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
OBJECTIVE = "ASEH-062"
OBJECTIVE_REVISION = "baguqeerauympe4ggo3n2ltsf726vcodpm45ldegoo2224ty3vieh4pwd3e3q"
TOOLCHAIN = "python3.12"
ENVIRONMENT = "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
SCHEMA_IDENTITY = "ipfs-datasets.proof-context.context-pack@0.1"

PACKAGE_SOURCES = {
    "ipfs_accelerate_py": ROOT / "ipfs_accelerate_py",
    "ipfs_datasets_py": ROOT / "ipfs_datasets_py" / "ipfs_datasets_py",
    "ipfs_kit_py": ROOT / "ipfs_kit_py" / "ipfs_kit_py",
}
REPO_ROOTS = {
    "ipfs_accelerate_py": ROOT,
    "ipfs_datasets_py": ROOT / "ipfs_datasets_py",
    "ipfs_kit_py": ROOT / "ipfs_kit_py",
}
REQUIRED_VECTOR_IDS = (
    "authority_boundaries",
    "canonical_bytes_and_cid_equality",
    "independent_current_head_installation",
    "missing_dependencies",
    "no_sibling_source_tree_test_imports",
    "numeric_bounds",
    "stale_identities",
    "unknown_fields",
)
FORBIDDEN_TEST_IMPORTS = (
    "ipfs_datasets_py.tests",
    "ipfs_kit_py.tests",
    "test.api.agent_supervisor.efficiency_state_hardening.test_",
    "test.api.test_agent_supervisor_",
    "tests.proof_context",
    "tests.test_context_pack_store",
)
REQUIRED_REPOSITORIES = ("ipfs_accelerate_py", "ipfs_datasets_py", "ipfs_kit_py")
RUNTIME_DEPENDENCIES = (
    "anyio",
    "idna",
    "multiformats",
    "multiformats_config",
    "sniffio",
    "typing_extensions",
    "varint",
)
CONTRACT_IMPORTS = (
    "ipfs_datasets_py.logic.software_contracts.content",
    "ipfs_datasets_py.proof_context.context_pack",
    "ipfs_datasets_py.proof_context.incremental_context",
    "ipfs_kit_py.proof_context.artifacts",
    "ipfs_kit_py.proof_context.state_store",
    "ipfs_accelerate_py.agent_supervisor.semantic_state.context_pack",
    "ipfs_accelerate_py.agent_supervisor.semantic_state.context_pack_selector",
)
STDLIB_MODULE_NAMES = set(getattr(sys, "stdlib_module_names", ())) | set(
    sys.builtin_module_names
)
SKIP_THIRD_PARTY_TOP_LEVEL = STDLIB_MODULE_NAMES | {
    "__main__",
    "conftest",
    "pytest",
    "_pytest",
    "pluggy",
    "py",
    "iniconfig",
    "packaging",
    "exceptiongroup",
    "tomli",
}
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
CID_RE = re.compile(r"^b[a-z2-7]{20,}$")
PSEUDO_CID_RE = re.compile(
    r"(?i)\b(?:Qm[1-9A-HJ-NP-Za-km-z]{44}|cid:[A-Za-z0-9_-]+|mock[_-]?cid|fake[_-]?cid|"
    r"pseudo[_-]?cid|test[_-]?cid|uuid:[0-9a-f-]{36})\b"
)
MOCK_CAPABILITY_RE = re.compile(
    r"(?i)(?:mock|fake|simulated|stub|pseudo)[_-]?(?:capability|provider|cid|prover)"
)
_IGNORE_NAMES = {
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "build",
    "dist",
    "node_modules",
    "test",
    "tests",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    canonical = _canonical(payload)
    assert raw.rstrip("\n") + "\n" == canonical
    return payload


def _git(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd or ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return completed.stdout.strip()


def _git_ok(*args: str, cwd: Path | None = None) -> bool:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd or ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _peel_head(cwd: Path) -> dict[str, str]:
    if not cwd.is_dir():
        raise AssertionError(f"unbound head: missing repository {cwd}")
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if completed.returncode != 0 or not SHA1_RE.fullmatch(commit):
        raise AssertionError(
            f"unbound head commit in {cwd}: {completed.stderr or completed.stdout}"
        )
    kind = _git("cat-file", "-t", commit, cwd=cwd)
    if kind != "commit":
        raise AssertionError(f"unbound head {commit} peels to {kind} in {cwd}")
    tree = _git("rev-parse", f"{commit}^{{tree}}", cwd=cwd)
    if not SHA1_RE.fullmatch(tree):
        raise AssertionError(f"unbound head tree in {cwd}")
    return {"commit": commit, "tree": tree}


def _porcelain(cwd: Path) -> list[str]:
    raw = _git("status", "--porcelain", cwd=cwd)
    if not raw:
        return []
    return [line[3:].split(" -> ", 1)[-1] for line in raw.splitlines() if line.strip()]


def _imported_module_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _assert_no_sibling_test_imports(names: Sequence[str]) -> None:
    sibling_stems = {
        path.stem
        for path in Path(__file__).parent.glob("test_*.py")
        if path.name != Path(__file__).name
    }
    for name in names:
        last = name.rsplit(".", 1)[-1]
        if last in sibling_stems and (
            name.startswith("test.") or name.startswith("tests.")
        ):
            raise AssertionError(name)
        for prefix in FORBIDDEN_TEST_IMPORTS:
            assert not name.startswith(prefix), name


def _assert_real_cid(value: Any, *, field: str) -> None:
    assert isinstance(value, str) and CID_RE.fullmatch(value), field
    assert not value.startswith("Qm"), field
    assert PSEUDO_CID_RE.search(value) is None, field
    assert MOCK_CAPABILITY_RE.search(value) is None, field


def _assert_sha1(value: Any, *, field: str) -> None:
    assert isinstance(value, str) and SHA1_RE.fullmatch(value), field


def _live_heads() -> dict[str, dict[str, Any]]:
    payload = _load(QUALIFICATION_PATH)
    policy = payload["current_head_policy"]
    allowed = set(policy["allowed_accelerate_dirty_paths"])
    ancestors = payload["planning_ancestors"]
    heads: dict[str, dict[str, Any]] = {}
    for name, repo in REPO_ROOTS.items():
        assert repo.is_dir(), name
        head = _peel_head(repo)
        planned = ancestors[name]
        _assert_sha1(planned["commit"], field=f"{name}.planning.commit")
        _assert_sha1(planned["tree"], field=f"{name}.planning.tree")
        assert _git_ok(
            "merge-base", "--is-ancestor", planned["commit"], head["commit"], cwd=repo
        ), f"{name} planning commit is not an ancestor of current HEAD"
        dirty = _porcelain(repo)
        if name == "ipfs_accelerate_py":
            dirty = [
                path
                for path in dirty
                if path not in allowed
                and path not in {"ipfs_datasets_py", "ipfs_kit_py"}
                and not path.startswith("ipfs_datasets_py/")
                and not path.startswith("ipfs_kit_py/")
            ]
        heads[name] = {
            **head,
            "dirty": dirty,
            "repository": str(repo),
            "planning_commit": planned["commit"],
            "planning_tree": planned["tree"],
        }
        if policy["unbound_head_invalidates"] and not (
            SHA1_RE.fullmatch(head["commit"]) and SHA1_RE.fullmatch(head["tree"])
        ):
            raise AssertionError(f"unbound head invalidates {name}")
    return heads


def _ignore_names(directory: str, names: list[str]) -> set[str]:
    dropped: set[str] = set()
    base = Path(directory)
    for name in names:
        if (
            name in _IGNORE_NAMES
            or name.endswith(".pyc")
            or name.endswith(".egg-info")
            or name.endswith(".dist-info")
        ):
            dropped.add(name)
            continue
        try:
            if (base / name).is_symlink():
                dropped.add(name)
        except OSError:
            dropped.add(name)
    return dropped


def _workspace_path(path: Path) -> bool:
    root = ROOT.resolve()
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved == root or root in resolved.parents


def _copy_tree(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(
            source,
            dest,
            dirs_exist_ok=True,
            ignore=_ignore_names,
            symlinks=False,
            ignore_dangling_symlinks=True,
            copy_function=shutil.copy2,
        )
        return
    except (OSError, shutil.Error):
        pass
    for item in source.iterdir():
        if item.name in _ignore_names(str(source), [item.name]):
            continue
        target = dest / item.name
        if item.is_symlink():
            try:
                resolved = item.resolve()
            except OSError:
                continue
            if _workspace_path(resolved) and not (
                source.resolve() == resolved or source.resolve() in resolved.parents
            ):
                continue
            if resolved.is_dir():
                _copy_tree(resolved, target)
            elif resolved.is_file():
                shutil.copy2(resolved, target)
            continue
        if item.is_dir():
            _copy_tree(item, target)
            continue
        if item.is_file():
            shutil.copy2(item, target)


def _install_packages(site: Path, names: Sequence[str]) -> None:
    site.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = PACKAGE_SOURCES[name]
        assert source.is_dir(), name
        dest = site / name
        if dest.exists():
            continue
        _copy_tree(source, dest)
        assert (dest / "__init__.py").is_file(), name


def _copy_module_into_site(module: Any, site: Path) -> None:
    root = ROOT.resolve()
    paths = getattr(module, "__path__", None)
    if paths:
        for entry in list(paths):
            source = Path(entry)
            if not source.is_dir():
                continue
            try:
                resolved = source.resolve()
            except OSError:
                continue
            if resolved == root or root in resolved.parents:
                continue
            dest = site / resolved.name
            if dest.exists():
                continue
            _copy_tree(resolved, dest)
        return
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str) or not origin:
        return
    source = Path(origin)
    try:
        resolved = source.resolve()
    except OSError:
        return
    if resolved == root or root in resolved.parents:
        return
    if resolved.name.startswith("__init__"):
        dest = site / resolved.parent.name
        if dest.exists():
            return
        _copy_tree(resolved.parent, dest)
        return
    dest = site / resolved.name
    if dest.exists() or not resolved.is_file():
        return
    shutil.copy2(resolved, dest)


def _install_runtime_dependencies(site: Path) -> None:
    site.mkdir(parents=True, exist_ok=True)
    root = ROOT.resolve()
    for name in RUNTIME_DEPENDENCIES:
        module = importlib.import_module(name)
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str) or not origin:
            raise AssertionError(f"missing runtime dependency {name}")
        source = Path(origin).resolve()
        if root == source or root in source.parents:
            raise AssertionError(f"runtime dependency {name} resolved inside workspace")
        _copy_module_into_site(module, site)
        dest_package = site / Path(origin).resolve().parent.name
        dest_file = site / Path(origin).name
        if not dest_package.exists() and not dest_file.exists():
            raise AssertionError(f"failed to copy runtime dependency {name}")


def _warmup_contract_imports() -> None:
    for name in CONTRACT_IMPORTS:
        importlib.import_module(name)


def _install_discovered_third_party(site: Path) -> None:
    site.mkdir(parents=True, exist_ok=True)
    owned = set(REQUIRED_REPOSITORIES)
    copied: set[str] = set()
    for name, module in list(sys.modules.items()):
        top = name.split(".", 1)[0]
        if (
            not top
            or top in copied
            or top in owned
            or top in SKIP_THIRD_PARTY_TOP_LEVEL
        ):
            continue
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str) or not origin:
            continue
        if _workspace_path(Path(origin)):
            continue
        _copy_module_into_site(module, site)
        copied.add(top)


def _is_python_prefix_path(path: Path) -> bool:
    prefixes = {
        Path(sys.base_prefix).resolve(),
        Path(sys.exec_prefix).resolve(),
        Path(sys.prefix).resolve(),
    }
    try:
        resolved = path.resolve()
    except OSError:
        return False
    if _workspace_path(resolved):
        return False
    for prefix in prefixes:
        if resolved == prefix or prefix in resolved.parents:
            return True
    return False


def _isolated_pythonpath(site: Path) -> list[str]:
    kept = [str(site.resolve())]
    seen = {kept[0]}
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for attr in ("base_prefix", "exec_prefix", "prefix"):
        prefix = Path(getattr(sys, attr)).resolve()
        for candidate in (
            prefix / "lib" / version,
            prefix / "lib" / version / "lib-dynload",
            prefix / "lib64" / version,
            prefix / "lib" / version / "site-packages",
            prefix / "lib64" / version / "site-packages",
        ):
            if candidate.is_dir():
                text = str(candidate)
                if text not in seen and not _workspace_path(candidate):
                    kept.append(text)
                    seen.add(text)
    for item in sys.path:
        if not item:
            continue
        try:
            candidate = Path(item)
            resolved = candidate.resolve()
        except OSError:
            continue
        if _workspace_path(resolved):
            continue
        if not (
            _is_python_prefix_path(candidate)
            or resolved.exists()
            or str(item).endswith(".zip")
        ):
            continue
        text = item if Path(item).is_absolute() else str(resolved)
        if text in seen:
            continue
        kept.append(text)
        seen.add(text)
    return kept


def _isolated_env(home: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PYTHON")
    }
    env["HOME"] = str(home)
    env["XDG_CACHE_HOME"] = str(home / ".cache")
    env["XDG_CONFIG_HOME"] = str(home / ".config")
    env["XDG_DATA_HOME"] = str(home / ".local" / "share")
    env["XDG_STATE_HOME"] = str(home / ".local" / "state")
    env["PATH"] = CANONICAL_PATH
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["IPFS_DATASETS_PY_MINIMAL_IMPORTS"] = "1"
    env["IPFS_DATASETS_AUTO_INSTALL"] = "0"
    env["IPFS_AUTO_INSTALL"] = "0"
    env["IPFS_KIT_AUTO_INSTALL_DEPS"] = "0"
    env["IPFS_DATASETS_ENSURE_INSTALLER"] = "0"
    env["IPFS_ACCEL_SKIP_CORE"] = "1"
    env["IPFS_DATASETS_PY_BENCHMARK"] = "1"
    env["IPFS_DATASETS_PROJECT_ROOT"] = str(home)
    env["IPFS_DATASETS_LOCAL_BIN"] = str(home / "bin")
    return env


def _run_isolated(
    *,
    site: Path,
    workdir: Path,
    home: Path,
    suite: str,
    store_root: Path,
) -> dict[str, Any]:
    result_path = workdir / f"{suite}-result.json"
    config = {
        "site_packages": str(site),
        "pythonpath": _isolated_pythonpath(site),
        "workdir": str(workdir),
        "store_root": str(store_root),
        "result_path": str(result_path),
        "suite": suite,
        "forbidden_roots": [str(ROOT.resolve())],
        "strip_forbidden": True,
        "tree_oid": TREE_OID,
        "stale_tree": STALE_TREE,
        "objective": OBJECTIVE,
        "objective_revision": OBJECTIVE_REVISION,
        "toolchain": TOOLCHAIN,
        "environment": ENVIRONMENT,
        "schema_identity": SCHEMA_IDENTITY,
        "json_max_int": JSON_MAX_INT,
    }
    completed = subprocess.run(
        [sys.executable, "-s", "-E", "-B", "-c", _ISOLATED_RUNNER, json.dumps(config)],
        cwd=str(workdir),
        env=_isolated_env(home),
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if completed.returncode != 0 or not result_path.is_file():
        raise AssertionError(
            f"isolated {suite} failed rc={completed.returncode}:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True, payload
    return payload


_ISOLATED_RUNNER = r"""
import json
import os
import sys
import traceback
from pathlib import Path

config = json.loads(sys.argv[1])
sys.path[:] = list(config["pythonpath"])
forbidden = [Path(item).resolve() for item in config.get("forbidden_roots") or []]
site = Path(config["site_packages"]).resolve()
owned = {"ipfs_accelerate_py", "ipfs_datasets_py", "ipfs_kit_py"}
installed = {item.name for item in site.iterdir() if item.is_dir()}

def _forbidden(path):
    if not path:
        return True
    resolved = Path(path).resolve()
    return any(resolved == root or root in resolved.parents for root in forbidden)

if config.get("strip_forbidden", True):
    sys.path[:] = [item for item in sys.path if not _forbidden(item)]
os.chdir(config["workdir"])

class BlockUninstalled:
    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".", 1)[0]
        if top in owned and top not in installed:
            raise ImportError(f"{top} is not independently installed")
        return None

sys.meta_path.insert(0, BlockUninstalled())

from importlib.resources import files as resource_files

def cid_label(label):
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
    return cid_for_bytes(label.encode("utf-8"))

def helper_dep():
    return {
        "symbol": "helper",
        "cid": cid_label("helper"),
        "path": "helper.py",
        "meaning": "pure helper used by target",
    }

def freshness(**overrides):
    payload = {
        "file_and_symbol_identities": [],
        "schema_identities": [config["schema_identity"]],
        "toolchain_identities": [config["toolchain"]],
        "environment_requirements": [config["environment"]],
        "reusable_until_conditions": ["tree-unchanged"],
    }
    payload.update(overrides)
    return payload

def pack_kwargs(**overrides):
    fields = {
        "repository_state_cid": cid_label("repo-state"),
        "task_id": "ASEH-062",
        "target_source_cid": cid_label("target"),
        "surrounding_source_cid": cid_label("surround"),
        "test_source_cid": cid_label("test"),
        "scanned_tree_oid": config["tree_oid"],
        "source_tree_oid": config["tree_oid"],
        "dependencies": [helper_dep()],
        "objective_identity": config["objective"],
        "objective_revision": config["objective_revision"],
        "policy_identity": cid_label("aseh-062-policy"),
        "freshness_bindings": freshness(),
        "invalidation": {
            "invalidation_triggers": ["tree-changed", "policy-changed"],
            "reusable_until_conditions": ["tree-unchanged"],
        },
    }
    fields.update(overrides)
    return fields

def imported_names():
    names = []
    markers = (
        "/ipfs_datasets_py/tests/",
        "/ipfs_kit_py/tests/",
        "/tests/proof_context/",
        "/tests/test_context_pack_store",
        "/test/api/agent_supervisor/efficiency_state_hardening/test_",
    )
    for name, module in list(sys.modules.items()):
        names.append(name)
        path = getattr(module, "__file__", None)
        resolved = str(Path(path).resolve()) if path else ""
        if any(marker in resolved for marker in markers):
            raise AssertionError(f"source-tree tests import: {name} -> {path}")
        top = name.split(".", 1)[0]
        if top in owned and path:
            located = Path(path).resolve()
            if site not in located.parents:
                raise AssertionError(f"editable sibling leakage: {name} -> {path}")
    return names

def assert_no_sibling_tests(names):
    prefixes = (
        "ipfs_datasets_py.tests",
        "ipfs_kit_py.tests",
        "test.api.",
        "tests.proof_context",
        "tests.test_context_pack_store",
    )
    for name in names:
        if name == "tests" or name.startswith("tests."):
            raise AssertionError(f"sibling tests import {name}")
        for prefix in prefixes:
            if name.startswith(prefix):
                raise AssertionError(f"sibling tests import {name}")

def datasets_only():
    import ipfs_kit_py  # noqa: F401
    raise AssertionError("datasets-only install imported kit")

def kit_only_datasets():
    import ipfs_datasets_py  # noqa: F401
    raise AssertionError("kit-only install imported datasets")

def run_suite(suite):
    result = {"suite": suite, "vectors": {}, "imports": []}
    names = imported_names()
    if suite == "datasets_only":
        from ipfs_datasets_py.proof_context.context_pack import (
            AUTHORITY,
            ContextPackConstructionError,
            build_minimal_semantic_pack,
        )
        record = build_minimal_semantic_pack(**pack_kwargs())
        assert record.producer == AUTHORITY
        assert record.verify_identity() == record.pack_cid
        again = build_minimal_semantic_pack(**pack_kwargs())
        assert again.pack_cid == record.pack_cid
        try:
            build_minimal_semantic_pack(**pack_kwargs(unexpected=True))
            raise AssertionError("unknown field accepted")
        except ContextPackConstructionError as exc:
            assert "unknown field" in str(exc)
        try:
            build_minimal_semantic_pack(
                **pack_kwargs(
                    budgets={
                        "maximum_bytes": None,
                        "maximum_tokens": config["json_max_int"] + 1,
                        "maximum_retrieval_operations": None,
                        "maximum_model_class": None,
                        "maximum_wall_time": None,
                        "validation_reserve_models_cannot_consume": True,
                    }
                )
            )
            raise AssertionError("numeric bound accepted")
        except ContextPackConstructionError as exc:
            assert "JSON-safe integer bound" in str(exc)
        try:
            build_minimal_semantic_pack(
                **pack_kwargs(
                    dependencies=[{
                        "symbol": "ghost",
                        "cid": None,
                        "path": "ghost.py",
                        "meaning": "unresolved helper",
                    }]
                )
            )
            raise AssertionError("missing dependency accepted")
        except ContextPackConstructionError as exc:
            assert "missing reference" in str(exc)
        try:
            build_minimal_semantic_pack(**pack_kwargs(admit_proof_reuse=True))
            raise AssertionError("datasets claimed execution authority")
        except ContextPackConstructionError:
            pass
        try:
            datasets_only()
        except ImportError:
            result["vectors"]["authority_boundaries"] = "accept"
        else:
            raise AssertionError("datasets-only leaked kit")
        result["vectors"]["canonical_bytes_and_cid_equality"] = "accept"
        result["vectors"]["unknown_fields"] = "reject"
        result["vectors"]["numeric_bounds"] = "reject"
        result["vectors"]["missing_dependencies"] = "reject"
        result["vectors"]["independent_current_head_installation"] = "accept"
    elif suite == "kit_only":
        from ipfs_kit_py.proof_context.artifacts import cid_for_bytes
        from ipfs_kit_py.proof_context.state_store import (
            CONTEXT_PACK_INTERFACE,
            CONTEXT_PACK_NAMESPACE,
            UnknownFieldError,
            apply_context_pack_contract_vectors,
            load_context_pack_contract_vectors,
            open_context_pack_store,
            validate_context_pack_record,
        )
        payload = b'{"namespace":"ContextPack","tag":"aseh-062-kit"}'
        store = open_context_pack_store(config["store_root"])
        reference = store.put_verified_bytes(payload)
        actual = cid_for_bytes(payload)
        assert reference.cid == actual == store.cid_for(payload)
        assert store.get_immutable(reference) == payload
        try:
            store.put_verified_bytes(payload, claimed_cid="b" + "a" * 58)
            raise AssertionError("cid inequality accepted")
        except Exception:
            pass
        vectors = load_context_pack_contract_vectors()
        installed = json.loads(
            resource_files("ipfs_kit_py.proof_context")
            .joinpath("context_pack_contract_vectors.json")
            .read_text(encoding="utf-8")
        )
        assert installed == vectors
        assert vectors["interface"] == CONTEXT_PACK_INTERFACE
        assert vectors["namespace"] == CONTEXT_PACK_NAMESPACE
        applied = apply_context_pack_contract_vectors(vectors)
        by_id = {item["id"]: item for item in applied}
        assert by_id["valid-admitted-seal"]["status"] == "accept"
        assert by_id["unknown-field-top-level"]["status"] == "reject"
        try:
            validate_context_pack_record(
                {
                    "schema": "ipfs-kit.proof-context.context-pack-store@1",
                    "interface": CONTEXT_PACK_INTERFACE,
                    "namespace": CONTEXT_PACK_NAMESPACE,
                    "role": "admitted",
                    "kind": "checkpoint_seal",
                    "generation": -1,
                }
            )
            raise AssertionError("negative generation accepted")
        except Exception:
            result["vectors"]["numeric_bounds"] = "reject"
        try:
            kit_only_datasets()
        except ImportError:
            result["vectors"]["authority_boundaries"] = "accept"
        else:
            raise AssertionError("kit-only leaked datasets")
        store.close()
        result["vectors"]["canonical_bytes_and_cid_equality"] = "accept"
        result["vectors"]["unknown_fields"] = "reject"
        result["vectors"]["independent_current_head_installation"] = "accept"
        result["vectors"]["missing_dependencies"] = "reject"
    elif suite == "accelerate_only":
        from ipfs_accelerate_py.agent_supervisor.semantic_state.context_pack import (
            ContextPackAuthorityUnavailable,
            load_datasets_context_pack_authority,
            load_kit_context_pack_store,
        )
        try:
            load_datasets_context_pack_authority()
            raise AssertionError("accelerate-only reminted datasets")
        except ContextPackAuthorityUnavailable:
            pass
        try:
            load_kit_context_pack_store()
            raise AssertionError("accelerate-only invented kit bytes")
        except ContextPackAuthorityUnavailable:
            pass
        result["vectors"]["authority_boundaries"] = "accept"
        result["vectors"]["missing_dependencies"] = "reject"
        result["vectors"]["independent_current_head_installation"] = "accept"
    elif suite == "combined":
        from ipfs_accelerate_py.agent_supervisor.semantic_state.context_pack import (
            DATASETS_CONTEXT_PACK_AUTHORITY,
            ContextPackError,
            CurrentPackIdentity,
            StaleIdentityError,
            encode_context_pack_envelope,
            evaluate_exact_freshness,
            load_datasets_context_pack_authority,
            verify_datasets_semantic_identity,
            verify_kit_bytes,
        )
        from ipfs_accelerate_py.agent_supervisor.semantic_state.context_pack_selector import (
            apply_installed_kit_contract_vectors,
            installed_datasets_context_pack_schema,
            select_current_minimal_pack,
        )
        from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes as datasets_cid
        from ipfs_datasets_py.proof_context.context_pack import (
            ContextPackConstructionError,
            build_minimal_semantic_pack,
        )
        from ipfs_datasets_py.proof_context.incremental_context import expand_incremental_pack
        from ipfs_kit_py.proof_context.artifacts import cid_for_bytes as kit_cid
        from ipfs_kit_py.proof_context.state_store import open_context_pack_store

        raw = b'{"namespace":"ContextPack","tag":"aseh-062-combined"}'
        assert datasets_cid(raw) == kit_cid(raw)
        assert datasets_cid(raw).startswith("b")
        store = open_context_pack_store(config["store_root"])
        reference = store.put_verified_bytes(raw)
        assert reference.cid == kit_cid(raw)
        assert store.get_immutable(reference) == raw
        assert verify_kit_bytes(store, raw, claimed_cid=reference.cid) == reference.cid
        try:
            verify_kit_bytes(store, raw, claimed_cid=kit_cid(b"other-bytes"))
            raise AssertionError("kit CID inequality accepted")
        except ContextPackError:
            pass
        authority = load_datasets_context_pack_authority()
        assert authority.producer == DATASETS_CONTEXT_PACK_AUTHORITY
        schema = installed_datasets_context_pack_schema()
        assert schema["additionalProperties"] is False
        kit_results = apply_installed_kit_contract_vectors()
        assert any(item["status"] == "accept" for item in kit_results)
        assert any(item["status"] == "reject" for item in kit_results)
        record = build_minimal_semantic_pack(**pack_kwargs())
        envelope = record.to_dict()
        verified = verify_datasets_semantic_identity(envelope)
        assert verified["pack_cid"] == record.pack_cid
        data = encode_context_pack_envelope(envelope)
        stored = store.put_verified_bytes(data)
        store.compare_and_swap_current_root(new_cid=stored.cid, generation=0)
        assert stored.cid != record.pack_cid
        current = CurrentPackIdentity.from_envelope(envelope)
        selected = select_current_minimal_pack(store, current)
        assert selected.selected.datasets_pack_cid == record.pack_cid
        stale = build_minimal_semantic_pack(
            **pack_kwargs(
                scanned_tree_oid=config["stale_tree"],
                source_tree_oid=config["stale_tree"],
            )
        )
        verdict = evaluate_exact_freshness(stale.to_dict(), current)
        assert verdict.fresh is False
        assert "tree" in verdict.stale_fields
        stale_bytes = encode_context_pack_envelope(stale.to_dict())
        stale_ref = store.put_candidate(stale_bytes, cache_key="pack:stale")
        store.compare_and_swap_current_root(
            new_cid=stale_ref.cid,
            expected_parent_cid=stored.cid,
            generation=1,
        )
        try:
            select_current_minimal_pack(store, current)
            raise AssertionError("stale identity accepted")
        except StaleIdentityError:
            result["vectors"]["stale_identities"] = "reject"
        try:
            build_minimal_semantic_pack(**pack_kwargs(unexpected=True))
            raise AssertionError("unknown field accepted")
        except ContextPackConstructionError as exc:
            assert "unknown field" in str(exc)
            result["vectors"]["unknown_fields"] = "reject"
        try:
            build_minimal_semantic_pack(
                **pack_kwargs(
                    budgets={
                        "maximum_bytes": None,
                        "maximum_tokens": config["json_max_int"] + 1,
                        "maximum_retrieval_operations": None,
                        "maximum_model_class": None,
                        "maximum_wall_time": None,
                        "validation_reserve_models_cannot_consume": True,
                    }
                )
            )
            raise AssertionError("numeric bound accepted")
        except ContextPackConstructionError as exc:
            assert "JSON-safe integer bound" in str(exc)
            result["vectors"]["numeric_bounds"] = "reject"
        incomplete = build_minimal_semantic_pack(
            **pack_kwargs(missing_evidence=["named-missing-contract"])
        )
        incomplete_bytes = encode_context_pack_envelope(incomplete.to_dict())
        incomplete_ref = store.put_candidate(incomplete_bytes, cache_key="pack:incomplete")
        store.compare_and_swap_current_root(
            new_cid=incomplete_ref.cid,
            expected_parent_cid=stale_ref.cid,
            generation=2,
        )
        recorded = select_current_minimal_pack(store, current, require_selection=False)
        assert recorded.selected is None
        expansion = expand_incremental_pack(
            parent=record,
            scanned_tree_oid=config["tree_oid"],
            named_missing=["symbol:helper"],
            catalog=[{
                "kind": "symbol",
                "name": "helper",
                "cid": cid_label("helper"),
                "path": "helper.py",
                "meaning": "pure helper used by target",
            }],
        )
        assert expansion.parent_pack_cid == record.pack_cid
        assert expansion.pack.verify_identity() == expansion.pack.pack_cid
        store.close()
        result["vectors"]["canonical_bytes_and_cid_equality"] = "accept"
        result["vectors"]["missing_dependencies"] = "reject"
        result["vectors"]["authority_boundaries"] = "accept"
        result["vectors"]["independent_current_head_installation"] = "accept"
    else:
        raise AssertionError(f"unknown suite {suite}")
    names = imported_names()
    assert_no_sibling_tests(names)
    result["imports"] = names
    result["vectors"]["no_sibling_source_tree_test_imports"] = "reject"
    return result

def _emit(payload):
    text = json.dumps(payload, sort_keys=True)
    Path(config["result_path"]).write_text(text, encoding="utf-8")
    sys.stdout.write(text)

try:
    payload = run_suite(config["suite"])
    payload["ok"] = True
    _emit(payload)
except Exception as exc:
    traceback.print_exc()
    _emit({
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "sys_path": list(sys.path),
        "installed": sorted(installed),
    })
    raise
"""


ALLOWED_TOP_LEVEL = {
    "authority",
    "authority_boundaries",
    "authority_requirement",
    "cid_profile",
    "contract_rules",
    "current_head_policy",
    "forbidden_test_imports",
    "installation",
    "nonclaims",
    "objective_id",
    "package_promotion",
    "plan_revision",
    "planning_ancestors",
    "policy_promotion",
    "production_qualified",
    "program_id",
    "qualification",
    "qualification_fingerprint",
    "schema",
    "task_id",
    "truth_state",
    "validation",
    "vectors",
}


def _validate_qualification(payload: dict[str, Any]) -> None:
    extra = set(payload) - ALLOWED_TOP_LEVEL
    assert not extra, f"unknown field {sorted(extra)}"
    assert payload["schema"] == SCHEMA
    assert payload["program_id"] == PROGRAM_ID
    assert payload["task_id"] == TASK_ID
    assert payload["objective_id"] == OBJECTIVE_ID
    assert payload["plan_revision"] == PLAN_REVISION
    assert payload["authority"] is False
    assert payload["package_promotion"] is False
    assert payload["policy_promotion"] is False
    assert payload["production_qualified"] is False
    assert payload["qualification"] is True
    assert payload["truth_state"] == (
        "current_head_installed_package_contract_qualification"
    )
    assert "promote" in payload["authority_requirement"]
    assert payload["installation"]["editable"] is False
    assert payload["installation"]["sibling_source_tree_tests"] is False
    assert payload["installation"]["python"] == CANONICAL_PYTHON
    assert payload["installation"]["path"] == CANONICAL_PATH
    assert payload["installation"]["mode"] == (
        "independent_package_trees_without_sibling_tests"
    )
    assert tuple(payload["qualification_fingerprint"]["repositories"]) == (
        REQUIRED_REPOSITORIES
    )
    assert tuple(payload["qualification_fingerprint"]["vectors"]) == REQUIRED_VECTOR_IDS
    assert payload["qualification_fingerprint"]["schema"] == SCHEMA
    assert payload["qualification_fingerprint"]["task_id"] == TASK_ID
    identity = content_identity(payload["qualification_fingerprint"])
    _assert_real_cid(identity, field="qualification_fingerprint")
    if "identity" in payload:
        assert payload["identity"] == identity

    cid_profile = payload["cid_profile"]
    assert cid_profile["algorithm"] == "cidv1_base32_sha2_256"
    assert cid_profile["producer"].endswith("content_identity")
    assert cid_profile["identity_field"] == "qualification_fingerprint"
    for forbidden in (
        "cidv0_multihash",
        "uuid_literal",
        "placeholder_identity",
        "mock_cid",
        "fake_cid",
    ):
        assert forbidden in cid_profile["forbidden_forms"]

    requirements = _load_json(REQUIREMENTS_PATH)["cross_repository_authority"]
    boundaries = payload["authority_boundaries"]
    assert set(boundaries) == set(REQUIRED_REPOSITORIES)
    for name in REQUIRED_REPOSITORIES:
        expected = requirements[name]
        record = boundaries[name]
        assert record["owns"] == expected["owns"], name
        assert record["must_not_own"] == expected["must_not_own"], name
        assert record["package"] == name
        assert isinstance(record["producer"], str) and record["producer"]
        assert MOCK_CAPABILITY_RE.search(record["producer"]) is None, name
    assert payload["contract_rules"] == requirements["contract_rules"]
    assert tuple(payload["forbidden_test_imports"]) == FORBIDDEN_TEST_IMPORTS

    policy = payload["current_head_policy"]
    for flag in (
        "cid_inequality_invalidates",
        "dirty_head_invalidates",
        "editable_sibling_leakage_invalidates",
        "mock_capability_invalidates",
        "planning_commit_must_be_ancestor",
        "source_tree_test_import_invalidates",
        "unbound_head_invalidates",
    ):
        assert policy[flag] is True, flag
    assert policy["binding"] == "live_peeled_head_must_be_commit_and_tree"
    assert policy["status"] == "measured_at_validation"
    allowed = set(policy["allowed_accelerate_dirty_paths"])
    assert QUALIFICATION_PATH.relative_to(ROOT).as_posix() in allowed
    assert Path(__file__).relative_to(ROOT).as_posix() in allowed

    baseline = _load_json(SEALED_BASELINE_PATH)["repositories"]
    ancestors = payload["planning_ancestors"]
    assert set(ancestors) == set(REQUIRED_REPOSITORIES)
    assert ancestors["ipfs_accelerate_py"]["commit"] == baseline["ipfs_accelerate_py"][
        "planning_commit"
    ]
    assert ancestors["ipfs_accelerate_py"]["tree"] == baseline["ipfs_accelerate_py"][
        "planning_tree"
    ]
    assert ancestors["ipfs_datasets_py"]["commit"] == baseline["ipfs_datasets_py"]["commit"]
    assert ancestors["ipfs_datasets_py"]["tree"] == baseline["ipfs_datasets_py"]["tree"]
    assert ancestors["ipfs_kit_py"]["commit"] == baseline["ipfs_kit_py"]["commit"]
    assert ancestors["ipfs_kit_py"]["tree"] == baseline["ipfs_kit_py"]["tree"]

    packages = payload["installation"]["packages"]
    assert set(packages) == set(REQUIRED_REPOSITORIES)
    for name, record in packages.items():
        source = Path(record["source_package"])
        tests = ROOT / record["tests_directory"]
        assert (ROOT / source).is_dir() or (ROOT / source.parts[0] / source.parts[-1]).is_dir()
        assert PACKAGE_SOURCES[name].is_dir(), name
        assert tests.is_dir(), record["tests_directory"]

    vectors = payload["vectors"]
    assert isinstance(vectors, list) and vectors
    by_id = {item["id"]: item for item in vectors}
    assert tuple(sorted(by_id)) == REQUIRED_VECTOR_IDS
    assert len(by_id) == len(vectors)
    for vector_id, item in by_id.items():
        assert set(item) == {"id", "expect", "proof"}, vector_id
        assert item["expect"] in {"accept", "reject"}, vector_id
        assert isinstance(item["proof"], str) and item["proof"], vector_id
        assert PSEUDO_CID_RE.search(item["proof"]) is None, vector_id
        assert MOCK_CAPABILITY_RE.search(item["proof"]) is None, vector_id
    assert by_id["canonical_bytes_and_cid_equality"]["expect"] == "accept"
    assert by_id["unknown_fields"]["expect"] == "reject"
    assert by_id["numeric_bounds"]["expect"] == "reject"
    assert by_id["stale_identities"]["expect"] == "reject"
    assert by_id["missing_dependencies"]["expect"] == "reject"
    assert by_id["authority_boundaries"]["expect"] == "accept"
    assert by_id["no_sibling_source_tree_test_imports"]["expect"] == "reject"
    assert by_id["independent_current_head_installation"]["expect"] == "accept"

    assert payload["validation"] == [
        "python3 -m pytest -q test/api/agent_supervisor/efficiency_state_hardening/test_cross_repository_contracts.py"
    ]
    assert payload["nonclaims"], "qualification must record nonclaims"
    for note in payload["nonclaims"]:
        assert isinstance(note, str) and note.strip()
        assert "not" in note.lower() or "never" in note.lower() or "cannot" in note.lower()

    identity = content_identity(payload["qualification_fingerprint"])
    _assert_real_cid(identity, field="qualification_fingerprint")


@pytest.fixture(scope="module")
def isolated_homes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("aseh062-isolated")
    homes = {
        "datasets": root / "datasets",
        "kit": root / "kit",
        "accelerate": root / "accelerate",
        "combined": root / "combined",
    }
    for home in homes.values():
        (home / "site-packages").mkdir(parents=True)
        (home / "work").mkdir()
        (home / "store").mkdir()
        (home / ".cache").mkdir()
        (home / ".config").mkdir()
        (home / ".local" / "share").mkdir(parents=True)
        (home / ".local" / "state").mkdir(parents=True)
    _install_packages(homes["datasets"] / "site-packages", ("ipfs_datasets_py",))
    _install_packages(homes["kit"] / "site-packages", ("ipfs_kit_py",))
    _install_packages(homes["accelerate"] / "site-packages", ("ipfs_accelerate_py",))
    _install_packages(
        homes["combined"] / "site-packages",
        REQUIRED_REPOSITORIES,
    )
    _warmup_contract_imports()
    for home in homes.values():
        _install_runtime_dependencies(home / "site-packages")
        _install_discovered_third_party(home / "site-packages")
    return homes


def test_qualification_schema_canonical_bytes_and_real_cid() -> None:
    payload = _load(QUALIFICATION_PATH)
    _validate_qualification(payload)
    identity = content_identity(payload["qualification_fingerprint"])
    _assert_real_cid(identity, field="qualification identity")


def test_current_heads_peel_and_planning_commits_are_ancestors() -> None:
    heads = _live_heads()
    assert set(heads) == set(REQUIRED_REPOSITORIES)
    for name, record in heads.items():
        _assert_sha1(record["commit"], field=f"{name}.commit")
        _assert_sha1(record["tree"], field=f"{name}.tree")
        assert record["planning_commit"]
        assert isinstance(record["dirty"], list)


def test_this_module_and_authorities_do_not_import_sibling_source_tree_tests() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    imported = _imported_module_names(ast.parse(source))
    _assert_no_sibling_test_imports(sorted(imported))
    assert "ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts" in imported
    production = [
        ROOT / "ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack.py",
        ROOT / "ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack_selector.py",
        ROOT / "ipfs_datasets_py/ipfs_datasets_py/proof_context/context_pack.py",
        ROOT / "ipfs_datasets_py/ipfs_datasets_py/proof_context/incremental_context.py",
        ROOT / "ipfs_kit_py/ipfs_kit_py/proof_context/state_store.py",
    ]
    for path in production:
        names = _imported_module_names(ast.parse(path.read_text(encoding="utf-8")))
        _assert_no_sibling_test_imports(sorted(names))
        source_text = path.read_text(encoding="utf-8")
        assert "tests.proof_context" not in source_text
        assert "tests.test_context_pack_store" not in source_text
        assert "tests/proof_context" not in source_text


def test_unknown_field_bounds_and_mock_capability_fail_closed() -> None:
    payload = _load(QUALIFICATION_PATH)

    unknown = copy.deepcopy(payload)
    unknown["unexpected"] = True
    with pytest.raises(AssertionError):
        _validate_qualification(unknown)

    mock_capability = copy.deepcopy(payload)
    mock_capability["authority_boundaries"]["ipfs_kit_py"]["producer"] = "mock_capability"
    with pytest.raises(AssertionError):
        _validate_qualification(mock_capability)

    fake_cid = copy.deepcopy(payload)
    fake_cid["qualification_fingerprint"]["task_id"] = "uuid:00000000-0000-4000-8000-000000000062"
    with pytest.raises(AssertionError):
        _validate_qualification(fake_cid)

    editable = copy.deepcopy(payload)
    editable["installation"]["editable"] = True
    with pytest.raises(AssertionError):
        _validate_qualification(editable)

    sibling_tests = copy.deepcopy(payload)
    sibling_tests["installation"]["sibling_source_tree_tests"] = True
    with pytest.raises(AssertionError):
        _validate_qualification(sibling_tests)

    unbound = copy.deepcopy(payload)
    unbound["planning_ancestors"]["ipfs_kit_py"]["commit"] = ""
    with pytest.raises(AssertionError):
        _validate_qualification(unbound)

    promote = copy.deepcopy(payload)
    promote["policy_promotion"] = True
    with pytest.raises(AssertionError):
        _validate_qualification(promote)

    missing_vector = copy.deepcopy(payload)
    missing_vector["vectors"] = [
        item for item in payload["vectors"] if item["id"] != "stale_identities"
    ]
    with pytest.raises(AssertionError):
        _validate_qualification(missing_vector)


def test_unbound_and_dirty_heads_invalidate_qualification() -> None:
    policy = _load(QUALIFICATION_PATH)["current_head_policy"]
    assert policy["unbound_head_invalidates"] is True
    assert policy["dirty_head_invalidates"] is True
    with pytest.raises(AssertionError, match="unbound head"):
        _peel_head(ROOT / "does-not-exist-aseh-062")
    bogus = copy.deepcopy(_load(QUALIFICATION_PATH))
    bogus["planning_ancestors"]["ipfs_datasets_py"]["tree"] = "not-a-tree"
    with pytest.raises(AssertionError):
        _validate_qualification(bogus)


def test_independent_datasets_install_proves_semantic_boundary(
    isolated_homes: dict[str, Path],
) -> None:
    home = isolated_homes["datasets"]
    result = _run_isolated(
        site=home / "site-packages",
        workdir=home / "work",
        home=home,
        suite="datasets_only",
        store_root=home / "store",
    )
    assert result["vectors"]["authority_boundaries"] == "accept"
    assert result["vectors"]["unknown_fields"] == "reject"
    assert result["vectors"]["numeric_bounds"] == "reject"
    assert result["vectors"]["missing_dependencies"] == "reject"
    assert result["vectors"]["no_sibling_source_tree_test_imports"] == "reject"
    _assert_no_sibling_test_imports(result["imports"])
    assert not any(name.startswith("ipfs_kit_py") for name in result["imports"])
    assert not any(name.startswith("ipfs_accelerate_py") for name in result["imports"])
    site = home / "site-packages"
    assert not (site / "ipfs_datasets_py" / "tests").exists()
    assert not (site / "ipfs_kit_py").exists()
    assert not (site / "ipfs_accelerate_py").exists()


def test_independent_kit_install_proves_durable_bytes_and_cid_equality(
    isolated_homes: dict[str, Path],
) -> None:
    home = isolated_homes["kit"]
    result = _run_isolated(
        site=home / "site-packages",
        workdir=home / "work",
        home=home,
        suite="kit_only",
        store_root=home / "store",
    )
    assert result["vectors"]["canonical_bytes_and_cid_equality"] == "accept"
    assert result["vectors"]["unknown_fields"] == "reject"
    assert result["vectors"]["numeric_bounds"] == "reject"
    assert result["vectors"]["authority_boundaries"] == "accept"
    _assert_no_sibling_test_imports(result["imports"])
    assert not any(name.startswith("ipfs_datasets_py") for name in result["imports"])
    site = home / "site-packages"
    assert not (site / "ipfs_datasets_py").exists()
    assert not (site / "ipfs_kit_py" / "tests").exists()


def test_independent_accelerate_install_does_not_remint_missing_authorities(
    isolated_homes: dict[str, Path],
) -> None:
    home = isolated_homes["accelerate"]
    result = _run_isolated(
        site=home / "site-packages",
        workdir=home / "work",
        home=home,
        suite="accelerate_only",
        store_root=home / "store",
    )
    assert result["vectors"]["authority_boundaries"] == "accept"
    assert result["vectors"]["missing_dependencies"] == "reject"
    _assert_no_sibling_test_imports(result["imports"])
    assert not any(name.startswith("ipfs_datasets_py") for name in result["imports"])
    assert not any(name.startswith("ipfs_kit_py") for name in result["imports"])
    site = home / "site-packages"
    assert not (site / "ipfs_datasets_py").exists()
    assert not (site / "ipfs_kit_py").exists()


def test_combined_isolated_install_passes_all_cross_repository_vectors(
    isolated_homes: dict[str, Path],
) -> None:
    home = isolated_homes["combined"]
    result = _run_isolated(
        site=home / "site-packages",
        workdir=home / "work",
        home=home,
        suite="combined",
        store_root=home / "store",
    )
    expected = {
        item["id"]: item["expect"] for item in _load(QUALIFICATION_PATH)["vectors"]
    }
    for vector_id, expect in expected.items():
        assert result["vectors"][vector_id] == expect, vector_id
    _assert_no_sibling_test_imports(result["imports"])
    site = str((home / "site-packages").resolve())
    for name in result["imports"]:
        if name.split(".", 1)[0] in REQUIRED_REPOSITORIES:
            module = name.split(".", 1)[0]
            assert any(
                item.startswith(module) for item in result["imports"]
            )


def test_editable_sibling_leakage_is_detected(isolated_homes: dict[str, Path]) -> None:
    home = isolated_homes["combined"]
    leaked_path = _isolated_pythonpath(home / "site-packages")
    leaked_path.insert(0, str(ROOT / "ipfs_kit_py"))
    leaked_path.insert(0, str(ROOT / "ipfs_datasets_py"))
    leaked_path.insert(0, str(ROOT))
    result_path = home / "work" / "leakage-result.json"
    config = {
        "site_packages": str(home / "site-packages"),
        "pythonpath": leaked_path,
        "workdir": str(home / "work"),
        "store_root": str(home / "store"),
        "result_path": str(result_path),
        "suite": "combined",
        "forbidden_roots": [],
        "strip_forbidden": False,
        "tree_oid": TREE_OID,
        "stale_tree": STALE_TREE,
        "objective": OBJECTIVE,
        "objective_revision": OBJECTIVE_REVISION,
        "toolchain": TOOLCHAIN,
        "environment": ENVIRONMENT,
        "schema_identity": SCHEMA_IDENTITY,
        "json_max_int": JSON_MAX_INT,
    }
    completed = subprocess.run(
        [sys.executable, "-s", "-c", _ISOLATED_RUNNER, json.dumps(config)],
        cwd=str(ROOT),
        env=_isolated_env(home),
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode != 0
    payload = json.loads(
        result_path.read_text(encoding="utf-8") if result_path.is_file() else '{"ok": false}'
    )
    assert payload.get("ok") is not True
