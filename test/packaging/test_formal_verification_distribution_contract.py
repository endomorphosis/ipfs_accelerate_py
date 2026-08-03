"""FormalVerificationDistributionContract@1 (FVT-084 / FVT-G215).

Machine-checked packaging and distribution gate for formal verification:

* root and datasets setup/pyproject/requirements share one dependency inventory;
* namespace-package discovery includes logic backends, software verification,
  installer plugins, and runtime assets that must ship in wheels;
* every declared installer plugin module exists on disk;
* a clean isolated wheel install imports and inventories the Logic API without
  network access, downloads, builds, user-site leakage, editable-source
  leakage, or installation side effects;
* optional native/external provers stay optional and never become mandatory
  pip dependencies that would break base installation.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path
from typing import Any

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py<3.11 fallback
    import tomli as tomllib  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = REPO_ROOT / "ipfs_datasets_py"
DATASETS_PACKAGE = DATASETS_ROOT / "ipfs_datasets_py"

DISTRIBUTION_CONTRACT_INTERFACE = "FormalVerificationDistributionContract@1"
GOAL_ID = "FVT-G215"
TASK_ID = "FVT-084"

# Python bindings that must stay consistent across datasets packaging surfaces.
# Native solver binaries remain lazy/user-local and must never appear here.
THEOREM_PYTHON_BINDINGS: frozenset[str] = frozenset(
    {
        "z3-solver",
        "cvc5",
        "pysmt",
        "beartype",
    }
)

# Optional advisor binding allowed only in extras / theorem profile, never as
# an undeclared drift from the theorem-provers requirements file.
THEOREM_OPTIONAL_PYTHON_BINDINGS: frozenset[str] = frozenset({"symbolicai"})

# Root accelerate package pins used by proof-gated contract repair.
ROOT_CONTRACT_REPAIR_DISTRIBUTIONS: frozenset[str] = frozenset(
    {"z3-solver", "cvc5", "mypy", "ruff"}
)

# Heavyweight native / external provers must not become mandatory pip deps.
FORBIDDEN_MANDATORY_PROVER_DISTRIBUTIONS: frozenset[str] = frozenset(
    {
        "tamarin-prover",
        "tamarin",
        "proverif",
        "apalache",
        "lean",
        "lean4",
        "elan",
        "coq",
        "rocq",
        "isabelle",
        "vampire",
        "eprover",
        "e-prover",
        "souffle",
        "maude",
        "tlc",
        "tlaplus",
        "hyperltl",
        "autohyper",
        "mchyper",
        "secpal",
        "ergoai",
        "provekit",
    }
)

NAMESPACE_PACKAGES: tuple[str, ...] = (
    "ipfs_datasets_py.logic.backends",
    "ipfs_datasets_py.logic.backends.installers",
    "ipfs_datasets_py.logic.software_verification",
    "ipfs_datasets_py.logic.software_verification.monitoring",
    "ipfs_datasets_py.logic.software_verification.counterexamples",
)

STABLE_MODULE_RELATIVE_PATHS: dict[str, str] = {
    "ipfs_datasets_py.logic.verification_api": "logic/verification_api.py",
    "ipfs_datasets_py.logic.backends.toolchains": "logic/backends/toolchains.py",
    "ipfs_datasets_py.logic.backends.process": "logic/backends/process.py",
    "ipfs_datasets_py.logic.backends.registry": "logic/backends/registry.py",
    "ipfs_datasets_py.logic.backends.installers.registry": (
        "logic/backends/installers/registry.py"
    ),
    "ipfs_datasets_py.logic.software_verification.vc": (
        "logic/software_verification/vc.py"
    ),
    "ipfs_datasets_py.logic.software_verification.contracts": (
        "logic/software_verification/contracts.py"
    ),
    "ipfs_datasets_py.logic.software_verification.receipts": (
        "logic/software_verification/receipts.py"
    ),
    "ipfs_datasets_py.logic.software_verification.monitoring.runtime_mtl": (
        "logic/software_verification/monitoring/runtime_mtl.py"
    ),
}

RUNTIME_ASSET_GLOBS: tuple[str, ...] = (
    "logic/zkp/provekit/circuits/*/Nargo.toml",
    "logic/zkp/provekit/circuits/*/src/*.nr",
    "logic/legal_ir/schemas/*.json",
    "processors/provekit_backend/README.md",
    "processors/provekit_backend/build.sh",
)

WHEEL_BUILD_TIMEOUT_SECONDS = 600.0
IMPORT_TIMEOUT_SECONDS = 90.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _distribution_name(requirement: str) -> str:
    token = re.split(r"[\s\[<>=!~;@]", requirement.strip(), maxsplit=1)[0]
    return token.replace("_", "-").lower()


def _parse_requirement_names(text: str) -> set[str]:
    names: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        # Editable / path / VCS markers are not distribution inventory rows.
        if stripped.startswith(("-e ", "--", "git+", "file:")):
            continue
        names.add(_distribution_name(stripped))
    return names


def _parse_requirement_lines(text: str) -> list[str]:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        if stripped.startswith(("-e ", "--", "git+", "file:")):
            continue
        lines.append(stripped)
    return lines


def _read_text(path: Path) -> str:
    assert path.is_file(), f"missing packaging file: {path}"
    return path.read_text(encoding="utf-8")


def _load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(_read_text(path))


def _offline_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base or os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_NO_BUILD_ISOLATION"] = "0"
    env["IPFS_DATASETS_PY_AUTO_NLTK_DOWNLOAD"] = "0"
    env["IPFS_DATASETS_PY_AUTO_GROTH16_BUILD"] = "0"
    env["IPFS_DATASETS_PY_INCLUDE_VCS_DEPENDENCIES"] = "0"
    env["ELAN_NO_AUTO_INSTALL"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env.setdefault("NO_PROXY", "*")
    env.setdefault("no_proxy", "*")
    # Drop editable / source leakage paths for child processes.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


def _bounded_run(
    argv: list[str],
    *,
    timeout: float,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(cwd) if cwd is not None else None,
    )


def _module_to_relative_path(module: str) -> str:
    parts = module.split(".")
    assert parts[0] == "ipfs_datasets_py"
    return "/".join(parts[1:]) + ".py"


def _wheel_member_suffixes(wheel_path: Path) -> set[str]:
    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()
    # Normalize both purelib layouts and .data/purelib layouts to package paths.
    normalized: set[str] = set()
    for name in names:
        if "/ipfs_datasets_py/" in name:
            normalized.add(name.split("/ipfs_datasets_py/", 1)[1])
        elif name.startswith("ipfs_datasets_py/"):
            normalized.add(name[len("ipfs_datasets_py/") :])
        normalized.add(name)
    return normalized


def _egg_metadata_snapshot(source_root: Path) -> dict[str, bytes]:
    """Capture source-checkout metadata that a wheel build must not rewrite."""

    metadata_root = source_root / "ipfs_datasets_py.egg-info"
    if not metadata_root.is_dir():
        return {}
    return {
        path.relative_to(metadata_root).as_posix(): path.read_bytes()
        for path in sorted(metadata_root.rglob("*"))
        if path.is_file()
    }


def _copy_distribution_source(source_root: Path, destination: Path) -> Path:
    """Stage only distribution inputs outside the tracked source checkout.

    Setuptools writes ``*.egg-info`` as part of ``bdist_wheel`` even when the
    requested wheel output directory is elsewhere.  Running that command in a
    task worktree therefore turns a read-only packaging check into unrelated
    tracked metadata changes.  A disposable source stage keeps every build
    side effect under pytest's temporary directory.
    """

    destination.mkdir(parents=True, exist_ok=False)
    for name in (
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "MANIFEST.in",
        "README.md",
        "__init__.py",
        "ipfs_datasets_cli.py",
        "pyproject.toml",
        "requirements-docs.txt",
        "requirements-lazy.txt",
        "requirements-theorem-provers.txt",
        "requirements.txt",
        "setup.py",
    ):
        source = source_root / name
        if source.is_file():
            shutil.copy2(source, destination / name)

    ignored = shutil.ignore_patterns(
        ".git",
        ".pytest_cache",
        "*.egg-info",
        "__pycache__",
        "*.pyc",
        "build",
        "dist",
        "node_modules",
    )
    shutil.copytree(
        source_root / "ipfs_datasets_py",
        destination / "ipfs_datasets_py",
        symlinks=True,
        ignore=ignored,
    )
    shutil.copytree(
        source_root / "typescript" / "logic-runtime-mtl",
        destination / "typescript" / "logic-runtime-mtl",
        symlinks=True,
        ignore=ignored,
    )
    return destination


# ---------------------------------------------------------------------------
# Interface + dependency inventory
# ---------------------------------------------------------------------------


def test_distribution_contract_interface_is_declared() -> None:
    """Stable interface identity for FormalVerificationDistributionContract@1."""

    assert DISTRIBUTION_CONTRACT_INTERFACE == "FormalVerificationDistributionContract@1"
    assert GOAL_ID == "FVT-G215"
    assert TASK_ID == "FVT-084"
    for relative in (
        "setup.py",
        "pyproject.toml",
        "requirements.txt",
        "ipfs_datasets_py/setup.py",
        "ipfs_datasets_py/pyproject.toml",
        "ipfs_datasets_py/requirements.txt",
        "ipfs_datasets_py/requirements-lazy.txt",
        "ipfs_datasets_py/requirements-theorem-provers.txt",
    ):
        assert (REPO_ROOT / relative).is_file(), relative


def test_root_and_datasets_dependency_inventory_is_machine_checked() -> None:
    """One inventory binds root contract-repair pins and datasets theorem extras."""

    root_requirements = _parse_requirement_names(
        _read_text(REPO_ROOT / "requirements.txt")
    )
    root_setup = _read_text(REPO_ROOT / "setup.py")
    root_pyproject = _load_toml(REPO_ROOT / "pyproject.toml")

    assert ROOT_CONTRACT_REPAIR_DISTRIBUTIONS <= root_requirements
    assert "CONTRACT_REPAIR_DISTRIBUTIONS" in root_setup
    for name in ROOT_CONTRACT_REPAIR_DISTRIBUTIONS:
        assert name in root_setup or name.replace("-", "_") in root_setup
    # Root packaging still surfaces optional-dependencies from pyproject.
    assert "optional-dependencies" in root_pyproject.get("project", {})

    datasets_req = _parse_requirement_names(
        _read_text(DATASETS_ROOT / "requirements.txt")
    )
    datasets_lazy = _parse_requirement_names(
        _read_text(DATASETS_ROOT / "requirements-lazy.txt")
    )
    datasets_theorem_lines = _parse_requirement_lines(
        _read_text(DATASETS_ROOT / "requirements-theorem-provers.txt")
    )
    datasets_theorem = {_distribution_name(line) for line in datasets_theorem_lines}

    datasets_pyproject = _load_toml(DATASETS_ROOT / "pyproject.toml")
    optional = datasets_pyproject["project"]["optional-dependencies"]
    theorem_extra = {_distribution_name(item) for item in optional["theorem-provers"]}
    lazy_extra = {_distribution_name(item) for item in optional["lazy"]}
    logic_extra = {_distribution_name(item) for item in optional["logic"]}

    datasets_setup = _read_text(DATASETS_ROOT / "setup.py")

    # Core solver Python bindings appear in the shared inventory surfaces.
    assert THEOREM_PYTHON_BINDINGS <= datasets_req
    assert THEOREM_PYTHON_BINDINGS <= datasets_lazy
    assert THEOREM_PYTHON_BINDINGS <= datasets_theorem
    assert THEOREM_PYTHON_BINDINGS <= theorem_extra
    assert THEOREM_PYTHON_BINDINGS <= lazy_extra

    for name in THEOREM_PYTHON_BINDINGS:
        assert name in datasets_setup

    # Theorem profile and lazy profile stay aligned for deterministic bindings.
    deterministic_theorem = datasets_theorem - THEOREM_OPTIONAL_PYTHON_BINDINGS
    assert deterministic_theorem <= datasets_lazy
    assert deterministic_theorem <= lazy_extra
    assert deterministic_theorem <= theorem_extra

    # SymbolicAI remains an explicit optional theorem/logic extra, not a base
    # install requirement that would force advisor stacks on every install.
    assert "symbolicai" in theorem_extra
    assert "symbolicai" in logic_extra
    assert "symbolicai" in datasets_theorem
    assert "symbolicai" not in datasets_req

    # Exact pins from requirements-theorem-provers.txt are reflected in extras.
    for line in datasets_theorem_lines:
        assert line in optional["theorem-provers"] or _distribution_name(line) in {
            _distribution_name(item) for item in optional["theorem-provers"]
        }
        assert line in datasets_setup or _distribution_name(line) in {
            "symbolicai",
            "jsonschema",
            *THEOREM_PYTHON_BINDINGS,
        }


def test_optional_native_provers_are_not_mandatory_pip_dependencies() -> None:
    """Heavyweight provers stay optional; base install must not require them."""

    inventories = (
        _parse_requirement_names(_read_text(REPO_ROOT / "requirements.txt")),
        _parse_requirement_names(_read_text(DATASETS_ROOT / "requirements.txt")),
        _parse_requirement_names(_read_text(DATASETS_ROOT / "requirements-lazy.txt")),
        _parse_requirement_names(
            _read_text(DATASETS_ROOT / "requirements-theorem-provers.txt")
        ),
    )
    for inventory in inventories:
        forbidden = FORBIDDEN_MANDATORY_PROVER_DISTRIBUTIONS & inventory
        assert not forbidden, (
            "native/external provers must remain optional installers, not "
            f"mandatory pip requirements: {sorted(forbidden)}"
        )

    datasets_setup = _read_text(DATASETS_ROOT / "setup.py")
    # setup.py must document lazy/user-local native installers.
    assert "ipfs-datasets-install-provers" in datasets_setup
    assert "lazily" in datasets_setup.lower() or "lazy" in datasets_setup.lower()

    datasets_pyproject = _load_toml(DATASETS_ROOT / "pyproject.toml")
    scripts = datasets_pyproject.get("project", {}).get("scripts", {})
    assert "ipfs-datasets-install-provers" in scripts


# ---------------------------------------------------------------------------
# Namespace discovery + plugin modules + runtime assets
# ---------------------------------------------------------------------------


def test_namespace_package_discovery_includes_formal_verification_surfaces() -> None:
    from setuptools import find_namespace_packages, find_packages

    datasets_setup = _read_text(DATASETS_ROOT / "setup.py")
    datasets_pyproject = _load_toml(DATASETS_ROOT / "pyproject.toml")

    assert "find_namespace_packages" in datasets_setup
    assert "packages=find_packages(" not in datasets_setup
    package_find = datasets_pyproject["tool"]["setuptools"]["packages"]["find"]
    assert package_find.get("namespaces") is True
    assert "ipfs_datasets_py.*" in package_find["include"]

    discovered = set(
        find_namespace_packages(
            where=str(DATASETS_ROOT),
            include=["ipfs_datasets_py*"],
        )
    )
    missing = [name for name in NAMESPACE_PACKAGES if name not in discovered]
    assert not missing, f"namespace discovery omitted formal-verification packages: {missing}"

    # Document classical find_packages gap; release packaging must not rely on it.
    classical = set(
        find_packages(
            where=str(DATASETS_ROOT),
            include=["ipfs_datasets_py*"],
        )
    )
    omitted_classically = [name for name in NAMESPACE_PACKAGES if name not in classical]
    if omitted_classically:
        assert "find_namespace" in datasets_setup or package_find.get("namespaces") is True


def test_declared_installer_plugin_modules_exist() -> None:
    """Every reviewed family plugin declared by the installer registry exists."""

    sys_path_inserted = False
    datasets_root = str(DATASETS_ROOT)
    if datasets_root not in sys.path:
        sys.path.insert(0, datasets_root)
        sys_path_inserted = True
    try:
        from ipfs_datasets_py.logic.backends.installers.registry import (
            PLUGIN_MODULE_PATHS,
            install_is_forbidden_on_import,
            list_installer_plugins,
            registry_side_effect_free_on_import,
        )

        assert registry_side_effect_free_on_import() is True
        assert install_is_forbidden_on_import() is True

        plugins = list_installer_plugins()
        assert plugins, "installer registry must declare family plugins"
        assert set(PLUGIN_MODULE_PATHS) == {plugin.family.value for plugin in plugins}

        for family, module_path in PLUGIN_MODULE_PATHS.items():
            relative = _module_to_relative_path(module_path)
            source = DATASETS_PACKAGE / relative
            assert source.is_file(), (
                f"declared installer plugin {family!r} missing at {source} "
                f"(module {module_path})"
            )
    finally:
        if sys_path_inserted:
            try:
                sys.path.remove(datasets_root)
            except ValueError:
                pass


def test_runtime_assets_are_declared_for_wheels() -> None:
    datasets_setup = _read_text(DATASETS_ROOT / "setup.py")
    datasets_pyproject = _load_toml(DATASETS_ROOT / "pyproject.toml")
    package_data = datasets_pyproject["tool"]["setuptools"]["package-data"][
        "ipfs_datasets_py"
    ]
    manifest = _read_text(DATASETS_ROOT / "MANIFEST.in")

    for asset in RUNTIME_ASSET_GLOBS:
        assert asset in package_data, asset
        assert asset in datasets_setup or asset.replace("*/", "") in datasets_setup

    # Runtime MTL TypeScript sources are vendored into the wheel by build_py.
    assert "_BuildPyWithFormalVerificationAssets" in datasets_setup
    assert "logic-runtime-mtl" in datasets_setup
    assert "recursive-include typescript/logic-runtime-mtl" in manifest
    assert "prune typescript/logic-runtime-mtl/node_modules" in manifest
    assert "prune typescript/logic-runtime-mtl/dist" in manifest

    source_root = DATASETS_ROOT / "typescript" / "logic-runtime-mtl"
    for relative in (
        "package.json",
        "package-lock.json",
        "tsconfig.json",
        "src/index.ts",
        "src/cli.ts",
    ):
        assert (source_root / relative).is_file(), relative


def test_source_tree_contains_stable_verification_modules() -> None:
    for module, relative in STABLE_MODULE_RELATIVE_PATHS.items():
        path = DATASETS_PACKAGE / relative
        assert path.is_file(), f"{module} missing at {path}"


# ---------------------------------------------------------------------------
# Clean isolated wheel install
# ---------------------------------------------------------------------------


def test_distribution_build_uses_disposable_source_without_egg_metadata(
    tmp_path: Path,
) -> None:
    before = _egg_metadata_snapshot(DATASETS_ROOT)
    staged = _copy_distribution_source(DATASETS_ROOT, tmp_path / "source")

    assert staged != DATASETS_ROOT
    assert (staged / "setup.py").is_file()
    assert (staged / "ipfs_datasets_py" / "logic" / "verification_api.py").is_file()
    assert (
        staged / "typescript" / "logic-runtime-mtl" / "package.json"
    ).is_file()
    assert not (staged / "ipfs_datasets_py.egg-info").exists()
    assert not (
        staged / "typescript" / "logic-runtime-mtl" / "node_modules"
    ).exists()
    assert _egg_metadata_snapshot(DATASETS_ROOT) == before


def test_clean_isolated_wheel_install_imports_and_inventories_logic_api(
    tmp_path: Path,
) -> None:
    """Build a local wheel, install offline into an isolated target, inventory.

    Conflict policy: do not hide missing wheel content with source PYTHONPATH,
    do not network/download/build during install/import, and do not treat user
    site or editable checkouts as substitutes for packaged content.
    """

    dist_dir = tmp_path / "dist"
    target_dir = tmp_path / "site"
    dist_dir.mkdir()
    target_dir.mkdir()
    metadata_before = _egg_metadata_snapshot(DATASETS_ROOT)
    build_source = _copy_distribution_source(
        DATASETS_ROOT,
        tmp_path / "source",
    )

    env = _offline_env()
    # Ensure setuptools/wheel from the current interpreter remain importable
    # while still forbidding index access.
    build = _bounded_run(
        [
            sys.executable,
            "setup.py",
            "bdist_wheel",
            "--dist-dir",
            str(dist_dir),
        ],
        timeout=WHEEL_BUILD_TIMEOUT_SECONDS,
        env=env,
        cwd=build_source,
    )
    assert build.returncode == 0, (
        "wheel build failed offline\n"
        f"stdout={build.stdout[-4000:]}\nstderr={build.stderr[-4000:]}"
    )
    assert _egg_metadata_snapshot(DATASETS_ROOT) == metadata_before, (
        "wheel build modified tracked source-checkout egg metadata; "
        "builds must run only in the disposable source stage"
    )

    wheels = sorted(dist_dir.glob("ipfs_datasets_py-*.whl"))
    assert wheels, f"no wheel produced in {dist_dir}"
    wheel_path = wheels[-1]

    members = _wheel_member_suffixes(wheel_path)
    for relative in STABLE_MODULE_RELATIVE_PATHS.values():
        assert relative in members or any(
            member.endswith(relative) for member in members
        ), f"wheel missing stable module path {relative}"

    for family_module in (
        "logic/backends/installers/registry.py",
        "logic/backends/installers/solver.py",
        "logic/backends/installers/atp.py",
        "logic/backends/installers/state_model.py",
        "logic/backends/installers/tamarin.py",
        "logic/backends/installers/proverif.py",
        "logic/backends/installers/rocq.py",
        "logic/backends/installers/isabelle.py",
        "logic/backends/installers/hyperproperty.py",
        "logic/backends/installers/authorization.py",
        "logic/backends/installers/runtime_mtl.py",
        "logic/backends/installers/advisors.py",
        "logic/backends/installers/kernel.py",
        "logic/backends/installers/zkp.py",
    ):
        assert family_module in members or any(
            member.endswith(family_module) for member in members
        ), f"wheel missing installer plugin {family_module}"

    assert any(
        member.endswith("_vendor/logic-runtime-mtl/package.json")
        or member.endswith("_vendor/logic-runtime-mtl/package.json")
        or "logic-runtime-mtl/package.json" in member
        for member in members
    ), "wheel missing vendored Runtime MTL package.json"

    install = _bounded_run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--upgrade",
            "--target",
            str(target_dir),
            str(wheel_path),
        ],
        timeout=120.0,
        env=env,
        cwd=tmp_path,
    )
    assert install.returncode == 0, (
        "offline wheel install failed\n"
        f"stdout={install.stdout[-4000:]}\nstderr={install.stderr[-4000:]}"
    )

    installed_api = target_dir / "ipfs_datasets_py" / "logic" / "verification_api.py"
    assert installed_api.is_file(), "wheel install omitted verification_api.py"
    assert (
        target_dir
        / "ipfs_datasets_py"
        / "_vendor"
        / "logic-runtime-mtl"
        / "package.json"
    ).is_file(), "wheel install omitted Runtime MTL vendor assets"

    # Probe import + inventory in a clean interpreter that cannot see the
    # monorepo source tree or user site packages.
    probe = textwrap.dedent(
        """
        import json
        import os
        import sys
        from pathlib import Path

        target = Path({target!r}).resolve()
        repo_root = Path({repo!r}).resolve()
        datasets_root = Path({datasets!r}).resolve()

        # Keep interpreter stdlib/site-packages so imports work, but never fall
        # back to the monorepo checkout or editable source trees.
        clean_path: list[str] = []
        for entry in list(sys.path):
            if not entry:
                clean_path.append(entry)
                continue
            try:
                resolved = str(Path(entry).resolve())
            except OSError:
                clean_path.append(entry)
                continue
            if resolved == str(target):
                continue
            if str(repo_root) in resolved or str(datasets_root) in resolved:
                continue
            clean_path.append(entry)
        sys.path[:] = [str(target)] + clean_path

        forbidden_events = []

        def _block(*_a, **_k):
            forbidden_events.append("subprocess")
            raise RuntimeError("install/network subprocess forbidden during inventory")

        import subprocess as _sp
        _sp.run = _block  # type: ignore[assignment]
        _sp.Popen = _block  # type: ignore[assignment]
        _sp.call = _block  # type: ignore[assignment]
        _sp.check_call = _block  # type: ignore[assignment]
        _sp.check_output = _block  # type: ignore[assignment]

        from ipfs_datasets_py.logic.verification_api import (
            LogicVerificationAPI,
            STABLE_OPERATIONS,
        )
        from ipfs_datasets_py.logic.backends.installers.registry import (
            PLUGIN_MODULE_PATHS,
            install_is_forbidden_on_import,
            list_installer_plugins,
            registry_side_effect_free_on_import,
        )

        api_module = __import__(
            "ipfs_datasets_py.logic.verification_api", fromlist=["LogicVerificationAPI"]
        )
        module_file = Path(api_module.__file__).resolve()
        assert str(target.resolve()) in str(module_file), module_file
        assert str(datasets_root.resolve()) not in str(module_file), module_file

        assert registry_side_effect_free_on_import() is True
        assert install_is_forbidden_on_import() is True

        api = LogicVerificationAPI()
        features = api.list_features().to_dict()
        providers = api.list_providers().to_dict()
        operations = set(features["result"]["operations"])
        assert operations >= set(STABLE_OPERATIONS)
        provider_list = providers["result"].get("providers") or providers["result"]
        assert provider_list, "Logic API inventory must declare providers"
        assert providers.get("status") in {{
            "declarative",
            "succeeded",
            "partial",
            "unsupported",
            "unavailable",
        }}

        plugins = list_installer_plugins()
        assert len(plugins) == len(PLUGIN_MODULE_PATHS)
        for family, module_path in PLUGIN_MODULE_PATHS.items():
            rel = "/".join(module_path.split(".")[1:]) + ".py"
            path = target / "ipfs_datasets_py" / Path(rel)
            assert path.is_file(), f"installed plugin missing: {{family}} -> {{path}}"

        # Optional external tools may be unavailable; inventory must not crash.
        unavailable_ok = True
        for name in ("list_providers", "list_logic_families", "list_features"):
            response = getattr(api, name)()
            payload = response.to_dict() if hasattr(response, "to_dict") else dict(response)
            assert payload.get("status") != "error"

        print(json.dumps({{
            "ok": True,
            "operations": sorted(STABLE_OPERATIONS),
            "provider_count": len(provider_list),
            "plugin_count": len(plugins),
            "module_file": str(module_file),
            "forbidden_events": forbidden_events,
            "unavailable_ok": unavailable_ok,
        }}))
        """
    ).format(
        target=str(target_dir),
        repo=str(REPO_ROOT),
        datasets=str(DATASETS_ROOT),
    )

    probe_env = _offline_env()
    completed = _bounded_run(
        [sys.executable, "-c", probe],
        timeout=IMPORT_TIMEOUT_SECONDS,
        env=probe_env,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, (
        "clean wheel import/inventory failed\n"
        f"stdout={completed.stdout}\nstderr={completed.stderr}"
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert payload["provider_count"] >= 1
    assert payload["plugin_count"] >= 1
    assert payload["forbidden_events"] == []
    assert str(target_dir.resolve()) in payload["module_file"]
    assert str(DATASETS_ROOT.resolve()) not in payload["module_file"]


def test_install_authorization_remains_fail_closed_during_distribution_checks() -> None:
    sys_path_inserted = False
    datasets_root = str(DATASETS_ROOT)
    if datasets_root not in sys.path:
        sys.path.insert(0, datasets_root)
        sys_path_inserted = True
    try:
        from ipfs_datasets_py.logic.backends.toolchains import (
            ToolchainError,
            authorize_provider_install,
            install_is_forbidden_on_import,
            registry_side_effect_free_on_import,
        )

        assert registry_side_effect_free_on_import() is True
        assert install_is_forbidden_on_import() is True
        with pytest.raises(ToolchainError):
            authorize_provider_install("apalache", yes=False, explicit_call=True)
        with pytest.raises(ToolchainError):
            authorize_provider_install(
                "apalache",
                yes=True,
                explicit_call=True,
                import_context=True,
            )
        with pytest.raises(ToolchainError):
            authorize_provider_install(
                "apalache",
                yes=True,
                explicit_call=True,
                capability_discovery=True,
            )
    finally:
        if sys_path_inserted:
            try:
                sys.path.remove(datasets_root)
            except ValueError:
                pass
