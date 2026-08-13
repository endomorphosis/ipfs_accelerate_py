"""SCH-013 built-wheel smoke: console entry and packaged interface schema.

Builds a local wheel offline, installs with ``--no-deps --no-index --target``,
and proves:

* the ``semantic-state`` console script entry is present in dist-info;
* the closed Profile A interface schema JSON is packaged;
* cold ``--help`` and schema load work from the installed tree without
  network, pip install, daemon, or environment mutation.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import site
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
WHEEL_BUILD_TIMEOUT = 300.0
WHEEL_INSTALL_TIMEOUT = 120.0
ENTRY_TARGET = "ipfs_accelerate_py.agent_supervisor.semantic_state.cli:main"
SCHEMA_SUFFIX = (
    "ipfs_accelerate_py/agent_supervisor/semantic_state/schemas/"
    "semantic-state-harness.interface.json"
)
CONSOLE_ENTRY = "semantic-state"


def _offline_env() -> dict[str, str]:
    env = dict(os.environ)
    # Fail closed on index access; keep interpreter tooling available.
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env.pop("PIP_INDEX_URL", None)
    env.pop("PIP_EXTRA_INDEX_URL", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _bounded_run(
    argv: list[str],
    *,
    timeout: float,
    env: dict[str, str],
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _wheel_members(wheel_path: Path) -> list[str]:
    with zipfile.ZipFile(wheel_path) as zf:
        return sorted(zf.namelist())


def _entry_points_text(wheel_path: Path) -> str:
    with zipfile.ZipFile(wheel_path) as zf:
        names = [
            name
            for name in zf.namelist()
            if name.endswith(".dist-info/entry_points.txt")
        ]
        assert names, "wheel missing entry_points.txt"
        return zf.read(names[0]).decode("utf-8")


def _metadata_declares_console_entry(text: str) -> bool:
    # setuptools writes:
    # [console_scripts]
    # semantic-state = module:main
    if CONSOLE_ENTRY not in text:
        return False
    compact = text.replace(" ", "")
    if ENTRY_TARGET not in compact and "semantic_state.cli:main" not in text:
        return False
    return "console_scripts" in text and CONSOLE_ENTRY in text


def _host_dependency_site_dirs() -> list[str]:
    """Directories that already hold pure runtime deps for the probe.

    The wheel is installed with ``--no-deps`` so pure transitive deps such as
    ``anyio`` (imported by packaged ``mcp_server.mcplusplus``) are absent from
    the target tree. Validation often uses a private HOME with
    ``PYTHONNOUSERSITE``, which hides user site-packages. Forward directories
    already resolvable in this pytest process so cold import completes offline
    without pip or an index. The install target is always prepended separately
    and wins package identity for ``ipfs_accelerate_py``.
    """

    dirs: list[str] = []
    seen: set[str] = set()
    repo_root = str(REPO_ROOT.resolve())

    def _add(path: Path | str | None) -> None:
        if path is None:
            return
        try:
            resolved = str(Path(path).resolve())
        except OSError:
            return
        if resolved in seen:
            return
        if resolved == repo_root:
            # Never prefer the live source tree over the installed wheel.
            return
        if not Path(resolved).is_dir():
            return
        seen.add(resolved)
        dirs.append(resolved)

    # Pure deps observed on the parent import path (site-packages parents).
    for name in (
        "anyio",
        "sniffio",
        "idna",
        "multiformats",
        "jsonschema",
        "typing_extensions",
        "attr",
        "attrs",
    ):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        file_name = getattr(mod, "__file__", None)
        if not file_name:
            # Namespace package: use first path entry.
            paths = list(getattr(mod, "__path__", []) or [])
            for item in paths:
                _add(Path(item).resolve().parent)
            continue
        # site-packages/<pkg>/__init__.py -> site-packages
        _add(Path(file_name).resolve().parents[1])

    try:
        for item in site.getsitepackages():
            _add(item)
    except Exception:
        pass
    try:
        _add(site.getusersitepackages())
    except Exception:
        pass

    # Forward every existing parent sys.path entry except the repo root so
    # private-HOME validation still sees the pure deps the parent already has.
    for entry in sys.path:
        if not entry:
            continue
        _add(entry)

    return dirs


def test_pyproject_and_setup_declare_semantic_state_console_entry() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    setup = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert (
        'semantic-state = "ipfs_accelerate_py.agent_supervisor.semantic_state.cli:main"'
        in pyproject
        or "semantic-state=" in pyproject
    )
    assert (
        "semantic-state=ipfs_accelerate_py.agent_supervisor.semantic_state.cli:main"
        in setup
    )
    assert "semantic_state/schemas" in manifest or "schemas/*.json" in manifest
    assert (
        'ipfs_accelerate_py.agent_supervisor.semantic_state"' in pyproject
        or "schemas/*.json" in pyproject
        or "schemas/*.json" in setup
    )


def test_source_tree_schema_loadable_via_cli_helper() -> None:
    cli = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.cli"
    )
    text = cli.load_interface_schema_text()
    body = json.loads(text)
    assert body["name"] == "semantic-state-harness"
    assert body["namespace"] == "ipfs-accelerate.agent-supervisor"


@pytest.mark.timeout(600)
def test_built_wheel_contains_console_entry_and_schema(tmp_path: Path) -> None:
    """Offline bdist_wheel + target install smoke test."""

    dist_dir = tmp_path / "dist"
    target_dir = tmp_path / "site"
    build_root = tmp_path / "src"
    dist_dir.mkdir()
    target_dir.mkdir()

    # Stage a minimal source tree sufficient for the semantic-state wheel path.
    # Copying the entire monorepo is slow; stage only what setuptools needs plus
    # the package tree and its schema.
    def _copy_needed() -> Path:
        stage = build_root
        stage.mkdir(parents=True)
        for name in ("setup.py", "pyproject.toml", "MANIFEST.in", "README.md", "LICENSE"):
            src = REPO_ROOT / name
            if src.exists():
                shutil.copy2(src, stage / name)
        # requirements referenced by setup.py
        for name in ("requirements.txt", "requirements-proof-reuse.txt"):
            src = REPO_ROOT / name
            if src.exists():
                shutil.copy2(src, stage / name)
            else:
                (stage / name).write_text("", encoding="utf-8")

        src_pkg = REPO_ROOT / "ipfs_accelerate_py"
        dst_pkg = stage / "ipfs_accelerate_py"

        def _ignore(directory: str, names: list[str]) -> set[str]:
            ignored: set[str] = set()
            for name in names:
                if name in {
                    "__pycache__",
                    ".git",
                    ".mypy_cache",
                    ".pytest_cache",
                    "node_modules",
                }:
                    ignored.add(name)
                elif name.endswith((".pyc", ".pyo", ".so")):
                    ignored.add(name)
            return ignored

        shutil.copytree(src_pkg, dst_pkg, ignore=_ignore)

        # scripts package is listed in find_packages
        scripts_src = REPO_ROOT / "scripts"
        if scripts_src.is_dir():
            scripts_dst = stage / "scripts"
            scripts_dst.mkdir(exist_ok=True)
            init = scripts_src / "__init__.py"
            if init.exists():
                shutil.copy2(init, scripts_dst / "__init__.py")
            else:
                (scripts_dst / "__init__.py").write_text("", encoding="utf-8")
            shared = scripts_src / "shared"
            if shared.is_dir():
                shutil.copytree(
                    shared,
                    scripts_dst / "shared",
                    ignore=_ignore,
                )
        return stage

    stage = _copy_needed()
    env = _offline_env()
    env_before = dict(os.environ)

    build = _bounded_run(
        [
            sys.executable,
            "setup.py",
            "bdist_wheel",
            "--dist-dir",
            str(dist_dir),
        ],
        timeout=WHEEL_BUILD_TIMEOUT,
        env=env,
        cwd=stage,
    )
    assert build.returncode == 0, (
        "wheel build failed offline\n"
        f"stdout={build.stdout[-4000:]}\nstderr={build.stderr[-4000:]}"
    )

    wheels = sorted(dist_dir.glob("ipfs_accelerate_py-*.whl"))
    assert wheels, f"no wheel produced in {dist_dir}: {list(dist_dir.iterdir())}"
    wheel_path = wheels[-1]

    members = _wheel_members(wheel_path)
    assert any(
        member.endswith(SCHEMA_SUFFIX)
        or member.replace("\\", "/").endswith(SCHEMA_SUFFIX)
        for member in members
    ), f"wheel missing interface schema; sample members={members[:40]}"

    entry_text = _entry_points_text(wheel_path)
    assert _metadata_declares_console_entry(entry_text), (
        f"console entry missing from wheel entry_points.txt:\n{entry_text}"
    )
    assert re.search(
        r"semantic-state\s*=\s*ipfs_accelerate_py\.agent_supervisor\.semantic_state\.cli:main",
        entry_text,
    ), entry_text

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
        timeout=WHEEL_INSTALL_TIMEOUT,
        env=env,
    )
    assert install.returncode == 0, (
        "wheel install failed offline\n"
        f"stdout={install.stdout[-4000:]}\nstderr={install.stderr[-4000:]}"
    )

    # Installed schema path.
    schema_path = (
        target_dir
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "semantic_state"
        / "schemas"
        / "semantic-state-harness.interface.json"
    )
    assert schema_path.is_file(), f"installed tree missing schema at {schema_path}"
    schema_body = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema_body["name"] == "semantic-state-harness"

    # Dist-info entry points on the installed target.
    dist_infos = list(target_dir.glob("*.dist-info"))
    assert dist_infos, "install produced no dist-info"
    ep_files = list(target_dir.glob("*.dist-info/entry_points.txt"))
    assert ep_files, "installed dist-info missing entry_points.txt"
    installed_ep = ep_files[0].read_text(encoding="utf-8")
    assert CONSOLE_ENTRY in installed_ep
    assert "semantic_state.cli:main" in installed_ep

    # Cold import + --help from the installed tree. Prefer the target install on
    # PYTHONPATH first, then pure host runtime deps already present in this
    # process (anyio, etc.) so --no-deps + private HOME still import offline.
    # Never enable an index or run pip here.
    host_sites = _host_dependency_site_dirs()
    assert host_sites, (
        "host dependency roots empty; parent interpreter cannot supply pure "
        "runtime deps (anyio) for offline --no-deps wheel smoke"
    )
    pythonpath_parts = [str(target_dir.resolve()), *host_sites]
    probe_env = dict(os.environ)
    probe_env["PIP_NO_INDEX"] = "1"
    probe_env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    probe_env.pop("PIP_INDEX_URL", None)
    probe_env.pop("PIP_EXTRA_INDEX_URL", None)
    # Keep user site disabled so only explicit PYTHONPATH host dirs are used.
    probe_env["PYTHONNOUSERSITE"] = "1"
    probe_env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    target_dir_s = str(target_dir.resolve())
    probe = _bounded_run(
        [
            sys.executable,
            "-c",
            (
                "import importlib, json, sys\n"
                # Target install must win package identity.
                f"assert sys.path[0] == {target_dir_s!r} or "
                f"{target_dir_s!r} in sys.path\n"
                "cli = importlib.import_module(\n"
                "    'ipfs_accelerate_py.agent_supervisor.semantic_state.cli'\n"
                ")\n"
                # Prove the installed wheel path is the one loaded.
                f"assert {target_dir_s!r} in (cli.__file__ or '')\n"
                "assert cli.CLI_INTERFACE == 'SemanticStateCLI@1'\n"
                "text = cli.load_interface_schema_text()\n"
                "body = json.loads(text)\n"
                "assert body['name'] == 'semantic-state-harness'\n"
                "code = cli.main(['--help'])\n"
                "assert code == 0\n"
                "code2 = cli.main(['interface-schema'])\n"
                "assert code2 == 0\n"
                "print('WHEEL_SMOKE_OK')\n"
            ),
        ],
        timeout=60.0,
        env=probe_env,
        cwd=tmp_path,
    )
    assert probe.returncode == 0, (
        "installed-tree smoke failed\n"
        f"stdout={probe.stdout[-4000:]}\nstderr={probe.stderr[-4000:]}\n"
        f"host_sites={host_sites!r}\n"
        f"PYTHONPATH={probe_env.get('PYTHONPATH')!r}"
    )
    assert "WHEEL_SMOKE_OK" in probe.stdout

    # Outer environment unchanged by the smoke path.
    assert dict(os.environ) == env_before
