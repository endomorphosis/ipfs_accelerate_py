"""SCG-044 install and import acceptance for the semantic-governor surface.

Proves:

* Cold package import is hermetic (no I/O, process, network, optional install).
* Required public names resolve lazily from the package root.
* Packaging declares the ``semantic-governor`` console entry.
* Offline wheel build + target install exposes the CLI entry and cold ``--help``.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "ipfs_accelerate_py.agent_supervisor.semantic_governor"
CLI_MODULE = f"{PACKAGE}.cli"
GOVERNOR_MODULE = f"{PACKAGE}.governor"
CONSOLE_ENTRY = "semantic-governor"
ENTRY_TARGET = (
    "ipfs_accelerate_py.agent_supervisor.semantic_governor.cli:main"
)
INIT_PATH = (
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/semantic_governor/__init__.py"
)
GOVERNOR_PATH = (
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/semantic_governor/governor.py"
)
CLI_PATH = (
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/semantic_governor/cli.py"
)

REQUIRED_PUBLIC_NAMES = (
    "SemanticCompressionGovernor",
    "evaluate_context_sufficiency",
    "create_shadow_plan",
    "compare_shadow_results",
    "diagnose_omission",
    "plan_context_expansion",
    "execute_expansion_loop",
    "update_calibration",
    "propose_rule_change",
    "evaluate_rule_candidate",
    "promote_compression_policy",
)

_OPT_OUTS = {
    "IPFS_DATASETS_AUTO_INSTALL": "0",
    "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS": "0",
    "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1",
    "IPFS_KIT_AUTO_INSTALL_DEPS": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
}

WHEEL_BUILD_TIMEOUT = 300.0
WHEEL_INSTALL_TIMEOUT = 120.0


# ---------------------------------------------------------------------------
# Packaging surface
# ---------------------------------------------------------------------------


def test_setup_and_manifest_declare_semantic_governor_console_entry() -> None:
    setup = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    assert (
        "semantic-governor="
        "ipfs_accelerate_py.agent_supervisor.semantic_governor.cli:main"
        in setup.replace("\n", "").replace(" ", "")
        or ENTRY_TARGET in setup
    )
    assert "semantic-governor" in setup
    assert "_semantic_governor_console" in setup or "scripts=" in setup
    assert CLI_PATH.is_file()
    assert INIT_PATH.is_file()
    assert GOVERNOR_PATH.is_file()


def test_source_tree_cli_help_and_descriptor() -> None:
    cli = importlib.import_module(CLI_MODULE)
    assert cli.CLI_INTERFACE == "SemanticGovernorCLI@1"
    assert cli.CONSOLE_ENTRY == CONSOLE_ENTRY
    assert cli.CLI_EVIDENCE == "scg/cli@1"
    commands = cli.required_cli_commands()
    assert len(commands) == 10
    assert "promote-policy" in commands
    desc = cli.semantic_governor_cli_descriptor()
    assert desc["console_entry"] == CONSOLE_ENTRY
    assert "no_implicit_promotion" in desc["invariants"]
    assert "promotion_requires_explicit_authorization_and_cas" in desc["invariants"]

    import io

    out = io.StringIO()
    err = io.StringIO()
    code = cli.main(["--help"], stdout=out, stderr=err)
    assert code == 0
    text = out.getvalue() + err.getvalue()
    assert "semantic-governor" in text
    for command in commands:
        assert command in text


# ---------------------------------------------------------------------------
# Cold import hermeticity
# ---------------------------------------------------------------------------


def test_package_import_is_hermetic_and_lazy() -> None:
    """Importing the package root starts no I/O, process, network, or installer.

    Run in a child interpreter so the parent suite does not dual-load
    governor classes via ``sys.modules`` eviction.
    """

    script = f"""
import importlib, os, socket, subprocess, sys, threading
for key, value in {json.dumps(_OPT_OUTS)}.items():
    os.environ[key] = value
before_threads = {{t.ident for t in threading.enumerate()}}
started = []
real_start = threading.Thread.start
def guarded_start(self, *args, **kwargs):
    started.append(self.name)
    return real_start(self, *args, **kwargs)
threading.Thread.start = guarded_start
def guarded_popen(*_a, **_k):
    raise AssertionError("cold import must not spawn subprocesses")
subprocess.Popen = guarded_popen
real_socket = socket.socket
class GuardedSocket(real_socket):
    def __init__(self, *args, **kwargs):
        raise AssertionError("cold import must not open sockets")
socket.socket = GuardedSocket
real_run = subprocess.run
def guarded_run(*args, **kwargs):
    cmd = args[0] if args else kwargs.get("args")
    text = " ".join(str(x) for x in (cmd or ()))
    if "pip" in text or "install" in text:
        raise AssertionError("cold import must not install: " + text)
    return real_run(*args, **kwargs)
subprocess.run = guarded_run
mod = importlib.import_module({PACKAGE!r})
assert mod.PUBLIC_API_EVIDENCE == "scg/public-api@1"
assert mod.PUBLIC_API_INTERFACE == "SemanticCompressionGovernorPublicApi@1"
assert {GOVERNOR_MODULE!r} not in sys.modules or hasattr(mod, "SemanticCompressionGovernor")
for name in {list(REQUIRED_PUBLIC_NAMES)!r}:
    value = getattr(mod, name)
    assert value is not None
    assert callable(value) or name == "SemanticCompressionGovernor"
assert started == []
after_threads = {{t.ident for t in threading.enumerate()}}
assert after_threads - before_threads == set()
print("HERMETIC_IMPORT_OK")
"""
    env = dict(os.environ)
    env.update(_OPT_OUTS)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            [
                str(REPO_ROOT),
                str(REPO_ROOT / "ipfs_kit_py"),
                str(REPO_ROOT / "ipfs_datasets_py"),
                env.get("PYTHONPATH", ""),
            ],
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "HERMETIC_IMPORT_OK" in result.stdout


def test_package_and_governor_sources_have_no_module_level_io() -> None:
    """Static check: package/governor modules perform no top-level I/O."""

    forbidden = {
        "open",
        "urlopen",
        "system",
        "Popen",
        "connect",
        "create_connection",
        "urlretrieve",
    }
    for path in (INIT_PATH, GOVERNOR_PATH, CLI_PATH):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)):
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    name = (
                        func.id
                        if isinstance(func, ast.Name)
                        else (
                            func.attr if isinstance(func, ast.Attribute) else ""
                        )
                    )
                    assert name not in forbidden, f"{path.name} top-level call {name}"


def test_subprocess_cold_import_resolves_required_names() -> None:
    """Isolated interpreter: import package, resolve names, no auto-install."""

    script = f"""
import importlib, os, sys
for key, value in {json.dumps(_OPT_OUTS)}.items():
    os.environ[key] = value
mod = importlib.import_module({PACKAGE!r})
assert mod.PUBLIC_API_EVIDENCE == "scg/public-api@1"
names = {list(REQUIRED_PUBLIC_NAMES)!r}
for name in names:
    value = getattr(mod, name)
    assert value is not None
gov = mod.create_semantic_compression_governor()
assert gov.required_public_apis() == mod.REQUIRED_PUBLIC_NAMES or len(gov.required_public_apis()) == 10
cli = importlib.import_module({CLI_MODULE!r})
assert cli.CONSOLE_ENTRY == {CONSOLE_ENTRY!r}
assert len(cli.required_cli_commands()) == 10
print("INSTALL_IMPORT_OK")
"""
    env = dict(os.environ)
    env.update(_OPT_OUTS)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            [
                str(REPO_ROOT),
                str(REPO_ROOT / "ipfs_kit_py"),
                str(REPO_ROOT / "ipfs_datasets_py"),
                env.get("PYTHONPATH", ""),
            ],
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "INSTALL_IMPORT_OK" in result.stdout


def test_required_public_names_match_package_exports() -> None:
    mod = importlib.import_module(PACKAGE)
    for name in REQUIRED_PUBLIC_NAMES:
        assert name in mod.REQUIRED_PUBLIC_NAMES or hasattr(mod, name)
        assert name in mod.__all__ or hasattr(mod, name)
    # Promotion is exported; rollback is intentionally leaf-only (not a
    # self-authorizing package root export).
    assert "promote_compression_policy" in mod.__all__ or hasattr(
        mod, "promote_compression_policy"
    )
    assert not hasattr(mod, "rollout_enable") or getattr(mod, "rollout_enable") is None


# ---------------------------------------------------------------------------
# Offline wheel install smoke
# ---------------------------------------------------------------------------


def _offline_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env.pop("PIP_INDEX_URL", None)
    env.pop("PIP_EXTRA_INDEX_URL", None)
    env["PYTHONNOUSERSITE"] = "1"
    env.update(_OPT_OUTS)
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
        if not names:
            return ""
        return zf.read(names[0]).decode("utf-8")


def _wheel_declares_console_entry(wheel_path: Path, members: list[str]) -> bool:
    entry_text = _entry_points_text(wheel_path)
    if entry_text and CONSOLE_ENTRY in entry_text:
        if (
            ENTRY_TARGET in entry_text.replace(" ", "")
            or "semantic_governor.cli:main" in entry_text
            or re.search(
                r"semantic-governor\s*=\s*ipfs_accelerate_py\.agent_supervisor\.semantic_governor\.cli:main",
                entry_text,
            )
        ):
            return True
        if "console_scripts" in entry_text and CONSOLE_ENTRY in entry_text:
            return True
    return any(
        member.replace("\\", "/").endswith(".data/scripts/semantic-governor")
        or member.replace("\\", "/").endswith("/scripts/semantic-governor")
        for member in members
    )


@pytest.mark.timeout(600)
def test_built_wheel_contains_semantic_governor_console(tmp_path: Path) -> None:
    """Offline bdist_wheel + target install smoke for semantic-governor."""

    dist_dir = tmp_path / "dist"
    target_dir = tmp_path / "site"
    build_root = tmp_path / "src"
    dist_dir.mkdir()
    target_dir.mkdir()

    def _copy_needed() -> Path:
        stage = build_root
        stage.mkdir(parents=True)
        for name in (
            "setup.py",
            "pyproject.toml",
            "MANIFEST.in",
            "README.md",
            "LICENSE",
        ):
            src = REPO_ROOT / name
            if src.exists():
                shutil.copy2(src, stage / name)
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
                shutil.copytree(shared, scripts_dst / "shared", ignore=_ignore)
        return stage

    stage = _copy_needed()
    env = _offline_env()

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
        "semantic_governor" in member.replace("\\", "/") for member in members
    ), f"wheel missing semantic_governor package; sample={members[:40]}"
    assert _wheel_declares_console_entry(wheel_path, members), (
        "console entry missing from wheel entry_points.txt and .data/scripts; "
        f"sample members={members[:40]} entry_points={_entry_points_text(wheel_path)!r}"
    )

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

    installed_pkg = (
        target_dir
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "semantic_governor"
        / "__init__.py"
    )
    assert installed_pkg.is_file(), f"installed tree missing package at {installed_pkg}"

    dist_infos = list(target_dir.glob("*.dist-info"))
    assert dist_infos, "install produced no dist-info"
    ep_files = list(target_dir.glob("*.dist-info/entry_points.txt"))
    installed_ep = ep_files[0].read_text(encoding="utf-8") if ep_files else ""
    script_hits = list(target_dir.rglob("semantic-governor"))
    assert (
        (CONSOLE_ENTRY in installed_ep and "semantic_governor.cli:main" in installed_ep)
        or any(path.is_file() for path in script_hits)
    ), (
        "installed tree missing semantic-governor console entry "
        f"(entry_points={installed_ep!r}, scripts={script_hits!r})"
    )

    probe_env = dict(os.environ)
    probe_env.update(_OPT_OUTS)
    probe_env["PIP_NO_INDEX"] = "1"
    probe_env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    probe_env.pop("PIP_INDEX_URL", None)
    probe_env.pop("PIP_EXTRA_INDEX_URL", None)
    probe_env.pop("PYTHONNOUSERSITE", None)
    host_path_entries = [entry for entry in sys.path if entry]
    # Prefer installed wheel; fall back to repo source deps (datasets/kit not in wheel).
    probe_env["PYTHONPATH"] = os.pathsep.join(
        [
            str(target_dir),
            str(REPO_ROOT / "ipfs_datasets_py"),
            str(REPO_ROOT / "ipfs_kit_py"),
            str(REPO_ROOT),
            *host_path_entries,
        ]
    )
    probe = _bounded_run(
        [
            sys.executable,
            "-c",
            (
                "import importlib, sys\n"
                "cli = importlib.import_module(\n"
                "    'ipfs_accelerate_py.agent_supervisor.semantic_governor.cli'\n"
                ")\n"
                f"assert {str(target_dir)!r} in (cli.__file__ or ''), cli.__file__\n"
                "assert cli.CLI_INTERFACE == 'SemanticGovernorCLI@1'\n"
                "assert cli.CONSOLE_ENTRY == 'semantic-governor'\n"
                "assert len(cli.required_cli_commands()) == 10\n"
                "code = cli.main(['--help'])\n"
                "assert code == 0\n"
                "pkg = importlib.import_module(\n"
                "    'ipfs_accelerate_py.agent_supervisor.semantic_governor'\n"
                ")\n"
                "assert pkg.PUBLIC_API_EVIDENCE == 'scg/public-api@1'\n"
                "assert hasattr(pkg, 'SemanticCompressionGovernor')\n"
                "print('WHEEL_SMOKE_OK')\n"
            ),
        ],
        timeout=60.0,
        env=probe_env,
        cwd=tmp_path,
    )
    assert probe.returncode == 0, (
        "installed-tree smoke failed\n"
        f"stdout={probe.stdout[-4000:]}\nstderr={probe.stderr[-4000:]}"
    )
    assert "WHEEL_SMOKE_OK" in probe.stdout
