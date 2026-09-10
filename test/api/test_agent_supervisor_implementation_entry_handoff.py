"""Regression tests for early implementation-supervisor authority redemption."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    IMPLEMENTATION_DAEMON_MODULE_SENTINEL,
    ORDINARY_IMPLEMENTATION_DAEMON_BOOTSTRAP,
)

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "scripts/ops/agent_supervisor/implementation_supervisor_entry.py"
PROCESS_SECURITY_MODULE = "ipfs_accelerate_py.agent_supervisor.runtime.process_security"
NATIVE_PRELOAD_MODULE = (
    "ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner"
)
IMPLEMENTATION_MODULE = "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor"


def test_ordinary_daemon_bootstrap_hardens_before_cold_imports() -> None:
    """Keep daemon imports outside the authority-redemption critical path."""

    tree = ast.parse(ORDINARY_IMPLEMENTATION_DAEMON_BOOTSTRAP)
    positions = {id(node): index for index, node in enumerate(tree.body)}
    sentinel_guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.BoolOp)
        and isinstance(node.value.op, ast.Or)
    )
    sentinel_pop = next(
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "pop"
    )
    process_security_import = next(
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == PROCESS_SECURITY_MODULE
    )
    harden_call = next(
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "harden_state_authority_process"
    )
    native_preload_import = next(
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == NATIVE_PRELOAD_MODULE
    )
    native_preload_call = next(
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "preload_sealed_native_dependency_from_environment"
    )
    daemon_import = next(
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == IMPLEMENTATION_DAEMON_MODULE_SENTINEL
    )
    daemon_exit = next(node for node in tree.body if isinstance(node, ast.Raise))

    assert [
        positions[id(node)]
        for node in (
            sentinel_guard,
            sentinel_pop,
            process_security_import,
            harden_call,
            native_preload_import,
            native_preload_call,
            daemon_import,
            daemon_exit,
        )
    ] == sorted(
        positions[id(node)]
        for node in (
            sentinel_guard,
            sentinel_pop,
            process_security_import,
            harden_call,
            native_preload_import,
            native_preload_call,
            daemon_import,
            daemon_exit,
        )
    )


def test_ordinary_daemon_bootstrap_rejects_wrong_module_sentinel() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-P",
            "-c",
            ORDINARY_IMPLEMENTATION_DAEMON_BOOTSTRAP,
            "wrong.daemon.module",
        ],
        cwd=ROOT,
        env={"PATH": "/usr/bin:/bin"},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )

    assert completed.returncode == 78


def test_entry_hardens_before_importing_implementation_supervisor() -> None:
    """Keep cold implementation imports outside the handoff critical path."""

    tree = ast.parse(ENTRY.read_text(encoding="utf-8"), filename=str(ENTRY))
    process_security_import = next(
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == PROCESS_SECURITY_MODULE
    )
    main_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    implementation_import = next(
        node
        for node in main_function.body
        if isinstance(node, ast.ImportFrom) and node.module == IMPLEMENTATION_MODULE
    )
    native_preload_import = next(
        node
        for node in main_function.body
        if isinstance(node, ast.ImportFrom) and node.module == NATIVE_PRELOAD_MODULE
    )
    harden_call = next(
        node
        for node in main_function.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "harden_state_authority_process"
    )
    native_preload_call = next(
        node
        for node in main_function.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "preload_sealed_native_dependency_from_environment"
    )

    assert process_security_import.lineno < harden_call.lineno
    assert harden_call.lineno < native_preload_import.lineno
    assert native_preload_import.lineno < native_preload_call.lineno
    assert native_preload_call.lineno < implementation_import.lineno
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module == IMPLEMENTATION_MODULE
        for node in tree.body
    )
    assert not any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "harden_state_authority_process"
        for node in tree.body
    )


def test_entry_import_is_inert_until_main_is_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Library imports must not redeem process authority as a side effect."""

    import ipfs_accelerate_py.agent_supervisor.runtime.process_security as security

    calls: list[str] = []
    monkeypatch.setattr(
        security,
        "harden_state_authority_process",
        lambda: calls.append("harden"),
    )
    module_name = "_aseh_test_implementation_supervisor_entry"
    spec = importlib.util.spec_from_file_location(module_name, ENTRY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    assert calls == []


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"),
    reason="sealed Linux memfd handoff is required",
)
def test_entry_redeems_handoff_before_delayed_implementation_import() -> None:
    """A slow implementation import must not consume the redemption deadline."""

    child_code = "\n".join(
        (
            "import importlib.abc, importlib.util, json, os, runpy, sys, time",
            f"target = {IMPLEMENTATION_MODULE!r}",
            "class DelayedLoader(importlib.abc.Loader):",
            "    def create_module(self, spec):",
            "        return None",
            "    def exec_module(self, module):",
            "        time.sleep(2.0)",
            "        module.main = lambda: 0",
            "class DelayedFinder(importlib.abc.MetaPathFinder):",
            "    def find_spec(self, fullname, path=None, target=None):",
            "        if fullname == globals()['target']:",
            "            return importlib.util.spec_from_loader(fullname, DelayedLoader())",
            "        return None",
            "sys.meta_path.insert(0, DelayedFinder())",
            "exit_code = None",
            "try:",
            f"    runpy.run_path({str(ENTRY)!r}, run_name='__main__')",
            "except SystemExit as exc:",
            "    exit_code = exc.code",
            (
                "from ipfs_accelerate_py.agent_supervisor.runtime.process_security "
                "import state_authority_pass_fds"
            ),
            "descriptors = state_authority_pass_fds(os.environ)",
            "print(json.dumps({'exit_code': exit_code, 'descriptor_count': len(descriptors)}))",
            "for descriptor in descriptors:",
            "    os.close(descriptor)",
        )
    )
    parent_code = "\n".join(
        (
            "import fcntl, json, os, subprocess, sys",
            (
                "from ipfs_accelerate_py.agent_supervisor.runtime.process_security "
                "import STATE_AUTHORITY_PARENT_LOSS_TERMINATE, "
                "prepare_state_authority_child_handoff"
            ),
            (
                "flags = int(getattr(os, 'MFD_CLOEXEC', 1)) | "
                "int(getattr(os, 'MFD_ALLOW_SEALING', 2))"
            ),
            "descriptor = os.memfd_create('implementation-entry-handoff', flags=flags)",
            "os.write(descriptor, b'x' * 64)",
            (
                "seals = int(getattr(fcntl, 'F_SEAL_SEAL', 1)) | "
                "int(getattr(fcntl, 'F_SEAL_SHRINK', 2)) | "
                "int(getattr(fcntl, 'F_SEAL_GROW', 4)) | "
                "int(getattr(fcntl, 'F_SEAL_WRITE', 8))"
            ),
            ("fcntl.fcntl(descriptor, int(getattr(fcntl, 'F_ADD_SEALS', 1033)), seals)"),
            (
                "environment = {'PATH': '/usr/bin:/bin', "
                "'IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET': "
                "'/tmp/implementation-entry-handoff.sock', "
                "'IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD': "
                "str(descriptor)}"
            ),
            (
                "handoff = prepare_state_authority_child_handoff("
                "environment, parent_loss_policy="
                "STATE_AUTHORITY_PARENT_LOSS_TERMINATE)"
            ),
            f"child_code = {child_code!r}",
            "process = None",
            "try:",
            (
                f"    process = subprocess.Popen([sys.executable, '-c', child_code], "
                f"cwd={str(ROOT)!r}, env=environment, stdin=subprocess.DEVNULL, "
                "stdout=subprocess.PIPE, stderr=subprocess.PIPE)"
            ),
            "    handoff.deliver(process, timeout_seconds=1.0)",
            "    stdout, stderr = process.communicate(timeout=5.0)",
            ("    if process.returncode != 0: raise RuntimeError(stderr.decode(errors='replace'))"),
            "    print(stdout.decode('utf-8').strip())",
            "finally:",
            "    handoff.close()",
            "    if process is not None and process.poll() is None:",
            "        process.kill()",
            "        process.wait(timeout=2.0)",
            "    os.close(descriptor)",
        )
    )

    completed = subprocess.run(
        [sys.executable, "-c", parent_code],
        cwd=ROOT,
        env={"PATH": "/usr/bin:/bin"},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "descriptor_count": 1,
        "exit_code": 0,
    }


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"),
    reason="sealed Linux memfd handoff is required",
)
def test_ordinary_daemon_bootstrap_redeems_before_delayed_daemon_import(
    tmp_path: Path,
) -> None:
    """The exact ordinary command must redeem before a slow daemon import."""

    import_hook_marker = tmp_path / "daemon-import-started"
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        "\n".join(
            (
                "import importlib.util, sys, time",
                f"_TARGET = {IMPLEMENTATION_DAEMON_MODULE_SENTINEL!r}",
                f"_MARKER = {str(import_hook_marker)!r}",
                "class _DelayedDaemonLoader:",
                "    def create_module(self, spec):",
                "        return None",
                "    def exec_module(self, module):",
                "        with open(_MARKER, 'w', encoding='utf-8') as stream:",
                "            stream.write('started')",
                "        time.sleep(2.0)",
                "        def main(*, native_dependency_preloaded=False):",
                "            assert native_dependency_preloaded is True",
                "            print(repr(sys.argv), flush=True)",
                "            return 0",
                "        module.main = main",
                "class _DelayedDaemonFinder:",
                "    def find_spec(self, fullname, path=None, target=None):",
                "        if fullname == _TARGET:",
                (
                    "            return importlib.util.spec_from_loader("
                    "fullname, _DelayedDaemonLoader())"
                ),
                "        return None",
                "sys.meta_path.insert(0, _DelayedDaemonFinder())",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    parent_code = "\n".join(
        (
            "import fcntl, json, os, subprocess, sys",
            (
                "from ipfs_accelerate_py.agent_supervisor.runtime.process_security "
                "import STATE_AUTHORITY_PARENT_LOSS_TERMINATE, "
                "prepare_state_authority_child_handoff"
            ),
            (
                "flags = int(getattr(os, 'MFD_CLOEXEC', 1)) | "
                "int(getattr(os, 'MFD_ALLOW_SEALING', 2))"
            ),
            "descriptor = os.memfd_create('ordinary-daemon-handoff', flags=flags)",
            "os.write(descriptor, b'x' * 64)",
            (
                "seals = int(getattr(fcntl, 'F_SEAL_SEAL', 1)) | "
                "int(getattr(fcntl, 'F_SEAL_SHRINK', 2)) | "
                "int(getattr(fcntl, 'F_SEAL_GROW', 4)) | "
                "int(getattr(fcntl, 'F_SEAL_WRITE', 8))"
            ),
            (
                "fcntl.fcntl(descriptor, "
                "int(getattr(fcntl, 'F_ADD_SEALS', 1033)), seals)"
            ),
            (
                "environment = {'PATH': '/usr/bin:/bin', "
                f"'PYTHONPATH': {str(tmp_path)!r} + os.pathsep + {str(ROOT)!r}, "
                "'IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET': "
                "'/tmp/ordinary-daemon-handoff.sock', "
                "'IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD': "
                "str(descriptor)}"
            ),
            (
                "handoff = prepare_state_authority_child_handoff("
                "environment, parent_loss_policy="
                "STATE_AUTHORITY_PARENT_LOSS_TERMINATE)"
            ),
            "process = None",
            "try:",
            "    process = subprocess.Popen(",
            "        [",
            "            sys.executable,",
            "            '-P',",
            "            '-c',",
            f"            {ORDINARY_IMPLEMENTATION_DAEMON_BOOTSTRAP!r},",
            f"            {IMPLEMENTATION_DAEMON_MODULE_SENTINEL!r},",
            "            '--probe',",
            "            'value',",
            "        ],",
            f"        cwd={str(ROOT)!r},",
            "        env=environment,",
            "        stdin=subprocess.DEVNULL,",
            "        stdout=subprocess.PIPE,",
            "        stderr=subprocess.PIPE,",
            "    )",
            "    handoff.deliver(process, timeout_seconds=1.0)",
            "    stdout, stderr = process.communicate(timeout=8.0)",
            (
                "    if process.returncode != 0: "
                "raise RuntimeError(stderr.decode(errors='replace'))"
            ),
            "    print(json.dumps({'argv': stdout.decode().strip()}))",
            "finally:",
            "    handoff.close()",
            "    if process is not None and process.poll() is None:",
            "        process.kill()",
            "        process.wait(timeout=2.0)",
            "    os.close(descriptor)",
        )
    )

    completed = subprocess.run(
        [sys.executable, "-c", parent_code],
        cwd=ROOT,
        env={"PATH": "/usr/bin:/bin"},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15.0,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert ast.literal_eval(json.loads(completed.stdout)["argv"]) == [
        "-c",
        "--probe",
        "value",
    ]
    assert import_hook_marker.read_text(encoding="utf-8") == "started"
