"""Original deadline transport with actual parent/child births and sealed FDs.

Only the deployed wrapper pathname is authored: a fresh inert executable
replaces /opt/ipfs-supervisor/bin/router-worker in both processes. Kernel
parentage, birth, namespaces, argv, cwd, descriptor seals and cleanup are real.
No model, target program, native database or new scheduler is invoked.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import owner_provider_deadline as module


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def _sealed(raw, *, seal=True, mode=0o400):
    descriptor = os.memfd_create("authored-deadline-refusal",
        os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
    os.write(descriptor, raw)
    os.fchmod(descriptor, mode)
    if seal:
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS,
            fcntl.F_SEAL_SEAL | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW)
    return descriptor


def _closed(descriptor):
    try:
        os.fstat(descriptor)
    except OSError:
        return True
    return False


@pytest.fixture
def executable(tmp_path, monkeypatch):
    root = Path(module.__file__).resolve().parents[3]
    script = tmp_path / "inert-owner-wrapper"
    script.write_text(f"#!{sys.executable} -I\n" +
        "import json,os,sys\nfrom pathlib import Path\n" +
        f"sys.path.insert(0,{str(root)!r})\n" +
        "from ipfs_accelerate_py.agent_supervisor.runtime import owner_provider_deadline as m\n" +
        "m.OWNER_WRAPPER=str(Path(__file__).resolve())\n" +
        "env=dict(os.environ)\nfd=int(env.get(m.DEADLINE_FD_ENV,'-1'))\n" +
        "try:\n" +
        " command=list(sys.argv)\n workspace=Path.cwd()\n" +
        " if sys.argv[1]=='wrong-command':command.append('unadmitted-extra')\n" +
        " if sys.argv[1]=='wrong-workspace':workspace=workspace.parent\n" +
        " deadline=m.consume_owner_provider_deadline_handoff(command=command,workspace=workspace,environment=env)\n" +
        " result={'ok':True,'deadline_ns':deadline}\n" +
        "except BaseException as error:\n result={'ok':False,'error_type':type(error).__name__}\n" +
        "try:os.fstat(fd);closed=False\nexcept OSError:closed=True\n" +
        "result.update(fd_closed=closed,private_environment_removed=not any(k in env for k in (m.DEADLINE_FD_ENV,m.DEADLINE_SHA256_ENV)))\n" +
        "if result['ok']:result['second_consume']=m.consume_owner_provider_deadline_handoff(command=list(sys.argv),workspace=Path.cwd(),environment=env)\n" +
        "print(json.dumps(result,sort_keys=True))\n")
    script.chmod(0o700)
    monkeypatch.setattr(module, "OWNER_WRAPPER", str(script))
    monkeypatch.chdir(tmp_path)
    return script


def _run(command, handoff, *, environment=None, pass_fds=None, cwd=None):
    completed = subprocess.run(command, cwd=Path.cwd() if cwd is None else cwd,
        env={"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1",
            **(handoff.environment if environment is None else environment)},
        pass_fds=handoff.pass_fds if pass_fds is None else pass_fds,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True, timeout=10)
    assert completed.stderr == b""
    return json.loads(completed.stdout)


def test_real_direct_child_consumes_exact_original_deadline_and_closes_private_fd(executable):
    command = [str(executable), "success"]
    deadline = time.monotonic() + 30
    with module.prepare_owner_provider_deadline_handoff(command=command,
            workspace=Path.cwd(), absolute_deadline=deadline) as handoff:
        assert len(handoff.pass_fds) == 1
        descriptor = handoff.pass_fds[0]
        assert descriptor >= 3 and stat.S_IMODE(os.fstat(descriptor).st_mode) == 0o400
        raw = os.pread(descriptor, module.MAX_HANDOFF_BYTES + 1, 0)
        body = json.loads(raw)
        assert raw == _canonical(body) and body["schema"] == module.SCHEMA
        numerator, denominator = deadline.as_integer_ratio()
        assert body["absolute_deadline_ns"] == numerator * 1_000_000_000 // denominator
        assert body["issuer_pid"] == os.getpid()
        assert set(handoff.environment) == {module.DEADLINE_FD_ENV, module.DEADLINE_SHA256_ENV}
        assert handoff.environment[module.DEADLINE_SHA256_ENV] == hashlib.sha256(raw).hexdigest()
        result = _run(command, handoff)
        assert result == dict(ok=True, deadline_ns=body["absolute_deadline_ns"],
            fd_closed=True, private_environment_removed=True, second_consume=None)
        assert not _closed(descriptor)
    assert _closed(descriptor)


@pytest.mark.parametrize("mode", ["wrong-command", "wrong-workspace"])
def test_real_child_refuses_changed_command_or_workspace_and_closes_descriptor(executable, mode):
    command = [str(executable), mode]
    with module.prepare_owner_provider_deadline_handoff(command=command,
            workspace=Path.cwd(), absolute_deadline=time.monotonic() + 30) as handoff:
        result = _run(command, handoff)
    assert result == dict(ok=False, error_type="OwnerProviderDeadlineError",
        fd_closed=True, private_environment_removed=True)


@pytest.mark.parametrize("change", ["digest", "unsealed", "writable", "unknown-field",
    "duplicate-key", "issuer-pid", "issuer-birth", "issuer-uid", "boot", "namespace",
    "workspace-inode", "expired", "noncanonical", "oversized"])
def test_real_child_rejects_corrupt_or_unbounded_handoff_without_adopting_it(executable, change):
    command = [str(executable), "success"]
    with module.prepare_owner_provider_deadline_handoff(command=command,
            workspace=Path.cwd(), absolute_deadline=time.monotonic() + 30) as handoff:
        original = handoff.pass_fds[0]
        body = json.loads(os.pread(original, module.MAX_HANDOFF_BYTES + 1, 0))
        if change == "unknown-field": body["execution_authority"] = True
        if change == "issuer-pid": body["issuer_pid"] += 1
        if change == "issuer-birth": body["issuer_start_ticks"] += 1
        if change == "issuer-uid": body["issuer_euid"] += 1
        if change == "boot": body["boot_id"] = "00000000-0000-0000-0000-000000000000"
        if change == "namespace": body["namespaces"]["mnt"] = "mnt:[0]"
        if change == "workspace-inode": body["workspace"]["inode"] += 1
        if change == "expired": body["absolute_deadline_ns"] = time.monotonic_ns() - 1
        raw = _canonical(body)
        if change == "duplicate-key": raw = b'{"schema":"duplicate",' + raw[1:]
        if change == "noncanonical": raw += b"\n"
        if change == "oversized": raw += b" " * (module.MAX_HANDOFF_BYTES + 1)
        descriptor = _sealed(raw, seal=change != "unsealed", mode=0o600 if change == "writable" else 0o400)
        try:
            environment = dict(handoff.environment,
                **{module.DEADLINE_FD_ENV: str(descriptor),
                   module.DEADLINE_SHA256_ENV: "0" * 64 if change == "digest" else hashlib.sha256(raw).hexdigest()})
            result = _run(command, handoff, environment=environment, pass_fds=(descriptor,))
            assert result == dict(ok=False, error_type="TimeoutError" if change == "expired" else "OwnerProviderDeadlineError",
                fd_closed=True, private_environment_removed=True)
        finally:
            os.close(descriptor)


def test_owner_cannot_consume_its_own_child_handoff(executable):
    command = [str(executable), "success"]
    with module.prepare_owner_provider_deadline_handoff(command=command,
            workspace=Path.cwd(), absolute_deadline=time.monotonic() + 30) as handoff:
        descriptor = os.dup(handoff.pass_fds[0])
        environment = dict(handoff.environment, **{module.DEADLINE_FD_ENV: str(descriptor)})
        with pytest.raises(module.OwnerProviderDeadlineError):
            module.consume_owner_provider_deadline_handoff(command=command,
                workspace=Path.cwd(), environment=environment)
        assert _closed(descriptor) and not environment


def test_real_grandchild_cannot_adopt_a_different_direct_parent_handoff(executable):
    command = [str(executable), "success"]
    with module.prepare_owner_provider_deadline_handoff(command=command,
            workspace=Path.cwd(), absolute_deadline=time.monotonic() + 30) as handoff:
        descriptor = handoff.pass_fds[0]
        relay = ("import os,subprocess,sys;"
            f"completed=subprocess.run({command!r},env=dict(os.environ),pass_fds=({descriptor},),"
            "stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=5);"
            "sys.stdout.buffer.write(completed.stdout);sys.stderr.buffer.write(completed.stderr);"
            "raise SystemExit(completed.returncode)")
        completed = subprocess.run([sys.executable, "-I", "-c", relay], cwd=Path.cwd(),
            env={"PATH": os.defpath, **handoff.environment}, pass_fds=handoff.pass_fds,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True, timeout=10)
        assert completed.stderr == b""
        assert json.loads(completed.stdout) == dict(ok=False,
            error_type="OwnerProviderDeadlineError", fd_closed=True,
            private_environment_removed=True)


def test_real_child_uses_target_workspace_while_issuer_keeps_its_distinct_cwd(executable):
    parent_workspace = Path.cwd()
    target_workspace = parent_workspace / "allocated-worktree"
    target_workspace.mkdir(mode=0o700)
    command = [str(executable), "success"]
    with module.prepare_owner_provider_deadline_handoff(command=command,
            workspace=target_workspace, absolute_deadline=time.monotonic() + 30) as handoff:
        result = _run(command, handoff, cwd=target_workspace)
        assert result["ok"] is True and result["fd_closed"] is True
        assert result["private_environment_removed"] is True
        assert result["second_consume"] is None
    assert Path.cwd() == parent_workspace


@pytest.mark.parametrize("missing", ["fd", "digest"])
def test_partial_private_pair_is_removed_and_any_named_valid_fd_is_closed(executable, missing):
    descriptor = _sealed(b"{}")
    environment = {module.DEADLINE_FD_ENV: str(descriptor), module.DEADLINE_SHA256_ENV: "a" * 64}
    environment.pop(module.DEADLINE_FD_ENV if missing == "fd" else module.DEADLINE_SHA256_ENV)
    try:
        with pytest.raises(module.OwnerProviderDeadlineError):
            module.consume_owner_provider_deadline_handoff(command=[str(executable)],
                workspace=Path.cwd(), environment=environment)
        assert not environment
        assert _closed(descriptor) is (missing == "digest")
    finally:
        if not _closed(descriptor): os.close(descriptor)


@pytest.mark.parametrize("deadline", [True, False, "30", float("nan"), float("inf"),
    float("-inf"), -1, 0, 10 ** 500])
def test_selected_wrapper_refuses_malformed_original_deadline(executable, deadline):
    with pytest.raises(module.OwnerProviderDeadlineError):
        with module.prepare_owner_provider_deadline_handoff(command=[str(executable)],
                workspace=Path.cwd(), absolute_deadline=deadline):
            pytest.fail("malformed deadline reached child preparation")


def test_expired_selected_wrapper_refuses_without_resetting_allowance(executable):
    with pytest.raises((module.OwnerProviderDeadlineError, TimeoutError)):
        with module.prepare_owner_provider_deadline_handoff(command=[str(executable)],
                workspace=Path.cwd(), absolute_deadline=time.monotonic() - 1):
            pytest.fail("expired deadline was replaced")


@pytest.mark.parametrize("command", [["/usr/bin/python3", "router-worker"],
    ["/usr/bin/env", "/opt/ipfs-supervisor/bin/router-worker"],
    ["relative/router-worker"], ["/opt/ipfs-supervisor/bin/router-worker-copy"], []])
def test_other_commands_cannot_prepare_a_deadline_handoff(command):
    assert module.owner_provider_deadline_selected(command) is False
    with pytest.raises(module.OwnerProviderDeadlineError):
        with module.prepare_owner_provider_deadline_handoff(command=command,
                workspace=Path.cwd(), absolute_deadline=time.monotonic() + 30):
            pytest.fail("a foreign command prepared an owner deadline")
    assert module.consume_owner_provider_deadline_handoff(command=command,
        workspace=Path.cwd(), environment={}) is None


def test_deadline_context_closes_descriptor_after_prelaunch_exception(executable):
    descriptor = None
    with pytest.raises(RuntimeError, match="authored prelaunch refusal"):
        with module.prepare_owner_provider_deadline_handoff(command=[str(executable)],
                workspace=Path.cwd(), absolute_deadline=time.monotonic() + 30) as handoff:
            descriptor = handoff.pass_fds[0]
            raise RuntimeError("authored prelaunch refusal")
    assert descriptor is not None and _closed(descriptor)
