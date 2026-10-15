"""Authored installer fixtures; no network, containers, npm packages or models."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding import terminal_setup_cache_advice as cache_advice


def executable(path, source):
    path.write_text(f"#!{sys.executable}\n" + source)
    path.chmod(0o755)
    return path


@pytest.fixture
def installer(tmp_path, monkeypatch):
    root = tmp_path / "supervisor"
    home = root / "home"
    home.mkdir(parents=True)
    binaries = tmp_path / "system"
    binaries.mkdir()
    node = executable(binaries / "node", "print('v18.20.4')\n")
    state = dict(root=root, home=home, binaries=binaries, node=node)

    def npm_script(mode="good"):
        expected_hashes = {}
        for name, output in (("codex", "codex-cli " + ("0.159.0" if mode == "native-version" else "0.160.0")),
                             ("codex-code-mode-host", "Usage: code-mode-host")):
            code = ("raise SystemExit(3)" if name == "codex-code-mode-host" and mode == "host-failed"
                    else "print(" + repr(output) + ")")
            expected_hashes[name] = hashlib.sha256(("#!" + sys.executable + "\n" + code + "\n").encode()).hexdigest()
        # These authored executable pins qualify fixture bytes only. Production
        # arm64/x64 hashes remain unchanged and are checked independently below.
        monkeypatch.setattr(deployment, "CODEX_X86_64_SHA256", expected_hashes.copy())
        monkeypatch.setattr(cache_advice, "CODEX_PINS_V2", expected_hashes.copy())
        return executable(binaries / "npm", f'''
import json,os,pathlib,signal,subprocess,sys,time
mode={mode!r}
args=sys.argv[1:]
work=pathlib.Path.cwd()
with (work/'commands.jsonl').open('a') as out:
 out.write(json.dumps({{'args':args,'env':dict(os.environ),'cwd':str(work)}})+'\\n')
if args==['--version']:
 print('9.2.0');raise SystemExit(0)
if mode=='fail':
 print('untrusted-install-output-sentinel');raise SystemExit(7)
if mode=='flood':
 print('z'*100000);raise SystemExit(0)
if mode=='stall':
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 child=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'])
 (work/'pids.json').write_text(json.dumps([os.getpid(),child.pid]))
 while True:time.sleep(0.01)
prefix=pathlib.Path(args[args.index('--prefix')+1])
package=prefix/'lib/node_modules/@openai/codex'
package.mkdir(parents=True)
metadata={{'name':'@openai/codex','version':'0.160.0'}}
if mode=='package-version':metadata['version']='0.159.0'
(package/'package.json').write_text(json.dumps(metadata) if mode!='metadata' else 'not json')
vendor=package/'node_modules/@openai/codex-test/vendor/test/bin'
vendor.mkdir(parents=True)
for name,output in [('codex','codex-cli '+('0.159.0' if mode=='native-version' else '0.160.0')),('codex-code-mode-host','Usage: code-mode-host')]:
 if name=='codex-code-mode-host' and mode=='missing-host':continue
 path=vendor/name
 code='raise SystemExit(3)' if name=='codex-code-mode-host' and mode=='host-failed' else 'print('+repr(output)+')'
 path.write_text('#!'+sys.executable+'\\n'+code+'\\n')
 path.chmod(0o755)
if mode=='symlink-host':
 (vendor/'codex-code-mode-host').unlink();(vendor/'codex-code-mode-host').symlink_to(vendor/'codex')
if mode=='duplicate-native':
 extra=package/'node_modules/@openai/codex-other/vendor/test/bin'
 extra.mkdir(parents=True);(extra/'codex').write_text('ambiguous')
if mode=='extra-native':
 (vendor/'unqualified-helper').write_text('extra');(vendor/'unqualified-helper').chmod(0o755)
if mode=='oversize-native':
 with (vendor/'codex').open('r+b') as stream:stream.truncate(512*1024**2+1)
if mode=='hardlink-native':
 os.link(vendor/'codex',work/'native-hardlink')
''')

    state["npm_script"] = npm_script
    npm_script()
    monkeypatch.setattr("shutil.which", lambda name, *, path: str(binaries / name)
                        if (binaries / name).exists() else None)
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    return state


def execute_installer(installer, capsys):
    namespace = {}
    with pytest.raises(SystemExit) as stopped:
        exec(compile(deployment.native_codex_install_script(str(installer["root"])),
                     "native-codex-install", "exec"), namespace)
    captured = capsys.readouterr()
    assert captured.err == ""
    report = json.loads(captured.out)
    assert stopped.value.code == (0 if report["completed"] else 1)
    return report


def test_compatible_system_node_install_is_exact_isolated_and_cache_compatible(installer, monkeypatch, capsys, tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / ".npmrc").write_text("registry=https://untrusted.invalid\n")
    (installer["home"] / ".npmrc").write_text("ignore-scripts=false\n")
    monkeypatch.chdir(task)
    for name in ("NPM_CONFIG_REGISTRY", "npm_config_userconfig", "NODE_OPTIONS",
                 "HTTP_PROXY", "HTTPS_PROXY", "CUSTOM_SECRET"):
        monkeypatch.setenv(name, "must-not-enter-child")
    report = execute_installer(installer, capsys)
    assert report["completed"] is True
    assert report["node_version"] == "v18.20.4"
    assert report["npm_version"] == "9.2.0"
    assert report["commands_completed"] == 5
    assert report["nvm_bootstrap_used"] is False
    assert report["provider_calls"] == 0
    assert report["independent_arm64_hashes_checked"] is False
    assert report["independent_native_hashes_checked"] is True
    assert report["architecture"] == "x86_64"
    work = installer["home"] / ".native-codex-install"
    commands = [json.loads(line) for line in (work / "commands.jsonl").read_text().splitlines()]
    assert len(commands) == 2
    for call in commands:
        assert call["cwd"] == str(work)
        assert "must-not-enter-child" not in json.dumps(call)
        assert call["env"]["PATH"] == "/usr/local/bin:/usr/bin:/bin"
        for name in ("USERCONFIG", "GLOBALCONFIG"):
            assert Path(call["env"]["NPM_CONFIG_" + name]).read_bytes() == b""
    args = commands[-1]["args"]
    assert args[-1] == "@openai/codex@0.160.0"
    for flag in ("--ignore-scripts", "--audit=false", "--fund=false", "--progress=false",
                 "--fetch-retries=1", "--fetch-timeout=30000", "--registry=https://registry.npmjs.org/"):
        assert flag in args
    matches = list((installer["home"] / ".nvm").glob(
        "versions/node/*/lib/node_modules/@openai/codex/node_modules/@openai/codex-*/vendor/*/bin/codex"))
    assert len(matches) == 1
    assert "/versions/node/system/" in str(matches[0])
    for name, digest in report["native_sha256"].items():
        assert hashlib.sha256((matches[0].parent / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("node_version", ["v15.9.0", "v16", "18.20.4", "v99.0.0-poison"])
def test_unsupported_node_fails_before_npm_install(installer, capsys, node_version):
    executable(installer["node"], f"print({node_version!r})\n")
    report = execute_installer(installer, capsys)
    assert report["completed"] is False
    assert report["reason"] == "system_node_16_required"
    assert report["phase"] == "system-node"
    assert not (installer["home"] / ".native-codex-install/commands.jsonl").exists()


@pytest.mark.parametrize("missing", ["node", "npm"])
def test_missing_system_binary_fails_without_bootstrap(installer, capsys, missing):
    (installer["binaries"] / missing).unlink()
    report = execute_installer(installer, capsys)
    assert report["reason"] == "system_node_npm_required"
    assert report["commands_completed"] == 0


@pytest.mark.parametrize("mode,phase,reason", [
    ("fail", "npm-install", "command_failed"),
    ("flood", "npm-install", "output_limit"),
    ("metadata", "package-verify", "invalid_package_metadata"),
    ("package-version", "package-verify", "package_version_mismatch"),
    ("missing-host", "package-verify", "native_bundle_incomplete"),
    ("symlink-host", "package-verify", "native_bundle_invalid"),
    ("duplicate-native", "package-verify", "native_bundle_ambiguous"),
    ("extra-native", "package-verify", "native_bundle_incomplete"),
    ("oversize-native", "package-verify", "native_bundle_invalid"),
    ("hardlink-native", "package-verify", "native_bundle_invalid"),
    ("native-version", "native-version", "native_version_mismatch"),
    ("host-failed", "native-code-mode-host", "command_failed"),
])
def test_failed_install_or_native_qualification_has_bounded_content_free_failure(installer, capsys, mode, phase, reason):
    installer["npm_script"](mode)
    report = execute_installer(installer, capsys)
    assert report["completed"] is False
    assert report["phase"] == phase
    assert report["reason"] == reason
    assert len(json.dumps(report)) < 1024
    assert "untrusted-install-output-sentinel" not in json.dumps(report)


@pytest.mark.parametrize("architecture", ["aarch64", "x86_64"])
def test_native_binary_hashes_must_match_independent_pins(installer, monkeypatch, capsys, architecture):
    monkeypatch.setattr("platform.machine", lambda: architecture)
    if architecture == "aarch64":
        monkeypatch.setattr(cache_advice, "CODEX_PINS_V2", {"codex": "0" * 64, "codex-code-mode-host": "1" * 64})
    else:
        monkeypatch.setattr(deployment, "CODEX_X86_64_SHA256", {"codex": "0" * 64, "codex-code-mode-host": "1" * 64})
    report = execute_installer(installer, capsys)
    assert report["reason"] == "native_bundle_hash_mismatch"
    assert report["commands_completed"] == 3


@pytest.mark.parametrize("machine,canonical", [("aarch64", "aarch64"), ("arm64", "aarch64"),
                                               ("x86_64", "x86_64"), ("AMD64", "x86_64")])
def test_both_supported_architectures_enforce_their_own_pins(installer, monkeypatch, capsys, machine, canonical):
    monkeypatch.setattr("platform.machine", lambda: machine)
    report = execute_installer(installer, capsys)
    assert report["completed"] is True
    assert report["architecture"] == canonical
    assert report["independent_native_hashes_checked"] is True
    assert report["independent_arm64_hashes_checked"] is (canonical == "aarch64")


@pytest.mark.parametrize("system,machine", [("Darwin", "arm64"), ("Windows", "AMD64"),
                                           ("Linux", "ppc64le"), ("Linux", "armv7l")])
def test_unsupported_platform_fails_before_installation(installer, monkeypatch, capsys, system, machine):
    monkeypatch.setattr("platform.system", lambda: system)
    monkeypatch.setattr("platform.machine", lambda: machine)
    report = execute_installer(installer, capsys)
    assert report["reason"] == "unsupported_system_architecture"
    assert report["phase"] == "system-platform"
    assert report["commands_completed"] == 0
    assert not (installer["home"] / ".native-codex-install").exists()


def test_unqualified_cli_upgrade_requires_new_native_pins(monkeypatch):
    monkeypatch.setattr(deployment, "CODEX_VERSION", "0.161.0")
    with pytest.raises(ValueError, match="independent current arm64 binary pins"):
        deployment.native_codex_install_script()


@pytest.mark.parametrize("unsafe", ["home", "prefix", "work"])
def test_symlink_install_paths_are_rejected(installer, capsys, tmp_path, unsafe):
    destination = tmp_path / "outside"
    destination.mkdir()
    if unsafe == "home":
        installer["home"].rmdir()
        installer["home"].symlink_to(destination)
    elif unsafe == "prefix":
        (installer["home"] / ".nvm").symlink_to(destination)
    else:
        (installer["home"] / ".native-codex-install").symlink_to(destination)
    report = execute_installer(installer, capsys)
    assert report["completed"] is False
    assert report["commands_completed"] == 0
    assert list(destination.iterdir()) == []


def not_running(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return True
    return raw[raw.rfind(")") + 2:].split()[0] == "Z"


def test_shared_deadline_kills_stalled_npm_and_descendant(installer, monkeypatch, capsys):
    installer["npm_script"]("stall")
    monkeypatch.setattr(deployment, "CODEX_INSTALL_TIMEOUT_SEC", 0.8)
    started = time.monotonic()
    report = execute_installer(installer, capsys)
    assert time.monotonic() - started < 4
    assert report["reason"] == "timeout"
    assert report["phase"] == "npm-install"
    pids = json.loads((installer["home"] / ".native-codex-install/pids.json").read_text())
    until = time.monotonic() + 2
    while not all(not_running(pid) for pid in pids) and time.monotonic() < until:
        time.sleep(0.01)
    assert all(not_running(pid) for pid in pids)
    with pytest.raises(ChildProcessError):
        os.waitpid(pids[0], os.WNOHANG)  # The direct npm child was reaped.


def test_interrupt_during_last_command_cleanup_cannot_report_success(installer, monkeypatch, capsys):
    original_popen = subprocess.Popen
    original_killpg = os.killpg
    final_pid = None

    def capture_final_process(command, *args, **kwargs):
        nonlocal final_pid
        process = original_popen(command, *args, **kwargs)
        if Path(command[0]).name == "codex-code-mode-host":
            final_pid = process.pid
        return process

    def interrupt_at_final_cleanup(pid, signum):
        if pid == final_pid and signum == signal.SIGTERM:
            os.kill(os.getpid(), signal.SIGINT)
        return original_killpg(pid, signum)

    monkeypatch.setattr(subprocess, "Popen", capture_final_process)
    monkeypatch.setattr(os, "killpg", interrupt_at_final_cleanup)
    report = execute_installer(installer, capsys)
    assert report["completed"] is False
    assert report["phase"] == "native-code-mode-host"
    assert report["reason"] == "interrupted"
    assert report["commands_completed"] == 5


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_signal_interrupt_fences_install_children(installer, signum):
    installer["npm_script"]("stall")
    script = ("import shutil,platform\n"
              + "shutil.which=lambda name,**kwargs: " + repr(str(installer["binaries"])) + "+'/'+name\n"
              + "platform.machine=lambda:'x86_64'\n"
              + deployment.native_codex_install_script(str(installer["root"])))
    process = subprocess.Popen([sys.executable, "-I", "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    marker = installer["home"] / ".native-codex-install/pids.json"
    try:
        until = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < until:
            assert process.poll() is None
            time.sleep(0.01)
        assert marker.exists()
        pids = json.loads(marker.read_text())
        process.send_signal(signum)
        output, error = process.communicate(timeout=4)
        assert error == b""
        report = json.loads(output)
        assert report["reason"] == "interrupted"
        assert process.returncode == 1
        until = time.monotonic() + 2
        while not all(not_running(pid) for pid in pids) and time.monotonic() < until:
            time.sleep(0.01)
        assert all(not_running(pid) for pid in pids)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
