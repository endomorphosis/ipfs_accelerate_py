"""Real Git custody and adversarial transfer checks; no provider is invoked.

Transfer fixtures are explicitly authored byte inventories. They do not claim
native context verification; the separate original-namespace probe covers that.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import zlib

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_context_evidence as evidence


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def public_git(tmp_path):
    root = tmp_path / "public-original"
    root.mkdir()
    files = {"bottle.py": b"def public_fixture():\n    return 1\n",
             "docs/guide.txt": b"Public upstream documentation.\n",
             ".supervisor-instruction.md": "Literal public caf\u00e9 task.\n".encode(),
             ".supervisor-public-smoke.py": b"# authored public structural check\n"}
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    for argv in [("init", "-q"), ("add", "."),
                 ("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                  "commit", "-qm", "public original")]:
        subprocess.run(["git", "-C", str(root), *argv], check=True, capture_output=True)
    baseline = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    sources = {name: {"sha256": sha(raw), "executable": False} for name, raw in files.items()}
    (root / ".runtime").mkdir()
    (root / ".runtime/private-owner-state").write_text("PRIVATE-OWNER-MARKER")
    return root, baseline, sources, files


def test_git_public_closure_replays_signed_old_sources_after_original_is_deleted(public_git, tmp_path):
    root, baseline, sources, files = public_git
    (root / "bottle.py").write_text("changed current source\n")
    held = evidence._git_public_closure(root, baseline, sources)
    assert b"PRIVATE-OWNER-MARKER" not in b"".join(held.values())
    assert not any(name.endswith("/config") or "/hooks/" in name for name in held)
    replay = tmp_path / "retained-public-repository"
    for member, raw in held.items():
        path = replay / Path(member).relative_to("repository")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        path.chmod(0o444)
    (replay / ".git/refs").mkdir()
    shutil.rmtree(root)
    for name, original in files.items():
        assert evidence._git(replay, "show", baseline + ":" + name) == original
    assert evidence._git(replay, "rev-parse", baseline + "^{commit}").strip().decode() == baseline


def test_git_public_closure_refuses_wrong_signed_hash(public_git):
    root, baseline, sources, _ = public_git
    sources["bottle.py"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="signed_public_git_source_differs"):
        evidence._git_public_closure(root, baseline, sources)


@pytest.mark.parametrize("name", [".git/config", ".runtime/private-owner-state", "../private", "a//b"])
def test_git_public_closure_cannot_nominate_private_or_noncanonical_sources(public_git, name):
    root, baseline, _, _ = public_git
    with pytest.raises(ValueError):
        evidence._git_public_closure(root, baseline, {name: {"sha256": "0" * 64}})


def test_git_public_closure_refuses_git_symlink_blobs(public_git):
    root, _, _, _ = public_git
    (root / "public-link").symlink_to("bottle.py")
    subprocess.run(["git", "-C", str(root), "add", "public-link"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Fixture",
                    "-c", "user.email=fixture@example.invalid", "commit", "-qm", "link fixture"],
                   check=True, capture_output=True)
    baseline = evidence._git(root, "rev-parse", "HEAD").strip().decode()
    with pytest.raises(ValueError, match="regular"):
        evidence._git_public_closure(root, baseline, {"public-link": {"sha256": sha(b"bottle.py")}})


def transfer(tmp_path, *, change=None, extra=None, link=False):
    data = {"state/prepared.json": ("prepared-state", b"{}"),
            "state/context-result.json": ("context-result", b"{}"),
            "state/one-base-context-capsule.json": ("native-capsule", b"{}"),
            "replay-request.json": ("replay-request", b"{}"),
            "repository/.git/HEAD": ("synthetic-git-head", b"1" * 40 + b"\n")}
    if change:
        change(data)
    rows = [{"path": name, "role": role, "bytes": len(raw), "sha256": sha(raw)}
            for name, (role, raw) in sorted(data.items())]
    manifest = {"schema": evidence.SCHEMA, "profile": evidence.PROFILE,
                "repository_origin": "/app", "state_origin": "/opt/ipfs-supervisor/state/authored-fixture",
                "workspace_root_origin": "/opt/ipfs-supervisor/worktrees", "native_prompt_bindings": ["0" * 64],
                "public_baseline_commit": "1" * 40, "public_source_count": 4, "files": rows,
                "content_bytes": sum(row["bytes"] for row in rows), "provider_calls": 0,
                "current_source_freshness_claimed": False, "private_state_tree_copied": False,
                "provider_transcripts_included": False, "execution_authority": False, "completion_authority": False}
    encoded = json.dumps(manifest, sort_keys=True).encode()
    path = tmp_path / "authored-transfer.tar"
    with tarfile.open(path, "w") as tar:
        for name, raw in [("manifest.json", encoded), *((n, v[1]) for n, v in data.items())]:
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
        if extra:
            info = tarfile.TarInfo(extra)
            if link:
                info.type = tarfile.SYMTYPE
                info.linkname = "/outside/private"
                tar.addfile(info)
            else:
                info.size = 1
                tar.addfile(info, io.BytesIO(b"x"))
    return path, sha(path.read_bytes()), sha(encoded), manifest


def test_transfer_verifies_before_fresh_unpack_without_claiming_native_verification(tmp_path):
    archive, digest, manifest_digest, manifest = transfer(tmp_path)
    assert evidence.verify_evidence_archive(archive=archive, expected_sha256=digest,
        manifest_sha256=manifest_digest) == manifest
    destination = tmp_path / "retained"
    evidence.verify_and_unpack(archive=archive, expected_sha256=digest,
        manifest_sha256=manifest_digest, destination=destination)
    assert (destination / "repository/.git/refs").is_dir()
    assert (destination / "state/prepared.json").stat().st_mode & 0o777 == 0o444
    with pytest.raises(ValueError, match="fresh"):
        evidence.verify_and_unpack(archive=archive, expected_sha256=digest,
            manifest_sha256=manifest_digest, destination=destination)


@pytest.mark.parametrize("extra,link", [("private-owner.json", False), ("../outside", False),
                                        ("state/link", True), ("state/prepared.json", False)])
def test_transfer_rejects_extra_traversal_links_and_duplicate_members(tmp_path, extra, link):
    archive, digest, manifest_digest, _ = transfer(tmp_path, extra=extra, link=link)
    with pytest.raises(ValueError):
        evidence.verify_evidence_archive(archive=archive, expected_sha256=digest,
            manifest_sha256=manifest_digest)


def test_rebound_transfer_manifest_cannot_expand_private_scope(tmp_path):
    def changed(data):
        data["repository/.git/config"] = ("public-git-object", b"private-marker")
    archive, digest, manifest_digest, _ = transfer(tmp_path, change=changed)
    with pytest.raises(ValueError, match="scope"):
        evidence.verify_evidence_archive(archive=archive, expected_sha256=digest,
            manifest_sha256=manifest_digest)


def test_transfer_requires_independent_archive_and_manifest_pins(tmp_path):
    archive, digest, manifest_digest, _ = transfer(tmp_path)
    with pytest.raises(ValueError, match="archive_digest"):
        evidence.verify_evidence_archive(archive=archive, expected_sha256="0" * 64,
            manifest_sha256=manifest_digest)
    with pytest.raises(ValueError, match="manifest_digest"):
        evidence.verify_evidence_archive(archive=archive, expected_sha256=digest,
            manifest_sha256="0" * 64)


def test_replay_namespace_argv_refuses_candidate_auditor_and_preserves_origins(tmp_path):
    archive, digest, manifest_digest, _ = transfer(tmp_path)
    export = tmp_path / "sealed"
    evidence.verify_and_unpack(archive=archive, expected_sha256=digest,
        manifest_sha256=manifest_digest, destination=export)
    output = tmp_path / "output"
    output.mkdir()
    arguments = dict(export=export, image="sha256:" + "1" * 64, output=output)
    with pytest.raises(ValueError, match="trusted_replay_auditor"):
        evidence.replay_docker_argv(**arguments, auditor_path="/app/candidate.py")
    argv = evidence.replay_docker_argv(**arguments,
        auditor_path="/opt/ipfs-supervisor/source/benchmarks/agent_supervisor/container_coding/terminal_context_evidence.py")
    assert "--network=none" in argv and "--pull=never" in argv and "--read-only" in argv
    assert any("dst=/app,readonly" in value for value in argv)
    assert any("dst=/opt/ipfs-supervisor/state/authored-fixture,readonly" in value for value in argv)
    assert all("credential" not in value or value == "credential" for value in argv)


def test_single_link_reader_rejects_links_and_keeps_zero_length_data_bounded(tmp_path):
    original = tmp_path / "data"
    original.write_bytes(b"")
    assert evidence._read(original) == b""
    linked = tmp_path / "hardlink"
    linked.hardlink_to(original)
    with pytest.raises(ValueError, match="single_link"):
        evidence._read(original)
    linked.unlink()
    linked.symlink_to(original)
    with pytest.raises(OSError):
        evidence._read(linked)


def _no_live_owned_processes(pid_file):
    assert pid_file.is_file()
    pids = json.loads(pid_file.read_text())
    deadline = time.monotonic() + 1
    while True:
        live = []
        for pid in pids:
            path = Path("/proc") / str(pid) / "stat"
            try:
                # An orphan zombie has exited; this caller cannot reap it.
                if path.read_text().rsplit(")", 1)[1].split()[0] not in {"Z", "X"}:
                    live.append(pid)
            except FileNotFoundError:
                pass
        if not live or time.monotonic() >= deadline:
            assert not live
            return
        time.sleep(.01)


def _authored_hanging_git(tmp_path, *, stdout=b"x"):
    pid_file = tmp_path / "owned-pids.json"
    script = tmp_path / "authored-git.py"
    script.write_text("import json,os,pathlib,signal,subprocess,sys,time\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        "child=subprocess.Popen([sys.executable,'-c',"
        "'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'])\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(json.dumps([os.getpid(),child.pid]))\n"
        f"os.write(1,{stdout!r})\n"
        "time.sleep(60)\n")
    return script, pid_file


def test_git_hanging_pipe_is_time_bounded_and_owned_group_is_removed(tmp_path, monkeypatch):
    script, pid_file = _authored_hanging_git(tmp_path)
    original = subprocess.Popen
    monkeypatch.setattr(evidence.subprocess, "Popen",
        lambda command, **kwargs: original([sys.executable, str(script)], **kwargs))
    monkeypatch.setattr(evidence, "_GIT_TIMEOUT_SECONDS", .3)
    began = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        evidence._git(tmp_path, "authored-hang")
    assert time.monotonic() - began < 2
    _no_live_owned_processes(pid_file)


def test_git_excess_output_is_bounded_before_process_finishes(tmp_path, monkeypatch):
    script, pid_file = _authored_hanging_git(tmp_path, stdout=b"x" * 10_001)
    original = subprocess.Popen
    monkeypatch.setattr(evidence.subprocess, "Popen",
        lambda command, **kwargs: original([sys.executable, str(script)], **kwargs))
    began = time.monotonic()
    with pytest.raises(ValueError, match="git_output_bound"):
        evidence._git(tmp_path, "authored-overflow", limit=100)
    assert time.monotonic() - began < 2
    _no_live_owned_processes(pid_file)


def test_collector_timeout_removes_git_descendants_and_does_not_block_on_stdin(tmp_path, monkeypatch):
    git, pid_file = _authored_hanging_git(tmp_path)
    exporter = tmp_path / "authored-exporter.py"
    exporter.write_text("import importlib.util,os,subprocess,sys\n"
        f"spec=importlib.util.spec_from_file_location('owned_evidence',{str(Path(evidence.__file__).resolve())!r})\n"
        "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n"
        "assert os.getpid()==os.getpgrp()==os.getsid(0)\n"
        "assert os.environ.get(module._COLLECTOR_GROUP_MARKER)=='owned'\n"
        "module._IN_OWNED_COLLECTOR_GROUP=True\n"
        "original=subprocess.Popen\n"
        f"subprocess.Popen=lambda command,**kwargs:original([sys.executable,{str(git)!r}],**kwargs)\n"
        "module._GIT_TIMEOUT_SECONDS=60\n"
        f"module._git({str(tmp_path)!r},'authored-hang')\n")
    original = subprocess.Popen
    launched = []
    def launch(command, **kwargs):
        assert kwargs["start_new_session"] is True
        child = original([sys.executable, str(exporter)], **kwargs)
        launched.append(child)
        return child
    monkeypatch.setattr(evidence.subprocess, "Popen", launch)
    # Larger than an OS pipe; the child deliberately never reads this input.
    # This authored request does not claim production request admission.
    monkeypatch.setattr(evidence, "_request", lambda receipts, report: {"authored": "x" * 60_000})
    began = time.monotonic()
    result = evidence.collect_terminal_context_evidence(state=tmp_path / "state", receipts=[],
        workspace_root=tmp_path, decision_report={}, timeout_seconds=.5)
    assert result["status"] == "unavailable" and result["reason"] == "export_timeout"
    assert time.monotonic() - began < 2
    assert len(launched) == 1 and launched[0].poll() is not None
    _no_live_owned_processes(pid_file)


def test_collector_cancellation_also_removes_owned_group(tmp_path, monkeypatch):
    script, pid_file = _authored_hanging_git(tmp_path)
    original = subprocess.Popen
    launched = []
    def launch(command, **kwargs):
        child = original([sys.executable, str(script)], **kwargs)
        launched.append(child)
        return child
    class Cancelled(BaseException):
        pass
    def cancelled_select(selector, timeout):
        deadline = time.monotonic() + 1
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        raise Cancelled()
    monkeypatch.setattr(evidence.subprocess, "Popen", launch)
    monkeypatch.setattr(evidence.selectors.DefaultSelector, "select", cancelled_select)
    monkeypatch.setattr(evidence, "_request", lambda receipts, report: {})
    with pytest.raises(Cancelled):
        evidence.collect_terminal_context_evidence(state=tmp_path / "state", receipts=[],
            workspace_root=tmp_path, decision_report={}, timeout_seconds=1)
    assert len(launched) == 1 and launched[0].poll() is not None
    _no_live_owned_processes(pid_file)
