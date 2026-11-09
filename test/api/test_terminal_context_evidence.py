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


def transfer(tmp_path, *, change=None, extra=None, link=False, profile=evidence.PROFILE, manifest_target_bytes=None):
    data = {"state/prepared.json": ("prepared-state", b"{}"),
            "state/context-result.json": ("context-result", b"{}"),
            "state/one-base-context-capsule.json": ("native-capsule", b"{}"),
            "replay-request.json": ("replay-request", b"{}"),
            "repository/.git/HEAD": ("synthetic-git-head", b"1" * 40 + b"\n")}
    if change:
        change(data)
    rows = [{"path": name, "role": role, "bytes": len(raw), "sha256": sha(raw)}
            for name, (role, raw) in sorted(data.items())]
    manifest = {"schema": evidence.SCHEMA, "profile": profile,
                "repository_origin": "/app", "state_origin": "/opt/ipfs-supervisor/state/authored-fixture",
                "workspace_root_origin": "/opt/ipfs-supervisor/worktrees", "native_prompt_bindings": ["0" * 64],
                "public_baseline_commit": "1" * 40, "public_source_count": 4, "files": rows,
                "content_bytes": sum(row["bytes"] for row in rows), "provider_calls": 0,
                "current_source_freshness_claimed": False, "private_state_tree_copied": False,
                "provider_transcripts_included": False, "execution_authority": False, "completion_authority": False}
    if manifest_target_bytes is not None:
        # Real bounded manifest rows, not hidden payload padding. Every added
        # leaf is an explicitly authored empty public CAS transport member.
        size = len(evidence._encode(manifest))
        counter = 0
        while True:
            name = "repository/.runtime/bounds/" + "public/" * 32 + str(counter) + ".json"
            row = {"path": name, "role": "semantic-cas", "bytes": 0, "sha256": sha(b"")}
            cost = len(evidence._encode(row)) + 1
            if size + cost > manifest_target_bytes - 2048:
                break
            data[name] = ("semantic-cas", b"")
            manifest["files"].append(row)
            size += cost
            counter += 1
        name = "repository/.runtime/bounds/final.json"
        row = {"path": name, "role": "semantic-cas", "bytes": 0, "sha256": sha(b"")}
        padding = manifest_target_bytes - size - len(evidence._encode(row)) - 1
        assert 0 <= padding < 4096
        prefix = ("p" * 99 + "/") * (padding // 100) + "p" * (padding % 100)
        row["path"] = "repository/.runtime/bounds/" + prefix + "final.json"
        data[row["path"]] = ("semantic-cas", b"")
        manifest["files"].append(row)
        assert len(manifest["files"]) <= evidence.MAX_FILES
    encoded = evidence._encode(manifest)
    if manifest_target_bytes is not None:
        assert len(encoded) == manifest_target_bytes
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


@pytest.fixture(params=("full", "projected", "retrieval"))
def native_typed_evidence(tmp_path, monkeypatch, request):
    """Actual producer + immutable native owner + historical verifier; no seams."""
    from test.api.test_semantic_retrieval_query_reference import build_retrieval_native_fixture
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_decision_owner as owner_api
    monkeypatch.setenv("IPFS_ACCELERATE_TYPED_VERIFICATION_CACHE", "0")
    native = build_retrieval_native_fixture(tmp_path)
    if request.param == "full":
        from dataclasses import replace
        from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
        from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import SemanticDecisionPolicy
        from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
            ContextCompiler, build_text_context_references, render_context_capsule,
        )
        from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
        native.semantic_artifact = native.repository / ".runtime/semantic-full/worker-context.json"
        native.prepared = prepare_semantic_context(repository=native.repository,
            paths=["mod.py", native.instruction_source_path],
            required_raw_paths=["mod.py", native.instruction_source_path], objective="Preserve public behavior.",
            task_id=native.binding.task_id, output=native.semantic_artifact.parent,
            context_input_tokens=32768)
        native.semantic_original_text = native.semantic_artifact.read_text()
        native.semantic_parent = json.loads(native.semantic_original_text)
        native.semantic_bindings = replace(native.semantic_bindings,
            artifact=native.semantic_artifact.relative_to(native.repository).as_posix(),
            expected_sha256=sha(native.semantic_original_text.encode()),
            semantic_root_cid=native.semantic_parent["semantic_root_cid"], scope_cid=native.semantic_parent["scope_cid"])
        refs = build_text_context_references(native.semantic_original_text, reference_prefix="semantic-context",
            kind="semantic-context", path=native.semantic_bindings.artifact,
            repository_id=native.semantic_bindings.repository_id, tree_id=native.semantic_bindings.tree_id,
            required=True, chunk_bytes=1201, coverage_ids=("semantic:original",))
        compiled = ContextCompiler(ContextBudget(max_input_tokens=32768, max_items=128,
            max_item_bytes=16384, max_serialized_bytes=262144)).compile(
            repository_id=native.semantic_bindings.repository_id, tree_id=native.semantic_bindings.tree_id,
            objective_id=native.binding.task_id, objective_revision="sha256:authored-task",
            policy_id="policy:authored", policy_revision="sha256:authored-policy", caller="supervisor:authored",
            stage="implementation", goal={"id": native.binding.task_id},
            authority={"mode": "candidate_only", "completion_authority": False},
            scope={"allowed_paths": ["mod.py", native.instruction_source_path]},
            acceptance={"criteria": ["authored native source validation"]}, evidence=refs)
        native.native_prompt = render_context_capsule(compiled.capsule) + native.suffix
        native.policy = SemanticDecisionPolicy(required_capsule_ids=tuple(
            row["capsule_cid"] for row in native.semantic_parent["capsules"]), optional_capsule_ids=())
    origin = tmp_path / "independent-public-owner.json"
    origin_raw = (json.dumps(owner_api.build_decision_nomination(native.policy),
                             sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    origin.write_bytes(origin_raw)
    origin.chmod(0o444)
    arguments = [
        "--semantic-repository", str(native.repository), "--purpose", "coding",
        "--semantic-decision-nomination-artifact", str(origin),
        "--semantic-decision-nomination-sha256", sha(origin_raw),
        "--semantic-decision-task-id", native.binding.task_id,
        "--semantic-decision-typed-nomination", "enabled"]
    if request.param == "retrieval":
        from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import write_task_context_bundle
        bundle = write_task_context_bundle(repository=native.repository, prepared=[{
            "schema": "supervisor-task-context-preparation@1", "task_cid": "authored-native-custody-task",
            "task_id": native.binding.task_id, "metadata": native.metadata}],
            output=native.repository / ".runtime/evidence-context-bundle.json")
        arguments += ["--semantic-decision-context-bundle-artifact", bundle["artifact"],
            "--semantic-decision-context-bundle-sha256", bundle["sha256"],
            "--semantic-decision-task-cid", "authored-native-custody-task"]
    handoff = owner_api.prepare_decision_owner_dispatch(arguments=arguments,
        prompt_bytes=native.native_prompt.encode(), workspace=native.workspace)
    owner = handoff.receipt
    dialogue = {"schema": "supervisor-semantic-decision-dialogue@1",
        "task_id": owner["task_id"], "parent_sha256": owner["parent_sha256"],
        "policy_sha256": owner["policy_sha256"], "workspace": str(native.workspace),
        "worker_launch": {"original_native_prompt_sha256": owner["native_prompt_sha256"],
            "original_native_prompt_bytes": owner["native_prompt_bytes"],
            "launch_policy_artifact": owner["launch_policy_artifact"],
            "launch_policy_sha256": owner["launch_policy_sha256"]}}
    if request.param == "retrieval":
        dialogue["worker_launch"]["retrieval_query_binding"] = native.binding.to_dict()
    report = {"semantic_decision_typed_nomination_expected": True,
        "semantic_decision_owner_binding_expected": True,
        "semantic_decision_expected_task_id": owner["task_id"],
        "semantic_decision_nomination_sha256": sha(origin_raw),
        "semantic_decision_owner_bindings": [owner], "semantic_decision_dialogues": [dialogue]}
    native.owner, native.dialogue, native.request = owner, dialogue, {"receipts": [], "decision_report": report}
    native.origin, native.origin_raw, native.allocation = origin, origin_raw, tmp_path
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
    native.export_prepared = {"spec": {"task_key": native.binding.task_id}, "query": native.query,
                             "manifest": {"payload": {"sources": native.semantic_parent["manifest"]}}}
    # The wrapper header is authored; its semantic result and parent are genuine
    # native producer bytes. This is not a signed task/admission fixture.
    descriptor = {"schema": "terminal-initial-indexed-planning@1", "request_cid": "authored-request",
        "scan_cid": "authored-scan", "task_alias": native.binding.task_id,
        "task_alias_is_declaration_only": True, "manifest_cid": content_identity(native.export_prepared["manifest"]),
        "public_query_sha256": sha(native.query.encode()), "repository_id": native.semantic_bindings.repository_id,
        "metadata": {"Semantic context artifact": native.semantic_bindings.artifact,
                     "Semantic context sha256": native.semantic_bindings.expected_sha256},
        "semantic": native.prepared, "retrieval": {}, "world": {"artifact": ".runtime/not-followed-private-world"},
        "world_capture_cid": "authored-world", "world_artifact_kind": "empty-intent-planning-capture",
        "index": {"artifact": ".runtime/not-followed-index"}, "learned_embeddings": False,
        "model_snapshot": "/private/not-followed-model", "model_revision": "",
        "execution_authority": False, "completion_authority": False}
    descriptor_path = native.repository / ".runtime/terminal-initial-context/descriptor.json"
    descriptor_path.parent.mkdir(mode=0o755)
    descriptor_raw = evidence._encode(descriptor)
    descriptor_path.write_bytes(descriptor_raw)
    native.context_result = {"semantic_root_cid": native.semantic_parent["semantic_root_cid"],
        "initial_context_descriptor": {"artifact": ".runtime/terminal-initial-context/descriptor.json",
                                       "sha256": sha(descriptor_raw)}}
    assert "semantic" not in native.context_result
    return native


def _native_public_blocks(native, *, forbidden_cid=None):
    from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import IpfsDatasetsSemanticStateProvider
    directory = native.semantic_artifact.parent / "blocks"
    reads = []
    def reader(cid):
        assert cid != forbidden_cid, "orphan bytes must never be read"
        reads.append(cid)
        return (directory / cid).read_bytes()
    view = IpfsDatasetsSemanticStateProvider().open_verified_view(
        native.semantic_parent["semantic_root_cid"], reader)
    captured = []
    def capture(path, member, expected=None, role=None):
        raw = evidence._read(path)
        evidence._require(sha(raw) == expected, "held_artifact_digest_differs")
        assert role == "initial-context-descriptor"
        captured.append(member)
        return raw
    semantic = evidence._capture_initial_semantic_preparation(repository=native.repository,
        prepared=native.export_prepared, expected=native.context_result, parent=native.semantic_parent,
        artifact=native.semantic_bindings.artifact, parent_sha256=native.semantic_bindings.expected_sha256,
        repository_id=view.root.repository_id, capture=capture)
    assert captured == ["repository/.runtime/terminal-initial-context/descriptor.json"]
    result = evidence._public_typed_blocks(repository=native.repository,
        artifact=native.semantic_bindings.artifact, parent=native.semantic_parent,
        parent_bytes=native.semantic_original_text.encode(), semantic_preparation=semantic, view=view)
    return result, reads


def _capture_native_lineage(native):
    retained = {}
    def capture(path, member, limit=evidence.MAX_FILE_BYTES, expected=None, role=None):
        raw = evidence._read(path, limit)
        assert sha(raw) == expected
        retained[member] = (role, raw)
        return raw
    owner, origin, verified = evidence._capture_typed_lineage(repository=native.repository,
        allocation=native.allocation, request=native.request, dialogue=native.dialogue,
        prompt=native.native_prompt, capture=capture)
    return retained, owner, origin, verified


def test_actual_native_typed_custody_preserves_embedded_origin_and_cold_historical_verification(native_typed_evidence):
    native = native_typed_evidence
    blocks, _ = _native_public_blocks(native)
    assert set(blocks) == {path.name for path in (native.semantic_artifact.parent / "blocks").iterdir()}
    native.origin.unlink()  # The independent pin survives; the external source path is not reread.
    (native.repository / "mod.py").write_text("# changed current source; historical custody only\n")
    retained, owner, origin, verified = _capture_native_lineage(native)
    assert owner == native.owner and origin == native.origin_raw
    assert {role for role, _ in retained.values()} == {"typed-nomination-lineage", "typed-derived-nomination"}
    assert verified["source_freshness_revalidated"] is False
    assert verified["provider_calls"] == 0 and verified["proof_authority"] is False
    assert verified["completion_authority"] is False and verified["execution_authority"] is False
    assert not any("cache" in path or ".owner-key" in path for path in retained)
    assert not (native.repository / ".runtime/semantic-typed-verification-cache").exists()


@pytest.mark.parametrize("raw", [b"PRIVATE-ORPHAN-MUST-NOT-EXPORT", b""])
def test_actual_public_typed_closure_rejects_valid_cid_orphan_before_reading_it(native_typed_evidence, raw):
    from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_bytes
    native = native_typed_evidence
    cid = cid_for_bytes(raw)
    path = native.semantic_artifact.parent / "blocks" / cid
    path.write_bytes(raw)
    # A valid raw CID is still outside this exact public producer inventory.
    with pytest.raises(ValueError, match="public_producer_block_inventory_differs"):
        _native_public_blocks(native, forbidden_cid=cid)


def test_actual_public_closure_exact_byte_limit_preserves_duplicate_native_reads(native_typed_evidence, monkeypatch):
    native = native_typed_evidence
    original, reads = _native_public_blocks(native)
    total = sum(len(raw) for raw in original.values())
    monkeypatch.setattr(evidence, "MAX_CONTENT_BYTES", total)
    repeated, repeated_reads = _native_public_blocks(native)
    assert repeated == original and repeated_reads == reads
    assert reads.count(native.semantic_parent["semantic_root_cid"]) == 2
    monkeypatch.setattr(evidence, "MAX_CONTENT_BYTES", total - 1)
    with pytest.raises(ValueError, match="export_content_bound"):
        _native_public_blocks(native)


def test_actual_duplicate_source_cids_count_once_at_public_closure_limit(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
    from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import IpfsDatasetsSemanticStateProvider
    root = tmp_path / "source"
    root.mkdir()
    source = b"raise RuntimeError('source must never execute')\n\ndef f(value):\n    return value + 1\n"
    for name in ("left.py", "right.py"):
        (root / name).write_bytes(source)
    (root / "instruction.md").write_text("Read both public modules before editing.\n")
    artifact = ".runtime/duplicate-public-sources/worker-context.json"
    prepared = prepare_semantic_context(repository=root, paths=["left.py", "right.py", "instruction.md"],
        required_raw_paths=["left.py", "right.py", "instruction.md"], objective="Preserve both modules.",
        task_id="ACTUAL-DUPLICATE-PUBLIC-SOURCES", output=(root / artifact).parent,
        context_input_tokens=32768)
    raw = (root / artifact).read_bytes()
    parent = json.loads(raw)
    duplicate = parent["manifest"]["left.py"]["source_cid"]
    assert duplicate == parent["manifest"]["right.py"]["source_cid"]
    reads = []
    def capture():
        def reader(cid):
            reads.append(cid)
            return ((root / artifact).parent / "blocks" / cid).read_bytes()
        view = IpfsDatasetsSemanticStateProvider().open_verified_view(parent["semantic_root_cid"], reader)
        return evidence._public_typed_blocks(repository=root, artifact=artifact, parent=parent,
            parent_bytes=raw, semantic_preparation=prepared, view=view)
    original = capture()
    total = sum(map(len, original.values()))
    assert reads.count(duplicate) == 2
    reads.clear()
    monkeypatch.setattr(evidence, "MAX_CONTENT_BYTES", total)
    assert capture() == original and reads.count(duplicate) == 2
    monkeypatch.setattr(evidence, "MAX_CONTENT_BYTES", total - 1)
    with pytest.raises(ValueError, match="export_content_bound"):
        capture()


def test_actual_typed_lineage_rejects_changed_independent_pins_and_ambiguous_owner(native_typed_evidence):
    from copy import deepcopy
    native = native_typed_evidence
    report = native.request["decision_report"]
    for key, value in (("semantic_decision_nomination_sha256", "0" * 64),
                       ("semantic_decision_expected_task_id", "OTHER-TASK")):
        original = report[key]
        report[key] = value
        with pytest.raises(ValueError):
            _capture_native_lineage(native)
        report[key] = original
    duplicate = deepcopy(native.owner)
    duplicate["binding_id"] = "authored-conflicting-binding"
    report["semantic_decision_owner_bindings"].append(duplicate)
    with pytest.raises(ValueError, match="unambiguous"):
        _capture_native_lineage(native)


def test_actual_typed_lineage_refuses_changed_digest_addressed_derived_bytes(native_typed_evidence):
    native = native_typed_evidence
    artifact = Path(native.owner["typed_nomination"]["derived_nomination_artifact"])
    artifact.chmod(0o600)
    artifact.write_bytes(b"{}")
    artifact.chmod(0o444)
    with pytest.raises(ValueError):
        _capture_native_lineage(native)


def _authored_typed_transfer(data):
    """Transport joins only; these bytes deliberately make no native proof claim."""
    origin, derived = b'{"schema":"authored-public-nominee"}', b'{}'
    lineage = evidence._encode({"schema": "supervisor-semantic-typed-nomination-lineage@1",
        "origin_nomination_text": origin.decode(), "typed_nomination": {
            "schema": "supervisor-semantic-typed-owner-nomination@1", "nomination": {}, "policy": {},
            "launch_policy": {"schema": "supervisor-semantic-decision-launch-policy@2"},
            "receipt": {"schema": "supervisor-semantic-typed-owner-policy-receipt@1"}}})
    commit = {"schema": "supervisor-semantic-typed-nomination-binding@1",
        "lineage_sha256": sha(lineage), "derived_nomination_sha256": sha(derived),
        "lineage_artifact": "/app/.runtime/semantic-typed-nominations/" + sha(lineage) + ".json",
        "derived_nomination_artifact": "/app/.runtime/semantic-decision-nominations/" + sha(derived) + ".json"}
    owner = {"schema": "supervisor-semantic-decision-owner-binding@1", "binding_id": "authored-owner",
        "task_id": "AUTHORED", "nomination_sha256": sha(origin), "typed_nomination": commit,
        "nomination_artifact": "/public/owner.json", "native_prompt_sha256": "0" * 64,
        "native_prompt_bytes": 1, "parent_artifact": ".runtime/semantic/worker-context.json",
        "parent_sha256": "1" * 64, "policy_sha256": "2" * 64,
        "launch_policy_artifact": "/app/.runtime/semantic-decision-policies/" + "3" * 64 + ".json",
        "launch_policy_sha256": "3" * 64, "provider_called": False, "completion_authority": False}
    dialogue = {"schema": "supervisor-semantic-decision-dialogue@1", "task_id": "AUTHORED",
        "parent_sha256": owner["parent_sha256"], "policy_sha256": owner["policy_sha256"],
        "worker_launch": {"original_native_prompt_sha256": owner["native_prompt_sha256"],
            "original_native_prompt_bytes": owner["native_prompt_bytes"],
            "launch_policy_artifact": owner["launch_policy_artifact"],
            "launch_policy_sha256": owner["launch_policy_sha256"]}}
    report = {"semantic_decision_typed_nomination_expected": True,
        "semantic_decision_owner_binding_expected": True,
        "semantic_decision_expected_task_id": "AUTHORED", "semantic_decision_nomination_sha256": sha(origin),
        "semantic_decision_owner_bindings": [owner], "semantic_decision_dialogues": [dialogue]}
    data["replay-request.json"] = ("replay-request", evidence._encode({"receipts": [], "decision_report": report}))
    descriptor = b'{"schema":"authored-descriptor-transport-only"}'
    data["repository/.runtime/terminal-initial-context/descriptor.json"] = ("initial-context-descriptor", descriptor)
    data["state/context-result.json"] = ("context-result", evidence._encode({"initial_context_descriptor": {
        "artifact": ".runtime/terminal-initial-context/descriptor.json", "sha256": sha(descriptor)}}))
    data["repository/.runtime/semantic-typed-nominations/" + sha(lineage) + ".json"] = ("typed-nomination-lineage", lineage)
    data["repository/.runtime/semantic-decision-nominations/" + sha(derived) + ".json"] = ("typed-derived-nomination", derived)
    data["typed-origins/" + sha(origin) + ".json"] = ("typed-origin-nomination", origin)


def _authored_typed_retrieval_transfer(data, *, referenced=True):
    """Explicit transport bytes, separate from the genuine native fixture below."""
    _authored_typed_transfer(data)
    request = json.loads(data["replay-request.json"][1])
    owner = request["decision_report"]["semantic_decision_owner_bindings"][0]
    dialogue = request["decision_report"]["semantic_decision_dialogues"][0]
    old_lineage = "repository/.runtime/semantic-typed-nominations/" + owner["typed_nomination"]["lineage_sha256"] + ".json"
    origin = json.loads(data.pop(old_lineage)[1])["origin_nomination_text"]
    snapshot = evidence._encode({"index_id": "authored-snapshot-transport-only"})
    raw = {"schema": "supervisor-code-retrieval-artifact@2" if referenced else "supervisor-code-retrieval-context@1",
        "task_id": owner["task_id"], "query_text": "Authored repeated public query.",
        "source_sha256": {}, "result": {}}
    if referenced:
        raw["snapshot_ref"] = {"path": ".runtime/authored-retrieval-snapshot.json", "sha256": sha(snapshot),
            "bytes": len(snapshot), "index_id": "authored-snapshot-transport-only"}
        data["repository/.runtime/authored-retrieval-snapshot.json"] = ("retrieval-snapshot", snapshot)
    else:
        raw["snapshot"] = json.loads(snapshot)
    raw = evidence._encode(raw)
    binding = {"artifact": ".runtime/authored-retrieval.json", "expected_sha256": sha(raw), "task_id": owner["task_id"]}
    descriptor = {"schema": "supervisor-semantic-decision-launch-policy@3", "task_id": owner["task_id"],
        "parent_artifact": owner["parent_artifact"], "parent_sha256": owner["parent_sha256"], "policy": {},
        "retrieval_query_binding": binding}
    descriptor_raw = evidence._encode(descriptor)
    owner["policy_sha256"] = sha(evidence._encode({}))
    owner["launch_policy_sha256"] = sha(descriptor_raw)
    owner["launch_policy_artifact"] = "/app/.runtime/semantic-decision-policies/" + sha(descriptor_raw) + ".json"
    dialogue["policy_sha256"] = owner["policy_sha256"]
    dialogue["worker_launch"].update(launch_policy_artifact=owner["launch_policy_artifact"],
        launch_policy_sha256=owner["launch_policy_sha256"], retrieval_query_binding=binding)
    lineage = evidence._encode({"schema": "supervisor-semantic-typed-nomination-lineage@2",
        "origin_nomination_text": origin, "typed_nomination": {
            "schema": "supervisor-semantic-typed-owner-nomination@2", "nomination": {}, "policy": {},
            "launch_policy": descriptor, "receipt": {"schema": "supervisor-semantic-typed-owner-policy-receipt@2",
                "retrieval_query_binding": binding}}})
    owner["typed_nomination"].update(lineage_sha256=sha(lineage),
        lineage_artifact="/app/.runtime/semantic-typed-nominations/" + sha(lineage) + ".json")
    data["repository/.runtime/semantic-typed-nominations/" + sha(lineage) + ".json"] = ("typed-nomination-lineage", lineage)
    data["repository/.runtime/semantic-decision-policies/" + sha(descriptor_raw) + ".json"] = ("launch-policy", descriptor_raw)
    data["repository/.runtime/authored-retrieval.json"] = ("raw-retrieval", raw)
    data["replay-request.json"] = ("replay-request", evidence._encode(request))


def test_typed_transfer_profile_requires_exact_closed_witnesses_without_claiming_native_verification(tmp_path):
    archive, digest, manifest_digest, manifest = transfer(tmp_path, change=_authored_typed_transfer,
                                                        profile=evidence.TYPED_PROFILE)
    assert evidence.verify_evidence_archive(archive=archive, expected_sha256=digest,
        manifest_sha256=manifest_digest) == manifest
    assert manifest["current_source_freshness_claimed"] is False


@pytest.mark.parametrize("referenced", [False, True])
def test_typed_retrieval_transfer_binds_exact_launch_lineage_and_raw_snapshot(tmp_path, referenced):
    archive, digest, manifest_digest, manifest = transfer(tmp_path,
        change=lambda data: _authored_typed_retrieval_transfer(data, referenced=referenced),
        profile=evidence.TYPED_RETRIEVAL_PROFILE)
    assert evidence.verify_evidence_archive(archive=archive, expected_sha256=digest,
        manifest_sha256=manifest_digest) == manifest
    assert manifest["provider_calls"] == 0 and manifest["current_source_freshness_claimed"] is False
    assert evidence._manifest_limit(evidence.TYPED_RETRIEVAL_PROFILE) == 4 * 1024 * 1024


@pytest.mark.parametrize("mutation", ["legacy-profile", "base-profile", "missing-binding", "mixed-bindings",
    "foreign-task", "raw-bytes", "snapshot-bytes", "snapshot-role", "extra-raw", "lineage-revision"])
def test_typed_retrieval_transfer_refuses_cross_profile_parent_and_snapshot_changes(tmp_path, mutation):
    def change(data):
        from copy import deepcopy
        _authored_typed_retrieval_transfer(data)
        request = json.loads(data["replay-request.json"][1])
        dialogue = request["decision_report"]["semantic_decision_dialogues"][0]
        owner = request["decision_report"]["semantic_decision_owner_bindings"][0]
        if mutation == "missing-binding":
            del dialogue["worker_launch"]["retrieval_query_binding"]
        elif mutation == "mixed-bindings":
            other = deepcopy(dialogue)
            del other["worker_launch"]["retrieval_query_binding"]
            request["decision_report"]["semantic_decision_dialogues"].append(other)
        elif mutation == "foreign-task":
            dialogue["worker_launch"]["retrieval_query_binding"]["task_id"] = "OTHER"
        elif mutation in {"raw-bytes", "snapshot-bytes", "snapshot-role"}:
            name = "repository/.runtime/authored-retrieval" + (".json" if mutation == "raw-bytes" else "-snapshot.json")
            role, raw = data[name]
            data[name] = ("semantic-cas", raw) if mutation == "snapshot-role" else (role, raw + b" ")
        elif mutation == "extra-raw":
            data["repository/.runtime/unrelated-private.json"] = ("raw-retrieval", b"PRIVATE-NOT-A-BOUND-PARENT")
        elif mutation == "lineage-revision":
            name = "repository/.runtime/semantic-typed-nominations/" + owner["typed_nomination"]["lineage_sha256"] + ".json"
            role, raw = data.pop(name)
            lineage = json.loads(raw)
            lineage["schema"] = "supervisor-semantic-typed-nomination-lineage@1"
            raw = evidence._encode(lineage)
            owner["typed_nomination"].update(lineage_sha256=sha(raw),
                lineage_artifact="/app/.runtime/semantic-typed-nominations/" + sha(raw) + ".json")
            data["repository/.runtime/semantic-typed-nominations/" + sha(raw) + ".json"] = (role, raw)
        data["replay-request.json"] = ("replay-request", evidence._encode(request))
    profile = evidence.PROFILE if mutation == "base-profile" else evidence.TYPED_PROFILE if mutation == "legacy-profile" else evidence.TYPED_RETRIEVAL_PROFILE
    archive, digest, manifest_digest, _ = transfer(tmp_path, change=change, profile=profile)
    with pytest.raises(ValueError):
        evidence.verify_evidence_archive(archive=archive, expected_sha256=digest,
            manifest_sha256=manifest_digest)


@pytest.mark.parametrize("mutation", ["base-profile", "false-flag", "missing-origin", "wrong-origin-role"])
def test_typed_transfer_cannot_downgrade_expectation_or_escape_closed_profile(tmp_path, mutation):
    def change(data):
        _authored_typed_transfer(data)
        if mutation == "false-flag":
            request = json.loads(data["replay-request.json"][1])
            request["decision_report"]["semantic_decision_typed_nomination_expected"] = False
            data["replay-request.json"] = ("replay-request", evidence._encode(request))
        if mutation in {"missing-origin", "wrong-origin-role"}:
            path = next(name for name in data if name.startswith("typed-origins/"))
            if mutation == "missing-origin":
                del data[path]
            else:
                data[path] = ("semantic-cas", data[path][1])
    archive, digest, manifest_digest, _ = transfer(tmp_path, change=change,
        profile=evidence.PROFILE if mutation == "base-profile" else evidence.TYPED_PROFILE)
    with pytest.raises(ValueError):
        evidence.verify_evidence_archive(archive=archive, expected_sha256=digest,
            manifest_sha256=manifest_digest)


@pytest.mark.parametrize("profile", [evidence.PROFILE, evidence.TYPED_PROFILE])
def test_missing_replay_request_is_deliberate_value_error_before_witness_lookup(tmp_path, profile):
    archive, digest, manifest_digest, _ = transfer(tmp_path, profile=profile,
        change=lambda data: data.pop("replay-request.json"))
    with pytest.raises(ValueError, match="replay_request_required"):
        evidence.verify_evidence_archive(archive=archive, expected_sha256=digest,
            manifest_sha256=manifest_digest)


@pytest.mark.parametrize("profile,target,accepted", [
    (evidence.PROFILE, evidence.MAX_MANIFEST_BYTES, True),
    (evidence.PROFILE, evidence.MAX_MANIFEST_BYTES + 1, False),
    (evidence.TYPED_PROFILE, evidence.MAX_MANIFEST_BYTES + 1, True),
    (evidence.TYPED_PROFILE, evidence.MAX_TYPED_MANIFEST_BYTES, True),
    (evidence.TYPED_PROFILE, evidence.MAX_TYPED_MANIFEST_BYTES + 1, False),
])
def test_profile_manifest_capacity_is_closed_at_archive_and_replay_readers(tmp_path, profile, target, accepted):
    archive, digest, manifest_digest, manifest = transfer(tmp_path, profile=profile,
        change=_authored_typed_transfer if profile == evidence.TYPED_PROFILE else None,
        manifest_target_bytes=target)
    raw_manifest = evidence._encode(manifest)
    held = tmp_path / "standalone-manifest.json"
    held.write_bytes(raw_manifest)
    if accepted:
        assert evidence.verify_evidence_archive(archive=archive, expected_sha256=digest,
            manifest_sha256=manifest_digest) == manifest
        assert evidence._read_manifest(held) == manifest
    else:
        with pytest.raises(ValueError):
            evidence.verify_evidence_archive(archive=archive, expected_sha256=digest,
                manifest_sha256=manifest_digest)
        with pytest.raises(ValueError):
            evidence._read_manifest(held)


@pytest.mark.parametrize("mutation", ["missing-ref", "wrong-pin", "foreign-path", "wrong-task", "wrong-parent", "wrong-root"])
def test_actual_native_metadata_requires_independently_bound_current_descriptor(native_typed_evidence, mutation):
    native = native_typed_evidence
    if mutation == "missing-ref":
        native.context_result.pop("initial_context_descriptor")
    elif mutation == "wrong-pin":
        native.context_result["initial_context_descriptor"]["sha256"] = "0" * 64
    elif mutation == "foreign-path":
        native.context_result["initial_context_descriptor"]["artifact"] = ".runtime/private.json"
    else:
        path = native.repository / native.context_result["initial_context_descriptor"]["artifact"]
        descriptor = json.loads(path.read_bytes())
        if mutation == "wrong-task":
            descriptor["task_alias"] = "OTHER"
        elif mutation == "wrong-parent":
            descriptor["metadata"]["Semantic context sha256"] = "0" * 64
        else:
            descriptor["semantic"]["semantic_root_cid"] = "foreign-root"
        raw = evidence._encode(descriptor)
        path.write_bytes(raw)
        native.context_result["initial_context_descriptor"]["sha256"] = sha(raw)
    with pytest.raises(ValueError):
        _native_public_blocks(native)


@pytest.mark.parametrize("profile", [evidence.PROFILE, evidence.TYPED_PROFILE])
@pytest.mark.parametrize("field", ["bytes", "manifest_bytes"])
def test_receipt_capacity_uses_selected_profile_without_expanding_base(tmp_path, profile, field):
    state = Path("/opt/ipfs-supervisor/state/authored-capacity")
    receipt = {"schema": evidence.RECEIPT_SCHEMA, "status": "exported", "artifact": str(state) + "-context-evidence.tar",
        "bytes": 1, "sha256": "1" * 64, "manifest_sha256": "2" * 64, "manifest_bytes": 1,
        "file_count": 1, "content_bytes": 1, "repository_origin": "/app", "state_origin": str(state),
        "workspace_root_origin": "/opt/ipfs-supervisor/worktrees", "profile": profile, "provider_calls": 0,
        "execution_authority": False, "completion_authority": False}
    limit = evidence._archive_limit(profile) if field == "bytes" else evidence._manifest_limit(profile)
    receipt[field] = limit
    assert evidence.validate_evidence_receipt(receipt, state=state) == receipt
    receipt[field] += 1
    with pytest.raises(ValueError, match="export_receipt_bound"):
        evidence.validate_evidence_receipt(receipt, state=state)
    assert evidence._manifest_limit(evidence.PROFILE) == 1024 * 1024
    assert evidence._manifest_limit(evidence.TYPED_PROFILE) == 4 * 1024 * 1024
