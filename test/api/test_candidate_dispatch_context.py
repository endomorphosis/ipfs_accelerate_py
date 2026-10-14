"""Real native candidate handoff and pre-provider source/grammar contracts.

Only container UID/ancestor admission and the fixed cleanup executable are
authored seams. Real Git, failed child, native artifacts, typed owner admission
and protected publication run normally. No provider or target tests run.
"""
from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import candidate_dispatch_context as module
from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_dispatch as owner
from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_round_prompt as prompts
from ipfs_accelerate_py.agent_supervisor.runtime import candidate_reply_contract as replies
from test.api.test_owner_candidate_round_context import source_service, native, SOURCE, CHANGED  # noqa: F401


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def worker_seams(monkeypatch, prepared):
    """Authored single-UID admission; retain exact files, schema and socket pins."""
    broker = prepared.source_broker
    monkeypatch.setattr(module.guard, "_worker_boundary", lambda *args:
        dict(owner_uid=broker.owner_uid, worker_uid=broker.worker_uid))
    access = os.access
    monkeypatch.setattr(module.os, "access", lambda path, mode: False
        if Path(path) == prepared.handoff.path and mode == os.W_OK else access(path, mode))
    def load(path, digest, *, workspace):
        assert Path(path) == broker.descriptor_path and workspace == broker.workspace
        raw = module.guard._read(Path(path), module.protocol.MAX_FRAME_BYTES)
        info = Path(path).lstat()
        if sha(raw) != digest or stat.S_IMODE(info.st_mode) != 0o440:
            raise ValueError("authored descriptor bytes or mode changed")
        value = module.protocol.validate_descriptor(json.loads(raw))
        socket_info = broker.socket_path.lstat()
        if (socket_info.st_dev, socket_info.st_ino) != (value["socket_device"], value["socket_inode"]):
            raise ValueError("authored socket changed")
        return value
    monkeypatch.setattr(module.protocol, "load_descriptor", load)


@pytest.fixture
def dispatch_case(source_service, monkeypatch):
    broker = source_service
    broker.original_requirements = prompts.capture_native_only_requirements(source_broker=broker)
    launch = owner.capture_original_coding_launch(source_broker=broker,
        argv=["--provider", "codex_cli", "--model", "gpt-6.1-sol", "--timeout", "180",
              "--semantic-repository", str(broker.repository), "--source-view-profile", broker._binding["profile"]],
        absolute_deadline=time.monotonic() + 179)
    child = subprocess.Popen([sys.executable, "-c",
        "import sys;from pathlib import Path;sys.stdin.read();Path('mod.py').write_bytes(bytes.fromhex(sys.argv[1]));raise SystemExit(7)",
        CHANGED.hex()], cwd=broker.workspace, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    broker.note_spawned(child)
    child.communicate(b"authored first attempt", timeout=10)
    assert child.returncode == 7
    monkeypatch.setattr(owner, "_isolated_cleanup", lambda: subprocess.CompletedProcess(["authored-cleanup"], 0))
    retirement = owner.retire_coding_worker(launch=launch, process=child, source_broker=broker)
    prepared = owner.prepare_second_coding_worker(launch=launch, retired_worker=retirement)
    worker_seams(monkeypatch, prepared)
    current = prepared.source_broker
    request = dict(prompt=prepared.handoff.provider_prompt.decode(), repository=current.repository,
        workspace=current.workspace, boundary=current.boundary,
        boundary_sha256=current.descriptor["boundary_manifest_sha256"], handoff=prepared.handoff.path,
        handoff_sha256=prepared.handoff.sha256, descriptor=current.descriptor,
        descriptor_sha256=current.descriptor_sha256, descriptor_path=current.descriptor_path,
        launch_settings=launch.launch_settings)
    try:
        yield SimpleNamespace(original=broker, launch=launch, retired=retirement,
                              prepared=prepared, request=request)
    finally:
        prepared.close()
        prepared.handoff.close()


def test_native_candidate_handoff_is_separate_from_baseline_and_preserves_all_requirements(dispatch_case):
    case = dispatch_case
    prepared = case.prepared
    metadata = prepared.handoff.metadata
    observation = module.validate_candidate_dispatch_context(**case.request)
    assert prepared.handoff.path.stat().st_mode & 0o777 == 0o440
    assert metadata["source_hashes"]["mod.py"] == sha(CHANGED)
    assert metadata["baseline_source_hashes"]["mod.py"] == sha(SOURCE)
    assert observation["candidate_semantic_root_cid"] != prepared.source_broker.descriptor["semantic_root_cid"]
    assert observation["workspace_generation"] == prepared.snapshot.metadata["workspace_generation"]
    assert observation["round_context_sha256"] == sha(prepared.snapshot.metadata_bytes)
    assert observation["candidate_artifacts"] == prepared.snapshot.metadata["artifacts"]
    assert observation["remaining_timeout_seconds"] > 90
    assert type(observation["remaining_timeout_seconds"]) is int
    assert metadata["expires_monotonic_ns"] - metadata["issued_monotonic_ns"] <= module.MAX_LIFETIME_NS
    assert observation["owner_private_state_replayed_here"] is False
    assert observation["producer_validation_inherited"] is True
    assert observation["neural_inference_replayed"] is False
    assert all(observation[key] == value for key, value in module.FLAGS.items())
    assert json.loads(prepared.handoff.provider_prompt)["native_prompt"] == case.original.prompt
    formal = json.loads(prepared.snapshot.artifacts["formal"])["reports"]["mod.py"]
    definitions = {row["definition"] for row in formal["projections"]["lean"]["bindings"]}
    assert "SelectionMin" in definitions and "SelectionMax" not in definitions
    assert CHANGED not in prepared.handoff.metadata_bytes
    assert case.original.prompt.encode() not in prepared.handoff.metadata_bytes
    with pytest.raises(FrozenInstanceError):
        prepared.handoff.sha256 = "f" * 64


@pytest.mark.parametrize("change", ["source", "support", "deleted", "canonical", "prompt", "settings",
    "digest", "mode", "symlink", "descriptor", "expired", "symbolic_pin", "population"])
def test_handoff_refuses_drift_rebinding_or_unprotected_files(dispatch_case, change, monkeypatch):
    case, request = dispatch_case, dict(dispatch_case.request)
    path = case.prepared.handoff.path
    if change in {"source", "support", "deleted", "canonical"}:
        root = case.original.repository if change == "canonical" else case.original.workspace
        target = root / ("harness.py" if change == "support" else "mod.py")
        target.unlink() if change == "deleted" else target.write_bytes(target.read_bytes() + b"\n# drift\n")
    elif change == "prompt":
        request["prompt"] += "\nworker retry mutation"
    elif change == "settings":
        request["launch_settings"] = {**request["launch_settings"], "model": "different-model"}
    elif change == "digest":
        request["handoff_sha256"] = "f" * 64
    elif change == "mode":
        path.chmod(0o640)
    elif change == "symlink":
        alias = path.with_name("candidate-dispatch-" + "f" * 32 + ".json")
        alias.symlink_to(path)
        request["handoff"] = alias
    elif change == "descriptor":
        request["descriptor"] = {**request["descriptor"], "workspace_generation": "f" * 64}
    else:
        value = json.loads(path.read_bytes())
        if change == "expired":
            monkeypatch.setattr(module.time, "monotonic_ns", lambda: value["expires_monotonic_ns"] + 1)
        elif change == "symbolic_pin":
            value["candidate_artifacts"]["formal"]["sha256"] = "f" * 64
        else:
            value["source_hashes"].pop("harness.py")
        if change != "expired":
            path.chmod(0o640)
            raw = module.canonical_json(value)
            path.write_bytes(raw)
            path.chmod(0o440)
            request["handoff_sha256"] = sha(raw)
    with pytest.raises((ValueError, OSError)):
        module.validate_candidate_dispatch_context(**request)


def test_serialized_checkpoint_cannot_publish_dispatch_and_postedit_ack_remains_literal(dispatch_case):
    case = dispatch_case
    prepared = case.prepared
    with pytest.raises(ValueError, match="exact retained"):
        module.prepare_candidate_dispatch_handoff(snapshot=prepared.snapshot.metadata,
            prompt=prepared.prompt, source_broker=prepared.source_broker, admission=prepared.admission)
    with pytest.raises(ValueError, match="exact retained"):
        module.prepare_candidate_dispatch_handoff(snapshot=prepared.snapshot,
            prompt=prepared.prompt, source_broker=prepared.source_broker, admission=prepared.admission._binding_bytes)
    observation = module.validate_candidate_dispatch_context(**case.request)
    text = json.dumps(dict(schema=replies.SCHEMA, status="candidate_ready",
        **{key: observation[key] for key in replies.BINDING_FIELDS}))
    (case.original.workspace / "mod.py").write_bytes(CHANGED + b"\n# expected new round edit\n")
    assert replies.validate_candidate_completion_reply(text, binding=observation) == text
    with pytest.raises(ValueError, match="stale"):
        module.validate_candidate_dispatch_context(**case.request)


def test_handoff_cleanup_preserves_replaced_inode(dispatch_case):
    handoff = dispatch_case.prepared.handoff
    path = handoff.path
    retained = path.with_name(path.name + ".retained")
    path.rename(retained)
    path.write_bytes(b"different owner file")
    handoff.close()
    assert path.read_bytes() == b"different owner file"
    path.unlink()
    retained.rename(path)
    handoff.close()
    handoff.close()
    assert not path.exists()


def test_worker_gate_never_replays_native_owner_state(dispatch_case, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_round_context as snapshots
    def forbidden(*args, **kwargs):
        pytest.fail("worker attempted owner snapshot replay")
    monkeypatch.setattr(snapshots.OwnerCandidateRoundContext, "revalidate", forbidden)
    assert module.validate_candidate_dispatch_context(**dispatch_case.request)["owner_private_state_replayed_here"] is False


def test_failed_publication_rolls_back_only_the_new_handoff(dispatch_case, monkeypatch):
    prepared = dispatch_case.prepared
    before = set(prepared.handoff.path.parent.iterdir())
    original = owner.CandidateDispatchAdmission.revalidate
    calls = []
    def changed(self, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise ValueError("authored admission drift after publication")
        return original(self, **kwargs)
    monkeypatch.setattr(owner.CandidateDispatchAdmission, "revalidate", changed)
    with pytest.raises(ValueError, match="authored admission drift"):
        module.prepare_candidate_dispatch_handoff(snapshot=prepared.snapshot, prompt=prepared.prompt,
            source_broker=prepared.source_broker, admission=prepared.admission)
    assert len(calls) == 2
    assert set(prepared.handoff.path.parent.iterdir()) == before
    assert prepared.handoff.path.read_bytes() == prepared.handoff.metadata_bytes


def test_launch_setting_types_are_exact(dispatch_case):
    request = dict(dispatch_case.request)
    request["launch_settings"] = {**request["launch_settings"], "grok_coding_max_turns": 128.0}
    with pytest.raises(ValueError, match="launch settings"):
        module.validate_candidate_dispatch_context(**request)
