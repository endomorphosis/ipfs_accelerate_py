"""Waited local child -> exact Git candidate -> native advisory formal artifacts.

Only container UID/path admission is authored. Native semantic production,
source capture, child wait, shutdown, capsules and formal grammar run normally.
No provider, target program, solver or accepted-state publication is invoked.
"""
from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_round_context as module
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_broker as brokers
from test.api.test_semantic_router_translation import native, _program_prompt  # noqa: F401

SOURCE = b"import numpy as np\ndef choose(values):\n    return np.argmax(values)\n"
CHANGED = SOURCE.replace(b"choose(values)", b"choose(values, offset=0)").replace(b"argmax", b"argmin")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def source_service(native, tmp_path, monkeypatch):
    repository = native[0]
    (repository / "mod.py").write_bytes(SOURCE)
    repository, _, prompt = _program_prompt(native, program=["mod.py"])
    def git(*args):
        subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True)
    git("init", "-q")
    git("add", "mod.py", "harness.py")
    git("-c", "user.name=Authored Fixture", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "authored baseline")
    workspace = tmp_path / "candidate"
    git("worktree", "add", "--detach", "-q", str(workspace))
    temporary = tempfile.TemporaryDirectory(prefix="candidate-round-")
    boundary = Path(temporary.name) / "boundary.json"
    boundary.write_text("authored boundary")
    (boundary.parent / "worker-observations").mkdir(mode=0o750)
    monkeypatch.setattr(guard, "_owner_boundary", lambda path: (
        dict(owner_uid=os.getuid(), worker_uid=os.getuid() + 1), "b" * 64))
    monkeypatch.setattr(guard, "_protected_directory", lambda *args, **kwargs: None)
    service = brokers.WorkerSourceBroker(repository=repository, workspace=workspace,
        boundary=boundary, prompt=prompt, timeout_seconds=90)
    yield service
    if service._process is not None and service._process.poll() is None:
        service._process.kill()
        service._process.wait()
    service.close()
    temporary.cleanup()


def waited(service, *, code=3, close=True):
    child = subprocess.Popen([sys.executable, "-c", "import sys;sys.stdin.read();sys.exit(int(sys.argv[1]))",
        str(code)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    service.note_spawned(child)
    child.communicate(b"authored fixture", timeout=10)
    assert child.returncode == code
    if close:
        service.close()


def prepare(service, **options):
    return module.prepare_candidate_round_context(**dict(source_broker=service,
        prior_returncode=3, round_number=2, absolute_deadline=time.monotonic()+90) | options)


def test_native_round_rebuild_preserves_baseline_and_exact_advisory_pins(source_service):
    service = source_service
    waited(service)
    before = prepare(service)
    (service.workspace / "mod.py").write_bytes(CHANGED)
    (service.workspace / "unadmitted.py").write_text("raise RuntimeError('must not execute or scan')\n")
    after = prepare(service)
    metadata = after.revalidate()
    old_formal = json.loads(before.artifacts["formal"])["reports"]["mod.py"]
    formal = json.loads(after.artifacts["formal"])["reports"]["mod.py"]
    old_definitions = {row["definition"] for row in old_formal["projections"]["lean"]["bindings"]}
    definitions = {row["definition"] for row in formal["projections"]["lean"]["bindings"]}
    assert "SelectionMax" in old_definitions and "SelectionMin" not in old_definitions
    assert "SelectionMin" in definitions and "SelectionMax" not in definitions
    assert before.metadata["semantic_root_cid"] != metadata["semantic_root_cid"]
    assert before.metadata["workspace_generation"] != metadata["workspace_generation"]
    capsules = json.loads(after.artifacts["capsules"])["capsules"]
    assert any("offset" in json.dumps(row["signature"]) for row in capsules)
    assert all(row["source_slice_path"] == "mod.py" for row in capsules)
    assert "excluded_harness_symbol" not in json.dumps(capsules)
    assert set(metadata["source_hashes"]) == {"mod.py", "harness.py"}
    assert metadata["source_hashes"]["mod.py"] == sha(CHANGED)
    assert metadata["program_paths"] == ["mod.py"] and metadata["source_files"] == 2
    support = next(row for row in metadata["coverage"] if row["path"] == "harness.py")
    assert support["role"] == "support" and support["formal_status"] == "not_selected_support"
    assert metadata["pending_obligation_count"] > 0
    assert metadata["prior_session_id"] == service.descriptor["session_id"]
    assert metadata["descriptor_sha256"] == service.descriptor_sha256
    assert metadata["native_prompt_sha256"] == sha(after.native_prompt)
    assert metadata["task_id"] == "TASK-1" and metadata["round_number"] == 2
    assert metadata["child_wait_observed"] is metadata["source_service_stopped"] is True
    for key in ("proof_authority", "execution_authority", "completion_authority",
                "baseline_evidence_applicable", "next_round_admitted", "accepted_index_published",
                "descendant_cleanup_verified", "source_scope_expanded"):
        assert metadata[key] is False
    assert metadata["provider_calls"] == metadata["prover_calls"] == metadata["training_steps"] == 0
    assert metadata["source_obligations_discharged"] == metadata["model_inference_calls"] == 0
    assert (service.repository / "mod.py").read_bytes() == SOURCE
    assert service.prompt.encode() == after.native_prompt
    assert b"return np.argmin" not in after.metadata_bytes and after.native_prompt not in after.metadata_bytes
    for name, raw in after.artifacts.items():
        assert metadata["artifacts"][name] == dict(sha256=sha(raw), bytes=len(raw))
    with pytest.raises(ValueError):
        before.revalidate()
    with pytest.raises(ValueError, match="stale"):
        service.views.admission()
    with pytest.raises(FrozenInstanceError):
        after.metadata_bytes = b"changed"
    with pytest.raises(TypeError):
        after.artifacts["formal"] = b"changed"
    metadata["proof_authority"] = True
    assert after.metadata["proof_authority"] is False


@pytest.mark.parametrize("location,path,delete", [
    ("workspace", "mod.py", False), ("workspace", "harness.py", False),
    ("repository", "mod.py", False), ("repository", "harness.py", False),
    ("workspace", "mod.py", True), ("workspace", "harness.py", True),
])
def test_revalidate_refuses_source_drift_and_deletion(source_service, location, path, delete):
    waited(source_service)
    snapshot = prepare(source_service)
    target = getattr(source_service, location) / path
    if delete:
        target.unlink()
    else:
        target.write_bytes(target.read_bytes()+b"\n# changed after preparation\n")
    with pytest.raises((ValueError, OSError)):
        snapshot.revalidate()


def test_build_race_does_not_return_snapshot(source_service, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_artifacts as artifacts
    waited(source_service)
    native_build = artifacts.build_candidate_artifacts
    def racing(**kwargs):
        result = native_build(**kwargs)
        (source_service.workspace / "mod.py").write_bytes(CHANGED)
        return result
    monkeypatch.setattr(artifacts, "build_candidate_artifacts", racing)
    with pytest.raises(ValueError):
        prepare(source_service)


def test_revalidate_refuses_producer_drift(source_service, monkeypatch):
    waited(source_service)
    snapshot = prepare(source_service)
    monkeypatch.setattr(module, "_producers", lambda: b'{"changed":"producer"}')
    with pytest.raises(ValueError, match="producer sources changed"):
        snapshot.revalidate()


def test_round_requires_registered_waited_failure_and_closed_service(source_service):
    with pytest.raises(ValueError, match="closed and stopped"):
        prepare(source_service)
    waited(source_service, close=False)
    with pytest.raises(ValueError, match="closed and stopped"):
        prepare(source_service)
    source_service.close()
    with pytest.raises(ValueError, match="already be waited"):
        prepare(source_service, prior_returncode=4)
    assert prepare(source_service).metadata["prior_returncode"] == 3


def test_closed_broker_without_child_is_not_prior_work(source_service):
    source_service.close()
    with pytest.raises(ValueError, match="already be waited"):
        prepare(source_service)


@pytest.mark.parametrize("options", [
    {"round_number": True}, {"round_number": 0}, {"round_number": 1.5},
    {"absolute_deadline": float("nan")}, {"absolute_deadline": float("inf")},
    {"absolute_deadline": True}, {"absolute_deadline": -1}, {"absolute_deadline": 10**500},
])
def test_invalid_budget_or_round_refused_before_owner_or_native_work(monkeypatch, options):
    monkeypatch.setattr(module, "_closed_baseline", lambda *_: pytest.fail("invalid input reached owner"))
    with pytest.raises(ValueError):
        prepare(object(), **options)


def test_success_and_boolean_returncodes_refused(source_service):
    waited(source_service, code=0)
    for code in (0, True, False, None, -9, 256):
        with pytest.raises(ValueError, match="positive waited"):
            prepare(source_service, prior_returncode=code)


def test_candidate_preparation_does_not_open_ambient_metadata_owner(source_service, monkeypatch, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.runtime import supervisor_meta_index as meta
    waited(source_service)
    database, lake = tmp_path / "not-opened.duckdb", tmp_path / "not-opened-lake"
    monkeypatch.setenv(meta.ENV_DUCKDB, str(database))
    monkeypatch.setenv(meta.ENV_DUCKLAKE, str(lake))
    monkeypatch.setattr(meta, "_connect", lambda *_: pytest.fail("candidate preparation opened metadata"))
    snapshot = prepare(source_service)
    assert snapshot.revalidate()["accepted_index_published"] is False
    assert not database.exists() and not lake.exists()


def test_retained_non_popen_child_is_not_waited_authority(source_service):
    from types import SimpleNamespace
    waited(source_service)
    actual = source_service._process
    source_service._process = SimpleNamespace(pid=actual.pid, returncode=3, poll=lambda: 3)
    try:
        with pytest.raises(ValueError, match="already be waited"):
            prepare(source_service)
    finally:
        source_service._process = actual


def test_deadline_expiry_during_native_build_refuses_snapshot(source_service, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_artifacts as artifacts
    waited(source_service)
    current = [time.monotonic()]
    monkeypatch.setattr(module.time, "monotonic", lambda: current[0])
    native_build = artifacts.build_candidate_artifacts
    def expires(**kwargs):
        result = native_build(**kwargs)
        current[0] += 100
        return result
    monkeypatch.setattr(artifacts, "build_candidate_artifacts", expires)
    with pytest.raises(ValueError, match="deadline exhausted"):
        prepare(source_service, absolute_deadline=current[0]+90)


def test_retained_descriptor_or_program_population_tampering_refused(source_service):
    waited(source_service)
    source_service.candidate_context.program_paths = ("harness.py", "mod.py")
    with pytest.raises(ValueError, match="program population"):
        prepare(source_service)
    source_service.candidate_context.program_paths = ("mod.py",)
    source_service.descriptor["native_prompt_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="descriptor changed"):
        prepare(source_service)
