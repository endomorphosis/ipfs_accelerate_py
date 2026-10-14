"""Native Doctor residual -> original custody -> freshly bound candidate advice.

Actual signed admission, Doctor diagnostics, Git, failed child wait and candidate
semantic producers run locally. Only this host's missing UID/path boundary is
authored. No model, prover, training or task-program execution is requested.
"""
from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import doctor_candidate_context as doctor
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_residual_context as residual
from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_round_prompt as prompts
from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_dispatch as dispatch
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_broker as brokers
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, build_text_context_references, render_context_capsule)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from test.api.test_doctor_residual_context import retained, scenario  # noqa: F401
from test.api.test_owner_candidate_round_context import waited, prepare
from test.api.test_candidate_round_native_pipeline import _prepared


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def doctor_service(retained, monkeypatch):
    _, _, _, request = retained
    root, workspace = request["repository"], request["workspace"]
    task_id = json.loads(request["artifact"].read_bytes())["task_id"]
    output = root / ".runtime/semantic-doctor-candidate"
    prepare_semantic_context(repository=root, paths=["answer.py"], program_paths=["answer.py"],
        required_raw_paths=["answer.py"], objective="Repair the admitted answer", task_id=task_id, output=output)
    artifact = output / "worker-context.json"
    refs = build_text_context_references(artifact.read_text(), reference_prefix="semantic-context",
        kind="semantic-context", path=artifact.relative_to(root).as_posix(),
        repository_id="repo:test", tree_id="tree:test", required=True, chunk_bytes=1201)
    compiled = ContextCompiler(ContextBudget(max_input_tokens=32768, max_items=128,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(
        repository_id="repo:test", tree_id="tree:test", objective_id=task_id,
        objective_revision="sha256:task", policy_id="policy:test", policy_revision="sha256:policy",
        caller="supervisor:test", stage="implementation", goal={"id": task_id},
        authority={"mode": "candidate_only", "completion_authority": False},
        scope={"allowed_paths": ["answer.py"]}, acceptance={"criteria": ["pending native validation"]}, evidence=refs)
    with tempfile.TemporaryDirectory(prefix="doctor-candidate-") as folder:
        boundary = Path(folder) / "boundary.json"
        boundary.write_text("authored local boundary")
        (boundary.parent / "worker-observations").mkdir(mode=0o750)
        monkeypatch.setattr(guard, "_owner_boundary", lambda path: (
            dict(owner_uid=os.getuid(), worker_uid=os.getuid()+1), "b"*64))
        monkeypatch.setattr(guard, "_protected_directory", lambda *args, **kwargs: None)
        service = brokers.WorkerSourceBroker(repository=root, workspace=workspace,
            boundary=boundary, prompt=render_context_capsule(compiled.capsule), timeout_seconds=90)
        try:
            yield service, {key: request[key] for key in ("artifact", "expected_sha256", "task_cid")}
        finally:
            service.close()


def capture(service, request):
    return prompts.capture_native_only_requirements(source_broker=service, doctor_request=request)


def candidate(service):
    waited(service)
    path = service.workspace / "answer.py"
    path.write_text(path.read_text().replace("process(count=2)", "process(amount=2)"))
    return prepare(service)


@pytest.mark.parametrize("retained", ["secret_analysis"], indirect=True)
def test_native_capture_carries_exact_obligations_to_new_generation_without_admitting_dispatch(doctor_service):
    service, request = doctor_service
    original = capture(service, request)
    service.original_requirements = original
    captured = original._doctor_requirements
    with pytest.raises(ValueError, match="cannot be omitted or substituted"):
        prompts.capture_native_only_requirements(source_broker=service)
    text, receipt = residual.load_doctor_residual_advisory(**request, repository=service.repository,
        workspace=service.workspace, prompt=service.prompt)
    assert captured.artifact_bytes == request["artifact"].read_bytes()
    assert captured.advisory_bytes == text.encode()
    assert json.loads(captured._receipt_bytes) == receipt
    assert captured.metadata["residual_budget"] == dict(max_calls=1, max_tokens=4096, max_rounds=1, max_cost_units=1)
    snapshot = candidate(service)
    with pytest.raises(ValueError, match="stale"):
        residual.load_doctor_residual_advisory(**request, repository=service.repository,
            workspace=service.workspace, prompt=service.prompt)
    prepared = doctor.prepare_doctor_candidate_context(snapshot=snapshot, requirements=original)
    metadata = prepared.revalidate()
    assert metadata["candidate_binding"]["workspace_generation"] == snapshot.metadata["workspace_generation"]
    assert metadata["candidate_binding"]["source_hashes"] == snapshot.metadata["source_hashes"]
    assert metadata["original_doctor"] == captured.metadata
    assert metadata["original_requirements_sha256"] == sha(original.metadata_bytes)
    assert metadata["original_advisory_sha256"] == sha(text.encode())
    assert prepared.advisory_bytes == captured.advisory_bytes
    assert metadata["candidate_inventory_paths"] == ["answer.py"]
    assert metadata["unchanged_inventory_paths"]
    assert metadata["full_doctor_source_inventory_current"] is metadata["original_obligations_preserved"] is True
    assert metadata["candidate_evidence_reproved"] is False
    assert metadata["continuation_status"] == "requires_independent_budget_enforcement"
    assert all(metadata[key] == expected for key, expected in doctor.FLAGS.items())
    assert text.encode() not in prepared.metadata_bytes
    assert captured.artifact_bytes not in prepared.advisory_bytes
    with pytest.raises(ValueError, match="independent residual budget"):
        prompts.prepare_candidate_round_prompt(snapshot=snapshot, requirements=original)
    with pytest.raises(FrozenInstanceError):
        prepared.advisory_bytes = b"replace constraints"
    prepared.metadata["additional_call_admitted"] = True
    assert prepared.metadata["additional_call_admitted"] is False
    with pytest.raises(ValueError, match="typed original"):
        doctor.prepare_doctor_candidate_context(snapshot=snapshot, requirements=original.metadata)


@pytest.mark.parametrize("retained", ["secret_analysis"], indirect=True)
def test_generation_bridge_refuses_full_inventory_drift_and_artifact_changes(doctor_service):
    service, request = doctor_service
    original = capture(service, request)
    snapshot = candidate(service)
    projection = doctor.prepare_doctor_candidate_context(snapshot=snapshot, requirements=original)
    # Changes outside the broker's selected inventory must still invalidate
    # Doctor's larger dependency inventory; merely rechecking candidate hashes
    # would miss this case.
    outside = projection.metadata["unchanged_inventory_paths"][0]
    targets = [service.workspace / outside, service.repository / outside, request["artifact"]]
    for target in targets:
        raw, mode = target.read_bytes(), target.stat().st_mode & 0o777
        target.chmod(0o600)
        target.write_bytes(raw+b"\nchanged after candidate binding\n")
        target.chmod(mode)
        try:
            with pytest.raises((ValueError, OSError)):
                projection.revalidate()
        finally:
            target.chmod(0o600)
            target.write_bytes(raw)
            target.chmod(mode)
        assert projection.revalidate()["full_doctor_source_inventory_current"] is True
    target = service.workspace / outside
    mode = target.stat().st_mode & 0o777
    target.chmod(mode ^ 0o100)
    try:
        with pytest.raises(ValueError, match="Doctor source"):
            projection.revalidate()
    finally:
        target.chmod(mode)
    # A source mismatch permanently invalidates the candidate generation;
    # restoring bytes must not revive that old typed snapshot.
    target = service.workspace / "answer.py"
    raw = target.read_bytes()
    target.write_bytes(raw+b"\n# source drift\n")
    with pytest.raises(ValueError):
        projection.revalidate()
    target.write_bytes(raw)
    with pytest.raises(ValueError, match="generation differs"):
        projection.revalidate()


@pytest.mark.parametrize("change", ["task", "digest", "source", "late"])
def test_initial_capture_requires_native_current_sources_and_prechild_custody(doctor_service, change):
    service, request = doctor_service
    request = dict(request)
    if change == "task":
        request["task_cid"] = "foreign-task"
    elif change == "digest":
        request["expected_sha256"] = "a"*64
    elif change == "source":
        (service.workspace / "answer.py").write_text("changed before capture\n")
    else:
        waited(service)
    with pytest.raises((ValueError, OSError)):
        capture(service, request)


def test_sealed_owner_broker_factory_captures_native_doctor_selection(doctor_service):
    service, request = doctor_service
    extra = ["--doctor-residual-artifact", str(request["artifact"]),
        "--doctor-residual-sha256", request["expected_sha256"], "--doctor-residual-task-cid", request["task_cid"]]
    captured = _prepared(service, extra)
    try:
        assert captured.original_requirements_unavailable_reason is None
        original = captured.original_requirements
        assert type(original._doctor_requirements) is doctor.OriginalDoctorRequirements
        assert original._doctor_requirements._source_broker is captured
        assert original.metadata["doctor_requirements"] == original._doctor_requirements.metadata
        for nomination in (None, {**request, "expected_sha256": "a"*64}):
            with pytest.raises(ValueError, match="cannot be omitted or substituted"):
                prompts.capture_native_only_requirements(source_broker=captured, doctor_request=nomination)
        # Omitting the residual CLI flags cannot admit even the first child of
        # an unsupported two-round launch carrying captured Doctor constraints.
        with pytest.raises(ValueError, match="independent residual budget"):
            dispatch.capture_original_coding_launch(source_broker=captured,
                argv=["--semantic-repository", str(captured.repository),
                    "--source-view-profile", "python-comments@1", "--timeout", "90"],
                absolute_deadline=time.monotonic()+89)
        assert captured._process is None and not captured._registered.is_set()
        snapshot = candidate(captured)
        projection = doctor.prepare_doctor_candidate_context(snapshot=snapshot, requirements=original)
        assert projection.revalidate()["full_doctor_source_inventory_current"] is True
        with pytest.raises(ValueError, match="independent residual budget"):
            prompts.prepare_candidate_round_prompt(snapshot=snapshot)
    finally:
        captured.close()
    with pytest.raises(ValueError, match="complete original Doctor"):
        _prepared(service, extra[:-2])


def test_capture_failure_closes_created_broker(doctor_service, monkeypatch):
    service, request = doctor_service
    seen = []
    native = prompts.capture_native_only_requirements
    def observe(**kwargs):
        seen.append(kwargs["source_broker"])
        return native(**kwargs)
    monkeypatch.setattr(prompts, "capture_native_only_requirements", observe)
    with pytest.raises(ValueError):
        _prepared(service, ["--doctor-residual-artifact", str(request["artifact"]),
            "--doctor-residual-sha256", "a"*64, "--doctor-residual-task-cid", request["task_cid"]])
    assert len(seen) == 1 and seen[0]._closed
    assert not seen[0].socket_path.exists() and not seen[0].descriptor_path.exists()


@pytest.mark.parametrize("retained", ["secret_analysis"], indirect=True)
@pytest.mark.parametrize("location", ["repository", "workspace"])
def test_late_snapshot_revalidation_cannot_hide_doctor_only_source_drift(doctor_service, monkeypatch, location):
    service, request = doctor_service
    original = capture(service, request)
    snapshot = candidate(service)
    captured = original._doctor_requirements
    outside = next(name for name in json.loads(captured.artifact_bytes)["source_manifest"]
        if name not in snapshot.metadata["source_hashes"])
    target = getattr(service, location) / outside
    validate, calls = type(snapshot).revalidate, []
    def drift(self):
        value = validate(self)
        if self is snapshot:
            calls.append(True)
            if len(calls) == 2:
                target.write_bytes(target.read_bytes()+b"\nlate Doctor dependency drift\n")
        return value
    monkeypatch.setattr(type(snapshot), "revalidate", drift)
    with pytest.raises(ValueError, match="complete Doctor source inventory"):
        captured.revalidate(snapshot=snapshot)
    assert len(calls) == 2


@pytest.mark.parametrize("retained", ["secret_analysis"], indirect=True)
def test_initial_capture_brackets_doctor_only_dependencies(doctor_service, monkeypatch):
    service, request = doctor_service
    payload = json.loads(request["artifact"].read_bytes())
    outside = next(name for name in payload["source_manifest"] if name != "answer.py")
    target = service.workspace / outside
    load, calls = residual.load_doctor_residual_advisory, []
    def drift(**kwargs):
        result = load(**kwargs)
        calls.append(True)
        if len(calls) == 1:
            target.write_bytes(target.read_bytes()+b"\nchanged before original capture finished\n")
        return result
    monkeypatch.setattr(residual, "load_doctor_residual_advisory", drift)
    with pytest.raises(ValueError, match="stale"):
        capture(service, request)
    assert service._doctor_request is None
