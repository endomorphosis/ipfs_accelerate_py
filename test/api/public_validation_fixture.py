"""Native signed public+Doctor fixture; only the host UID boundary is authored."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

from benchmarks.agent_supervisor.container_coding.test_terminal_public_acceptance_profile import prepared_public_case
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as preparation
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as doctor_dispatch
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, build_text_context_references, render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.runtime import router_public_instruction as public
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_broker as brokers
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context


def _capture_broker(service, extra, timeout_seconds):
    fd = os.memfd_create("authored-public-doctor-requirements", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
    try:
        os.write(fd, service.prompt.encode())
        os.lseek(fd, 0, os.SEEK_SET)
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, 15)
        return brokers.prepare_worker_source_broker(argv=["--purpose", "coding",
            "--semantic-repository", str(service.repository), "--source-view-profile", "python-comments@1",
            "--timeout", str(timeout_seconds), *extra], boundary=service.boundary,
            workspace=service.workspace, input_fd=fd)
    finally:
        os.close(fd)


@contextmanager
def public_doctor_service(tmp_path, monkeypatch, *, timeout_seconds=600):
    case = prepared_public_case(tmp_path, planning_mode="declared-contract@1")
    planning = preparation.plan(case.state)
    assert planning["qualified"] and planning["provider_calls"] == 0, planning
    task_cid, = planning["task_cids"]
    task_id = case.prepared["spec"]["task_key"]
    admission = json.loads((case.state / "admission.json").read_text())
    public_context = public.prepare_public_instruction_context(repository=case.repository,
        admission=admission, task_cid=task_cid, source_path=".supervisor-instruction.md",
        expected_source_sha256=case.profile["instruction_sha256"])
    monkeypatch.setattr(doctor_dispatch, "_installed_provers", lambda: (
        Path("/unavailable/authored-z3"), Path("/unavailable/authored-lean")))
    dispatch = doctor_dispatch.prepare_terminal_doctor_dispatch(repository=case.repository,
        state=case.state, admission=admission, task_cid=task_cid)
    assert dispatch["route"] == "model_router" and dispatch["status"] == "residual", dispatch
    assert dispatch["provider_calls"] == 0 and "residual_context" in dispatch
    doctor_context = dispatch["residual_context"]
    public_request = dict(artifact=Path(public_context["artifact"]), expected_sha256=public_context["sha256"], task_cid=task_cid)
    doctor_request = dict(artifact=Path(doctor_context["artifact"]), expected_sha256=doctor_context["sha256"], task_cid=task_cid)
    workspace = tmp_path / "allocated"
    subprocess.run(["git", "-C", str(case.repository), "worktree", "add", "--detach", "-q", str(workspace)], check=True)
    output = case.repository / ".runtime/semantic-public-validation"
    prepare_semantic_context(repository=case.repository, paths=["eigen.py"], program_paths=["eigen.py"],
        required_raw_paths=["eigen.py"], objective="Implement the signed public eigenpair and performance obligations",
        task_id=task_id, output=output)
    artifact = output / "worker-context.json"
    references = build_text_context_references(artifact.read_text(), reference_prefix="semantic-context",
        kind="semantic-context", path=artifact.relative_to(case.repository).as_posix(),
        repository_id="repo:authored-public", tree_id="tree:authored-public", required=True, chunk_bytes=1201)
    compiled = ContextCompiler(ContextBudget(max_input_tokens=32768, max_items=128,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(
        repository_id="repo:authored-public", tree_id="tree:authored-public", objective_id=task_id,
        objective_revision="sha256:task", policy_id="policy:authored-public", policy_revision="sha256:policy",
        caller="supervisor:authored", stage="implementation", goal={"id": task_id},
        authority={"mode": "candidate_only", "completion_authority": False}, scope={"allowed_paths": ["eigen.py"]},
        acceptance={"criteria": [row["criterion_key"] for row in case.prepared["spec"]["acceptance"]]}, evidence=references)
    with tempfile.TemporaryDirectory(prefix="public-doctor-candidate-") as directory:
        boundary = Path(directory) / "boundary.json"
        boundary.write_text("authored local boundary")
        (boundary.parent / "worker-observations").mkdir(mode=0o750)
        monkeypatch.setattr(guard, "_owner_boundary", lambda path: (
            dict(owner_uid=os.getuid(), worker_uid=os.getuid()+1), "b" * 64))
        monkeypatch.setattr(guard, "_protected_directory", lambda *args, **kwargs: None)
        service = brokers.WorkerSourceBroker(repository=case.repository, workspace=workspace, boundary=boundary,
            prompt=render_context_capsule(compiled.capsule), timeout_seconds=timeout_seconds)
        original = None
        extra = []
        for prefix, selected in (("public-instruction", public_request), ("doctor-residual", doctor_request)):
            extra.extend(["--" + prefix + "-artifact", str(selected["artifact"]),
                "--" + prefix + "-sha256", selected["expected_sha256"], "--" + prefix + "-task-cid", task_cid])
        try:
            original = _capture_broker(service, extra, timeout_seconds)
            assert original.original_requirements is not None
            assert original.original_requirements._doctor_requirements is not None
            yield SimpleNamespace(original=original, public_request=public_request, doctor_request=doctor_request,
                prepared=case.prepared, planning=planning, dispatch=dispatch, repository=case.repository,
                workspace=workspace, state=case.state, admission=admission, task_cid=task_cid,
                profile=case.profile, extra_argv=extra, boundary=boundary)
        finally:
            if original is not None:
                original.close()
            service.close()
