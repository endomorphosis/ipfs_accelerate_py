"""Host native claim/allocation admission around a separately born child.

DuckDB claims/grants, Portal projection, Git allocation, lifecycle, source
production, run lease and child birth are real. Quack transport and deployment
UID/path admission are inherited authored host seams. Children never import
target code, invoke a provider, validate a task, or publish accepted state.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, build_text_context_references, render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DatabaseCoordinationExpiredError, open_database_coordinator,
)
from ipfs_accelerate_py.agent_supervisor.runtime import native_candidate_round_admission as admission
from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_round_context as contexts
from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_round_prompt as prompts
from ipfs_accelerate_py.agent_supervisor.runtime import owner_worker_attempt as attempts
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from ipfs_accelerate_py.agent_supervisor.runtime.worker_source_broker import WorkerSourceBroker
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import canonical_json_bytes
from test.api.test_current_source_task_admission import ALIAS, _update_body
from test.api.test_current_source_worktree_allocation import _allocation_case, _record_change
from test.api.test_owner_candidate_round_context import SOURCE, CHANGED
from test.api.test_owner_worker_attempt import _sealed


class CandidateCase(SimpleNamespace):
    def __repr__(self):
        return "CandidateCase(native owner resources retained privately)"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def capture_kwargs(case, **changes):
    return dict(allocation=case.allocation, run_coordinator=case.run_coordinator,
        run_lease=case.run_lease, candidate_prompt=case.candidate_prompt,
        prior_worker_attempt=case.prior_worker_attempt, worker_attempt=case.worker_attempt) | changes


def capture(case, **changes):
    return admission.capture_native_candidate_round_admission(**capture_kwargs(case, **changes))


def spawn_child(case, *, code=0):
    """Return an unblocked child only after the caller releases its stdin."""
    child = subprocess.Popen([sys.executable, "-I", "-c",
        "import sys;sys.stdin.buffer.read();raise SystemExit(int(sys.argv[1]))", str(code)],
        cwd=case.native.workspace,
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    case.children.append(child)
    return child


@contextmanager
def _candidate_case(tmp_path, monkeypatch, *, prior_flags=True,
        provision_workspace=None, capture_allocation=None):
    """Reusable real owner fixture; no native source-reader binding is spent."""
    with _allocation_case(tmp_path, monkeypatch, defer_source_binding=True,
            source_files={b"module.py": SOURCE}) as allocated:
        native = allocated.native
        if provision_workspace is not None:
            provision_workspace(allocated)
        semantic_output = native.canonical / ".runtime/native-candidate-round-context"
        prepare_semantic_context(repository=native.canonical, paths=["module.py"],
            program_paths=["module.py"], required_raw_paths=["module.py"],
            objective="Retain the native task requirements while observing candidate source.",
            task_id=ALIAS, output=semantic_output)
        artifact = semantic_output / "worker-context.json"
        task_body = canonical_json_bytes(dict(native.native_task.body)).decode("utf-8")
        references = build_text_context_references(artifact.read_text(),
            reference_prefix="semantic-context", kind="semantic-context",
            path=artifact.relative_to(native.canonical).as_posix(),
            repository_id=native.server.identity.repository_id,
            tree_id="git-tree:" + native.retained.request.expected_tree,
            required=True, chunk_bytes=1201)
        # The full native body is retained literally, independently of the
        # generated source summary. Neither source selection nor a failure can
        # replace the original acceptance requirements or declared outputs.
        required_body = build_text_context_references(task_body,
            reference_prefix="native-task-body", kind="native-task-requirements",
            repository_id=native.server.identity.repository_id,
            tree_id="git-tree:" + native.retained.request.expected_tree,
            required=True, chunk_bytes=1201)
        compiled = ContextCompiler(ContextBudget(max_input_tokens=32768, max_items=128,
            max_item_bytes=16384, max_serialized_bytes=262144)).compile(
                repository_id=native.server.identity.repository_id,
                tree_id="git-tree:" + native.retained.request.expected_tree,
                objective_id=ALIAS, objective_revision="sha256:native-task-revision",
                policy_id="policy:native-candidate-host-fixture", policy_revision="sha256:host-policy-v1",
                caller="native-owner-host-fixture", stage="implementation",
                goal={"task_id": ALIAS, "native_task_cid": native.task_cid},
                authority={"mode": "candidate_only", "completion_authority": False},
                scope={"allowed_paths": ["module.py"]},
                acceptance={"native_task_body": json.loads(task_body), "validation_pending": True},
                evidence=(*references, *required_body))
        native_prompt = render_context_capsule(compiled.capsule) + "\nPreserve every complete original native requirement.\n"
        boundary = native.factory["boundary"]
        source_broker = WorkerSourceBroker(repository=native.canonical, workspace=native.workspace,
            boundary=boundary, prompt=native_prompt, timeout_seconds=90)
        descriptors, children = [], []
        coordinator = None
        run_lease = None
        try:
            original = prompts.capture_native_only_requirements(source_broker=source_broker)
            source_broker.original_requirements = original
            deadline = time.monotonic() + 90
            initial_fd = _sealed(native_prompt.encode("utf-8"))
            descriptors.append(initial_fd)
            prior = attempts.prepare_owner_worker_attempt(boundary=boundary, workspace=native.workspace,
                input_fd=initial_fd, absolute_deadline=deadline)
            if capture_allocation is not None:
                allocated.guard = capture_allocation(allocated, prior)
                native.factory["worktree_admission"] = allocated.guard
            first = subprocess.Popen([sys.executable, "-I", "-c",
                "import sys;from pathlib import Path;sys.stdin.buffer.read();"
                "Path(sys.argv[1]).write_bytes(bytes.fromhex(sys.argv[2]));raise SystemExit(7)",
                str(native.workspace / "module.py"), CHANGED.hex()],
                cwd=native.workspace,
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            children.append(first)
            prior.note_spawned(first)
            source_broker.note_spawned(first)
            first.communicate(b"release authored inert source edit", timeout=10)
            assert first.returncode == 7
            source_broker.close()
            if prior_flags is not None:
                flags = dict(waited=True, cleanup_succeeded=True, brokers_closed=True)
                if type(prior_flags) is dict:
                    flags.update(prior_flags)
                prior.finish(**flags)
            snapshot = contexts.prepare_candidate_round_context(source_broker=source_broker,
                prior_returncode=7, round_number=2, absolute_deadline=deadline)
            candidate_prompt = prompts.prepare_candidate_round_prompt(snapshot=snapshot, requirements=original)
            candidate_fd = _sealed(candidate_prompt.provider_prompt)
            descriptors.append(candidate_fd)
            worker_attempt = attempts.prepare_owner_worker_attempt(boundary=boundary, workspace=native.workspace,
                input_fd=candidate_fd, absolute_deadline=deadline)
            run_root = tmp_path / "candidate-run-authority"
            run_root.mkdir(mode=0o700)
            coordinator = open_database_coordinator(run_root / "coordination.duckdb")
            coordinator.database_path.chmod(0o600)
            run_id = "native-candidate-host-run"
            run_lease = coordinator.acquire(lease_kind="resource", scope=run_id,
                owner_session_id=native.attempt.owner_session_id, lease_ms=120000,
                resource_kind="supervisor_run", resource_id=run_id,
                repository_id=native.server.identity.repository_id)
            case = CandidateCase(allocation_case=allocated, native=native, allocation=allocated.guard,
                candidate_prompt=candidate_prompt, snapshot=snapshot, original_requirements=original,
                prior_worker_attempt=prior, worker_attempt=worker_attempt, source_broker=source_broker,
                first_process=first, run_coordinator=coordinator, run_lease=run_lease,
                native_prompt=native_prompt, native_task_body=task_body,
                boundary=boundary, deadline=deadline, descriptors=descriptors, children=children)
            yield case
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)
                if child.stdin is not None:
                    child.stdin.close()
            source_broker.close()
            for descriptor in descriptors:
                os.close(descriptor)
            if coordinator is not None:
                try:
                    if run_lease is not None:
                        try:
                            coordinator.release(run_lease, reason="native candidate host fixture finished")
                        except DatabaseCoordinationExpiredError:
                            # Released/expired authority is an explicit test
                            # case. Teardown still closes this exact database.
                            pass
                finally:
                    coordinator.close()


@pytest.fixture
def candidate_case(tmp_path, monkeypatch):
    with _candidate_case(tmp_path, monkeypatch) as case:
        yield case


@pytest.fixture
def native_candidate(candidate_case):
    return candidate_case


def assert_never_spawns(handle):
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        handle.execute_spawn(lambda: pytest.fail("stale native candidate admission spawned a child"), lambda _process: None)
    assert handle.process is None


def test_real_native_gate_registers_actual_birth_and_keeps_complete_original_requirements(native_candidate):
    case = native_candidate
    gate = capture(case)
    metadata = gate.revalidate()
    payload = json.loads(case.candidate_prompt.provider_prompt)
    assert payload["native_prompt"] == case.native_prompt
    assert payload["original_requirements_binding"]["task_id"] == ALIAS
    native_wire, _end = json.JSONDecoder().raw_decode(payload["native_prompt"])
    required_body = sorted((ref for ref in native_wire["evidence"]
        if ref["kind"] == "native-task-requirements"), key=lambda ref: ref["metadata"]["chunk_index"])
    assert "".join(ref["summary"] for ref in required_body) == case.native_task_body
    assert payload["baseline_evidence_applicable"] is payload["proof_authority"] is False
    registered = []
    def register(actual):
        assert type(actual) is subprocess.Popen and actual.returncode is None
        event_path = case.boundary.parent / "worker-observations" / (
            "worker-attempt-" + case.worker_attempt.attempt_id + "-spawned.json")
        born = json.loads(event_path.read_bytes())
        assert born["child_pid"] == actual.pid
        registered.append(actual)
    child = gate.execute_spawn(lambda: spawn_child(case), register)
    assert type(child) is subprocess.Popen and child is gate.process
    assert registered == [child] and child.pid != case.first_process.pid
    assert child.returncode is None  # The terminal stdin barrier is still held.
    child.communicate(b"finish authored admission child", timeout=10)
    assert child.returncode == 0
    assert metadata["completion_authority"] is False
    assert case.native.source.get_task(case.native.task_cid).status == "in_progress"
    with pytest.raises((ValueError, RuntimeError)):
        gate.execute_spawn(lambda: pytest.fail("admission launched twice"), lambda _process: None)


def test_a_predecessor_cannot_issue_two_admissions(native_candidate):
    capture(native_candidate)
    with pytest.raises((ValueError, RuntimeError)):
        capture(native_candidate)


@pytest.mark.parametrize("field", ["allocation", "run_coordinator", "run_lease", "candidate_prompt", "prior_worker_attempt", "worker_attempt"])
def test_serialized_or_alleged_handles_never_replace_native_owner_custody(native_candidate, field):
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        capture(native_candidate, **{field: {"claimed_current": True}})


@pytest.mark.parametrize("changed", ["claim", "grant", "task_body", "candidate", "canonical", "producer", "run_lease"])
def test_actual_owner_or_source_drift_refuses_before_birth(native_candidate, monkeypatch, changed):
    case = native_candidate
    gate = capture(case)
    if changed == "claim":
        claim = case.native.daemon.coordinator.get_task_claim(case.native.attempt.claim_id)
        lease = case.native.daemon.coordinator.protect_task_claim(claim)
        case.native.daemon.coordinator.release(lease, reason="authored native claim release")
    elif changed == "grant":
        case.native.server._command_gateway.revoke_grant(case.native.daemon_grant_id)
    elif changed == "task_body":
        body = json.loads(canonical_json_bytes(dict(case.native.native_task.body)))
        body["title"] = "Changed complete original requirement"
        _update_body(case.native, body)
    elif changed in {"candidate", "canonical"}:
        root = case.native.workspace if changed == "candidate" else case.native.canonical
        path = root / "module.py"
        path.write_bytes(path.read_bytes() + b"\n# drift after native admission\n")
    elif changed == "producer":
        monkeypatch.setattr(contexts, "_producers", lambda: b'{"changed":"native producer"}')
    else:
        case.run_coordinator.release(case.run_lease, reason="authored run lease release")
        case.run_lease = None
    assert_never_spawns(gate)


@pytest.mark.parametrize("changed", ["lifecycle", "control"])
def test_native_lifecycle_or_portal_control_drift_refuses_before_birth(native_candidate, changed):
    case = native_candidate
    gate = capture(case)
    if changed == "lifecycle":
        with _record_change(case.allocation_case, lambda value: value.update(state="settling")):
            assert_never_spawns(gate)
    else:
        path = case.allocation_case.paths.binding
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n")
            assert_never_spawns(gate)
        finally:
            path.write_bytes(original)


def test_postbirth_registration_failure_keeps_actual_process_and_consumes_admission(native_candidate):
    case = native_candidate
    gate = capture(case)
    def refused(actual):
        assert type(actual) is subprocess.Popen
        raise RuntimeError("authored registration refusal")
    with pytest.raises(RuntimeError, match="authored registration refusal"):
        gate.execute_spawn(lambda: spawn_child(case), refused)
    assert gate.process is case.children[-1] and type(gate.process) is subprocess.Popen
    gate.process.communicate(b"retire retained child", timeout=10)
    with pytest.raises((ValueError, RuntimeError)):
        gate.execute_spawn(lambda: pytest.fail("failed registration caused another birth"), lambda _process: None)


def test_original_deadline_expiry_refuses_without_reset(native_candidate, monkeypatch):
    case = native_candidate
    gate = capture(case)
    members = dict(vars(time))
    members.update(monotonic=lambda: case.deadline + 1, monotonic_ns=lambda: int((case.deadline + 1) * 1e9))
    monkeypatch.setattr(contexts, "time", SimpleNamespace(**members))
    assert_never_spawns(gate)


def test_run_lease_renewal_preserves_native_fence(native_candidate):
    case = native_candidate
    gate = capture(case)
    renewed = case.run_coordinator.renew(case.run_lease, lease_ms=120000)
    assert renewed.lease_id == case.run_lease.lease_id
    assert renewed.fencing_token == case.run_lease.fencing_token
    assert renewed.fence_epoch == case.run_lease.fence_epoch
    case.run_lease = renewed
    gate.revalidate()


@pytest.mark.parametrize("flags", [None, dict(waited=False), dict(cleanup_succeeded=False), dict(brokers_closed=False)])
def test_pending_or_incompletely_retired_predecessor_cannot_admit_a_child(tmp_path, monkeypatch, flags):
    with _candidate_case(tmp_path, monkeypatch, prior_flags=flags) as case:
        with pytest.raises((ValueError, RuntimeError)):
            capture(case)


@pytest.mark.parametrize("mismatch", ["prompt", "deadline"])
def test_new_attempt_must_bind_exact_candidate_prompt_and_original_deadline(native_candidate, mismatch):
    case = native_candidate
    descriptor = _sealed(b"Different original prompt" if mismatch == "prompt" else case.candidate_prompt.provider_prompt)
    case.descriptors.append(descriptor)
    foreign = attempts.prepare_owner_worker_attempt(boundary=case.boundary, workspace=case.native.workspace,
        input_fd=descriptor, absolute_deadline=case.deadline + (1 if mismatch == "deadline" else 0))
    with pytest.raises((ValueError, RuntimeError)):
        capture(case, worker_attempt=foreign)


def test_matching_round_three_prompt_and_attempt_do_not_acquire_round_two_admission(native_candidate):
    case = native_candidate
    snapshot = contexts.prepare_candidate_round_context(source_broker=case.source_broker,
        prior_returncode=7, round_number=3, absolute_deadline=case.deadline)
    prompt = prompts.prepare_candidate_round_prompt(snapshot=snapshot, requirements=case.original_requirements)
    descriptor = _sealed(prompt.provider_prompt)
    case.descriptors.append(descriptor)
    worker_attempt = attempts.prepare_owner_worker_attempt(boundary=case.boundary,
        workspace=case.native.workspace, input_fd=descriptor, absolute_deadline=case.deadline)
    # Every prompt/attempt pin is genuine and coherent. The independent native
    # factory itself limits this host admission to the supported second round.
    assert snapshot.metadata["round_number"] == 3
    with pytest.raises((ValueError, RuntimeError)):
        capture(case, candidate_prompt=prompt, worker_attempt=worker_attempt)
    assert len(case.children) == 1


def test_source_edit_after_real_native_coordinator_returns_refuses_final_revalidation(native_candidate, monkeypatch):
    case = native_candidate
    gate = capture(case)
    coordinator = case.native.daemon.coordinator
    original = coordinator.execute_with_task_claim_fence
    edited = []

    def closing_observer(*args, **kwargs):
        # Keep the genuine coordinator, claim, callback and both native fences.
        # Author only a late closing observer after that operation returned;
        # the earlier source checks cannot have observed these changed bytes.
        result = original(*args, **kwargs)
        if not edited:
            path = case.native.workspace / "module.py"
            path.write_bytes(path.read_bytes() + b"\n# authored late closing observer edit\n")
            edited.append(True)
        return result

    monkeypatch.setattr(coordinator, "execute_with_task_claim_fence", closing_observer)
    with pytest.raises((ValueError, RuntimeError)):
        gate.revalidate()
    assert edited == [True]
    assert gate.process is None and len(case.children) == 1


def test_complete_retired_file_without_retained_publication_witness_cannot_be_adopted(tmp_path, monkeypatch):
    with _candidate_case(tmp_path, monkeypatch, prior_flags=None) as case:
        original = attempts.OwnerWorkerAttempt._publish

        def published_then_failed(self, value, **kwargs):
            result = original(self, value, **kwargs)
            if self is case.prior_worker_attempt and value["event"] == "retired":
                raise OSError("authored failure after successful retirement publication")
            return result

        monkeypatch.setattr(attempts.OwnerWorkerAttempt, "_publish", published_then_failed)
        with pytest.raises(OSError, match="after successful retirement publication"):
            case.prior_worker_attempt.finish(waited=True, cleanup_succeeded=True, brokers_closed=True)
        event_path = case.boundary.parent / "worker-observations" / (
            "worker-attempt-" + case.prior_worker_attempt.attempt_id + "-retired.json")
        retained_file = json.loads(event_path.read_bytes())
        assert retained_file["waited"] is retained_file["cleanup_succeeded"] is retained_file["brokers_closed"] is True
        assert case.prior_worker_attempt._state.retired is True
        assert case.prior_worker_attempt._state.retired_event is None
        with pytest.raises((ValueError, RuntimeError)):
            capture(case)
        assert len(case.children) == 1


def test_final_run_clock_expiry_refuses_after_birth_and_retains_actual_child(native_candidate, monkeypatch):
    case = native_candidate
    # Keep the independently protected run lease valid beyond the task lease.
    # Only the final pure task-expiry check should reject this child birth.
    case.run_lease = case.run_coordinator.renew(case.run_lease, lease_ms=600000)
    actual_run_clock = case.run_coordinator._clock_ms
    state = dict(native_closed=False, task_expires_at_ms=None, advanced=False, effective_wall_ns=None)
    actual_wall_ns = time.time_ns
    members = dict(vars(time))
    members["time_ns"] = lambda: (actual_wall_ns() if state["effective_wall_ns"] is None
        else state["effective_wall_ns"])
    monkeypatch.setattr(admission, "time", SimpleNamespace(**members))

    def final_run_observer():
        now = actual_run_clock()
        if state["native_closed"] and not state["advanced"]:
            # Author this clock observer's elapsed-time effect, without
            # changing the actual native lease, claim, or coordinator result.
            state["effective_wall_ns"] = (state["task_expires_at_ms"] + 1) * 1000000
            state["advanced"] = True
        return now

    # The live admission captures this original clock object. Changing its
    # identity after issuance would test custody refusal instead of expiry.
    monkeypatch.setattr(case.run_coordinator, "_clock_ms", final_run_observer)
    gate = capture(case)
    coordinator = case.native.daemon.coordinator
    actual_native_fence = coordinator.execute_with_task_claim_fence

    def native_closing_boundary(claim, callback, **kwargs):
        def observe_actual_task_lease(protected):
            state["task_expires_at_ms"] = protected.expires_at_ms
            return callback(protected)
        result = actual_native_fence(claim, observe_actual_task_lease, **kwargs)
        state["native_closed"] = True
        return result

    monkeypatch.setattr(coordinator, "execute_with_task_claim_fence", native_closing_boundary)
    registered = []
    with pytest.raises(TimeoutError, match="native lease lacks remaining"):
        gate.execute_spawn(lambda: spawn_child(case), lambda process: registered.append(process))
    assert state["advanced"] is True and type(state["task_expires_at_ms"]) is int
    assert gate.process is case.children[-1] and type(gate.process) is subprocess.Popen
    assert registered == [gate.process] and gate.process.returncode is None
    assert gate._state.consumed is gate._state.spawn_started is True
    assert gate._state.spawn_completed is False
    assert case.worker_attempt._state.spawned is not None
    # Stdin remains withheld; fixture teardown reaps this exact born child.
