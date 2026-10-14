"""Declared byte observations are bounded diagnostics, never task authority."""
from copy import deepcopy
import builtins
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_output_observation as observation
from benchmarks.agent_supervisor.container_coding import terminal_native_progress as progress


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def private_json(path, value):
    path.write_text(json.dumps(value) + "\n")
    path.chmod(0o600)


def seal(binding):
    payload = {key: value for key, value in binding.items() if key != "binding_id"}
    binding["binding_id"] = "sha256:" + sha(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return binding


@pytest.fixture
def authored(tmp_path):
    task, alias, attempt, claim, lease = "task:authored", "authored", "attempt:authored", "claim:authored", "lease:authored"
    native = tmp_path / "native"; native.mkdir(mode=0o700)
    root = tmp_path / "worktrees"; root.mkdir(mode=0o750)
    workspace = root / "owned"; workspace.mkdir()
    app = tmp_path / "app"; app.mkdir()
    (workspace / "eigen.py").write_bytes(b"authored changed candidate\n")
    (app / "eigen.py").write_bytes(b"authored baseline\n")
    baseline = "a" * 40
    scope = {"task_cid": task, "task_alias": alias, "baseline_commit": baseline,
        "outputs": [{"path": "eigen.py", "effect": "modify", "baseline_sha256": sha(b"authored baseline\n")}]}
    directory = native / "run/admitted_database_portal_attempts" / sha(attempt)[:24]
    directory.mkdir(mode=0o700, parents=True)
    binding = seal(dict(schema="ipfs_accelerate_py/agent-supervisor/database-portal-attempt-binding@1",
        interface="DatabasePortalExecutionBridge@1", attempt_id=attempt, claim_id=claim, lease_id=lease,
        task_cid=task, task_alias=alias, goal_cid="goal:authored", plan_cid="plan:authored",
        task_revision=3, fencing_token=1, fence_epoch=1, task_body_digest="sha256:" + "b" * 64,
        projection_seed_digest="sha256:" + "c" * 64, projection_immutable_digest="sha256:" + "d" * 64,
        authoritative_task_store="duckdb", projection_authority=False))
    private_json(directory / "database-attempt-binding.json", binding)
    event = dict(type="implementation_finished", task_id=alias, task_cid=task, canonical_task_cid=task,
        worktree_path=str(workspace), baseline_ref=baseline, attempt=1, returncode=1,
        provider_dispatched=True, attempt_consumed=True, lifecycle_finalize={"finalized": True},
        unrelated_private_body="PRIVATE-EVENT-CANARY")
    private_json(directory / "portal-events.jsonl", event)
    terminal = dict(task_cid=task, status="blocked", revision=4, availability="observed",
        completion_receipt=dict(operation="database_task_claim_failure", failure_kind="terminal_portal_bridge_error",
            automatic_retry_admitted=False, settlement_id_sha256=sha("settlement:authored"), attempt_id_sha256=sha(attempt),
            claim_id_sha256=sha(claim), lease_id_sha256=sha(lease), attempt_number=1,
            control_expected_revision=3, fencing_token=1, fence_epoch=1))
    report = dict(arm="no-index", task_state={key: terminal[key] for key in ("task_cid", "status", "revision")},
        native_progress={"latest": {"task": terminal}}, stop={"status": "succeeded"}, remaining_processes=0,
        runtime_close={"succeeded": True}, worker_cleanup_returncode=0, task_completed=False,
        provider_invocations=[dict(schema="router-implementation-invocation@1", phase="coding", purpose="coding",
            invocation_id="invocation:authored", workspace=str(workspace), public_instruction={"task_cid": task},
            raw_body="PRIVATE-PROVIDER-CANARY")])
    return SimpleNamespace(native=native, root=root, workspace=workspace, app=app, directory=directory,
        binding=binding, event=event, scope=scope, report=report)


def observe(f):
    before = deepcopy(f.report)
    result = observation.observe(native_state=f.native, workspace_root=f.root, repository=f.app,
        scope=f.scope, report=f.report)
    assert before == f.report
    assert result["observation_only"] is True
    for key in ("completion_authority", "settlement_authority", "retry_authority", "task_correctness_established",
                "patch_application_established", "writer_identity_established", "raw_source_exported"):
        assert result[key] is False
    encoded = json.dumps(result)
    assert "PRIVATE" not in encoded and "authored changed candidate" not in encoded
    assert str(f.workspace) not in encoded and str(f.native) not in encoded
    return result


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_bound_candidate_and_canonical_byte_facts_without_capsules(authored, arm):
    authored.report["arm"] = arm
    result = observe(authored)
    assert result["status"] == "observed"
    assert result["attempt_id_sha256"] == sha("attempt:authored")
    assert result["invocation_id_sha256"] == sha("invocation:authored")
    assert result["binding_id"] == authored.binding["binding_id"]
    assert result["original_allocation_inode_verified"] is False
    row, = result["outputs"]
    assert row["candidate"] == dict(status="observed", exists=True, size_bytes=27,
        sha256=sha(b"authored changed candidate\n"), equals_admitted_baseline=False)
    assert row["canonical"]["equals_admitted_baseline"] is True
    assert not list(authored.native.rglob("*capsule*"))


def test_only_signed_outputs_are_selected(authored):
    (authored.workspace / "eval.py").write_text("NOT-DECLARED-CANARY")
    (authored.workspace / "private").mkdir()
    (authored.workspace / "private/auth.json").write_text("PRIVATE-SECRET-CANARY")
    assert [row["path"] for row in observe(authored)["outputs"]] == ["eigen.py"]


def test_missing_output_is_absence_but_removed_allocation_is_unknown(authored):
    (authored.workspace / "eigen.py").unlink()
    result = observe(authored)
    assert result["outputs"][0]["candidate"]["exists"] is False
    authored.workspace.rmdir()
    result = observe(authored)
    assert result["status"] == "partial"
    assert result["outputs"][0]["candidate"]["exists"] is None
    assert result["outputs"][0]["canonical"]["exists"] is True


@pytest.mark.parametrize("part", ["task", "attempt", "claim", "lease", "fence", "baseline", "alias", "invocation_task", "invocation_path", "ambiguous_invocation", "ambiguous_event"])
def test_foreign_or_ambiguous_owner_join_never_reads_outputs(authored, part, monkeypatch):
    if part in {"task", "attempt", "claim", "lease", "alias"}:
        key = {"task": "task_cid", "attempt": "attempt_id", "claim": "claim_id", "lease": "lease_id", "alias": "task_alias"}[part]
        authored.binding[key] += ":foreign"
        private_json(authored.directory / "database-attempt-binding.json", seal(authored.binding))
    elif part == "fence":
        authored.report["native_progress"]["latest"]["task"]["completion_receipt"]["fence_epoch"] += 1
    elif part == "baseline":
        authored.event["baseline_ref"] = "b" * 40
        private_json(authored.directory / "portal-events.jsonl", authored.event)
    elif part == "invocation_task":
        authored.report["provider_invocations"][0]["public_instruction"]["task_cid"] += ":foreign"
    elif part == "invocation_path":
        authored.report["provider_invocations"][0]["workspace"] = str(authored.root / "foreign")
    elif part == "ambiguous_invocation":
        authored.report["provider_invocations"].append(deepcopy(authored.report["provider_invocations"][0]))
    else:
        path = authored.directory / "portal-events.jsonl"
        path.write_text(path.read_text() * 2)
    monkeypatch.setattr(observation, "_file", lambda *args: pytest.fail("unbound output read"))
    assert observe(authored)["status"] == "unavailable"


@pytest.mark.parametrize("part", ["binding", "events", "native", "attempt", "parent"])
def test_missing_private_or_insecure_parent_binding_is_unavailable(authored, part, monkeypatch):
    if part == "binding":
        (authored.directory / "database-attempt-binding.json").unlink()
    elif part == "events":
        (authored.directory / "portal-events.jsonl").unlink()
    else:
        {"native": authored.native, "attempt": authored.directory, "parent": authored.root}[part].chmod(0o777)
    monkeypatch.setattr(observation, "_file", lambda *args: pytest.fail("unbound output read"))
    assert observe(authored)["status"] == "unavailable"


@pytest.mark.parametrize("part", ["status", "operation", "settlement", "retry", "unsettled", "stop", "processes", "close", "worker", "revision"])
def test_unsettled_or_live_worker_cannot_be_observed(authored, part, monkeypatch):
    receipt = authored.report["native_progress"]["latest"]["task"]["completion_receipt"]
    if part == "status": authored.report["task_state"]["status"] = "in_progress"
    elif part == "operation": receipt["operation"] = "database_claim"
    elif part == "settlement": receipt.pop("settlement_id_sha256")
    elif part == "retry": receipt["automatic_retry_admitted"] = True
    elif part == "unsettled": authored.report["unavailable_settlement"] = True
    elif part == "stop": authored.report["stop"]["status"] = "failed"
    elif part == "processes": authored.report["remaining_processes"] = 1
    elif part == "close": authored.report["runtime_close"]["succeeded"] = False
    elif part == "worker": authored.report["worker_cleanup_returncode"] = 1
    else: authored.report["task_state"]["revision"] += 1
    monkeypatch.setattr(observation, "_open_directory", lambda *args: pytest.fail("unsettled filesystem read"))
    assert observe(authored)["status"] == "unavailable"


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory", "fifo", "oversize"])
@pytest.mark.parametrize("location", ["candidate", "canonical"])
def test_unsafe_output_is_unknown_and_never_hashed(authored, kind, location):
    path = (authored.workspace if location == "candidate" else authored.app) / "eigen.py"
    path.unlink()
    target = authored.app.parent / "unrelated-private"; target.write_text("PRIVATE-CANARY")
    if kind == "symlink": path.symlink_to(target)
    elif kind == "hardlink": os.link(target, path)
    elif kind == "directory": path.mkdir()
    elif kind == "fifo": os.mkfifo(path)
    else:
        with path.open("wb") as stream: stream.truncate(observation.MAX_FILE_BYTES + 1)
    result = observe(authored)
    assert result["status"] == "partial"
    row = result["outputs"][0][location]
    assert row["status"] == "unavailable" and row["exists"] is None and row["sha256"] is None


@pytest.mark.parametrize("path", ["../hidden.py", "/tests/test.py", ".git/config", "x/../eigen.py", "x//y", ".", ".runtime/secrets"])
def test_unsafe_scope_cannot_nominate_files(authored, path, monkeypatch):
    authored.scope["outputs"][0]["path"] = path
    monkeypatch.setattr(observation, "_open_directory", lambda *args: pytest.fail("unsafe scope read"))
    assert observe(authored)["status"] == "unavailable"


def test_create_scope_compares_absent_baseline_without_source_invention(authored):
    authored.scope["outputs"][0].update(effect="create", baseline_sha256=None)
    (authored.app / "eigen.py").unlink()
    result = observe(authored)
    assert result["outputs"][0]["candidate"]["equals_admitted_baseline"] is False
    assert result["outputs"][0]["canonical"]["equals_admitted_baseline"] is True


def test_capture_uses_admitted_scope_not_changed_checkout(authored):
    task = SimpleNamespace(task_cid=authored.scope["task_cid"], task_key=authored.scope["task_alias"],
        outputs=[SimpleNamespace(path="eigen.py", effect="modify")])
    verified = {"manifest": {"baseline_commit": authored.scope["baseline_commit"],
        "sources": {"eigen.py": {"sha256": sha(b"authored baseline\n")}}}}
    assert observation.capture_scope(verified, task) == authored.scope


def test_current_v2_owner_binding_is_accepted(authored):
    authored.binding.update(schema="ipfs_accelerate_py/agent-supervisor/database-portal-attempt-binding@2",
        control_binding_id="binding:authored", control_task_projection_cid="projection:authored",
        control_expected_revision=3, control_portal_binding_basis_cid="basis:authored")
    private_json(authored.directory / "database-attempt-binding.json", seal(authored.binding))
    assert observe(authored)["status"] == "observed"


def test_completed_task_byte_observation_is_not_publication_or_correctness(authored):
    authored.report["task_state"]["status"] = "completed"
    task = authored.report["native_progress"]["latest"]["task"]
    task["status"] = "completed"
    task["completion_receipt"]["operation"] = "database_task_completion"
    authored.event["returncode"] = 0
    private_json(authored.directory / "portal-events.jsonl", authored.event)
    assert observe(authored)["status"] == "observed"


@pytest.fixture
def native_completed(authored):
    """Capture the real daemon's emitted receipt at its authored CAS boundary.

    No live owner mutation or provider execution is performed. The production
    receipt constructor is exercised, so a copied spelling cannot mask drift.
    """
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon

    current = SimpleNamespace(**{key: authored.binding[key] for key in
        ("task_cid", "attempt_id", "claim_id", "lease_id", "fencing_token", "fence_epoch")})
    task = SimpleNamespace(task_cid=current.task_cid, status="in_progress", revision=3,
        body={"completion_receipt": {"operation": "database_attempt_admitted"}})
    captured = {}

    class CapturedCAS(Exception):
        pass

    def capture_cas(task_cid, **kwargs):
        assert task_cid == current.task_cid
        captured.update(kwargs)
        raise CapturedCAS

    daemon = SimpleNamespace(get_attempt=lambda _: current,
        provider_invocation_recorded=lambda *args, **kwargs: {}, require_real_execution=True,
        owner_session_id="owner:authored", _attempt_claim=lambda _: "claim:authored",
        _protect_attempt_claim=lambda *args, **kwargs: None, _now_ms=lambda: 1,
        _cas_task_status_database=capture_cas,
        task_source=SimpleNamespace(get=lambda _: task, record_validation_result=lambda **kwargs: None),
        coordinator=SimpleNamespace(prepare_task_completion=lambda *args, **kwargs: {"private": True}))
    with pytest.raises(CapturedCAS):
        DatabaseImplementationDaemon.complete_attempt(daemon, current,
            validation_result={"outcome": "passed", "evidence_digest": "sha256:" + "e" * 64,
                               "private": "PRIVATE-VALIDATION-CANARY"})
    assert captured["new_status"] == "completed"
    assert captured["expected_revision"] == 3
    assert captured["receipt"]["operation"] == "database_complete"
    assert captured["evidence_digests"] == [captured["receipt"]["evidence_digest"]]
    task.status, task.revision = "completed", 4
    task.body = {"completion_receipt": captured["receipt"]}
    authored.report["task_state"].update(status="completed", revision=4)
    authored.report["native_progress"]["latest"]["task"] = progress.project_task(task, task_cid=current.task_cid)
    authored.event["returncode"] = 0
    private_json(authored.directory / "portal-events.jsonl", authored.event)
    return authored, task


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_real_completion_receipt_survives_projection_and_owner_bound_observation(native_completed, arm):
    authored, task = native_completed
    authored.report["arm"] = arm
    before = deepcopy(task.body)
    projected = authored.report["native_progress"]["latest"]["task"]
    assert projected["completion_receipt"]["operation"] == "database_complete"
    result = observe(authored)
    assert result["status"] == "observed"
    assert result["binding_id"] == authored.binding["binding_id"]
    assert result["outputs"][0]["candidate"]["sha256"] == sha(b"authored changed candidate\n")
    assert result["outputs"][0]["canonical"]["equals_admitted_baseline"] is True
    assert authored.report["task_completed"] is False  # Observation does not grant completion.
    assert task.body == before


@pytest.mark.parametrize("part", ["operation", "internal_authority", "status", "attempt", "claim", "lease",
    "fence", "revision", "binding", "event", "stop", "processes", "close", "worker"])
def test_real_completed_projection_still_requires_exact_owner_and_cleanup(native_completed, part, monkeypatch):
    authored, task = native_completed
    receipt = task.body["completion_receipt"]
    if part == "operation": receipt["operation"] = "database_claim"
    elif part == "internal_authority": receipt["operation"] = "task.database.completion"
    elif part == "status":
        task.status = "blocked"
        authored.report["task_state"]["status"] = "blocked"
    elif part in {"attempt", "claim", "lease"}: receipt[part + "_id"] += ":foreign"
    elif part == "fence": receipt["fence_epoch"] += 1
    elif part == "revision": task.revision += 1
    elif part == "binding": (authored.directory / "database-attempt-binding.json").unlink()
    elif part == "event":
        authored.event["lifecycle_finalize"]["finalized"] = False
        private_json(authored.directory / "portal-events.jsonl", authored.event)
    elif part == "stop": authored.report["stop"]["status"] = "failed"
    elif part == "processes": authored.report["remaining_processes"] = 1
    elif part == "close": authored.report["runtime_close"]["succeeded"] = False
    elif part == "worker": authored.report["worker_cleanup_returncode"] = 1
    authored.report["native_progress"]["latest"]["task"] = progress.project_task(task, task_cid=task.task_cid)
    monkeypatch.setattr(observation, "_file", lambda *args: pytest.fail("unbound completed output read"))
    assert observe(authored)["status"] == "unavailable"


@pytest.mark.parametrize("replaced", ["file", "directory", "root", "delete"])
def test_named_path_replacement_during_read_is_unavailable(authored, monkeypatch, replaced):
    path = authored.workspace / "eigen.py"
    if replaced == "directory":
        (authored.workspace / "src").mkdir()
        path.rename(authored.workspace / "src/eigen.py")
        path = authored.workspace / "src/eigen.py"
        authored.scope["outputs"][0]["path"] = "src/eigen.py"
    target_inode = path.stat().st_ino
    original_read = observation.os.read
    changed = []
    def replace(fd, size):
        raw = original_read(fd, size)
        if not changed and os.fstat(fd).st_ino == target_inode:
            changed.append(True)
            if replaced == "delete":
                path.unlink()
            elif replaced == "file":
                path.rename(path.with_name("old.py")); path.write_text("PRIVATE-REPLACEMENT")
            elif replaced == "directory":
                path.parent.rename(authored.workspace / "previous-src")
                path.parent.mkdir(); path.write_text("PRIVATE-REPLACEMENT")
            else:
                authored.workspace.rename(authored.root / "previous-root")
                authored.workspace.mkdir(); path.write_text("PRIVATE-REPLACEMENT")
        return raw
    monkeypatch.setattr(observation.os, "read", replace)
    result = observe(authored)
    assert changed
    assert result["outputs"][0]["candidate"]["status"] == "unavailable"


def test_real_bounded_child_exports_only_closed_facts(authored):
    result = observation.collect(native_state=authored.native, workspace_root=authored.root,
        repository=authored.app, scope=authored.scope, report=authored.report, timeout_seconds=5)
    assert result["status"] == "observed"
    assert "PRIVATE" not in json.dumps(result)


def test_real_child_uses_loaded_benchmark_owner_before_foreign_namespace(authored, tmp_path, monkeypatch):
    foreign = tmp_path / "foreign"
    package = foreign / "benchmarks"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('raise RuntimeError("PRIVATE-FOREIGN-PACKAGE-CANARY")\n')
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        value for value in (str(foreign), os.environ.get("PYTHONPATH")) if value))
    result = observation.collect(native_state=authored.native, workspace_root=authored.root,
        repository=authored.app, scope=authored.scope, report=authored.report, timeout_seconds=5)
    assert result["status"] == "observed"
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize("umask", [0o022, 0o027, 0o077])
@pytest.mark.parametrize("child", [False, True])
def test_native_event_writer_permissions_under_private_attempt(authored, umask, child):
    """Use the production append path, whose log inherits the process umask."""
    from ipfs_accelerate_py.agent_supervisor.runtime.event_log import append_jsonl_event

    path = authored.directory / "portal-events.jsonl"
    path.unlink()
    previous = os.umask(umask)
    try:
        append_jsonl_event(path, "implementation_finished", authored.event)
    finally:
        os.umask(previous)
    assert path.stat().st_mode & 0o777 == 0o666 & ~umask
    if child:
        result = observation.collect(native_state=authored.native, workspace_root=authored.root,
            repository=authored.app, scope=authored.scope, report=authored.report, timeout_seconds=5)
    else:
        result = observe(authored)
    assert result["status"] == "observed"
    assert result["outputs"][0]["candidate"]["sha256"] == sha(b"authored changed candidate\n")
    assert result["outputs"][0]["canonical"]["sha256"] == sha(b"authored baseline\n")
    assert result["patch_application_established"] is False
    assert result["writer_identity_established"] is False
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize("mode", [0o620, 0o602, 0o664, 0o666])
def test_owner_event_log_with_other_write_permission_is_rejected(authored, mode, monkeypatch):
    (authored.directory / "portal-events.jsonl").chmod(mode)
    monkeypatch.setattr(observation, "_file", lambda *args: pytest.fail("unsafe event selected outputs"))
    result = observe(authored)
    assert result["status"] == "unavailable"
    assert result["evidence_stage"] == "owner_event_log"


@pytest.mark.parametrize("kind", ["foreign_owner", "symlink", "hardlink", "fifo"])
def test_owner_event_log_requires_native_owned_regular_file(authored, kind, monkeypatch):
    path = authored.directory / "portal-events.jsonl"
    if kind == "foreign_owner":
        inode = path.stat().st_ino
        original = observation.os.fstat
        def foreign(fd):
            value = original(fd)
            if value.st_ino != inode:
                return value
            fields = {key: getattr(value, key) for key in dir(value) if key.startswith("st_")}
            return SimpleNamespace(**(fields | {"st_uid": os.geteuid() + 1}))
        monkeypatch.setattr(observation.os, "fstat", foreign)
    else:
        path.unlink()
        target = authored.native / "unrelated-event.jsonl"
        private_json(target, authored.event)
        if kind == "symlink": path.symlink_to(target)
        elif kind == "hardlink": os.link(target, path)
        else: os.mkfifo(path)
    monkeypatch.setattr(observation, "_file", lambda *args: pytest.fail("unsafe event selected outputs"))
    result = observe(authored)
    assert result["status"] == "unavailable"
    assert result["evidence_stage"] == "owner_event_log"


def test_readable_event_does_not_relax_binding_privacy(authored, monkeypatch):
    (authored.directory / "portal-events.jsonl").chmod(0o644)
    (authored.directory / "database-attempt-binding.json").chmod(0o644)
    monkeypatch.setattr(observation, "_file", lambda *args: pytest.fail("nonprivate binding selected outputs"))
    result = observe(authored)
    assert result["status"] == "unavailable"
    assert result["evidence_stage"] == "attempt_binding"


def test_readable_event_requires_private_attempt_directory(authored, monkeypatch):
    (authored.directory / "portal-events.jsonl").chmod(0o644)
    authored.directory.chmod(0o755)
    monkeypatch.setattr(observation, "_file", lambda *args: pytest.fail("nonprivate attempt selected outputs"))
    result = observe(authored)
    assert result["status"] == "unavailable"
    assert result["evidence_stage"] == "attempt_directory"


def test_binding_change_between_safe_read_and_native_verification_is_rejected(authored, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge

    original = DatabasePortalExecutionBridge._strict_binding
    def replaced(path):
        changed = dict(authored.binding, lease_id="PRIVATE-FOREIGN-LEASE")
        private_json(authored.directory / "database-attempt-binding.json", seal(changed))
        return original(path)
    monkeypatch.setattr(DatabasePortalExecutionBridge, "_strict_binding", replaced)
    monkeypatch.setattr(observation, "_file", lambda *args: pytest.fail("changed binding selected outputs"))
    result = observe(authored)
    assert result["status"] == "unavailable"
    assert result["evidence_stage"] == "attempt_binding"


@pytest.mark.parametrize("replacement", ["file", "write_permissions"])
def test_event_changes_during_read_are_rejected(authored, monkeypatch, replacement):
    path = authored.directory / "portal-events.jsonl"
    path.chmod(0o644)
    inode = path.stat().st_ino
    original = observation.os.read
    changed = []
    def replace(fd, size):
        raw = original(fd, size)
        if not changed and os.fstat(fd).st_ino == inode:
            changed.append(True)
            if replacement == "file":
                path.rename(path.with_suffix(".old"))
                private_json(path, authored.event)
            else:
                path.chmod(0o666)
        return raw
    monkeypatch.setattr(observation.os, "read", replace)
    monkeypatch.setattr(observation, "_file", lambda *args: pytest.fail("changed event selected outputs"))
    result = observe(authored)
    assert changed
    assert result["status"] == "unavailable"
    assert result["evidence_stage"] == "owner_event_log"


@pytest.mark.parametrize("kind", ["timeout", "failure", "malformed", "oversize"])
def test_diagnostic_child_failure_never_changes_outcome(authored, monkeypatch, kind):
    before = deepcopy(authored.report)
    def run(argv, **kwargs):
        assert "PRIVATE" not in kwargs["input"].decode()
        assert argv[:4] == [observation.sys.executable, "-B", "-P", "-m"]
        assert 0 < kwargs["timeout"] <= 5
        if kind == "timeout": raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return SimpleNamespace(returncode=1 if kind == "failure" else 0,
            stdout=b"{" if kind == "malformed" else b"x" * (observation.MAX_RESULT_BYTES + 1))
    monkeypatch.setattr(observation.subprocess, "run", run)
    result = observation.collect(native_state=authored.native, workspace_root=authored.root,
        repository=authored.app, scope=authored.scope, report=authored.report)
    assert result["status"] == "unavailable" and authored.report == before


@pytest.mark.parametrize("scope,seconds", [(None, 3), ({}, 0)])
def test_missing_scope_or_budget_starts_no_child(authored, monkeypatch, scope, seconds):
    monkeypatch.setattr(observation.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected child"))
    result = observation.collect(native_state=authored.native, workspace_root=authored.root,
        repository=authored.app, scope=scope, report=authored.report, timeout_seconds=seconds)
    assert result["status"] == "unavailable"


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_driver_observer_failure_preserves_settlement_and_durable_report(tmp_path, monkeypatch, arm):
    from benchmarks.agent_supervisor.container_coding.test_formalization_resource_profile import _simulate
    calls = []
    def failed(**kwargs):
        report = kwargs["report"]
        assert report["stop"]["status"] == "succeeded"
        assert report["runtime_close"]["succeeded"] is True
        assert report["worker_cleanup_returncode"] == 0
        calls.append(kwargs)
        raise RuntimeError("PRIVATE-DIAGNOSTIC-CANARY")
    monkeypatch.setattr(observation, "collect", failed)
    result, *_ = _simulate(tmp_path, monkeypatch, arm=arm)
    assert len(calls) == 1
    assert result["task_state"]["status"] == "blocked" and result["task_completed"] is False
    assert result["declared_output_observation"]["status"] == "unavailable"
    assert "PRIVATE" not in json.dumps(result)


def test_driver_missing_observer_import_still_writes_primary_report(tmp_path, monkeypatch):
    from benchmarks.agent_supervisor.container_coding.test_formalization_resource_profile import _simulate
    original_import = builtins.__import__
    def missing(name, *args, **kwargs):
        if name.endswith("terminal_output_observation"):
            raise ModuleNotFoundError("PRIVATE-IMPORT-CANARY")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing)
    result, *_ = _simulate(tmp_path, monkeypatch, arm="no-index")
    assert result["declared_output_observation"]["status"] == "unavailable"
    assert result["task_state"]["status"] == "blocked" and result["task_completed"] is False
    assert "PRIVATE" not in json.dumps(result)
