"""Real typed owner cannot replace local acceptance with generic Portal evidence."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import subprocess

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import run_owner_local_task_validations
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import probe_quack_capabilities
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
from ipfs_accelerate_py.agent_supervisor.task_sources.task_source import TaskSourceConflictError
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import TransactionError
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon


@contextmanager
def typed_claim(scenario, tmp_path):
    report = probe_quack_capabilities()
    if not report.passes_health_check:
        pytest.skip(f"installed native Quack unavailable: {report.reason_code}")
    admission = local.admit_local_benchmark_plan(
        graph=scenario["graph"], manifest=scenario["manifest"],
    )
    local.materialize_local_benchmark_plan(admission=admission, intent=scenario["intent"])
    task = scenario["intent"].get_task(scenario["task_cid"])
    with open_existing_native_owner(
        database=scenario["intent"].database_path, checkout=scenario["repository"],
        state_dir=tmp_path / "owner", repository_id=scenario["manifest"]["payload"]["repository_cid"],
        execution_routes={task["task_alias"]: GROK_CODEX_EXECUTION_MODE},
    ) as owner:
        daemon = DatabaseImplementationDaemon(
            database_path=owner.database, coordination_path=tmp_path / "coordination.duckdb",
            execution_path=tmp_path / "execution.duckdb", authority_mode="quack",
            task_source_kind="duckdb", owner_session_id="session:local-completion-test",
            process_instance_id=owner.identity.process_birth_id,
            quack_uri=owner.identity.listen_uri, task_source=owner.source,
            close_task_source=False, state_owner_bootstrap_credentials=owner.credentials,
            strict_task_sharding=True, max_task_attempts=2, lease_ms=60_000,
            require_real_execution=True,
        ).open()
        try:
            attempt = daemon.claim_next()
            assert attempt is not None
            yield owner, daemon, attempt
        finally:
            daemon.close()


def complete(owner, attempt, digest):
    task = owner.source.get_task(attempt.task_cid)
    claimed = task.body["completion_receipt"]
    return owner.source.compare_and_set_status(
        task.task_cid, task.revision, "completed",
        receipt={
            "operation": "database_complete", "evidence_digest": digest,
            **{key: claimed[key] for key in (
                "attempt_id", "claim_id", "lease_id", "owner_session_id", "fencing_token", "fence_epoch",
            )},
        },
        expected_control_receipt=claimed, evidence_digests=[digest],
    )


def test_typed_owner_requires_actual_signed_current_local_checks(scenario, tmp_path):
    with typed_claim(scenario, tmp_path) as (owner, _daemon, attempt):
        task = owner.source.get_task(attempt.task_cid)
        generic = "sha256:" + hashlib.sha256(b"generic Portal validation passed").hexdigest()
        owner.client.record_task_validation(
            task_cid=task.task_cid, outcome="passed", evidence_digest=generic,
            argv=["portal-supervisor-gates"], attempt_id=attempt.attempt_id,
            body={"validator": "DatabasePortalExecutionBridge@1"},
            idempotency_key="generic-observation", command_id="generic-observation",
        )
        with pytest.raises((TaskSourceConflictError, TransactionError)):
            complete(owner, attempt, generic)
        assert owner.source.get_task(task.task_cid).revision == task.revision
        failed = run_owner_local_task_validations(
            server=owner.server, task_cid=task.task_cid, attempt_id=attempt.attempt_id,
            expected_revision=task.revision,
        )
        assert failed["passed"] is False
        with pytest.raises((TaskSourceConflictError, TransactionError)):
            complete(owner, attempt, failed["results"][0]["evidence_digest"])
        (scenario["repository"] / "answer.py").write_text("def answer():\n    return 2\n")
        passed = run_owner_local_task_validations(
            server=owner.server, task_cid=task.task_cid, attempt_id=attempt.attempt_id,
            expected_revision=task.revision,
        )
        assert passed["passed"] is True
        complete(owner, attempt, passed["results"][0]["evidence_digest"])
        assert owner.source.get_task(task.task_cid).status == "completed"


@pytest.mark.parametrize("change", ["remove", "replace"])
def test_typed_cas_cannot_drop_or_replace_pending_contract(scenario, tmp_path, change):
    with typed_claim(scenario, tmp_path) as (owner, _daemon, attempt):
        task = owner.source.get_task(attempt.task_cid)
        body = deepcopy(dict(task.body))
        if change == "remove":
            body.pop(local.CONTRACT_KEY)
        else:
            body[local.CONTRACT_KEY]["payload"]["pending_requirements"] = []
        with pytest.raises(TransactionError, match="authorization_denied"):
            owner.client.cas_task_status(
                task_cid=task.task_cid, goal_cid=task.goal_cid,
                expected_task_revision=task.revision, new_status="retrying",
                idempotency_key="local-contract-tamper:" + change,
                command_id="local-contract-tamper:" + change, body=body,
            )
        assert owner.source.get_task(task.task_cid).revision == task.revision
        assert owner.source.get_task(task.task_cid).body[local.CONTRACT_KEY] == task.body[local.CONTRACT_KEY]


def test_owner_validator_refuses_wrong_claim_or_stale_revision(scenario, tmp_path):
    with typed_claim(scenario, tmp_path) as (owner, _daemon, attempt):
        task = owner.source.get_task(attempt.task_cid)
        for attempt_id, revision in (("foreign-attempt", task.revision), (attempt.attempt_id, task.revision - 1)):
            with pytest.raises(local.LocalPlanningError):
                run_owner_local_task_validations(
                    server=owner.server, task_cid=task.task_cid,
                    attempt_id=attempt_id, expected_revision=revision,
                )
        assert owner.source.get_task(task.task_cid).revision == task.revision


def test_local_projection_keeps_contract_identity_without_owner_metadata(scenario, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge

    with typed_claim(scenario, tmp_path) as (owner, _daemon, attempt):
        task = owner.source.get_task(attempt.task_cid)
        contract = task.body[local.CONTRACT_KEY]
        projection = DatabasePortalExecutionBridge._render_projection_seed(attempt, task)
        assert "- Local planning contract CID: " + local.content_identity(contract) in projection
        assert "- Local Planning Contract:" not in projection
        assert str(contract["payload"]["manifest"]["payload"]["profile_dir"]) not in projection
        assert local.CONTRACT_KEY not in projection
        assert owner.source.get_task(task.task_cid).body[local.CONTRACT_KEY] == contract


def test_native_runner_uses_exact_single_task_prefix_for_database_projection(scenario, tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon_runner as runner
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalImplementationDaemon, PortalTaskState, parse_args
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane import resolve_control_plane

    with typed_claim(scenario, tmp_path) as (owner, daemon, attempt):
        parsed = parse_args([
            "--task-source-kind", "duckdb", "--authority-mode", "quack",
            "--database-path", str(owner.database), "--task-prefix", "##",
            "--state-dir", str(tmp_path / "runner"), "--state-prefix", "single",
            "--board-namespace", "intent",
            "--max-task-attempts", "1", "--implement", "--once",
        ])
        monkeypatch.setattr(runner, "_mirror_database_portal_execution_binding", lambda _: None)
        bridge = runner.bind_database_portal_execution_from_args(
            daemon, parsed, repo_root=scenario["repository"], portal_daemon_class=PortalImplementationDaemon,
        )
        record = owner.source.get_task(attempt.task_cid)
        paths, binding = bridge._ensure_attempt_projection(attempt, record)
        monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_ENDPOINT", "quack:127.0.0.1:1")
        portal = bridge.portal_factory(paths, record.task_alias)
        try:
            bound = portal.bind_database_attempt_authority(
                task_id=record.task_alias, database_task_cid=attempt.task_cid,
                database_attempt_id=attempt.attempt_id, database_claim_id=attempt.claim_id,
                database_attempt_number=attempt.attempt_number, database_binding_id=binding["binding_id"],
            )
            assert bound["task_id"] == record.task_alias
            assert portal.task_header_prefix == "## " + record.task_alias
            assert bound["completion_authority"] is False
            projected_task = portal._load_tasks()[0]
            assert projected_task.board_namespace == bridge.board_namespace == portal.board_namespace == 'intent'
            assert portal._identity_for_task(projected_task).board_namespace == 'intent'
            assert resolve_control_plane(paths.state) == (str(paths.root / "portal-state.duckdb"), "bound")
            state = PortalTaskState.load(paths.state)
            state.active_task_id = record.task_alias
            state.save(paths.state)
            assert PortalTaskState.load(paths.state).active_task_id == record.task_alias
            assert not paths.state.exists()
        finally:
            portal.close_event_runtime()


def test_native_validation_invocations_are_distinct_in_the_same_second(scenario, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.task_sources import intent_repository
    from test.api.test_agent_supervisor_local_planning_admission import _start

    _start(scenario)
    monkeypatch.setattr(intent_repository, "_utc_iso", lambda *_: "2026-09-29T00:00:00Z")
    # Execute the actual failing public check twice under the same task and
    # attempt. Even equal observations must produce separate native runs.
    for _ in range(2):
        observed = local.run_local_task_validations(
            intent=scenario["intent"], task_cid=scenario["task_cid"],
            attempt_id="same-real-validation-attempt",
        )
        assert observed["passed"] is False
    with scenario["intent"]._connection() as connection:
        rows = connection.execute(
            "SELECT run_id, started_at, status FROM validation_runs WHERE task_cid = ?",
            [scenario["task_cid"]],
        ).fetchall()
    assert len(rows) == 2 and len({row[0] for row in rows}) == 2
    assert {row[1] for row in rows} == {"2026-09-29T00:00:00Z"}
    assert {row[2] for row in rows} == {"failed"}


def test_latest_observed_check_uses_native_sequence_with_same_second_and_bytes(scenario, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.task_sources import intent_repository
    from test.api.test_agent_supervisor_local_planning_admission import _start

    _start(scenario)
    monkeypatch.setattr(intent_repository, "_utc_iso", lambda *_: "2026-09-29T00:00:00Z")
    root = scenario["repository"]
    (root / "answer.py").write_text(
        "from pathlib import Path\ndef answer():\n"
        "    return 1 if Path('.runtime/fail').exists() else 2\n"
    )
    runtime = root / ".runtime"
    runtime.mkdir()
    task = scenario["intent"].get_task(scenario["task_cid"])
    for fail in (False, True, False):
        flag = runtime / "fail"
        if fail:
            flag.touch()
        elif flag.exists():
            flag.unlink()
        observed = local.run_local_task_validations(
            intent=scenario["intent"], task_cid=scenario["task_cid"], attempt_id="same-source-attempt",
        )
        assert observed["passed"] is (not fail)
        with scenario["intent"]._connection() as connection:
            missing = local.local_completion_missing(connection, task["task_cid"], task["body"], task["revision"])
        assert bool(missing) is fail


def native_published_transition(scenario, tmp_path, owner, attempt, *, created_outputs=None, modified_outputs=None):
    """Produce real Git/queue publication and native Portal observation proofs.

    No provider is dispatched: this exercises the publication/owner join with
    deterministic code bytes, and does not claim a live coding benchmark.
    """
    from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue
    from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import checkout_repository_id
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalImplementationDaemon, parse_task_text
    from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import authorize_owner_portal_source_transition

    root = scenario["repository"]
    def git(*args):
        return subprocess.check_output([
            "git", "-C", str(root), "-c", "user.name=Local validation",
            "-c", "user.email=local@example.invalid", *args,
        ], text=True).strip()

    baseline = git("rev-parse", "HEAD")
    branch = git("symbolic-ref", "--short", "HEAD")
    task = owner.source.get_task(attempt.task_cid)
    bridge = DatabasePortalExecutionBridge(
        task_source=owner.source, attempt_root=tmp_path / "portal-attempts",
        portal_factory=lambda *_: None, repo_root=root, board_namespace="local-publication-test",
        merge_target_branch=branch, task_header_prefix="## " + task.task_alias,
    )
    paths, binding = bridge._ensure_attempt_projection(attempt, task)
    portal = PortalImplementationDaemon(
        todo_path=paths.task_projection, state_path=paths.state, strategy_path=paths.strategy,
        events_path=paths.events, repo_root=root, task_header_prefix="## " + task.task_alias,
        board_namespace=bridge.board_namespace, merge_target_branch=branch,
        max_task_attempts=1, worktree_pool_enabled=False,
    )
    try:
        parsed = parse_task_text(paths.task_projection.read_text(), path=paths.task_projection,
                                 task_header_prefix="## " + task.task_alias)[0]
        canonical = portal._identity_for_task(parsed)
        git("checkout", "-q", "-b", "implementation/local-check")
        modifications = {"answer.py": "def answer():\n    return 2\n"} if modified_outputs is None else modified_outputs
        for name, contents in modifications.items():
            (root / name).write_text(contents)
        for name, contents in (created_outputs or {}).items():
            (root / name).write_text(contents)
        git("add", *modifications, *(created_outputs or {}))
        git("commit", "-qm", "Repair declared output")
        implementation = git("rev-parse", "HEAD")
        git("checkout", "-q", branch)
        queue = MergeQueue(tmp_path / "merge-queue", target_repository_id=checkout_repository_id(root),
                           target_branch=branch, require_target_binding=True)
        request = queue.enqueue(
            branch_name="implementation/local-check", task_id=task.task_alias, attempt=1,
            commit_sha=implementation, canonical_task_cid=canonical.canonical_task_cid,
            canonical_task_key=canonical.canonical_task_key,
            metadata={
                "baseline_ref": baseline, "implementation_commit": implementation,
                "repo_root": str(root), "completion_task_cids": {task.task_alias: canonical.canonical_task_cid},
                "task": {
                    "task_id": task.task_alias, "board_namespace": bridge.board_namespace,
                    "canonical_task_cid": canonical.canonical_task_cid,
                    "canonical_task_key": canonical.canonical_task_key,
                    "metadata": {"database attempt id": attempt.attempt_id,
                                 "database claim id": attempt.claim_id, "database task cid": attempt.task_cid},
                },
            },
        )
        claimed = queue.claim_pending_request(request, consumer_id="local-owner-publication-check")
        assert claimed is not None
        git("merge", "--no-ff", "-qm", "Accept repair", implementation)
        published = git("rev-parse", "HEAD")
        proof = portal._immutable_integration_commit(
            {"merge_commit": published}, implementation_commit=implementation, target_branch=branch,
        )
        invariant = portal._declared_output_tracking_invariant([parsed], repository_ref=published)
        assert proof["passed"] and invariant["passed"]
        queue.complete(claimed, metadata={"merge_commit": published})
        portal._record_event("implementation_finished", {
            "returncode": 0, "task_id": task.task_alias, "attempt": 1,
            "board_namespace": bridge.board_namespace,
            "board_completion": {"complete": True, "pending_merge": False, "reason": "merged_into_target"},
            "baseline_ref": baseline, "implementation_commit": implementation,
            "canonical_task_cid": canonical.canonical_task_cid, "canonical_task_key": canonical.canonical_task_key,
            "merge_result": {
                "merged": True, "returncode": 0, "request_id": request.request_id,
                "baseline_ref": baseline, "implementation_commit": implementation,
                "merge_commit": published, "target_branch": branch,
                "target_repository_id": checkout_repository_id(root),
                "canonical_task_cid": canonical.canonical_task_cid, "canonical_task_key": canonical.canonical_task_key,
                "integration_commit_proof": proof, "post_merge_declared_output_invariant": invariant,
            },
        })
        portal._record_event("task_completed", {"task_id": task.task_alias})
        paths.task_projection.write_text(paths.task_projection.read_text().replace("- Status: ready", "- Status: completed"))
        receipt = bridge._acceptance_receipt(
            attempt=attempt, paths=paths, binding=binding, summaries=[], merge_request_loader=queue.get,
        )
        return authorize_owner_portal_source_transition(
            server=owner.server, bridge=bridge, merge_queue=queue, attempt=attempt, provider_result=receipt,
        )
    finally:
        portal.close_event_runtime()


def test_exact_native_published_merge_needs_owner_checks_before_typed_completion(scenario, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.control.profile_authority import LocalProfileTampered
    from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import verify_owner_local_benchmark_observation

    admission = local.admit_local_benchmark_plan(graph=scenario["graph"], manifest=scenario["manifest"])
    with typed_claim(scenario, tmp_path) as (owner, _daemon, attempt):
        task = owner.source.get_task(attempt.task_cid)
        assert verify_owner_local_benchmark_observation(server=owner.server, admission=admission)["receipt"]["completion_authority"] is False
        transition = native_published_transition(scenario, tmp_path, owner, attempt)
        with pytest.raises(local.LocalPlanningError, match="no exact native owner observation"):
            verify_owner_local_benchmark_observation(server=owner.server, admission=admission)
        # A genuine published source transition still does not mean tests ran.
        with pytest.raises((TaskSourceConflictError, TransactionError)):
            complete(owner, attempt, local.content_identity(transition))
        with pytest.raises(local.LocalPlanningError, match="baseline drift"):
            run_owner_local_task_validations(
                server=owner.server, task_cid=task.task_cid, attempt_id=attempt.attempt_id,
                expected_revision=task.revision,
            )
        altered = deepcopy(transition)
        altered["payload"]["changed_paths"].append("test_answer.py")
        with pytest.raises(LocalProfileTampered):
            run_owner_local_task_validations(
                server=owner.server, task_cid=task.task_cid, attempt_id=attempt.attempt_id,
                expected_revision=task.revision, source_transition=altered,
            )
        passed = run_owner_local_task_validations(
            server=owner.server, task_cid=task.task_cid, attempt_id=attempt.attempt_id,
            expected_revision=task.revision, source_transition=transition,
        )
        assert passed["passed"] is True
        assert task.body[local.CONTRACT_KEY]["payload"]["planning_receipt_cid"] == local.content_identity(admission["receipt"])
        verified = verify_owner_local_benchmark_observation(server=owner.server, admission=admission)
        assert verified["current_source_tree_id"] == passed["source_tree_id"]
        with pytest.raises(local.LocalPlanningError, match="baseline drift"):
            local.verify_local_benchmark_admission(admission, initial=True)
        complete(owner, attempt, passed["results"][0]["evidence_digest"])
        assert owner.source.get_task(task.task_cid).status == "completed"
        assert verify_owner_local_benchmark_observation(server=owner.server, admission=admission)["current_source_tree_id"] == passed["source_tree_id"]
        (scenario["repository"] / "test_answer.py").write_text("assert True\n")
        with pytest.raises(local.LocalPlanningError):
            verify_owner_local_benchmark_observation(server=owner.server, admission=admission)


def test_passed_portal_hook_runs_closed_owner_validation_service(scenario, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import bind_owner_local_completion_service
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import TypedStateOwnerRemoteError

    with typed_claim(scenario, tmp_path) as (owner, _daemon, attempt):
        task = owner.source.get_task(attempt.task_cid)
        native_published_transition(scenario, tmp_path, owner, attempt)
        branch = subprocess.check_output(["git", "-C", str(scenario["repository"]), "symbolic-ref", "--short", "HEAD"], text=True).strip()
        bind_owner_local_completion_service(
            server=owner.server, portal_attempt_root=tmp_path / "portal-attempts",
            repo_root=scenario["repository"], merge_queue_dir=tmp_path / "merge-queue",
            board_namespace="local-publication-test", target_branch=branch,
        )
        connection = owner.client._adapter.raw
        for extra in ({"argv": ["arbitrary-command"]}, {"expected_revision": task.revision - 1}, {"attempt_id": "wrong-attempt"}):
            request = {"task_cid": task.task_cid, "attempt_id": attempt.attempt_id, "expected_revision": task.revision, **extra}
            with pytest.raises(TypedStateOwnerRemoteError):
                connection._request("local.task.validation.run", **request)
        generic = "sha256:" + hashlib.sha256(b"actual Portal publication observed").hexdigest()
        # This actual worker-side API requests owner execution over the native
        # authenticated socket before recording ordinary Portal evidence.
        owner.source.record_validation_result(
            task_cid=task.task_cid, outcome="passed", evidence_digest=generic,
            argv=["portal-supervisor-gates"], attempt_id=attempt.attempt_id,
            body={"validator": "DatabasePortalExecutionBridge@1"},
        )
        complete(owner, attempt, generic)
        assert owner.source.get_task(task.task_cid).status == "completed"


def test_exact_created_output_is_bound_through_native_published_completion(scenario, tmp_path):
    from test.api.test_local_planning_declared_create import _with_creation

    graph, manifest = _with_creation(scenario)
    scenario = {**scenario, "graph": graph, "manifest": manifest, "task_cid": graph.tasks[0].task_cid}
    assert not (scenario["repository"] / "report.jsonl").exists()
    with typed_claim(scenario, tmp_path) as (owner, _daemon, attempt):
        task = owner.source.get_task(attempt.task_cid)
        transition = native_published_transition(
            scenario, tmp_path, owner, attempt, created_outputs={"report.jsonl": '{"answer":2}\n'},
        )
        assert transition["payload"]["changed_paths"] == ["answer.py", "report.jsonl"]
        assert "report.jsonl" in transition["payload"]["sources"]
        observed = run_owner_local_task_validations(
            server=owner.server, task_cid=task.task_cid, attempt_id=attempt.attempt_id,
            expected_revision=task.revision, source_transition=transition,
        )
        assert observed["passed"] is True
        complete(owner, attempt, observed["results"][0]["evidence_digest"])
        assert owner.source.get_task(task.task_cid).status == "completed"
