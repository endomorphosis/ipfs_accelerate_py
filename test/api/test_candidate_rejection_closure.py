"""Protected cleanup CAS and actual lifecycle/typed claim disposition.

Docker absence/dispatch observations are explicit doubles. Signed route and CAS,
Git preservation, lifecycle transitions, Portal events and typed task control are
real disposable fixtures; these tests grant no live provider closure authority.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.control import provider_attempt_store as cas
from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner as runner
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    candidate_rejection_closure as closure,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as impl,
)
from test.api import test_llm_router_agent_supervisor_fallback_route as routes
from test.api.test_agent_supervisor_grok_quota_terra_gate import (
    _created_docker_termination_fence,
)
from test.api.test_lifecycle_terminal_callback_replay import (
    lifecycle as _lifecycle_fixture,
)
from test.api.test_protected_watchdog_cleanup_handoff import _publish_prepared


@pytest.fixture
def lifecycle(tmp_path):
    return _lifecycle_fixture.__wrapped__(tmp_path)


def _native_cleanup(
    tmp_path, monkeypatch, *, repository, route, invocation, complete=True
):
    now = int(time.time() * 1000)
    nonce = "c" * 64
    primary = llm_router.build_agent_implementation_failure_receipt(
        probe_stderr_text="not signed in",
        nonce=nonce,
        model="grok-4.5",
        probe_returncode=1,
        observed_at_ms=now,
    )
    decision = llm_router.decide_agent_implementation_fallback(
        route,
        repo_root=repository,
        failure_receipt=primary,
        expected_nonce=nonce,
        expected_model="grok-4.5",
        expected_probe_returncode=1,
        expected_invocation_binding=invocation.signed_payload(),
        now_ms=now,
        max_age_ms=60000,
    )
    assert decision.authorized
    authorization = llm_router.build_agent_implementation_effect_authorization_context(
        route=route,
        repo_root=repository,
        failure_receipt=primary,
        decision=decision,
        expected_nonce=nonce,
        expected_model="grok-4.5",
        expected_probe_returncode=1,
    )
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    state = tmp_path / "runner-state"
    (state / "run").mkdir(parents=True)
    for name in runner._DOCKER_WATCHDOG_LIFECYCLE_ENV_NAMES:
        monkeypatch.setenv(name, "fixture")
    for name, value in {
        runner.STATE_ROOT_ENV: str(state),
        runner.RUN_ROOT_ENV: str(state / "run"),
        runner.REPOSITORY_ROOT_ENV: str(repository),
        runner.FENCING_EPOCH_ENV: "1",
    }.items():
        monkeypatch.setenv(name, value)
    original = routes._protected_effect_launch_context
    with monkeypatch.context() as mp:
        mp.setattr(tempfile, "tempdir", str(root))
        mp.setattr(
            routes,
            "_protected_effect_launch_context",
            lambda: original(workspace=invocation.workspace_path),
        )
        context, paths = routes._live_cleanup_launch_context()
    cleanup = context["cleanup_receipt"]
    cleanup["watchdog_pid"] = os.getpid()
    cleanup["watchdog_start_ticks"] = runner._runner_process_start_ticks(os.getpid())
    cleanup.pop("receipt_id")
    cleanup["receipt_id"] = routes._effect_detail_identity(cleanup)
    context["cleanup_id"] = cleanup["receipt_id"]
    routes._refresh_effect_detail_identities(context)
    store = cas.DurableProviderAttemptCAS(
        invocation.provider_attempt_store,
        expected_directory_identity=invocation.provider_attempt_store_identity,
    )
    started = store.reserve_or_adopt(
        logical_attempt_id=invocation.logical_attempt_id,
        route_id=route.route_id,
        decision_id=decision.content_id,
        task_id=invocation.task_id,
        worktree_id=invocation.worktree_id,
        authorized=True,
        authorization_context=authorization,
        launch_context=context,
    )
    case = SimpleNamespace(
        store=store,
        started=started,
        context=context,
        paths=paths,
        binding_path=runner._docker_cleanup_binding_path(context["container_name"]),
        observation={
            "logical_attempt_id": invocation.logical_attempt_id,
            "provider_attempt_store": str(store.directory),
            "provider_attempt_store_identity": store.directory_identity,
        },
    )
    binding = _publish_prepared(case)
    fence = _created_docker_termination_fence(
        container_name=context["container_name"],
        container_id=str(context["container_id"]).removeprefix("sha256:"),
        image_id=context["image_id"],
    )
    binding.update(
        binding_state="command_bound",
        create_command_id="sha256:" + "a" * 64,
        create_cwd=str(root),
        create_environment_id="sha256:" + "b" * 64,
        termination_fence=fence,
    )
    binding.pop("record_id")
    binding["record_id"] = runner._effect_receipt_identity(binding)
    runner._write_private_control_record(
        case.binding_path.parent, case.binding_path.name, binding, replace_existing=True
    )
    identity = runner._cleanup_path_identity(case.binding_path, directory=False)
    outcome = llm_router.build_agent_implementation_route_outcome(
        receipt=primary,
        route=route,
        decision="fallback_succeeded",
        verifier_status=decision.verifier_status,
        fallback_dispatched=True,
        fallback_returncode=0,
        decision_id=decision.content_id,
        reservation_id=started.reservation.reservation_id,
        effect_launch_receipt=started.reservation.effect_launch_receipt,
    )
    terminal = store.complete(
        started.reservation,
        returncode=0,
        outcome=outcome,
        completion_capability=started.completion_capability,
        terminal_cleanup_evidence={
            "binding_path": str(case.binding_path),
            "binding_record_id": binding["record_id"],
            "termination_fence_id": fence["fence_id"],
        },
    )
    if complete:
        birth = runner.read_process_birth(os.getpid())
        dispatch = runner._docker_removal_dispatch_value(
            binding_path=case.binding_path,
            binding_record=binding,
            termination_fence=fence,
            issuer_process_birth={
                "pid": os.getpid(),
                "start_time_ticks": birth.start_time_ticks,
                "boot_id": birth.boot_id,
                "parent_pid": os.getppid(),
            },
            state="request_completed",
            generation=1,
            previous_dispatch_id="sha256:" + "c" * 64,
            docker_returncode=0,
            failure_kind="",
        )
        dispatch_path = runner._docker_removal_dispatch_path(case.binding_path)
        runner._write_private_control_record(
            dispatch_path.parent, dispatch_path.name, dispatch, replace_existing=False
        )
        assert runner._finalize_verified_cleanup_completion(
            binding_path=case.binding_path,
            binding_identity=identity,
            binding_record=binding,
            terminal_cleanup_store=store,
            terminal_cleanup_reservation=terminal,
        )
    command = [
        "python",
        "-m",
        "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner",
        "--agent-implementation-route-json",
        json.dumps(route.as_binding_dict()),
    ]
    text = (
        routes.render_grok_failure_receipt(primary)
        + "\n"
        + llm_router.render_agent_implementation_route_outcome(outcome)
        + "\n"
    )
    case.command, case.receipt_text = command, text
    return case


@pytest.mark.parametrize("complete", [False, True])
def test_signed_native_terminal_requires_actual_cleanup_completion(
    tmp_path, monkeypatch, complete
):
    repository, _key, route, invocation = routes._reviewed_route(tmp_path)
    case = _native_cleanup(
        tmp_path,
        monkeypatch,
        repository=repository,
        route=route,
        invocation=invocation,
        complete=complete,
    )
    try:
        observed = impl.PortalImplementationDaemon._protected_provider_effect_audit(
            repo_root=repository,
            command_items=case.command,
            receipt_text=case.receipt_text,
            returncode=0,
        )
        assert bool(observed.get("candidate_provider_cleanup")) is complete
        if complete:
            proof = observed["candidate_provider_cleanup"]
            assert closure.exact_seal(proof, "proof_id")
            assert (
                proof["cleanup_progress_id"]
                == case.store.observe(
                    invocation.logical_attempt_id
                ).terminal_cleanup_progress["progress_id"]
            )
            wrong = impl.PortalImplementationDaemon._protected_provider_effect_audit(
                repo_root=repository,
                command_items=case.command,
                receipt_text=case.receipt_text,
                returncode=78,
            )
            assert not wrong.get("candidate_provider_cleanup")
    finally:
        routes._discard_live_cleanup_inputs(case.paths)


def _handoff(case, events):
    record = case.prior
    proof = closure.sealed(
        {
            "schema": closure.PROVIDER_SCHEMA,
            "task_id": record.task_id,
            "attempt": record.attempt,
            "workspace_path": record.workspace_path,
            "task_revision_cid": record.canonical_task_cid,
        },
        "proof_id",
    )
    return closure.CandidateLifecycleHandoff(
        cleanup=proof,
        record_event=lambda typ, v: events.append((typ, v)),
        task_id=record.task_id,
        attempt=record.attempt,
        workspace_path=record.workspace_path,
        branch=record.branch,
        preserved_commit="a" * 40,
        rescue_branch="rescue/candidate",
    )


def test_real_lifecycle_records_both_sides_of_exact_delete(lifecycle):
    events = []
    handoff = _handoff(lifecycle, events)
    result = lifecycle.daemon._finalize_exact_worktree_lifecycle(
        lifecycle.prior,
        reason="worktree_cleaned",
        terminal_callback=handoff.terminal,
        released_callback=handoff.released,
    )
    assert result["finalized"] is True
    assert [e[0] for e in events] == [closure.TERMINAL_EVENT, closure.RELEASED_EVENT]
    assert lifecycle.store.load_workspace(lifecycle.prior.workspace_path) is None
    assert (
        result["released_callback"]["terminal_receipt_id"]
        == result["terminal_callback"]["receipt_id"]
    )


def test_post_delete_event_failure_never_synthesizes_release_from_missing_row(
    lifecycle,
):
    events = []
    handoff = _handoff(lifecycle, events)

    def failed(*_):
        raise OSError("release publication lost")

    with pytest.raises(OSError, match="publication lost"):
        lifecycle.daemon._finalize_exact_worktree_lifecycle(
            lifecycle.prior,
            reason="worktree_cleaned",
            terminal_callback=handoff.terminal,
            released_callback=failed,
        )
    assert (
        len(events) == 1
        and lifecycle.store.load_workspace(lifecycle.prior.workspace_path) is None
    )
    result = lifecycle.daemon._finalize_exact_worktree_lifecycle(
        lifecycle.prior,
        reason="worktree_cleaned",
        terminal_callback=handoff.terminal,
        released_callback=handoff.released,
    )
    assert result["finalized"] is False
    assert len(events) == 1


def _rebind_route(repository, route, invocation, key, *, task_id, task_cid, workspace):
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
        content_identity,
    )

    baseline = routes._git(repository, "rev-parse", "HEAD")
    worktree_id = content_identity(
        {"workspace_path": str(workspace.resolve()), "baseline_commit": baseline}
    )
    logical = {
        "task_id": task_id,
        "task_revision_cid": task_cid,
        "attempt": 1,
        "prompt_cid": invocation.prompt_cid,
        "worktree_id": worktree_id,
        "route_id": route.route_id,
    }
    logical_id = content_identity(logical)
    unsigned = replace(
        invocation,
        task_id=task_id,
        task_revision_cid=task_cid,
        worktree_id=worktree_id,
        workspace_path=str(workspace.resolve()),
        baseline_commit=baseline,
        logical_attempt_id=logical_id,
        invocation_id=content_identity({**logical, "logical_attempt_id": logical_id}),
        reviewer_signature="pending",
    )
    signed = replace(
        unsigned, reviewer_signature=routes._sign(key, unsigned.signed_payload())
    )
    bound = llm_router.bind_agent_implementation_route_invocation(
        replace(route, invocation_binding=None),
        signed,
        repo_root=repository,
        workspace=workspace,
        expected_binding=signed.signed_payload(),
        now_ms=signed.issued_at_ms,
        max_age_ms=60000,
    )
    return bound, signed


def _preserved_candidate_pass(
    tmp_path,
    monkeypatch,
    *,
    bridge,
    paths,
    task,
    route_fixture,
    complete=True,
    omit_post=False,
    journal=False,
):
    """Exercise native audit + real preservation helper and lifecycle callbacks.

    Candidate bytes/rejection policy are explicit fixture inputs. No accepted
    outputs/terminal task receipts or coordinator release are fabricated.
    """
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        WorktreeLifecycleStore,
        current_process_birth,
    )

    append_jsonl_event = impl.append_jsonl_event
    repository, key, route, invocation = route_fixture
    binding = json.loads(paths.binding.read_text())
    projection = paths.task_projection.read_text()
    portal_key, portal_cid = bridge._portal_completion_event_identity(
        paths=paths, projection_text=projection, binding=binding
    )
    worktree = tmp_path / "workspaces" / ("candidate-" + binding["attempt_id"][-10:])
    worktree.parent.mkdir(exist_ok=True)
    branch = "implementation/candidate-" + binding["attempt_id"][-10:]
    routes._git(repository, "worktree", "add", "-qb", branch, str(worktree), "HEAD")
    bound, signed = _rebind_route(
        repository,
        route,
        invocation,
        key,
        task_id=task.task_id,
        task_cid=portal_cid,
        workspace=worktree,
    )
    local = tmp_path / ("provider-" + binding["attempt_id"][-10:])
    local.mkdir()
    case = _native_cleanup(
        local,
        monkeypatch,
        repository=repository,
        route=bound,
        invocation=signed,
        complete=complete,
    )
    paths.implementation_logs.mkdir(exist_ok=True)
    log = paths.implementation_logs / "provider.log"
    log.write_text(case.receipt_text)
    log.chmod(0o600)
    daemon = object.__new__(impl.PortalImplementationDaemon)
    daemon.repo_root = repository
    daemon.worktree_root = worktree.parent
    daemon.state_path = paths.state
    daemon._worktree_pool_leases = {}
    daemon._worktree_pool_effective_paths = {}
    daemon.worktree_submodule_paths = ()
    daemon.worktree_lifecycle = WorktreeLifecycleStore(
        repo_root=repository, **({} if journal else {"store_dir": tmp_path / "workspace-lifecycle"})
    )
    prior = daemon.worktree_lifecycle.begin_preparing(
        task_id=task.task_id,
        canonical_task_cid=portal_cid,
        attempt=1,
        lane_id="lane-fixture",
        workspace_path=worktree,
        branch=branch,
        merge_target="main",
        owner=current_process_birth(),
        state_dir=str(paths.state.parent.resolve()),
    )
    prior = daemon.worktree_lifecycle.mark_settling(
        worktree, lease_id=prior.lease_id, expected_fence=prior.fence
    )
    daemon._active_worktree_lifecycle = prior

    def event(kind, value):
        if omit_post and kind == closure.RELEASED_EVENT:
            raise OSError("post-delete event unavailable")
        append_jsonl_event(
            paths.events,
            kind,
            {
                **value,
                "task_id": task.task_id,
                "attempt": 1,
                "canonical_task_key": portal_key,
                "canonical_task_cid": portal_cid,
                "task_cid": portal_cid,
            },
        )

    daemon._record_event = event
    event(
        "implementation_started",
        {
            "command": case.command,
            "log_path": str(log),
            "worktree_path": str(worktree),
            "branch": branch,
            "provider_dispatched": False,
        },
    )
    proposal = {
        "attempted": True,
        "accepted": False,
        "proposal_id": "proposal:rejected",
        "receipt_id": "receipt:policy-denied",
        "policy_id": "policy:fixture",
        "reason_codes": ["validation_channel_tampering_forbidden"],
    }
    validation = {
        "attempted": True,
        "passed": False,
        "returncode": 78,
        "reason": "proposal_gate_failed",
        "proposal_gate": proposal,
    }
    event("implementation_proposal_validated", proposal)
    (worktree / "candidate.txt").write_text("preserved rejected candidate\n")
    daemon._drop_unchanged_seeded_worktree_context = lambda *_a, **_k: []

    def commit(workspace, *_a, **_k):
        routes._git(workspace, "add", "candidate.txt")
        routes._git(
            workspace,
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "preserved rejected candidate",
        )
        return {
            "committed": True,
            "commit": routes._git(workspace, "rev-parse", "HEAD"),
        }

    daemon._commit_worktree_changes = commit
    daemon._forget_seeded_worktree_context = lambda *_a: None
    daemon._cleanup_worktree_submodules = lambda *_a, **_k: []
    daemon._submodule_cleanup_failures = lambda _v: []
    daemon._git_ref_exists = lambda ref: bool(
        __import__("subprocess")
        .run(
            ["git", "show-ref", "--verify", "--quiet", "refs/heads/" + ref],
            cwd=repository,
        )
        .returncode
        == 0
    )
    daemon._run_git = lambda args, **kw: __import__("subprocess").run(
        ["git", *args],
        cwd=kw.get("cwd", repository),
        text=True,
        capture_output=True,
        check=True,
    )
    daemon._worktree_path_registered_in_repo = lambda *_: False
    # Cleanup authorization uses the actual owner/lifecycle CAS. The helper
    # needs the same current lane state dir that was supplied at begin.
    daemon._worktree_lifecycle_lane_state_dir = lambda: str(
        paths.state.parent.resolve()
    )
    audit = daemon._protected_provider_effect_audit(
        repo_root=repository,
        command_items=case.command,
        receipt_text=case.receipt_text,
        returncode=0,
    )
    proof = audit.get("candidate_provider_cleanup")
    # Existing tests retain the legacy @1 producer contract. New journal tests
    # use the actual future constructor and canonical native lifecycle store.
    constructor = closure.CandidateLifecycleHandoff.__init__
    def legacy_handoff(self, **kwargs):
        kwargs.pop("journal_validation", None)
        constructor(self, **kwargs)
    with monkeypatch.context() as legacy:
        if not journal:
            legacy.setattr(closure.CandidateLifecycleHandoff, "__init__", legacy_handoff)
        preserved = daemon._preserve_failed_validation_worktree(
            worktree,
            branch,
            task,
            1,
            validation,
            baseline_ref=signed.baseline_commit,
            candidate_cleanup_evidence=proof,
            candidate_cleanup_required=True,
        )

    result = {
        "task_id": task.task_id,
        "task_cid": portal_cid,
        "canonical_task_cid": portal_cid,
        "canonical_task_key": portal_key,
        "attempt": 1,
        "returncode": 78,
        "provider_dispatched": True,
        "attempt_consumed": True,
        "validation_result": validation,
        "failed_preservation_result": preserved,
        "cleanup_result": preserved.get("cleanup_result", {"cleaned": False}),
        "merge_result": {"merged": False, "reason": "not_attempted"},
        "board_completion": {"complete": False},
        "protected_path_violation": {},
        "implementation_commit": preserved.get("implementation_commit", {}),
    }
    event("implementation_finished", result)
    return {"implementation_result": result}


from contextlib import contextmanager


@contextmanager
def _typed_outer(tmp_path, monkeypatch, *, repository, factory, tick=False):
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        OwnerLiveness,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        FakeQuackTransport,
        build_server,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        open_duckdb_connection,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
        QuackStateClient,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
        GROK_CODEX_EXECUTION_MODE,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
        TypedDatabaseTaskSource,
        daemon_required_owner_command_operations,
        daemon_required_owner_operations,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        TYPED_STATE_OWNER_SOCKET_ENV,
        TYPED_STATE_OWNER_TOKEN_ENV,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
        DatabasePortalExecutionBridge,
    )
    from test.api.causal_federation.test_admitted_executor import (
        _typed_bootstrap_credentials,
    )
    from test.api.causal_federation.test_bootstrap_runtime import _capability, _migrate
    from test.api.test_agent_supervisor_database_implementation_daemon import (
        _population,
    )

    database = tmp_path / "owner.duckdb"
    initial = DatabaseTaskSource(database)
    initial.materialize(_population(1))
    initial.close()
    server = build_server(
        database_path=database,
        state_dir=tmp_path / "owner",
        repository_id="repository:candidate-closure",
        store_id="candidate-closure-v1",
        transport=FakeQuackTransport(),
        capability_probe=_capability,
        migrate=_migrate,
        connection_factory=open_duckdb_connection,
        owner_liveness_probe=lambda _: OwnerLiveness.DEAD,
    )
    owner = server.start()
    client_id = "database-implementation-daemon:closure-test"
    token, _ = server.issue_typed_client_grant_record(
        client_id=client_id,
        process_birth_id=owner.process_birth_id,
        allowed_operations=daemon_required_owner_operations(),
        allowed_command_operations=daemon_required_owner_command_operations(),
        peer_pid=os.getpid(),
    )
    monkeypatch.setenv(
        TYPED_STATE_OWNER_SOCKET_ENV, str(server.typed_command_socket_path())
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    client = QuackStateClient(
        owner_id=client_id,
        store_id=owner.store_id,
        process_birth_id=owner.process_birth_id,
    )
    outer = source = None
    try:
        client.attach(owner.listen_uri, server_id=owner.server_id)
        unsealed = TypedDatabaseTaskSource(client, owns_client=False)
        tasks = unsealed.list_tasks().tasks
        policy = unsealed.seal_execution_route_policy(
            {task.task_alias: GROK_CODEX_EXECUTION_MODE for task in tasks}
        )
        unsealed.close()
        source = TypedDatabaseTaskSource(client, execution_route_policy=policy)
        credentials = _typed_bootstrap_credentials(
            server=server,
            identity=owner,
            client_id=client_id,
            token=token,
            route_policy=policy,
        )
        bridge = DatabasePortalExecutionBridge(
            task_source=source,
            attempt_root=tmp_path / "attempts",
            portal_factory=factory,
            repository_root=repository,
            max_task_attempts=4,
        )
        outer = impl.DatabaseImplementationDaemon(
            database_path=database,
            coordination_path=tmp_path / "coordination.duckdb",
            execution_path=tmp_path / "execution.duckdb",
            owner_session_id="session:closure",
            process_instance_id=owner.process_birth_id,
            authority_mode="quack",
            task_source_kind="duckdb",
            quack_uri=owner.listen_uri,
            task_source=source,
            close_task_source=False,
            state_owner_bootstrap_credentials=credentials,
            lease_ms=30000,
            max_task_attempts=4,
            provider_fn=bridge.run_provider,
            effect_fn=lambda *_: pytest.fail("no effect callback"),
            validation_fn=lambda *_: pytest.fail("no validation callback"),
            strict_task_sharding=True,
            require_real_execution=True,
        ).open()
        if tick:
            outer.bind_post_merge_recovery(
                lambda: bridge.recover_post_merge_declared_outputs(outer)
            )
        yield outer, bridge, source
    finally:
        if outer is not None:
            outer.close()
        if source is not None:
            source.close()
        client.close()
        server.stop()


def test_actual_typed_candidate_closure_releases_exact_claim_for_natural_successor(
    tmp_path, monkeypatch
):
    from test.api.test_database_attempt_diagnostic_feedback import prompt_daemon

    route_fixture = routes._reviewed_route(tmp_path)
    calls = []

    def factory(paths, alias):
        daemon = prompt_daemon(monkeypatch)
        daemon.repo_root = route_fixture[0]
        daemon.bind_launch_task_execution_route = lambda _: None
        daemon.close_event_runtime = lambda: None

        def run():
            task = impl.parse_task_text(
                paths.task_projection.read_text(),
                path=paths.task_projection,
                task_header_prefix=f"## {alias}",
            )[0]
            calls.append(task.task_id)
            return _preserved_candidate_pass(
                tmp_path,
                monkeypatch,
                bridge=bridge,
                paths=paths,
                task=task,
                route_fixture=route_fixture,
            )

        daemon.run_once = run
        return daemon

    with _typed_outer(
        tmp_path, monkeypatch, repository=route_fixture[0], factory=factory
    ) as (outer, bridge, source):
        attempt = outer.claim_next()
        assert attempt is not None
        result = outer._resume_attempt_without_process_crash(attempt)
        assert result["status"] == "failed"
        prior = outer.provider_invocation_recorded(
            attempt.attempt_id, idempotency_key=f"provider:{attempt.attempt_id}"
        )
        assert prior["schema"] == closure.CALLBACK_SCHEMA
        assert (
            outer.coordinator.get_task_claim(attempt.claim_id).state.value == "released"
        )
        successor = outer.claim_next()
        assert successor is not None
        assert successor.attempt_number == attempt.attempt_number + 1
        assert len(calls) == 1
        assert source.get_task(successor.task_cid).status == "in_progress"


@pytest.mark.parametrize("missing", ["cleanup", "post_delete"])
def test_actual_typed_incomplete_candidate_keeps_original_unknown_custody(
    tmp_path, monkeypatch, missing
):
    from test.api.test_database_attempt_diagnostic_feedback import prompt_daemon

    route_fixture = routes._reviewed_route(tmp_path)
    calls = []

    def factory(paths, alias):
        daemon = prompt_daemon(monkeypatch)
        daemon.repo_root = route_fixture[0]
        daemon.bind_launch_task_execution_route = lambda _: None
        daemon.close_event_runtime = lambda: None

        def run():
            task = impl.parse_task_text(
                paths.task_projection.read_text(),
                path=paths.task_projection,
                task_header_prefix=f"## {alias}",
            )[0]
            calls.append(task.task_id)
            return _preserved_candidate_pass(
                tmp_path,
                monkeypatch,
                bridge=bridge,
                paths=paths,
                task=task,
                route_fixture=route_fixture,
                complete=missing != "cleanup",
                omit_post=missing == "post_delete",
            )

        daemon.run_once = run
        return daemon

    with _typed_outer(
        tmp_path, monkeypatch, repository=route_fixture[0], factory=factory
    ) as (outer, bridge, _source):
        attempt = outer.claim_next()
        assert attempt is not None
        with pytest.raises(closure.CandidateClosureObservationUnknown):
            outer._resume_attempt_without_process_crash(attempt)
        prior = outer.provider_invocation_recorded(
            attempt.attempt_id, idempotency_key=f"provider:{attempt.attempt_id}"
        )
        assert prior["callback_state"] == "started_outcome_unknown"
        assert prior["provider_effect_state"] == "unknown_may_have_started"
        assert (
            outer.coordinator.get_task_claim(attempt.claim_id).state.value == "accepted"
        )
        assert outer.claim_next() is None
        assert len(calls) == 1
        if missing == "cleanup":
            assert list((tmp_path / "workspaces").glob("candidate-*/candidate.txt"))
        assert bridge.verify_candidate_rejection_closure(attempt) is None


@pytest.mark.parametrize(
    "mode",
    ["removed", "incomplete", "missing", "foreign_store", "symlink", "changed_cas"],
)
def test_terminal_observer_is_not_live_effect_authority(tmp_path, monkeypatch, mode):
    from ipfs_accelerate_py import agent_implementation_route as authority

    repository, key, route, original = routes._reviewed_route(tmp_path)
    parent = tmp_path / "candidate-parent"
    parent.mkdir()
    workspace = parent / "candidat-é"
    routes._git(repository, "worktree", "add", "--detach", str(workspace), "HEAD")
    route, invocation = _rebind_route(
        repository,
        route,
        original,
        key,
        task_id=original.task_id,
        task_cid=original.task_revision_cid,
        workspace=workspace,
    )
    case = _native_cleanup(
        tmp_path,
        monkeypatch,
        repository=repository,
        route=route,
        invocation=invocation,
        complete=mode != "incomplete",
    )
    try:
        produced = impl.PortalImplementationDaemon._protected_provider_effect_audit(
            repo_root=repository,
            command_items=case.command,
            receipt_text=case.receipt_text,
            returncode=0,
        ).get("candidate_provider_cleanup")
        routes._git(repository, "worktree", "remove", str(workspace))
        assert not workspace.exists()
        terminal = case.store.observe(invocation.logical_attempt_id)
        assert (
            llm_router.parse_agent_implementation_effect_authorization_context(
                terminal.authorization_context,
                repo_root=repository,
                effect_started_at_ms=terminal.effect_started_at_ms,
                expected_signer_parent_pid=terminal.effect_launch_receipt[
                    "effect_owner_pid"
                ],
                max_age_ms=60000,
            )
            is None
        )
        with pytest.raises(ValueError):
            llm_router.bind_agent_implementation_route_invocation(
                route,
                invocation,
                repo_root=repository,
                workspace=workspace,
                now_ms=invocation.issued_at_ms,
                max_age_ms=60000,
            )
        store_identity = invocation.provider_attempt_store_identity
        logical = invocation.logical_attempt_id
        if mode == "foreign_store":
            store_identity = "sha256:" + "f" * 64
        elif mode == "missing":
            logical = "sha256:" + "e" * 64
        elif mode == "symlink":
            moved = tmp_path / "candidate-parent-moved"
            parent.rename(moved)
            parent.symlink_to(moved, target_is_directory=True)
        elif mode == "changed_cas":
            observe = cas.DurableProviderAttemptCAS.observe
            calls = []

            def changed(self, logical):
                calls.append(logical)
                return observe(self, logical) if len(calls) == 1 else None

            monkeypatch.setattr(cas.DurableProviderAttemptCAS, "observe", changed)
        evidence = authority.observe_agent_implementation_terminal_cleanup(
            store_path=invocation.provider_attempt_store,
            expected_store_identity=store_identity,
            logical_attempt_id=logical,
            repo_root=repository,
            max_age_ms=60000,
        )
        if mode == "removed":
            assert (
                type(evidence) is authority.AgentImplementationTerminalCleanupEvidence
            )
            value = json.loads(evidence.evidence_json)
            assert value["provider_cleanup"] == produced
            assert (
                value["provider_cleanup"]["cleanup_progress_id"]
                == terminal.terminal_cleanup_progress["progress_id"]
            )
            assert value["new_effect_authority"] is False
            assert not hasattr(evidence, "route") and not hasattr(
                evidence, "authorized"
            )
            # Historical timestamps remain strict on every public effect API.
            common = {
                "repo_root": repository,
                "now_ms": terminal.effect_started_at_ms,
                "max_age_ms": 60000,
                "historical_effect_started_at_ms": terminal.effect_started_at_ms,
            }
            for fn, args, kwargs in (
                (
                    authority.verify_agent_implementation_invocation_binding,
                    (invocation,),
                    {**common, "route": route, "workspace": workspace},
                ),
                (
                    authority.bind_agent_implementation_route_invocation,
                    (route, invocation),
                    {**common, "workspace": workspace},
                ),
                (
                    authority.decide_agent_implementation_fallback,
                    (route,),
                    {
                        **common,
                        "failure_receipt": terminal.authorization_context[
                            "failure_receipt"
                        ],
                        "expected_nonce": terminal.authorization_context[
                            "expected_nonce"
                        ],
                        "expected_model": terminal.authorization_context[
                            "expected_model"
                        ],
                        "expected_probe_returncode": terminal.authorization_context[
                            "expected_probe_returncode"
                        ],
                        "expected_invocation_binding": invocation.signed_payload(),
                    },
                ),
            ):
                with pytest.raises(ValueError):
                    fn(*args, **kwargs)
                with pytest.raises(TypeError, match="_terminal_workspace"):
                    fn(*args, **kwargs, _terminal_workspace=str(workspace))
            with pytest.raises(TypeError, match="_terminal_workspace"):
                authority.parse_agent_implementation_effect_authorization_context(
                    terminal.authorization_context,
                    repo_root=repository,
                    effect_started_at_ms=terminal.effect_started_at_ms,
                    expected_signer_parent_pid=terminal.effect_launch_receipt[
                        "effect_owner_pid"
                    ],
                    max_age_ms=60000,
                    _terminal_workspace=str(workspace),
                )
            assert (
                closure.observe_provider_cleanup(
                    repo_root=repository,
                    command_items=case.command,
                    receipt_text=case.receipt_text,
                )
                == value["provider_cleanup"]
            )
        else:
            assert evidence is None
    finally:
        routes._discard_live_cleanup_inputs(case.paths)


@pytest.mark.parametrize(
    "boundary",
    ["finished_event", "callback_cas", "failed_phase", "retry_cas", "release_cas"],
)
def test_actual_typed_candidate_crash_replay_never_dispatches_provider_twice(
    tmp_path, monkeypatch, boundary
):
    from test.api.test_database_attempt_diagnostic_feedback import prompt_daemon

    route_fixture = routes._reviewed_route(tmp_path)
    calls = []

    def factory(paths, alias):
        daemon = prompt_daemon(monkeypatch)
        daemon.repo_root = route_fixture[0]
        daemon.bind_launch_task_execution_route = lambda _: None
        daemon.close_event_runtime = lambda: None

        def run():
            task = impl.parse_task_text(
                paths.task_projection.read_text(),
                path=paths.task_projection,
                task_header_prefix=f"## {alias}",
            )[0]
            calls.append(task.task_id)
            return _preserved_candidate_pass(
                tmp_path,
                monkeypatch,
                bridge=bridge,
                paths=paths,
                task=task,
                route_fixture=route_fixture,
            )

        daemon.run_once = run
        return daemon

    class InterruptedProcess(BaseException):
        pass

    with _typed_outer(
        tmp_path, monkeypatch, repository=route_fixture[0], factory=factory
    ) as (outer, bridge, source):
        attempt = outer.claim_next()
        assert attempt is not None
        name = (
            "_commit_candidate_callback_closure"
            if boundary in {"finished_event", "callback_cas"}
            else "_release_closed_candidate_claim"
            if boundary == "release_cas"
            else "_persist_task_retry_state"
        )
        original = getattr(outer, name)

        def interrupted(*args, **kwargs):
            if boundary not in {"finished_event", "failed_phase"}:
                original(*args, **kwargs)
            raise InterruptedProcess(boundary)

        with monkeypatch.context() as fault:
            fault.setattr(outer, name, interrupted)
            with pytest.raises(InterruptedProcess, match=boundary):
                outer._resume_attempt_without_process_crash(attempt)
        assert len(calls) == 1
        captured = outer.provider_invocation_recorded(
            attempt.attempt_id, idempotency_key=f"provider:{attempt.attempt_id}"
        )
        current = outer.get_attempt(attempt.attempt_id)
        assert current is not None
        if current.status == "failed":
            # This exact reconciler is called by every ordinary run_once.
            outer.reconcile_terminal_retry_states()
        else:
            outer._resume_attempt_without_process_crash(current)
        replayed = outer.provider_invocation_recorded(
            attempt.attempt_id, idempotency_key=f"provider:{attempt.attempt_id}"
        )
        assert replayed["schema"] == closure.CALLBACK_SCHEMA
        assert replayed["failure_fingerprint"] == captured["failure_fingerprint"]
        if boundary != "finished_event":
            assert replayed == captured
        assert (
            outer.coordinator.get_task_claim(attempt.claim_id).state.value == "released"
        )
        assert source.get_task(attempt.task_cid).status == "retrying"
        # Repeating terminal reconciliation is idempotent after release.
        outer.reconcile_terminal_retry_states()
        assert (
            outer.provider_invocation_recorded(
                attempt.attempt_id, idempotency_key=f"provider:{attempt.attempt_id}"
            )
            == replayed
        )
        successor = outer.claim_next()
        assert (
            successor is not None
            and successor.attempt_number == attempt.attempt_number + 1
        )
        assert len(calls) == 1


@pytest.mark.parametrize(
    "fault",
    [
        "absent_before",
        "absent_at_unlink",
        "foreign_record",
        "foreign_index",
        "directory_fsync",
    ],
)
def test_lifecycle_release_requires_observed_exact_delete(
    lifecycle, monkeypatch, fault
):
    import stat

    events = []
    handoff = _handoff(lifecycle, events)
    record_path = lifecycle.store.workspace_path_for(lifecycle.prior.workspace_path)
    original_unlink = Path.unlink
    original_fsync = os.fsync

    def disappearing(path, *args, **kwargs):
        if path == record_path:
            original_unlink(path, *args, **kwargs)
            raise FileNotFoundError("record disappeared before our unlink")
        return original_unlink(path, *args, **kwargs)

    def failed_directory_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("directory persistence unobserved")
        return original_fsync(fd)

    def terminal(prior, after):
        value = handoff.terminal(prior, after)
        if fault == "absent_before":
            record_path.unlink()
        elif fault == "absent_at_unlink":
            monkeypatch.setattr(Path, "unlink", disappearing)
        elif fault == "foreign_record":
            raw = json.loads(record_path.read_text())
            raw["terminal_reason"] = "another_terminal_transition"
            record_path.write_text(json.dumps(raw))
        elif fault == "foreign_index":
            index = lifecycle.store.task_index_path_for(
                canonical_task_cid=after.canonical_task_cid,
                task_id=after.task_id,
                attempt=after.attempt,
            )
            raw = json.loads(index.read_text())
            raw["lease_id"] = "foreign-lease"
            index.write_text(json.dumps(raw))
        elif fault == "directory_fsync":
            monkeypatch.setattr(os, "fsync", failed_directory_fsync)
        return value

    result = lifecycle.daemon._finalize_exact_worktree_lifecycle(
        lifecycle.prior,
        reason="worktree_cleaned",
        terminal_callback=terminal,
        released_callback=handoff.released,
    )
    assert result["finalized"] is False
    assert [kind for kind, _ in events] == [closure.TERMINAL_EVENT]
    if fault in {"foreign_record", "foreign_index"}:
        assert record_path.exists()


def test_legacy_missing_lifecycle_delete_stays_idempotent(lifecycle):
    record = lifecycle.store.mark_terminal(
        lifecycle.prior.workspace_path,
        lease_id=lifecycle.prior.lease_id,
        expected_fence=lifecycle.prior.fence,
    )
    assert lifecycle.store.compare_and_delete(
        record.workspace_path, lease_id=record.lease_id, expected_fence=record.fence
    )
    assert lifecycle.store.compare_and_delete(
        record.workspace_path, lease_id=record.lease_id, expected_fence=record.fence
    )
    assert lifecycle.store.compare_and_delete_observed(record) is False


@contextmanager
def _signed_typed_candidate(tmp_path, monkeypatch):
    from test.api.test_database_attempt_diagnostic_feedback import prompt_daemon

    route_fixture = routes._reviewed_route(tmp_path)
    calls = []

    def factory(paths, alias):
        daemon = prompt_daemon(monkeypatch)
        daemon.repo_root = route_fixture[0]
        daemon.bind_launch_task_execution_route = lambda _: None
        daemon.close_event_runtime = lambda: None

        def run():
            task = impl.parse_task_text(
                paths.task_projection.read_text(),
                path=paths.task_projection,
                task_header_prefix=f"## {alias}",
            )[0]
            calls.append(task.task_id)
            return _preserved_candidate_pass(
                tmp_path,
                monkeypatch,
                bridge=bridge,
                paths=paths,
                task=task,
                route_fixture=route_fixture,
            )

        daemon.run_once = run
        return daemon

    with _typed_outer(
        tmp_path, monkeypatch, repository=route_fixture[0], factory=factory
    ) as (outer, bridge, source):
        yield outer, bridge, source, calls


@pytest.mark.parametrize(
    "foreign", ["closure_claim", "original_claim", "original_contract", "original_tree"]
)
def test_actual_callback_cas_rejects_foreign_original_or_closure(
    tmp_path, monkeypatch, foreign
):
    with _signed_typed_candidate(tmp_path, monkeypatch) as (
        outer,
        _bridge,
        _source,
        calls,
    ):
        attempt = outer.claim_next()
        assert attempt is not None
        original = outer._commit_candidate_callback_closure
        captured = []

        def changed(*args, **kwargs):
            captured.append(copy.deepcopy(kwargs["original"]))
            if foreign == "closure_claim":
                value = dict(kwargs["closure"])
                value.pop("receipt_id")
                value["claim_id"] = "claim:foreign"
                kwargs["closure"] = closure.sealed(value)
            else:
                value = dict(kwargs["original"])
                key = {
                    "original_claim": "claim_id",
                    "original_contract": "task_contract_digest",
                    "original_tree": "repository_tree_id",
                }[foreign]
                value[key] = "sha256:" + "f" * 64
                value["failure_fingerprint"] = (
                    impl._database_provider_callback_unknown_fingerprint(value)
                )
                kwargs["original"] = value
            return original(*args, **kwargs)

        with monkeypatch.context() as fault:
            fault.setattr(outer, "_commit_candidate_callback_closure", changed)
            with pytest.raises(impl.DatabaseImplementationAuthorityError):
                outer.run_provider(attempt)
        prior = outer.provider_invocation_recorded(
            attempt.attempt_id, idempotency_key=f"provider:{attempt.attempt_id}"
        )
        assert prior == captured[0]
        assert prior["callback_state"] == "started_outcome_unknown"
        assert (
            outer.coordinator.get_task_claim(attempt.claim_id).state.value == "accepted"
        )
        assert outer.claim_next() is None
        assert len(calls) == 1


def test_closed_candidate_claim_expiry_never_becomes_release_authority(
    tmp_path, monkeypatch
):
    class LostReleaseResponse(BaseException):
        pass

    with _signed_typed_candidate(tmp_path, monkeypatch) as (
        outer,
        _bridge,
        source,
        calls,
    ):
        attempt = outer.claim_next()
        assert attempt is not None
        with monkeypatch.context() as fault:

            def before_release(*_a, **_k):
                raise LostReleaseResponse()

            fault.setattr(outer, "_release_closed_candidate_claim", before_release)
            with pytest.raises(LostReleaseResponse):
                outer._resume_attempt_without_process_crash(attempt)
        claim = outer.coordinator.get_task_claim(attempt.claim_id)
        prior = outer.provider_invocation_recorded(
            attempt.attempt_id, idempotency_key=f"provider:{attempt.attempt_id}"
        )
        assert (
            prior["schema"] == closure.CALLBACK_SCHEMA
            and claim.state.value == "accepted"
        )
        # Real wall time crosses this fixture's actual native coordination lease.
        time.sleep(
            max(0, (claim.expires_at_ms - int(time.time() * 1000)) / 1000) + 0.15
        )
        with pytest.raises(
            impl.DatabaseImplementationAuthorityError, match="live accepted"
        ):
            outer._release_closed_candidate_claim(outer.get_attempt(attempt.attempt_id))
        assert (
            outer.provider_invocation_recorded(
                attempt.attempt_id, idempotency_key=f"provider:{attempt.attempt_id}"
            )
            == prior
        )
        successor = outer.claim_next()
        assert (
            successor is not None
            and successor.attempt_number == attempt.attempt_number + 1
        )
        assert successor.fencing_token > attempt.fencing_token
        assert (
            outer.coordinator.get_task_claim(attempt.claim_id).state.value == "expired"
        )
        assert source.get_task(attempt.task_cid).status == "in_progress"
        Path("/tmp/candidate-closure-expiry-observation.json").write_text(
            json.dumps(
                {
                    "successor": successor.to_dict() if successor else None,
                    "prior_claim_state": outer.coordinator.get_task_claim(
                        attempt.claim_id
                    ).state.value,
                    "task_status": source.get_task(attempt.task_cid).status,
                    "provider_calls": len(calls),
                },
                indent=2,
            )
        )
        assert len(calls) == 1
