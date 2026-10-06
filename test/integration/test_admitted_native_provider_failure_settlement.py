"""Signed native failure settlement with an authored provider boundary, no model calls."""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import sys
import time

import pytest

from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from ipfs_accelerate_py.agent_supervisor.control import profile_authority
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission import verify_local_benchmark_admission
from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import write_task_context_bundle
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA
from test.integration.test_admitted_benchmark_runtime import _prepare_implementation_fixture


def _authored_router_timeout(path: Path) -> None:
    # The runner and its timeout observation are production code. The provider
    # boundary is explicitly authored; this never invokes Grok or another model.
    root = str(Path(__file__).resolve().parents[2])
    path.write_text(f'''import subprocess, sys
from types import SimpleNamespace
sys.path.insert(0, {root!r})
from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
from ipfs_accelerate_py.llm_allocation import intelligence_index
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
intelligence_index.discover_available_providers = lambda: ["codex_cli"]
intelligence_index.select_efficient_route = lambda **kwargs: SimpleNamespace(provider="codex_cli", model_name="fixture", reasoning_effort="high", catalog_revision="fixture")
llm_router.get_llm_provider = lambda *args, **kwargs: object()
def timeout(*args, **kwargs):
    try:
        subprocess.run([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired:
        set_last_cli_observation("codex_cli", {{"timed_out": True}})
        raise
llm_router.generate_text = timeout
runner.run(prompt="Repair the authored fixture", provider="codex_cli", model="fixture", timeout=1, max_output_tokens=128)
''')


def test_signed_native_router_timeout_settles_failed_attempt_and_releases_claim(tmp_path, monkeypatch):
    """Actual START/owner/claim/Portal/STOP; no authority guard is replaced."""
    import duckdb
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import open_database_coordinator

    monkeypatch.setattr(profile_authority, "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE", tmp_path / "account")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(tmp_path / "ambient-unrelated"))
    monkeypatch.delenv("IPFS_DATASETS_PROOF_RESOURCE_PROFILE", raising=False)
    prepared = _prepare_implementation_fixture(tmp_path / "task")
    verified = verify_local_benchmark_admission(prepared["admission"])
    repository = Path(prepared["repository"])
    database = Path(prepared["intent_database"])
    initial_answer = (repository / "answer.py").read_bytes()
    initial_check = (repository / "test_answer.py").read_bytes()
    script = tmp_path / "authored_router_timeout.py"
    _authored_router_timeout(script)
    with open_existing_native_owner(database=database, checkout=repository,
            state_dir=tmp_path / "owner", repository_id=verified["manifest"]["repository_cid"],
            execution_routes={prepared["task_id"]: GROK_CODEX_EXECUTION_MODE}) as owner:
        bundle = write_task_context_bundle(repository=repository, prepared=[{
            "schema": "supervisor-task-context-preparation@1", "task_cid": prepared["task_cid"],
            "task_id": prepared["task_id"], "metadata": {},
        }], output=repository / ".runtime/context.json")
        runtime = AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission=prepared["admission"],
            server=owner.server, source=owner.source, context_bundle=bundle,
            implement=True, implementation_command=shlex.join([sys.executable, "-B", str(script)]),
            implementation_timeout_seconds=20, max_task_attempts=1)
        try:
            argv = runtime.profile.argv
            assert argv[argv.index("--authority-mode") + 1] == "quack"
            assert "--state-owner-bootstrap-fd" in argv
            assert "--state-owner-client-id" in argv
            started = runtime.start()
            assert started.succeeded, started.error
            deadline = time.monotonic() + 90
            while True:
                task = owner.source.get_task(prepared["task_cid"])
                if task.status in {"failed", "blocked", "completed", "cancelled"}:
                    break
                assert time.monotonic() < deadline, (task.status, task.revision)
                assert runtime.process.snapshot(runtime.profile).members
                time.sleep(.25)
            assert task.status == "blocked"
            assert runtime.bootstrap_receipts and runtime.bootstrap_errors == []
            history = owner.source.task_revision_diagnostic_window(task.task_cid, current_revision=task.revision)
            admitted = [row for row in history["revisions"]
                        if row["body"].get("completion_receipt", {}).get("claim_phase_schema")
                        == TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA]
            assert len(admitted) == 1
            admission = admitted[0]["body"]["completion_receipt"]
            assert admission["operation"] == "database_attempt_admitted"
            assert admission["claim_process_attestation"]
            # Let another native selection pass run: terminal failure must not
            # automatically launch another provider callback.
            time.sleep(.6)
            assert owner.source.get_task(task.task_cid).status == "blocked"
            assert runtime.stop().succeeded
            assert not runtime.process.snapshot(runtime.profile).members
            assert (repository / "answer.py").read_bytes() == initial_answer
            assert (repository / "test_answer.py").read_bytes() == initial_check
            execution = database.with_name(f"{database.stem}.execution.duckdb")
            with duckdb.connect(str(execution), read_only=True) as connection:
                attempts = connection.execute("SELECT attempt_id, claim_id, status, body_json FROM database_task_attempts").fetchall()
                assert len(attempts) == 1
                attempt_id, claim_id, status, body = attempts[0]
                assert status == "failed"
                assert admission["attempt_id"] == attempt_id and admission["claim_id"] == claim_id
                assert json.loads(body)["control_binding"]
                rows = connection.execute("SELECT result_json FROM provider_invocations").fetchall()
                assert len(rows) == 1
                receipt = json.loads(rows[0][0])
                assert receipt["schema"] == "database-native-provider-failure@1"
                assert receipt["callback_state"] == "failed_outcome_settled"
                assert receipt["native_exit"]["subreaper_children_absent"] is True
                assert receipt["native_exit"]["automatic_retry_admitted"] is False
                assert receipt["native_exit"]["completion_authority"] is False
                assert connection.execute("SELECT count(*) FROM effect_claims").fetchone()[0] == 0
                assert connection.execute("SELECT count(*) FROM daemon_execution_events WHERE event_type='native_provider_failure_observed'").fetchone()[0] == 1
            coordinator = open_database_coordinator(database.with_name(f"{database.stem}.coordination.duckdb"))
            try:
                assert coordinator.get_task_claim(claim_id).state.value == "released"
            finally:
                coordinator.close()
            logs = list((runtime.state / "run/admitted_database_portal_attempts").rglob("*.log"))
            invocations = [json.loads(line) for path in logs for line in path.read_text().splitlines()
                           if line.startswith('{"') and 'router-implementation-invocation@1' in line]
            assert len(invocations) == 1 and invocations[0]["error_type"] == "TimeoutExpired"
        finally:
            if runtime.process.snapshot(runtime.profile).members:
                assert runtime.stop().succeeded
            runtime.close()


@pytest.mark.parametrize("change", ["phase", "attestation", "source_revision", "released_lease"])
def test_settlement_rechecks_live_typed_admission_before_persisting_failure(tmp_path, monkeypatch, change):
    """A real signed owner claim cannot be replaced by a stale source snapshot."""
    from copy import deepcopy
    from dataclasses import replace
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import DatabaseCoordinationError
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DatabaseImplementationDaemon, DatabaseImplementationAuthorityError,
    )
    prepared = _prepare_implementation_fixture(tmp_path / "task")
    verified = verify_local_benchmark_admission(prepared["admission"])
    with open_existing_native_owner(database=Path(prepared["intent_database"]),
            checkout=Path(prepared["repository"]), state_dir=tmp_path / "owner",
            repository_id=verified["manifest"]["repository_cid"],
            execution_routes={prepared["task_id"]: GROK_CODEX_EXECUTION_MODE}) as owner:
        with DatabaseImplementationDaemon(database_path=owner.database,
                coordination_path=tmp_path / "coordination.duckdb", execution_path=tmp_path / "execution.duckdb",
                authority_mode="quack", task_source_kind="duckdb",
                owner_session_id="session:signed-failure-recheck", process_instance_id=owner.identity.process_birth_id,
                quack_uri=owner.identity.listen_uri, task_source=owner.source, close_task_source=False,
                state_owner_bootstrap_credentials=owner.credentials, strict_task_sharding=True,
                max_task_attempts=1, lease_ms=60_000, require_real_execution=True) as daemon:
            assert daemon._typed_quack_authority_binding is not None
            attempt = daemon.claim_next()
            assert attempt is not None
            daemon._require_provider_admission(attempt)
            if change == "released_lease":
                lease = daemon.coordinator.get_lease(attempt.lease_id)
                daemon.coordinator.release(lease, expected_fencing_token=attempt.fencing_token,
                                           expected_fence_epoch=attempt.fence_epoch)
            else:
                task = daemon.task_source.get(attempt.task_cid)
                body = deepcopy(dict(task.body))
                receipt = body["completion_receipt"]
                if change == "phase":
                    receipt["claim_phase_schema"] = "unadmitted-fixture@1"
                elif change == "attestation":
                    receipt["claim_process_attestation"] = {}
                else:
                    receipt["admitted_from_revision"] += 1
                stale = replace(task, body=body)
                # Only the candidate read is replaced. Neither admission nor
                # claim protection is patched; the live owner remains intact.
                original = daemon.task_source.get
                monkeypatch.setattr(daemon.task_source, "get", lambda cid: stale if cid == attempt.task_cid else original(cid))
            with pytest.raises((DatabaseImplementationAuthorityError, DatabaseCoordinationError)) as error:
                daemon._settle_native_provider_failure(attempt, native_exit={},
                    idempotency_key=f"provider:{attempt.attempt_id}")
            if change != "released_lease":
                assert isinstance(error.value, DatabaseImplementationAuthorityError)
                assert "exact owner-admitted" in str(error.value)
            else:
                assert isinstance(error.value, DatabaseCoordinationError)
            assert daemon.provider_invocation_recorded(attempt.attempt_id,
                idempotency_key=f"provider:{attempt.attempt_id}") is None
            assert daemon.get_attempt(attempt.attempt_id).status == "running"
