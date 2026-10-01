"""Real Quack transport preserves large signed tasks behind bounded CAS patches."""

from contextlib import contextmanager
from dataclasses import replace
import json
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original as original
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.terminal_native_preflight import _authored_graph
from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import run_owner_local_task_validations
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    MAX_COMMAND_BYTES, canonical_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import TransactionError
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository, MAX_BODY_BYTES
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import probe_quack_capabilities
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import QuackClientError
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon


@pytest.fixture
def large_admission(original, request):
    root, instruction, state = original
    inventory = root / "public-inputs"
    inventory.mkdir()
    for index in range(217):
        (inventory / (f"document-{index:03}-" + "x" * 5 + ".txt")).write_text("Independent public input\n")
    subprocess.run(["git", "-C", str(root), "add", "public-inputs"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c",
        "user.email=test@example.invalid", "commit", "-qm", "complete source inventory"], check=True)
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    graph = _authored_graph(prepared)
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=prepared["manifest"])
    database = state / "intent.duckdb"
    with IntentRepository(database) as intent:
        result = local.materialize_local_benchmark_plan(admission=admission, intent=intent)
        task = intent.get_task(result["task_cids"][0])
        assert MAX_COMMAND_BYTES < len(canonical_json_bytes(task["body"])) < MAX_BODY_BYTES
        if getattr(request, "param", False):
            body = dict(task["body"])
            # Harmless owner-authored text exercises the independent native
            # task-size limit. It grants no proof or completion authority.
            body["owner_documentation"] = "x" * (MAX_BODY_BYTES - 4096 - len(canonical_json_bytes(body)))
            intent.upsert_task(task_cid=task["task_cid"], task_alias=task["task_alias"],
                goal_cid=task["goal_cid"], plan_cid=task["plan_cid"],
                objective_id=task["objective_id"], body=body, identity=task["identity"])
            task = intent.get_task(task["task_cid"])
            assert MAX_BODY_BYTES - 8192 < len(canonical_json_bytes(task["body"])) < MAX_BODY_BYTES
    assert probe_quack_capabilities().passes_health_check, "installed real Quack required; never skip transport"
    with open_existing_native_owner(database=database, checkout=root, state_dir=state / "owner",
            repository_id=prepared["manifest"]["payload"]["repository_cid"],
            execution_routes={task["task_alias"]: GROK_CODEX_EXECUTION_MODE}) as owner:
        yield owner, task, prepared


@contextmanager
def _daemon(owner, state):
    daemon = DatabaseImplementationDaemon(
        database_path=owner.database, coordination_path=state / "coordination.duckdb",
        execution_path=state / "execution.duckdb", authority_mode="quack",
        task_source_kind="duckdb", owner_session_id="session:large-native-task",
        process_instance_id=owner.identity.process_birth_id, quack_uri=owner.identity.listen_uri,
        task_source=owner.source, close_task_source=False,
        state_owner_bootstrap_credentials=owner.credentials, strict_task_sharding=True,
        max_task_attempts=1, lease_ms=60_000, require_real_execution=True,
    ).open()
    try:
        yield daemon
    finally:
        daemon.close()


def test_real_large_task_reservation_admission_retains_contract_and_completion_gate(large_admission, monkeypatch):
    owner, original_task, prepared = large_admission
    original_body = original_task["body"]
    observed = []
    native_submit = owner.client.submit_command
    def capture(command, **kwargs):
        if command.parameters.get("operation") == "task.status.cas.receipt":
            observed.append(command.to_dict())
        return native_submit(command, **kwargs)
    monkeypatch.setattr(owner.client, "submit_command", capture)
    from pathlib import Path
    with _daemon(owner, Path(prepared["state"])) as daemon:
        attempt = daemon.claim_next()
        assert attempt is not None
        task = owner.source.get_task(attempt.task_cid)
        assert task.status == "in_progress" and task.revision == original_task["revision"] + 2
        assert task.body[local.CONTRACT_KEY] == original_body[local.CONTRACT_KEY]
        assert {key: value for key, value in task.body.items() if key != "completion_receipt"} == original_body
        assert task.body["completion_receipt"]["operation"] == "database_attempt_admitted"
        history = owner.source.task_revision_diagnostic_window(task.task_cid, current_revision=task.revision)
        revisions = {row["revision"]: row for row in history["revisions"]}
        assert revisions[task.revision - 1]["body"]["completion_receipt"]["operation"] == "database_claim"
        for revision in (task.revision - 1, task.revision):
            assert revisions[revision]["body"][local.CONTRACT_KEY] == original_body[local.CONTRACT_KEY]
        assert observed and all(len(canonical_json_bytes(command)) <= MAX_COMMAND_BYTES for command in observed)
        for command in observed:
            params = command["parameters"]
            assert params["body_patch_schema"] == "task-completion-receipt-patch@1"
            assert params["expected_task_body_cid"]
            assert set(json.loads(params["body_json"])) <= {"completion_receipt"}
            assert len(params["body_json"].encode()) < 16_384
        failed = run_owner_local_task_validations(server=owner.server, task_cid=task.task_cid,
            attempt_id=attempt.attempt_id, expected_revision=task.revision)
        assert failed["passed"] is False
        digest = failed["results"][0]["evidence_digest"]
        claimed = task.body["completion_receipt"]
        completion = {"operation": "database_complete", "evidence_digest": digest,
            **{key: claimed[key] for key in (
                "attempt_id", "claim_id", "lease_id", "owner_session_id", "fencing_token", "fence_epoch")}}
        native_gate = local.local_completion_missing
        missing_observed = []
        def observe_gate(*args, **kwargs):
            missing = native_gate(*args, **kwargs)
            missing_observed.append((args[1], missing))
            return missing
        monkeypatch.setattr(local, "local_completion_missing", observe_gate)
        # Valid claim/control/evidence shape reaches the local pending gate;
        # an actual failed owner check cannot become completion authority.
        with pytest.raises(TransactionError, match="authorization_denied"):
            owner.client.cas_task_status(task_cid=task.task_cid, goal_cid=task.goal_cid,
                expected_task_revision=task.revision, new_status="completed",
                idempotency_key="large-local-completion-denied", body={"completion_receipt": completion},
                expected_control_receipt=claimed, evidence_digests=[digest],
                expected_task_body_cid=local.content_identity(task.body))
        assert missing_observed and all(cid == task.task_cid and missing for cid, missing in missing_observed)
        assert owner.source.get_task(task.task_cid).revision == task.revision
        assert owner.source.get_task(task.task_cid).body[local.CONTRACT_KEY] == original_body[local.CONTRACT_KEY]


def test_real_owner_rejects_stale_identity_revision_and_nonreceipt_patch(large_admission, monkeypatch):
    owner, original_task, _ = large_admission
    task = owner.source.get_task(original_task["task_cid"])
    cid = local.content_identity(task.body)
    cases = [
        (task.revision, local.content_identity({"foreign": True}), {}, "body-cid"),
        (task.revision + 1, cid, {}, "revision"),
        (task.revision, cid, {"title": "different task"}, "non-receipt"),
        (task.revision, cid, {local.CONTRACT_KEY: {}}, "contract-tamper"),
    ]
    for revision, expected_cid, patch, label in cases:
        with pytest.raises((TransactionError, QuackClientError, ValueError)):
            owner.client.cas_task_status(task_cid=task.task_cid, goal_cid=task.goal_cid,
                expected_task_revision=revision, new_status="retrying", body=patch,
                expected_task_body_cid=expected_cid, idempotency_key="large-patch-denied:" + label)
        retained = owner.source.get_task(task.task_cid)
        assert retained.revision == task.revision and retained.body == task.body
    # Bypass the client's input checks to exercise the actual owner's closed
    # command boundary, while still using the real authenticated transport.
    native_submit = owner.client.submit_command
    for label, changes in (
        ("raw-contract", {"body_json": json.dumps({local.CONTRACT_KEY: {}}, separators=(",", ":"))}),
        ("raw-title", {"body_json": '{"title":"replaced"}'}),
        ("raw-schema", {"body_patch_schema": "unrecognized-patch@1"}),
    ):
        def inject(command, _changes=changes, **kwargs):
            altered = replace(command, parameters={**dict(command.parameters), **_changes})
            return native_submit(altered, **kwargs)
        with monkeypatch.context() as patcher:
            patcher.setattr(owner.client, "submit_command", inject)
            with pytest.raises(TransactionError):
                owner.client.cas_task_status(task_cid=task.task_cid, goal_cid=task.goal_cid,
                    expected_task_revision=task.revision, new_status="retrying", body={},
                    expected_task_body_cid=cid, idempotency_key="large-patch-denied:" + label)
        retained = owner.source.get_task(task.task_cid)
        assert retained.revision == task.revision and retained.body == task.body
    owner.source.compare_and_set_status(task.task_cid, task.revision, "retrying")
    retained = owner.source.get_task(task.task_cid)
    assert retained.revision == task.revision + 1 and retained.status == "retrying"
    assert retained.body == task.body


@pytest.mark.parametrize("large_admission", [True], indirect=True)
def test_owner_reconstruction_keeps_existing_native_body_bound(large_admission):
    owner, original_task, _ = large_admission
    task = owner.source.get_task(original_task["task_cid"])
    owner.source.compare_and_set_status(task.task_cid, task.revision, "retrying",
        receipt={"operation": "owner-authorized-retry"})
    task = owner.source.get_task(task.task_cid)
    receipt = {"operation": "owner-authorized-retry", "note": "x" * 8192}
    assert len(canonical_json_bytes({"completion_receipt": receipt})) < MAX_COMMAND_BYTES
    assert len(canonical_json_bytes({**dict(task.body), "completion_receipt": receipt})) > MAX_BODY_BYTES
    with pytest.raises(TransactionError):
        owner.client.cas_task_status(task_cid=task.task_cid, goal_cid=task.goal_cid,
            expected_task_revision=task.revision, new_status="retrying",
            body={"completion_receipt": receipt}, expected_task_body_cid=local.content_identity(task.body),
            idempotency_key="large-body-owner-overflow")
    retained = owner.source.get_task(task.task_cid)
    assert retained.revision == task.revision and retained.body == task.body
