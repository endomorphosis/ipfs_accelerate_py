"""Real native custody and public checks drive read-only requirement progress.

No coding provider is dispatched. Publication uses actual Git, merge queue and
Portal evidence; every measured check is an owner-signed subprocess result.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_intent_symbolic_planning import _prepare
from test.api.test_intent_symbolic_planning_ordered import _ordered_case
from test.api.test_local_completion_bridge import complete, native_published_transition
from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.planning.intent_requirement_repair import build_intent_requirement_repair_proposal
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.intent_requirement_observation import observe_owner_intent_requirements
from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import (
    run_owner_local_task_validations, verify_owner_local_benchmark_observation,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import TransactionError
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import probe_quack_capabilities
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
from ipfs_accelerate_py.agent_supervisor.task_sources.task_source import TaskSourceConflictError
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon

AUTHORITY = ("source_semantics_verified", "semantic_alignment_verified", "proof_authority",
             "execution_authority", "completion_authority", "canonical_state_mutated")


def _admit(case, *, ordered=False):
    contract, manifest, proposed = _ordered_case(case) if ordered else _prepare(case, create=True)
    admission = local.admit_local_benchmark_plan(graph=proposed["graph"], manifest=manifest,
        requirement_bindings=proposed["requirement_bindings"])
    local.materialize_local_benchmark_plan(admission=admission, intent=case["intent"])
    tasks = {task.task_key: task for task in proposed["graph"].tasks}
    return {**case, "contract": contract, "intent_manifest": manifest,
            "admission": admission, "proposed": proposed, "tasks": tasks,
            "task_cid": tasks["LOCAL-TASK"].task_cid}


@contextmanager
def _owner(case, tmp_path):
    capabilities = probe_quack_capabilities()
    if not capabilities.passes_health_check:
        pytest.skip(f"installed native Quack unavailable: {capabilities.reason_code}")
    with open_existing_native_owner(
        database=case["intent"].database_path, checkout=case["repository"],
        state_dir=tmp_path / "requirement-native-owner",
        repository_id=case["intent_manifest"]["payload"]["repository_cid"],
        execution_routes={key: GROK_CODEX_EXECUTION_MODE for key in case["tasks"]},
    ) as owner:
        yield owner


@contextmanager
def _claim(owner, tmp_path):
    daemon = DatabaseImplementationDaemon(
        database_path=owner.database, coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb", authority_mode="quack",
        task_source_kind="duckdb", owner_session_id="session:requirement-observation-test",
        process_instance_id=owner.identity.process_birth_id,
        quack_uri=owner.identity.listen_uri, task_source=owner.source,
        close_task_source=False, state_owner_bootstrap_credentials=owner.credentials,
        strict_task_sharding=True, max_task_attempts=2, lease_ms=60_000,
        require_real_execution=True,
    ).open()
    try:
        attempt = daemon.claim_next()
        assert attempt is not None
        yield daemon, attempt
    finally:
        daemon.close()


def _native_snapshot(owner):
    with owner.server._lock:
        reader = IntentRepository(bound_connection=owner.server._connection, install_schema=False)
        return reader.event_watermark(), reader.list_tasks()


def _observe(case, owner, **kwargs):
    before = _native_snapshot(owner)
    observed = observe_owner_intent_requirements(server=owner.server, admission=case["admission"], **kwargs)
    assert _native_snapshot(owner) == before
    assert observed["native_event_watermark"] == before[0]
    assert all(observed[key] is False for key in AUTHORITY)
    assert observed["provider_calls"] == 0 and observed["official_reward"] is None
    return observed


def _run(owner, attempt, **kwargs):
    task = owner.source.get_task(attempt.task_cid)
    return run_owner_local_task_validations(server=owner.server, task_cid=task.task_cid,
        attempt_id=attempt.attempt_id, expected_revision=task.revision, **kwargs)


def _complete_with_status(owner, attempt, digest, status):
    task = owner.source.get_task(attempt.task_cid)
    claimed = task.body["completion_receipt"]
    return owner.source.compare_and_set_status(task.task_cid, task.revision, status,
        receipt={"operation": "database_complete", "evidence_digest": digest,
            **{key: claimed[key] for key in (
                "attempt_id", "claim_id", "lease_id", "owner_session_id", "fencing_token", "fence_epoch")}},
        expected_control_receipt=claimed, evidence_digests=[digest])


def _record_signed_observation(owner, envelope):
    """Use the existing owner connection for CID-valued local result records.

    The public generic Quack validation API accepts only SHA-256 digests. Local
    checks use their native CID envelope and original admitted intent owner.
    """
    payload = envelope["payload"]
    with owner.server._lock:
        reader = IntentRepository(bound_connection=owner.server._connection, install_schema=False,
            owner_id=payload["intent_owner_id"], session_id=owner.identity.process_birth_id)
        return reader.record_validation_result(task_cid=payload["task_cid"], outcome=payload["outcome"],
            evidence_digest=local.content_identity(envelope), argv=payload["validation"]["argv"],
            attempt_id=payload["attempt_id"], body={"local_observed_validation": envelope})


def _completion_diagnostic(owner, task_cid, observed):
    with owner.server._lock:
        row = owner.server._connection.execute(
            "SELECT attempt_id,body_json FROM completion_receipts WHERE task_cid=?",
            [task_cid]).fetchone()
    retained = dict(owner.source.get_task(task_cid).body["completion_receipt"])
    body = json.loads(row[1])
    return json.dumps({"stale_reasons": observed["tasks"][0]["validations"][0]["stale_reasons"],
        "indexed_attempt_id": row[0], "body_attempt_id": body["receipt"].get("attempt_id"),
        "retained_attempt_id": retained.get("attempt_id"),
        "body_operation": body["receipt"].get("operation"), "retained_operation": retained.get("operation")})


def test_real_runtime_attaches_requirement_observation_without_task_mutation(scenario, tmp_path):
    case = _admit(scenario)
    with _owner(case, tmp_path) as owner:
        base = verify_owner_local_benchmark_observation(server=owner.server, admission=case["admission"])
        observed = _observe(case, owner)
        assert observed["requirements"][0]["measurement_status"] == "unobserved"
        assert observed["requirements"][0]["missing_output_paths"] == ["report.jsonl"]
        runtime = AdmittedBenchmarkRuntime.create(tmp_path / "requirement-runtime",
            admission=case["admission"], server=owner.server, source=owner.source)
        try:
            before = _native_snapshot(owner)
            result = runtime.observe()
            assert _native_snapshot(owner) == before
            assert result["schema"] == "isolated-supervisor-observation@1"
            assert result["provider_dispatch_allowed"] is False
            assert result["completion_authority"] is False
            assert result["intent_requirements"] == observed
            proposal = result["intent_requirement_repair"]
            assert proposal["observation_cid"] == observed["observation_cid"]
            assert proposal["intent_revision_cid"] == observed["intent_revision_cid"]
            assert proposal["nominations"][0]["work_kind"] == "validation"
            assert proposal["nominations"][0]["write_paths"] == []
            assert proposal["nomination_only"] is True
            assert verify_owner_local_benchmark_observation(server=owner.server, admission=case["admission"]) == base
        finally:
            runtime.close()


def test_projection_failure_does_not_grant_authority_or_bypass_base_source_verification(scenario, tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import intent_requirement_observation as module

    case = _admit(scenario)
    with _owner(case, tmp_path) as owner:
        runtime = AdmittedBenchmarkRuntime.create(tmp_path / "requirement-runtime",
            admission=case["admission"], server=owner.server, source=owner.source)
        def unavailable(**kwargs):
            raise local.LocalPlanningError("derived requirement measurement unavailable")
        monkeypatch.setattr(module, "observe_owner_intent_requirements", unavailable)
        try:
            before = _native_snapshot(owner)
            result = runtime.observe()
            assert _native_snapshot(owner) == before
            assert result["intent_requirements"]["status"] == "unavailable"
            assert result["intent_requirements"]["reason"] == "LocalPlanningError"
            assert result["intent_requirements"]["official_reward"] is None
            assert all(result["intent_requirements"][key] is False for key in (
                "source_semantics_verified", "execution_authority", "completion_authority", "canonical_state_mutated"))
            assert "intent_requirement_repair" not in result
            (case["repository"] / "test_answer.py").write_text("changed immutable requirements\n")
            with pytest.raises(ValueError):
                runtime.observe()
        finally:
            runtime.close()


def test_repair_projection_failure_retains_verified_observation(scenario, tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.planning import intent_requirement_repair as module

    case = _admit(scenario)
    with _owner(case, tmp_path) as owner:
        runtime = AdmittedBenchmarkRuntime.create(tmp_path / "requirement-runtime",
            admission=case["admission"], server=owner.server, source=owner.source)
        verified = _observe(case, owner)
        def unavailable(value):
            raise ValueError("bounded repair packet unavailable")
        monkeypatch.setattr(module, "build_intent_requirement_repair_proposal", unavailable)
        try:
            before = _native_snapshot(owner)
            result = runtime.observe()
            assert _native_snapshot(owner) == before
            assert result["intent_requirements"] == verified
            assert result["intent_requirement_repair"]["status"] == "unavailable"
            assert all(result["intent_requirement_repair"][key] is False for key in (
                "execution_authority", "completion_authority", "canonical_state_mutated"))
            assert result["intent_requirement_repair"]["official_reward"] is None
        finally:
            runtime.close()


@pytest.mark.parametrize("completion_status", ["completed", "complete", "done"])
def test_actual_claim_failure_missing_output_stale_and_completion_remain_distinct(scenario, tmp_path, completion_status):
    case = _admit(scenario)
    with _owner(case, tmp_path) as owner:
        initial = _observe(case, owner)
        with _claim(owner, tmp_path) as (_daemon, attempt):
            task = owner.source.get_task(attempt.task_cid)
            owner.client.record_task_validation(task_cid=task.task_cid, outcome="passed",
                evidence_digest="sha256:" + hashlib.sha256(b"generic-portal-success").hexdigest(),
                argv=["portal-supervisor-gates"],
                attempt_id=attempt.attempt_id, body={"private_dump": "not a signed declared public check"},
                idempotency_key="generic-intent-observation", command_id="generic-intent-observation")
            unobserved = _observe(case, owner)
            assert unobserved["requirements"][0]["measurement_status"] == "unobserved"
            assert "private_dump" not in json.dumps(unobserved)
            assert _run(owner, attempt)["passed"] is False
            failed = _observe(case, owner)
            assert failed["requirements"][0]["measurement_status"] == "failed"
            repair = build_intent_requirement_repair_proposal(failed)["nominations"][0]
            assert repair["work_kind"] == "repair" and repair["write_paths"] == ["answer.py", "report.jsonl"]
            assert repair["residual_packet"]["nomination_only"] is True
            (case["repository"] / "answer.py").write_text("def answer():\n    return 2\n")
            missing_check = _run(owner, attempt)
            assert missing_check["passed"] is True
            missing = _observe(case, owner)
            assert missing["requirements"][0]["measurement_status"] == "missing_outputs"
            assert missing["requirements"][0]["public_checks_passed"] is True
            assert missing["requirements"][0]["missing_output_paths"] == ["report.jsonl"]
            with pytest.raises((TaskSourceConflictError, TransactionError)):
                complete(owner, attempt, missing_check["results"][0]["evidence_digest"])
            (case["repository"] / "report.jsonl").write_text('{"answer":2}\n')
            stale = _observe(case, owner)
            assert stale["requirements"][0]["measurement_status"] == "stale"
            assert stale["tasks"][0]["validations"][0]["stale_reasons"] == ["source_tree_changed"]
            nomination = build_intent_requirement_repair_proposal(stale)["nominations"][0]
            assert nomination["work_kind"] == "validation" and nomination["write_paths"] == []
            passed_check = _run(owner, attempt)
            passed = _observe(case, owner)
            assert passed["requirements"][0]["measurement_status"] == "public_checks_passed"
            assert passed["residual_requirement_ids"] == []
            _complete_with_status(owner, attempt, passed_check["results"][0]["evidence_digest"], completion_status)
            completed = _observe(case, owner)
            assert completed["tasks"][0]["status"] == completion_status
            assert completed["tasks"][0]["current_completion_receipt_cid"]
            assert completed["requirements"][0]["measurement_status"] == "public_checks_passed", _completion_diagnostic(
                owner, attempt.task_cid, completed)
            (case["repository"] / "answer.py").write_text("def answer():\n    return 3\n")
            after = _observe(case, owner)
            assert after["tasks"][0]["status"] == completion_status
            assert after["requirements"][0]["measurement_status"] == "stale"
            assert after["requirements"][0]["public_checks_passed"] is False
            assert {value["intent_revision_cid"] for value in (
                initial, unobserved, failed, missing, stale, passed, completed, after)} == {initial["intent_revision_cid"]}
            assert len({value["observation_cid"] for value in (initial, failed, missing, stale, passed, completed, after)}) == 7


def test_native_sequence_selects_latest_public_result_when_timestamps_and_source_match(scenario, tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.task_sources import intent_repository

    case = _admit(scenario)
    root = case["repository"]
    (root / "answer.py").write_text(
        "from pathlib import Path\ndef answer():\n    return 1 if Path('.runtime/fail').exists() else 2\n")
    (root / "report.jsonl").write_text('{"answer":2}\n')
    (root / ".runtime").mkdir(exist_ok=True)
    monkeypatch.setattr(intent_repository, "_utc_iso", lambda *_: "2026-10-01T00:00:00Z")
    with _owner(case, tmp_path) as owner, _claim(owner, tmp_path) as (_daemon, attempt):
        snapshots = []
        for fails in (False, True, False):
            flag = root / ".runtime/fail"
            if fails:
                flag.touch()
            elif flag.exists():
                flag.unlink()
            check = _run(owner, attempt)
            assert check["passed"] is (not fails)
            observed = _observe(case, owner)
            assert observed["requirements"][0]["measurement_status"] == ("failed" if fails else "public_checks_passed")
            snapshots.append(observed)
        assert len({row["current_source_tree_id"] for row in snapshots}) == 1
        sequences = [row["tasks"][0]["validations"][0]["event_sequence"] for row in snapshots]
        assert sequences == sorted(set(sequences))
        with owner.server._lock:
            runs = owner.server._connection.execute("SELECT run_id,started_at FROM validation_runs WHERE task_cid=?",
                [attempt.task_cid]).fetchall()
        assert len(runs) == 3 and len({row[0] for row in runs}) == 3
        assert {row[1] for row in runs} == {"2026-10-01T00:00:00Z"}


def test_owner_signed_mismatched_check_bindings_cannot_replace_real_failed_result(scenario, tmp_path):
    case = _admit(scenario)
    with _owner(case, tmp_path) as owner, _claim(owner, tmp_path) as (_daemon, attempt):
        failed = _run(owner, attempt)
        with owner.server._lock:
            body = owner.server._connection.execute(
                "SELECT body_json FROM validation_results WHERE evidence_digest=?",
                [failed["results"][0]["evidence_digest"]]).fetchone()[0]
        original = json.loads(body)["local_observed_validation"]["payload"]
        for field in ("manifest_cid", "contract_cid", "pending_cid", "intent_owner_id", "task_cid", "validation"):
            forged = deepcopy(original)
            forged.update(outcome="passed", exit_code=0)
            if field == "validation":
                forged[field]["argv"] = ["forged-declared-check"]
            else:
                forged[field] = "foreign-observation-binding"
            envelope = local._signed(forged, case["intent_manifest"]["payload"])
            # Keep the real native row/run/event subject and owner. The signed
            # envelope's one changed binding must still fail independent replay.
            with owner.server._lock:
                reader = IntentRepository(bound_connection=owner.server._connection, install_schema=False,
                    owner_id=original["intent_owner_id"], session_id=owner.identity.process_birth_id)
                reader.record_validation_result(task_cid=attempt.task_cid, outcome="passed",
                    evidence_digest=local.content_identity(envelope), argv=original["validation"]["argv"],
                    attempt_id=attempt.attempt_id, body={"local_observed_validation": envelope})
            observed = _observe(case, owner)
            assert observed["requirements"][0]["measurement_status"] == "failed"
            assert observed["tasks"][0]["validations"][0]["evidence_digest"] == failed["results"][0]["evidence_digest"]


@pytest.mark.parametrize("field", ["attempt_id", "task_revision"])
def test_signed_check_for_previous_custody_is_stale_not_current_success(scenario, tmp_path, field):
    case = _admit(scenario)
    (case["repository"] / "answer.py").write_text("def answer():\n    return 2\n")
    (case["repository"] / "report.jsonl").write_text('{"answer":2}\n')
    with _owner(case, tmp_path) as owner, _claim(owner, tmp_path) as (_daemon, attempt):
        passed = _run(owner, attempt)
        with owner.server._lock:
            body = owner.server._connection.execute(
                "SELECT body_json FROM validation_results WHERE evidence_digest=?",
                [passed["results"][0]["evidence_digest"]]).fetchone()[0]
        payload = json.loads(body)["local_observed_validation"]["payload"]
        payload[field] = "previous-attempt" if field == "attempt_id" else payload[field] - 1
        envelope = local._signed(payload, case["intent_manifest"]["payload"])
        _record_signed_observation(owner, envelope)
        observed = _observe(case, owner)
        assert observed["requirements"][0]["measurement_status"] == "stale"
        assert observed["requirements"][0]["public_checks_passed"] is False
        assert observed["tasks"][0]["validations"][0]["stale_reasons"] == [
            "attempt_changed" if field == "attempt_id" else "task_revision_changed"]
        nomination = build_intent_requirement_repair_proposal(observed)["nominations"][0]
        assert nomination["work_kind"] == "validation" and nomination["write_paths"] == []


def test_real_publication_needs_exact_transition_and_owner_check_before_progress(scenario, tmp_path):
    case = _admit(scenario)
    with _owner(case, tmp_path) as owner, _claim(owner, tmp_path) as (_daemon, attempt):
        initial = _observe(case, owner)
        transition = native_published_transition(case, tmp_path, owner, attempt,
            created_outputs={"report.jsonl": '{"answer":2}\n'})
        with pytest.raises(local.LocalPlanningError):
            _observe(case, owner)
        published = _observe(case, owner, source_transition=transition)
        assert published["requirements"][0]["measurement_status"] == "unobserved"
        assert published["requirements"][0]["missing_output_paths"] == []
        assert published["residual_requirement_ids"] == initial["residual_requirement_ids"]
        with pytest.raises((TaskSourceConflictError, TransactionError)):
            complete(owner, attempt, local.content_identity(transition))
        check = _run(owner, attempt, source_transition=transition)
        assert check["passed"] is True
        measured = _observe(case, owner)
        assert measured["requirements"][0]["measurement_status"] == "public_checks_passed"
        complete(owner, attempt, check["results"][0]["evidence_digest"])
        completed = _observe(case, owner)
        assert completed["requirements"][0]["measurement_status"] == "public_checks_passed", _completion_diagnostic(
            owner, attempt.task_cid, completed)
        assert {row["intent_revision_cid"] for row in (initial, published, measured, completed)} == {
            initial["intent_revision_cid"]}
        assert measured["current_source_tree_id"] == completed["current_source_tree_id"] == check["source_tree_id"]
        forged = deepcopy(transition)
        forged["payload"]["attempt_id"] = "foreign-publication-attempt"
        forged = local._signed(forged["payload"], case["intent_manifest"]["payload"])
        with pytest.raises(local.LocalPlanningError):
            _observe(case, owner, source_transition=forged)
        (case["repository"] / "answer.py").write_text("def answer():\n    return 3\n")
        with pytest.raises(local.LocalPlanningError):
            _observe(case, owner)


def test_native_ordered_failure_nominates_exact_dependent_without_fake_facts(scenario, tmp_path):
    case = _admit(scenario, ordered=True)
    with _owner(case, tmp_path) as owner, _claim(owner, tmp_path) as (_daemon, attempt):
        assert attempt.task_cid == case["tasks"]["LOCAL-TASK"].task_cid
        assert _run(owner, attempt)["passed"] is False
        observed = _observe(case, owner)
        nomination = build_intent_requirement_repair_proposal(observed)
        assert nomination["affected_task_cids"] == sorted(task.task_cid for task in case["tasks"].values())
        rows = {row["task_key"]: row for row in nomination["nominations"]}
        assert rows["LOCAL-TASK"]["work_kind"] == "repair"
        assert rows["LOCAL-TASK"]["write_paths"] == ["answer.py"]
        assert rows["REPORT-TASK"]["work_kind"] == "validation"
        assert rows["REPORT-TASK"]["write_paths"] == []
        assert rows["REPORT-TASK"]["dependency_task_cids"] == [attempt.task_cid]
        assert rows["REPORT-TASK"]["depends_on_affected_task"] is True
        assert case["proposed"]["receipt"]["observed_facts_supplied"] == 0


def test_publication_requires_new_custody_check_even_when_source_bytes_match(scenario, tmp_path):
    case = _admit(scenario)
    with _owner(case, tmp_path) as owner, _claim(owner, tmp_path) as (_daemon, attempt):
        (case["repository"] / "answer.py").write_text("def answer():\n    return 2\n")
        (case["repository"] / "report.jsonl").write_text('{"answer":2}\n')
        assert _run(owner, attempt)["passed"] is True
        before = _observe(case, owner)
        assert before["requirements"][0]["measurement_status"] == "public_checks_passed"
        transition = native_published_transition(case, tmp_path, owner, attempt,
            created_outputs={"report.jsonl": '{"answer":2}\n'})
        published = _observe(case, owner, source_transition=transition)
        assert published["current_source_tree_id"] == before["current_source_tree_id"]
        assert published["requirements"][0]["measurement_status"] == "stale"
        assert published["tasks"][0]["validations"][0]["stale_reasons"] == ["publication_binding_unavailable"]
        assert published["intent_revision_cid"] == before["intent_revision_cid"]
        assert _run(owner, attempt, source_transition=transition)["passed"] is True
        after = _observe(case, owner)
        assert after["requirements"][0]["measurement_status"] == "public_checks_passed"
        assert after["current_source_tree_id"] == before["current_source_tree_id"]


@pytest.mark.parametrize("mutation", ["indexed_attempt", "receipt_attempt", "retained_attempt", "receipt_claim"])
def test_corrupted_retained_completion_custody_cannot_preserve_public_success(scenario, tmp_path, mutation):
    case = _admit(scenario)
    with _owner(case, tmp_path) as owner, _claim(owner, tmp_path) as (_daemon, attempt):
        (case["repository"] / "answer.py").write_text("def answer():\n    return 2\n")
        (case["repository"] / "report.jsonl").write_text('{"answer":2}\n')
        passed = _run(owner, attempt)
        assert passed["passed"] is True
        complete(owner, attempt, passed["results"][0]["evidence_digest"])
        original = _observe(case, owner)
        assert original["requirements"][0]["measurement_status"] == "public_checks_passed"
        with owner.server._lock:
            connection = owner.server._connection
            if mutation == "retained_attempt":
                row = connection.execute("SELECT body_json FROM tasks WHERE task_cid=?", [attempt.task_cid]).fetchone()
                body = json.loads(row[0])
                body["completion_receipt"]["attempt_id"] = "foreign-retained-attempt"
                connection.execute("UPDATE tasks SET body_json=? WHERE task_cid=?", [json.dumps(body), attempt.task_cid])
            elif mutation == "indexed_attempt":
                connection.execute("UPDATE completion_receipts SET attempt_id=? WHERE task_cid=?",
                    ["foreign-indexed-attempt", attempt.task_cid])
            else:
                row = connection.execute("SELECT body_json FROM completion_receipts WHERE task_cid=?", [attempt.task_cid]).fetchone()
                body = json.loads(row[0])
                body["receipt"]["attempt_id" if mutation == "receipt_attempt" else "claim_id"] = "foreign-completion-custody"
                connection.execute("UPDATE completion_receipts SET body_json=? WHERE task_cid=?", [json.dumps(body), attempt.task_cid])
        changed = _observe(case, owner)
        assert changed["requirements"][0]["measurement_status"] == "stale"
        assert changed["requirements"][0]["public_checks_passed"] is False
        assert changed["intent_revision_cid"] == original["intent_revision_cid"]
        assert changed["residual_requirement_ids"] == [original["requirements"][0]["requirement_id"]]


@pytest.mark.parametrize("mutation", ["missing_event", "malformed_event", "run_digest"])
def test_runtime_reports_corrupt_check_relation_unavailable_without_bypassing_base_observation(scenario, tmp_path, mutation):
    case = _admit(scenario)
    with _owner(case, tmp_path) as owner:
        runtime = AdmittedBenchmarkRuntime.create(tmp_path / "requirement-runtime",
            admission=case["admission"], server=owner.server, source=owner.source)
        try:
            with _claim(owner, tmp_path) as (_daemon, attempt):
                assert _run(owner, attempt)["passed"] is False
                base = verify_owner_local_benchmark_observation(server=owner.server, admission=case["admission"])
                with owner.server._lock:
                    connection = owner.server._connection
                    if mutation == "missing_event":
                        connection.execute("DELETE FROM domain_events WHERE task_cid=? AND event_type='intent.validation_recorded'",
                            [attempt.task_cid])
                    elif mutation == "malformed_event":
                        connection.execute("UPDATE domain_events SET body_json=? WHERE task_cid=? AND event_type='intent.validation_recorded'",
                            ['{"malformed_event":', attempt.task_cid])
                    else:
                        connection.execute("UPDATE validation_runs SET command_digest=? WHERE task_cid=?",
                            ["foreign-validation-command", attempt.task_cid])
                before = _native_snapshot(owner)
                result = runtime.observe()
                assert _native_snapshot(owner) == before
                assert result["intent_requirements"]["status"] == "unavailable"
                assert result["intent_requirements"]["reason"] == "LocalPlanningError"
                assert all(result["intent_requirements"][key] is False for key in AUTHORITY)
                assert result["intent_requirements"]["provider_calls"] == 0
                assert result["intent_requirements"]["official_reward"] is None
                assert "intent_requirement_repair" not in result
                assert verify_owner_local_benchmark_observation(server=owner.server, admission=case["admission"]) == base
        finally:
            runtime.close()
