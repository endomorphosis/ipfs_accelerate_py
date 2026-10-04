"""Public requirement observations are source-bound, read-only projections."""
from copy import deepcopy

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_intent_symbolic_planning import _prepare
from test.api.test_intent_symbolic_planning_ordered import _ordered_case
from ipfs_accelerate_py.agent_supervisor.planning.intent_requirement_repair import (
    IntentRequirementRepairError, build_intent_requirement_repair_proposal,
)
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.intent_requirement_observation import (
    observe_local_intent_requirements, observe_owner_intent_requirements,
)


def _admit(case, *, create=False, ordered=False):
    _, manifest, proposed = _ordered_case(case) if ordered else _prepare(case, create=create)
    admission = local.admit_local_benchmark_plan(graph=proposed["graph"], manifest=manifest,
        requirement_bindings=proposed["requirement_bindings"])
    local.materialize_local_benchmark_plan(admission=admission, intent=case["intent"])
    return admission, {task.task_key: task.task_cid for task in proposed["graph"].tasks}


def _observe(case, admission):
    before = case["intent"].event_watermark()
    tasks_before = case["intent"].list_tasks()
    value = observe_local_intent_requirements(admission=admission, intent=case["intent"])
    assert case["intent"].event_watermark() == before
    assert case["intent"].list_tasks() == tasks_before
    assert value["native_event_watermark"] == before
    assert value["official_reward"] is None and value["provider_calls"] == 0
    assert all(value[key] is False for key in (
        "source_semantics_verified", "semantic_alignment_verified", "proof_authority",
        "execution_authority", "completion_authority", "canonical_state_mutated"))
    return value


def _start(case, cid):
    task = case["intent"].get_task(cid)
    case["intent"].cas_task_status(task_cid=cid, expected_revision=task["revision"], new_status="in_progress")


def test_unobserved_intent_nominates_only_validation_and_keeps_source_revision(scenario):
    admission, _ = _admit(scenario, create=True)
    before = deepcopy(admission)
    observed = _observe(scenario, admission)
    assert observed == _observe(scenario, admission)
    row = observed["requirements"][0]
    assert row["measurement_status"] == "unobserved"
    assert row["missing_output_paths"] == ["report.jsonl"]
    proposal = build_intent_requirement_repair_proposal(observed)
    assert proposal["observation_cid"] == observed["observation_cid"]
    nomination = proposal["nominations"][0]
    assert nomination["work_kind"] == "validation"
    assert nomination["write_paths"] == [] and nomination["residual_packet"] is None
    assert proposal["intent_revision_cid"] == observed["intent_revision_cid"]
    assert admission == before


def test_actual_failure_missing_creation_staleness_and_pass_are_distinct(scenario):
    admission, tasks = _admit(scenario, create=True)
    cid = tasks["LOCAL-TASK"]
    initial = _observe(scenario, admission)
    _start(scenario, cid)
    local.run_local_task_validations(intent=scenario["intent"], task_cid=cid, attempt_id="observed:failure")
    failed = _observe(scenario, admission)
    assert failed["requirements"][0]["measurement_status"] == "failed"
    proposal = build_intent_requirement_repair_proposal(failed)
    assert proposal["observation_cid"] == failed["observation_cid"]
    nomination = proposal["nominations"][0]
    assert nomination["work_kind"] == "repair"
    assert nomination["write_paths"] == ["answer.py", "report.jsonl"]
    assert nomination["residual_packet"]["nomination_only"] is True
    (scenario["repository"] / "answer.py").write_text("def answer():\n    return 2\n")
    local.run_local_task_validations(intent=scenario["intent"], task_cid=cid, attempt_id="observed:missing")
    missing = _observe(scenario, admission)
    assert missing["requirements"][0]["measurement_status"] == "missing_outputs"
    assert missing["requirements"][0]["public_checks_passed"] is True
    (scenario["repository"] / "report.jsonl").write_text('{"answer":2}\n')
    stale = _observe(scenario, admission)
    assert stale["requirements"][0]["measurement_status"] == "stale"
    stale_nomination = build_intent_requirement_repair_proposal(stale)["nominations"][0]
    assert stale_nomination["work_kind"] == "validation"
    assert stale_nomination["write_paths"] == []
    local.run_local_task_validations(intent=scenario["intent"], task_cid=cid, attempt_id="observed:pass")
    passed = _observe(scenario, admission)
    assert passed["requirements"][0]["measurement_status"] == "public_checks_passed"
    assert passed["residual_requirement_ids"] == []
    assert build_intent_requirement_repair_proposal(passed)["nominations"] == []
    assert {value["intent_revision_cid"] for value in (initial, failed, missing, stale, passed)} == {
        initial["intent_revision_cid"]}
    assert passed["requirements"][0]["source_semantic_status"] == "unresolved"


def test_repair_includes_exact_dependent_task_without_mutating_original_plan(scenario):
    admission, tasks = _admit(scenario, ordered=True)
    _start(scenario, tasks["LOCAL-TASK"])
    local.run_local_task_validations(intent=scenario["intent"], task_cid=tasks["LOCAL-TASK"],
        attempt_id="ordered:failure")
    observed = _observe(scenario, admission)
    proposal = build_intent_requirement_repair_proposal(observed)
    assert proposal["affected_task_cids"] == sorted(tasks.values())
    nominations = {row["task_key"]: row for row in proposal["nominations"]}
    assert nominations["LOCAL-TASK"]["work_kind"] == "repair"
    assert nominations["LOCAL-TASK"]["write_paths"] == ["answer.py"]
    assert "test_answer.py" not in nominations["LOCAL-TASK"]["write_paths"]
    assert nominations["REPORT-TASK"]["work_kind"] == "validation"
    assert nominations["REPORT-TASK"]["dependency_task_cids"] == [tasks["LOCAL-TASK"]]
    assert proposal["graph_cid"] == observed["graph_cid"]
    assert scenario["intent"].event_watermark() == observed["native_event_watermark"]


@pytest.mark.parametrize("mutation", ["receipt", "native_identity", "native_output", "native_check", "plan_head", "extra_task", "source"])
def test_mutated_authority_or_immutable_source_refuses_observation(scenario, mutation):
    admission, tasks = _admit(scenario)
    if mutation == "receipt":
        admission["receipt"]["payload"]["graph_cid"] = "forged"
    elif mutation == "native_identity":
        with scenario["intent"]._connection(write=True) as connection:
            connection.execute("UPDATE tasks SET identity_json='{}' WHERE task_cid=?", [tasks["LOCAL-TASK"]])
    elif mutation in {"native_output", "native_check"}:
        with scenario["intent"]._connection(write=True) as connection:
            if mutation == "native_output":
                connection.execute("UPDATE task_outputs SET path='test_answer.py' WHERE task_cid=?", [tasks["LOCAL-TASK"]])
            else:
                connection.execute("UPDATE task_validations SET argv_json='[\"arbitrary-check\"]' WHERE task_cid=?", [tasks["LOCAL-TASK"]])
    elif mutation == "plan_head":
        with scenario["intent"]._connection(write=True) as connection:
            connection.execute("UPDATE plans SET status='historical'")
    elif mutation == "extra_task":
        task = scenario["intent"].get_task(tasks["LOCAL-TASK"])
        scenario["intent"].upsert_task(task_cid="extra-task", task_alias="EXTRA", goal_cid=task["goal_cid"])
    else:
        (scenario["repository"] / "test_answer.py").write_text("assert True\n")
    with pytest.raises(ValueError):
        observe_local_intent_requirements(admission=admission, intent=scenario["intent"])


def test_unqualified_owner_and_changed_observation_cannot_nominate(scenario):
    admission, _ = _admit(scenario)
    with pytest.raises(local.LocalPlanningError, match="actual native owner"):
        observe_owner_intent_requirements(server=object(), admission=admission)
    observation = _observe(scenario, admission)
    observation["requirements"][0]["measurement_status"] = "public_checks_passed"
    with pytest.raises(IntentRequirementRepairError):
        build_intent_requirement_repair_proposal(observation)


def test_generic_validation_and_bodies_are_not_exported_as_requirement_evidence(scenario):
    admission, tasks = _admit(scenario)
    _start(scenario, tasks["LOCAL-TASK"])
    scenario["intent"].record_validation_result(task_cid=tasks["LOCAL-TASK"], outcome="passed",
        evidence_digest="generic-success", argv=["portal-supervisor-gates"], attempt_id="generic",
        body={"private_source_dump": "not a signed owner check"})
    observed = _observe(scenario, admission)
    assert observed["requirements"][0]["measurement_status"] == "unobserved"
    import json
    assert "private_source_dump" not in json.dumps(observed)
    assert "candidate_intent_ir" not in json.dumps(observed)
    assert "def answer" not in json.dumps(observed)


def test_newest_stale_observation_does_not_reuse_an_earlier_matching_pass(scenario):
    admission, tasks = _admit(scenario)
    cid = tasks["LOCAL-TASK"]
    _start(scenario, cid)
    answer = scenario["repository"] / "answer.py"
    answer.write_text("def answer():\n    return 2\n")
    first = local.run_local_task_validations(intent=scenario["intent"], task_cid=cid, attempt_id="same-attempt")
    assert first["passed"]
    answer.write_text("def answer():\n    return 3\n")
    newer = local.run_local_task_validations(intent=scenario["intent"], task_cid=cid, attempt_id="same-attempt")
    assert not newer["passed"]
    answer.write_text("def answer():\n    return 2\n")
    observed = _observe(scenario, admission)
    assert observed["current_source_tree_id"] == first["source_tree_id"]
    check = observed["tasks"][0]["validations"][0]
    assert check["evidence_digest"] == newer["results"][0]["evidence_digest"]
    assert check["status"] == "stale"
    assert observed["requirements"][0]["public_checks_passed"] is False
    assert build_intent_requirement_repair_proposal(observed)["nominations"][0]["work_kind"] == "validation"


def test_source_change_during_projection_is_rejected(scenario, monkeypatch):
    admission, _ = _admit(scenario)
    original = local._manifest
    calls = 0
    def changing(*args, **kwargs):
        nonlocal calls
        value = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            (scenario["repository"] / "answer.py").write_text("def answer():\n    return 2\n")
        return value
    monkeypatch.setattr(local, "_manifest", changing)
    before = scenario["intent"].event_watermark()
    with pytest.raises(local.LocalPlanningError, match="changed during observation"):
        observe_local_intent_requirements(admission=admission, intent=scenario["intent"])
    assert scenario["intent"].event_watermark() == before


def test_prohibition_is_preserved_without_inventing_a_satisfaction_measure(scenario):
    from test.api.test_intent_requirement_admission import _author, _reviewed_contract, _sha, _wire
    from ipfs_datasets_py.logic.intent_ir.formalize.requirements import build_intent_requirement_ledger
    contract = _reviewed_contract(scenario)
    report = deepcopy(contract["ledger"]["source_report"])
    statement = deepcopy(report["candidates"][0]["candidate_intent_ir"]["statements"][0])
    statement.update(statement_id="prohibited-check-edit", modality="prohibited",
        arguments=["agent", "test_answer.py"], normalized_text="Do not edit the public check.")
    report["candidates"][0]["candidate_intent_ir"]["statements"].append(statement)
    report["report_sha256"] = _sha(_wire({key: value for key, value in report.items() if key != "report_sha256"}))
    ledger = build_intent_requirement_ledger((scenario["repository"] / "test_answer.py").read_text(),
        source_report=report, source_identity=contract["ledger"]["source"]["source_identity"])
    mandatory = next(row["requirement_id"] for row in ledger["requirements"] if row["modality"] == "required")
    prohibited = next(row["requirement_id"] for row in ledger["requirements"] if row["modality"] == "prohibited")
    contract["ledger"] = ledger
    contract["requirements"][0]["requirement_id"] = mandatory
    contract["requirements"].append({"requirement_id": prohibited,
        "outputs": [{"path": "test_answer.py", "effect": "modify", "media_type": "text/x-python"}],
        "validation_keys": [], "dependency_requirement_ids": []})
    manifest = _author(scenario, contract)
    admission = local.admit_local_benchmark_plan(graph=scenario["graph"], manifest=manifest,
        requirement_bindings=[{"requirement_id": mandatory, "task_keys": ["LOCAL-TASK"],
            "validation_keys": ["public-answer"]}])
    local.materialize_local_benchmark_plan(admission=admission, intent=scenario["intent"])
    observed = _observe(scenario, admission)
    prohibition = next(row for row in observed["requirements"] if row["requirement_id"] == prohibited)
    assert prohibition["measurement_status"] == "not_measured"
    assert prohibition["public_checks_passed"] is False
    assert prohibited in observed["unmeasured_requirement_ids"]
    assert prohibited not in build_intent_requirement_repair_proposal(observed)["residual_requirement_ids"]


@pytest.mark.parametrize("mutation", ["run_status", "run_command", "run_body", "drop_run",
    "drop_event", "event_body", "malformed_event", "duplicate_event", "drop_result",
    "malformed_result", "result_outcome", "result_digest"])
def test_damaged_latest_owner_check_never_resurrects_an_older_pass(scenario, mutation):
    import json
    admission, tasks = _admit(scenario)
    cid = tasks["LOCAL-TASK"]
    _start(scenario, cid)
    root = scenario["repository"]
    (root / "answer.py").write_text("from pathlib import Path\ndef answer():\n"
        "    return 1 if Path('.runtime/fail').exists() else 2\n")
    runtime = root / ".runtime"
    runtime.mkdir()
    passed = local.run_local_task_validations(intent=scenario["intent"], task_cid=cid, attempt_id="same-attempt")
    assert passed["passed"]
    (runtime / "fail").touch()
    failed = local.run_local_task_validations(intent=scenario["intent"], task_cid=cid, attempt_id="same-attempt")
    assert not failed["passed"] and failed["source_tree_id"] == passed["source_tree_id"]
    with scenario["intent"]._connection(write=True) as connection:
        result = connection.execute("SELECT result_id,run_id FROM validation_results WHERE evidence_digest=?",
            [failed["results"][0]["evidence_digest"]]).fetchone()
        result_id, run_id = result[0], result[1]
        if mutation == "run_status":
            connection.execute("UPDATE validation_runs SET status='passed' WHERE run_id=?", [run_id])
        elif mutation == "run_command":
            connection.execute("UPDATE validation_runs SET command_digest='other-command' WHERE run_id=?", [run_id])
        elif mutation == "run_body":
            connection.execute("UPDATE validation_runs SET body_json='{}' WHERE run_id=?", [run_id])
        elif mutation == "drop_run":
            connection.execute("DELETE FROM validation_runs WHERE run_id=?", [run_id])
        elif mutation in {"drop_event", "event_body", "malformed_event", "duplicate_event"}:
            event = connection.execute("SELECT event_id,body_json FROM domain_events "
                "WHERE event_type='intent.validation_recorded' AND json_extract_string(body_json,'$.subject_id')=?",
                [result_id]).fetchone()
            if mutation == "drop_event":
                connection.execute("DELETE FROM domain_events WHERE event_id=?", [event[0]])
            elif mutation == "event_body":
                body = json.loads(event[1])
                body["body"]["outcome"] = "passed"
                connection.execute("UPDATE domain_events SET body_json=? WHERE event_id=?",
                    [json.dumps(body), event[0]])
            elif mutation == "malformed_event":
                connection.execute("UPDATE domain_events SET body_json='{' WHERE event_id=?", [event[0]])
            else:
                connection.execute("INSERT INTO domain_events (event_id,stream_id,sequence,global_sequence,"
                    "event_type,task_cid,attempt_id,session_id,recorded_at,body_json) "
                    "SELECT 'duplicate-check-event',stream_id,"
                    "sequence+100,global_sequence+100,event_type,task_cid,attempt_id,session_id,recorded_at,body_json "
                    "FROM domain_events WHERE event_id=?", [event[0]])
        elif mutation == "drop_result":
            connection.execute("DELETE FROM validation_results WHERE result_id=?", [result_id])
        elif mutation == "malformed_result":
            connection.execute("UPDATE validation_results SET body_json='{' WHERE result_id=?", [result_id])
        elif mutation == "result_outcome":
            connection.execute("UPDATE validation_results SET outcome='passed' WHERE result_id=?", [result_id])
        else:
            connection.execute("UPDATE validation_results SET evidence_digest='another-digest' WHERE result_id=?", [result_id])
    before = scenario["intent"].event_watermark()
    with pytest.raises(local.LocalPlanningError, match="inconsistent native evidence links"):
        observe_local_intent_requirements(admission=admission, intent=scenario["intent"])
    assert scenario["intent"].event_watermark() == before
