"""Reviewed finite integer scheduling operations retain signed admission and native replay.

These authored fixtures supply the interpretation explicitly. They do not claim
that learned inference recovered the meaning of arbitrary source instructions.
"""
from copy import deepcopy
import hashlib
import json
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
from ipfs_accelerate_py.agent_supervisor.planning import intent_symbolic_planning as symbolic
from ipfs_accelerate_py.agent_supervisor.planning.intent_interval_schedule import (
    CONTRACT_SCHEMA, CONSTRAINT_POLICY, FALSE, SCHEMA, validate_reviewed_interval_schedule,
)
from ipfs_accelerate_py.agent_supervisor.planning.intent_requirement_adapter import (
    build_intent_planning_materials,
)
from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import (
    build_intent_requirement_contract, validate_intent_requirement_contract,
)
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def _public_checker(input_value):
    return '''import json
from pathlib import Path


def exact_object(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result
        result[key] = value
    return result


def validate():
    problem = json.loads(Path('input.json').read_text(), object_pairs_hook=exact_object)
    assert problem == ''' + repr(input_value) + '''
    witness = json.loads(Path('output.json').read_text(), object_pairs_hook=exact_object)
    assert type(witness) is dict and set(witness) == {'schema', 'assignments'}
    assert witness['schema'] == 'finite-interval-schedule-witness@1'
    assignments = witness['assignments']
    assert type(assignments) is list and len(assignments) == len(problem['jobs'])
    resources = {row['id']: row for row in problem['resources']}
    spans = {key: [] for key in resources}
    for job, assignment in zip(problem['jobs'], assignments):
        assert type(assignment) is dict and set(assignment) == {'id', 'start', 'end'}
        assert assignment['id'] == job['id']
        start, end = assignment['start'], assignment['end']
        assert type(start) is int and type(end) is int
        assert problem['horizon_start'] <= start < end <= problem['horizon_end']
        assert start >= job['release'] and end <= job['deadline']
        assert end - start == job['duration']
        resource = resources[job['resource']]
        assert any(left <= start and end <= right for left, right in resource['availability'])
        spans[job['resource']].append((start, end, job['demand']))
    for resource_id, intervals in spans.items():
        endpoints = sorted({point for start, end, _ in intervals for point in (start, end)})
        for point in endpoints:
            demand = sum(amount for start, end, amount in intervals if start <= point < end)
            assert demand <= resources[resource_id]['capacity']


if __name__ == '__main__':
    validate()
'''


def _schedule_case(tmp_path, input_value=None):
    """Fresh public fixture plus signed proposal, with no open database owner."""
    from ipfs_datasets_py.logic.intent_ir.formalize.requirements import build_intent_requirement_ledger
    from ipfs_datasets_py.logic.intent_ir.schema import (
        IntentIRDocument, IntentKind, IntentModality, IntentStatement, NodeGrounding,
        ReviewStatus, SourceRef, SourceSpan, StatementKind,
    )
    repository = tmp_path / "repository"
    repository.mkdir(parents=True)
    input_value = deepcopy(input_value) if input_value is not None else {
        "schema": "finite-interval-schedule-input@1", "horizon_start": 0, "horizon_end": 12,
        "resources": [{"id": "worker", "capacity": 2, "availability": [[0, 12]]}],
        "jobs": [{"id": "alpha", "resource": "worker", "duration": 4, "release": 0,
                  "deadline": 8, "demand": 1},
                 {"id": "beta", "resource": "worker", "duration": 3, "release": 1,
                  "deadline": 10, "demand": 2}]}
    input_bytes = _wire(input_value) + b"\n"
    source = ("Create output.json with an integer interval schedule for the jobs in input.json. "
              "Use half-open intervals and satisfy every duration, release, deadline, resource "
              "availability and capacity constraint within the horizon. Preserve the job order.\n")
    instruction_path = "instruction.txt"
    validation_key = "public-finite-schedule"
    (repository / instruction_path).write_text(source)
    (repository / "input.json").write_bytes(input_bytes)
    # An independently authored public validator, not a wrapper around the
    # candidate solver or the shared formal witness checker.
    check = _public_checker(input_value)
    (repository / "public_check.py").write_text(check)
    for args in (("init", "-q"), ("add", "."),
            ("-c", "user.name=Authored schedule qualification", "-c", "user.email=schedule@example.invalid",
             "commit", "-qm", "Authored finite schedule fixture")):
        subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True)
    baseline = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    with (repository / ".git" / "info" / "exclude").open("a") as stream:
        stream.write("\n.runtime/\n")
    profile, lifecycle = tmp_path / "profile", tmp_path / "lifecycle"
    Supervisor.init_local(repository=repository, consent=True, profile_dir=profile, lifecycle_dir=lifecycle)
    raw = source.encode()
    digest = hashlib.sha256(raw).hexdigest()
    source_ref = SourceRef(ref_id="public-instruction", source_uri="fixture:finite-schedule", source_id=digest,
        source_revision=digest, content_sha256=digest, span=SourceSpan(0, len(source)),
        review_status=ReviewStatus.MACHINE_EXTRACTED)
    statement = IntentStatement(statement_id="create-output", kind=StatementKind.GOAL,
        modality=IntentModality.REQUIRED, normalized_text=source.strip(),
        source_ref_ids=(source_ref.ref_id,), predicate="create", arguments=("agent", "output.json"),
        confidence=0, grounding=NodeGrounding.INFERRED, review_status=ReviewStatus.MACHINE_EXTRACTED)
    document = IntentIRDocument(document_id="reviewed-schedule:" + digest, title="Reviewed finite interval schedule",
        intent_kind=IntentKind.DECLARATIVE, sources=(source_ref,), statements=(statement,))
    document.validate()
    report = {"schema": "intent-reviewed-source-report@1", "source_sha256": digest,
        "source_bytes": len(raw), "source_characters": len(source),
        "producer": {"name": "authored-finite-schedule-fixture", "revision": "1"},
        "interpretation_status": "reviewed_candidate",
        "units": [{"unit_id": "unit:schedule", "start_char": 0, "end_char": len(source),
            "start_byte": 0, "end_byte": len(raw), "text": source, "sha256": digest,
            "disposition": "interpreted_candidate", "reason": "explicit_reviewed_fixture"}],
        "candidates": [{"unit_id": "unit:schedule", "candidate_intent_ir": document.to_dict()}],
        "proof_authority": False, "execution_authority": False,
        "completion_authority": False, "source_semantics_verified": False}
    report["report_sha256"] = hashlib.sha256(_wire(report)).hexdigest()
    ledger = build_intent_requirement_ledger(source, source_report=report,
        source_identity={"path": instruction_path, "revision": baseline})
    requirement = ledger["requirements"][0]
    output = {"path": "output.json", "effect": "create", "media_type": "application/json"}
    native = document.to_dict()["statements"][0]
    operation = {"operation_id": "operation:schedule", "task_key": "SCHEDULE-TASK",
        "matchers": [{"requirement_id": requirement["requirement_id"],
            "native_document_sha256": requirement["native_document_sha256"],
            **{key: native[key] for key in ("statement_id", "predicate", "arguments", "modality")}}],
        "outputs": [output], "validation_keys": [validation_key], "dependency_operation_ids": []}
    operations = {"schema": "intent-symbolic-operation-contract@1", "ledger_sha256": ledger["ledger_sha256"],
        "review_ref": "review:authored-finite-schedule@1", "interpretation_scope": "administrative_requirement_task_coverage",
        "operations": [operation], "semantic_alignment_verified": False, "proof_authority": False,
        "execution_authority": False, "completion_authority": False}
    selector = {"schema": SCHEMA, "review_ref": "review:authored-schedule-selector@1",
        "operation_id": operation["operation_id"], "input_path": "input.json", "output_path": "output.json",
        "constraint_policy": CONSTRAINT_POLICY, "validation_key": validation_key, **FALSE}
    contract = build_intent_requirement_contract(source_path=instruction_path, ledger=ledger,
        requirements=[{"requirement_id": requirement["requirement_id"], "outputs": [output],
            "validation_keys": [validation_key], "dependency_requirement_ids": []}],
        source_text=source, symbolic_operations=operations, reviewed_interval_schedule=selector)
    sources = local._sources(repository, [instruction_path, "input.json", "public_check.py"])
    roots = {"request_cid": local.content_identity({"prompt": source}),
        "scan_cid": local.content_identity({"sources": sources}),
        "program_root": local.content_identity({"git_tree": subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD^{tree}"], text=True).strip()})}
    spec = {"task_key": "SCHEDULE-TASK", "scope_paths": [*sorted(sources), "output.json"],
        "dependencies": [], "outputs": [output],
        "validations": [{"validation_key": validation_key,
            "argv": ["python3", "-I", "-B", "public_check.py"], "cwd": ".", "expected_exit_codes": [0],
            "policy_cid": local.content_identity(local.LOCAL_POLICY)}],
        "acceptance": [{"criterion_key": "public-finite-schedule-constraints",
            "criterion": "The public check validates the ordered finite integer schedule against all authored input constraints.",
            "validation_keys": [validation_key], "evidence_cids": []}]}
    manifest = local.author_local_benchmark_manifest(repository=repository, profile_dir=profile,
        lifecycle_dir=lifecycle, task_specs=[spec], planning_roots=roots, intent_requirements=contract)
    proposed = symbolic.build_intent_symbolic_plan(contract, manifest=manifest)
    return dict(repository=repository, manifest=manifest, contract=contract, proposed=proposed,
        profile=profile, lifecycle=lifecycle, baseline_commit=baseline, input_bytes=input_bytes,
        input_value=input_value, public_check_bytes=check.encode(), source=source)


def test_reviewed_schedule_survives_signed_planning_and_public_replay(tmp_path):
    from benchmarks.agent_supervisor.container_coding.terminal_indexed_preparation import _planning_strategy
    case = _schedule_case(tmp_path)
    contract, manifest, proposed = (case[key] for key in ("contract", "manifest", "proposed"))
    assert contract["schema"] == CONTRACT_SCHEMA
    assert _planning_strategy(contract) == "intent_symbolic"
    assert proposed["receipt"]["provider_calls"] == proposed["receipt"]["observed_facts_supplied"] == 0
    materials = build_intent_planning_materials(contract, manifest=manifest)
    assert materials.current_facts == () and materials.source_applicability is None
    admission = local.admit_local_benchmark_plan(graph=proposed["graph"], manifest=manifest,
        requirement_bindings=proposed["requirement_bindings"])
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        materialized = local.materialize_local_benchmark_plan(admission=admission, intent=intent)
        task = intent.get_task(materialized["task_cids"][0])
        pending, _, _, _ = local._contract(task["body"], task["task_cid"])
        assert pending["intent_plan"]["schema"] == "supervisor-local-intent-plan@2"
        assert pending["intent_plan"]["symbolic_planning"] == proposed["receipt"]
        reference = intent.get_plan(materialized["plan_id"])["body"]["local_planning_receipt_ref"]
        assert local.load_local_planning_receipt(reference, manifest=manifest) == admission["receipt"]
        assert task["status"] == "ready"
    assert local._verify_intent_requirement_text(manifest["payload"], case["source"]) == contract
    assert not (case["repository"] / "output.json").exists()


@pytest.mark.parametrize("field,value", [
    ("schema", "reviewed-integer-interval-schedule@2"), ("review_ref", ""),
    ("operation_id", "operation:foreign"), ("validation_key", "other"),
    ("input_path", "output.json"), ("output_path", "../outside.json"),
    ("output_path", ".git/output.json"), ("input_path", "input.jsonl"),
    ("input_path", "nested/../input.json"), ("output_path", "output.JSON"),
    ("constraint_policy", "finite-closed-capacity@1"),
    *((key, True) for key in FALSE), *((key, 0) for key in FALSE),
    ("solver_checked", True), ("optimality_required", True), ("time_unit", "UTC"),
])
def test_malformed_or_authority_granting_selector_is_rejected(tmp_path, field, value):
    case = _schedule_case(tmp_path)
    changed = deepcopy(case["contract"])
    changed["reviewed_interval_schedule"][field] = value
    with pytest.raises(ValueError):
        validate_intent_requirement_contract(changed, source_text=case["source"])


@pytest.mark.parametrize("mutation", ["missing_input", "input_outside_scope", "output_exists",
    "extra_creation", "changed_validation", "extra_validation", "extra_output", "extra_task",
    "task_dependency", "changed_task", "output_outside_scope", "wrong_media_type", "wrong_effect"])
def test_selector_cannot_escape_independent_manifest_scope(tmp_path, mutation):
    case = _schedule_case(tmp_path)
    manifest = deepcopy(case["manifest"])
    payload = manifest["payload"]
    task = payload["tasks"][0]
    if mutation == "missing_input":
        del payload["sources"]["input.json"]
    elif mutation == "input_outside_scope":
        task["scope_paths"].remove("input.json")
    elif mutation == "output_outside_scope":
        task["scope_paths"].remove("output.json")
    elif mutation == "output_exists":
        payload["sources"]["output.json"] = deepcopy(payload["sources"]["input.json"])
    elif mutation == "extra_creation":
        payload["created_outputs"].append("second.json")
    elif mutation == "changed_validation":
        task["validations"][0]["validation_key"] = "other"
    elif mutation == "extra_validation":
        task["validations"].append(deepcopy(task["validations"][0]))
    elif mutation == "extra_output":
        task["outputs"].append({"path": "second.json", "effect": "create", "media_type": "application/json"})
    elif mutation == "extra_task":
        payload["tasks"].append(deepcopy(task))
    elif mutation == "task_dependency":
        task["dependencies"] = ["OTHER-TASK"]
    elif mutation == "changed_task":
        task["task_key"] = "OTHER-TASK"
    elif mutation == "wrong_media_type":
        task["outputs"][0]["media_type"] = "text/plain"
    else:
        task["outputs"][0]["effect"] = "modify"
    with pytest.raises(ValueError):
        build_intent_planning_materials(case["contract"], manifest=manifest)
    with pytest.raises(local.LocalPlanningError):
        local._verify_intent_requirement_text(payload, case["source"])


@pytest.mark.parametrize("mutation", ["missing", "second", "dependency", "extra_output",
    "extra_validation", "wrong_effect", "wrong_media_type"])
def test_selector_requires_one_closed_operation(tmp_path, mutation):
    case = _schedule_case(tmp_path)
    operations = deepcopy(case["contract"]["symbolic_operations"]["operations"])
    operation = operations[0]
    if mutation == "missing":
        operations.clear()
    elif mutation == "second":
        operations.append(deepcopy(operation))
    elif mutation == "dependency":
        operation["dependency_operation_ids"] = ["operation:other"]
    elif mutation == "extra_output":
        operation["outputs"].append({"path": "other.json", "effect": "create", "media_type": "application/json"})
    elif mutation == "extra_validation":
        operation["validation_keys"].append("other")
    elif mutation == "wrong_effect":
        operation["outputs"][0]["effect"] = "modify"
    else:
        operation["outputs"][0]["media_type"] = "text/plain"
    with pytest.raises(ValueError):
        validate_reviewed_interval_schedule(case["contract"]["reviewed_interval_schedule"],
            operations=operations)


def test_schedule_selector_cannot_be_injected_into_other_contract_versions(tmp_path):
    case = _schedule_case(tmp_path)
    for schema in range(1, 5):
        changed = deepcopy(case["contract"])
        changed["schema"] = f"intent-plan-requirement-contract@{schema}"
        with pytest.raises(ValueError):
            validate_intent_requirement_contract(changed)
    for extension in ("source_applicability", "reviewed_data_transform"):
        changed = deepcopy(case["contract"])
        changed[extension] = {"caller_checked": True}
        with pytest.raises(ValueError):
            validate_intent_requirement_contract(changed)
    changed = deepcopy(case["contract"])
    changed["schema"] = "intent-plan-requirement-contract@2"
    changed.pop("reviewed_interval_schedule")
    assert validate_intent_requirement_contract(changed)["schema"] == "intent-plan-requirement-contract@2"
    with pytest.raises(ValueError, match="differs from signed manifest"):
        build_intent_planning_materials(changed, manifest=case["manifest"])
    with pytest.raises(ValueError, match="requires explicit v3"):
        build_intent_planning_materials(case["contract"], manifest=case["manifest"],
            source_applicability_nomination={"caller_checked": True})


def test_builder_requires_explicit_exclusive_symbolic_selector(tmp_path):
    case = _schedule_case(tmp_path)
    contract = case["contract"]
    kwargs = {key: contract[key] for key in ("source_path", "ledger", "requirements")}
    kwargs["reviewed_interval_schedule"] = contract["reviewed_interval_schedule"]
    with pytest.raises(ValueError, match="explicit symbolic operations"):
        build_intent_requirement_contract(**kwargs)
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_intent_requirement_contract(**kwargs, symbolic_operations=contract["symbolic_operations"],
            reviewed_data_transform={"caller_checked": True})


def test_signed_source_drift_refuses_schedule_admission(tmp_path):
    case = _schedule_case(tmp_path)
    changed = deepcopy(case["input_value"])
    changed["resources"][0]["capacity"] = 10
    (case["repository"] / "input.json").write_bytes(_wire(changed))
    with pytest.raises(local.LocalPlanningError):
        local.admit_local_benchmark_plan(graph=case["proposed"]["graph"], manifest=case["manifest"],
            requirement_bindings=case["proposed"]["requirement_bindings"])


@pytest.mark.parametrize("assignments,valid", [
    ([{"id": "alpha", "start": 0, "end": 4}, {"id": "beta", "start": 4, "end": 7}], True),
    ([{"id": "alpha", "start": 4, "end": 8}, {"id": "beta", "start": 1, "end": 4}], True),
    ([{"id": "alpha", "start": 0, "end": 4}, {"id": "beta", "start": 3, "end": 6}], False),
    ([{"id": "alpha", "start": 0, "end": 5}, {"id": "beta", "start": 5, "end": 8}], False),
    ([{"id": "alpha", "start": 4, "end": 8}, {"id": "beta", "start": 0, "end": 3}], False),
    ([{"id": "alpha", "start": 5, "end": 9}, {"id": "beta", "start": 1, "end": 4}], False),
    ([{"id": "alpha", "start": False, "end": 4}, {"id": "beta", "start": 4, "end": 7}], False),
    ([{"id": "beta", "start": 4, "end": 7}, {"id": "alpha", "start": 0, "end": 4}], False),
    ([{"id": "alpha", "start": 0, "end": 4}], False),
])
def test_independent_public_check_validates_schedule_behavior(tmp_path, assignments, valid):
    case = _schedule_case(tmp_path)
    witness = {"schema": "finite-interval-schedule-witness@1", "assignments": assignments}
    (case["repository"] / "output.json").write_bytes(_wire(witness))
    observed = subprocess.run(["python3", "-I", "-B", "public_check.py"],
        cwd=case["repository"], capture_output=True, timeout=10)
    assert (observed.returncode == 0) is valid


def test_independent_public_check_rejects_crossing_resource_unavailability(tmp_path):
    default = _schedule_case(tmp_path / "default")["input_value"]
    default["resources"][0]["availability"] = [[0, 4], [5, 12]]
    case = _schedule_case(tmp_path / "gapped", input_value=default)
    witness = {"schema": "finite-interval-schedule-witness@1", "assignments": [
        {"id": "alpha", "start": 0, "end": 4}, {"id": "beta", "start": 4, "end": 7}]}
    (case["repository"] / "output.json").write_bytes(_wire(witness))
    observed = subprocess.run(["python3", "-I", "-B", "public_check.py"],
        cwd=case["repository"], capture_output=True, timeout=10)
    assert observed.returncode != 0
