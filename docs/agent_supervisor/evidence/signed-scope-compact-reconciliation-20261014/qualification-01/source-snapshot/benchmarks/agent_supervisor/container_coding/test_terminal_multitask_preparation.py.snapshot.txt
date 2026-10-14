"""Reviewed multi-task declarations reach actual admission and native Quack.

Candidate meanings and operations are explicitly authored test inputs. Native
source accounting, symbolic selection, signed admission, storage and replay are
real; these observations confer no execution or semantic proof authority.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.planning import intent_symbolic_planning as symbolic
from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import build_intent_requirement_contract
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import verify_owner_local_benchmark_observation
from ipfs_accelerate_py.agent_supervisor.runtime.terminal_task_profile import instruction_sha256
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import probe_quack_capabilities
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
from ipfs_datasets_py.logic.intent_ir.formalize.requirements import build_intent_requirement_ledger
from ipfs_datasets_py.logic.intent_ir.formalize.roundtrip import frame_to_intent_ir


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _git(repository, *args):
    return subprocess.check_output(["git", "-C", str(repository), *args])


def _declarations(text):
    outputs = [
        {"path": "left.py", "effect": "modify", "media_type": "text/x-python"},
        {"path": "right.py", "effect": "modify", "media_type": "text/x-python"},
        {"path": "report.json", "effect": "create", "media_type": "application/json"},
    ]
    bindings = [
        {"task_key": "TB-LEFT", "operation_id": "operation:left", "output_paths": ["left.py"],
         "dependencies": [], "validation_key": "public-left", "criterion_key": "declared-left"},
        {"task_key": "TB-RIGHT", "operation_id": "operation:right", "output_paths": ["right.py"],
         "dependencies": [], "validation_key": "public-right", "criterion_key": "declared-right"},
        {"task_key": "TB-JOIN", "operation_id": "operation:join", "output_paths": ["report.json"],
         "dependencies": ["TB-LEFT", "TB-RIGHT"], "validation_key": "public-join",
         "criterion_key": "declared-join"},
    ]
    document = frame_to_intent_ir(
        {"actor": "agent", "action": "repair", "object": "left", "modality": "required"},
        instruction=text,
    ).to_dict()
    first = deepcopy(document["statements"][0])
    statements = []
    for label, path, action in (("left", "left.py", "repair"), ("right", "right.py", "repair"),
                                ("join", "report.json", "create")):
        statement = deepcopy(first)
        statement.update(statement_id="required-" + label, predicate=action,
                         arguments=["agent", path], normalized_text=action + " " + path)
        statements.append(statement)
    document["statements"] = statements
    raw = text.encode("utf-8")
    report = {
        "schema": "intent-reviewed-source-report@1", "source_sha256": _sha(raw),
        "source_bytes": len(raw), "source_characters": len(text),
        "producer": {"name": "reviewed-parallel-task-test-input", "revision": "1"},
        "interpretation_status": "reviewed_candidate",
        "units": [{"unit_id": "unit:public-instruction", "start_char": 0, "end_char": len(text),
                   "start_byte": 0, "end_byte": len(raw), "text": text, "sha256": _sha(raw),
                   "disposition": "interpreted_candidate", "reason": "explicit_reviewed_fixture"}],
        "candidates": [{"unit_id": "unit:public-instruction", "candidate_intent_ir": document}],
        "proof_authority": False, "execution_authority": False, "completion_authority": False,
        "source_semantics_verified": False,
    }
    report["report_sha256"] = _sha(_wire(report))
    ledger = build_intent_requirement_ledger(text, source_report=report,
        source_identity={"path": prep.INSTRUCTION, "revision": _sha(raw)})
    by_statement = {item["statement_ids"][0]: item for item in ledger["requirements"]}
    task_to_requirement = {binding["task_key"]: by_statement[statement["statement_id"]]
                           for binding, statement in zip(bindings, statements)}
    task_to_operation = {row["task_key"]: row["operation_id"] for row in bindings}
    requirements, operations = [], []
    for binding, statement, output in zip(bindings, statements, outputs):
        requirement = task_to_requirement[binding["task_key"]]
        requirements.append({"requirement_id": requirement["requirement_id"], "outputs": [output],
            "validation_keys": [binding["validation_key"]], "dependency_requirement_ids": sorted(
                task_to_requirement[key]["requirement_id"] for key in binding["dependencies"])})
        operations.append({"operation_id": binding["operation_id"], "task_key": binding["task_key"],
            "matchers": [{"requirement_id": requirement["requirement_id"],
                "native_document_sha256": requirement["native_document_sha256"],
                "statement_id": statement["statement_id"], "predicate": statement["predicate"],
                "arguments": statement["arguments"], "modality": statement["modality"]}],
            "outputs": [output], "validation_keys": [binding["validation_key"]],
            "dependency_operation_ids": sorted(task_to_operation[key] for key in binding["dependencies"])})
    contract = build_intent_requirement_contract(source_path=prep.INSTRUCTION, ledger=ledger,
        requirements=requirements, source_text=text,
        symbolic_operations={"schema": "intent-symbolic-operation-contract@1",
            "ledger_sha256": ledger["ledger_sha256"], "review_ref": "review:parallel-task-test@1",
            "interpretation_scope": "administrative_requirement_task_coverage", "operations": operations,
            "semantic_alignment_verified": False, "proof_authority": False,
            "execution_authority": False, "completion_authority": False})
    profile = {"schema": "terminal-public-task-profile@3", "instruction_sha256": instruction_sha256(text),
        "input_paths": ["left.py", "right.py"], "outputs": outputs,
        "intent_requirement_contract_cid": cid_for_dag_json(contract), "tasks": bindings}
    return profile, contract


@pytest.fixture
def multitask_case(tmp_path):
    repository = tmp_path / "app"
    repository.mkdir()
    (repository / "left.py").write_text("def left():\n    return 'original left'\n")
    (repository / "right.py").write_text("def right():\n    return 'original right'\n")
    _git(repository, "init", "-q")
    _git(repository, "add", ".")
    _git(repository, "-c", "user.name=Native multi-task qualification", "-c",
         "user.email=native@example.invalid", "commit", "-qm", "Original public sources")
    (repository / "left.py").write_text("def left():\n    return 'retained dirty image bytes'\n")
    text = "Repair left.py and right.py independently. Then create report.json after both repairs.\n"
    instruction = tmp_path / "instruction.md"
    instruction.write_text(text)
    profile, contract = _declarations(text)
    contract_path = tmp_path / "reviewed-requirements.json"
    contract_path.write_bytes(_wire(contract))
    return {"repository": repository, "instruction": instruction, "state": tmp_path / "state",
        "profile": profile, "contract": contract, "contract_path": contract_path,
        "source_bytes": {name: (repository / name).read_bytes() for name in profile["input_paths"]}}


def _prepare(case, **kwargs):
    return prep.prepare(repository=case["repository"], instruction=case["instruction"], state=case["state"],
        task_profile=case["profile"], intent_requirement_contract=case["contract_path"],
        disable_intent_autoencoder=True, **kwargs)


def _assert_no_admission(case):
    assert not (case["state"] / "admission.json").exists()
    database = case["state"] / "intent.duckdb"
    if database.exists():
        with IntentRepository(database) as intent:
            assert intent.list_tasks() == ()


def test_three_reviewed_tasks_prepare_without_changing_original_sources(multitask_case):
    case = multitask_case
    prepared = _prepare(case)
    assert prepared["schema"] == "terminal-indexed-public-preparation@2"
    assert "spec" not in prepared and len(prepared["specs"]) == 3
    assert prepared["manifest"]["payload"]["tasks"] == prepared["specs"]
    assert prepared["request"]["budget"]["max_tasks"] == 3
    assert prepared["query"] == case["instruction"].read_text()
    assert prepared["intent_requirement_contract"] == case["contract"]
    assert prepared["planning_strategy"] == "intent_symbolic"
    assert prepared["provider_calls"] == 0 and prepared["benchmark_success"] is None
    assert not (case["state"] / "provider-request.json").exists()
    assert not (case["repository"] / "report.json").exists()
    assert {name: (case["repository"] / name).read_bytes() for name in case["source_bytes"]} == case["source_bytes"]
    assert b"retained dirty image bytes" in (case["state"] / "original-image.diff").read_bytes()
    assert prep._load_prepared(case["state"]) == prepared


def test_parallel_roots_and_join_replay_into_actual_native_owner(multitask_case, tmp_path):
    case = multitask_case
    prepared = _prepare(case)
    planned = symbolic.build_intent_symbolic_plan(case["contract"], manifest=prepared["manifest"])
    result = prep.plan(case["state"])
    assert result["qualified"] is True, result
    assert result["tasks"] == len(result["task_cids"]) == 3
    assert result["provider_calls"] == planned["receipt"]["provider_calls"] == 0
    assert result["symbolic_planning"] == planned["receipt"]
    assert result["benchmark_success"] is None
    for key in ("source_semantics_verified", "semantic_alignment_verified", "proof_authority",
                "execution_authority", "completion_authority"):
        assert result[key] is False
    schedule = result["symbolic_planning"]["schedule"]
    assert sorted(map(len, schedule["waves"])) == [1, 2]
    assert len(schedule["dependency_edges"]) == 2
    admission = json.loads((case["state"] / "admission.json").read_bytes())
    verified = local.verify_local_benchmark_admission(admission)
    assert verified["receipt"]["completion_authority"] is False
    tasks = {row.task_key: row for row in planned["graph"].tasks}
    assert set(tasks) == {"TB-LEFT", "TB-RIGHT", "TB-JOIN"}
    assert tasks["TB-LEFT"].dependency_task_cids == tasks["TB-RIGHT"].dependency_task_cids == ()
    assert set(tasks["TB-JOIN"].dependency_task_cids) == {tasks["TB-LEFT"].task_cid, tasks["TB-RIGHT"].task_cid}
    with IntentRepository(case["state"] / "intent.duckdb") as intent:
        for task in tasks.values():
            stored = intent.get_task(task.task_cid)
            assert stored["status"] == "ready"
            assert set(stored["dependencies"]) == set(task.dependency_task_cids)
            pending, manifest, _, _ = local._contract(stored["body"], stored["task_cid"])
            assert local.decode_intent_requirement_contract(manifest) == case["contract"]
            assert pending["intent_plan"]["symbolic_planning"] == planned["receipt"]
    capabilities = probe_quack_capabilities()
    assert capabilities.passes_health_check, capabilities.reason_code
    with open_existing_native_owner(database=case["state"] / "intent.duckdb", checkout=case["repository"],
        state_dir=tmp_path / "native-owner", repository_id=prepared["manifest"]["payload"]["repository_cid"],
        execution_routes={key: GROK_CODEX_EXECUTION_MODE for key in tasks}) as owner:
        observed = verify_owner_local_benchmark_observation(server=owner.server, admission=admission)
        runtime = AdmittedBenchmarkRuntime()
        runtime.source, runtime.server, runtime.admission = owner.source, owner.server, admission
        runtime._verify_tasks(observed)
        assert {row.task_cid for row in owner.source.ready_tasks().tasks} == {
            tasks["TB-LEFT"].task_cid, tasks["TB-RIGHT"].task_cid}
        with owner.server._lock:
            rows = owner.server._connection.execute(
                "SELECT task_cid, dependency_task_cid, kind FROM task_dependencies ORDER BY dependency_task_cid"
            ).fetchall()
        assert {tuple(row[index] for index in range(3)) for row in rows} == {
            (tasks["TB-JOIN"].task_cid, tasks["TB-LEFT"].task_cid, "depends_on"),
            (tasks["TB-JOIN"].task_cid, tasks["TB-RIGHT"].task_cid, "depends_on")}
        assert all(owner.source.get_task(task.task_cid).status == "ready" for task in tasks.values())
    assert owner.server.status()["lifecycle"] == "stopped"
    assert not (case["repository"] / "report.json").exists()


def test_real_per_task_checks_do_not_require_later_join_output(multitask_case):
    case = multitask_case
    prepared = _prepare(case)
    specs = {row["task_key"]: row for row in prepared["specs"]}
    for key in ("TB-LEFT", "TB-RIGHT"):
        observed = subprocess.run(specs[key]["validations"][0]["argv"], cwd=case["repository"], capture_output=True)
        assert observed.returncode == 0, observed.stderr.decode()
    join = subprocess.run(specs["TB-JOIN"]["validations"][0]["argv"], cwd=case["repository"], capture_output=True)
    assert join.returncode != 0 and not (case["repository"] / "report.json").exists()
    (case["repository"] / "report.json").write_text('{"status":"authored test output"}\n')
    assert subprocess.run(specs["TB-JOIN"]["validations"][0]["argv"], cwd=case["repository"], capture_output=True).returncode == 0
    _assert_no_admission(case)


@pytest.mark.parametrize("mutation", ["instruction", "contract_cid", "operation", "ordering", "ownership",
                                      "authority", "legacy_contract"])
def test_invalid_reviewed_binding_refused_before_repository_or_state_changes(multitask_case, mutation):
    case = multitask_case
    original_head = _git(case["repository"], "rev-parse", "HEAD")
    if mutation == "instruction":
        case["instruction"].write_text("Perform a different public task.\n")
    elif mutation == "contract_cid":
        case["profile"]["intent_requirement_contract_cid"] = cid_for_dag_json({"foreign": "requirements"})
    elif mutation == "operation":
        case["profile"]["tasks"][0]["operation_id"] = "operation:foreign"
    elif mutation == "ordering":
        case["profile"]["tasks"][2]["dependencies"] = ["TB-LEFT"]
    elif mutation == "ownership":
        case["profile"]["tasks"][1]["output_paths"] = ["left.py"]
    else:
        contract = deepcopy(case["contract"])
        if mutation == "authority":
            contract["symbolic_operations"]["execution_authority"] = True
        else:
            contract["schema"] = "intent-plan-requirement-contract@1"
            contract.pop("symbolic_operations")
        case["contract_path"].write_bytes(_wire(contract))
        case["profile"]["intent_requirement_contract_cid"] = cid_for_dag_json(contract)
    with pytest.raises(ValueError):
        _prepare(case)
    assert not case["state"].exists()
    assert _git(case["repository"], "rev-parse", "HEAD") == original_head
    assert {name: (case["repository"] / name).read_bytes() for name in case["source_bytes"]} == case["source_bytes"]
    assert not (case["repository"] / prep.INSTRUCTION).exists()
    assert not (case["repository"] / prep.SMOKE).exists()


def test_multitask_profile_requires_the_external_reviewed_contract(multitask_case):
    case = multitask_case
    with pytest.raises(ValueError):
        prep.prepare(repository=case["repository"], instruction=case["instruction"], state=case["state"],
            task_profile=case["profile"], disable_intent_autoencoder=True)
    assert not case["state"].exists()


@pytest.mark.parametrize("mutation", ["profile_omitted", "spec_dropped", "spec_extra", "dependencies",
                                      "task_validation", "administrative_flag", "budget", "authority"])
def test_mutated_preparation_cannot_enter_planning_or_native_storage(multitask_case, mutation):
    case = multitask_case
    prepared = _prepare(case)
    if mutation == "profile_omitted":
        prepared.pop("task_profile")
    elif mutation == "spec_dropped":
        prepared["specs"].pop()
    elif mutation == "spec_extra":
        prepared["specs"].append(deepcopy(prepared["specs"][0]))
    elif mutation == "dependencies":
        next(row for row in prepared["specs"] if row["task_key"] == "TB-JOIN")["dependencies"] = []
    elif mutation == "task_validation":
        prepared["specs"][0]["validations"][0]["argv"] = ["python3", "-c", "pass"]
    elif mutation == "administrative_flag":
        prepared["administrative_only"] = False
    elif mutation == "budget":
        prepared["request"]["budget"]["max_tasks"] = 16
    else:
        prepared["intent_requirement_contract"]["symbolic_operations"]["completion_authority"] = True
    (case["state"] / "prepared.json").write_bytes(_wire(prepared))
    with pytest.raises(ValueError):
        prep.plan(case["state"])
    assert not (case["state"] / "planner-invoked.json").exists()
    _assert_no_admission(case)


@pytest.mark.parametrize("path", [prep.INSTRUCTION, prep.SMOKE, ".supervisor-task-profile.json", "left.py"])
def test_changed_signed_source_refused_before_symbolic_selection(multitask_case, path):
    case = multitask_case
    _prepare(case)
    selected = case["repository"] / path
    selected.write_bytes(selected.read_bytes() + b"\n# source changed\n")
    with pytest.raises(ValueError):
        prep.plan(case["state"])
    assert not (case["state"] / "planner-invoked.json").exists()
    _assert_no_admission(case)


def test_revoked_actual_owner_profile_cannot_admit_prepared_tasks(multitask_case):
    from ipfs_accelerate_py.agent_supervisor.control.profile_authority import revoke_local_profile
    case = multitask_case
    prepared = _prepare(case)
    payload = prepared["manifest"]["payload"]
    revoke_local_profile(profile_dir=Path(payload["profile_dir"]), lifecycle_dir=Path(payload["lifecycle_dir"]))
    with pytest.raises(ValueError):
        prep.plan(case["state"])
    assert not (case["state"] / "planner-invoked.json").exists()
    _assert_no_admission(case)


def test_broken_owner_signature_cannot_enter_native_storage(multitask_case):
    case = multitask_case
    prepared = _prepare(case)
    prepared["manifest"]["binding"]["signature"] = "invalid-signature"
    (case["state"] / "prepared.json").write_bytes(_wire(prepared))
    with pytest.raises(ValueError):
        prep.plan(case["state"])
    _assert_no_admission(case)


def _resigned_contract(case, prepared):
    payload = deepcopy(prepared["manifest"]["payload"])
    contract = deepcopy(case["contract"])
    contract["symbolic_operations"]["review_ref"] = "review:substituted-owner-contract@1"
    payload["intent_requirements"]["contract_json"] = json.dumps(contract, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    payload["intent_requirements"]["contract_cid"] = cid_for_dag_json(contract)
    return contract, local._signed(payload, payload)


def test_genuinely_resigned_contract_cannot_substitute_signed_profile_identity(multitask_case):
    case = multitask_case
    prepared = _prepare(case)
    contract, resigned = _resigned_contract(case, prepared)
    profile = local._manifest(prepared["manifest"], initial=True)[1]
    assert local._verify_signature(resigned, profile) == resigned["payload"]
    assert cid_for_dag_json(contract) != case["profile"]["intent_requirement_contract_cid"]
    planned = symbolic.build_intent_symbolic_plan(contract, manifest=resigned)
    with pytest.raises(ValueError):
        local.admit_local_benchmark_plan(graph=planned["graph"], manifest=resigned,
                                        requirement_bindings=planned["requirement_bindings"])
    _assert_no_admission(case)


def test_dropping_intent_contract_from_resigned_manifest_cannot_erase_profile_join(multitask_case):
    case = multitask_case
    prepared = _prepare(case)
    payload = deepcopy(prepared["manifest"]["payload"])
    payload["schema"] = local.CREATE_MANIFEST_SCHEMA
    payload.pop("intent_requirements")
    resigned = local._signed(payload, payload)
    profile = local._manifest(prepared["manifest"], initial=True)[1]
    assert local._verify_signature(resigned, profile) == payload
    with pytest.raises(ValueError):
        local._manifest(resigned, initial=True)
    _assert_no_admission(case)


def test_resigned_contract_substitution_is_refused_by_materialization_and_native_replay(multitask_case, tmp_path):
    case = multitask_case
    prepared = _prepare(case)
    result = prep.plan(case["state"])
    assert result["qualified"] is True, result
    admission = json.loads((case["state"] / "admission.json").read_bytes())
    _, resigned = _resigned_contract(case, prepared)
    forged = deepcopy(admission)
    forged["manifest"] = resigned
    with pytest.raises(ValueError, match="multi-task"):
        local.verify_local_benchmark_admission(forged)
    with IntentRepository(tmp_path / "refused-materialization.duckdb") as intent:
        with pytest.raises(ValueError, match="multi-task"):
            local.materialize_local_benchmark_plan(admission=forged, intent=intent)
        assert intent.list_tasks() == ()
    with open_existing_native_owner(database=case["state"] / "intent.duckdb", checkout=case["repository"],
        state_dir=tmp_path / "native-owner", repository_id=prepared["manifest"]["payload"]["repository_cid"],
        execution_routes={row["task_key"]: GROK_CODEX_EXECUTION_MODE for row in prepared["specs"]}) as owner:
        before = [(row.task_cid, row.status, row.revision) for row in owner.source.list_tasks().tasks]
        with pytest.raises(ValueError):
            verify_owner_local_benchmark_observation(server=owner.server, admission=forged)
        assert [(row.task_cid, row.status, row.revision) for row in owner.source.list_tasks().tasks] == before
    assert owner.server.status()["lifecycle"] == "stopped"


def test_independently_resigned_smoke_cannot_replace_canonical_task_checks(multitask_case, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
    case = multitask_case
    prepared = _prepare(case)
    (case["repository"] / prep.SMOKE).write_text("raise SystemExit(0)\n")
    _git(case["repository"], "add", "--", prep.SMOKE)
    _git(case["repository"], "-c", "user.name=Native independent owner", "-c",
         "user.email=native@example.invalid", "commit", "-qm", "Change the public smoke input")
    profile_dir, lifecycle_dir = tmp_path / "new-owner-profile", tmp_path / "new-owner-lifecycle"
    Supervisor.init_local(repository=case["repository"], consent=True,
                          profile_dir=profile_dir, lifecycle_dir=lifecycle_dir)
    specs = deepcopy(prepared["specs"])
    for spec in specs:
        for criterion in spec["acceptance"]:
            criterion["evidence_cids"] = []
    with pytest.raises(ValueError, match="structural check"):
        local.author_local_benchmark_manifest(repository=case["repository"], profile_dir=profile_dir,
            lifecycle_dir=lifecycle_dir, task_specs=specs,
            planning_roots=prepared["manifest"]["payload"]["planning_roots"],
            intent_requirements=case["contract"])
    _assert_no_admission(case)


@pytest.mark.parametrize("mutation", ["malformed", "duplicate_schema", "legacy_downgrade", "unknown_schema"])
def test_resigned_source_profile_cannot_hide_a_multitask_declaration(multitask_case, tmp_path, mutation):
    from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
    case = multitask_case
    prepared = _prepare(case)
    legacy = {name: "terminal-public-task-profile@1" if name == "schema" else case["profile"][name]
              for name in ("schema", "instruction_sha256", "input_paths", "outputs")}
    if mutation == "malformed":
        raw = b'{"schema":"terminal-public-task-profile@3", malformed}\n'
    elif mutation == "duplicate_schema":
        raw = b'{"schema":"terminal-public-task-profile@3",' + _wire(legacy)[1:] + b"\n"
    elif mutation == "legacy_downgrade":
        raw = _wire(legacy) + b"\n"
    else:
        unknown = deepcopy(case["profile"])
        unknown["schema"] = "terminal-public-task-profile@999"
        raw = _wire(unknown) + b"\n"
    (case["repository"] / ".supervisor-task-profile.json").write_bytes(raw)
    _git(case["repository"], "add", "--", ".supervisor-task-profile.json")
    _git(case["repository"], "-c", "user.name=Native independent owner", "-c",
         "user.email=native@example.invalid", "commit", "-qm", "Change public profile source")
    profile_dir, lifecycle_dir = tmp_path / "new-owner-profile", tmp_path / "new-owner-lifecycle"
    Supervisor.init_local(repository=case["repository"], consent=True,
                          profile_dir=profile_dir, lifecycle_dir=lifecycle_dir)
    specs = deepcopy(prepared["specs"])
    for spec in specs:
        for criterion in spec["acceptance"]:
            criterion["evidence_cids"] = []
    with pytest.raises(ValueError):
        local.author_local_benchmark_manifest(repository=case["repository"], profile_dir=profile_dir,
            lifecycle_dir=lifecycle_dir, task_specs=specs,
            planning_roots=prepared["manifest"]["payload"]["planning_roots"],
            intent_requirements=case["contract"])
    _assert_no_admission(case)


def test_resigned_manifest_cannot_drop_native_planning_inputs(multitask_case):
    case = multitask_case
    prepared = _prepare(case)
    payload = deepcopy(prepared["manifest"]["payload"])
    payload.pop("planning_inputs")
    for spec in payload["tasks"]:
        for criterion in spec["acceptance"]:
            criterion["evidence_cids"] = []
    resigned = local._signed(payload, payload)
    with pytest.raises(ValueError, match="symbolic planning inputs"):
        local._manifest(resigned, initial=True)
    _assert_no_admission(case)


def test_source_change_after_real_graph_compilation_refuses_admission(multitask_case, monkeypatch):
    """Inject a controlled file race after the actual compiler finishes."""
    case = multitask_case
    prepared = _prepare(case)
    planned = symbolic.build_intent_symbolic_plan(case["contract"], manifest=prepared["manifest"])
    compile_actual = local._graph_contract
    compilations = []

    def change_source_after_compilation(*args, **kwargs):
        compiled = compile_actual(*args, **kwargs)
        compilations.append(True)
        (case["repository"] / "left.py").write_text("def left():\n    return 'changed after actual compiler'\n")
        return compiled

    monkeypatch.setattr(local, "_graph_contract", change_source_after_compilation)
    with pytest.raises(ValueError):
        local.admit_local_benchmark_plan(graph=planned["graph"], manifest=prepared["manifest"],
                                        requirement_bindings=planned["requirement_bindings"])
    assert compilations == [True]
    _assert_no_admission(case)


def test_source_change_after_real_task_insert_rolls_back_native_transaction(multitask_case, monkeypatch):
    """Mutate a real source after a real insert; the owner must roll back."""
    case = multitask_case
    prepared = _prepare(case)
    planned = symbolic.build_intent_symbolic_plan(case["contract"], manifest=prepared["manifest"])
    admission = local.admit_local_benchmark_plan(graph=planned["graph"], manifest=prepared["manifest"],
                                               requirement_bindings=planned["requirement_bindings"])
    insert_actual = IntentRepository.upsert_task
    native_insert_counts = []

    def change_source_after_insert(self, *args, **kwargs):
        inserted = insert_actual(self, *args, **kwargs)
        if not native_insert_counts:
            with self._connection(write=False) as connection:
                native_insert_counts.append(connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
            (case["repository"] / "left.py").write_text("def left():\n    return 'changed after native insert'\n")
        return inserted

    monkeypatch.setattr(IntentRepository, "upsert_task", change_source_after_insert)
    with IntentRepository(case["state"] / "intent.duckdb") as intent:
        with pytest.raises(ValueError):
            local.materialize_local_benchmark_plan(admission=admission, intent=intent)
        assert native_insert_counts == [1]
        assert intent.list_tasks() == ()
        with intent._connection(write=False) as connection:
            assert connection.execute("SELECT COUNT(*) FROM task_dependencies").fetchone()[0] == 0
            assert connection.execute("SELECT COUNT(*) FROM plans").fetchone()[0] == 0
            assert connection.execute("SELECT COUNT(*) FROM goals").fetchone()[0] == 0
            assert connection.execute("SELECT COUNT(*) FROM objectives").fetchone()[0] == 0


@pytest.mark.parametrize("operation", ["initial_context", "context"])
def test_administrative_profile_refuses_unqualified_task_context_paths(multitask_case, operation):
    case = multitask_case
    _prepare(case)
    with pytest.raises(ValueError, match="administrative planning only"):
        getattr(prep, operation)(state=case["state"])
    assert not (case["state"] / "initial-context-result.json").exists()
    assert not (case["state"] / "context-result.json").exists()
    assert not (case["state"] / "planner-invoked.json").exists()
    _assert_no_admission(case)


def test_administrative_plan_cannot_consume_a_singleton_initial_context_marker(multitask_case):
    case = multitask_case
    _prepare(case)
    (case["state"] / "initial-context-result.json").write_text("{}\n")
    with pytest.raises(ValueError, match="unqualified initial task context"):
        prep.plan(case["state"])
    assert not (case["state"] / "planner-invoked.json").exists()
    _assert_no_admission(case)


@pytest.mark.parametrize("implement", [False, True])
def test_actual_native_runtime_refuses_multitask_launch_before_allocation(multitask_case, tmp_path, implement):
    case = multitask_case
    prepared = _prepare(case)
    result = prep.plan(case["state"])
    assert result["qualified"] is True, result
    admission = json.loads((case["state"] / "admission.json").read_bytes())
    launch = tmp_path / "unallocated-run"
    with open_existing_native_owner(database=case["state"] / "intent.duckdb", checkout=case["repository"],
        state_dir=tmp_path / "native-owner", repository_id=prepared["manifest"]["payload"]["repository_cid"],
        execution_routes={row["task_key"]: GROK_CODEX_EXECUTION_MODE for row in prepared["specs"]}) as owner:
        before = [(row.task_cid, row.status, row.revision) for row in owner.source.list_tasks().tasks]
        with pytest.raises(ValueError, match="administrative"):
            AdmittedBenchmarkRuntime.create(launch, admission=admission, server=owner.server,
                source=owner.source, implement=implement,
                implementation_command="python3 -c 'print(1)'" if implement else "")
        assert not launch.exists()
        assert [(row.task_cid, row.status, row.revision) for row in owner.source.list_tasks().tasks] == before
        assert all(status == "ready" for _, status, _ in before)
    assert owner.server.status()["lifecycle"] == "stopped"
