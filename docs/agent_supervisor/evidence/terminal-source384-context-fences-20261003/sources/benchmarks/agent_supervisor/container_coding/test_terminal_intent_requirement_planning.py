"""Public source requirements gate actual scanner, planner parser and admission.

The model response and interpretation are explicitly authored fixtures. These
tests exercise the entry point without provider calls or benchmark score claims.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original, _proposal_json  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import build_intent_requirement_contract
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_datasets_py.logic.intent_ir.formalize.requirements import build_intent_requirement_ledger
from ipfs_datasets_py.logic.intent_ir.formalize.roundtrip import frame_to_intent_ir


def _requirements(instruction, *, symbolic=False):
    text = instruction.read_text()
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    raw = text.encode()
    document = frame_to_intent_ir(
        {"actor": "agent", "action": "repair", "object": "bottle", "modality": "required"},
        instruction=text,
    )
    report = {
        "schema": "intent-reviewed-source-report@1", "source_sha256": sha(raw),
        "source_bytes": len(raw), "source_characters": len(text),
        "producer": {"name": "public-instruction-test-fixture", "revision": "1"},
        "interpretation_status": "reviewed_candidate",
        "units": [{"unit_id": "unit:public-instruction", "start_char": 0, "end_char": len(text),
                   "start_byte": 0, "end_byte": len(raw), "text": text, "sha256": sha(raw),
                   "disposition": "interpreted_candidate", "reason": "explicit_reviewed_fixture"}],
        "candidates": [{"unit_id": "unit:public-instruction", "candidate_intent_ir": document.to_dict()}],
        "proof_authority": False, "execution_authority": False, "completion_authority": False,
        "source_semantics_verified": False,
    }
    report["report_sha256"] = sha(json.dumps(report, sort_keys=True, separators=(",", ":"),
                                            ensure_ascii=True, allow_nan=False).encode())
    ledger = build_intent_requirement_ledger(
        text, source_report=report, source_identity={"path": prep.INSTRUCTION, "revision": sha(raw)},
    )
    requirements = [{"requirement_id": item["requirement_id"],
                     "outputs": [{"path": "bottle.py", "effect": "modify", "media_type": "text/x-python"},
                                 {"path": "report.jsonl", "effect": "create", "media_type": "text/plain"}],
                     "validation_keys": ["public-structural-smoke"], "dependency_requirement_ids": []}
                    for item in ledger["requirements"]]
    operation_contract = None
    if symbolic:
        statement = document.to_dict()["statements"][0]
        operation_contract = {"schema": "intent-symbolic-operation-contract@1",
            "ledger_sha256": ledger["ledger_sha256"], "review_ref": "reviewed:public-test-operation@1",
            "interpretation_scope": "administrative_requirement_task_coverage",
            "operations": [{"operation_id": "operation:repair-bottle", "task_key": "TB-CODE-TASK",
                "matchers": [{"requirement_id": ledger["requirements"][0]["requirement_id"],
                    "native_document_sha256": ledger["requirements"][0]["native_document_sha256"],
                    "statement_id": statement["statement_id"], "predicate": statement["predicate"],
                    "arguments": statement["arguments"], "modality": statement["modality"]}],
                "outputs": deepcopy(requirements[0]["outputs"]),
                "validation_keys": requirements[0]["validation_keys"], "dependency_operation_ids": []}],
            "semantic_alignment_verified": False, "proof_authority": False,
            "execution_authority": False, "completion_authority": False}
    return build_intent_requirement_contract(source_path=prep.INSTRUCTION, ledger=ledger,
        requirements=requirements, source_text=text, symbolic_operations=operation_contract)


def _prepare(original, *, symbolic=False):
    root, instruction, state = original
    contract = _requirements(instruction, symbolic=symbolic)
    artifact = instruction.parent / "requirements.json"
    artifact.write_text(json.dumps(contract))
    prepared = prep.prepare(repository=root, instruction=instruction, state=state,
                            intent_requirement_contract=artifact)
    return prepared, contract


def _plan(prepared, contract, monkeypatch, *, change=None):
    original_run = subprocess.run
    def version(argv, *args, **kwargs):
        if argv == ["codex", "--version"]:
            return subprocess.CompletedProcess(argv, 0, "codex-cli 0.158.0\n", "")
        return original_run(argv, *args, **kwargs)
    monkeypatch.setattr(subprocess, "run", version)
    captured = []
    def provider(prompt, **kwargs):
        captured.append(json.loads(prompt))
        envelope = {"schema": "intent-plan-proposal@1", "contract_cid": cid_for_dag_json(contract),
                    "graph_proposal": json.loads(_proposal_json(prepared)),
                    "requirement_bindings": [{"requirement_id": row["requirement_id"],
                                              "task_keys": ["TB-CODE-TASK"],
                                              "validation_keys": row["validation_keys"]}
                                             for row in contract["requirements"]]}
        if change == "omit":
            envelope["requirement_bindings"] = []
        elif change == "stale":
            envelope["contract_cid"] = cid_for_dag_json({"stale": "contract"})
        response = _proposal_json(prepared) if change == "legacy" else json.dumps(envelope)
        return {"text": response, "observation": {"input_tokens": 12, "output_tokens": 8},
                "execution_receipt": None}
    result = prep.plan(Path(prepared["state"]), provider_callable=provider)
    return result, captured


def test_requirements_gate_entry_point_and_survive_native_storage(original, monkeypatch):
    prepared, contract = _prepare(original)
    assert prepared["manifest"]["payload"]["schema"] == local.INTENT_MANIFEST_SCHEMA
    assert prepared["query"] == original[1].read_text()
    preview = json.loads((original[2] / "provider-request.json").read_text())
    assert preview["schema"] == "intent-plan-provider-request@1"
    result, calls = _plan(prepared, contract, monkeypatch)
    assert result["qualified"] is True, result
    assert result["planning_strategy"] == "intent_coverage"
    assert result["provider_calls"] == len(calls) == 1
    assert calls[0]["intent_contract"] == contract
    assert calls[0]["contract_cid"] == cid_for_dag_json(contract)
    assert result["requirement_coverage"]["accepted"] is True
    assert result["requirement_coverage"]["semantic_alignment_verified"] is False
    assert result["benchmark_success"] is None
    admission = json.loads((original[2] / "admission.json").read_text())
    local.verify_local_benchmark_admission(admission)
    with IntentRepository(original[2] / "intent.duckdb") as intent:
        task = intent.get_task(result["task_cids"][0])
        pending, manifest, _, _ = local._contract(task["body"], task["task_cid"])
        assert pending["intent_plan"]["requirement_bindings"] == admission["requirement_bindings"]
        assert local.decode_intent_requirement_contract(manifest) == contract


@pytest.mark.parametrize("change", ["omit", "stale", "legacy"])
def test_bad_provider_bindings_cannot_materialize_native_tasks(original, monkeypatch, change):
    prepared, contract = _prepare(original)
    result, calls = _plan(prepared, contract, monkeypatch, change=change)
    assert result["qualified"] is False
    assert result["provider_calls"] == len(calls) == 1
    assert result["provider_observation"]["input_tokens"] == 12
    assert not (original[2] / "admission.json").exists()
    if change == "omit":
        assert result["requirement_coverage"]["accepted"] is False
        assert result["requirement_coverage"]["uncovered_requirement_ids"]


def test_mismatched_source_contract_refused_before_preparation(original):
    root, instruction, state = original
    contract = _requirements(instruction)
    artifact = instruction.parent / "requirements.json"
    artifact.write_text(json.dumps(contract))
    instruction.write_text("A different public instruction.")
    with pytest.raises(ValueError, match="ledger|source"):
        prep.prepare(repository=root, instruction=instruction, state=state,
                     intent_requirement_contract=artifact)
    assert not state.exists()


def test_changed_prepared_contract_cannot_reach_provider(original):
    prepared, _ = _prepare(original)
    forged = deepcopy(prepared)
    forged["intent_requirement_contract"]["requirements"][0]["outputs"][0]["path"] = "foreign.py"
    (original[2] / "prepared.json").write_text(json.dumps(forged))
    calls = []
    def provider(*args, **kwargs):
        calls.append(args)
        raise AssertionError("stale requirements must be rejected before dispatch")
    with pytest.raises(ValueError, match="preparation differs"):
        prep.plan(original[2], provider_callable=provider)
    assert calls == []


def test_symbolic_entrypoint_selects_and_materializes_without_planner_provider(original, monkeypatch):
    def provider_forbidden(*args, **kwargs):
        raise AssertionError("symbolic operations must not invoke a planning provider")
    monkeypatch.setattr(prep, "generate_prompt_goal_graph", provider_forbidden)
    monkeypatch.setattr(prep, "build_prompt_goal_provider_request", provider_forbidden)
    actual_run = subprocess.run
    def no_codex(argv, *args, **kwargs):
        if argv and argv[0] == "codex":
            raise AssertionError("symbolic selection must not depend on the model CLI")
        return actual_run(argv, *args, **kwargs)
    monkeypatch.setattr(subprocess, "run", no_codex)
    prepared, contract = _prepare(original, symbolic=True)
    assert prepared["planning_strategy"] == "intent_symbolic"
    assert prepared["request"]["planning_policy"]["allow_model"] is False
    assert not (original[2] / "provider-request.json").exists()
    result = prep.plan(original[2], provider_callable=provider_forbidden)
    assert result["qualified"] is True, result
    assert result["planning_strategy"] == "intent_symbolic"
    assert result["provider_calls"] == 0
    assert result["provider_observation"] == {}
    assert result["model_request_sha256"] is None
    assert result["symbolic_planning"]["provider_calls"] == 0
    assert result["symbolic_planning"]["observed_facts_supplied"] == 0
    assert result["symbolic_planning"]["source_semantics_verified"] is False
    assert result["symbolic_planning"]["input_snapshot_cid"]
    from ipfs_accelerate_py.agent_supervisor.planning.intent_symbolic_planning import build_intent_symbolic_plan
    replay = build_intent_symbolic_plan(contract, manifest=prepared["manifest"])
    assert replay["receipt"] == result["symbolic_planning"]
    assert replay["graph"].content_id == result["symbolic_planning"]["graph_cid"]
    assert replay["requirement_bindings"] == json.loads((original[2] / "admission.json").read_text())["requirement_bindings"]
    assert result["requirement_coverage"]["accepted"] is True
    assert result["benchmark_success"] is None
    admission = json.loads((original[2] / "admission.json").read_text())
    local.verify_local_benchmark_admission(admission)
    with IntentRepository(original[2] / "intent.duckdb") as intent:
        task = intent.get_task(result["task_cids"][0])
        pending, manifest, _, _ = local._contract(task["body"], task["task_cid"])
        assert local.decode_intent_requirement_contract(manifest) == contract
        assert pending["intent_plan"]["requirement_bindings"] == admission["requirement_bindings"]
    assert not (original[0] / "report.jsonl").exists()
    assert not (original[2] / "planner-provider-request.json").exists()


def test_symbolic_entrypoint_needs_no_provider_callable(original):
    _prepare(original, symbolic=True)
    result = prep.plan(original[2])
    assert result["qualified"] is True, result
    assert result["provider_calls"] == 0


@pytest.mark.parametrize("drift", ["world", "descriptor"])
def test_symbolic_context_drift_during_selection_prevents_admission(original, monkeypatch, drift):
    from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
    from ipfs_accelerate_py.agent_supervisor.planning import intent_symbolic_planning

    prepared, _ = _prepare(original, symbolic=True)
    root, _, state = original
    prep.initial_context(state=state)
    loaded = initial.load_initial_context(state=state, prepared=prepared, require_empty_owner=True)
    selected_path = root / (loaded["descriptor"]["world"]["artifact"] if drift == "world"
                            else loaded["receipt"]["descriptor"]["artifact"])
    real_build = intent_symbolic_planning.build_intent_symbolic_plan
    calls = []

    def change_after_selection(*args, **kwargs):
        selected = real_build(*args, **kwargs)
        selected_path.write_bytes(selected_path.read_bytes() + b" ")
        return selected

    def admission_forbidden(**kwargs):
        calls.append(kwargs)
        pytest.fail("changed indexed context must be rejected before admission")

    monkeypatch.setattr(intent_symbolic_planning, "build_intent_symbolic_plan", change_after_selection)
    monkeypatch.setattr(local, "admit_local_benchmark_plan", admission_forbidden)
    result = prep.plan(state)
    assert result["qualified"] is False, result
    assert result["failure"]["type"] == "ValueError"
    assert result["provider_calls"] == 0
    assert calls == []
    assert not (state / "admission.json").exists()
    with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
        assert intent.plan_projection()["tasks"] == []
        assert intent.event_watermark() == 0


def test_symbolic_selection_admits_unchanged_initial_context(original):
    _prepare(original, symbolic=True)
    prep.initial_context(state=original[2])
    result = prep.plan(original[2])
    assert result["qualified"] is True, result
    assert result["provider_calls"] == 0
    assert result["initial_indexed_context"]["completion_authority"] is False


@pytest.mark.parametrize("strategy", ["symbolic", "provider"])
def test_valid_replacement_initial_descriptor_cannot_change_selection_during_planning(original, monkeypatch, strategy):
    from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
    from benchmarks.agent_supervisor.container_coding.test_terminal_initial_context import _version
    from ipfs_accelerate_py.agent_supervisor.planning import intent_symbolic_planning

    root, instruction, state = original
    prepared = (_prepare(original, symbolic=True)[0] if strategy == "symbolic"
                else prep.prepare(repository=root, instruction=instruction, state=state))
    prep.initial_context(state=state)
    original_selection = initial.load_initial_context(state=state, prepared=prepared, require_empty_owner=True)
    replacements = []

    def replace_valid_descriptor():
        receipt_path = state / "initial-context-result.json"
        receipt = json.loads(receipt_path.read_bytes())
        descriptor_path = root / receipt["descriptor"]["artifact"]
        descriptor_path.write_bytes(descriptor_path.read_bytes() + b" ")
        receipt["descriptor"]["sha256"] = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
        receipt_path.write_text(json.dumps(receipt))
        # A new internally consistent selection still cannot replace what the
        # planner was given. Ordinary freshness checks alone accept this one.
        current = initial.load_initial_context(state=state, prepared=prepared, require_empty_owner=True)
        assert current["descriptor"] == original_selection["descriptor"]
        assert current["summaries"] == original_selection["summaries"]
        assert current["receipt"]["descriptor"] != original_selection["receipt"]["descriptor"]
        replacements.append(current)

    def forbidden_admission(**kwargs):
        pytest.fail("a newly selected valid context cannot reach admission")
    monkeypatch.setattr(local, "admit_local_benchmark_plan", forbidden_admission)
    if strategy == "symbolic":
        real_build = intent_symbolic_planning.build_intent_symbolic_plan
        def build(*args, **kwargs):
            result = real_build(*args, **kwargs)
            replace_valid_descriptor()
            return result
        monkeypatch.setattr(intent_symbolic_planning, "build_intent_symbolic_plan", build)
        result = prep.plan(state)
    else:
        _version(monkeypatch)
        def provider(*args, **kwargs):
            replace_valid_descriptor()
            return {"text": _proposal_json(prepared), "observation": {}, "execution_receipt": None}
        result = prep.plan(state, provider_callable=provider)
    assert len(replacements) == 1
    assert result["qualified"] is False
    assert result["failure"]["message"] == "selected initial context changed during planning"
    assert not (state / "admission.json").exists()
    with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
        assert intent.plan_projection()["tasks"] == []


def test_symbolic_failure_is_retained_without_provider_fallback_or_native_tasks(original, monkeypatch):
    _prepare(original, symbolic=True)
    from ipfs_accelerate_py.agent_supervisor.planning import intent_symbolic_planning
    def no_selection(*args, **kwargs):
        error = intent_symbolic_planning.IntentSymbolicPlanningError("no symbolic candidate passed the existing planner gates")
        error.symbolic_issues = [{"code": "uncovered_goal", "severity": "error"}]
        raise error
    monkeypatch.setattr(intent_symbolic_planning, "build_intent_symbolic_plan", no_selection)
    def provider_forbidden(*args, **kwargs):
        raise AssertionError("symbolic failure cannot call the provider")
    result = prep.plan(original[2], provider_callable=provider_forbidden)
    assert result["qualified"] is False
    assert result["provider_calls"] == 0
    assert result["failure"]["type"] == "IntentSymbolicPlanningError"
    assert result["symbolic_issues"] == [{"code": "uncovered_goal", "severity": "error"}]
    assert "symbolic_planning" not in result
    assert not (original[2] / "admission.json").exists()
    assert not (original[2] / "intent.duckdb").exists()


def test_symbolic_budget_expiry_prevents_admission_and_materialization(original, monkeypatch):
    _prepare(original, symbolic=True)
    from ipfs_accelerate_py.agent_supervisor.planning import intent_symbolic_planning
    real_build = intent_symbolic_planning.build_intent_symbolic_plan
    monotonic = [0.0]
    def over_budget(*args, **kwargs):
        selected = real_build(*args, **kwargs)
        monotonic[0] = 2.0
        return selected
    monkeypatch.setattr(intent_symbolic_planning, "build_intent_symbolic_plan", over_budget)
    monkeypatch.setattr(prep.time, "monotonic", lambda: monotonic[0])
    result = prep.plan(original[2], timeout_seconds=1)
    assert result["qualified"] is False
    assert result["provider_calls"] == 0
    assert result["failure"]["type"] == "TimeoutError"
    assert "symbolic_planning" not in result
    assert not (original[2] / "admission.json").exists()
    assert not (original[2] / "intent.duckdb").exists()


def test_full_benchmark_strategy_metadata_tracks_versioned_contract_identity(original):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark import _intent_selection, config_for
    v1 = _requirements(original[1])
    v2 = _requirements(original[1], symbolic=True)
    assert _intent_selection(None)["planning_strategy"] == "direct"
    assert _intent_selection(v1)["planning_strategy"] == "intent_coverage"
    selected = _intent_selection(v2)
    assert selected["planning_strategy"] == "intent_symbolic"
    assert selected["intent_requirement_contract_cid"] == cid_for_dag_json(v2)
    for arm in ("full", "no-index"):
        config = config_for(Path("/dataset"), Path("/output"), Path("/archive"), arm,
                            intent_requirement_contract=v2)
        assert _intent_selection(config["agents"][0]["kwargs"]["intent_requirement_contract"]) == selected
