"""Real scalar proposal/operator/Lake checks with explicit numerical test controls."""
from copy import deepcopy
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import scalar_repair_advisor as subject

INSTRUCTION = "the runner must compute result; requires left > 0; ensures result = old(right) * old(left) and returned."
SOURCE = "def derive(capacity: int, threshold: int) -> int:\n    return capacity + threshold\n"


def sha(value):
    return hashlib.sha256(value).hexdigest()


def test_checkpoint_read_may_update_access_time_without_changing_its_identity(tmp_path):
    path = tmp_path / "checkpoint.json"
    raw = b'{"fixture":"inert checkpoint bytes"}'
    path.write_bytes(raw)
    os.utime(path, ns=(1, path.stat().st_mtime_ns))
    pin = subject._checkpoint_pin(str(path), sha(raw))
    assert pin["sha256"] == sha(raw) and pin["bytes"] == len(raw)


class IntentControl:
    """Authored numerical control; real codec/source audit are still exercised."""
    def report(self, instruction, **options):
        from ipfs_datasets_py.logic.intent_ir.formalize import action_contracts as codec
        raw = codec.source_to_target(instruction)
        binding = codec.bind_candidate_source(instruction, raw)
        result = dict(schema=subject.intent_owner.REPORT_SCHEMA, status="source_supported_action_contract",
            source_sha256=sha(instruction.encode()), checkpoint_sha256=options["expected_sha256"],
            checkpoint_path=options["checkpoint_path"], snapshot_path=options["snapshot_path"],
            raw_candidate_ir=raw, native_intent_ir=binding["bound_candidate"]["document"], binding=binding,
            **subject.intent_owner.FALSE)
        result["report_sha256"] = subject._digest(result)
        return result

    prepare_intent_action_inference = report

    def verify_intent_action_inference(self, report, instruction, **options):
        expected = self.report(instruction, **options)
        if report != expected: raise ValueError("control differs")
        return expected


class SecurityControl:
    """Explicit authored predictions, real datasets source qualification."""
    def __init__(self):
        self.calls = []
        self.operator_override = None
        self.after = None

    def __call__(self, *, config, source_rows):
        from ipfs_datasets_py.logic.formalization.autoencoder import source_program_runtime_384 as owner
        from ipfs_datasets_py.logic.software_verification.program import ProgramExpression
        self.calls.append(deepcopy(source_rows))
        def infer(rows, **options):
            predicted = []
            for row in rows:
                tree = ast.parse(row["source_text"])
                operation = next(node for node in ast.walk(tree) if type(node) is ast.BinOp)
                operator = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}[type(operation.op)]
                if self.operator_override is not None:
                    operator = self.operator_override(len(self.calls), operator)
                operands = ("expr:" + operation.left.id, "expr:" + operation.right.id)
                candidate = dict(kind="program_expression", document=ProgramExpression("expr:result", "binary", "integer",
                    operand_ids=operands, evaluation_order=operands, operator=operator, source_ref_ids=("source",)).to_dict())
                predicted.append(dict(id=row["id"], source_sha256=sha(row["source_text"].encode()), candidate_ir=candidate,
                    status="unqualified_candidate"))
            return dict(domain_id="security_ir", rows=predicted)
        runtime = owner.SourceProgramDecoder384(SimpleNamespace(infer=infer,
            describe=lambda: dict(domain_id="security_ir")), checkpoint_sha256=config["checkpoint_sha256"])
        inference = runtime.infer([dict(id="input-" + str(index), source_text=row["source_text"], embedding=[0.] * 384)
            for index, row in enumerate(source_rows)])
        result = subject.security_owner._base("source_candidate_advice")
        result.update(checkpoint_selection=deepcopy(config), inference=inference,
            source_hashes={row["id"]: row["source_sha256"] for row in source_rows},
            input_bindings=[dict(source_id=row["id"], inference_id="input-" + str(index), source_sha256=row["source_sha256"])
                for index, row in enumerate(source_rows)])
        if any(row["source_contract"]["status"] != "qualified" for row in inference["rows"]):
            result["status"] = "fail_open_no_qualified_candidates"
        if self.after is not None: self.after(len(self.calls))
        return result


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    lake = os.environ.get("IR384_TEST_LAKE_EXECUTABLE")
    if not lake: pytest.skip("explicit actual Lake executable required")
    intent_path, security_path = tmp_path / "intent.json", tmp_path / "security.json"
    intent_path.write_text('{"fixture":"Intent numerical control"}')
    security_path.write_text('{"fixture":"Security numerical control"}')
    domains = {name: dict(lower=-1, upper=1) for name in ("capacity", "threshold")}
    values = dict(instruction=INSTRUCTION,
        intent_config=dict(schema=subject.intent_owner.CONFIG_SCHEMA, checkpoint_path=str(intent_path),
            checkpoint_sha256=sha(intent_path.read_bytes()), embedding_snapshot_path=None),
        security_config=dict(schema=subject.security_owner.STATE_CONFIG_SCHEMA, checkpoint_path=str(security_path),
            checkpoint_sha256=sha(security_path.read_bytes()), decoder="structured", embedding_snapshot_path=None,
            finite_state_domains={"program.py": deepcopy(domains)}, lake=None),
        source_rows=[dict(id="program.py", source_text=SOURCE, source_sha256=sha(SOURCE.encode()))],
        effect_config=dict(schema=subject.effects.ACTION_CONFIG_SCHEMA, contracts=[dict(id="repair", source_id="program.py",
            input_domains=domains, action_id="action", input_parameter_mapping={"left": "capacity", "right": "threshold"})],
            lake=dict(executable=lake, timeout_seconds=60)))
    intent, security = IntentControl(), SecurityControl()
    monkeypatch.setattr(subject.intent_owner, "_owner", lambda: intent)
    monkeypatch.setattr(subject.security_owner, "prepare_security_source_program_advice", security)
    return values, security


def test_live_counterexample_checks_each_exact_operator_candidate_and_preserves_source(inputs):
    values, security = inputs
    before = deepcopy(values)
    result = subject.prepare_scalar_repair_advice(**values)
    assert result["status"] == "candidate_evidence", result
    assert result["initial_refutation_live_verified"] and result["input_pins_rechecked"]
    assert values == before and result["original_source_unchanged"] and result["source_writes"] == 0
    assert len(security.calls) == result["security_advisor_calls"] == 3
    assert [call[0]["source_text"] for call in security.calls] == [SOURCE, SOURCE.replace(" + ", " - "), SOURCE.replace(" + ", " * ")]
    assert [row["effect_status"] for row in result["candidates"]] == ["refuted", "satisfied"]
    assert result["satisfied_candidate_ids"] == [result["candidates"][1]["id"]]
    assert all(row["enabled_case_count"] == 3 and row["live_build_verified"] for row in result["candidates"])
    for row in result["candidates"]:
        assert row["operator_application"]["proposal_only"]
        assert row["rows"][0]["code_source_text"] == row["source_text"]
        assert row["native"]["rows"][0]["code_source_sha256"] == row["source_sha256"]
        assert all(row[key] is False for key in subject.FALSE)
    assert all(result[key] is False for key in subject.FALSE)


@pytest.mark.parametrize("disposition", ["satisfied", "no_enabled_cases"])
def test_no_initial_counterexample_does_not_propose_or_infer_candidates(inputs, disposition):
    values, security = inputs
    values["instruction"] = INSTRUCTION.replace("*", "+") if disposition == "satisfied" else INSTRUCTION.replace("left > 0", "left > 100")
    result = subject.prepare_scalar_repair_advice(**values)
    assert result["status"] == "no_starting_counterexample" and result["initial"]["effect_status"] == disposition
    assert result["input_pins_rechecked"] and result["candidates"] == [] and len(security.calls) == 1
    assert not result["initial_refutation_live_verified"] and result["proposal_report"] is None


def test_wrong_fresh_prediction_is_retained_and_never_repaired_to_match_proposal(inputs):
    values, security = inputs
    security.operator_override = lambda call, operator: "+" if call > 1 else operator
    result = subject.prepare_scalar_repair_advice(**values)
    assert result["status"] == "candidate_evidence" and len(security.calls) == 3
    assert not result["satisfied_candidate_ids"]
    for row in result["candidates"]:
        assert row["status"] == "fail_open_candidate" and row["failure_stage"] == "security_inference"
        assert row["security_advice"]["inference"]["rows"][0]["candidate_ir"]["document"]["operator"] == "+"
        assert row["native"] is None and not row["bounded_effects_satisfied"]


def test_operator_only_alternatives_do_not_silently_reverse_intent_operands(inputs):
    values, _ = inputs
    values["instruction"] = INSTRUCTION.replace("*", "-")
    result = subject.prepare_scalar_repair_advice(**values)
    assert result["status"] == "candidate_evidence"
    assert [row["effect_status"] for row in result["candidates"]] == ["refuted", "refuted"]
    assert result["satisfied_candidate_ids"] == []


def test_explicit_candidate_budget_retains_unchecked_alternative(inputs):
    values, security = inputs
    result = subject.prepare_scalar_repair_advice(**values, maximum_candidates=1)
    assert result["status"] == "candidate_evidence" and len(security.calls) == 2
    assert result["candidates"][1]["status"] == "not_checked_candidate_budget"
    assert result["candidates"][1]["effect_status"] is None and not result["satisfied_candidate_ids"]


@pytest.mark.parametrize("change", ["no_lake", "v1", "more_sources", "source_hash", "source_path", "different_source", "mapping", "checkpoint_hash"])
def test_invalid_or_ambiguous_declarations_do_not_infer_a_repair(inputs, change):
    values, security = inputs
    if change == "no_lake": values["effect_config"]["lake"] = None
    elif change == "v1": values["effect_config"]["schema"] = subject.effects.CONFIG_SCHEMA
    elif change == "more_sources": values["source_rows"].append(deepcopy(values["source_rows"][0]))
    elif change == "source_hash": values["source_rows"][0]["source_sha256"] = "0" * 64
    elif change == "source_path": values["source_rows"][0]["id"] = "../program.py"
    elif change == "different_source": values["effect_config"]["contracts"][0]["source_id"] = "different.py"
    elif change == "mapping": values["effect_config"]["contracts"][0]["input_parameter_mapping"] = {"left": "capacity"}
    else: values["intent_config"]["checkpoint_sha256"] = "0" * 64
    result = subject.prepare_scalar_repair_advice(**values)
    assert result["status"] == "fail_open_unavailable" and not result["candidates"] and security.calls == []


@pytest.mark.parametrize("change", ["intent_checkpoint", "security_checkpoint", "caller_source", "caller_domains"])
def test_changed_checkpoint_or_original_inputs_invalidates_all_candidate_nominations(inputs, change):
    values, security = inputs
    def mutate(call):
        if call != 3: return
        if change.endswith("checkpoint"):
            key = "intent_config" if change.startswith("intent") else "security_config"
            path = Path(values[key]["checkpoint_path"]); path.write_bytes(path.read_bytes() + b"\n")
        elif change == "caller_source": values["source_rows"][0]["source_text"] += "\n"
        else: values["effect_config"]["contracts"][0]["input_domains"]["capacity"]["upper"] = 0
    security.after = mutate
    result = subject.prepare_scalar_repair_advice(**values)
    assert result["status"] == "fail_open_unavailable" and result["failure_stage"] == "input_pin_recheck"
    assert len(result["candidates"]) == 2 and not result["input_pins_rechecked"] and not result["satisfied_candidate_ids"]


def test_saved_starting_receipt_cannot_replace_fresh_live_handle(inputs, monkeypatch):
    values, security = inputs
    builder, gate, proposer, operators = subject._owners()
    original = gate.build_intent_code_effects_lake
    monkeypatch.setattr(gate, "build_intent_code_effects_lake", lambda *args, **kwargs: original(*args, **kwargs).to_dict())
    result = subject.prepare_scalar_repair_advice(**values)
    assert result["status"] == "fail_open_unavailable" and result["failure_stage"] == "initial_live_counterexample"
    assert len(security.calls) == 1 and not result["initial_refutation_live_verified"] and not result["candidates"]


def test_changed_program_world_result_never_reaches_fresh_candidate_inference(inputs, monkeypatch):
    values, security = inputs
    *_, operators = subject._owners()
    original = operators.apply_program_world_repair_operator
    def wrong(**options):
        value = original(**options)
        return SimpleNamespace(to_dict=value.to_dict, accepted_sketch=True, proposal_only=True,
            after_source=SOURCE, before_hash=value.before_hash)
    monkeypatch.setattr(operators, "apply_program_world_repair_operator", wrong)
    result = subject.prepare_scalar_repair_advice(**values)
    assert result["status"] == "candidate_evidence" and len(security.calls) == 1
    assert all(row["status"] == "fail_open_candidate" and row["failure_stage"] == "program_world_operator"
        for row in result["candidates"])


def test_forged_saved_intent_prediction_cannot_start_repair(inputs):
    values, security = inputs
    advice = subject.intent_owner.prepare_intent_384_advice(instruction=INSTRUCTION, config=values["intent_config"])
    advice["candidate_intent_ir"]["invented_effect"] = True
    advice.pop("advice_sha256"); subject.intent_owner._finish(advice)
    result = subject.prepare_scalar_repair_advice(**values, intent_advice=advice)
    assert result["status"] == "fail_open_unavailable" and result["failure_stage"] == "intent_inference_replay"
    assert security.calls == [] and not result["candidates"]
