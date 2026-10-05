"""Intent384 transport keeps raw learned fields and replays provenance binding."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import intent_384_advisor as subject
from ipfs_accelerate_py.agent_supervisor.runtime import intent_code_effect_advisor as effects

INSTRUCTION = "Compute the result under the declared return contract."
PIN = "a" * 64


def config():
    return dict(schema=subject.CONFIG_SCHEMA, checkpoint_path="/models/intent-action.json",
        checkpoint_sha256=PIN, embedding_snapshot_path="/models/gte-small")


def finish(report):
    report.pop("report_sha256", None)
    report["report_sha256"] = subject._sha(subject._wire(report))
    return report


def resign(advice):
    advice.pop("advice_sha256", None)
    return subject._finish(advice)


class Owner:
    """Deterministic transport fake; no claim of learned or formal semantics."""
    def __init__(self, *, status="source_supported_action_contract", change=None):
        self.status, self.change, self.calls = status, change, []

    def report(self, instruction, **options):
        raw = {"kind": "document", "document": {"authored_transport_fixture": "raw predicted slots",
            "sources": [{"content_sha256": "unbound"}]}}
        native = deepcopy(raw["document"])
        native["sources"][0]["content_sha256"] = hashlib.sha256(instruction.encode()).hexdigest()
        return finish(dict(schema=subject.REPORT_SCHEMA, status=self.status,
            source_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
            checkpoint_sha256=options["expected_sha256"], checkpoint_path=options["checkpoint_path"],
            snapshot_path=options["snapshot_path"], raw_candidate_ir=raw,
            native_intent_ir=native if self.status == "source_supported_action_contract" else None,
            binding={"bound_candidate": {"kind": "document", "document": native}},
            **subject.FALSE))

    def prepare_intent_action_inference(self, instruction, **options):
        self.calls.append(("prepare", instruction, deepcopy(options)))
        report = self.report(instruction, **options)
        if self.change: self.change(report)
        return report

    def verify_intent_action_inference(self, report, instruction, **options):
        self.calls.append(("verify", instruction, deepcopy(options)))
        expected = self.report(instruction, **options)
        if subject._wire(report) != subject._wire(expected):
            raise ValueError("does not match numerical checkpoint replay")
        return deepcopy(expected)


def owner(monkeypatch, **options):
    selected = Owner(**options)
    monkeypatch.setattr(subject, "_owner", lambda: selected)
    return selected


def test_no_selection_does_not_import_or_infer_a_model(monkeypatch):
    monkeypatch.setattr(subject, "_owner", lambda: pytest.fail("disabled model imported"))
    report = subject.prepare_intent_384_advice(instruction=INSTRUCTION)
    assert report["status"] == "disabled" and report["config"] is None
    assert subject.validate_intent_384_advice(report, instruction=INSTRUCTION) == report


def test_selected_shared_checkpoint_is_replayed_and_both_candidates_remain_separate(monkeypatch):
    native = owner(monkeypatch); selected = config(); before = deepcopy(selected)
    report = subject.prepare_intent_384_advice(instruction=INSTRUCTION, config=selected)
    assert report["status"] == "semantic_candidate_advice" and report["numerical_replay_verified"]
    assert selected == before and [call[0] for call in native.calls] == ["prepare", "verify"]
    assert native.calls[0][2] == dict(checkpoint_path=selected["checkpoint_path"],
        expected_sha256=PIN, snapshot_path=selected["embedding_snapshot_path"])
    assert report["raw_candidate_ir"] == report["report"]["raw_candidate_ir"]
    assert report["candidate_intent_ir"] == report["report"]["native_intent_ir"]
    assert report["raw_candidate_ir"]["document"]["sources"][0]["content_sha256"] == "unbound"
    assert report["candidate_intent_ir"]["sources"][0]["content_sha256"] == hashlib.sha256(INSTRUCTION.encode()).hexdigest()
    assert all(report[key] is False for key in subject.FALSE)
    assert subject.validate_intent_384_advice(report, instruction=INSTRUCTION) == report
    assert [call[0] for call in native.calls] == ["prepare", "verify", "verify"]


def test_unsupported_native_candidate_is_preserved_without_becoming_an_effect(monkeypatch):
    owner(monkeypatch, status="fail_open_source_mismatch")
    report = subject.prepare_intent_384_advice(instruction=INSTRUCTION, config=config())
    assert report["status"] == "fail_open_source_mismatch" and report["raw_candidate_ir"] is not None
    assert report["candidate_intent_ir"] is None and not report["numerical_replay_verified"]
    assert subject.validate_intent_384_advice(report, instruction=INSTRUCTION) == report
    with pytest.raises(ValueError, match="source-supported"):
        effects._intent(INSTRUCTION, report)


@pytest.mark.parametrize("change", ["schema", "extra", "hash", "relative_checkpoint", "relative_snapshot", "missing"])
def test_unknown_or_incomplete_configuration_fails_open_without_owner_import(monkeypatch, change):
    native = owner(monkeypatch); selected = config()
    if change == "schema": selected["schema"] = "published-old-atom"
    elif change == "extra": selected["fallback_to_parser"] = True
    elif change == "hash": selected["checkpoint_sha256"] = "main"
    elif change == "relative_checkpoint": selected["checkpoint_path"] = "checkpoint.json"
    elif change == "relative_snapshot": selected["embedding_snapshot_path"] = "gte"
    else: selected.pop("checkpoint_sha256")
    report = subject.prepare_intent_384_advice(instruction=INSTRUCTION, config=selected)
    assert report["status"] == "fail_open_unavailable" and report["failure_stage"] == "configuration"
    assert native.calls == [] and report["continue_planning"]
    assert subject.validate_intent_384_advice(report, instruction=INSTRUCTION) == report


@pytest.mark.parametrize("change", ["raw_candidate", "native_candidate", "source", "checkpoint", "authority", "extra"])
def test_changed_shared_report_is_rejected_by_numerical_replay(monkeypatch, change):
    def mutate(report):
        if change == "raw_candidate": report["raw_candidate_ir"]["document"]["authored_transport_fixture"] = "repaired"
        elif change == "native_candidate": report["native_intent_ir"]["invented_effect"] = True
        elif change == "source": report["source_sha256"] = "0" * 64
        elif change == "checkpoint": report["checkpoint_sha256"] = "0" * 64
        elif change == "authority": report["proof_authority"] = True
        else: report["trust_me"] = True
        finish(report)
    owner(monkeypatch, change=mutate)
    report = subject.prepare_intent_384_advice(instruction=INSTRUCTION, config=config())
    assert report["status"] == "fail_open_unavailable"
    assert report["failure_stage"] == "datasets_numerical_replay" and report["report"] is None


@pytest.mark.parametrize("change", ["source", "checkpoint", "path", "snapshot", "authority", "digest", "missing_native", "raw_kind"])
def test_owner_transport_cannot_silently_drop_identity_or_authority_fields(monkeypatch, change):
    def mutate(report):
        if change == "source": report["source_sha256"] = "0" * 64
        elif change == "checkpoint": report["checkpoint_sha256"] = "0" * 64
        elif change == "path": report["checkpoint_path"] = "/models/different.json"
        elif change == "snapshot": report["snapshot_path"] = "/models/other-embedding"
        elif change == "authority": report["execution_authority"] = True
        elif change == "missing_native": report["native_intent_ir"] = None
        elif change == "raw_kind": report["raw_candidate_ir"]["kind"] = "intent_rich_ast"
        else: report["report_sha256"] = "0" * 64
        if change != "digest": finish(report)
    shared = owner(monkeypatch, change=mutate)
    # Exercise the independent consumer boundary even if an owner incorrectly
    # echoes its own malformed transport report as the verification result.
    shared.verify_intent_action_inference = lambda report, *args, **kwargs: deepcopy(report)
    result = subject.prepare_intent_384_advice(instruction=INSTRUCTION, config=config())
    assert result["status"] == "fail_open_unavailable" and result["failure_stage"] == "datasets_result"


def _owner_with_contract(monkeypatch):
    shared = owner(monkeypatch)
    original = shared.report
    def authored_report(*args, **kwargs):
        report = original(*args, **kwargs)
        report["binding"]["source_audit"] = {"candidate_contract": {"equation": {"operator": "add"}}}
        return finish(report)
    shared.report = authored_report
    return shared


@pytest.mark.parametrize("phase", ["prepare", "saved"])
@pytest.mark.parametrize("returned,rehash", [("changed", False), ("changed", True), ("original", True)])
def test_numerical_replay_cannot_mutate_its_contract_input(monkeypatch, phase, returned, rehash):
    shared = _owner_with_contract(monkeypatch)
    advice = subject.prepare_intent_384_advice(instruction=INSTRUCTION, config=config())
    before = deepcopy(advice)
    def changed(report, *args, **kwargs):
        original = deepcopy(report)
        report["binding"]["source_audit"]["candidate_contract"]["equation"]["operator"] = "subtract"
        if rehash:
            finish(report)
        return report if returned == "changed" else original
    shared.verify_intent_action_inference = changed
    if phase == "prepare":
        result = subject.prepare_intent_384_advice(instruction=INSTRUCTION, config=config())
        assert result["status"] == "fail_open_unavailable"
        assert result["failure_stage"] == "datasets_numerical_replay"
        assert result["report"] is None and result["candidate_intent_ir"] is None
        assert result["numerical_replay_verified"] is False and result["continue_planning"]
    else:
        with pytest.raises(ValueError, match="numerical replay differs"):
            subject.validate_intent_384_advice(advice, instruction=INSTRUCTION)
    assert advice == before


@pytest.mark.parametrize("change", ["contract", "authority", "configuration"])
def test_caller_advice_changed_during_replay_cannot_be_accepted(monkeypatch, change):
    shared = _owner_with_contract(monkeypatch)
    advice = subject.prepare_intent_384_advice(instruction=INSTRUCTION, config=config())
    def changed(report, *args, **kwargs):
        assert report is not advice["report"]
        if change == "contract":
            advice["report"]["binding"]["source_audit"]["candidate_contract"]["equation"]["operator"] = "subtract"
            finish(advice["report"])
        elif change == "authority":
            advice["proof_authority"] = True
        else:
            advice["config"]["checkpoint_path"] = "/models/replaced.json"
        resign(advice)
        return deepcopy(report)
    shared.verify_intent_action_inference = changed
    with pytest.raises(ValueError, match="saved advice changed during numerical replay"):
        subject.validate_intent_384_advice(advice, instruction=INSTRUCTION)
    # The caller retains its own mutated object; validation must not rewrite it.
    assert advice["advice_sha256"] == subject._sha(subject._wire(
        {key: value for key, value in advice.items() if key != "advice_sha256"}))


def test_preparation_detaches_producer_report_before_replay(monkeypatch):
    shared = _owner_with_contract(monkeypatch)
    prepare = shared.prepare_intent_action_inference
    retained = {}
    def capture(*args, **kwargs):
        report = prepare(*args, **kwargs)
        retained["report"] = report
        return report
    def changed(report, *args, **kwargs):
        assert report is not retained["report"]
        retained["report"]["binding"]["source_audit"]["candidate_contract"]["equation"]["operator"] = "subtract"
        finish(retained["report"])
        return deepcopy(report)
    shared.prepare_intent_action_inference = capture
    shared.verify_intent_action_inference = changed
    advice = subject.prepare_intent_384_advice(instruction=INSTRUCTION, config=config())
    assert advice["status"] == "semantic_candidate_advice" and advice["numerical_replay_verified"]
    assert advice["report"]["binding"]["source_audit"]["candidate_contract"]["equation"]["operator"] == "add"
    assert retained["report"]["binding"]["source_audit"]["candidate_contract"]["equation"]["operator"] == "subtract"
    assert advice["report"]["report_sha256"] == subject._sha(subject._wire(
        {key: value for key, value in advice["report"].items() if key != "report_sha256"}))


def test_mutated_replay_cannot_supply_planner_contract_slots(monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime.intent_advisor_selection import intent_384_planner_summary
    shared = _owner_with_contract(monkeypatch)
    advice = subject.prepare_intent_384_advice(instruction=INSTRUCTION, config=config())
    before = deepcopy(advice)
    def changed(report, *args, **kwargs):
        report["binding"]["source_audit"]["candidate_contract"]["equation"]["operator"] = "subtract"
        return finish(report)
    shared.verify_intent_action_inference = changed
    summary, result = intent_384_planner_summary(advice, instruction=INSTRUCTION)
    assert summary is None and result["status"] == "fail_open_unavailable"
    assert result["continue_planning"] and result["raw_instruction_preserved"]
    assert all(result[key] is False for key in subject.FALSE)
    assert advice == before


@pytest.mark.parametrize("change", ["raw_candidate", "native_candidate", "report_native", "raw_hash", "native_hash", "instruction", "authority", "scope", "replay_claim", "extra"])
def test_rehashed_saved_advice_cannot_replace_either_candidate_or_provenance(monkeypatch, change):
    owner(monkeypatch)
    advice = subject.prepare_intent_384_advice(instruction=INSTRUCTION, config=config())
    if change == "raw_candidate": advice["raw_candidate_ir"]["document"]["invented"] = True
    elif change == "native_candidate": advice["candidate_intent_ir"]["invented"] = True
    elif change == "report_native":
        advice["report"]["native_intent_ir"]["invented"] = True; finish(advice["report"])
    elif change == "raw_hash": advice["raw_candidate_sha256"] = "0" * 64
    elif change == "native_hash": advice["native_candidate_sha256"] = "0" * 64
    elif change == "instruction": advice["instruction_sha256"] = "0" * 64
    elif change == "authority": advice["proof_authority"] = True
    elif change == "scope": advice["scope"] = "verified everything"
    elif change == "replay_claim": advice["numerical_replay_verified"] = False
    else: advice["trusted"] = True
    resign(advice)
    with pytest.raises(ValueError): subject.validate_intent_384_advice(advice, instruction=INSTRUCTION)


def test_new_schema_dispatch_uses_verified_native_document_and_keeps_old_route_separate(monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import intent_autoencoder_advisor as old
    native = owner(monkeypatch)
    advice = subject.prepare_intent_384_advice(instruction=INSTRUCTION, config=config())
    monkeypatch.setattr(old, "validate_intent_advice", lambda *args, **kwargs: pytest.fail("new route used old codec"))
    document, checksum = effects._intent(INSTRUCTION, advice)
    assert document == advice["report"]["binding"]["bound_candidate"] and checksum == PIN
    assert document["document"] == advice["candidate_intent_ir"]
    assert document != advice["raw_candidate_ir"] and native.calls[-1][0] == "verify"
    before = deepcopy(advice)
    document["caller_mutated_copy"] = True
    assert advice == before


def test_cross_source_transport_uses_bound_native_document_and_preserves_raw_candidate(monkeypatch):
    from test.api.semantic_state.test_intent_code_effect_advisor import args, owner as effect_owner
    native = owner(monkeypatch)
    advice = subject.prepare_intent_384_advice(instruction=INSTRUCTION, config=config())
    values = args()
    values.update(instruction=INSTRUCTION, intent_advice=advice)
    gate = effect_owner(monkeypatch)
    before = deepcopy(values)
    result = effects.prepare_intent_code_effect_advice(**values)
    assert result["status"] == "contract_interpretation_advice"
    assert gate.calls[0][1][0]["intent_candidate_ir"] == advice["report"]["binding"]["bound_candidate"]
    assert gate.calls[0][1][0]["intent_candidate_ir"] != advice["raw_candidate_ir"]
    assert result["intent_checkpoint_sha256"] == PIN and result["intent_advice_sha256"] == advice["advice_sha256"]
    assert values == before and native.calls[-1][0] == "verify"


def test_missing_shared_owner_is_sanitized_and_preserves_original_instruction(monkeypatch):
    def missing(): raise ImportError("private machine information")
    monkeypatch.setattr(subject, "_owner", missing)
    report = subject.prepare_intent_384_advice(instruction=INSTRUCTION, config=config())
    assert report["status"] == "fail_open_unavailable" and report["continue_planning"]
    assert report["instruction_sha256"] == hashlib.sha256(INSTRUCTION.encode()).hexdigest()
    assert report["error_type"] == "ImportError" and "private machine information" not in json.dumps(report)
    assert subject.validate_intent_384_advice(report, instruction=INSTRUCTION) == report


def _assert_native_builder_identity(instruction, candidate):
    """Real typed association owner, independent of the transport fakes."""
    from ipfs_datasets_py.logic.formalization.autoencoder import intent_action_association as association_owner
    from ipfs_datasets_py.logic.formalization.autoencoder import intent_code_effects as effect_owner
    from ipfs_datasets_py.logic.software_verification.program import ProgramExpression
    source = "def compute(capacity: int, threshold: int) -> int:\n    return capacity + threshold\n"
    operands = ("expr:capacity", "expr:threshold")
    code = dict(kind="program_expression", document=ProgramExpression("expr:result", "binary", "integer",
        operand_ids=operands, evaluation_order=operands, operator="+", source_ref_ids=("source",)).to_dict())
    domains = {name: dict(lower=-1, upper=1) for name in ("capacity", "threshold")}
    association = association_owner.build_intent_action_association(instruction, candidate, source, code, domains,
        action_id="action", input_parameter_mapping={"left": "capacity", "right": "threshold"})
    report = effect_owner.prepare_intent_code_effects(instruction, candidate, source, code, domains, association)
    assert association["intent_candidate_sha256"] == subject._sha(subject._wire(candidate))
    assert report["intent_candidate_ir"] == candidate and report["status"] == "satisfied"
    assert report["enabled_case_count"] > 0 and not report["proof_authority"]


def test_adapter_document_envelope_matches_real_association_builder_identity(monkeypatch):
    from ipfs_datasets_py.logic.intent_ir.formalize import action_contracts as codec
    instruction = "the calculator must compute result; requires true; ensures result = old(left) + old(right) and returned."
    shared = owner(monkeypatch)
    transport = shared.report

    def authored_report(source, **options):
        report = transport(source, **options)
        raw = codec.source_to_target(source)
        binding = codec.bind_candidate_source(source, raw)
        report.update(raw_candidate_ir=raw, native_intent_ir=binding["bound_candidate"]["document"], binding=binding)
        return finish(report)

    # Authored native control isolates the consumer's envelope identity. The
    # optional checkpoint test below covers actual learned numerical replay.
    shared.report = authored_report
    advice = subject.prepare_intent_384_advice(instruction=instruction, config=config())
    candidate, _ = effects._intent(instruction, advice)
    _assert_native_builder_identity(instruction, candidate)


@pytest.mark.parametrize("change", ["missing", "bare_document", "changed_document", "extra"])
def test_effect_boundary_requires_exact_replayed_bound_envelope(monkeypatch, change):
    shared = owner(monkeypatch)
    transport = shared.report

    def malformed_binding(source, **options):
        report = transport(source, **options)
        if change == "missing": report.pop("binding")
        elif change == "bare_document": report["binding"]["bound_candidate"] = report["native_intent_ir"]
        elif change == "changed_document": report["binding"]["bound_candidate"]["document"] = {"changed": True}
        else: report["binding"]["bound_candidate"]["extra"] = True
        return finish(report)

    shared.report = malformed_binding
    advice = subject.prepare_intent_384_advice(instruction=INSTRUCTION, config=config())
    with pytest.raises(ValueError, match="source-bound Intent384 envelope"):
        effects._intent(INSTRUCTION, advice)


def test_real_effect_checkpoint_replay_keeps_learned_and_bound_documents_distinct():
    """Optional real local owner regression; root owns the complete evidence run."""
    selected = os.environ.get("IR384_TEST_INTENT_ACTION_CONFIG")
    instruction = os.environ.get("IR384_TEST_INTENT_ACTION_INSTRUCTION")
    if not selected or not instruction:
        pytest.skip("explicit action checkpoint configuration and source instruction required")
    configuration = json.loads(Path(selected).read_bytes())
    artifact = Path(configuration["checkpoint_path"]).read_bytes()
    report = subject.prepare_intent_384_advice(instruction=instruction, config=configuration)
    assert report["status"] == "semantic_candidate_advice", report
    assert report["numerical_replay_verified"] and report["raw_candidate_ir"] is not None
    assert any(action["effect_ids"] for action in report["candidate_intent_ir"]["actions"])
    assert subject.validate_intent_384_advice(report, instruction=instruction) == report
    native, checksum = effects._intent(instruction, report)
    assert native == report["report"]["binding"]["bound_candidate"] and checksum == configuration["checkpoint_sha256"]
    assert native["document"] == report["candidate_intent_ir"]
    _assert_native_builder_identity(instruction, native)
    assert Path(configuration["checkpoint_path"]).read_bytes() == artifact
    assert all(report[key] is False for key in subject.FALSE)
