"""Explicit cross-source joins and advisory scope; no guessed Intent effects."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import intent_code_effect_advisor as subject
from ipfs_accelerate_py.agent_supervisor.runtime import intent_autoencoder_advisor as intent_owner

INSTRUCTION = "The agent must preserve the result."
SOURCE = "def compute(left: int, right: int) -> int:\n    return left + right\n"
PIN = "a" * 64
INTENT_PIN = "b" * 64


def args():
    checksum = hashlib.sha256(SOURCE.encode()).hexdigest()
    return dict(instruction=INSTRUCTION, intent_advice={
        "status": "semantic_candidate_advice", "advice_sha256": "c" * 64,
        "report": {"schema": intent_owner.ROUNDTRIP_REPORT_SCHEMA, "status": "semantic_candidate_advice",
            "checkpoint_sha256": INTENT_PIN, "candidate_intent_ir": {"authored_transport_fixture": "unchanged"}}},
        security_advice={"schema": "supervisor-security-source-program-384-advice/v2",
            "checkpoint_selection": {"checkpoint_sha256": PIN}, "source_hashes": {"code.py": checksum},
            "input_bindings": [{"source_id": "code.py", "inference_id": "input-0", "source_sha256": checksum}],
            "inference": {"domain_id": "security_ir", "checkpoint_sha256": PIN,
                "rows": [{"id": "input-0", "source_sha256": checksum, "candidate_ir": {"operator": "+"}}]},
            **{key: False for key in ("proof_authority", "execution_authority", "completion_authority", "source_semantics_verified")}},
        source_rows=[dict(id="code.py", source_text=SOURCE, source_sha256=checksum)],
        config=dict(schema=subject.CONFIG_SCHEMA, lake=None, contracts=[dict(id="contract:first", source_id="code.py",
            input_domains={"left": {"lower": -1, "upper": 1}, "right": {"lower": 0, "upper": 1}},
            association={"authored_transport_fixture": "caller interpretation"})]))


@pytest.fixture
def validate_transport(monkeypatch):
    """Only exercise envelope transport; native integration tests are separate."""
    calls = []
    def validate(advice, *, instruction):
        calls.append((deepcopy(advice), instruction))
        if instruction != INSTRUCTION:
            raise ValueError("instruction binding differs")
        return advice
    monkeypatch.setattr(intent_owner, "validate_intent_advice", validate)
    return calls


class Owner:
    """Transport fake with explicit checked/refuted/empty dispositions."""
    def __init__(self, effect_status="satisfied", change=None):
        self.calls = []
        self.effect_status = effect_status
        self.change = change
        self.issued = {}

    def report(self, rows, checked):
        values = []
        for row in rows:
            supported = self.effect_status is not None
            done = checked and supported
            value = dict(id=row["id"], status="passed" if done else "prepared" if supported else "unsupported",
                reason=None, effect_status=self.effect_status, semantic_lowering_supported=supported,
                case_count=6, enabled_case_count=0 if self.effect_status == "no_enabled_cases" else 6,
                bounded_effects_satisfied=done and self.effect_status == "satisfied",
                finite_effects_kernel_checked=done, counterexample_kernel_checked=done and self.effect_status == "refuted",
                lake_status="passed" if done else "not_run", **subject.FALSE, source_executed=False)
            value["contract"] = {**deepcopy(row), "status": self.effect_status} if supported else None
            for name in ("intent_source", "intent_candidate", "code_source", "code_candidate", "input_domains", "association"):
                key = name + "_text" if name.endswith("source") else name + "_ir" if name.endswith("candidate") else name
                raw = row[key].encode() if name.endswith("source") else subject._wire(row[key])
                value[name + "_sha256"] = subject._sha(raw)
            values.append(value)
        result = dict(schema="intent-code-effects-lake/v1", input_sha256=subject._sha(subject._wire(rows)),
            source_replay_passed=True,
            backend_executed=checked, rows=values,
            all_candidates_checked=all(row["finite_effects_kernel_checked"] for row in values),
            bounded_effects_satisfied=all(row["bounded_effects_satisfied"] for row in values),
            **subject.FALSE, source_executed=False)
        if self.change:
            self.change(result)
        return result

    def prepare_intent_code_effects_lean(self, rows):
        self.calls.append(("prepare", deepcopy(rows)))
        return self.report(rows, False)

    def build_intent_code_effects_lake(self, rows, **options):
        self.calls.append(("build", deepcopy(rows), options))
        result = self.report(rows, True)
        handle = SimpleNamespace(to_dict=lambda: deepcopy(result))
        self.issued[id(handle)] = (deepcopy(rows), result)
        return handle

    def verify_intent_code_effects_lake(self, execution, rows):
        self.calls.append(("verify", deepcopy(rows)))
        original, report = self.issued[id(execution)]
        assert original == rows
        return deepcopy(report)


def owner(monkeypatch, **options):
    value = Owner(**options)
    monkeypatch.setattr(subject, "_gate", lambda: value)
    return value


def test_disabled_consumer_does_not_import_or_replay_optional_owners(monkeypatch):
    monkeypatch.setattr(subject, "_gate", lambda: pytest.fail("disabled optional gate imported"))
    monkeypatch.setattr(subject, "_intent", lambda *a: pytest.fail("disabled Intent replay"))
    assert subject.prepare_intent_code_effect_advice()["status"] == "disabled"


def test_explicit_contract_joins_separate_originals_without_changing_either_advisor(monkeypatch, validate_transport):
    gate = owner(monkeypatch)
    inputs = args(); before = deepcopy(inputs)
    result = subject.prepare_intent_code_effect_advice(**inputs)
    assert result["status"] == "contract_interpretation_advice" and inputs == before
    assert validate_transport == [(inputs["intent_advice"], INSTRUCTION)]
    row, = gate.calls[0][1]
    assert row == dict(id="contract:first", intent_source_text=INSTRUCTION,
        intent_candidate_ir=inputs["intent_advice"]["report"]["candidate_intent_ir"], code_source_text=SOURCE,
        code_candidate_ir=inputs["security_advice"]["inference"]["rows"][0]["candidate_ir"],
        input_domains=inputs["config"]["contracts"][0]["input_domains"],
        association=inputs["config"]["contracts"][0]["association"])
    assert result["input_bindings"] == [dict(id="contract:first", source_id="code.py", inference_id="input-0")]
    assert result["intent_checkpoint_sha256"] == INTENT_PIN and result["security_checkpoint_sha256"] == PIN
    assert all(result[key] is False for key in subject.FALSE)
    assert not result["all_selected_contracts_checked"] and not result["selected_bounded_effects_satisfied"]


def _second_population(values):
    source = values["source_rows"][0]["source_text"] + "\n# second original input\n"
    checksum = subject._sha(source.encode())
    values["source_rows"].append(dict(id="other.py", source_text=source, source_sha256=checksum))
    security = values["security_advice"]
    security["source_hashes"]["other.py"] = checksum
    security["input_bindings"].append(dict(source_id="other.py", inference_id="input-1", source_sha256=checksum))
    prediction = deepcopy(security["inference"]["rows"][0])
    prediction.update(id="input-1", source_sha256=checksum)
    security["inference"]["rows"].append(prediction)
    contract = deepcopy(values["config"]["contracts"][0])
    contract.update(id="contract:second", source_id="other.py")
    values["config"]["contracts"].append(contract)
    return values


def _change_caller_inputs(values, change):
    if change == "source":
        source = values["source_rows"][1]
        source["source_text"] += "# callback changed original\n"
        checksum = subject._sha(source["source_text"].encode())
        source["source_sha256"] = checksum
        security = values["security_advice"]
        security["source_hashes"][source["id"]] = checksum
        security["input_bindings"][1]["source_sha256"] = checksum
        security["inference"]["rows"][1]["source_sha256"] = checksum
    elif change == "prediction":
        values["security_advice"]["inference"]["rows"][1]["candidate_ir"] = {"callback": "changed prediction"}
    elif change == "security_checkpoint":
        security = values["security_advice"]
        security["inference"]["checkpoint_sha256"] = "d" * 64
        security["checkpoint_selection"]["checkpoint_sha256"] = "d" * 64
    elif change == "intent":
        values["intent_advice"]["advice_sha256"] = "d" * 64
    else:
        domains = values["config"]["contracts"][1]["input_domains"]
        domains[next(iter(domains))]["lower"] = 0


@pytest.mark.parametrize("change", ["source", "prediction", "security_checkpoint", "intent", "configuration"])
def test_intent_callback_cannot_replace_inputs_before_source_capture(monkeypatch, change):
    values = _second_population(args())
    gate = owner(monkeypatch)
    captured = []
    def changed(advice, *, instruction):
        assert advice is not values["intent_advice"]
        captured.append(deepcopy(advice))
        _change_caller_inputs(values, change)
        return advice
    monkeypatch.setattr(intent_owner, "validate_intent_advice", changed)
    report = subject.prepare_intent_code_effect_advice(**values)
    assert len(captured) == 1 and gate.calls == []
    assert report["status"] == "fail_open_unavailable" and report["failure_stage"] == "intent_advice_replay"
    assert report["native"] is None and report["continue_planning"]
    assert all(report[key] is False for key in subject.FALSE)
    # Callbacks own their original mutations; the consumer never rewrites them.
    if change == "source":
        assert "callback changed original" in values["source_rows"][1]["source_text"]
    elif change == "prediction":
        assert values["security_advice"]["inference"]["rows"][1]["candidate_ir"] == {"callback": "changed prediction"}
    elif change == "configuration":
        domains = values["config"]["contracts"][1]["input_domains"]
        assert domains[next(iter(domains))]["lower"] == 0 and report["error_type"] == "ValueError"


def test_intent_callback_cannot_change_the_detached_replay_advice(monkeypatch):
    values = _second_population(args())
    before = deepcopy(values)
    gate = owner(monkeypatch)
    def changed(advice, *, instruction):
        advice["report"]["candidate_intent_ir"] = {"callback": "changed intent"}
        advice["advice_sha256"] = subject._sha(subject._wire(advice))
        return advice
    monkeypatch.setattr(intent_owner, "validate_intent_advice", changed)
    report = subject.prepare_intent_code_effect_advice(**values)
    assert report["status"] == "fail_open_unavailable" and report["failure_stage"] == "intent_advice_replay"
    assert values == before and gate.calls == [] and report["native"] is None


@pytest.mark.parametrize("change", ["source_fields", "source_hash", "prediction_join", "security_size", "intent_size"])
def test_populations_are_bounded_and_validated_before_intent_callback(monkeypatch, change):
    values = args()
    if change == "source_fields": values["source_rows"][0]["target"] = "not an input"
    elif change == "source_hash": values["source_rows"][0]["source_text"] += "# unbound\n"
    elif change == "prediction_join": values["security_advice"]["input_bindings"][0]["source_id"] = "foreign.py"
    elif change == "security_size": values["security_advice"]["oversized"] = "x" * subject.MAX_BYTES
    else: values["intent_advice"]["oversized"] = "x" * 278_528
    monkeypatch.setattr(subject, "_intent", lambda *args: pytest.fail("invalid input reached Intent replay"))
    gate = owner(monkeypatch)
    before = deepcopy(values)
    report = subject.prepare_intent_code_effect_advice(**values)
    assert report["status"] == "fail_open_unavailable" and report["native"] is None
    assert gate.calls == [] and values == before


def _action_population(monkeypatch):
    from test.api.semantic_state.test_intent_action_effect_selection import inputs, authored_transport
    candidate, values = inputs()
    authored_transport(monkeypatch, candidate)
    return _second_population(values)


@pytest.mark.parametrize("change", ["source", "prediction", "security_checkpoint", "intent", "configuration"])
def test_first_association_callback_cannot_change_next_contract_population(monkeypatch, change):
    from ipfs_datasets_py.logic.formalization.autoencoder import intent_action_association as builder
    values = _action_population(monkeypatch)
    gate = owner(monkeypatch)
    original = builder.build_intent_action_association
    calls = []
    def changed(*arguments, **options):
        association = original(*arguments, **options)
        calls.append(arguments[2])
        if len(calls) == 1:
            _change_caller_inputs(values, change)
        return association
    monkeypatch.setattr(builder, "build_intent_action_association", changed)
    report = subject.prepare_intent_code_effect_advice(**values)
    assert calls == [values["source_rows"][0]["source_text"]] and gate.calls == []
    assert report["status"] == "fail_open_unavailable" and report["failure_stage"] == "datasets_action_association"
    assert report["native"] is None and report["continue_planning"]
    assert all(report[key] is False for key in subject.FALSE)
    if change == "configuration":
        domains = values["config"]["contracts"][1]["input_domains"]
        assert domains[next(iter(domains))]["lower"] == 0 and report["error_type"] == "ValueError"


@pytest.mark.parametrize("returned,change", [("changed", "effect_binding"), ("changed", "source_identity"), ("original", "effect_binding")])
def test_association_verifier_cannot_redefine_its_comparison_reference(monkeypatch, returned, change):
    from ipfs_datasets_py.logic.formalization.autoencoder import intent_action_association as builder
    values = _action_population(monkeypatch)
    before = deepcopy(values)
    gate = owner(monkeypatch)
    def changed(association, *arguments, **options):
        original = deepcopy(association)
        if change == "effect_binding":
            association["effect_bindings"][0]["expression_id"] = "contract:returned"
        else:
            association["intent_candidate_sha256"] = "0" * 64
        return association if returned == "changed" else original
    monkeypatch.setattr(builder, "verify_intent_action_association", changed)
    report = subject.prepare_intent_code_effect_advice(**values)
    assert report["status"] == "fail_open_unavailable" and report["failure_stage"] == "datasets_action_association"
    assert values == before and gate.calls == [] and report["native"] is None
    assert not report.get("association_replay_verified", False)


def test_multi_source_success_retains_original_supplied_security_identity(monkeypatch, validate_transport):
    values = _second_population(args())
    before = deepcopy(values)
    gate = owner(monkeypatch)
    report = subject.prepare_intent_code_effect_advice(**values)
    assert report["status"] == "contract_interpretation_advice" and values == before
    assert report["selected_contract_count"] == 2
    assert report["security_advice_sha256"] == subject._sha(subject._wire(before["security_advice"]))
    assert report["intent_advice_sha256"] == before["intent_advice"]["advice_sha256"]
    assert report["security_checkpoint_sha256"] == PIN and report["intent_checkpoint_sha256"] == INTENT_PIN
    assert [row["code_source_text"] for row in gate.calls[0][1]] == [row["source_text"] for row in before["source_rows"]]
    assert report["fresh_security_inference_replayed"] is False
    assert report["security_inference_provenance"] == "supplied inference identity; no independent numerical replay"


@pytest.mark.parametrize("effect_status", ["satisfied", "refuted", "no_enabled_cases"])
def test_live_checks_keep_positive_negative_and_empty_dispositions_distinct(monkeypatch, validate_transport, effect_status):
    gate = owner(monkeypatch, effect_status=effect_status)
    inputs = args(); inputs["config"]["lake"] = dict(executable="/tools/lake", timeout_seconds=5)
    result = subject.prepare_intent_code_effect_advice(**inputs)
    assert result["live_build_verified"] and result["all_selected_contracts_checked"]
    assert result["selected_bounded_effects_satisfied"] is (effect_status == "satisfied")
    assert result["rows"][0]["counterexample_kernel_checked"] is (effect_status == "refuted")
    assert result["rows"][0]["effect_status"] == effect_status
    assert result["status"] == ("contract_no_enabled_inputs" if effect_status == "no_enabled_cases" else "contract_interpretation_advice")
    assert [call[0] for call in gate.calls] == ["build", "verify"]


@pytest.mark.parametrize("change", ["instruction", "source", "duplicate_prediction", "source_population", "binding", "checkpoint", "no_intent", "feature_only", "target", "foreign_selection"])
def test_missing_or_mismatched_originals_fail_open_before_native_gate(monkeypatch, validate_transport, change):
    gate = owner(monkeypatch); values = args()
    if change == "instruction": values["instruction"] = "A different instruction."
    elif change == "source": values["source_rows"][0]["source_text"] += "\n"
    elif change == "duplicate_prediction": values["security_advice"]["inference"]["rows"] *= 2
    elif change == "source_population": values["security_advice"]["source_hashes"].clear()
    elif change == "binding": values["security_advice"]["input_bindings"][0]["source_id"] = "foreign.py"
    elif change == "checkpoint": values["security_advice"]["inference"]["checkpoint_sha256"] = "0" * 64
    elif change == "no_intent": values["intent_advice"]["report"]["candidate_intent_ir"] = None
    elif change == "feature_only": values["intent_advice"]["status"] = "feature_advice"
    elif change == "target": values["source_rows"][0]["target"] = "forbidden"
    else: values["config"]["contracts"][0]["source_id"] = "foreign.py"
    before = deepcopy(values)
    report = subject.prepare_intent_code_effect_advice(**values)
    assert report["status"] == "fail_open_unavailable" and report["continue_planning"]
    assert values == before and gate.calls == [] and report["native"] is None


@pytest.mark.parametrize("change", ["intent_source", "intent_candidate", "code_source", "code_candidate", "input_domains", "association", "population", "authority", "top_identity", "false_satisfaction", "unchecked_evidence", "repaired_candidate", "replay"])
def test_native_binding_and_claim_mutations_cannot_be_forwarded(monkeypatch, validate_transport, change):
    def alter(native):
        if change in {"intent_source", "intent_candidate", "code_source", "code_candidate", "input_domains", "association"}:
            native["rows"][0][change + "_sha256"] = "0" * 64
        elif change == "population": native["rows"] *= 2
        elif change == "authority": native["rows"][0]["proof_authority"] = True
        elif change == "top_identity": native["input_sha256"] = "0" * 64
        elif change == "false_satisfaction": native["rows"][0]["bounded_effects_satisfied"] = True
        elif change == "repaired_candidate": native["rows"][0]["contract"]["code_candidate_ir"] = {"repaired": True}
        elif change == "replay": native["source_replay_passed"] = False
        else: native["rows"][0]["finite_effects_kernel_checked"] = True
    owner(monkeypatch, effect_status="refuted", change=alter)
    result = subject.prepare_intent_code_effect_advice(**args())
    assert result["status"] == "fail_open_unavailable" and result["native"] is None
    assert result["failure_stage"] == "datasets_contract_result"


def test_abstaining_source_prediction_stays_missing_and_is_not_repaired(monkeypatch, validate_transport):
    gate = owner(monkeypatch, effect_status=None)
    values = args(); values["security_advice"]["inference"]["rows"][0]["candidate_ir"] = None
    result = subject.prepare_intent_code_effect_advice(**values)
    assert result["status"] == "fail_open_no_supported_contracts"
    assert gate.calls[0][1][0]["code_candidate_ir"] is None
    assert values["security_advice"]["inference"]["rows"][0]["candidate_ir"] is None


def test_selected_subset_does_not_claim_all_sources_checked(monkeypatch, validate_transport):
    gate = owner(monkeypatch); values = args()
    extra = deepcopy(values["source_rows"][0]); extra["id"] = "other.py"
    values["source_rows"].append(extra)
    advice = values["security_advice"]
    advice["source_hashes"]["other.py"] = extra["source_sha256"]
    advice["input_bindings"].append(dict(source_id="other.py", inference_id="input-1", source_sha256=extra["source_sha256"]))
    row = deepcopy(advice["inference"]["rows"][0]); row["id"] = "input-1"; advice["inference"]["rows"].append(row)
    result = subject.prepare_intent_code_effect_advice(**values)
    assert result["selected_contract_count"] == 1 and len(gate.calls[0][1]) == 1
    assert result["input_bindings"][0]["source_id"] == "code.py"
    assert "all_sources_checked" not in result


@pytest.mark.parametrize("failure", ["unavailable", "budget", "live_replay"])
def test_optional_failures_leave_original_advice_untouched(monkeypatch, validate_transport, failure):
    gate = owner(monkeypatch); values = args()
    if failure == "unavailable":
        def missing(): raise ImportError("private machine information")
        monkeypatch.setattr(subject, "_gate", missing)
    elif failure == "budget": values["maximum_bytes"] = 1024
    else:
        values["config"]["lake"] = dict(executable="/tools/lake", timeout_seconds=5)
        def reject(*a): raise ValueError("saved receipt")
        gate.verify_intent_code_effects_lake = reject
    before = deepcopy(values)
    result = subject.prepare_intent_code_effect_advice(**values)
    assert result["status"] == "fail_open_unavailable" and result["continue_planning"]
    assert result["native"] is None and not result["live_build_verified"]
    assert values == before and "private machine information" not in json.dumps(result)


@pytest.mark.parametrize("change", ["unknown_schema", "extra_target", "duplicate_contract", "foreign_field", "relative_lake", "unbounded_timeout"])
def test_configuration_requires_explicit_closed_contracts(monkeypatch, validate_transport, change):
    gate = owner(monkeypatch); values = args(); selection = values["config"]
    if change == "unknown_schema": selection["schema"] = "unknown"
    elif change == "extra_target": selection["target"] = "not input"
    elif change == "duplicate_contract": selection["contracts"] *= 2
    elif change == "foreign_field": selection["contracts"][0]["inferred_action"] = "action"
    elif change == "relative_lake": selection["lake"] = dict(executable="lake", timeout_seconds=5)
    else: selection["lake"] = dict(executable="/tools/lake", timeout_seconds=100)
    result = subject.prepare_intent_code_effect_advice(**values)
    assert result["status"] == "fail_open_unavailable" and result["failure_stage"] == "configuration"
    assert gate.calls == [] and validate_transport == []


@pytest.mark.parametrize("change", ["none", "different_source", "unpermitted", "link", "during_check"])
def test_repository_recapture_uses_only_exact_preexisting_security_sources(tmp_path, monkeypatch, validate_transport, change):
    owner(monkeypatch); values = args()
    source = tmp_path / "code.py"; source.write_text(SOURCE)
    selected = ["code.py"]
    if change == "different_source": source.write_text(SOURCE + "\n")
    elif change == "unpermitted": selected = ["unrelated.py"]
    elif change == "link":
        original = tmp_path / "actual.py"; source.rename(original); source.symlink_to(original)
    elif change == "during_check":
        original = subject.prepare_intent_code_effect_advice
        def modified(**options):
            report = original(**options)
            source.write_text(SOURCE + "\n")
            return report
        monkeypatch.setattr(subject, "prepare_intent_code_effect_advice", modified)
    values.pop("source_rows")
    result = subject.prepare_repository_intent_code_effect_advice(repository=tmp_path, paths=selected, **values)
    if change == "none":
        assert result["status"] == "contract_interpretation_advice"
    else:
        assert result["status"] == "fail_open_unavailable" and result["failure_stage"] == "source_capture"
        assert result["native"] is None


def _native_fixture(status="satisfied"):
    import runpy
    import ipfs_datasets_py
    # Both repositories have a top-level tests package. Resolve this explicit
    # integration fixture from the actual datasets checkout, not sys.path order.
    root = Path(ipfs_datasets_py.__file__).resolve().parent.parent
    fixture = runpy.run_path(str(root / "tests/unit/logic/formalization/autoencoder/test_intent_code_effects.py"))["fixture"]
    return fixture(status)


@pytest.mark.parametrize("effect_status", ["satisfied", "refuted", "no_enabled_cases"])
def test_real_owner_and_lake_preserve_authored_boundary_dispositions(monkeypatch, effect_status):
    """Authored Intent transport fixture; actual shared derivation and Lean gate."""
    lake = os.environ.get("IR384_TEST_LAKE_EXECUTABLE")
    if not lake:
        pytest.skip("actual Lake explicitly selected; no download")
    instruction, intent_candidate, source, code_candidate, domains, association = _native_fixture(effect_status)
    values = args()
    values["instruction"] = instruction
    values["intent_advice"]["report"]["candidate_intent_ir"] = intent_candidate
    values["security_advice"]["inference"]["rows"][0]["candidate_ir"] = code_candidate
    assert source == SOURCE
    values["config"]["contracts"][0].update(input_domains=domains, association=association)
    values["config"]["lake"] = dict(executable=lake, timeout_seconds=60)
    # The fixture is explicitly authored. This patch replaces only Intent
    # numerical transport in this test; the production consumer has no bypass.
    monkeypatch.setattr(intent_owner, "validate_intent_advice", lambda advice, **kwargs: advice)
    before = deepcopy(values)
    result = subject.prepare_intent_code_effect_advice(**values)
    assert result["native"] is not None, result
    assert result["live_build_verified"] and result["all_selected_contracts_checked"]
    assert result["rows"][0]["effect_status"] == effect_status
    assert result["selected_bounded_effects_satisfied"] is (effect_status == "satisfied")
    assert result["rows"][0]["counterexample_kernel_checked"] is (effect_status == "refuted")
    assert values == before and all(result[key] is False for key in subject.FALSE)


def test_real_trained_intent_without_effects_remains_unsupported():
    """Actual local Intent weights; Security prediction is a supplied test input."""
    descriptor_path = os.environ.get("IR384_TEST_INTENT_ROUNDTRIP_DESCRIPTOR")
    if not descriptor_path:
        pytest.skip("existing local Intent descriptor explicitly selected; no training/download")
    from ipfs_datasets_py.logic.formalization.autoencoder import intent_code_effects as native
    _, _, source, code_candidate, domains, association = _native_fixture()
    instruction = "the agent may delete the report."
    descriptor = json.loads(Path(descriptor_path).read_bytes())
    decoded = intent_owner.prepare_intent_advice(instruction=instruction, checkpoint_descriptor=descriptor)
    assert decoded["status"] == "semantic_candidate_advice", decoded["status"]
    candidate = decoded["report"]["candidate_intent_ir"]
    assert candidate["actions"] and all(not action["effect_ids"] for action in candidate["actions"])
    requirements = native.intent_code_effect_requirements(instruction, candidate, source, code_candidate, domains)
    association.update({key: requirements[key] for key in native.IDENTITY_FIELDS})
    association.update(action_id=candidate["actions"][0]["action_id"],
        intent_evidence_ref=candidate["actions"][0]["source_ref_ids"][0])
    values = args()
    values.update(instruction=instruction, intent_advice=decoded)
    values["security_advice"]["inference"]["rows"][0]["candidate_ir"] = code_candidate
    values["config"]["contracts"][0].update(input_domains=domains, association=association)
    before = deepcopy(values)
    result = subject.prepare_intent_code_effect_advice(**values)
    assert result["status"] == "fail_open_no_supported_contracts", result
    assert "no declared effects" in result["native"]["rows"][0]["reason"]
    assert not result["all_selected_contracts_checked"] and not result["selected_bounded_effects_satisfied"]
    assert values == before and result["continue_planning"]
