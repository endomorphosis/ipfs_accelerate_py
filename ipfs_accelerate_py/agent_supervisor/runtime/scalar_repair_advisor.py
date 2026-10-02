"""Bounded in-memory operator repairs from fresh Intent/code counterexamples.

Each candidate goes through the existing ProgramWorld operator, fresh Security
inference, a newly bound Intent association, and a live Lake check. This module
never changes source files or grants admission, proof, or completion authority.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

from . import intent_384_advisor as intent_owner
from . import intent_code_effect_advisor as effects
from . import security_source_program_advisor_384 as security_owner

SCHEMA = "supervisor-scalar-operator-repair-advice/v1"
MAX_BYTES = 8 * 1024 * 1024
FALSE = dict(proof_authority=False, execution_authority=False, completion_authority=False,
    mutation_authority=False, admission_authority=False, source_semantics_verified=False,
    whole_program_semantics_verified=False, intent_meaning_verified=False, whole_instruction_verified=False,
    normative_compliance_verified=False, claim_proved=False, default_model_promoted=False,
    input_domains_inferred=False, association_inferred=False, candidate_repaired=False,
    source_executed=False, saved_receipt_is_live_authority=False, repair_applied=False)


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def _wire(value):
    return effects._wire(value)


def _digest(value):
    return hashlib.sha256(_wire(value)).hexdigest()


def _checkpoint_pin(path, expected):
    selected = Path(path)
    _require(selected.is_absolute(), "absolute checkpoint path required")
    resolved = selected.resolve(strict=True)
    before = resolved.stat()
    _require(resolved.is_file() and 0 < before.st_size <= 64 * 1024 * 1024,
        "bounded regular checkpoint required")
    with resolved.open("rb") as stream:
        raw = stream.read(64 * 1024 * 1024 + 1)
    after = resolved.stat()
    stable = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    _require(stable(before) == stable(after) and selected.resolve(strict=True) == resolved
        and len(raw) == before.st_size and effects._sha(raw) == expected,
        "selected checkpoint bytes or identity differ")
    return dict(path=str(selected), resolved_path=str(resolved), sha256=expected, bytes=len(raw))


def _owners():
    from ipfs_datasets_py.logic.formalization.autoencoder import intent_action_association as builder
    from ipfs_datasets_py.logic.formalization.autoencoder import intent_code_effects_lake as gate
    from ipfs_datasets_py.logic.formalization.autoencoder.security import source_scalar_repair as proposer
    from ..autonomous_repair import program_world_operators as operators
    return builder, gate, proposer, operators


def _producer_pins(owners):
    import sys
    modules = (sys.modules[__name__], intent_owner, effects, security_owner, *owners)
    return {module.__name__: effects._sha(Path(module.__file__).read_bytes()) for module in modules}


def _base():
    return dict(schema=SCHEMA, status="fail_open_unavailable", continue_planning=True,
        scope="one explicit scalar contract and source; two operator alternatives under caller finite bounds",
        original_source=None, intent_advice=None, initial=None, proposal_report=None, candidates=[],
        satisfied_candidate_ids=[], initial_refutation_live_verified=False, input_pins_rechecked=False,
        original_source_unchanged=False, security_advisor_calls=0, training_steps=0, provider_calls=0,
        download_calls=0, source_writes=0, **FALSE)


def _source(source_rows, selection):
    _require(type(source_rows) is list and len(source_rows) == 1,
        "one explicit complete source required")
    row = source_rows[0]
    _require(type(row) is dict and set(row) == {"id", "source_text", "source_sha256"}
        and type(row["id"]) is str and 0 < len(row["id"]) <= 256
        and Path(row["id"]).as_posix() == row["id"] and not Path(row["id"]).is_absolute()
        and ".." not in Path(row["id"]).parts and row["id"].endswith(".py"),
        "closed canonical Python source identity required")
    _require(type(row["source_text"]) is str and 0 < len(row["source_text"].encode()) <= 65_536
        and row["source_sha256"] == effects._sha(row["source_text"].encode())
        and selection["source_id"] == row["id"], "exact selected source bytes required")
    from ..analysis.planning_analysis_factory import _contains_secret, _credential_path_reason
    _require(not _credential_path_reason(row["id"]) and not _contains_secret(row["source_text"].encode()),
        "screened source required")
    return deepcopy(row)


def _association(builder, instruction, intent_candidate, source, prediction, contract):
    arguments = (instruction, deepcopy(intent_candidate), source["source_text"],
        deepcopy(prediction["candidate_ir"]), deepcopy(contract["input_domains"]))
    options = dict(action_id=contract["action_id"], input_parameter_mapping=deepcopy(contract["input_parameter_mapping"]))
    before = _wire([arguments, options])
    association = builder.build_intent_action_association(*arguments, **options)
    checked = builder.verify_intent_action_association(association, *arguments, **options)
    _require(_wire(association) == _wire(checked) and _wire([arguments, options]) == before,
        "association changed declared inputs or inferred candidates")
    return [dict(id=contract["id"], intent_source_text=instruction, intent_candidate_ir=deepcopy(intent_candidate),
        code_source_text=source["source_text"], code_candidate_ir=deepcopy(prediction["candidate_ir"]),
        input_domains=deepcopy(contract["input_domains"]), association=deepcopy(association))]


def _check_source(record, *, instruction, intent_candidate, source, security_config, contract, lake, owners, result):
    builder, gate, _, _ = owners
    record["failure_stage"] = "security_inference"
    sources = [deepcopy(source)]
    before = _wire([sources, security_config])
    result["security_advisor_calls"] += 1
    record["security_advisor_invoked"] = True
    advice = security_owner.prepare_security_source_program_advice(config=security_config, source_rows=sources)
    record["security_advice"] = deepcopy(advice)
    _require(_wire([sources, security_config]) == before and advice.get("checkpoint_selection") == security_config,
        "Security inference changed selected inputs or checkpoint configuration")
    joined, _ = effects._sources(sources, advice)
    _, prediction = joined[source["id"]]
    _require(advice.get("status") == "source_candidate_advice"
        and prediction.get("source_contract", {}).get("status") == "qualified",
        "fresh Security prediction must qualify against its exact candidate source")
    record["failure_stage"] = "intent_association"
    rows = _association(builder, instruction, intent_candidate, source, prediction, contract)
    record["rows"] = deepcopy(rows)
    before = _wire(rows)
    record["failure_stage"] = "live_lake"
    handle = gate.build_intent_code_effects_lake(rows, lake_executable=lake["executable"],
        timeout_seconds=lake["timeout_seconds"])
    native = gate.verify_intent_code_effects_lake(handle, rows)
    _require(native == handle.to_dict() and _wire(rows) == before, "live check changed candidate contract inputs")
    effects._validate_native(native, rows)
    record["native"] = deepcopy(native)
    row = native["rows"][0]
    _require(native["all_candidates_checked"] is True and row["finite_effects_kernel_checked"] is True,
        "finite candidate disposition must have a successful live kernel check")
    record.update(live_build_verified=True, effect_status=row["effect_status"],
        enabled_case_count=row["enabled_case_count"], bounded_effects_satisfied=row["bounded_effects_satisfied"],
        counterexample_kernel_checked=row["counterexample_kernel_checked"], status="checked_candidate",
        failure_stage=None, error_type=None)
    return handle, rows


def _record(source):
    return dict(status="pending", source_text=source["source_text"], source_sha256=source["source_sha256"],
        security_advisor_invoked=False, security_advice=None, rows=None, native=None,
        live_build_verified=False, effect_status=None, enabled_case_count=None,
        bounded_effects_satisfied=False, counterexample_kernel_checked=False,
        failure_stage=None, error_type=None, **FALSE)


def prepare_scalar_repair_advice(*, instruction, intent_config, security_config, source_rows,
        effect_config, maximum_candidates=2, intent_advice=None):
    """Check inert proposals; no serialized prior counterexample is accepted.

    The complete starting check is fresh even when caller-supplied Intent advice
    is selected. Such Intent advice must replay numerically against the selected
    checkpoint. Source rows contain caller snapshots; repository admission and
    concurrent on-disk source checks belong to the invoking owner boundary.
    """
    result = _base()
    stage = "configuration"
    try:
        _require(type(maximum_candidates) is int and 1 <= maximum_candidates <= 2,
            "explicit candidate cap must be one or two")
        _require(type(intent_config) is dict and type(security_config) is dict,
            "explicit closed checkpoint configurations required")
        original_inputs = _wire([instruction, intent_config, security_config, source_rows, effect_config, intent_advice])
        intent_selection = intent_owner._config(intent_config)
        security_selection = security_owner._config(security_config)
        selection = effects._selection(effect_config)
        _require(selection["schema"] == effects.ACTION_CONFIG_SCHEMA and len(selection["contracts"]) == 1
            and selection["lake"] is not None, "one explicit scalar association and live Lake required")
        contract, lake = selection["contracts"][0], selection["lake"]
        source = _source(source_rows, contract)
        result["original_source"] = deepcopy(source)
        result["configuration_sha256"] = _digest([intent_selection, security_selection, selection, maximum_candidates])
        result["input_sha256"] = effects._sha(original_inputs)
        stage = "checkpoint_pins"
        pins = {name: _checkpoint_pin(config["checkpoint_path"], config["checkpoint_sha256"])
            for name, config in (("intent", intent_selection), ("security", security_selection))}
        result["checkpoint_pins"] = pins
        owners = _owners()
        producer_pins = _producer_pins(owners)
        result["producer_pins"] = producer_pins
        result["producer_pin_scope"] = "listed consumer and direct owner source files; native live gates additionally pin their owners"
        stage = "intent_inference_replay"
        advice = (intent_owner.prepare_intent_384_advice(instruction=instruction, config=intent_selection)
            if intent_advice is None else deepcopy(intent_advice))
        _require(advice.get("config") == intent_selection, "Intent advice must use the exact selected configuration")
        intent_candidate, _ = effects._intent(instruction, advice)
        result["intent_advice"] = deepcopy(advice)
        stage = "initial_live_counterexample"
        initial = _record(source)
        result["initial"] = initial
        initial_handle, initial_rows = _check_source(initial, instruction=instruction,
            intent_candidate=intent_candidate, source=source, security_config=security_selection,
            contract=contract, lake=lake, owners=owners, result=result)
        if initial["effect_status"] != "refuted":
            result["status"] = "no_starting_counterexample"
        else:
            _require(initial["counterexample_kernel_checked"] is True and initial["enabled_case_count"] > 0,
                "live nonvacuous starting counterexample required")
            result["initial_refutation_live_verified"] = True
            stage = "symbolic_proposals"
            _, _, proposer, operators = owners
            proposal_report = proposer.prepare_scalar_operator_repair(initial_handle, initial_rows, row_id=contract["id"])
            verified = proposer.verify_scalar_operator_repair(proposal_report, initial_handle, initial_rows, row_id=contract["id"])
            _require(_wire(verified) == _wire(proposal_report) and proposal_report["status"] == "proposed"
                and type(proposal_report.get("proposals")) is list and len(proposal_report["proposals"]) == 2,
                "complete replayed scalar proposal population required")
            result["proposal_report"] = deepcopy(proposal_report)
            result["proposal_report_sha256"] = _digest(proposal_report)
            proposal_ids = set()
            for index, proposal in enumerate(proposal_report["proposals"]):
                _require(type(proposal.get("id")) is str and proposal["id"] not in proposal_ids,
                    "unique deterministic proposal IDs required")
                proposal_ids.add(proposal["id"])
                candidate_source = dict(id=source["id"], source_text=proposal["source_text"],
                    source_sha256=proposal["after_source_sha256"])
                candidate = _record(candidate_source)
                candidate.update(id=proposal["id"], proposal=deepcopy(proposal),
                    proposal_sha256=_digest(proposal), before_source_sha256=source["source_sha256"], operator_application=None)
                result["candidates"].append(candidate)
                if index >= maximum_candidates:
                    candidate["status"] = "not_checked_candidate_budget"
                    continue
                try:
                    candidate["failure_stage"] = "program_world_operator"
                    _require(proposal["before_source_sha256"] == source["source_sha256"]
                        and candidate_source["source_sha256"] == effects._sha(candidate_source["source_text"].encode()),
                        "proposal original or candidate source digest differs")
                    edit = proposal["expression_edit"]
                    application = operators.apply_program_world_repair_operator(operator_kind="replace_exact_bytes",
                        path=source["id"], source_text=source["source_text"], before_span=edit["before"], after_span=edit["after"],
                        admitted_scope=(source["id"],), proof_obligation_ids=("intent-code:" + contract["id"],))
                    candidate["operator_application"] = application.to_dict()
                    _require(application.accepted_sketch and application.proposal_only
                        and application.after_source == candidate_source["source_text"]
                        and application.before_hash == "sha256:" + source["source_sha256"],
                        "ProgramWorld operator must produce the exact proposed inert source")
                    _check_source(candidate, instruction=instruction, intent_candidate=intent_candidate,
                        source=candidate_source, security_config=security_selection, contract=contract,
                        lake=lake, owners=owners, result=result)
                except Exception as error:
                    candidate.update(status="fail_open_candidate", error_type=type(error).__name__,
                        live_build_verified=False, bounded_effects_satisfied=False, counterexample_kernel_checked=False)
            result["status"] = "candidate_evidence"
        stage = "input_pin_recheck"
        _require(original_inputs == _wire([instruction, intent_config, security_config, source_rows, effect_config, intent_advice]),
            "caller instruction, configurations or original source changed")
        _require(result["configuration_sha256"] == _digest([intent_selection, security_selection, selection, maximum_candidates]),
            "working configurations changed during candidate checking")
        _require(pins == {name: _checkpoint_pin(config["checkpoint_path"], config["checkpoint_sha256"])
            for name, config in (("intent", intent_selection), ("security", security_selection))},
            "checkpoint files changed during repair checking")
        _require(producer_pins == _producer_pins(owners), "consumer or proposal producer changed during checking")
        result.update(input_pins_rechecked=True, original_source_unchanged=True)
        result["satisfied_candidate_ids"] = [row["id"] for row in result["candidates"]
            if row["status"] == "checked_candidate" and row["live_build_verified"]
            and row["bounded_effects_satisfied"] and row["effect_status"] == "satisfied" and row["enabled_case_count"] > 0]
        stage = "serialization"
        _require(len(_wire(result)) <= MAX_BYTES, "repair evidence exceeds bounded output size")
    except Exception as error:
        result.update(status="fail_open_unavailable", failure_stage=stage, error_type=type(error).__name__,
            satisfied_candidate_ids=[], input_pins_rechecked=False)
        if len(_wire(result)) > MAX_BYTES:
            result = dict(_base(), failure_stage=stage, error_type="ValueError", evidence_omitted_over_budget=True)
    result["report_sha256"] = _digest(result)
    return result


__all__ = ["prepare_scalar_repair_advice"]
