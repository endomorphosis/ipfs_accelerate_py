"""Optional explicit contracts between decoded Intent and code observations.

Associations and finite input domains are caller declarations. The datasets
owner interprets them against unchanged candidates; this consumer cannot infer
a missing effect, repair a prediction, or turn a saved receipt into authority.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

CONFIG_SCHEMA = "supervisor-intent-code-effect-config/v1"
ACTION_CONFIG_SCHEMA = "supervisor-intent-code-effect-config/v2"
SCHEMA = "supervisor-intent-code-effect-advice/v1"
MAX_BYTES = 2_097_152
MAX_ROWS = 16
FALSE = dict(proof_authority=False, execution_authority=False, completion_authority=False,
    mutation_authority=False, source_semantics_verified=False, whole_program_semantics_verified=False,
    normative_compliance_verified=False, intent_source_semantics_verified=False,
    security_specification_inferred=False, claim_proved=False, default_model_promoted=False,
    association_inferred=False, input_domains_inferred=False, candidate_repaired=False,
    intent_meaning_verified=False, whole_instruction_verified=False,
    fresh_security_inference_replayed=False, saved_receipt_is_live_authority=False)


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _hash(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _base(status):
    return dict(schema=SCHEMA, status=status, continue_planning=True, native=None, rows=[],
        selected_contract_count=0, live_build_verified=False, intent_advice_sha256=None,
        instruction_sha256=None, intent_checkpoint_sha256=None, security_checkpoint_sha256=None,
        security_advice_sha256=None, training_steps=0, provider_calls=0, download_calls=0,
        source_executed=False, scope="selected explicit Intent/code interpretations under finite caller input bounds",
        security_inference_provenance="supplied inference identity; no independent numerical replay",
        **FALSE)


def _gate():
    from ipfs_datasets_py.logic.formalization.autoencoder import intent_code_effects_lake
    return intent_code_effects_lake


def _selection(config):
    _require(type(config) is dict and set(config) == {"schema", "contracts", "lake"}
        and config["schema"] in {CONFIG_SCHEMA, ACTION_CONFIG_SCHEMA} and len(_wire(config)) <= 262_144,
        "closed explicit Intent/code configuration required")
    contracts = config["contracts"]
    _require(type(contracts) is list and 1 <= len(contracts) <= MAX_ROWS,
        "bounded explicit contract selection required")
    seen = set()
    for item in contracts:
        fields = {"association"} if config["schema"] == CONFIG_SCHEMA else {"action_id", "input_parameter_mapping"}
        _require(type(item) is dict and set(item) == {"id", "source_id", "input_domains"} | fields,
            "closed contract selection required")
        for key in ("id", "source_id"):
            _require(type(item[key]) is str and 0 < len(item[key]) <= 256, "bounded contract identity required")
        _require(item["id"] not in seen, "unique contract IDs required")
        seen.add(item["id"])
        _require(type(item["input_domains"]) is dict, "explicit domains required")
        if config["schema"] == CONFIG_SCHEMA:
            _require(type(item["association"]) is dict, "explicit association required")
        else:
            mapping = item["input_parameter_mapping"]
            _require(item["action_id"] == "action" and type(mapping) is dict
                and set(mapping) == {"left", "right"}
                and all(type(value) is str and 0 < len(value) <= 256 for value in mapping.values())
                and len(set(mapping.values())) == 2,
                "explicit scalar action and bijective input mapping required")
    lake = config["lake"]
    if lake is not None:
        _require(type(lake) is dict and set(lake) == {"executable", "timeout_seconds"}
            and type(lake["executable"]) is str and Path(lake["executable"]).is_absolute()
            and type(lake["timeout_seconds"]) in (int, float) and 0 < lake["timeout_seconds"] <= 60,
            "explicit bounded local Lake selection required")
    return deepcopy(config)


def _intent(instruction, advice):
    if type(advice) is dict and advice.get("schema") == "supervisor-intent-action-384-advice/v1":
        from .intent_384_advisor import validate_intent_384_advice
        checked = validate_intent_384_advice(advice, instruction=instruction)
        _require(checked["status"] == "semantic_candidate_advice" and checked["numerical_replay_verified"] is True
            and type(checked["candidate_intent_ir"]) is dict, "source-supported decoded Intent384 candidate required")
        binding = checked["report"].get("binding")
        candidate = binding.get("bound_candidate") if type(binding) is dict else None
        _require(type(candidate) is dict and set(candidate) == {"kind", "document"}
            and candidate["kind"] == "document"
            and _wire(candidate["document"]) == _wire(checked["candidate_intent_ir"]),
            "exact replayed source-bound Intent384 envelope required")
        return deepcopy(candidate), checked["checkpoint_sha256"]
    from .intent_autoencoder_advisor import validate_intent_advice, SEMANTIC_REPORT_SCHEMAS
    _require(type(instruction) is str and 0 < len(instruction.encode()) <= 65536,
        "exact original instruction required")
    _require(type(advice) is dict and len(_wire(advice)) <= 278_528,
        "bounded original Intent advice required")
    validate_intent_advice(advice, instruction=instruction)
    report = advice.get("report")
    _require(advice.get("status") == "semantic_candidate_advice" and type(report) is dict
        and report.get("schema") in SEMANTIC_REPORT_SCHEMAS
        and report.get("status") == "semantic_candidate_advice"
        and type(report.get("candidate_intent_ir")) is dict and _hash(report.get("checkpoint_sha256")),
        "decoded semantic Intent candidate required")
    return deepcopy(report["candidate_intent_ir"]), report["checkpoint_sha256"]


def _sources(source_rows, advice):
    _require(type(source_rows) is list and 1 <= len(source_rows) <= MAX_ROWS,
        "complete bounded original source population required")
    sources = {}
    for row in source_rows:
        _require(type(row) is dict and set(row) == {"id", "source_text", "source_sha256"},
            "closed target-free original source required")
        sid, text = row["id"], row["source_text"]
        _require(type(sid) is str and 0 < len(sid) <= 256 and sid not in sources,
            "unique original source identity required")
        _require(type(text) is str and 0 < len(text.encode()) <= 65536
            and row["source_sha256"] == _sha(text.encode()), "exact original source bytes required")
        sources[sid] = row
    _require(type(advice) is dict and len(_wire(advice)) <= MAX_BYTES
        and advice.get("schema") in {"supervisor-security-source-program-384-advice/v1",
            "supervisor-security-source-program-384-advice/v2"}, "Security source-program advice required")
    for field in ("proof_authority", "execution_authority", "completion_authority", "source_semantics_verified"):
        _require(advice.get(field) is False, "source advice cannot supply authority")
    _require(advice.get("source_hashes") == {key: row["source_sha256"] for key, row in sources.items()},
        "full original source population differs")
    inference = advice.get("inference")
    _require(type(inference) is dict and inference.get("domain_id") == "security_ir"
        and _hash(inference.get("checkpoint_sha256")), "supplied Security inference identity required")
    selection = advice.get("checkpoint_selection")
    _require(type(selection) is dict and selection.get("checkpoint_sha256") == inference["checkpoint_sha256"],
        "selected Security checkpoint differs")
    predictions = inference.get("rows")
    _require(type(predictions) is list and len(predictions) == len(sources), "complete prediction population required")
    by_id = {}
    for row in predictions:
        _require(type(row) is dict and type(row.get("id")) is str and row["id"] not in by_id
            and "candidate_ir" in row and "source_sha256" in row, "unique unchanged prediction required")
        by_id[row["id"]] = row
    bindings = advice.get("input_bindings")
    _require(type(bindings) is list and len(bindings) == len(sources), "full input bindings required")
    joined, seen = {}, set()
    for binding in bindings:
        _require(type(binding) is dict and set(binding) == {"source_id", "inference_id", "source_sha256"},
            "closed source prediction binding required")
        sid, iid = binding["source_id"], binding["inference_id"]
        _require(type(sid) is str and sid in sources and sid not in joined
            and type(iid) is str and iid in by_id and iid not in seen, "exact unique source prediction join required")
        _require(binding["source_sha256"] == sources[sid]["source_sha256"] == by_id[iid]["source_sha256"],
            "prediction and original source hash differ")
        joined[sid] = (sources[sid], by_id[iid])
        seen.add(iid)
    return deepcopy(joined), inference["checkpoint_sha256"]


def _validate_native(native, rows):
    _require(type(native) is dict and native.get("schema") == "intent-code-effects-lake/v1"
        and native.get("input_sha256") == _sha(_wire(rows))
        and native.get("source_replay_passed") is True, "native contract input identity differs")
    required_false = ("proof_authority", "execution_authority", "completion_authority", "mutation_authority",
        "source_semantics_verified", "intent_meaning_verified", "whole_instruction_verified", "claim_proved",
        "source_executed", "input_domains_inferred", "association_inferred", "candidate_repaired")
    for key in required_false:
        _require(native.get(key) is False, "native contract cannot grant authority or infer association")
    for key in set(native) & set(FALSE):
        _require(native[key] is False, "native contract authority or scope differs")
    reported = native.get("rows")
    _require(type(reported) is list and len(reported) == len(rows), "complete selected contract population required")
    inputs = {row["id"]: row for row in rows}
    seen = set()
    for row in reported:
        _require(type(row) is dict and type(row.get("id")) is str and row["id"] in inputs
            and row["id"] not in seen, "duplicate or foreign native contract")
        expected = inputs[row["id"]]
        for key in ("intent_source", "intent_candidate", "code_source", "code_candidate", "input_domains", "association"):
            source_key = key + "_text" if key.endswith("source") else key + "_ir" if key.endswith("candidate") else key
            value = expected[source_key]
            checksum = _sha(value.encode()) if key.endswith("source") else _sha(_wire(value))
            _require(row.get(key + "_sha256") == checksum, "native contract source/candidate/interpretation binding differs")
        _require(type(row.get("semantic_lowering_supported")) is bool
            and row.get("effect_status") in {None, "satisfied", "refuted", "no_enabled_cases"},
            "explicit native contract disposition required")
        _require(row["semantic_lowering_supported"] == (row["effect_status"] is not None),
            "native support and effect disposition disagree")
        for key in required_false:
            _require(row.get(key) is False, "native contract row cannot grant authority")
        for key in set(row) & set(FALSE):
            _require(row[key] is False, "native contract row authority or scope differs")
        if row["semantic_lowering_supported"]:
            contract = row.get("contract")
            _require(type(contract) is dict and contract.get("status") == row["effect_status"],
                "original native contract disposition required")
            for key in ("intent_source_text", "intent_candidate_ir", "code_source_text", "code_candidate_ir", "input_domains", "association"):
                _require(_wire(contract.get(key)) == _wire(expected[key]), "native contract changed original candidates or interpretation")
        for key in ("bounded_effects_satisfied", "finite_effects_kernel_checked", "counterexample_kernel_checked"):
            _require(type(row.get(key)) is bool, "explicit finite contract check disposition required")
        checked = row["finite_effects_kernel_checked"]
        _require(not checked or (native.get("backend_executed") is True and row.get("lake_status") == "passed"
            and row["semantic_lowering_supported"] and row["effect_status"] in {"satisfied", "refuted", "no_enabled_cases"}),
            "native finite contract evidence lacks successful kernel check")
        _require(row["bounded_effects_satisfied"] == (checked and row["effect_status"] == "satisfied")
            and row["counterexample_kernel_checked"] == (checked and row["effect_status"] == "refuted"),
            "refutation or empty domain must not become positive satisfaction")
        seen.add(row["id"])
    checked = all(row["finite_effects_kernel_checked"] for row in reported)
    _require(native.get("all_candidates_checked") is checked
        and native.get("bounded_effects_satisfied") is (checked and all(row["effect_status"] == "satisfied" for row in reported)),
        "selected contract aggregate differs from row checks")
    return native


def prepare_intent_code_effect_advice(*, instruction=None, intent_advice=None,
        security_advice=None, source_rows=(), config=None, maximum_bytes=MAX_BYTES):
    """Check explicit selected interpretations while preserving both advisors.

    Intent advice is replayed by its existing owner. Security numerical origin
    is the supplied inference's identity, never an independent inference claim.
    The task-context caller supplies its freshly prepared Security advice.
    """
    if config is None:
        return _base("disabled")
    stage = "configuration"
    try:
        _require(type(maximum_bytes) is int and 1024 <= maximum_bytes <= MAX_BYTES,
            "bounded optional advice budget required")
        selected = _selection(config)
        stage = "intent_advice_replay"
        _require(type(instruction) is str and 0 < len(instruction.encode()) <= 65536,
            "exact original instruction required")
        _require(type(intent_advice) is dict, "bounded original Intent advice required")
        intent_wire = _wire(intent_advice)
        intent_maximum = (1_048_576 if intent_advice.get("schema") ==
            "supervisor-intent-action-384-advice/v1" else 278_528)
        _require(len(intent_wire) <= intent_maximum, "bounded original Intent advice required")
        intent_snapshot = deepcopy(intent_advice)
        stage = "source_prediction_join"
        joined, code_checkpoint = _sources(source_rows, security_advice)
        security_wire = _wire(security_advice)
        captured_inputs = ((intent_advice, intent_wire), (security_advice, security_wire),
            (source_rows, _wire(source_rows)), (config, _wire(selected)))
        def require_current_inputs():
            _require(all(_wire(value) == captured for value, captured in captured_inputs),
                "original Intent/code inputs changed during checking")
        stage = "intent_advice_replay"
        candidate, intent_checkpoint = _intent(instruction, intent_snapshot)
        _require(_wire(intent_snapshot) == intent_wire, "Intent replay changed its supplied advice")
        require_current_inputs()
        rows, bindings = [], []
        for item in selected["contracts"]:
            _require(item["source_id"] in joined, "selected contract source is outside captured inference")
            source, prediction = joined[item["source_id"]]
            if selected["schema"] == ACTION_CONFIG_SCHEMA:
                stage = "datasets_action_association"
                from ipfs_datasets_py.logic.formalization.autoencoder import intent_action_association as builder
                arguments = (instruction, deepcopy(candidate), source["source_text"],
                    deepcopy(prediction["candidate_ir"]), deepcopy(item["input_domains"]))
                options = dict(action_id=item["action_id"],
                    input_parameter_mapping=deepcopy(item["input_parameter_mapping"]))
                before_binding = _wire([arguments, options])
                association = builder.build_intent_action_association(*arguments, **options)
                _require(before_binding == _wire([arguments, options]),
                    "generated association changed predictions or caller declarations")
                require_current_inputs()
                _require(type(association) is dict and len(_wire(association)) <= MAX_BYTES,
                    "bounded generated association required")
                association_wire = _wire(association)
                association = deepcopy(association)
                replay_association = deepcopy(association)
                replay_arguments, replay_options = deepcopy(arguments), deepcopy(options)
                verified = builder.verify_intent_action_association(
                    replay_association, *replay_arguments, **replay_options)
                _require(association_wire == _wire(replay_association) == _wire(verified)
                    and before_binding == _wire([replay_arguments, replay_options]),
                    "generated association changed predictions or caller declarations")
                require_current_inputs()
            else:
                association = item["association"]
            rows.append(dict(id=item["id"], intent_source_text=instruction, intent_candidate_ir=deepcopy(candidate),
                code_source_text=source["source_text"], code_candidate_ir=deepcopy(prediction["candidate_ir"]),
                input_domains=deepcopy(item["input_domains"]), association=deepcopy(association)))
            bindings.append(dict(id=item["id"], source_id=item["source_id"], inference_id=prediction["id"]))
        before = _wire(rows)
        result = _base("contract_interpretation_advice")
        result.update(instruction_sha256=_sha(instruction.encode()), intent_advice_sha256=intent_snapshot["advice_sha256"],
            intent_checkpoint_sha256=intent_checkpoint, security_checkpoint_sha256=code_checkpoint,
            security_advice_sha256=_sha(security_wire), selected_contract_count=len(rows),
            input_bindings=bindings)
        if selected["schema"] == ACTION_CONFIG_SCHEMA:
            result.update(association_profile=builder.PROFILE, association_replay_verified=True,
                configuration_sha256=_sha(_wire(selected)),
                action_selections=[{key: deepcopy(item[key]) for key in
                    ("id", "source_id", "action_id", "input_parameter_mapping", "input_domains")}
                    for item in selected["contracts"]])
        stage = "datasets_contract_preparation"
        owner = _gate()
        if selected["lake"] is None:
            native = owner.prepare_intent_code_effects_lean(rows)
        else:
            lake = selected["lake"]
            stage = "datasets_contract_build"
            execution = owner.build_intent_code_effects_lake(rows, lake_executable=lake["executable"],
                timeout_seconds=lake["timeout_seconds"])
            stage = "datasets_contract_live_verification"
            native = owner.verify_intent_code_effects_lake(execution, rows)
            _require(native == execution.to_dict(), "live contract receipt differs")
            result["live_build_verified"] = True
        _require(_wire(rows) == before, "native contract changed an input candidate or interpretation")
        require_current_inputs()
        stage = "datasets_contract_result"
        _validate_native(native, rows)
        by_id = {row["id"]: row for row in bindings}
        result["rows"] = [{**by_id[row["id"]], **{key: deepcopy(row.get(key)) for key in (
            "status", "reason", "effect_status", "semantic_lowering_supported", "case_count", "enabled_case_count",
            "bounded_effects_satisfied", "finite_effects_kernel_checked", "counterexample_kernel_checked", "lake_status")}}
            for row in native["rows"]]
        result.update(native=native, all_selected_contracts_checked=native["all_candidates_checked"],
            selected_bounded_effects_satisfied=native["bounded_effects_satisfied"],
            refuted_contract_count=sum(row["effect_status"] == "refuted" for row in native["rows"]))
        if not any(row["semantic_lowering_supported"] for row in native["rows"]):
            result["status"] = "fail_open_no_supported_contracts"
        elif all(row["effect_status"] == "no_enabled_cases" for row in native["rows"]):
            result["status"] = "contract_no_enabled_inputs"
        stage = "contract_advice_serialization"
        _require(len(_wire(result)) <= maximum_bytes, "optional contract advice exceeds byte budget")
        require_current_inputs()
        return result
    except Exception as error:
        return dict(_base("fail_open_unavailable"), failure_stage=stage, error_type=type(error).__name__)


def prepare_repository_intent_code_effect_advice(*, repository, paths, instruction=None,
        intent_advice=None, security_advice=None, config=None):
    """Recapture the existing permitted source set; never choose files from Intent."""
    if config is None:
        return _base("disabled")
    try:
        from ..analysis.planning_analysis_factory import _contains_secret, _credential_path_reason
        from .security_source_program_advisor_384 import _read
        root = Path(repository).resolve(strict=True)
        _require(type(paths) in (list, tuple) and 1 <= len(paths) <= 256
            and all(type(name) is str for name in paths) and len(paths) == len(set(paths)),
            "bounded explicit permitted source paths required")
        expected = security_advice.get("source_hashes") if type(security_advice) is dict else None
        _require(type(expected) is dict and 1 <= len(expected) <= MAX_ROWS
            and set(expected) <= set(paths), "captured Security source population required")
        captured, rows = {}, []
        for name in sorted(expected):
            _require(type(name) is str and Path(name).as_posix() == name and not Path(name).is_absolute()
                and ".." not in Path(name).parts and name.endswith(".py") and not _credential_path_reason(name),
                "canonical screened permitted Python source required")
            raw = _read(root / name, 65536)
            _require(not _contains_secret(raw) and _sha(raw) == expected[name], "captured source changed or is screened")
            captured[name] = raw
            rows.append(dict(id=name, source_text=raw.decode("utf-8"), source_sha256=expected[name]))
        result = prepare_intent_code_effect_advice(instruction=instruction, intent_advice=intent_advice,
            security_advice=security_advice, source_rows=rows, config=config)
        _require(all(_read(root / name, 65536) == raw for name, raw in captured.items()),
            "source changed during optional contract check")
        return result
    except Exception as error:
        return dict(_base("fail_open_unavailable"), failure_stage="source_capture", error_type=type(error).__name__)


__all__ = ["prepare_intent_code_effect_advice", "prepare_repository_intent_code_effect_advice"]
