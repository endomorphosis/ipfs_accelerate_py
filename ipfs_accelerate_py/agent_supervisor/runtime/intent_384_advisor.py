"""Opt-in shared 384D Intent action inference with exact numerical replay.

The raw learned native-document envelope is preserved separately from the
datasets-owned provenance-bound document. Neither a model prediction nor its
source audit infers permission, an association with code, or task authority.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

CONFIG_SCHEMA = "supervisor-intent-action-384-config/v1"
SCHEMA = "supervisor-intent-action-384-advice/v1"
REPORT_SCHEMA = "intent-action-inference-384/v1"
MAX_BYTES = 1_048_576
FALSE = dict(proof_authority=False, execution_authority=False, completion_authority=False,
    mutation_authority=False, omission_authority=False, source_semantics_verified=False,
    intent_meaning_verified=False, whole_instruction_verified=False, normative_compliance_verified=False,
    association_inferred=False, candidate_repaired=False, claim_proved=False,
    default_model_promoted=False, source_executed=False)


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _owner():
    from ipfs_datasets_py.logic.formalization.autoencoder import intent_action_runtime_384
    return intent_action_runtime_384


def _config(value):
    _require(type(value) is dict and set(value) == {"schema", "checkpoint_path", "checkpoint_sha256", "embedding_snapshot_path"}
        and value["schema"] == CONFIG_SCHEMA and len(_wire(value)) <= 32768,
        "closed explicit local Intent384 configuration required")
    _require(_hash(value["checkpoint_sha256"]), "independent exact Intent checkpoint SHA256 required")
    for key in ("checkpoint_path", "embedding_snapshot_path"):
        path = value[key]
        if key == "embedding_snapshot_path" and path is None:
            continue
        _require(type(path) is str and 0 < len(path) <= 4096 and Path(path).is_absolute(),
            "explicit absolute local artifact path required")
    return deepcopy(value)


def _instruction(instruction):
    _require(type(instruction) is str and instruction.strip() and len(instruction.encode()) <= 65536,
        "exact bounded original instruction required")


def _options(config):
    return dict(checkpoint_path=config["checkpoint_path"], expected_sha256=config["checkpoint_sha256"],
        snapshot_path=config["embedding_snapshot_path"])


def _base(instruction, status):
    _instruction(instruction)
    return dict(schema=SCHEMA, status=status, instruction_sha256=_sha(instruction.encode()),
        instruction_bytes=len(instruction.encode()), config=None, report=None,
        raw_candidate_ir=None, candidate_intent_ir=None,
        raw_candidate_sha256=None, native_candidate_sha256=None, checkpoint_sha256=None,
        numerical_replay_verified=False, continue_planning=True, raw_instruction_preserved=True,
        error_type=None, failure_stage=None, training_steps=0, provider_calls=0, download_calls=0,
        scope="explicit source-audited 384D action candidate; provenance binding is separate from learned semantic fields",
        **FALSE)


def _finish(advice):
    advice["advice_sha256"] = _sha(_wire(advice))
    return advice


def _report(report, instruction, config):
    _require(type(report) is dict and len(_wire(report)) <= MAX_BYTES
        and report.get("schema") == REPORT_SCHEMA, "shared Intent384 report required")
    _require(report.get("source_sha256") == _sha(instruction.encode())
        and report.get("checkpoint_sha256") == config["checkpoint_sha256"]
        and report.get("checkpoint_path") == config["checkpoint_path"]
        and report.get("snapshot_path") == config["embedding_snapshot_path"],
        "shared source/checkpoint/input selection differs")
    for key in ("proof_authority", "execution_authority", "completion_authority", "source_semantics_verified"):
        _require(report.get(key) is False, "shared inference cannot grant authority")
    for key in set(report) & set(FALSE):
        _require(report[key] is False, "shared inference changed authority or candidate scope")
    _require(_hash(report.get("report_sha256")) and report["report_sha256"] ==
        _sha(_wire({key: value for key, value in report.items() if key != "report_sha256"})),
        "shared inference report digest differs")
    status = report.get("status")
    _require(type(status) is str and (status == "source_supported_action_contract" or status.startswith("fail_open_")),
        "explicit shared inference disposition required")
    if status == "source_supported_action_contract":
        candidate = report.get("raw_candidate_ir")
        _require(type(candidate) is dict and set(candidate) == {"kind", "document"}
            and candidate["kind"] == "document" and type(candidate["document"]) is dict
            and type(report.get("native_intent_ir")) is dict,
            "separate learned native candidate and provenance-bound document required")
    return report


def prepare_intent_384_advice(*, instruction, config=None):
    """Infer and replay explicit local weights; unavailable optional advice fails open."""
    value = _base(instruction, "disabled" if config is None else "fail_open_unavailable")
    if config is None:
        return _finish(value)
    stage = "configuration"
    try:
        selected = _config(config)
        stage = "datasets_inference"
        owner = _owner()
        report = owner.prepare_intent_action_inference(instruction, **_options(selected))
        stage = "datasets_numerical_replay"
        checked = owner.verify_intent_action_inference(report, instruction, **_options(selected))
        _require(_wire(checked) == _wire(report), "shared inference replay differs")
        stage = "datasets_result"
        _report(report, instruction, selected)
        active = report["status"] == "source_supported_action_contract"
        raw = deepcopy(report.get("raw_candidate_ir"))
        native = deepcopy(report.get("native_intent_ir")) if active else None
        value.update(config=selected, report=deepcopy(report), checkpoint_sha256=selected["checkpoint_sha256"],
            status="semantic_candidate_advice" if active else report["status"],
            raw_candidate_ir=raw, candidate_intent_ir=native,
            raw_candidate_sha256=_sha(_wire(raw)) if raw is not None else None,
            native_candidate_sha256=_sha(_wire(native)) if native is not None else None,
            numerical_replay_verified=active)
        stage = "advice_serialization"
        _require(len(_wire(value)) <= MAX_BYTES, "optional Intent384 advice exceeds byte bound")
    except Exception as error:
        value = _base(instruction, "fail_open_unavailable")
        value.update(failure_stage=stage, error_type=type(error).__name__)
    return _finish(value)


def validate_intent_384_advice(advice, *, instruction):
    """Replay the shared numerical owner before accepting a saved native candidate."""
    baseline = _finish(_base(instruction, "disabled"))
    _require(type(advice) is dict and set(advice) == set(baseline) and len(_wire(advice)) <= MAX_BYTES,
        "closed bounded Intent384 advice required")
    _require(advice["schema"] == SCHEMA and advice["advice_sha256"] ==
        _sha(_wire({key: value for key, value in advice.items() if key != "advice_sha256"})),
        "Intent384 advice digest differs")
    variable = {"advice_sha256", "status", "config", "report", "raw_candidate_ir", "candidate_intent_ir",
        "raw_candidate_sha256", "native_candidate_sha256", "checkpoint_sha256", "numerical_replay_verified",
        "error_type", "failure_stage"}
    for key in set(baseline) - variable:
        _require(type(advice[key]) is type(baseline[key]) and advice[key] == baseline[key],
            "Intent384 instruction or authority binding differs")
    if advice["report"] is None:
        _require(advice["status"] in {"disabled", "fail_open_unavailable"}, "missing native report disposition")
        for key in ("config", "raw_candidate_ir", "candidate_intent_ir", "raw_candidate_sha256", "native_candidate_sha256", "checkpoint_sha256"):
            _require(advice[key] is None, "fallback advice cannot supply a candidate or selected weights")
        _require(advice["numerical_replay_verified"] is False, "fallback advice cannot claim inference replay")
        if advice["status"] == "disabled":
            _require(advice["error_type"] is None and advice["failure_stage"] is None, "disabled advice has error metadata")
        else:
            _require(type(advice["error_type"]) is str and advice["error_type"].isidentifier()
                and len(advice["error_type"]) <= 128 and advice["failure_stage"] in {
                    "configuration", "datasets_inference", "datasets_numerical_replay", "datasets_result", "advice_serialization"},
                "bounded fallback error category required")
        return deepcopy(advice)
    config = _config(advice["config"])
    report = _report(advice["report"], instruction, config)
    checked = _owner().verify_intent_action_inference(report, instruction, **_options(config))
    _require(_wire(checked) == _wire(report), "Intent384 numerical replay differs from saved report")
    active = report["status"] == "source_supported_action_contract"
    raw = report.get("raw_candidate_ir")
    native = report.get("native_intent_ir") if active else None
    _require(advice["status"] == ("semantic_candidate_advice" if active else report["status"])
        and advice["checkpoint_sha256"] == config["checkpoint_sha256"]
        and advice["numerical_replay_verified"] is active and advice["error_type"] is None and advice["failure_stage"] is None,
        "Intent384 inference disposition differs")
    _require(_wire(advice["raw_candidate_ir"]) == _wire(raw) and _wire(advice["candidate_intent_ir"]) == _wire(native)
        and advice["raw_candidate_sha256"] == (_sha(_wire(raw)) if raw is not None else None)
        and advice["native_candidate_sha256"] == (_sha(_wire(native)) if native is not None else None),
        "Intent384 raw prediction or native provenance binding differs")
    return deepcopy(advice)


__all__ = ["prepare_intent_384_advice", "validate_intent_384_advice"]
