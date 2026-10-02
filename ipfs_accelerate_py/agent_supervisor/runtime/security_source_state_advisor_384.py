"""Optional consumer of datasets-owned finite source-program state models.

Input domains are explicit caller assumptions. A derived operational model does
not establish an Intent effect, source security, or permission to execute code.
Saved receipts remain observations; live handles are verified before export.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

SCHEMA = "supervisor-security-source-state-384-advice/v1"
MAX_ROWS = 16
MAX_BYTES = 1_048_576
FALSE = dict(proof_authority=False, execution_authority=False, completion_authority=False,
    mutation_authority=False, source_semantics_verified=False, whole_program_semantics_verified=False,
    security_specification_inferred=False, claim_proved=False, default_model_promoted=False,
    input_domains_inferred=False, prediction_repair_performed=False,
    intent_effect_compliance_verified=False, saved_receipt_is_live_authority=False)


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _base(status):
    return dict(schema=SCHEMA, status=status, continue_planning=True, native=None, rows=[],
        input_bindings=[], checkpoint_sha256=None, live_build_verified=False,
        training_steps=0, provider_calls=0, download_calls=0, source_executed=False,
        scope="finite code-source operational model under explicit input bounds; not prompt Intent effect compliance",
        **FALSE)


def _gate():
    from ipfs_datasets_py.logic.formalization.autoencoder import source_state_lake
    return source_state_lake


def _join(inference, source_rows, input_bindings, domains):
    _require(type(source_rows) is list and 1 <= len(source_rows) <= MAX_ROWS, "bounded source rows required")
    sources = {}
    for row in source_rows:
        _require(type(row) is dict and set(row) == {"id", "source_text", "source_sha256"}, "closed target-free source row required")
        identity, text = row["id"], row["source_text"]
        _require(type(identity) is str and 0 < len(identity) <= 256 and identity not in sources, "unique source IDs required")
        _require(type(text) is str and 0 < len(text.encode()) <= 65536, "bounded exact source text required")
        _require(row["source_sha256"] == _sha(text.encode()), "source hash differs")
        sources[identity] = row
    _require(type(domains) is dict and set(domains) == set(sources), "explicit domains must cover exactly the supplied source IDs")
    for domain in domains.values():
        _require(type(domain) is dict and len(domain) == 2 and all(type(name) is str and name for name in domain),
            "two explicit named parameter domains required")
        combinations = 1
        for bound in domain.values():
            _require(type(bound) is dict and set(bound) == {"lower", "upper"}
                and type(bound["lower"]) is int and type(bound["upper"]) is int
                and bound["lower"] <= bound["upper"], "closed exactly typed integer bounds required")
            combinations *= bound["upper"] - bound["lower"] + 1
        _require(combinations <= 64, "finite input Cartesian product exceeds 64 cases")
    _require(type(inference) is dict and inference.get("domain_id") == "security_ir"
        and type(inference.get("checkpoint_sha256")) is str
        and re.fullmatch(r"[a-f0-9]{64}", inference["checkpoint_sha256"]) is not None,
        "exact Security checkpoint inference required")
    predictions = inference.get("rows")
    _require(type(predictions) is list and len(predictions) == len(sources), "complete source prediction population required")
    by_id = {}
    for row in predictions:
        _require(type(row) is dict and type(row.get("id")) is str and row["id"] not in by_id
            and "candidate_ir" in row and "source_sha256" in row, "unique unchanged candidate rows required")
        by_id[row["id"]] = row
    _require(type(input_bindings) is list and len(input_bindings) == len(sources), "complete input bindings required")
    gate_rows, seen_sources, seen_predictions = [], set(), set()
    for binding in input_bindings:
        _require(type(binding) is dict and set(binding) == {"source_id", "inference_id", "source_sha256"}, "closed input binding required")
        sid, iid = binding["source_id"], binding["inference_id"]
        _require(type(sid) is str and sid in sources and sid not in seen_sources
            and type(iid) is str and iid in by_id and iid not in seen_predictions, "exact unique source/prediction join required")
        source, prediction = sources[sid], by_id[iid]
        _require(binding["source_sha256"] == source["source_sha256"] == prediction["source_sha256"], "prediction source binding differs")
        gate_rows.append(dict(id=iid, source_text=source["source_text"],
            candidate_ir=deepcopy(prediction["candidate_ir"]), input_domains=deepcopy(domains[sid])))
        seen_sources.add(sid)
        seen_predictions.add(iid)
    return gate_rows


def _validate_native(native, rows):
    _require(type(native) is dict and native.get("schema") == "source-state-384-lake/v1", "datasets state gate schema differs")
    _require(native.get("source_executed") is False and native.get("automatic_operational_model") is True,
        "source state derivation scope differs")
    _require(native.get("source_replay_passed") is True
        and native.get("input_sha256") == _sha(_wire(rows)), "native state input replay differs")
    _require(native.get("candidate_repaired") is False and native.get("model_inference_performed") is False
        and native.get("input_domains_inferred") is False,
        "native state prediction or input-bound contract differs")
    for field in ("source_semantics_verified", "proof_authority", "execution_authority", "completion_authority", "claim_proved"):
        _require(native.get(field) is False, "native state advice cannot carry authority")
    for field, value in native.items():
        if field in FALSE:
            _require(value is False, "native state authority or inference scope differs")
    reported = native.get("rows")
    _require(type(reported) is list and len(reported) == len(rows), "native state population differs")
    by_id = {row["id"]: row for row in rows}
    seen = set()
    for row in reported:
        _require(type(row) is dict and type(row.get("id")) is str and row["id"] in by_id
            and row["id"] not in seen, "duplicate or foreign native state row")
        expected = by_id[row["id"]]
        _require(row["source_sha256"] == _sha(expected["source_text"].encode())
            and row["candidate_sha256"] == _sha(_wire(expected["candidate_ir"]))
            and row["input_domains_sha256"] == _sha(_wire(expected["input_domains"])), "native source/candidate/domain identity differs")
        _require(type(row.get("semantic_lowering_supported")) is bool, "explicit native state support disposition required")
        for field, value in row.items():
            if field in FALSE:
                _require(value is False, "native state row cannot carry authority")
        seen.add(row["id"])
    return native


def consume_source_state_advice(*, inference, source_rows, input_bindings,
        finite_state_domains, lake=None, maximum_bytes=MAX_BYTES):
    """Derive optional bounded state advice without changing any prediction.

    The native owner replays qualification; a supplied source_contract success
    is never accepted as proof. Missing candidates remain native abstentions.
    Optional-model or tool failure leaves the caller's inference untouched.
    """
    result = _base("fail_open_unavailable")
    stage = "source_state_inputs"
    try:
        _require(type(maximum_bytes) is int and 1024 <= maximum_bytes <= 2_097_152, "bounded advice budget required")
        rows = _join(inference, source_rows, input_bindings, finite_state_domains)
        before = _wire(rows)
        result.update(checkpoint_sha256=inference["checkpoint_sha256"], input_bindings=deepcopy(input_bindings))
        stage = "datasets_state_preparation"
        owner = _gate()
        if lake is None:
            native = owner.prepare_source_state_lean(rows)
        else:
            _require(type(lake) is dict and set(lake) == {"executable", "timeout_seconds"}
                and type(lake["executable"]) is str and Path(lake["executable"]).is_absolute()
                and type(lake["timeout_seconds"]) in (int, float) and 0 < lake["timeout_seconds"] <= 60,
                "explicit bounded local Lake selection required")
            stage = "datasets_state_build"
            execution = owner.build_source_state_lake(rows, lake_executable=lake["executable"],
                timeout_seconds=lake["timeout_seconds"])
            stage = "datasets_state_live_verification"
            native = owner.verify_source_state_lake(execution, rows)
            _require(native == execution.to_dict(), "live state execution receipt differs")
            result["live_build_verified"] = True
        _require(_wire(rows) == before, "native state inputs changed during derivation")
        stage = "datasets_state_result"
        _validate_native(native, rows)
        bindings = {row["inference_id"]: row for row in input_bindings}
        result["rows"] = [dict(source_id=bindings[row["id"]]["source_id"],
            **{key: deepcopy(row[key]) for key in ("id", "source_sha256", "candidate_sha256", "input_domains_sha256",
                "status", "semantic_lowering_supported", "lake_status", "sany_status", "reason")}) for row in native["rows"]]
        supported = sum(row["semantic_lowering_supported"] for row in native["rows"])
        result.update(native=native, status="state_candidate_advice" if supported else "fail_open_no_supported_state_candidates",
            supported_candidate_count=supported, source_count=len(rows))
        stage = "source_state_serialization"
        if len(_wire(result)) > maximum_bytes:
            raise ValueError("optional source state advice exceeds byte budget")
    except Exception as error:
        result = dict(_base("fail_open_unavailable"), failure_stage=stage, error_type=type(error).__name__)
    return result


__all__ = ["consume_source_state_advice"]
