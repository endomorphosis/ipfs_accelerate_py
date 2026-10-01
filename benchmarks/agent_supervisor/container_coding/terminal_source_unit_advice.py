"""Opt-in datasets-owned clause/function advice before benchmark planning.

The sidecar is optional descriptive evidence. It does not replace the public
instruction, domain roots, admission policy or the existing Intent advice API.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

SCHEMA = "terminal-source-unit-preplanning@1"
FAMILY_SCHEMA = "terminal-source-unit-preplanning@2"
MAX_DESCRIPTOR_BYTES = 32_768
MAX_ADVICE_BYTES = 2_097_152
AUTHORITY = {"proof_authority": False, "execution_authority": False,
    "completion_authority": False, "source_semantics_verified": False}


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, limit):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular local artifact required")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("optional source-unit artifact exceeds explicit byte bound")
    return raw, {"path": str(path), "sha256": _sha(raw), "bytes": len(raw)}


def _descriptor(path):
    if path is None:
        return None, None
    raw, pin = _read(path, MAX_DESCRIPTOR_BYTES)
    value = json.loads(raw)
    if type(value) is not dict:
        raise ValueError("closed checkpoint descriptor object required")
    return value, pin


def _native(instruction, intent, security, family_request=None):
    from ipfs_datasets_py.logic.formalization.autoencoder.source_document import prepare_source_document
    options = {} if family_request is None else {
        "project_logic_families":True,"intent_family_context":family_request["context"],
        "requested_intent_families":family_request["requested_families"]}
    return prepare_source_document(instruction, source_path=".supervisor-instruction.md",
        source_format="markdown", intent_checkpoint=intent, security_checkpoint=security, **options)


def _base(instruction, status, *, report=None, descriptors=None, descriptor_pins=None,
          elapsed_ns=0, error_type=None, family_request=None):
    value = {"schema": SCHEMA, "status": status,
        "instruction_sha256": _sha(instruction.encode()), "instruction_bytes": len(instruction.encode()),
        "report": report, "descriptors": descriptors or {"intent": None, "security": None},
        "descriptor_pins": descriptor_pins or {"intent": None, "security": None},
        "elapsed_ns": elapsed_ns, "error_type": error_type,
        "raw_instruction_preserved": True, "continue_planning": True,
        "before_goal_decomposition": True, **AUTHORITY}
    if family_request is not None:
        value["schema"] = FAMILY_SCHEMA
        value["family_request"] = family_request
    value["advice_sha256"] = _sha(_wire(value))
    return value


def prepare_source_unit_advice(*, instruction, enabled=False,
        intent_descriptor_path=None, security_descriptor_path=None,
        project_logic_families=False, intent_family_context_path=None, requested_intent_families=None):
    """No downloads, training, source execution, Lake or provider calls."""
    if (type(instruction) is not str or not instruction.strip() or type(enabled) is not bool
            or type(project_logic_families) is not bool):
        raise ValueError("exact nonempty instruction and explicit boolean selection required")
    if project_logic_families and not enabled:
        raise ValueError("family projections require enabled source-unit advice")
    if not project_logic_families and (intent_family_context_path is not None or requested_intent_families is not None):
        raise ValueError("family context and selection require enabled family projections")
    if not enabled:
        return _base(instruction, "disabled")
    started = time.monotonic_ns()
    try:
        intent, intent_pin = _descriptor(intent_descriptor_path)
        security, security_pin = _descriptor(security_descriptor_path)
        family_request = None
        if project_logic_families:
            context, context_pin = None, None
            if intent_family_context_path is not None:
                context_raw, context_pin = _read(intent_family_context_path, 262_144)
                context = json.loads(context_raw)
            family_request = {"context":context,"context_pin":context_pin,
                              "requested_families":requested_intent_families}
        report = (_native(instruction, intent, security) if family_request is None else
                  _native(instruction, intent, security, family_request=family_request))
        advice = _base(instruction, "source_unit_candidate_advice" if report["candidates"] else "fail_open_no_candidates",
            report=report, descriptors={"intent": intent, "security": security},
            descriptor_pins={"intent": intent_pin, "security": security_pin},
            elapsed_ns=time.monotonic_ns()-started, family_request=family_request)
        if len(_wire(advice)) > MAX_ADVICE_BYTES:
            raise ValueError("optional source-unit report exceeds sidecar byte bound")
        return advice
    except Exception as exc:
        return _base(instruction, "fail_open_error", error_type=type(exc).__name__,
            elapsed_ns=time.monotonic_ns()-started)


def load_source_unit_advice(*, path, expected_sha256, instruction):
    """Verify exact sidecar/source pins and replay frozen inference once.

    Invalid advice fails closed while the caller continues with its unchanged
    public task. Return replay cost independently from initial preparation cost.
    """
    started = time.monotonic_ns()
    replay = {"inference_replays": 0, "replay_seconds": 0.0}
    try:
        raw, pin = _read(path, MAX_ADVICE_BYTES)
        if pin["sha256"] != expected_sha256:
            raise ValueError("source-unit sidecar digest differs")
        value = json.loads(raw)
        if type(value) is not dict:
            raise ValueError("source-unit advice object required")
        contextual = value.get("schema") == FAMILY_SCHEMA
        expected_fields = set(_base(instruction, "disabled", family_request={} if contextual else None))
        if (set(value) != expected_fields or value["schema"] not in {SCHEMA,FAMILY_SCHEMA}
                or type(value["elapsed_ns"]) is not int or value["elapsed_ns"] < 0
                or any(value[key] is not False for key in AUTHORITY)
                or any(value[key] is not True for key in ("raw_instruction_preserved", "continue_planning", "before_goal_decomposition"))
                or value["instruction_sha256"] != _sha(instruction.encode())
                or value["instruction_bytes"] != len(instruction.encode())
                or value["advice_sha256"] != _sha(_wire({k:v for k,v in value.items() if k != "advice_sha256"}))):
            raise ValueError("source-unit advice identity or authority differs")
        if value["report"] is None:
            if contextual or value["status"] not in {"disabled", "fail_open_error"}:
                raise ValueError("unexpected report-free status")
        else:
            if set(value["descriptors"]) != {"intent", "security"} or set(value["descriptor_pins"]) != {"intent", "security"}:
                raise ValueError("closed domain checkpoint selection required")
            for domain, expected_pin in value["descriptor_pins"].items():
                if expected_pin is None:
                    if value["descriptors"][domain] is not None:
                        raise ValueError("unbound checkpoint descriptor")
                    continue
                descriptor, current_pin = _descriptor(expected_pin["path"])
                if current_pin != expected_pin or descriptor != value["descriptors"][domain]:
                    raise ValueError("checkpoint descriptor changed since preparation")
            family_request = value.get("family_request")
            if contextual:
                if type(family_request) is not dict or set(family_request) != {"context","context_pin","requested_families"}:
                    raise ValueError("closed family request required")
                context_pin = family_request["context_pin"]
                if context_pin is None:
                    if family_request["context"] is not None:
                        raise ValueError("family context requires an independently pinned artifact")
                else:
                    context_raw, current_pin = _read(context_pin["path"],262_144)
                    if current_pin != context_pin or json.loads(context_raw) != family_request["context"]:
                        raise ValueError("family context changed since preparation")
            replay["inference_replays"] += 1
            report = (_native(instruction, **value["descriptors"]) if family_request is None else
                      _native(instruction, **value["descriptors"], family_request=family_request))
            status = "source_unit_candidate_advice" if report["candidates"] else "fail_open_no_candidates"
            if _wire(report) != _wire(value["report"]) or value["status"] != status:
                raise ValueError("source-unit report differs from frozen learned replay")
        advice = value
    except Exception as exc:
        advice = _base(instruction, "fail_open_invalid_sidecar", error_type=type(exc).__name__)
    replay["replay_seconds"] = (time.monotonic_ns()-started)/1_000_000_000
    return advice, replay


def source_unit_planner_summary(advice, *, maximum_bytes):
    """Build an advisory summary solely from a previously replayed report."""
    if advice["status"] != "source_unit_candidate_advice":
        return None
    report = advice["report"]
    candidates = []
    if report["intent"]:
        for unit in report["intent"]["units"]:
            if unit["accepted"]:
                candidates.append({"domain": "intent_ir", "start_char": unit["start_char"],
                    "end_char": unit["end_char"], "frame": unit["inference"]["learned"]["frame"]})
    rich = report.get("rich_intent")
    if rich is not None:
        for unit in rich["units"]:
            if unit["accepted"]:
                candidates.append({"domain":"intent_ir","start_char":unit["start_char"],
                    "end_char":unit["end_char"],"rich_ir":unit["inference"]["rich_ir"],
                    "decoding_method":unit.get("decoding_method","direct_neural_roundtrip"),
                    "syntax_constraint_policy":unit["inference"].get("policy"),
                    "symbolic_structure":({k:v for k,v in unit["inference"]["composition"].items()
                        if k!='source_parts'} if unit["inference"].get("composition") is not None else None),
                    "logic_families":sorted({p["family_id"] for p in unit["selected_projections"]
                        if p["status"] in {"projected","partial","candidate"}} |
                        {p["logic_family"] for p in unit.get("selected_native_targets",[])}),
                    "logic_projection_sha256":unit["inference"]["logic"]["projection_sha256"]})
    for region in report["security_regions"]:
        for equation in region["equations"]:
            projection = equation["projection"]
            if projection["status"] == "candidate":
                candidates.append({"domain": "security_ir", "unit_id": equation["unit_id"],
                    "pure_model": projection["candidate_model"], "assumptions": projection["assumptions"],
                    "typed_ir_available":bool(projection.get("typed_ir")),
                    "logic_projections":([{"family":p["family_id"],"backend":p["backend"],
                        "profile":p["representation_profile"]} for p in equation["family_views"]]
                        if "family_views" in equation else
                        [{"family":p["family_id"],"profile":p["profile_id"]}
                         for p in projection.get("projections",[])]),
                    "projection_sha256":projection["projection_sha256"]})
    summary = {"schema": advice["schema"], "instruction_sha256": advice["instruction_sha256"],
        "report_sha256": report["report_sha256"], "counts": report["counts"],
        "candidates": candidates,
        "limitations": "Partial source-bound candidates. Composed candidates use explicit source grammar for structure and neural round trips for action clauses. Keep all original task requirements; no proof, execution, completion, omission or whole-document authority.",
        **AUTHORITY}
    families = report.get("intent_family_projection")
    if families is not None:
        summary["intent_family_views"] = {
            "report_sha256":families["report_sha256"],
            "available_families":[p["family_id"] for p in families["family_inventory"]
                                  if p["status"] == "available_views"],
            "units":[{"unit_id":unit["unit_id"],
                "environment_sha256":unit["slot_environment"]["environment_sha256"],
                "slot_status":unit["slot_environment"]["status"],
                "slots":[{"slot_id":slot["slot_id"],"sort":slot["resolved_sort"],
                    "origin":slot["origin"],"referent_resolved":slot["referent_resolved"]}
                    for slot in unit["slot_environment"]["slots"]],
                "formula":None if unit["typed_fixture"] is None else unit["typed_fixture"]["formula"]}
                for unit in families["units"]],
            "semantics":"Available views include partial declarations. Typed slots and KG selections are modeling assumptions; no existence, facts, or task correctness established."}
    text = "Optional source-unit autoencoder advice (unverified planning context):\n" + _wire(summary).decode()
    if type(maximum_bytes) is not int or maximum_bytes <= 0 or len(text.encode()) > maximum_bytes:
        raise ValueError("source-unit summary exceeds existing planner bound")
    return text
