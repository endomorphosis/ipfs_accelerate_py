"""Explicit file-backed Intent384 selection at the preplanning boundary.

This additive route pins configuration bytes as well as the selected checkpoint.
Saved candidates are replayed before supplying bounded, advisory planner slots.
The caller retains the original instruction and independent admission roots.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

from . import intent_384_advisor as advisor

SCHEMA = "supervisor-intent-action-384-selection/v1"
MAX_CONFIG_BYTES = 32_768


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def _read(path, maximum):
    path = Path(path)
    _require(path.is_absolute() and not path.is_symlink() and path.is_file(),
        "explicit regular local advisory artifact required")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    _require(len(raw) <= maximum, "optional advisory artifact exceeds its byte bound")
    return raw


def _json(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            _require(key not in value, "duplicate advisory JSON field")
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=unique)


def _fallback(instruction, error):
    value = advisor._base(instruction, "fail_open_unavailable")
    value.update(failure_stage="configuration", error_type=type(error).__name__)
    return advisor._finish(value)


def _selection(selection):
    _require(type(selection) is dict and set(selection) == {"schema", "enabled", "config_path", "config_sha256"}
        and selection["schema"] == SCHEMA and type(selection["enabled"]) is bool,
        "closed Intent384 startup selection required")
    path, checksum = selection["config_path"], selection["config_sha256"]
    _require(path is None or (type(path) is str and 0 < len(path) <= 4096 and Path(path).is_absolute()),
        "bounded absolute Intent384 configuration path required")
    _require(checksum is None or (advisor._hash(checksum) and path is not None and selection["enabled"]),
        "exact selected configuration digest required")
    return deepcopy(selection)


def prepare_intent_384_selection(*, instruction, config_path, enabled=True):
    """Return advice, explicit selection, and elapsed cost before declarations."""
    _require(type(enabled) is bool, "explicit boolean Intent384 selection required")
    selection = dict(schema=SCHEMA, enabled=enabled, config_path=None, config_sha256=None)
    started = time.monotonic_ns()
    if not enabled:
        return advisor.prepare_intent_384_advice(instruction=instruction), selection, 0
    try:
        path = Path(config_path).absolute()
        _require(len(str(path)) <= 4096, "bounded configuration path required")
        selection["config_path"] = str(path)
        raw = _read(path, MAX_CONFIG_BYTES)
        selection["config_sha256"] = advisor._sha(raw)
        config = advisor._config(_json(raw))
        advice = advisor.prepare_intent_384_advice(instruction=instruction, config=config)
        _require(_read(path, MAX_CONFIG_BYTES) == raw, "Intent384 configuration changed during inference")
    except Exception as error:
        advice = _fallback(instruction, error)
    return advice, selection, time.monotonic_ns() - started


def _require_current_selection(*, path, selection, captured):
    _require(advisor._wire(_selection(selection)) == advisor._wire(captured["selection"]),
        "Intent384 startup selection changed during numerical replay")
    _require(_read(path, advisor.MAX_BYTES) == captured["advice_bytes"],
        "saved advice changed during numerical replay")
    if captured["config_bytes"] is not None:
        _require(_read(captured["selection"]["config_path"], MAX_CONFIG_BYTES) == captured["config_bytes"],
            "selected configuration changed during numerical replay")


def _load_verified_selection(*, path, expected_sha256, instruction, selection):
    selected = _selection(selection)
    _require(advisor._hash(expected_sha256), "independent saved advice digest required")
    raw = _read(path, advisor.MAX_BYTES)
    _require(advisor._sha(raw) == expected_sha256, "saved Intent384 advice changed")
    advice = _json(raw)
    config_raw = None
    if selected["enabled"] and selected["config_sha256"] is not None:
        config_raw = _read(selected["config_path"], MAX_CONFIG_BYTES)
        _require(advisor._sha(config_raw) == selected["config_sha256"],
            "selected Intent384 configuration changed")
        config = advisor._config(_json(config_raw))
        if advice.get("report") is not None:
            _require(advice.get("config") == config, "saved advice changed its selected checkpoint configuration")
    else:
        _require(advice.get("report") is None and advice.get("status") == (
            "fail_open_unavailable" if selected["enabled"] else "disabled"),
            "unselected model cannot supply active saved advice")
    captured = dict(selection=selected, advice_bytes=raw, config_bytes=config_raw)
    checked = advisor.validate_intent_384_advice(advice, instruction=instruction)
    _require_current_selection(path=path, selection=selection, captured=captured)
    return checked, captured


def load_intent_384_selection(*, path, expected_sha256, instruction, selection):
    """Reject stale files, source or numerical predictions without blocking planning."""
    try:
        checked, _ = _load_verified_selection(path=path, expected_sha256=expected_sha256,
            instruction=instruction, selection=selection)
        return checked
    except Exception as error:
        return _fallback(instruction, error)


def intent_384_planner_summary(advice, *, instruction, maximum_bytes=8192):
    """Expose only replayed learned contract slots, never new planning authority."""
    try:
        _require(type(maximum_bytes) is int and 1 <= maximum_bytes <= 65_536,
            "bounded planner advisory budget required")
        checked = advisor.validate_intent_384_advice(advice, instruction=instruction)
        if checked["status"] != "semantic_candidate_advice":
            return None, checked
        contract = checked["report"]["binding"]["source_audit"]["candidate_contract"]
        _require(type(contract) is dict, "audited learned contract slots required")
        summary = dict(kind="intent_instruction_advisory", mode="shared_384_action_contract_candidate",
            authority="unverified_candidate_only", advice_sha256=checked["advice_sha256"],
            instruction_sha256=checked["instruction_sha256"], checkpoint_sha256=checked["checkpoint_sha256"],
            raw_candidate_sha256=checked["raw_candidate_sha256"], native_candidate_sha256=checked["native_candidate_sha256"],
            contract=deepcopy(contract), numerical_replay_verified=True,
            reconstruction_is_proof=False, raw_instruction_preserved=True,
            notice="Candidate planning context only. Preserve the original instruction and independently admitted constraints.",
            **advisor.FALSE)
        encoded = advisor._wire(summary)
        _require(len(encoded) <= maximum_bytes, "Intent384 candidate exceeds planner advisory bound")
        return encoded.decode("utf-8"), checked
    except Exception as error:
        return None, _fallback(instruction, error)


def load_intent_384_planner_summary(*, path, expected_sha256, instruction, selection, maximum_bytes=8192):
    """Replay saved advice and retain its file/selection binding through summary replay."""
    try:
        advice, captured = _load_verified_selection(path=path, expected_sha256=expected_sha256,
            instruction=instruction, selection=selection)
        summary, checked = intent_384_planner_summary(advice,
            instruction=instruction, maximum_bytes=maximum_bytes)
        _require_current_selection(path=path, selection=selection, captured=captured)
        return summary, checked
    except Exception as error:
        return None, _fallback(instruction, error)


__all__ = ["prepare_intent_384_selection", "load_intent_384_selection", "intent_384_planner_summary",
    "load_intent_384_planner_summary"]
