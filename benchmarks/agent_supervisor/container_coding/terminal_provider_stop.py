"""Content-free provider stop observations, never retry or completion authority.

Only failed router invocation receipts are considered. A previously projected
``provider_stop`` field, exception messages and model/tool text are not inputs.
The launch gate must independently bind this observation to its frozen route.
"""
from __future__ import annotations

import math
import re


MAX_INVOCATIONS = 1024
MAX_RETRY_AFTER_SECONDS = 604800
_PROVIDERS = frozenset({"codex_cli", "grok_cli"})
_PHASES = frozenset({"planning", "coding"})
_REASONS = frozenset({"rate_limit", "authentication", "billing", "quota"})
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,255}\Z")
_DIAGNOSTIC_KEYS = frozenset({"phase", "reason_code", "http_status",
    "retry_after_seconds", "retry_after_unbounded"})


def _project(row: dict, diagnostic: dict) -> dict:
    reason = diagnostic["reason_code"]
    provider = row.get("provider")
    provider = provider if type(provider) is str and provider in _PROVIDERS else None
    model = row.get("model")
    model = model if type(model) is str and len(model) <= 256 and _IDENTIFIER.fullmatch(model) else None
    phase = row.get("phase")
    phase = phase if type(phase) is str and phase in _PHASES else None
    valid = (row.get("schema") == "router-implementation-invocation@1"
        and row.get("failure_phase") == "provider_invocation"
        and diagnostic.get("phase") == "provider_invocation"
        and len(diagnostic) <= len(_DIAGNOSTIC_KEYS)
        and all(key in _DIAGNOSTIC_KEYS for key in diagnostic)
        and provider is not None and model is not None and phase is not None)
    if "http_status" in diagnostic:
        value = diagnostic["http_status"]
        valid = valid and type(value) is int and 100 <= value <= 599

    delay = None
    unbounded = False
    has_delay = "retry_after_seconds" in diagnostic
    has_unbounded = "retry_after_unbounded" in diagnostic
    if has_delay:
        value = diagnostic["retry_after_seconds"]
        numeric = (type(value) is int or type(value) is float and math.isfinite(value))
        if numeric and 0 < value <= MAX_RETRY_AFTER_SECONDS and reason == "rate_limit":
            delay = value
        else:
            valid = False
            unbounded = bool(numeric and value > MAX_RETRY_AFTER_SECONDS)
    if has_unbounded:
        unbounded = diagnostic["retry_after_unbounded"] is True
        valid = valid and unbounded and reason == "rate_limit" and not has_delay
    if not valid:
        delay = None
    result = {
        "schema": "terminal-provider-stop@1",
        "state": "rate_limited" if reason == "rate_limit" and valid and not unbounded else "provider_blocked",
        "reason_code": reason, "provider": provider, "model": model, "phase": phase,
        "retry_after_seconds": delay, "observation_valid": valid,
        "retry_authority": False, "automatic_replay_allowed": False,
        "task_completion_authority": False,
    }
    if unbounded:
        result["retry_after_unbounded"] = True
    return result


def provider_stop_observation(report: dict) -> dict | None:
    """Derive a bounded stop from retained failures without changing usage.

    Missing or unrelated failures produce no provider-specific observation.
    A recognized stop with malformed metadata remains blocked; invalid input
    populations raise rather than silently examining only a prefix. Multiple
    failures cannot weaken a block or shorten an observed cooldown.
    """
    if type(report) is not dict:
        raise ValueError("provider stop requires a report object")
    rows = report.get("provider_invocations", [])
    if type(rows) is not list or len(rows) > MAX_INVOCATIONS:
        raise ValueError("provider stop requires a bounded invocation list")
    observations = []
    population_valid = not report.get("unreceipted_provider_attempt")
    identities = {}
    for row in rows:
        if type(row) is not dict:
            population_valid = False
            continue
        identity = row.get("invocation_id")
        identity_valid = (type(identity) is str and len(identity) <= 256
            and _IDENTIFIER.fullmatch(identity) is not None)
        if not identity_valid:
            population_valid = False
        if row.get("status") != "failed":
            if identity_valid:
                if identity in identities and identities[identity] != row.get("status"):
                    population_valid = False
                identities[identity] = row.get("status")
            continue
        diagnostic = row.get("provider_failure")
        if type(diagnostic) is not dict:
            population_valid = False
            continue
        reason = diagnostic.get("reason_code")
        if type(reason) is not str or reason not in _REASONS:
            population_valid = False
            continue
        projected = _project(row, diagnostic)
        if identity_valid:
            if identity in identities and identities[identity] != projected:
                population_valid = False
            identities[identity] = projected
        observations.append(projected)
    if not observations:
        return None
    result = dict(max(observations, key=lambda value: (
        value["state"] == "provider_blocked", not value["observation_valid"],
        value.get("retry_after_unbounded") is True, value["retry_after_seconds"] or 0)))
    routes = {(value["provider"], value["model"]) for value in observations}
    if len(routes) != 1:
        result.update(state="provider_blocked", observation_valid=False,
                      provider=None, model=None, retry_after_seconds=None)
    elif not population_valid:
        result.update(state="provider_blocked", observation_valid=False, retry_after_seconds=None)
    return result
