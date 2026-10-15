"""Retain dialogue accounting evidence without inventing provider charges.

Codex native receipts can occur both as printed observations and nested in a
dialogue. Their original invocation identity is the accounting identity. A
dialogue wrapper, prompt estimate, or CLI usage summary is never another token
charge. Coverage is derived from attempts, not a cached producer summary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
import re

from .cumulative_usage_reconciliation import INVOCATION_SCHEMA, MAX_RECEIPTS

DIALOGUE_SCHEMA = "supervisor-semantic-decision-dialogue@1"
PROVIDER_SCHEMA = "router-semantic-decision-provider@1"
COVERAGE_SCHEMA = "supervisor-semantic-decision-usage-coverage@1"
OWNER_BINDING_SCHEMA = "supervisor-semantic-decision-owner-binding@1"
TYPED_NOMINATION_BINDING_SCHEMA = "supervisor-semantic-typed-nomination-binding@1"
TYPED_NOMINATION_VERIFICATION_SCHEMA = "supervisor-semantic-typed-nomination-verification@1"
TYPED_NOMINATION_BINDING_FIELDS = frozenset({
    "schema", "derived_nomination_artifact", "derived_nomination_sha256",
    "lineage_artifact", "lineage_sha256",
})
TYPED_NOMINATION_VERIFICATION_FIELDS = frozenset({
    "schema", "task_id", "origin_nomination_sha256", "derived_nomination_sha256",
    "lineage_sha256", "native_prompt_sha256", "native_prompt_bytes",
    "parent_artifact", "parent_sha256", "policy_sha256", "launch_policy_sha256",
    "owner_binding_sha256", "source_freshness_revalidated", "provider_calls",
    "execution_authority", "proof_authority", "completion_authority",
})
MAX_OWNER_BINDING_BYTES = 65_536
MAX_NATIVE_PROMPT_BYTES = 256_000
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _identity(value):
    return type(value) is str and _ID.fullmatch(value) is not None


def _digest(value):
    return type(value) is str and _SHA.fullmatch(value) is not None


def _task_identity(value):
    if type(value) is not str:
        return False
    try:
        size = len(value.encode())
    except UnicodeError:
        return False
    return (1 <= size <= 512 and value == value.strip()
            and not any(char in value for char in ("\x00", "\n", "\r")))


def _artifact_path(value, *, absolute):
    if type(value) is not str:
        return False
    try:
        size = len(value.encode())
    except UnicodeError:
        return False
    if not 1 <= size <= 4096 or any(char in value for char in ("\x00", "\\", ":")):
        return False
    path = PurePosixPath(value)
    return (bool(path.name) and path.is_absolute() is absolute and path.as_posix() == value
            and not any(part in {".", ".."} for part in value.split("/")))


def _owner_binding_digest(receipt):
    """Bind all owner fields while exporting only their digest."""
    raw = json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _valid_typed_nomination_binding(binding):
    return (type(binding) is dict and set(binding) == TYPED_NOMINATION_BINDING_FIELDS
            and binding.get("schema") == TYPED_NOMINATION_BINDING_SCHEMA
            and all(_artifact_path(binding.get(key), absolute=True) for key in (
                "derived_nomination_artifact", "lineage_artifact"))
            and all(_digest(binding.get(key)) for key in (
                "derived_nomination_sha256", "lineage_sha256")))


def _valid_typed_nomination_verification(receipt):
    if (type(receipt) is not dict or set(receipt) != TYPED_NOMINATION_VERIFICATION_FIELDS
            or receipt.get("schema") != TYPED_NOMINATION_VERIFICATION_SCHEMA):
        return False
    try:
        if len(json.dumps(receipt, allow_nan=False).encode()) > MAX_OWNER_BINDING_BYTES:
            return False
    except (ValueError, TypeError, RecursionError, OverflowError):
        return False
    amount = receipt.get("native_prompt_bytes")
    return (
        _task_identity(receipt.get("task_id"))
        and type(amount) is int and 1 <= amount <= MAX_NATIVE_PROMPT_BYTES
        and all(_digest(receipt.get(key)) for key in (
            "origin_nomination_sha256", "derived_nomination_sha256", "lineage_sha256",
            "native_prompt_sha256", "parent_sha256", "policy_sha256",
            "launch_policy_sha256", "owner_binding_sha256"))
        and _artifact_path(receipt.get("parent_artifact"), absolute=False)
        and receipt.get("source_freshness_revalidated") is False
        and type(receipt.get("provider_calls")) is int and receipt["provider_calls"] == 0
        and all(receipt.get(key) is False for key in (
            "execution_authority", "proof_authority", "completion_authority"))
    )


def _typed_verification_matches_owner(verification, owner):
    """Join a trusted observer result to every field of the actual owner row."""
    if not _valid_typed_nomination_verification(verification):
        return False
    binding = owner.get("typed_nomination")
    if not _valid_typed_nomination_binding(binding):
        return False
    try:
        owner_digest = _owner_binding_digest(owner)
    except (ValueError, TypeError, RecursionError, OverflowError, UnicodeError):
        return False
    expected = {
        "task_id": owner["task_id"],
        "origin_nomination_sha256": owner["nomination_sha256"],
        "derived_nomination_sha256": binding["derived_nomination_sha256"],
        "lineage_sha256": binding["lineage_sha256"],
        "native_prompt_sha256": owner["native_prompt_sha256"],
        "native_prompt_bytes": owner["native_prompt_bytes"],
        "parent_artifact": owner["parent_artifact"], "parent_sha256": owner["parent_sha256"],
        "policy_sha256": owner["policy_sha256"],
        "launch_policy_sha256": owner["launch_policy_sha256"],
        "owner_binding_sha256": owner_digest,
    }
    return all(type(verification.get(key)) is type(value) and verification.get(key) == value
               for key, value in expected.items())


def _valid_owner_binding(receipt):
    if type(receipt) is not dict or receipt.get("schema") != OWNER_BINDING_SCHEMA:
        return False
    try:
        if len(json.dumps(receipt, allow_nan=False).encode()) > MAX_OWNER_BINDING_BYTES:
            return False
    except (ValueError, TypeError, RecursionError, OverflowError):
        return False
    amount = receipt.get("native_prompt_bytes")
    return (
        _identity(receipt.get("binding_id")) and _task_identity(receipt.get("task_id"))
        and type(amount) is int and 1 <= amount <= MAX_NATIVE_PROMPT_BYTES
        and all(_digest(receipt.get(key)) for key in (
            "native_prompt_sha256", "parent_sha256", "launch_policy_sha256",
            "nomination_sha256", "policy_sha256"))
        and _artifact_path(receipt.get("parent_artifact"), absolute=False)
        and _artifact_path(receipt.get("launch_policy_artifact"), absolute=True)
        and _artifact_path(receipt.get("nomination_artifact"), absolute=True)
        and ("typed_nomination" not in receipt
             or _valid_typed_nomination_binding(receipt["typed_nomination"]))
        and receipt.get("provider_called") is False
        and receipt.get("completion_authority") is False
    )


def _owner_binding_key(receipt):
    return tuple(receipt[key] for key in (
        "task_id", "parent_sha256", "policy_sha256", "native_prompt_sha256",
        "native_prompt_bytes", "launch_policy_artifact", "launch_policy_sha256"))


def _dialogue_owner_key(dialogue):
    launch = dialogue.get("worker_launch")
    if type(launch) is not dict:
        return None
    amount = launch.get("original_native_prompt_bytes")
    if (not _task_identity(dialogue.get("task_id"))
            or not all(_digest(dialogue.get(key)) for key in ("parent_sha256", "policy_sha256"))
            or not _digest(launch.get("original_native_prompt_sha256"))
            or type(amount) is not int or not 1 <= amount <= MAX_NATIVE_PROMPT_BYTES
            or not _artifact_path(launch.get("launch_policy_artifact"), absolute=True)
            or not _digest(launch.get("launch_policy_sha256"))):
        return None
    return (dialogue["task_id"], dialogue["parent_sha256"], dialogue["policy_sha256"],
            launch["original_native_prompt_sha256"], amount,
            launch["launch_policy_artifact"], launch["launch_policy_sha256"])


def _owner_binding_coverage(report, dialogues, coverage):
    """Join owner observations to one dialogue each, without charging them.

    Independent task/nomination pins are required for the selected owner route.
    An orphan owner observation says nothing about whether a child dispatched
    before interruption, even though binding itself calls no provider.
    """
    expected = report.get("semantic_decision_owner_binding_expected") is True
    typed_mode_present = "semantic_decision_typed_nomination_expected" in report
    typed_expected = report.get("semantic_decision_typed_nomination_expected") is True
    if not expected and "semantic_decision_owner_bindings" not in report and not typed_expected:
        return
    coverage.update(owner_binding_expected=expected, owner_bindings_observed=0,
                    owner_bound_dialogues_observed=0, owner_bindings_without_dialogue_count=0,
                    owner_binding_coverage_complete=True)

    def issue(reason, **ordinals):
        if len(coverage["issues"]) < MAX_RECEIPTS:
            coverage["issues"].append({"reason": reason, **ordinals})
        coverage.update(owner_binding_coverage_complete=False,
                        authoritative_usage_coverage_complete=False,
                        all_attempts_receipted=False, provider_call_count_complete=False)

    if typed_mode_present:
        coverage.update(typed_nomination_expected=typed_expected,
                        typed_nomination_verifications_observed=0,
                        typed_nomination_verified_owner_bindings_observed=0)
        if type(report["semantic_decision_typed_nomination_expected"]) is not bool:
            issue("invalid_expected_typed_nomination_mode")
    if typed_expected and not expected:
        issue("typed_nomination_without_expected_owner_route")

    # This subtree comes from the trusted historical context observer. It is
    # deliberately outside collect_provider_usage_observation: printed rows
    # with this schema must never become verification evidence.
    verifications = []
    if typed_expected:
        audit = report.get("context_input_audit")
        population = audit.get("semantic_decision_inputs") if type(audit) is dict else None
        trusted_envelope = (
            type(audit) is dict and audit.get("schema") == "terminal-final-context-audit@1"
            and type(audit.get("provider_calls")) is int and audit["provider_calls"] == 0
            and audit.get("raw_prompts_exported") is False
            and audit.get("completion_authority") is False
            and type(population) is dict
            and population.get("schema") == "terminal-semantic-decision-input-population-audit@1"
            and type(population.get("provider_calls")) is int and population["provider_calls"] == 0
            and population.get("raw_prompts_exported") is False
            and population.get("completion_authority") is False
        )
        observed = population.get("typed_nomination_verifications") if trusted_envelope else None
        if type(observed) is not list or len(observed) > MAX_RECEIPTS:
            issue("typed_nomination_verification_population_unavailable")
        else:
            for ordinal, verification in enumerate(observed):
                if not _valid_typed_nomination_verification(verification):
                    issue("invalid_typed_nomination_verification", typed_verification_ordinal=ordinal)
                elif verification not in verifications:
                    verifications.append(verification)
            coverage["typed_nomination_verifications_observed"] = len(verifications)

    task_pin = report.get("semantic_decision_expected_task_id")
    nomination_pin = report.get("semantic_decision_nomination_sha256")
    valid_pins = _task_identity(task_pin) and _digest(nomination_pin)
    if expected and not valid_pins:
        issue("invalid_expected_owner_binding_pins")
    owners = report.get("semantic_decision_owner_bindings", [])
    if type(owners) is not list or len(owners) > MAX_RECEIPTS:
        issue("invalid_owner_binding_population")
        return
    if expected and not owners:
        issue("expected_owner_binding_receipt_missing")
    if expected and not dialogues:
        issue("expected_owner_bound_dialogue_missing")
    unique, seen_ids, conflicting_ids = [], {}, set()
    for ordinal, owner in enumerate(owners):
        if type(owner) is dict and _identity(owner.get("binding_id")):
            identity = owner["binding_id"]
            if identity in seen_ids:
                if owner == seen_ids[identity]:
                    continue
                conflicting_ids.add(identity)
                issue("owner_binding_identity_conflict", owner_binding_ordinal=ordinal)
            else:
                seen_ids[identity] = owner
        unique.append((ordinal, owner))
    coverage["owner_bindings_observed"] = len(unique)
    candidates, matched_verifications = {}, set()
    for ordinal, owner in unique:
        if not _valid_owner_binding(owner):
            issue("invalid_owner_binding_receipt", owner_binding_ordinal=ordinal)
            continue
        if owner["binding_id"] in conflicting_ids:
            continue
        if expected and (not valid_pins or owner["task_id"] != task_pin
                         or owner["nomination_sha256"] != nomination_pin):
            issue("owner_binding_expected_pin_conflict", owner_binding_ordinal=ordinal)
            continue
        has_typed_nomination = "typed_nomination" in owner
        if typed_expected:
            if not has_typed_nomination:
                issue("expected_typed_nomination_binding_missing", owner_binding_ordinal=ordinal)
                continue
            matching = [index for index, verification in enumerate(verifications)
                        if _typed_verification_matches_owner(verification, owner)
                        and verification["task_id"] == task_pin
                        and verification["origin_nomination_sha256"] == nomination_pin]
            if len(matching) != 1:
                issue("typed_nomination_verification_missing_or_ambiguous", owner_binding_ordinal=ordinal)
                continue
            if matching[0] in matched_verifications:
                issue("typed_nomination_verification_reused", owner_binding_ordinal=ordinal)
                continue
            matched_verifications.add(matching[0])
            coverage["typed_nomination_verified_owner_bindings_observed"] += 1
        elif has_typed_nomination:
            # Neither an opt-out command nor a legacy report authorized this
            # policy transformation. The original pin remains independently held.
            issue("unexpected_typed_nomination_binding", owner_binding_ordinal=ordinal)
            continue
        candidates.setdefault(_owner_binding_key(owner), []).append(ordinal)
    if typed_expected and len(matched_verifications) != len(verifications):
        issue("typed_nomination_verification_without_owner_binding")
    matched, seen_dialogues = set(), {}
    for ordinal, dialogue in enumerate(dialogues):
        if type(dialogue) is not dict or dialogue.get("schema") != DIALOGUE_SCHEMA:
            issue("unmatchable_dialogue_owner_binding", dialogue_ordinal=ordinal)
            continue
        session = dialogue.get("session_id")
        if _identity(session):
            if session in seen_dialogues and dialogue == seen_dialogues[session]:
                continue
            seen_dialogues.setdefault(session, dialogue)
        key = _dialogue_owner_key(dialogue)
        matching = candidates.get(key, []) if key is not None else []
        if not matching:
            issue("dialogue_owner_binding_missing_or_foreign", dialogue_ordinal=ordinal)
        elif len(matching) != 1:
            issue("ambiguous_dialogue_owner_binding", dialogue_ordinal=ordinal)
        elif matching[0] in matched:
            issue("owner_binding_reused_for_dialogues", dialogue_ordinal=ordinal)
        else:
            matched.add(matching[0])
            coverage["owner_bound_dialogues_observed"] += 1
    orphaned = {ordinal for ordinal, _ in unique} - matched
    coverage["owner_bindings_without_dialogue_count"] = len(orphaned)
    if orphaned:
        issue("owner_binding_without_dialogue")


def _native_receipts(dialogue):
    attempts = dialogue.get("attempts")
    if type(attempts) is not list or len(attempts) > MAX_RECEIPTS:
        return
    for attempt in attempts:
        provider = attempt.get("provider_receipt") if type(attempt) is dict else None
        native = provider.get("native_invocation") if type(provider) is dict else None
        if type(native) is dict and native.get("schema") == INVOCATION_SCHEMA:
            yield native


def _retain_native(rows, item):
    # Preserve changed evidence with the same identity. Only an exact repeated
    # observation may be removed; the collector's inferred phase is incidental.
    identity = item.get("invocation_id")
    phase = item.get("purpose") if item.get("purpose") in ("planning", "coding") else "coding"
    body = {key: value for key, value in item.items() if key != "phase"}
    prior_phases = set()
    for prior in rows:
        if type(prior) is not dict or prior.get("invocation_id") != identity:
            continue
        if {key: value for key, value in prior.items() if key != "phase"} == body:
            return
        prior_phase = prior.get("phase", prior.get("purpose"))
        if prior_phase in ("planning", "coding"):
            prior_phases.add(prior_phase)
    if (_identity(identity) and "purpose" not in item and "phase" not in item
            and len(prior_phases) == 1):
        phase = next(iter(prior_phases))
    rows.append({"phase": phase, **item})


def collect_provider_usage_observation(report: dict, item: dict) -> bool:
    """Collect a recognized log row, including failed dialogue attempts.

    No declared binding or success flag can discard a retained native receipt:
    invalid bindings with observed usage still represent possible charges.
    The independent coverage projection withholds exact totals for such rows.
    """
    if type(item) is not dict:
        return False
    if item.get("schema") == INVOCATION_SCHEMA:
        _retain_native(report.setdefault("provider_invocations", []), item)
        return True
    if item.get("schema") == OWNER_BINDING_SCHEMA:
        owners = report.setdefault("semantic_decision_owner_bindings", [])
        if item not in owners:
            owners.append(item)
        return True
    if item.get("schema") != DIALOGUE_SCHEMA:
        return False
    dialogues = report.setdefault("semantic_decision_dialogues", [])
    if item not in dialogues:
        dialogues.append(item)
    for native in _native_receipts(item):
        _retain_native(report.setdefault("provider_invocations", []), native)
    return True


def semantic_decision_accounting_inputs(report: dict) -> tuple[list, dict]:
    """Return retained native inputs and a bounded, secret-free coverage view.

    This is pure so final-report and Harbor accounting both derive results
    from the evidence, even if a stored summary or collector was bypassed.
    Grok CLI observations remain unsupported for authoritative usage totals.
    """
    supplied = report.get("provider_invocations", [])
    if type(supplied) is not list or len(supplied) > MAX_RECEIPTS:
        raise ValueError("bounded retained provider invocation list required")
    rows = list(supplied)
    coverage = {
        "schema": COVERAGE_SCHEMA, "dialogues_observed": 0, "attempts_observed": 0,
        "dialogue_expected": report.get("semantic_decision_dialogue_expected") is True,
        "reported_provider_dispatches": 0, "reported_local_refusals": 0,
        "unknown_dispatch_status_count": 0, "known_additional_provider_calls": 0,
        "provider_call_count_complete": True, "all_attempts_receipted": True,
        "authoritative_usage_coverage_complete": True, "issues": [],
        "provider_token_definition": "input_tokens + output_tokens",
        "local_prompt_estimates_added": False, "completion_authority": False,
    }
    dialogues = report.get("semantic_decision_dialogues", [])
    if type(dialogues) is not list or len(dialogues) > MAX_RECEIPTS:
        coverage["issues"].append({"reason": "invalid_dialogue_population"})
        coverage.update(authoritative_usage_coverage_complete=False,
                        all_attempts_receipted=False, provider_call_count_complete=False)
        _owner_binding_coverage(report, [], coverage)
        return rows, coverage
    if coverage["dialogue_expected"] and not dialogues:
        coverage["issues"].append({"reason": "expected_dialogue_receipt_missing"})
        coverage.update(authoritative_usage_coverage_complete=False,
                        all_attempts_receipted=False, provider_call_count_complete=False)

    seen_sessions, seen_calls, additional_calls, covered_native_ids = {}, {}, set(), set()
    for ordinal, dialogue in enumerate(dialogues):
        def issue(reason, attempt_ordinal=None, *, missing=False, count_unknown=False):
            entry = {"dialogue_ordinal": ordinal, "reason": reason}
            if attempt_ordinal is not None:
                entry["attempt_ordinal"] = attempt_ordinal
            if len(coverage["issues"]) < MAX_RECEIPTS:
                coverage["issues"].append(entry)
            coverage["authoritative_usage_coverage_complete"] = False
            if missing:
                coverage["all_attempts_receipted"] = False
            if count_unknown:
                coverage["provider_call_count_complete"] = False

        if type(dialogue) is not dict or dialogue.get("schema") != DIALOGUE_SCHEMA:
            issue("invalid_dialogue_receipt", missing=True, count_unknown=True)
            continue
        session = dialogue.get("session_id")
        if not _identity(session):
            issue("invalid_dialogue_identity", missing=True, count_unknown=True)
        elif session in seen_sessions:
            if dialogue == seen_sessions[session]:
                continue
            issue("dialogue_session_identity_conflict", count_unknown=True)
        else:
            seen_sessions[session] = dialogue
        coverage["dialogues_observed"] += 1
        attempts = dialogue.get("attempts")
        if type(attempts) is not list or len(attempts) > MAX_RECEIPTS:
            issue("invalid_dialogue_attempt_population", missing=True, count_unknown=True)
            continue
        if len(attempts) + coverage["attempts_observed"] > MAX_RECEIPTS:
            raise ValueError("aggregate retained dialogue attempt population exceeds bound")
        for native in _native_receipts(dialogue):
            _retain_native(rows, native)
            if _identity(native.get("invocation_id")):
                covered_native_ids.add(native["invocation_id"])
        declared = dialogue.get("invocation_attempt_count")
        if type(declared) is not int or declared != len(attempts):
            issue("dialogue_attempt_count_conflict", missing=True, count_unknown=True)
        for attempt_ordinal, attempt in enumerate(attempts):
            if type(attempt) is not dict:
                coverage["attempts_observed"] += 1
                issue("invalid_dialogue_attempt", attempt_ordinal, missing=True, count_unknown=True)
                continue
            call_id = attempt.get("call_id")
            if not _identity(call_id):
                issue("invalid_dialogue_call_identity", attempt_ordinal, count_unknown=True)
            elif call_id in seen_calls:
                if attempt == seen_calls[call_id]:
                    continue
                issue("dialogue_call_identity_conflict", attempt_ordinal, count_unknown=True)
            else:
                seen_calls[call_id] = attempt
            coverage["attempts_observed"] += 1
            receipt = attempt.get("provider_receipt")
            if type(receipt) is not dict:
                coverage["unknown_dispatch_status_count"] += 1
                issue("missing_dialogue_provider_receipt", attempt_ordinal, missing=True, count_unknown=True)
                continue
            bound = (attempt.get("provider_receipt_bound") is True
                     and receipt.get("schema") == PROVIDER_SCHEMA
                     and receipt.get("call_id") == call_id
                     and receipt.get("session_id") == session
                     and receipt.get("provider") == dialogue.get("provider")
                     and receipt.get("model") == dialogue.get("model")
                     and receipt.get("workspace") == dialogue.get("workspace")
                     and receipt.get("model_prompt_sha256") == attempt.get("model_prompt_sha256"))
            if not bound:
                issue("unbound_dialogue_provider_receipt", attempt_ordinal, missing=True, count_unknown=True)
            dispatched = receipt.get("provider_called")
            if type(dispatched) is not bool:
                coverage["unknown_dispatch_status_count"] += 1
                issue("unknown_dialogue_provider_dispatch", attempt_ordinal, missing=True, count_unknown=True)
                continue
            if not dispatched:
                coverage["reported_local_refusals"] += 1
                native = receipt.get("native_invocation")
                if type(native) is dict and native.get("schema") == INVOCATION_SCHEMA:
                    issue("dialogue_provider_dispatch_contradiction", attempt_ordinal,
                          missing=True, count_unknown=True)
                continue
            coverage["reported_provider_dispatches"] += 1
            native = receipt.get("native_invocation")
            if receipt.get("provider") != "codex_cli":
                issue("unsupported_dialogue_provider_usage", attempt_ordinal)
                if _identity(call_id):
                    additional_calls.add(call_id)
                continue
            if type(native) is not dict or native.get("schema") != INVOCATION_SCHEMA:
                issue("missing_dialogue_native_invocation", attempt_ordinal, missing=True, count_unknown=True)
                continue
            if not all(native.get(key) == receipt.get(key) for key in (
                    "provider", "model", "workspace", "model_prompt_sha256")):
                issue("dialogue_native_invocation_binding_conflict", attempt_ordinal,
                      missing=True, count_unknown=True)
            if not _identity(native.get("invocation_id")):
                issue("invalid_dialogue_native_invocation_identity", attempt_ordinal, count_unknown=True)
    coverage["known_additional_provider_calls"] = len(additional_calls)
    if coverage["dialogue_expected"]:
        for row in rows:
            if (type(row) is dict and row.get("schema") == INVOCATION_SCHEMA
                    and row.get("phase", row.get("purpose")) == "coding"
                    and (not _identity(row.get("invocation_id"))
                         or row.get("invocation_id") not in covered_native_ids)):
                if len(coverage["issues"]) < MAX_RECEIPTS:
                    coverage["issues"].append({"reason": "uncovered_dialogue_native_invocation"})
                coverage.update(authoritative_usage_coverage_complete=False,
                                all_attempts_receipted=False, provider_call_count_complete=False)
    if len(rows) > MAX_RECEIPTS:
        raise ValueError("aggregate retained provider receipt population exceeds bound")
    _owner_binding_coverage(report, dialogues, coverage)
    return rows, coverage
