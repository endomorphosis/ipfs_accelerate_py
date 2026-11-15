"""Explicit owner selection for at most one continuation of ordinary coding."""

from __future__ import annotations

import hashlib
import json
import re

DOCTOR_CODING_POLICY = "doctor-coding-two-attempts@1"
PUBLIC_VALIDATION_POLICY = "doctor-coding-public-validation-two-attempts@1"
DOCTOR_CODING_POLICIES = (DOCTOR_CODING_POLICY, PUBLIC_VALIDATION_POLICY)


def validate_doctor_coding_policy(value, *, candidate_rounds):
    validate_candidate_rounds(candidate_rounds)
    if value is not None and (type(value) is not str or value not in DOCTOR_CODING_POLICIES
                              or candidate_rounds != 2):
        raise ValueError("Doctor coding policy requires its exact explicit two-attempt selection")
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def build_doctor_coding_policy(*, doctor_requirements, original_requirements_sha256,
                              deadline_monotonic_ns, policy_id=DOCTOR_CODING_POLICY,
                              public_validation_contract=None):
    """Bind an independent coding-attempt allocation, never a proposal budget increase."""
    from .doctor_residual_budget import validate_residual_budget
    validate_doctor_coding_policy(policy_id, candidate_rounds=2)
    if policy_id is None:
        raise ValueError("explicit Doctor coding policy required")
    value = doctor_requirements
    if (type(value) is not dict or value.get("schema") != "supervisor-owner-doctor-requirements@1"
            or type(deadline_monotonic_ns) is not int or deadline_monotonic_ns <= 0):
        raise ValueError("exact captured Doctor requirements and original deadline required")
    budget = value["residual_budget"]
    if type(budget) is not dict:
        raise ValueError("closed positive integer Doctor residual budget required")
    # The strict native helper also checks its duplicate capsule token fields.
    # Reconstruct only those redundant counters; no budget value is enlarged.
    validated = validate_residual_budget(dict(residual_budget=budget,
        token_budget={"residual_"+key: budget.get(key) for key in
            ("max_calls", "max_tokens", "max_rounds", "max_cost_units")}))
    for digest in (original_requirements_sha256, value["artifact_sha256"], value["advisory_sha256"],
                   value["native_prompt_sha256"], value["descriptor_sha256"]):
        if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("exact Doctor coding policy digests required")
    for key in ("context_cid", "native_capsule_id", "task_cid", "task_id", "prior_session_id"):
        if type(value.get(key)) is not str or not value[key]:
            raise ValueError("complete original Doctor coding policy identity required")
    result = dict(schema="supervisor-doctor-coding-policy@1", policy_id=DOCTOR_CODING_POLICY,
        max_child_attempts=2, max_outer_router_calls=2, deadline_monotonic_ns=deadline_monotonic_ns,
        original_requirements_sha256=original_requirements_sha256,
        doctor_artifact_sha256=value["artifact_sha256"], doctor_advisory_sha256=value["advisory_sha256"],
        doctor_context_cid=value["context_cid"], native_capsule_id=value["native_capsule_id"],
        task_cid=value["task_cid"], task_id=value["task_id"], native_prompt_sha256=value["native_prompt_sha256"],
        prior_session_id=value["prior_session_id"], prior_descriptor_sha256=value["descriptor_sha256"],
        proposal_budget=validated, proposal_budget_sha256=hashlib.sha256(_json(validated)).hexdigest(),
        unit="isolated_coding_attempt", provider_internal_calls_bounded=False,
        total_tokens_bounded=False, total_cost_bounded=False, original_deadline_preserved=True,
        requires_failed_retirement=True, scope_expansion_authority=False)
    if policy_id == PUBLIC_VALIDATION_POLICY:
        from .candidate_public_validation import validate_public_validation_contract_metadata
        contract = validate_public_validation_contract_metadata(public_validation_contract)
        if (contract["original_requirements_sha256"] != original_requirements_sha256
                or contract["deadline_monotonic_ns"] != deadline_monotonic_ns
                or contract["task_id"] != value["task_id"] or contract["task_cid"] != value["task_cid"]):
            raise ValueError("public validation contract differs from original Doctor task or allocation")
        result.pop("requires_failed_retirement")
        result.update(schema="supervisor-doctor-coding-policy@2", policy_id=policy_id,
            requires_retirement=True,
            continuation_reasons=["worker_exit_nonzero", "public_validation_failed"],
            public_validation_contract_sha256=hashlib.sha256(_json(contract)).hexdigest(),
            public_validation_contract_bytes=len(_json(contract)))
    elif public_validation_contract is not None:
        raise ValueError("legacy Doctor coding policy cannot acquire public validation")
    return result


def validate_doctor_coding_policy_binding(value, **bindings):
    expected = build_doctor_coding_policy(**bindings)
    if type(value) is not dict or _json(value) != _json(expected):
        raise ValueError("Doctor coding policy differs from captured constraints or original allocation")
    return expected


def validate_doctor_coding_policy_shape(value):
    """Closed historical shape only; callers still join it to retained native evidence."""
    if type(value) is not dict:
        raise ValueError("exact Doctor coding policy required")
    try:
        doctor = dict(schema="supervisor-owner-doctor-requirements@1",
            artifact_sha256=value["doctor_artifact_sha256"], advisory_sha256=value["doctor_advisory_sha256"],
            context_cid=value["doctor_context_cid"], native_capsule_id=value["native_capsule_id"],
            task_cid=value["task_cid"], task_id=value["task_id"], native_prompt_sha256=value["native_prompt_sha256"],
            prior_session_id=value["prior_session_id"], descriptor_sha256=value["prior_descriptor_sha256"],
            residual_budget=value["proposal_budget"])
        if value.get("policy_id") == PUBLIC_VALIDATION_POLICY:
            # This parser checks historical shape only. The live validator must
            # independently join the contract bytes to the selected policy.
            legacy = dict(value, schema="supervisor-doctor-coding-policy@1", policy_id=DOCTOR_CODING_POLICY,
                requires_failed_retirement=True)
            for key in ("requires_retirement", "continuation_reasons",
                        "public_validation_contract_sha256", "public_validation_contract_bytes"):
                legacy.pop(key, None)
            validate_doctor_coding_policy_shape(legacy)
            if (value.get("schema") != "supervisor-doctor-coding-policy@2"
                    or value.get("requires_retirement") is not True
                    or value.get("continuation_reasons") != ["worker_exit_nonzero", "public_validation_failed"]
                    or type(value.get("public_validation_contract_bytes")) is not int
                    or not 0 < value["public_validation_contract_bytes"] <= 65536
                    or type(value.get("public_validation_contract_sha256")) is not str
                    or re.fullmatch(r"[0-9a-f]{64}", value["public_validation_contract_sha256"]) is None
                    or set(value) != (set(legacy)-{"requires_failed_retirement"}) | {
                        "requires_retirement", "continuation_reasons", "public_validation_contract_sha256",
                        "public_validation_contract_bytes"}):
                raise ValueError("closed public-validation Doctor policy required")
            return json.loads(_json(value))
        return validate_doctor_coding_policy_binding(value, doctor_requirements=doctor,
            original_requirements_sha256=value["original_requirements_sha256"],
            deadline_monotonic_ns=value["deadline_monotonic_ns"])
    except (KeyError, TypeError) as error:
        raise ValueError("complete Doctor coding policy binding required") from error


def split_owner_coding_controls(argv):
    """Strip both owner controls without changing the worker's argument grammar."""
    if type(argv) is not list or any(type(arg) is not str for arg in argv):
        raise ValueError("literal owner argv required")
    policy, result, index = None, [], 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--doctor-coding-policy":
            index += 1
            if index == len(argv):
                raise ValueError("Doctor coding policy value required")
            selected = argv[index]
        elif arg.startswith("--doctor-coding-policy="):
            selected = arg.split("=", 1)[1]
        else:
            result.append(arg)
            index += 1
            continue
        if policy is not None or selected not in DOCTOR_CODING_POLICIES:
            raise ValueError("one exact Doctor coding policy selection required")
        policy = selected
        index += 1
    rounds, result = split_owner_candidate_rounds(result)
    validate_doctor_coding_policy(policy, candidate_rounds=rounds)
    return rounds, policy, result


def validate_candidate_rounds(value):
    if type(value) is not int or value not in (1, 2):
        raise ValueError("candidate rounds must be exactly one or two")
    return value


def split_owner_candidate_rounds(argv):
    """Remove the owner-only control before preparing the original worker.

    Duplicates and incomplete selections refuse rather than change which
    interpretation the owner and the child see. An absent flag retains one
    round; this function does not admit a continuation.
    """
    if type(argv) is not list or any(type(arg) is not str for arg in argv):
        raise ValueError("literal owner argv required")
    result, selected = [], None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--candidate-rounds":
            index += 1
            if index == len(argv):
                raise ValueError("candidate rounds value required")
            value = argv[index]
        elif arg.startswith("--candidate-rounds="):
            value = arg.split("=", 1)[1]
        else:
            result.append(arg)
            index += 1
            continue
        if selected is not None or value not in ("1", "2"):
            raise ValueError("one exact candidate rounds selection required")
        selected = int(value)
        index += 1
    return 1 if selected is None else selected, result
