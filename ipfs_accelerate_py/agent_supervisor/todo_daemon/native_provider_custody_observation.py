"""Closed observations of native custody checks; never settlement evidence."""

SCHEMA = "database-native-provider-custody-observation@1"
STAGE_REASONS = {
    "issuance": frozenset({"eligible", "process_did_not_return", "missing_process",
        "missing_attempt_authority", "not_nonzero_exit", "process_exit_mismatch",
        "provider_not_dispatched", "attempt_not_consumed", "implementation_exception",
        "outer_timeout", "protected_path_violation", "validation_attempted",
        "lifecycle_not_finalized"}),
    "native_exit": frozenset({"verified", "not_issued", "custody_not_established",
        "custody_not_quiesced", "custody_owner_changed", "custody_owner_unavailable",
        "subreaper_status_unavailable", "subreaper_disabled", "child_census_not_empty_or_unstable",
        "child_census_unavailable", "not_native_process", "attempt_authority_changed",
        "implementation_result_changed", "process_not_reaped", "returncode_changed",
        "process_group_present", "process_group_unreadable", "check_unavailable"}),
    "bridge_join": frozenset({"verified", "not_native_daemon", "native_exit_unavailable",
        "binding_changed", "finished_event_mismatch", "implementation_mismatch",
        "state_mismatch", "protected_path_not_rearmed", "check_unavailable",
        "directory_unavailable", "binding_unavailable", "projection_unverified",
        "events_unavailable", "state_unavailable", "rearm_check_unavailable"}),
    "capability_take": frozenset({"verified", "not_issued", "foreign_exception",
        "attempt_changed", "binding_changed", "directory_changed", "state_changed",
        "events_changed", "protected_path_not_rearmed", "check_unavailable",
        "attempt_check_unavailable", "binding_unavailable", "directory_unavailable",
        "state_unavailable", "events_unavailable", "rearm_check_unavailable", "projection_unverified"}),
}
_SUCCESS = {"issuance": "eligible", "native_exit": "verified",
            "bridge_join": "verified", "capability_take": "verified"}


def validate_native_provider_custody_observation(value):
    """Return a defensive closed copy, or None for any malformed observation."""
    if type(value) is not dict or set(value) != {
        "schema", "checks", "observation_only", "completion_authority",
        "retry_authority", "settlement_authority",
    }:
        return None
    if (value["schema"] != SCHEMA or value["observation_only"] is not True
            or any(value[name] is not False for name in (
                "completion_authority", "retry_authority", "settlement_authority"))):
        return None
    checks = value["checks"]
    if type(checks) is not list or not 1 <= len(checks) <= 4:
        return None
    copied, last = [], -1
    for check in checks:
        if type(check) is not dict or set(check) != {"stage", "status", "reason_code"}:
            return None
        stage, status, reason = (check[name] for name in ("stage", "status", "reason_code"))
        if (type(stage) is not str or stage not in STAGE_REASONS
                or type(status) is not str or status not in {"passed", "denied"}
                or type(reason) is not str or reason not in STAGE_REASONS[stage]):
            return None
        ordinal = tuple(STAGE_REASONS).index(stage)
        if ordinal <= last or (status == "passed") != (reason == _SUCCESS[stage]):
            return None
        last = ordinal
        copied.append(dict(check))
    return {**value, "checks": copied}


def append_native_provider_custody_check(value, stage, reason_code):
    """Append one checked stage; reject unknown/private fields rather than echoing them."""
    previous = validate_native_provider_custody_observation(value)
    if value is not None and previous is None:
        return None
    if (type(stage) is not str or stage not in STAGE_REASONS
            or type(reason_code) is not str or reason_code not in STAGE_REASONS[stage]):
        return None
    result = previous or {"schema": SCHEMA, "checks": [], "observation_only": True,
        "completion_authority": False, "retry_authority": False, "settlement_authority": False}
    result["checks"].append({"stage": stage,
        "status": "passed" if reason_code == _SUCCESS[stage] else "denied",
        "reason_code": reason_code})
    return validate_native_provider_custody_observation(result)
