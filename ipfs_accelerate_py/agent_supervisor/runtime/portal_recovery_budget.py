"""Conservative replay budget for source-qualified portal failure recovery.

This is only a budget check. The caller must verify the live sealed source and
retain the existing settlement, task CAS, coordination and provider checks.
"""
from collections.abc import Iterable, Mapping


def portal_recovery_budget_consumed(
    events: Iterable[Mapping[str, object]],
    *,
    settlement_id: str,
    accepted_source: Mapping[str, object] | None = None,
) -> bool:
    history = tuple(events)
    if not history:
        return False
    # Keep the original lifetime budget when no sealed source was verified.
    if accepted_source is None:
        return True
    keys = ("source_head", "source_tree")
    if any(
        type(accepted_source.get(key)) is not str
        or len(accepted_source[key]) != 40
        or any(ch not in "0123456789abcdef" for ch in accepted_source[key])
        for key in keys
    ) or not settlement_id:
        return True
    for event in history:
        if not isinstance(event, Mapping) or not event.get("settlement_id"):
            return True
        # A settled failure can never gain another allowance by restarting.
        if event["settlement_id"] == settlement_id:
            return True
        prior = event.get("accepted_recovery_source")
        if prior is not None:
            if not isinstance(prior, Mapping) or any(key not in prior for key in keys):
                return True
            if all(prior[key] == accepted_source[key] for key in keys):
                return True
    return False
