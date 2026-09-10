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
    keys = ("source_head", "source_tree")
    def valid_source(source):
        return isinstance(source, Mapping) and all(
            type(source.get(key)) is str and len(source[key]) == 40
            and all(ch in "0123456789abcdef" for ch in source[key])
            for key in keys
        )
    history = tuple(events)
    if accepted_source is not None and (not valid_source(accepted_source) or not settlement_id):
        return True
    if not history:
        return False
    # Keep the original lifetime budget when no sealed source was verified.
    if accepted_source is None:
        return True
    for event in history:
        if not isinstance(event, Mapping) or not event.get("settlement_id"):
            return True
        # A settled failure can never gain another allowance by restarting.
        if event["settlement_id"] == settlement_id:
            return True
        prior = event.get("accepted_recovery_source")
        if prior is not None:
            if not valid_source(prior):
                return True
            if all(prior[key] == accepted_source[key] for key in keys):
                return True
    return False
