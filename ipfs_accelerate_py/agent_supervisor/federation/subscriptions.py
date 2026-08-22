"""Bounded, SQL-free event subscription matching."""

# ``typing.Callable`` keeps the evaluated selector alias importable on Python
# 3.8, where ``collections.abc.Callable`` was not yet subscriptable.
# ruff: noqa: UP035

from __future__ import annotations

from typing import Callable

from .contracts import FederationAuthorityError, FederationContractError
from .events import (
    DomainEvent,
    EventClass,
    EventSelector,
    EventSubscription,
    SelectorKind,
    SubscriptionState,
)


class SubscriptionError(FederationContractError):
    """Base subscription validation error."""


class CausalSelectorUnavailable(SubscriptionError):
    """Causal selectors were requested before an exact graph is admitted."""


CausalSelectorEvaluator = Callable[[EventSelector, DomainEvent], bool]


_EVENT_FIELD = {
    SelectorKind.REPOSITORY: "repository_id",
    SelectorKind.TREE: "tree_id",
    SelectorKind.GOAL: "goal_id",
    SelectorKind.SUBGOAL: "subgoal_id",
    SelectorKind.TASK: "task_id",
    SelectorKind.SYMBOL: "symbol_id",
    SelectorKind.CONTRACT: "contract_id",
    SelectorKind.PROOF_OBLIGATION: "proof_obligation_id",
    SelectorKind.SUPERVISOR: "supervisor_id",
    SelectorKind.RESOURCE_CLASS: "resource_class",
}


def event_matches_subscription(
    event: DomainEvent,
    subscription: EventSubscription,
    *,
    causal_selector_evaluator: CausalSelectorEvaluator | None = None,
) -> bool:
    """Match one event against a closed selector contract.

    Selectors of the same kind are ORed; different kinds are ANDed.  No SQL or
    caller-provided predicate is accepted.  Causal selectors require an
    injected exact/admitted-conservative evaluator from the later causal
    tranche; otherwise they fail closed.
    """

    if subscription.state is not SubscriptionState.ACTIVE:
        return False
    if event.tenant_id != subscription.tenant_id:
        return False
    if event.federation_id != subscription.federation_id:
        return False
    if event.event_type not in subscription.event_classes:
        return False

    grouped: dict[SelectorKind, list[EventSelector]] = {}
    for selector in subscription.selectors:
        grouped.setdefault(selector.kind, []).append(selector)
    for kind, selectors in grouped.items():
        if kind is SelectorKind.EVENT_CLASS:
            if not any(event.event_type is EventClass(item.value) for item in selectors):
                return False
            continue
        if kind in {SelectorKind.CAUSAL_ANCESTOR, SelectorKind.CAUSAL_DESCENDANT}:
            if causal_selector_evaluator is None:
                raise CausalSelectorUnavailable(
                    "causal selector requires an admitted exact causal graph"
                )
            if not any(causal_selector_evaluator(item, event) for item in selectors):
                return False
            continue
        field = _EVENT_FIELD.get(kind)
        if field is None:
            raise SubscriptionError(f"selector kind {kind.value} has no closed matcher")
        observed = str(getattr(event, field))
        if not any(observed == item.value for item in selectors):
            return False
    return True


def assert_subscription_scope(
    subscription: EventSubscription,
    *,
    tenant_id: str,
    federation_id: str,
    consumer_id: str,
) -> None:
    if subscription.tenant_id != tenant_id:
        raise FederationAuthorityError("subscription tenant scope differs")
    if subscription.federation_id != federation_id:
        raise FederationAuthorityError("subscription federation scope differs")
    if subscription.consumer_id != consumer_id:
        raise FederationAuthorityError("subscription consumer scope differs")


__all__ = [
    "CausalSelectorEvaluator",
    "CausalSelectorUnavailable",
    "SubscriptionError",
    "assert_subscription_scope",
    "event_matches_subscription",
]
