"""P0 wire ownership and boundary validation for task families.

P0 helpers reject inconsistent boundaries, memberships, and already-known
counterexamples in immutable task-family contracts.  PCPC-010/PCPC-011 may
add discovery and live boundary validators in this same module.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import (
    FamilyMembershipClass,
    ProcedureContractError,
    TaskFamily,
    TaskFamilyBoundary,
    TaskFamilyCounterexample,
    TaskFamilyMembership,
)


class TaskFamilyContractError(ProcedureContractError):
    """A task-family wire artifact violates its declared safe boundary."""


def validate_task_family_membership(
    membership: TaskFamilyMembership,
    family: TaskFamily,
) -> TaskFamilyMembership:
    """Require membership class to agree with the exact declared example set."""

    if not isinstance(membership, TaskFamilyMembership) or not isinstance(family, TaskFamily):
        raise TaskFamilyContractError("membership and family must use typed contracts")
    if membership.bindings != family.bindings:
        raise TaskFamilyContractError("membership and family exact bindings differ")
    if membership.task_family_cid != family.content_id:
        raise TaskFamilyContractError("membership does not bind the exact task-family CID")
    boundary = family.boundary
    expected = {
        FamilyMembershipClass.POSITIVE: set(boundary.positive_member_cids),
        FamilyMembershipClass.NEGATIVE: set(boundary.negative_example_cids),
        FamilyMembershipClass.BOUNDARY: set(boundary.boundary_example_cids),
        FamilyMembershipClass.UNKNOWN: set(boundary.unknown_case_cids),
    }[membership.membership]
    if membership.trajectory_cid not in expected:
        raise TaskFamilyContractError("membership class contradicts the declared boundary")
    return membership


def validate_task_family_contract(
    family: TaskFamily,
    *,
    counterexamples: Sequence[TaskFamilyCounterexample] = (),
) -> TaskFamily:
    """Reject a family invalidated by a known authority/effect/validation split."""

    if not isinstance(family, TaskFamily):
        raise TaskFamilyContractError("family must be TaskFamily")
    if not isinstance(counterexamples, Sequence) or isinstance(
        counterexamples, (str, bytes, bytearray, memoryview)
    ):
        raise TaskFamilyContractError("counterexamples must be a bounded sequence")
    if len(counterexamples) > 128:
        raise TaskFamilyContractError("counterexamples exceeds its item bound")
    for counterexample in counterexamples:
        if not isinstance(counterexample, TaskFamilyCounterexample):
            raise TaskFamilyContractError("counterexamples must be typed contracts")
        if counterexample.bindings != family.bindings:
            raise TaskFamilyContractError("counterexample exact bindings differ")
        if counterexample.task_family_cid != family.content_id:
            raise TaskFamilyContractError("counterexample does not bind the exact family CID")
        if (
            counterexample.conflicting_authority_classes
            or counterexample.conflicting_effect_classes
            or counterexample.conflicting_validation_classes
        ):
            raise TaskFamilyContractError(
                "known counterexample materially splits authority, effects, or validation"
            )
        raise TaskFamilyContractError("known counterexample invalidates the family boundary")
    return family


def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TaskFamilyContractError("task-family JSON contains a duplicate field")
        result[key] = value
    return result


def _reject_float(_: str) -> Any:
    raise TaskFamilyContractError("task-family JSON cannot contain floating point values")


def _decode_json(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            value = bytes(value).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise TaskFamilyContractError("task-family bytes must be UTF-8") from exc
    if isinstance(value, str):
        try:
            return json.loads(
                value,
                object_pairs_hook=_closed_object,
                parse_float=_reject_float,
                parse_constant=_reject_float,
            )
        except json.JSONDecodeError as exc:
            raise TaskFamilyContractError("task-family JSON is malformed") from exc
    return value


def parse_task_family(value: Any) -> TaskFamily:
    if isinstance(value, TaskFamily):
        return validate_task_family_contract(value)
    value = _decode_json(value)
    if not isinstance(value, Mapping):
        raise TaskFamilyContractError("task family must be a mapping or JSON object")
    return validate_task_family_contract(TaskFamily.from_dict(value))


def parse_task_family_membership(
    value: Any,
    family: TaskFamily,
) -> TaskFamilyMembership:
    if not isinstance(value, TaskFamilyMembership):
        value = _decode_json(value)
        if not isinstance(value, Mapping):
            raise TaskFamilyContractError("membership must be a mapping or JSON object")
        value = TaskFamilyMembership.from_dict(value)
    return validate_task_family_membership(value, family)


__all__ = [
    "FamilyMembershipClass",
    "TaskFamily",
    "TaskFamilyBoundary",
    "TaskFamilyContractError",
    "TaskFamilyCounterexample",
    "TaskFamilyMembership",
    "parse_task_family",
    "parse_task_family_membership",
    "validate_task_family_contract",
    "validate_task_family_membership",
]
