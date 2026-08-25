"""Typed goal contracts compiled from handoff objectives (EAAEF-070)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


GOAL_CONTRACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-goal-contract@1"
)
GOAL_CONTRACT_INTERFACE: Final[str] = "ExternalGoalContract@1"


class GoalContractError(ValueError):
    """Malformed or unsafe goal contract."""


def _text(value: object, name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise GoalContractError(f"{name} is required")
    return text


def _tuple_text(values: object, name: str) -> tuple[str, ...]:
    if values is None:
        items: Sequence[object] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Sequence):
        items = values
    else:
        raise GoalContractError(f"{name} must be a list of strings")
    result = tuple(_text(item, name, required=True) for item in items)
    if len(set(result)) != len(result):
        raise GoalContractError(f"{name} contains duplicates")
    return result


@dataclass(frozen=True)
class ExternalGoalContract:
    """Desired/prohibited outcomes, scope, budgets, authority and evidence."""

    objective_id: str
    desired_outcomes: tuple[str, ...]
    prohibited_outcomes: tuple[str, ...]
    write_scope: tuple[str, ...]
    authority_ceiling: str
    verification_requirements: tuple[str, ...]
    proof_requirements: tuple[str, ...]
    review_requirements: tuple[str, ...]
    completion_evidence: tuple[str, ...]
    timeout_seconds: int
    cpu_millicores: int
    ram_mib: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective_id", _text(self.objective_id, "objective_id"))
        object.__setattr__(
            self, "desired_outcomes", _tuple_text(self.desired_outcomes, "desired_outcomes")
        )
        object.__setattr__(
            self,
            "prohibited_outcomes",
            _tuple_text(self.prohibited_outcomes, "prohibited_outcomes"),
        )
        object.__setattr__(self, "write_scope", _tuple_text(self.write_scope, "write_scope"))
        object.__setattr__(
            self, "authority_ceiling", _text(self.authority_ceiling, "authority_ceiling")
        )
        if self.authority_ceiling in {"unbounded", "root", "self_approve"}:
            raise GoalContractError("authority ceiling cannot be unbounded or self-granted")
        if not self.desired_outcomes:
            raise GoalContractError("desired_outcomes is required")
        overlap = set(self.desired_outcomes).intersection(self.prohibited_outcomes)
        if overlap:
            raise GoalContractError("desired and prohibited outcomes overlap")
        for field in (
            "verification_requirements",
            "proof_requirements",
            "review_requirements",
            "completion_evidence",
        ):
            object.__setattr__(self, field, _tuple_text(getattr(self, field), field))
        if not self.verification_requirements or not self.completion_evidence:
            raise GoalContractError("verification and completion evidence are required")
        for numeric, name in (
            (self.timeout_seconds, "timeout_seconds"),
            (self.cpu_millicores, "cpu_millicores"),
            (self.ram_mib, "ram_mib"),
        ):
            if int(numeric) <= 0:
                raise GoalContractError(f"{name} must be positive")

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": GOAL_CONTRACT_SCHEMA,
                "interface": GOAL_CONTRACT_INTERFACE,
                "objective_id": self.objective_id,
                "desired_outcomes": list(self.desired_outcomes),
                "prohibited_outcomes": list(self.prohibited_outcomes),
                "write_scope": list(self.write_scope),
                "authority_ceiling": self.authority_ceiling,
                "verification_requirements": list(self.verification_requirements),
                "proof_requirements": list(self.proof_requirements),
                "review_requirements": list(self.review_requirements),
                "completion_evidence": list(self.completion_evidence),
                "timeout_seconds": int(self.timeout_seconds),
                "cpu_millicores": int(self.cpu_millicores),
                "ram_mib": int(self.ram_mib),
            }
        )

    @property
    def content_id(self) -> str:
        return content_identity(dict(self.to_dict()))

    @classmethod
    def compile(cls, objective: Mapping[str, Any]) -> "ExternalGoalContract":
        if not isinstance(objective, Mapping):
            raise GoalContractError("objective must be an object")
        if objective.get("self_granted_authority"):
            raise GoalContractError("goals cannot self-grant authority")
        return cls(
            objective_id=str(objective.get("objective_id") or ""),
            desired_outcomes=objective.get("desired_outcomes") or (),
            prohibited_outcomes=objective.get("prohibited_outcomes") or (),
            write_scope=objective.get("write_scope") or (),
            authority_ceiling=str(objective.get("authority_ceiling") or ""),
            verification_requirements=objective.get("verification_requirements") or (),
            proof_requirements=objective.get("proof_requirements") or (),
            review_requirements=objective.get("review_requirements") or (),
            completion_evidence=objective.get("completion_evidence") or (),
            timeout_seconds=int(objective.get("timeout_seconds") or 0),
            cpu_millicores=int(objective.get("cpu_millicores") or 0),
            ram_mib=int(objective.get("ram_mib") or 0),
        )
