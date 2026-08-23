"""Allowlisted supervisor skills: typed operations, never arbitrary commands.

``SupervisorSkillRegistry@1`` admits :class:`SupervisorSkill` programs whose
steps are closed :class:`MetaAction` values.  Each step is admitted on its
own schema/effect/scope tuple.  Failure rolls back already-applied steps or
takes the skill fallback.  Skills cannot expand authority, spawn a shell, or
execute unsigned/forged programs.

Cold import performs no filesystem, network, or provider action.  Existing
operation implementations remain the canonical services; this registry only
assembles and bounds them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    CONTRACT_VERSION,
    content_identity,
)
from .contracts import (
    MAX_IDENTIFIER_BYTES,
    MAX_MAPPING_ITEMS,
    MAX_SEQUENCE_ITEMS,
    AutonomyContractError,
    MetaAction,
    RiskClass,
    SupervisorSkill,
)

SUPERVISOR_SKILL_REGISTRY_INTERFACE: Final[str] = "SupervisorSkillRegistry@1"
SUPERVISOR_SKILL_INTERFACE: Final[str] = "SupervisorSkill@1"
SUPERVISOR_SKILL_REGISTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/supervisor-skill-registry@1"
)
SKILL_ADMISSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/skill-step-admission@1"
)
SKILL_EXECUTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/skill-execution-receipt@1"
)

ALLOWLISTED_OPERATIONS: Final[frozenset[MetaAction]] = frozenset(MetaAction)
FORBIDDEN_SKILL_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "bash",
        "cmd.exe",
        "executable_code",
        "os.system",
        "powershell",
        "shell",
        "shell_command",
        "sh -c",
        "subprocess",
        "/bin/sh",
        "/bin/bash",
    }
)


class SupervisorSkillError(AutonomyContractError):
    """Raised when a skill is forged, out of domain, or cannot be admitted."""


class SkillStepStatus(str, Enum):
    ADMITTED = "admitted"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    SKIPPED_FALLBACK = "skipped_fallback"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SkillStepAdmission:
    SCHEMA: ClassVar[str] = SKILL_ADMISSION_SCHEMA
    step: MetaAction
    status: SkillStepStatus
    reason: str = ""

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": self.SCHEMA,
                "step": self.step.value,
                "status": self.status.value,
                "reason": self.reason,
            }
        )


@dataclass(frozen=True)
class SkillExecutionReceipt:
    SCHEMA: ClassVar[str] = SKILL_EXECUTION_SCHEMA
    skill_id: str
    status: str
    applied_steps: tuple[MetaAction, ...]
    rolled_back_steps: tuple[MetaAction, ...]
    fallback: MetaAction | None
    admissions: tuple[SkillStepAdmission, ...]
    cancelled: bool = False

    def to_dict(self) -> Mapping[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "skill_id": self.skill_id,
            "status": self.status,
            "applied_steps": [item.value for item in self.applied_steps],
            "rolled_back_steps": [item.value for item in self.rolled_back_steps],
            "fallback": None if self.fallback is None else self.fallback.value,
            "admissions": [dict(item.to_dict()) for item in self.admissions],
            "cancelled": self.cancelled,
        }
        return MappingProxyType(
            {**payload, "receipt_id": content_identity(payload)}
        )


def _identifier(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text or len(text.encode("utf-8")) > MAX_IDENTIFIER_BYTES:
        raise SupervisorSkillError(f"{name} must be a bounded identifier")
    lowered = text.lower()
    for marker in FORBIDDEN_SKILL_MARKERS:
        if marker in lowered:
            raise SupervisorSkillError(f"{name} contains a forbidden command marker")
    return text


def _reject_forbidden_payload(payload: Mapping[str, Any], noun: str) -> None:
    if len(payload) > MAX_MAPPING_ITEMS:
        raise SupervisorSkillError(f"{noun} contains too many fields")
    encoded = str(payload).lower()
    for marker in FORBIDDEN_SKILL_MARKERS:
        if marker in encoded:
            raise SupervisorSkillError(
                f"{noun} contains a forbidden shell or executable marker"
            )


def _coerce_skill(value: Any) -> SupervisorSkill:
    if isinstance(value, SupervisorSkill):
        _reject_forbidden_payload(value.to_dict(), "skill")
        return value
    if not isinstance(value, Mapping):
        raise SupervisorSkillError("skill must be a SupervisorSkill or mapping")
    _reject_forbidden_payload(value, "skill")
    extra = set(value) - {
        "schema",
        "contract_version",
        "content_id",
        "skill_id",
        "version",
        "precondition_ids",
        "input_schema_id",
        "effect_class",
        "steps",
        "postcondition_ids",
        "validation_ids",
        "rollback_action_ids",
        "fallback",
        "scope_paths",
        "scope_symbols",
        "risk_class",
    }
    if extra:
        raise SupervisorSkillError("forged skill contains unsigned extra fields")
    try:
        skill = SupervisorSkill.from_dict(value)
    except AutonomyContractError as exc:
        raise SupervisorSkillError(str(exc)) from exc
    claimed = value.get("skill_id") or value.get("content_id")
    if claimed and str(claimed) != skill.skill_id:
        raise SupervisorSkillError("forged skill identity does not match content")
    return skill


def _in_scope(path: str, allowed: Sequence[str]) -> bool:
    relative = path.strip().replace("\\", "/").strip("/")
    if not relative or relative.startswith("..") or "/../" in f"/{relative}/":
        return False
    for prefix in allowed:
        root = str(prefix).strip().replace("\\", "/").strip("/")
        if relative == root or relative.startswith(root + "/"):
            return True
    return False


@dataclass
class SupervisorSkillRegistry:
    """Bounded registry of allowlisted supervisor skills."""

    INTERFACE: ClassVar[str] = SUPERVISOR_SKILL_REGISTRY_INTERFACE
    SCHEMA: ClassVar[str] = SUPERVISOR_SKILL_REGISTRY_SCHEMA

    _lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _skills: dict[str, SupervisorSkill] = field(default_factory=dict, init=False)
    _operations: Callable[[MetaAction, Mapping[str, Any]], Mapping[str, Any]] | None = (
        None
    )

    def register(self, skill: SupervisorSkill | Mapping[str, Any]) -> SupervisorSkill:
        admitted = _coerce_skill(skill)
        for step in admitted.steps:
            if step not in ALLOWLISTED_OPERATIONS:
                raise SupervisorSkillError("skill step is not an allowlisted operation")
        if admitted.fallback not in ALLOWLISTED_OPERATIONS:
            raise SupervisorSkillError("skill fallback is not an allowlisted operation")
        if admitted.risk_class.rank >= RiskClass.R4_SECURITY_OR_PROTOCOL_SENSITIVE.rank:
            raise SupervisorSkillError("skills cannot expand into sensitive authority")
        with self._lock:
            current = self._skills.get(admitted.skill_id)
            if current is not None and current != admitted:
                raise SupervisorSkillError("skill identity already registered")
            self._skills[admitted.skill_id] = admitted
        return admitted

    def get(self, skill_id: str) -> SupervisorSkill | None:
        identifier = _identifier(skill_id, "skill_id")
        with self._lock:
            return self._skills.get(identifier)

    def skills(self) -> tuple[SupervisorSkill, ...]:
        with self._lock:
            return tuple(self._skills.values())

    def admit_step(
        self,
        skill: SupervisorSkill,
        step: MetaAction,
        *,
        allowed_paths: Sequence[str],
        admitted_preconditions: Sequence[str],
        cancelled: bool = False,
    ) -> SkillStepAdmission:
        if cancelled:
            return SkillStepAdmission(
                step=step,
                status=SkillStepStatus.CANCELLED,
                reason="skill execution cancelled",
            )
        if step not in ALLOWLISTED_OPERATIONS:
            return SkillStepAdmission(
                step=step,
                status=SkillStepStatus.REJECTED,
                reason="operation is not allowlisted",
            )
        if step not in skill.steps and step != skill.fallback:
            return SkillStepAdmission(
                step=step,
                status=SkillStepStatus.REJECTED,
                reason="operation is outside the skill program",
            )
        missing = [
            item
            for item in skill.precondition_ids
            if item not in set(admitted_preconditions)
        ]
        if missing:
            return SkillStepAdmission(
                step=step,
                status=SkillStepStatus.REJECTED,
                reason="preconditions are not independently admitted",
            )
        if skill.scope_paths and not all(
            _in_scope(path, allowed_paths) for path in skill.scope_paths
        ):
            return SkillStepAdmission(
                step=step,
                status=SkillStepStatus.REJECTED,
                reason="skill scope is outside the admitted envelope",
            )
        return SkillStepAdmission(step=step, status=SkillStepStatus.ADMITTED)

    def execute(
        self,
        skill_id: str,
        *,
        allowed_paths: Sequence[str],
        admitted_preconditions: Sequence[str],
        cancelled: bool = False,
        fail_step: MetaAction | None = None,
        operation: Callable[[MetaAction, Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
    ) -> SkillExecutionReceipt:
        skill = self.get(skill_id)
        if skill is None:
            raise SupervisorSkillError("skill is not registered")
        runner = operation or self._operations
        admissions: list[SkillStepAdmission] = []
        applied: list[MetaAction] = []
        rolled_back: list[MetaAction] = []
        fallback: MetaAction | None = None

        def rollback() -> None:
            for action_id in reversed(skill.rollback_action_ids):
                rolled_back.append(MetaAction.NO_OP)
                admissions.append(
                    SkillStepAdmission(
                        step=MetaAction.NO_OP,
                        status=SkillStepStatus.ROLLED_BACK,
                        reason=action_id,
                    )
                )
            while applied:
                applied.pop()

        for step in skill.steps:
            admission = self.admit_step(
                skill,
                step,
                allowed_paths=allowed_paths,
                admitted_preconditions=admitted_preconditions,
                cancelled=cancelled,
            )
            admissions.append(admission)
            if admission.status is SkillStepStatus.CANCELLED:
                rollback()
                return SkillExecutionReceipt(
                    skill_id=skill.skill_id,
                    status="cancelled",
                    applied_steps=tuple(applied),
                    rolled_back_steps=tuple(rolled_back),
                    fallback=skill.fallback,
                    admissions=tuple(admissions),
                    cancelled=True,
                )
            if admission.status is SkillStepStatus.REJECTED:
                rollback()
                fallback = skill.fallback
                admissions.append(
                    SkillStepAdmission(
                        step=skill.fallback,
                        status=SkillStepStatus.SKIPPED_FALLBACK,
                        reason=admission.reason,
                    )
                )
                return SkillExecutionReceipt(
                    skill_id=skill.skill_id,
                    status="fallback",
                    applied_steps=(),
                    rolled_back_steps=tuple(rolled_back),
                    fallback=fallback,
                    admissions=tuple(admissions),
                )
            if fail_step is step:
                rollback()
                fallback = skill.fallback
                admissions.append(
                    SkillStepAdmission(
                        step=skill.fallback,
                        status=SkillStepStatus.SKIPPED_FALLBACK,
                        reason="typed operation failed",
                    )
                )
                return SkillExecutionReceipt(
                    skill_id=skill.skill_id,
                    status="rolled_back",
                    applied_steps=(),
                    rolled_back_steps=tuple(rolled_back),
                    fallback=fallback,
                    admissions=tuple(admissions),
                )
            if runner is not None:
                result = runner(step, {"skill_id": skill.skill_id})
                if not isinstance(result, Mapping):
                    raise SupervisorSkillError("operation result must be a mapping")
            applied.append(step)
            admissions[-1] = SkillStepAdmission(
                step=step,
                status=SkillStepStatus.APPLIED,
            )
        return SkillExecutionReceipt(
            skill_id=skill.skill_id,
            status="succeeded",
            applied_steps=tuple(applied),
            rolled_back_steps=(),
            fallback=None,
            admissions=tuple(admissions),
        )
