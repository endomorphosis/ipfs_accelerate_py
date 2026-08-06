"""Safe rollout gate and public controls for prompt bootstrap and rescue.

The rollout report is evidence, never action or completion authority.
``automatic`` requires a qualifying observation and a distinct, later
current-root observation.  Binding, safety, or metric regression returns only
the affected behavior to ``shadow``.

Python, CLI, and MCP adapters below all decode the same canonical request and
invoke one service.  Import and discovery are provider-free and side-effect
free.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any, Final

from .prompt_workflow_benchmark import (
    PromptWorkflowBenchmark,
    PromptWorkflowBenchmarkError,
    PromptWorkflowGateReport,
    recompute_prompt_workflow_gate,
)


PROMPT_WORKFLOW_ROLLOUT_VERSION: Final = 1
PROMPT_WORKFLOW_ROLLOUT_EVALUATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/prompt-workflow-rollout-evaluation@1"
)
PROMPT_WORKFLOW_ROLLOUT_DECISION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/prompt-workflow-rollout-decision@1"
)
PROMPT_WORKFLOW_CONTROL_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/prompt-workflow-control-request@1"
)
PROMPT_WORKFLOW_CONTROL_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/prompt-workflow-control-result@1"
)
PROMPT_WORKFLOW_ROLLOUT_REQUIREMENT_ID: Final = (
    "asi-159:prompt-workflow-bootstrap-rescue-rollout"
)
PROMPT_WORKFLOW_BEHAVIOR_ID: Final = (
    "behavior:prompt-workflow-bootstrap-rescue@1"
)


class PromptWorkflowRolloutError(ValueError):
    """Rollout evidence, policy, or control input is invalid."""


class PromptWorkflowRolloutMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ASSIST = "assist"
    AUTOMATIC = "automatic"


class PromptWorkflowControlAction(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ASSIST = "assist"
    AUTOMATIC = "automatic"
    STATUS = "status"
    EXPLANATION = "explanation"
    ROLLBACK = "rollback"

    @property
    def requested_mode(self) -> PromptWorkflowRolloutMode | None:
        try:
            return PromptWorkflowRolloutMode(self.value)
        except ValueError:
            return None


class PromptWorkflowControlSurface(str, Enum):
    PYTHON = "python"
    CLI = "cli"
    MCP = "mcp"


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _plain(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PromptWorkflowRolloutError(
            "rollout data must be canonical JSON"
        ) from exc


def _identity(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_json(value: str | bytes | bytearray, name: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise PromptWorkflowRolloutError(
                    f"{name} contains duplicate JSON key {key!r}"
                )
            result[key] = item
        return result

    try:
        if isinstance(value, (bytes, bytearray)):
            value = bytes(value).decode("utf-8")
        if not isinstance(value, str):
            raise PromptWorkflowRolloutError(f"{name} must be JSON text")
        return json.loads(value, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromptWorkflowRolloutError(f"{name} is invalid JSON") from exc


def _text(value: Any, name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PromptWorkflowRolloutError(
            f"{name} must be non-empty canonical text"
        )
    if "\x00" in value or len(value.encode("utf-8")) > maximum:
        raise PromptWorkflowRolloutError(f"{name} is unsafe or too large")
    return value


def _timestamp(value: datetime | str, name: str) -> str:
    if isinstance(value, datetime):
        selected = value
    elif isinstance(value, str):
        try:
            selected = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise PromptWorkflowRolloutError(f"{name} is invalid") from exc
    else:
        raise PromptWorkflowRolloutError(f"{name} must be a timestamp")
    if selected.tzinfo is None:
        raise PromptWorkflowRolloutError(f"{name} must include a timezone")
    return (
        selected.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _mode(value: Any) -> PromptWorkflowRolloutMode:
    if isinstance(value, PromptWorkflowRolloutMode):
        return value
    try:
        return PromptWorkflowRolloutMode(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise PromptWorkflowRolloutError("unknown rollout mode") from exc


@dataclass(frozen=True)
class PromptWorkflowRolloutBinding:
    """Exact current deployment identity for one affected behavior."""

    repository_id: str
    tree_id: str
    behavior_id: str
    objective_id: str
    objective_revision: str
    policy_id: str
    policy_revision: str
    capability_id: str
    capability_revision: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self, name, _text(getattr(self, name), name, maximum=512)
            )

    @property
    def binding_id(self) -> str:
        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "PromptWorkflowRolloutBinding":
        if set(value) != set(cls.__dataclass_fields__):
            raise PromptWorkflowRolloutError("invalid rollout binding fields")
        return cls(**dict(value))


@dataclass(frozen=True)
class PromptWorkflowRolloutPolicy:
    """Reviewed promotion policy.  It cannot waive a safety gate."""

    policy_id: str
    policy_revision: str
    approved_behavior_ids: tuple[str, ...]
    approved_modes: tuple[PromptWorkflowRolloutMode | str, ...] = (
        PromptWorkflowRolloutMode.OFF,
        PromptWorkflowRolloutMode.SHADOW,
        PromptWorkflowRolloutMode.ASSIST,
    )
    require_distinct_current_evaluation: bool = True
    rollback_on_metric_regression: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id")
        )
        object.__setattr__(
            self,
            "policy_revision",
            _text(self.policy_revision, "policy_revision"),
        )
        behaviors = tuple(
            sorted(
                _text(item, "approved_behavior_ids")
                for item in self.approved_behavior_ids
            )
        )
        if len(behaviors) != len(set(behaviors)):
            raise PromptWorkflowRolloutError(
                "approved behavior IDs must be unique"
            )
        object.__setattr__(self, "approved_behavior_ids", behaviors)
        modes = tuple(
            sorted(
                {_mode(item) for item in self.approved_modes},
                key=lambda x: x.value,
            )
        )
        object.__setattr__(self, "approved_modes", modes)
        if not isinstance(
            self.require_distinct_current_evaluation, bool
        ) or not isinstance(self.rollback_on_metric_regression, bool):
            raise PromptWorkflowRolloutError("policy flags must be booleans")

    @property
    def policy_binding_id(self) -> str:
        return _identity(self.to_dict())

    def approves(
        self, behavior_id: str, mode: PromptWorkflowRolloutMode
    ) -> bool:
        return (
            behavior_id in self.approved_behavior_ids
            and mode in self.approved_modes
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "approved_behavior_ids": list(self.approved_behavior_ids),
            "approved_modes": [item.value for item in self.approved_modes],
            "require_distinct_current_evaluation": (
                self.require_distinct_current_evaluation
            ),
            "rollback_on_metric_regression": self.rollback_on_metric_regression,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "PromptWorkflowRolloutPolicy":
        if set(value) != set(cls.__dataclass_fields__):
            raise PromptWorkflowRolloutError("invalid rollout policy fields")
        return cls(**dict(value))


@dataclass(frozen=True)
class PromptWorkflowRolloutEvaluation:
    """A time-bound benchmark observation; report values are always replayed."""

    evaluation_id: str
    observed_at: datetime | str
    benchmark: PromptWorkflowBenchmark

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "evaluation_id", _text(self.evaluation_id, "evaluation_id")
        )
        object.__setattr__(
            self, "observed_at", _timestamp(self.observed_at, "observed_at")
        )
        if not isinstance(self.benchmark, PromptWorkflowBenchmark):
            raise PromptWorkflowRolloutError(
                "evaluation benchmark has the wrong type"
            )

    @property
    def evaluation_receipt_id(self) -> str:
        return _identity(self.to_dict())

    @property
    def report(self) -> PromptWorkflowGateReport:
        return recompute_prompt_workflow_gate(self.benchmark)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROMPT_WORKFLOW_ROLLOUT_EVALUATION_SCHEMA,
            "version": PROMPT_WORKFLOW_ROLLOUT_VERSION,
            "evaluation_id": self.evaluation_id,
            "observed_at": self.observed_at,
            "benchmark_id": self.benchmark.benchmark_id,
            "report_id": self.report.report_id,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        benchmark: PromptWorkflowBenchmark,
    ) -> "PromptWorkflowRolloutEvaluation":
        allowed = {
            "schema",
            "version",
            "evaluation_id",
            "observed_at",
            "benchmark_id",
            "report_id",
        }
        if set(value) != allowed:
            raise PromptWorkflowRolloutError("invalid evaluation fields")
        result = cls(
            evaluation_id=value["evaluation_id"],
            observed_at=value["observed_at"],
            benchmark=benchmark,
        )
        if _canonical_bytes(value) != _canonical_bytes(result.to_dict()):
            raise PromptWorkflowRolloutError(
                "evaluation does not match producer receipt replay"
            )
        return result


def _binding_failures(
    evaluation: PromptWorkflowRolloutEvaluation,
    binding: PromptWorkflowRolloutBinding,
    *,
    require_current_tree: bool,
) -> tuple[str, ...]:
    failures: set[str] = set()
    for receipt in evaluation.benchmark.receipts:
        identity = receipt.identity
        names = (
            "repository_id",
            "objective_id",
            "objective_revision",
            "policy_id",
            "policy_revision",
            "capability_id",
            "capability_revision",
        )
        for name in names:
            if getattr(identity, name) != getattr(binding, name):
                failures.add(f"stale-binding:{name}")
        if require_current_tree and identity.tree_id != binding.tree_id:
            failures.add("stale-binding:tree_id")
    return tuple(sorted(failures))


def _population_key(
    evaluation: PromptWorkflowRolloutEvaluation,
) -> tuple[tuple[str, ...], ...]:
    """Compare frozen interventions, excluding observation-local receipt IDs."""

    return tuple(
        sorted(
            (
                receipt.identity.repository_id,
                receipt.identity.prompt_fixture_id,
                receipt.identity.objective_id,
                receipt.identity.objective_revision,
                receipt.identity.policy_id,
                receipt.identity.policy_revision,
                receipt.identity.capability_id,
                receipt.identity.capability_revision,
                receipt.identity.partition_id,
                receipt.planning_mode.value,
                receipt.task_source.value,
                receipt.transport.value,
                (
                    receipt.adversarial_fixture.value
                    if receipt.adversarial_fixture is not None
                    else (
                        receipt.chaos_boundary.value
                        if receipt.chaos_boundary is not None
                        else (
                            receipt.optional_dependency.value
                            if receipt.optional_dependency is not None
                            else "paired"
                        )
                    )
                ),
            )
            for receipt in evaluation.benchmark.receipts
        )
    )


def _metric_regressions(
    qualifying: PromptWorkflowGateReport,
    current: PromptWorkflowGateReport,
) -> tuple[str, ...]:
    failures: list[str] = []
    lower_is_better = (
        "model_calls",
        "total_tokens",
        "retries",
        "storage_bytes",
        "process_count",
    )
    higher_is_better = (
        "admitted_task_cid_count",
        "ready_task_cid_count",
        "accepted_effect_count",
        "paired_path_count",
    )
    for name in lower_is_better:
        if getattr(current, name) > getattr(qualifying, name):
            failures.append(f"metric-regression:{name}")
    for name in higher_is_better:
        if getattr(current, name) < getattr(qualifying, name):
            failures.append(f"metric-regression:{name}")
    for flag in (
        "task_cid_parity_passed",
        "ready_set_parity_passed",
        "effect_parity_passed",
        "terminal_parity_passed",
        "transport_parity_passed",
        "adversarial_passed",
        "chaos_passed",
        "bounds_passed",
        "secret_hygiene_passed",
        "deterministic_degraded_passed",
        "lazy_discovery_passed",
    ):
        if getattr(qualifying, flag) and not getattr(current, flag):
            failures.append(f"metric-regression:{flag}")
    return tuple(failures)


@dataclass(frozen=True)
class PromptWorkflowRolloutDecision:
    """Desired/effective mode with exact evidence and rollback reasons."""

    binding: PromptWorkflowRolloutBinding
    policy: PromptWorkflowRolloutPolicy
    desired_mode: PromptWorkflowRolloutMode
    effective_mode: PromptWorkflowRolloutMode
    qualification_evaluation_id: str
    qualification_report_id: str
    current_evaluation_id: str
    current_report_id: str
    reason_codes: tuple[str, ...]
    qualification_passed: bool
    current_root_passed: bool
    automatic_ready: bool
    rollback_applied: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "desired_mode", _mode(self.desired_mode))
        object.__setattr__(self, "effective_mode", _mode(self.effective_mode))
        reasons = tuple(sorted(set(self.reason_codes)))
        object.__setattr__(self, "reason_codes", reasons)
        if self.desired_mode is PromptWorkflowRolloutMode.OFF:
            if self.effective_mode is not PromptWorkflowRolloutMode.OFF:
                raise PromptWorkflowRolloutError("off cannot gain authority")
        elif self.desired_mode is PromptWorkflowRolloutMode.SHADOW:
            if self.effective_mode is not PromptWorkflowRolloutMode.SHADOW:
                raise PromptWorkflowRolloutError("shadow cannot gain authority")
        elif self.effective_mode not in {
            self.desired_mode,
            PromptWorkflowRolloutMode.SHADOW,
        }:
            raise PromptWorkflowRolloutError(
                "failed promotion must return to shadow"
            )
        if (
            self.effective_mode is PromptWorkflowRolloutMode.AUTOMATIC
            and not self.automatic_ready
        ):
            raise PromptWorkflowRolloutError(
                "automatic requires the complete two-observation gate"
            )

    @property
    def decision_id(self) -> str:
        return _identity(self.to_dict(include_decision_id=False))

    @property
    def affected_behavior_ids(self) -> tuple[str, ...]:
        return (self.binding.behavior_id,)

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def completion_authoritative(self) -> bool:
        return False

    def explain(self) -> str:
        if self.effective_mode is self.desired_mode and not self.reason_codes:
            return (
                f"{self.binding.behavior_id} is {self.effective_mode.value}; "
                "all gates required for that mode passed."
            )
        return (
            f"{self.binding.behavior_id} returned to shadow: "
            + ", ".join(self.reason_codes)
        )

    def to_dict(self, *, include_decision_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": PROMPT_WORKFLOW_ROLLOUT_DECISION_SCHEMA,
            "version": PROMPT_WORKFLOW_ROLLOUT_VERSION,
            "requirement_id": PROMPT_WORKFLOW_ROLLOUT_REQUIREMENT_ID,
            "binding": self.binding.to_dict(),
            "binding_id": self.binding.binding_id,
            "policy": self.policy.to_dict(),
            "policy_binding_id": self.policy.policy_binding_id,
            "desired_mode": self.desired_mode.value,
            "effective_mode": self.effective_mode.value,
            "qualification_evaluation_id": self.qualification_evaluation_id,
            "qualification_report_id": self.qualification_report_id,
            "current_evaluation_id": self.current_evaluation_id,
            "current_report_id": self.current_report_id,
            "reason_codes": list(self.reason_codes),
            "qualification_passed": self.qualification_passed,
            "current_root_passed": self.current_root_passed,
            "automatic_ready": self.automatic_ready,
            "rollback_applied": self.rollback_applied,
            "affected_behavior_ids": list(self.affected_behavior_ids),
            "explanation": self.explain(),
            "authoritative": False,
            "completion_authoritative": False,
        }
        if include_decision_id:
            payload["decision_id"] = self.decision_id
        return payload

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        qualification: PromptWorkflowRolloutEvaluation,
        binding: PromptWorkflowRolloutBinding,
        policy: PromptWorkflowRolloutPolicy,
        current_evaluation: PromptWorkflowRolloutEvaluation | None = None,
    ) -> "PromptWorkflowRolloutDecision":
        """Restore a decision only by replaying its complete source evidence."""

        try:
            desired_mode = value["desired_mode"]
        except (KeyError, TypeError) as exc:
            raise PromptWorkflowRolloutError(
                "rollout decision is missing desired_mode"
            ) from exc
        replayed = evaluate_prompt_workflow_rollout(
            qualification,
            binding=binding,
            policy=policy,
            desired_mode=desired_mode,
            current_evaluation=current_evaluation,
        )
        if _canonical_bytes(value) != _canonical_bytes(replayed.to_dict()):
            raise PromptWorkflowRolloutError(
                "rollout decision does not match source evidence replay"
            )
        return replayed

    @classmethod
    def from_json(
        cls,
        value: str | bytes | bytearray,
        **sources: Any,
    ) -> "PromptWorkflowRolloutDecision":
        return cls.from_dict(
            _load_json(value, "prompt workflow rollout decision"),
            **sources,
        )


def evaluate_prompt_workflow_rollout(
    qualification: PromptWorkflowRolloutEvaluation,
    *,
    binding: PromptWorkflowRolloutBinding,
    policy: PromptWorkflowRolloutPolicy,
    desired_mode: PromptWorkflowRolloutMode | str = (
        PromptWorkflowRolloutMode.SHADOW
    ),
    current_evaluation: PromptWorkflowRolloutEvaluation | None = None,
) -> PromptWorkflowRolloutDecision:
    """Recompute all gates and derive a non-authoritative rollout decision."""

    desired = _mode(desired_mode)
    if not isinstance(qualification, PromptWorkflowRolloutEvaluation):
        raise PromptWorkflowRolloutError("qualification has the wrong type")
    if not isinstance(binding, PromptWorkflowRolloutBinding):
        raise PromptWorkflowRolloutError("binding has the wrong type")
    if not isinstance(policy, PromptWorkflowRolloutPolicy):
        raise PromptWorkflowRolloutError("policy has the wrong type")

    qualifying_report = qualification.report
    reasons = list(
        _binding_failures(qualification, binding, require_current_tree=False)
    )
    if (
        policy.policy_id != binding.policy_id
        or policy.policy_revision != binding.policy_revision
    ):
        reasons.append("stale-binding:rollout-policy")
    if not qualifying_report.passed:
        reasons.extend(
            f"qualification:{item}" for item in qualifying_report.failure_codes
        )
    qualification_passed = not reasons

    current_report: PromptWorkflowGateReport | None = None
    current_passed = False
    if current_evaluation is not None:
        if not isinstance(current_evaluation, PromptWorkflowRolloutEvaluation):
            raise PromptWorkflowRolloutError(
                "current_evaluation has the wrong type"
            )
        current_report = current_evaluation.report
        current_reasons = list(
            _binding_failures(
                current_evaluation, binding, require_current_tree=True
            )
        )
        if not current_report.passed:
            current_reasons.extend(
                f"current:{item}" for item in current_report.failure_codes
            )
        if (
            current_evaluation.evaluation_id == qualification.evaluation_id
            or current_evaluation.evaluation_receipt_id
            == qualification.evaluation_receipt_id
            or current_evaluation.benchmark.benchmark_id
            == qualification.benchmark.benchmark_id
        ):
            current_reasons.append("current-evaluation-not-distinct")
        if _datetime(current_evaluation.observed_at) <= _datetime(
            qualification.observed_at
        ):
            current_reasons.append("current-evaluation-not-later")
        if _population_key(current_evaluation) != _population_key(qualification):
            current_reasons.append("benchmark-population-narrowed")
        if policy.rollback_on_metric_regression:
            current_reasons.extend(
                _metric_regressions(qualifying_report, current_report)
            )
        reasons.extend(current_reasons)
        current_passed = not current_reasons
    elif desired is PromptWorkflowRolloutMode.AUTOMATIC:
        reasons.append("current-evaluation-required")

    if desired in {
        PromptWorkflowRolloutMode.ASSIST,
        PromptWorkflowRolloutMode.AUTOMATIC,
    } and not policy.approves(binding.behavior_id, desired):
        reasons.append("mode-not-policy-approved")
    reasons = sorted(set(reasons))
    automatic_ready = (
        desired is PromptWorkflowRolloutMode.AUTOMATIC
        and qualification_passed
        and current_passed
        and not reasons
    )

    if desired is PromptWorkflowRolloutMode.OFF:
        effective = PromptWorkflowRolloutMode.OFF
    elif desired is PromptWorkflowRolloutMode.SHADOW:
        effective = PromptWorkflowRolloutMode.SHADOW
    elif desired is PromptWorkflowRolloutMode.ASSIST:
        effective = (
            PromptWorkflowRolloutMode.ASSIST
            if qualification_passed and not reasons
            else PromptWorkflowRolloutMode.SHADOW
        )
    else:
        effective = (
            PromptWorkflowRolloutMode.AUTOMATIC
            if automatic_ready
            else PromptWorkflowRolloutMode.SHADOW
        )
    rollback = effective is PromptWorkflowRolloutMode.SHADOW and desired in {
        PromptWorkflowRolloutMode.ASSIST,
        PromptWorkflowRolloutMode.AUTOMATIC,
    }
    return PromptWorkflowRolloutDecision(
        binding=binding,
        policy=policy,
        desired_mode=desired,
        effective_mode=effective,
        qualification_evaluation_id=qualification.evaluation_id,
        qualification_report_id=qualifying_report.report_id,
        current_evaluation_id=(
            current_evaluation.evaluation_id if current_evaluation else ""
        ),
        current_report_id=current_report.report_id if current_report else "",
        reason_codes=tuple(reasons),
        qualification_passed=qualification_passed,
        current_root_passed=current_passed,
        automatic_ready=automatic_ready,
        rollback_applied=rollback,
    )


def verify_prompt_workflow_rollout(
    decision: PromptWorkflowRolloutDecision,
    qualification: PromptWorkflowRolloutEvaluation,
    *,
    binding: PromptWorkflowRolloutBinding,
    policy: PromptWorkflowRolloutPolicy,
    current_evaluation: PromptWorkflowRolloutEvaluation | None = None,
) -> bool:
    try:
        replayed = evaluate_prompt_workflow_rollout(
            qualification,
            binding=binding,
            policy=policy,
            desired_mode=decision.desired_mode,
            current_evaluation=current_evaluation,
        )
    except (PromptWorkflowRolloutError, PromptWorkflowBenchmarkError):
        return False
    return _canonical_bytes(decision.to_dict()) == _canonical_bytes(
        replayed.to_dict()
    )


@dataclass(frozen=True)
class PromptWorkflowControlRequest:
    action: PromptWorkflowControlAction | str
    expected_binding_id: str = ""
    expected_decision_id: str = ""

    def __post_init__(self) -> None:
        try:
            selected = (
                self.action
                if isinstance(self.action, PromptWorkflowControlAction)
                else PromptWorkflowControlAction(str(self.action))
            )
        except ValueError as exc:
            raise PromptWorkflowRolloutError("unknown control action") from exc
        object.__setattr__(self, "action", selected)
        if self.expected_binding_id:
            object.__setattr__(
                self,
                "expected_binding_id",
                _text(self.expected_binding_id, "expected_binding_id"),
            )
        if self.expected_decision_id:
            object.__setattr__(
                self,
                "expected_decision_id",
                _text(self.expected_decision_id, "expected_decision_id"),
            )

    @property
    def request_id(self) -> str:
        return _identity(self.to_dict(include_request_id=False))

    def to_dict(self, *, include_request_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": PROMPT_WORKFLOW_CONTROL_REQUEST_SCHEMA,
            "version": PROMPT_WORKFLOW_ROLLOUT_VERSION,
            "action": self.action.value,
            "expected_binding_id": self.expected_binding_id,
            "expected_decision_id": self.expected_decision_id,
        }
        if include_request_id:
            payload["request_id"] = self.request_id
        return payload

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "PromptWorkflowControlRequest":
        allowed = {
            "schema",
            "version",
            "action",
            "expected_binding_id",
            "expected_decision_id",
            "request_id",
        }
        required = {
            "schema",
            "version",
            "action",
            "expected_binding_id",
            "expected_decision_id",
        }
        if set(value).difference(allowed) or not required.issubset(value):
            raise PromptWorkflowRolloutError("unknown control request fields")
        if (
            value.get("schema") != PROMPT_WORKFLOW_CONTROL_REQUEST_SCHEMA
            or value.get("version") != PROMPT_WORKFLOW_ROLLOUT_VERSION
        ):
            raise PromptWorkflowRolloutError("unsupported control request")
        result = cls(
            action=value["action"],
            expected_binding_id=value.get("expected_binding_id", ""),
            expected_decision_id=value.get("expected_decision_id", ""),
        )
        if value.get("request_id", result.request_id) != result.request_id:
            raise PromptWorkflowRolloutError("control request ID mismatch")
        return result

    @classmethod
    def from_json(
        cls, value: str | bytes | bytearray
    ) -> "PromptWorkflowControlRequest":
        return cls.from_dict(
            _load_json(value, "prompt workflow control request")
        )


@dataclass(frozen=True)
class PromptWorkflowControlResult:
    request_id: str
    action: PromptWorkflowControlAction
    decision: PromptWorkflowRolloutDecision
    changed: bool
    explanation: str

    @property
    def result_id(self) -> str:
        return _identity(self.to_dict(include_result_id=False))

    def to_dict(self, *, include_result_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": PROMPT_WORKFLOW_CONTROL_RESULT_SCHEMA,
            "version": PROMPT_WORKFLOW_ROLLOUT_VERSION,
            "request_id": self.request_id,
            "action": self.action.value,
            "decision": self.decision.to_dict(),
            "changed": self.changed,
            "explanation": self.explanation,
        }
        if include_result_id:
            payload["result_id"] = self.result_id
        return payload


class PromptWorkflowPublicAPI:
    """One canonical stateful control service used by all three surfaces."""

    def __init__(
        self,
        qualification: PromptWorkflowRolloutEvaluation,
        *,
        binding: PromptWorkflowRolloutBinding,
        policy: PromptWorkflowRolloutPolicy,
        current_evaluation: PromptWorkflowRolloutEvaluation | None = None,
        initial_mode: PromptWorkflowRolloutMode | str = (
            PromptWorkflowRolloutMode.SHADOW
        ),
    ) -> None:
        self.qualification = qualification
        self.binding = binding
        self.policy = policy
        self.current_evaluation = current_evaluation
        self._lock = RLock()
        self._decision = evaluate_prompt_workflow_rollout(
            qualification,
            binding=binding,
            policy=policy,
            desired_mode=initial_mode,
            current_evaluation=current_evaluation,
        )

    @staticmethod
    def discovery() -> dict[str, Any]:
        """Static discovery; does not construct providers or inspect the host."""

        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/"
            "prompt-workflow-public-api@1",
            "version": PROMPT_WORKFLOW_ROLLOUT_VERSION,
            "requirement_id": PROMPT_WORKFLOW_ROLLOUT_REQUIREMENT_ID,
            "behavior_id": PROMPT_WORKFLOW_BEHAVIOR_ID,
            "surfaces": [
                item.value for item in PromptWorkflowControlSurface
            ],
            "actions": [item.value for item in PromptWorkflowControlAction],
            "modes": [item.value for item in PromptWorkflowRolloutMode],
            "optional_providers_loaded": False,
            "processes_started": False,
        }

    @property
    def decision(self) -> PromptWorkflowRolloutDecision:
        with self._lock:
            return self._decision

    def _decode(
        self, request: PromptWorkflowControlRequest | Mapping[str, Any] | str
    ) -> PromptWorkflowControlRequest:
        if isinstance(request, PromptWorkflowControlRequest):
            return request
        if isinstance(request, str):
            return PromptWorkflowControlRequest(action=request)
        if isinstance(request, Mapping):
            return PromptWorkflowControlRequest.from_dict(request)
        raise PromptWorkflowRolloutError("invalid control request")

    def execute(
        self, request: PromptWorkflowControlRequest | Mapping[str, Any] | str
    ) -> PromptWorkflowControlResult:
        selected = self._decode(request)
        with self._lock:
            previous = self._decision
            # Status and explanation are live safety checks, not cached prose.
            self._decision = evaluate_prompt_workflow_rollout(
                self.qualification,
                binding=self.binding,
                policy=self.policy,
                desired_mode=previous.desired_mode,
                current_evaluation=self.current_evaluation,
            )
            if (
                selected.expected_binding_id
                and selected.expected_binding_id != self.binding.binding_id
            ):
                raise PromptWorkflowRolloutError("stale control binding")
            if (
                selected.expected_decision_id
                and selected.expected_decision_id != self._decision.decision_id
            ):
                raise PromptWorkflowRolloutError("stale control decision")
            mode = selected.action.requested_mode
            if selected.action is PromptWorkflowControlAction.ROLLBACK:
                mode = PromptWorkflowRolloutMode.SHADOW
            if mode is not None:
                candidate = evaluate_prompt_workflow_rollout(
                    self.qualification,
                    binding=self.binding,
                    policy=self.policy,
                    desired_mode=mode,
                    current_evaluation=self.current_evaluation,
                )
                self._decision = candidate
            decision = self._decision
            explanation = (
                decision.explain()
                if selected.action
                in {
                    PromptWorkflowControlAction.EXPLANATION,
                    PromptWorkflowControlAction.ROLLBACK,
                }
                else (
                    f"desired={decision.desired_mode.value}; "
                    f"effective={decision.effective_mode.value}"
                )
            )
            return PromptWorkflowControlResult(
                request_id=selected.request_id,
                action=selected.action,
                decision=decision,
                changed=decision.decision_id != previous.decision_id,
                explanation=explanation,
            )

    # These aliases deliberately contain no surface-specific policy.
    python = execute
    cli = execute
    mcp = execute

    def status(self) -> PromptWorkflowControlResult:
        return self.execute("status")

    def explanation(self) -> PromptWorkflowControlResult:
        return self.execute("explanation")

    def rollback(self) -> PromptWorkflowControlResult:
        return self.execute("rollback")


def build_default_prompt_workflow_binding(
    *,
    tree_id: str = "sha256:frozen-prompt-workflow-tree",
) -> PromptWorkflowRolloutBinding:
    return PromptWorkflowRolloutBinding(
        repository_id="repository:prompt-workflow-benchmark@1",
        tree_id=tree_id,
        behavior_id=PROMPT_WORKFLOW_BEHAVIOR_ID,
        objective_id="ASI-G470",
        objective_revision="sha256:frozen-prompt-objective",
        policy_id="policy:prompt-workflow-rollout@1",
        policy_revision="sha256:frozen-prompt-policy",
        capability_id="capability:prompt-workflow-local@1",
        capability_revision="sha256:frozen-prompt-capability",
    )


def build_default_prompt_workflow_policy(
    *,
    approve_automatic: bool = True,
) -> PromptWorkflowRolloutPolicy:
    modes: list[PromptWorkflowRolloutMode] = [
        PromptWorkflowRolloutMode.OFF,
        PromptWorkflowRolloutMode.SHADOW,
        PromptWorkflowRolloutMode.ASSIST,
    ]
    if approve_automatic:
        modes.append(PromptWorkflowRolloutMode.AUTOMATIC)
    return PromptWorkflowRolloutPolicy(
        policy_id="policy:prompt-workflow-rollout@1",
        policy_revision="sha256:frozen-prompt-policy",
        approved_behavior_ids=(PROMPT_WORKFLOW_BEHAVIOR_ID,),
        approved_modes=tuple(modes),
    )


__all__ = (
    "PROMPT_WORKFLOW_BEHAVIOR_ID",
    "PROMPT_WORKFLOW_CONTROL_REQUEST_SCHEMA",
    "PROMPT_WORKFLOW_CONTROL_RESULT_SCHEMA",
    "PROMPT_WORKFLOW_ROLLOUT_DECISION_SCHEMA",
    "PROMPT_WORKFLOW_ROLLOUT_EVALUATION_SCHEMA",
    "PROMPT_WORKFLOW_ROLLOUT_REQUIREMENT_ID",
    "PROMPT_WORKFLOW_ROLLOUT_VERSION",
    "PromptWorkflowControlAction",
    "PromptWorkflowControlRequest",
    "PromptWorkflowControlResult",
    "PromptWorkflowControlSurface",
    "PromptWorkflowPublicAPI",
    "PromptWorkflowRolloutBinding",
    "PromptWorkflowRolloutDecision",
    "PromptWorkflowRolloutError",
    "PromptWorkflowRolloutEvaluation",
    "PromptWorkflowRolloutMode",
    "PromptWorkflowRolloutPolicy",
    "build_default_prompt_workflow_binding",
    "build_default_prompt_workflow_policy",
    "evaluate_prompt_workflow_rollout",
    "verify_prompt_workflow_rollout",
)
