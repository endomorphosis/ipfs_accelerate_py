"""Hierarchical supervisor usage envelopes and endpoint accounting bridge.

Defines immutable, provider-neutral contracts for nested supervisor budgets:

    deployment policy
      -> supervisor run
        -> goal / objective
          -> task
            -> attempt
              -> stage / lane
                -> provider request

A child budget may only lower its parent across every typed endpoint usage
dimension, limit window, and cost currency.  Identities bind repository,
state, tree, policy, run, goal/objective, task, attempt, stage, lane,
request, catalog/usage revisions, endpoint scope, caller, deadline,
idempotency, lease, and fence.  Records never carry prompts, source, media,
model output, credentials, or raw endpoints.

This module is pure: cold import and schema discovery perform no network,
provider, process, database, or secret-store I/O.  The bridge projects
reconciled endpoint events into supervisor attribution without authorizing
usage, rewriting provider settlement, or treating usage as correctness or
completion evidence.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final, Optional

from ipfs_accelerate_py.endpoint_usage.identity import (
    assert_no_prompt_media_or_output,
    contains_bearer_url,
    contains_raw_endpoint,
    is_secret_key,
    is_secret_value,
)
from ipfs_accelerate_py.endpoint_usage.schema import (
    LimitWindow,
    Quantity,
    QuantityKind,
    SchemaValidationError,
    UsageDimension,
    UsageEvent,
    UsageEventKind,
    UsageVector,
    UsageVectorEntry,
    WindowKind,
)

from .formal_verification_contracts import CanonicalContract


SUPERVISOR_USAGE_ENVELOPE_REQUIREMENT_ID: Final[str] = (
    "requirement:supervisor-usage-envelope.v1"
)
SUPERVISOR_USAGE_ENVELOPE_GOAL_ID: Final[str] = "ASI-G510"
PROVIDER_USAGE_CONTRACT_VERSION: Final[int] = 1
SCHEMA_VERSION: Final[int] = PROVIDER_USAGE_CONTRACT_VERSION

SUPERVISOR_USAGE_SCOPE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/supervisor-usage-scope@1"
)
SUPERVISOR_USAGE_BUDGET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/supervisor-usage-budget@1"
)
SUPERVISOR_BUDGET_LIMIT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/supervisor-budget-limit@1"
)
SUPERVISOR_USAGE_ENVELOPE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/supervisor-usage-envelope@1"
)
SUPERVISOR_USAGE_ATTRIBUTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/supervisor-usage-attribution@1"
)
SUPERVISOR_TO_ENDPOINT_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/supervisor-to-endpoint-request@1"
)
SUPERVISOR_USAGE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/supervisor-usage-receipt@1"
)

MAX_TEXT_BYTES: Final[int] = 512
MAX_NESTING_DEPTH: Final[int] = 8
MAX_BUDGET_LIMITS: Final[int] = 64
MAX_CHILDREN: Final[int] = 256
MAX_EVENT_IDS: Final[int] = 4_096
MAX_ABS_CEILING: Final[int] = (1 << 63) - 1
MAX_SERIALIZED_BYTES: Final[int] = 4 * 1024 * 1024

# Accounting bridge authority bounds.  These never flip true: the ledger and
# efficiency path may attribute work but cannot admit capacity or settle.
BRIDGE_AUTHORIZES_USAGE: Final[bool] = False
BRIDGE_REWRITES_PROVIDER_SETTLEMENT: Final[bool] = False
BRIDGE_IS_COMPLETION_EVIDENCE: Final[bool] = False
BRIDGE_IS_CORRECTNESS_EVIDENCE: Final[bool] = False

_TEXT_SAFE = re.compile(r"^[^\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+$")
_CURRENCY = re.compile(r"^[A-Z]{3}$")
_LEVEL_ORDER: Final[tuple[str, ...]] = (
    "deployment",
    "run",
    "goal",
    "task",
    "attempt",
    "stage",
    "lane",
    "request",
)
_SETTLEMENT_KINDS: Final[frozenset[UsageEventKind]] = frozenset(
    {
        UsageEventKind.COMMIT,
        UsageEventKind.STREAM_SETTLEMENT,
        UsageEventKind.OBSERVATION_SUCCESS,
        UsageEventKind.OBSERVATION_FAILURE,
        UsageEventKind.CORRECTION,
        UsageEventKind.RELEASE,
        UsageEventKind.REFUND,
    }
)


class ProviderUsageValidationError(ValueError):
    """Supervisor usage envelope, budget, or bridge contract is invalid."""


class SupervisorUsageLevel(str, Enum):
    """Nested budget lineage levels from deployment policy to request."""

    DEPLOYMENT = "deployment"
    RUN = "run"
    GOAL = "goal"
    TASK = "task"
    ATTEMPT = "attempt"
    STAGE = "stage"
    LANE = "lane"
    REQUEST = "request"

    @property
    def depth(self) -> int:
        return _LEVEL_ORDER.index(self.value)

    def may_parent(self, child: "SupervisorUsageLevel") -> bool:
        return child.depth == self.depth + 1


class SupervisorUsageFinalStatus(str, Enum):
    COMMITTED = "committed"
    RELEASED = "released"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    FAILED = "failed"
    CAPACITY_UNAVAILABLE = "capacity_unavailable"
    UNKNOWN = "unknown"


def _fail(message: str) -> None:
    raise ProviderUsageValidationError(message)


def _text(value: Any, name: str, *, required: bool = True, maximum: int = MAX_TEXT_BYTES) -> str:
    if value is None:
        if required:
            _fail(f"{name} must be text")
        return ""
    if not isinstance(value, str):
        _fail(f"{name} must be text")
    result = value.strip()
    if required and not result:
        _fail(f"{name} must not be empty")
    if not required and not result:
        return ""
    if len(result.encode("utf-8")) > maximum:
        _fail(f"{name} is too large")
    if not _TEXT_SAFE.fullmatch(result):
        _fail(f"{name} contains control characters")
    if is_secret_value(result) or contains_bearer_url(result):
        _fail(f"{name} contains credential-shaped data")
    if contains_raw_endpoint(result):
        _fail(f"{name} must not embed a raw endpoint or URL")
    lowered = name.casefold()
    if any(
        marker in lowered
        for marker in (
            "prompt",
            "source",
            "media",
            "output",
            "credential",
            "password",
            "token",
            "secret",
            "api_key",
        )
    ) and name not in {
        "idempotency_key",
        "endpoint_scope_id",
        "usage_revision",
        "catalog_revision",
        "endpoint_event_id",
        "supersedes_event_id",
    }:
        # Field names are fixed; values still go through secret/URL checks above.
        pass
    return result


def _optional_text(value: Any, name: str, *, maximum: int = MAX_TEXT_BYTES) -> str:
    return _text(value, name, required=False, maximum=maximum)


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_ABS_CEILING,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{name} must be an integer")
    if value < minimum or value > maximum:
        _fail(f"{name} must be between {minimum} and {maximum}")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(raw)
    except (TypeError, ValueError) as exc:
        raise ProviderUsageValidationError(
            f"{name} is not a supported {enum_type.__name__}"
        ) from exc


def _currency(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not _CURRENCY.fullmatch(value):
        _fail("currency must be a three-letter ISO code")
    return value


def _closed(
    payload: Mapping[str, Any],
    *,
    schema: str,
    allowed: Iterable[str],
    name: str,
) -> None:
    if not isinstance(payload, Mapping):
        _fail(f"{name} must be an object")
    if payload.get("schema") != schema:
        _fail(f"unsupported {name} schema")
    version = payload.get("contract_version", payload.get("schema_version"))
    if version != PROVIDER_USAGE_CONTRACT_VERSION:
        _fail(f"unsupported {name} version")
    unknown = set(payload).difference(allowed)
    if unknown:
        _fail(f"{name} contains unknown fields: {sorted(unknown)}")


def _claim(payload: Mapping[str, Any], actual: str, *names: str) -> None:
    for name in names:
        claimed = payload.get(name)
        if claimed not in (None, "", actual):
            _fail("content identity does not match canonical contents")


def _strict_json(value: str | bytes | bytearray, name: str) -> Mapping[str, Any]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                _fail(f"{name} JSON contains duplicate object keys")
            result[key] = item
        return result

    try:
        if isinstance(value, (bytes, bytearray)):
            value = bytes(value).decode("utf-8")
        if not isinstance(value, str):
            _fail(f"{name} JSON must be text")
        decoded = json.loads(value, object_pairs_hook=unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderUsageValidationError(f"{name} JSON is malformed") from exc
    if not isinstance(decoded, Mapping):
        _fail(f"{name} JSON must contain an object")
    return decoded


def _window(value: Any) -> LimitWindow:
    if isinstance(value, LimitWindow):
        return value
    if isinstance(value, Mapping):
        try:
            return LimitWindow.from_dict(value)
        except SchemaValidationError as exc:
            raise ProviderUsageValidationError(str(exc)) from exc
    _fail("window must be a LimitWindow")


def _usage_vector(value: Any) -> UsageVector:
    if isinstance(value, UsageVector):
        return value
    if value is None:
        return UsageVector()
    if isinstance(value, Mapping) or isinstance(value, Sequence):
        try:
            return UsageVector.from_dict(value)
        except SchemaValidationError as exc:
            raise ProviderUsageValidationError(str(exc)) from exc
    _fail("usage vector is malformed")


def _usage_event(value: Any) -> UsageEvent:
    if isinstance(value, UsageEvent):
        return value
    if isinstance(value, Mapping):
        try:
            return UsageEvent.from_dict(value)
        except SchemaValidationError as exc:
            raise ProviderUsageValidationError(str(exc)) from exc
    _fail("endpoint event must be a UsageEvent")


def _window_key(window: LimitWindow) -> tuple[Any, ...]:
    return (
        window.kind.value if isinstance(window.kind, WindowKind) else str(window.kind),
        window.length_ms,
        window.anchor_at or "",
        window.reset_at or "",
        window.refill_per_second,
        window.burst,
        window.safety_reserve,
    )


def _limit_key(
    dimension: UsageDimension, window: LimitWindow, currency: Optional[str]
) -> tuple[Any, ...]:
    return (dimension.value, _window_key(window), currency or "")


def _reject_forbidden_payload(payload: Mapping[str, Any]) -> None:
    try:
        assert_no_prompt_media_or_output(dict(payload))
    except Exception as exc:  # UsageIdentityError and related
        raise ProviderUsageValidationError(str(exc)) from exc
    for key in payload:
        if is_secret_key(str(key)) and "pseudonym" not in str(key).casefold():
            _fail(f"forbidden credential field: {key}")


class _UsageContract(CanonicalContract):
    @property
    def schema_version(self) -> int:
        return PROVIDER_USAGE_CONTRACT_VERSION

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "_UsageContract":
        return cls.from_dict(_strict_json(value, cls.__name__))  # type: ignore[attr-defined,no-any-return]


@dataclass(frozen=True)
class SupervisorBudgetLimit(_UsageContract):
    """One typed ceiling on a dimension/window/currency triple."""

    SCHEMA: ClassVar[str] = SUPERVISOR_BUDGET_LIMIT_SCHEMA

    dimension: UsageDimension
    ceiling: int
    window: LimitWindow
    currency: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dimension", _enum(self.dimension, UsageDimension, "dimension")
        )
        object.__setattr__(
            self,
            "ceiling",
            _integer(self.ceiling, "ceiling", minimum=0, maximum=MAX_ABS_CEILING),
        )
        object.__setattr__(self, "window", _window(self.window))
        currency = _currency(self.currency)
        if self.dimension is UsageDimension.COST_MICROS:
            if currency is None:
                _fail("currency is required for cost_micros")
        elif currency is not None:
            _fail("currency is only valid for cost_micros")
        object.__setattr__(self, "currency", currency)

    @property
    def key(self) -> tuple[Any, ...]:
        return _limit_key(self.dimension, self.window, self.currency)

    def lowers_or_matches(self, parent: "SupervisorBudgetLimit") -> bool:
        if self.key != parent.key:
            return False
        return self.ceiling <= parent.ceiling

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROVIDER_USAGE_CONTRACT_VERSION,
            "dimension": self.dimension.value,
            "ceiling": self.ceiling,
            "window": self.window.to_dict(),
            "currency": self.currency,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SupervisorBudgetLimit":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "dimension",
            "ceiling",
            "window",
            "currency",
            "content_id",
        }
        _closed(payload, schema=cls.SCHEMA, allowed=allowed, name="budget limit")
        result = cls(
            dimension=payload.get("dimension", ""),
            ceiling=payload.get("ceiling", -1),
            window=payload.get("window", {}),
            currency=payload.get("currency"),
        )
        _claim(payload, result.content_id, "content_id")
        return result


@dataclass(frozen=True)
class SupervisorUsageBudget(_UsageContract):
    """Closed multi-dimension budget; mixed cost currencies are rejected."""

    SCHEMA: ClassVar[str] = SUPERVISOR_USAGE_BUDGET_SCHEMA

    limits: tuple[SupervisorBudgetLimit, ...] = ()

    def __post_init__(self) -> None:
        raw = self.limits
        if raw is None:
            raw = ()
        if isinstance(raw, (str, bytes, Mapping)) or not isinstance(raw, Sequence):
            _fail("limits must be a sequence")
        if len(raw) > MAX_BUDGET_LIMITS:
            _fail("budget exceeds maximum limits")
        parsed: list[SupervisorBudgetLimit] = []
        for item in raw:
            if isinstance(item, SupervisorBudgetLimit):
                parsed.append(item)
            elif isinstance(item, Mapping):
                parsed.append(SupervisorBudgetLimit.from_dict(item))
            else:
                _fail("limits must contain SupervisorBudgetLimit records")
        seen: dict[tuple[Any, ...], SupervisorBudgetLimit] = {}
        currencies: set[str] = set()
        for limit in parsed:
            if limit.key in seen:
                _fail(
                    f"duplicate budget limit for dimension {limit.dimension.value}"
                )
            seen[limit.key] = limit
            if limit.currency is not None:
                currencies.add(limit.currency)
        if len(currencies) > 1:
            _fail("mixed cost currency is not permitted within one budget")
        object.__setattr__(
            self,
            "limits",
            tuple(
                sorted(
                    parsed,
                    key=lambda item: (
                        item.dimension.value,
                        _window_key(item.window),
                        item.currency or "",
                    ),
                )
            ),
        )

    def by_key(self) -> dict[tuple[Any, ...], SupervisorBudgetLimit]:
        return {item.key: item for item in self.limits}

    def cost_currency(self) -> Optional[str]:
        for item in self.limits:
            if item.dimension is UsageDimension.COST_MICROS:
                return item.currency
        return None

    def is_lower_or_equal(self, parent: "SupervisorUsageBudget") -> bool:
        """Return True when every child limit is present on parent and not raised."""

        parent_map = parent.by_key()
        child_map = self.by_key()
        parent_currency = parent.cost_currency()
        child_currency = self.cost_currency()
        if (
            child_currency is not None
            and parent_currency is not None
            and child_currency != parent_currency
        ):
            return False
        for key, child_limit in child_map.items():
            parent_limit = parent_map.get(key)
            if parent_limit is None:
                return False
            if child_limit.ceiling > parent_limit.ceiling:
                return False
        return True

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROVIDER_USAGE_CONTRACT_VERSION,
            "limits": tuple(item.to_record() for item in self.limits),
            "cost_currency": self.cost_currency(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SupervisorUsageBudget":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "limits",
            "cost_currency",
            "content_id",
        }
        _closed(payload, schema=cls.SCHEMA, allowed=allowed, name="usage budget")
        result = cls(limits=payload.get("limits", ()))
        if payload.get("cost_currency", result.cost_currency()) != result.cost_currency():
            _fail("cost_currency claim does not match budget limits")
        _claim(payload, result.content_id, "content_id")
        return result

    @classmethod
    def of(
        cls,
        *,
        window: LimitWindow | Mapping[str, Any] | None = None,
        currency: Optional[str] = None,
        **dimensions: int,
    ) -> "SupervisorUsageBudget":
        """Build a budget from finite dimension ceilings sharing one window."""

        if window is None:
            window = LimitWindow(kind=WindowKind.LIFETIME)
        else:
            window = _window(window)
        limits = []
        for name, amount in dimensions.items():
            dimension = _enum(name, UsageDimension, "dimension")
            limits.append(
                SupervisorBudgetLimit(
                    dimension=dimension,
                    ceiling=amount,
                    window=window,
                    currency=currency if dimension is UsageDimension.COST_MICROS else None,
                )
            )
        return cls(limits=tuple(limits))


@dataclass(frozen=True)
class SupervisorUsageScope(_UsageContract):
    """Exact secret-free supervisor lineage for one usage budget node."""

    SCHEMA: ClassVar[str] = SUPERVISOR_USAGE_SCOPE_SCHEMA

    level: SupervisorUsageLevel
    repository_id: str
    state_id: str
    tree_id: str
    policy_id: str
    policy_revision: str
    supervisor_run_id: str
    goal_id: str
    objective_id: str
    objective_revision: str
    task_id: str
    attempt: int
    stage: str
    lane: str
    request_id: str
    catalog_revision: str
    usage_revision: str
    endpoint_scope_id: str
    caller_id: str
    deadline_at: str
    idempotency_key: str
    lease_id: str
    fence_id: str
    parent_scope_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "level", _enum(self.level, SupervisorUsageLevel, "level")
        )
        for name in (
            "repository_id",
            "state_id",
            "tree_id",
            "policy_id",
            "policy_revision",
            "catalog_revision",
            "usage_revision",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "supervisor_run_id",
            _text(
                self.supervisor_run_id,
                "supervisor_run_id",
                required=self.level.depth >= SupervisorUsageLevel.RUN.depth,
            ),
        )
        for name in ("goal_id", "objective_id", "objective_revision"):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name),
                    name,
                    required=self.level.depth >= SupervisorUsageLevel.GOAL.depth,
                ),
            )
        object.__setattr__(
            self,
            "task_id",
            _text(
                self.task_id,
                "task_id",
                required=self.level.depth >= SupervisorUsageLevel.TASK.depth,
            ),
        )
        if self.level.depth >= SupervisorUsageLevel.ATTEMPT.depth:
            object.__setattr__(
                self,
                "attempt",
                _integer(self.attempt, "attempt", minimum=1, maximum=100_000),
            )
        else:
            object.__setattr__(
                self, "attempt", _integer(self.attempt, "attempt", minimum=0, maximum=0)
            )
        object.__setattr__(
            self,
            "stage",
            _text(
                self.stage,
                "stage",
                required=self.level.depth >= SupervisorUsageLevel.STAGE.depth,
            ),
        )
        object.__setattr__(
            self,
            "lane",
            _text(
                self.lane,
                "lane",
                required=self.level.depth >= SupervisorUsageLevel.LANE.depth,
            ),
        )
        for name in (
            "request_id",
            "endpoint_scope_id",
            "caller_id",
            "deadline_at",
            "idempotency_key",
            "lease_id",
            "fence_id",
        ):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name),
                    name,
                    required=self.level is SupervisorUsageLevel.REQUEST,
                ),
            )
        object.__setattr__(
            self,
            "parent_scope_id",
            _optional_text(self.parent_scope_id, "parent_scope_id"),
        )
        if self.level is not SupervisorUsageLevel.DEPLOYMENT and not self.parent_scope_id:
            _fail("missing parent ancestry for non-deployment scope")
        if self.level is SupervisorUsageLevel.DEPLOYMENT and self.parent_scope_id:
            _fail("deployment scope cannot declare a parent")

    @property
    def scope_id(self) -> str:
        return self.content_id

    def shares_binding_with(self, other: "SupervisorUsageScope") -> bool:
        return (
            self.repository_id == other.repository_id
            and self.state_id == other.state_id
            and self.tree_id == other.tree_id
            and self.policy_id == other.policy_id
            and self.policy_revision == other.policy_revision
            and self.catalog_revision == other.catalog_revision
            and self.usage_revision == other.usage_revision
        )

    def is_child_of(self, parent: "SupervisorUsageScope") -> bool:
        if not parent.level.may_parent(self.level):
            return False
        if not self.shares_binding_with(parent):
            return False
        if self.parent_scope_id != parent.scope_id:
            return False
        # Child may refine identity fields introduced at or above its level,
        # but must never change ancestor-bound values.
        if self.level.depth >= SupervisorUsageLevel.RUN.depth:
            if parent.supervisor_run_id and self.supervisor_run_id != parent.supervisor_run_id:
                return False
        if self.level.depth >= SupervisorUsageLevel.GOAL.depth:
            for name in ("goal_id", "objective_id", "objective_revision"):
                parent_value = getattr(parent, name)
                if parent_value and getattr(self, name) != parent_value:
                    return False
        if self.level.depth >= SupervisorUsageLevel.TASK.depth:
            if parent.task_id and self.task_id != parent.task_id:
                return False
        if self.level.depth >= SupervisorUsageLevel.ATTEMPT.depth:
            if parent.attempt and self.attempt != parent.attempt:
                return False
        if self.level.depth >= SupervisorUsageLevel.STAGE.depth:
            if parent.stage and self.stage != parent.stage:
                return False
        if self.level.depth >= SupervisorUsageLevel.LANE.depth:
            if parent.lane and self.lane != parent.lane:
                return False
        return True

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROVIDER_USAGE_CONTRACT_VERSION,
            "level": self.level.value,
            "repository_id": self.repository_id,
            "state_id": self.state_id,
            "tree_id": self.tree_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "supervisor_run_id": self.supervisor_run_id,
            "goal_id": self.goal_id,
            "objective_id": self.objective_id,
            "objective_revision": self.objective_revision,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "stage": self.stage,
            "lane": self.lane,
            "request_id": self.request_id,
            "catalog_revision": self.catalog_revision,
            "usage_revision": self.usage_revision,
            "endpoint_scope_id": self.endpoint_scope_id,
            "caller_id": self.caller_id,
            "deadline_at": self.deadline_at,
            "idempotency_key": self.idempotency_key,
            "lease_id": self.lease_id,
            "fence_id": self.fence_id,
            "parent_scope_id": self.parent_scope_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SupervisorUsageScope":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "level",
            "repository_id",
            "state_id",
            "tree_id",
            "policy_id",
            "policy_revision",
            "supervisor_run_id",
            "goal_id",
            "objective_id",
            "objective_revision",
            "task_id",
            "attempt",
            "stage",
            "lane",
            "request_id",
            "catalog_revision",
            "usage_revision",
            "endpoint_scope_id",
            "caller_id",
            "deadline_at",
            "idempotency_key",
            "lease_id",
            "fence_id",
            "parent_scope_id",
            "scope_id",
            "content_id",
        }
        _closed(payload, schema=cls.SCHEMA, allowed=allowed, name="usage scope")
        _reject_forbidden_payload(payload)
        result = cls(
            level=payload.get("level", ""),
            repository_id=payload.get("repository_id", ""),
            state_id=payload.get("state_id", ""),
            tree_id=payload.get("tree_id", ""),
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            supervisor_run_id=payload.get("supervisor_run_id", ""),
            goal_id=payload.get("goal_id", ""),
            objective_id=payload.get("objective_id", ""),
            objective_revision=payload.get("objective_revision", ""),
            task_id=payload.get("task_id", ""),
            attempt=payload.get("attempt", 0),
            stage=payload.get("stage", ""),
            lane=payload.get("lane", ""),
            request_id=payload.get("request_id", ""),
            catalog_revision=payload.get("catalog_revision", ""),
            usage_revision=payload.get("usage_revision", ""),
            endpoint_scope_id=payload.get("endpoint_scope_id", ""),
            caller_id=payload.get("caller_id", ""),
            deadline_at=payload.get("deadline_at", ""),
            idempotency_key=payload.get("idempotency_key", ""),
            lease_id=payload.get("lease_id", ""),
            fence_id=payload.get("fence_id", ""),
            parent_scope_id=payload.get("parent_scope_id", ""),
        )
        _claim(payload, result.scope_id, "scope_id", "content_id")
        return result


def _scope(value: Any) -> SupervisorUsageScope:
    if isinstance(value, SupervisorUsageScope):
        return value
    if isinstance(value, Mapping):
        return SupervisorUsageScope.from_dict(value)
    _fail("scope must be a SupervisorUsageScope")


def _budget(value: Any) -> SupervisorUsageBudget:
    if isinstance(value, SupervisorUsageBudget):
        return value
    if isinstance(value, Mapping):
        return SupervisorUsageBudget.from_dict(value)
    _fail("budget must be a SupervisorUsageBudget")


@dataclass(frozen=True)
class SupervisorUsageEnvelope(_UsageContract):
    """Immutable nested usage envelope with parent-lowering budgets."""

    SCHEMA: ClassVar[str] = SUPERVISOR_USAGE_ENVELOPE_SCHEMA

    scope: SupervisorUsageScope
    budget: SupervisorUsageBudget
    children: tuple["SupervisorUsageEnvelope", ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _scope(self.scope))
        object.__setattr__(self, "budget", _budget(self.budget))
        raw = self.children
        if raw is None:
            raw = ()
        if isinstance(raw, (str, bytes, Mapping)) or not isinstance(raw, Sequence):
            _fail("children must be a sequence")
        if len(raw) > MAX_CHILDREN:
            _fail("envelope exceeds maximum children")
        parsed: list[SupervisorUsageEnvelope] = []
        for item in raw:
            if isinstance(item, SupervisorUsageEnvelope):
                parsed.append(item)
            elif isinstance(item, Mapping):
                parsed.append(SupervisorUsageEnvelope.from_dict(item))
            else:
                _fail("children must contain SupervisorUsageEnvelope records")
        self._validate_tree(tuple(parsed), depth=1)
        object.__setattr__(
            self,
            "children",
            tuple(
                sorted(
                    parsed,
                    key=lambda item: (
                        item.scope.level.value,
                        item.scope.task_id,
                        item.scope.attempt,
                        item.scope.stage,
                        item.scope.lane,
                        item.scope.request_id,
                        item.envelope_id,
                    ),
                )
            ),
        )
        if len(self.canonical_bytes()) > MAX_SERIALIZED_BYTES:
            _fail("usage envelope exceeds its serialized size bound")

    def _validate_tree(
        self, children: tuple["SupervisorUsageEnvelope", ...], *, depth: int
    ) -> None:
        if depth > MAX_NESTING_DEPTH:
            _fail("usage envelope exceeds maximum nesting depth")
        attempt_keys: set[tuple[str, str, int, str, str]] = set()
        request_ids: set[str] = set()
        for child in children:
            if not child.scope.is_child_of(self.scope):
                if not child.scope.shares_binding_with(self.scope):
                    if (
                        child.scope.catalog_revision != self.scope.catalog_revision
                        or child.scope.usage_revision != self.scope.usage_revision
                        or child.scope.tree_id != self.scope.tree_id
                        or child.scope.policy_revision != self.scope.policy_revision
                    ):
                        _fail("stale or foreign ancestry between parent and child")
                    _fail("foreign ancestry between parent and child")
                if child.scope.parent_scope_id != self.scope.scope_id:
                    _fail("missing or foreign parent ancestry")
                if not self.scope.level.may_parent(child.scope.level):
                    _fail("child level is not the immediate descendant of parent")
                _fail("child scope is foreign to parent lineage")
            if not child.budget.is_lower_or_equal(self.budget):
                _fail("child budget widens or raises a parent limit")
            if child.scope.level is SupervisorUsageLevel.ATTEMPT:
                key = (
                    child.scope.task_id,
                    child.scope.stage or "",
                    child.scope.attempt,
                    child.scope.lane or "",
                    child.scope.request_id or "",
                )
                if key in attempt_keys:
                    _fail("duplicate attempt in envelope lineage")
                attempt_keys.add(key)
            if child.scope.level is SupervisorUsageLevel.REQUEST:
                if child.scope.request_id in request_ids:
                    _fail("duplicate request identity in envelope lineage")
                request_ids.add(child.scope.request_id)
            child._validate_tree(child.children, depth=depth + 1)

    @property
    def envelope_id(self) -> str:
        return self.content_id

    def walk(self) -> tuple["SupervisorUsageEnvelope", ...]:
        nodes = [self]
        for child in self.children:
            nodes.extend(child.walk())
        return tuple(nodes)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROVIDER_USAGE_CONTRACT_VERSION,
            "scope": self.scope.to_record(),
            "budget": self.budget.to_record(),
            "children": tuple(item.to_record() for item in self.children),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SupervisorUsageEnvelope":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "scope",
            "budget",
            "children",
            "envelope_id",
            "content_id",
        }
        _closed(payload, schema=cls.SCHEMA, allowed=allowed, name="usage envelope")
        _reject_forbidden_payload(payload)
        result = cls(
            scope=payload.get("scope", {}),
            budget=payload.get("budget", {}),
            children=payload.get("children", ()),
        )
        _claim(payload, result.envelope_id, "envelope_id", "content_id")
        return result


def build_child_envelope(
    parent: SupervisorUsageEnvelope,
    *,
    level: SupervisorUsageLevel | str,
    budget: SupervisorUsageBudget | Mapping[str, Any],
    children: Sequence[SupervisorUsageEnvelope | Mapping[str, Any]] = (),
    **scope_overrides: Any,
) -> SupervisorUsageEnvelope:
    """Construct a child that lowers the parent budget and refines lineage."""

    level_value = _enum(level, SupervisorUsageLevel, "level")
    if not parent.scope.level.may_parent(level_value):
        _fail("child level is not the immediate descendant of parent")
    parent_scope = parent.scope
    fields = {
        "level": level_value,
        "repository_id": parent_scope.repository_id,
        "state_id": parent_scope.state_id,
        "tree_id": parent_scope.tree_id,
        "policy_id": parent_scope.policy_id,
        "policy_revision": parent_scope.policy_revision,
        "supervisor_run_id": parent_scope.supervisor_run_id,
        "goal_id": parent_scope.goal_id,
        "objective_id": parent_scope.objective_id,
        "objective_revision": parent_scope.objective_revision,
        "task_id": parent_scope.task_id,
        "attempt": parent_scope.attempt,
        "stage": parent_scope.stage,
        "lane": parent_scope.lane,
        "request_id": parent_scope.request_id,
        "catalog_revision": parent_scope.catalog_revision,
        "usage_revision": parent_scope.usage_revision,
        "endpoint_scope_id": parent_scope.endpoint_scope_id,
        "caller_id": parent_scope.caller_id,
        "deadline_at": parent_scope.deadline_at,
        "idempotency_key": parent_scope.idempotency_key,
        "lease_id": parent_scope.lease_id,
        "fence_id": parent_scope.fence_id,
        "parent_scope_id": parent_scope.scope_id,
    }
    fields.update(scope_overrides)
    child_scope = SupervisorUsageScope(**fields)
    if not child_scope.is_child_of(parent.scope):
        if not child_scope.shares_binding_with(parent.scope):
            if (
                child_scope.catalog_revision != parent.scope.catalog_revision
                or child_scope.usage_revision != parent.scope.usage_revision
                or child_scope.tree_id != parent.scope.tree_id
                or child_scope.policy_revision != parent.scope.policy_revision
            ):
                _fail("stale or foreign ancestry between parent and child")
            _fail("foreign ancestry between parent and child")
        _fail("child scope is foreign to parent lineage")
    child_budget = _budget(budget)
    if not child_budget.is_lower_or_equal(parent.budget):
        _fail("child budget widens or raises a parent limit")
    return SupervisorUsageEnvelope(
        scope=child_scope,
        budget=child_budget,
        children=tuple(children),  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class SupervisorToEndpointRequest(_UsageContract):
    """Supervisor request bridge into the endpoint usage coordinator."""

    SCHEMA: ClassVar[str] = SUPERVISOR_TO_ENDPOINT_REQUEST_SCHEMA

    scope: SupervisorUsageScope
    envelope_id: str
    endpoint_scope_id: str
    catalog_revision: str
    usage_revision: str
    estimated: UsageVector
    request_id: str
    attempt: int
    idempotency_key: str
    caller_id: str
    deadline_at: str
    lease_id: str
    fence_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _scope(self.scope))
        if self.scope.level is not SupervisorUsageLevel.REQUEST:
            _fail("endpoint bridge request requires request-level scope")
        for name in (
            "envelope_id",
            "endpoint_scope_id",
            "catalog_revision",
            "usage_revision",
            "request_id",
            "idempotency_key",
            "caller_id",
            "deadline_at",
            "lease_id",
            "fence_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "attempt",
            _integer(self.attempt, "attempt", minimum=1, maximum=100_000),
        )
        object.__setattr__(self, "estimated", _usage_vector(self.estimated))
        if self.scope.request_id != self.request_id:
            _fail("request_id is foreign to supervisor scope")
        if self.scope.endpoint_scope_id != self.endpoint_scope_id:
            _fail("endpoint_scope_id is foreign to supervisor scope")
        if self.scope.catalog_revision != self.catalog_revision:
            _fail("catalog_revision is stale relative to supervisor scope")
        if self.scope.usage_revision != self.usage_revision:
            _fail("usage_revision is stale relative to supervisor scope")
        if self.scope.attempt != self.attempt:
            _fail("attempt is foreign to supervisor scope")
        if self.scope.idempotency_key != self.idempotency_key:
            _fail("idempotency_key is foreign to supervisor scope")
        if self.scope.caller_id != self.caller_id:
            _fail("caller_id is foreign to supervisor scope")
        if self.scope.lease_id != self.lease_id or self.scope.fence_id != self.fence_id:
            _fail("lease or fence is foreign to supervisor scope")
        _reject_forbidden_payload(self._payload())

    @property
    def bridge_request_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROVIDER_USAGE_CONTRACT_VERSION,
            "scope": self.scope.to_record(),
            "envelope_id": self.envelope_id,
            "endpoint_scope_id": self.endpoint_scope_id,
            "catalog_revision": self.catalog_revision,
            "usage_revision": self.usage_revision,
            "estimated": self.estimated.to_dict(),
            "request_id": self.request_id,
            "attempt": self.attempt,
            "idempotency_key": self.idempotency_key,
            "caller_id": self.caller_id,
            "deadline_at": self.deadline_at,
            "lease_id": self.lease_id,
            "fence_id": self.fence_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SupervisorToEndpointRequest":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "scope",
            "envelope_id",
            "endpoint_scope_id",
            "catalog_revision",
            "usage_revision",
            "estimated",
            "request_id",
            "attempt",
            "idempotency_key",
            "caller_id",
            "deadline_at",
            "lease_id",
            "fence_id",
            "bridge_request_id",
            "content_id",
        }
        _closed(
            payload,
            schema=cls.SCHEMA,
            allowed=allowed,
            name="supervisor-to-endpoint request",
        )
        result = cls(
            scope=payload.get("scope", {}),
            envelope_id=payload.get("envelope_id", ""),
            endpoint_scope_id=payload.get("endpoint_scope_id", ""),
            catalog_revision=payload.get("catalog_revision", ""),
            usage_revision=payload.get("usage_revision", ""),
            estimated=payload.get("estimated", {}),
            request_id=payload.get("request_id", ""),
            attempt=payload.get("attempt", 0),
            idempotency_key=payload.get("idempotency_key", ""),
            caller_id=payload.get("caller_id", ""),
            deadline_at=payload.get("deadline_at", ""),
            lease_id=payload.get("lease_id", ""),
            fence_id=payload.get("fence_id", ""),
        )
        _claim(payload, result.bridge_request_id, "bridge_request_id", "content_id")
        return result


@dataclass(frozen=True)
class SupervisorUsageReceipt(_UsageContract):
    """Bounded settlement receipt joining supervisor scope to endpoint events."""

    SCHEMA: ClassVar[str] = SUPERVISOR_USAGE_RECEIPT_SCHEMA

    scope: SupervisorUsageScope
    envelope_id: str
    request_id: str
    endpoint_scope_id: str
    catalog_revision: str
    usage_revision: str
    reservation_id: str
    endpoint_event_ids: tuple[str, ...]
    settled: UsageVector
    final_status: SupervisorUsageFinalStatus
    authorizes_usage: bool = False
    rewrites_provider_settlement: bool = False
    is_completion_evidence: bool = False
    is_correctness_evidence: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _scope(self.scope))
        for name in (
            "envelope_id",
            "request_id",
            "endpoint_scope_id",
            "catalog_revision",
            "usage_revision",
            "reservation_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "final_status",
            _enum(self.final_status, SupervisorUsageFinalStatus, "final_status"),
        )
        object.__setattr__(self, "settled", _usage_vector(self.settled))
        raw_ids = self.endpoint_event_ids
        if raw_ids is None:
            raw_ids = ()
        if isinstance(raw_ids, (str, bytes, Mapping)) or not isinstance(raw_ids, Sequence):
            _fail("endpoint_event_ids must be a sequence")
        if len(raw_ids) > MAX_EVENT_IDS:
            _fail("endpoint_event_ids exceeds its bound")
        ids = tuple(_text(item, "endpoint_event_id") for item in raw_ids)
        if len(ids) != len(set(ids)):
            _fail("endpoint events must be referenced exactly once")
        object.__setattr__(self, "endpoint_event_ids", ids)
        for flag_name, expected in (
            ("authorizes_usage", BRIDGE_AUTHORIZES_USAGE),
            ("rewrites_provider_settlement", BRIDGE_REWRITES_PROVIDER_SETTLEMENT),
            ("is_completion_evidence", BRIDGE_IS_COMPLETION_EVIDENCE),
            ("is_correctness_evidence", BRIDGE_IS_CORRECTNESS_EVIDENCE),
        ):
            value = getattr(self, flag_name)
            if not isinstance(value, bool):
                _fail(f"{flag_name} must be boolean")
            if value is not expected:
                _fail(
                    f"{flag_name} cannot be true; usage receipts are operational only"
                )
        if self.scope.request_id != self.request_id:
            _fail("receipt request_id is foreign to supervisor scope")
        if self.scope.endpoint_scope_id != self.endpoint_scope_id:
            _fail("receipt endpoint_scope_id is foreign to supervisor scope")
        if self.scope.catalog_revision != self.catalog_revision:
            _fail("receipt catalog_revision is stale")
        if self.scope.usage_revision != self.usage_revision:
            _fail("receipt usage_revision is stale")
        _reject_forbidden_payload(self._payload())

    @property
    def receipt_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROVIDER_USAGE_CONTRACT_VERSION,
            "scope": self.scope.to_record(),
            "envelope_id": self.envelope_id,
            "request_id": self.request_id,
            "endpoint_scope_id": self.endpoint_scope_id,
            "catalog_revision": self.catalog_revision,
            "usage_revision": self.usage_revision,
            "reservation_id": self.reservation_id,
            "endpoint_event_ids": self.endpoint_event_ids,
            "settled": self.settled.to_dict(),
            "final_status": self.final_status.value,
            "authorizes_usage": self.authorizes_usage,
            "rewrites_provider_settlement": self.rewrites_provider_settlement,
            "is_completion_evidence": self.is_completion_evidence,
            "is_correctness_evidence": self.is_correctness_evidence,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SupervisorUsageReceipt":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "scope",
            "envelope_id",
            "request_id",
            "endpoint_scope_id",
            "catalog_revision",
            "usage_revision",
            "reservation_id",
            "endpoint_event_ids",
            "settled",
            "final_status",
            "authorizes_usage",
            "rewrites_provider_settlement",
            "is_completion_evidence",
            "is_correctness_evidence",
            "receipt_id",
            "content_id",
        }
        _closed(payload, schema=cls.SCHEMA, allowed=allowed, name="usage receipt")
        result = cls(
            scope=payload.get("scope", {}),
            envelope_id=payload.get("envelope_id", ""),
            request_id=payload.get("request_id", ""),
            endpoint_scope_id=payload.get("endpoint_scope_id", ""),
            catalog_revision=payload.get("catalog_revision", ""),
            usage_revision=payload.get("usage_revision", ""),
            reservation_id=payload.get("reservation_id", ""),
            endpoint_event_ids=payload.get("endpoint_event_ids", ()),
            settled=payload.get("settled", {}),
            final_status=payload.get("final_status", ""),
            authorizes_usage=payload.get("authorizes_usage", False),
            rewrites_provider_settlement=payload.get(
                "rewrites_provider_settlement", False
            ),
            is_completion_evidence=payload.get("is_completion_evidence", False),
            is_correctness_evidence=payload.get("is_correctness_evidence", False),
        )
        _claim(payload, result.receipt_id, "receipt_id", "content_id")
        return result


@dataclass(frozen=True)
class SupervisorUsageAttribution(_UsageContract):
    """One-to-one join of supervisor lifecycle work to a reconciled endpoint event."""

    SCHEMA: ClassVar[str] = SUPERVISOR_USAGE_ATTRIBUTION_SCHEMA

    scope: SupervisorUsageScope
    endpoint_event_id: str
    endpoint_scope_id: str
    lifecycle_event_id: str
    settled: UsageVector
    request_id: str
    reservation_id: str = ""
    supersedes_event_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _scope(self.scope))
        for name in (
            "endpoint_event_id",
            "endpoint_scope_id",
            "lifecycle_event_id",
            "request_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "reservation_id", _optional_text(self.reservation_id, "reservation_id")
        )
        object.__setattr__(
            self,
            "supersedes_event_id",
            _optional_text(self.supersedes_event_id, "supersedes_event_id"),
        )
        object.__setattr__(self, "settled", _usage_vector(self.settled))
        if self.scope.endpoint_scope_id and self.scope.endpoint_scope_id != self.endpoint_scope_id:
            _fail("attribution endpoint_scope_id is foreign to supervisor scope")
        if self.scope.request_id and self.scope.request_id != self.request_id:
            _fail("attribution request_id is foreign to supervisor scope")
        _reject_forbidden_payload(self._payload())

    @property
    def attribution_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROVIDER_USAGE_CONTRACT_VERSION,
            "scope": self.scope.to_record(),
            "endpoint_event_id": self.endpoint_event_id,
            "endpoint_scope_id": self.endpoint_scope_id,
            "lifecycle_event_id": self.lifecycle_event_id,
            "settled": self.settled.to_dict(),
            "request_id": self.request_id,
            "reservation_id": self.reservation_id,
            "supersedes_event_id": self.supersedes_event_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SupervisorUsageAttribution":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "scope",
            "endpoint_event_id",
            "endpoint_scope_id",
            "lifecycle_event_id",
            "settled",
            "request_id",
            "reservation_id",
            "supersedes_event_id",
            "attribution_id",
            "content_id",
        }
        _closed(
            payload, schema=cls.SCHEMA, allowed=allowed, name="usage attribution"
        )
        result = cls(
            scope=payload.get("scope", {}),
            endpoint_event_id=payload.get("endpoint_event_id", ""),
            endpoint_scope_id=payload.get("endpoint_scope_id", ""),
            lifecycle_event_id=payload.get("lifecycle_event_id", ""),
            settled=payload.get("settled", {}),
            request_id=payload.get("request_id", ""),
            reservation_id=payload.get("reservation_id", ""),
            supersedes_event_id=payload.get("supersedes_event_id", ""),
        )
        _claim(payload, result.attribution_id, "attribution_id", "content_id")
        return result


def consume_reconciled_endpoint_events_exactly_once(
    events: Sequence[UsageEvent | Mapping[str, Any]],
) -> tuple[UsageEvent, ...]:
    """Normalize settlement events and reject missing or duplicate identities."""

    if events is None:
        _fail("endpoint events must be a sequence")
    if isinstance(events, (str, bytes, Mapping)) or not isinstance(events, Sequence):
        _fail("endpoint events must be a sequence")
    if len(events) > MAX_EVENT_IDS:
        _fail("endpoint event population exceeds its bound")
    parsed: list[UsageEvent] = []
    seen: set[str] = set()
    for item in events:
        event = _usage_event(item)
        event_id = event.event_id or ""
        if not event_id:
            _fail("endpoint event is missing event_id")
        if event_id in seen:
            _fail("endpoint event population contains duplicated event_id")
        seen.add(event_id)
        if event.kind not in _SETTLEMENT_KINDS:
            _fail(
                f"endpoint event kind {event.kind.value} is not a reconciled settlement"
            )
        # Projected units must be finite and non-negative; unknown is not a charge.
        for entry in event.units.entries:
            if entry.amount.kind is QuantityKind.FINITE:
                if entry.amount.value is None or entry.amount.value < 0:
                    _fail("endpoint event contains negative or missing units")
            elif entry.amount.kind is QuantityKind.UNLIMITED:
                _fail("settled endpoint units cannot be unlimited")
        parsed.append(event)
    return tuple(sorted(parsed, key=lambda item: (item.sequence or 0, item.event_id or "")))


def attribute_endpoint_events(
    *,
    scope: SupervisorUsageScope,
    events: Sequence[UsageEvent | Mapping[str, Any]],
    lifecycle_event_ids: Sequence[str],
) -> tuple[SupervisorUsageAttribution, ...]:
    """Join each reconciled endpoint event to exactly one lifecycle identity."""

    consumed = consume_reconciled_endpoint_events_exactly_once(events)
    if len(lifecycle_event_ids) != len(consumed):
        _fail("lifecycle and endpoint event populations must match one-to-one")
    life_ids = [_text(item, "lifecycle_event_id") for item in lifecycle_event_ids]
    if len(life_ids) != len(set(life_ids)):
        _fail("lifecycle event population contains duplicates")
    attributions = []
    for event, life_id in zip(consumed, life_ids):
        attributions.append(
            SupervisorUsageAttribution(
                scope=scope,
                endpoint_event_id=event.event_id or "",
                endpoint_scope_id=event.scope_id,
                lifecycle_event_id=life_id,
                settled=event.units,
                request_id=event.request_id or scope.request_id,
                reservation_id=event.reservation_id or "",
                supersedes_event_id=event.supersedes_event_id or "",
            )
        )
    return tuple(attributions)


def finite_units(
    vector: UsageVector,
    dimension: UsageDimension,
    *,
    currency: Optional[str] = None,
) -> int:
    """Return the finite amount for *dimension* or zero when absent/unknown."""

    if dimension is UsageDimension.COST_MICROS and currency is None:
        matches = [
            entry
            for entry in vector.entries
            if entry.dimension is UsageDimension.COST_MICROS
        ]
        if not matches:
            return 0
        if len({entry.currency for entry in matches}) > 1:
            _fail("endpoint cost_micros mixes currencies")
        entry = matches[0]
    else:
        entry = vector.get(dimension, currency=currency)
    if entry is None or entry.amount.kind is not QuantityKind.FINITE:
        return 0
    return int(entry.amount.value or 0)


def discover_schemas() -> dict[str, str]:
    """Provider-free schema discovery for supervisor usage contracts."""

    return {
        "requirement_id": SUPERVISOR_USAGE_ENVELOPE_REQUIREMENT_ID,
        "goal_id": SUPERVISOR_USAGE_ENVELOPE_GOAL_ID,
        "contract_version": str(PROVIDER_USAGE_CONTRACT_VERSION),
        "scope": SUPERVISOR_USAGE_SCOPE_SCHEMA,
        "budget": SUPERVISOR_USAGE_BUDGET_SCHEMA,
        "budget_limit": SUPERVISOR_BUDGET_LIMIT_SCHEMA,
        "envelope": SUPERVISOR_USAGE_ENVELOPE_SCHEMA,
        "attribution": SUPERVISOR_USAGE_ATTRIBUTION_SCHEMA,
        "request_bridge": SUPERVISOR_TO_ENDPOINT_REQUEST_SCHEMA,
        "receipt": SUPERVISOR_USAGE_RECEIPT_SCHEMA,
        "authorizes_usage": str(BRIDGE_AUTHORIZES_USAGE).lower(),
        "rewrites_provider_settlement": str(
            BRIDGE_REWRITES_PROVIDER_SETTLEMENT
        ).lower(),
        "is_completion_evidence": str(BRIDGE_IS_COMPLETION_EVIDENCE).lower(),
        "is_correctness_evidence": str(BRIDGE_IS_CORRECTNESS_EVIDENCE).lower(),
    }


def accounting_bridge_bounds() -> dict[str, bool]:
    """Explicit non-authority bounds for ledger and efficiency consumers."""

    return {
        "authorizes_usage": BRIDGE_AUTHORIZES_USAGE,
        "rewrites_provider_settlement": BRIDGE_REWRITES_PROVIDER_SETTLEMENT,
        "is_completion_evidence": BRIDGE_IS_COMPLETION_EVIDENCE,
        "is_correctness_evidence": BRIDGE_IS_CORRECTNESS_EVIDENCE,
    }


__all__ = [
    "BRIDGE_AUTHORIZES_USAGE",
    "BRIDGE_IS_COMPLETION_EVIDENCE",
    "BRIDGE_IS_CORRECTNESS_EVIDENCE",
    "BRIDGE_REWRITES_PROVIDER_SETTLEMENT",
    "MAX_NESTING_DEPTH",
    "PROVIDER_USAGE_CONTRACT_VERSION",
    "SCHEMA_VERSION",
    "SUPERVISOR_USAGE_ENVELOPE_GOAL_ID",
    "SUPERVISOR_USAGE_ENVELOPE_REQUIREMENT_ID",
    "ProviderUsageValidationError",
    "SupervisorBudgetLimit",
    "SupervisorToEndpointRequest",
    "SupervisorUsageAttribution",
    "SupervisorUsageBudget",
    "SupervisorUsageEnvelope",
    "SupervisorUsageFinalStatus",
    "SupervisorUsageLevel",
    "SupervisorUsageReceipt",
    "SupervisorUsageScope",
    "accounting_bridge_bounds",
    "attribute_endpoint_events",
    "build_child_envelope",
    "consume_reconciled_endpoint_events_exactly_once",
    "discover_schemas",
    "finite_units",
]
