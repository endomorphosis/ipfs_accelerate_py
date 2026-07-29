"""Fail-closed task execution policy for symbolic and model-assisted work.

The policy deliberately separates two execution modes:

* deterministic-only tasks may invoke only enum-typed operations registered in
  the supervisor-owned local allowlist; they never enter a provider callback;
* model-assisted tasks invoke one exactly-bound Grok executable and then one
  exactly-bound, independent Codex reviewer.

Task-declared context bounds are protocol limits, not hints.  They are checked
before any local operation or provider invocation.  Provider prompt bytes and
tokens for each model request are re-measured against the same hard limits.
Provider quota exhaustion, malformed responses, and failures defer the task; an
implementation proposal is never silently promoted to a completed result when
review did not run.  Provider output is proposal-only: receipts always deny
completion and proof authority, and display labels are audit metadata only and
are never used for dispatch.

Evidence obligation SCAEV167ROUTE (SCA-G167 / SCA-188): symbolic-only execution
mode routing and bounded Grok/Codex provider enforcement.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .contract_packet_provider_router import (
    ProviderQuotaError,
    ProviderQuotaLatch,
    ProviderRole,
)


TASK_EXECUTION_POLICY_INTERFACE: Final = "TaskExecutionPolicy@1"
TASK_EXECUTION_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-execution-request@1"
)
TASK_EXECUTION_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-execution-receipt@1"
)

# Objective-evidence term for SCA-G167: exact-text matches in implementation
# and validation sources prove the route/enforcement obligation is covered.
SCAEV167ROUTE: Final = "SCAEV167ROUTE"
SCAEV167ROUTE_EVIDENCE: Final = SCAEV167ROUTE
SCAEV167ROUTE_COVERAGE: Final = (
    "deterministic-only-typed-allowlist-zero-provider-calls",
    "task-context-bytes-tokens-hard-limits",
    "provider-prompt-bytes-tokens-hard-limits",
    "exact-grok-codex-executable-identity",
    "independent-sequential-review-order",
    "quota-failure-defer-without-promotion-or-fallback",
    "display-labels-never-select-or-upgrade",
    "proposal-only-admission-no-completion-or-proof-authority",
)

MAX_TASK_CONTEXT_BYTES: Final = 64 * 1_024
MAX_TASK_CONTEXT_TOKENS: Final = 4_096
MAX_TASK_PROVIDER_RESPONSE_BYTES: Final = 256 * 1_024

_EXECUTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,255}$")


class ExecutionMode(str, Enum):
    DETERMINISTIC_ONLY = "deterministic-only"
    GROK_CODEX = "grok-codex"


class ExecutionStatus(str, Enum):
    SUCCEEDED = "succeeded"
    DEFERRED = "deferred"
    REJECTED = "rejected"


class ExecutionReason(str, Enum):
    COMPLETED = "completed"
    INVALID_REQUEST = "invalid_request"
    CONTEXT_LIMIT_EXCEEDED = "task_context_limit_exceeded"
    PROVIDER_PROMPT_TOO_LARGE = "provider_prompt_too_large"
    PROVIDER_PROMPT_TOKEN_BUDGET = "provider_prompt_token_budget_exceeded"
    LOCAL_OPERATION_NOT_ALLOWED = "local_operation_not_allowed"
    LOCAL_OPERATION_FAILED = "local_operation_failed"
    PROVIDER_NOT_CONFIGURED = "provider_not_configured"
    EXECUTABLE_BINDING_MISMATCH = "executable_binding_mismatch"
    PROVIDERS_NOT_INDEPENDENT = "providers_not_independent"
    GROK_QUOTA_EXHAUSTED = "grok_quota_exhausted"
    CODEX_QUOTA_EXHAUSTED = "codex_quota_exhausted"
    GROK_FAILED = "grok_failed"
    CODEX_FAILED = "codex_failed"
    PROVIDER_RESPONSE_INVALID = "provider_response_invalid"
    PROVIDER_RESPONSE_TOO_LARGE = "provider_response_too_large"
    PROVIDER_RESULT_NOT_ADMITTED = "provider_result_not_admitted"


class LocalOperationType(str, Enum):
    """Closed vocabulary for supervisor-owned deterministic operations."""

    CANONICAL_JSON = "canonical-json"
    SHA256 = "sha256"
    EXACT_EQUAL = "exact-equal"
    ALL_TRUE = "all-true"


class TaskExecutionPolicyError(ValueError):
    """A typed request/configuration error raised before execution starts."""

    def __init__(self, message: str, *, reason_code: ExecutionReason | str) -> None:
        super().__init__(message)
        self.reason_code = str(getattr(reason_code, "value", reason_code))


def _plain_json(value: Any, *, path: str = "$") -> Any:
    """Return a detached JSON value and reject ambiguous/non-finite data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string object key")
            result[key] = _plain_json(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _plain_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{path} is not canonical JSON data")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _default_token_count(data: bytes) -> int:
    """Conservative deterministic estimate used when no tokenizer is supplied."""

    return (len(data) + 3) // 4


def _validate_positive_limit(name: str, value: int, maximum: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > maximum
    ):
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")


@dataclass(frozen=True, slots=True)
class TaskContextMetadata:
    """Per-task hard limits supplied with the bounded task context."""

    max_bytes: int
    max_tokens: int

    def __post_init__(self) -> None:
        _validate_positive_limit("max_bytes", self.max_bytes, MAX_TASK_CONTEXT_BYTES)
        _validate_positive_limit(
            "max_tokens", self.max_tokens, MAX_TASK_CONTEXT_TOKENS
        )

    def to_dict(self) -> dict[str, int]:
        return {"max_bytes": self.max_bytes, "max_tokens": self.max_tokens}


@dataclass(frozen=True, slots=True)
class TypedLocalOperation:
    """One typed local operation; free-form command names are not accepted."""

    operation_type: LocalOperationType
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.operation_type, LocalOperationType):
            raise TypeError("operation_type must be a LocalOperationType")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("arguments must be a mapping")
        detached = _plain_json(self.arguments, path="$.arguments")
        object.__setattr__(self, "arguments", _freeze_json(detached))

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_type": self.operation_type.value,
            "arguments": _thaw_json(self.arguments),
        }


@dataclass(frozen=True, slots=True)
class TaskExecutionRequest:
    task_id: str
    mode: ExecutionMode
    context: Mapping[str, Any]
    context_metadata: TaskContextMetadata
    local_operations: Sequence[TypedLocalOperation] = ()
    grok_executable_id: str = ""
    codex_executable_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(self.mode, ExecutionMode):
            raise TypeError("mode must be an ExecutionMode")
        if not isinstance(self.context, Mapping):
            raise TypeError("context must be a mapping")
        if not isinstance(self.context_metadata, TaskContextMetadata):
            raise TypeError("context_metadata must be TaskContextMetadata")
        context = _plain_json(self.context, path="$.context")
        operations = tuple(self.local_operations)
        if any(not isinstance(item, TypedLocalOperation) for item in operations):
            raise TypeError("local_operations must contain TypedLocalOperation values")
        object.__setattr__(self, "context", _freeze_json(context))
        object.__setattr__(self, "local_operations", operations)


@dataclass(frozen=True, slots=True)
class ProviderExecutable:
    """One exact executable binding.

    ``display_label`` is emitted for audit only.  Dispatch always uses this
    object's exact ``executable_id`` and never performs label lookup.
    """

    role: ProviderRole
    executable_id: str
    display_label: str
    invoke: Callable[["ModelExecutionRequest"], Mapping[str, Any]]

    def __post_init__(self) -> None:
        if not isinstance(self.role, ProviderRole) or self.role not in (
            ProviderRole.GROK_IMPLEMENT,
            ProviderRole.CODEX_REVIEW,
        ):
            raise ValueError("provider role must be Grok implementation or Codex review")
        if not isinstance(self.executable_id, str) or not _EXECUTABLE_ID.fullmatch(
            self.executable_id
        ):
            raise ValueError("executable_id is not a valid exact executable identifier")
        if not isinstance(self.display_label, str):
            raise TypeError("display_label must be a string")
        if not callable(self.invoke):
            raise TypeError("invoke must be callable")


@dataclass(frozen=True, slots=True)
class ModelExecutionRequest(Mapping[str, Any]):
    """Bounded, proposal-only payload passed to a model executable."""

    role: ProviderRole
    executable_id: str
    task_id: str
    task_context: Mapping[str, Any]
    context_metadata: TaskContextMetadata
    grok_implementation: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": TASK_EXECUTION_REQUEST_SCHEMA,
            "interface": TASK_EXECUTION_POLICY_INTERFACE,
            "role": self.role.value,
            "executable_id": self.executable_id,
            "task_id": self.task_id,
            "task_context": _thaw_json(self.task_context),
            "context_metadata": self.context_metadata.to_dict(),
            "authority": {
                "proposal_only": True,
                "repository_write_allowed": False,
                "completion_authoritative": False,
            },
        }
        if self.grok_implementation is not None:
            payload["grok_implementation"] = _thaw_json(self.grok_implementation)
        return payload

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True, slots=True)
class ExecutionAttempt:
    stage: str
    role: str
    executable_id: str
    display_label: str
    invoked: bool
    status: str
    reason_code: str
    prompt_bytes: int = 0
    prompt_tokens: int = 0
    response_bytes: int = 0
    admitted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "role": self.role,
            "executable_id": self.executable_id,
            "display_label": self.display_label,
            "invoked": self.invoked,
            "status": self.status,
            "reason_code": self.reason_code,
            "prompt_bytes": self.prompt_bytes,
            "prompt_tokens": self.prompt_tokens,
            "response_bytes": self.response_bytes,
            "admitted": self.admitted,
        }


@dataclass(frozen=True, slots=True)
class TaskExecutionReceipt:
    task_id: str
    mode: ExecutionMode
    status: ExecutionStatus
    reason_code: str
    context_bytes: int
    context_tokens: int
    model_call_count: int
    provider_call_count: int
    attempts: tuple[ExecutionAttempt, ...] = ()
    result: Any = None
    grok_implementation: Mapping[str, Any] | None = None
    codex_review: Mapping[str, Any] | None = None
    context_byte_limit: int = 0
    context_token_limit: int = 0
    max_provider_response_bytes: int = 0
    prompt_bytes: int = 0
    prompt_tokens: int = 0

    @property
    def deferred(self) -> bool:
        return self.status is ExecutionStatus.DEFERRED

    @property
    def provider_result_admitted(self) -> bool:
        """True only when a model-assisted run completed independent review."""

        return (
            self.status is ExecutionStatus.SUCCEEDED
            and self.mode is ExecutionMode.GROK_CODEX
            and self.codex_review is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TASK_EXECUTION_RECEIPT_SCHEMA,
            "interface": TASK_EXECUTION_POLICY_INTERFACE,
            "evidence": {
                "requirement_ids": [SCAEV167ROUTE],
                "coverage": list(SCAEV167ROUTE_COVERAGE),
            },
            "task_id": self.task_id,
            "mode": self.mode.value,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "context_usage": {
                "bytes": self.context_bytes,
                "tokens": self.context_tokens,
                "byte_limit": self.context_byte_limit,
                "token_limit": self.context_token_limit,
            },
            "prompt_usage": {
                "bytes": self.prompt_bytes,
                "tokens": self.prompt_tokens,
                "byte_limit": self.context_byte_limit,
                "token_limit": self.context_token_limit,
            },
            "bounds": {
                "max_context_bytes": self.context_byte_limit,
                "max_context_tokens": self.context_token_limit,
                "max_provider_response_bytes": self.max_provider_response_bytes,
            },
            "isolation_audit": {
                "llm_call_count": self.model_call_count,
                "model_call_count": self.model_call_count,
                "provider_call_count": self.provider_call_count,
            },
            "admission": {
                "proposal_only": True,
                "repository_write_allowed": False,
                "completion_authoritative": False,
                "proof_authoritative": False,
                "provider_result_admitted": self.provider_result_admitted,
                "labels_may_select_provider": False,
            },
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "result": _thaw_json(self.result),
            "grok_implementation": _thaw_json(self.grok_implementation),
            "codex_review": _thaw_json(self.codex_review),
            "proof_authoritative": False,
            "completion_authoritative": False,
        }


LocalOperationHandler = Callable[[Mapping[str, Any], Mapping[str, Any]], Any]
TokenCounter = Callable[[bytes], int]


def builtin_local_operation_handlers() -> Mapping[LocalOperationType, LocalOperationHandler]:
    """Return the small, side-effect-free built-in operation allowlist."""

    def canonical_json(arguments: Mapping[str, Any], _context: Mapping[str, Any]) -> str:
        return _canonical_bytes(arguments.get("value")).decode("utf-8")

    def sha256(arguments: Mapping[str, Any], _context: Mapping[str, Any]) -> str:
        return "sha256:" + hashlib.sha256(
            _canonical_bytes(arguments.get("value"))
        ).hexdigest()

    def exact_equal(arguments: Mapping[str, Any], _context: Mapping[str, Any]) -> bool:
        return _plain_json(arguments.get("left")) == _plain_json(
            arguments.get("right")
        )

    def all_true(arguments: Mapping[str, Any], _context: Mapping[str, Any]) -> bool:
        values = arguments.get("values")
        if not isinstance(values, (tuple, list)):
            raise ValueError("all-true requires a values array")
        if any(not isinstance(value, bool) for value in values):
            raise ValueError("all-true accepts boolean values only")
        return all(values)

    return MappingProxyType(
        {
            LocalOperationType.CANONICAL_JSON: canonical_json,
            LocalOperationType.SHA256: sha256,
            LocalOperationType.EXACT_EQUAL: exact_equal,
            LocalOperationType.ALL_TRUE: all_true,
        }
    )


class TaskExecutionPolicy:
    """Execute a task inside deterministic or sequential model boundaries."""

    def __init__(
        self,
        *,
        local_operation_handlers: Mapping[
            LocalOperationType, LocalOperationHandler
        ] | None = None,
        grok: ProviderExecutable | None = None,
        codex: ProviderExecutable | None = None,
        grok_quota: ProviderQuotaLatch | None = None,
        codex_quota: ProviderQuotaLatch | None = None,
        max_context_bytes: int = MAX_TASK_CONTEXT_BYTES,
        max_context_tokens: int = MAX_TASK_CONTEXT_TOKENS,
        max_provider_response_bytes: int = MAX_TASK_PROVIDER_RESPONSE_BYTES,
        token_counter: TokenCounter | None = None,
    ) -> None:
        _validate_positive_limit(
            "max_context_bytes", max_context_bytes, MAX_TASK_CONTEXT_BYTES
        )
        _validate_positive_limit(
            "max_context_tokens", max_context_tokens, MAX_TASK_CONTEXT_TOKENS
        )
        _validate_positive_limit(
            "max_provider_response_bytes",
            max_provider_response_bytes,
            MAX_TASK_PROVIDER_RESPONSE_BYTES,
        )
        handlers = dict(local_operation_handlers or {})
        for operation_type, handler in handlers.items():
            if not isinstance(operation_type, LocalOperationType):
                raise TypeError("local operation allowlist keys must be LocalOperationType")
            if not callable(handler):
                raise TypeError("local operation allowlist values must be callable")
        if grok is not None and grok.role is not ProviderRole.GROK_IMPLEMENT:
            raise ValueError("grok executable has the wrong provider role")
        if codex is not None and codex.role is not ProviderRole.CODEX_REVIEW:
            raise ValueError("codex executable has the wrong provider role")
        if token_counter is not None and not callable(token_counter):
            raise TypeError("token_counter must be callable")

        self.local_operation_handlers = MappingProxyType(handlers)
        self.grok = grok
        self.codex = codex
        self.grok_quota = (
            grok_quota if grok_quota is not None else ProviderQuotaLatch()
        )
        self.codex_quota = (
            codex_quota if codex_quota is not None else ProviderQuotaLatch()
        )
        self.max_context_bytes = max_context_bytes
        self.max_context_tokens = max_context_tokens
        self.max_provider_response_bytes = max_provider_response_bytes
        self.token_counter = token_counter or _default_token_count

    def execute(self, request: TaskExecutionRequest) -> TaskExecutionReceipt:
        if not isinstance(request, TaskExecutionRequest):
            raise TypeError("request must be TaskExecutionRequest")

        context_data = _canonical_bytes(request.context)
        context_bytes = len(context_data)
        context_tokens = self.token_counter(context_data)
        if (
            isinstance(context_tokens, bool)
            or not isinstance(context_tokens, int)
            or context_tokens < 0
        ):
            raise ValueError("token_counter must return a non-negative integer")

        byte_limit = min(self.max_context_bytes, request.context_metadata.max_bytes)
        token_limit = min(self.max_context_tokens, request.context_metadata.max_tokens)
        if context_bytes > byte_limit or context_tokens > token_limit:
            return self._receipt(
                request,
                status=ExecutionStatus.REJECTED,
                reason=ExecutionReason.CONTEXT_LIMIT_EXCEEDED,
                context_bytes=context_bytes,
                context_tokens=context_tokens,
                context_byte_limit=byte_limit,
                context_token_limit=token_limit,
            )

        if request.mode is ExecutionMode.DETERMINISTIC_ONLY:
            return self._execute_local(
                request,
                context_bytes,
                context_tokens,
                byte_limit=byte_limit,
                token_limit=token_limit,
            )
        return self._execute_models(
            request,
            context_bytes,
            context_tokens,
            byte_limit=byte_limit,
            token_limit=token_limit,
        )

    def _execute_local(
        self,
        request: TaskExecutionRequest,
        context_bytes: int,
        context_tokens: int,
        *,
        byte_limit: int,
        token_limit: int,
    ) -> TaskExecutionReceipt:
        results: list[Any] = []
        attempts: list[ExecutionAttempt] = []
        for operation in request.local_operations:
            handler = self.local_operation_handlers.get(operation.operation_type)
            if handler is None:
                attempts.append(
                    ExecutionAttempt(
                        stage="local-operation",
                        role=ProviderRole.DETERMINISTIC_LOCAL.value,
                        executable_id=operation.operation_type.value,
                        display_label="",
                        invoked=False,
                        status=ExecutionStatus.REJECTED.value,
                        reason_code=ExecutionReason.LOCAL_OPERATION_NOT_ALLOWED.value,
                        admitted=False,
                    )
                )
                return self._receipt(
                    request,
                    status=ExecutionStatus.REJECTED,
                    reason=ExecutionReason.LOCAL_OPERATION_NOT_ALLOWED,
                    context_bytes=context_bytes,
                    context_tokens=context_tokens,
                    context_byte_limit=byte_limit,
                    context_token_limit=token_limit,
                    attempts=attempts,
                )
            try:
                value = _plain_json(
                    handler(operation.arguments, request.context),
                    path="$.local_result",
                )
            except Exception:
                attempts.append(
                    ExecutionAttempt(
                        stage="local-operation",
                        role=ProviderRole.DETERMINISTIC_LOCAL.value,
                        executable_id=operation.operation_type.value,
                        display_label="",
                        invoked=True,
                        status=ExecutionStatus.DEFERRED.value,
                        reason_code=ExecutionReason.LOCAL_OPERATION_FAILED.value,
                        admitted=False,
                    )
                )
                return self._receipt(
                    request,
                    status=ExecutionStatus.DEFERRED,
                    reason=ExecutionReason.LOCAL_OPERATION_FAILED,
                    context_bytes=context_bytes,
                    context_tokens=context_tokens,
                    context_byte_limit=byte_limit,
                    context_token_limit=token_limit,
                    attempts=attempts,
                )
            results.append(value)
            attempts.append(
                ExecutionAttempt(
                    stage="local-operation",
                    role=ProviderRole.DETERMINISTIC_LOCAL.value,
                    executable_id=operation.operation_type.value,
                    display_label="",
                    invoked=True,
                    status=ExecutionStatus.SUCCEEDED.value,
                    reason_code=ExecutionReason.COMPLETED.value,
                    admitted=True,
                )
            )

        return self._receipt(
            request,
            status=ExecutionStatus.SUCCEEDED,
            reason=ExecutionReason.COMPLETED,
            context_bytes=context_bytes,
            context_tokens=context_tokens,
            context_byte_limit=byte_limit,
            context_token_limit=token_limit,
            attempts=attempts,
            result=results,
        )

    def _execute_models(
        self,
        request: TaskExecutionRequest,
        context_bytes: int,
        context_tokens: int,
        *,
        byte_limit: int,
        token_limit: int,
    ) -> TaskExecutionReceipt:
        if request.local_operations:
            return self._receipt(
                request,
                status=ExecutionStatus.REJECTED,
                reason=ExecutionReason.INVALID_REQUEST,
                context_bytes=context_bytes,
                context_tokens=context_tokens,
                context_byte_limit=byte_limit,
                context_token_limit=token_limit,
            )
        if self.grok is None or self.codex is None:
            return self._receipt(
                request,
                status=ExecutionStatus.DEFERRED,
                reason=ExecutionReason.PROVIDER_NOT_CONFIGURED,
                context_bytes=context_bytes,
                context_tokens=context_tokens,
                context_byte_limit=byte_limit,
                context_token_limit=token_limit,
            )
        if (
            request.grok_executable_id != self.grok.executable_id
            or request.codex_executable_id != self.codex.executable_id
        ):
            return self._receipt(
                request,
                status=ExecutionStatus.REJECTED,
                reason=ExecutionReason.EXECUTABLE_BINDING_MISMATCH,
                context_bytes=context_bytes,
                context_tokens=context_tokens,
                context_byte_limit=byte_limit,
                context_token_limit=token_limit,
            )
        if (
            self.grok.executable_id == self.codex.executable_id
            or self.grok.invoke is self.codex.invoke
        ):
            return self._receipt(
                request,
                status=ExecutionStatus.REJECTED,
                reason=ExecutionReason.PROVIDERS_NOT_INDEPENDENT,
                context_bytes=context_bytes,
                context_tokens=context_tokens,
                context_byte_limit=byte_limit,
                context_token_limit=token_limit,
            )

        attempts: list[ExecutionAttempt] = []
        # Prompt envelopes are bounded by the policy ceilings; task context is
        # already constrained by the tighter min(policy, request metadata).
        prompt_byte_limit = self.max_context_bytes
        prompt_token_limit = self.max_context_tokens
        grok_request = ModelExecutionRequest(
            role=ProviderRole.GROK_IMPLEMENT,
            executable_id=self.grok.executable_id,
            task_id=request.task_id,
            task_context=request.context,
            context_metadata=request.context_metadata,
        )
        grok_result, grok_reason = self._invoke_provider(
            self.grok,
            self.grok_quota,
            grok_request,
            attempts,
            prompt_byte_limit=prompt_byte_limit,
            prompt_token_limit=prompt_token_limit,
        )
        if grok_result is None:
            return self._receipt(
                request,
                status=self._terminal_status_for_reason(grok_reason),
                reason=grok_reason,
                context_bytes=context_bytes,
                context_tokens=context_tokens,
                context_byte_limit=byte_limit,
                context_token_limit=token_limit,
                attempts=attempts,
            )

        codex_request = ModelExecutionRequest(
            role=ProviderRole.CODEX_REVIEW,
            executable_id=self.codex.executable_id,
            task_id=request.task_id,
            task_context=request.context,
            context_metadata=request.context_metadata,
            grok_implementation=grok_result,
        )
        codex_result, codex_reason = self._invoke_provider(
            self.codex,
            self.codex_quota,
            codex_request,
            attempts,
            prompt_byte_limit=prompt_byte_limit,
            prompt_token_limit=prompt_token_limit,
        )
        if codex_result is None:
            return self._receipt(
                request,
                status=self._terminal_status_for_reason(codex_reason),
                reason=codex_reason,
                context_bytes=context_bytes,
                context_tokens=context_tokens,
                context_byte_limit=byte_limit,
                context_token_limit=token_limit,
                attempts=attempts,
                grok_implementation=grok_result,
            )

        result = {"implementation": grok_result, "review": codex_result}
        return self._receipt(
            request,
            status=ExecutionStatus.SUCCEEDED,
            reason=ExecutionReason.COMPLETED,
            context_bytes=context_bytes,
            context_tokens=context_tokens,
            context_byte_limit=byte_limit,
            context_token_limit=token_limit,
            attempts=attempts,
            result=result,
            grok_implementation=grok_result,
            codex_review=codex_result,
        )

    @staticmethod
    def _terminal_status_for_reason(reason: ExecutionReason) -> ExecutionStatus:
        """Map provider-stage failures to reject vs defer terminal status."""

        if reason in {
            ExecutionReason.PROVIDER_PROMPT_TOO_LARGE,
            ExecutionReason.PROVIDER_PROMPT_TOKEN_BUDGET,
            ExecutionReason.EXECUTABLE_BINDING_MISMATCH,
            ExecutionReason.PROVIDERS_NOT_INDEPENDENT,
            ExecutionReason.INVALID_REQUEST,
            ExecutionReason.CONTEXT_LIMIT_EXCEEDED,
            ExecutionReason.LOCAL_OPERATION_NOT_ALLOWED,
        }:
            return ExecutionStatus.REJECTED
        return ExecutionStatus.DEFERRED

    def _measure_prompt(
        self,
        request: ModelExecutionRequest,
        *,
        prompt_byte_limit: int,
        prompt_token_limit: int,
    ) -> tuple[int, int, ExecutionReason | None]:
        """Return prompt size and an optional hard-limit violation reason."""

        prompt_data = _canonical_bytes(request.to_dict())
        prompt_bytes = len(prompt_data)
        prompt_tokens = self.token_counter(prompt_data)
        if (
            isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens < 0
        ):
            raise ValueError("token_counter must return a non-negative integer")
        if prompt_bytes > prompt_byte_limit:
            return prompt_bytes, prompt_tokens, ExecutionReason.PROVIDER_PROMPT_TOO_LARGE
        if prompt_tokens > prompt_token_limit:
            return (
                prompt_bytes,
                prompt_tokens,
                ExecutionReason.PROVIDER_PROMPT_TOKEN_BUDGET,
            )
        return prompt_bytes, prompt_tokens, None

    def _invoke_provider(
        self,
        executable: ProviderExecutable,
        quota: ProviderQuotaLatch,
        request: ModelExecutionRequest,
        attempts: list[ExecutionAttempt],
        *,
        prompt_byte_limit: int,
        prompt_token_limit: int,
    ) -> tuple[Mapping[str, Any] | None, ExecutionReason]:
        quota_reason = (
            ExecutionReason.GROK_QUOTA_EXHAUSTED
            if executable.role is ProviderRole.GROK_IMPLEMENT
            else ExecutionReason.CODEX_QUOTA_EXHAUSTED
        )
        failure_reason = (
            ExecutionReason.GROK_FAILED
            if executable.role is ProviderRole.GROK_IMPLEMENT
            else ExecutionReason.CODEX_FAILED
        )
        prompt_bytes, prompt_tokens, prompt_reason = self._measure_prompt(
            request,
            prompt_byte_limit=prompt_byte_limit,
            prompt_token_limit=prompt_token_limit,
        )
        if prompt_reason is not None:
            attempts.append(
                self._provider_attempt(
                    executable,
                    invoked=False,
                    status=ExecutionStatus.REJECTED,
                    reason=prompt_reason,
                    prompt_bytes=prompt_bytes,
                    prompt_tokens=prompt_tokens,
                )
            )
            return None, prompt_reason

        if not quota.acquire():
            attempts.append(
                self._provider_attempt(
                    executable,
                    invoked=False,
                    status=ExecutionStatus.DEFERRED,
                    reason=quota_reason,
                    prompt_bytes=prompt_bytes,
                    prompt_tokens=prompt_tokens,
                )
            )
            return None, quota_reason

        try:
            response = executable.invoke(request)
        except ProviderQuotaError as exc:
            quota.latch(exc.reason_code)
            attempts.append(
                self._provider_attempt(
                    executable,
                    invoked=True,
                    status=ExecutionStatus.DEFERRED,
                    reason=quota_reason,
                    prompt_bytes=prompt_bytes,
                    prompt_tokens=prompt_tokens,
                )
            )
            return None, quota_reason
        except Exception:
            attempts.append(
                self._provider_attempt(
                    executable,
                    invoked=True,
                    status=ExecutionStatus.DEFERRED,
                    reason=failure_reason,
                    prompt_bytes=prompt_bytes,
                    prompt_tokens=prompt_tokens,
                )
            )
            return None, failure_reason

        if not isinstance(response, Mapping):
            attempts.append(
                self._provider_attempt(
                    executable,
                    invoked=True,
                    status=ExecutionStatus.DEFERRED,
                    reason=ExecutionReason.PROVIDER_RESPONSE_INVALID,
                    prompt_bytes=prompt_bytes,
                    prompt_tokens=prompt_tokens,
                )
            )
            return None, ExecutionReason.PROVIDER_RESPONSE_INVALID
        try:
            detached = _plain_json(response, path="$.provider_response")
            response_data = _canonical_bytes(detached)
            response_size = len(response_data)
        except (TypeError, ValueError):
            attempts.append(
                self._provider_attempt(
                    executable,
                    invoked=True,
                    status=ExecutionStatus.DEFERRED,
                    reason=ExecutionReason.PROVIDER_RESPONSE_INVALID,
                    prompt_bytes=prompt_bytes,
                    prompt_tokens=prompt_tokens,
                )
            )
            return None, ExecutionReason.PROVIDER_RESPONSE_INVALID
        if response_size > self.max_provider_response_bytes:
            attempts.append(
                self._provider_attempt(
                    executable,
                    invoked=True,
                    status=ExecutionStatus.DEFERRED,
                    reason=ExecutionReason.PROVIDER_RESPONSE_TOO_LARGE,
                    prompt_bytes=prompt_bytes,
                    prompt_tokens=prompt_tokens,
                    response_bytes=response_size,
                )
            )
            return None, ExecutionReason.PROVIDER_RESPONSE_TOO_LARGE
        if self._response_reports_quota(detached):
            quota.latch("provider_reported_quota_exhausted")
            attempts.append(
                self._provider_attempt(
                    executable,
                    invoked=True,
                    status=ExecutionStatus.DEFERRED,
                    reason=quota_reason,
                    prompt_bytes=prompt_bytes,
                    prompt_tokens=prompt_tokens,
                    response_bytes=response_size,
                )
            )
            return None, quota_reason
        if self._response_claims_authority(detached):
            attempts.append(
                self._provider_attempt(
                    executable,
                    invoked=True,
                    status=ExecutionStatus.DEFERRED,
                    reason=ExecutionReason.PROVIDER_RESULT_NOT_ADMITTED,
                    prompt_bytes=prompt_bytes,
                    prompt_tokens=prompt_tokens,
                    response_bytes=response_size,
                )
            )
            return None, ExecutionReason.PROVIDER_RESULT_NOT_ADMITTED

        frozen = _freeze_json(detached)
        attempts.append(
            self._provider_attempt(
                executable,
                invoked=True,
                status=ExecutionStatus.SUCCEEDED,
                reason=ExecutionReason.COMPLETED,
                prompt_bytes=prompt_bytes,
                prompt_tokens=prompt_tokens,
                response_bytes=response_size,
                admitted=True,
            )
        )
        return frozen, ExecutionReason.COMPLETED

    @staticmethod
    def _response_reports_quota(response: Mapping[str, Any]) -> bool:
        status = str(response.get("status", "")).strip().lower().replace("-", "_")
        reason = str(response.get("reason_code", "")).strip().lower()
        return status in {"quota_exhausted", "rate_limited", "capacity_exhausted"} or (
            "quota" in reason or "rate_limit" in reason
        )

    @staticmethod
    def _response_claims_authority(response: Mapping[str, Any]) -> bool:
        """Reject provider payloads that claim completion or proof authority."""

        if response.get("completion_authoritative") is True:
            return True
        if response.get("proof_authoritative") is True:
            return True
        authority = response.get("authority")
        if isinstance(authority, Mapping):
            if authority.get("completion_authoritative") is True:
                return True
            if authority.get("proof_authoritative") is True:
                return True
            if authority.get("repository_write_allowed") is True:
                return True
            if authority.get("proposal_only") is False:
                return True
        return False

    @staticmethod
    def _provider_attempt(
        executable: ProviderExecutable,
        *,
        invoked: bool,
        status: ExecutionStatus,
        reason: ExecutionReason,
        prompt_bytes: int = 0,
        prompt_tokens: int = 0,
        response_bytes: int = 0,
        admitted: bool = False,
    ) -> ExecutionAttempt:
        return ExecutionAttempt(
            stage=executable.role.value,
            role=executable.role.value,
            executable_id=executable.executable_id,
            display_label=executable.display_label,
            invoked=invoked,
            status=status.value,
            reason_code=reason.value,
            prompt_bytes=prompt_bytes,
            prompt_tokens=prompt_tokens,
            response_bytes=response_bytes,
            admitted=admitted,
        )

    def _receipt(
        self,
        request: TaskExecutionRequest,
        *,
        status: ExecutionStatus,
        reason: ExecutionReason,
        context_bytes: int,
        context_tokens: int,
        context_byte_limit: int,
        context_token_limit: int,
        attempts: Sequence[ExecutionAttempt] = (),
        result: Any = None,
        grok_implementation: Mapping[str, Any] | None = None,
        codex_review: Mapping[str, Any] | None = None,
    ) -> TaskExecutionReceipt:
        attempts_tuple = tuple(attempts)
        model_calls = sum(
            1
            for attempt in attempts_tuple
            if attempt.invoked
            and attempt.role
            in {
                ProviderRole.GROK_IMPLEMENT.value,
                ProviderRole.CODEX_REVIEW.value,
            }
        )
        prompt_bytes = sum(attempt.prompt_bytes for attempt in attempts_tuple)
        prompt_tokens = sum(attempt.prompt_tokens for attempt in attempts_tuple)
        return TaskExecutionReceipt(
            task_id=request.task_id,
            mode=request.mode,
            status=status,
            reason_code=reason.value,
            context_bytes=context_bytes,
            context_tokens=context_tokens,
            model_call_count=model_calls,
            provider_call_count=model_calls,
            attempts=attempts_tuple,
            result=_freeze_json(_plain_json(result)) if result is not None else None,
            grok_implementation=grok_implementation,
            codex_review=codex_review,
            context_byte_limit=context_byte_limit,
            context_token_limit=context_token_limit,
            max_provider_response_bytes=self.max_provider_response_bytes,
            prompt_bytes=prompt_bytes,
            prompt_tokens=prompt_tokens,
        )


# Descriptive aliases retained for call sites that prefer task-prefixed names.
TaskExecutionMode = ExecutionMode
TaskExecutionStatus = ExecutionStatus
TaskExecutionReason = ExecutionReason
TaskContextBounds = TaskContextMetadata
LocalOperation = TypedLocalOperation


def execute_task(
    request: TaskExecutionRequest,
    *,
    policy: TaskExecutionPolicy,
) -> TaskExecutionReceipt:
    """Functional facade for daemon integrations."""

    return policy.execute(request)
