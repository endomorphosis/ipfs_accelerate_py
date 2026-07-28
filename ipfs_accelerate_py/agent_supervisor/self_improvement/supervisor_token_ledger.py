"""Provider-native token accounting joined to terminal acceptance criteria.

The generation-1 efficiency receipt records useful task totals, but cannot
prove that every provider measurement and every lifecycle event was counted
exactly once.  This module supplies that stricter boundary.  A
``SupervisorTokenLedger`` is immutable, content addressed, and closed over:

* one generation-2 result binding;
* the complete lifecycle-event population for that binding;
* one token attribution for every event;
* calibrated provider/model envelopes for estimated measurements; and
* terminal criterion records, including rejected and abandoned work.

Input, output, and tool tokens form the charged token total.  Reused tokens
are a subset of input tokens and speculative tokens are a subset of output
tokens.  Retry and failed-attempt tokens are explicit, checked classifications
of that same total rather than additional tokens.  This prevents either
double charging or making failed work disappear from efficiency reports.

No prompts, model output, or source bodies cross this contract.  Fallback
calibrations contain only byte/token observations and are scoped to an exact
provider/model envelope.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import CanonicalContract
from .supervisor_v2_contracts import (
    ResultBinding,
    StageEvent,
    StageEventKind,
)


TOKEN_LEDGER_CONTRACT_VERSION: Final[int] = 1
SCHEMA_VERSION: Final[int] = TOKEN_LEDGER_CONTRACT_VERSION
ACCEPTED_CRITERION_TOKEN_REQUIREMENT_ID: Final[str] = (
    "121282056926752432472380808295780602698"
)
ACCEPTED_CRITERION_TOKEN_GOAL_ID: Final[str] = "ASI-G210"

PROVIDER_MODEL_ENVELOPE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/provider-model-envelope@1"
)
TOKENIZER_SAMPLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/tokenizer-calibration-sample@1"
)
TOKENIZER_CALIBRATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/tokenizer-calibration@1"
)
PROVIDER_TOKEN_USAGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/provider-token-usage@1"
)
TERMINAL_CRITERION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/terminal-criterion-attribution@1"
)
TOKEN_ATTRIBUTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/token-attribution@1"
)
TOKEN_RATIO_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/token-ledger-exact-ratio@1"
)
CRITERION_COST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/criterion-token-cost@1"
)
TOKEN_LEDGER_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/token-ledger-report@1"
)
SUPERVISOR_TOKEN_LEDGER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/supervisor-token-ledger@1"
)

MAX_TEXT_BYTES: Final[int] = 512
MAX_TOKENS: Final[int] = 10**12
MAX_COST_MICROUNITS: Final[int] = 10**15
MAX_CONTEXT_TOKENS: Final[int] = 10**9
MAX_CALIBRATION_SAMPLES: Final[int] = 4_096
MAX_EVENTS: Final[int] = 100_000
MAX_CRITERIA: Final[int] = 4_096
MAX_SERIALIZED_LEDGER_BYTES: Final[int] = 16 * 1024 * 1024


class TokenLedgerValidationError(ValueError):
    """Token accounting is malformed, duplicated, or detached."""


class UsageSource(str, Enum):
    PROVIDER_NATIVE = "provider_native"
    CALIBRATED_FALLBACK = "calibrated_fallback"


class CacheDecision(str, Enum):
    HIT = "hit"
    MISS = "miss"
    BYPASS = "bypass"
    INVALIDATED = "invalidated"
    ERROR = "error"
    NOT_APPLICABLE = "not_applicable"


class ValidationResult(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"
    NOT_REQUIRED = "not_required"


class TerminalDisposition(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ABANDONED = "abandoned"

    @property
    def accepted(self) -> bool:
        return self is TerminalDisposition.ACCEPTED


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise TokenLedgerValidationError(f"{name} must be text")
    result = value.strip()
    if required and not result:
        raise TokenLedgerValidationError(f"{name} must not be empty")
    if "\x00" in result or len(result.encode("utf-8")) > MAX_TEXT_BYTES:
        raise TokenLedgerValidationError(f"{name} is unsafe or too large")
    return result


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_TOKENS,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TokenLedgerValidationError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise TokenLedgerValidationError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(raw)
    except (TypeError, ValueError) as exc:
        raise TokenLedgerValidationError(
            f"{name} is not a supported {enum_type.__name__}"
        ) from exc


def _closed(
    payload: Mapping[str, Any],
    *,
    schema: str,
    allowed: Iterable[str],
    name: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise TokenLedgerValidationError(f"{name} must be an object")
    if payload.get("schema") != schema:
        raise TokenLedgerValidationError(f"unsupported {name} schema")
    version = payload.get("contract_version", payload.get("schema_version"))
    if version != TOKEN_LEDGER_CONTRACT_VERSION:
        raise TokenLedgerValidationError(f"unsupported {name} version")
    if set(payload).difference(allowed):
        raise TokenLedgerValidationError(f"{name} contains unsupported fields")


def _claim(payload: Mapping[str, Any], actual: str, *names: str) -> None:
    for name in names:
        claimed = payload.get(name)
        if claimed not in (None, "", actual):
            raise TokenLedgerValidationError(
                "content identity does not match canonical contents"
            )


def _records(
    values: Any,
    record_type: type,
    *,
    field_name: str,
    maximum: int,
) -> tuple[Any, ...]:
    if not isinstance(values, Sequence) or isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        raise TokenLedgerValidationError(f"{field_name} must be a sequence")
    if len(values) > maximum:
        raise TokenLedgerValidationError(
            f"{field_name} exceeds its {maximum}-item bound"
        )
    result = []
    for item in values:
        if isinstance(item, record_type):
            result.append(item)
        elif isinstance(item, Mapping):
            result.append(record_type.from_dict(item))
        else:
            raise TokenLedgerValidationError(
                f"{field_name} must contain {record_type.__name__} records"
            )
    return tuple(result)


def _binding(value: Any) -> ResultBinding:
    if isinstance(value, ResultBinding):
        return value
    if isinstance(value, Mapping):
        return ResultBinding.from_dict(value)
    raise TokenLedgerValidationError("binding must be a ResultBinding")


def _strict_json(value: str | bytes | bytearray, name: str) -> Mapping[str, Any]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise TokenLedgerValidationError(
                    f"{name} JSON contains duplicate object keys"
                )
            result[key] = item
        return result

    try:
        if isinstance(value, (bytes, bytearray)):
            value = bytes(value).decode("utf-8")
        if not isinstance(value, str):
            raise TokenLedgerValidationError(f"{name} JSON must be text")
        decoded = json.loads(value, object_pairs_hook=unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TokenLedgerValidationError(f"{name} JSON is malformed") from exc
    if not isinstance(decoded, Mapping):
        raise TokenLedgerValidationError(f"{name} JSON must contain an object")
    return decoded


class _LedgerContract(CanonicalContract):
    @property
    def schema_version(self) -> int:
        return TOKEN_LEDGER_CONTRACT_VERSION

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "_LedgerContract":
        return cls.from_dict(_strict_json(value, cls.__name__))  # type: ignore[attr-defined,no-any-return]


@dataclass(frozen=True)
class ProviderModelEnvelope(_LedgerContract):
    """Exact provider/model/tokenizer boundary for native or fallback counts."""

    SCHEMA: ClassVar[str] = PROVIDER_MODEL_ENVELOPE_SCHEMA

    provider_id: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    envelope_revision: str
    max_context_tokens: int

    def __post_init__(self) -> None:
        for name in (
            "provider_id",
            "model_id",
            "model_revision",
            "tokenizer_id",
            "envelope_revision",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "max_context_tokens",
            _integer(
                self.max_context_tokens,
                "max_context_tokens",
                minimum=1,
                maximum=MAX_CONTEXT_TOKENS,
            ),
        )

    @property
    def envelope_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": TOKEN_LEDGER_CONTRACT_VERSION,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_id": self.tokenizer_id,
            "envelope_revision": self.envelope_revision,
            "max_context_tokens": self.max_context_tokens,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderModelEnvelope":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "provider_id",
            "model_id",
            "model_revision",
            "tokenizer_id",
            "envelope_revision",
            "max_context_tokens",
            "envelope_id",
            "content_id",
        }
        _closed(
            payload,
            schema=cls.SCHEMA,
            allowed=allowed,
            name="provider/model envelope",
        )
        result = cls(
            provider_id=payload.get("provider_id", ""),
            model_id=payload.get("model_id", ""),
            model_revision=payload.get("model_revision", ""),
            tokenizer_id=payload.get("tokenizer_id", ""),
            envelope_revision=payload.get("envelope_revision", ""),
            max_context_tokens=payload.get("max_context_tokens", 0),
        )
        _claim(payload, result.envelope_id, "envelope_id", "content_id")
        return result


@dataclass(frozen=True)
class TokenizerCalibrationSample(_LedgerContract):
    """Content-free provider observation used to calibrate a fallback."""

    SCHEMA: ClassVar[str] = TOKENIZER_SAMPLE_SCHEMA

    sample_id: str
    utf8_bytes: int
    provider_tokens: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", _text(self.sample_id, "sample_id"))
        object.__setattr__(
            self,
            "utf8_bytes",
            _integer(self.utf8_bytes, "utf8_bytes", minimum=1),
        )
        object.__setattr__(
            self,
            "provider_tokens",
            _integer(self.provider_tokens, "provider_tokens", minimum=1),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": TOKEN_LEDGER_CONTRACT_VERSION,
            "sample_id": self.sample_id,
            "utf8_bytes": self.utf8_bytes,
            "provider_tokens": self.provider_tokens,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "TokenizerCalibrationSample":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "sample_id",
            "utf8_bytes",
            "provider_tokens",
            "content_id",
        }
        _closed(
            payload,
            schema=cls.SCHEMA,
            allowed=allowed,
            name="tokenizer calibration sample",
        )
        result = cls(
            sample_id=payload.get("sample_id", ""),
            utf8_bytes=payload.get("utf8_bytes", 0),
            provider_tokens=payload.get("provider_tokens", 0),
        )
        _claim(payload, result.content_id, "content_id")
        return result


@dataclass(frozen=True)
class FallbackTokenizerCalibration(_LedgerContract):
    """Replayable weighted byte/token calibration for one exact envelope."""

    SCHEMA: ClassVar[str] = TOKENIZER_CALIBRATION_SCHEMA

    envelope: ProviderModelEnvelope
    calibration_revision: str
    samples: tuple[TokenizerCalibrationSample, ...]

    def __post_init__(self) -> None:
        envelope = self.envelope
        if isinstance(envelope, Mapping):
            envelope = ProviderModelEnvelope.from_dict(envelope)
        if not isinstance(envelope, ProviderModelEnvelope):
            raise TokenLedgerValidationError(
                "envelope must be a ProviderModelEnvelope"
            )
        object.__setattr__(self, "envelope", envelope)
        object.__setattr__(
            self,
            "calibration_revision",
            _text(self.calibration_revision, "calibration_revision"),
        )
        samples = _records(
            self.samples,
            TokenizerCalibrationSample,
            field_name="samples",
            maximum=MAX_CALIBRATION_SAMPLES,
        )
        if not samples:
            raise TokenLedgerValidationError(
                "fallback calibration requires provider-native samples"
            )
        sample_ids = [item.sample_id for item in samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise TokenLedgerValidationError(
                "fallback calibration contains duplicated samples"
            )
        object.__setattr__(
            self, "samples", tuple(sorted(samples, key=lambda item: item.sample_id))
        )

    @property
    def calibration_id(self) -> str:
        return self.content_id

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def token_numerator(self) -> int:
        return sum(item.provider_tokens for item in self.samples)

    @property
    def byte_denominator(self) -> int:
        return sum(item.utf8_bytes for item in self.samples)

    @property
    def maximum_absolute_error_bps(self) -> int:
        errors = []
        for item in self.samples:
            estimate = self.estimate_bytes(item.utf8_bytes)
            errors.append(
                abs(estimate - item.provider_tokens) * 10_000
                // item.provider_tokens
            )
        return max(errors, default=0)

    def estimate_bytes(self, utf8_bytes: int) -> int:
        size = _integer(utf8_bytes, "utf8_bytes")
        if not size:
            return 0
        return (
            size * self.token_numerator + self.byte_denominator - 1
        ) // self.byte_denominator

    def estimate_text(self, text: str) -> int:
        if not isinstance(text, str):
            raise TokenLedgerValidationError("text must be text")
        return self.estimate_bytes(len(text.encode("utf-8")))

    def supports(self, envelope: ProviderModelEnvelope) -> bool:
        return (
            isinstance(envelope, ProviderModelEnvelope)
            and envelope.envelope_id == self.envelope.envelope_id
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": TOKEN_LEDGER_CONTRACT_VERSION,
            "envelope": self.envelope.to_record(),
            "calibration_revision": self.calibration_revision,
            "samples": tuple(item.to_record() for item in self.samples),
            "sample_count": self.sample_count,
            "token_numerator": self.token_numerator,
            "byte_denominator": self.byte_denominator,
            "maximum_absolute_error_bps": self.maximum_absolute_error_bps,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "FallbackTokenizerCalibration":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "envelope",
            "calibration_revision",
            "samples",
            "sample_count",
            "token_numerator",
            "byte_denominator",
            "maximum_absolute_error_bps",
            "calibration_id",
            "content_id",
        }
        _closed(
            payload,
            schema=cls.SCHEMA,
            allowed=allowed,
            name="fallback tokenizer calibration",
        )
        result = cls(
            envelope=payload.get("envelope", {}),
            calibration_revision=payload.get("calibration_revision", ""),
            samples=payload.get("samples", ()),
        )
        claims = {
            "sample_count": result.sample_count,
            "token_numerator": result.token_numerator,
            "byte_denominator": result.byte_denominator,
            "maximum_absolute_error_bps": result.maximum_absolute_error_bps,
        }
        for name, actual in claims.items():
            if payload.get(name, actual) != actual:
                raise TokenLedgerValidationError(
                    f"{name} does not match calibration samples"
                )
        _claim(
            payload,
            result.calibration_id,
            "calibration_id",
            "content_id",
        )
        return result


TokenizerCalibration = FallbackTokenizerCalibration
ProviderTokenizerEnvelope = ProviderModelEnvelope
CalibrationSample = TokenizerCalibrationSample


def calibrate_fallback_tokenizer(
    envelope: ProviderModelEnvelope,
    samples: Sequence[TokenizerCalibrationSample | Mapping[str, Any]],
    *,
    calibration_revision: str,
) -> FallbackTokenizerCalibration:
    """Build a replayable fallback tokenizer from native count observations."""

    return FallbackTokenizerCalibration(
        envelope=envelope,
        calibration_revision=calibration_revision,
        samples=tuple(samples),  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class ProviderTokenUsage(_LedgerContract):
    """One provider measurement; classified counters do not double charge."""

    SCHEMA: ClassVar[str] = PROVIDER_TOKEN_USAGE_SCHEMA

    measurement_id: str
    envelope: ProviderModelEnvelope
    source: UsageSource
    input_tokens: int = 0
    output_tokens: int = 0
    reused_tokens: int = 0
    speculative_tokens: int = 0
    tool_tokens: int = 0
    retry_tokens: int = 0
    failed_attempt_tokens: int = 0
    cost_microunits: int = 0
    calibration_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "measurement_id", _text(self.measurement_id, "measurement_id")
        )
        envelope = self.envelope
        if isinstance(envelope, Mapping):
            envelope = ProviderModelEnvelope.from_dict(envelope)
        if not isinstance(envelope, ProviderModelEnvelope):
            raise TokenLedgerValidationError(
                "envelope must be a ProviderModelEnvelope"
            )
        object.__setattr__(self, "envelope", envelope)
        object.__setattr__(
            self, "source", _enum(self.source, UsageSource, "source")
        )
        for name in (
            "input_tokens",
            "output_tokens",
            "reused_tokens",
            "speculative_tokens",
            "tool_tokens",
            "retry_tokens",
            "failed_attempt_tokens",
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "cost_microunits",
            _integer(
                self.cost_microunits,
                "cost_microunits",
                maximum=MAX_COST_MICROUNITS,
            ),
        )
        object.__setattr__(
            self,
            "calibration_id",
            _text(
                self.calibration_id,
                "calibration_id",
                required=False,
            ),
        )
        if self.reused_tokens > self.input_tokens:
            raise TokenLedgerValidationError(
                "reused_tokens cannot exceed input_tokens"
            )
        if self.speculative_tokens > self.output_tokens:
            raise TokenLedgerValidationError(
                "speculative_tokens cannot exceed output_tokens"
            )
        if self.retry_tokens > self.total_tokens:
            raise TokenLedgerValidationError(
                "retry_tokens cannot exceed total_tokens"
            )
        if self.failed_attempt_tokens > self.total_tokens:
            raise TokenLedgerValidationError(
                "failed_attempt_tokens cannot exceed total_tokens"
            )
        if self.input_tokens > self.envelope.max_context_tokens:
            raise TokenLedgerValidationError(
                "input_tokens exceed the provider/model context envelope"
            )
        if self.source is UsageSource.PROVIDER_NATIVE and self.calibration_id:
            raise TokenLedgerValidationError(
                "provider-native usage cannot cite a fallback calibration"
            )
        if (
            self.source is UsageSource.CALIBRATED_FALLBACK
            and not self.calibration_id
        ):
            raise TokenLedgerValidationError(
                "fallback usage requires a calibration_id"
            )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.tool_tokens

    @property
    def fresh_input_tokens(self) -> int:
        return self.input_tokens - self.reused_tokens

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": TOKEN_LEDGER_CONTRACT_VERSION,
            "measurement_id": self.measurement_id,
            "envelope": self.envelope.to_record(),
            "source": self.source,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reused_tokens": self.reused_tokens,
            "speculative_tokens": self.speculative_tokens,
            "tool_tokens": self.tool_tokens,
            "retry_tokens": self.retry_tokens,
            "failed_attempt_tokens": self.failed_attempt_tokens,
            "cost_microunits": self.cost_microunits,
            "calibration_id": self.calibration_id,
            "total_tokens": self.total_tokens,
            "fresh_input_tokens": self.fresh_input_tokens,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderTokenUsage":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "measurement_id",
            "envelope",
            "source",
            "input_tokens",
            "output_tokens",
            "reused_tokens",
            "speculative_tokens",
            "tool_tokens",
            "retry_tokens",
            "failed_attempt_tokens",
            "cost_microunits",
            "calibration_id",
            "total_tokens",
            "fresh_input_tokens",
            "content_id",
        }
        _closed(
            payload,
            schema=cls.SCHEMA,
            allowed=allowed,
            name="provider token usage",
        )
        result = cls(
            measurement_id=payload.get("measurement_id", ""),
            envelope=payload.get("envelope", {}),
            source=payload.get("source", ""),
            input_tokens=payload.get("input_tokens", 0),
            output_tokens=payload.get("output_tokens", 0),
            reused_tokens=payload.get("reused_tokens", 0),
            speculative_tokens=payload.get("speculative_tokens", 0),
            tool_tokens=payload.get("tool_tokens", 0),
            retry_tokens=payload.get("retry_tokens", 0),
            failed_attempt_tokens=payload.get("failed_attempt_tokens", 0),
            cost_microunits=payload.get("cost_microunits", 0),
            calibration_id=payload.get("calibration_id", ""),
        )
        for name in ("total_tokens", "fresh_input_tokens"):
            if payload.get(name, getattr(result, name)) != getattr(result, name):
                raise TokenLedgerValidationError(
                    f"{name} does not match provider counters"
                )
        _claim(payload, result.content_id, "content_id")
        return result


NativeTokenUsage = ProviderTokenUsage


@dataclass(frozen=True)
class TerminalCriterionAttribution(_LedgerContract):
    """Terminal result for the criterion one attempt tried to satisfy."""

    SCHEMA: ClassVar[str] = TERMINAL_CRITERION_SCHEMA

    binding: ResultBinding
    terminal_event_id: str
    criterion_id: str
    disposition: TerminalDisposition
    validation_result: ValidationResult
    evidence_gain: int = 0
    reason_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding", _binding(self.binding))
        for name in ("terminal_event_id", "criterion_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, TerminalDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "validation_result",
            _enum(
                self.validation_result,
                ValidationResult,
                "validation_result",
            ),
        )
        object.__setattr__(
            self,
            "evidence_gain",
            _integer(self.evidence_gain, "evidence_gain"),
        )
        object.__setattr__(
            self,
            "reason_code",
            _text(self.reason_code, "reason_code", required=False),
        )
        if self.disposition.accepted:
            if self.validation_result is not ValidationResult.PASSED:
                raise TokenLedgerValidationError(
                    "accepted criterion requires passed validation"
                )
        else:
            if self.evidence_gain:
                raise TokenLedgerValidationError(
                    "rejected or abandoned criteria cannot claim evidence gain"
                )
            if not self.reason_code:
                raise TokenLedgerValidationError(
                    "rejected or abandoned criteria require reason_code"
                )

    @property
    def terminal_attribution_id(self) -> str:
        return self.content_id

    @property
    def accepted(self) -> bool:
        return self.disposition.accepted

    @property
    def accepted_criterion_id(self) -> str:
        return self.criterion_id if self.accepted else ""

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": TOKEN_LEDGER_CONTRACT_VERSION,
            "binding": self.binding.to_record(),
            "terminal_event_id": self.terminal_event_id,
            "criterion_id": self.criterion_id,
            "disposition": self.disposition,
            "validation_result": self.validation_result,
            "evidence_gain": self.evidence_gain,
            "reason_code": self.reason_code,
            "accepted": self.accepted,
            "accepted_criterion_id": self.accepted_criterion_id,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "TerminalCriterionAttribution":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "binding",
            "terminal_event_id",
            "criterion_id",
            "disposition",
            "validation_result",
            "evidence_gain",
            "reason_code",
            "accepted",
            "accepted_criterion_id",
            "terminal_attribution_id",
            "content_id",
        }
        _closed(
            payload,
            schema=cls.SCHEMA,
            allowed=allowed,
            name="terminal criterion attribution",
        )
        result = cls(
            binding=payload.get("binding", {}),
            terminal_event_id=payload.get("terminal_event_id", ""),
            criterion_id=payload.get("criterion_id", ""),
            disposition=payload.get("disposition", ""),
            validation_result=payload.get("validation_result", ""),
            evidence_gain=payload.get("evidence_gain", 0),
            reason_code=payload.get("reason_code", ""),
        )
        if payload.get("accepted", result.accepted) is not result.accepted:
            raise TokenLedgerValidationError("accepted claim is inconsistent")
        if (
            payload.get("accepted_criterion_id", result.accepted_criterion_id)
            != result.accepted_criterion_id
        ):
            raise TokenLedgerValidationError(
                "accepted_criterion_id claim is inconsistent"
            )
        _claim(
            payload,
            result.terminal_attribution_id,
            "terminal_attribution_id",
            "content_id",
        )
        return result


TerminalAttribution = TerminalCriterionAttribution


@dataclass(frozen=True)
class TokenAttribution(_LedgerContract):
    """One lifecycle event's complete provider usage attribution."""

    SCHEMA: ClassVar[str] = TOKEN_ATTRIBUTION_SCHEMA

    binding: ResultBinding
    event_id: str
    stage: str
    attempt: int
    context_id: str
    cache_decision: CacheDecision
    validation_result: ValidationResult
    terminal_attribution_id: str
    usage: ProviderTokenUsage

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding", _binding(self.binding))
        for name in (
            "event_id",
            "stage",
            "context_id",
            "terminal_attribution_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "attempt",
            _integer(self.attempt, "attempt", minimum=1, maximum=100_000),
        )
        object.__setattr__(
            self,
            "cache_decision",
            _enum(self.cache_decision, CacheDecision, "cache_decision"),
        )
        object.__setattr__(
            self,
            "validation_result",
            _enum(
                self.validation_result,
                ValidationResult,
                "validation_result",
            ),
        )
        usage = self.usage
        if isinstance(usage, Mapping):
            usage = ProviderTokenUsage.from_dict(usage)
        if not isinstance(usage, ProviderTokenUsage):
            raise TokenLedgerValidationError(
                "usage must be ProviderTokenUsage"
            )
        object.__setattr__(self, "usage", usage)
        if (
            usage.reused_tokens
            and self.cache_decision is not CacheDecision.HIT
        ):
            raise TokenLedgerValidationError(
                "reused tokens require an attributed cache hit"
            )
        if self.attempt == 1 and usage.retry_tokens:
            raise TokenLedgerValidationError(
                "first-attempt usage cannot be classified as retry tokens"
            )
        if self.attempt > 1 and usage.retry_tokens != usage.total_tokens:
            raise TokenLedgerValidationError(
                "all tokens after attempt one must be classified as retry tokens"
            )

    @property
    def attribution_id(self) -> str:
        return self.content_id

    @property
    def task_id(self) -> str:
        return self.binding.task_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": TOKEN_LEDGER_CONTRACT_VERSION,
            "binding": self.binding.to_record(),
            "event_id": self.event_id,
            "stage": self.stage,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "context_id": self.context_id,
            "cache_decision": self.cache_decision,
            "validation_result": self.validation_result,
            "terminal_attribution_id": self.terminal_attribution_id,
            "usage": self.usage.to_record(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TokenAttribution":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "binding",
            "event_id",
            "stage",
            "task_id",
            "attempt",
            "context_id",
            "cache_decision",
            "validation_result",
            "terminal_attribution_id",
            "usage",
            "attribution_id",
            "content_id",
        }
        _closed(
            payload,
            schema=cls.SCHEMA,
            allowed=allowed,
            name="token attribution",
        )
        result = cls(
            binding=payload.get("binding", {}),
            event_id=payload.get("event_id", ""),
            stage=payload.get("stage", ""),
            attempt=payload.get("attempt", 0),
            context_id=payload.get("context_id", ""),
            cache_decision=payload.get("cache_decision", ""),
            validation_result=payload.get("validation_result", ""),
            terminal_attribution_id=payload.get(
                "terminal_attribution_id", ""
            ),
            usage=payload.get("usage", {}),
        )
        if payload.get("task_id", result.task_id) != result.task_id:
            raise TokenLedgerValidationError("task_id is foreign to binding")
        _claim(
            payload,
            result.attribution_id,
            "attribution_id",
            "content_id",
        )
        return result


LifecycleTokenAttribution = TokenAttribution


@dataclass(frozen=True)
class ExactTokenRatio(_LedgerContract):
    SCHEMA: ClassVar[str] = TOKEN_RATIO_SCHEMA

    numerator: int
    denominator: int
    multiplier: int = 1

    def __post_init__(self) -> None:
        for name in ("numerator", "denominator", "multiplier"):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name,
                    maximum=MAX_COST_MICROUNITS * MAX_EVENTS,
                ),
            )

    @property
    def defined(self) -> bool:
        return self.denominator > 0

    @property
    def value(self) -> float:
        if not self.defined:
            return 0.0
        return self.numerator * self.multiplier / self.denominator

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": TOKEN_LEDGER_CONTRACT_VERSION,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "multiplier": self.multiplier,
            "defined": self.defined,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExactTokenRatio":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "numerator",
            "denominator",
            "multiplier",
            "defined",
            "content_id",
        }
        _closed(
            payload,
            schema=cls.SCHEMA,
            allowed=allowed,
            name="exact token ratio",
        )
        result = cls(
            numerator=payload.get("numerator", 0),
            denominator=payload.get("denominator", 0),
            multiplier=payload.get("multiplier", 1),
        )
        if payload.get("defined", result.defined) is not result.defined:
            raise TokenLedgerValidationError("ratio defined claim is inconsistent")
        _claim(payload, result.content_id, "content_id")
        return result


@dataclass(frozen=True)
class CriterionTokenCost(_LedgerContract):
    """All attempts directed at one criterion, successful or otherwise."""

    SCHEMA: ClassVar[str] = CRITERION_COST_SCHEMA

    criterion_id: str
    accepted: bool
    attempt_count: int
    total_tokens: int
    cost_microunits: int
    evidence_gain: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "criterion_id", _text(self.criterion_id, "criterion_id")
        )
        if not isinstance(self.accepted, bool):
            raise TokenLedgerValidationError("accepted must be boolean")
        for name, minimum, maximum in (
            ("attempt_count", 1, MAX_EVENTS),
            ("total_tokens", 0, MAX_TOKENS * MAX_EVENTS),
            ("cost_microunits", 0, MAX_COST_MICROUNITS * MAX_EVENTS),
            ("evidence_gain", 0, MAX_TOKENS),
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name,
                    minimum=minimum,
                    maximum=maximum,
                ),
            )
        if self.evidence_gain and not self.accepted:
            raise TokenLedgerValidationError(
                "unaccepted criterion cannot claim evidence gain"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": TOKEN_LEDGER_CONTRACT_VERSION,
            "criterion_id": self.criterion_id,
            "accepted": self.accepted,
            "attempt_count": self.attempt_count,
            "total_tokens": self.total_tokens,
            "cost_microunits": self.cost_microunits,
            "evidence_gain": self.evidence_gain,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CriterionTokenCost":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "criterion_id",
            "accepted",
            "attempt_count",
            "total_tokens",
            "cost_microunits",
            "evidence_gain",
            "content_id",
        }
        _closed(
            payload,
            schema=cls.SCHEMA,
            allowed=allowed,
            name="criterion token cost",
        )
        result = cls(
            criterion_id=payload.get("criterion_id", ""),
            accepted=payload.get("accepted", False),
            attempt_count=payload.get("attempt_count", 0),
            total_tokens=payload.get("total_tokens", 0),
            cost_microunits=payload.get("cost_microunits", 0),
            evidence_gain=payload.get("evidence_gain", 0),
        )
        _claim(payload, result.content_id, "content_id")
        return result


@dataclass(frozen=True)
class TokenLedgerReport(_LedgerContract):
    SCHEMA: ClassVar[str] = TOKEN_LEDGER_REPORT_SCHEMA

    binding_id: str
    lifecycle_event_count: int
    attribution_count: int
    accepted_criterion_count: int
    input_tokens: int
    output_tokens: int
    reused_tokens: int
    speculative_tokens: int
    tool_tokens: int
    retry_tokens: int
    failed_attempt_tokens: int
    provider_native_tokens: int
    fallback_tokens: int
    rejected_tokens: int
    abandoned_tokens: int
    total_cost_microunits: int
    accepted_evidence_gain: int
    criterion_costs: tuple[CriterionTokenCost, ...]
    cost_per_accepted_criterion_ratio: ExactTokenRatio
    tokens_per_accepted_criterion_ratio: ExactTokenRatio
    evidence_gain_per_thousand_tokens_ratio: ExactTokenRatio

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding_id", _text(self.binding_id, "binding_id"))
        for name in (
            "lifecycle_event_count",
            "attribution_count",
            "accepted_criterion_count",
            "input_tokens",
            "output_tokens",
            "reused_tokens",
            "speculative_tokens",
            "tool_tokens",
            "retry_tokens",
            "failed_attempt_tokens",
            "provider_native_tokens",
            "fallback_tokens",
            "rejected_tokens",
            "abandoned_tokens",
            "total_cost_microunits",
            "accepted_evidence_gain",
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name,
                    maximum=MAX_COST_MICROUNITS * MAX_EVENTS,
                ),
            )
        costs = _records(
            self.criterion_costs,
            CriterionTokenCost,
            field_name="criterion_costs",
            maximum=MAX_CRITERIA,
        )
        if len({item.criterion_id for item in costs}) != len(costs):
            raise TokenLedgerValidationError(
                "criterion_costs contains duplicate criteria"
            )
        object.__setattr__(
            self, "criterion_costs", tuple(sorted(costs, key=lambda x: x.criterion_id))
        )
        for name in (
            "cost_per_accepted_criterion_ratio",
            "tokens_per_accepted_criterion_ratio",
            "evidence_gain_per_thousand_tokens_ratio",
        ):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                value = ExactTokenRatio.from_dict(value)
            if not isinstance(value, ExactTokenRatio):
                raise TokenLedgerValidationError(
                    f"{name} must be ExactTokenRatio"
                )
            object.__setattr__(self, name, value)
        if self.lifecycle_event_count != self.attribution_count:
            raise TokenLedgerValidationError(
                "every lifecycle event must be attributed exactly once"
            )
        total_tokens = self.total_tokens
        if self.reused_tokens > self.input_tokens:
            raise TokenLedgerValidationError(
                "total reused tokens cannot exceed total input tokens"
            )
        if self.speculative_tokens > self.output_tokens:
            raise TokenLedgerValidationError(
                "total speculative tokens cannot exceed total output tokens"
            )
        if self.retry_tokens > total_tokens:
            raise TokenLedgerValidationError(
                "total retry tokens cannot exceed total tokens"
            )
        if self.failed_attempt_tokens > total_tokens:
            raise TokenLedgerValidationError(
                "total failed-attempt tokens cannot exceed total tokens"
            )
        if self.provider_native_tokens + self.fallback_tokens != self.total_tokens:
            raise TokenLedgerValidationError(
                "native and fallback token totals do not reconcile"
            )
        if self.rejected_tokens + self.abandoned_tokens > self.failed_attempt_tokens:
            raise TokenLedgerValidationError(
                "terminally unsuccessful tokens exceed failed-attempt tokens"
            )
        accepted_costs = tuple(item for item in costs if item.accepted)
        if self.accepted_criterion_count != len(accepted_costs):
            raise TokenLedgerValidationError(
                "accepted criterion count does not match criterion costs"
            )
        if sum(item.total_tokens for item in costs) != total_tokens:
            raise TokenLedgerValidationError(
                "criterion token costs do not reconcile with total tokens"
            )
        if (
            sum(item.cost_microunits for item in costs)
            != self.total_cost_microunits
        ):
            raise TokenLedgerValidationError(
                "criterion costs do not reconcile with total cost"
            )
        if (
            sum(item.evidence_gain for item in accepted_costs)
            != self.accepted_evidence_gain
        ):
            raise TokenLedgerValidationError(
                "criterion evidence does not reconcile with accepted evidence gain"
            )
        expected_ratios = {
            "cost_per_accepted_criterion_ratio": ExactTokenRatio(
                self.total_cost_microunits, self.accepted_criterion_count
            ),
            "tokens_per_accepted_criterion_ratio": ExactTokenRatio(
                total_tokens, self.accepted_criterion_count
            ),
            "evidence_gain_per_thousand_tokens_ratio": ExactTokenRatio(
                self.accepted_evidence_gain, total_tokens, 1_000
            ),
        }
        for name, expected in expected_ratios.items():
            if getattr(self, name) != expected:
                raise TokenLedgerValidationError(
                    f"{name} does not reconcile with report totals"
                )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.tool_tokens

    @property
    def cost_per_accepted_criterion_microunits(self) -> float:
        return self.cost_per_accepted_criterion_ratio.value

    @property
    def tokens_per_accepted_criterion(self) -> float:
        return self.tokens_per_accepted_criterion_ratio.value

    @property
    def evidence_gain_per_thousand_tokens(self) -> float:
        return self.evidence_gain_per_thousand_tokens_ratio.value

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": TOKEN_LEDGER_CONTRACT_VERSION,
            "binding_id": self.binding_id,
            "lifecycle_event_count": self.lifecycle_event_count,
            "attribution_count": self.attribution_count,
            "accepted_criterion_count": self.accepted_criterion_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reused_tokens": self.reused_tokens,
            "speculative_tokens": self.speculative_tokens,
            "tool_tokens": self.tool_tokens,
            "retry_tokens": self.retry_tokens,
            "failed_attempt_tokens": self.failed_attempt_tokens,
            "provider_native_tokens": self.provider_native_tokens,
            "fallback_tokens": self.fallback_tokens,
            "rejected_tokens": self.rejected_tokens,
            "abandoned_tokens": self.abandoned_tokens,
            "total_cost_microunits": self.total_cost_microunits,
            "accepted_evidence_gain": self.accepted_evidence_gain,
            "total_tokens": self.total_tokens,
            "criterion_costs": tuple(item.to_record() for item in self.criterion_costs),
            "cost_per_accepted_criterion_ratio": (
                self.cost_per_accepted_criterion_ratio.to_record()
            ),
            "tokens_per_accepted_criterion_ratio": (
                self.tokens_per_accepted_criterion_ratio.to_record()
            ),
            "evidence_gain_per_thousand_tokens_ratio": (
                self.evidence_gain_per_thousand_tokens_ratio.to_record()
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TokenLedgerReport":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "binding_id",
            "lifecycle_event_count",
            "attribution_count",
            "accepted_criterion_count",
            "input_tokens",
            "output_tokens",
            "reused_tokens",
            "speculative_tokens",
            "tool_tokens",
            "retry_tokens",
            "failed_attempt_tokens",
            "provider_native_tokens",
            "fallback_tokens",
            "rejected_tokens",
            "abandoned_tokens",
            "total_cost_microunits",
            "accepted_evidence_gain",
            "total_tokens",
            "criterion_costs",
            "cost_per_accepted_criterion_ratio",
            "tokens_per_accepted_criterion_ratio",
            "evidence_gain_per_thousand_tokens_ratio",
            "cost_per_accepted_criterion_microunits",
            "tokens_per_accepted_criterion",
            "evidence_gain_per_thousand_tokens",
            "content_id",
        }
        _closed(
            payload,
            schema=cls.SCHEMA,
            allowed=allowed,
            name="token ledger report",
        )
        result = cls(
            binding_id=payload.get("binding_id", ""),
            lifecycle_event_count=payload.get("lifecycle_event_count", 0),
            attribution_count=payload.get("attribution_count", 0),
            accepted_criterion_count=payload.get("accepted_criterion_count", 0),
            input_tokens=payload.get("input_tokens", 0),
            output_tokens=payload.get("output_tokens", 0),
            reused_tokens=payload.get("reused_tokens", 0),
            speculative_tokens=payload.get("speculative_tokens", 0),
            tool_tokens=payload.get("tool_tokens", 0),
            retry_tokens=payload.get("retry_tokens", 0),
            failed_attempt_tokens=payload.get("failed_attempt_tokens", 0),
            provider_native_tokens=payload.get("provider_native_tokens", 0),
            fallback_tokens=payload.get("fallback_tokens", 0),
            rejected_tokens=payload.get("rejected_tokens", 0),
            abandoned_tokens=payload.get("abandoned_tokens", 0),
            total_cost_microunits=payload.get("total_cost_microunits", 0),
            accepted_evidence_gain=payload.get("accepted_evidence_gain", 0),
            criterion_costs=payload.get("criterion_costs", ()),
            cost_per_accepted_criterion_ratio=payload.get(
                "cost_per_accepted_criterion_ratio", {}
            ),
            tokens_per_accepted_criterion_ratio=payload.get(
                "tokens_per_accepted_criterion_ratio", {}
            ),
            evidence_gain_per_thousand_tokens_ratio=payload.get(
                "evidence_gain_per_thousand_tokens_ratio", {}
            ),
        )
        scalar_claims = {
            "total_tokens": result.total_tokens,
            "cost_per_accepted_criterion_microunits": (
                result.cost_per_accepted_criterion_microunits
            ),
            "tokens_per_accepted_criterion": (
                result.tokens_per_accepted_criterion
            ),
            "evidence_gain_per_thousand_tokens": (
                result.evidence_gain_per_thousand_tokens
            ),
        }
        for name, actual in scalar_claims.items():
            if payload.get(name, actual) != actual:
                raise TokenLedgerValidationError(
                    f"{name} claim does not match report contents"
                )
        _claim(payload, result.content_id, "content_id")
        return result


@dataclass(frozen=True)
class SupervisorTokenLedger(_LedgerContract):
    """Population-complete token attribution for one bound supervisor task."""

    SCHEMA: ClassVar[str] = SUPERVISOR_TOKEN_LEDGER_SCHEMA

    binding: ResultBinding
    lifecycle_events: tuple[StageEvent, ...]
    terminal_attributions: tuple[TerminalCriterionAttribution, ...]
    attributions: tuple[TokenAttribution, ...]
    calibrations: tuple[FallbackTokenizerCalibration, ...] = ()

    def __post_init__(self) -> None:
        binding = _binding(self.binding)
        object.__setattr__(self, "binding", binding)
        events = _records(
            self.lifecycle_events,
            StageEvent,
            field_name="lifecycle_events",
            maximum=MAX_EVENTS,
        )
        terminals = _records(
            self.terminal_attributions,
            TerminalCriterionAttribution,
            field_name="terminal_attributions",
            maximum=MAX_EVENTS,
        )
        attributions = _records(
            self.attributions,
            TokenAttribution,
            field_name="attributions",
            maximum=MAX_EVENTS,
        )
        calibrations = _records(
            self.calibrations,
            FallbackTokenizerCalibration,
            field_name="calibrations",
            maximum=MAX_CALIBRATION_SAMPLES,
        )
        if not events:
            raise TokenLedgerValidationError(
                "ledger requires the complete non-empty lifecycle population"
            )
        event_ids = [item.event_id for item in events]
        if len(event_ids) != len(set(event_ids)):
            raise TokenLedgerValidationError(
                "lifecycle event population contains duplicates"
            )
        terminal_ids = [
            item.terminal_attribution_id for item in terminals
        ]
        if len(terminal_ids) != len(set(terminal_ids)):
            raise TokenLedgerValidationError(
                "terminal attribution population contains duplicates"
            )
        calibration_ids = [item.calibration_id for item in calibrations]
        if len(calibration_ids) != len(set(calibration_ids)):
            raise TokenLedgerValidationError(
                "calibration population contains duplicates"
            )
        for collection_name, records in (
            ("lifecycle event", events),
            ("terminal attribution", terminals),
            ("token attribution", attributions),
        ):
            for record in records:
                if record.binding.binding_id != binding.binding_id:
                    raise TokenLedgerValidationError(
                        f"{collection_name} is foreign-bound"
                    )
        events_by_id = {item.event_id: item for item in events}
        terminals_by_id = {
            item.terminal_attribution_id: item for item in terminals
        }
        event_attribution_ids = [item.event_id for item in attributions]
        if set(event_attribution_ids) != set(event_ids) or len(
            event_attribution_ids
        ) != len(event_ids):
            raise TokenLedgerValidationError(
                "every lifecycle event must be reconciled exactly once"
            )
        measurement_ids = [item.usage.measurement_id for item in attributions]
        if len(measurement_ids) != len(set(measurement_ids)):
            raise TokenLedgerValidationError(
                "provider usage population contains duplicated measurements"
            )
        accepted_pairs: set[tuple[str, str]] = set()
        terminal_attempt_pairs: set[tuple[str, int]] = set()
        used_terminal_ids: set[str] = set()
        used_calibration_ids: set[str] = set()
        calibration_by_id = {
            item.calibration_id: item for item in calibrations
        }
        for terminal in terminals:
            event = events_by_id.get(terminal.terminal_event_id)
            if event is None or not event.kind.terminal:
                raise TokenLedgerValidationError(
                    "terminal attribution references a missing or non-terminal event"
                )
            attempt_key = (terminal.criterion_id, event.attempt)
            if attempt_key in terminal_attempt_pairs:
                raise TokenLedgerValidationError(
                    "criterion attempt has duplicated terminal attribution"
                )
            terminal_attempt_pairs.add(attempt_key)
            if terminal.accepted:
                key = (terminal.binding.task_id, terminal.criterion_id)
                if key in accepted_pairs:
                    raise TokenLedgerValidationError(
                        "criterion has duplicated terminal acceptance"
                    )
                accepted_pairs.add(key)
        for attribution in attributions:
            event = events_by_id[attribution.event_id]
            if (
                attribution.stage != event.stage
                or attribution.attempt != event.attempt
            ):
                raise TokenLedgerValidationError(
                    "token attribution stage or attempt is foreign to lifecycle event"
                )
            terminal = terminals_by_id.get(
                attribution.terminal_attribution_id
            )
            if terminal is None:
                raise TokenLedgerValidationError(
                    "token usage is terminally unattributed"
                )
            used_terminal_ids.add(terminal.terminal_attribution_id)
            terminal_event = events_by_id[terminal.terminal_event_id]
            if attribution.attempt != terminal_event.attempt:
                raise TokenLedgerValidationError(
                    "token attribution attempt is foreign to terminal criterion"
                )
            if attribution.validation_result is not terminal.validation_result:
                raise TokenLedgerValidationError(
                    "token attribution validation result is inconsistent"
                )
            expected_failed = (
                0 if terminal.accepted else attribution.usage.total_tokens
            )
            if attribution.usage.failed_attempt_tokens != expected_failed:
                raise TokenLedgerValidationError(
                    "failed-attempt token classification is incomplete"
                )
            usage = attribution.usage
            if usage.source is UsageSource.CALIBRATED_FALLBACK:
                used_calibration_ids.add(usage.calibration_id)
                calibration = calibration_by_id.get(usage.calibration_id)
                if calibration is None:
                    raise TokenLedgerValidationError(
                        "fallback usage cites a missing calibration"
                    )
                if not calibration.supports(usage.envelope):
                    raise TokenLedgerValidationError(
                        "fallback calibration is foreign to provider/model envelope"
                    )
        if set(terminal_ids) != used_terminal_ids:
            raise TokenLedgerValidationError(
                "terminal attribution population contains unused records"
            )
        if set(calibration_ids) != used_calibration_ids:
            raise TokenLedgerValidationError(
                "calibration population contains unused records"
            )
        object.__setattr__(
            self,
            "lifecycle_events",
            tuple(
                sorted(
                    events,
                    key=lambda item: (
                        item.attempt,
                        item.sequence,
                        item.stage,
                        item.event_id,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "terminal_attributions",
            tuple(
                sorted(
                    terminals,
                    key=lambda item: (
                        item.criterion_id,
                        item.disposition.value,
                        item.terminal_attribution_id,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "attributions",
            tuple(
                sorted(
                    attributions,
                    key=lambda item: (
                        item.attempt,
                        item.stage,
                        item.event_id,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "calibrations",
            tuple(sorted(calibrations, key=lambda item: item.calibration_id)),
        )
        if len(self.canonical_bytes()) > MAX_SERIALIZED_LEDGER_BYTES:
            raise TokenLedgerValidationError(
                "token ledger exceeds its serialized size bound"
            )

    @property
    def ledger_id(self) -> str:
        return self.content_id

    @property
    def report(self) -> TokenLedgerReport:
        return self.build_report()

    def build_report(self) -> TokenLedgerReport:
        terminals = {
            item.terminal_attribution_id: item
            for item in self.terminal_attributions
        }
        usages = [item.usage for item in self.attributions]
        criterion_ids = sorted(
            {item.criterion_id for item in self.terminal_attributions}
        )
        costs = []
        for criterion_id in criterion_ids:
            criterion_terminals = [
                item
                for item in self.terminal_attributions
                if item.criterion_id == criterion_id
            ]
            ids = {
                item.terminal_attribution_id for item in criterion_terminals
            }
            criterion_usages = [
                item.usage
                for item in self.attributions
                if item.terminal_attribution_id in ids
            ]
            accepted = [item for item in criterion_terminals if item.accepted]
            costs.append(
                CriterionTokenCost(
                    criterion_id=criterion_id,
                    accepted=bool(accepted),
                    attempt_count=len(
                        {
                            item.attempt
                            for item in self.attributions
                            if item.terminal_attribution_id in ids
                        }
                    ),
                    total_tokens=sum(item.total_tokens for item in criterion_usages),
                    cost_microunits=sum(
                        item.cost_microunits for item in criterion_usages
                    ),
                    evidence_gain=sum(item.evidence_gain for item in accepted),
                )
            )
        accepted_criteria = [item for item in costs if item.accepted]
        accepted_count = len(accepted_criteria)
        total_tokens = sum(item.total_tokens for item in usages)
        total_cost = sum(item.cost_microunits for item in usages)
        evidence_gain = sum(item.evidence_gain for item in accepted_criteria)
        return TokenLedgerReport(
            binding_id=self.binding.binding_id,
            lifecycle_event_count=len(self.lifecycle_events),
            attribution_count=len(self.attributions),
            accepted_criterion_count=accepted_count,
            input_tokens=sum(item.input_tokens for item in usages),
            output_tokens=sum(item.output_tokens for item in usages),
            reused_tokens=sum(item.reused_tokens for item in usages),
            speculative_tokens=sum(item.speculative_tokens for item in usages),
            tool_tokens=sum(item.tool_tokens for item in usages),
            retry_tokens=sum(item.retry_tokens for item in usages),
            failed_attempt_tokens=sum(
                item.failed_attempt_tokens for item in usages
            ),
            provider_native_tokens=sum(
                item.total_tokens
                for item in usages
                if item.source is UsageSource.PROVIDER_NATIVE
            ),
            fallback_tokens=sum(
                item.total_tokens
                for item in usages
                if item.source is UsageSource.CALIBRATED_FALLBACK
            ),
            rejected_tokens=sum(
                attribution.usage.total_tokens
                for attribution in self.attributions
                if terminals[
                    attribution.terminal_attribution_id
                ].disposition
                is TerminalDisposition.REJECTED
            ),
            abandoned_tokens=sum(
                attribution.usage.total_tokens
                for attribution in self.attributions
                if terminals[
                    attribution.terminal_attribution_id
                ].disposition
                is TerminalDisposition.ABANDONED
            ),
            total_cost_microunits=total_cost,
            accepted_evidence_gain=evidence_gain,
            criterion_costs=tuple(costs),
            cost_per_accepted_criterion_ratio=ExactTokenRatio(
                total_cost, accepted_count
            ),
            tokens_per_accepted_criterion_ratio=ExactTokenRatio(
                total_tokens, accepted_count
            ),
            evidence_gain_per_thousand_tokens_ratio=ExactTokenRatio(
                evidence_gain, total_tokens, 1_000
            ),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": TOKEN_LEDGER_CONTRACT_VERSION,
            "binding": self.binding.to_record(),
            "lifecycle_events": tuple(
                item.to_record() for item in self.lifecycle_events
            ),
            "terminal_attributions": tuple(
                item.to_record() for item in self.terminal_attributions
            ),
            "attributions": tuple(
                item.to_record() for item in self.attributions
            ),
            "calibrations": tuple(
                item.to_record() for item in self.calibrations
            ),
            "report": self.report.to_record(),
        }

    def to_dict(self, *, include_ledger_id: bool = False) -> dict[str, Any]:
        payload = super().to_dict()
        if include_ledger_id:
            payload["ledger_id"] = self.ledger_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SupervisorTokenLedger":
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "binding",
            "lifecycle_events",
            "terminal_attributions",
            "attributions",
            "calibrations",
            "report",
            "ledger_id",
            "content_id",
        }
        _closed(
            payload,
            schema=cls.SCHEMA,
            allowed=allowed,
            name="supervisor token ledger",
        )
        result = cls(
            binding=payload.get("binding", {}),
            lifecycle_events=payload.get("lifecycle_events", ()),
            terminal_attributions=payload.get(
                "terminal_attributions", ()
            ),
            attributions=payload.get("attributions", ()),
            calibrations=payload.get("calibrations", ()),
        )
        supplied_report = payload.get("report")
        if supplied_report is not None:
            try:
                report = TokenLedgerReport.from_dict(supplied_report)
            except TokenLedgerValidationError as exc:
                raise TokenLedgerValidationError(
                    "ledger report does not reconcile from lifecycle usage"
                ) from exc
            if report.content_id != result.report.content_id:
                raise TokenLedgerValidationError(
                    "ledger report does not reconcile from lifecycle usage"
                )
        _claim(payload, result.ledger_id, "ledger_id", "content_id")
        return result


TokenLedger = SupervisorTokenLedger


def build_token_ledger(
    *,
    binding: ResultBinding,
    lifecycle_events: Sequence[StageEvent | Mapping[str, Any]],
    terminal_attributions: Sequence[
        TerminalCriterionAttribution | Mapping[str, Any]
    ],
    attributions: Sequence[TokenAttribution | Mapping[str, Any]],
    calibrations: Sequence[
        FallbackTokenizerCalibration | Mapping[str, Any]
    ] = (),
) -> SupervisorTokenLedger:
    """Construct and reconcile a population-complete supervisor token ledger."""

    return SupervisorTokenLedger(
        binding=binding,
        lifecycle_events=tuple(lifecycle_events),  # type: ignore[arg-type]
        terminal_attributions=tuple(terminal_attributions),  # type: ignore[arg-type]
        attributions=tuple(attributions),  # type: ignore[arg-type]
        calibrations=tuple(calibrations),  # type: ignore[arg-type]
    )


def _v1_validation_result(receipt: Any) -> ValidationResult:
    raw = str(
        getattr(getattr(receipt.validation, "status", ""), "value", "")
    )
    return {
        "passed": ValidationResult.PASSED,
        "failed": ValidationResult.FAILED,
        "not_required": ValidationResult.NOT_REQUIRED,
        "skipped": ValidationResult.NOT_RUN,
    }.get(raw, ValidationResult.NOT_RUN)


def adapt_efficiency_receipt(
    receipt: Any,
    *,
    binding: ResultBinding,
    envelope: ProviderModelEnvelope,
    criterion_id: str,
    context_id: str | None = None,
    usage_source: UsageSource | str = UsageSource.PROVIDER_NATIVE,
    calibration: FallbackTokenizerCalibration | None = None,
) -> SupervisorTokenLedger:
    """Adapt one v1 aggregate receipt without dropping its retry accounting.

    Synthetic stage events are used because v1 receipts store aggregate stage
    timing rather than the original v2 event population.  Token counts remain
    provider-native and every retry observation becomes its own attempt.
    """

    try:
        retries = tuple(receipt.retries)
        final_attempt = int(receipt.attempt)
        accepted = bool(receipt.accepted)
        tokens = receipt.tokens
        total_cost = int(receipt.total_cost_microunits)
        context = context_id or str(receipt.context_digest)
        terminal_outcome = str(
            getattr(
                receipt.terminal.outcome,
                "value",
                receipt.terminal.outcome,
            )
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise TokenLedgerValidationError(
            "receipt does not expose the v1 EfficiencyReceipt contract"
        ) from exc
    if final_attempt != len(retries) + 1:
        raise TokenLedgerValidationError(
            "v1 retry population does not reconcile with final attempt"
        )
    source = _enum(usage_source, UsageSource, "usage_source")
    if source is UsageSource.CALIBRATED_FALLBACK:
        if calibration is None:
            raise TokenLedgerValidationError(
                "fallback v1 adaptation requires a calibration"
            )
        if not calibration.supports(envelope):
            raise TokenLedgerValidationError(
                "fallback v1 calibration is foreign to provider/model envelope"
            )
    elif calibration is not None:
        raise TokenLedgerValidationError(
            "provider-native v1 adaptation cannot attach a fallback calibration"
        )
    retry_input = sum(item.tokens.input_tokens for item in retries)
    retry_output = sum(item.tokens.output_tokens for item in retries)
    retry_reused = sum(item.tokens.reused_tokens for item in retries)
    initial = (
        tokens.input_tokens - retry_input,
        tokens.output_tokens - retry_output,
        tokens.reused_tokens - retry_reused,
    )
    if min(initial) < 0:
        raise TokenLedgerValidationError(
            "v1 retry tokens exceed aggregate provider usage"
        )
    attempt_counts = [initial] + [
        (
            item.tokens.input_tokens,
            item.tokens.output_tokens,
            item.tokens.reused_tokens,
        )
        for item in retries
    ]
    events: list[StageEvent] = []
    terminals: list[TerminalCriterionAttribution] = []
    attributions: list[TokenAttribution] = []
    validation = _v1_validation_result(receipt)
    for offset, (input_tokens, output_tokens, reused_tokens) in enumerate(
        attempt_counts, start=1
    ):
        is_final = offset == final_attempt
        disposition = (
            TerminalDisposition.ACCEPTED
            if is_final and accepted
            else (
                TerminalDisposition.ABANDONED
                if is_final and terminal_outcome == "cancelled"
                else TerminalDisposition.REJECTED
            )
        )
        attempt_validation = (
            validation if is_final else ValidationResult.FAILED
        )
        event = StageEvent(
            binding=binding,
            stage="inference",
            attempt=offset,
            sequence=0,
            kind=(
                StageEventKind.COMPLETED
                if disposition.accepted
                else StageEventKind.FAILED
            ),
            authority="validation",
            occurred_at=f"1970-01-01T00:00:{offset - 1:02d}.000000Z",
            reason_code="" if disposition.accepted else "v1-attempt-rejected",
        )
        events.append(event)
        terminal = TerminalCriterionAttribution(
            binding=binding,
            terminal_event_id=event.event_id,
            criterion_id=criterion_id,
            disposition=disposition,
            validation_result=attempt_validation,
            evidence_gain=(
                int(getattr(receipt, "accepted_evidence_gain", 0))
                if disposition.accepted
                else 0
            ),
            reason_code="" if disposition.accepted else "v1-attempt-rejected",
        )
        terminals.append(terminal)
        total = input_tokens + output_tokens
        usage = ProviderTokenUsage(
            measurement_id=f"{receipt.receipt_id}:attempt:{offset}",
            envelope=envelope,
            source=source,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reused_tokens=reused_tokens,
            retry_tokens=total if offset > 1 else 0,
            failed_attempt_tokens=0 if disposition.accepted else total,
            cost_microunits=total_cost if is_final else 0,
            calibration_id=(
                calibration.calibration_id
                if calibration is not None
                else ""
            ),
        )
        attributions.append(
            TokenAttribution(
                binding=binding,
                event_id=event.event_id,
                stage=event.stage,
                attempt=offset,
                context_id=context,
                cache_decision=(
                    CacheDecision.HIT
                    if reused_tokens
                    else CacheDecision.MISS
                ),
                validation_result=attempt_validation,
                terminal_attribution_id=terminal.terminal_attribution_id,
                usage=usage,
            )
        )
    return SupervisorTokenLedger(
        binding=binding,
        lifecycle_events=tuple(events),
        terminal_attributions=tuple(terminals),
        attributions=tuple(attributions),
        calibrations=(calibration,) if calibration is not None else (),
    )


adapt_v1_efficiency_receipt = adapt_efficiency_receipt


__all__ = [
    "ACCEPTED_CRITERION_TOKEN_GOAL_ID",
    "ACCEPTED_CRITERION_TOKEN_REQUIREMENT_ID",
    "CacheDecision",
    "CalibrationSample",
    "CriterionTokenCost",
    "ExactTokenRatio",
    "FallbackTokenizerCalibration",
    "LifecycleTokenAttribution",
    "NativeTokenUsage",
    "ProviderModelEnvelope",
    "ProviderTokenUsage",
    "ProviderTokenizerEnvelope",
    "SCHEMA_VERSION",
    "SupervisorTokenLedger",
    "TerminalAttribution",
    "TerminalCriterionAttribution",
    "TerminalDisposition",
    "TokenAttribution",
    "TokenLedger",
    "TokenLedgerReport",
    "TokenLedgerValidationError",
    "TokenizerCalibration",
    "TokenizerCalibrationSample",
    "UsageSource",
    "ValidationResult",
    "adapt_efficiency_receipt",
    "adapt_v1_efficiency_receipt",
    "build_token_ledger",
    "calibrate_fallback_tokenizer",
]
