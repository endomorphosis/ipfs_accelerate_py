"""Closed, canonical receipts for ASEH task-efficiency measurements.

This is a data contract, not an attestation mechanism.  In particular, an
``observed`` or ``verified`` value records the state claimed by its supplied
evidence reference; callers still need to apply their own admission policy.
The deliberately small wire format keeps prompts, source bodies, decoded model
output, and unbounded diagnostic graphs out of durable benchmark receipts.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final


CONTRACT_VERSION: Final[int] = 1
SCHEMA_VERSION: Final[int] = CONTRACT_VERSION
TASK_EFFICIENCY_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/task-efficiency-receipt@1"
)
PAIRED_BENCHMARK_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/paired-benchmark-manifest@1"
)
MAX_TEXT_BYTES: Final[int] = 512
MAX_IDENTIFIER_BYTES: Final[int] = 256
MAX_INTEGER: Final[int] = 10**18
MAX_FACTS: Final[int] = 64
MAX_REFERENCES: Final[int] = 128
MAX_ARMS: Final[int] = 3


class EfficiencyReceiptValidationError(ValueError):
    """Raised when a receipt or paired manifest is not an admitted contract."""


# A shorter compatibility spelling is useful to contract consumers.
ContractValidationError = EfficiencyReceiptValidationError


class TruthState(str, Enum):
    """Truth status of one field; these states are intentionally not ordered."""

    MEASURED = "measured"
    ESTIMATED = "estimated"
    SIMULATED = "simulated"
    UNAVAILABLE = "unavailable"
    ATTEMPTED = "attempted"
    OBSERVED = "observed"
    VERIFIED = "verified"


class TerminalOutcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class BenchmarkArm(str, Enum):
    DIRECT = "direct"
    SEALED_CURRENT = "sealed_current"
    CANDIDATE = "candidate"


_REQUIRED_RECEIPT_SECTIONS: Final[tuple[str, ...]] = (
    "identity",
    "model",
    "compute",
    "cost",
    "work",
    "quality",
    "evidence",
    "terminal",
)
_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "task_id",
    "attempt_id",
    "goal_id",
    "repository",
    "repository_commit",
    "repository_tree",
    "objective_revision",
    "policy_revision",
    "environment_id",
    "input_digest",
)
_MODEL_FIELDS: Final[tuple[str, ...]] = (
    "provider",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "request_id",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
    "call_count",
)
_COMPUTE_FIELDS: Final[tuple[str, ...]] = (
    "wall_clock_ms",
    "cpu_ms",
    "peak_rss_bytes",
    "gpu_ms",
    "read_bytes",
    "write_bytes",
    "network_bytes",
)
_COST_FIELDS: Final[tuple[str, ...]] = (
    "reported_cost_microusd",
    "estimated_cost_microusd",
    "price_snapshot_id",
    "currency",
)
_WORK_FIELDS: Final[tuple[str, ...]] = (
    "deterministic_steps",
    "model_calls",
    "validation_runs",
    "proof_runs",
    "retries",
    "merge_attempts",
    "human_interventions",
    "changed_files",
)
_QUALITY_FIELDS: Final[tuple[str, ...]] = (
    "acceptance_criteria_total",
    "acceptance_criteria_passed",
    "acceptance_criteria_failed",
    "validation_verdict",
    "proof_verdict",
)
_NUMERIC_FACT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
        "call_count",
        "wall_clock_ms",
        "cpu_ms",
        "peak_rss_bytes",
        "gpu_ms",
        "read_bytes",
        "write_bytes",
        "network_bytes",
        "reported_cost_microusd",
        "estimated_cost_microusd",
        "deterministic_steps",
        "model_calls",
        "validation_runs",
        "proof_runs",
        "retries",
        "merge_attempts",
        "human_interventions",
        "changed_files",
        "acceptance_criteria_total",
        "acceptance_criteria_passed",
        "acceptance_criteria_failed",
    }
)
_SECTION_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "identity": _IDENTITY_FIELDS,
    "model": _MODEL_FIELDS,
    "compute": _COMPUTE_FIELDS,
    "cost": _COST_FIELDS,
    "work": _WORK_FIELDS,
    "quality": _QUALITY_FIELDS,
}


def _fail(message: str) -> None:
    raise EfficiencyReceiptValidationError(message)


def _text(value: Any, name: str, *, required: bool = True, limit: int = MAX_TEXT_BYTES) -> str:
    if not isinstance(value, str):
        _fail(f"{name} must be text")
    result = value.strip()
    if required and not result:
        _fail(f"{name} is required")
    if "\x00" in result or len(result.encode("utf-8")) > limit:
        _fail(f"{name} is unsafe or exceeds its bound")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{name} must be an integer")
    if value < minimum or value > MAX_INTEGER:
        _fail(f"{name} is outside supported bounds")
    return value


def _closed(payload: Any, allowed: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        _fail(f"{name} must be an object")
    if set(payload) - allowed:
        _fail(f"{name} contains unknown fields")
    return payload


def _enum(value: Any, enum_type: type[Enum], name: str) -> Enum:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        raise EfficiencyReceiptValidationError(f"{name} has an unsupported value") from exc


def _canonical_value(value: Any, name: str = "value") -> Any:
    """Normalize only JSON values and reject floats/non-finite numeric forms."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return _integer(value, name, minimum=-MAX_INTEGER)
    if isinstance(value, float):
        _fail(f"{name} must not be a floating-point number")
    if isinstance(value, Mapping):
        if len(value) > MAX_FACTS:
            _fail(f"{name} exceeds its object bound")
        output: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = _text(key, f"{name} key", limit=MAX_IDENTIFIER_BYTES)
            output[normalized_key] = _canonical_value(item, f"{name}.{normalized_key}")
        return output
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        if len(value) > MAX_REFERENCES:
            _fail(f"{name} exceeds its sequence bound")
        return [_canonical_value(item, name) for item in value]
    _fail(f"{name} is not JSON-compatible")


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes without whitespace or floats."""

    return json.dumps(
        _canonical_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def content_identity(value: Any) -> str:
    """Return a CIDv1 dag-json SHA-256 identity of canonical bytes."""

    digest = hashlib.sha256(canonical_json_bytes(value)).digest()
    return "b" + base64.b32encode(b"\x01\xa9\x02\x12\x20" + digest).decode("ascii").rstrip("=").lower()


@dataclass(frozen=True)
class TelemetryFact:
    """One bounded field value with an explicit, non-upgradeable truth state."""

    state: TruthState | str
    value: Any = None
    unit: str = ""
    reason_code: str = ""
    basis_ref: str = ""
    evidence_ref: str = ""
    verifier_ref: str = ""

    def __post_init__(self) -> None:
        state = _enum(self.state, TruthState, "fact state")
        object.__setattr__(self, "state", state)
        value = _canonical_value(self.value, "fact value")
        object.__setattr__(self, "value", value)
        for name in ("unit", "reason_code", "basis_ref", "evidence_ref", "verifier_ref"):
            object.__setattr__(self, name, _text(getattr(self, name), name, required=False))
        has_value = value is not None
        if state is TruthState.UNAVAILABLE:
            if has_value or not self.reason_code:
                _fail("unavailable facts require a reason_code and no value")
            if self.basis_ref or self.evidence_ref or self.verifier_ref:
                _fail("unavailable facts cannot claim a basis or evidence")
        elif state is TruthState.ATTEMPTED:
            if has_value or not self.evidence_ref:
                _fail("attempted facts require an evidence_ref and no value")
            if self.reason_code or self.basis_ref or self.verifier_ref:
                _fail("attempted facts cannot claim a result, basis, or verifier")
        else:
            if not has_value:
                _fail(f"{state.value} facts require a value")
            if state is TruthState.ESTIMATED:
                if not self.basis_ref or self.evidence_ref or self.verifier_ref:
                    _fail("estimated facts require basis_ref only")
            elif state is TruthState.SIMULATED:
                if not self.basis_ref or self.evidence_ref or self.verifier_ref:
                    _fail("simulated facts require basis_ref only")
            elif state is TruthState.MEASURED:
                if not self.evidence_ref or self.basis_ref or self.verifier_ref:
                    _fail("measured facts require evidence_ref only")
            elif state is TruthState.OBSERVED:
                if not self.evidence_ref or self.basis_ref or self.verifier_ref:
                    _fail("observed facts require evidence_ref only")
            elif state is TruthState.VERIFIED:
                if not self.evidence_ref or not self.verifier_ref or self.basis_ref:
                    _fail("verified facts require evidence_ref and verifier_ref")
        if has_value and isinstance(value, int) and not self.unit:
            _fail("numeric facts require an explicit unit")
        if not has_value and self.unit:
            _fail("facts without values cannot have a unit")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"state": self.state.value}
        if self.value is not None:
            result["value"] = self.value
            if self.unit:
                result["unit"] = self.unit
        if self.reason_code:
            result["reason_code"] = self.reason_code
        if self.basis_ref:
            result["basis_ref"] = self.basis_ref
        if self.evidence_ref:
            result["evidence_ref"] = self.evidence_ref
        if self.verifier_ref:
            result["verifier_ref"] = self.verifier_ref
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TelemetryFact":
        data = _closed(payload, {"state", "value", "unit", "reason_code", "basis_ref", "evidence_ref", "verifier_ref"}, "telemetry fact")
        return cls(
            state=data.get("state"), value=data.get("value"), unit=data.get("unit", ""),
            reason_code=data.get("reason_code", ""), basis_ref=data.get("basis_ref", ""),
            evidence_ref=data.get("evidence_ref", ""), verifier_ref=data.get("verifier_ref", ""),
        )


# Measurement is a natural spelling at provider/compute call sites.
Measurement = TelemetryFact


def _facts(payload: Any, section: str) -> dict[str, TelemetryFact]:
    fields = _SECTION_FIELDS[section]
    data = _closed(payload, set(fields), section)
    if set(data) != set(fields):
        _fail(f"{section} must contain every required field")
    facts = {field: TelemetryFact.from_dict(data[field]) for field in fields}
    for field, fact in facts.items():
        if field in _NUMERIC_FACT_FIELDS and fact.value is not None:
            _integer(fact.value, field)
    return facts


def _identity(payload: Any) -> dict[str, str]:
    data = _closed(payload, set(_IDENTITY_FIELDS), "identity")
    if set(data) != set(_IDENTITY_FIELDS):
        _fail("identity must contain every required field")
    return {field: _text(data[field], field, limit=MAX_IDENTIFIER_BYTES) for field in _IDENTITY_FIELDS}


def _references(value: Any, name: str, *, minimum: int = 0) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(f"{name} must be an array")
    if len(value) < minimum or len(value) > MAX_REFERENCES:
        _fail(f"{name} is outside supported bounds")
    result = tuple(_text(item, name, limit=MAX_IDENTIFIER_BYTES) for item in value)
    if len(set(result)) != len(result):
        _fail(f"{name} must be unique")
    return tuple(sorted(result))


@dataclass(frozen=True)
class TaskEfficiencyReceipt:
    """Complete per-attempt telemetry with all field families present."""

    SCHEMA: ClassVar[str] = TASK_EFFICIENCY_RECEIPT_SCHEMA
    identity: Mapping[str, Any]
    model: Mapping[str, Any]
    compute: Mapping[str, Any]
    cost: Mapping[str, Any]
    work: Mapping[str, Any]
    quality: Mapping[str, Any]
    evidence: Mapping[str, Any]
    terminal: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity", _identity(self.identity))
        for section in ("model", "compute", "cost", "work", "quality"):
            object.__setattr__(self, section, _facts(getattr(self, section), section))
        object.__setattr__(self, "evidence", self._validate_evidence(self.evidence))
        object.__setattr__(self, "terminal", self._validate_terminal(self.terminal))
        self._validate_cross_fields()

    @staticmethod
    def _validate_evidence(payload: Any) -> dict[str, Any]:
        data = _closed(payload, {"state", "receipt_references", "raw_log_references", "validator", "verification"}, "evidence")
        if set(data) != {"state", "receipt_references", "raw_log_references", "validator", "verification"}:
            _fail("evidence must contain every required field")
        state = _enum(data["state"], TruthState, "evidence state")
        receipts = _references(data["receipt_references"], "receipt_references")
        raw_logs = _references(data["raw_log_references"], "raw_log_references")
        validator = TelemetryFact.from_dict(data["validator"])
        verification = TelemetryFact.from_dict(data["verification"])
        if state is TruthState.UNAVAILABLE and (receipts or raw_logs):
            _fail("unavailable evidence cannot contain references")
        if state in {TruthState.ATTEMPTED, TruthState.OBSERVED, TruthState.VERIFIED} and not receipts:
            _fail("attempted, observed, and verified evidence require receipt references")
        if state is TruthState.VERIFIED and verification.state is not TruthState.VERIFIED:
            _fail("verified evidence requires verified verification")
        return {"state": state, "receipt_references": receipts, "raw_log_references": raw_logs, "validator": validator, "verification": verification}

    @staticmethod
    def _validate_terminal(payload: Any) -> dict[str, Any]:
        data = _closed(payload, {"outcome", "state", "reason_code", "terminal_receipt_ref"}, "terminal")
        if set(data) != {"outcome", "state", "reason_code", "terminal_receipt_ref"}:
            _fail("terminal must contain every required field")
        outcome = _enum(data["outcome"], TerminalOutcome, "terminal outcome")
        state = _enum(data["state"], TruthState, "terminal state")
        reason = _text(data["reason_code"], "terminal reason_code", required=False)
        reference = _text(data["terminal_receipt_ref"], "terminal_receipt_ref", required=False, limit=MAX_IDENTIFIER_BYTES)
        if state in {TruthState.UNAVAILABLE, TruthState.ATTEMPTED} and outcome is TerminalOutcome.ACCEPTED:
            _fail("an unobserved terminal state cannot be accepted")
        if state in {TruthState.OBSERVED, TruthState.VERIFIED} and not reference:
            _fail("observed and verified terminal states require a terminal receipt")
        if state is TruthState.UNAVAILABLE and not reason:
            _fail("unavailable terminal states require a reason")
        if outcome is TerminalOutcome.ACCEPTED and state is not TruthState.VERIFIED:
            _fail("accepted terminal outcomes require verified state")
        return {"outcome": outcome, "state": state, "reason_code": reason, "terminal_receipt_ref": reference}

    def _validate_cross_fields(self) -> None:
        quality = self.quality
        total = quality["acceptance_criteria_total"].value
        passed = quality["acceptance_criteria_passed"].value
        failed = quality["acceptance_criteria_failed"].value
        if all(isinstance(item, int) for item in (total, passed, failed)) and passed + failed > total:
            _fail("quality passed plus failed criteria cannot exceed total")
        if self.terminal["outcome"] is TerminalOutcome.ACCEPTED:
            if self.evidence["state"] is not TruthState.VERIFIED:
                _fail("accepted terminal outcomes require verified evidence")
            verdict = quality["validation_verdict"]
            if verdict.value != "passed" or verdict.state is not TruthState.VERIFIED:
                _fail("accepted terminal outcomes require verified passed validation")

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "identity": dict(self.identity),
            "model": {key: value.to_dict() for key, value in self.model.items()},
            "compute": {key: value.to_dict() for key, value in self.compute.items()},
            "cost": {key: value.to_dict() for key, value in self.cost.items()},
            "work": {key: value.to_dict() for key, value in self.work.items()},
            "quality": {key: value.to_dict() for key, value in self.quality.items()},
            "evidence": {
                "state": self.evidence["state"].value,
                "receipt_references": list(self.evidence["receipt_references"]),
                "raw_log_references": list(self.evidence["raw_log_references"]),
                "validator": self.evidence["validator"].to_dict(),
                "verification": self.evidence["verification"].to_dict(),
            },
            "terminal": {
                "outcome": self.terminal["outcome"].value,
                "state": self.terminal["state"].value,
                "reason_code": self.terminal["reason_code"],
                "terminal_receipt_ref": self.terminal["terminal_receipt_ref"],
            },
        }

    @property
    def content_id(self) -> str:
        return content_identity(self._body())

    @property
    def receipt_id(self) -> str:
        return self.content_id

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self._body())

    def canonical_json(self) -> str:
        return self.canonical_bytes().decode("utf-8")

    def to_dict(self, *, include_content_id: bool = True) -> dict[str, Any]:
        result = self._body()
        if include_content_id:
            result["content_id"] = self.content_id
            result["receipt_id"] = self.receipt_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskEfficiencyReceipt":
        data = _closed(payload, {"schema", "schema_version", "contract_version", *_REQUIRED_RECEIPT_SECTIONS, "content_id", "receipt_id"}, "task efficiency receipt")
        if data.get("schema") != cls.SCHEMA or data.get("schema_version") != SCHEMA_VERSION or data.get("contract_version") != CONTRACT_VERSION:
            _fail("task efficiency receipt has an unsupported schema or version")
        if not set(_REQUIRED_RECEIPT_SECTIONS) <= set(data):
            _fail("task efficiency receipt is missing a required section")
        result = cls(**{section: data[section] for section in _REQUIRED_RECEIPT_SECTIONS})
        for field in ("content_id", "receipt_id"):
            if field in data and data[field] != result.content_id:
                _fail(f"{field} does not match canonical content")
        return result

    @classmethod
    def from_canonical_bytes(cls, value: bytes) -> "TaskEfficiencyReceipt":
        if not isinstance(value, bytes):
            _fail("canonical receipt must be bytes")
        try:
            payload = json.loads(value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EfficiencyReceiptValidationError("canonical receipt bytes are invalid JSON") from exc
        result = cls.from_dict(payload)
        if value != result.canonical_bytes():
            _fail("receipt bytes are not canonical")
        return result


EfficiencyReceipt = TaskEfficiencyReceipt


@dataclass(frozen=True)
class PairedBenchmarkManifest:
    """Sealed comparison of direct, sealed-current, and candidate receipts."""

    SCHEMA: ClassVar[str] = PAIRED_BENCHMARK_MANIFEST_SCHEMA
    benchmark_id: str
    cohort_id: str
    input_digest: str
    gate_digest: str
    arms: Mapping[str, Any]
    evidence_state: TruthState | str

    def __post_init__(self) -> None:
        for name in ("benchmark_id", "cohort_id", "input_digest", "gate_digest"):
            object.__setattr__(self, name, _text(getattr(self, name), name, limit=MAX_IDENTIFIER_BYTES))
        object.__setattr__(self, "evidence_state", _enum(self.evidence_state, TruthState, "evidence_state"))
        data = _closed(self.arms, {member.value for member in BenchmarkArm}, "paired benchmark arms")
        if set(data) != {member.value for member in BenchmarkArm}:
            _fail("paired benchmark manifests require exactly direct, sealed_current, and candidate arms")
        normalized: dict[str, dict[str, Any]] = {}
        for arm in BenchmarkArm:
            item = _closed(data[arm.value], {"receipt_id", "input_digest", "gate_digest", "state"}, f"{arm.value} arm")
            if set(item) != {"receipt_id", "input_digest", "gate_digest", "state"}:
                _fail(f"{arm.value} arm is incomplete")
            receipt_id = _text(item["receipt_id"], f"{arm.value} receipt_id", limit=MAX_IDENTIFIER_BYTES)
            input_digest = _text(item["input_digest"], f"{arm.value} input_digest", limit=MAX_IDENTIFIER_BYTES)
            gate_digest = _text(item["gate_digest"], f"{arm.value} gate_digest", limit=MAX_IDENTIFIER_BYTES)
            state = _enum(item["state"], TruthState, f"{arm.value} state")
            if input_digest != self.input_digest or gate_digest != self.gate_digest:
                _fail("paired benchmark arms must use the manifest input and gate digests")
            normalized[arm.value] = {"receipt_id": receipt_id, "input_digest": input_digest, "gate_digest": gate_digest, "state": state}
        if len({entry["receipt_id"] for entry in normalized.values()}) != MAX_ARMS:
            _fail("paired benchmark arms must reference distinct receipts")
        if self.evidence_state is TruthState.VERIFIED and any(entry["state"] is not TruthState.VERIFIED for entry in normalized.values()):
            _fail("verified paired manifests require verified arms")
        object.__setattr__(self, "arms", normalized)

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "benchmark_id": self.benchmark_id,
            "cohort_id": self.cohort_id,
            "input_digest": self.input_digest,
            "gate_digest": self.gate_digest,
            "evidence_state": self.evidence_state.value,
            "arms": {name: {**item, "state": item["state"].value} for name, item in self.arms.items()},
        }

    @property
    def content_id(self) -> str:
        return content_identity(self._body())

    @property
    def manifest_id(self) -> str:
        return self.content_id

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self._body())

    def canonical_json(self) -> str:
        return self.canonical_bytes().decode("utf-8")

    def to_dict(self, *, include_content_id: bool = True) -> dict[str, Any]:
        result = self._body()
        if include_content_id:
            result["content_id"] = self.content_id
            result["manifest_id"] = self.manifest_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairedBenchmarkManifest":
        data = _closed(payload, {"schema", "schema_version", "contract_version", "benchmark_id", "cohort_id", "input_digest", "gate_digest", "evidence_state", "arms", "content_id", "manifest_id"}, "paired benchmark manifest")
        if data.get("schema") != cls.SCHEMA or data.get("schema_version") != SCHEMA_VERSION or data.get("contract_version") != CONTRACT_VERSION:
            _fail("paired benchmark manifest has an unsupported schema or version")
        result = cls(benchmark_id=data.get("benchmark_id"), cohort_id=data.get("cohort_id"), input_digest=data.get("input_digest"), gate_digest=data.get("gate_digest"), arms=data.get("arms"), evidence_state=data.get("evidence_state"))
        for field in ("content_id", "manifest_id"):
            if field in data and data[field] != result.content_id:
                _fail(f"{field} does not match canonical content")
        return result

    @classmethod
    def from_canonical_bytes(cls, value: bytes) -> "PairedBenchmarkManifest":
        if not isinstance(value, bytes):
            _fail("canonical manifest must be bytes")
        try:
            payload = json.loads(value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EfficiencyReceiptValidationError("canonical manifest bytes are invalid JSON") from exc
        result = cls.from_dict(payload)
        if value != result.canonical_bytes():
            _fail("manifest bytes are not canonical")
        return result


__all__ = (
    "BenchmarkArm", "CONTRACT_VERSION", "ContractValidationError", "EfficiencyReceipt",
    "EfficiencyReceiptValidationError", "Measurement", "PAIRED_BENCHMARK_MANIFEST_SCHEMA",
    "PairedBenchmarkManifest", "SCHEMA_VERSION", "TASK_EFFICIENCY_RECEIPT_SCHEMA",
    "TaskEfficiencyReceipt", "TelemetryFact", "TerminalOutcome", "TruthState",
    "canonical_json", "canonical_json_bytes", "content_identity",
)
