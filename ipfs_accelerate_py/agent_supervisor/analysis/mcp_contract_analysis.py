"""Deterministic MCP++ schema, invocation, and safety parity analysis.

Interface: ``McpContractAnalysis@1``

The analyzer consumes a reviewed expected operation contract, observed route
contracts, and (optionally) an exact :mod:`mcp_invocation_trace`.  It emits one
typed claim for every parity dimension owned by SCA-051.  Missing or partial
evidence is preserved as such; it is never coerced into a successful claim.

The accepted operation dictionaries are deliberately small and serializable.
An expected contract may contain ``input_schema``, ``output_schema``,
``result_envelope``, ``failure_states``, ``required_policies``, ``transports``,
``require_provenance`` and ``require_receipt``.  An observed contract contains
``routes`` plus ``discovery``.  Each route may carry those same schema and
envelope fields together with ``argument_map``, ``failure_mapping``, ordered
``events``, ``path_class``, ``transport``, and ``callable``.

Argument renames are authority-sensitive: an ``argument_map`` only describes
an observed translation.  A non-identity translation is accepted exclusively
when a matching :class:`ReviewedAlias` is supplied.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity
from .mcp_contract_catalog import McpClaimFamily
from .mcp_invocation_trace import (
    InvocationTerminalState,
    McpInvocationTrace,
)
from .symbolic_contract_graph import canonical_contract_graph_bytes


MCP_CONTRACT_ANALYSIS_INTERFACE: Final = "McpContractAnalysis@1"
MCP_CONTRACT_ANALYSIS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-analysis@1"
)
MCP_CONTRACT_CLAIM_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-parity-claim@1"
)
MCP_CONTRACT_COUNTEREXAMPLE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-counterexample@1"
)
MCP_REVIEWED_ALIAS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-reviewed-alias@1"
)
MCP_CONTRACT_ANALYSIS_VERSION: Final = "1"

DEFAULT_FAILURE_STATES: Final[tuple[str, ...]] = (
    "unsupported",
    "unavailable",
    "denied",
    "timed_out",
    "malformed",
    "partial",
)
SUPPORTED_JSON_SCHEMA_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "$defs",
        "$id",
        "$schema",
        "additionalProperties",
        "const",
        "default",
        "description",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "items",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "pattern",
        "properties",
        "required",
        "title",
        "type",
        "uniqueItems",
    }
)
PARITY_CLAIM_FAMILIES: Final[tuple[McpClaimFamily, ...]] = (
    McpClaimFamily.DESCRIPTOR_SCHEMA_MATCHES,
    McpClaimFamily.ARGUMENTS_PRESERVED,
    McpClaimFamily.RESULT_ENVELOPE_PRESERVED,
    McpClaimFamily.POLICY_BEFORE_EFFECT,
    McpClaimFamily.NO_COMPATIBILITY_BYPASS,
    McpClaimFamily.TRANSPORT_PARITY,
    McpClaimFamily.DISCOVERY_EXECUTION_PARITY,
    McpClaimFamily.FAILURE_PARITY,
)


class McpContractAnalysisError(ValueError):
    """Input or serialized parity evidence is malformed."""


class ParityState(str, Enum):
    """Closed claim states; only ``satisfied`` is an acceptance result."""

    SATISFIED = "satisfied"
    REFUTED = "refuted"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"
    NOT_MEASURED = "not_measured"
    PARTIAL = "partial"


class RoutePathClass(str, Enum):
    DIRECT = "direct"
    COMPATIBILITY = "compatibility"


class SchemaVariance(str, Enum):
    """Set-inclusion direction for a boundary schema."""

    INPUT = "input"
    OUTPUT = "output"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise McpContractAnalysisError(f"{name} must be a string")
    if value != value.strip() or "\x00" in value:
        raise McpContractAnalysisError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not value:
        raise McpContractAnalysisError(f"{name} is required")
    if len(value.encode("utf-8")) > 16_384:
        raise McpContractAnalysisError(f"{name} is oversized")
    return value


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 32:
        raise McpContractAnalysisError("contract evidence exceeds nesting bound")
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise McpContractAnalysisError(
            "floating values are not canonical contract evidence"
        )
    if isinstance(value, Mapping):
        if len(value) > 4_096 or not all(
            isinstance(key, str) for key in value
        ):
            raise McpContractAnalysisError(
                "contract objects require at most 4096 string keys"
            )
        return {
            key: _plain(value[key], depth=depth + 1)
            for key in sorted(value)
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if len(value) > 65_536:
            raise McpContractAnalysisError("contract sequence is oversized")
        return [_plain(item, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict(), depth=depth + 1)
    raise McpContractAnalysisError(
        f"unsupported contract value: {type(value).__name__}"
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise McpContractAnalysisError(f"{name} must be an object")
    return MappingProxyType(_plain(value))


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise McpContractAnalysisError(f"{name} must be a sequence")
    return tuple(sorted({_text(str(item), name) for item in value}))


def _canonical(value: Any) -> bytes:
    return canonical_contract_graph_bytes(_plain(value))


def _cid(value: Any) -> str:
    return content_identity(_plain(value))


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(str(raw))
    except (TypeError, ValueError) as exc:
        raise McpContractAnalysisError(
            f"unknown {name}: {value!r}"
        ) from exc


@dataclass(frozen=True, slots=True)
class ReviewedAlias:
    """An exact, reviewed source-to-target argument-name translation."""

    source_name: str
    target_name: str
    review_id: str
    source_ids: tuple[str, ...]
    alias_id: str = ""

    def __post_init__(self) -> None:
        for name in ("source_name", "target_name", "review_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "source_ids", _strings(self.source_ids, "source_ids")
        )
        if not self.source_ids:
            raise McpContractAnalysisError(
                "reviewed alias requires authority-bearing source_ids"
            )
        expected = _cid(self._identity_payload())
        if self.alias_id and self.alias_id != expected:
            raise McpContractAnalysisError("alias identity mismatch")
        object.__setattr__(self, "alias_id", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MCP_REVIEWED_ALIAS_SCHEMA,
            "source_name": self.source_name,
            "target_name": self.target_name,
            "review_id": self.review_id,
            "source_ids": list(self.source_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"alias_id": self.alias_id, **self._identity_payload()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReviewedAlias":
        if value.get("schema") not in (None, MCP_REVIEWED_ALIAS_SCHEMA):
            raise McpContractAnalysisError("unsupported alias schema")
        return cls(
            source_name=str(value.get("source_name") or value.get("source") or ""),
            target_name=str(value.get("target_name") or value.get("target") or ""),
            review_id=str(value.get("review_id") or ""),
            source_ids=tuple(value.get("source_ids") or ()),
            alias_id=str(value.get("alias_id") or ""),
        )


@dataclass(frozen=True, slots=True)
class ContractCounterexample:
    """A compact deterministic refutation or uncertainty witness."""

    reason_code: str
    boundary_id: str
    path: str
    expected: Any
    actual: Any
    source_ids: tuple[str, ...] = ()
    counterexample_id: str = ""

    def __post_init__(self) -> None:
        for name in ("reason_code", "boundary_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "path", _text(self.path, "path", required=False)
        )
        object.__setattr__(self, "expected", _plain(self.expected))
        object.__setattr__(self, "actual", _plain(self.actual))
        object.__setattr__(
            self, "source_ids", _strings(self.source_ids, "source_ids")
        )
        expected_id = _cid(self._identity_payload())
        if self.counterexample_id and self.counterexample_id != expected_id:
            raise McpContractAnalysisError("counterexample identity mismatch")
        object.__setattr__(self, "counterexample_id", expected_id)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MCP_CONTRACT_COUNTEREXAMPLE_SCHEMA,
            "reason_code": self.reason_code,
            "boundary_id": self.boundary_id,
            "path": self.path,
            "expected": self.expected,
            "actual": self.actual,
            "source_ids": list(self.source_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "counterexample_id": self.counterexample_id,
            **self._identity_payload(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContractCounterexample":
        if value.get("schema") not in (
            None,
            MCP_CONTRACT_COUNTEREXAMPLE_SCHEMA,
        ):
            raise McpContractAnalysisError("unsupported counterexample schema")
        return cls(
            reason_code=str(value.get("reason_code") or ""),
            boundary_id=str(value.get("boundary_id") or ""),
            path=str(value.get("path") or ""),
            expected=value.get("expected"),
            actual=value.get("actual"),
            source_ids=tuple(value.get("source_ids") or ()),
            counterexample_id=str(value.get("counterexample_id") or ""),
        )


@dataclass(frozen=True, slots=True)
class ContractParityClaim:
    """One reviewed parity claim with exact premises and witnesses."""

    family: McpClaimFamily
    state: ParityState
    operation_id: str
    premise_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    counterexamples: tuple[ContractCounterexample, ...] = ()
    claim_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "family",
            _enum(self.family, McpClaimFamily, "claim family"),
        )
        if self.family not in PARITY_CLAIM_FAMILIES:
            raise McpContractAnalysisError(
                f"{self.family.value} is not an SCA-051 parity family"
            )
        object.__setattr__(
            self, "state", _enum(self.state, ParityState, "parity state")
        )
        object.__setattr__(
            self, "operation_id", _text(self.operation_id, "operation_id")
        )
        object.__setattr__(
            self, "premise_ids", _strings(self.premise_ids, "premise_ids")
        )
        object.__setattr__(
            self, "reason_codes", _strings(self.reason_codes, "reason_codes")
        )
        if not self.reason_codes:
            raise McpContractAnalysisError("claim requires a reason_code")
        items = tuple(
            item
            if isinstance(item, ContractCounterexample)
            else ContractCounterexample.from_dict(item)
            for item in self.counterexamples
        )
        by_id = {item.counterexample_id: item for item in items}
        object.__setattr__(
            self,
            "counterexamples",
            tuple(by_id[key] for key in sorted(by_id)),
        )
        if (
            self.state is ParityState.SATISFIED
            and self.counterexamples
        ):
            raise McpContractAnalysisError(
                "satisfied claim cannot contain counterexamples"
            )
        if (
            self.state is ParityState.REFUTED
            and not self.counterexamples
        ):
            raise McpContractAnalysisError(
                "refuted claim requires a counterexample"
            )
        expected = _cid(self._identity_payload())
        if self.claim_id and self.claim_id != expected:
            raise McpContractAnalysisError("claim identity mismatch")
        object.__setattr__(self, "claim_id", expected)

    @property
    def passed(self) -> bool:
        return self.state is ParityState.SATISFIED

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MCP_CONTRACT_CLAIM_SCHEMA,
            "family": self.family.value,
            "state": self.state.value,
            "operation_id": self.operation_id,
            "premise_ids": list(self.premise_ids),
            "reason_codes": list(self.reason_codes),
            "counterexamples": [
                item.to_dict() for item in self.counterexamples
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {"claim_id": self.claim_id, **self._identity_payload()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContractParityClaim":
        if value.get("schema") not in (None, MCP_CONTRACT_CLAIM_SCHEMA):
            raise McpContractAnalysisError("unsupported parity-claim schema")
        return cls(
            family=value.get("family", ""),
            state=value.get("state", ""),
            operation_id=str(value.get("operation_id") or ""),
            premise_ids=tuple(value.get("premise_ids") or ()),
            reason_codes=tuple(value.get("reason_codes") or ()),
            counterexamples=tuple(
                ContractCounterexample.from_dict(item)
                for item in value.get("counterexamples") or ()
            ),
            claim_id=str(value.get("claim_id") or ""),
        )


@dataclass(frozen=True, slots=True)
class McpContractAnalysis:
    """Content-addressed parity report for exactly one operation."""

    operation_id: str
    expected_contract_id: str
    observed_contract_id: str
    claims: tuple[ContractParityClaim, ...]
    trace_id: str = ""
    complete: bool = True
    analysis_id: str = ""
    version: str = MCP_CONTRACT_ANALYSIS_VERSION

    def __post_init__(self) -> None:
        for name in (
            "operation_id",
            "expected_contract_id",
            "observed_contract_id",
            "version",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "trace_id", _text(self.trace_id, "trace_id", required=False)
        )
        claims = tuple(
            item
            if isinstance(item, ContractParityClaim)
            else ContractParityClaim.from_dict(item)
            for item in self.claims
        )
        by_family = {item.family: item for item in claims}
        if len(by_family) != len(claims):
            raise McpContractAnalysisError("duplicate parity claim family")
        missing = set(PARITY_CLAIM_FAMILIES) - set(by_family)
        if missing:
            raise McpContractAnalysisError(
                "analysis missing parity families: "
                + ", ".join(sorted(item.value for item in missing))
            )
        if any(item.operation_id != self.operation_id for item in claims):
            raise McpContractAnalysisError("claim operation_id mismatch")
        object.__setattr__(
            self,
            "claims",
            tuple(sorted(claims, key=lambda item: item.family.value)),
        )
        object.__setattr__(self, "complete", bool(self.complete))
        if self.complete and any(
            item.state
            in {
                ParityState.AMBIGUOUS,
                ParityState.NOT_MEASURED,
                ParityState.PARTIAL,
            }
            for item in claims
        ):
            raise McpContractAnalysisError(
                "complete report cannot contain incomplete claim states"
            )
        expected = _cid(self._identity_payload())
        if self.analysis_id and self.analysis_id != expected:
            raise McpContractAnalysisError("analysis identity mismatch")
        object.__setattr__(self, "analysis_id", expected)

    @property
    def passed(self) -> bool:
        return self.complete and all(item.passed for item in self.claims)

    @property
    def state(self) -> ParityState:
        states = {item.state for item in self.claims}
        for state in (
            ParityState.REFUTED,
            ParityState.PARTIAL,
            ParityState.AMBIGUOUS,
            ParityState.NOT_MEASURED,
            ParityState.UNSUPPORTED,
        ):
            if state in states:
                return state
        if not self.complete:
            return ParityState.PARTIAL
        return ParityState.SATISFIED

    def claim(self, family: McpClaimFamily | str) -> ContractParityClaim:
        selected = _enum(family, McpClaimFamily, "claim family")
        for item in self.claims:
            if item.family is selected:
                return item
        raise KeyError(selected.value)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MCP_CONTRACT_ANALYSIS_SCHEMA,
            "interface": MCP_CONTRACT_ANALYSIS_INTERFACE,
            "version": self.version,
            "operation_id": self.operation_id,
            "expected_contract_id": self.expected_contract_id,
            "observed_contract_id": self.observed_contract_id,
            "trace_id": self.trace_id,
            "claims": [item.to_dict() for item in self.claims],
            "complete": self.complete,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "passed": self.passed,
            "state": self.state.value,
            **self._identity_payload(),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            _plain(self.to_dict()),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "McpContractAnalysis":
        if value.get("schema") not in (None, MCP_CONTRACT_ANALYSIS_SCHEMA):
            raise McpContractAnalysisError("unsupported analysis schema")
        if value.get("interface") not in (
            None,
            MCP_CONTRACT_ANALYSIS_INTERFACE,
        ):
            raise McpContractAnalysisError("unsupported analysis interface")
        result = cls(
            operation_id=str(value.get("operation_id") or ""),
            expected_contract_id=str(value.get("expected_contract_id") or ""),
            observed_contract_id=str(value.get("observed_contract_id") or ""),
            trace_id=str(value.get("trace_id") or ""),
            claims=tuple(
                ContractParityClaim.from_dict(item)
                for item in value.get("claims") or ()
            ),
            complete=value.get("complete", False),
            analysis_id=str(value.get("analysis_id") or ""),
            version=str(
                value.get("version") or MCP_CONTRACT_ANALYSIS_VERSION
            ),
        )
        if "passed" in value and bool(value["passed"]) != result.passed:
            raise McpContractAnalysisError("analysis passed claim mismatch")
        if "state" in value and str(value["state"]) != result.state.value:
            raise McpContractAnalysisError("analysis state claim mismatch")
        return result

    @classmethod
    def from_json(cls, value: str | bytes) -> "McpContractAnalysis":
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise McpContractAnalysisError(
                "analysis JSON is malformed"
            ) from exc
        if not isinstance(payload, Mapping):
            raise McpContractAnalysisError(
                "analysis JSON must contain an object"
            )
        return cls.from_dict(payload)


def _schema(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise McpContractAnalysisError("JSON schema must be an object")
    return _mapping(value, "JSON schema")


def _schema_types(schema: Mapping[str, Any]) -> frozenset[str] | None:
    raw = schema.get("type")
    if raw is None:
        return None
    if isinstance(raw, str):
        values = (raw,)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = tuple(raw)
    else:
        raise McpContractAnalysisError("schema type must be a string or list")
    allowed = {
        "array",
        "boolean",
        "integer",
        "null",
        "number",
        "object",
        "string",
    }
    result = frozenset(str(item) for item in values)
    if not result or not result <= allowed:
        raise McpContractAnalysisError("schema contains an unknown JSON type")
    return result


def _accepted_type_subset(
    narrower: frozenset[str] | None,
    wider: frozenset[str] | None,
) -> bool | None:
    if wider is None:
        return True
    if narrower is None:
        return None
    # JSON Schema's integer instances are also number instances.
    expanded_wider = set(wider)
    if "number" in expanded_wider:
        expanded_wider.add("integer")
    return set(narrower) <= expanded_wider


def _unsupported_schema_keywords(
    schema: Mapping[str, Any], path: str = "$"
) -> tuple[str, ...]:
    found: set[str] = set()
    for key, value in schema.items():
        if key not in SUPPORTED_JSON_SCHEMA_KEYWORDS:
            found.add(f"{path}/{key}")
        if key == "properties" and isinstance(value, Mapping):
            for name, child in value.items():
                if isinstance(child, Mapping):
                    found.update(
                        _unsupported_schema_keywords(
                            child, f"{path}/properties/{name}"
                        )
                    )
        elif key in {"items", "additionalProperties"} and isinstance(
            value, Mapping
        ):
            found.update(
                _unsupported_schema_keywords(value, f"{path}/{key}")
            )
    return tuple(sorted(found))


def _counterexample(
    reason: str,
    boundary: str,
    path: str,
    expected: Any,
    actual: Any,
    source_ids: Sequence[str] = (),
) -> ContractCounterexample:
    return ContractCounterexample(
        reason_code=reason,
        boundary_id=boundary,
        path=path,
        expected=expected,
        actual=actual,
        source_ids=tuple(source_ids),
    )


def _schema_inclusion(
    narrower: Mapping[str, Any],
    wider: Mapping[str, Any],
    *,
    boundary: str,
    path: str,
    reason_prefix: str,
    source_ids: Sequence[str],
) -> tuple[ContractCounterexample, ...]:
    """Prove a conservative subset relation for the supported schema fragment.

    ``narrower`` instances must all be accepted by ``wider``.  Unsupported
    keywords are reported separately by the caller and never silently ignored.
    """

    issues: list[ContractCounterexample] = []
    narrow_types = _schema_types(narrower)
    wide_types = _schema_types(wider)
    included = _accepted_type_subset(narrow_types, wide_types)
    if included is not True:
        issues.append(
            _counterexample(
                f"{reason_prefix}_type_variance",
                boundary,
                f"{path}/type",
                sorted(narrow_types) if narrow_types else None,
                sorted(wide_types) if wide_types else None,
                source_ids,
            )
        )
        return tuple(issues)

    if "const" in wider and narrower.get("const", object()) != wider["const"]:
        issues.append(
            _counterexample(
                f"{reason_prefix}_const_variance",
                boundary,
                f"{path}/const",
                narrower.get("const"),
                wider["const"],
                source_ids,
            )
        )
    if "enum" in wider:
        narrow_values = (
            {json.dumps(narrower["const"], sort_keys=True)}
            if "const" in narrower
            else {
                json.dumps(item, sort_keys=True)
                for item in narrower.get("enum", ())
            }
        )
        wide_values = {
            json.dumps(item, sort_keys=True) for item in wider["enum"]
        }
        if not narrow_values or not narrow_values <= wide_values:
            issues.append(
                _counterexample(
                    f"{reason_prefix}_enum_variance",
                    boundary,
                    f"{path}/enum",
                    narrower.get("enum", narrower.get("const")),
                    wider["enum"],
                    source_ids,
                )
            )

    # A wider schema cannot impose a stronger lower bound or a weaker upper
    # bound than the narrower set being checked.
    for key in ("minimum", "exclusiveMinimum", "minLength", "minItems"):
        if key in wider and (
            key not in narrower or narrower[key] < wider[key]
        ):
            issues.append(
                _counterexample(
                    f"{reason_prefix}_{key}_variance",
                    boundary,
                    f"{path}/{key}",
                    narrower.get(key),
                    wider[key],
                    source_ids,
                )
            )
    for key in ("maximum", "exclusiveMaximum", "maxLength", "maxItems"):
        if key in wider and (
            key not in narrower or narrower[key] > wider[key]
        ):
            issues.append(
                _counterexample(
                    f"{reason_prefix}_{key}_variance",
                    boundary,
                    f"{path}/{key}",
                    narrower.get(key),
                    wider[key],
                    source_ids,
                )
            )
    for key in ("pattern", "format", "uniqueItems"):
        if key in wider and narrower.get(key) != wider[key]:
            issues.append(
                _counterexample(
                    f"{reason_prefix}_{key}_variance",
                    boundary,
                    f"{path}/{key}",
                    narrower.get(key),
                    wider[key],
                    source_ids,
                )
            )

    narrow_props = narrower.get("properties", {})
    wide_props = wider.get("properties", {})
    if not isinstance(narrow_props, Mapping) or not isinstance(
        wide_props, Mapping
    ):
        raise McpContractAnalysisError("schema properties must be an object")
    narrow_required = set(_strings(narrower.get("required"), "schema required"))
    wide_required = set(_strings(wider.get("required"), "schema required"))
    # Every wider-required property must necessarily exist in narrower.
    for name in sorted(wide_required - narrow_required):
        issues.append(
            _counterexample(
                f"{reason_prefix}_required_variance",
                boundary,
                f"{path}/required/{name}",
                sorted(narrow_required),
                sorted(wide_required),
                source_ids,
            )
        )
    for name in sorted(set(narrow_props) & set(wide_props)):
        left = narrow_props[name]
        right = wide_props[name]
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise McpContractAnalysisError(
                "property schemas must be objects"
            )
        issues.extend(
            _schema_inclusion(
                left,
                right,
                boundary=boundary,
                path=f"{path}/properties/{name}",
                reason_prefix=reason_prefix,
                source_ids=source_ids,
            )
        )
    for name in sorted(set(narrow_props) - set(wide_props)):
        if wider.get("additionalProperties") is False:
            issues.append(
                _counterexample(
                    f"{reason_prefix}_additional_property_variance",
                    boundary,
                    f"{path}/properties/{name}",
                    narrow_props[name],
                    "forbidden",
                    source_ids,
                )
            )
    if (
        narrower.get("additionalProperties") is not False
        and wider.get("additionalProperties") is False
    ):
        issues.append(
            _counterexample(
                f"{reason_prefix}_additional_properties_variance",
                boundary,
                f"{path}/additionalProperties",
                narrower.get("additionalProperties", True),
                False,
                source_ids,
            )
        )
    if isinstance(narrower.get("items"), Mapping) and isinstance(
        wider.get("items"), Mapping
    ):
        issues.extend(
            _schema_inclusion(
                narrower["items"],
                wider["items"],
                boundary=boundary,
                path=f"{path}/items",
                reason_prefix=reason_prefix,
                source_ids=source_ids,
            )
        )
    elif "items" in wider and "items" not in narrower:
        issues.append(
            _counterexample(
                f"{reason_prefix}_items_variance",
                boundary,
                f"{path}/items",
                None,
                wider["items"],
                source_ids,
            )
        )
    return tuple(issues)


@dataclass(slots=True)
class _ClaimBuilder:
    family: McpClaimFamily
    operation_id: str
    premise_ids: set[str] = field(default_factory=set)
    issues: list[ContractCounterexample] = field(default_factory=list)
    unknown_reasons: set[str] = field(default_factory=set)
    unsupported_reasons: set[str] = field(default_factory=set)
    not_measured_reasons: set[str] = field(default_factory=set)
    partial_reasons: set[str] = field(default_factory=set)

    def finish(self) -> ContractParityClaim:
        reasons = {
            item.reason_code for item in self.issues
        } | {
            *self.unknown_reasons,
            *self.unsupported_reasons,
            *self.not_measured_reasons,
            *self.partial_reasons,
        }
        if self.issues:
            state = ParityState.REFUTED
        elif self.partial_reasons:
            state = ParityState.PARTIAL
        elif self.unknown_reasons:
            state = ParityState.AMBIGUOUS
        elif self.not_measured_reasons:
            state = ParityState.NOT_MEASURED
        elif self.unsupported_reasons:
            state = ParityState.UNSUPPORTED
        else:
            state = ParityState.SATISFIED
            reasons = {"parity_satisfied"}
        return ContractParityClaim(
            family=self.family,
            state=state,
            operation_id=self.operation_id,
            premise_ids=tuple(self.premise_ids),
            reason_codes=tuple(reasons),
            counterexamples=tuple(self.issues),
        )


def _operation_id(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> str:
    left = str(expected.get("operation_id") or expected.get("tool_name") or "")
    right = str(observed.get("operation_id") or observed.get("tool_name") or "")
    if left and right and left != right:
        raise McpContractAnalysisError("expected/observed operation_id mismatch")
    return _text(left or right, "operation_id")


_UNORDERED_SEQUENCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "enum",
        "failure_states",
        "fields",
        "required",
        "required_fields",
        "required_policies",
        "result_envelope",
        "source_ids",
        "tools",
        "transports",
        "type",
    }
)


def _normalized_contract_value(value: Any, *, field_name: str = "") -> Any:
    """Canonicalize only fields whose contract semantics are set-like."""

    if isinstance(value, Mapping):
        return {
            key: _normalized_contract_value(value[key], field_name=key)
            for key in sorted(value)
            if not key.startswith("_")
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        items = [
            _normalized_contract_value(item)
            for item in value
        ]
        if field_name in _UNORDERED_SEQUENCE_FIELDS:
            unique = {_canonical(item): item for item in items}
            return [unique[key] for key in sorted(unique)]
        return items
    return _plain(value)


def _identity_contract(
    contract: Mapping[str, Any],
    *,
    routes: Sequence[Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    normalized = _normalized_contract_value(contract)
    if routes is not None:
        normalized["routes"] = [
            _normalized_contract_value(route)
            for route in sorted(routes, key=lambda item: item["route_id"])
        ]
    return normalized


def _routes(observed: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = observed.get("routes")
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise McpContractAnalysisError("observed routes must be a sequence")
    result: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, route in enumerate(raw):
        item = _mapping(route, "route")
        route_id = str(item.get("route_id") or item.get("id") or "")
        route_id = _text(route_id, "route_id")
        if route_id in seen:
            raise McpContractAnalysisError(f"duplicate route_id: {route_id}")
        seen.add(route_id)
        normalized = dict(item)
        normalized["route_id"] = route_id
        normalized["_index"] = index
        result.append(MappingProxyType(normalized))
    return tuple(sorted(result, key=lambda item: item["route_id"]))


def _route_sources(route: Mapping[str, Any]) -> tuple[str, ...]:
    return _strings(route.get("source_ids"), "route source_ids")


def _route_path_class(route: Mapping[str, Any]) -> RoutePathClass:
    if route.get("compatibility") is True:
        return RoutePathClass.COMPATIBILITY
    return _enum(
        route.get("path_class", RoutePathClass.DIRECT.value),
        RoutePathClass,
        "route path_class",
    )


def _schema_claim(
    builder: _ClaimBuilder,
    expected: Mapping[str, Any],
    routes: Sequence[Mapping[str, Any]],
    aliases: Sequence[ReviewedAlias],
) -> None:
    expected_input = _schema(expected.get("input_schema"))
    expected_output = _schema(expected.get("output_schema"))
    if expected_input is None or expected_output is None:
        builder.unknown_reasons.add("expected_schema_missing")
        return
    for unsupported in (
        *_unsupported_schema_keywords(expected_input),
        *_unsupported_schema_keywords(expected_output),
    ):
        builder.unsupported_reasons.add(
            f"unsupported_schema_keyword:{unsupported}"
        )
    if not routes:
        builder.unknown_reasons.add("route_evidence_missing")
        return
    for route in routes:
        route_id = route["route_id"]
        sources = _route_sources(route)
        builder.premise_ids.update(sources)
        actual_input = _schema(route.get("input_schema"))
        actual_output = _schema(route.get("output_schema"))
        if actual_input is None or actual_output is None:
            builder.partial_reasons.add("route_schema_missing")
            continue
        actual_input = _translate_reviewed_input_schema(
            actual_input, route, aliases
        )
        for unsupported in (
            *_unsupported_schema_keywords(actual_input),
            *_unsupported_schema_keywords(actual_output),
        ):
            builder.unsupported_reasons.add(
                f"unsupported_schema_keyword:{unsupported}"
            )
        # Expected callers must remain accepted by the implementation.
        builder.issues.extend(
            _schema_inclusion(
                expected_input,
                actual_input,
                boundary=route_id,
                path="$/input",
                reason_prefix="input_schema",
                source_ids=sources,
            )
        )
        # Every actual result must be accepted by the descriptor.
        builder.issues.extend(
            _schema_inclusion(
                actual_output,
                expected_output,
                boundary=route_id,
                path="$/output",
                reason_prefix="output_schema",
                source_ids=sources,
            )
        )


def _translate_reviewed_input_schema(
    schema: Mapping[str, Any],
    route: Mapping[str, Any],
    aliases: Sequence[ReviewedAlias],
) -> Mapping[str, Any]:
    """Project reviewed handler names back to descriptor names."""

    raw_map = route.get("argument_map", {})
    if not isinstance(raw_map, Mapping):
        raise McpContractAnalysisError("argument_map must be an object")
    approved = {
        (item.source_name, item.target_name) for item in aliases
    }
    translations = {
        str(target): str(source)
        for source, target in raw_map.items()
        if str(source) != str(target)
        and (str(source), str(target)) in approved
    }
    if not translations:
        return schema
    result = _plain(schema)
    properties = result.get("properties")
    if isinstance(properties, Mapping):
        renamed: dict[str, Any] = {}
        for name, value in properties.items():
            projected = translations.get(name, name)
            if projected in renamed:
                raise McpContractAnalysisError(
                    "reviewed alias creates a schema-property collision"
                )
            renamed[projected] = value
        result["properties"] = renamed
    required = result.get("required")
    if isinstance(required, Sequence) and not isinstance(
        required, (str, bytes)
    ):
        result["required"] = [
            translations.get(str(name), str(name)) for name in required
        ]
    return _mapping(result, "translated input schema")


def _argument_claim(
    builder: _ClaimBuilder,
    expected: Mapping[str, Any],
    routes: Sequence[Mapping[str, Any]],
    aliases: Sequence[ReviewedAlias],
) -> None:
    expected_schema = _schema(expected.get("input_schema"))
    if expected_schema is None:
        builder.unknown_reasons.add("expected_input_schema_missing")
        return
    expected_props = expected_schema.get("properties", {})
    if not isinstance(expected_props, Mapping):
        raise McpContractAnalysisError("input schema properties must be object")
    expected_required = set(
        _strings(expected_schema.get("required"), "schema required")
    )
    reviewed = {(item.source_name, item.target_name): item for item in aliases}
    if not routes:
        builder.unknown_reasons.add("route_evidence_missing")
        return
    for route in routes:
        route_id = route["route_id"]
        sources = _route_sources(route)
        actual_schema = _schema(route.get("input_schema"))
        if actual_schema is None:
            builder.partial_reasons.add("route_input_schema_missing")
            continue
        actual_props = actual_schema.get("properties", {})
        if not isinstance(actual_props, Mapping):
            raise McpContractAnalysisError(
                "route input schema properties must be object"
            )
        actual_required = set(
            _strings(actual_schema.get("required"), "schema required")
        )
        raw_map = route.get("argument_map", {})
        if not isinstance(raw_map, Mapping):
            raise McpContractAnalysisError("argument_map must be an object")
        mapping = {
            _text(str(key), "argument_map source"): _text(
                str(value), "argument_map target"
            )
            for key, value in raw_map.items()
        }
        targets: list[str] = []
        for source_name in sorted(expected_props):
            target_name = mapping.get(source_name, source_name)
            targets.append(target_name)
            path = f"$/arguments/{source_name}"
            expected_property = expected_props[source_name]
            actual_property = actual_props.get(target_name)
            if target_name != source_name:
                alias = reviewed.get((source_name, target_name))
                if alias is None:
                    builder.issues.append(
                        _counterexample(
                            "argument_rename_unreviewed",
                            route_id,
                            path,
                            source_name,
                            target_name,
                            sources,
                        )
                    )
                else:
                    builder.premise_ids.add(alias.alias_id)
                    builder.premise_ids.update(alias.source_ids)
            if actual_property is None:
                builder.issues.append(
                    _counterexample(
                        "argument_dropped",
                        route_id,
                        path,
                        expected_property,
                        None,
                        sources,
                    )
                )
                continue
            if not isinstance(expected_property, Mapping) or not isinstance(
                actual_property, Mapping
            ):
                raise McpContractAnalysisError(
                    "argument property schema must be an object"
                )
            expected_default = (
                ("present", expected_property["default"])
                if "default" in expected_property
                else ("absent", None)
            )
            actual_default = (
                ("present", actual_property["default"])
                if "default" in actual_property
                else ("absent", None)
            )
            if expected_default != actual_default:
                builder.issues.append(
                    _counterexample(
                        "argument_default_changed",
                        route_id,
                        f"{path}/default",
                        list(expected_default),
                        list(actual_default),
                        sources,
                    )
                )
            if (source_name in expected_required) != (
                target_name in actual_required
            ):
                builder.issues.append(
                    _counterexample(
                        "argument_requiredness_changed",
                        route_id,
                        f"{path}/required",
                        source_name in expected_required,
                        target_name in actual_required,
                        sources,
                    )
                )
            expected_types = _schema_types(expected_property)
            actual_types = _schema_types(actual_property)
            if expected_types != actual_types:
                builder.issues.append(
                    _counterexample(
                        (
                            "argument_type_lost"
                            if expected_types is not None
                            and actual_types is None
                            else "argument_type_changed"
                        ),
                        route_id,
                        f"{path}/type",
                        sorted(expected_types) if expected_types else None,
                        sorted(actual_types) if actual_types else None,
                        sources,
                    )
                )
        duplicates = sorted(
            name for name, count in Counter(targets).items() if count > 1
        )
        for target in duplicates:
            builder.issues.append(
                _counterexample(
                    "argument_mapping_collision",
                    route_id,
                    "$/argument_map",
                    "one-to-one",
                    target,
                    sources,
                )
            )
        unknown_sources = sorted(set(mapping) - set(expected_props))
        for name in unknown_sources:
            builder.issues.append(
                _counterexample(
                    "argument_mapping_unknown_source",
                    route_id,
                    f"$/argument_map/{name}",
                    sorted(expected_props),
                    name,
                    sources,
                )
            )


def _envelope_fields(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        raw = value.get(
            "required_fields",
            value.get("required", value.get("fields")),
        )
        if raw is None:
            raw = tuple(value.keys())
    else:
        raw = value
    return _strings(raw, "result envelope fields")


def _result_claim(
    builder: _ClaimBuilder,
    expected: Mapping[str, Any],
    routes: Sequence[Mapping[str, Any]],
) -> None:
    required = _envelope_fields(expected.get("result_envelope"))
    expected_variants = _result_variants(expected.get("result_envelopes"))
    require_provenance = bool(expected.get("require_provenance", False))
    require_receipt = bool(expected.get("require_receipt", False))
    if required is None and expected_variants is None:
        builder.unknown_reasons.add("expected_result_envelope_missing")
        return
    if not routes:
        builder.unknown_reasons.add("route_evidence_missing")
        return
    for route in routes:
        route_id = route["route_id"]
        sources = _route_sources(route)
        builder.premise_ids.update(sources)
        actual = _envelope_fields(route.get("result_envelope"))
        if required is not None and actual is None:
            builder.partial_reasons.add("route_result_envelope_missing")
        elif required is not None:
            for name in sorted(set(required) - set(actual or ())):
                builder.issues.append(
                    _counterexample(
                        "result_envelope_field_lost",
                        route_id,
                        f"$/result_envelope/{name}",
                        True,
                        False,
                        sources,
                    )
                )
        variant_issues = _result_variant_issues(expected, route)
        if variant_issues is None:
            builder.partial_reasons.add("route_result_variants_missing")
        else:
            builder.issues.extend(variant_issues)
        for flag, reason in (
            ("provenance", "result_provenance_lost"),
            ("receipt", "result_receipt_lost"),
        ):
            required_flag = (
                require_provenance if flag == "provenance" else require_receipt
            )
            actual_flag = route.get(
                flag,
                route.get(f"preserves_{flag}"),
            )
            if required_flag and actual_flag is not True:
                builder.issues.append(
                    _counterexample(
                        reason,
                        route_id,
                        f"$/result/{flag}",
                        True,
                        actual_flag,
                        sources,
                    )
                )


def _result_variants(value: Any) -> Mapping[str, tuple[str, ...]] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise McpContractAnalysisError("result_envelopes must be an object")
    result: dict[str, tuple[str, ...]] = {}
    for name, envelope in value.items():
        fields = _envelope_fields(envelope)
        if fields is None:
            raise McpContractAnalysisError(
                "result envelope variant requires fields"
            )
        result[_text(str(name), "result variant")] = fields
    return MappingProxyType(result)


def _result_variant_issues(
    expected: Mapping[str, Any],
    route: Mapping[str, Any],
) -> tuple[ContractCounterexample, ...] | None:
    expected_variants = _result_variants(expected.get("result_envelopes"))
    if expected_variants is None:
        return ()
    actual_variants = _result_variants(route.get("result_envelopes"))
    if actual_variants is None:
        return None
    issues: list[ContractCounterexample] = []
    sources = _route_sources(route)
    route_id = route["route_id"]
    for variant, fields in expected_variants.items():
        if variant not in actual_variants:
            issues.append(
                _counterexample(
                    "result_envelope_variant_lost",
                    route_id,
                    f"$/result_envelopes/{variant}",
                    list(fields),
                    None,
                    sources,
                )
            )
            continue
        for name in sorted(set(fields) - set(actual_variants[variant])):
            issues.append(
                _counterexample(
                    "result_envelope_variant_field_lost",
                    route_id,
                    f"$/result_envelopes/{variant}/{name}",
                    True,
                    False,
                    sources,
                )
            )
    raw_mapping = route.get("envelope_mapping", {})
    if not isinstance(raw_mapping, Mapping):
        raise McpContractAnalysisError("envelope_mapping must be an object")
    mapping = {
        variant: str(raw_mapping.get(variant, variant))
        for variant in expected_variants
    }
    targets = Counter(mapping.values())
    for variant, target in sorted(mapping.items()):
        if target != variant:
            issues.append(
                _counterexample(
                    "result_envelope_remapped",
                    route_id,
                    f"$/envelope_mapping/{variant}",
                    variant,
                    target,
                    sources,
                )
            )
        if targets[target] > 1:
            issues.append(
                _counterexample(
                    "result_envelopes_collapsed",
                    route_id,
                    f"$/envelope_mapping/{variant}",
                    variant,
                    target,
                    sources,
                )
            )
    return tuple(issues)


def _events(route: Mapping[str, Any]) -> tuple[tuple[str, str], ...] | None:
    raw = route.get("events")
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise McpContractAnalysisError("route events must be a sequence")
    result: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, str):
            if ":" not in item:
                raise McpContractAnalysisError(
                    "string event must use kind:name"
                )
            kind, name = item.split(":", 1)
        elif isinstance(item, Mapping):
            kind = str(item.get("kind") or "")
            name = str(item.get("name") or item.get("id") or "")
        else:
            raise McpContractAnalysisError(
                "event must be a kind:name string or object"
            )
        if kind == "authorization":
            kind = "policy"
        if kind == "mutation":
            kind = "effect"
        if kind not in {"policy", "effect"}:
            raise McpContractAnalysisError(f"unknown event kind: {kind!r}")
        result.append((_text(kind, "event kind"), _text(name, "event name")))
    return tuple(result)


def _policy_issues(
    expected: Mapping[str, Any],
    route: Mapping[str, Any],
) -> tuple[ContractCounterexample, ...] | None:
    policies = _strings(expected.get("required_policies"), "required policies")
    events = _events(route)
    route_id = route["route_id"]
    sources = _route_sources(route)
    mutation_capable = bool(
        route.get("mutation_capable")
        or route.get("effects")
        or (events and any(kind == "effect" for kind, _ in events))
    )
    if not mutation_capable:
        return ()
    if not policies:
        return (
            _counterexample(
                "mutation_policy_contract_missing",
                route_id,
                "$/required_policies",
                "at least one policy",
                [],
                sources,
            ),
        )
    if events is None:
        return None
    effect_indexes = [
        index for index, (kind, _) in enumerate(events) if kind == "effect"
    ]
    if not effect_indexes:
        return None
    issues: list[ContractCounterexample] = []
    for effect_index in effect_indexes:
        effect_name = events[effect_index][1]
        before = {
            name
            for kind, name in events[:effect_index]
            if kind == "policy"
        }
        after = {
            name
            for kind, name in events[effect_index + 1 :]
            if kind == "policy"
        }
        for policy in policies:
            if policy in before:
                continue
            reason = (
                "policy_after_effect"
                if policy in after
                else "required_policy_missing"
            )
            issues.append(
                _counterexample(
                    reason,
                    route_id,
                    f"$/events/{effect_index}/{effect_name}",
                    f"policy:{policy} before effect",
                    [f"{kind}:{name}" for kind, name in events],
                    sources,
                )
            )
    return tuple(issues)


def _policy_claim(
    builder: _ClaimBuilder,
    expected: Mapping[str, Any],
    routes: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[ContractCounterexample, ...] | None]:
    result: dict[str, tuple[ContractCounterexample, ...] | None] = {}
    if not routes:
        builder.unknown_reasons.add("route_evidence_missing")
        return result
    for route in routes:
        issues = _policy_issues(expected, route)
        result[route["route_id"]] = issues
        builder.premise_ids.update(_route_sources(route))
        if issues is None:
            builder.partial_reasons.add("policy_order_evidence_missing")
        else:
            builder.issues.extend(issues)
    return result


def _failure_issues(
    expected: Mapping[str, Any],
    route: Mapping[str, Any],
) -> tuple[ContractCounterexample, ...] | None:
    expected_states = _strings(
        expected.get("failure_states"), "failure states"
    )
    if not expected_states:
        return None
    raw_actual = route.get("failure_states")
    if raw_actual is None:
        return None
    actual_states = _strings(raw_actual, "failure states")
    route_id = route["route_id"]
    sources = _route_sources(route)
    issues: list[ContractCounterexample] = []
    for state in sorted(set(expected_states) - set(actual_states)):
        issues.append(
            _counterexample(
                "failure_state_lost",
                route_id,
                f"$/failure_states/{state}",
                state,
                None,
                sources,
            )
        )
    raw_mapping = route.get("failure_mapping", {})
    if not isinstance(raw_mapping, Mapping):
        raise McpContractAnalysisError("failure_mapping must be an object")
    mapping = {
        str(state): str(raw_mapping.get(state, state))
        for state in expected_states
    }
    targets = Counter(mapping.values())
    for state in expected_states:
        target = mapping[state]
        if target != state:
            issues.append(
                _counterexample(
                    "failure_state_remapped",
                    route_id,
                    f"$/failure_mapping/{state}",
                    state,
                    target,
                    sources,
                )
            )
        if targets[target] > 1:
            issues.append(
                _counterexample(
                    "failure_states_collapsed",
                    route_id,
                    f"$/failure_mapping/{state}",
                    state,
                    target,
                    sources,
                )
            )
    return tuple(issues)


def _failure_claim(
    builder: _ClaimBuilder,
    expected: Mapping[str, Any],
    routes: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[ContractCounterexample, ...] | None]:
    result: dict[str, tuple[ContractCounterexample, ...] | None] = {}
    if not expected.get("failure_states"):
        builder.unknown_reasons.add("expected_failure_states_missing")
        return result
    if not routes:
        builder.unknown_reasons.add("route_evidence_missing")
        return result
    for route in routes:
        issues = _failure_issues(expected, route)
        result[route["route_id"]] = issues
        builder.premise_ids.update(_route_sources(route))
        if issues is None:
            builder.partial_reasons.add("route_failure_evidence_missing")
        else:
            builder.issues.extend(issues)
    return result


def _call_reachability(
    observed: Mapping[str, Any],
    routes: Sequence[Mapping[str, Any]],
    trace: McpInvocationTrace | None,
    builder: _ClaimBuilder,
) -> bool | None:
    if trace is not None:
        builder.premise_ids.add(trace.trace_id)
        if trace.terminal_state is InvocationTerminalState.REACHABLE:
            return True
        if trace.terminal_state is InvocationTerminalState.REFUTED:
            return False
        if trace.terminal_state is InvocationTerminalState.UNSUPPORTED:
            builder.unsupported_reasons.add("invocation_unsupported")
        elif trace.terminal_state is InvocationTerminalState.NOT_MEASURED:
            builder.not_measured_reasons.add("invocation_not_measured")
        else:
            builder.unknown_reasons.add("invocation_ambiguous")
        return None
    values = {
        bool(route["callable"])
        for route in routes
        if "callable" in route and route["callable"] is not None
    }
    if not values:
        value = observed.get("callable")
        return bool(value) if value is not None else None
    return any(values)


def _discovery_claim(
    builder: _ClaimBuilder,
    operation_id: str,
    observed: Mapping[str, Any],
    routes: Sequence[Mapping[str, Any]],
    trace: McpInvocationTrace | None,
) -> None:
    discovery = observed.get("discovery")
    listed: bool | None = None
    if isinstance(discovery, Mapping):
        if "listed" in discovery:
            listed = bool(discovery["listed"])
        elif "tools" in discovery:
            tools = _strings(discovery["tools"], "discovered tools")
            listed = operation_id in tools
    elif discovery is not None:
        raise McpContractAnalysisError("discovery must be an object")
    callable_value = _call_reachability(
        observed, routes, trace, builder
    )
    if listed is None:
        builder.unknown_reasons.add("tools_list_evidence_missing")
    if callable_value is None:
        builder.unknown_reasons.add("tools_call_evidence_missing")
    if listed is not None and callable_value is not None and (
        listed != callable_value
    ):
        builder.issues.append(
            _counterexample(
                "tools_list_call_drift",
                "discovery",
                "$/discovery",
                {"listed": listed},
                {"callable": callable_value},
                (trace.trace_id,) if trace else (),
            )
        )
    discovery_transports = (
        discovery.get("transports", {})
        if isinstance(discovery, Mapping)
        else {}
    )
    if not isinstance(discovery_transports, Mapping):
        raise McpContractAnalysisError(
            "discovery transports must be an object"
        )
    for route in routes:
        transport = str(route.get("transport") or "")
        route_listed = route.get("discoverable", route.get("listed"))
        if route_listed is None and transport in discovery_transports:
            transport_evidence = discovery_transports[transport]
            if isinstance(transport_evidence, bool):
                route_listed = transport_evidence
            elif isinstance(transport_evidence, Sequence) and not isinstance(
                transport_evidence, (str, bytes)
            ):
                route_listed = operation_id in transport_evidence
            else:
                raise McpContractAnalysisError(
                    "transport discovery must be boolean or a tool list"
                )
        route_callable = route.get("callable")
        if route_listed is not None and route_callable is not None and (
            bool(route_listed) != bool(route_callable)
        ):
            builder.issues.append(
                _counterexample(
                    "tools_list_call_route_drift",
                    route["route_id"],
                    "$/discovery/transports",
                    {"listed": bool(route_listed)},
                    {"callable": bool(route_callable)},
                    _route_sources(route),
                )
            )


def _route_compliance_issues(
    expected: Mapping[str, Any],
    route: Mapping[str, Any],
    policy: tuple[ContractCounterexample, ...] | None,
    failure: tuple[ContractCounterexample, ...] | None,
    additional: Sequence[ContractCounterexample] = (),
) -> tuple[ContractCounterexample, ...] | None:
    """Return route-local bypass witnesses, or ``None`` for incomplete data."""

    issues: list[ContractCounterexample] = list(additional)
    if policy is None or failure is None:
        return None
    issues.extend(policy)
    issues.extend(failure)
    callable_value = route.get("callable")
    if callable_value is None:
        return None
    if callable_value is not True:
        issues.append(
            _counterexample(
                "route_invocation_unreachable",
                route["route_id"],
                "$/callable",
                True,
                callable_value,
                _route_sources(route),
            )
        )
    required_envelope = _envelope_fields(expected.get("result_envelope"))
    actual_envelope = _envelope_fields(route.get("result_envelope"))
    if required_envelope is None or actual_envelope is None:
        return None
    sources = _route_sources(route)
    for name in sorted(set(required_envelope) - set(actual_envelope)):
        issues.append(
            _counterexample(
                "route_envelope_bypass",
                route["route_id"],
                f"$/result_envelope/{name}",
                True,
                False,
                sources,
            )
        )
    for flag in ("provenance", "receipt"):
        required_flag = bool(expected.get(f"require_{flag}", False))
        actual_flag = route.get(flag, route.get(f"preserves_{flag}"))
        if required_flag and actual_flag is not True:
            issues.append(
                _counterexample(
                    f"route_{flag}_bypass",
                    route["route_id"],
                    f"$/result/{flag}",
                    True,
                    actual_flag,
                    sources,
                )
            )
    if route.get("input_schema") is None or route.get("output_schema") is None:
        return None
    return tuple(issues)


def _compatibility_claim(
    builder: _ClaimBuilder,
    expected: Mapping[str, Any],
    routes: Sequence[Mapping[str, Any]],
    compliance: Mapping[
        str, tuple[ContractCounterexample, ...] | None
    ],
    trace: McpInvocationTrace | None,
) -> None:
    compatibility = [
        route
        for route in routes
        if _route_path_class(route) is RoutePathClass.COMPATIBILITY
    ]
    if trace is not None and trace.compatibility_paths:
        builder.premise_ids.add(trace.trace_id)
        if not compatibility:
            builder.partial_reasons.add(
                "compatibility_trace_route_evidence_missing"
            )
            return
    if not compatibility:
        return
    for route in compatibility:
        route_id = route["route_id"]
        issues = compliance[route_id]
        if issues is None:
            builder.partial_reasons.add(
                "compatibility_route_evidence_missing"
            )
            continue
        for item in issues:
            builder.issues.append(
                _counterexample(
                    "compatibility_bypass",
                    route_id,
                    item.path,
                    {
                        "semantic_requirement": item.reason_code,
                        "value": item.expected,
                    },
                    item.actual,
                    item.source_ids,
                )
            )


def _transport_claim(
    builder: _ClaimBuilder,
    expected: Mapping[str, Any],
    routes: Sequence[Mapping[str, Any]],
    compliance: Mapping[
        str, tuple[ContractCounterexample, ...] | None
    ],
) -> None:
    expected_transports = _strings(
        expected.get("transports"), "expected transports"
    )
    if not expected_transports:
        builder.unknown_reasons.add("expected_transport_set_missing")
        return
    by_transport: dict[str, list[Mapping[str, Any]]] = {}
    for route in routes:
        transport = route.get("transport")
        if transport is None:
            builder.partial_reasons.add("route_transport_missing")
            continue
        selected = _text(str(transport), "route transport")
        by_transport.setdefault(selected, []).append(route)
    for transport in sorted(set(expected_transports) - set(by_transport)):
        builder.issues.append(
            _counterexample(
                "transport_route_missing",
                transport,
                "$/transports",
                True,
                False,
            )
        )
    for transport in sorted(set(by_transport) - set(expected_transports)):
        builder.issues.append(
            _counterexample(
                "unexpected_transport_route",
                transport,
                "$/transports",
                list(expected_transports),
                transport,
            )
        )
    compliant_transports: set[str] = set()
    noncompliant: dict[str, tuple[ContractCounterexample, ...]] = {}
    for transport, transport_routes in sorted(by_transport.items()):
        route_results = [compliance[route["route_id"]] for route in transport_routes]
        if any(item is None for item in route_results):
            builder.partial_reasons.add("transport_semantics_incomplete")
            continue
        flattened = tuple(
            witness
            for result in route_results
            for witness in (result or ())
        )
        if flattened:
            noncompliant[transport] = flattened
        else:
            compliant_transports.add(transport)
    for transport, issues in sorted(noncompliant.items()):
        other_is_compliant = bool(compliant_transports - {transport})
        for item in issues:
            builder.issues.append(
                _counterexample(
                    (
                        "transport_only_bypass"
                        if other_is_compliant
                        else "transport_semantics_mismatch"
                    ),
                    item.boundary_id,
                    item.path,
                    {
                        "transport": transport,
                        "semantic_requirement": item.reason_code,
                        "value": item.expected,
                    },
                    item.actual,
                    item.source_ids,
                )
            )


class McpContractAnalyzer:
    """Analyze one expected/observed operation pair without executing code."""

    interface: Final = MCP_CONTRACT_ANALYSIS_INTERFACE

    def analyze(
        self,
        expected: Mapping[str, Any],
        observed: Mapping[str, Any],
        *,
        trace: McpInvocationTrace | Mapping[str, Any] | None = None,
        aliases: Iterable[ReviewedAlias | Mapping[str, Any]] = (),
    ) -> McpContractAnalysis:
        expected_m = _mapping(expected, "expected contract")
        observed_m = _mapping(observed, "observed contract")
        operation_id = _operation_id(expected_m, observed_m)
        if trace is not None and not isinstance(trace, McpInvocationTrace):
            trace = McpInvocationTrace.from_dict(trace)
        if trace is not None and trace.operation_id != operation_id:
            raise McpContractAnalysisError("trace operation_id mismatch")
        reviewed_aliases = tuple(
            item
            if isinstance(item, ReviewedAlias)
            else ReviewedAlias.from_dict(item)
            for item in aliases
        )
        alias_pairs = [
            (item.source_name, item.target_name)
            for item in reviewed_aliases
        ]
        if len(alias_pairs) != len(set(alias_pairs)):
            raise McpContractAnalysisError("duplicate reviewed alias mapping")

        routes = _routes(observed_m)
        builders = {
            family: _ClaimBuilder(family, operation_id)
            for family in PARITY_CLAIM_FAMILIES
        }
        expected_id = _cid(
            {
                "schema": "mcp-expected-operation-contract@1",
                "contract": _identity_contract(expected_m),
            }
        )
        observed_id = _cid(
            {
                "schema": "mcp-observed-operation-contract@1",
                "contract": _identity_contract(observed_m, routes=routes),
            }
        )
        for builder in builders.values():
            builder.premise_ids.update((expected_id, observed_id))
            if not expected_m.get("complete", True):
                builder.partial_reasons.add("expected_contract_incomplete")
            if not observed_m.get("complete", True):
                builder.partial_reasons.add("observed_contract_incomplete")
            if trace is not None and not trace.complete:
                builder.partial_reasons.add("invocation_trace_incomplete")

        _schema_claim(
            builders[McpClaimFamily.DESCRIPTOR_SCHEMA_MATCHES],
            expected_m,
            routes,
            reviewed_aliases,
        )
        _argument_claim(
            builders[McpClaimFamily.ARGUMENTS_PRESERVED],
            expected_m,
            routes,
            reviewed_aliases,
        )
        _result_claim(
            builders[McpClaimFamily.RESULT_ENVELOPE_PRESERVED],
            expected_m,
            routes,
        )
        policy = _policy_claim(
            builders[McpClaimFamily.POLICY_BEFORE_EFFECT],
            expected_m,
            routes,
        )
        failure = _failure_claim(
            builders[McpClaimFamily.FAILURE_PARITY],
            expected_m,
            routes,
        )
        _discovery_claim(
            builders[McpClaimFamily.DISCOVERY_EXECUTION_PARITY],
            operation_id,
            observed_m,
            routes,
            trace,
        )
        route_semantic_families = (
            McpClaimFamily.DESCRIPTOR_SCHEMA_MATCHES,
            McpClaimFamily.ARGUMENTS_PRESERVED,
            McpClaimFamily.RESULT_ENVELOPE_PRESERVED,
        )
        compliance = {
            route["route_id"]: _route_compliance_issues(
                expected_m,
                route,
                policy.get(route["route_id"]),
                failure.get(route["route_id"]),
                tuple(
                    issue
                    for family in route_semantic_families
                    for issue in builders[family].issues
                    if issue.boundary_id == route["route_id"]
                ),
            )
            for route in routes
        }
        _compatibility_claim(
            builders[McpClaimFamily.NO_COMPATIBILITY_BYPASS],
            expected_m,
            routes,
            compliance,
            trace,
        )
        _transport_claim(
            builders[McpClaimFamily.TRANSPORT_PARITY],
            expected_m,
            routes,
            compliance,
        )

        claims = tuple(builder.finish() for builder in builders.values())
        complete = bool(
            expected_m.get("complete", True)
            and observed_m.get("complete", True)
            and (trace is None or trace.complete)
            and all(
                claim.state
                not in {
                    ParityState.AMBIGUOUS,
                    ParityState.NOT_MEASURED,
                    ParityState.PARTIAL,
                }
                for claim in claims
            )
        )
        return McpContractAnalysis(
            operation_id=operation_id,
            expected_contract_id=expected_id,
            observed_contract_id=observed_id,
            trace_id=trace.trace_id if trace else "",
            claims=claims,
            complete=complete,
        )

    def analyze_many(
        self,
        pairs: Iterable[
            tuple[Mapping[str, Any], Mapping[str, Any]]
            | Mapping[str, Any]
        ],
    ) -> tuple[McpContractAnalysis, ...]:
        """Analyze a deterministic batch of operation pairs."""

        results: list[McpContractAnalysis] = []
        for item in pairs:
            if isinstance(item, Mapping):
                results.append(
                    self.analyze(
                        item.get("expected", {}),
                        item.get("observed", {}),
                        trace=item.get("trace"),
                        aliases=item.get("aliases", ()),
                    )
                )
            else:
                expected, observed = item
                results.append(self.analyze(expected, observed))
        by_operation: dict[str, McpContractAnalysis] = {}
        for result in results:
            if result.operation_id in by_operation:
                raise McpContractAnalysisError(
                    f"duplicate operation_id: {result.operation_id}"
                )
            by_operation[result.operation_id] = result
        return tuple(by_operation[key] for key in sorted(by_operation))


def analyze_mcp_contract(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    trace: McpInvocationTrace | Mapping[str, Any] | None = None,
    aliases: Iterable[ReviewedAlias | Mapping[str, Any]] = (),
) -> McpContractAnalysis:
    """Convenience entry point for one MCP++ parity analysis."""

    return McpContractAnalyzer().analyze(
        expected, observed, trace=trace, aliases=aliases
    )


def analyze_mcp_contracts(
    pairs: Iterable[
        tuple[Mapping[str, Any], Mapping[str, Any]]
        | Mapping[str, Any]
    ],
) -> tuple[McpContractAnalysis, ...]:
    """Convenience entry point for deterministic batch analysis."""

    return McpContractAnalyzer().analyze_many(pairs)


def analyze_schema_variance(
    expected_schema: Mapping[str, Any],
    actual_schema: Mapping[str, Any],
    *,
    variance: SchemaVariance | str,
    boundary_id: str = "schema",
    source_ids: Sequence[str] = (),
) -> tuple[ContractCounterexample, ...]:
    """Check JSON-schema set inclusion at one input or output boundary.

    Inputs are contravariant: every descriptor-valid input must remain valid
    for the implementation.  Outputs are covariant: every implementation
    result must remain valid for the descriptor.
    """

    direction = _enum(variance, SchemaVariance, "schema variance")
    expected = _schema(expected_schema)
    actual = _schema(actual_schema)
    assert expected is not None and actual is not None
    unsupported = (
        *_unsupported_schema_keywords(expected),
        *_unsupported_schema_keywords(actual),
    )
    if unsupported:
        return tuple(
            _counterexample(
                "unsupported_schema_keyword",
                boundary_id,
                path,
                "supported JSON Schema fragment",
                path,
                source_ids,
            )
            for path in sorted(set(unsupported))
        )
    narrower, wider = (
        (expected, actual)
        if direction is SchemaVariance.INPUT
        else (actual, expected)
    )
    return _schema_inclusion(
        narrower,
        wider,
        boundary=boundary_id,
        path="$",
        reason_prefix=f"{direction.value}_schema",
        source_ids=source_ids,
    )


# Compact spellings for downstream obligation compilation.
McpContractClaim = ContractParityClaim
McpParityCounterexample = ContractCounterexample
ContractAnalysisState = ParityState
analyze_contract_parity = analyze_mcp_contract
check_schema_variance = analyze_schema_variance


__all__ = [
    "DEFAULT_FAILURE_STATES",
    "MCP_CONTRACT_ANALYSIS_INTERFACE",
    "MCP_CONTRACT_ANALYSIS_SCHEMA",
    "MCP_CONTRACT_ANALYSIS_VERSION",
    "MCP_CONTRACT_CLAIM_SCHEMA",
    "MCP_CONTRACT_COUNTEREXAMPLE_SCHEMA",
    "MCP_REVIEWED_ALIAS_SCHEMA",
    "PARITY_CLAIM_FAMILIES",
    "SUPPORTED_JSON_SCHEMA_KEYWORDS",
    "ContractAnalysisState",
    "ContractCounterexample",
    "ContractParityClaim",
    "McpContractAnalysis",
    "McpContractAnalysisError",
    "McpContractAnalyzer",
    "McpContractClaim",
    "McpParityCounterexample",
    "ParityState",
    "ReviewedAlias",
    "RoutePathClass",
    "SchemaVariance",
    "analyze_contract_parity",
    "analyze_mcp_contract",
    "analyze_mcp_contracts",
    "analyze_schema_variance",
    "check_schema_variance",
]
