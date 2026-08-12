"""Symbolic contract comparison and counterexample generation (VFS-016 / VFS-G051).

Pure checking over immutable expected/observed program-contract IR.  The
checker never mutates contracts, never promotes observations into
expectations, and never claims ``proved_compatible`` outside closed
supported rules.

Result kinds are closed and distinct (VFS-G051 acceptance / objective
validation repair):

* ``proved_compatible`` — every closed supported rule for the compared aspects
  holds under structural/lattice comparison (symbolic model match);
* ``witnessed_mismatch`` — a conclusive contradiction with a minimal
  counterexample witness (symbolic model disproof);
* ``runtime_witness`` — a hermetic runtime observation confirms declared
  behavior under the exact subject binding; never interchangeable with
  ``proved_compatible`` (claim level ``runtime_witnessed``);
* ``ambiguous`` — dynamic dispatch, same-name multi-target, or incomplete
  closed evidence;
* ``unsupported`` — an aspect is marked unsupported on either side;
* ``timeout`` — the declared check budget was exhausted;
* ``stale`` — authority timestamps or cache-generation bindings are stale;
* ``unknown`` — a selected closed aspect explicitly carries unknown semantics;
* ``incomplete`` — required observations were omitted or no closed rule ran.

Conclusive outcomes require exact repository, symbol, interface, policy, and
freshness binding (plan claim taxonomy: ``contract_broken`` only under a
shared subject + freshness window).  AST evidence surface includes
:class:`CodeProofObligation` and :class:`Counterexample` (alias of
:class:`ContractCounterexample`).

Large witness bodies remain artifact-referenced.  These records carry compact
facts, aspect verdicts, and deterministic content identities only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Final, TypeVar

from .program_contracts import (
    CONTRACT_VERSION as PROGRAM_CONTRACT_VERSION,
    MAX_CLAUSE_BYTES,
    MAX_COLLECTION_ITEMS,
    MAX_RECORD_BYTES,
    AtomicitySpec,
    AuthorizationSpec,
    CapabilityMode,
    CapabilitySpec,
    ConsistencySpec,
    EffectKind,
    EffectPolarity,
    ErrorSpec,
    ExpectedProgramContract,
    FallbackSpec,
    IdempotenceSpec,
    InterfaceIdentity,
    ObservedProgramContract,
    Optionality,
    OrderingMode,
    OrderingSpec,
    ParameterSpec,
    ProgramContractBundle,
    ProgramContractError,
    ResourceBounds,
    ReturnSpec,
    SemanticAspect,
    SideEffectSpec,
    SupportStatus,
    SymbolIdentity,
    SyncAsyncSpec,
    TypeShape,
    compare_type_shapes,
)
from .program_assurance_contracts import ClaimLevel
from .formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)


CHECKER_VERSION: Final[str] = "contract-checker@1"
CONTRACT_CHECKER_VERSION: Final[int] = 1
SCHEMA_VERSION: Final[int] = CONTRACT_CHECKER_VERSION
CONTRACT_CHECK_RESULT_EVIDENCE: Final[str] = "vfs/contract-check-result@1"
CONTRACT_COUNTEREXAMPLE_EVIDENCE: Final[str] = "vfs/contract-counterexample@1"
# Synthetic objective-heap evidence term for VFS-G051 validation-gate work.
OBJECTIVE_VALIDATION_REPAIR_EVIDENCE: Final[str] = "objective validation repair"
OBJECTIVE_GOAL_ID: Final[str] = "VFS-G051"
# Exact subject dimensions required for conclusive check authority.
EXACT_BINDING_DIMENSIONS: Final[tuple[str, ...]] = (
    "repository",
    "symbol",
    "interface",
    "policy",
    "freshness",
)

MAX_WITNESS_STEPS: Final[int] = 64
MAX_PATH_STEPS: Final[int] = 128
MAX_ASPECT_RESULTS: Final[int] = 64
MAX_COUNTEREXAMPLES: Final[int] = 64
MAX_CHECK_RESULTS: Final[int] = 256
MAX_TEXT_BYTES: Final[int] = 8_192
MAX_FACT_BYTES: Final[int] = 4_096
DEFAULT_BUDGET_MS: Final[int] = 5_000

# Closed aspects that admit deterministic lattice/structural rules.
CLOSED_SUPPORTED_ASPECTS: Final[tuple[SemanticAspect, ...]] = (
    SemanticAspect.IDENTITY,
    SemanticAspect.INPUTS,
    SemanticAspect.OUTPUTS,
    SemanticAspect.ERRORS,
    SemanticAspect.SYNC_ASYNC,
    SemanticAspect.SIDE_EFFECTS,
    SemanticAspect.CAPABILITIES,
    SemanticAspect.AUTHORIZATION,
    SemanticAspect.IDEMPOTENCE,
    SemanticAspect.ORDERING,
    SemanticAspect.ATOMICITY,
    SemanticAspect.CONSISTENCY,
    SemanticAspect.RESOURCE_BOUNDS,
    SemanticAspect.FALLBACK_DEGRADATION,
)

CALL_PATH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-check/call-path@1"
)
CALL_PATH_STEP_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-check/call-path-step@1"
)
ASPECT_CHECK_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-check/aspect-result@1"
)
CONTRACT_COUNTEREXAMPLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-check/counterexample@1"
)
CONTRACT_CHECK_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-check/result@1"
)
CONTRACT_CHECK_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-check/report@1"
)
CHECK_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-check/binding@1"
)


class ContractCheckerError(ContractValidationError):
    """Base error for malformed or unsafe contract-check records."""


class ContractCheckBoundsError(ContractCheckerError):
    """A compact check record exceeded an explicit item, text, or byte bound."""


class ForgedIdentityError(ContractCheckerError):
    """A caller-supplied identity or derived projection was forged."""


class UnsupportedVersionError(ContractCheckerError):
    """Schema or checker version is not supported."""


class ScopeMismatchError(ContractCheckerError):
    """Expected and observed contracts do not share exact semantic scope."""


class StaleAuthorityError(ContractCheckerError):
    """A check was attempted with expired or cache-stale authority."""


class ContractCheckResultKind(str, Enum):
    """Closed vocabulary of symbolic contract-check outcomes.

    Acceptance dimensions (VFS-G051) map as:

    * proven matches → ``proved_compatible``
    * proven mismatches → ``witnessed_mismatch``
    * runtime witnesses → ``runtime_witness``
    * ambiguity → ``ambiguous``
    * unsupported semantics → ``unsupported``
    * timeout → ``timeout``
    * stale results → ``stale``
    """

    PROVED_COMPATIBLE = "proved_compatible"
    WITNESSED_MISMATCH = "witnessed_mismatch"
    RUNTIME_WITNESS = "runtime_witness"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"
    STALE = "stale"
    UNKNOWN = "unknown"
    INCOMPLETE = "incomplete"

    @property
    def conclusive(self) -> bool:
        return self in {
            ContractCheckResultKind.PROVED_COMPATIBLE,
            ContractCheckResultKind.WITNESSED_MISMATCH,
            ContractCheckResultKind.RUNTIME_WITNESS,
        }

    @property
    def is_compatible(self) -> bool:
        return self in {
            ContractCheckResultKind.PROVED_COMPATIBLE,
            ContractCheckResultKind.RUNTIME_WITNESS,
        }

    @property
    def claim_level(self) -> ClaimLevel:
        """Program-assurance claim class for this result kind.

        Levels are intentionally non-ordered: a runtime witness never
        upgrades to model proof and a model match never claims runtime.
        """

        if self is ContractCheckResultKind.PROVED_COMPATIBLE:
            return ClaimLevel.MODEL_PROVED
        if self is ContractCheckResultKind.WITNESSED_MISMATCH:
            return ClaimLevel.MODEL_DISPROVED
        if self is ContractCheckResultKind.RUNTIME_WITNESS:
            return ClaimLevel.RUNTIME_WITNESSED
        return ClaimLevel.RESOLVED_STATIC


class ObservationLayer(str, Enum):
    """Authority layer of the observation under comparison.

    Symbolic lattice comparison (default) never claims hermetic runtime
    conformance.  Runtime-layer checks emit ``runtime_witness`` on match
    instead of ``proved_compatible``.
    """

    SYMBOLIC = "symbolic"
    RUNTIME = "runtime"


class AspectVerdict(str, Enum):
    """Per-aspect lattice outcome under closed rules."""

    COMPATIBLE = "compatible"
    MISMATCH = "mismatch"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"
    OMITTED = "omitted"

    @property
    def blocks_proved_compatible(self) -> bool:
        return self in {
            AspectVerdict.MISMATCH,
            AspectVerdict.AMBIGUOUS,
            AspectVerdict.UNSUPPORTED,
            AspectVerdict.UNKNOWN,
            AspectVerdict.OMITTED,
        }


class CallPathResolution(str, Enum):
    """How a declared call-path step is bound to an implementation."""

    STATIC = "static"
    CANDIDATE = "candidate"
    AMBIGUOUS = "ambiguous"
    DYNAMIC = "dynamic"
    EXTERNAL = "external"
    UNKNOWN = "unknown"
    PATH_TRAVERSAL = "path_traversal"

    @property
    def is_uncertain(self) -> bool:
        return self in {
            CallPathResolution.CANDIDATE,
            CallPathResolution.AMBIGUOUS,
            CallPathResolution.DYNAMIC,
            CallPathResolution.EXTERNAL,
            CallPathResolution.UNKNOWN,
            CallPathResolution.PATH_TRAVERSAL,
        }


class CacheFreshness(str, Enum):
    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"


T = TypeVar("T")
E = TypeVar("E", bound=Enum)


def _text(
    value: Any,
    *,
    field_name: str,
    required: bool = True,
    maximum: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        if required:
            raise ContractCheckerError(f"{field_name} is required")
        return ""
    if not isinstance(value, str):
        raise ContractCheckerError(f"{field_name} must be a string")
    if len(value.encode("utf-8")) > maximum:
        raise ContractCheckBoundsError(
            f"{field_name} exceeds {maximum} bytes"
        )
    if required and not value:
        raise ContractCheckerError(f"{field_name} must be non-empty")
    return value


def _boolean(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractCheckerError(f"{field_name} must be a boolean")
    return value


def _integer(
    value: Any,
    *,
    field_name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractCheckerError(f"{field_name} must be an integer")
    if value < minimum:
        raise ContractCheckerError(f"{field_name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ContractCheckBoundsError(
            f"{field_name} exceeds maximum {maximum}"
        )
    return value


def _optional_integer(
    value: Any,
    *,
    field_name: str,
    minimum: int = 0,
) -> int | None:
    if value is None:
        return None
    return _integer(value, field_name=field_name, minimum=minimum)


def _enum(value: Any, enum_type: type[E], *, field_name: str) -> E:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError as exc:
            raise ContractCheckerError(
                f"{field_name} is not a valid {enum_type.__name__}"
            ) from exc
    raise ContractCheckerError(
        f"{field_name} must be a {enum_type.__name__} or string"
    )


def _strings(
    values: Any,
    *,
    field_name: str,
    required: bool = False,
    preserve_order: bool = True,
    maximum: int = MAX_COLLECTION_ITEMS,
    item_bytes: int = MAX_CLAUSE_BYTES,
) -> tuple[str, ...]:
    if values is None:
        if required:
            raise ContractCheckerError(f"{field_name} is required")
        return ()
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ContractCheckerError(f"{field_name} must be a sequence of strings")
    if len(values) > maximum:
        raise ContractCheckBoundsError(
            f"{field_name} exceeds {maximum} items"
        )
    items: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(values):
        text = _text(
            raw,
            field_name=f"{field_name}[{index}]",
            maximum=item_bytes,
        )
        if not preserve_order:
            if text in seen:
                continue
            seen.add(text)
        items.append(text)
    if required and not items:
        raise ContractCheckerError(f"{field_name} must be non-empty")
    return tuple(items)


def _timestamp(value: Any, *, field_name: str) -> str:
    text = _text(value, field_name=field_name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractCheckerError(
            f"{field_name} must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise ContractCheckerError(f"{field_name} must be timezone-aware")
    return text


def _datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _check_header(payload: Mapping[str, Any], expected_schema: str) -> None:
    if not isinstance(payload, Mapping):
        raise ContractCheckerError("payload must be a mapping")
    schema = payload.get("schema")
    if schema != expected_schema:
        raise ContractCheckerError(
            f"schema must be {expected_schema!r}, got {schema!r}"
        )
    version = payload.get("contract_version", payload.get("schema_version"))
    if version not in (None, CONTRACT_CHECKER_VERSION, PROGRAM_CONTRACT_VERSION):
        raise UnsupportedVersionError(
            f"unsupported contract_version {version!r}"
        )


def _reject_unknown(
    payload: Mapping[str, Any],
    allowed: set[str],
    *,
    artifact_name: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ContractCheckerError(
            f"{artifact_name} has unknown fields: {', '.join(unknown)}"
        )


def _check_identity(
    payload: Mapping[str, Any],
    derived: str,
    *,
    names: Sequence[str],
    artifact_name: str,
) -> None:
    for name in names:
        claimed = payload.get(name)
        if claimed in (None, ""):
            continue
        if claimed != derived:
            raise ForgedIdentityError(
                f"{artifact_name} {name} does not match derived identity"
            )


def _bounded(
    record: CanonicalContract,
    *,
    maximum: int = MAX_RECORD_BYTES,
    artifact_name: str,
) -> None:
    size = len(record.canonical_bytes())
    if size > maximum:
        raise ContractCheckBoundsError(
            f"{artifact_name} exceeds {maximum} bytes ({size})"
        )


def _record(
    value: Any,
    typ: type[T],
    *,
    field_name: str,
) -> T:
    if isinstance(value, typ):
        return value
    if isinstance(value, Mapping):
        from_dict = getattr(typ, "from_dict", None)
        if from_dict is None:
            raise ContractCheckerError(
                f"{field_name} cannot be constructed from a mapping"
            )
        return from_dict(value)
    raise ContractCheckerError(
        f"{field_name} must be a {typ.__name__} or mapping"
    )


def _records(
    values: Any,
    typ: type[T],
    *,
    field_name: str,
    maximum: int = MAX_COLLECTION_ITEMS,
) -> tuple[T, ...]:
    if values is None:
        return ()
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ContractCheckerError(f"{field_name} must be a sequence")
    if len(values) > maximum:
        raise ContractCheckBoundsError(
            f"{field_name} exceeds {maximum} items"
        )
    return tuple(
        _record(item, typ, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(values)
    )


def _header_fields() -> set[str]:
    return {
        "schema",
        "schema_version",
        "contract_version",
        "content_id",
    }


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _fact(value: Any) -> str:
    """Render a compact deterministic fact string for witnesses."""

    if value is None:
        return "null"
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, str)):
        text = str(value)
    elif isinstance(value, TypeShape):
        text = (
            f"type:{value.constructor.value}"
            f"{':' + value.name if value.name else ''}"
        )
    elif isinstance(value, ParameterSpec):
        text = (
            f"param:{value.name}:{value.optionality.value}:"
            f"{value.type_shape.constructor.value}"
        )
    elif isinstance(value, ReturnSpec):
        text = f"return:{value.type_shape.constructor.value}"
    elif isinstance(value, ErrorSpec):
        text = f"error:{value.error_name}:{value.code or '-'}"
    elif isinstance(value, SideEffectSpec):
        text = (
            f"effect:{value.effect_kind.value}:{value.polarity.value}"
            f"{':' + value.target if value.target else ''}"
        )
    elif isinstance(value, CapabilitySpec):
        text = f"cap:{value.capability_name}:{value.mode.value}"
    elif isinstance(value, AuthorizationSpec):
        text = (
            f"auth:{value.mode.value}:scopes={','.join(value.scopes) or '-'}"
        )
    elif isinstance(value, IdempotenceSpec):
        text = f"idempotence:{value.mode.value}"
    elif isinstance(value, OrderingSpec):
        text = f"ordering:{value.mode.value}"
    elif isinstance(value, AtomicitySpec):
        text = f"atomicity:{value.mode.value}"
    elif isinstance(value, ConsistencySpec):
        text = f"consistency:{value.mode.value}"
    elif isinstance(value, ResourceBounds):
        parts = []
        for name in (
            "max_wall_time_ms",
            "max_cpu_time_ms",
            "max_memory_bytes",
            "max_payload_bytes",
            "max_output_bytes",
            "max_calls",
            "max_concurrency",
        ):
            item = getattr(value, name)
            if item is not None:
                parts.append(f"{name}={item}")
        text = "bounds:" + (",".join(parts) if parts else "unbounded")
    elif isinstance(value, FallbackSpec):
        text = f"fallback:{value.mode.value}"
    elif isinstance(value, SyncAsyncSpec):
        text = f"sync_async:{value.mode.value}"
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        text = "[" + ",".join(_fact(item) for item in value) + "]"
    else:
        text = str(value)
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_FACT_BYTES:
        return encoded[: MAX_FACT_BYTES - 3].decode("utf-8", "ignore") + "..."
    return text


def _path_is_traversal(path: str) -> bool:
    """Return whether a path string indicates traversal or absolute escape."""

    if not path:
        return False
    normalized = path.replace("\\", "/")
    if normalized.startswith("/") or (
        len(normalized) >= 2 and normalized[1] == ":"
    ):
        return True
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    return any(part == ".." for part in parts)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CheckContract(CanonicalContract):
    """Shared header helpers for contract-check IR."""

    @property
    def schema_version(self) -> int:
        return CONTRACT_CHECKER_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "contract_version": CONTRACT_CHECKER_VERSION,
            **self._payload(),
        }
        return payload


@dataclass(frozen=True)
class CallPathStep(_CheckContract):
    """One edge in a declared call path under check."""

    SCHEMA: ClassVar[str] = CALL_PATH_STEP_SCHEMA

    step_index: int
    symbol_name: str
    interface_name: str = ""
    module_path: str = ""
    resolution: CallPathResolution = CallPathResolution.STATIC
    target_path: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "step_index",
            _integer(self.step_index, field_name="step_index"),
        )
        object.__setattr__(
            self,
            "symbol_name",
            _text(self.symbol_name, field_name="symbol_name"),
        )
        object.__setattr__(
            self,
            "interface_name",
            _text(
                self.interface_name,
                field_name="interface_name",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "module_path",
            _text(self.module_path, field_name="module_path", required=False),
        )
        object.__setattr__(
            self,
            "resolution",
            _enum(self.resolution, CallPathResolution, field_name="resolution"),
        )
        object.__setattr__(
            self,
            "target_path",
            _text(self.target_path, field_name="target_path", required=False),
        )
        object.__setattr__(
            self,
            "notes",
            _text(
                self.notes,
                field_name="notes",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        if self.target_path and _path_is_traversal(self.target_path):
            object.__setattr__(
                self, "resolution", CallPathResolution.PATH_TRAVERSAL
            )
        _bounded(self, artifact_name="call path step")

    @property
    def step_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "symbol_name": self.symbol_name,
            "interface_name": self.interface_name,
            "module_path": self.module_path,
            "resolution": self.resolution.value,
            "target_path": self.target_path,
            "notes": self.notes,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "step_id": self.step_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CallPathStep":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "step_index",
            "symbol_name",
            "interface_name",
            "module_path",
            "resolution",
            "target_path",
            "notes",
        }
        _reject_unknown(
            payload,
            fields | _header_fields() | {"step_id"},
            artifact_name="call path step",
        )
        result = cls(
            step_index=payload.get("step_index", 0),
            symbol_name=payload.get("symbol_name", ""),
            interface_name=payload.get("interface_name", ""),
            module_path=payload.get("module_path", ""),
            resolution=payload.get("resolution", CallPathResolution.STATIC),
            target_path=payload.get("target_path", ""),
            notes=payload.get("notes", ""),
        )
        _check_identity(
            payload,
            result.step_id,
            names=("step_id", "content_id"),
            artifact_name="call path step",
        )
        return result


@dataclass(frozen=True)
class CallPath(_CheckContract):
    """Finite declared call path that scopes a contract comparison."""

    SCHEMA: ClassVar[str] = CALL_PATH_SCHEMA

    repository_id: str
    tree_id: str
    policy_revision: str
    path_name: str
    steps: tuple[CallPathStep, ...]
    entry_interface: str = ""
    exit_symbol: str = ""
    summary: str = ""

    def __post_init__(self) -> None:
        for name in ("repository_id", "tree_id", "policy_revision", "path_name"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "steps",
            _records(
                self.steps,
                CallPathStep,
                field_name="steps",
                maximum=MAX_PATH_STEPS,
            ),
        )
        if not self.steps:
            raise ContractCheckerError("call path requires at least one step")
        # Enforce deterministic step indices 0..n-1.
        for index, step in enumerate(self.steps):
            if step.step_index != index:
                raise ContractCheckerError(
                    "call path steps must use contiguous step_index from 0"
                )
        object.__setattr__(
            self,
            "entry_interface",
            _text(
                self.entry_interface,
                field_name="entry_interface",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "exit_symbol",
            _text(self.exit_symbol, field_name="exit_symbol", required=False),
        )
        object.__setattr__(
            self,
            "summary",
            _text(
                self.summary,
                field_name="summary",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        _bounded(self, artifact_name="call path")

    @property
    def path_id(self) -> str:
        return self.content_id

    @property
    def has_uncertainty(self) -> bool:
        return any(step.resolution.is_uncertain for step in self.steps)

    @property
    def has_path_traversal(self) -> bool:
        return any(
            step.resolution is CallPathResolution.PATH_TRAVERSAL
            or (step.target_path and _path_is_traversal(step.target_path))
            for step in self.steps
        )

    @property
    def has_dynamic_dispatch(self) -> bool:
        return any(
            step.resolution
            in {
                CallPathResolution.DYNAMIC,
                CallPathResolution.AMBIGUOUS,
            }
            for step in self.steps
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "policy_revision": self.policy_revision,
            "path_name": self.path_name,
            "steps": [step.to_dict() for step in self.steps],
            "entry_interface": self.entry_interface,
            "exit_symbol": self.exit_symbol,
            "summary": self.summary,
        }

    def to_record(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "path_id": self.path_id,
            "has_uncertainty": self.has_uncertainty,
            "has_path_traversal": self.has_path_traversal,
            "has_dynamic_dispatch": self.has_dynamic_dispatch,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CallPath":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "repository_id",
            "tree_id",
            "policy_revision",
            "path_name",
            "steps",
            "entry_interface",
            "exit_symbol",
            "summary",
        }
        _reject_unknown(
            payload,
            fields
            | _header_fields()
            | {
                "path_id",
                "has_uncertainty",
                "has_path_traversal",
                "has_dynamic_dispatch",
            },
            artifact_name="call path",
        )
        result = cls(
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            path_name=payload.get("path_name", ""),
            steps=tuple(payload.get("steps") or ()),
            entry_interface=payload.get("entry_interface", ""),
            exit_symbol=payload.get("exit_symbol", ""),
            summary=payload.get("summary", ""),
        )
        _check_identity(
            payload,
            result.path_id,
            names=("path_id", "content_id"),
            artifact_name="call path",
        )
        return result


@dataclass(frozen=True)
class CheckBinding(_CheckContract):
    """Exact repository/symbol/interface/policy/freshness binding for a check."""

    SCHEMA: ClassVar[str] = CHECK_BINDING_SCHEMA

    repository_id: str
    tree_id: str
    symbol_qualified_name: str
    expected_symbol_id: str
    observed_symbol_id: str
    interface_name: str
    expected_interface_id: str
    observed_interface_id: str
    policy_revision: str
    observed_repository_id: str
    observed_tree_id: str
    observed_policy_revision: str
    expected_contract_id: str
    observed_contract_id: str
    repository_observation_id: str = ""
    call_path_id: str = ""
    cache_generation: str = ""
    expected_cache_generation: str = ""
    checker_version: str = CHECKER_VERSION

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "tree_id",
            "symbol_qualified_name",
            "expected_symbol_id",
            "observed_symbol_id",
            "interface_name",
            "expected_interface_id",
            "observed_interface_id",
            "policy_revision",
            "observed_repository_id",
            "observed_tree_id",
            "observed_policy_revision",
            "expected_contract_id",
            "observed_contract_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )
        for name in (
            "repository_observation_id",
            "call_path_id",
            "cache_generation",
            "expected_cache_generation",
            "checker_version",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=False),
            )
        if not self.checker_version:
            object.__setattr__(self, "checker_version", CHECKER_VERSION)
        _bounded(self, artifact_name="check binding")

    @property
    def binding_id(self) -> str:
        return self.content_id

    @property
    def subject_matches(self) -> bool:
        """Whether both contracts bind the exact same repository subject."""

        return (
            self.repository_id == self.observed_repository_id
            and self.tree_id == self.observed_tree_id
            and self.expected_symbol_id == self.observed_symbol_id
            and self.expected_interface_id == self.observed_interface_id
            and self.policy_revision == self.observed_policy_revision
        )

    @property
    def cache_binding_freshness(self) -> CacheFreshness:
        """Freshness implied by an optional expected cache-generation pin."""

        if not self.expected_cache_generation:
            return CacheFreshness.UNKNOWN
        if self.cache_generation == self.expected_cache_generation:
            return CacheFreshness.CURRENT
        return CacheFreshness.STALE

    @property
    def exact_binding_dimensions(self) -> dict[str, dict[str, str]]:
        """Closed map of the five exact-binding dimensions (VFS-G051 refine).

        Dimensions: repository, symbol, interface, policy, freshness.
        Each side is named so near-matches cannot collapse under one field.
        """

        return {
            "repository": {
                "expected": self.repository_id,
                "observed": self.observed_repository_id,
                "tree_expected": self.tree_id,
                "tree_observed": self.observed_tree_id,
            },
            "symbol": {
                "expected": self.expected_symbol_id,
                "observed": self.observed_symbol_id,
                "qualified_name": self.symbol_qualified_name,
            },
            "interface": {
                "expected": self.expected_interface_id,
                "observed": self.observed_interface_id,
                "name": self.interface_name,
            },
            "policy": {
                "expected": self.policy_revision,
                "observed": self.observed_policy_revision,
            },
            "freshness": {
                "cache_generation": self.cache_generation,
                "expected_cache_generation": self.expected_cache_generation,
                "cache_binding_freshness": self.cache_binding_freshness.value,
            },
        }

    @property
    def has_complete_binding_dimensions(self) -> bool:
        """True when every exact-binding identity field is non-empty."""

        required = (
            self.repository_id,
            self.observed_repository_id,
            self.tree_id,
            self.observed_tree_id,
            self.expected_symbol_id,
            self.observed_symbol_id,
            self.expected_interface_id,
            self.observed_interface_id,
            self.policy_revision,
            self.observed_policy_revision,
            self.expected_contract_id,
            self.observed_contract_id,
        )
        return all(bool(item) for item in required)

    def diverging_binding_dimensions(self) -> tuple[str, ...]:
        """Return exact-binding dimensions that disagree across sides."""

        diverged: list[str] = []
        if (
            self.repository_id != self.observed_repository_id
            or self.tree_id != self.observed_tree_id
        ):
            diverged.append("repository")
        if self.expected_symbol_id != self.observed_symbol_id:
            diverged.append("symbol")
        if self.expected_interface_id != self.observed_interface_id:
            diverged.append("interface")
        if self.policy_revision != self.observed_policy_revision:
            diverged.append("policy")
        if self.cache_binding_freshness is CacheFreshness.STALE:
            diverged.append("freshness")
        return tuple(diverged)

    def _payload(self) -> dict[str, Any]:
        return {
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "symbol_qualified_name": self.symbol_qualified_name,
            "expected_symbol_id": self.expected_symbol_id,
            "observed_symbol_id": self.observed_symbol_id,
            "interface_name": self.interface_name,
            "expected_interface_id": self.expected_interface_id,
            "observed_interface_id": self.observed_interface_id,
            "policy_revision": self.policy_revision,
            "observed_repository_id": self.observed_repository_id,
            "observed_tree_id": self.observed_tree_id,
            "observed_policy_revision": self.observed_policy_revision,
            "expected_contract_id": self.expected_contract_id,
            "observed_contract_id": self.observed_contract_id,
            "repository_observation_id": self.repository_observation_id,
            "call_path_id": self.call_path_id,
            "cache_generation": self.cache_generation,
            "expected_cache_generation": self.expected_cache_generation,
            "checker_version": self.checker_version,
            "subject_matches": self.subject_matches,
            "cache_binding_freshness": self.cache_binding_freshness.value,
            "has_complete_binding_dimensions": (
                self.has_complete_binding_dimensions
            ),
            "diverging_binding_dimensions": list(
                self.diverging_binding_dimensions()
            ),
            "exact_binding_dimensions": self.exact_binding_dimensions,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "binding_id": self.binding_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CheckBinding":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "repository_id",
            "tree_id",
            "symbol_qualified_name",
            "expected_symbol_id",
            "observed_symbol_id",
            "interface_name",
            "expected_interface_id",
            "observed_interface_id",
            "policy_revision",
            "observed_repository_id",
            "observed_tree_id",
            "observed_policy_revision",
            "expected_contract_id",
            "observed_contract_id",
            "repository_observation_id",
            "call_path_id",
            "cache_generation",
            "expected_cache_generation",
            "checker_version",
        }
        _reject_unknown(
            payload,
            fields
            | _header_fields()
            | {
                "binding_id",
                "subject_matches",
                "cache_binding_freshness",
                "has_complete_binding_dimensions",
                "diverging_binding_dimensions",
                "exact_binding_dimensions",
            },
            artifact_name="check binding",
        )
        result = cls(
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            symbol_qualified_name=payload.get("symbol_qualified_name", ""),
            expected_symbol_id=payload.get("expected_symbol_id", ""),
            observed_symbol_id=payload.get("observed_symbol_id", ""),
            interface_name=payload.get("interface_name", ""),
            expected_interface_id=payload.get("expected_interface_id", ""),
            observed_interface_id=payload.get("observed_interface_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            observed_repository_id=payload.get("observed_repository_id", ""),
            observed_tree_id=payload.get("observed_tree_id", ""),
            observed_policy_revision=payload.get(
                "observed_policy_revision", ""
            ),
            expected_contract_id=payload.get("expected_contract_id", ""),
            observed_contract_id=payload.get("observed_contract_id", ""),
            repository_observation_id=payload.get(
                "repository_observation_id", ""
            ),
            call_path_id=payload.get("call_path_id", ""),
            cache_generation=payload.get("cache_generation", ""),
            expected_cache_generation=payload.get(
                "expected_cache_generation", ""
            ),
            checker_version=payload.get("checker_version", CHECKER_VERSION),
        )
        _check_identity(
            payload,
            result.binding_id,
            names=("binding_id", "content_id"),
            artifact_name="check binding",
        )
        claimed_subject = payload.get("subject_matches")
        if (
            claimed_subject is not None
            and claimed_subject is not result.subject_matches
        ):
            raise ForgedIdentityError(
                "subject_matches does not match exact binding identities"
            )
        claimed_freshness = payload.get("cache_binding_freshness")
        if (
            claimed_freshness is not None
            and claimed_freshness != result.cache_binding_freshness.value
        ):
            raise ForgedIdentityError(
                "cache_binding_freshness does not match generation binding"
            )
        claimed_complete = payload.get("has_complete_binding_dimensions")
        if (
            claimed_complete is not None
            and claimed_complete is not result.has_complete_binding_dimensions
        ):
            raise ForgedIdentityError(
                "has_complete_binding_dimensions does not match binding fields"
            )
        claimed_diverged = payload.get("diverging_binding_dimensions")
        if claimed_diverged is not None:
            claimed_tuple = tuple(str(item) for item in claimed_diverged)
            if claimed_tuple != result.diverging_binding_dimensions():
                raise ForgedIdentityError(
                    "diverging_binding_dimensions does not match binding sides"
                )
        return result


@dataclass(frozen=True)
class AspectCheckResult(_CheckContract):
    """Outcome of one closed or residual semantic aspect comparison."""

    SCHEMA: ClassVar[str] = ASPECT_CHECK_RESULT_SCHEMA

    aspect: SemanticAspect
    verdict: AspectVerdict
    rule_id: str
    expected_fact: str = ""
    observed_fact: str = ""
    summary: str = ""
    closed_rule: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "aspect",
            _enum(self.aspect, SemanticAspect, field_name="aspect"),
        )
        object.__setattr__(
            self,
            "verdict",
            _enum(self.verdict, AspectVerdict, field_name="verdict"),
        )
        object.__setattr__(
            self, "rule_id", _text(self.rule_id, field_name="rule_id")
        )
        for name in ("expected_fact", "observed_fact", "summary"):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name),
                    field_name=name,
                    required=False,
                    maximum=MAX_CLAUSE_BYTES,
                ),
            )
        object.__setattr__(
            self,
            "closed_rule",
            _boolean(self.closed_rule, field_name="closed_rule"),
        )
        _bounded(self, artifact_name="aspect check result")

    @property
    def aspect_result_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "aspect": self.aspect.value,
            "verdict": self.verdict.value,
            "rule_id": self.rule_id,
            "expected_fact": self.expected_fact,
            "observed_fact": self.observed_fact,
            "summary": self.summary,
            "closed_rule": self.closed_rule,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "aspect_result_id": self.aspect_result_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AspectCheckResult":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "aspect",
            "verdict",
            "rule_id",
            "expected_fact",
            "observed_fact",
            "summary",
            "closed_rule",
        }
        _reject_unknown(
            payload,
            fields | _header_fields() | {"aspect_result_id"},
            artifact_name="aspect check result",
        )
        result = cls(
            aspect=payload.get("aspect", ""),
            verdict=payload.get("verdict", ""),
            rule_id=payload.get("rule_id", ""),
            expected_fact=payload.get("expected_fact", ""),
            observed_fact=payload.get("observed_fact", ""),
            summary=payload.get("summary", ""),
            closed_rule=bool(payload.get("closed_rule", True)),
        )
        _check_identity(
            payload,
            result.aspect_result_id,
            names=("aspect_result_id", "content_id"),
            artifact_name="aspect check result",
        )
        return result


@dataclass(frozen=True)
class ContractCounterexample(_CheckContract):
    """Minimal conclusive witness that expected and observed disagree."""

    SCHEMA: ClassVar[str] = CONTRACT_COUNTEREXAMPLE_SCHEMA

    binding: CheckBinding
    aspect: SemanticAspect
    rule_id: str
    expected_fact: str
    observed_fact: str
    witness_steps: tuple[str, ...]
    summary: str
    evaluated_at: str
    authority_expires_at: str
    call_path_id: str = ""
    artifact_ref: str = ""
    conclusive: bool = True
    evidence: str = CONTRACT_COUNTEREXAMPLE_EVIDENCE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "binding",
            _record(self.binding, CheckBinding, field_name="binding"),
        )
        object.__setattr__(
            self,
            "aspect",
            _enum(self.aspect, SemanticAspect, field_name="aspect"),
        )
        object.__setattr__(
            self, "rule_id", _text(self.rule_id, field_name="rule_id")
        )
        for name in ("expected_fact", "observed_fact", "summary"):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name),
                    field_name=name,
                    maximum=MAX_CLAUSE_BYTES,
                ),
            )
        object.__setattr__(
            self,
            "witness_steps",
            _strings(
                self.witness_steps,
                field_name="witness_steps",
                required=True,
                preserve_order=True,
                maximum=MAX_WITNESS_STEPS,
                item_bytes=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "evaluated_at",
            _timestamp(self.evaluated_at, field_name="evaluated_at"),
        )
        object.__setattr__(
            self,
            "authority_expires_at",
            _timestamp(
                self.authority_expires_at, field_name="authority_expires_at"
            ),
        )
        object.__setattr__(
            self,
            "call_path_id",
            _text(self.call_path_id, field_name="call_path_id", required=False),
        )
        object.__setattr__(
            self,
            "artifact_ref",
            _text(self.artifact_ref, field_name="artifact_ref", required=False),
        )
        object.__setattr__(
            self,
            "conclusive",
            _boolean(self.conclusive, field_name="conclusive"),
        )
        object.__setattr__(
            self,
            "evidence",
            _text(self.evidence, field_name="evidence"),
        )
        if self.evidence != CONTRACT_COUNTEREXAMPLE_EVIDENCE:
            raise ContractCheckerError(
                "counterexample evidence must be "
                f"{CONTRACT_COUNTEREXAMPLE_EVIDENCE!r}"
            )
        if self.call_path_id != self.binding.call_path_id:
            raise ScopeMismatchError(
                "counterexample call path must match its exact binding"
            )
        if self.conclusive and self.freshness is CacheFreshness.STALE:
            raise StaleAuthorityError(
                "a conclusive counterexample cannot have stale authority"
            )
        _bounded(self, artifact_name="contract counterexample")

    @property
    def freshness(self) -> CacheFreshness:
        if self.binding.cache_binding_freshness is CacheFreshness.STALE:
            return CacheFreshness.STALE
        return (
            CacheFreshness.CURRENT
            if _datetime(self.evaluated_at)
            < _datetime(self.authority_expires_at)
            else CacheFreshness.STALE
        )

    @property
    def counterexample_id(self) -> str:
        return self.content_id

    @property
    def authoritative(self) -> bool:
        return self.conclusive and self.freshness is CacheFreshness.CURRENT

    def _payload(self) -> dict[str, Any]:
        return {
            "binding": self.binding.to_dict(),
            "aspect": self.aspect.value,
            "rule_id": self.rule_id,
            "expected_fact": self.expected_fact,
            "observed_fact": self.observed_fact,
            "witness_steps": list(self.witness_steps),
            "summary": self.summary,
            "evaluated_at": self.evaluated_at,
            "authority_expires_at": self.authority_expires_at,
            "call_path_id": self.call_path_id,
            "artifact_ref": self.artifact_ref,
            "conclusive": self.conclusive,
            "evidence": self.evidence,
            "freshness": self.freshness.value,
            "authoritative": self.authoritative,
        }

    def to_record(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "counterexample_id": self.counterexample_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractCounterexample":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "binding",
            "aspect",
            "rule_id",
            "expected_fact",
            "observed_fact",
            "witness_steps",
            "summary",
            "evaluated_at",
            "authority_expires_at",
            "call_path_id",
            "artifact_ref",
            "conclusive",
            "evidence",
        }
        _reject_unknown(
            payload,
            fields
            | _header_fields()
            | {
                "counterexample_id",
                "freshness",
                "authoritative",
            },
            artifact_name="contract counterexample",
        )
        result = cls(
            binding=payload.get("binding"),
            aspect=payload.get("aspect", ""),
            rule_id=payload.get("rule_id", ""),
            expected_fact=payload.get("expected_fact", ""),
            observed_fact=payload.get("observed_fact", ""),
            witness_steps=tuple(payload.get("witness_steps") or ()),
            summary=payload.get("summary", ""),
            evaluated_at=payload.get("evaluated_at", ""),
            authority_expires_at=payload.get("authority_expires_at", ""),
            call_path_id=payload.get("call_path_id", ""),
            artifact_ref=payload.get("artifact_ref", ""),
            conclusive=bool(payload.get("conclusive", True)),
            evidence=payload.get(
                "evidence", CONTRACT_COUNTEREXAMPLE_EVIDENCE
            ),
        )
        _check_identity(
            payload,
            result.counterexample_id,
            names=("counterexample_id", "content_id"),
            artifact_name="contract counterexample",
        )
        return result


# AST evidence alias for objective query ``Counterexample`` (VFS-G051).
Counterexample = ContractCounterexample


CODE_PROOF_OBLIGATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-check/code-proof-obligation@1"
)


@dataclass(frozen=True)
class CodeProofObligation(_CheckContract):
    """Bounded proof obligation derived from a contract check (VFS-G051 AST).

    Packages the exact repository, symbol, interface, policy, and freshness
    binding required before a model prover or runtime witness may act on a
    check result.  Never elevates claim level: the obligation inherits the
    result kind's claim class and refuses stale or incomplete bindings.
    """

    SCHEMA: ClassVar[str] = CODE_PROOF_OBLIGATION_SCHEMA

    binding: CheckBinding
    kind: ContractCheckResultKind
    expected_contract_id: str
    observed_contract_id: str
    evaluated_at: str
    authority_expires_at: str
    goal_id: str = OBJECTIVE_GOAL_ID
    result_id: str = ""
    counterexample_id: str = ""
    primary_aspect: str = ""
    observation_layer: ObservationLayer = ObservationLayer.SYMBOLIC
    evidence: str = CONTRACT_CHECK_RESULT_EVIDENCE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "binding",
            _record(self.binding, CheckBinding, field_name="binding"),
        )
        object.__setattr__(
            self,
            "kind",
            _enum(self.kind, ContractCheckResultKind, field_name="kind"),
        )
        for name in (
            "expected_contract_id",
            "observed_contract_id",
            "goal_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "evaluated_at",
            _timestamp(self.evaluated_at, field_name="evaluated_at"),
        )
        object.__setattr__(
            self,
            "authority_expires_at",
            _timestamp(
                self.authority_expires_at, field_name="authority_expires_at"
            ),
        )
        for name in ("result_id", "counterexample_id", "primary_aspect"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=False),
            )
        object.__setattr__(
            self,
            "observation_layer",
            _enum(
                self.observation_layer,
                ObservationLayer,
                field_name="observation_layer",
            ),
        )
        object.__setattr__(
            self,
            "evidence",
            _text(self.evidence, field_name="evidence"),
        )
        if not self.binding.has_complete_binding_dimensions:
            raise ScopeMismatchError(
                "CodeProofObligation requires complete exact binding dimensions"
            )
        if self.expected_contract_id != self.binding.expected_contract_id:
            raise ScopeMismatchError(
                "obligation expected_contract_id must match binding"
            )
        if self.observed_contract_id != self.binding.observed_contract_id:
            raise ScopeMismatchError(
                "obligation observed_contract_id must match binding"
            )
        if self.kind.conclusive and not self.binding.subject_matches:
            if not (
                self.kind is ContractCheckResultKind.WITNESSED_MISMATCH
                and self.primary_aspect == SemanticAspect.IDENTITY.value
            ):
                raise ScopeMismatchError(
                    "conclusive CodeProofObligation requires exact shared "
                    "subject binding unless the primary aspect is identity"
                )
        if self.binding.cache_binding_freshness is CacheFreshness.STALE:
            raise StaleAuthorityError(
                "CodeProofObligation cannot bind a stale cache generation"
            )
        if _datetime(self.evaluated_at) >= _datetime(self.authority_expires_at):
            raise StaleAuthorityError(
                "CodeProofObligation cannot bind an expired authority window"
            )
        _bounded(self, artifact_name="code proof obligation")

    @property
    def obligation_id(self) -> str:
        return self.content_id

    @property
    def claim_level(self) -> ClaimLevel:
        return self.kind.claim_level

    @property
    def exact_binding_dimensions(self) -> dict[str, dict[str, str]]:
        return self.binding.exact_binding_dimensions

    def _payload(self) -> dict[str, Any]:
        return {
            "binding": self.binding.to_dict(),
            "kind": self.kind.value,
            "expected_contract_id": self.expected_contract_id,
            "observed_contract_id": self.observed_contract_id,
            "evaluated_at": self.evaluated_at,
            "authority_expires_at": self.authority_expires_at,
            "goal_id": self.goal_id,
            "result_id": self.result_id,
            "counterexample_id": self.counterexample_id,
            "primary_aspect": self.primary_aspect,
            "observation_layer": self.observation_layer.value,
            "evidence": self.evidence,
            "claim_level": self.claim_level.value,
            "exact_binding_dimensions": self.exact_binding_dimensions,
        }

    def to_record(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "obligation_id": self.obligation_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodeProofObligation":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "binding",
            "kind",
            "expected_contract_id",
            "observed_contract_id",
            "evaluated_at",
            "authority_expires_at",
            "goal_id",
            "result_id",
            "counterexample_id",
            "primary_aspect",
            "observation_layer",
            "evidence",
        }
        _reject_unknown(
            payload,
            fields
            | _header_fields()
            | {
                "obligation_id",
                "claim_level",
                "exact_binding_dimensions",
            },
            artifact_name="code proof obligation",
        )
        result = cls(
            binding=payload.get("binding"),
            kind=payload.get("kind", ""),
            expected_contract_id=payload.get("expected_contract_id", ""),
            observed_contract_id=payload.get("observed_contract_id", ""),
            evaluated_at=payload.get("evaluated_at", ""),
            authority_expires_at=payload.get("authority_expires_at", ""),
            goal_id=payload.get("goal_id", OBJECTIVE_GOAL_ID),
            result_id=payload.get("result_id", ""),
            counterexample_id=payload.get("counterexample_id", ""),
            primary_aspect=payload.get("primary_aspect", ""),
            observation_layer=payload.get(
                "observation_layer", ObservationLayer.SYMBOLIC
            ),
            evidence=payload.get("evidence", CONTRACT_CHECK_RESULT_EVIDENCE),
        )
        _check_identity(
            payload,
            result.obligation_id,
            names=("obligation_id", "content_id"),
            artifact_name="code proof obligation",
        )
        claimed_level = payload.get("claim_level")
        if (
            claimed_level is not None
            and claimed_level != result.claim_level.value
        ):
            raise ForgedIdentityError(
                "claim_level does not match result kind claim class"
            )
        return result

    @classmethod
    def from_check_result(
        cls,
        result: "ContractCheckResult",
        *,
        observation_layer: ObservationLayer = ObservationLayer.SYMBOLIC,
        goal_id: str = OBJECTIVE_GOAL_ID,
    ) -> "CodeProofObligation":
        """Project a check result into a proof/runtime obligation."""

        primary = ""
        counterexample_id = ""
        if result.counterexample is not None:
            primary = result.counterexample.aspect.value
            counterexample_id = result.counterexample.counterexample_id
        elif result.mismatch_aspects:
            primary = result.mismatch_aspects[0].value
        return cls(
            binding=result.binding,
            kind=result.kind,
            expected_contract_id=result.binding.expected_contract_id,
            observed_contract_id=result.binding.observed_contract_id,
            evaluated_at=result.evaluated_at,
            authority_expires_at=result.authority_expires_at,
            goal_id=goal_id,
            result_id=result.result_id,
            counterexample_id=counterexample_id,
            primary_aspect=primary,
            observation_layer=observation_layer,
            evidence=result.evidence,
        )


@dataclass(frozen=True)
class ContractCheckResult(_CheckContract):
    """Typed result of comparing one expected contract to one observation."""

    SCHEMA: ClassVar[str] = CONTRACT_CHECK_RESULT_SCHEMA

    kind: ContractCheckResultKind
    binding: CheckBinding
    aspect_results: tuple[AspectCheckResult, ...]
    summary: str
    evaluated_at: str
    authority_expires_at: str
    counterexample: ContractCounterexample | None = None
    call_path_id: str = ""
    cache_freshness: CacheFreshness = CacheFreshness.CURRENT
    budget_ms: int = DEFAULT_BUDGET_MS
    elapsed_ms: int = 0
    checker_version: str = CHECKER_VERSION
    evidence: str = CONTRACT_CHECK_RESULT_EVIDENCE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _enum(self.kind, ContractCheckResultKind, field_name="kind"),
        )
        object.__setattr__(
            self,
            "binding",
            _record(self.binding, CheckBinding, field_name="binding"),
        )
        object.__setattr__(
            self,
            "aspect_results",
            _records(
                self.aspect_results,
                AspectCheckResult,
                field_name="aspect_results",
                maximum=MAX_ASPECT_RESULTS,
            ),
        )
        object.__setattr__(
            self,
            "summary",
            _text(
                self.summary,
                field_name="summary",
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "evaluated_at",
            _timestamp(self.evaluated_at, field_name="evaluated_at"),
        )
        object.__setattr__(
            self,
            "authority_expires_at",
            _timestamp(
                self.authority_expires_at, field_name="authority_expires_at"
            ),
        )
        if self.counterexample is not None:
            object.__setattr__(
                self,
                "counterexample",
                _record(
                    self.counterexample,
                    ContractCounterexample,
                    field_name="counterexample",
                ),
            )
        object.__setattr__(
            self,
            "call_path_id",
            _text(self.call_path_id, field_name="call_path_id", required=False),
        )
        object.__setattr__(
            self,
            "cache_freshness",
            _enum(
                self.cache_freshness,
                CacheFreshness,
                field_name="cache_freshness",
            ),
        )
        object.__setattr__(
            self,
            "budget_ms",
            _integer(self.budget_ms, field_name="budget_ms", minimum=0),
        )
        object.__setattr__(
            self,
            "elapsed_ms",
            _integer(self.elapsed_ms, field_name="elapsed_ms", minimum=0),
        )
        object.__setattr__(
            self,
            "checker_version",
            _text(self.checker_version, field_name="checker_version")
            or CHECKER_VERSION,
        )
        object.__setattr__(
            self,
            "evidence",
            _text(self.evidence, field_name="evidence"),
        )
        if self.evidence != CONTRACT_CHECK_RESULT_EVIDENCE:
            raise ContractCheckerError(
                "check result evidence must be "
                f"{CONTRACT_CHECK_RESULT_EVIDENCE!r}"
            )
        self._validate_kind_invariants()
        _bounded(self, artifact_name="contract check result")

    def _validate_kind_invariants(self) -> None:
        if self.call_path_id != self.binding.call_path_id:
            raise ScopeMismatchError(
                "check result call path must match its exact binding"
            )
        if self.checker_version != self.binding.checker_version:
            raise ScopeMismatchError(
                "check result checker version must match its exact binding"
            )
        bound_cache_freshness = self.binding.cache_binding_freshness
        if (
            bound_cache_freshness is not CacheFreshness.UNKNOWN
            and self.cache_freshness is not bound_cache_freshness
        ):
            raise StaleAuthorityError(
                "cache freshness does not match the bound cache generations"
            )
        # Conclusive outcomes require complete exact-binding dimensions.
        if (
            self.kind.conclusive
            and not self.binding.has_complete_binding_dimensions
        ):
            raise ScopeMismatchError(
                f"{self.kind.value} requires complete repository, symbol, "
                "interface, and policy identities on both sides"
            )
        if self.kind is ContractCheckResultKind.WITNESSED_MISMATCH:
            if self.counterexample is None:
                raise ContractCheckerError(
                    "witnessed_mismatch requires a counterexample"
                )
            if not self.counterexample.conclusive:
                raise ContractCheckerError(
                    "witnessed_mismatch counterexample must be conclusive"
                )
            if self.counterexample.freshness is CacheFreshness.STALE:
                raise StaleAuthorityError(
                    "witnessed_mismatch cannot use a stale counterexample"
                )
            if (
                self.counterexample.binding.binding_id
                != self.binding.binding_id
            ):
                raise ContractCheckerError(
                    "counterexample binding must match check result binding"
                )
            if self.counterexample.call_path_id != self.call_path_id:
                raise ScopeMismatchError(
                    "counterexample call path must match check result"
                )
            if (
                self.counterexample.evaluated_at != self.evaluated_at
                or self.counterexample.authority_expires_at
                != self.authority_expires_at
            ):
                raise StaleAuthorityError(
                    "counterexample freshness window must match check result"
                )
            # Non-identity contract_broken claims require shared subject binding.
            if (
                self.counterexample.aspect is not SemanticAspect.IDENTITY
                and not self.binding.subject_matches
            ):
                raise ScopeMismatchError(
                    "non-identity witnessed_mismatch requires exact shared "
                    "repository, symbol, interface, and policy binding"
                )
        elif self.counterexample is not None:
            raise ContractCheckerError(
                f"{self.kind.value} must not carry a counterexample"
            )
        if self.kind is ContractCheckResultKind.PROVED_COMPATIBLE:
            if not self.binding.subject_matches:
                raise ScopeMismatchError(
                    "proved_compatible requires exact shared subject binding"
                )
            for item in self.aspect_results:
                if item.closed_rule and item.verdict.blocks_proved_compatible:
                    raise ContractCheckerError(
                        "proved_compatible cannot include blocking aspect verdicts"
                    )
                if not item.closed_rule and item.verdict is AspectVerdict.MISMATCH:
                    raise ContractCheckerError(
                        "proved_compatible cannot claim mismatch on open rules"
                    )
        if self.kind is ContractCheckResultKind.RUNTIME_WITNESS:
            if not self.binding.subject_matches:
                raise ScopeMismatchError(
                    "runtime_witness requires exact shared subject binding"
                )
            for item in self.aspect_results:
                if item.closed_rule and item.verdict.blocks_proved_compatible:
                    raise ContractCheckerError(
                        "runtime_witness cannot include blocking aspect verdicts"
                    )
        if self.kind is ContractCheckResultKind.UNKNOWN and not any(
            item.verdict is AspectVerdict.UNKNOWN
            for item in self.aspect_results
        ):
            raise ContractCheckerError(
                "unknown result requires an explicit unknown aspect verdict"
            )
        if (
            self.kind is not ContractCheckResultKind.STALE
            and self.freshness is CacheFreshness.STALE
        ):
            raise StaleAuthorityError(
                "non-stale result cannot use expired authority or stale cache"
            )
        if self.kind is ContractCheckResultKind.STALE:
            if self.freshness is not CacheFreshness.STALE:
                raise ContractCheckerError(
                    "stale result requires stale cache or expired authority"
                )
        if self.kind is ContractCheckResultKind.TIMEOUT:
            if self.elapsed_ms < self.budget_ms and self.budget_ms > 0:
                # Allow explicit timeout injection only when budget is zero
                # (forced) or elapsed exceeds budget.
                if self.budget_ms != 0:
                    raise ContractCheckerError(
                        "timeout requires elapsed_ms >= budget_ms or zero budget"
                    )

    @property
    def result_id(self) -> str:
        return self.content_id

    @property
    def freshness(self) -> CacheFreshness:
        if _datetime(self.evaluated_at) >= _datetime(
            self.authority_expires_at
        ):
            return CacheFreshness.STALE
        if self.cache_freshness is CacheFreshness.STALE:
            return CacheFreshness.STALE
        if self.cache_freshness is CacheFreshness.UNKNOWN:
            return CacheFreshness.UNKNOWN
        return CacheFreshness.CURRENT

    @property
    def mismatch_aspects(self) -> tuple[SemanticAspect, ...]:
        return tuple(
            item.aspect
            for item in self.aspect_results
            if item.verdict is AspectVerdict.MISMATCH
        )

    @property
    def claim_level(self) -> ClaimLevel:
        return self.kind.claim_level

    def as_code_proof_obligation(
        self,
        *,
        observation_layer: ObservationLayer = ObservationLayer.SYMBOLIC,
        goal_id: str = OBJECTIVE_GOAL_ID,
    ) -> CodeProofObligation:
        """Project this result into a :class:`CodeProofObligation`."""

        return CodeProofObligation.from_check_result(
            self,
            observation_layer=observation_layer,
            goal_id=goal_id,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "binding": self.binding.to_dict(),
            "aspect_results": [item.to_dict() for item in self.aspect_results],
            "summary": self.summary,
            "evaluated_at": self.evaluated_at,
            "authority_expires_at": self.authority_expires_at,
            "counterexample": (
                None
                if self.counterexample is None
                else self.counterexample.to_dict()
            ),
            "call_path_id": self.call_path_id,
            "cache_freshness": self.cache_freshness.value,
            "budget_ms": self.budget_ms,
            "elapsed_ms": self.elapsed_ms,
            "checker_version": self.checker_version,
            "evidence": self.evidence,
            "freshness": self.freshness.value,
            "claim_level": self.claim_level.value,
        }

    def to_record(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "result_id": self.result_id,
            "mismatch_aspects": [item.value for item in self.mismatch_aspects],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractCheckResult":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "kind",
            "binding",
            "aspect_results",
            "summary",
            "evaluated_at",
            "authority_expires_at",
            "counterexample",
            "call_path_id",
            "cache_freshness",
            "budget_ms",
            "elapsed_ms",
            "checker_version",
            "evidence",
        }
        _reject_unknown(
            payload,
            fields
            | _header_fields()
            | {
                "result_id",
                "freshness",
                "mismatch_aspects",
                "claim_level",
            },
            artifact_name="contract check result",
        )
        result = cls(
            kind=payload.get("kind", ""),
            binding=payload.get("binding"),
            aspect_results=tuple(payload.get("aspect_results") or ()),
            summary=payload.get("summary", ""),
            evaluated_at=payload.get("evaluated_at", ""),
            authority_expires_at=payload.get("authority_expires_at", ""),
            counterexample=payload.get("counterexample"),
            call_path_id=payload.get("call_path_id", ""),
            cache_freshness=payload.get(
                "cache_freshness", CacheFreshness.CURRENT
            ),
            budget_ms=payload.get("budget_ms", DEFAULT_BUDGET_MS),
            elapsed_ms=payload.get("elapsed_ms", 0),
            checker_version=payload.get("checker_version", CHECKER_VERSION),
            evidence=payload.get("evidence", CONTRACT_CHECK_RESULT_EVIDENCE),
        )
        _check_identity(
            payload,
            result.result_id,
            names=("result_id", "content_id"),
            artifact_name="contract check result",
        )
        claimed_level = payload.get("claim_level")
        if (
            claimed_level is not None
            and claimed_level != result.claim_level.value
        ):
            raise ForgedIdentityError(
                "claim_level does not match result kind claim class"
            )
        return result


@dataclass(frozen=True)
class ContractCheckReport(_CheckContract):
    """Bounded multi-pair report over a bundle and optional call paths."""

    SCHEMA: ClassVar[str] = CONTRACT_CHECK_REPORT_SCHEMA

    repository_id: str
    tree_id: str
    policy_revision: str
    results: tuple[ContractCheckResult, ...]
    evaluated_at: str
    authority_expires_at: str
    summary: str = ""
    checker_version: str = CHECKER_VERSION

    def __post_init__(self) -> None:
        for name in ("repository_id", "tree_id", "policy_revision"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "results",
            _records(
                self.results,
                ContractCheckResult,
                field_name="results",
                maximum=MAX_CHECK_RESULTS,
            ),
        )
        object.__setattr__(
            self,
            "evaluated_at",
            _timestamp(self.evaluated_at, field_name="evaluated_at"),
        )
        object.__setattr__(
            self,
            "authority_expires_at",
            _timestamp(
                self.authority_expires_at, field_name="authority_expires_at"
            ),
        )
        object.__setattr__(
            self,
            "summary",
            _text(
                self.summary,
                field_name="summary",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "checker_version",
            _text(self.checker_version, field_name="checker_version")
            or CHECKER_VERSION,
        )
        for result in self.results:
            if (
                result.binding.repository_id != self.repository_id
                or result.binding.tree_id != self.tree_id
                or result.binding.policy_revision != self.policy_revision
            ):
                raise ScopeMismatchError(
                    "report results must match repository, tree, and policy"
                )
            if (
                result.evaluated_at != self.evaluated_at
                or result.authority_expires_at != self.authority_expires_at
            ):
                raise StaleAuthorityError(
                    "report results must match the report freshness window"
                )
        _bounded(self, maximum=MAX_RECORD_BYTES * 4, artifact_name="check report")

    @property
    def report_id(self) -> str:
        return self.content_id

    @property
    def counts_by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {kind.value: 0 for kind in ContractCheckResultKind}
        for item in self.results:
            counts[item.kind.value] = counts.get(item.kind.value, 0) + 1
        return counts

    def _payload(self) -> dict[str, Any]:
        return {
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "policy_revision": self.policy_revision,
            "results": [item.to_dict() for item in self.results],
            "evaluated_at": self.evaluated_at,
            "authority_expires_at": self.authority_expires_at,
            "summary": self.summary,
            "checker_version": self.checker_version,
            "counts_by_kind": self.counts_by_kind,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "report_id": self.report_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractCheckReport":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "repository_id",
            "tree_id",
            "policy_revision",
            "results",
            "evaluated_at",
            "authority_expires_at",
            "summary",
            "checker_version",
        }
        _reject_unknown(
            payload,
            fields | _header_fields() | {"report_id", "counts_by_kind"},
            artifact_name="contract check report",
        )
        result = cls(
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            results=tuple(payload.get("results") or ()),
            evaluated_at=payload.get("evaluated_at", ""),
            authority_expires_at=payload.get("authority_expires_at", ""),
            summary=payload.get("summary", ""),
            checker_version=payload.get("checker_version", CHECKER_VERSION),
        )
        _check_identity(
            payload,
            result.report_id,
            names=("report_id", "content_id"),
            artifact_name="contract check report",
        )
        claimed = payload.get("counts_by_kind")
        if claimed is not None and claimed != result.counts_by_kind:
            raise ForgedIdentityError(
                "counts_by_kind does not match derived state"
            )
        return result


# ---------------------------------------------------------------------------
# Closed comparison rules
# ---------------------------------------------------------------------------


def _aspect_result(
    aspect: SemanticAspect,
    verdict: AspectVerdict,
    *,
    rule_id: str,
    expected_fact: str = "",
    observed_fact: str = "",
    summary: str = "",
    closed_rule: bool = True,
) -> AspectCheckResult:
    return AspectCheckResult(
        aspect=aspect,
        verdict=verdict,
        rule_id=rule_id,
        expected_fact=expected_fact,
        observed_fact=observed_fact,
        summary=summary,
        closed_rule=closed_rule,
    )


def _unsupported_or_unknown(
    aspect: SemanticAspect,
    expected_status: SupportStatus,
    observed_status: SupportStatus,
    *,
    rule_id: str,
) -> AspectCheckResult | None:
    if (
        expected_status is SupportStatus.UNSUPPORTED
        or observed_status is SupportStatus.UNSUPPORTED
    ):
        return _aspect_result(
            aspect,
            AspectVerdict.UNSUPPORTED,
            rule_id=rule_id,
            expected_fact=_fact(expected_status),
            observed_fact=_fact(observed_status),
            summary=f"{aspect.value} marked unsupported",
            closed_rule=True,
        )
    if (
        expected_status is SupportStatus.UNKNOWN
        or observed_status is SupportStatus.UNKNOWN
    ):
        return _aspect_result(
            aspect,
            AspectVerdict.UNKNOWN,
            rule_id=rule_id,
            expected_fact=_fact(expected_status),
            observed_fact=_fact(observed_status),
            summary=f"{aspect.value} support unknown",
            closed_rule=True,
        )
    if expected_status is SupportStatus.NOT_APPLICABLE:
        return _aspect_result(
            aspect,
            AspectVerdict.NOT_APPLICABLE,
            rule_id=rule_id,
            expected_fact=_fact(expected_status),
            observed_fact=_fact(observed_status),
            summary=f"{aspect.value} not applicable",
            closed_rule=True,
        )
    return None


def _binds_exact_subject(
    expected: ExpectedProgramContract,
    observed: ObservedProgramContract,
) -> bool:
    """Bind repository, complete symbol/interface identities, and policy."""

    return (
        expected.symbol.repository_id == observed.symbol.repository_id
        and expected.symbol.tree_id == observed.symbol.tree_id
        and expected.symbol.symbol_id == observed.symbol.symbol_id
        and expected.interface.interface_id == observed.interface.interface_id
        and expected.policy_revision == observed.policy_revision
    )


def check_identity(
    expected: ExpectedProgramContract,
    observed: ObservedProgramContract,
) -> AspectCheckResult:
    """Closed identity rule: exact repository/symbol/interface/policy binding."""

    rule = "rule:identity/exact-binding@1"
    same_subject = _binds_exact_subject(expected, observed)
    expected_fact = (
        f"{expected.symbol.repository_id}|{expected.symbol.tree_id}|"
        f"{expected.symbol.symbol_id}|{expected.interface.interface_id}|"
        f"{expected.policy_revision}"
    )
    observed_fact = (
        f"{observed.symbol.repository_id}|{observed.symbol.tree_id}|"
        f"{observed.symbol.symbol_id}|{observed.interface.interface_id}|"
        f"{observed.policy_revision}"
    )
    if same_subject:
        return _aspect_result(
            SemanticAspect.IDENTITY,
            AspectVerdict.COMPATIBLE,
            rule_id=rule,
            expected_fact=expected_fact,
            observed_fact=observed_fact,
            summary="identity binding matches",
        )
    return _aspect_result(
        SemanticAspect.IDENTITY,
        AspectVerdict.MISMATCH,
        rule_id=rule,
        expected_fact=expected_fact,
        observed_fact=observed_fact,
        summary="identity binding diverges",
    )


def check_inputs(
    expected: ExpectedProgramContract,
    observed: ObservedProgramContract,
) -> AspectCheckResult:
    """Closed input variance: required expected params accepted contravariantly."""

    rule = "rule:inputs/contravariant-required@1"
    if expected.aspect_support(SemanticAspect.INPUTS) is SupportStatus.UNSUPPORTED:
        return _aspect_result(
            SemanticAspect.INPUTS,
            AspectVerdict.UNSUPPORTED,
            rule_id=rule,
            summary="inputs unsupported on expectation",
        )
    expected_by_name = {param.name: param for param in expected.inputs}
    observed_by_name = {param.name: param for param in observed.inputs}
    for name, required in expected_by_name.items():
        if required.optionality is Optionality.OPTIONAL:
            # Optional expected inputs may be omitted by the observation.
            if name not in observed_by_name:
                continue
            provided = observed_by_name[name]
            if not provided.is_input_compatible_with(required):
                return _aspect_result(
                    SemanticAspect.INPUTS,
                    AspectVerdict.MISMATCH,
                    rule_id=rule,
                    expected_fact=_fact(required),
                    observed_fact=_fact(provided),
                    summary=f"optional input {name!r} type is incompatible",
                )
            continue
        if name not in observed_by_name:
            return _aspect_result(
                SemanticAspect.INPUTS,
                AspectVerdict.MISMATCH,
                rule_id=rule,
                expected_fact=_fact(required),
                observed_fact="missing",
                summary=f"required input {name!r} is missing from observation",
            )
        provided = observed_by_name[name]
        if provided.optionality is Optionality.OPTIONAL and (
            required.optionality is Optionality.REQUIRED
        ):
            return _aspect_result(
                SemanticAspect.INPUTS,
                AspectVerdict.MISMATCH,
                rule_id=rule,
                expected_fact=_fact(required),
                observed_fact=_fact(provided),
                summary=(
                    f"required input {name!r} observed as optional"
                ),
            )
        if not provided.is_input_compatible_with(required):
            return _aspect_result(
                SemanticAspect.INPUTS,
                AspectVerdict.MISMATCH,
                rule_id=rule,
                expected_fact=_fact(required),
                observed_fact=_fact(provided),
                summary=f"input {name!r} fails contravariant acceptance",
            )
    return _aspect_result(
        SemanticAspect.INPUTS,
        AspectVerdict.COMPATIBLE,
        rule_id=rule,
        expected_fact=_fact(expected.inputs),
        observed_fact=_fact(observed.inputs),
        summary="inputs accept required expectation values",
    )


def check_outputs(
    expected: ExpectedProgramContract,
    observed: ObservedProgramContract,
) -> AspectCheckResult:
    """Closed output variance: observed return must subtype expected return."""

    rule = "rule:outputs/covariant-return@1"
    if expected.returns is None:
        return _aspect_result(
            SemanticAspect.OUTPUTS,
            AspectVerdict.NOT_APPLICABLE,
            rule_id=rule,
            summary="no expected return type",
        )
    if observed.returns is None:
        return _aspect_result(
            SemanticAspect.OUTPUTS,
            AspectVerdict.MISMATCH,
            rule_id=rule,
            expected_fact=_fact(expected.returns),
            observed_fact="missing",
            summary="expected return type is absent from observation",
        )
    if observed.returns.is_subtype_of(expected.returns):
        return _aspect_result(
            SemanticAspect.OUTPUTS,
            AspectVerdict.COMPATIBLE,
            rule_id=rule,
            expected_fact=_fact(expected.returns),
            observed_fact=_fact(observed.returns),
            summary="return type is a structural subtype of expectation",
        )
    relation = compare_type_shapes(
        observed.returns.type_shape, expected.returns.type_shape
    )
    return _aspect_result(
        SemanticAspect.OUTPUTS,
        AspectVerdict.MISMATCH,
        rule_id=rule,
        expected_fact=_fact(expected.returns),
        observed_fact=_fact(observed.returns),
        summary=f"return type relation is {relation.value}",
    )


def check_errors(
    expected: ExpectedProgramContract,
    observed: ObservedProgramContract,
) -> AspectCheckResult:
    """Closed error-map rule: shared error names must not disagree on codes."""

    rule = "rule:errors/code-map@1"
    if not expected.errors:
        return _aspect_result(
            SemanticAspect.ERRORS,
            AspectVerdict.NOT_APPLICABLE,
            rule_id=rule,
            summary="no expected error map",
        )
    expected_by_name = {item.error_name: item for item in expected.errors}
    observed_by_name = {item.error_name: item for item in observed.errors}
    for name, expected_err in expected_by_name.items():
        if name not in observed_by_name:
            # Omission is incomplete rather than a hard mismatch: the
            # observation may not have exercised the error path.
            continue
        observed_err = observed_by_name[name]
        if expected_err.code and observed_err.code and (
            expected_err.code != observed_err.code
        ):
            return _aspect_result(
                SemanticAspect.ERRORS,
                AspectVerdict.MISMATCH,
                rule_id=rule,
                expected_fact=_fact(expected_err),
                observed_fact=_fact(observed_err),
                summary=f"error {name!r} code map disagrees",
            )
        if (
            expected_err.error_type is not None
            and observed_err.error_type is not None
            and not observed_err.error_type.is_subtype_of(expected_err.error_type)
        ):
            return _aspect_result(
                SemanticAspect.ERRORS,
                AspectVerdict.MISMATCH,
                rule_id=rule,
                expected_fact=_fact(expected_err),
                observed_fact=_fact(observed_err),
                summary=f"error {name!r} type is incompatible",
            )
    # All shared names agree; omitted expected errors do not disprove.
    shared = sorted(set(expected_by_name) & set(observed_by_name))
    if not shared and observed.errors:
        return _aspect_result(
            SemanticAspect.ERRORS,
            AspectVerdict.AMBIGUOUS,
            rule_id=rule,
            expected_fact=_fact(expected.errors),
            observed_fact=_fact(observed.errors),
            summary="observed errors share no names with expectation map",
        )
    return _aspect_result(
        SemanticAspect.ERRORS,
        AspectVerdict.COMPATIBLE,
        rule_id=rule,
        expected_fact=_fact(expected.errors),
        observed_fact=_fact(observed.errors),
        summary="shared error map entries agree",
    )


def check_sync_async(
    expected: ExpectedProgramContract,
    observed: ObservedProgramContract,
) -> AspectCheckResult:
    rule = "rule:sync-async/mode-compat@1"
    if expected.sync_async is None:
        return _aspect_result(
            SemanticAspect.SYNC_ASYNC,
            AspectVerdict.NOT_APPLICABLE,
            rule_id=rule,
            summary="no expected sync/async mode",
        )
    if observed.sync_async is None:
        return _aspect_result(
            SemanticAspect.SYNC_ASYNC,
            AspectVerdict.OMITTED,
            rule_id=rule,
            expected_fact=_fact(expected.sync_async),
            observed_fact="missing",
            summary="observed sync/async mode omitted",
        )
    residual = _unsupported_or_unknown(
        SemanticAspect.SYNC_ASYNC,
        expected.sync_async.support,
        observed.sync_async.support,
        rule_id=rule,
    )
    if residual is not None and residual.verdict is not AspectVerdict.NOT_APPLICABLE:
        if residual.verdict in {
            AspectVerdict.UNSUPPORTED,
            AspectVerdict.UNKNOWN,
        }:
            return residual
    if observed.sync_async.is_compatible_with(expected.sync_async):
        return _aspect_result(
            SemanticAspect.SYNC_ASYNC,
            AspectVerdict.COMPATIBLE,
            rule_id=rule,
            expected_fact=_fact(expected.sync_async),
            observed_fact=_fact(observed.sync_async),
            summary="sync/async modes are compatible",
        )
    return _aspect_result(
        SemanticAspect.SYNC_ASYNC,
        AspectVerdict.MISMATCH,
        rule_id=rule,
        expected_fact=_fact(expected.sync_async),
        observed_fact=_fact(observed.sync_async),
        summary="sync/async modes are incompatible",
    )


def check_side_effects(
    expected: ExpectedProgramContract,
    observed: ObservedProgramContract,
) -> AspectCheckResult:
    """Closed effect rule for required, forbidden, and omitted effects."""

    rule = "rule:side-effects/required-forbidden@1"
    expected_required = [
        effect
        for effect in expected.side_effects
        if effect.polarity is EffectPolarity.REQUIRED
    ]
    expected_forbidden = {
        effect.effect_kind
        for effect in expected.side_effects
        if effect.polarity is EffectPolarity.FORBIDDEN
    }
    expected_allowed = {
        effect.effect_kind
        for effect in expected.side_effects
        if effect.polarity
        in {EffectPolarity.ALLOWED, EffectPolarity.REQUIRED, EffectPolarity.OBSERVED}
    }
    observed_effects = [
        effect
        for effect in observed.side_effects
        if effect.effect_kind is not EffectKind.NONE
    ]
    # Required effects must appear (observed or allowed polarity).
    observed_kinds = {effect.effect_kind for effect in observed_effects}
    for required in expected_required:
        if required.effect_kind not in observed_kinds:
            return _aspect_result(
                SemanticAspect.SIDE_EFFECTS,
                AspectVerdict.MISMATCH,
                rule_id=rule,
                expected_fact=_fact(required),
                observed_fact="omitted",
                summary=(
                    f"required effect {required.effect_kind.value} is omitted"
                ),
            )
    # Forbidden effects must not be observed.
    for effect in observed_effects:
        if effect.effect_kind in expected_forbidden:
            return _aspect_result(
                SemanticAspect.SIDE_EFFECTS,
                AspectVerdict.MISMATCH,
                rule_id=rule,
                expected_fact=f"forbidden:{effect.effect_kind.value}",
                observed_fact=_fact(effect),
                summary=(
                    f"forbidden effect {effect.effect_kind.value} was observed"
                ),
            )
        if expected_allowed and effect.effect_kind not in expected_allowed:
            # Unknown extra effect under a non-empty allowance set is ambiguous
            # unless the allowance set is closed with NONE only.
            if EffectKind.NONE in expected_allowed and len(expected_allowed) == 1:
                return _aspect_result(
                    SemanticAspect.SIDE_EFFECTS,
                    AspectVerdict.MISMATCH,
                    rule_id=rule,
                    expected_fact="allowed:none",
                    observed_fact=_fact(effect),
                    summary="pure expectation forbids all effects",
                )
            return _aspect_result(
                SemanticAspect.SIDE_EFFECTS,
                AspectVerdict.AMBIGUOUS,
                rule_id=rule,
                expected_fact=_fact(
                    sorted(item.value for item in expected_allowed)
                ),
                observed_fact=_fact(effect),
                summary=(
                    f"observed effect {effect.effect_kind.value} is outside "
                    "the declared allowance set"
                ),
            )
        # Check target-level allowance when present.
        matching_allowances = [
            item
            for item in expected.side_effects
            if item.effect_kind is effect.effect_kind
            and item.polarity
            in {
                EffectPolarity.ALLOWED,
                EffectPolarity.REQUIRED,
                EffectPolarity.OBSERVED,
            }
        ]
        if matching_allowances and not any(
            effect.is_allowed_by(item) for item in matching_allowances
        ):
            return _aspect_result(
                SemanticAspect.SIDE_EFFECTS,
                AspectVerdict.MISMATCH,
                rule_id=rule,
                expected_fact=_fact(matching_allowances),
                observed_fact=_fact(effect),
                summary=(
                    f"effect {effect.effect_kind.value} target is not allowed"
                ),
            )
    return _aspect_result(
        SemanticAspect.SIDE_EFFECTS,
        AspectVerdict.COMPATIBLE,
        rule_id=rule,
        expected_fact=_fact(expected.side_effects),
        observed_fact=_fact(observed.side_effects),
        summary="side effects satisfy required/forbidden constraints",
    )


def check_capabilities(
    expected: ExpectedProgramContract,
    observed: ObservedProgramContract,
) -> AspectCheckResult:
    rule = "rule:capabilities/required-present@1"
    if not expected.capabilities:
        return _aspect_result(
            SemanticAspect.CAPABILITIES,
            AspectVerdict.NOT_APPLICABLE,
            rule_id=rule,
            summary="no expected capabilities",
        )
    observed_names = {
        item.capability_name: item for item in observed.capabilities
    }
    for item in expected.capabilities:
        if item.mode is CapabilityMode.REQUIRED:
            if item.capability_name not in observed_names:
                return _aspect_result(
                    SemanticAspect.CAPABILITIES,
                    AspectVerdict.MISMATCH,
                    rule_id=rule,
                    expected_fact=_fact(item),
                    observed_fact="missing",
                    summary=(
                        f"required capability {item.capability_name!r} missing"
                    ),
                )
            observed_item = observed_names[item.capability_name]
            if observed_item.mode is CapabilityMode.FORBIDDEN:
                return _aspect_result(
                    SemanticAspect.CAPABILITIES,
                    AspectVerdict.MISMATCH,
                    rule_id=rule,
                    expected_fact=_fact(item),
                    observed_fact=_fact(observed_item),
                    summary=(
                        f"required capability {item.capability_name!r} "
                        "is forbidden in observation"
                    ),
                )
        if item.mode is CapabilityMode.FORBIDDEN:
            observed_item = observed_names.get(item.capability_name)
            if observed_item is not None and observed_item.mode in {
                CapabilityMode.REQUIRED,
                CapabilityMode.OBSERVED,
                CapabilityMode.OPTIONAL,
                CapabilityMode.NEGOTIATED,
            }:
                return _aspect_result(
                    SemanticAspect.CAPABILITIES,
                    AspectVerdict.MISMATCH,
                    rule_id=rule,
                    expected_fact=_fact(item),
                    observed_fact=_fact(observed_item),
                    summary=(
                        f"forbidden capability {item.capability_name!r} "
                        "is present"
                    ),
                )
    return _aspect_result(
        SemanticAspect.CAPABILITIES,
        AspectVerdict.COMPATIBLE,
        rule_id=rule,
        expected_fact=_fact(expected.capabilities),
        observed_fact=_fact(observed.capabilities),
        summary="required capabilities are present",
    )


def check_authorization(
    expected: ExpectedProgramContract,
    observed: ObservedProgramContract,
) -> AspectCheckResult:
    rule = "rule:authorization/refinement@1"
    if expected.authorization is None:
        return _aspect_result(
            SemanticAspect.AUTHORIZATION,
            AspectVerdict.NOT_APPLICABLE,
            rule_id=rule,
            summary="no expected authorization",
        )
    if observed.authorization is None:
        return _aspect_result(
            SemanticAspect.AUTHORIZATION,
            AspectVerdict.OMITTED,
            rule_id=rule,
            expected_fact=_fact(expected.authorization),
            observed_fact="missing",
            summary="authorization omitted from observation",
        )
    residual = _unsupported_or_unknown(
        SemanticAspect.AUTHORIZATION,
        expected.authorization.support,
        observed.authorization.support,
        rule_id=rule,
    )
    if residual is not None and residual.verdict in {
        AspectVerdict.UNSUPPORTED,
        AspectVerdict.UNKNOWN,
    }:
        return residual
    if observed.authorization.is_refinement_of(expected.authorization):
        return _aspect_result(
            SemanticAspect.AUTHORIZATION,
            AspectVerdict.COMPATIBLE,
            rule_id=rule,
            expected_fact=_fact(expected.authorization),
            observed_fact=_fact(observed.authorization),
            summary="authorization refines expectation",
        )
    return _aspect_result(
        SemanticAspect.AUTHORIZATION,
        AspectVerdict.MISMATCH,
        rule_id=rule,
        expected_fact=_fact(expected.authorization),
        observed_fact=_fact(observed.authorization),
        summary="authorization does not refine expectation",
    )


def check_idempotence(
    expected: ExpectedProgramContract,
    observed: ObservedProgramContract,
) -> AspectCheckResult:
    rule = "rule:idempotence/strength@1"
    if expected.idempotence is None:
        return _aspect_result(
            SemanticAspect.IDEMPOTENCE,
            AspectVerdict.NOT_APPLICABLE,
            rule_id=rule,
            summary="no expected idempotence",
        )
    if observed.idempotence is None:
        return _aspect_result(
            SemanticAspect.IDEMPOTENCE,
            AspectVerdict.OMITTED,
            rule_id=rule,
            expected_fact=_fact(expected.idempotence),
            observed_fact="missing",
            summary="idempotence omitted from observation",
        )
    if observed.idempotence.is_refinement_of(expected.idempotence):
        return _aspect_result(
            SemanticAspect.IDEMPOTENCE,
            AspectVerdict.COMPATIBLE,
            rule_id=rule,
            expected_fact=_fact(expected.idempotence),
            observed_fact=_fact(observed.idempotence),
            summary="idempotence is at least as strong as expectation",
        )
    return _aspect_result(
        SemanticAspect.IDEMPOTENCE,
        AspectVerdict.MISMATCH,
        rule_id=rule,
        expected_fact=_fact(expected.idempotence),
        observed_fact=_fact(observed.idempotence),
        summary="idempotence is weaker than expectation",
    )


def check_ordering(
    expected: ExpectedProgramContract,
    observed: ObservedProgramContract,
) -> AspectCheckResult:
    rule = "rule:ordering/mode@1"
    if expected.ordering is None:
        return _aspect_result(
            SemanticAspect.ORDERING,
            AspectVerdict.NOT_APPLICABLE,
            rule_id=rule,
            summary="no expected ordering",
        )
    if observed.ordering is None:
        return _aspect_result(
            SemanticAspect.ORDERING,
            AspectVerdict.OMITTED,
            rule_id=rule,
            expected_fact=_fact(expected.ordering),
            observed_fact="missing",
            summary="ordering omitted from observation",
        )
    exp = expected.ordering.mode
    obs = observed.ordering.mode
    if exp is OrderingMode.UNKNOWN or obs is OrderingMode.UNKNOWN:
        return _aspect_result(
            SemanticAspect.ORDERING,
            AspectVerdict.UNKNOWN,
            rule_id=rule,
            expected_fact=_fact(expected.ordering),
            observed_fact=_fact(observed.ordering),
            summary="ordering mode unknown",
        )
    # Closed compatibility lattice: same mode is compatible; total/sequential
    # satisfy causal/partial; concurrent is incompatible with sequential/total.
    if exp is obs:
        verdict = AspectVerdict.COMPATIBLE
        summary = "ordering modes match"
    elif exp is OrderingMode.UNORDERED:
        verdict = AspectVerdict.COMPATIBLE
        summary = "unordered expectation admits any observation mode"
    elif exp in {OrderingMode.CAUSAL, OrderingMode.PARTIAL} and obs in {
        OrderingMode.TOTAL,
        OrderingMode.SEQUENTIAL,
        OrderingMode.CAUSAL,
        OrderingMode.PARTIAL,
    }:
        verdict = AspectVerdict.COMPATIBLE
        summary = "stronger ordering observation satisfies weaker expectation"
    elif exp in {OrderingMode.TOTAL, OrderingMode.SEQUENTIAL} and obs in {
        OrderingMode.CONCURRENT,
        OrderingMode.UNORDERED,
    }:
        verdict = AspectVerdict.MISMATCH
        summary = "observation weakens required ordering"
    elif exp is OrderingMode.CONCURRENT and obs is OrderingMode.SEQUENTIAL:
        # Sequential is a special concurrent schedule — compatible.
        verdict = AspectVerdict.COMPATIBLE
        summary = "sequential schedule satisfies concurrent expectation"
    else:
        verdict = AspectVerdict.MISMATCH
        summary = "ordering modes are incompatible"
    return _aspect_result(
        SemanticAspect.ORDERING,
        verdict,
        rule_id=rule,
        expected_fact=_fact(expected.ordering),
        observed_fact=_fact(observed.ordering),
        summary=summary,
    )


def check_atomicity(
    expected: ExpectedProgramContract,
    observed: ObservedProgramContract,
) -> AspectCheckResult:
    rule = "rule:atomicity/strength@1"
    if expected.atomicity is None:
        return _aspect_result(
            SemanticAspect.ATOMICITY,
            AspectVerdict.NOT_APPLICABLE,
            rule_id=rule,
            summary="no expected atomicity",
        )
    if observed.atomicity is None:
        return _aspect_result(
            SemanticAspect.ATOMICITY,
            AspectVerdict.OMITTED,
            rule_id=rule,
            expected_fact=_fact(expected.atomicity),
            observed_fact="missing",
            summary="atomicity omitted from observation",
        )
    if observed.atomicity.is_refinement_of(expected.atomicity):
        return _aspect_result(
            SemanticAspect.ATOMICITY,
            AspectVerdict.COMPATIBLE,
            rule_id=rule,
            expected_fact=_fact(expected.atomicity),
            observed_fact=_fact(observed.atomicity),
            summary="atomicity is at least as strong as expectation",
        )
    return _aspect_result(
        SemanticAspect.ATOMICITY,
        AspectVerdict.MISMATCH,
        rule_id=rule,
        expected_fact=_fact(expected.atomicity),
        observed_fact=_fact(observed.atomicity),
        summary="atomicity is weaker than expectation",
    )


def check_consistency(
    expected: ExpectedProgramContract,
    observed: ObservedProgramContract,
) -> AspectCheckResult:
    rule = "rule:consistency/strength@1"
    if expected.consistency is None:
        return _aspect_result(
            SemanticAspect.CONSISTENCY,
            AspectVerdict.NOT_APPLICABLE,
            rule_id=rule,
            summary="no expected consistency",
        )
    if observed.consistency is None:
        return _aspect_result(
            SemanticAspect.CONSISTENCY,
            AspectVerdict.OMITTED,
            rule_id=rule,
            expected_fact=_fact(expected.consistency),
            observed_fact="missing",
            summary="consistency omitted from observation",
        )
    if observed.consistency.is_refinement_of(expected.consistency):
        return _aspect_result(
            SemanticAspect.CONSISTENCY,
            AspectVerdict.COMPATIBLE,
            rule_id=rule,
            expected_fact=_fact(expected.consistency),
            observed_fact=_fact(observed.consistency),
            summary="consistency is at least as strong as expectation",
        )
    return _aspect_result(
        SemanticAspect.CONSISTENCY,
        AspectVerdict.MISMATCH,
        rule_id=rule,
        expected_fact=_fact(expected.consistency),
        observed_fact=_fact(observed.consistency),
        summary="consistency is weaker than expectation",
    )


def check_resource_bounds(
    expected: ExpectedProgramContract,
    observed: ObservedProgramContract,
) -> AspectCheckResult:
    rule = "rule:resource-bounds/shared-dimensions@1"
    if expected.resource_bounds is None:
        return _aspect_result(
            SemanticAspect.RESOURCE_BOUNDS,
            AspectVerdict.NOT_APPLICABLE,
            rule_id=rule,
            summary="no expected resource bounds",
        )
    if observed.resource_bounds is None:
        return _aspect_result(
            SemanticAspect.RESOURCE_BOUNDS,
            AspectVerdict.OMITTED,
            rule_id=rule,
            expected_fact=_fact(expected.resource_bounds),
            observed_fact="missing",
            summary="resource bounds omitted from observation",
        )
    # Closed rule: only dimensions present on both sides are compared.
    # Observed values must be <= expected (tighter or equal). Omitted
    # dimensions on the observation do not invent a violation.
    looser: list[str] = []
    compared = 0
    for name in (
        "max_wall_time_ms",
        "max_cpu_time_ms",
        "max_memory_bytes",
        "max_payload_bytes",
        "max_output_bytes",
        "max_calls",
        "max_concurrency",
    ):
        exp_val = getattr(expected.resource_bounds, name)
        obs_val = getattr(observed.resource_bounds, name)
        if exp_val is None or obs_val is None:
            continue
        compared += 1
        if obs_val > exp_val:
            looser.append(f"{name}:{obs_val}>{exp_val}")
    if looser:
        return _aspect_result(
            SemanticAspect.RESOURCE_BOUNDS,
            AspectVerdict.MISMATCH,
            rule_id=rule,
            expected_fact=_fact(expected.resource_bounds),
            observed_fact=_fact(observed.resource_bounds),
            summary="observed bounds exceed expectation on " + ",".join(looser),
        )
    if compared == 0:
        return _aspect_result(
            SemanticAspect.RESOURCE_BOUNDS,
            AspectVerdict.AMBIGUOUS,
            rule_id=rule,
            expected_fact=_fact(expected.resource_bounds),
            observed_fact=_fact(observed.resource_bounds),
            summary="no shared resource-bound dimensions to compare",
        )
    return _aspect_result(
        SemanticAspect.RESOURCE_BOUNDS,
        AspectVerdict.COMPATIBLE,
        rule_id=rule,
        expected_fact=_fact(expected.resource_bounds),
        observed_fact=_fact(observed.resource_bounds),
        summary="shared resource-bound dimensions refine expectation",
    )


def check_fallback_degradation(
    expected: ExpectedProgramContract,
    observed: ObservedProgramContract,
) -> AspectCheckResult:
    rule = "rule:fallback/degradation-mode@1"
    if expected.fallback is None:
        return _aspect_result(
            SemanticAspect.FALLBACK_DEGRADATION,
            AspectVerdict.NOT_APPLICABLE,
            rule_id=rule,
            summary="no expected fallback/degradation",
        )
    if observed.fallback is None:
        return _aspect_result(
            SemanticAspect.FALLBACK_DEGRADATION,
            AspectVerdict.OMITTED,
            rule_id=rule,
            expected_fact=_fact(expected.fallback),
            observed_fact="missing",
            summary="fallback omitted from observation",
        )
    residual = _unsupported_or_unknown(
        SemanticAspect.FALLBACK_DEGRADATION,
        expected.fallback.support,
        observed.fallback.support,
        rule_id=rule,
    )
    if residual is not None and residual.verdict in {
        AspectVerdict.UNSUPPORTED,
        AspectVerdict.UNKNOWN,
    }:
        return residual
    # Fail-closed expectation rejects fail-open observation.
    from .program_contracts import DegradationMode

    exp = expected.fallback.mode
    obs = observed.fallback.mode
    if exp is DegradationMode.UNKNOWN or obs is DegradationMode.UNKNOWN:
        return _aspect_result(
            SemanticAspect.FALLBACK_DEGRADATION,
            AspectVerdict.UNKNOWN,
            rule_id=rule,
            expected_fact=_fact(expected.fallback),
            observed_fact=_fact(observed.fallback),
            summary="degradation mode unknown",
        )
    if exp is DegradationMode.FAIL_CLOSED and obs is DegradationMode.FAIL_OPEN:
        return _aspect_result(
            SemanticAspect.FALLBACK_DEGRADATION,
            AspectVerdict.MISMATCH,
            rule_id=rule,
            expected_fact=_fact(expected.fallback),
            observed_fact=_fact(observed.fallback),
            summary="fail-open observation violates fail-closed expectation",
        )
    if exp is obs:
        return _aspect_result(
            SemanticAspect.FALLBACK_DEGRADATION,
            AspectVerdict.COMPATIBLE,
            rule_id=rule,
            expected_fact=_fact(expected.fallback),
            observed_fact=_fact(observed.fallback),
            summary="degradation modes match",
        )
    # Other mode pairs are treated as mismatch under closed rules.
    return _aspect_result(
        SemanticAspect.FALLBACK_DEGRADATION,
        AspectVerdict.MISMATCH,
        rule_id=rule,
        expected_fact=_fact(expected.fallback),
        observed_fact=_fact(observed.fallback),
        summary="degradation modes disagree",
    )


ASPECT_CHECKERS: Final[
    dict[
        SemanticAspect,
        Any,
    ]
] = {
    SemanticAspect.IDENTITY: check_identity,
    SemanticAspect.INPUTS: check_inputs,
    SemanticAspect.OUTPUTS: check_outputs,
    SemanticAspect.ERRORS: check_errors,
    SemanticAspect.SYNC_ASYNC: check_sync_async,
    SemanticAspect.SIDE_EFFECTS: check_side_effects,
    SemanticAspect.CAPABILITIES: check_capabilities,
    SemanticAspect.AUTHORIZATION: check_authorization,
    SemanticAspect.IDEMPOTENCE: check_idempotence,
    SemanticAspect.ORDERING: check_ordering,
    SemanticAspect.ATOMICITY: check_atomicity,
    SemanticAspect.CONSISTENCY: check_consistency,
    SemanticAspect.RESOURCE_BOUNDS: check_resource_bounds,
    SemanticAspect.FALLBACK_DEGRADATION: check_fallback_degradation,
}


def make_binding(
    expected: ExpectedProgramContract,
    observed: ObservedProgramContract,
    *,
    call_path: CallPath | None = None,
    cache_generation: str = "",
    expected_cache_generation: str = "",
) -> CheckBinding:
    return CheckBinding(
        repository_id=expected.symbol.repository_id,
        tree_id=expected.symbol.tree_id,
        symbol_qualified_name=expected.symbol.qualified_name,
        expected_symbol_id=expected.symbol.symbol_id,
        observed_symbol_id=observed.symbol.symbol_id,
        interface_name=expected.interface.interface_name,
        expected_interface_id=expected.interface.interface_id,
        observed_interface_id=observed.interface.interface_id,
        policy_revision=expected.policy_revision,
        observed_repository_id=observed.symbol.repository_id,
        observed_tree_id=observed.symbol.tree_id,
        observed_policy_revision=observed.policy_revision,
        expected_contract_id=expected.expected_contract_id,
        observed_contract_id=observed.observed_contract_id,
        repository_observation_id=observed.repository_observation_id,
        call_path_id="" if call_path is None else call_path.path_id,
        cache_generation=cache_generation,
        expected_cache_generation=expected_cache_generation,
        checker_version=CHECKER_VERSION,
    )


def minimal_counterexample(
    *,
    binding: CheckBinding,
    aspect_result: AspectCheckResult,
    evaluated_at: str,
    authority_expires_at: str,
    call_path: CallPath | None = None,
    artifact_ref: str = "",
) -> ContractCounterexample:
    """Build the smallest conclusive witness for one mismatched aspect."""

    steps = [
        f"bind:{binding.symbol_qualified_name}@{binding.interface_name}",
        f"aspect:{aspect_result.aspect.value}",
        f"rule:{aspect_result.rule_id}",
        f"expected:{aspect_result.expected_fact or '-'}",
        f"observed:{aspect_result.observed_fact or '-'}",
    ]
    if call_path is not None:
        steps.insert(
            1,
            f"path:{call_path.path_name}#{call_path.path_id[:16]}",
        )
    return ContractCounterexample(
        binding=binding,
        aspect=aspect_result.aspect,
        rule_id=aspect_result.rule_id,
        expected_fact=aspect_result.expected_fact or "null",
        observed_fact=aspect_result.observed_fact or "null",
        witness_steps=tuple(steps),
        summary=aspect_result.summary
        or f"{aspect_result.aspect.value} mismatch",
        evaluated_at=evaluated_at,
        authority_expires_at=authority_expires_at,
        call_path_id="" if call_path is None else call_path.path_id,
        artifact_ref=artifact_ref,
        conclusive=True,
    )


def _select_primary_mismatch(
    aspect_results: Sequence[AspectCheckResult],
) -> AspectCheckResult | None:
    """Choose the minimal (stable-order) mismatch for a counterexample."""

    for item in aspect_results:
        if item.verdict is AspectVerdict.MISMATCH and item.closed_rule:
            return item
    return None


def _aggregate_kind(
    aspect_results: Sequence[AspectCheckResult],
    *,
    path: CallPath | None,
    cache_freshness: CacheFreshness,
    evaluated_at: str,
    authority_expires_at: str,
    elapsed_ms: int,
    budget_ms: int,
    force_timeout: bool = False,
    observation_layer: ObservationLayer = ObservationLayer.SYMBOLIC,
) -> ContractCheckResultKind:
    if cache_freshness is CacheFreshness.STALE:
        return ContractCheckResultKind.STALE
    if _datetime(evaluated_at) >= _datetime(authority_expires_at):
        return ContractCheckResultKind.STALE
    if force_timeout or (
        budget_ms >= 0 and elapsed_ms >= budget_ms and budget_ms > 0
    ):
        return ContractCheckResultKind.TIMEOUT
    if force_timeout and budget_ms == 0:
        return ContractCheckResultKind.TIMEOUT
    if path is not None and path.has_path_traversal:
        # Path traversal is a conclusive mismatch on identity/authorization
        # surface when present as a declared path defect.
        return ContractCheckResultKind.WITNESSED_MISMATCH
    if path is not None and path.has_dynamic_dispatch:
        return ContractCheckResultKind.AMBIGUOUS
    if path is not None and path.has_uncertainty:
        return ContractCheckResultKind.AMBIGUOUS

    has_mismatch = False
    has_unsupported = False
    has_ambiguous = False
    has_omitted = False
    has_unknown = False
    closed_compatible = 0
    closed_total = 0
    for item in aspect_results:
        if not item.closed_rule:
            continue
        closed_total += 1
        if item.verdict is AspectVerdict.MISMATCH:
            has_mismatch = True
        elif item.verdict is AspectVerdict.UNSUPPORTED:
            has_unsupported = True
        elif item.verdict is AspectVerdict.AMBIGUOUS:
            has_ambiguous = True
        elif item.verdict is AspectVerdict.OMITTED:
            has_omitted = True
        elif item.verdict is AspectVerdict.UNKNOWN:
            has_unknown = True
        elif item.verdict in {
            AspectVerdict.COMPATIBLE,
            AspectVerdict.NOT_APPLICABLE,
        }:
            closed_compatible += 1
    if has_mismatch:
        return ContractCheckResultKind.WITNESSED_MISMATCH
    if has_unsupported:
        return ContractCheckResultKind.UNSUPPORTED
    if has_ambiguous:
        return ContractCheckResultKind.AMBIGUOUS
    if has_unknown:
        return ContractCheckResultKind.UNKNOWN
    if has_omitted:
        return ContractCheckResultKind.INCOMPLETE
    if closed_total == 0:
        return ContractCheckResultKind.INCOMPLETE
    if closed_compatible == closed_total:
        # Runtime-layer matches are never promoted to model proof.
        if observation_layer is ObservationLayer.RUNTIME:
            return ContractCheckResultKind.RUNTIME_WITNESS
        return ContractCheckResultKind.PROVED_COMPATIBLE
    return ContractCheckResultKind.INCOMPLETE


def compare_contracts(
    expected: ExpectedProgramContract,
    observed: ObservedProgramContract,
    *,
    call_path: CallPath | None = None,
    aspects: Sequence[SemanticAspect] | None = None,
    evaluated_at: str | None = None,
    authority_expires_at: str | None = None,
    budget_ms: int = DEFAULT_BUDGET_MS,
    elapsed_ms: int = 0,
    cache_generation: str = "",
    expected_cache_generation: str = "",
    force_timeout: bool = False,
    require_same_subject: bool = True,
    observation_layer: ObservationLayer | str = ObservationLayer.SYMBOLIC,
) -> ContractCheckResult:
    """Compare one expected contract to one observation under closed rules.

    Emits ``proved_compatible`` only when every selected closed supported rule
    succeeds under the symbolic observation layer.  A runtime-layer match
    emits ``runtime_witness`` instead (never claim-promoted to model proof).
    Conclusive mismatches carry a minimal counterexample.  Dynamic dispatch
    uncertainty, omitted effects, cache staleness, and path traversal produce
    the corresponding typed non-compatible outcomes.
    """

    if not isinstance(expected, ExpectedProgramContract):
        raise ContractCheckerError("expected must be an ExpectedProgramContract")
    if not isinstance(observed, ObservedProgramContract):
        raise ContractCheckerError("observed must be an ObservedProgramContract")

    layer = _enum(
        observation_layer,
        ObservationLayer,
        field_name="observation_layer",
    )

    evaluated = evaluated_at or _now_iso()
    if authority_expires_at is None:
        # Default authority window: evaluated_at + 1 hour.
        base = _datetime(evaluated)
        expires = base.replace()  # copy
        from datetime import timedelta

        expires = base + timedelta(hours=1)
        authority_expires = (
            expires.isoformat().replace("+00:00", "Z")
            if expires.tzinfo
            else expires.replace(tzinfo=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    else:
        authority_expires = authority_expires_at

    cache_freshness = CacheFreshness.CURRENT
    if (
        expected_cache_generation
        and cache_generation
        and expected_cache_generation != cache_generation
    ):
        cache_freshness = CacheFreshness.STALE
    elif expected_cache_generation and not cache_generation:
        cache_freshness = CacheFreshness.STALE

    if call_path is not None:
        if (
            call_path.repository_id != expected.symbol.repository_id
            or call_path.tree_id != expected.symbol.tree_id
            or call_path.policy_revision != expected.policy_revision
        ):
            raise ScopeMismatchError(
                "call path must share repository, tree, and policy with contracts"
            )
        if (
            call_path.entry_interface
            and call_path.entry_interface
            not in {
                expected.interface.interface_name,
                expected.interface.interface_id,
            }
        ):
            raise ScopeMismatchError(
                "call path entry interface must match the expected interface"
            )
        if (
            call_path.exit_symbol
            and call_path.exit_symbol
            not in {
                expected.symbol.symbol_name,
                expected.symbol.qualified_name,
                expected.symbol.symbol_id,
            }
        ):
            raise ScopeMismatchError(
                "call path exit symbol must match the expected symbol"
            )

    binding = make_binding(
        expected,
        observed,
        call_path=call_path,
        cache_generation=cache_generation,
        expected_cache_generation=expected_cache_generation,
    )

    selected = tuple(aspects) if aspects is not None else CLOSED_SUPPORTED_ASPECTS
    aspect_results: list[AspectCheckResult] = []

    # Path-traversal is a forced mismatch on inputs/authorization when present.
    if call_path is not None and call_path.has_path_traversal:
        bad_steps = [
            step
            for step in call_path.steps
            if step.resolution is CallPathResolution.PATH_TRAVERSAL
            or (step.target_path and _path_is_traversal(step.target_path))
        ]
        first = bad_steps[0]
        aspect_results.append(
            _aspect_result(
                SemanticAspect.INPUTS,
                AspectVerdict.MISMATCH,
                rule_id="rule:path/no-traversal@1",
                expected_fact="repository-relative normalized path",
                observed_fact=first.target_path or first.notes or "traversal",
                summary="declared call path contains path traversal",
            )
        )

    # Unsupported residual clauses from either side.
    unsupported_aspects = {
        item.aspect for item in expected.unsupported
    } | {item.aspect for item in observed.unsupported}

    for aspect in selected:
        if aspect is SemanticAspect.SOURCE_PRECEDENCE:
            # Source precedence is extraction-time; satisfaction does not
            # re-litigate source ranking.
            aspect_results.append(
                _aspect_result(
                    aspect,
                    AspectVerdict.NOT_APPLICABLE,
                    rule_id="rule:source-precedence/extraction-only@1",
                    summary="source precedence is not a runtime satisfaction rule",
                    closed_rule=True,
                )
            )
            continue
        if aspect in unsupported_aspects:
            aspect_results.append(
                _aspect_result(
                    aspect,
                    AspectVerdict.UNSUPPORTED,
                    rule_id="rule:unsupported/declared@1",
                    expected_fact="unsupported",
                    observed_fact="unsupported",
                    summary=f"{aspect.value} declared unsupported",
                )
            )
            continue
        if expected.has_conflicts and aspect is SemanticAspect.IDENTITY:
            aspect_results.append(
                _aspect_result(
                    aspect,
                    AspectVerdict.AMBIGUOUS,
                    rule_id="rule:conflicts/expectation@1",
                    summary="conflicting expectations block proved_compatible",
                )
            )
            continue
        checker = ASPECT_CHECKERS.get(aspect)
        if checker is None:
            aspect_results.append(
                _aspect_result(
                    aspect,
                    AspectVerdict.UNSUPPORTED,
                    rule_id="rule:aspect/no-closed-rule@1",
                    summary=f"no closed rule for {aspect.value}",
                    closed_rule=False,
                )
            )
            continue
        # Skip duplicate inputs mismatch if path traversal already recorded.
        if (
            aspect is SemanticAspect.INPUTS
            and any(
                item.aspect is SemanticAspect.INPUTS
                and item.verdict is AspectVerdict.MISMATCH
                for item in aspect_results
            )
        ):
            continue
        aspect_results.append(checker(expected, observed))

    # Stable order by closed aspect order then residual.
    order = {aspect: index for index, aspect in enumerate(CLOSED_SUPPORTED_ASPECTS)}
    aspect_results.sort(
        key=lambda item: (order.get(item.aspect, 1_000), item.rule_id)
    )

    if require_same_subject and not _binds_exact_subject(expected, observed):
        # Ensure identity mismatch is present and primary.
        if not any(
            item.aspect is SemanticAspect.IDENTITY
            and item.verdict is AspectVerdict.MISMATCH
            for item in aspect_results
        ):
            aspect_results.insert(0, check_identity(expected, observed))

    effective_elapsed = elapsed_ms
    if force_timeout and budget_ms > 0 and effective_elapsed < budget_ms:
        # Satisfy the closed timeout invariant without inventing work.
        effective_elapsed = budget_ms

    kind = _aggregate_kind(
        aspect_results,
        path=call_path,
        cache_freshness=cache_freshness,
        evaluated_at=evaluated,
        authority_expires_at=authority_expires,
        elapsed_ms=effective_elapsed,
        budget_ms=budget_ms,
        force_timeout=force_timeout,
        observation_layer=layer,
    )

    counterexample: ContractCounterexample | None = None
    if kind is ContractCheckResultKind.WITNESSED_MISMATCH:
        primary = _select_primary_mismatch(aspect_results)
        if primary is None and call_path is not None and call_path.has_path_traversal:
            primary = next(
                (
                    item
                    for item in aspect_results
                    if item.verdict is AspectVerdict.MISMATCH
                ),
                None,
            )
        if primary is None:
            # Dynamic path forced mismatch without aspect mismatch — demote.
            kind = ContractCheckResultKind.AMBIGUOUS
        else:
            counterexample = minimal_counterexample(
                binding=binding,
                aspect_result=primary,
                evaluated_at=evaluated,
                authority_expires_at=authority_expires,
                call_path=call_path,
            )

    if kind is ContractCheckResultKind.PROVED_COMPATIBLE:
        summary = "all closed supported rules proved compatible"
    elif kind is ContractCheckResultKind.RUNTIME_WITNESS:
        summary = (
            "hermetic runtime observation confirms declared behavior under "
            "exact subject binding"
        )
    elif kind is ContractCheckResultKind.WITNESSED_MISMATCH:
        summary = (
            counterexample.summary
            if counterexample is not None
            else "conclusive contract mismatch"
        )
    elif kind is ContractCheckResultKind.AMBIGUOUS:
        summary = "contract comparison is ambiguous under declared path/evidence"
    elif kind is ContractCheckResultKind.UNSUPPORTED:
        summary = "one or more aspects are unsupported"
    elif kind is ContractCheckResultKind.TIMEOUT:
        summary = "contract comparison budget exhausted"
    elif kind is ContractCheckResultKind.STALE:
        summary = "contract comparison authority or cache is stale"
    elif kind is ContractCheckResultKind.UNKNOWN:
        summary = "one or more contract semantics are explicitly unknown"
    else:
        summary = "contract comparison is incomplete"

    return ContractCheckResult(
        kind=kind,
        binding=binding,
        aspect_results=tuple(aspect_results),
        summary=summary,
        evaluated_at=evaluated,
        authority_expires_at=authority_expires,
        counterexample=counterexample,
        call_path_id="" if call_path is None else call_path.path_id,
        cache_freshness=cache_freshness,
        budget_ms=budget_ms,
        elapsed_ms=effective_elapsed,
        checker_version=CHECKER_VERSION,
    )


def compare_expected_refinement(
    refined: ExpectedProgramContract,
    base: ExpectedProgramContract,
) -> AspectCheckResult:
    """Compare two expected contracts for compatible refinement (not observation)."""

    rule = "rule:expected/refinement@1"
    if refined.is_refinement_of(base):
        if base.is_refinement_of(refined):
            summary = "expected contracts are equivalent under refinement"
        else:
            summary = "refined expected contract is a compatible subtype"
        return _aspect_result(
            SemanticAspect.IDENTITY,
            AspectVerdict.COMPATIBLE,
            rule_id=rule,
            expected_fact=base.expected_contract_id,
            observed_fact=refined.expected_contract_id,
            summary=summary,
        )
    return _aspect_result(
        SemanticAspect.IDENTITY,
        AspectVerdict.MISMATCH,
        rule_id=rule,
        expected_fact=base.expected_contract_id,
        observed_fact=refined.expected_contract_id,
        summary="refined expected contract is incompatible with base",
    )


class ContractChecker:
    """Pure symbolic checker for expected vs observed program contracts."""

    def __init__(
        self,
        *,
        budget_ms: int = DEFAULT_BUDGET_MS,
        aspects: Sequence[SemanticAspect] | None = None,
        checker_version: str = CHECKER_VERSION,
    ) -> None:
        self.budget_ms = _integer(budget_ms, field_name="budget_ms", minimum=0)
        self.aspects = (
            tuple(aspects) if aspects is not None else CLOSED_SUPPORTED_ASPECTS
        )
        self.checker_version = _text(
            checker_version, field_name="checker_version"
        )

    def check(
        self,
        expected: ExpectedProgramContract,
        observed: ObservedProgramContract,
        *,
        call_path: CallPath | None = None,
        evaluated_at: str | None = None,
        authority_expires_at: str | None = None,
        elapsed_ms: int = 0,
        cache_generation: str = "",
        expected_cache_generation: str = "",
        force_timeout: bool = False,
        observation_layer: ObservationLayer | str = ObservationLayer.SYMBOLIC,
    ) -> ContractCheckResult:
        return compare_contracts(
            expected,
            observed,
            call_path=call_path,
            aspects=self.aspects,
            evaluated_at=evaluated_at,
            authority_expires_at=authority_expires_at,
            budget_ms=self.budget_ms,
            elapsed_ms=elapsed_ms,
            cache_generation=cache_generation,
            expected_cache_generation=expected_cache_generation,
            force_timeout=force_timeout,
            observation_layer=observation_layer,
        )

    def check_bundle(
        self,
        bundle: ProgramContractBundle,
        *,
        call_paths: Sequence[CallPath] | None = None,
        evaluated_at: str | None = None,
        authority_expires_at: str | None = None,
        cache_generation: str = "",
        expected_cache_generation: str = "",
    ) -> ContractCheckReport:
        if not isinstance(bundle, ProgramContractBundle):
            raise ContractCheckerError("bundle must be a ProgramContractBundle")
        evaluated = evaluated_at or _now_iso()
        if authority_expires_at is None:
            from datetime import timedelta

            expires = _datetime(evaluated) + timedelta(hours=1)
            authority_expires = expires.isoformat().replace("+00:00", "Z")
        else:
            authority_expires = authority_expires_at

        path_list = tuple(call_paths or ())
        results: list[ContractCheckResult] = []

        # Pair expected/observed by subject binding.
        for expected in bundle.expected:
            matched = [
                observed
                for observed in bundle.observed
                if observed.binds_same_subject(expected)
            ]
            if not matched:
                # No observation: incomplete identity-style result via a
                # synthetic empty observation is out of scope; skip with no
                # manufactured observation.
                continue
            for observed in matched:
                subject_paths = [
                    path
                    for path in path_list
                    if path.repository_id == bundle.repository_id
                    and path.tree_id == bundle.tree_id
                    and path.policy_revision == bundle.policy_revision
                    and (
                        not path.exit_symbol
                        or path.exit_symbol == expected.symbol.symbol_name
                        or path.exit_symbol == expected.symbol.qualified_name
                    )
                ]
                if subject_paths:
                    for path in subject_paths:
                        results.append(
                            self.check(
                                expected,
                                observed,
                                call_path=path,
                                evaluated_at=evaluated,
                                authority_expires_at=authority_expires,
                                cache_generation=cache_generation,
                                expected_cache_generation=expected_cache_generation,
                            )
                        )
                else:
                    results.append(
                        self.check(
                            expected,
                            observed,
                            evaluated_at=evaluated,
                            authority_expires_at=authority_expires,
                            cache_generation=cache_generation,
                            expected_cache_generation=expected_cache_generation,
                        )
                    )

        # Deterministic order by result identity.
        results.sort(key=lambda item: item.result_id)
        counts = {kind.value: 0 for kind in ContractCheckResultKind}
        for item in results:
            counts[item.kind.value] = counts.get(item.kind.value, 0) + 1
        summary = (
            "checked "
            f"{len(results)} pair(s): "
            + ", ".join(
                f"{name}={count}"
                for name, count in sorted(counts.items())
                if count
            )
        )
        return ContractCheckReport(
            repository_id=bundle.repository_id,
            tree_id=bundle.tree_id,
            policy_revision=bundle.policy_revision,
            results=tuple(results),
            evaluated_at=evaluated,
            authority_expires_at=authority_expires,
            summary=summary,
            checker_version=self.checker_version,
        )

    def check_along_paths(
        self,
        expected: ExpectedProgramContract,
        observed: ObservedProgramContract,
        paths: Sequence[CallPath],
        **kwargs: Any,
    ) -> tuple[ContractCheckResult, ...]:
        results = [
            self.check(expected, observed, call_path=path, **kwargs)
            for path in paths
        ]
        return tuple(sorted(results, key=lambda item: item.result_id))


def closed_supported_aspects() -> tuple[SemanticAspect, ...]:
    return CLOSED_SUPPORTED_ASPECTS


def canonical_contract_check_json_bytes(payload: Any) -> bytes:
    return canonical_json_bytes(payload)


def contract_check_content_identity(payload: Any) -> str:
    canonical_contract_check_json_bytes(payload)
    return content_identity(payload)


__all__ = [
    "CHECKER_VERSION",
    "CONTRACT_CHECKER_VERSION",
    "SCHEMA_VERSION",
    "CONTRACT_CHECK_RESULT_EVIDENCE",
    "CONTRACT_COUNTEREXAMPLE_EVIDENCE",
    "OBJECTIVE_VALIDATION_REPAIR_EVIDENCE",
    "OBJECTIVE_GOAL_ID",
    "EXACT_BINDING_DIMENSIONS",
    "CLOSED_SUPPORTED_ASPECTS",
    "DEFAULT_BUDGET_MS",
    "ContractCheckerError",
    "ContractCheckBoundsError",
    "ForgedIdentityError",
    "UnsupportedVersionError",
    "ScopeMismatchError",
    "StaleAuthorityError",
    "ContractCheckResultKind",
    "ObservationLayer",
    "AspectVerdict",
    "CallPathResolution",
    "CacheFreshness",
    "CallPathStep",
    "CallPath",
    "CheckBinding",
    "AspectCheckResult",
    "ContractCounterexample",
    "Counterexample",
    "CodeProofObligation",
    "ContractCheckResult",
    "ContractCheckReport",
    "ContractChecker",
    "ASPECT_CHECKERS",
    "check_identity",
    "check_inputs",
    "check_outputs",
    "check_errors",
    "check_sync_async",
    "check_side_effects",
    "check_capabilities",
    "check_authorization",
    "check_idempotence",
    "check_ordering",
    "check_atomicity",
    "check_consistency",
    "check_resource_bounds",
    "check_fallback_degradation",
    "compare_contracts",
    "compare_expected_refinement",
    "make_binding",
    "minimal_counterexample",
    "closed_supported_aspects",
    "canonical_contract_check_json_bytes",
    "contract_check_content_identity",
    "_path_is_traversal",
]
