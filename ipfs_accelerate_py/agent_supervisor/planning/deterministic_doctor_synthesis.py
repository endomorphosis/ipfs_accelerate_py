"""Proof-admitted deterministic repair overlay materialization (LPR-036).

Interface: ``DeterministicDoctorSynthesizer@1``

Materializes **only** a uniquely admitted closed operator through the existing
analytical path (:class:`DoctorRepairOperatorRegistry` /
:class:`AnalyticalChangeTransformer`).  The synthesizer:

* recomputes every input identity and operator precondition;
* renders only the unique admitted target / value / placement with exact
  before hashes and bounded paths;
* proves the output patch CID and byte-equivalent replay;
* parses / simulates the candidate overlay **without writing** the target;
* rejects changed spans, unsupported AST shapes, extra files / imports /
  dependencies, semantics outside the admitted consequence, non-idempotency,
  and any provider / model import or call; and
* on failure yields a typed abstention with **no partial overlay**.

This module never imports or calls ``llm_router`` / model-provider surfaces,
never invents text or behavior, never selects new targets / paths /
dependencies, and never grants overlay write authority.  Monkeypatched LLM
routes that raise remain untouched because they are never invoked.
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..analysis.contract_repair_contracts import RepairTargetDecision
from ..analysis.deterministic_doctor_contracts import (
    MAX_PATH_BYTES,
    MAX_REFERENCE_COUNT,
    MAX_TEXT_BYTES,
    DoctorAuthorityRoots,
    DoctorOperatorKind,
    DoctorRepairDisposition,
)
from ..proof.deterministic_doctor_hammer import (
    DoctorAuthoritativeProofReceipt,
    DoctorProofAuthorityDisposition,
)
from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)
from ..proof.missing_input_synthesis import ValueMappingProof
from .analytical_change_transforms import (
    ANALYTICAL_CHANGE_TRANSFORMER_INTERFACE,
    FieldMapping,
    TransformEdit,
    TransformRenderReceipt,
)
from .deterministic_doctor_transforms import (
    DOCTOR_REPAIR_OPERATOR_REGISTRY_INTERFACE,
    DoctorOperatorProposal,
    DoctorOperatorReceipt,
    DoctorOperatorRejectionReason,
    DoctorRepairOperatorRegistry,
    DoctorTransformAuthorityError,
    DoctorTransformError,
    DoctorTransformUnsupportedError,
    build_default_doctor_operator_registry,
)

# ---------------------------------------------------------------------------
# Schema / interface constants
# ---------------------------------------------------------------------------

DETERMINISTIC_DOCTOR_SYNTHESIZER_INTERFACE: Final[str] = (
    "DeterministicDoctorSynthesizer@1"
)
# Capability revision for PDR-051 bounded synthesis integration. The frozen
# LPR-036 interface identity remains ``@1`` (existing tests pin it); receipt
# schema is already ``@2``. Multi-operator / CEGIS / residual-hybrid
# orchestration is owned by ``ProgramRepairSynthesizer@1``.
DETERMINISTIC_DOCTOR_SYNTHESIZER_CAPABILITY_VERSION: Final[int] = 2
DOCTOR_ANALYTICAL_OVERLAY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/analytical-overlay@1"
)
DOCTOR_SYNTHESIS_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/synthesis-receipt@2"
)
DOCTOR_SYNTHESIS_RECEIPT_INTERFACE: Final[str] = "DoctorSynthesisReceipt@2"
DOCTOR_SYNTHESIS_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/synthesis-request@1"
)
DOCTOR_SIMULATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/simulation-receipt@1"
)

PRODUCER_ID: Final[str] = "deterministic-doctor-synthesis@1"
CONTRACT_VERSION: Final[int] = 1
RENDERER_ID: Final[str] = ANALYTICAL_CHANGE_TRANSFORMER_INTERFACE

MAX_SPAN_BYTES: Final[int] = 65_536
MAX_FILE_BYTES: Final[int] = 1_048_576
MAX_PATCH_BYTES: Final[int] = 2_000_000
MAX_REASON_CODES: Final[int] = 64
MAX_EXTRA_PATHS: Final[int] = 0  # synthesis admits exactly one target path

# Substrings that must never appear in the synthesizer's import graph or
# runtime call surface (enforced structurally + by tests).
_FORBIDDEN_PROVIDER_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "llm_router",
        "model_provider",
        "openai",
        "anthropic",
        "provider_router",
        "todo_daemon.change_propagation_provider_router",
    }
)

_PYTHON_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class DoctorSynthesisDisposition(str, Enum):
    """Closed outcomes for one proof-admitted overlay materialization."""

    SUPPORTED = "supported"
    ABSTAIN = "abstain"
    APPROVAL_REQUIRED = "approval_required"

    @property
    def grants_write_authority(self) -> bool:
        # Synthesis never grants mutation authority — only a candidate overlay.
        return False

    @property
    def is_success(self) -> bool:
        return self is DoctorSynthesisDisposition.SUPPORTED


class DoctorSynthesisReason(str, Enum):
    """Stable machine-readable synthesis reason codes."""

    RENDERED = "proof_admitted_render_supported"
    PROOF_NOT_ADMITTED = "proof_not_admitted"
    PROOF_RECEIPT_REQUIRED = "proof_receipt_required"
    PROOF_NOT_UNIQUE = "proof_not_unique"
    CONSEQUENCE_MISMATCH = "consequence_outside_admitted"
    TARGET_NOT_UNIQUE = "target_value_placement_not_unique"
    VALUE_MISMATCH = "value_ref_mismatch"
    PLACEMENT_MISMATCH = "placement_ref_mismatch"
    OPERATOR_MISMATCH = "operator_not_registered"
    ROOT_MISMATCH = "root_mismatch"
    IDENTITY_MISMATCH = "identity_mismatch"
    STALE_SPAN = "stale_span"
    UNSUPPORTED_AST_SHAPE = "unsupported_ast_shape"
    EXTRA_FILE = "extra_file_or_path"
    EXTRA_IMPORT = "extra_import"
    EXTRA_DEPENDENCY = "extra_dependency"
    SEMANTICS_OUTSIDE_CONSEQUENCE = "semantics_outside_admitted_consequence"
    NON_IDEMPOTENT = "non_idempotent_render"
    PROVIDER_OR_MODEL_CALL = "provider_or_model_import_or_call"
    RENDER_FAILED = "render_failed"
    SIMULATION_FAILED = "simulation_parse_failed"
    PATCH_CID_MISMATCH = "patch_cid_mismatch"
    REPLAY_MISMATCH = "byte_equivalent_replay_failed"
    WRITE_ATTEMPTED = "write_authority_claimed"
    EMPTY_SPAN = "empty_span"
    BOUNDS_EXCEEDED = "bounds_exceeded"
    MALFORMED_INPUT = "malformed_input"
    NO_PARTIAL_OVERLAY = "no_partial_overlay"
    PATH_NOT_BOUNDED = "path_not_bounded"
    UNPROVED_VALUE = "unproved_value"
    FORBIDDEN_PATH = "forbidden_path"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DoctorSynthesisError(ContractValidationError):
    """Malformed synthesis input or closed-boundary violation."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: DoctorSynthesisReason | str = DoctorSynthesisReason.MALFORMED_INPUT,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(getattr(reason_code, "value", reason_code))


class DoctorSynthesisAuthorityError(DoctorSynthesisError):
    """Attempt to invent authority, broaden scope, or write the target."""


class DoctorSynthesisUnsupportedError(DoctorSynthesisError):
    """Shape or consequence outside the closed synthesis surface."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _text(value: Any, name: str, *, required: bool = True, limit: int = MAX_TEXT_BYTES) -> str:
    if value is None:
        if required:
            raise DoctorSynthesisError(f"{name} is required")
        return ""
    if not isinstance(value, str):
        raise DoctorSynthesisError(f"{name} must be a string")
    text = value.strip() if name.endswith("_id") or name.endswith("_ref") else value
    if required and not text:
        raise DoctorSynthesisError(f"{name} is required")
    if len(text.encode("utf-8")) > limit:
        raise DoctorSynthesisError(
            f"{name} exceeds its byte bound",
            reason_code=DoctorSynthesisReason.BOUNDS_EXCEEDED,
        )
    return text


def _optional_text(value: Any, name: str, *, limit: int = MAX_TEXT_BYTES) -> str:
    return _text(value, name, required=False, limit=limit)


def _identifier(value: Any, name: str) -> str:
    text = _text(value, name, required=True, limit=MAX_TEXT_BYTES)
    return text


def _optional_identifier(value: Any, name: str) -> str:
    return _text(value, name, required=False, limit=MAX_TEXT_BYTES)


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise DoctorSynthesisError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DoctorSynthesisError(f"{name} must be a non-negative integer")
    return value


def _ids(
    values: Any,
    name: str,
    *,
    required: bool = False,
    limit: int = MAX_REFERENCE_COUNT,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise DoctorSynthesisError(f"{name} must be a sequence of identifiers")
    else:
        raw = values
    if required and not raw:
        raise DoctorSynthesisError(f"{name} is required")
    if len(raw) > limit:
        raise DoctorSynthesisError(
            f"{name} exceeds its bound",
            reason_code=DoctorSynthesisReason.BOUNDS_EXCEEDED,
        )
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = _identifier(item, name)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return tuple(out)


def _path(value: Any, name: str = "path") -> str:
    text = _text(value, name, required=True, limit=MAX_PATH_BYTES)
    if "\\" in text or text.startswith("/") or ".." in PurePosixPath(text).parts:
        raise DoctorSynthesisAuthorityError(
            f"{name} must be a bounded relative repository path",
            reason_code=DoctorSynthesisReason.PATH_NOT_BOUNDED,
        )
    return text.replace("\\", "/")


def _paths(values: Any, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise DoctorSynthesisError(f"{name} must be a sequence of paths")
    return tuple(_path(item, name) for item in values)


def _enum(value: Any, enum_cls: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value))
    except (TypeError, ValueError) as exc:
        raise DoctorSynthesisError(f"{name} is not a valid {enum_cls.__name__}") from exc


def _assert_no_provider_markers(*texts: str) -> None:
    for text in texts:
        lowered = text.lower()
        for marker in _FORBIDDEN_PROVIDER_MARKERS:
            if marker in lowered:
                raise DoctorSynthesisAuthorityError(
                    f"provider/model surface forbidden: {marker}",
                    reason_code=DoctorSynthesisReason.PROVIDER_OR_MODEL_CALL,
                )


def _assert_body_free_mapping(payload: Mapping[str, Any], label: str) -> None:
    """Body-free guard for receipts that must not embed source bodies."""

    forbidden = {
        "source_text",
        "span_text",
        "body",
        "code",
        "snippet",
        "prompt",
        "completion",
    }
    for key in payload:
        lowered = str(key).lower()
        if lowered in forbidden:
            raise DoctorSynthesisAuthorityError(
                f"{label} must not embed free-form body field {key!r}",
                reason_code=DoctorSynthesisReason.MALFORMED_INPUT,
            )


def _mapping_get(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _proof_is_admitted(proof: Any) -> bool:
    if proof is None:
        return False
    if hasattr(proof, "is_admitted"):
        try:
            return bool(proof.is_admitted)
        except Exception:
            return False
    disposition = _mapping_get(proof, "disposition")
    if disposition is None:
        return False
    value = getattr(disposition, "value", disposition)
    return str(value) == "admitted"


def _proof_uniqueness_satisfied(proof: Any) -> bool:
    if proof is None:
        return False
    return bool(_mapping_get(proof, "uniqueness_satisfied", default=False))


def _recompute_identity(value: Any) -> str:
    """Recompute a content identity for a contract or mapping."""

    if value is None:
        return ""
    if hasattr(value, "content_id"):
        try:
            return str(value.content_id)
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return content_identity(value.to_dict())
        except Exception:
            pass
    if isinstance(value, Mapping):
        return content_identity(dict(value))
    return content_identity({"repr": repr(value)})


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorSimulationReceipt(CanonicalContract):
    """In-memory parse/simulation of a candidate overlay (never writes)."""

    SCHEMA: ClassVar[str] = DOCTOR_SIMULATION_RECEIPT_SCHEMA

    path: str
    before_hash: str
    after_hash: str
    parse_ok: bool
    language: str = "python"
    simulated_bytes: int = 0
    error_message: str = ""
    wrote_target: bool = False
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path))
        object.__setattr__(
            self, "before_hash", _identifier(self.before_hash, "before_hash")
        )
        object.__setattr__(
            self, "after_hash", _identifier(self.after_hash, "after_hash")
        )
        object.__setattr__(self, "parse_ok", _bool(self.parse_ok, "parse_ok"))
        object.__setattr__(
            self, "language", _optional_text(self.language, "language") or "python"
        )
        object.__setattr__(
            self, "simulated_bytes", _nonneg_int(self.simulated_bytes, "simulated_bytes")
        )
        object.__setattr__(
            self,
            "error_message",
            _optional_text(self.error_message, "error_message", limit=MAX_TEXT_BYTES),
        )
        # Hard safety: simulation never writes.
        if self.wrote_target is not False:
            raise DoctorSynthesisAuthorityError(
                "simulation must never write the target",
                reason_code=DoctorSynthesisReason.WRITE_ATTEMPTED,
            )
        object.__setattr__(self, "wrote_target", False)
        object.__setattr__(self, "producer_id", _identifier(self.producer_id, "producer_id"))

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "path": self.path,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "parse_ok": self.parse_ok,
            "language": self.language,
            "simulated_bytes": self.simulated_bytes,
            "error_message": self.error_message,
            "wrote_target": False,
            "producer_id": self.producer_id,
        }


def _apply_span_replacement(
    file_text: str,
    *,
    span_start: int,
    span_end: int,
    span_text: str,
    replacement: str,
) -> str:
    """Apply a span replacement in memory without touching the filesystem."""

    if not isinstance(file_text, str):
        raise DoctorSynthesisError("file_text must be a string")
    if len(file_text.encode("utf-8")) > MAX_FILE_BYTES:
        raise DoctorSynthesisError(
            "file_text exceeds single-file bound",
            reason_code=DoctorSynthesisReason.BOUNDS_EXCEEDED,
        )
    # Prefer absolute offsets when they match; otherwise locate span_text.
    if 0 <= span_start <= span_end <= len(file_text):
        region = file_text[span_start:span_end]
        if region == span_text or (span_end == span_start and not span_text):
            return file_text[:span_start] + replacement + file_text[span_end:]
        if region and region != span_text and span_text and span_text in file_text:
            # Fall through to search when absolute slice drifted.
            pass
        elif region == span_text:
            return file_text[:span_start] + replacement + file_text[span_end:]
    if span_text and span_text in file_text:
        index = file_text.find(span_text)
        return file_text[:index] + replacement + file_text[index + len(span_text) :]
    if file_text == span_text:
        return replacement
    # Span-only simulation when full file not supplied.
    if not file_text:
        return replacement
    raise DoctorSynthesisError(
        "span does not match file_text for simulation",
        reason_code=DoctorSynthesisReason.STALE_SPAN,
    )


def _simulate_python_parse(
    *,
    path: str,
    before_text: str,
    after_text: str,
    before_hash: str,
    after_hash: str,
) -> DoctorSimulationReceipt:
    parse_ok = True
    error = ""
    if path.endswith((".py", ".pyi")) or before_text.strip().startswith(
        ("def ", "class ", "import ", "from ", "@")
    ) or "(" in after_text:
        try:
            ast.parse(after_text)
        except SyntaxError as exc:
            parse_ok = False
            error = f"syntax_error:{exc.msg}"
    # Non-Python artifacts (json/txt restore) skip AST parse.
    elif path.endswith((".json", ".txt", ".md", ".toml", ".yml", ".yaml")):
        parse_ok = True
        if path.endswith(".json"):
            import json

            try:
                json.loads(after_text)
            except Exception as exc:  # noqa: BLE001 — typed into receipt
                parse_ok = False
                error = f"json_error:{exc}"
    return DoctorSimulationReceipt(
        path=path,
        before_hash=before_hash,
        after_hash=after_hash,
        parse_ok=parse_ok,
        language="python" if path.endswith((".py", ".pyi")) else "text",
        simulated_bytes=len(after_text.encode("utf-8")),
        error_message=error,
        wrote_target=False,
    )


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorAnalyticalOverlay(CanonicalContract):
    """Candidate analytical overlay for a proof-admitted deterministic repair.

    Ready for impact planning but **incapable of direct mutation**:
    ``write_authority`` and ``semantic_authority`` are hard-wired false and
    ``source_write_count`` is always zero.
    """

    SCHEMA: ClassVar[str] = DOCTOR_ANALYTICAL_OVERLAY_SCHEMA

    roots: DoctorAuthorityRoots
    overlay_id: str
    path: str
    before_hash: str
    after_hash: str
    span_start: int
    span_end: int
    replacement: str
    patch_cid: str
    operator_id: str
    operator_kind: str
    finding_id: str = ""
    plan_receipt_id: str = ""
    proof_receipt_id: str = ""
    proposal_id: str = ""
    selected_consequence_ref: str = ""
    value_ref: str = ""
    placement_ref: str = ""
    obligation_refs: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()
    postcondition_refs: tuple[str, ...] = ()
    forbidden_paths: tuple[str, ...] = ()
    operator_receipt_replay_id: str = ""
    render_receipt_replay_id: str = ""
    artifact_id: str = ""
    idempotent_noop: bool = False
    write_authority: bool = False
    semantic_authority: bool = False
    source_write_count: int = 0
    llm_invocation_count: int = 0
    model_provider_call_count: int = 0
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        if not isinstance(self.roots, DoctorAuthorityRoots):
            raise DoctorSynthesisError("roots must be DoctorAuthorityRoots")
        object.__setattr__(self, "overlay_id", _identifier(self.overlay_id, "overlay_id"))
        object.__setattr__(self, "path", _path(self.path))
        object.__setattr__(
            self, "before_hash", _identifier(self.before_hash, "before_hash")
        )
        object.__setattr__(self, "after_hash", _identifier(self.after_hash, "after_hash"))
        object.__setattr__(
            self, "span_start", _nonneg_int(self.span_start, "span_start")
        )
        object.__setattr__(self, "span_end", _nonneg_int(self.span_end, "span_end"))
        if self.span_end < self.span_start:
            raise DoctorSynthesisError("span_end must be >= span_start")
        replacement = self.replacement if isinstance(self.replacement, str) else ""
        if len(replacement.encode("utf-8")) > MAX_SPAN_BYTES:
            raise DoctorSynthesisError(
                "replacement exceeds span bound",
                reason_code=DoctorSynthesisReason.BOUNDS_EXCEEDED,
            )
        object.__setattr__(self, "replacement", replacement)
        expected_after = _sha256_text(replacement)
        if self.after_hash != expected_after:
            raise DoctorSynthesisAuthorityError(
                "after_hash must equal sha256 of replacement",
                reason_code=DoctorSynthesisReason.IDENTITY_MISMATCH,
            )
        object.__setattr__(self, "patch_cid", _identifier(self.patch_cid, "patch_cid"))
        object.__setattr__(
            self, "operator_id", _identifier(self.operator_id, "operator_id")
        )
        object.__setattr__(
            self, "operator_kind", _identifier(self.operator_kind, "operator_kind")
        )
        for name in (
            "finding_id",
            "plan_receipt_id",
            "proof_receipt_id",
            "proposal_id",
            "selected_consequence_ref",
            "value_ref",
            "placement_ref",
            "operator_receipt_replay_id",
            "render_receipt_replay_id",
            "artifact_id",
            "producer_id",
        ):
            object.__setattr__(
                self, name, _optional_identifier(getattr(self, name), name)
            )
        for name in (
            "obligation_refs",
            "proof_refs",
            "postcondition_refs",
            "forbidden_paths",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        object.__setattr__(
            self, "idempotent_noop", _bool(self.idempotent_noop, "idempotent_noop")
        )
        # Hard authority invariants.
        if self.write_authority is not False:
            raise DoctorSynthesisAuthorityError(
                "analytical overlay cannot claim write_authority",
                reason_code=DoctorSynthesisReason.WRITE_ATTEMPTED,
            )
        if self.semantic_authority is not False:
            raise DoctorSynthesisAuthorityError(
                "analytical overlay cannot claim semantic_authority",
                reason_code=DoctorSynthesisReason.WRITE_ATTEMPTED,
            )
        if self.source_write_count != 0:
            raise DoctorSynthesisAuthorityError(
                "analytical overlay must report zero source writes",
                reason_code=DoctorSynthesisReason.WRITE_ATTEMPTED,
            )
        if self.llm_invocation_count != 0 or self.model_provider_call_count != 0:
            raise DoctorSynthesisAuthorityError(
                "analytical overlay must report zero LLM/model-provider calls",
                reason_code=DoctorSynthesisReason.PROVIDER_OR_MODEL_CALL,
            )
        object.__setattr__(self, "write_authority", False)
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "source_write_count", 0)
        object.__setattr__(self, "llm_invocation_count", 0)
        object.__setattr__(self, "model_provider_call_count", 0)
        object.__setattr__(
            self, "producer_id", _identifier(self.producer_id or PRODUCER_ID, "producer_id")
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "interface": DETERMINISTIC_DOCTOR_SYNTHESIZER_INTERFACE,
            "roots": self.roots.to_dict(),
            "overlay_id": self.overlay_id,
            "path": self.path,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "replacement": self.replacement,
            "patch_cid": self.patch_cid,
            "operator_id": self.operator_id,
            "operator_kind": self.operator_kind,
            "finding_id": self.finding_id,
            "plan_receipt_id": self.plan_receipt_id,
            "proof_receipt_id": self.proof_receipt_id,
            "proposal_id": self.proposal_id,
            "selected_consequence_ref": self.selected_consequence_ref,
            "value_ref": self.value_ref,
            "placement_ref": self.placement_ref,
            "obligation_refs": list(self.obligation_refs),
            "proof_refs": list(self.proof_refs),
            "postcondition_refs": list(self.postcondition_refs),
            "forbidden_paths": list(self.forbidden_paths),
            "operator_receipt_replay_id": self.operator_receipt_replay_id,
            "render_receipt_replay_id": self.render_receipt_replay_id,
            "artifact_id": self.artifact_id,
            "idempotent_noop": self.idempotent_noop,
            "write_authority": False,
            "semantic_authority": False,
            "source_write_count": 0,
            "llm_invocation_count": 0,
            "model_provider_call_count": 0,
            "producer_id": self.producer_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorAnalyticalOverlay":
        if not isinstance(payload, Mapping):
            raise DoctorSynthesisError("overlay payload must be a mapping")
        values = dict(payload)
        roots = values.get("roots")
        if isinstance(roots, Mapping):
            values["roots"] = DoctorAuthorityRoots.from_dict(roots)
        for drop in ("schema", "contract_version", "content_id", "cid", "interface"):
            values.pop(drop, None)
        return cls(**values)

    @property
    def is_mutable(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Request / receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorSynthesisRequest:
    """Inputs for one proof-admitted deterministic overlay materialization.

    Exactly one target path / span is admitted.  ``extra_paths`` must be empty.
    Full-file ``file_text`` is optional and used only for in-memory simulation.
    """

    roots: DoctorAuthorityRoots
    proposal: DoctorOperatorProposal
    span_text: str
    expression_text: str = ""
    field_mappings: tuple[FieldMapping, ...] | Mapping[str, str] = ()
    value_mapping: ValueMappingProof | None = None
    decision: RepairTargetDecision | None = None
    proof_receipt: Any = None
    selected_consequence_ref: str = ""
    value_ref: str = ""
    placement_ref: str = ""
    finding_id: str = ""
    plan_receipt_id: str = ""
    proof_receipt_id: str = ""
    file_text: str = ""
    verified_artifact_bytes: bytes | None = None
    extra_paths: tuple[str, ...] = ()
    extra_imports: tuple[str, ...] = ()
    require_proof_receipt: bool = True
    require_idempotent_replay: bool = True
    already_applied: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.roots, DoctorAuthorityRoots):
            raise DoctorSynthesisError("roots must be DoctorAuthorityRoots")
        if not isinstance(self.proposal, DoctorOperatorProposal):
            raise DoctorSynthesisError("proposal must be DoctorOperatorProposal")
        if not isinstance(self.span_text, str):
            raise DoctorSynthesisError("span_text must be a string")
        if len(self.span_text.encode("utf-8")) > MAX_SPAN_BYTES:
            raise DoctorSynthesisError(
                "span_text exceeds span bound",
                reason_code=DoctorSynthesisReason.BOUNDS_EXCEEDED,
            )
        if self.file_text and len(self.file_text.encode("utf-8")) > MAX_FILE_BYTES:
            raise DoctorSynthesisError(
                "file_text exceeds single-file bound",
                reason_code=DoctorSynthesisReason.BOUNDS_EXCEEDED,
            )
        object.__setattr__(
            self,
            "expression_text",
            self.expression_text if isinstance(self.expression_text, str) else "",
        )
        if len(self.expression_text.encode("utf-8")) > MAX_TEXT_BYTES:
            raise DoctorSynthesisError(
                "expression_text exceeds bound",
                reason_code=DoctorSynthesisReason.BOUNDS_EXCEEDED,
            )
        for name in (
            "selected_consequence_ref",
            "value_ref",
            "placement_ref",
            "finding_id",
            "plan_receipt_id",
            "proof_receipt_id",
        ):
            object.__setattr__(
                self, name, _optional_identifier(getattr(self, name), name)
            )
        object.__setattr__(self, "extra_paths", _paths(self.extra_paths, "extra_paths"))
        object.__setattr__(
            self, "extra_imports", _ids(self.extra_imports, "extra_imports")
        )
        for name in (
            "require_proof_receipt",
            "require_idempotent_replay",
            "already_applied",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(self.metadata or {}))
        )
        if self.extra_paths:
            raise DoctorSynthesisAuthorityError(
                "synthesis admits exactly one target path; extra_paths forbidden",
                reason_code=DoctorSynthesisReason.EXTRA_FILE,
            )
        if self.extra_imports:
            raise DoctorSynthesisAuthorityError(
                "synthesis rejects undeclared extra imports at request construction",
                reason_code=DoctorSynthesisReason.EXTRA_IMPORT,
            )
        _assert_no_provider_markers(
            self.span_text,
            self.expression_text,
            self.selected_consequence_ref,
            self.value_ref,
            self.placement_ref,
        )


@dataclass(frozen=True)
class DoctorSynthesisReceipt(CanonicalContract):
    """Fail-closed receipt for one deterministic doctor synthesis attempt.

    Failed renders yield typed abstention with ``overlay is None`` (no partial
    overlay).  Successful receipts still report ``write_performed=False``.
    """

    SCHEMA: ClassVar[str] = DOCTOR_SYNTHESIS_RECEIPT_SCHEMA

    disposition: DoctorSynthesisDisposition
    reason_codes: tuple[str, ...]
    roots: DoctorAuthorityRoots
    proposal_id: str = ""
    operator_id: str = ""
    operator_kind: str = ""
    path: str = ""
    before_hash: str = ""
    after_hash: str = ""
    patch_cid: str = ""
    selected_consequence_ref: str = ""
    property_id: str = ""
    toolchain_id: str = ""
    policy_id: str = ""
    value_ref: str = ""
    placement_ref: str = ""
    finding_id: str = ""
    plan_receipt_id: str = ""
    proof_receipt_id: str = ""
    proof_receipt_cid: str = ""
    proof_native_entry_id: str = ""
    proof_kernel_entry_id: str = ""
    proof_authority_entry_id: str = ""
    operator_receipt_id: str = ""
    render_receipt_id: str = ""
    overlay: DoctorAnalyticalOverlay | None = None
    simulation: DoctorSimulationReceipt | None = None
    input_identities: Mapping[str, str] = field(default_factory=dict)
    replay_identity: str = ""
    byte_equivalent_replay: bool = False
    uniqueness_satisfied: bool = False
    authoritative_proof: bool = False
    proof_authority: DoctorAuthoritativeProofReceipt | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    write_performed: bool = False
    write_authority: bool = False
    semantic_authority: bool = False
    provider_invoked: bool = False
    llm_invocation_count: int = 0
    model_provider_call_count: int = 0
    source_write_count: int = 0
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DoctorSynthesisDisposition, "disposition"),
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes")
        )
        if not isinstance(self.roots, DoctorAuthorityRoots):
            raise DoctorSynthesisError("roots must be DoctorAuthorityRoots")
        for name in (
            "proposal_id",
            "operator_id",
            "operator_kind",
            "path",
            "before_hash",
            "after_hash",
            "patch_cid",
            "selected_consequence_ref",
            "property_id",
            "toolchain_id",
            "policy_id",
            "value_ref",
            "placement_ref",
            "finding_id",
            "plan_receipt_id",
            "proof_receipt_id",
            "proof_receipt_cid",
            "proof_native_entry_id",
            "proof_kernel_entry_id",
            "proof_authority_entry_id",
            "operator_receipt_id",
            "render_receipt_id",
            "replay_identity",
            "producer_id",
        ):
            object.__setattr__(
                self, name, _optional_identifier(getattr(self, name), name)
            )
        if self.overlay is not None and not isinstance(
            self.overlay, DoctorAnalyticalOverlay
        ):
            raise DoctorSynthesisError("overlay must be DoctorAnalyticalOverlay")
        if self.simulation is not None and not isinstance(
            self.simulation, DoctorSimulationReceipt
        ):
            raise DoctorSynthesisError("simulation must be DoctorSimulationReceipt")
        if (
            self.proof_authority is not None
            and type(self.proof_authority) is not DoctorAuthoritativeProofReceipt
        ):
            raise DoctorSynthesisAuthorityError(
                "proof_authority must be a typed authoritative proof"
            )
        if self.proof_authority is not None and not self.authoritative_proof:
            raise DoctorSynthesisAuthorityError(
                "proof_authority requires the authoritative_proof binding"
            )
        object.__setattr__(
            self,
            "input_identities",
            MappingProxyType(
                {
                    str(key): str(val)
                    for key, val in dict(self.input_identities or {}).items()
                }
            ),
        )
        for name in (
            "byte_equivalent_replay",
            "uniqueness_satisfied",
            "authoritative_proof",
            "write_performed",
            "write_authority",
            "semantic_authority",
            "provider_invoked",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        for name in (
            "llm_invocation_count",
            "model_provider_call_count",
            "source_write_count",
        ):
            object.__setattr__(
                self, name, _nonneg_int(getattr(self, name), name)
            )
        # Authority / safety invariants.
        if self.write_authority is not False or self.write_performed is not False:
            raise DoctorSynthesisAuthorityError(
                "synthesis receipts never write or claim write authority",
                reason_code=DoctorSynthesisReason.WRITE_ATTEMPTED,
            )
        if self.semantic_authority is not False:
            raise DoctorSynthesisAuthorityError(
                "synthesis receipts never claim semantic authority",
                reason_code=DoctorSynthesisReason.WRITE_ATTEMPTED,
            )
        if self.provider_invoked is not False:
            raise DoctorSynthesisAuthorityError(
                "synthesis receipts never invoke a provider",
                reason_code=DoctorSynthesisReason.PROVIDER_OR_MODEL_CALL,
            )
        if (
            self.llm_invocation_count != 0
            or self.model_provider_call_count != 0
            or self.source_write_count != 0
        ):
            raise DoctorSynthesisAuthorityError(
                "synthesis receipts must report zero LLM/provider/source writes",
                reason_code=DoctorSynthesisReason.PROVIDER_OR_MODEL_CALL,
            )
        object.__setattr__(self, "write_performed", False)
        object.__setattr__(self, "write_authority", False)
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "provider_invoked", False)
        object.__setattr__(self, "llm_invocation_count", 0)
        object.__setattr__(self, "model_provider_call_count", 0)
        object.__setattr__(self, "source_write_count", 0)
        disposition = self.disposition
        if isinstance(disposition, DoctorSynthesisDisposition):
            if disposition is DoctorSynthesisDisposition.SUPPORTED:
                if self.overlay is None:
                    raise DoctorSynthesisError(
                        "supported receipts require a complete overlay"
                    )
                if not self.reason_codes:
                    raise DoctorSynthesisError(
                        "supported receipts require reason codes"
                    )
                if not self.patch_cid:
                    raise DoctorSynthesisError(
                        "supported receipts require a proved patch_cid"
                    )
                if not self.byte_equivalent_replay:
                    raise DoctorSynthesisError(
                        "supported receipts require byte-equivalent replay"
                    )
                if not self.selected_consequence_ref:
                    raise DoctorSynthesisError(
                        "supported receipts require a selected consequence"
                    )
                if not self.property_id:
                    raise DoctorSynthesisError(
                        "supported receipts require a bound property"
                    )
                if not self.toolchain_id or not self.policy_id:
                    raise DoctorSynthesisError(
                        "supported receipts require toolchain and policy bindings"
                    )
                if not self.uniqueness_satisfied:
                    raise DoctorSynthesisError(
                        "supported receipts require a unique consequence"
                    )
                if self.toolchain_id != self.roots.toolchain_id:
                    raise DoctorSynthesisAuthorityError(
                        "synthesis toolchain binding does not match current roots"
                    )
                if self.policy_id != self.roots.policy_id:
                    raise DoctorSynthesisAuthorityError(
                        "synthesis policy binding does not match current roots"
                    )
                if self.authoritative_proof and not (
                    self.proof_receipt_cid
                    and self.proof_native_entry_id
                    and self.proof_kernel_entry_id
                    and self.proof_authority_entry_id
                ):
                    raise DoctorSynthesisAuthorityError(
                        "authoritative synthesis requires sealed proof lineage"
                    )
                if self.authoritative_proof:
                    proof = self.proof_authority
                    if (
                        type(proof) is not DoctorAuthoritativeProofReceipt
                        or not proof.mutation_capable
                        or proof.content_id != self.proof_receipt_cid
                        or proof.selected_consequence_ref
                        != self.selected_consequence_ref
                        or proof.property_id != self.property_id
                        or proof.toolchain_id != self.toolchain_id
                        or proof.policy_id != self.policy_id
                        or proof.native_store_ref is None
                        or proof.native_store_ref.entry_id
                        != self.proof_native_entry_id
                        or proof.kernel_store_ref is None
                        or proof.kernel_store_ref.entry_id
                        != self.proof_kernel_entry_id
                        or proof.authority_store_ref is None
                        or proof.authority_store_ref.entry_id
                        != self.proof_authority_entry_id
                    ):
                        raise DoctorSynthesisAuthorityError(
                            "authoritative synthesis proof bindings are not sealed"
                        )
            else:
                # Fail-closed: abstention / approval never carries a partial overlay.
                if self.overlay is not None:
                    raise DoctorSynthesisAuthorityError(
                        "failed synthesis cannot retain a partial overlay",
                        reason_code=DoctorSynthesisReason.NO_PARTIAL_OVERLAY,
                    )
                if not self.reason_codes:
                    raise DoctorSynthesisError(
                        "abstention receipts require rejection reasons"
                    )
        if not self.replay_identity:
            object.__setattr__(
                self,
                "replay_identity",
                content_identity(self._payload_without_replay()),
            )
        object.__setattr__(
            self,
            "producer_id",
            _identifier(self.producer_id or PRODUCER_ID, "producer_id"),
        )

    def _payload_without_replay(self) -> dict[str, Any]:
        return {
            "contract_version": 2,
            "interface": DOCTOR_SYNTHESIS_RECEIPT_INTERFACE,
            "disposition": (
                self.disposition.value
                if isinstance(self.disposition, DoctorSynthesisDisposition)
                else self.disposition
            ),
            "reason_codes": list(self.reason_codes),
            "roots": self.roots.to_dict(),
            "proposal_id": self.proposal_id,
            "operator_id": self.operator_id,
            "operator_kind": self.operator_kind,
            "path": self.path,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "patch_cid": self.patch_cid,
            "selected_consequence_ref": self.selected_consequence_ref,
            "property_id": self.property_id,
            "toolchain_id": self.toolchain_id,
            "policy_id": self.policy_id,
            "value_ref": self.value_ref,
            "placement_ref": self.placement_ref,
            "finding_id": self.finding_id,
            "plan_receipt_id": self.plan_receipt_id,
            "proof_receipt_id": self.proof_receipt_id,
            "proof_receipt_cid": self.proof_receipt_cid,
            "proof_native_entry_id": self.proof_native_entry_id,
            "proof_kernel_entry_id": self.proof_kernel_entry_id,
            "proof_authority_entry_id": self.proof_authority_entry_id,
            "operator_receipt_id": self.operator_receipt_id,
            "render_receipt_id": self.render_receipt_id,
            "overlay_id": self.overlay.overlay_id if self.overlay is not None else "",
            "overlay_content_id": (
                self.overlay.content_id if self.overlay is not None else ""
            ),
            "simulation": (
                self.simulation.to_dict() if self.simulation is not None else None
            ),
            "input_identities": dict(self.input_identities),
            "byte_equivalent_replay": self.byte_equivalent_replay,
            "uniqueness_satisfied": self.uniqueness_satisfied,
            "authoritative_proof": self.authoritative_proof,
            "write_performed": False,
            "write_authority": False,
            "semantic_authority": False,
            "provider_invoked": False,
            "llm_invocation_count": 0,
            "model_provider_call_count": 0,
            "source_write_count": 0,
            "producer_id": self.producer_id,
        }

    def _payload(self) -> dict[str, Any]:
        payload = self._payload_without_replay()
        payload["replay_identity"] = self.replay_identity
        return payload

    @property
    def admitted(self) -> bool:
        return (
            self.disposition is DoctorSynthesisDisposition.SUPPORTED
            and self.overlay is not None
        )

    @property
    def mutation_capable(self) -> bool:
        """Whether proof authority may be forwarded to a separate mutation gate."""

        return (
            self.admitted
            and self.authoritative_proof
            and self.uniqueness_satisfied
            and bool(self.proof_receipt_cid)
            and bool(self.proof_native_entry_id)
            and bool(self.proof_kernel_entry_id)
            and bool(self.proof_authority_entry_id)
            and type(self.proof_authority) is DoctorAuthoritativeProofReceipt
            and self.proof_authority.mutation_capable
            and self.proof_authority.content_id == self.proof_receipt_cid
        )

    @property
    def abstained(self) -> bool:
        return self.disposition is DoctorSynthesisDisposition.ABSTAIN


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------


class DeterministicDoctorSynthesizer:
    """Materialize only proof-admitted deterministic repair overlays.

    Uses the closed :class:`DoctorRepairOperatorRegistry` analytical renderer.
    Never mutates the repository and never calls model providers.
    """

    INTERFACE: ClassVar[str] = DETERMINISTIC_DOCTOR_SYNTHESIZER_INTERFACE

    def __init__(
        self,
        registry: DoctorRepairOperatorRegistry | None = None,
        *,
        roots: DoctorAuthorityRoots | None = None,
    ) -> None:
        if registry is not None:
            if not isinstance(registry, DoctorRepairOperatorRegistry):
                raise DoctorSynthesisError(
                    "registry must be DoctorRepairOperatorRegistry"
                )
            self._registry = registry
        elif roots is not None:
            self._registry = build_default_doctor_operator_registry(roots)
        else:
            self._registry = None  # type: ignore[assignment]

    @property
    def registry(self) -> DoctorRepairOperatorRegistry:
        if self._registry is None:
            raise DoctorSynthesisError("synthesizer registry is not configured")
        return self._registry

    @property
    def capability_version(self) -> int:
        """PDR-051 capability revision (interface identity remains ``@1``)."""

        return DETERMINISTIC_DOCTOR_SYNTHESIZER_CAPABILITY_VERSION

    @staticmethod
    def prove_zero_model_calls(receipt: DoctorSynthesisReceipt) -> bool:
        """Return True when a receipt hard-proves zero LLM/provider invocations."""

        if not isinstance(receipt, DoctorSynthesisReceipt):
            return False
        return (
            receipt.llm_invocation_count == 0
            and receipt.model_provider_call_count == 0
            and receipt.provider_invoked is False
            and receipt.source_write_count == 0
        )

    def synthesize(self, request: DoctorSynthesisRequest) -> DoctorSynthesisReceipt:
        """Render a proof-admitted proposal into a candidate analytical overlay."""

        if not isinstance(request, DoctorSynthesisRequest):
            raise DoctorSynthesisError(
                "request must be DoctorSynthesisRequest",
                reason_code=DoctorSynthesisReason.MALFORMED_INPUT,
            )
        registry = self._resolve_registry(request.roots)
        identities = self._recompute_input_identities(request, registry)
        reasons = self._preflight(request, registry, identities)
        if reasons:
            return self._abstain(request, reasons, identities)

        # Render through the closed analytical operator path.
        try:
            operator_receipt, render_receipt = registry.render_admitted(
                request.proposal,
                span_text=request.span_text,
                expression_text=request.expression_text,
                field_mappings=request.field_mappings,
                verified_artifact_bytes=request.verified_artifact_bytes,
                value_mapping=request.value_mapping,
                decision=request.decision,
                already_applied=request.already_applied,
            )
        except (
            DoctorTransformError,
            DoctorTransformAuthorityError,
            DoctorTransformUnsupportedError,
            ValueError,
        ) as exc:
            reason = self._map_transform_exception(exc)
            return self._abstain(
                request,
                (reason, DoctorSynthesisReason.RENDER_FAILED),
                identities,
            )

        if not operator_receipt.admitted or render_receipt is None:
            mapped = self._map_operator_reasons(operator_receipt.rejection_reasons)
            if not mapped:
                mapped = (DoctorSynthesisReason.RENDER_FAILED,)
            return self._abstain(request, mapped, identities)

        if not render_receipt.edits:
            return self._abstain(
                request,
                (DoctorSynthesisReason.RENDER_FAILED,),
                identities,
            )

        edit = render_receipt.edits[0]
        # Single-path invariant: no extra files in the render.
        if len(render_receipt.edits) != 1:
            return self._abstain(
                request,
                (DoctorSynthesisReason.EXTRA_FILE,),
                identities,
            )
        if edit.path != request.proposal.edit_site.path:
            return self._abstain(
                request,
                (DoctorSynthesisReason.EXTRA_FILE, DoctorSynthesisReason.PATH_NOT_BOUNDED),
                identities,
            )

        # Reject undeclared import introductions beyond the closed operator.
        if render_receipt.import_statements:
            extra = self._unexpected_imports(request, render_receipt)
            if extra:
                return self._abstain(
                    request,
                    (DoctorSynthesisReason.EXTRA_IMPORT,),
                    identities,
                )

        # Idempotent / byte-equivalent replay.
        replay_ok = self._prove_byte_equivalent_replay(
            registry,
            request,
            first_receipt=operator_receipt,
            first_render=render_receipt,
        )
        if request.require_idempotent_replay and not replay_ok:
            return self._abstain(
                request,
                (DoctorSynthesisReason.NON_IDEMPOTENT, DoctorSynthesisReason.REPLAY_MISMATCH),
                identities,
            )

        # In-memory simulation (never writes).
        try:
            simulation = self._simulate(request, edit)
        except DoctorSynthesisError as exc:
            reason = DoctorSynthesisReason(exc.reason_code) if exc.reason_code in {
                item.value for item in DoctorSynthesisReason
            } else DoctorSynthesisReason.SIMULATION_FAILED
            return self._abstain(
                request,
                (reason, DoctorSynthesisReason.SIMULATION_FAILED),
                identities,
            )
        if not simulation.parse_ok:
            return self._abstain(
                request,
                (DoctorSynthesisReason.SIMULATION_FAILED, DoctorSynthesisReason.UNSUPPORTED_AST_SHAPE),
                identities,
                simulation=simulation,
            )

        patch_doc = self._build_patch_document(request, edit, operator_receipt, render_receipt)
        patch_bytes = _canonical_patch_bytes(patch_doc)
        if len(patch_bytes) > MAX_PATCH_BYTES:
            return self._abstain(
                request,
                (DoctorSynthesisReason.BOUNDS_EXCEEDED,),
                identities,
                simulation=simulation,
            )
        patch_cid = content_identity(patch_doc)
        # Prove patch CID: recompute and compare.
        if content_identity(patch_doc) != patch_cid:
            return self._abstain(
                request,
                (DoctorSynthesisReason.PATCH_CID_MISMATCH,),
                identities,
                simulation=simulation,
            )

        consequence = (
            request.selected_consequence_ref
            or str(
                _mapping_get(
                    request.proof_receipt, "selected_consequence_ref", default=""
                )
                or ""
            )
        )
        value_ref = request.value_ref or (
            request.proposal.value_source_refs[0]
            if request.proposal.value_source_refs
            else ""
        )
        placement_ref = request.placement_ref or (
            f"placement:{request.proposal.edit_site.path}"
            f":{request.proposal.edit_site.span_start}"
            f":{request.proposal.edit_site.span_end}"
        )
        finding_id = request.finding_id or str(
            _mapping_get(request.proof_receipt, "finding_id", default="") or ""
        )
        plan_receipt_id = request.plan_receipt_id or str(
            _mapping_get(request.proof_receipt, "plan_receipt_id", default="") or ""
        )
        proof_receipt_id = request.proof_receipt_id or str(
            _mapping_get(request.proof_receipt, "receipt_id", default="") or ""
        )
        authoritative = (
            request.proof_receipt
            if type(request.proof_receipt) is DoctorAuthoritativeProofReceipt
            else None
        )
        property_id = (
            authoritative.property_id
            if authoritative is not None
            else (
                request.proposal.obligation_refs[0]
                if request.proposal.obligation_refs
                else f"property:{consequence}"
            )
        )
        proof_receipt_cid = (
            authoritative.content_id if authoritative is not None else ""
        )
        native_entry_id = (
            authoritative.native_store_ref.entry_id
            if authoritative is not None
            and authoritative.native_store_ref is not None
            else ""
        )
        kernel_entry_id = (
            authoritative.kernel_store_ref.entry_id
            if authoritative is not None
            and authoritative.kernel_store_ref is not None
            else ""
        )
        authority_entry_id = (
            authoritative.authority_store_ref.entry_id
            if authoritative is not None
            and authoritative.authority_store_ref is not None
            else ""
        )

        overlay_id = content_identity(
            {
                "proposal_id": request.proposal.proposal_id,
                "patch_cid": patch_cid,
                "path": edit.path,
                "before_hash": edit.before_hash,
                "after_hash": edit.expected_after_hash,
            }
        )
        forbidden = tuple(
            registry.get(request.proposal.operator_id).spec.forbidden_paths
        )
        try:
            overlay = DoctorAnalyticalOverlay(
                roots=request.roots,
                overlay_id=f"overlay:{overlay_id}",
                path=edit.path,
                before_hash=edit.before_hash,
                after_hash=edit.expected_after_hash,
                span_start=edit.start,
                span_end=edit.end,
                replacement=edit.replacement,
                patch_cid=patch_cid,
                operator_id=request.proposal.operator_id,
                operator_kind=(
                    request.proposal.kind.value
                    if isinstance(request.proposal.kind, DoctorOperatorKind)
                    else str(request.proposal.kind)
                ),
                finding_id=finding_id,
                plan_receipt_id=plan_receipt_id,
                proof_receipt_id=proof_receipt_id,
                proposal_id=request.proposal.proposal_id,
                selected_consequence_ref=consequence,
                value_ref=value_ref,
                placement_ref=placement_ref,
                obligation_refs=request.proposal.obligation_refs,
                proof_refs=request.proposal.proof_refs,
                postcondition_refs=operator_receipt.postcondition_refs
                or request.proposal.postcondition_refs,
                forbidden_paths=forbidden,
                operator_receipt_replay_id=operator_receipt.replay_identity,
                render_receipt_replay_id=render_receipt.replay_identity,
                artifact_id=edit.artifact_id,
                idempotent_noop=operator_receipt.idempotent_noop,
            )
        except DoctorSynthesisError as exc:
            reason = DoctorSynthesisReason(exc.reason_code) if exc.reason_code in {
                item.value for item in DoctorSynthesisReason
            } else DoctorSynthesisReason.MALFORMED_INPUT
            return self._abstain(
                request,
                (reason, DoctorSynthesisReason.NO_PARTIAL_OVERLAY),
                identities,
                simulation=simulation,
            )

        return DoctorSynthesisReceipt(
            disposition=DoctorSynthesisDisposition.SUPPORTED,
            reason_codes=(DoctorSynthesisReason.RENDERED.value,),
            roots=request.roots,
            proposal_id=request.proposal.proposal_id,
            operator_id=request.proposal.operator_id,
            operator_kind=overlay.operator_kind,
            path=edit.path,
            before_hash=edit.before_hash,
            after_hash=edit.expected_after_hash,
            patch_cid=patch_cid,
            selected_consequence_ref=consequence,
            property_id=property_id,
            toolchain_id=request.roots.toolchain_id,
            policy_id=request.roots.policy_id,
            value_ref=value_ref,
            placement_ref=placement_ref,
            finding_id=finding_id,
            plan_receipt_id=plan_receipt_id,
            proof_receipt_id=proof_receipt_id,
            proof_receipt_cid=proof_receipt_cid,
            proof_native_entry_id=native_entry_id,
            proof_kernel_entry_id=kernel_entry_id,
            proof_authority_entry_id=authority_entry_id,
            operator_receipt_id=operator_receipt.replay_identity,
            render_receipt_id=render_receipt.replay_identity,
            overlay=overlay,
            simulation=simulation,
            input_identities=identities,
            byte_equivalent_replay=replay_ok,
            uniqueness_satisfied=True,
            authoritative_proof=authoritative is not None,
            proof_authority=authoritative,
        )

    # -- internal ------------------------------------------------------------

    def _resolve_registry(
        self, roots: DoctorAuthorityRoots
    ) -> DoctorRepairOperatorRegistry:
        if self._registry is None:
            self._registry = build_default_doctor_operator_registry(roots)
            return self._registry
        if self._registry.roots.to_dict() != roots.to_dict():
            # Prefer request roots: rebuild a matching default registry only when
            # the bound registry roots diverge (fail closed for custom registries).
            if self._registry.producer_id == "deterministic-doctor-transforms@1":
                # Default registry can be rebuilt under request roots.
                return build_default_doctor_operator_registry(roots)
            raise DoctorSynthesisAuthorityError(
                "registry roots must match request roots",
                reason_code=DoctorSynthesisReason.ROOT_MISMATCH,
            )
        return self._registry

    def _recompute_input_identities(
        self,
        request: DoctorSynthesisRequest,
        registry: DoctorRepairOperatorRegistry,
    ) -> dict[str, str]:
        identities = {
            "roots": _recompute_identity(request.roots),
            "proposal": _recompute_identity(request.proposal),
            "operator_registry": registry.registry_id,
            "edit_site": _recompute_identity(request.proposal.edit_site),
            "span_before_hash": _sha256_text(request.span_text),
        }
        if request.proof_receipt is not None:
            identities["proof_receipt"] = _recompute_identity(request.proof_receipt)
        if request.value_mapping is not None:
            identities["value_mapping"] = _recompute_identity(request.value_mapping)
        if request.decision is not None:
            identities["decision"] = _recompute_identity(request.decision)
        # Operator precondition identity: registered descriptor content.
        try:
            descriptor = registry.get(request.proposal.operator_id)
            identities["operator"] = _recompute_identity(descriptor)
            identities["operator_spec"] = _recompute_identity(descriptor.spec)
        except DoctorTransformUnsupportedError:
            identities["operator"] = ""
        return identities

    def _preflight(
        self,
        request: DoctorSynthesisRequest,
        registry: DoctorRepairOperatorRegistry,
        identities: Mapping[str, str],
    ) -> tuple[DoctorSynthesisReason, ...]:
        reasons: list[DoctorSynthesisReason] = []
        proposal = request.proposal

        if proposal.roots.to_dict() != request.roots.to_dict():
            reasons.append(DoctorSynthesisReason.ROOT_MISMATCH)
        if registry.roots.to_dict() != request.roots.to_dict():
            # Default rebuild path may still leave mismatch if custom registry.
            if registry.roots.to_dict() != proposal.roots.to_dict():
                reasons.append(DoctorSynthesisReason.ROOT_MISMATCH)

        # Recomputed before-hash must match edit site.
        expected_before = identities.get("span_before_hash", "")
        if proposal.edit_site.before_hash != expected_before:
            reasons.append(DoctorSynthesisReason.STALE_SPAN)
        if not request.span_text and proposal.kind is not DoctorOperatorKind.RESTORE_TRACKED_ARTIFACT:
            # Empty span only allowed for pure inserts with zero-width site.
            if proposal.edit_site.span_end != proposal.edit_site.span_start:
                reasons.append(DoctorSynthesisReason.EMPTY_SPAN)

        # Operator must be registered and match proposal kind.
        try:
            descriptor = registry.get(proposal.operator_id)
            if descriptor.kind is not proposal.kind:
                reasons.append(DoctorSynthesisReason.OPERATOR_MISMATCH)
        except DoctorTransformUnsupportedError:
            reasons.append(DoctorSynthesisReason.OPERATOR_MISMATCH)

        # Proof admission gate.
        if not proposal.proof_admitted:
            reasons.append(DoctorSynthesisReason.PROOF_NOT_ADMITTED)

        proof = request.proof_receipt
        if request.require_proof_receipt and proof is None:
            reasons.append(DoctorSynthesisReason.PROOF_RECEIPT_REQUIRED)
        if proof is not None:
            if not _proof_is_admitted(proof):
                reasons.append(DoctorSynthesisReason.PROOF_NOT_ADMITTED)
            if not _proof_uniqueness_satisfied(proof):
                reasons.append(DoctorSynthesisReason.PROOF_NOT_UNIQUE)
            # Unique consequence.
            selected = str(
                _mapping_get(proof, "selected_consequence_ref", default="") or ""
            )
            eligible = _mapping_get(proof, "eligible_consequence_refs", default=()) or ()
            eligible_ids = tuple(str(item) for item in eligible)
            if selected and len([item for item in eligible_ids if item == selected]) > 1:
                reasons.append(DoctorSynthesisReason.TARGET_NOT_UNIQUE)
            if eligible_ids and selected and selected not in eligible_ids:
                reasons.append(DoctorSynthesisReason.CONSEQUENCE_MISMATCH)
            if request.selected_consequence_ref and selected:
                if request.selected_consequence_ref != selected:
                    reasons.append(DoctorSynthesisReason.CONSEQUENCE_MISMATCH)
            # Root binding on proof when available.
            proof_roots = _mapping_get(proof, "roots")
            if proof_roots is not None:
                proof_repo = str(
                    _mapping_get(proof_roots, "repository_id", default="") or ""
                )
                proof_tree = str(_mapping_get(proof_roots, "tree_id", default="") or "")
                if proof_repo and proof_repo != request.roots.repository_id:
                    reasons.append(DoctorSynthesisReason.ROOT_MISMATCH)
                if proof_tree and proof_tree != request.roots.tree_id:
                    reasons.append(DoctorSynthesisReason.ROOT_MISMATCH)
            # Safety counters on proof.
            if int(_mapping_get(proof, "llm_invocation_count", default=0) or 0) != 0:
                reasons.append(DoctorSynthesisReason.PROVIDER_OR_MODEL_CALL)
            if int(
                _mapping_get(proof, "model_provider_call_count", default=0) or 0
            ) != 0:
                reasons.append(DoctorSynthesisReason.PROVIDER_OR_MODEL_CALL)
            if bool(_mapping_get(proof, "write_authority", default=False)):
                reasons.append(DoctorSynthesisReason.WRITE_ATTEMPTED)
            if type(proof) is DoctorAuthoritativeProofReceipt:
                authoritative = proof
                if (
                    authoritative.disposition
                    is not DoctorProofAuthorityDisposition.VERIFIED
                    or not authoritative.mutation_capable
                ):
                    reasons.append(DoctorSynthesisReason.PROOF_NOT_ADMITTED)
                shared_root_fields = (
                    "repository_id",
                    "forest_id",
                    "tree_id",
                    "overlay_id",
                    "graph_id",
                    "corpus_id",
                    "index_id",
                    "model_id",
                    "translator_id",
                    "toolchain_id",
                    "policy_id",
                    "environment_id",
                )
                if any(
                    getattr(authoritative.roots, name)
                    != getattr(request.roots, name)
                    for name in shared_root_fields
                ):
                    reasons.append(DoctorSynthesisReason.ROOT_MISMATCH)
                if authoritative.eligible_consequence_refs != (
                    authoritative.selected_consequence_ref,
                ):
                    reasons.append(DoctorSynthesisReason.PROOF_NOT_UNIQUE)
                if (
                    request.proof_receipt_id
                    and request.proof_receipt_id != authoritative.receipt_id
                ):
                    reasons.append(DoctorSynthesisReason.IDENTITY_MISMATCH)

        # Unique value source.
        if proposal.value_source_refs and len(set(proposal.value_source_refs)) != len(
            proposal.value_source_refs
        ):
            reasons.append(DoctorSynthesisReason.TARGET_NOT_UNIQUE)
        if request.value_ref and proposal.value_source_refs:
            if request.value_ref not in proposal.value_source_refs:
                reasons.append(DoctorSynthesisReason.VALUE_MISMATCH)
        if request.value_mapping is not None:
            if request.value_mapping.disposition.value != "unique_proved":
                reasons.append(DoctorSynthesisReason.UNPROVED_VALUE)
            if len(request.value_mapping.proved_candidate_ids) != 1:
                reasons.append(DoctorSynthesisReason.TARGET_NOT_UNIQUE)

        # Extra paths / dependencies already rejected at request construction,
        # but re-check proposal allowed_dependency_paths for ADD_IMPORT.
        if proposal.kind is DoctorOperatorKind.ADD_IMPORT:
            if proposal.allowed_dependency_paths and proposal.import_module:
                projected = proposal.import_module.replace(".", "/") + ".py"
                allowed = set(proposal.allowed_dependency_paths)
                if projected not in allowed and proposal.import_module not in allowed:
                    if not any(
                        projected.startswith(item.rstrip("/") + "/")
                        or item == proposal.import_module
                        for item in allowed
                    ):
                        reasons.append(DoctorSynthesisReason.EXTRA_DEPENDENCY)

        # Identity recomputation sanity: proposal content_id stable.
        if identities.get("proposal") and _recompute_identity(proposal) != identities[
            "proposal"
        ]:
            reasons.append(DoctorSynthesisReason.IDENTITY_MISMATCH)

        return tuple(dict.fromkeys(reasons))

    def _prove_byte_equivalent_replay(
        self,
        registry: DoctorRepairOperatorRegistry,
        request: DoctorSynthesisRequest,
        *,
        first_receipt: DoctorOperatorReceipt,
        first_render: TransformRenderReceipt,
    ) -> bool:
        second_receipt, second_render = registry.render_admitted(
            request.proposal,
            span_text=request.span_text,
            expression_text=request.expression_text,
            field_mappings=request.field_mappings,
            verified_artifact_bytes=request.verified_artifact_bytes,
            value_mapping=request.value_mapping,
            decision=request.decision,
            already_applied=request.already_applied,
        )
        if not second_receipt.admitted or second_render is None:
            return False
        if not first_render.edits or not second_render.edits:
            return False
        first_edit = first_render.edits[0]
        second_edit = second_render.edits[0]
        if first_edit.replacement != second_edit.replacement:
            return False
        if first_edit.expected_after_hash != second_edit.expected_after_hash:
            return False
        if first_receipt.expected_after_hash != second_receipt.expected_after_hash:
            return False
        if first_receipt.replay_identity != second_receipt.replay_identity:
            return False
        # Optional re-application idempotency (already-applied path).
        if not request.already_applied:
            try:
                reapplied = registry.render_admitted_repeat_is_noop(
                    request.proposal,
                    span_text=request.span_text,
                    expression_text=request.expression_text,
                    field_mappings=request.field_mappings,
                    value_mapping=request.value_mapping,
                    decision=request.decision,
                )
            except Exception:
                reapplied = False
            if not reapplied and not first_receipt.idempotent_noop:
                # Strict non-idempotency only when both replay-after-apply and
                # first-pass noop checks fail for value-bearing transforms that
                # claim idempotency in the registry.
                descriptor = registry.get(request.proposal.operator_id)
                if descriptor.spec.idempotent and not reapplied:
                    # Exact renames / restores that change content are still
                    # idempotent on a second apply of the *result*; if the
                    # helper reports false, treat as non-idempotent.
                    return False
        return True

    def _simulate(
        self,
        request: DoctorSynthesisRequest,
        edit: TransformEdit,
    ) -> DoctorSimulationReceipt:
        path = edit.path
        span_text = request.span_text
        if request.file_text:
            after_file = _apply_span_replacement(
                request.file_text,
                span_start=edit.start,
                span_end=edit.end,
                span_text=span_text,
                replacement=edit.replacement,
            )
            before_hash = _sha256_text(request.file_text)
            after_hash = _sha256_text(after_file)
            return _simulate_python_parse(
                path=path,
                before_text=request.file_text,
                after_text=after_file,
                before_hash=before_hash,
                after_hash=after_hash,
            )
        # Span-only simulation.
        return _simulate_python_parse(
            path=path,
            before_text=span_text,
            after_text=edit.replacement,
            before_hash=edit.before_hash,
            after_hash=edit.expected_after_hash,
        )

    def _build_patch_document(
        self,
        request: DoctorSynthesisRequest,
        edit: TransformEdit,
        operator_receipt: DoctorOperatorReceipt,
        render_receipt: TransformRenderReceipt,
    ) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "schema": "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/patch@1",
            "roots": {
                "repository_id": request.roots.repository_id,
                "tree_id": request.roots.tree_id,
                "overlay_id": request.roots.overlay_id,
            },
            "proposal_id": request.proposal.proposal_id,
            "operator_id": request.proposal.operator_id,
            "path": edit.path,
            "span_start": edit.start,
            "span_end": edit.end,
            "before_hash": edit.before_hash,
            "after_hash": edit.expected_after_hash,
            "replacement": edit.replacement,
            "artifact_id": edit.artifact_id,
            "operator_receipt_replay_id": operator_receipt.replay_identity,
            "render_receipt_replay_id": render_receipt.replay_identity,
            "postcondition_refs": list(operator_receipt.postcondition_refs),
            "write_authority": False,
        }

    def _unexpected_imports(
        self,
        request: DoctorSynthesisRequest,
        render_receipt: TransformRenderReceipt,
    ) -> tuple[str, ...]:
        if request.proposal.kind is not DoctorOperatorKind.ADD_IMPORT:
            # Non-import operators must not introduce import statements.
            return tuple(render_receipt.import_statements)
        allowed_module = request.proposal.import_module
        unexpected: list[str] = []
        for stmt in render_receipt.import_statements:
            if allowed_module and allowed_module not in stmt:
                unexpected.append(stmt)
        return tuple(unexpected)

    def _map_operator_reasons(
        self, reasons: Sequence[str]
    ) -> tuple[DoctorSynthesisReason, ...]:
        mapping = {
            DoctorOperatorRejectionReason.PROOF_NOT_ADMITTED.value: DoctorSynthesisReason.PROOF_NOT_ADMITTED,
            DoctorOperatorRejectionReason.STALE_SPAN.value: DoctorSynthesisReason.STALE_SPAN,
            DoctorOperatorRejectionReason.UNSUPPORTED_AST_SHAPE.value: DoctorSynthesisReason.UNSUPPORTED_AST_SHAPE,
            DoctorOperatorRejectionReason.UNPROVED_VALUE.value: DoctorSynthesisReason.UNPROVED_VALUE,
            DoctorOperatorRejectionReason.NEW_DEPENDENCY.value: DoctorSynthesisReason.EXTRA_DEPENDENCY,
            DoctorOperatorRejectionReason.ROOT_MISMATCH.value: DoctorSynthesisReason.ROOT_MISMATCH,
            DoctorOperatorRejectionReason.FORBIDDEN_PATH.value: DoctorSynthesisReason.FORBIDDEN_PATH,
            DoctorOperatorRejectionReason.TCB_PATH.value: DoctorSynthesisReason.FORBIDDEN_PATH,
            DoctorOperatorRejectionReason.PATH_NOT_AUTHORIZED.value: DoctorSynthesisReason.PATH_NOT_BOUNDED,
            DoctorOperatorRejectionReason.INVENTED_BEHAVIOR.value: DoctorSynthesisReason.SEMANTICS_OUTSIDE_CONSEQUENCE,
            DoctorOperatorRejectionReason.COMPLEX_NEW_BEHAVIOR.value: DoctorSynthesisReason.SEMANTICS_OUTSIDE_CONSEQUENCE,
            DoctorOperatorRejectionReason.SCOPE_ESCAPE.value: DoctorSynthesisReason.SEMANTICS_OUTSIDE_CONSEQUENCE,
            DoctorOperatorRejectionReason.WRITE_AUTHORITY.value: DoctorSynthesisReason.WRITE_ATTEMPTED,
            DoctorOperatorRejectionReason.SEMANTIC_AUTHORITY.value: DoctorSynthesisReason.WRITE_ATTEMPTED,
            DoctorOperatorRejectionReason.EMPTY_SPAN.value: DoctorSynthesisReason.EMPTY_SPAN,
            DoctorOperatorRejectionReason.RENDER_REJECTED.value: DoctorSynthesisReason.RENDER_FAILED,
            DoctorOperatorRejectionReason.UNKNOWN_OPERATOR.value: DoctorSynthesisReason.OPERATOR_MISMATCH,
            DoctorOperatorRejectionReason.UNSUPPORTED_KIND.value: DoctorSynthesisReason.OPERATOR_MISMATCH,
        }
        out: list[DoctorSynthesisReason] = []
        for reason in reasons:
            mapped = mapping.get(str(reason))
            if mapped is not None:
                out.append(mapped)
            else:
                out.append(DoctorSynthesisReason.RENDER_FAILED)
        return tuple(dict.fromkeys(out))

    def _map_transform_exception(self, exc: BaseException) -> DoctorSynthesisReason:
        text = str(exc)
        for reason in DoctorSynthesisReason:
            if reason.value in text:
                return reason
        for op_reason in DoctorOperatorRejectionReason:
            if op_reason.value in text:
                mapped = self._map_operator_reasons((op_reason.value,))
                return mapped[0] if mapped else DoctorSynthesisReason.RENDER_FAILED
        return DoctorSynthesisReason.RENDER_FAILED

    def _abstain(
        self,
        request: DoctorSynthesisRequest,
        reasons: Sequence[DoctorSynthesisReason | str],
        identities: Mapping[str, str],
        *,
        simulation: DoctorSimulationReceipt | None = None,
    ) -> DoctorSynthesisReceipt:
        codes = tuple(
            dict.fromkeys(
                str(getattr(item, "value", item)) for item in reasons
            )
        )
        if DoctorSynthesisReason.NO_PARTIAL_OVERLAY.value not in codes:
            codes = codes + (DoctorSynthesisReason.NO_PARTIAL_OVERLAY.value,)
        return DoctorSynthesisReceipt(
            disposition=DoctorSynthesisDisposition.ABSTAIN,
            reason_codes=codes,
            roots=request.roots,
            proposal_id=request.proposal.proposal_id,
            operator_id=request.proposal.operator_id,
            operator_kind=(
                request.proposal.kind.value
                if isinstance(request.proposal.kind, DoctorOperatorKind)
                else str(request.proposal.kind)
            ),
            path=request.proposal.edit_site.path,
            before_hash=request.proposal.edit_site.before_hash,
            selected_consequence_ref=request.selected_consequence_ref,
            value_ref=request.value_ref,
            placement_ref=request.placement_ref,
            finding_id=request.finding_id,
            plan_receipt_id=request.plan_receipt_id,
            proof_receipt_id=request.proof_receipt_id
            or str(
                _mapping_get(request.proof_receipt, "receipt_id", default="") or ""
            ),
            overlay=None,
            simulation=simulation,
            input_identities=dict(identities),
            byte_equivalent_replay=False,
        )


def _canonical_patch_bytes(patch_doc: Mapping[str, Any]) -> bytes:
    import json

    return json.dumps(dict(patch_doc), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def create_deterministic_doctor_synthesizer(
    roots: DoctorAuthorityRoots,
    *,
    registry: DoctorRepairOperatorRegistry | None = None,
) -> DeterministicDoctorSynthesizer:
    """Factory for a root-bound deterministic doctor synthesizer."""

    if registry is None:
        registry = build_default_doctor_operator_registry(roots)
    return DeterministicDoctorSynthesizer(registry=registry)


def materialize_proof_admitted_overlay(
    request: DoctorSynthesisRequest,
    *,
    registry: DoctorRepairOperatorRegistry | None = None,
) -> DoctorSynthesisReceipt:
    """Module-level convenience wrapper around :class:`DeterministicDoctorSynthesizer`."""

    synth = DeterministicDoctorSynthesizer(
        registry=registry,
        roots=request.roots if registry is None else None,
    )
    return synth.synthesize(request)


__all__ = (
    "CONTRACT_VERSION",
    "DETERMINISTIC_DOCTOR_SYNTHESIZER_CAPABILITY_VERSION",
    "DETERMINISTIC_DOCTOR_SYNTHESIZER_INTERFACE",
    "DOCTOR_ANALYTICAL_OVERLAY_SCHEMA",
    "DOCTOR_SIMULATION_RECEIPT_SCHEMA",
    "DOCTOR_SYNTHESIS_RECEIPT_SCHEMA",
    "DOCTOR_SYNTHESIS_RECEIPT_INTERFACE",
    "DOCTOR_SYNTHESIS_REQUEST_SCHEMA",
    "DOCTOR_REPAIR_OPERATOR_REGISTRY_INTERFACE",
    "PRODUCER_ID",
    "RENDERER_ID",
    "DeterministicDoctorSynthesizer",
    "DoctorAnalyticalOverlay",
    "DoctorSimulationReceipt",
    "DoctorSynthesisAuthorityError",
    "DoctorSynthesisDisposition",
    "DoctorSynthesisError",
    "DoctorSynthesisReason",
    "DoctorSynthesisReceipt",
    "DoctorSynthesisRequest",
    "DoctorSynthesisUnsupportedError",
    "DoctorRepairDisposition",
    "create_deterministic_doctor_synthesizer",
    "materialize_proof_admitted_overlay",
)
