"""Allowlisted deterministic AST repair transforms for the doctor (LPR-033).

Interface: ``DoctorRepairOperatorRegistry@1``

Adapts the closed :class:`AnalyticalChangeTransformer` surface into an
immutable, root-bound operator registry.  Every registered operator declares:

* closed input/output type refs and supported language/AST shapes;
* typed preconditions, semantic postconditions, and frame conditions;
* exact read/write path constraints, value-source requirements, and
  placement/forbidden-path sets;
* a deterministic byte-stable renderer identity;
* idempotency plus an inverse or compensation reference.

Proposals remain **body-free** until an explicit proof admission gate is
satisfied.  The registry never invents values, never escalates to a model,
never mutates the repository, and never grants write authority.

Fail-closed rejections cover splats, ambiguous overloads, reflection, monkey
patches, stale spans, incomplete mappings, unproved values, generated/native/
FFI/unsafe/concurrency targets, forbidden/TCB/cross-root writes, and complex
new behavior.  Repeating an already-applied transform is a no-op (identical
replacement) or a deterministic rejection.
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..analysis.change_propagation_contracts import (
    PropagationAuthorityRoots,
    TransformKind,
)
from ..analysis.contract_repair_contracts import RepairTargetDecision
from ..analysis.deterministic_doctor_contracts import (
    DOCTOR_TCB_PATH_MARKERS,
    MAX_OPERATOR_COUNT,
    MAX_PATH_BYTES,
    MAX_REFERENCE_COUNT,
    MAX_TEXT_BYTES,
    DoctorApprovalClass,
    DoctorAuthorityRoots,
    DoctorEditSite,
    DoctorOperatorKind,
    DoctorRepairDisposition,
    DoctorRepairOperatorSpec,
    is_doctor_tcb_path,
)
from ..proof.formal_verification_contracts import (
    CanonicalContract,
    content_identity,
)
from ..proof.missing_input_synthesis import ValueMappingProof
from .analytical_change_transforms import (
    ANALYTICAL_CHANGE_TRANSFORMER_INTERFACE,
    AnalyticalChangeTransformAuthorityError,
    AnalyticalChangeTransformError,
    AnalyticalChangeTransformUnsupportedError,
    AnalyticalChangeTransformer,
    FieldMapping,
    TransformRejectionReason,
    TransformRenderReceipt,
    TransformSite,
    TransformSourceSpan,
    make_span,
    render_analytical_transform,
)


# ---------------------------------------------------------------------------
# Schema / interface constants
# ---------------------------------------------------------------------------

DOCTOR_REPAIR_OPERATOR_REGISTRY_INTERFACE: Final[str] = "DoctorRepairOperatorRegistry@1"
DOCTOR_OPERATOR_PROPOSAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/operator-proposal@1"
)
DOCTOR_OPERATOR_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/operator-receipt@1"
)
DOCTOR_OPERATOR_DESCRIPTOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/operator-descriptor@1"
)
DOCTOR_REPAIR_OPERATOR_REGISTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor/operator-registry@1"
)
PRODUCER_ID: Final[str] = "deterministic-doctor-transforms@1"
CONTRACT_VERSION: Final[int] = 1
RENDERER_ID: Final[str] = ANALYTICAL_CHANGE_TRANSFORMER_INTERFACE

MAX_AST_SHAPES: Final[int] = 32
MAX_TYPE_REFS: Final[int] = 64
MAX_SPAN_BYTES: Final[int] = 65_536
MAX_EXPRESSION_BYTES: Final[int] = 4_096
MAX_PROPOSALS: Final[int] = 256

_PYTHON_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SIMPLE_ATTR_EXPR: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)
_MODULE_PATH: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)

# Closed reject markers for free-form / reflective / concurrent targets.
_FORBIDDEN_TARGET_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "reflection",
        "reflect",
        "monkey_patch",
        "monkeypatch",
        "setattr",
        "getattr",
        "globals",
        "locals",
        "eval",
        "exec",
        "compile",
        "__import__",
        "importlib",
        "ctypes",
        "cffi",
        "cffi_",
        "native",
        "ffi",
        "unsafe",
        "concurrency",
        "threading",
        "multiprocessing",
        "asyncio",
        "generated",
        "codegen",
        "template",
        "jinja",
        "shell",
        "subprocess",
        "os.system",
    }
)

_DEFAULT_FORBIDDEN_PATHS: Final[tuple[str, ...]] = tuple(
    sorted(
        {
            marker.rstrip("/")
            for marker in DOCTOR_TCB_PATH_MARKERS
            if not marker.endswith("_") and not marker.endswith(".py")
        }
        | {
            "ipfs_accelerate_py/agent_supervisor/proof",
            "ipfs_accelerate_py/agent_supervisor/control",
            "ipfs_accelerate_py/agent_supervisor/merge",
            "ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_policy.py",
            "ipfs_accelerate_py/agent_supervisor/analysis/deterministic_doctor_contracts.py",
            # Non-TCB but never autonomously writable by doctor operators.
            "vendor",
            "third_party",
            "node_modules",
        }
    )
)

_DEFAULT_APPROVAL_EXCLUSIONS: Final[tuple[str, ...]] = tuple(
    item.value for item in DoctorApprovalClass
)

_DEFAULT_UNSUPPORTED_FRONTIERS: Final[tuple[str, ...]] = (
    "dynamic_splat",
    "ambiguous_overload",
    "reflection",
    "monkey_patch",
    "native_or_ffi",
    "unsafe_memory",
    "concurrency",
    "generated_code",
    "public_api_or_schema",
    "stateful_behavior",
    "cross_repository_edit",
    "new_external_dependency",
    "complex_new_behavior",
)


# ---------------------------------------------------------------------------
# Errors and closed rejection vocabulary
# ---------------------------------------------------------------------------


class DoctorTransformError(ValueError):
    """Malformed doctor transform input or closed-boundary violation."""


class DoctorTransformAuthorityError(DoctorTransformError):
    """Root, path, proof, or write-authority mismatch."""


class DoctorTransformUnsupportedError(DoctorTransformError):
    """Shape is outside the closed deterministic operator set."""


class DoctorOperatorRejectionReason(str, Enum):
    """Closed, audit-stable rejection codes for doctor operator proposals."""

    UNKNOWN_OPERATOR = "unknown_operator"
    UNSUPPORTED_KIND = "unsupported_kind"
    UNSUPPORTED_AST_SHAPE = "unsupported_ast_shape"
    DYNAMIC_SPLAT = "dynamic_splat"
    AMBIGUOUS_OVERLOAD = "ambiguous_overload"
    REFLECTION = "reflection"
    MONKEY_PATCH = "monkey_patch"
    STALE_SPAN = "stale_span"
    INCOMPLETE_MAPPING = "incomplete_mapping"
    UNPROVED_VALUE = "unproved_value"
    GENERATED_TARGET = "generated_target"
    NATIVE_OR_FFI = "native_or_ffi"
    UNSAFE_TARGET = "unsafe_target"
    CONCURRENCY_TARGET = "concurrency_target"
    FORBIDDEN_PATH = "forbidden_path"
    TCB_PATH = "trusted_computing_base_path"
    CROSS_ROOT_WRITE = "cross_root_write"
    COMPLEX_NEW_BEHAVIOR = "complex_new_behavior"
    BODY_IN_PROPOSAL = "body_in_proposal"
    PROOF_NOT_ADMITTED = "proof_not_admitted"
    MISSING_PROOF = "missing_proof"
    ROOT_MISMATCH = "root_mismatch"
    PATH_NOT_AUTHORIZED = "path_not_authorized"
    INVALID_IDENTIFIER = "invalid_identifier"
    NEW_DEPENDENCY = "new_dependency"
    SCOPE_ESCAPE = "scope_escape"
    INVENTED_BEHAVIOR = "invented_behavior"
    ALREADY_APPLIED_NOOP = "already_present_noop"
    RENDER_REJECTED = "render_rejected"
    REGISTRY_MISMATCH = "registry_mismatch"
    SEMANTIC_AUTHORITY = "semantic_authority"
    WRITE_AUTHORITY = "write_authority"
    OVERLOAD_COUNT = "ambiguous_overload"
    EMPTY_SPAN = "empty_span"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    RESTORE_CID_MISMATCH = "restore_cid_mismatch"
    RESTORE_MISSING_CONTENT = "restore_missing_content"
    RENDER_REQUIRED = "render_required"


_TRANSFORM_REASON_MAP: Final[Mapping[str, DoctorOperatorRejectionReason]] = MappingProxyType(
    {
        TransformRejectionReason.DYNAMIC_SPLAT.value: DoctorOperatorRejectionReason.DYNAMIC_SPLAT,
        TransformRejectionReason.AMBIGUOUS_OVERLOAD.value: DoctorOperatorRejectionReason.AMBIGUOUS_OVERLOAD,
        TransformRejectionReason.UNSUPPORTED_SYNTAX.value: DoctorOperatorRejectionReason.UNSUPPORTED_AST_SHAPE,
        TransformRejectionReason.STALE_SPAN.value: DoctorOperatorRejectionReason.STALE_SPAN,
        TransformRejectionReason.NON_TOTAL_MAPPING.value: DoctorOperatorRejectionReason.INCOMPLETE_MAPPING,
        TransformRejectionReason.NEW_DEPENDENCY.value: DoctorOperatorRejectionReason.NEW_DEPENDENCY,
        TransformRejectionReason.SCOPE_ESCAPE.value: DoctorOperatorRejectionReason.SCOPE_ESCAPE,
        TransformRejectionReason.INVENTED_BEHAVIOR.value: DoctorOperatorRejectionReason.INVENTED_BEHAVIOR,
        TransformRejectionReason.NO_CODE_AUTHORITY.value: DoctorOperatorRejectionReason.UNPROVED_VALUE,
        TransformRejectionReason.ROOT_MISMATCH.value: DoctorOperatorRejectionReason.ROOT_MISMATCH,
        TransformRejectionReason.PATH_NOT_AUTHORIZED.value: DoctorOperatorRejectionReason.PATH_NOT_AUTHORIZED,
        TransformRejectionReason.MISSING_PROOF.value: DoctorOperatorRejectionReason.MISSING_PROOF,
        TransformRejectionReason.EXPRESSION_MISMATCH.value: DoctorOperatorRejectionReason.UNPROVED_VALUE,
        TransformRejectionReason.UNSUPPORTED_KIND.value: DoctorOperatorRejectionReason.UNSUPPORTED_KIND,
        TransformRejectionReason.EMPTY_SPAN.value: DoctorOperatorRejectionReason.EMPTY_SPAN,
        TransformRejectionReason.INVALID_IDENTIFIER.value: DoctorOperatorRejectionReason.INVALID_IDENTIFIER,
        TransformRejectionReason.ALREADY_PRESENT_NOOP.value: DoctorOperatorRejectionReason.ALREADY_APPLIED_NOOP,
    }
)


# ---------------------------------------------------------------------------
# Kind → analytical transform mapping and closed descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorOperatorKindBinding:
    """Closed binding from a doctor operator kind to analytical machinery."""

    kind: DoctorOperatorKind
    operator_id: str
    analytical_kind: TransformKind | None
    input_type_refs: tuple[str, ...]
    output_type_refs: tuple[str, ...]
    supported_ast_shapes: tuple[str, ...]
    precondition_refs: tuple[str, ...]
    postcondition_refs: tuple[str, ...]
    frame_condition_refs: tuple[str, ...]
    proof_template_refs: tuple[str, ...]
    value_source_required: bool
    placement_constraints: tuple[str, ...]
    inverse_or_compensation_ref: str
    idempotent: bool = True


def _binding(
    kind: DoctorOperatorKind,
    *,
    analytical: TransformKind | None,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    shapes: tuple[str, ...],
    pre: tuple[str, ...],
    post: tuple[str, ...],
    frame: tuple[str, ...],
    proof: tuple[str, ...],
    value_required: bool,
    placement: tuple[str, ...],
    inverse: str,
) -> DoctorOperatorKindBinding:
    return DoctorOperatorKindBinding(
        kind=kind,
        operator_id=f"operator:{kind.value}",
        analytical_kind=analytical,
        input_type_refs=inputs,
        output_type_refs=outputs,
        supported_ast_shapes=shapes,
        precondition_refs=pre,
        postcondition_refs=post,
        frame_condition_refs=frame,
        proof_template_refs=proof,
        value_source_required=value_required,
        placement_constraints=placement,
        inverse_or_compensation_ref=inverse,
        idempotent=True,
    )


# Exhaustive closed table — one entry per DoctorOperatorKind.
_KIND_BINDINGS: Final[tuple[DoctorOperatorKindBinding, ...]] = (
    _binding(
        DoctorOperatorKind.EXACT_RENAME,
        analytical=TransformKind.RENAME_ARGUMENT,
        inputs=("ast:Name", "ast:arg", "ast:keyword"),
        outputs=("ast:Name", "ast:arg", "ast:keyword"),
        shapes=("identifier", "function_parameter", "keyword_argument"),
        pre=("pre:unique_symbol", "pre:proved_equivalence", "pre:single_overload"),
        post=("post:name_replaced", "post:referents_equivalent"),
        frame=("frame:non_target_symbols_unchanged", "frame:types_preserved"),
        proof=("proof:rename_equivalence",),
        value_required=False,
        placement=("placement:exact_span",),
        inverse="compensation:rename_inverse",
    ),
    _binding(
        DoctorOperatorKind.ADD_ARGUMENT,
        analytical=TransformKind.ADD_ARGUMENT,
        inputs=("ast:Call", "type:Expression"),
        outputs=("ast:Call",),
        shapes=("call", "keyword_argument", "positional_argument"),
        pre=("pre:unique_call_site", "pre:proved_expression", "pre:no_splat"),
        post=("post:argument_present", "post:call_typechecks"),
        frame=("frame:other_arguments_unchanged", "frame:callee_unchanged"),
        proof=("proof:argument_insertion",),
        value_required=True,
        placement=("placement:call_argument_list",),
        inverse="compensation:remove_argument",
    ),
    _binding(
        DoctorOperatorKind.RENAME_ARGUMENT,
        analytical=TransformKind.RENAME_ARGUMENT,
        inputs=("ast:Call", "ast:FunctionDef", "ast:AsyncFunctionDef"),
        outputs=("ast:Call", "ast:FunctionDef", "ast:AsyncFunctionDef"),
        shapes=("keyword_argument", "function_parameter"),
        pre=("pre:unique_parameter", "pre:proved_rename"),
        post=("post:parameter_renamed", "post:callers_updated"),
        frame=("frame:body_semantics_preserved",),
        proof=("proof:argument_rename",),
        # Analytical layer treats rename as value-bearing (expression authority).
        value_required=True,
        placement=("placement:parameter_or_keyword",),
        inverse="compensation:rename_argument_inverse",
    ),
    _binding(
        DoctorOperatorKind.REORDER_ARGUMENT,
        analytical=TransformKind.REORDER_ARGUMENT,
        inputs=("ast:Call",),
        outputs=("ast:Call",),
        shapes=("keyword_call",),
        pre=("pre:total_keyword_order", "pre:no_positional_ambiguity"),
        post=("post:argument_order_matches",),
        frame=("frame:argument_values_unchanged",),
        proof=("proof:argument_reorder",),
        # Analytical layer treats reorder as value-bearing (expression authority).
        value_required=True,
        placement=("placement:keyword_argument_list",),
        inverse="compensation:reorder_argument_inverse",
    ),
    _binding(
        DoctorOperatorKind.THREAD_ARGUMENT,
        analytical=TransformKind.THREAD_PARAMETER,
        inputs=("ast:Call", "type:Expression"),
        outputs=("ast:Call",),
        shapes=("call_route_hop",),
        pre=("pre:complete_route", "pre:proved_expression", "pre:finite_hops"),
        post=("post:parameter_threaded", "post:route_typechecks"),
        frame=("frame:non_route_sites_unchanged",),
        proof=("proof:parameter_thread",),
        value_required=True,
        placement=("placement:call_route",),
        inverse="compensation:unthread_parameter",
    ),
    _binding(
        DoctorOperatorKind.ADD_IMPORT,
        analytical=TransformKind.ADD_IMPORT,
        inputs=("ast:Module", "module:path"),
        outputs=("ast:Import", "ast:ImportFrom"),
        shapes=("module_header", "import_from"),
        pre=("pre:module_in_tree", "pre:no_new_external_dependency"),
        post=("post:import_present", "post:name_bound"),
        frame=("frame:other_imports_unchanged",),
        proof=("proof:import_presence",),
        value_required=False,
        placement=("placement:module_import_block",),
        inverse="compensation:remove_import",
    ),
    _binding(
        DoctorOperatorKind.ADD_EXPORT,
        analytical=TransformKind.ADD_EXPORT,
        inputs=("ast:Assign", "ast:List"),
        outputs=("ast:Assign",),
        shapes=("dunder_all", "export_list"),
        pre=("pre:export_name_defined", "pre:all_is_list"),
        post=("post:export_listed",),
        frame=("frame:other_exports_unchanged",),
        proof=("proof:export_registration",),
        value_required=False,
        placement=("placement:module_all",),
        inverse="compensation:remove_export",
    ),
    _binding(
        DoctorOperatorKind.ADD_REGISTRATION,
        analytical=TransformKind.ADD_REGISTRATION,
        inputs=("ast:Call",),
        outputs=("ast:Call",),
        shapes=("registration_call",),
        pre=("pre:registry_call_closed", "pre:target_in_tree"),
        post=("post:registration_present",),
        frame=("frame:registry_identity_unchanged",),
        proof=("proof:registration_entry",),
        value_required=False,
        placement=("placement:registration_site",),
        inverse="compensation:unregister",
    ),
    _binding(
        DoctorOperatorKind.ADD_CONSTRUCTOR_ROUTE,
        analytical=TransformKind.UPDATE_CONSTRUCTOR,
        inputs=("ast:Call", "type:Expression"),
        outputs=("ast:Call",),
        shapes=("constructor_call",),
        pre=("pre:constructor_in_tree", "pre:proved_expression"),
        post=("post:constructor_argument_present",),
        frame=("frame:other_constructor_args_unchanged",),
        proof=("proof:constructor_route",),
        value_required=True,
        placement=("placement:constructor_call",),
        inverse="compensation:remove_constructor_argument",
    ),
    _binding(
        DoctorOperatorKind.ADD_FACTORY_ROUTE,
        analytical=TransformKind.UPDATE_CONSTRUCTOR,
        inputs=("ast:Call", "type:Expression"),
        outputs=("ast:Call",),
        shapes=("factory_call",),
        pre=("pre:factory_in_tree", "pre:proved_expression"),
        post=("post:factory_argument_present",),
        frame=("frame:other_factory_args_unchanged",),
        proof=("proof:factory_route",),
        value_required=True,
        placement=("placement:factory_call",),
        inverse="compensation:remove_factory_argument",
    ),
    _binding(
        DoctorOperatorKind.FINITE_ADAPTER,
        analytical=TransformKind.ADD_ADAPTER,
        inputs=("type:Expression", "type:Adapter"),
        outputs=("ast:Call",),
        shapes=("adapter_wrap", "simple_expression"),
        pre=("pre:total_adapter_mapping", "pre:adapter_in_tree"),
        post=("post:adapter_applied", "post:type_compatible"),
        frame=("frame:adapted_value_source_unchanged",),
        proof=("proof:finite_adapter",),
        value_required=True,
        placement=("placement:expression_site",),
        inverse="compensation:unwrap_adapter",
    ),
    _binding(
        DoctorOperatorKind.SCHEMA_PROJECTION,
        analytical=TransformKind.UPDATE_SCHEMA_FIELD,
        inputs=("schema:object", "mapping:total"),
        outputs=("schema:object", "fixture:object", "manifest:object"),
        shapes=("json_object", "schema_literal", "fixture_literal", "manifest_literal"),
        pre=("pre:total_field_mapping", "pre:authoritative_generator"),
        post=("post:fields_projected", "post:keys_complete"),
        frame=("frame:unmapped_fields_unchanged",),
        proof=("proof:schema_projection",),
        value_required=False,
        placement=("placement:schema_or_fixture",),
        inverse="compensation:schema_projection_inverse",
    ),
    _binding(
        DoctorOperatorKind.RESTORE_TRACKED_ARTIFACT,
        analytical=None,
        inputs=("artifact:cid", "path:tracked"),
        outputs=("artifact:bytes",),
        shapes=("whole_file", "tracked_blob"),
        pre=("pre:verified_cid", "pre:canonical_preimage", "pre:tracked_path"),
        post=("post:artifact_restored", "post:cid_matches"),
        frame=("frame:other_paths_unchanged",),
        proof=("proof:artifact_restoration",),
        value_required=True,
        placement=("placement:whole_file",),
        inverse="compensation:restore_previous_cid",
    ),
)

_KIND_BY_OPERATOR: Final[Mapping[DoctorOperatorKind, DoctorOperatorKindBinding]] = MappingProxyType(
    {item.kind: item for item in _KIND_BINDINGS}
)

assert frozenset(_KIND_BY_OPERATOR) == frozenset(DoctorOperatorKind), (
    "default doctor operator table must cover every DoctorOperatorKind"
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    limit: int = MAX_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise DoctorTransformError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise DoctorTransformError(f"{name} is required")
    if len(result.encode("utf-8")) > limit:
        raise DoctorTransformError(f"{name} exceeds its byte bound")
    if "\0" in result:
        raise DoctorTransformError(f"{name} must not contain NUL")
    return result


def _optional_text(value: Any, name: str, *, limit: int = MAX_TEXT_BYTES) -> str:
    if value in (None, ""):
        return ""
    return _text(value, name, required=True, limit=limit)


def _identifier(value: Any, name: str) -> str:
    result = _text(value, name, required=True)
    if any(char.isspace() for char in result):
        raise DoctorTransformError(f"{name} must be a compact identifier")
    return result


def _optional_identifier(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _identifier(value, name)


def _path(value: Any, name: str = "path") -> str:
    raw = _text(value, name, required=True, limit=MAX_PATH_BYTES).replace("\\", "/")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or ".." in candidate.parts or raw in {".", ""}:
        raise DoctorTransformAuthorityError(
            f"{name} must be a relative repository path without escape"
        )
    return candidate.as_posix()


def _ids(
    values: Any,
    name: str,
    *,
    required: bool = False,
    limit: int = MAX_REFERENCE_COUNT,
) -> tuple[str, ...]:
    if values is None:
        values = ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise DoctorTransformError(f"{name} must be a sequence of identifiers")
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _identifier(item, name)
        if text not in seen:
            seen.add(text)
            result.append(text)
    if required and not result:
        raise DoctorTransformError(f"{name} must not be empty")
    if len(result) > limit:
        raise DoctorTransformError(f"{name} exceeds its item bound")
    return tuple(result)


def _paths(values: Any, name: str, *, limit: int = MAX_REFERENCE_COUNT) -> tuple[str, ...]:
    if values is None:
        values = ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise DoctorTransformError(f"{name} must be a sequence of paths")
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _path(item, name)
        if path not in seen:
            seen.add(path)
            result.append(path)
    if len(result) > limit:
        raise DoctorTransformError(f"{name} exceeds its item bound")
    return tuple(result)


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise DoctorTransformError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DoctorTransformError(f"{name} must be a non-negative integer")
    if value < 0:
        raise DoctorTransformError(f"{name} must be a non-negative integer")
    return value


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _is_python_identifier(name: str) -> bool:
    return bool(_PYTHON_IDENTIFIER.fullmatch(name))


def _assert_body_free_mapping(payload: Mapping[str, Any], name: str = "record") -> None:
    """Reject body/secret keys in body-free doctor proposals and specs."""

    body_markers = {
        "body",
        "source",
        "source_body",
        "source_text",
        "span_text",
        "contents",
        "content",
        "snippet",
        "code",
        "file_text",
        "raw_ast",
        "ast_body",
        "replacement",
        "secret",
        "secrets",
        "password",
        "token",
        "api_key",
        "private_key",
        "credential",
    }
    for key, value in payload.items():
        lowered = str(key).strip().lower()
        if lowered in body_markers or any(
            marker in lowered for marker in ("secret", "password", "private_key", "api_key")
        ):
            # content_id / before_hash / cid fields are digests, not bodies.
            if lowered.endswith(("_id", "_cid", "_hash", "_ref", "_refs")):
                continue
            if lowered in {"content_id", "before_hash", "expected_after_hash", "artifact_cid"}:
                continue
            raise DoctorTransformAuthorityError(
                f"{DoctorOperatorRejectionReason.BODY_IN_PROPOSAL.value}: "
                f"{name} must not carry body field {key!r}"
            )
        if isinstance(value, Mapping):
            _assert_body_free_mapping(value, name)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                if isinstance(item, Mapping):
                    _assert_body_free_mapping(item, name)


def _token_has_marker(tokens: set[str], markers: set[str]) -> bool:
    """True when any token equals a marker or is underscore-delimited around one."""

    for token in tokens:
        if token in markers:
            return True
        parts = token.split("_")
        if any(part in markers for part in parts if part):
            return True
    return False


def _scan_forbidden_markers(*texts: str) -> tuple[DoctorOperatorRejectionReason, ...]:
    reasons: list[DoctorOperatorRejectionReason] = []
    for raw in texts:
        if not raw:
            continue
        lowered = raw.lower().replace("-", "_")
        tokens = set(re.findall(r"[a-z_][a-z0-9_]*", lowered))
        if _token_has_marker(
            tokens, {"reflection", "reflect", "getattr", "setattr", "globals", "locals"}
        ):
            reasons.append(DoctorOperatorRejectionReason.REFLECTION)
        if _token_has_marker(tokens, {"monkey_patch", "monkeypatch"}) or "monkey_patch" in lowered:
            reasons.append(DoctorOperatorRejectionReason.MONKEY_PATCH)
        if _token_has_marker(tokens, {"generated", "codegen", "jinja", "template"}):
            reasons.append(DoctorOperatorRejectionReason.GENERATED_TARGET)
        if _token_has_marker(tokens, {"native", "ffi", "ctypes", "cffi"}):
            reasons.append(DoctorOperatorRejectionReason.NATIVE_OR_FFI)
        if _token_has_marker(tokens, {"unsafe"}):
            reasons.append(DoctorOperatorRejectionReason.UNSAFE_TARGET)
        if _token_has_marker(
            tokens, {"concurrency", "threading", "multiprocessing", "asyncio"}
        ):
            reasons.append(DoctorOperatorRejectionReason.CONCURRENCY_TARGET)
        if (
            _token_has_marker(tokens, {"eval", "exec", "compile"})
            or "os.system" in lowered
            or "subprocess" in lowered
        ):
            reasons.append(DoctorOperatorRejectionReason.COMPLEX_NEW_BEHAVIOR)
        if "*" in raw or "**" in raw:
            reasons.append(DoctorOperatorRejectionReason.DYNAMIC_SPLAT)
    return tuple(dict.fromkeys(reasons))


def _map_transform_reasons(
    reasons: Sequence[str],
) -> tuple[DoctorOperatorRejectionReason, ...]:
    mapped: list[DoctorOperatorRejectionReason] = []
    for reason in reasons:
        known = _TRANSFORM_REASON_MAP.get(reason)
        if known is not None:
            mapped.append(known)
        else:
            mapped.append(DoctorOperatorRejectionReason.RENDER_REJECTED)
    return tuple(dict.fromkeys(mapped))


def doctor_roots_to_propagation_roots(
    roots: DoctorAuthorityRoots,
) -> PropagationAuthorityRoots:
    """Bridge doctor roots into the analytical transform root schema.

    Candidate identities reuse the current doctor forest/tree/overlay.  Base
    identities are derived deterministically so the analytical contract's
    base≠candidate invariant holds without inventing external roots.
    """

    if not isinstance(roots, DoctorAuthorityRoots):
        raise DoctorTransformError("roots must be DoctorAuthorityRoots")
    return PropagationAuthorityRoots(
        repository_id=roots.repository_id,
        base_forest_id=f"base-of:{roots.forest_id}",
        base_tree_id=f"base-of:{roots.tree_id}",
        base_overlay_id=f"base-of:{roots.overlay_id}",
        candidate_forest_id=roots.forest_id,
        candidate_tree_id=roots.tree_id,
        candidate_overlay_id=roots.overlay_id,
        graph_id=roots.graph_id,
        index_id=roots.index_id,
        model_id=roots.model_id,
        config_id=roots.corpus_id,
        translator_id=roots.translator_id,
        toolchain_id=roots.toolchain_id,
        policy_id=roots.policy_id,
    )


def _path_is_forbidden(path: str, forbidden: Sequence[str]) -> bool:
    if is_doctor_tcb_path(path):
        return True
    normalized = PurePosixPath(path).as_posix()
    for item in forbidden:
        marker = item.rstrip("/")
        if normalized == marker or normalized.startswith(marker + "/"):
            return True
        if marker.endswith(".py") and normalized == marker:
            return True
    return False


def _expression_looks_invented(expression_text: str) -> bool:
    """Reject non-closed expressions that invent behavior (calls, lambdas, …)."""

    text = expression_text.strip()
    if not text:
        return False
    if _SIMPLE_ATTR_EXPR.fullmatch(text):
        return False
    # Allow simple literals used as proved values.
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return True
    node = tree.body
    if isinstance(node, (ast.Name, ast.Attribute, ast.Constant)):
        return False
    return True


# ---------------------------------------------------------------------------
# Descriptor / proposal / receipt contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorOperatorDescriptor(CanonicalContract):
    """Closed operator metadata layered on :class:`DoctorRepairOperatorSpec`."""

    SCHEMA: ClassVar[str] = DOCTOR_OPERATOR_DESCRIPTOR_SCHEMA

    spec: DoctorRepairOperatorSpec
    analytical_kind: str
    input_type_refs: tuple[str, ...]
    output_type_refs: tuple[str, ...]
    supported_ast_shapes: tuple[str, ...]
    value_source_required: bool

    def __post_init__(self) -> None:
        if not isinstance(self.spec, DoctorRepairOperatorSpec):
            raise DoctorTransformError("spec must be DoctorRepairOperatorSpec")
        object.__setattr__(
            self,
            "analytical_kind",
            _optional_text(self.analytical_kind, "analytical_kind", limit=128),
        )
        object.__setattr__(
            self,
            "input_type_refs",
            _ids(self.input_type_refs, "input_type_refs", required=True, limit=MAX_TYPE_REFS),
        )
        object.__setattr__(
            self,
            "output_type_refs",
            _ids(self.output_type_refs, "output_type_refs", required=True, limit=MAX_TYPE_REFS),
        )
        object.__setattr__(
            self,
            "supported_ast_shapes",
            _ids(
                self.supported_ast_shapes,
                "supported_ast_shapes",
                required=True,
                limit=MAX_AST_SHAPES,
            ),
        )
        object.__setattr__(
            self,
            "value_source_required",
            _bool(self.value_source_required, "value_source_required"),
        )
        if self.spec.grants_write_authority:
            raise DoctorTransformAuthorityError(
                DoctorOperatorRejectionReason.WRITE_AUTHORITY.value
            )
        if self.spec.semantic_authority:
            raise DoctorTransformAuthorityError(
                DoctorOperatorRejectionReason.SEMANTIC_AUTHORITY.value
            )
        if not self.spec.idempotent:
            raise DoctorTransformUnsupportedError(
                "registered operators must be idempotent"
            )
        if not self.spec.inverse_or_compensation_ref:
            raise DoctorTransformUnsupportedError(
                "registered operators require inverse_or_compensation_ref"
            )
        if not self.spec.renderer_id:
            raise DoctorTransformUnsupportedError(
                "registered operators require renderer_id"
            )
        _assert_body_free_mapping(self._payload(), "operator descriptor")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "spec": self.spec.to_dict(),
            "analytical_kind": self.analytical_kind,
            "input_type_refs": list(self.input_type_refs),
            "output_type_refs": list(self.output_type_refs),
            "supported_ast_shapes": list(self.supported_ast_shapes),
            "value_source_required": self.value_source_required,
        }

    @property
    def kind(self) -> DoctorOperatorKind:
        return self.spec.kind

    @property
    def operator_id(self) -> str:
        return self.spec.operator_id


@dataclass(frozen=True)
class DoctorOperatorProposal(CanonicalContract):
    """Body-free operator proposal.  Carries refs/hashes only — never source."""

    SCHEMA: ClassVar[str] = DOCTOR_OPERATOR_PROPOSAL_SCHEMA

    roots: DoctorAuthorityRoots
    proposal_id: str
    operator_id: str
    kind: DoctorOperatorKind
    edit_site: DoctorEditSite
    obligation_refs: tuple[str, ...]
    proof_refs: tuple[str, ...] = ()
    value_source_refs: tuple[str, ...] = ()
    expression_ref: str = ""
    parameter_name: str = ""
    previous_parameter_name: str = ""
    argument_order: tuple[str, ...] = ()
    keyword_style: bool = True
    insert_position: int | None = None
    import_module: str = ""
    import_name: str = ""
    export_name: str = ""
    registration_name: str = ""
    registration_target: str = ""
    adapter_expression: str = ""
    field_mapping_refs: tuple[str, ...] = ()
    allowed_dependency_paths: tuple[str, ...] = ()
    route_site_ids: tuple[str, ...] = ()
    dependency_transform_ids: tuple[str, ...] = ()
    postcondition_refs: tuple[str, ...] = ()
    artifact_cid: str = ""
    artifact_preimage_hash: str = ""
    language: str = "python"
    overload_count: int = 1
    proof_admitted: bool = False
    grants_write_authority: bool = False
    semantic_authority: bool = False
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        if not isinstance(self.roots, DoctorAuthorityRoots):
            raise DoctorTransformError("roots must be DoctorAuthorityRoots")
        object.__setattr__(self, "proposal_id", _identifier(self.proposal_id, "proposal_id"))
        object.__setattr__(self, "operator_id", _identifier(self.operator_id, "operator_id"))
        kind = (
            self.kind
            if isinstance(self.kind, DoctorOperatorKind)
            else DoctorOperatorKind(self.kind)
        )
        object.__setattr__(self, "kind", kind)
        if not isinstance(self.edit_site, DoctorEditSite):
            raise DoctorTransformError("edit_site must be DoctorEditSite")
        object.__setattr__(
            self,
            "obligation_refs",
            _ids(self.obligation_refs, "obligation_refs", required=True),
        )
        object.__setattr__(self, "proof_refs", _ids(self.proof_refs, "proof_refs"))
        object.__setattr__(
            self, "value_source_refs", _ids(self.value_source_refs, "value_source_refs")
        )
        for name in (
            "expression_ref",
            "parameter_name",
            "previous_parameter_name",
            "import_module",
            "import_name",
            "export_name",
            "registration_name",
            "registration_target",
            "adapter_expression",
            "artifact_cid",
            "artifact_preimage_hash",
            "language",
        ):
            object.__setattr__(
                self, name, _optional_text(getattr(self, name), name, limit=MAX_EXPRESSION_BYTES)
            )
        object.__setattr__(
            self, "argument_order", _ids(self.argument_order, "argument_order")
        )
        object.__setattr__(self, "keyword_style", _bool(self.keyword_style, "keyword_style"))
        if self.insert_position is not None:
            object.__setattr__(
                self,
                "insert_position",
                _nonneg_int(self.insert_position, "insert_position"),
            )
        object.__setattr__(
            self,
            "field_mapping_refs",
            _ids(self.field_mapping_refs, "field_mapping_refs"),
        )
        object.__setattr__(
            self,
            "allowed_dependency_paths",
            _paths(self.allowed_dependency_paths, "allowed_dependency_paths"),
        )
        object.__setattr__(
            self, "route_site_ids", _ids(self.route_site_ids, "route_site_ids")
        )
        object.__setattr__(
            self,
            "dependency_transform_ids",
            _ids(self.dependency_transform_ids, "dependency_transform_ids"),
        )
        object.__setattr__(
            self, "postcondition_refs", _ids(self.postcondition_refs, "postcondition_refs")
        )
        object.__setattr__(
            self, "overload_count", _nonneg_int(self.overload_count, "overload_count")
        )
        object.__setattr__(
            self, "proof_admitted", _bool(self.proof_admitted, "proof_admitted")
        )
        # Proposals never grant write or semantic authority.
        if self.grants_write_authority is not False:
            raise DoctorTransformAuthorityError(
                DoctorOperatorRejectionReason.WRITE_AUTHORITY.value
            )
        object.__setattr__(self, "grants_write_authority", False)
        if self.semantic_authority is not False:
            raise DoctorTransformAuthorityError(
                DoctorOperatorRejectionReason.SEMANTIC_AUTHORITY.value
            )
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "producer_id", _identifier(self.producer_id, "producer_id"))
        if self.language != "python":
            raise DoctorTransformUnsupportedError(
                DoctorOperatorRejectionReason.UNSUPPORTED_LANGUAGE.value
            )
        if self.overload_count != 1:
            raise DoctorTransformUnsupportedError(
                DoctorOperatorRejectionReason.AMBIGUOUS_OVERLOAD.value
            )
        payload = self._payload()
        _assert_body_free_mapping(payload, "operator proposal")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "roots": self.roots.to_dict(),
            "proposal_id": self.proposal_id,
            "operator_id": self.operator_id,
            "kind": self.kind.value,
            "edit_site": self.edit_site.to_dict(),
            "obligation_refs": list(self.obligation_refs),
            "proof_refs": list(self.proof_refs),
            "value_source_refs": list(self.value_source_refs),
            "expression_ref": self.expression_ref,
            "parameter_name": self.parameter_name,
            "previous_parameter_name": self.previous_parameter_name,
            "argument_order": list(self.argument_order),
            "keyword_style": self.keyword_style,
            "insert_position": self.insert_position,
            "import_module": self.import_module,
            "import_name": self.import_name,
            "export_name": self.export_name,
            "registration_name": self.registration_name,
            "registration_target": self.registration_target,
            "adapter_expression": self.adapter_expression,
            "field_mapping_refs": list(self.field_mapping_refs),
            "allowed_dependency_paths": list(self.allowed_dependency_paths),
            "route_site_ids": list(self.route_site_ids),
            "dependency_transform_ids": list(self.dependency_transform_ids),
            "postcondition_refs": list(self.postcondition_refs),
            "artifact_cid": self.artifact_cid,
            "artifact_preimage_hash": self.artifact_preimage_hash,
            "language": self.language,
            "overload_count": self.overload_count,
            "proof_admitted": self.proof_admitted,
            "grants_write_authority": False,
            "semantic_authority": False,
            "producer_id": self.producer_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorOperatorProposal":
        if not isinstance(payload, Mapping):
            raise DoctorTransformError("proposal payload must be a mapping")
        roots = payload.get("roots")
        if isinstance(roots, DoctorAuthorityRoots):
            root_obj = roots
        elif isinstance(roots, Mapping):
            root_obj = DoctorAuthorityRoots.from_dict(roots)
        else:
            raise DoctorTransformError("proposal roots are required")
        edit = payload.get("edit_site")
        if isinstance(edit, DoctorEditSite):
            site = edit
        elif isinstance(edit, Mapping):
            site = DoctorEditSite.from_dict(edit)
        else:
            raise DoctorTransformError("edit_site is required")
        return cls(
            roots=root_obj,
            proposal_id=str(payload.get("proposal_id", "")),
            operator_id=str(payload.get("operator_id", "")),
            kind=DoctorOperatorKind(str(payload.get("kind", ""))),
            edit_site=site,
            obligation_refs=tuple(payload.get("obligation_refs") or ()),
            proof_refs=tuple(payload.get("proof_refs") or ()),
            value_source_refs=tuple(payload.get("value_source_refs") or ()),
            expression_ref=str(payload.get("expression_ref") or ""),
            parameter_name=str(payload.get("parameter_name") or ""),
            previous_parameter_name=str(payload.get("previous_parameter_name") or ""),
            argument_order=tuple(payload.get("argument_order") or ()),
            keyword_style=bool(payload.get("keyword_style", True)),
            insert_position=payload.get("insert_position"),
            import_module=str(payload.get("import_module") or ""),
            import_name=str(payload.get("import_name") or ""),
            export_name=str(payload.get("export_name") or ""),
            registration_name=str(payload.get("registration_name") or ""),
            registration_target=str(payload.get("registration_target") or ""),
            adapter_expression=str(payload.get("adapter_expression") or ""),
            field_mapping_refs=tuple(payload.get("field_mapping_refs") or ()),
            allowed_dependency_paths=tuple(payload.get("allowed_dependency_paths") or ()),
            route_site_ids=tuple(payload.get("route_site_ids") or ()),
            dependency_transform_ids=tuple(payload.get("dependency_transform_ids") or ()),
            postcondition_refs=tuple(payload.get("postcondition_refs") or ()),
            artifact_cid=str(payload.get("artifact_cid") or ""),
            artifact_preimage_hash=str(payload.get("artifact_preimage_hash") or ""),
            language=str(payload.get("language") or "python"),
            overload_count=int(payload.get("overload_count", 1)),
            proof_admitted=bool(payload.get("proof_admitted", False)),
            grants_write_authority=bool(payload.get("grants_write_authority", False)),
            semantic_authority=bool(payload.get("semantic_authority", False)),
            producer_id=str(payload.get("producer_id") or PRODUCER_ID),
        )


@dataclass(frozen=True)
class DoctorOperatorReceipt(CanonicalContract):
    """Proposal evaluation receipt; edits appear only after proof admission."""

    SCHEMA: ClassVar[str] = DOCTOR_OPERATOR_RECEIPT_SCHEMA

    proposal: DoctorOperatorProposal
    disposition: DoctorRepairDisposition
    rejection_reasons: tuple[str, ...] = ()
    render_receipt_id: str = ""
    expected_after_hash: str = ""
    replacement_hash: str = ""
    postcondition_refs: tuple[str, ...] = ()
    idempotent_noop: bool = False
    producer_id: str = PRODUCER_ID
    replay_identity: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, DoctorOperatorProposal):
            raise DoctorTransformError("proposal must be DoctorOperatorProposal")
        disposition = (
            self.disposition
            if isinstance(self.disposition, DoctorRepairDisposition)
            else DoctorRepairDisposition(self.disposition)
        )
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self, "rejection_reasons", _ids(self.rejection_reasons, "rejection_reasons")
        )
        object.__setattr__(
            self,
            "render_receipt_id",
            _optional_identifier(self.render_receipt_id, "render_receipt_id"),
        )
        object.__setattr__(
            self,
            "expected_after_hash",
            _optional_text(self.expected_after_hash, "expected_after_hash"),
        )
        object.__setattr__(
            self,
            "replacement_hash",
            _optional_text(self.replacement_hash, "replacement_hash"),
        )
        object.__setattr__(
            self, "postcondition_refs", _ids(self.postcondition_refs, "postcondition_refs")
        )
        object.__setattr__(
            self, "idempotent_noop", _bool(self.idempotent_noop, "idempotent_noop")
        )
        object.__setattr__(self, "producer_id", _identifier(self.producer_id, "producer_id"))
        if disposition is DoctorRepairDisposition.SUPPORTED:
            if self.rejection_reasons:
                raise DoctorTransformError(
                    "supported receipts cannot carry rejection reasons"
                )
            if not self.proposal.proof_admitted:
                raise DoctorTransformAuthorityError(
                    "supported receipts require a proof-admitted proposal"
                )
        else:
            if not self.rejection_reasons and disposition is DoctorRepairDisposition.ABSTAIN:
                # Abstention always carries at least one reason.
                raise DoctorTransformError("abstain receipts require rejection reasons")
        # Body-free: never embed replacement text — only hashes.
        if not self.replay_identity:
            object.__setattr__(
                self,
                "replay_identity",
                content_identity(self._payload_without_replay()),
            )
        else:
            object.__setattr__(
                self,
                "replay_identity",
                _identifier(self.replay_identity, "replay_identity"),
            )
        _assert_body_free_mapping(self._payload(), "operator receipt")

    def _payload_without_replay(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "proposal": self.proposal.to_dict(),
            "disposition": self.disposition.value,
            "rejection_reasons": list(self.rejection_reasons),
            "render_receipt_id": self.render_receipt_id,
            "expected_after_hash": self.expected_after_hash,
            "replacement_hash": self.replacement_hash,
            "postcondition_refs": list(self.postcondition_refs),
            "idempotent_noop": self.idempotent_noop,
            "producer_id": self.producer_id,
        }

    def _payload(self) -> dict[str, Any]:
        payload = self._payload_without_replay()
        payload["replay_identity"] = self.replay_identity
        return payload

    @property
    def admitted(self) -> bool:
        return self.disposition is DoctorRepairDisposition.SUPPORTED


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorRepairOperatorRegistry(CanonicalContract):
    """Immutable closed registry of allowlisted doctor repair operators."""

    SCHEMA: ClassVar[str] = DOCTOR_REPAIR_OPERATOR_REGISTRY_SCHEMA
    INTERFACE: ClassVar[str] = DOCTOR_REPAIR_OPERATOR_REGISTRY_INTERFACE

    roots: DoctorAuthorityRoots
    descriptors: tuple[DoctorOperatorDescriptor, ...]
    registry_id: str = ""
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        if not isinstance(self.roots, DoctorAuthorityRoots):
            raise DoctorTransformError("roots must be DoctorAuthorityRoots")
        if not self.descriptors:
            raise DoctorTransformError("registry must contain at least one operator")
        if len(self.descriptors) > MAX_OPERATOR_COUNT:
            raise DoctorTransformError("registry exceeds operator bound")
        if not all(isinstance(item, DoctorOperatorDescriptor) for item in self.descriptors):
            raise DoctorTransformError("descriptors must be DoctorOperatorDescriptor values")
        # Deterministic order by operator_id then kind.
        ordered = tuple(
            sorted(self.descriptors, key=lambda item: (item.operator_id, item.kind.value))
        )
        object.__setattr__(self, "descriptors", ordered)
        seen_ids: set[str] = set()
        seen_kinds: set[DoctorOperatorKind] = set()
        for item in ordered:
            if item.operator_id in seen_ids:
                raise DoctorTransformError(f"duplicate operator_id: {item.operator_id}")
            seen_ids.add(item.operator_id)
            if item.kind in seen_kinds:
                raise DoctorTransformError(f"duplicate operator kind: {item.kind.value}")
            seen_kinds.add(item.kind)
            if item.spec.roots.to_dict() != self.roots.to_dict():
                raise DoctorTransformAuthorityError(
                    DoctorOperatorRejectionReason.ROOT_MISMATCH.value
                )
        object.__setattr__(self, "producer_id", _identifier(self.producer_id, "producer_id"))
        computed = content_identity(self._payload_without_registry_id())
        provided = _optional_identifier(self.registry_id, "registry_id")
        if provided and provided != computed:
            raise DoctorTransformAuthorityError(
                DoctorOperatorRejectionReason.REGISTRY_MISMATCH.value
            )
        object.__setattr__(self, "registry_id", computed)
        _assert_body_free_mapping(self._payload(), "operator registry")

    def _payload_without_registry_id(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "roots": self.roots.to_dict(),
            "descriptors": [item.to_dict() for item in self.descriptors],
            "producer_id": self.producer_id,
        }

    def _payload(self) -> dict[str, Any]:
        payload = self._payload_without_registry_id()
        payload["registry_id"] = self.registry_id
        return payload

    def get(self, operator_id: str) -> DoctorOperatorDescriptor:
        for item in self.descriptors:
            if item.operator_id == operator_id:
                return item
        raise DoctorTransformUnsupportedError(
            DoctorOperatorRejectionReason.UNKNOWN_OPERATOR.value
        )

    def get_by_kind(self, kind: DoctorOperatorKind | str) -> DoctorOperatorDescriptor:
        resolved = kind if isinstance(kind, DoctorOperatorKind) else DoctorOperatorKind(kind)
        for item in self.descriptors:
            if item.kind is resolved:
                return item
        raise DoctorTransformUnsupportedError(
            DoctorOperatorRejectionReason.UNKNOWN_OPERATOR.value
        )

    def specs(self) -> tuple[DoctorRepairOperatorSpec, ...]:
        return tuple(item.spec for item in self.descriptors)

    def kinds(self) -> tuple[DoctorOperatorKind, ...]:
        return tuple(item.kind for item in self.descriptors)

    def propose(
        self,
        kind: DoctorOperatorKind | str,
        edit_site: DoctorEditSite,
        *,
        obligation_refs: Sequence[str],
        proposal_id: str = "",
        proof_refs: Sequence[str] = (),
        value_source_refs: Sequence[str] = (),
        expression_ref: str = "",
        parameter_name: str = "",
        previous_parameter_name: str = "",
        argument_order: Sequence[str] = (),
        keyword_style: bool = True,
        insert_position: int | None = None,
        import_module: str = "",
        import_name: str = "",
        export_name: str = "",
        registration_name: str = "",
        registration_target: str = "",
        adapter_expression: str = "",
        field_mapping_refs: Sequence[str] = (),
        allowed_dependency_paths: Sequence[str] = (),
        route_site_ids: Sequence[str] = (),
        dependency_transform_ids: Sequence[str] = (),
        postcondition_refs: Sequence[str] = (),
        artifact_cid: str = "",
        artifact_preimage_hash: str = "",
        language: str = "python",
        overload_count: int = 1,
        proof_admitted: bool = False,
    ) -> DoctorOperatorProposal:
        """Build a body-free proposal for a registered operator.

        Raises on closed-set violations that are structural (unknown kind,
        forbidden path, ambiguous overload).  Soft rejections (unproved
        values, stale spans) are deferred to :meth:`evaluate` /
        :meth:`render_admitted`.
        """

        descriptor = self.get_by_kind(kind)
        if not isinstance(edit_site, DoctorEditSite):
            raise DoctorTransformError("edit_site must be DoctorEditSite")
        if edit_site.path and _path_is_forbidden(
            edit_site.path, descriptor.spec.forbidden_paths or _DEFAULT_FORBIDDEN_PATHS
        ):
            raise DoctorTransformAuthorityError(
                DoctorOperatorRejectionReason.TCB_PATH.value
                if is_doctor_tcb_path(edit_site.path)
                else DoctorOperatorRejectionReason.FORBIDDEN_PATH.value
            )
        for write_path in descriptor.spec.write_paths:
            if _path_is_forbidden(write_path, descriptor.spec.forbidden_paths):
                raise DoctorTransformAuthorityError(
                    DoctorOperatorRejectionReason.FORBIDDEN_PATH.value
                )
        marker_reasons = _scan_forbidden_markers(
            parameter_name,
            previous_parameter_name,
            expression_ref,
            import_module,
            import_name,
            export_name,
            registration_name,
            registration_target,
            adapter_expression,
            artifact_cid,
            *field_mapping_refs,
            *route_site_ids,
        )
        if marker_reasons:
            raise DoctorTransformUnsupportedError(marker_reasons[0].value)
        if language != "python":
            raise DoctorTransformUnsupportedError(
                DoctorOperatorRejectionReason.UNSUPPORTED_LANGUAGE.value
            )
        if overload_count != 1:
            raise DoctorTransformUnsupportedError(
                DoctorOperatorRejectionReason.AMBIGUOUS_OVERLOAD.value
            )
        if not proposal_id:
            proposal_id = f"proposal:{descriptor.operator_id}:{edit_site.path}:{edit_site.before_hash[:24]}"
        return DoctorOperatorProposal(
            roots=self.roots,
            proposal_id=proposal_id,
            operator_id=descriptor.operator_id,
            kind=descriptor.kind,
            edit_site=edit_site,
            obligation_refs=tuple(obligation_refs),
            proof_refs=tuple(proof_refs),
            value_source_refs=tuple(value_source_refs),
            expression_ref=expression_ref,
            parameter_name=parameter_name,
            previous_parameter_name=previous_parameter_name,
            argument_order=tuple(argument_order),
            keyword_style=keyword_style,
            insert_position=insert_position,
            import_module=import_module,
            import_name=import_name,
            export_name=export_name,
            registration_name=registration_name,
            registration_target=registration_target,
            adapter_expression=adapter_expression,
            field_mapping_refs=tuple(field_mapping_refs),
            allowed_dependency_paths=tuple(allowed_dependency_paths),
            route_site_ids=tuple(route_site_ids),
            dependency_transform_ids=tuple(dependency_transform_ids),
            postcondition_refs=tuple(postcondition_refs)
            or descriptor.spec.postcondition_refs,
            artifact_cid=artifact_cid,
            artifact_preimage_hash=artifact_preimage_hash,
            language=language,
            overload_count=overload_count,
            proof_admitted=proof_admitted,
        )

    def evaluate(
        self,
        proposal: DoctorOperatorProposal,
        *,
        value_mapping: ValueMappingProof | None = None,
        decision: RepairTargetDecision | None = None,
    ) -> DoctorOperatorReceipt:
        """Evaluate a body-free proposal without rendering source bodies."""

        if not isinstance(proposal, DoctorOperatorProposal):
            raise DoctorTransformError("proposal must be DoctorOperatorProposal")
        reasons = self._preflight_reasons(
            proposal, value_mapping=value_mapping, decision=decision
        )
        if reasons:
            return DoctorOperatorReceipt(
                proposal=proposal,
                disposition=DoctorRepairDisposition.ABSTAIN,
                rejection_reasons=tuple(reason.value for reason in reasons),
                postcondition_refs=proposal.postcondition_refs,
            )
        if not proposal.proof_admitted:
            return DoctorOperatorReceipt(
                proposal=proposal,
                disposition=DoctorRepairDisposition.ABSTAIN,
                rejection_reasons=(
                    DoctorOperatorRejectionReason.PROOF_NOT_ADMITTED.value,
                ),
                postcondition_refs=proposal.postcondition_refs,
            )
        # Proof-admitted but body-free evaluate path never renders; materializers
        # must call render_admitted with the exact span under the bound hashes.
        return DoctorOperatorReceipt(
            proposal=proposal,
            disposition=DoctorRepairDisposition.ABSTAIN,
            rejection_reasons=(
                DoctorOperatorRejectionReason.RENDER_REQUIRED.value,
            ),
            postcondition_refs=proposal.postcondition_refs,
        )

    def render_admitted(
        self,
        proposal: DoctorOperatorProposal,
        *,
        span_text: str,
        expression_text: str = "",
        field_mappings: Sequence[FieldMapping] | Mapping[str, str] = (),
        verified_artifact_bytes: bytes | None = None,
        value_mapping: ValueMappingProof | None = None,
        decision: RepairTargetDecision | None = None,
        already_applied: bool = False,
    ) -> tuple[DoctorOperatorReceipt, TransformRenderReceipt | None]:
        """Render a proof-admitted proposal through the analytical transformer.

        Returns a body-free :class:`DoctorOperatorReceipt` plus the analytical
        render receipt (which may carry replacement text for the materializer).
        Until ``proposal.proof_admitted`` is true this method abstains and
        returns ``None`` for the analytical receipt.
        """

        if not isinstance(proposal, DoctorOperatorProposal):
            raise DoctorTransformError("proposal must be DoctorOperatorProposal")
        reasons = list(
            self._preflight_reasons(
                proposal, value_mapping=value_mapping, decision=decision
            )
        )
        if not proposal.proof_admitted:
            reasons.append(DoctorOperatorRejectionReason.PROOF_NOT_ADMITTED)
        if not isinstance(span_text, str):
            raise DoctorTransformError("span_text must be a string")
        if len(span_text.encode("utf-8")) > MAX_SPAN_BYTES:
            raise DoctorTransformError("span_text exceeds its byte bound")
        # Stale span: before_hash must match supplied text length/content.
        expected_hash = _sha256_text(span_text)
        site = proposal.edit_site
        if site.before_hash != expected_hash:
            reasons.append(DoctorOperatorRejectionReason.STALE_SPAN)
        if site.span_end < site.span_start:
            reasons.append(DoctorOperatorRejectionReason.STALE_SPAN)
        if (site.span_end - site.span_start) not in {0, len(span_text)} and site.span_end != site.span_start:
            # Allow absolute offsets when end-start equals span length.
            if (site.span_end - site.span_start) != len(span_text):
                reasons.append(DoctorOperatorRejectionReason.STALE_SPAN)

        marker_reasons = _scan_forbidden_markers(span_text, expression_text)
        reasons.extend(marker_reasons)

        if expression_text and _expression_looks_invented(expression_text):
            reasons.append(DoctorOperatorRejectionReason.INVENTED_BEHAVIOR)

        unique_reasons = tuple(dict.fromkeys(reasons))
        if unique_reasons:
            return (
                DoctorOperatorReceipt(
                    proposal=proposal,
                    disposition=DoctorRepairDisposition.ABSTAIN,
                    rejection_reasons=tuple(item.value for item in unique_reasons),
                    postcondition_refs=proposal.postcondition_refs,
                ),
                None,
            )

        descriptor = self.get(proposal.operator_id)
        if descriptor.kind is DoctorOperatorKind.RESTORE_TRACKED_ARTIFACT:
            return self._render_restore(
                proposal,
                span_text=span_text,
                verified_artifact_bytes=verified_artifact_bytes,
                already_applied=already_applied,
            )

        if descriptor.kind is DoctorOperatorKind.EXACT_RENAME and _is_python_identifier(
            span_text
        ):
            return self._render_exact_identifier_rename(
                proposal,
                span_text=span_text,
                already_applied=already_applied,
            )

        analytical = self._to_transform_site(
            proposal,
            span_text=span_text,
            expression_text=expression_text,
            field_mappings=field_mappings,
            analytical_kind=descriptor.analytical_kind or "",
        )
        try:
            render_receipt = render_analytical_transform(
                analytical,
                value_mapping=value_mapping,
                decision=decision,
            )
        except (
            AnalyticalChangeTransformError,
            AnalyticalChangeTransformAuthorityError,
            AnalyticalChangeTransformUnsupportedError,
        ) as exc:
            reason = str(exc).split(":", 1)[0].strip()
            mapped = _map_transform_reasons((reason,))
            return (
                DoctorOperatorReceipt(
                    proposal=proposal,
                    disposition=DoctorRepairDisposition.ABSTAIN,
                    rejection_reasons=tuple(item.value for item in mapped)
                    or (DoctorOperatorRejectionReason.RENDER_REJECTED.value,),
                    postcondition_refs=proposal.postcondition_refs,
                ),
                None,
            )

        if not render_receipt.admitted:
            mapped = _map_transform_reasons(render_receipt.rejection_reasons)
            return (
                DoctorOperatorReceipt(
                    proposal=proposal,
                    disposition=DoctorRepairDisposition.ABSTAIN,
                    rejection_reasons=tuple(item.value for item in mapped)
                    or (DoctorOperatorRejectionReason.RENDER_REJECTED.value,),
                    postcondition_refs=proposal.postcondition_refs,
                ),
                render_receipt,
            )

        edit = render_receipt.edits[0]
        noop = edit.replacement == span_text or already_applied
        receipt = DoctorOperatorReceipt(
            proposal=proposal,
            disposition=DoctorRepairDisposition.SUPPORTED,
            render_receipt_id=render_receipt.replay_identity,
            expected_after_hash=edit.expected_after_hash,
            replacement_hash=edit.expected_after_hash,
            postcondition_refs=proposal.postcondition_refs
            or descriptor.spec.postcondition_refs,
            idempotent_noop=noop,
        )
        return receipt, render_receipt

    def render_admitted_repeat_is_noop(
        self,
        proposal: DoctorOperatorProposal,
        *,
        span_text: str,
        expression_text: str = "",
        field_mappings: Sequence[FieldMapping] | Mapping[str, str] = (),
        value_mapping: ValueMappingProof | None = None,
        decision: RepairTargetDecision | None = None,
    ) -> bool:
        """Return True when a second render is byte-identical (idempotent)."""

        first, first_render = self.render_admitted(
            proposal,
            span_text=span_text,
            expression_text=expression_text,
            field_mappings=field_mappings,
            value_mapping=value_mapping,
            decision=decision,
        )
        if not first.admitted or first_render is None or not first_render.edits:
            return False
        applied = first_render.edits[0].replacement
        # Update edit site before_hash to the applied text for the second pass.
        applied_site = DoctorEditSite(
            path=proposal.edit_site.path,
            before_hash=_sha256_text(applied),
            span_start=proposal.edit_site.span_start,
            span_end=proposal.edit_site.span_start + len(applied),
            artifact_id=proposal.edit_site.artifact_id,
        )
        second_proposal = DoctorOperatorProposal(
            roots=proposal.roots,
            proposal_id=f"{proposal.proposal_id}:repeat",
            operator_id=proposal.operator_id,
            kind=proposal.kind,
            edit_site=applied_site,
            obligation_refs=proposal.obligation_refs,
            proof_refs=proposal.proof_refs,
            value_source_refs=proposal.value_source_refs,
            expression_ref=proposal.expression_ref,
            parameter_name=proposal.parameter_name,
            previous_parameter_name=proposal.previous_parameter_name,
            argument_order=proposal.argument_order,
            keyword_style=proposal.keyword_style,
            insert_position=proposal.insert_position,
            import_module=proposal.import_module,
            import_name=proposal.import_name,
            export_name=proposal.export_name,
            registration_name=proposal.registration_name,
            registration_target=proposal.registration_target,
            adapter_expression=proposal.adapter_expression,
            field_mapping_refs=proposal.field_mapping_refs,
            allowed_dependency_paths=proposal.allowed_dependency_paths,
            route_site_ids=proposal.route_site_ids,
            dependency_transform_ids=proposal.dependency_transform_ids,
            postcondition_refs=proposal.postcondition_refs,
            artifact_cid=proposal.artifact_cid,
            artifact_preimage_hash=proposal.artifact_preimage_hash,
            language=proposal.language,
            overload_count=proposal.overload_count,
            proof_admitted=True,
        )
        second, second_render = self.render_admitted(
            second_proposal,
            span_text=applied,
            expression_text=expression_text,
            field_mappings=field_mappings,
            value_mapping=value_mapping,
            decision=decision,
            already_applied=True,
        )
        if not second.admitted or second_render is None or not second_render.edits:
            # Deterministic rejection of a re-application is also acceptable.
            return not second.admitted
        return second_render.edits[0].replacement == applied

    # -- internal ------------------------------------------------------------

    def _preflight_reasons(
        self,
        proposal: DoctorOperatorProposal,
        *,
        value_mapping: ValueMappingProof | None,
        decision: RepairTargetDecision | None,
    ) -> tuple[DoctorOperatorRejectionReason, ...]:
        reasons: list[DoctorOperatorRejectionReason] = []
        if proposal.roots.to_dict() != self.roots.to_dict():
            reasons.append(DoctorOperatorRejectionReason.ROOT_MISMATCH)
        try:
            descriptor = self.get(proposal.operator_id)
        except DoctorTransformUnsupportedError:
            reasons.append(DoctorOperatorRejectionReason.UNKNOWN_OPERATOR)
            return tuple(reasons)
        if descriptor.kind is not proposal.kind:
            reasons.append(DoctorOperatorRejectionReason.UNSUPPORTED_KIND)
        if proposal.language != "python":
            reasons.append(DoctorOperatorRejectionReason.UNSUPPORTED_LANGUAGE)
        if proposal.overload_count != 1:
            reasons.append(DoctorOperatorRejectionReason.AMBIGUOUS_OVERLOAD)
        path = proposal.edit_site.path
        forbidden = descriptor.spec.forbidden_paths or _DEFAULT_FORBIDDEN_PATHS
        if _path_is_forbidden(path, forbidden):
            reasons.append(
                DoctorOperatorRejectionReason.TCB_PATH
                if is_doctor_tcb_path(path)
                else DoctorOperatorRejectionReason.FORBIDDEN_PATH
            )
        if decision is not None:
            if decision.disposition.value != "admitted":
                reasons.append(DoctorOperatorRejectionReason.PATH_NOT_AUTHORIZED)
            elif path not in set(decision.permitted_write_paths):
                reasons.append(DoctorOperatorRejectionReason.PATH_NOT_AUTHORIZED)
        marker_reasons = _scan_forbidden_markers(
            proposal.parameter_name,
            proposal.previous_parameter_name,
            proposal.expression_ref,
            proposal.import_module,
            proposal.import_name,
            proposal.export_name,
            proposal.registration_name,
            proposal.registration_target,
            proposal.adapter_expression,
            proposal.artifact_cid,
            *proposal.field_mapping_refs,
            *proposal.route_site_ids,
        )
        reasons.extend(marker_reasons)
        if not proposal.proof_refs and not proposal.proof_admitted:
            reasons.append(DoctorOperatorRejectionReason.MISSING_PROOF)
        if descriptor.value_source_required:
            if not proposal.value_source_refs and value_mapping is None:
                reasons.append(DoctorOperatorRejectionReason.UNPROVED_VALUE)
            if value_mapping is not None:
                if value_mapping.disposition.value != "unique_proved":
                    reasons.append(DoctorOperatorRejectionReason.UNPROVED_VALUE)
                if len(value_mapping.proved_candidate_ids) != 1:
                    reasons.append(DoctorOperatorRejectionReason.UNPROVED_VALUE)
                if (
                    value_mapping.repository_id
                    and value_mapping.repository_id != proposal.roots.repository_id
                ):
                    reasons.append(DoctorOperatorRejectionReason.ROOT_MISMATCH)
                if value_mapping.tree_id and value_mapping.tree_id != proposal.roots.tree_id:
                    reasons.append(DoctorOperatorRejectionReason.CROSS_ROOT_WRITE)
                if (
                    proposal.expression_ref
                    and value_mapping.expression_ref
                    and proposal.expression_ref != value_mapping.expression_ref
                ):
                    reasons.append(DoctorOperatorRejectionReason.UNPROVED_VALUE)
        if descriptor.kind is DoctorOperatorKind.SCHEMA_PROJECTION:
            if not proposal.field_mapping_refs:
                reasons.append(DoctorOperatorRejectionReason.INCOMPLETE_MAPPING)
        if descriptor.kind is DoctorOperatorKind.REORDER_ARGUMENT:
            if not proposal.argument_order:
                reasons.append(DoctorOperatorRejectionReason.INCOMPLETE_MAPPING)
        if descriptor.kind is DoctorOperatorKind.THREAD_ARGUMENT:
            if not proposal.route_site_ids:
                reasons.append(DoctorOperatorRejectionReason.INCOMPLETE_MAPPING)
        if descriptor.kind is DoctorOperatorKind.RESTORE_TRACKED_ARTIFACT:
            if not proposal.artifact_cid or not proposal.artifact_preimage_hash:
                reasons.append(DoctorOperatorRejectionReason.RESTORE_MISSING_CONTENT)
        if descriptor.kind is DoctorOperatorKind.ADD_IMPORT and proposal.allowed_dependency_paths:
            if proposal.import_module:
                projected = proposal.import_module.replace(".", "/") + ".py"
                allowed = set(proposal.allowed_dependency_paths)
                if projected not in allowed and proposal.import_module not in allowed:
                    if not any(
                        projected.startswith(item.rstrip("/") + "/")
                        or item == proposal.import_module
                        for item in allowed
                    ):
                        reasons.append(DoctorOperatorRejectionReason.NEW_DEPENDENCY)
        return tuple(dict.fromkeys(reasons))

    def _to_transform_site(
        self,
        proposal: DoctorOperatorProposal,
        *,
        span_text: str,
        expression_text: str,
        field_mappings: Sequence[FieldMapping] | Mapping[str, str],
        analytical_kind: str,
    ) -> TransformSite:
        if not analytical_kind:
            raise DoctorTransformUnsupportedError(
                DoctorOperatorRejectionReason.UNSUPPORTED_KIND.value
            )
        kind = TransformKind(analytical_kind)
        mappings: tuple[FieldMapping, ...]
        if isinstance(field_mappings, Mapping):
            mappings = tuple(
                FieldMapping(str(before), str(after))
                for before, after in field_mappings.items()
            )
        else:
            mappings = tuple(field_mappings)
        prop_roots = doctor_roots_to_propagation_roots(proposal.roots)
        span = TransformSourceSpan(
            path=proposal.edit_site.path,
            start=proposal.edit_site.span_start,
            end=proposal.edit_site.span_start + len(span_text)
            if proposal.edit_site.span_end == proposal.edit_site.span_start
            else proposal.edit_site.span_end,
            artifact_id=proposal.edit_site.artifact_id
            or f"blob:{proposal.edit_site.before_hash[7:23]}",
            span_text=span_text,
            before_hash=proposal.edit_site.before_hash,
        )
        # When end was absolute and mismatched length, make_span style binding wins.
        if span.end - span.start != len(span_text):
            span = make_span(
                proposal.edit_site.path,
                span_text,
                start=proposal.edit_site.span_start,
                artifact_id=proposal.edit_site.artifact_id,
            )
            if span.before_hash != proposal.edit_site.before_hash:
                raise DoctorTransformAuthorityError(
                    DoctorOperatorRejectionReason.STALE_SPAN.value
                )
        return TransformSite(
            roots=prop_roots,
            site_id=proposal.proposal_id,
            kind=kind,
            span=span,
            obligation_ids=proposal.obligation_refs,
            proof_refs=proposal.proof_refs,
            expression_ref=proposal.expression_ref,
            expression_text=expression_text,
            parameter_name=proposal.parameter_name,
            previous_parameter_name=proposal.previous_parameter_name,
            argument_order=proposal.argument_order,
            keyword_style=proposal.keyword_style,
            insert_position=proposal.insert_position,
            import_module=proposal.import_module,
            import_name=proposal.import_name,
            export_name=proposal.export_name,
            registration_name=proposal.registration_name,
            registration_target=proposal.registration_target,
            adapter_expression=proposal.adapter_expression,
            field_mappings=mappings,
            allowed_dependency_paths=proposal.allowed_dependency_paths,
            route_site_ids=proposal.route_site_ids,
            dependency_transform_ids=proposal.dependency_transform_ids,
            postcondition_refs=proposal.postcondition_refs,
            overload_count=proposal.overload_count,
            language=proposal.language,
        )

    def _render_exact_identifier_rename(
        self,
        proposal: DoctorOperatorProposal,
        *,
        span_text: str,
        already_applied: bool,
    ) -> tuple[DoctorOperatorReceipt, TransformRenderReceipt | None]:
        new_name = proposal.parameter_name
        if not new_name or not _is_python_identifier(new_name):
            return (
                DoctorOperatorReceipt(
                    proposal=proposal,
                    disposition=DoctorRepairDisposition.ABSTAIN,
                    rejection_reasons=(
                        DoctorOperatorRejectionReason.INVALID_IDENTIFIER.value,
                    ),
                ),
                None,
            )
        if not _is_python_identifier(span_text):
            return (
                DoctorOperatorReceipt(
                    proposal=proposal,
                    disposition=DoctorRepairDisposition.ABSTAIN,
                    rejection_reasons=(
                        DoctorOperatorRejectionReason.UNSUPPORTED_AST_SHAPE.value,
                    ),
                ),
                None,
            )
        if proposal.previous_parameter_name and proposal.previous_parameter_name != span_text:
            return (
                DoctorOperatorReceipt(
                    proposal=proposal,
                    disposition=DoctorRepairDisposition.ABSTAIN,
                    rejection_reasons=(
                        DoctorOperatorRejectionReason.STALE_SPAN.value,
                    ),
                ),
                None,
            )
        replacement = new_name
        after_hash = _sha256_text(replacement)
        noop = replacement == span_text or already_applied
        # Build a synthetic analytical receipt via the rename path for parity.
        prop_roots = doctor_roots_to_propagation_roots(proposal.roots)
        site = TransformSite(
            roots=prop_roots,
            site_id=proposal.proposal_id,
            kind=TransformKind.RENAME_ARGUMENT,
            span=TransformSourceSpan(
                path=proposal.edit_site.path,
                start=proposal.edit_site.span_start,
                end=proposal.edit_site.span_start + len(span_text),
                artifact_id=proposal.edit_site.artifact_id
                or f"blob:{proposal.edit_site.before_hash[7:23]}",
                span_text=span_text,
                before_hash=proposal.edit_site.before_hash,
            ),
            obligation_ids=proposal.obligation_refs,
            proof_refs=proposal.proof_refs or ("proof:rename",),
            parameter_name=new_name,
            previous_parameter_name=span_text,
        )
        # Identifier-only rename: produce receipt without full analytical call
        # when the span is a bare Name (analytical rename expects Call/Def).
        from .analytical_change_transforms import TransformEdit, TransformDisposition
        from ..analysis.change_propagation_contracts import AnalyticalTransform

        edit = TransformEdit(
            path=site.span.path,
            start=site.span.start,
            end=site.span.end,
            artifact_id=site.span.artifact_id,
            before_hash=site.span.before_hash,
            replacement=replacement,
            expected_after_hash=after_hash,
        )
        transform = AnalyticalTransform(
            roots=prop_roots,
            transform_id=f"transform:{proposal.proposal_id}",
            kind=TransformKind.RENAME_ARGUMENT,
            disposition=TransformDisposition.ADMITTED,
            obligation_ids=proposal.obligation_refs,
            target_paths=(site.span.path,),
            expression_refs=(),
            proof_refs=site.proof_refs,
        )
        render_receipt = TransformRenderReceipt(
            transform=transform,
            site_id=proposal.proposal_id,
            edits=(edit,),
            postcondition_refs=proposal.postcondition_refs,
        )
        receipt = DoctorOperatorReceipt(
            proposal=proposal,
            disposition=DoctorRepairDisposition.SUPPORTED,
            render_receipt_id=render_receipt.replay_identity,
            expected_after_hash=after_hash,
            replacement_hash=after_hash,
            postcondition_refs=proposal.postcondition_refs,
            idempotent_noop=noop,
        )
        return receipt, render_receipt

    def _render_restore(
        self,
        proposal: DoctorOperatorProposal,
        *,
        span_text: str,
        verified_artifact_bytes: bytes | None,
        already_applied: bool,
    ) -> tuple[DoctorOperatorReceipt, TransformRenderReceipt | None]:
        if verified_artifact_bytes is None:
            return (
                DoctorOperatorReceipt(
                    proposal=proposal,
                    disposition=DoctorRepairDisposition.ABSTAIN,
                    rejection_reasons=(
                        DoctorOperatorRejectionReason.RESTORE_MISSING_CONTENT.value,
                    ),
                ),
                None,
            )
        preimage = _sha256_bytes(verified_artifact_bytes)
        if proposal.artifact_preimage_hash and proposal.artifact_preimage_hash != preimage:
            return (
                DoctorOperatorReceipt(
                    proposal=proposal,
                    disposition=DoctorRepairDisposition.ABSTAIN,
                    rejection_reasons=(
                        DoctorOperatorRejectionReason.RESTORE_CID_MISMATCH.value,
                    ),
                ),
                None,
            )
        try:
            replacement = verified_artifact_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return (
                DoctorOperatorReceipt(
                    proposal=proposal,
                    disposition=DoctorRepairDisposition.ABSTAIN,
                    rejection_reasons=(
                        DoctorOperatorRejectionReason.UNSUPPORTED_AST_SHAPE.value,
                    ),
                ),
                None,
            )
        after_hash = _sha256_text(replacement)
        noop = replacement == span_text or already_applied
        prop_roots = doctor_roots_to_propagation_roots(proposal.roots)
        from .analytical_change_transforms import TransformEdit, TransformDisposition
        from ..analysis.change_propagation_contracts import AnalyticalTransform

        # Represent restore as a schema/fixture-style whole-span replacement
        # using UPDATE_GENERATED_MANIFEST kind for the analytical record shape.
        edit = TransformEdit(
            path=proposal.edit_site.path,
            start=proposal.edit_site.span_start,
            end=proposal.edit_site.span_start + len(span_text),
            artifact_id=proposal.edit_site.artifact_id
            or f"blob:{proposal.edit_site.before_hash[7:23]}",
            before_hash=proposal.edit_site.before_hash,
            replacement=replacement,
            expected_after_hash=after_hash,
        )
        transform = AnalyticalTransform(
            roots=prop_roots,
            transform_id=f"transform:{proposal.proposal_id}",
            kind=TransformKind.UPDATE_GENERATED_MANIFEST,
            disposition=TransformDisposition.ADMITTED,
            obligation_ids=proposal.obligation_refs,
            target_paths=(proposal.edit_site.path,),
            expression_refs=(),
            proof_refs=proposal.proof_refs or ("proof:restore",),
        )
        render_receipt = TransformRenderReceipt(
            transform=transform,
            site_id=proposal.proposal_id,
            edits=(edit,),
            postcondition_refs=proposal.postcondition_refs,
        )
        receipt = DoctorOperatorReceipt(
            proposal=proposal,
            disposition=DoctorRepairDisposition.SUPPORTED,
            render_receipt_id=render_receipt.replay_identity,
            expected_after_hash=after_hash,
            replacement_hash=after_hash,
            postcondition_refs=proposal.postcondition_refs,
            idempotent_noop=noop,
        )
        return receipt, render_receipt


def _build_descriptor(
    roots: DoctorAuthorityRoots,
    binding: DoctorOperatorKindBinding,
) -> DoctorOperatorDescriptor:
    analytical = binding.analytical_kind.value if binding.analytical_kind else ""
    renderer = (
        RENDERER_ID
        if binding.analytical_kind is not None
        else f"{PRODUCER_ID}:restore"
    )
    spec = DoctorRepairOperatorSpec(
        roots=roots,
        operator_id=binding.operator_id,
        kind=binding.kind,
        supported_languages=("python",),
        precondition_refs=binding.precondition_refs,
        postcondition_refs=binding.postcondition_refs,
        frame_condition_refs=binding.frame_condition_refs,
        proof_template_refs=binding.proof_template_refs,
        read_paths=(),
        write_paths=(),
        value_source_refs=(
            ("value:proved_expression",) if binding.value_source_required else ()
        ),
        placement_constraints=binding.placement_constraints,
        forbidden_paths=_DEFAULT_FORBIDDEN_PATHS,
        renderer_id=renderer,
        idempotent=binding.idempotent,
        inverse_or_compensation_ref=binding.inverse_or_compensation_ref,
        resource_bound_ref="bounds:doctor-operator-default",
        approval_exclusions=_DEFAULT_APPROVAL_EXCLUSIONS,
        unsupported_frontier_exclusions=_DEFAULT_UNSUPPORTED_FRONTIERS,
        semantic_authority=False,
        grants_write_authority=False,
    )
    return DoctorOperatorDescriptor(
        spec=spec,
        analytical_kind=analytical,
        input_type_refs=binding.input_type_refs,
        output_type_refs=binding.output_type_refs,
        supported_ast_shapes=binding.supported_ast_shapes,
        value_source_required=binding.value_source_required,
    )


def build_default_doctor_operator_registry(
    roots: DoctorAuthorityRoots,
) -> DoctorRepairOperatorRegistry:
    """Construct the closed default doctor operator registry for ``roots``."""

    if not isinstance(roots, DoctorAuthorityRoots):
        raise DoctorTransformError("roots must be DoctorAuthorityRoots")
    descriptors = tuple(_build_descriptor(roots, binding) for binding in _KIND_BINDINGS)
    return DoctorRepairOperatorRegistry(roots=roots, descriptors=descriptors)


def make_edit_site(
    path: str,
    span_text: str,
    *,
    start: int = 0,
    artifact_id: str = "",
) -> DoctorEditSite:
    """Build a body-free edit site bound to ``sha256(span_text)``."""

    text = span_text if isinstance(span_text, str) else ""
    before = _sha256_text(text)
    return DoctorEditSite(
        path=path,
        before_hash=before,
        span_start=start,
        span_end=start + len(text),
        artifact_id=artifact_id or f"blob:{before[7:23]}",
    )


def default_operator_registry_id(roots: DoctorAuthorityRoots) -> str:
    """Return the content id of the default registry under ``roots``."""

    return build_default_doctor_operator_registry(roots).registry_id


__all__ = (
    "CONTRACT_VERSION",
    "DOCTOR_OPERATOR_DESCRIPTOR_SCHEMA",
    "DOCTOR_OPERATOR_PROPOSAL_SCHEMA",
    "DOCTOR_OPERATOR_RECEIPT_SCHEMA",
    "DOCTOR_REPAIR_OPERATOR_REGISTRY_INTERFACE",
    "DOCTOR_REPAIR_OPERATOR_REGISTRY_SCHEMA",
    "PRODUCER_ID",
    "RENDERER_ID",
    "DoctorOperatorDescriptor",
    "DoctorOperatorKindBinding",
    "DoctorOperatorProposal",
    "DoctorOperatorReceipt",
    "DoctorOperatorRejectionReason",
    "DoctorRepairOperatorRegistry",
    "DoctorTransformAuthorityError",
    "DoctorTransformError",
    "DoctorTransformUnsupportedError",
    "build_default_doctor_operator_registry",
    "default_operator_registry_id",
    "doctor_roots_to_propagation_roots",
    "make_edit_site",
)
