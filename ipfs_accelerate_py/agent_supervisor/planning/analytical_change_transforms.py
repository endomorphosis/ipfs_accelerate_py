"""Deterministic analytical change transforms for closed Python shapes.

RPR-037 / ``AnalyticalChangeTransformer@1``

Given unique reconstructed value mappings (RPR-036), exact AST/source spans,
and optional repair-target write authority, this module *renders* closed,
replayable Python edits.  It never mutates the repository, never invents a
value source, and never escalates to a model.

Authority rules (fail-closed):

* Only closed supported shapes are rendered (add/rename/reorder argument,
  parameter threading, import/export/registration, finite adapter,
  typed constructor/factory update, authorized schema/fixture mapping).
* Unique proved expressions and current roots are required for value-bearing
  transforms; search order, nominations, and incomplete proofs grant nothing.
* Exact before-hash, span, proof, and expected-after identity are bound on every
  admitted edit.  Stale spans, dynamic splats, ambiguous overloads, non-total
  mappings, new dependencies, scope escape, and invented behavior are rejected.
* Formatting of the surrounding span text is preserved; repeated rendering of
  the same site is byte-equivalent and idempotent on an already-applied site.
* The canonical RPR-022 :class:`AnalyticalTransform` is imported and returned;
  this module does not redefine it.  Plans/receipts are emitted only.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final

from ..analysis.change_propagation_contracts import (
    AnalyticalTransform,
    ConsumerMigrationObligation,
    PropagationAuthorityRoots,
    TransformDisposition,
    TransformKind,
)
from ..analysis.contract_repair_contracts import (
    DecisionDisposition,
    RepairTargetDecision,
)
from ..program_ast_adapters import adapt_python_source
from ..proof.formal_verification_contracts import (
    CanonicalContract,
    content_identity,
)
from ..proof.missing_input_synthesis import (
    SynthesisDisposition,
    ValueMappingProof,
)


# ---------------------------------------------------------------------------
# Schema / producer constants
# ---------------------------------------------------------------------------

ANALYTICAL_CHANGE_TRANSFORMER_INTERFACE: Final[str] = (
    "AnalyticalChangeTransformer@1"
)
TRANSFORM_SITE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analytical-transform-site@1"
)
TRANSFORM_EDIT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analytical-transform-edit@1"
)
TRANSFORM_RENDER_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analytical-transform-render-receipt@1"
)
TRANSFORM_BATCH_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analytical-transform-batch-receipt@1"
)
PRODUCER_ID: Final[str] = "analytical-change-transforms@1"

MAX_SPAN_BYTES: Final[int] = 65_536
MAX_EXPRESSION_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024
MAX_IMPORT_BYTES: Final[int] = 1_024
MAX_SITES: Final[int] = 256
MAX_ROUTE_HOPS: Final[int] = 64
MAX_RECORD_BYTES: Final[int] = 262_144
CONTRACT_VERSION: Final[int] = 1

_PYTHON_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SIMPLE_ATTR_EXPR: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)
_MODULE_PATH: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)


# ---------------------------------------------------------------------------
# Errors and closed rejection vocabulary
# ---------------------------------------------------------------------------


class AnalyticalChangeTransformError(ValueError):
    """Malformed transform input or an attempt to weaken a fail-closed boundary."""


class AnalyticalChangeTransformAuthorityError(AnalyticalChangeTransformError):
    """Root, path, proof, or write-authority mismatch."""


class AnalyticalChangeTransformUnsupportedError(AnalyticalChangeTransformError):
    """Shape is outside the closed deterministic codemod set."""


class TransformRejectionReason(str, Enum):
    """Closed, audit-stable rejection codes (never free-form model text)."""

    DYNAMIC_SPLAT = "dynamic_splat"
    AMBIGUOUS_OVERLOAD = "ambiguous_overload"
    UNSUPPORTED_SYNTAX = "unsupported_syntax"
    STALE_SPAN = "stale_span"
    NON_TOTAL_MAPPING = "non_total_mapping"
    NEW_DEPENDENCY = "new_dependency"
    SCOPE_ESCAPE = "scope_escape"
    INVENTED_BEHAVIOR = "invented_behavior"
    NO_CODE_AUTHORITY = "no_code_authority"
    ROOT_MISMATCH = "root_mismatch"
    PATH_NOT_AUTHORIZED = "path_not_authorized"
    MISSING_PROOF = "missing_proof"
    EXPRESSION_MISMATCH = "expression_mismatch"
    UNSUPPORTED_KIND = "unsupported_kind"
    EMPTY_SPAN = "empty_span"
    INVALID_IDENTIFIER = "invalid_identifier"
    ALREADY_PRESENT_NOOP = "already_present_noop"


# Kinds that insert or rewrite a proved value expression.
_VALUE_BEARING_KINDS: Final[frozenset[TransformKind]] = frozenset(
    {
        TransformKind.ADD_ARGUMENT,
        TransformKind.RENAME_ARGUMENT,
        TransformKind.REORDER_ARGUMENT,
        TransformKind.THREAD_PARAMETER,
        TransformKind.ADD_ADAPTER,
        TransformKind.UPDATE_CONSTRUCTOR,
    }
)

# Kinds that may rewrite schema/fixture/serializer surfaces under total maps.
_MAPPING_KINDS: Final[frozenset[TransformKind]] = frozenset(
    {
        TransformKind.UPDATE_SCHEMA_FIELD,
        TransformKind.UPDATE_SERIALIZER,
        TransformKind.UPDATE_FIXTURE,
        TransformKind.UPDATE_GENERATED_MANIFEST,
    }
)

_WIRING_KINDS: Final[frozenset[TransformKind]] = frozenset(
    {
        TransformKind.ADD_IMPORT,
        TransformKind.ADD_EXPORT,
        TransformKind.ADD_REGISTRATION,
    }
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, required: bool = True, limit: int = MAX_EXPRESSION_BYTES) -> str:
    if not isinstance(value, str):
        raise AnalyticalChangeTransformError(f"{name} must be a string")
    # Preserve interior formatting for span/source text; only strip ends for ids.
    if name in {"span_text", "source_text", "replacement", "expression_text"}:
        result = value
    else:
        result = value.strip()
    if required and not result:
        raise AnalyticalChangeTransformError(f"{name} is required")
    if len(result.encode("utf-8")) > limit:
        raise AnalyticalChangeTransformError(f"{name} exceeds its byte bound")
    return result


def _identifier(value: Any, name: str) -> str:
    result = _text(value, name, required=True, limit=MAX_EXPRESSION_BYTES)
    if any(char.isspace() for char in result):
        raise AnalyticalChangeTransformError(f"{name} must be a compact identifier")
    return result


def _path(value: Any, name: str = "path") -> str:
    raw = _text(value, name, required=True, limit=MAX_PATH_BYTES).replace("\\", "/")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or ".." in candidate.parts or raw in {".", ""}:
        raise AnalyticalChangeTransformAuthorityError(
            f"{name} must be a relative repository path"
        )
    return candidate.as_posix()


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AnalyticalChangeTransformError(f"{name} must be a non-negative integer")
    return value


def _ids(values: Sequence[Any], name: str, *, required: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise AnalyticalChangeTransformError(f"{name} must be a sequence")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        ident = _identifier(item, name)
        if ident not in seen:
            seen.add(ident)
            ordered.append(ident)
    if required and not ordered:
        raise AnalyticalChangeTransformError(f"{name} must not be empty")
    return tuple(ordered)


def _sha256_text(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _is_python_identifier(name: str) -> bool:
    return bool(_PYTHON_IDENTIFIER.fullmatch(name))


def _is_simple_expression(text: str) -> bool:
    """Accept only closed attribute/name expressions (no calls, ops, or literals)."""

    candidate = text.strip()
    if not candidate or len(candidate.encode("utf-8")) > MAX_EXPRESSION_BYTES:
        return False
    if not _SIMPLE_ATTR_EXPR.fullmatch(candidate):
        return False
    try:
        tree = ast.parse(candidate, mode="eval")
    except SyntaxError:
        return False
    return isinstance(tree.body, (ast.Name, ast.Attribute))


def _parse_module(source: str, *, path: str = "<transform>") -> ast.Module:
    try:
        return ast.parse(source, filename=path, type_comments=True)
    except SyntaxError as exc:
        raise AnalyticalChangeTransformUnsupportedError(
            f"{TransformRejectionReason.UNSUPPORTED_SYNTAX.value}: {exc}"
        ) from exc


def _single_expr(source: str) -> ast.AST:
    module = _parse_module(source)
    if len(module.body) != 1:
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.UNSUPPORTED_SYNTAX.value
        )
    stmt = module.body[0]
    if isinstance(stmt, ast.Expr):
        return stmt.value
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign, ast.Import, ast.ImportFrom)):
        return stmt
    raise AnalyticalChangeTransformUnsupportedError(
        TransformRejectionReason.UNSUPPORTED_SYNTAX.value
    )


def _call_has_splat(call: ast.Call) -> bool:
    if any(isinstance(arg, ast.Starred) for arg in call.args):
        return True
    return any(keyword.arg is None for keyword in call.keywords)


def _args_has_splat(args: ast.arguments) -> bool:
    return args.vararg is not None or args.kwarg is not None


def _source_segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.UNSUPPORTED_SYNTAX.value
        )
    return segment


def _find_closing_paren(source: str, open_index: int) -> int:
    """Return the index of the matching ')' for the '(' at open_index."""

    if open_index < 0 or open_index >= len(source) or source[open_index] != "(":
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.UNSUPPORTED_SYNTAX.value
        )
    depth = 0
    in_str: str | None = None
    escape = False
    i = open_index
    while i < len(source):
        ch = source[i]
        if in_str is not None:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in {'"', "'"}:
            # Handle simple quotes; triple quotes are closed shapes we still scan.
            if source[i : i + 3] in {'"""', "'''"}:
                in_str = source[i : i + 3]
                i += 3
                continue
            in_str = ch
            i += 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise AnalyticalChangeTransformUnsupportedError(
        TransformRejectionReason.UNSUPPORTED_SYNTAX.value
    )


def _comma_style(source: str, call: ast.Call) -> str:
    """Infer ', ' vs ',' spacing from existing arguments; default to ', '."""

    if not call.args and not call.keywords:
        return ", "
    pieces: list[str] = []
    for arg in call.args:
        pieces.append(_source_segment(source, arg))
    for keyword in call.keywords:
        pieces.append(_source_segment(source, keyword))
    if len(pieces) < 2:
        # Look between first arg and closing paren for trailing style cues.
        return ", "
    first_end = call.args[0].end_col_offset if call.args else call.keywords[0].end_col_offset
    second_start = (
        call.args[1].col_offset
        if len(call.args) > 1
        else call.keywords[0].col_offset
    )
    if first_end is None or second_start is None:
        return ", "
    # Single-line heuristic using absolute offsets when available.
    try:
        between = source[first_end:second_start]
    except Exception:  # pragma: no cover - defensive
        return ", "
    if between.startswith(",\n"):
        return ",\n"
    if between.startswith(", "):
        return ", "
    if between.startswith(","):
        return ","
    return ", "


def _absolute_offsets(source: str, node: ast.AST) -> tuple[int, int]:
    """Map lineno/col_offset to absolute UTF-8-agnostic character offsets."""

    if (
        not hasattr(node, "lineno")
        or not hasattr(node, "col_offset")
        or not hasattr(node, "end_lineno")
        or not hasattr(node, "end_col_offset")
    ):
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.UNSUPPORTED_SYNTAX.value
        )
    lines = source.splitlines(keepends=True)
    lineno = int(node.lineno)  # type: ignore[attr-defined]
    col = int(node.col_offset)  # type: ignore[attr-defined]
    end_lineno = int(node.end_lineno)  # type: ignore[attr-defined]
    end_col = int(node.end_col_offset)  # type: ignore[attr-defined]
    if lineno < 1 or end_lineno < 1:
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.UNSUPPORTED_SYNTAX.value
        )

    def line_start(line_no: int) -> int:
        return sum(len(lines[i]) for i in range(line_no - 1))

    start = line_start(lineno) + col
    end = line_start(end_lineno) + end_col
    if not (0 <= start <= end <= len(source)):
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.UNSUPPORTED_SYNTAX.value
        )
    return start, end


def _roots_tuple(roots: PropagationAuthorityRoots) -> tuple[str, ...]:
    return (
        roots.repository_id,
        roots.base_tree_id,
        roots.candidate_tree_id,
        roots.graph_id,
        roots.index_id,
        roots.translator_id,
        roots.toolchain_id,
        roots.policy_id,
    )


# ---------------------------------------------------------------------------
# Site / edit / receipt contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformSourceSpan:
    """Exact half-open character span plus the bound source slice and hash."""

    path: str
    start: int
    end: int
    artifact_id: str
    span_text: str
    before_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path))
        object.__setattr__(self, "start", _nonneg_int(self.start, "start"))
        object.__setattr__(self, "end", _nonneg_int(self.end, "end"))
        if self.end < self.start:
            raise AnalyticalChangeTransformError("span end must be at or after start")
        object.__setattr__(self, "artifact_id", _identifier(self.artifact_id, "artifact_id"))
        # Empty span_text is allowed for pure insertion points (e.g. ADD_IMPORT
        # at a zero-width file offset).  Length must still equal end - start.
        text = _text(self.span_text, "span_text", required=False, limit=MAX_SPAN_BYTES)
        object.__setattr__(self, "span_text", text)
        if len(text) != (self.end - self.start):
            raise AnalyticalChangeTransformError(
                "span_text length must equal end - start"
            )
        expected = _sha256_text(text)
        provided = _text(self.before_hash, "before_hash", required=True)
        if provided != expected:
            raise AnalyticalChangeTransformAuthorityError(
                f"{TransformRejectionReason.STALE_SPAN.value}: before_hash mismatch"
            )
        object.__setattr__(self, "before_hash", provided)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start": self.start,
            "end": self.end,
            "artifact_id": self.artifact_id,
            "before_hash": self.before_hash,
            "span_text_sha256": self.before_hash,
        }


@dataclass(frozen=True)
class FieldMapping:
    """Total, authorized field rename/add mapping for schema/fixture surfaces."""

    before: str
    after: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "before", _text(self.before, "before", required=True))
        object.__setattr__(self, "after", _text(self.after, "after", required=True))

    def to_dict(self) -> dict[str, str]:
        return {"before": self.before, "after": self.after}


@dataclass(frozen=True)
class TransformSite:
    """One closed, authority-bound analytical transform request.

    The caller supplies the exact span text; this module never opens files.
    """

    SCHEMA: ClassVar[str] = TRANSFORM_SITE_SCHEMA

    roots: PropagationAuthorityRoots
    site_id: str
    kind: TransformKind
    span: TransformSourceSpan
    obligation_ids: tuple[str, ...]
    proof_refs: tuple[str, ...] = ()
    expression_ref: str = ""
    expression_text: str = ""
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
    field_mappings: tuple[FieldMapping, ...] = ()
    allowed_dependency_paths: tuple[str, ...] = ()
    route_site_ids: tuple[str, ...] = ()
    dependency_transform_ids: tuple[str, ...] = ()
    postcondition_refs: tuple[str, ...] = ()
    overload_count: int = 1
    language: str = "python"

    def __post_init__(self) -> None:
        if not isinstance(self.roots, PropagationAuthorityRoots):
            raise AnalyticalChangeTransformError("roots must be PropagationAuthorityRoots")
        object.__setattr__(self, "site_id", _identifier(self.site_id, "site_id"))
        kind = self.kind if isinstance(self.kind, TransformKind) else TransformKind(self.kind)
        object.__setattr__(self, "kind", kind)
        if not isinstance(self.span, TransformSourceSpan):
            raise AnalyticalChangeTransformError("span must be TransformSourceSpan")
        object.__setattr__(
            self, "obligation_ids", _ids(self.obligation_ids, "obligation_ids", required=True)
        )
        object.__setattr__(self, "proof_refs", _ids(self.proof_refs, "proof_refs"))
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
            "language",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        object.__setattr__(
            self,
            "expression_text",
            _text(self.expression_text, "expression_text", required=False, limit=MAX_EXPRESSION_BYTES),
        )
        object.__setattr__(
            self, "argument_order", _ids(self.argument_order, "argument_order")
        )
        if not isinstance(self.keyword_style, bool):
            raise AnalyticalChangeTransformError("keyword_style must be boolean")
        if self.insert_position is not None:
            object.__setattr__(
                self, "insert_position", _nonneg_int(self.insert_position, "insert_position")
            )
        if not all(isinstance(item, FieldMapping) for item in self.field_mappings):
            raise AnalyticalChangeTransformError("field_mappings must be FieldMapping values")
        object.__setattr__(
            self,
            "allowed_dependency_paths",
            tuple(_path(item, "allowed_dependency_paths") for item in self.allowed_dependency_paths),
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
        object.__setattr__(self, "overload_count", _nonneg_int(self.overload_count, "overload_count"))
        if self.language != "python":
            raise AnalyticalChangeTransformUnsupportedError(
                TransformRejectionReason.UNSUPPORTED_SYNTAX.value
            )
        if self.overload_count != 1:
            raise AnalyticalChangeTransformUnsupportedError(
                TransformRejectionReason.AMBIGUOUS_OVERLOAD.value
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "roots": self.roots.to_dict(),
            "site_id": self.site_id,
            "kind": self.kind.value,
            "span": self.span.to_dict(),
            "obligation_ids": list(self.obligation_ids),
            "proof_refs": list(self.proof_refs),
            "expression_ref": self.expression_ref,
            "expression_text": self.expression_text,
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
            "field_mappings": [item.to_dict() for item in self.field_mappings],
            "allowed_dependency_paths": list(self.allowed_dependency_paths),
            "route_site_ids": list(self.route_site_ids),
            "dependency_transform_ids": list(self.dependency_transform_ids),
            "postcondition_refs": list(self.postcondition_refs),
            "overload_count": self.overload_count,
            "language": self.language,
        }

    @property
    def content_id(self) -> str:
        # Exclude span_text body from identity; before_hash binds it.
        payload = self.to_dict()
        return content_identity(payload)


@dataclass(frozen=True)
class TransformEdit(CanonicalContract):
    """One exact replacement over a verified span; never applied by this module."""

    SCHEMA: ClassVar[str] = TRANSFORM_EDIT_SCHEMA

    path: str
    start: int
    end: int
    artifact_id: str
    before_hash: str
    replacement: str
    expected_after_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path))
        object.__setattr__(self, "start", _nonneg_int(self.start, "start"))
        object.__setattr__(self, "end", _nonneg_int(self.end, "end"))
        object.__setattr__(self, "artifact_id", _identifier(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "before_hash", _text(self.before_hash, "before_hash"))
        replacement = _text(
            self.replacement, "replacement", required=True, limit=MAX_SPAN_BYTES
        )
        object.__setattr__(self, "replacement", replacement)
        expected = _sha256_text(replacement)
        provided = _text(self.expected_after_hash, "expected_after_hash")
        if provided != expected:
            raise AnalyticalChangeTransformError(
                "expected_after_hash must equal sha256 of replacement"
            )
        object.__setattr__(self, "expected_after_hash", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "path": self.path,
            "start": self.start,
            "end": self.end,
            "artifact_id": self.artifact_id,
            "before_hash": self.before_hash,
            "replacement": self.replacement,
            "expected_after_hash": self.expected_after_hash,
        }


@dataclass(frozen=True)
class TransformRenderReceipt(CanonicalContract):
    """Rendered analytical transform with exact edits and replay identity."""

    SCHEMA: ClassVar[str] = TRANSFORM_RENDER_RECEIPT_SCHEMA

    transform: AnalyticalTransform
    site_id: str
    edits: tuple[TransformEdit, ...]
    postcondition_refs: tuple[str, ...] = ()
    import_statements: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    producer_id: str = PRODUCER_ID
    replay_identity: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.transform, AnalyticalTransform):
            raise AnalyticalChangeTransformError(
                "transform must be the canonical AnalyticalTransform@1"
            )
        object.__setattr__(self, "site_id", _identifier(self.site_id, "site_id"))
        if not all(isinstance(item, TransformEdit) for item in self.edits):
            raise AnalyticalChangeTransformError("edits must be TransformEdit values")
        object.__setattr__(
            self, "postcondition_refs", _ids(self.postcondition_refs, "postcondition_refs")
        )
        imports: list[str] = []
        for item in self.import_statements:
            text = _text(item, "import_statements", required=True, limit=MAX_IMPORT_BYTES)
            imports.append(text)
        object.__setattr__(self, "import_statements", tuple(imports))
        object.__setattr__(
            self, "rejection_reasons", _ids(self.rejection_reasons, "rejection_reasons")
        )
        object.__setattr__(self, "producer_id", _identifier(self.producer_id, "producer_id"))
        if self.transform.disposition is TransformDisposition.ADMITTED:
            if not self.edits:
                raise AnalyticalChangeTransformError(
                    "admitted transforms require at least one edit"
                )
            if self.rejection_reasons:
                raise AnalyticalChangeTransformError(
                    "admitted transforms cannot carry rejection reasons"
                )
        else:
            if self.edits:
                raise AnalyticalChangeTransformAuthorityError(
                    "non-admitted transforms cannot grant edits"
                )
            if (
                self.transform.disposition is TransformDisposition.REJECTED
                and not self.rejection_reasons
            ):
                raise AnalyticalChangeTransformError(
                    "rejected transforms require rejection reasons"
                )
        replay = self.replay_identity
        if not replay:
            replay = content_identity(self._payload_without_replay())
        object.__setattr__(self, "replay_identity", _identifier(replay, "replay_identity"))

    def _payload_without_replay(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "schema": self.SCHEMA,
            "transform": self.transform.to_record(),
            "site_id": self.site_id,
            "edits": [item.to_record() for item in self.edits],
            "postcondition_refs": list(self.postcondition_refs),
            "import_statements": list(self.import_statements),
            "rejection_reasons": list(self.rejection_reasons),
            "producer_id": self.producer_id,
        }

    def _payload(self) -> dict[str, Any]:
        payload = self._payload_without_replay()
        payload["replay_identity"] = self.replay_identity
        return payload

    @property
    def admitted(self) -> bool:
        return self.transform.disposition is TransformDisposition.ADMITTED


@dataclass(frozen=True)
class TransformBatchReceipt(CanonicalContract):
    """Deterministically ordered multi-site render result."""

    SCHEMA: ClassVar[str] = TRANSFORM_BATCH_RECEIPT_SCHEMA

    receipts: tuple[TransformRenderReceipt, ...]
    roots: PropagationAuthorityRoots
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        if not isinstance(self.roots, PropagationAuthorityRoots):
            raise AnalyticalChangeTransformError("roots must be PropagationAuthorityRoots")
        if len(self.receipts) > MAX_SITES:
            raise AnalyticalChangeTransformError("batch exceeds site bound")
        if not all(isinstance(item, TransformRenderReceipt) for item in self.receipts):
            raise AnalyticalChangeTransformError(
                "receipts must be TransformRenderReceipt values"
            )
        object.__setattr__(self, "producer_id", _identifier(self.producer_id, "producer_id"))

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "schema": self.SCHEMA,
            "roots": self.roots.to_dict(),
            "receipts": [item.to_record() for item in self.receipts],
            "producer_id": self.producer_id,
            "admitted_count": sum(1 for item in self.receipts if item.admitted),
            "rejected_count": sum(
                1
                for item in self.receipts
                if item.transform.disposition is TransformDisposition.REJECTED
            ),
            "abstained_count": sum(
                1
                for item in self.receipts
                if item.transform.disposition is TransformDisposition.ABSTAINED
            ),
        }

    @property
    def admitted_transforms(self) -> tuple[AnalyticalTransform, ...]:
        return tuple(
            item.transform for item in self.receipts if item.admitted
        )


# ---------------------------------------------------------------------------
# Shape-specific renderers (pure, deterministic)
# ---------------------------------------------------------------------------


def _render_add_argument(
    span_text: str,
    *,
    parameter_name: str,
    expression_text: str,
    keyword_style: bool,
    insert_position: int | None,
) -> str:
    node = _single_expr(span_text)
    if not isinstance(node, ast.Call):
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.UNSUPPORTED_SYNTAX.value
        )
    if _call_has_splat(node):
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.DYNAMIC_SPLAT.value
        )
    if not _is_python_identifier(parameter_name):
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.INVALID_IDENTIFIER.value
        )
    if not _is_simple_expression(expression_text):
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.INVENTED_BEHAVIOR.value
        )

    # Idempotency: already present as keyword or trailing matching expression.
    for keyword in node.keywords:
        if keyword.arg == parameter_name:
            return span_text
    if not keyword_style:
        for arg in node.args:
            if _source_segment(span_text, arg).strip() == expression_text.strip():
                return span_text

    open_paren = span_text.find("(")
    close_paren = _find_closing_paren(span_text, open_paren)
    interior = span_text[open_paren + 1 : close_paren]
    insertion = (
        f"{parameter_name}={expression_text.strip()}"
        if keyword_style
        else expression_text.strip()
    )
    sep = _comma_style(span_text, node)

    if not interior.strip():
        new_interior = insertion
    elif insert_position is None or insert_position >= (
        len(node.args) + len(node.keywords)
    ):
        # Append: preserve interior (including trailing whitespace/comments-free).
        stripped_right = interior.rstrip()
        trailing = interior[len(stripped_right) :]
        if stripped_right.endswith(","):
            new_interior = f"{stripped_right}{sep}{insertion}{trailing}"
        else:
            new_interior = f"{stripped_right}{sep}{insertion}{trailing}"
    else:
        # Positional insert among existing positional args only.
        if insert_position > len(node.args):
            raise AnalyticalChangeTransformUnsupportedError(
                TransformRejectionReason.UNSUPPORTED_SYNTAX.value
            )
        if keyword_style:
            # Keyword args always append to keep deterministic closed form.
            stripped_right = interior.rstrip()
            trailing = interior[len(stripped_right) :]
            new_interior = f"{stripped_right}{sep}{insertion}{trailing}"
        else:
            if insert_position == 0:
                if not node.args:
                    new_interior = insertion
                else:
                    first_start, _ = _absolute_offsets(span_text, node.args[0])
                    rel = first_start - (open_paren + 1)
                    new_interior = f"{insertion}{sep}{interior[rel:]}"
            else:
                prev = node.args[insert_position - 1]
                _, prev_end = _absolute_offsets(span_text, prev)
                rel = prev_end - (open_paren + 1)
                new_interior = f"{interior[:rel]}{sep}{insertion}{interior[rel:]}"

    return f"{span_text[: open_paren + 1]}{new_interior}{span_text[close_paren:]}"


def _render_rename_argument(
    span_text: str,
    *,
    previous_name: str,
    new_name: str,
) -> str:
    if not _is_python_identifier(previous_name) or not _is_python_identifier(new_name):
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.INVALID_IDENTIFIER.value
        )
    node = _single_expr(span_text)

    if isinstance(node, ast.Call):
        if _call_has_splat(node):
            raise AnalyticalChangeTransformUnsupportedError(
                TransformRejectionReason.DYNAMIC_SPLAT.value
            )
        matches = [kw for kw in node.keywords if kw.arg == previous_name]
        if not matches:
            # Idempotent if already renamed.
            if any(kw.arg == new_name for kw in node.keywords):
                return span_text
            raise AnalyticalChangeTransformUnsupportedError(
                TransformRejectionReason.UNSUPPORTED_SYNTAX.value
            )
        if len(matches) != 1:
            raise AnalyticalChangeTransformUnsupportedError(
                TransformRejectionReason.AMBIGUOUS_OVERLOAD.value
            )
        keyword = matches[0]
        start, end = _absolute_offsets(span_text, keyword)
        # Keyword source is "name=value"; replace only the name prefix.
        segment = span_text[start:end]
        eq = segment.find("=")
        if eq < 0:
            raise AnalyticalChangeTransformUnsupportedError(
                TransformRejectionReason.UNSUPPORTED_SYNTAX.value
            )
        return f"{span_text[:start]}{new_name}{segment[eq:]}{span_text[end:]}"

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if _args_has_splat(node.args):
            raise AnalyticalChangeTransformUnsupportedError(
                TransformRejectionReason.DYNAMIC_SPLAT.value
            )
        formals = [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]
        matches = [arg for arg in formals if arg.arg == previous_name]
        if not matches:
            if any(arg.arg == new_name for arg in formals):
                return span_text
            raise AnalyticalChangeTransformUnsupportedError(
                TransformRejectionReason.UNSUPPORTED_SYNTAX.value
            )
        if len(matches) != 1:
            raise AnalyticalChangeTransformUnsupportedError(
                TransformRejectionReason.AMBIGUOUS_OVERLOAD.value
            )
        arg = matches[0]
        start, end = _absolute_offsets(span_text, arg)
        # Annotation form: name: Type — replace only the identifier head.
        segment = span_text[start:end]
        if ":" in segment:
            head, _, tail = segment.partition(":")
            if head.strip() != previous_name:
                raise AnalyticalChangeTransformUnsupportedError(
                    TransformRejectionReason.UNSUPPORTED_SYNTAX.value
                )
            # Preserve whitespace between name and colon.
            ws = head[len(previous_name) :]
            return f"{span_text[:start]}{new_name}{ws}:{tail}{span_text[end:]}"
        if segment != previous_name:
            raise AnalyticalChangeTransformUnsupportedError(
                TransformRejectionReason.UNSUPPORTED_SYNTAX.value
            )
        return f"{span_text[:start]}{new_name}{span_text[end:]}"

    raise AnalyticalChangeTransformUnsupportedError(
        TransformRejectionReason.UNSUPPORTED_SYNTAX.value
    )


def _render_reorder_argument(
    span_text: str,
    *,
    argument_order: Sequence[str],
) -> str:
    node = _single_expr(span_text)
    if not isinstance(node, ast.Call):
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.UNSUPPORTED_SYNTAX.value
        )
    if _call_has_splat(node):
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.DYNAMIC_SPLAT.value
        )
    if node.args:
        # Only pure-keyword reorders are closed and total without invented positions.
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.UNSUPPORTED_SYNTAX.value
        )
    current = [kw.arg for kw in node.keywords]
    if any(name is None for name in current):
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.DYNAMIC_SPLAT.value
        )
    current_names = [str(name) for name in current]
    desired = list(argument_order)
    if sorted(current_names) != sorted(desired) or len(set(desired)) != len(desired):
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.NON_TOTAL_MAPPING.value
        )
    if current_names == desired:
        return span_text

    by_name = {kw.arg: _source_segment(span_text, kw) for kw in node.keywords}
    sep = _comma_style(span_text, node)
    open_paren = span_text.find("(")
    close_paren = _find_closing_paren(span_text, open_paren)
    new_interior = sep.join(by_name[name] for name in desired)
    return f"{span_text[: open_paren + 1]}{new_interior}{span_text[close_paren:]}"


def _render_add_import(
    span_text: str,
    *,
    module: str,
    name: str,
) -> str:
    if not _MODULE_PATH.fullmatch(module):
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.INVALID_IDENTIFIER.value
        )
    if name and not _is_python_identifier(name):
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.INVALID_IDENTIFIER.value
        )
    # Span may be empty insert-point (zero-length) or an existing import block.
    statement = f"from {module} import {name}" if name else f"import {module}"

    if not span_text:
        return statement

    module_ast = _parse_module(span_text)
    for stmt in module_ast.body:
        if isinstance(stmt, ast.Import) and not name:
            if any(alias.name == module for alias in stmt.names):
                return span_text
        if isinstance(stmt, ast.ImportFrom) and name:
            mod = stmt.module or ""
            if mod == module and any(alias.name == name for alias in stmt.names):
                return span_text
        # Reject dynamic / relative-star imports in the span.
        if isinstance(stmt, ast.ImportFrom) and any(
            alias.name == "*" for alias in stmt.names
        ):
            raise AnalyticalChangeTransformUnsupportedError(
                TransformRejectionReason.DYNAMIC_SPLAT.value
            )

    if span_text.endswith("\n"):
        return f"{span_text}{statement}\n"
    return f"{span_text}\n{statement}" if span_text else statement


def _render_add_export(span_text: str, *, export_name: str) -> str:
    if not _is_python_identifier(export_name):
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.INVALID_IDENTIFIER.value
        )
    node = _single_expr(span_text)
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.UNSUPPORTED_SYNTAX.value
        )
    target = node.targets[0]
    if not isinstance(target, ast.Name) or target.id != "__all__":
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.UNSUPPORTED_SYNTAX.value
        )
    value = node.value
    if not isinstance(value, (ast.List, ast.Tuple)):
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.UNSUPPORTED_SYNTAX.value
        )
    existing: list[str] = []
    for elt in value.elts:
        if not isinstance(elt, ast.Constant) or not isinstance(elt.value, str):
            raise AnalyticalChangeTransformUnsupportedError(
                TransformRejectionReason.UNSUPPORTED_SYNTAX.value
            )
        existing.append(elt.value)
    if export_name in existing:
        return span_text

    open_idx = span_text.find("[" if isinstance(value, ast.List) else "(")
    close_ch = "]" if isinstance(value, ast.List) else ")"
    if open_idx < 0:
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.UNSUPPORTED_SYNTAX.value
        )
    close_idx = span_text.rfind(close_ch)
    if close_idx < 0:
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.UNSUPPORTED_SYNTAX.value
        )
    interior = span_text[open_idx + 1 : close_idx]
    literal = f'"{export_name}"'
    if not interior.strip():
        new_interior = literal
    else:
        stripped = interior.rstrip()
        trailing = interior[len(stripped) :]
        sep = ", " if "\n" not in interior else ",\n"
        if stripped.endswith(","):
            new_interior = f"{stripped}{sep}{literal}{trailing}"
        else:
            new_interior = f"{stripped}{sep}{literal}{trailing}"
    return f"{span_text[: open_idx + 1]}{new_interior}{span_text[close_idx:]}"


def _render_add_registration(
    span_text: str,
    *,
    registration_name: str,
    registration_target: str,
) -> str:
    if not _is_python_identifier(registration_name) and not (
        registration_name.startswith('"') and registration_name.endswith('"')
    ):
        # Allow "name" string keys only as fully quoted closed literals.
        if not (
            len(registration_name) >= 2
            and registration_name[0] == registration_name[-1]
            and registration_name[0] in {'"', "'"}
        ):
            raise AnalyticalChangeTransformUnsupportedError(
                TransformRejectionReason.INVALID_IDENTIFIER.value
            )
    if not _is_simple_expression(registration_target):
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.INVENTED_BEHAVIOR.value
        )
    node = _single_expr(span_text)
    # Closed form: REG["name"] = target  or  REG.register("name", target)
    if isinstance(node, ast.Call):
        if _call_has_splat(node):
            raise AnalyticalChangeTransformUnsupportedError(
                TransformRejectionReason.DYNAMIC_SPLAT.value
            )
        # Idempotent when same registration already present.
        if len(node.args) >= 2:
            key = _source_segment(span_text, node.args[0]).strip()
            target = _source_segment(span_text, node.args[1]).strip()
            if key.strip("\"'") == registration_name.strip("\"'") and target == registration_target:
                return span_text
        open_paren = span_text.find("(")
        close_paren = _find_closing_paren(span_text, open_paren)
        key_literal = (
            registration_name
            if registration_name[0] in {'"', "'"}
            else json.dumps(registration_name)
        )
        insertion = f"{key_literal}, {registration_target}"
        interior = span_text[open_paren + 1 : close_paren]
        if interior.strip():
            raise AnalyticalChangeTransformUnsupportedError(
                TransformRejectionReason.UNSUPPORTED_SYNTAX.value
            )
        return f"{span_text[: open_paren + 1]}{insertion}{span_text[close_paren:]}"

    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        # REG["name"] = target
        return (
            f"{_source_segment(span_text, node.targets[0])} = {registration_target}"
        )

    raise AnalyticalChangeTransformUnsupportedError(
        TransformRejectionReason.UNSUPPORTED_SYNTAX.value
    )


def _render_add_adapter(
    span_text: str,
    *,
    adapter_expression: str,
    expression_text: str,
) -> str:
    """Wrap a proved expression with a finite adapter call: Adapter(expr)."""

    if not _is_simple_expression(adapter_expression):
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.INVENTED_BEHAVIOR.value
        )
    if not _is_simple_expression(expression_text):
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.INVENTED_BEHAVIOR.value
        )
    expected = f"{adapter_expression.strip()}({expression_text.strip()})"
    if span_text.strip() == expected:
        return span_text
    # Span is the bare expression being adapted.
    if span_text.strip() != expression_text.strip():
        # Or already an adapter application to a different source — reject.
        node = _single_expr(span_text)
        if isinstance(node, ast.Call) and not _call_has_splat(node):
            func = _source_segment(span_text, node.func).strip()
            if func == adapter_expression.strip() and len(node.args) == 1:
                inner = _source_segment(span_text, node.args[0]).strip()
                if inner == expression_text.strip():
                    return span_text
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.EXPRESSION_MISMATCH.value
        )
    return expected


def _render_update_constructor(
    span_text: str,
    *,
    parameter_name: str,
    expression_text: str,
    keyword_style: bool,
) -> str:
    node = _single_expr(span_text)
    if not isinstance(node, ast.Call):
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.UNSUPPORTED_SYNTAX.value
        )
    # Constructor: Name(...) or Attribute(...) — not an arbitrary call chain.
    if not isinstance(node.func, (ast.Name, ast.Attribute)):
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.UNSUPPORTED_SYNTAX.value
        )
    return _render_add_argument(
        span_text,
        parameter_name=parameter_name,
        expression_text=expression_text,
        keyword_style=keyword_style,
        insert_position=None,
    )


def _render_field_mapping(span_text: str, mappings: Sequence[FieldMapping]) -> str:
    if not mappings:
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.NON_TOTAL_MAPPING.value
        )
    befores = [item.before for item in mappings]
    if len(set(befores)) != len(befores):
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.NON_TOTAL_MAPPING.value
        )
    try:
        data = json.loads(span_text)
    except json.JSONDecodeError as exc:
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.UNSUPPORTED_SYNTAX.value
        ) from exc
    if not isinstance(data, dict):
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.UNSUPPORTED_SYNTAX.value
        )
    # Totality: every mapping.before must exist unless before == after (identity).
    missing = [item.before for item in mappings if item.before not in data and item.before != item.after]
    if missing:
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.NON_TOTAL_MAPPING.value
        )
    result = dict(data)
    for item in mappings:
        if item.before in result:
            value = result.pop(item.before)
            result[item.after] = value
        elif item.before == item.after:
            continue
        else:
            # Explicit new key only when after not present and before absent —
            # still non-total without a proved default value.
            raise AnalyticalChangeTransformUnsupportedError(
                TransformRejectionReason.INVENTED_BEHAVIOR.value
            )
    # Deterministic JSON: sorted keys, stable separators, UTF-8, trailing layout.
    return json.dumps(result, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Authority checks
# ---------------------------------------------------------------------------


def _verify_value_mapping(
    site: TransformSite,
    value_mapping: ValueMappingProof | None,
) -> tuple[str, ...]:
    """Return rejection reasons; empty means the mapping authorizes the site."""

    if site.kind not in _VALUE_BEARING_KINDS and site.kind not in _MAPPING_KINDS:
        return ()
    if value_mapping is None:
        # Wiring-only kinds may omit value mappings; value-bearing may not.
        if site.kind in _VALUE_BEARING_KINDS:
            return (TransformRejectionReason.NO_CODE_AUTHORITY.value,)
        return ()
    if not isinstance(value_mapping, ValueMappingProof):
        raise AnalyticalChangeTransformError("value_mapping must be ValueMappingProof")
    reasons: list[str] = []
    if value_mapping.disposition is not SynthesisDisposition.UNIQUE_PROVED:
        reasons.append(TransformRejectionReason.NO_CODE_AUTHORITY.value)
    if len(value_mapping.proved_candidate_ids) != 1:
        reasons.append(TransformRejectionReason.NO_CODE_AUTHORITY.value)
    if site.expression_ref and value_mapping.expression_ref:
        if site.expression_ref != value_mapping.expression_ref:
            reasons.append(TransformRejectionReason.EXPRESSION_MISMATCH.value)
    elif site.kind in _VALUE_BEARING_KINDS and not (
        site.expression_ref or value_mapping.expression_ref
    ):
        reasons.append(TransformRejectionReason.MISSING_PROOF.value)
    # Root binding when the proof carries repository/tree/toolchain/policy.
    if value_mapping.repository_id and value_mapping.repository_id != site.roots.repository_id:
        reasons.append(TransformRejectionReason.ROOT_MISMATCH.value)
    if value_mapping.tree_id and value_mapping.tree_id not in {
        site.roots.base_tree_id,
        site.roots.candidate_tree_id,
    }:
        reasons.append(TransformRejectionReason.ROOT_MISMATCH.value)
    if value_mapping.toolchain_id and value_mapping.toolchain_id != site.roots.toolchain_id:
        reasons.append(TransformRejectionReason.ROOT_MISMATCH.value)
    if value_mapping.policy_id and value_mapping.policy_id != site.roots.policy_id:
        reasons.append(TransformRejectionReason.ROOT_MISMATCH.value)
    # When facets exist, require full reconstruction authority.
    if value_mapping.facet_results and not value_mapping.code_authority:
        reasons.append(TransformRejectionReason.NO_CODE_AUTHORITY.value)
    return tuple(dict.fromkeys(reasons))


def _verify_obligation(
    site: TransformSite,
    obligation: ConsumerMigrationObligation | None,
) -> tuple[str, ...]:
    if obligation is None:
        return ()
    if not isinstance(obligation, ConsumerMigrationObligation):
        raise AnalyticalChangeTransformError(
            "obligation must be ConsumerMigrationObligation"
        )
    reasons: list[str] = []
    if obligation.obligation_id not in site.obligation_ids:
        reasons.append(TransformRejectionReason.SCOPE_ESCAPE.value)
    if obligation.roots.to_dict() != site.roots.to_dict():
        reasons.append(TransformRejectionReason.ROOT_MISMATCH.value)
    if obligation.node.path != site.span.path:
        # Threading may involve multi-path routes checked separately.
        if site.kind is not TransformKind.THREAD_PARAMETER:
            reasons.append(TransformRejectionReason.SCOPE_ESCAPE.value)
    return tuple(dict.fromkeys(reasons))


def _verify_target_decision(
    site: TransformSite,
    decision: RepairTargetDecision | None,
) -> tuple[str, ...]:
    if decision is None:
        return ()
    if not isinstance(decision, RepairTargetDecision):
        raise AnalyticalChangeTransformError("decision must be RepairTargetDecision")
    if decision.disposition is not DecisionDisposition.ADMITTED:
        return (TransformRejectionReason.PATH_NOT_AUTHORIZED.value,)
    writes = set(decision.permitted_write_paths)
    if site.span.path not in writes:
        return (TransformRejectionReason.PATH_NOT_AUTHORIZED.value,)
    return ()


def _verify_dependencies(site: TransformSite) -> tuple[str, ...]:
    """Reject transforms that would introduce undeclared dependencies."""

    if not site.allowed_dependency_paths:
        return ()
    allowed = set(site.allowed_dependency_paths)
    # Import module path projected as dotted → slash path for membership.
    if site.kind is TransformKind.ADD_IMPORT and site.import_module:
        projected = site.import_module.replace(".", "/") + ".py"
        # Allow either exact path membership or package prefix admission.
        if projected not in allowed and site.import_module not in allowed:
            if not any(
                projected.startswith(item.rstrip("/") + "/")
                or item == site.import_module
                for item in allowed
            ):
                return (TransformRejectionReason.NEW_DEPENDENCY.value,)
    return ()


def _validate_python_shape(span_text: str, *, path: str) -> None:
    """Use ProgramASTAdapter as a fail-closed Python parser admission gate."""

    if span_text == "":
        return
    # Zero-length insert points are valid for pure import insertion.
    result = adapt_python_source(span_text, path=path)
    if result.status == "malformed":
        raise AnalyticalChangeTransformUnsupportedError(
            TransformRejectionReason.UNSUPPORTED_SYNTAX.value
        )


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


class AnalyticalChangeTransformer:
    """Render closed analytical transforms; never execute repository edits."""

    INTERFACE: ClassVar[str] = ANALYTICAL_CHANGE_TRANSFORMER_INTERFACE

    def render(
        self,
        site: TransformSite,
        *,
        value_mapping: ValueMappingProof | None = None,
        obligation: ConsumerMigrationObligation | None = None,
        decision: RepairTargetDecision | None = None,
    ) -> TransformRenderReceipt:
        if not isinstance(site, TransformSite):
            raise AnalyticalChangeTransformError("site must be TransformSite")

        rejection: list[str] = []
        rejection.extend(_verify_value_mapping(site, value_mapping))
        rejection.extend(_verify_obligation(site, obligation))
        rejection.extend(_verify_target_decision(site, decision))
        rejection.extend(_verify_dependencies(site))

        proof_refs = list(site.proof_refs)
        if value_mapping is not None and value_mapping.proof_id:
            if value_mapping.proof_id not in proof_refs:
                proof_refs.append(value_mapping.proof_id)
        if not proof_refs and not rejection:
            rejection.append(TransformRejectionReason.MISSING_PROOF.value)

        expression_refs: tuple[str, ...] = ()
        if site.expression_ref:
            expression_refs = (site.expression_ref,)
        elif value_mapping is not None and value_mapping.expression_ref:
            expression_refs = (value_mapping.expression_ref,)

        if rejection:
            return self._reject(site, tuple(rejection), proof_refs=tuple(proof_refs))

        try:
            replacement, imports = self._render_replacement(site)
        except AnalyticalChangeTransformUnsupportedError as exc:
            reason = str(exc).split(":", 1)[0].strip() or (
                TransformRejectionReason.UNSUPPORTED_SYNTAX.value
            )
            if reason not in {item.value for item in TransformRejectionReason}:
                reason = TransformRejectionReason.UNSUPPORTED_SYNTAX.value
            return self._reject(site, (reason,), proof_refs=tuple(proof_refs))
        except AnalyticalChangeTransformAuthorityError as exc:
            reason = str(exc).split(":", 1)[0].strip()
            if reason not in {item.value for item in TransformRejectionReason}:
                reason = TransformRejectionReason.SCOPE_ESCAPE.value
            return self._reject(site, (reason,), proof_refs=tuple(proof_refs))

        edit = TransformEdit(
            path=site.span.path,
            start=site.span.start,
            end=site.span.end,
            artifact_id=site.span.artifact_id,
            before_hash=site.span.before_hash,
            replacement=replacement,
            expected_after_hash=_sha256_text(replacement),
        )
        transform = AnalyticalTransform(
            roots=site.roots,
            transform_id=f"transform:{site.site_id}",
            kind=site.kind,
            disposition=TransformDisposition.ADMITTED,
            obligation_ids=site.obligation_ids,
            target_paths=(site.span.path,),
            expression_refs=expression_refs,
            proof_refs=tuple(proof_refs),
            dependency_transform_ids=site.dependency_transform_ids,
        )
        return TransformRenderReceipt(
            transform=transform,
            site_id=site.site_id,
            edits=(edit,),
            postcondition_refs=site.postcondition_refs,
            import_statements=imports,
        )

    def render_many(
        self,
        sites: Sequence[TransformSite],
        *,
        value_mappings: Mapping[str, ValueMappingProof] | None = None,
        obligations: Mapping[str, ConsumerMigrationObligation] | None = None,
        decisions: Mapping[str, RepairTargetDecision] | None = None,
    ) -> TransformBatchReceipt:
        if isinstance(sites, (str, bytes, bytearray)) or not isinstance(sites, Sequence):
            raise AnalyticalChangeTransformError("sites must be a sequence")
        if len(sites) > MAX_SITES:
            raise AnalyticalChangeTransformError("sites exceed the batch bound")
        if not sites:
            raise AnalyticalChangeTransformError("sites must not be empty")
        if not all(isinstance(item, TransformSite) for item in sites):
            raise AnalyticalChangeTransformError("sites must contain TransformSite values")

        roots = sites[0].roots
        for item in sites:
            if item.roots.to_dict() != roots.to_dict():
                raise AnalyticalChangeTransformAuthorityError(
                    TransformRejectionReason.ROOT_MISMATCH.value
                )

        mappings = value_mappings or {}
        obligs = obligations or {}
        decs = decisions or {}

        # Deterministic order: content-addressed site_id sort (never input order).
        ordered = tuple(sorted(sites, key=lambda item: (item.site_id, item.content_id)))
        receipts: list[TransformRenderReceipt] = []
        for site in ordered:
            # Thread parameter: require every route hop to be present in the batch.
            if site.kind is TransformKind.THREAD_PARAMETER:
                route = set(site.route_site_ids)
                if not route or len(route) > MAX_ROUTE_HOPS:
                    receipts.append(
                        self._reject(
                            site,
                            (TransformRejectionReason.UNSUPPORTED_SYNTAX.value,),
                            proof_refs=site.proof_refs,
                        )
                    )
                    continue
                present = {item.site_id for item in ordered}
                if not route.issubset(present):
                    receipts.append(
                        self._reject(
                            site,
                            (TransformRejectionReason.SCOPE_ESCAPE.value,),
                            proof_refs=site.proof_refs,
                        )
                    )
                    continue
            mapping = mappings.get(site.site_id) or mappings.get(
                site.expression_ref, None
            )
            # Also allow lookup by requirement binding via any single mapping.
            if mapping is None and len(mappings) == 1:
                mapping = next(iter(mappings.values()))
            obligation = None
            for oblig_id in site.obligation_ids:
                if oblig_id in obligs:
                    obligation = obligs[oblig_id]
                    break
            decision = decs.get(site.site_id) or decs.get(site.span.path)
            receipts.append(
                self.render(
                    site,
                    value_mapping=mapping,
                    obligation=obligation,
                    decision=decision,
                )
            )
        return TransformBatchReceipt(receipts=tuple(receipts), roots=roots)

    def _render_replacement(
        self, site: TransformSite
    ) -> tuple[str, tuple[str, ...]]:
        span_text = site.span.span_text
        kind = site.kind
        imports: tuple[str, ...] = ()

        if kind not in _MAPPING_KINDS and kind is not TransformKind.ADD_IMPORT:
            _validate_python_shape(span_text, path=site.span.path)

        if kind is TransformKind.ADD_ARGUMENT:
            replacement = _render_add_argument(
                span_text,
                parameter_name=site.parameter_name,
                expression_text=site.expression_text,
                keyword_style=site.keyword_style,
                insert_position=site.insert_position,
            )
        elif kind is TransformKind.RENAME_ARGUMENT:
            previous = site.previous_parameter_name
            new_name = site.parameter_name
            if not previous or not new_name:
                raise AnalyticalChangeTransformUnsupportedError(
                    TransformRejectionReason.INVALID_IDENTIFIER.value
                )
            replacement = _render_rename_argument(
                span_text,
                previous_name=previous,
                new_name=new_name,
            )
        elif kind is TransformKind.REORDER_ARGUMENT:
            replacement = _render_reorder_argument(
                span_text, argument_order=site.argument_order
            )
        elif kind is TransformKind.THREAD_PARAMETER:
            # Threading renders the local call-site hop as add-argument; the
            # batch admits the full route.  Dependency ids bind the chain.
            replacement = _render_add_argument(
                span_text,
                parameter_name=site.parameter_name,
                expression_text=site.expression_text,
                keyword_style=site.keyword_style,
                insert_position=site.insert_position,
            )
        elif kind is TransformKind.ADD_IMPORT:
            replacement = _render_add_import(
                span_text,
                module=site.import_module,
                name=site.import_name,
            )
            imports = (replacement.splitlines()[-1],)
        elif kind is TransformKind.ADD_EXPORT:
            replacement = _render_add_export(
                span_text, export_name=site.export_name or site.parameter_name
            )
        elif kind is TransformKind.ADD_REGISTRATION:
            replacement = _render_add_registration(
                span_text,
                registration_name=site.registration_name or site.parameter_name,
                registration_target=site.registration_target or site.expression_text,
            )
        elif kind is TransformKind.ADD_ADAPTER:
            replacement = _render_add_adapter(
                span_text,
                adapter_expression=site.adapter_expression,
                expression_text=site.expression_text,
            )
        elif kind is TransformKind.UPDATE_CONSTRUCTOR:
            replacement = _render_update_constructor(
                span_text,
                parameter_name=site.parameter_name,
                expression_text=site.expression_text,
                keyword_style=site.keyword_style,
            )
        elif kind in _MAPPING_KINDS:
            replacement = _render_field_mapping(span_text, site.field_mappings)
        else:
            raise AnalyticalChangeTransformUnsupportedError(
                TransformRejectionReason.UNSUPPORTED_KIND.value
            )

        # Byte-stable: replacement is a pure function of span_text + closed params.
        return replacement, imports

    def _reject(
        self,
        site: TransformSite,
        reasons: tuple[str, ...],
        *,
        proof_refs: tuple[str, ...],
    ) -> TransformRenderReceipt:
        unique_reasons = tuple(dict.fromkeys(reasons))
        transform = AnalyticalTransform(
            roots=site.roots,
            transform_id=f"transform:{site.site_id}",
            kind=site.kind,
            disposition=TransformDisposition.REJECTED,
            obligation_ids=site.obligation_ids,
            target_paths=(),
            expression_refs=(site.expression_ref,) if site.expression_ref else (),
            proof_refs=proof_refs,
            dependency_transform_ids=site.dependency_transform_ids,
            rejection_reasons=unique_reasons,
        )
        return TransformRenderReceipt(
            transform=transform,
            site_id=site.site_id,
            edits=(),
            postcondition_refs=site.postcondition_refs,
            rejection_reasons=unique_reasons,
        )


def render_analytical_transform(
    site: TransformSite,
    *,
    value_mapping: ValueMappingProof | None = None,
    obligation: ConsumerMigrationObligation | None = None,
    decision: RepairTargetDecision | None = None,
) -> TransformRenderReceipt:
    """Module-level convenience wrapper around :class:`AnalyticalChangeTransformer`."""

    return AnalyticalChangeTransformer().render(
        site,
        value_mapping=value_mapping,
        obligation=obligation,
        decision=decision,
    )


def make_span(
    path: str,
    source: str,
    *,
    start: int = 0,
    artifact_id: str = "",
) -> TransformSourceSpan:
    """Build a span over ``source[start:start+len(source)]`` with a bound hash.

    When ``start`` is 0 and the span covers the whole string, ``end`` equals
    ``len(source)``.  Callers that embed a slice in a larger file pass the true
    absolute offsets while still supplying only the slice text.
    """

    text = source
    end = start + len(text)
    return TransformSourceSpan(
        path=path,
        start=start,
        end=end,
        artifact_id=artifact_id or f"blob:{_sha256_text(text)[7:23]}",
        span_text=text,
        before_hash=_sha256_text(text),
    )


__all__ = (
    "ANALYTICAL_CHANGE_TRANSFORMER_INTERFACE",
    "AnalyticalChangeTransformAuthorityError",
    "AnalyticalChangeTransformError",
    "AnalyticalChangeTransformUnsupportedError",
    "AnalyticalChangeTransformer",
    "FieldMapping",
    "PRODUCER_ID",
    "TransformBatchReceipt",
    "TransformEdit",
    "TransformRejectionReason",
    "TransformRenderReceipt",
    "TransformSite",
    "TransformSourceSpan",
    "make_span",
    "render_analytical_transform",
)
