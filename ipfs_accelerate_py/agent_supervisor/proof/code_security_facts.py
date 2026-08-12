"""Canonical, non-authoritative security facts extracted from changed code.

This module is deliberately on the observation side of the supervisor's
authority boundary.  It parses complete before/after sources, attributes facts
only to changed AST scopes, and binds every fact to the exact tree, blob, diff,
and AST identities from which it was derived.  It does not evaluate policy,
issue permits, or interpret comments and string literals as declarations.

The Python extractor is intentionally small and deterministic.  It observes
syntax (calls, imports, assignments, returns, guards, and selected
capabilities), not programmer intent.  Unsupported languages, missing source,
parse failures, and dynamic calls are retained as explicit diagnostics instead
of being silently dropped or converted into guessed facts.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Final


CODE_SECURITY_FACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/code-security-fact@1"
)
CODE_SECURITY_FACT_SET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/code-security-fact-set@1"
)
CODE_SECURITY_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/code-security-binding@1"
)
CODE_SECURITY_SCOPE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/code-security-source-scope@1"
)
CODE_SECURITY_DIAGNOSTIC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/code-security-diagnostic@1"
)
CODE_SECURITY_EXTRACTOR_VERSION: Final[int] = 1

_MAX_SOURCE_BYTES: Final[int] = 8 * 1024 * 1024
_MAX_FACTS: Final[int] = 100_000
_MAX_TEXT_BYTES: Final[int] = 8_192
_PYTHON_SUFFIXES: Final[frozenset[str]] = frozenset({".py", ".pyi"})


class CodeSecurityFactError(ValueError):
    """A code-security extraction input or canonical contract is malformed."""


class CodeSecurityFactKind(str, Enum):
    """Orthogonal dimensions observed in changed code."""

    ACTION = "action"
    TARGET = "target"
    DATA_FLOW = "data_flow"
    EFFECT = "effect"
    EXPECTED_EFFECT = "effect"
    CAPABILITY = "capability"
    GUARD = "guard"
    LANGUAGE = "language"
    SOURCE_SCOPE = "source_scope"


class CodeSecurityDelta(str, Enum):
    """Which side of a changed AST scope produced a fact."""

    ADDED = "added"
    REMOVED = "removed"


class CodeSecurityExtractionStatus(str, Enum):
    """Completeness of one extraction result."""

    EXTRACTED = "extracted"
    PARTIAL = "partial"
    EMPTY = "empty"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"
    INVALID = "invalid"


class CodeSecurityDiagnosticCode(str, Enum):
    """Stable reason codes for extraction that could not be exact."""

    UNSUPPORTED_LANGUAGE = "unsupported_language"
    MISSING_SOURCE = "missing_source"
    SOURCE_TOO_LARGE = "source_too_large"
    PARSE_ERROR = "parse_error"
    DYNAMIC_CALL_TARGET = "dynamic_call_target"
    FACT_LIMIT_EXCEEDED = "fact_limit_exceeded"
    NO_SEMANTIC_CHANGE = "no_semantic_change"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise CodeSecurityFactError(f"{name} must be a string")
    if value != value.strip() or "\x00" in value:
        raise CodeSecurityFactError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not value:
        raise CodeSecurityFactError(f"{name} must not be empty")
    if len(value.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise CodeSecurityFactError(f"{name} exceeds {_MAX_TEXT_BYTES} UTF-8 bytes")
    return value


def _path(value: Any, name: str, *, required: bool = True) -> str:
    raw = _text(value, name, required=required).replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    if not raw:
        if required:
            raise CodeSecurityFactError(f"{name} must not be empty")
        return ""
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise CodeSecurityFactError(f"{name} must remain repository-relative")
    return candidate.as_posix()


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CodeSecurityFactError("value is not canonical JSON") from exc


def _identity(namespace: str, value: Any) -> str:
    return f"{namespace}:sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _source_sha256(source: str) -> str:
    return "sha256:" + hashlib.sha256(
        source.encode("utf-8", errors="surrogatepass")
    ).hexdigest()


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    try:
        return value if isinstance(value, enum_type) else enum_type(str(value))
    except ValueError as exc:
        raise CodeSecurityFactError(f"unsupported {name}: {value!r}") from exc


def _verify_claim(payload: Mapping[str, Any], name: str, actual: str) -> None:
    claimed = str(payload.get(name) or "")
    if claimed and claimed != actual:
        raise CodeSecurityFactError(f"{name} does not match its canonical payload")


def _contract(
    payload: Mapping[str, Any],
    *,
    schema: str,
    allowed: set[str],
    name: str,
) -> None:
    supplied_schema = payload.get("schema")
    if supplied_schema not in (None, "", schema):
        raise CodeSecurityFactError(f"unsupported {name} schema")
    supplied_version = payload.get("extractor_version")
    if supplied_version not in (None, CODE_SECURITY_EXTRACTOR_VERSION):
        raise CodeSecurityFactError(f"unsupported {name} extractor version")
    unknown = set(payload) - allowed
    if unknown:
        raise CodeSecurityFactError(
            f"{name} contains unsupported fields: "
            + ", ".join(sorted(map(str, unknown)))
        )


@dataclass(frozen=True)
class ChangedCodeDiff:
    """Complete, identity-bound before/after input for one changed file.

    ``tree_id`` is the candidate repository tree and ``diff_id`` identifies
    the complete candidate diff.  Blob IDs may be Git object IDs, CIDs, or
    another repository identity.  When omitted for a present text side, a
    SHA-256 source identity is used.  Source text is never serialized by any
    output contract.
    """

    tree_id: str
    diff_id: str
    old_path: str = ""
    new_path: str = ""
    before_source: str | None = None
    after_source: str | None = None
    before_blob_id: str = ""
    after_blob_id: str = ""
    language: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(self, "diff_id", _text(self.diff_id, "diff_id"))
        old_path = _path(self.old_path, "old_path", required=False)
        new_path = _path(self.new_path, "new_path", required=False)
        if not old_path and not new_path:
            raise CodeSecurityFactError("a changed diff requires old_path or new_path")
        object.__setattr__(self, "old_path", old_path)
        object.__setattr__(self, "new_path", new_path)
        for name in ("before_source", "after_source"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise CodeSecurityFactError(f"{name} must be text or None")
        before_blob = _text(
            self.before_blob_id, "before_blob_id", required=False
        )
        after_blob = _text(self.after_blob_id, "after_blob_id", required=False)
        if self.before_source is not None and not before_blob:
            before_blob = _source_sha256(self.before_source)
        if self.after_source is not None and not after_blob:
            after_blob = _source_sha256(self.after_source)
        object.__setattr__(self, "before_blob_id", before_blob)
        object.__setattr__(self, "after_blob_id", after_blob)
        language = str(self.language or "").strip().lower()
        if not language:
            suffix = PurePosixPath(new_path or old_path).suffix.lower()
            language = "python" if suffix in _PYTHON_SUFFIXES else suffix.lstrip(".")
        object.__setattr__(self, "language", _text(language, "language"))

    @property
    def path(self) -> str:
        return self.new_path or self.old_path

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChangedCodeDiff":
        allowed = {
            "tree_id",
            "repository_tree_id",
            "diff_id",
            "old_path",
            "new_path",
            "path",
            "before_source",
            "after_source",
            "before_blob_id",
            "after_blob_id",
            "language",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise CodeSecurityFactError(
                "changed diff contains unsupported fields: "
                + ", ".join(sorted(map(str, unknown)))
            )
        return cls(
            tree_id=str(payload.get("tree_id") or payload.get("repository_tree_id") or ""),
            diff_id=str(payload.get("diff_id") or ""),
            old_path=str(payload.get("old_path") or ""),
            new_path=str(payload.get("new_path") or payload.get("path") or ""),
            before_source=payload.get("before_source"),  # type: ignore[arg-type]
            after_source=payload.get("after_source"),  # type: ignore[arg-type]
            before_blob_id=str(payload.get("before_blob_id") or ""),
            after_blob_id=str(payload.get("after_blob_id") or ""),
            language=str(payload.get("language") or ""),
        )


@dataclass(frozen=True)
class CodeSecurityIdentityBinding:
    """Exact source identities carried by every emitted fact."""

    tree_id: str
    diff_id: str
    blob_id: str
    source_sha256: str
    ast_id: str

    def __post_init__(self) -> None:
        for name in ("tree_id", "diff_id", "blob_id", "source_sha256", "ast_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))

    @property
    def binding_id(self) -> str:
        return _identity("code-security-binding", self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": CODE_SECURITY_BINDING_SCHEMA,
            "extractor_version": CODE_SECURITY_EXTRACTOR_VERSION,
            "tree_id": self.tree_id,
            "diff_id": self.diff_id,
            "blob_id": self.blob_id,
            "source_sha256": self.source_sha256,
            "ast_id": self.ast_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"binding_id": self.binding_id, **self._payload()}

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "CodeSecurityIdentityBinding":
        _contract(
            payload,
            schema=CODE_SECURITY_BINDING_SCHEMA,
            allowed={
                "schema",
                "extractor_version",
                "binding_id",
                "tree_id",
                "diff_id",
                "blob_id",
                "source_sha256",
                "ast_id",
            },
            name="code-security binding",
        )
        result = cls(
            tree_id=str(payload.get("tree_id") or ""),
            diff_id=str(payload.get("diff_id") or ""),
            blob_id=str(payload.get("blob_id") or ""),
            source_sha256=str(payload.get("source_sha256") or ""),
            ast_id=str(payload.get("ast_id") or ""),
        )
        _verify_claim(payload, "binding_id", result.binding_id)
        return result


@dataclass(frozen=True)
class CodeSecuritySourceScope:
    """The changed AST unit and exact line interval attributed to a fact."""

    path: str
    symbol: str
    line_start: int
    line_end: int
    delta: CodeSecurityDelta

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path, "path"))
        object.__setattr__(self, "symbol", _text(self.symbol, "symbol"))
        if (
            isinstance(self.line_start, bool)
            or isinstance(self.line_end, bool)
            or not isinstance(self.line_start, int)
            or not isinstance(self.line_end, int)
            or self.line_start < 1
            or self.line_end < self.line_start
        ):
            raise CodeSecurityFactError("source scope requires a valid 1-based line range")
        object.__setattr__(
            self, "delta", _enum(self.delta, CodeSecurityDelta, "source delta")
        )

    @property
    def scope_id(self) -> str:
        return _identity("code-security-scope", self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": CODE_SECURITY_SCOPE_SCHEMA,
            "path": self.path,
            "symbol": self.symbol,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "delta": self.delta.value,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"scope_id": self.scope_id, **self._payload()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodeSecuritySourceScope":
        _contract(
            payload,
            schema=CODE_SECURITY_SCOPE_SCHEMA,
            allowed={
                "schema",
                "scope_id",
                "path",
                "symbol",
                "line_start",
                "line_end",
                "delta",
            },
            name="code-security source scope",
        )
        result = cls(
            path=str(payload.get("path") or ""),
            symbol=str(payload.get("symbol") or ""),
            line_start=payload.get("line_start", 0),  # type: ignore[arg-type]
            line_end=payload.get("line_end", 0),  # type: ignore[arg-type]
            delta=payload.get("delta", ""),  # type: ignore[arg-type]
        )
        _verify_claim(payload, "scope_id", result.scope_id)
        return result


@dataclass(frozen=True)
class CodeSecurityFact:
    """One canonical syntax observation; never an authorization grant."""

    kind: CodeSecurityFactKind
    value: str
    binding: CodeSecurityIdentityBinding
    source_scope: CodeSecuritySourceScope

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, CodeSecurityFactKind, "fact kind"))
        object.__setattr__(self, "value", _text(self.value, "fact value"))
        if not isinstance(self.binding, CodeSecurityIdentityBinding):
            raise CodeSecurityFactError("fact binding must be canonical")
        if not isinstance(self.source_scope, CodeSecuritySourceScope):
            raise CodeSecurityFactError("fact source_scope must be canonical")

    @property
    def fact_id(self) -> str:
        return _identity("code-security-fact", self._payload())

    @property
    def grants_execution_authority(self) -> bool:
        return False

    @property
    def authorizes_completion(self) -> bool:
        return False

    @property
    def establishes_generated_code_correctness(self) -> bool:
        return False

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def grants_authority(self) -> bool:
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": CODE_SECURITY_FACT_SCHEMA,
            "extractor_version": CODE_SECURITY_EXTRACTOR_VERSION,
            "kind": self.kind.value,
            "value": self.value,
            "binding": self.binding.to_dict(),
            "source_scope": self.source_scope.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            **self._payload(),
            "grants_execution_authority": False,
            "authorizes_completion": False,
            "establishes_generated_code_correctness": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodeSecurityFact":
        _contract(
            payload,
            schema=CODE_SECURITY_FACT_SCHEMA,
            allowed={
                "schema",
                "extractor_version",
                "fact_id",
                "kind",
                "value",
                "binding",
                "source_scope",
                "grants_execution_authority",
                "authorizes_completion",
                "establishes_generated_code_correctness",
            },
            name="code-security fact",
        )
        for field_name in (
            "grants_execution_authority",
            "authorizes_completion",
            "establishes_generated_code_correctness",
        ):
            if payload.get(field_name) not in (None, False):
                raise CodeSecurityFactError(
                    f"code-security facts cannot set {field_name}"
                )
        binding = payload.get("binding")
        scope = payload.get("source_scope")
        if not isinstance(binding, Mapping) or not isinstance(scope, Mapping):
            raise CodeSecurityFactError("fact requires binding and source_scope")
        result = cls(
            kind=payload.get("kind", ""),  # type: ignore[arg-type]
            value=str(payload.get("value") or ""),
            binding=CodeSecurityIdentityBinding.from_dict(binding),
            source_scope=CodeSecuritySourceScope.from_dict(scope),
        )
        _verify_claim(payload, "fact_id", result.fact_id)
        return result


@dataclass(frozen=True)
class CodeSecurityDiagnostic:
    """Bound explanation for an unsupported or ambiguous observation."""

    code: CodeSecurityDiagnosticCode
    message: str
    tree_id: str
    diff_id: str
    path: str
    symbol: str = ""
    line: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "code", _enum(self.code, CodeSecurityDiagnosticCode, "diagnostic code")
        )
        object.__setattr__(self, "message", _text(self.message, "message"))
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(self, "diff_id", _text(self.diff_id, "diff_id"))
        object.__setattr__(self, "path", _path(self.path, "path"))
        object.__setattr__(
            self, "symbol", _text(self.symbol, "symbol", required=False)
        )
        if isinstance(self.line, bool) or not isinstance(self.line, int) or self.line < 0:
            raise CodeSecurityFactError("diagnostic line must be a non-negative integer")

    @property
    def diagnostic_id(self) -> str:
        return _identity("code-security-diagnostic", self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": CODE_SECURITY_DIAGNOSTIC_SCHEMA,
            "code": self.code.value,
            "message": self.message,
            "tree_id": self.tree_id,
            "diff_id": self.diff_id,
            "path": self.path,
            "symbol": self.symbol,
            "line": self.line,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"diagnostic_id": self.diagnostic_id, **self._payload()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodeSecurityDiagnostic":
        _contract(
            payload,
            schema=CODE_SECURITY_DIAGNOSTIC_SCHEMA,
            allowed={
                "schema",
                "diagnostic_id",
                "code",
                "message",
                "tree_id",
                "diff_id",
                "path",
                "symbol",
                "line",
            },
            name="code-security diagnostic",
        )
        result = cls(
            code=payload.get("code", ""),  # type: ignore[arg-type]
            message=str(payload.get("message") or ""),
            tree_id=str(payload.get("tree_id") or ""),
            diff_id=str(payload.get("diff_id") or ""),
            path=str(payload.get("path") or ""),
            symbol=str(payload.get("symbol") or ""),
            line=payload.get("line", 0),  # type: ignore[arg-type]
        )
        _verify_claim(payload, "diagnostic_id", result.diagnostic_id)
        return result


@dataclass(frozen=True)
class CodeSecurityFactSet:
    """Deterministically ordered facts and loss-aware extraction diagnostics."""

    tree_id: str
    diff_id: str
    status: CodeSecurityExtractionStatus
    facts: tuple[CodeSecurityFact, ...] = ()
    diagnostics: tuple[CodeSecurityDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(self, "diff_id", _text(self.diff_id, "diff_id"))
        object.__setattr__(
            self,
            "status",
            _enum(self.status, CodeSecurityExtractionStatus, "extraction status"),
        )
        if any(not isinstance(item, CodeSecurityFact) for item in self.facts):
            raise CodeSecurityFactError("fact set contains a non-canonical fact")
        if any(
            not isinstance(item, CodeSecurityDiagnostic) for item in self.diagnostics
        ):
            raise CodeSecurityFactError("fact set contains a non-canonical diagnostic")
        facts = {item.fact_id: item for item in self.facts}
        diagnostics = {item.diagnostic_id: item for item in self.diagnostics}
        if len(facts) > _MAX_FACTS:
            raise CodeSecurityFactError("fact set exceeds its hard fact bound")
        for item in (*facts.values(), *diagnostics.values()):
            item_tree_id = (
                item.tree_id
                if isinstance(item, CodeSecurityDiagnostic)
                else item.binding.tree_id
            )
            item_diff_id = (
                item.diff_id
                if isinstance(item, CodeSecurityDiagnostic)
                else item.binding.diff_id
            )
            if item_tree_id != self.tree_id:
                raise CodeSecurityFactError("fact-set tree binding mismatch")
            if item_diff_id != self.diff_id:
                raise CodeSecurityFactError("fact-set diff binding mismatch")
        object.__setattr__(
            self, "facts", tuple(facts[key] for key in sorted(facts))
        )
        object.__setattr__(
            self,
            "diagnostics",
            tuple(diagnostics[key] for key in sorted(diagnostics)),
        )

    @property
    def fact_set_id(self) -> str:
        return _identity("code-security-fact-set", self._payload())

    @property
    def grants_execution_authority(self) -> bool:
        return False

    @property
    def authorizes_completion(self) -> bool:
        return False

    @property
    def establishes_generated_code_correctness(self) -> bool:
        return False

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def grants_authority(self) -> bool:
        return False

    @property
    def complete(self) -> bool:
        return self.status is CodeSecurityExtractionStatus.EXTRACTED

    @property
    def unsupported(self) -> bool:
        return self.status is CodeSecurityExtractionStatus.UNSUPPORTED

    @property
    def ambiguous(self) -> bool:
        return self.status in {
            CodeSecurityExtractionStatus.AMBIGUOUS,
            CodeSecurityExtractionStatus.PARTIAL,
        } and any(
            item.code is CodeSecurityDiagnosticCode.DYNAMIC_CALL_TARGET
            for item in self.diagnostics
        )

    def by_kind(
        self, kind: CodeSecurityFactKind | str
    ) -> tuple[CodeSecurityFact, ...]:
        normalized = _enum(kind, CodeSecurityFactKind, "fact kind")
        return tuple(item for item in self.facts if item.kind is normalized)

    def _values(self, kind: CodeSecurityFactKind) -> tuple[str, ...]:
        return tuple(sorted({item.value for item in self.by_kind(kind)}))

    @property
    def actions(self) -> tuple[str, ...]:
        return self._values(CodeSecurityFactKind.ACTION)

    @property
    def targets(self) -> tuple[str, ...]:
        return self._values(CodeSecurityFactKind.TARGET)

    @property
    def data_flows(self) -> tuple[str, ...]:
        return self._values(CodeSecurityFactKind.DATA_FLOW)

    @property
    def effects(self) -> tuple[str, ...]:
        return self._values(CodeSecurityFactKind.EFFECT)

    @property
    def expected_effects(self) -> tuple[str, ...]:
        return self.effects

    @property
    def capabilities(self) -> tuple[str, ...]:
        return self._values(CodeSecurityFactKind.CAPABILITY)

    @property
    def guards(self) -> tuple[str, ...]:
        return self._values(CodeSecurityFactKind.GUARD)

    @property
    def languages(self) -> tuple[str, ...]:
        return self._values(CodeSecurityFactKind.LANGUAGE)

    @property
    def source_scopes(self) -> tuple[str, ...]:
        return self._values(CodeSecurityFactKind.SOURCE_SCOPE)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": CODE_SECURITY_FACT_SET_SCHEMA,
            "extractor_version": CODE_SECURITY_EXTRACTOR_VERSION,
            "tree_id": self.tree_id,
            "diff_id": self.diff_id,
            "status": self.status.value,
            "facts": [item.to_dict() for item in self.facts],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_set_id": self.fact_set_id,
            **self._payload(),
            "grants_execution_authority": False,
            "authorizes_completion": False,
            "establishes_generated_code_correctness": False,
        }

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodeSecurityFactSet":
        _contract(
            payload,
            schema=CODE_SECURITY_FACT_SET_SCHEMA,
            allowed={
                "schema",
                "extractor_version",
                "fact_set_id",
                "tree_id",
                "diff_id",
                "status",
                "facts",
                "diagnostics",
                "grants_execution_authority",
                "authorizes_completion",
                "establishes_generated_code_correctness",
            },
            name="code-security fact set",
        )
        for field_name in (
            "grants_execution_authority",
            "authorizes_completion",
            "establishes_generated_code_correctness",
        ):
            if payload.get(field_name) not in (None, False):
                raise CodeSecurityFactError(
                    f"code-security fact sets cannot set {field_name}"
                )
        result = cls(
            tree_id=str(payload.get("tree_id") or ""),
            diff_id=str(payload.get("diff_id") or ""),
            status=payload.get("status", ""),  # type: ignore[arg-type]
            facts=tuple(
                CodeSecurityFact.from_dict(item)
                for item in payload.get("facts") or ()
            ),
            diagnostics=tuple(
                CodeSecurityDiagnostic.from_dict(item)
                for item in payload.get("diagnostics") or ()
            ),
        )
        _verify_claim(payload, "fact_set_id", result.fact_set_id)
        return result

    @classmethod
    def from_json(cls, text: str) -> "CodeSecurityFactSet":
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise CodeSecurityFactError("fact-set JSON is invalid") from exc
        if not isinstance(payload, Mapping):
            raise CodeSecurityFactError("fact-set JSON must contain an object")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class _ASTUnit:
    symbol: str
    node: ast.AST
    digest: str
    line_start: int
    line_end: int
    imports: Mapping[str, str] = field(default_factory=dict)


def _node_digest(node: ast.AST) -> str:
    semantic = ast.dump(node, annotate_fields=True, include_attributes=False)
    return "ast-node:sha256:" + hashlib.sha256(semantic.encode("utf-8")).hexdigest()


def _ast_identity(tree: ast.AST) -> str:
    return _identity(
        "ast",
        {
            "language": "python",
            "semantic_ast": ast.dump(
                tree, annotate_fields=True, include_attributes=False
            ),
        },
    )


def _expression_name(node: ast.AST | None) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expression_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _expression_shape(node: ast.AST | None) -> str:
    """Return structure only; constant bodies cannot become facts."""

    if node is None:
        return "none"
    if isinstance(node, ast.Constant):
        return f"constant:{type(node.value).__name__}"
    if isinstance(node, ast.Name):
        return "name"
    if isinstance(node, ast.Attribute):
        return "attribute"
    if isinstance(node, ast.Call):
        return "call"
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return "collection"
    if isinstance(node, ast.Dict):
        return "mapping"
    if isinstance(node, ast.Subscript):
        return "subscript"
    if isinstance(node, ast.BinOp):
        return "binary_expression"
    if isinstance(node, ast.BoolOp):
        return "boolean_expression"
    if isinstance(node, ast.Compare):
        return "comparison"
    return type(node).__name__.lower()


def _imports(tree: ast.AST) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                result[alias.asname or alias.name.split(".", 1)[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                result[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return result


def _qualify_call(name: str, aliases: Mapping[str, str]) -> str:
    root, dot, suffix = name.partition(".")
    replacement = aliases.get(root, root)
    return f"{replacement}{dot}{suffix}" if dot else replacement


def _scope_units(tree: ast.Module) -> dict[str, _ASTUnit]:
    aliases = _imports(tree)
    units: dict[str, _ASTUnit] = {}

    # Module-level executable statements and class headers are a distinct unit;
    # function bodies are deliberately excluded so stable sibling functions
    # cannot be attributed to a changed function.
    module_body: list[ast.stmt] = []
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(statement, ast.ClassDef):
            module_body.append(
                ast.ClassDef(
                    name=statement.name,
                    bases=statement.bases,
                    keywords=statement.keywords,
                    body=[
                        item
                        for item in statement.body
                        if not isinstance(
                            item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                        )
                    ]
                    or [ast.Pass()],
                    decorator_list=statement.decorator_list,
                    type_params=getattr(statement, "type_params", []),
                )
            )
        else:
            module_body.append(statement)
    module = ast.Module(body=module_body, type_ignores=[])
    end = max((int(getattr(item, "end_lineno", 1) or 1) for item in tree.body), default=1)
    units["<module>"] = _ASTUnit(
        symbol="<module>",
        node=module,
        digest=_node_digest(module),
        line_start=1,
        line_end=end,
        imports=aliases,
    )

    class Collector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []

        def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            symbol = ".".join([*self.scope, node.name])
            units[symbol] = _ASTUnit(
                symbol=symbol,
                node=node,
                digest=_node_digest(node),
                line_start=int(getattr(node, "lineno", 1) or 1),
                line_end=int(getattr(node, "end_lineno", node.lineno) or node.lineno),
                imports=aliases,
            )
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._function(node)

    Collector().visit(tree)
    return units


def _capabilities(callee: str) -> tuple[str, ...]:
    value = callee.lower()
    result = {"call"}
    if value == "open" or value.startswith(
        ("os.remove", "os.unlink", "os.rename", "os.replace", "pathlib.path.")
    ):
        result.add("filesystem")
    if value.startswith(
        ("requests.", "httpx.", "urllib.", "socket.", "aiohttp.")
    ):
        result.add("network")
    if value in {"eval", "exec", "compile"} or value.startswith(
        ("subprocess.", "os.system", "os.exec", "asyncio.create_subprocess")
    ):
        result.add("code_execution")
    if value.startswith(("os.environ", "os.getenv", "os.putenv", "os.unsetenv")):
        result.add("environment")
    return tuple(sorted(result))


def _call_action(callee: str) -> str:
    value = callee.lower()
    if value.startswith(
        (
            "os.remove",
            "os.unlink",
            "pathlib.path.unlink",
            "shutil.rmtree",
        )
    ):
        return "delete"
    if value.startswith(
        (
            "pathlib.path.write",
            "os.rename",
            "os.replace",
            "shutil.copy",
            "shutil.move",
        )
    ):
        return "write"
    if value.startswith(
        (
            "pathlib.path.read",
            "requests.get",
            "httpx.get",
            "urllib.request.urlopen",
        )
    ):
        return "read"
    if value in {"eval", "exec", "compile"} or value.startswith(
        ("subprocess.", "os.system", "os.exec")
    ):
        return "execute"
    return "invoke"


class _UnitFactVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        emit: Any,
        aliases: Mapping[str, str],
        diagnostic: Any,
    ) -> None:
        self.emit = emit
        self.aliases = aliases
        self.diagnostic = diagnostic
        self.guards: list[str] = []
        self._root_seen = False

    def _visit_function_body(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        for statement in node.body:
            self.visit(statement)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if not self._root_seen:
            self._root_seen = True
            self._visit_function_body(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if not self._root_seen:
            self._root_seen = True
            self._visit_function_body(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Nested definitions have their own changed-scope unit.
        return

    def visit_If(self, node: ast.If) -> None:
        guard = _identity(
            "guard",
            ast.dump(node.test, annotate_fields=True, include_attributes=False),
        )
        self.emit(CodeSecurityFactKind.GUARD, guard, node)
        self.guards.append(guard)
        for statement in node.body:
            self.visit(statement)
        self.guards.pop()
        for statement in node.orelse:
            self.visit(statement)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        guard = _identity(
            "guard",
            ast.dump(node.test, annotate_fields=True, include_attributes=False),
        )
        self.emit(CodeSecurityFactKind.GUARD, guard, node)
        self.guards.append(guard)
        self.visit(node.body)
        self.guards.pop()
        self.visit(node.orelse)

    def visit_Call(self, node: ast.Call) -> None:
        raw = _expression_name(node.func)
        if not raw:
            self.diagnostic(node)
            self.generic_visit(node)
            return
        callee = _qualify_call(raw, self.aliases)
        self.emit(CodeSecurityFactKind.ACTION, _call_action(callee), node)
        self.emit(CodeSecurityFactKind.TARGET, callee, node)
        self.emit(CodeSecurityFactKind.EFFECT, "call", node)
        for capability in _capabilities(callee):
            self.emit(CodeSecurityFactKind.CAPABILITY, capability, node)
        for argument in (*node.args, *(item.value for item in node.keywords)):
            self.emit(
                CodeSecurityFactKind.DATA_FLOW,
                f"{_expression_shape(argument)}->argument:{callee}",
                node,
            )
        for guard in self.guards:
            self.emit(CodeSecurityFactKind.GUARD, guard, node)
        self.generic_visit(node)

    def _assignment(
        self, target: ast.AST, value: ast.AST | None, node: ast.AST
    ) -> None:
        target_name = _expression_name(target) or _expression_shape(target)
        self.emit(CodeSecurityFactKind.ACTION, "write", node)
        self.emit(CodeSecurityFactKind.TARGET, target_name, node)
        self.emit(CodeSecurityFactKind.EFFECT, "state_update", node)
        self.emit(CodeSecurityFactKind.CAPABILITY, "state_mutation", node)
        self.emit(
            CodeSecurityFactKind.DATA_FLOW,
            f"{_expression_shape(value)}->{_expression_shape(target)}",
            node,
        )
        for guard in self.guards:
            self.emit(CodeSecurityFactKind.GUARD, guard, node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._assignment(target, node.value, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._assignment(node.target, node.value, node)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._assignment(node.target, node.value, node)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.emit(CodeSecurityFactKind.ACTION, "return", node)
        self.emit(CodeSecurityFactKind.TARGET, "caller", node)
        self.emit(CodeSecurityFactKind.EFFECT, "function_return", node)
        self.emit(
            CodeSecurityFactKind.DATA_FLOW,
            f"{_expression_shape(node.value)}->caller",
            node,
        )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.emit(CodeSecurityFactKind.ACTION, "import", node)
            self.emit(CodeSecurityFactKind.TARGET, alias.name, node)
            self.emit(CodeSecurityFactKind.EFFECT, "module_load", node)
            self.emit(CodeSecurityFactKind.CAPABILITY, "module_import", node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            target = f"{module}.{alias.name}".strip(".")
            self.emit(CodeSecurityFactKind.ACTION, "import", node)
            self.emit(CodeSecurityFactKind.TARGET, target, node)
            self.emit(CodeSecurityFactKind.EFFECT, "module_load", node)
            self.emit(CodeSecurityFactKind.CAPABILITY, "module_import", node)


def _parse_side(
    source: str | None,
    *,
    diff: ChangedCodeDiff,
    side: CodeSecurityDelta,
) -> tuple[ast.Module | None, CodeSecurityDiagnostic | None]:
    path = diff.new_path if side is CodeSecurityDelta.ADDED else diff.old_path
    path = path or diff.path
    if source is None:
        return None, CodeSecurityDiagnostic(
            code=CodeSecurityDiagnosticCode.MISSING_SOURCE,
            message=f"complete {side.value} source is unavailable",
            tree_id=diff.tree_id,
            diff_id=diff.diff_id,
            path=path,
        )
    if len(source.encode("utf-8", errors="surrogatepass")) > _MAX_SOURCE_BYTES:
        return None, CodeSecurityDiagnostic(
            code=CodeSecurityDiagnosticCode.SOURCE_TOO_LARGE,
            message=f"{side.value} source exceeds the hard parser bound",
            tree_id=diff.tree_id,
            diff_id=diff.diff_id,
            path=path,
        )
    try:
        return ast.parse(source), None
    except (SyntaxError, ValueError) as exc:
        line = int(getattr(exc, "lineno", 0) or 0)
        return None, CodeSecurityDiagnostic(
            code=CodeSecurityDiagnosticCode.PARSE_ERROR,
            message=f"{side.value} Python source is not parseable",
            tree_id=diff.tree_id,
            diff_id=diff.diff_id,
            path=path,
            line=line,
        )


def _status(
    facts: Sequence[CodeSecurityFact],
    diagnostics: Sequence[CodeSecurityDiagnostic],
) -> CodeSecurityExtractionStatus:
    codes = {item.code for item in diagnostics}
    if facts and diagnostics:
        return CodeSecurityExtractionStatus.PARTIAL
    if facts:
        return CodeSecurityExtractionStatus.EXTRACTED
    if CodeSecurityDiagnosticCode.PARSE_ERROR in codes:
        return CodeSecurityExtractionStatus.INVALID
    if CodeSecurityDiagnosticCode.DYNAMIC_CALL_TARGET in codes:
        return CodeSecurityExtractionStatus.AMBIGUOUS
    if codes & {
        CodeSecurityDiagnosticCode.UNSUPPORTED_LANGUAGE,
        CodeSecurityDiagnosticCode.MISSING_SOURCE,
        CodeSecurityDiagnosticCode.SOURCE_TOO_LARGE,
    }:
        return CodeSecurityExtractionStatus.UNSUPPORTED
    return CodeSecurityExtractionStatus.EMPTY


def _extract_one_code_security_facts(
    changed_diff: ChangedCodeDiff | Mapping[str, Any],
) -> CodeSecurityFactSet:
    """Extract canonical security facts from exactly one changed file.

    Extraction never reads a repository or follows imports.  Callers must
    supply complete source sides and their external identities, making the
    result deterministic and suitable for later exact request correlation.
    """

    diff = (
        changed_diff
        if isinstance(changed_diff, ChangedCodeDiff)
        else ChangedCodeDiff.from_dict(changed_diff)
    )
    if diff.language not in {"python", "py", "pyi"}:
        diagnostic = CodeSecurityDiagnostic(
            code=CodeSecurityDiagnosticCode.UNSUPPORTED_LANGUAGE,
            message=f"no deterministic extractor is registered for {diff.language}",
            tree_id=diff.tree_id,
            diff_id=diff.diff_id,
            path=diff.path,
        )
        return CodeSecurityFactSet(
            tree_id=diff.tree_id,
            diff_id=diff.diff_id,
            status=CodeSecurityExtractionStatus.UNSUPPORTED,
            diagnostics=(diagnostic,),
        )

    before_tree, before_error = _parse_side(
        diff.before_source, diff=diff, side=CodeSecurityDelta.REMOVED
    )
    after_tree, after_error = _parse_side(
        diff.after_source, diff=diff, side=CodeSecurityDelta.ADDED
    )
    diagnostics = [item for item in (before_error, after_error) if item is not None]

    # A file addition/deletion intentionally lacks one side.  It is exact when
    # the present path and source make that absence unambiguous.
    if not diff.old_path and diff.before_source is None:
        diagnostics = [
            item
            for item in diagnostics
            if not (
                item.code is CodeSecurityDiagnosticCode.MISSING_SOURCE
                and "removed" in item.message
            )
        ]
    if not diff.new_path and diff.after_source is None:
        diagnostics = [
            item
            for item in diagnostics
            if not (
                item.code is CodeSecurityDiagnosticCode.MISSING_SOURCE
                and "added" in item.message
            )
        ]

    if before_tree is None and after_tree is None:
        return CodeSecurityFactSet(
            tree_id=diff.tree_id,
            diff_id=diff.diff_id,
            status=_status((), diagnostics),
            diagnostics=tuple(diagnostics),
        )

    before_units = _scope_units(before_tree) if before_tree is not None else {}
    after_units = _scope_units(after_tree) if after_tree is not None else {}
    changed = {
        symbol
        for symbol in set(before_units) | set(after_units)
        if symbol not in before_units
        or symbol not in after_units
        or before_units[symbol].digest != after_units[symbol].digest
    }
    facts: list[CodeSecurityFact] = []

    def extract_unit(
        unit: _ASTUnit,
        *,
        side: CodeSecurityDelta,
        tree: ast.Module,
        source: str,
        blob_id: str,
        path: str,
    ) -> None:
        binding = CodeSecurityIdentityBinding(
            tree_id=diff.tree_id,
            diff_id=diff.diff_id,
            blob_id=blob_id or _source_sha256(source),
            source_sha256=_source_sha256(source),
            ast_id=_ast_identity(tree),
        )
        def emit(kind: CodeSecurityFactKind, value: str, node: ast.AST) -> None:
            if len(facts) >= _MAX_FACTS:
                return
            line_start = int(getattr(node, "lineno", unit.line_start) or unit.line_start)
            line_end = int(getattr(node, "end_lineno", line_start) or line_start)
            scope = CodeSecuritySourceScope(
                path=path,
                symbol=unit.symbol,
                line_start=max(unit.line_start, line_start),
                line_end=max(max(unit.line_start, line_start), line_end),
                delta=side,
            )
            facts.append(
                CodeSecurityFact(
                    kind=kind,
                    value=value,
                    binding=binding,
                    source_scope=scope,
                )
            )

        # These two dimensions make even a changed constant-only scope
        # observable without pretending the constant has security meaning.
        emit(CodeSecurityFactKind.LANGUAGE, "python", unit.node)
        emit(CodeSecurityFactKind.SOURCE_SCOPE, unit.symbol, unit.node)

        def dynamic_call(node: ast.Call) -> None:
            diagnostics.append(
                CodeSecurityDiagnostic(
                    code=CodeSecurityDiagnosticCode.DYNAMIC_CALL_TARGET,
                    message="dynamic call target cannot be resolved exactly",
                    tree_id=diff.tree_id,
                    diff_id=diff.diff_id,
                    path=path,
                    symbol=unit.symbol,
                    line=int(getattr(node, "lineno", 0) or 0),
                )
            )

        visitor = _UnitFactVisitor(
            emit=emit, aliases=unit.imports, diagnostic=dynamic_call
        )
        visitor.visit(unit.node)

    for symbol in sorted(changed):
        if symbol in before_units and before_tree is not None and diff.before_source is not None:
            extract_unit(
                before_units[symbol],
                side=CodeSecurityDelta.REMOVED,
                tree=before_tree,
                source=diff.before_source,
                blob_id=diff.before_blob_id,
                path=diff.old_path or diff.path,
            )
        if symbol in after_units and after_tree is not None and diff.after_source is not None:
            extract_unit(
                after_units[symbol],
                side=CodeSecurityDelta.ADDED,
                tree=after_tree,
                source=diff.after_source,
                blob_id=diff.after_blob_id,
                path=diff.new_path or diff.path,
            )

    if len(facts) >= _MAX_FACTS:
        diagnostics.append(
            CodeSecurityDiagnostic(
                code=CodeSecurityDiagnosticCode.FACT_LIMIT_EXCEEDED,
                message="fact extraction reached its hard output bound",
                tree_id=diff.tree_id,
                diff_id=diff.diff_id,
                path=diff.path,
            )
        )
    if not changed and not diagnostics:
        diagnostics.append(
            CodeSecurityDiagnostic(
                code=CodeSecurityDiagnosticCode.NO_SEMANTIC_CHANGE,
                message="before and after AST identities contain no semantic change",
                tree_id=diff.tree_id,
                diff_id=diff.diff_id,
                path=diff.path,
            )
        )

    return CodeSecurityFactSet(
        tree_id=diff.tree_id,
        diff_id=diff.diff_id,
        status=_status(facts, diagnostics),
        facts=tuple(facts),
        diagnostics=tuple(diagnostics),
    )


def extract_code_security_facts(
    changed_diff: (
        ChangedCodeDiff
        | Mapping[str, Any]
        | Sequence[ChangedCodeDiff | Mapping[str, Any]]
    ),
) -> CodeSecurityFactSet:
    """Extract one canonical result from a file change or complete multi-file diff.

    Every entry in a multi-file input must bind the same candidate tree and
    diff identity.  This prevents callers from accidentally aggregating facts
    across revisions while preserving the convenient single-file interface.
    """

    if isinstance(changed_diff, (ChangedCodeDiff, Mapping)):
        return _extract_one_code_security_facts(changed_diff)
    if isinstance(changed_diff, (str, bytes, bytearray)) or not isinstance(
        changed_diff, Sequence
    ):
        raise CodeSecurityFactError(
            "changed_diff must be a changed file or a sequence of changed files"
        )
    if not changed_diff:
        raise CodeSecurityFactError("a multi-file changed diff must not be empty")
    results = tuple(_extract_one_code_security_facts(item) for item in changed_diff)
    tree_id = results[0].tree_id
    diff_id = results[0].diff_id
    if any(
        item.tree_id != tree_id or item.diff_id != diff_id for item in results[1:]
    ):
        raise CodeSecurityFactError(
            "all changed files must bind the same tree_id and diff_id"
        )
    facts = tuple(fact for item in results for fact in item.facts)
    diagnostics = tuple(
        diagnostic for item in results for diagnostic in item.diagnostics
    )
    return CodeSecurityFactSet(
        tree_id=tree_id,
        diff_id=diff_id,
        status=_status(facts, diagnostics),
        facts=facts,
        diagnostics=diagnostics,
    )


# Explicit compatibility spellings for downstream gate code.
CodeSecurityDiff = ChangedCodeDiff
CodeFactKind = CodeSecurityFactKind
CodeFactDelta = CodeSecurityDelta
CodeFactExtractionStatus = CodeSecurityExtractionStatus
CodeFactDiagnosticCode = CodeSecurityDiagnosticCode
CodeFactIdentityBinding = CodeSecurityIdentityBinding
CodeFactSourceScope = CodeSecuritySourceScope
GeneratedCodeSecurityFact = CodeSecurityFact
GeneratedCodeSecurityFacts = CodeSecurityFactSet
extract_changed_diff_security_facts = extract_code_security_facts
extract_generated_code_security_facts = extract_code_security_facts

# Narrow compatibility surface for software-verification security domain
# adapters (LFV SecuritySoftwareVerificationAdapter@1).  Facts remain
# observational: they never authorize completion or prove source correctness.
SOFTWARE_VERIFICATION_SECURITY_FACT_COMPAT = (
    "ipfs_accelerate_py/agent-supervisor/software-verification-security-fact-compat@1"
)


def security_observation_for_software_verification(
    changed_diff: (
        ChangedCodeDiff
        | Mapping[str, Any]
        | Sequence[ChangedCodeDiff | Mapping[str, Any]]
    ),
) -> CodeSecurityFactSet:
    """Extract non-authoritative security observations for shared IR lowering.

    Compatibility helper used by
    ``ipfs_datasets_py.logic.software_verification.domain_adapters``.  The
    returned fact set never grants proof, completion, or execution authority.
    """

    result = extract_code_security_facts(changed_diff)
    if result.authorizes_completion or result.grants_authority:
        raise CodeSecurityFactError(
            "code security facts must not authorize completion or grant authority "
            "when reused by software-verification domain adapters"
        )
    return result


def security_observation_payload(
    fact_set: CodeSecurityFactSet | Mapping[str, Any],
) -> dict[str, Any]:
    """Compact observational payload for SecuritySoftwareVerificationAdapter."""

    if isinstance(fact_set, CodeSecurityFactSet):
        payload = fact_set.to_dict()
    elif isinstance(fact_set, Mapping):
        payload = dict(fact_set)
    else:
        raise CodeSecurityFactError(
            "fact_set must be a CodeSecurityFactSet or mapping"
        )
    return {
        "compat": SOFTWARE_VERIFICATION_SECURITY_FACT_COMPAT,
        "authoritative": False,
        "authorizes_completion": False,
        "grants_authority": False,
        "status": payload.get("status"),
        "tree_id": payload.get("tree_id"),
        "diff_id": payload.get("diff_id"),
        "fact_count": len(payload.get("facts") or ()),
        "diagnostic_count": len(payload.get("diagnostics") or ()),
        "facts": payload.get("facts") or [],
        "diagnostics": payload.get("diagnostics") or [],
    }


__all__ = [
    "CODE_SECURITY_BINDING_SCHEMA",
    "CODE_SECURITY_DIAGNOSTIC_SCHEMA",
    "CODE_SECURITY_EXTRACTOR_VERSION",
    "CODE_SECURITY_FACT_SCHEMA",
    "CODE_SECURITY_FACT_SET_SCHEMA",
    "CODE_SECURITY_SCOPE_SCHEMA",
    "SOFTWARE_VERIFICATION_SECURITY_FACT_COMPAT",
    "ChangedCodeDiff",
    "CodeSecurityDiff",
    "CodeFactDelta",
    "CodeFactDiagnosticCode",
    "CodeFactExtractionStatus",
    "CodeFactIdentityBinding",
    "CodeFactKind",
    "CodeFactSourceScope",
    "CodeSecurityDelta",
    "CodeSecurityDiagnostic",
    "CodeSecurityDiagnosticCode",
    "CodeSecurityExtractionStatus",
    "CodeSecurityFact",
    "CodeSecurityFactError",
    "CodeSecurityFactKind",
    "CodeSecurityFactSet",
    "CodeSecurityIdentityBinding",
    "CodeSecuritySourceScope",
    "GeneratedCodeSecurityFact",
    "GeneratedCodeSecurityFacts",
    "extract_changed_diff_security_facts",
    "extract_code_security_facts",
    "extract_generated_code_security_facts",
    "security_observation_for_software_verification",
    "security_observation_payload",
]
