"""Compile candidate source changes into bounded, typed code-proof scopes.

This module does not invent proof statements.  It produces deterministic inputs
for reviewed obligation templates: exact changed paths, changed Python facts,
source bindings, and explicit conservative fallbacks.  Repository-wide source
and opaque model-generated AST summaries are intentionally not proof inputs.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .conflict_graph import (
    ASTBlobRecord,
    ConflictSurface,
    TaskConflictGraph,
    _looks_generated,
    build_python_ast_blob_record,
    index_ast_blob_records,
    normalize_repo_path,
)
from .formal_verification_contracts import (
    AssuranceLevel,
    CodeProofObligation,
    ContractValidationError,
    EvidenceFreshness,
    ProofReceipt,
    ProofVerdict,
    assurance_satisfies,
    canonical_json,
    content_identity,
)
from .proof_obligation_templates import (
    DEFAULT_TEMPLATE_REGISTRY,
    ProofObligationTemplateRegistry,
    ReviewedCodeShape,
    UnsupportedProofTemplateError,
)


PROOF_SCOPE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/code-proof-scope@1"
PROOF_SCOPE_SET_SCHEMA = "ipfs_accelerate_py/agent-supervisor/code-proof-scope-set@1"
CODE_OBLIGATION_REQUEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/code-obligation-request@1"
)
CODE_OBLIGATION_CACHE_KEY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/code-obligation-cache-key@1"
)


class DiffChangeKind(str, Enum):
    ADD = "add"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"
    COPY = "copy"
    TYPE_CHANGE = "type_change"
    UNKNOWN = "unknown"


class ProofScopeKind(str, Enum):
    CHANGED_PATH = "changed_path"
    QUALIFIED_SYMBOL = "qualified_symbol"
    IMPORT = "import"
    CALL = "call"
    STATE_TRANSITION = "state_transition"
    INTERFACE = "interface"
    CONSERVATIVE_FILE = "conservative_file"


# Compatibility spelling for callers that use "type" at the API boundary.
ProofScopeType = ProofScopeKind
CandidateChangeKind = DiffChangeKind


def _sha256_source(source: str) -> str:
    return "sha256:" + hashlib.sha256(
        source.encode("utf-8", errors="surrogatepass")
    ).hexdigest()


def _enum_change_kind(value: Any) -> DiffChangeKind:
    if isinstance(value, DiffChangeKind):
        return value
    raw = str(value or "").strip().lower().replace("-", "_")
    git_status = raw.upper()
    if git_status.startswith("R"):
        return DiffChangeKind.RENAME
    if git_status.startswith("C"):
        return DiffChangeKind.COPY
    aliases = {
        "a": DiffChangeKind.ADD,
        "added": DiffChangeKind.ADD,
        "new": DiffChangeKind.ADD,
        "m": DiffChangeKind.MODIFY,
        "modified": DiffChangeKind.MODIFY,
        "d": DiffChangeKind.DELETE,
        "deleted": DiffChangeKind.DELETE,
        "removed": DiffChangeKind.DELETE,
        "t": DiffChangeKind.TYPE_CHANGE,
        "typechange": DiffChangeKind.TYPE_CHANGE,
        "type_changed": DiffChangeKind.TYPE_CHANGE,
        "u": DiffChangeKind.UNKNOWN,
        "unmerged": DiffChangeKind.UNKNOWN,
    }
    return aliases.get(raw, DiffChangeKind(raw) if raw in {item.value for item in DiffChangeKind} else DiffChangeKind.UNKNOWN)


@dataclass(frozen=True)
class CandidateDiffEntry:
    """One normalized before/after path in a candidate change."""

    old_path: str = ""
    new_path: str = ""
    change_kind: DiffChangeKind = DiffChangeKind.MODIFY
    before_source: str | None = None
    after_source: str | None = None
    before_blob_id: str = ""
    after_blob_id: str = ""
    binary: bool = False
    generated: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        binary = bool(self.binary)
        for name in ("before_source", "after_source"):
            source = getattr(self, name)
            if isinstance(source, bytes):
                try:
                    source = source.decode("utf-8")
                except UnicodeDecodeError:
                    source = None
                    binary = True
                object.__setattr__(self, name, source)
            elif source is not None and not isinstance(source, str):
                raise TypeError(f"{name} must be text, bytes, or None")
        old_path = normalize_repo_path(self.old_path)
        new_path = normalize_repo_path(self.new_path)
        kind = _enum_change_kind(self.change_kind)
        if kind == DiffChangeKind.ADD and not new_path:
            new_path = old_path
            old_path = ""
        elif kind == DiffChangeKind.DELETE and not old_path:
            old_path = new_path
            new_path = ""
        elif kind == DiffChangeKind.RENAME and old_path == new_path:
            kind = DiffChangeKind.MODIFY
        object.__setattr__(self, "old_path", old_path)
        object.__setattr__(self, "new_path", new_path)
        object.__setattr__(self, "change_kind", kind)
        object.__setattr__(self, "before_blob_id", str(self.before_blob_id or "").strip())
        object.__setattr__(self, "after_blob_id", str(self.after_blob_id or "").strip())
        object.__setattr__(self, "binary", binary)
        object.__setattr__(
            self,
            "metadata",
            {str(key): value for key, value in sorted(dict(self.metadata).items())},
        )
        if not old_path and not new_path:
            raise ValueError("candidate diff entry requires an old or new repository path")

    @property
    def path(self) -> str:
        return self.new_path or self.old_path

    @property
    def is_python(self) -> bool:
        return self.path.lower().endswith((".py", ".pyi"))

    def to_dict(self, *, include_sources: bool = True) -> dict[str, Any]:
        payload = {
            "old_path": self.old_path,
            "new_path": self.new_path,
            "change_kind": self.change_kind.value,
            "before_blob_id": self.before_blob_id,
            "after_blob_id": self.after_blob_id,
            "binary": self.binary,
            "generated": self.generated,
            "metadata": dict(self.metadata),
        }
        if include_sources:
            payload["before_source"] = self.before_source
            payload["after_source"] = self.after_source
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CandidateDiffEntry":
        payload = dict(value)
        status = (
            payload.get("change_kind")
            or payload.get("change_type")
            or payload.get("status")
            or payload.get("kind")
            or ""
        )
        old_path = str(
            payload.get("old_path")
            or payload.get("before_path")
            or payload.get("source_path")
            or ""
        )
        new_path = str(
            payload.get("new_path")
            or payload.get("after_path")
            or payload.get("target_path")
            or payload.get("path")
            or payload.get("changed_path")
            or ""
        )
        if not status:
            if old_path and not new_path:
                status = "delete"
            elif new_path and not old_path:
                status = "add"
            elif old_path and new_path and old_path != new_path:
                status = "rename"
            else:
                status = "modify"
        kind = _enum_change_kind(status)
        if kind == DiffChangeKind.MODIFY and not old_path:
            old_path = new_path
        if kind == DiffChangeKind.MODIFY and not new_path:
            new_path = old_path
        known = {
            "old_path", "before_path", "source_path", "new_path", "after_path",
            "target_path", "path", "changed_path", "change_kind", "change_type",
            "status", "kind", "before_source", "old_source", "base_source",
            "after_source", "new_source", "candidate_source", "source",
            "before_blob_id", "old_blob_id", "old_blob", "base_blob",
            "after_blob_id", "new_blob_id", "new_blob", "candidate_blob",
            "blob_id", "binary", "is_binary", "generated", "is_generated",
            "metadata",
        }
        metadata = dict(payload.get("metadata") or {})
        metadata.update({key: payload[key] for key in payload if key not in known})
        result = cls(
            old_path=old_path,
            new_path=new_path,
            change_kind=kind,
            before_source=payload.get("before_source", payload.get("old_source", payload.get("base_source"))),
            after_source=payload.get(
                "after_source",
                payload.get("new_source", payload.get("candidate_source", payload.get("source"))),
            ),
            before_blob_id=str(
                payload.get("before_blob_id")
                or payload.get("old_blob_id")
                or payload.get("old_blob")
                or payload.get("base_blob")
                or ""
            ),
            after_blob_id=str(
                payload.get("after_blob_id")
                or payload.get("new_blob_id")
                or payload.get("new_blob")
                or payload.get("candidate_blob")
                or payload.get("blob_id")
                or ""
            ),
            binary=bool(payload.get("binary", payload.get("is_binary", False))),
            generated=payload.get("generated", payload.get("is_generated")),
            metadata=metadata,
        )
        return result


@dataclass(frozen=True)
class CodeProofScope:
    """One content-addressed path or AST fact selected for proof planning."""

    kind: ProofScopeKind
    path: str
    change_kind: DiffChangeKind
    value: str = ""
    qualified_symbol: str = ""
    owner_symbol: str = ""
    delta: str = "context"
    old_path: str = ""
    before_source_hash: str = ""
    after_source_hash: str = ""
    before_blob_id: str = ""
    after_blob_id: str = ""
    line_start: int = 0
    line_end: int = 0
    conservative: bool = False
    conservative_reasons: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ProofScopeKind(self.kind))
        object.__setattr__(self, "change_kind", _enum_change_kind(self.change_kind))
        object.__setattr__(self, "path", normalize_repo_path(self.path))
        object.__setattr__(self, "old_path", normalize_repo_path(self.old_path))
        if not self.path:
            raise ValueError("code proof scope requires a repository path")
        for name in (
            "value", "qualified_symbol", "owner_symbol", "delta",
            "before_source_hash", "after_source_hash", "before_blob_id", "after_blob_id",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
        object.__setattr__(self, "line_start", max(0, int(self.line_start or 0)))
        object.__setattr__(self, "line_end", max(0, int(self.line_end or 0)))
        reasons = tuple(sorted({str(reason).strip() for reason in self.conservative_reasons if str(reason).strip()}))
        object.__setattr__(self, "conservative_reasons", reasons)
        object.__setattr__(self, "conservative", bool(self.conservative or reasons))
        object.__setattr__(
            self,
            "metadata",
            {str(key): value for key, value in sorted(dict(self.metadata).items())},
        )

    def _identity_payload(self) -> dict[str, Any]:
        # Blob/cache identifiers are intentionally excluded.  Exact source
        # hashes and semantic facts are sufficient and stay stable when the
        # same source is discovered through a cold read or a warm Git cache.
        return {
            "schema": PROOF_SCOPE_SCHEMA,
            "kind": self.kind.value,
            "path": self.path,
            "old_path": self.old_path,
            "change_kind": self.change_kind.value,
            "value": self.value,
            "qualified_symbol": self.qualified_symbol,
            "owner_symbol": self.owner_symbol,
            "delta": self.delta,
            "before_source_hash": self.before_source_hash,
            "after_source_hash": self.after_source_hash,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "conservative": self.conservative,
            "conservative_reasons": self.conservative_reasons,
        }

    @property
    def scope_id(self) -> str:
        return content_identity(self._identity_payload())

    @property
    def content_id(self) -> str:
        return self.scope_id

    @property
    def scope_type(self) -> ProofScopeKind:
        return self.kind

    @property
    def scope_kind(self) -> ProofScopeKind:
        return self.kind

    @property
    def source_hashes(self) -> tuple[str, ...]:
        return tuple(
            item
            for item in (self.before_source_hash, self.after_source_hash)
            if item
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "scope_id": self.scope_id,
            "before_blob_id": self.before_blob_id,
            "after_blob_id": self.after_blob_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodeProofScope":
        schema = str(payload.get("schema") or PROOF_SCOPE_SCHEMA)
        if schema != PROOF_SCOPE_SCHEMA:
            raise ValueError(f"unsupported code proof scope schema: {schema}")
        scope = cls(
            kind=payload.get("kind", payload.get("scope_type", "")),
            path=str(payload.get("path") or ""),
            old_path=str(payload.get("old_path") or ""),
            change_kind=payload.get("change_kind", DiffChangeKind.UNKNOWN),
            value=str(payload.get("value") or ""),
            qualified_symbol=str(payload.get("qualified_symbol") or ""),
            owner_symbol=str(payload.get("owner_symbol") or ""),
            delta=str(payload.get("delta") or "context"),
            before_source_hash=str(payload.get("before_source_hash") or ""),
            after_source_hash=str(payload.get("after_source_hash") or ""),
            before_blob_id=str(payload.get("before_blob_id") or ""),
            after_blob_id=str(payload.get("after_blob_id") or ""),
            line_start=int(payload.get("line_start") or 0),
            line_end=int(payload.get("line_end") or 0),
            conservative=bool(payload.get("conservative", False)),
            conservative_reasons=tuple(payload.get("conservative_reasons") or ()),
            metadata=payload.get("metadata") or {},
        )
        claimed_id = str(payload.get("scope_id") or payload.get("content_id") or "")
        if claimed_id and claimed_id != scope.scope_id:
            raise ValueError("code proof scope identity does not match payload")
        return scope


# More explicit spelling retained for template/index work.
ASTProofScope = CodeProofScope
TypedASTProofScope = CodeProofScope


@dataclass(frozen=True)
class ProofScopeCompilationStats:
    entry_count: int = 0
    python_entry_count: int = 0
    parsed_blob_count: int = 0
    reused_blob_count: int = 0
    conservative_entry_count: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "entry_count": self.entry_count,
            "python_entry_count": self.python_entry_count,
            "parsed_blob_count": self.parsed_blob_count,
            "reused_blob_count": self.reused_blob_count,
            "conservative_entry_count": self.conservative_entry_count,
        }


@dataclass(frozen=True)
class CodeProofScopeSet:
    """Canonical scope selection plus non-canonical cache measurements."""

    scopes: tuple[CodeProofScope, ...]
    changed_paths: tuple[str, ...]
    source_hashes: tuple[str, ...]
    ast_records: tuple[ASTBlobRecord, ...] = ()
    stats: ProofScopeCompilationStats = field(default_factory=ProofScopeCompilationStats)

    def __post_init__(self) -> None:
        unique_scopes = {scope.scope_id: scope for scope in self.scopes}
        object.__setattr__(
            self,
            "scopes",
            tuple(unique_scopes[key] for key in sorted(unique_scopes)),
        )
        normalized_paths: set[str] = set()
        for raw_path in self.changed_paths:
            path = normalize_repo_path(str(raw_path))
            if not path:
                raise ValueError("proof scope set contains an invalid changed path")
            normalized_paths.add(path)
        object.__setattr__(self, "changed_paths", tuple(sorted(normalized_paths)))
        object.__setattr__(self, "source_hashes", tuple(sorted(set(self.source_hashes))))
        unique_records = {record.record_id: record for record in self.ast_records}
        object.__setattr__(
            self,
            "ast_records",
            tuple(unique_records[key] for key in sorted(unique_records)),
        )

    @property
    def scope_ids(self) -> tuple[str, ...]:
        return tuple(scope.scope_id for scope in self.scopes)

    @property
    def scope_identities(self) -> tuple[str, ...]:
        return self.scope_ids

    @property
    def qualified_symbols(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    scope.qualified_symbol
                    for scope in self.by_kind(ProofScopeKind.QUALIFIED_SYMBOL)
                    if scope.qualified_symbol
                }
            )
        )

    def _values(self, kind: ProofScopeKind) -> tuple[str, ...]:
        return tuple(
            sorted({scope.value for scope in self.by_kind(kind) if scope.value})
        )

    @property
    def imports(self) -> tuple[str, ...]:
        return self._values(ProofScopeKind.IMPORT)

    @property
    def calls(self) -> tuple[str, ...]:
        return self._values(ProofScopeKind.CALL)

    @property
    def state_transitions(self) -> tuple[str, ...]:
        return self._values(ProofScopeKind.STATE_TRANSITION)

    @property
    def interfaces(self) -> tuple[str, ...]:
        return self._values(ProofScopeKind.INTERFACE)

    @property
    def changed_path_scopes(self) -> tuple[CodeProofScope, ...]:
        return self.by_kind(ProofScopeKind.CHANGED_PATH)

    @property
    def compilation_id(self) -> str:
        return content_identity(
            {
                "schema": PROOF_SCOPE_SET_SCHEMA,
                "scope_ids": self.scope_ids,
                "changed_paths": self.changed_paths,
                "source_hashes": self.source_hashes,
            }
        )

    @property
    def scope_set_id(self) -> str:
        return self.compilation_id

    @property
    def conservative(self) -> bool:
        return any(scope.conservative for scope in self.scopes)

    @property
    def conservative_reasons(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    reason
                    for scope in self.scopes
                    for reason in scope.conservative_reasons
                }
            )
        )

    def by_kind(self, kind: ProofScopeKind | str) -> tuple[CodeProofScope, ...]:
        normalized = ProofScopeKind(kind)
        return tuple(scope for scope in self.scopes if scope.kind == normalized)

    def __iter__(self):
        return iter(self.scopes)

    def __len__(self) -> int:
        return len(self.scopes)

    def __getitem__(self, index: int) -> CodeProofScope:
        return self.scopes[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_SCOPE_SET_SCHEMA,
            "scope_set_id": self.scope_set_id,
            "compilation_id": self.compilation_id,
            "scope_ids": list(self.scope_ids),
            "changed_paths": list(self.changed_paths),
            "source_hashes": list(self.source_hashes),
            "conservative": self.conservative,
            "conservative_reasons": list(self.conservative_reasons),
            "scopes": [scope.to_dict() for scope in self.scopes],
            "ast_records": [record.to_dict() for record in self.ast_records],
            "stats": self.stats.to_dict(),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodeProofScopeSet":
        schema = str(payload.get("schema") or PROOF_SCOPE_SET_SCHEMA)
        if schema != PROOF_SCOPE_SET_SCHEMA:
            raise ValueError(f"unsupported code proof scope-set schema: {schema}")
        stats_payload = payload.get("stats") or {}
        stats = ProofScopeCompilationStats(
            entry_count=int(stats_payload.get("entry_count") or 0),
            python_entry_count=int(stats_payload.get("python_entry_count") or 0),
            parsed_blob_count=int(stats_payload.get("parsed_blob_count") or 0),
            reused_blob_count=int(stats_payload.get("reused_blob_count") or 0),
            conservative_entry_count=int(
                stats_payload.get("conservative_entry_count") or 0
            ),
        )
        records = tuple(
            ASTBlobRecord.from_dict(record)
            for record in payload.get("ast_records") or ()
        )
        result = cls(
            scopes=tuple(
                CodeProofScope.from_dict(scope)
                for scope in payload.get("scopes") or ()
            ),
            changed_paths=tuple(payload.get("changed_paths") or ()),
            source_hashes=tuple(payload.get("source_hashes") or ()),
            ast_records=records,
            stats=stats,
        )
        claimed_id = str(
            payload.get("scope_set_id") or payload.get("compilation_id") or ""
        )
        if claimed_id and claimed_id != result.scope_set_id:
            raise ValueError("code proof scope-set identity does not match payload")
        claimed_scope_ids = tuple(payload.get("scope_ids") or ())
        if claimed_scope_ids and claimed_scope_ids != result.scope_ids:
            raise ValueError("code proof scope identities do not match payload")
        return result

    @classmethod
    def from_json(cls, text: str) -> "CodeProofScopeSet":
        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ValueError("code proof scope-set JSON must be an object")
        return cls.from_dict(payload)


CompiledProofScopes = CodeProofScopeSet
ProofScopeSet = CodeProofScopeSet
ProofScopeCompilation = CodeProofScopeSet
CandidateFileDiff = CandidateDiffEntry


@dataclass(frozen=True)
class CodeObligationRequest:
    """Explicit binding of a reviewed template to exact compiled scopes.

    ``template_id`` is mandatory.  ``code_shape`` is optional because a
    reviewed policy may select a template directly; when present it must be an
    exact shape supported by that same template.  No field is interpreted as
    free-form evidence for a similar template.
    """

    template_id: str
    template_version: str = ""
    ast_scope_ids: tuple[str, ...] = ()
    code_shape: str = ""
    premise_ids: tuple[str, ...] = ()
    required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED
    task_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("template_id", "template_version", "code_shape", "task_id"):
            value = str(getattr(self, name) or "").strip()
            if name == "template_id" and not value:
                raise ValueError("template_id is required")
            object.__setattr__(self, name, value)
        for name in ("ast_scope_ids", "premise_ids"):
            raw = getattr(self, name)
            if isinstance(raw, str):
                raw = (raw,)
            values = tuple(
                sorted({str(value).strip() for value in raw if str(value).strip()})
            )
            object.__setattr__(self, name, values)
        object.__setattr__(
            self, "required_assurance", AssuranceLevel(self.required_assurance)
        )
        # Reuse the canonical contract boundary to reject floats, opaque
        # objects, and non-string mapping keys.
        normalized_metadata = json.loads(canonical_json(dict(self.metadata)))
        object.__setattr__(self, "metadata", normalized_metadata)

    @property
    def request_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CODE_OBLIGATION_REQUEST_SCHEMA,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "ast_scope_ids": self.ast_scope_ids,
            "code_shape": self.code_shape,
            "premise_ids": self.premise_ids,
            "required_assurance": self.required_assurance.value,
            "task_id": self.task_id,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodeObligationRequest":
        schema = str(payload.get("schema") or CODE_OBLIGATION_REQUEST_SCHEMA)
        if schema != CODE_OBLIGATION_REQUEST_SCHEMA:
            raise ValueError(f"unsupported code obligation request schema: {schema}")
        result = cls(
            template_id=str(payload.get("template_id") or ""),
            template_version=str(payload.get("template_version") or ""),
            ast_scope_ids=tuple(payload.get("ast_scope_ids") or ()),
            code_shape=str(payload.get("code_shape") or ""),
            premise_ids=tuple(payload.get("premise_ids") or ()),
            required_assurance=payload.get(
                "required_assurance", AssuranceLevel.KERNEL_VERIFIED
            ),
            task_id=str(payload.get("task_id") or ""),
            metadata=payload.get("metadata") or {},
        )
        claimed_id = str(payload.get("request_id") or "")
        if claimed_id and claimed_id != result.request_id:
            raise ValueError("code obligation request identity does not match payload")
        return result

    @classmethod
    def from_json(cls, text: str) -> "CodeObligationRequest":
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("code obligation request JSON is malformed") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("code obligation request JSON must be an object")
        return cls.from_dict(payload)


CodeProofObligationRequest = CodeObligationRequest
ProofObligationRequest = CodeObligationRequest


def _selected_obligation_scopes(
    scope_set: CodeProofScopeSet,
    requested_scope_ids: Sequence[str],
) -> tuple[CodeProofScope, ...]:
    by_id = {scope.scope_id: scope for scope in scope_set.scopes}
    if requested_scope_ids:
        unknown = sorted(set(requested_scope_ids) - set(by_id))
        if unknown:
            raise ValueError(
                "obligation request references scopes outside the compiled scope set: "
                + ", ".join(unknown)
            )
        selected = tuple(by_id[value] for value in sorted(set(requested_scope_ids)))
    else:
        # Path inventory is compilation context, not an AST theorem premise.
        # Conservative-file scopes can never become formal premises.
        selected = tuple(
            scope
            for scope in scope_set.scopes
            if scope.kind
            not in (ProofScopeKind.CHANGED_PATH, ProofScopeKind.CONSERVATIVE_FILE)
        )
    if not selected:
        raise UnsupportedProofTemplateError(
            "no non-conservative AST scopes are available for a code obligation"
        )
    if any(scope.conservative for scope in selected):
        reasons = sorted(
            {
                reason
                for scope in selected
                for reason in scope.conservative_reasons
            }
        )
        raise UnsupportedProofTemplateError(
            "conservative scopes cannot satisfy a reviewed code obligation"
            + (": " + ", ".join(reasons) if reasons else "")
        )
    return tuple(sorted(selected, key=lambda scope: scope.scope_id))


def materialize_code_proof_obligation(
    scope_set: CodeProofScopeSet,
    *,
    repository_tree_id: str,
    template_id: str = "",
    template_version: str | None = None,
    request: CodeObligationRequest | None = None,
    repository_id: str = "",
    ast_scope_ids: Sequence[str] = (),
    code_shape: str | ReviewedCodeShape = "",
    premise_ids: Sequence[str] = (),
    required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED,
    task_id: str = "",
    backend_id: str = "",
    metadata: Mapping[str, Any] | None = None,
    registry: ProofObligationTemplateRegistry = DEFAULT_TEMPLATE_REGISTRY,
) -> CodeProofObligation:
    """Apply one explicitly selected reviewed template to exact AST scopes.

    The canonical statement, fallback tests, version, and semantic hash always
    come from the registry.  Callers cannot replace them with model-generated
    prose.  An optional code shape is checked by exact membership only.
    """

    if not isinstance(scope_set, CodeProofScopeSet):
        raise TypeError("scope_set must be a CodeProofScopeSet")
    tree_id = str(repository_tree_id or "").strip()
    if not tree_id:
        raise ValueError("repository_tree_id is required")
    if request is None:
        shape_value = str(getattr(code_shape, "value", code_shape) or "").strip()
        request = CodeObligationRequest(
            template_id=template_id,
            template_version=str(template_version or ""),
            ast_scope_ids=tuple(ast_scope_ids),
            code_shape=shape_value,
            premise_ids=tuple(premise_ids),
            required_assurance=required_assurance,
            task_id=task_id,
            metadata=metadata or {},
        )
    elif any(
        (
            template_id,
            template_version,
            tuple(ast_scope_ids),
            str(getattr(code_shape, "value", code_shape) or "").strip(),
            tuple(premise_ids),
            task_id,
            metadata,
        )
    ):
        raise ValueError(
            "request cannot be combined with direct template, scope, premise, task, "
            "or metadata arguments"
        )

    template = registry.require(
        request.template_id, request.template_version or None
    )
    if request.code_shape and not template.supports_code_shape(request.code_shape):
        raise UnsupportedProofTemplateError(
            f"template {template.template_id!r} does not support exact code shape "
            f"{request.code_shape!r}"
        )
    normalized_backend = str(backend_id or "").strip()
    if normalized_backend and not template.supports_backend(normalized_backend):
        raise UnsupportedProofTemplateError(
            f"template {template.template_id!r} does not support backend "
            f"{normalized_backend!r}"
        )
    selected = _selected_obligation_scopes(scope_set, request.ast_scope_ids)
    obligation_metadata = dict(request.metadata)
    obligation_metadata.update({"code_shape": request.code_shape})
    return CodeProofObligation(
        repository_id=str(repository_id or "").strip(),
        repository_tree_id=tree_id,
        ast_scope_ids=tuple(scope.scope_id for scope in selected),
        statement=template.canonical_statement,
        premise_ids=request.premise_ids,
        template_id=template.template_id,
        template_version=template.version,
        template_semantic_hash=template.semantic_hash,
        invariant_class=template.invariant_class,
        task_id=request.task_id,
        required_assurance=request.required_assurance,
        fallback_checks=template.fallback_tests,
        metadata=obligation_metadata,
    )


def build_code_proof_obligation(
    scope_set: CodeProofScopeSet,
    **kwargs: Any,
) -> CodeProofObligation:
    """Compatibility facade for :func:`materialize_code_proof_obligation`."""

    return materialize_code_proof_obligation(scope_set, **kwargs)


def obligation_cache_identity(
    obligation: CodeProofObligation,
    *,
    backend_id: str = "",
    translator_id: str = "",
    toolchain_id: str = "",
    semantic_input_ids: Iterable[str] = (),
) -> str:
    """Return a proof-cache identity including all reviewed semantics.

    Template version and semantic hash are repeated explicitly instead of
    relying only on their transitive inclusion in ``obligation_id``.  This
    makes incomplete cache-key implementations visible during review.
    """

    if not isinstance(obligation, CodeProofObligation):
        raise TypeError("obligation must be a CodeProofObligation")
    raw_inputs = (
        (semantic_input_ids,)
        if isinstance(semantic_input_ids, str)
        else semantic_input_ids
    )
    inputs = tuple(
        sorted(
            {
                str(value).strip()
                for value in raw_inputs
                if str(value).strip()
            }
        )
    )
    return content_identity(
        {
            "schema": CODE_OBLIGATION_CACHE_KEY_SCHEMA,
            "obligation_id": obligation.obligation_id,
            "repository_tree_id": obligation.repository_tree_id,
            "ast_scope_ids": obligation.ast_scope_ids,
            "template_id": obligation.template_id,
            "template_version": obligation.template_version,
            "template_semantic_hash": obligation.template_semantic_hash,
            "backend_id": str(backend_id or "").strip(),
            "translator_id": str(translator_id or "").strip(),
            "toolchain_id": str(toolchain_id or "").strip(),
            "semantic_input_ids": inputs,
        }
    )


code_proof_obligation_cache_identity = obligation_cache_identity
build_obligation_cache_key = obligation_cache_identity


def _module_name(path: str) -> str:
    normalized = normalize_repo_path(path)
    pure = PurePosixPath(normalized)
    parts = list(pure.parts)
    if not parts:
        return ""
    name = parts[-1]
    if name.endswith(".pyi"):
        parts[-1] = name[:-4]
    elif name.endswith(".py"):
        parts[-1] = name[:-3]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(part for part in parts if part)


def _qualify(path: str, lexical: str) -> str:
    module = _module_name(path)
    if not lexical or lexical == "<module>":
        return module or "<module>"
    return f"{module}.{lexical}" if module else lexical


def _split_owner(value: str, separator: str) -> tuple[str, str]:
    if separator not in value:
        return "<module>", value
    return tuple(value.split(separator, 1))  # type: ignore[return-value]


def _record_matches_source(record: ASTBlobRecord, source_hash: str) -> bool:
    return record.source_sha256 == source_hash


def _candidate_records(
    ast_records: Iterable[Any],
    conflict_graph: TaskConflictGraph | None,
    conflict_surfaces: Iterable[ConflictSurface | Mapping[str, Any]],
) -> list[Any]:
    values = list(ast_records)
    surfaces: list[Any] = list(conflict_surfaces)
    if conflict_graph is not None:
        surfaces.extend(conflict_graph.surfaces.values())
    for surface in surfaces:
        if isinstance(surface, ConflictSurface):
            values.extend(surface.ast_records)
        elif isinstance(surface, Mapping):
            records = surface.get("ast_records") or surface.get("ast_blob_records") or ()
            if isinstance(records, Mapping):
                values.extend(records.values())
            elif isinstance(records, Sequence) and not isinstance(records, (str, bytes, bytearray)):
                values.extend(records)
    return values


def _facts_for_record(
    record: ASTBlobRecord,
    *,
    path: str,
    before: ASTBlobRecord | None,
    kind: DiffChangeKind,
    side: str,
) -> list[tuple[ProofScopeKind, str, str, str, int, int]]:
    """Return kind, value, owner, delta, start, end for selected facts."""

    whole_blob = before is None or kind in {
        DiffChangeKind.ADD,
        DiffChangeKind.DELETE,
        DiffChangeKind.RENAME,
        DiffChangeKind.COPY,
    }
    default_delta = "removed" if side == "before" else "added"
    if whole_blob:
        changed_symbols = set(record.qualified_symbols)
    else:
        all_symbols = set(record.qualified_symbols) | set(before.qualified_symbols)
        changed_symbols = {
            symbol
            for symbol in all_symbols
            if record.symbol_hashes.get(symbol) != before.symbol_hashes.get(symbol)
        }

    def fact_delta(value: str, comparison_values: set[str]) -> str:
        if whole_blob or value not in comparison_values:
            return default_delta
        return f"modified_{side}"

    facts: list[tuple[ProofScopeKind, str, str, str, int, int]] = []
    for symbol in sorted(changed_symbols & set(record.qualified_symbols)):
        qualified = _qualify(path, symbol)
        start, end = record.symbol_lines.get(symbol, (0, 0))
        comparison_symbols = set(before.qualified_symbols) if before else set()
        facts.append(
            (
                ProofScopeKind.QUALIFIED_SYMBOL,
                qualified,
                qualified,
                fact_delta(symbol, comparison_symbols),
                start,
                end,
            )
        )

    comparison = before
    before_imports = set(comparison.imports) if comparison else set()
    imports = set(record.imports)
    selected_imports = imports if whole_blob else imports - before_imports
    for value in sorted(selected_imports):
        facts.append(
            (
                ProofScopeKind.IMPORT,
                value,
                _module_name(path),
                fact_delta(value, before_imports),
                0,
                0,
            )
        )

    def owner_changed(owner: str) -> bool:
        return owner == "<module>" or owner in changed_symbols or any(
            owner.startswith(symbol + ".") for symbol in changed_symbols
        )

    before_calls = set(comparison.calls) if comparison else set()
    for value in sorted(record.calls):
        owner, callee = _split_owner(value, "->")
        if whole_blob or value not in before_calls or owner_changed(owner):
            facts.append(
                (
                    ProofScopeKind.CALL,
                    callee,
                    _qualify(path, owner),
                    fact_delta(value, before_calls),
                    0,
                    0,
                )
            )

    before_states = set(comparison.state_transitions) if comparison else set()
    for value in sorted(record.state_transitions):
        owner, remainder = _split_owner(value, ":")
        if whole_blob or value not in before_states or owner_changed(owner):
            facts.append(
                (
                    ProofScopeKind.STATE_TRANSITION,
                    remainder,
                    _qualify(path, owner),
                    fact_delta(value, before_states),
                    0,
                    0,
                )
            )

    before_interfaces = set(comparison.interfaces) if comparison else set()
    for value in sorted(record.interfaces):
        lexical = value.split(":", 1)[0].split("(", 1)[0]
        if whole_blob or value not in before_interfaces or owner_changed(lexical):
            qualified_value = value.replace(lexical, _qualify(path, lexical), 1)
            facts.append(
                (
                    ProofScopeKind.INTERFACE,
                    qualified_value,
                    _qualify(path, lexical),
                    fact_delta(value, before_interfaces),
                    *record.symbol_lines.get(lexical, (0, 0)),
                )
            )
    return facts


def _coerce_entries(value: Any) -> list[CandidateDiffEntry]:
    if value is None:
        return []
    if isinstance(value, CandidateDiffEntry):
        return [value]
    if isinstance(value, str):
        return parse_unified_diff(value)
    if isinstance(value, Mapping):
        for name in ("entries", "changes", "diff_entries", "changed_files", "files"):
            nested = value.get(name)
            if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes, bytearray)):
                return _coerce_entries(nested)
        return [CandidateDiffEntry.from_mapping(value)]
    entries: list[CandidateDiffEntry] = []
    for item in value:
        if isinstance(item, CandidateDiffEntry):
            entries.append(item)
        elif isinstance(item, Mapping):
            entries.append(CandidateDiffEntry.from_mapping(item))
        else:
            raise TypeError("candidate diff entries must be mappings or CandidateDiffEntry values")
    return entries


def parse_unified_diff(text: str) -> list[CandidateDiffEntry]:
    """Parse path/status metadata from a unified Git diff.

    Unified hunks do not contain a trustworthy full before/after source.  The
    resulting entries therefore compile to explicit ``missing_source`` scopes
    unless the caller supplies repository revisions through
    :func:`collect_git_candidate_diff`.
    """

    entries: list[CandidateDiffEntry] = []
    current: dict[str, Any] | None = None
    for line in str(text or "").splitlines():
        if line.startswith("diff --git "):
            if current:
                entries.append(CandidateDiffEntry.from_mapping(current))
            match = re.match(r"diff --git a/(.*?) b/(.*)$", line)
            current = {
                "old_path": match.group(1) if match else "",
                "new_path": match.group(2) if match else "",
                "status": "modify",
            }
        elif current is not None and line.startswith("new file mode"):
            current["status"] = "add"
            current["old_path"] = ""
        elif current is not None and line.startswith("deleted file mode"):
            current["status"] = "delete"
            current["new_path"] = ""
        elif current is not None and line.startswith("rename from "):
            current["status"] = "rename"
            current["old_path"] = line[len("rename from ") :]
        elif current is not None and line.startswith("rename to "):
            current["status"] = "rename"
            current["new_path"] = line[len("rename to ") :]
        elif current is not None and line.startswith("Binary files "):
            current["binary"] = True
    if current:
        entries.append(CandidateDiffEntry.from_mapping(current))
    return entries


def _git(repo_root: Path, *arguments: str, binary: bool = False) -> str | bytes | None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return None
    if binary:
        return result.stdout
    return result.stdout.decode("utf-8", errors="surrogateescape")


def _git_source(repo_root: Path, revision: str | None, path: str) -> tuple[str | None, str]:
    if revision is None:
        absolute = repo_root / path
        if absolute.is_symlink():
            blob = _git(repo_root, "hash-object", "--", path)
            return None, str(blob or "").strip()
        try:
            raw = absolute.read_bytes()
        except OSError:
            return None, ""
        blob = _git(repo_root, "hash-object", "--", path)
    else:
        raw = _git(repo_root, "show", f"{revision}:{path}", binary=True)
        if not isinstance(raw, bytes):
            return None, ""
        blob = _git(repo_root, "rev-parse", f"{revision}:{path}")
    if b"\0" in raw:
        return None, str(blob or "").strip()
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None, str(blob or "").strip()
    return source, str(blob or "").strip()


def collect_git_candidate_diff(
    repo_root: Path | str,
    *,
    base_revision: str = "HEAD",
    candidate_revision: str | None = None,
    include_untracked: bool = True,
) -> list[CandidateDiffEntry]:
    """Collect complete sources for a revision-to-revision or working-tree diff."""

    root = Path(repo_root).resolve()
    arguments = ["diff", "--name-status", "-z", "-M", "-C", base_revision]
    if candidate_revision is not None:
        arguments.append(candidate_revision)
    raw = _git(root, *arguments, binary=True)
    if not isinstance(raw, bytes):
        raise ValueError(f"unable to inspect Git diff at {root}")
    fields = [
        item.decode("utf-8", errors="surrogateescape")
        for item in raw.split(b"\0")
        if item
    ]
    changes: list[tuple[str, str, str]] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        if status.startswith(("R", "C")):
            if index + 1 >= len(fields):
                break
            old_path, new_path = fields[index], fields[index + 1]
            index += 2
        else:
            if index >= len(fields):
                break
            path = fields[index]
            index += 1
            old_path = "" if status.startswith("A") else path
            new_path = "" if status.startswith("D") else path
        changes.append((status, old_path, new_path))

    if candidate_revision is None and include_untracked:
        untracked_raw = _git(
            root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            binary=True,
        )
        if isinstance(untracked_raw, bytes):
            for raw_path in untracked_raw.split(b"\0"):
                if raw_path:
                    changes.append(
                        ("A", "", raw_path.decode("utf-8", errors="surrogateescape"))
                    )

    entries: list[CandidateDiffEntry] = []
    for status, old_path, new_path in changes:
        before_source, before_blob = (
            _git_source(root, base_revision, old_path) if old_path else (None, "")
        )
        after_source, after_blob = (
            _git_source(root, candidate_revision, new_path) if new_path else (None, "")
        )
        binary = bool(
            (old_path and before_source is None and before_blob)
            or (new_path and after_source is None and after_blob)
        )
        entries.append(
            CandidateDiffEntry(
                old_path=old_path,
                new_path=new_path,
                change_kind=_enum_change_kind(status),
                before_source=before_source,
                after_source=after_source,
                before_blob_id=before_blob,
                after_blob_id=after_blob,
                binary=binary,
            )
        )

    # Git cannot pair an unstaged deletion with an untracked destination.
    # Reconcile only unambiguous, byte-identical one-to-one pairs by Git blob
    # identity; ambiguous duplicates remain explicit add/delete scopes.
    deleted_by_blob: dict[str, list[CandidateDiffEntry]] = {}
    added_by_blob: dict[str, list[CandidateDiffEntry]] = {}
    for entry in entries:
        if entry.change_kind == DiffChangeKind.DELETE and entry.before_blob_id:
            deleted_by_blob.setdefault(entry.before_blob_id, []).append(entry)
        elif entry.change_kind == DiffChangeKind.ADD and entry.after_blob_id:
            added_by_blob.setdefault(entry.after_blob_id, []).append(entry)
    replaced: set[int] = set()
    reconciled: list[CandidateDiffEntry] = []
    for blob in sorted(set(deleted_by_blob) & set(added_by_blob)):
        deleted = deleted_by_blob[blob]
        added = added_by_blob[blob]
        if len(deleted) != 1 or len(added) != 1:
            continue
        old_entry, new_entry = deleted[0], added[0]
        replaced.update((id(old_entry), id(new_entry)))
        reconciled.append(
            CandidateDiffEntry(
                old_path=old_entry.old_path,
                new_path=new_entry.new_path,
                change_kind=DiffChangeKind.RENAME,
                before_source=old_entry.before_source,
                after_source=new_entry.after_source,
                before_blob_id=old_entry.before_blob_id,
                after_blob_id=new_entry.after_blob_id,
                binary=old_entry.binary or new_entry.binary,
                metadata={"detected_from_unstaged_blob_identity": True},
            )
        )
    reconciled.extend(entry for entry in entries if id(entry) not in replaced)
    return sorted(reconciled, key=lambda item: (item.path, item.old_path, item.change_kind.value))


def compile_candidate_proof_scopes(
    candidate_diff: Any,
    *,
    ast_records: Iterable[Any] = (),
    conflict_graph: TaskConflictGraph | None = None,
    conflict_surfaces: Iterable[ConflictSurface | Mapping[str, Any]] = (),
) -> CodeProofScopeSet:
    """Compile normalized candidate entries into deterministic proof scopes."""

    entries = _coerce_entries(candidate_diff)
    cache_values = _candidate_records(ast_records, conflict_graph, conflict_surfaces)
    cache = index_ast_blob_records(cache_values)
    scopes: list[CodeProofScope] = []
    records: dict[str, ASTBlobRecord] = {}
    changed_paths: set[str] = set()
    source_hashes: set[str] = set()
    parsed = reused = python_count = conservative_count = 0

    def resolve(source: str | None, blob_id: str) -> ASTBlobRecord | None:
        nonlocal parsed, reused
        if source is None:
            record = cache.get(blob_id) if blob_id else None
            if record is not None:
                reused += 1
                records[record.record_id] = record
            return record
        source_hash = _sha256_source(source)
        record = cache.get(blob_id) or cache.get(source_hash)
        if record is not None and not _record_matches_source(record, source_hash):
            record = None
        if record is not None:
            reused += 1
        else:
            record = build_python_ast_blob_record(
                source,
                blob_identity=blob_id or source_hash,
                source_sha256=source_hash,
            )
            parsed += 1
            for identity in (record.blob_identity, record.source_sha256, record.record_id):
                if identity:
                    cache.setdefault(identity, record)
        records[record.record_id] = record
        return record

    for entry in sorted(entries, key=lambda item: (item.path, item.old_path, item.change_kind.value)):
        path = entry.path
        changed_paths.update(item for item in (entry.old_path, entry.new_path) if item)
        before_hash = _sha256_source(entry.before_source) if entry.before_source is not None else ""
        after_hash = _sha256_source(entry.after_source) if entry.after_source is not None else ""
        source_hashes.update(item for item in (before_hash, after_hash) if item)
        reasons: list[str] = []
        generated = bool(
            entry.generated
            if entry.generated is not None
            else _looks_generated(path) or (entry.old_path and _looks_generated(entry.old_path))
        )
        if entry.binary:
            reasons.append("binary_change")
        if generated:
            reasons.append("generated_file")
        if not entry.is_python:
            reasons.append("non_python_change")
        if entry.change_kind == DiffChangeKind.DELETE:
            reasons.append("deleted_path")
        elif entry.change_kind == DiffChangeKind.RENAME:
            reasons.append("rename_requires_reference_validation")
        elif entry.change_kind == DiffChangeKind.COPY:
            reasons.append("copy_requires_reference_validation")
        elif entry.change_kind in {DiffChangeKind.TYPE_CHANGE, DiffChangeKind.UNKNOWN}:
            reasons.append("unsupported_change_kind")
        entry_counted_conservative = bool(reasons)
        if entry_counted_conservative:
            conservative_count += 1

        def append_path_scope(scope_reasons: Sequence[str]) -> None:
            scopes.append(
                CodeProofScope(
                    kind=ProofScopeKind.CHANGED_PATH,
                    path=path,
                    old_path=entry.old_path if entry.old_path != path else "",
                    change_kind=entry.change_kind,
                    value=path,
                    delta=entry.change_kind.value,
                    before_source_hash=before_hash,
                    after_source_hash=after_hash,
                    before_blob_id=entry.before_blob_id,
                    after_blob_id=entry.after_blob_id,
                    conservative=bool(scope_reasons),
                    conservative_reasons=tuple(scope_reasons),
                )
            )

        if entry.binary or generated or not entry.is_python:
            append_path_scope(reasons)
            scopes.append(
                CodeProofScope(
                    kind=ProofScopeKind.CONSERVATIVE_FILE,
                    path=path,
                    old_path=entry.old_path if entry.old_path != path else "",
                    change_kind=entry.change_kind,
                    value=";".join(reasons),
                    delta=entry.change_kind.value,
                    before_source_hash=before_hash,
                    after_source_hash=after_hash,
                    before_blob_id=entry.before_blob_id,
                    after_blob_id=entry.after_blob_id,
                    conservative=True,
                    conservative_reasons=tuple(reasons),
                )
            )
            continue

        python_count += 1
        before = resolve(entry.before_source, entry.before_blob_id) if entry.old_path else None
        after = resolve(entry.after_source, entry.after_blob_id) if entry.new_path else None
        before_hash = before_hash or (before.source_sha256 if before is not None else "")
        after_hash = after_hash or (after.source_sha256 if after is not None else "")
        source_hashes.update(item for item in (before_hash, after_hash) if item)
        missing_expected = (
            entry.change_kind != DiffChangeKind.ADD and entry.old_path and before is None
        ) or (
            entry.change_kind != DiffChangeKind.DELETE and entry.new_path and after is None
        )
        parse_errors = [
            f"{side}_syntax_error:{record.parse_error}"
            for side, record in (("before", before), ("after", after))
            if record is not None and record.parse_error
        ]
        failure_reasons = list(reasons)
        if missing_expected:
            failure_reasons.append("missing_source")
        failure_reasons.extend(parse_errors)
        append_path_scope(failure_reasons)
        if failure_reasons and (missing_expected or parse_errors):
            if not entry_counted_conservative:
                conservative_count += 1
            scopes.append(
                CodeProofScope(
                    kind=ProofScopeKind.CONSERVATIVE_FILE,
                    path=path,
                    old_path=entry.old_path if entry.old_path != path else "",
                    change_kind=entry.change_kind,
                    value=";".join(failure_reasons),
                    delta=entry.change_kind.value,
                    before_source_hash=before_hash,
                    after_source_hash=after_hash,
                    before_blob_id=entry.before_blob_id,
                    after_blob_id=entry.after_blob_id,
                    conservative=True,
                    conservative_reasons=tuple(failure_reasons),
                )
            )
            continue

        fact_groups: list[tuple[ASTBlobRecord, str, ASTBlobRecord | None, str]] = []
        if entry.change_kind == DiffChangeKind.RENAME:
            if before is not None:
                fact_groups.append((before, entry.old_path, None, "before"))
            if after is not None:
                fact_groups.append((after, entry.new_path, None, "after"))
        elif entry.change_kind == DiffChangeKind.DELETE:
            if before is not None:
                fact_groups.append((before, entry.old_path, None, "before"))
        elif after is not None:
            if before is not None:
                fact_groups.append((before, entry.old_path, after, "before"))
            fact_groups.append((after, entry.new_path, before, "after"))

        for record, fact_path, comparison, side in fact_groups:
            for fact_kind, value, owner, delta, line_start, line_end in _facts_for_record(
                record,
                path=fact_path,
                before=comparison,
                kind=entry.change_kind,
                side=side,
            ):
                scopes.append(
                    CodeProofScope(
                        kind=fact_kind,
                        path=fact_path,
                        old_path=entry.old_path if entry.old_path != fact_path else "",
                        change_kind=entry.change_kind,
                        value=value,
                        qualified_symbol=value if fact_kind == ProofScopeKind.QUALIFIED_SYMBOL else "",
                        owner_symbol=owner,
                        delta=delta,
                        before_source_hash=before_hash,
                        after_source_hash=after_hash,
                        before_blob_id=entry.before_blob_id,
                        after_blob_id=entry.after_blob_id,
                        line_start=line_start,
                        line_end=line_end,
                        conservative=bool(reasons),
                        conservative_reasons=tuple(reasons),
                    )
                )

    return CodeProofScopeSet(
        scopes=tuple(scopes),
        changed_paths=tuple(changed_paths),
        source_hashes=tuple(source_hashes),
        ast_records=tuple(records.values()),
        stats=ProofScopeCompilationStats(
            entry_count=len(entries),
            python_entry_count=python_count,
            parsed_blob_count=parsed,
            reused_blob_count=reused,
            conservative_entry_count=conservative_count,
        ),
    )


def compile_candidate_diff_scopes(
    candidate_diff: Any = None,
    *,
    repo_root: Path | str | None = None,
    base_revision: str = "HEAD",
    candidate_revision: str | None = None,
    ast_records: Iterable[Any] = (),
    conflict_graph: TaskConflictGraph | None = None,
    conflict_surfaces: Iterable[ConflictSurface | Mapping[str, Any]] = (),
) -> CodeProofScopeSet:
    """Compile supplied entries, or collect a complete Git candidate diff."""

    if candidate_diff is None:
        if repo_root is None:
            raise ValueError("candidate_diff or repo_root is required")
        candidate_diff = collect_git_candidate_diff(
            repo_root,
            base_revision=base_revision,
            candidate_revision=candidate_revision,
        )
    return compile_candidate_proof_scopes(
        candidate_diff,
        ast_records=ast_records,
        conflict_graph=conflict_graph,
        conflict_surfaces=conflict_surfaces,
    )


def compile_candidate_diff(
    candidate_diff: Any = None,
    **kwargs: Any,
) -> CodeProofScopeSet:
    """Compatibility facade for :func:`compile_candidate_diff_scopes`.

    Passing a directory as the first argument is treated as ``repo_root``.
    """

    possible_directory = isinstance(candidate_diff, Path) or (
        isinstance(candidate_diff, str)
        and "\n" not in candidate_diff
        and "\0" not in candidate_diff
        and len(candidate_diff) < 4096
    )
    if possible_directory:
        try:
            is_directory = Path(str(candidate_diff)).is_dir()
        except OSError:
            is_directory = False
        if is_directory:
            kwargs.setdefault("repo_root", candidate_diff)
            candidate_diff = None
    return compile_candidate_diff_scopes(candidate_diff, **kwargs)


compile_code_proof_scopes = compile_candidate_diff_scopes
compile_proof_scopes = compile_candidate_diff_scopes
compile_ast_proof_scopes = compile_candidate_diff_scopes
compile_candidate_diffs = compile_candidate_diff_scopes


IMPLEMENTATION_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/implementation-result-evidence@1"
)
IMPLEMENTATION_BINDING_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/implementation-result-binding@1"
)
IMPLEMENTATION_OBLIGATION_SET_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/implementation-obligation-set@1"
)
CODE_PROOF_BINDING_RESULT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/code-proof-binding-result@1"
)


class ImplementationEvidenceKind(str, Enum):
    """Observed implementation evidence; none is itself a code proof."""

    TEST = "test"
    RUNTIME = "runtime"
    STATIC_ANALYSIS = "static_analysis"
    TYPE_CHECK = "type_check"


class ImplementationObligationKind(str, Enum):
    """Closed implementation-conformance families derived after execution."""

    CHANGED_SYMBOL = "changed_symbol"
    INTERFACE = "interface"
    EFFECT = "effect"
    TEST = "test"
    RUNTIME_EVIDENCE = "runtime_evidence"
    STATIC_ANALYSIS = "static_analysis"


def _canonical_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize a semantic mapping using the proof codec."""

    return json.loads(canonical_json(dict(value or {})))


def _canonical_strings(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    elif isinstance(values, Mapping):
        values = values.keys()
    try:
        iterator = iter(values)
    except TypeError:
        iterator = iter((values,))
    return tuple(
        sorted({str(value).strip() for value in iterator if str(value).strip()})
    )


def _timestamp(value: str | datetime | None) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            value = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError("observed_at must be an ISO-8601 timestamp") from exc
    if not isinstance(value, datetime):
        raise TypeError("observed_at must be a datetime or ISO-8601 string")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class ImplementationResultEvidence:
    """One content-addressed test, runtime, or static-analysis observation.

    This record deliberately has no assurance field.  A passing test, runtime
    trace, or type check is a bounded observation and cannot be promoted into
    a theorem about generated code.
    """

    kind: ImplementationEvidenceKind
    repository_tree_id: str
    subject: str = ""
    evidence_id: str = ""
    accepted_plan_id: str = ""
    repository_id: str = ""
    scope_ids: tuple[str, ...] = ()
    subject_ids: tuple[str, ...] = ()
    passed: bool = False
    observed_at: str | datetime | None = None
    validation_bounds: Mapping[str, Any] = field(default_factory=dict)
    assumption_ids: tuple[str, ...] = ()
    producer_id: str = ""
    command: str = ""
    artifact_id: str = ""
    contradictory: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ImplementationEvidenceKind(self.kind))
        for name in (
            "repository_tree_id",
            "accepted_plan_id",
            "repository_id",
            "producer_id",
            "subject",
            "command",
            "artifact_id",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
        if not self.repository_tree_id:
            raise ValueError("implementation evidence requires repository_tree_id")
        for name in ("scope_ids", "subject_ids", "assumption_ids"):
            object.__setattr__(
                self, name, _canonical_strings(getattr(self, name))
            )
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be boolean")
        if not isinstance(self.contradictory, bool):
            raise TypeError("contradictory must be boolean")
        object.__setattr__(self, "observed_at", _timestamp(self.observed_at))
        object.__setattr__(
            self, "validation_bounds", _canonical_mapping(self.validation_bounds)
        )
        object.__setattr__(self, "metadata", _canonical_mapping(self.metadata))
        supplied = str(self.evidence_id or "").strip()
        if not supplied:
            supplied = content_identity(self._identity_payload())
        object.__setattr__(self, "evidence_id", supplied)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": IMPLEMENTATION_EVIDENCE_SCHEMA,
            "kind": self.kind.value,
            "subject": self.subject,
            "accepted_plan_id": self.accepted_plan_id,
            "repository_id": self.repository_id,
            "repository_tree_id": self.repository_tree_id,
            "scope_ids": self.scope_ids,
            "subject_ids": self.subject_ids,
            "passed": self.passed,
            "observed_at": self.observed_at,
            "validation_bounds": self.validation_bounds,
            "assumption_ids": self.assumption_ids,
            "producer_id": self.producer_id,
            "command": self.command,
            "artifact_id": self.artifact_id,
            "contradictory": self.contradictory,
            "metadata": self.metadata,
        }

    @property
    def evidence_digest(self) -> str:
        """Content digest separate from the producer's receipt identity."""

        return content_identity(self._identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "evidence_id": self.evidence_id,
            "evidence_digest": self.evidence_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImplementationResultEvidence":
        schema = str(payload.get("schema") or IMPLEMENTATION_EVIDENCE_SCHEMA)
        if schema != IMPLEMENTATION_EVIDENCE_SCHEMA:
            raise ValueError(f"unsupported implementation evidence schema: {schema}")
        result = cls(
            kind=payload.get("kind", ImplementationEvidenceKind.TEST),
            subject=str(payload.get("subject") or ""),
            accepted_plan_id=str(payload.get("accepted_plan_id") or payload.get("plan_id") or ""),
            repository_id=str(payload.get("repository_id") or ""),
            repository_tree_id=str(payload.get("repository_tree_id") or payload.get("tree_id") or ""),
            scope_ids=tuple(payload.get("scope_ids") or payload.get("ast_scope_ids") or ()),
            subject_ids=tuple(payload.get("subject_ids") or ()),
            passed=payload.get("passed", False),
            observed_at=payload.get("observed_at"),
            validation_bounds=payload.get("validation_bounds") or payload.get("bounds") or {},
            assumption_ids=tuple(payload.get("assumption_ids") or ()),
            producer_id=str(payload.get("producer_id") or ""),
            command=str(payload.get("command") or ""),
            artifact_id=str(payload.get("artifact_id") or ""),
            contradictory=payload.get("contradictory", False),
            metadata=payload.get("metadata") or {},
            evidence_id=str(payload.get("evidence_id") or payload.get("content_id") or ""),
        )
        claimed_digest = str(payload.get("evidence_digest") or "")
        if claimed_digest and claimed_digest != result.evidence_digest:
            raise ValueError("implementation evidence digest does not match payload")
        return result


@dataclass(frozen=True)
class ImplementationResultBinding:
    """Frozen semantic context for all post-Codex implementation receipts."""

    accepted_plan_id: str
    repository_id: str
    repository_tree_id: str
    changed_scope_set_id: str
    changed_scope_ids: tuple[str, ...]
    changed_paths: tuple[str, ...]
    assumption_ids: tuple[str, ...] = ()
    assumptions: Mapping[str, Any] = field(default_factory=dict)
    validation_bounds: Mapping[str, Any] = field(default_factory=dict)
    test_evidence_ids: tuple[str, ...] = ()
    runtime_evidence_ids: tuple[str, ...] = ()
    static_analysis_evidence_ids: tuple[str, ...] = ()
    evidence_digests: Mapping[str, str] = field(default_factory=dict)
    plan_effect_ids: tuple[str, ...] = ()
    plan_requirement_ids: tuple[str, ...] = ()
    plan_trace_bound: int | None = None
    task_id: str = ""
    binding_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "accepted_plan_id",
            "repository_id",
            "repository_tree_id",
            "changed_scope_set_id",
            "task_id",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
        for name in (
            "accepted_plan_id",
            "repository_tree_id",
            "changed_scope_set_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        for name in (
            "changed_scope_ids",
            "changed_paths",
            "assumption_ids",
            "test_evidence_ids",
            "runtime_evidence_ids",
            "static_analysis_evidence_ids",
            "plan_effect_ids",
            "plan_requirement_ids",
        ):
            object.__setattr__(
                self, name, _canonical_strings(getattr(self, name))
            )
        if not self.changed_scope_ids or not self.changed_paths:
            raise ValueError("implementation binding requires a nonempty changed scope")
        object.__setattr__(self, "assumptions", _canonical_mapping(self.assumptions))
        object.__setattr__(self, "validation_bounds", _canonical_mapping(self.validation_bounds))
        digests = {
            str(key).strip(): str(value).strip()
            for key, value in dict(self.evidence_digests or {}).items()
            if str(key).strip() and str(value).strip()
        }
        object.__setattr__(self, "evidence_digests", dict(sorted(digests.items())))
        if (
            self.plan_trace_bound is not None
            and (
                isinstance(self.plan_trace_bound, bool)
                or not isinstance(self.plan_trace_bound, int)
                or self.plan_trace_bound <= 0
            )
        ):
            raise ValueError("plan_trace_bound must be a positive integer or None")
        supplied = str(self.binding_id or "").strip()
        object.__setattr__(self, "binding_id", "")
        derived = content_identity(self._identity_payload())
        if supplied and supplied != derived:
            raise ValueError("implementation binding identity does not match payload")
        object.__setattr__(self, "binding_id", derived)

    @property
    def plan_id(self) -> str:
        return self.accepted_plan_id

    @property
    def ast_scope_ids(self) -> tuple[str, ...]:
        return self.changed_scope_ids

    @property
    def scope_set_id(self) -> str:
        return self.changed_scope_set_id

    @property
    def planned_effect_ids(self) -> tuple[str, ...]:
        return self.plan_effect_ids

    @property
    def tree_id(self) -> str:
        return self.repository_tree_id

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": IMPLEMENTATION_BINDING_SCHEMA,
            "accepted_plan_id": self.accepted_plan_id,
            "repository_id": self.repository_id,
            "repository_tree_id": self.repository_tree_id,
            "changed_scope_set_id": self.changed_scope_set_id,
            "changed_scope_ids": self.changed_scope_ids,
            "changed_paths": self.changed_paths,
            "assumption_ids": self.assumption_ids,
            "assumptions": self.assumptions,
            "validation_bounds": self.validation_bounds,
            "test_evidence_ids": self.test_evidence_ids,
            "runtime_evidence_ids": self.runtime_evidence_ids,
            "static_analysis_evidence_ids": self.static_analysis_evidence_ids,
            "evidence_digests": self.evidence_digests,
            "plan_effect_ids": self.plan_effect_ids,
            "plan_requirement_ids": self.plan_requirement_ids,
            "plan_trace_bound": self.plan_trace_bound,
            "task_id": self.task_id,
        }

    @property
    def assumptions_digest(self) -> str:
        return content_identity(
            {"assumption_ids": self.assumption_ids, "assumptions": self.assumptions}
        )

    @property
    def validation_bounds_digest(self) -> str:
        return content_identity(self.validation_bounds)

    def receipt_metadata(
        self,
        *,
        obligation: CodeProofObligation | None = None,
    ) -> dict[str, Any]:
        """Return the complete exact metadata required on a code-proof receipt."""

        payload = {
            "receipt_purpose": "code_proof",
            "implementation_binding_id": self.binding_id,
            "accepted_plan_id": self.accepted_plan_id,
            "repository_id": self.repository_id,
            "repository_tree_id": self.repository_tree_id,
            "changed_scope_set_id": self.changed_scope_set_id,
            "changed_scope_ids": self.changed_scope_ids,
            "changed_paths": self.changed_paths,
            "assumption_ids": self.assumption_ids,
            "assumptions_digest": self.assumptions_digest,
            "validation_bounds_digest": self.validation_bounds_digest,
            "test_evidence_ids": self.test_evidence_ids,
            "runtime_evidence_ids": self.runtime_evidence_ids,
            "static_analysis_evidence_ids": self.static_analysis_evidence_ids,
            "evidence_digests": self.evidence_digests,
            "plan_effect_ids": self.plan_effect_ids,
            "plan_requirement_ids": self.plan_requirement_ids,
            "plan_trace_bound": self.plan_trace_bound,
            "task_id": self.task_id,
        }
        if obligation is not None:
            payload["code_proof_obligation_id"] = obligation.obligation_id
            payload["code_proof_scope_ids"] = obligation.ast_scope_ids
        return json.loads(canonical_json(payload))

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "binding_id": self.binding_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImplementationResultBinding":
        schema = str(payload.get("schema") or IMPLEMENTATION_BINDING_SCHEMA)
        if schema != IMPLEMENTATION_BINDING_SCHEMA:
            raise ValueError(f"unsupported implementation binding schema: {schema}")
        return cls(
            accepted_plan_id=str(payload.get("accepted_plan_id") or payload.get("plan_id") or ""),
            repository_id=str(payload.get("repository_id") or ""),
            repository_tree_id=str(payload.get("repository_tree_id") or payload.get("tree_id") or ""),
            changed_scope_set_id=str(payload.get("changed_scope_set_id") or payload.get("scope_set_id") or ""),
            changed_scope_ids=tuple(payload.get("changed_scope_ids") or payload.get("ast_scope_ids") or ()),
            changed_paths=tuple(payload.get("changed_paths") or ()),
            assumption_ids=tuple(payload.get("assumption_ids") or ()),
            assumptions=payload.get("assumptions") or {},
            validation_bounds=payload.get("validation_bounds") or payload.get("bounds") or {},
            test_evidence_ids=tuple(payload.get("test_evidence_ids") or ()),
            runtime_evidence_ids=tuple(payload.get("runtime_evidence_ids") or ()),
            static_analysis_evidence_ids=tuple(payload.get("static_analysis_evidence_ids") or ()),
            evidence_digests=payload.get("evidence_digests") or {},
            plan_effect_ids=tuple(payload.get("plan_effect_ids") or payload.get("planned_effect_ids") or payload.get("effect_ids") or ()),
            plan_requirement_ids=tuple(payload.get("plan_requirement_ids") or ()),
            plan_trace_bound=payload.get("plan_trace_bound"),
            task_id=str(payload.get("task_id") or ""),
            binding_id=str(payload.get("binding_id") or payload.get("content_id") or ""),
        )


_OBLIGATION_STATEMENTS = {
    ImplementationObligationKind.CHANGED_SYMBOL: (
        "Every changed executable symbol satisfies its reviewed implementation contract."
    ),
    ImplementationObligationKind.INTERFACE: (
        "Every changed public interface remains compatible with its reviewed consumers."
    ),
    ImplementationObligationKind.EFFECT: (
        "Every changed implementation effect conforms to the accepted plan effects."
    ),
    ImplementationObligationKind.TEST: (
        "Required tests pass against the exact candidate tree and changed scope."
    ),
    ImplementationObligationKind.RUNTIME_EVIDENCE: (
        "Required runtime observations satisfy their declared finite validation bounds."
    ),
    ImplementationObligationKind.STATIC_ANALYSIS: (
        "Required static analysis passes against the exact candidate tree and changed scope."
    ),
}


@dataclass(frozen=True)
class ImplementationProofObligation(CodeProofObligation):
    """A canonical code obligation annotated with its derivation family."""

    kind: ImplementationObligationKind = ImplementationObligationKind.CHANGED_SYMBOL
    subject: str = ""
    binding_id: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "kind", ImplementationObligationKind(self.kind))
        object.__setattr__(self, "subject", str(self.subject or "").strip())
        object.__setattr__(self, "binding_id", str(self.binding_id or "").strip())
        if not self.subject:
            raise ValueError("implementation obligation requires a subject")
        if not self.binding_id:
            raise ValueError("implementation obligation requires a binding_id")
        if self.metadata.get("implementation_binding_id") != self.binding_id:
            raise ValueError("implementation obligation metadata binding is inconsistent")

    def _payload(self) -> dict[str, Any]:
        payload = super()._payload()
        payload.update(
            {
                "implementation_obligation_kind": self.kind.value,
                "implementation_subject": self.subject,
                "implementation_binding_id": self.binding_id,
            }
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImplementationProofObligation":
        metadata = payload.get("metadata") or {}
        result = cls(
            repository_id=payload.get("repository_id", ""),
            repository_tree_id=payload.get("repository_tree_id", ""),
            ast_scope_ids=tuple(payload.get("ast_scope_ids") or ()),
            statement=payload.get("statement", ""),
            premise_ids=tuple(payload.get("premise_ids") or ()),
            template_id=payload.get("template_id", ""),
            template_version=payload.get("template_version", ""),
            template_semantic_hash=payload.get("template_semantic_hash", ""),
            invariant_class=payload.get("invariant_class", ""),
            task_id=payload.get("task_id", ""),
            required_assurance=payload.get(
                "required_assurance", AssuranceLevel.KERNEL_VERIFIED
            ),
            fallback_checks=tuple(payload.get("fallback_checks") or ()),
            metadata=metadata,
            kind=payload.get(
                "implementation_obligation_kind",
                metadata.get("obligation_kind", ImplementationObligationKind.CHANGED_SYMBOL),
            ),
            subject=payload.get(
                "implementation_subject", metadata.get("subject", "")
            ),
            binding_id=payload.get(
                "implementation_binding_id",
                metadata.get("implementation_binding_id", ""),
            ),
        )
        claimed = payload.get("obligation_id") or payload.get("content_id")
        if claimed and claimed != result.obligation_id:
            raise ValueError("implementation obligation identity does not match payload")
        return result


@dataclass(frozen=True)
class ImplementationObligationSet:
    """Fresh obligations plus the exact evidence and binding that derived them."""

    binding: ImplementationResultBinding
    obligations: tuple[ImplementationProofObligation, ...]
    evidence: tuple[ImplementationResultEvidence, ...] = ()
    obligation_kinds: Mapping[str, str] = field(default_factory=dict)
    incomplete_reason_codes: tuple[str, ...] = ()
    set_id: str = ""

    def __post_init__(self) -> None:
        binding = (
            self.binding
            if isinstance(self.binding, ImplementationResultBinding)
            else ImplementationResultBinding.from_dict(self.binding)
        )
        object.__setattr__(self, "binding", binding)
        obligations = tuple(
            sorted(
                (
                    item
                    if isinstance(item, ImplementationProofObligation)
                    else ImplementationProofObligation.from_dict(item)
                    for item in self.obligations
                ),
                key=lambda item: item.obligation_id,
            )
        )
        if len({item.obligation_id for item in obligations}) != len(obligations):
            raise ValueError("implementation obligations contain duplicate identities")
        for item in obligations:
            if item.repository_tree_id != binding.repository_tree_id:
                raise ValueError("implementation obligation tree does not match binding")
            if item.repository_id != binding.repository_id:
                raise ValueError("implementation obligation repository does not match binding")
            if item.metadata.get("implementation_binding_id") != binding.binding_id:
                raise ValueError("implementation obligation is not bound to its result")
        object.__setattr__(self, "obligations", obligations)
        evidence = tuple(
            sorted(
                (
                    item
                    if isinstance(item, ImplementationResultEvidence)
                    else ImplementationResultEvidence.from_dict(item)
                    for item in self.evidence
                ),
                key=lambda item: item.evidence_id,
            )
        )
        if len({item.evidence_id for item in evidence}) != len(evidence):
            raise ValueError("implementation evidence contains duplicate identities")
        evidence_ids_by_kind = {
            ImplementationEvidenceKind.TEST: tuple(
                item.evidence_id
                for item in evidence
                if item.kind is ImplementationEvidenceKind.TEST
            ),
            ImplementationEvidenceKind.RUNTIME: tuple(
                item.evidence_id
                for item in evidence
                if item.kind is ImplementationEvidenceKind.RUNTIME
            ),
            ImplementationEvidenceKind.STATIC_ANALYSIS: tuple(
                item.evidence_id
                for item in evidence
                if item.kind
                in {
                    ImplementationEvidenceKind.STATIC_ANALYSIS,
                    ImplementationEvidenceKind.TYPE_CHECK,
                }
            ),
        }
        if (
            evidence_ids_by_kind[ImplementationEvidenceKind.TEST]
            != binding.test_evidence_ids
            or evidence_ids_by_kind[ImplementationEvidenceKind.RUNTIME]
            != binding.runtime_evidence_ids
            or evidence_ids_by_kind[ImplementationEvidenceKind.STATIC_ANALYSIS]
            != binding.static_analysis_evidence_ids
        ):
            raise ValueError(
                "implementation evidence identities do not match binding"
            )
        evidence_digests = {
            item.evidence_id: item.evidence_digest for item in evidence
        }
        if evidence_digests != binding.evidence_digests:
            raise ValueError("implementation evidence digests do not match binding")
        object.__setattr__(self, "evidence", evidence)
        kinds = {
            str(key): ImplementationObligationKind(value).value
            for key, value in dict(self.obligation_kinds).items()
        }
        if set(kinds) != {item.obligation_id for item in obligations}:
            raise ValueError("obligation_kinds must classify every obligation exactly")
        object.__setattr__(self, "obligation_kinds", dict(sorted(kinds.items())))
        object.__setattr__(
            self,
            "incomplete_reason_codes",
            _canonical_strings(self.incomplete_reason_codes),
        )
        supplied = str(self.set_id or "").strip()
        object.__setattr__(self, "set_id", "")
        derived = content_identity(self._identity_payload())
        if supplied and supplied != derived:
            raise ValueError("implementation obligation-set identity does not match payload")
        object.__setattr__(self, "set_id", derived)

    @property
    def binding_id(self) -> str:
        return self.binding.binding_id

    @property
    def obligation_set_id(self) -> str:
        """Descriptive alias for the canonical set identity."""

        return self.set_id

    @property
    def obligation_ids(self) -> tuple[str, ...]:
        return tuple(item.obligation_id for item in self.obligations)

    @property
    def complete(self) -> bool:
        return bool(self.obligations) and not self.incomplete_reason_codes

    def by_kind(
        self, kind: ImplementationObligationKind | str
    ) -> tuple[CodeProofObligation, ...]:
        normalized = ImplementationObligationKind(kind).value
        return tuple(
            item
            for item in self.obligations
            if self.obligation_kinds.get(item.obligation_id) == normalized
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": IMPLEMENTATION_OBLIGATION_SET_SCHEMA,
            "binding": self.binding.to_dict(),
            "obligations": tuple(item.to_dict() for item in self.obligations),
            "evidence": tuple(item.to_dict() for item in self.evidence),
            "obligation_kinds": self.obligation_kinds,
            "incomplete_reason_codes": self.incomplete_reason_codes,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "set_id": self.set_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImplementationObligationSet":
        schema = str(payload.get("schema") or IMPLEMENTATION_OBLIGATION_SET_SCHEMA)
        if schema != IMPLEMENTATION_OBLIGATION_SET_SCHEMA:
            raise ValueError(f"unsupported implementation obligation-set schema: {schema}")
        return cls(
            binding=ImplementationResultBinding.from_dict(payload.get("binding") or {}),
            obligations=tuple(
                ImplementationProofObligation.from_dict(item)
                for item in payload.get("obligations") or ()
            ),
            evidence=tuple(
                ImplementationResultEvidence.from_dict(item)
                for item in payload.get("evidence") or ()
            ),
            obligation_kinds=payload.get("obligation_kinds") or {},
            incomplete_reason_codes=tuple(payload.get("incomplete_reason_codes") or ()),
            set_id=str(payload.get("set_id") or payload.get("content_id") or ""),
        )


def _evidence_values(
    values: Iterable[ImplementationResultEvidence | Mapping[str, Any]],
) -> tuple[ImplementationResultEvidence, ...]:
    return tuple(
        item
        if isinstance(item, ImplementationResultEvidence)
        else ImplementationResultEvidence.from_dict(item)
        for item in values
    )


def derive_fresh_implementation_obligations(
    scope_set: CodeProofScopeSet,
    *,
    accepted_plan_id: str = "",
    accepted_plan: Any = None,
    repository_id: str = "",
    repository_tree_id: str = "",
    assumption_ids: Iterable[str] = (),
    assumptions: Mapping[str, Any] | Iterable[str] = (),
    validation_bounds: Mapping[str, Any] | None = None,
    test_evidence: Iterable[ImplementationResultEvidence | Mapping[str, Any]] = (),
    runtime_evidence: Iterable[ImplementationResultEvidence | Mapping[str, Any]] = (),
    static_analysis_evidence: Iterable[ImplementationResultEvidence | Mapping[str, Any]] = (),
    planned_effect_ids: Iterable[str] = (),
    plan_requirement_ids: Iterable[str] = (),
    plan_trace_bound: int | None = None,
    task_id: str = "",
    required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED,
) -> ImplementationObligationSet:
    """Derive fresh post-execution obligations from code and bounded evidence."""

    if not isinstance(scope_set, CodeProofScopeSet):
        raise TypeError("scope_set must be a CodeProofScopeSet")
    plan = accepted_plan
    if plan is not None:
        candidate_plan_id = str(
            getattr(plan, "plan_id", "")
            or getattr(plan, "content_id", "")
            or (plan.get("plan_id", "") if isinstance(plan, Mapping) else "")
        ).strip()
        if accepted_plan_id and accepted_plan_id != candidate_plan_id:
            raise ValueError("accepted plan identity does not match accepted_plan")
        accepted_plan_id = candidate_plan_id
        plan_tree = str(
            getattr(plan, "repository_tree_id", "")
            or (plan.get("repository_tree_id", "") if isinstance(plan, Mapping) else "")
        ).strip()
        if repository_tree_id and plan_tree and repository_tree_id != plan_tree:
            raise ValueError("accepted plan and implementation tree do not match")
        repository_tree_id = repository_tree_id or plan_tree
        plan_effects = (
            getattr(plan, "effects", ())
            if not isinstance(plan, Mapping)
            else plan.get("effects", ())
        )
        extracted_effects = [
            str(
                getattr(item, "effect_id", "")
                or (item.get("effect_id", "") if isinstance(item, Mapping) else "")
            )
            for item in plan_effects
        ]
        planned_effect_ids = (*planned_effect_ids, *extracted_effects)
        trace_bound = (
            getattr(plan, "trace_bound", None)
            if not isinstance(plan, Mapping)
            else plan.get("trace_bound")
        )
        if validation_bounds is None:
            validation_bounds = (
                {"trace_bound": trace_bound} if trace_bound is not None else {}
            )
        if plan_trace_bound is None:
            plan_trace_bound = trace_bound
        plan_requirements = (
            getattr(plan, "evidence_requirements", ())
            if not isinstance(plan, Mapping)
            else plan.get("evidence_requirements", ())
        )
        extracted_requirements = [
            str(
                getattr(item, "requirement_id", "")
                or (
                    item.get("requirement_id", "")
                    if isinstance(item, Mapping)
                    else ""
                )
            )
            for item in plan_requirements
        ]
        plan_requirement_ids = (*plan_requirement_ids, *extracted_requirements)
    accepted_plan_id = str(accepted_plan_id or "").strip()
    repository_tree_id = str(repository_tree_id or "").strip()
    if not accepted_plan_id:
        raise ValueError("accepted_plan_id is required")
    if not repository_tree_id:
        raise ValueError("repository_tree_id is required")
    if not scope_set.scopes or not scope_set.changed_paths:
        raise ValueError("fresh obligations require a nonempty changed scope")

    evidence = _evidence_values(
        (*tuple(test_evidence), *tuple(runtime_evidence), *tuple(static_analysis_evidence))
    )
    expected_kinds = {
        ImplementationEvidenceKind.TEST: tuple(
            item.evidence_id for item in evidence if item.kind is ImplementationEvidenceKind.TEST
        ),
        ImplementationEvidenceKind.RUNTIME: tuple(
            item.evidence_id for item in evidence if item.kind is ImplementationEvidenceKind.RUNTIME
        ),
        ImplementationEvidenceKind.STATIC_ANALYSIS: tuple(
            item.evidence_id
            for item in evidence
            if item.kind in {
                ImplementationEvidenceKind.STATIC_ANALYSIS,
                ImplementationEvidenceKind.TYPE_CHECK,
            }
        ),
    }
    assumptions_mapping = (
        _canonical_mapping(assumptions)
        if isinstance(assumptions, Mapping)
        else {}
    )
    assumptions_combined = _canonical_strings(
        (
            *tuple(assumption_ids),
            *(() if isinstance(assumptions, Mapping) else tuple(assumptions)),
        )
    )
    binding = ImplementationResultBinding(
        accepted_plan_id=accepted_plan_id,
        repository_id=str(repository_id or "").strip(),
        repository_tree_id=repository_tree_id,
        changed_scope_set_id=scope_set.scope_set_id,
        changed_scope_ids=scope_set.scope_ids,
        changed_paths=scope_set.changed_paths,
        assumption_ids=assumptions_combined,
        assumptions=assumptions_mapping,
        validation_bounds=validation_bounds or {},
        test_evidence_ids=expected_kinds[ImplementationEvidenceKind.TEST],
        runtime_evidence_ids=expected_kinds[ImplementationEvidenceKind.RUNTIME],
        static_analysis_evidence_ids=expected_kinds[ImplementationEvidenceKind.STATIC_ANALYSIS],
        evidence_digests={
            item.evidence_id: item.evidence_digest for item in evidence
        },
        plan_effect_ids=tuple(planned_effect_ids),
        plan_requirement_ids=tuple(plan_requirement_ids),
        plan_trace_bound=plan_trace_bound,
        task_id=task_id,
    )

    incomplete: list[str] = []
    if scope_set.conservative:
        incomplete.append("conservative_changed_scope")
    for item in evidence:
        if item.repository_tree_id != repository_tree_id:
            incomplete.append("evidence_tree_mismatch")
        if item.repository_id and item.repository_id != binding.repository_id:
            incomplete.append("evidence_repository_mismatch")
        if item.accepted_plan_id and item.accepted_plan_id != accepted_plan_id:
            incomplete.append("evidence_plan_mismatch")
        if item.scope_ids and not set(item.scope_ids).issubset(binding.changed_scope_ids):
            incomplete.append("evidence_scope_mismatch")
        if item.assumption_ids and item.assumption_ids != binding.assumption_ids:
            incomplete.append("evidence_assumptions_mismatch")
        if not item.passed:
            incomplete.append("failed_implementation_evidence")
        if item.contradictory:
            incomplete.append("contradictory_implementation_evidence")

    groups: list[tuple[ImplementationObligationKind, tuple[CodeProofScope, ...], tuple[str, ...]]] = []
    symbols = scope_set.by_kind(ProofScopeKind.QUALIFIED_SYMBOL)
    interfaces = scope_set.by_kind(ProofScopeKind.INTERFACE)
    effects = tuple(
        sorted(
            (
                *scope_set.by_kind(ProofScopeKind.CALL),
                *scope_set.by_kind(ProofScopeKind.STATE_TRANSITION),
            ),
            key=lambda item: item.scope_id,
        )
    )
    if symbols:
        groups.append((ImplementationObligationKind.CHANGED_SYMBOL, symbols, ()))
    if interfaces:
        groups.append((ImplementationObligationKind.INTERFACE, interfaces, ()))
    if effects or binding.planned_effect_ids:
        groups.append((ImplementationObligationKind.EFFECT, effects or symbols, binding.planned_effect_ids))
    if binding.test_evidence_ids:
        groups.append((ImplementationObligationKind.TEST, symbols or tuple(scope_set.scopes), binding.test_evidence_ids))
    if binding.runtime_evidence_ids:
        groups.append((ImplementationObligationKind.RUNTIME_EVIDENCE, effects or symbols or tuple(scope_set.scopes), binding.runtime_evidence_ids))
    if binding.static_analysis_evidence_ids:
        groups.append((ImplementationObligationKind.STATIC_ANALYSIS, symbols or tuple(scope_set.scopes), binding.static_analysis_evidence_ids))
    if not groups:
        incomplete.append("no_derivable_implementation_obligations")

    obligations: list[ImplementationProofObligation] = []
    kinds: dict[str, str] = {}
    for kind, scopes, evidence_ids in groups:
        selected_scope_ids = tuple(
            sorted(
                {
                    item.scope_id
                    for item in scopes
                    if item.kind not in {
                        ProofScopeKind.CHANGED_PATH,
                        ProofScopeKind.CONSERVATIVE_FILE,
                    }
                    and not item.conservative
                }
            )
        )
        if not selected_scope_ids:
            incomplete.append(f"no_supported_{kind.value}_scope")
            continue
        semantic_definition = {
            "kind": kind.value,
            "statement": _OBLIGATION_STATEMENTS[kind],
            "version": "1",
        }
        subject_values = tuple(
            sorted({item.value for item in scopes if item.value})
        )
        subject = ", ".join(subject_values or evidence_ids or binding.plan_effect_ids)
        obligation = ImplementationProofObligation(
            repository_id=binding.repository_id,
            repository_tree_id=binding.repository_tree_id,
            ast_scope_ids=selected_scope_ids,
            statement=_OBLIGATION_STATEMENTS[kind],
            premise_ids=binding.assumption_ids,
            template_id=f"reviewed-implementation-{kind.value}",
            template_version="1",
            template_semantic_hash=content_identity(semantic_definition),
            invariant_class=f"implementation_{kind.value}",
            task_id=binding.task_id,
            required_assurance=required_assurance,
            metadata={
                "implementation_binding_id": binding.binding_id,
                "scope_set_id": binding.changed_scope_set_id,
                "accepted_plan_id": binding.accepted_plan_id,
                "assumption_ids": binding.assumption_ids,
                "validation_bounds": binding.validation_bounds,
                "evidence_ids": evidence_ids,
                "obligation_kind": kind.value,
                "subject": subject,
            },
            kind=kind,
            subject=subject,
            binding_id=binding.binding_id,
        )
        obligations.append(obligation)
        kinds[obligation.obligation_id] = kind.value

    return ImplementationObligationSet(
        binding=binding,
        obligations=tuple(obligations),
        evidence=evidence,
        obligation_kinds=kinds,
        incomplete_reason_codes=tuple(incomplete),
    )


@dataclass(frozen=True)
class CodeProofReceiptBindingResult:
    receipt_id: str
    obligation_id: str
    binding_id: str
    valid: bool
    stale: bool = False
    contradictory: bool = False
    reason_codes: tuple[str, ...] = ()
    authoritative_assurance: AssuranceLevel = AssuranceLevel.UNVERIFIED
    authoritative_verdict: ProofVerdict = ProofVerdict.INCONCLUSIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CODE_PROOF_BINDING_RESULT_SCHEMA,
            "receipt_id": self.receipt_id,
            "obligation_id": self.obligation_id,
            "binding_id": self.binding_id,
            "valid": self.valid,
            "stale": self.stale,
            "contradictory": self.contradictory,
            "reason_codes": list(self.reason_codes),
            "authoritative_assurance": self.authoritative_assurance.value,
            "authoritative_verdict": self.authoritative_verdict.value,
        }


def validate_code_proof_receipt_bindings(
    receipt: ProofReceipt | Mapping[str, Any],
    binding: ImplementationResultBinding | ImplementationObligationSet | Mapping[str, Any],
    *,
    obligation: CodeProofObligation | Mapping[str, Any] | None = None,
    required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED,
    plan_assurance: Any = None,
) -> CodeProofReceiptBindingResult:
    """Re-derive every binding needed to accept a code-proof receipt."""

    try:
        proof = (
            receipt
            if isinstance(receipt, ProofReceipt)
            else ProofReceipt.from_dict(receipt)
        )
    except (ContractValidationError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid canonical code-proof receipt: {exc}") from exc
    obligation_set = (
        binding if isinstance(binding, ImplementationObligationSet) else None
    )
    if obligation_set is not None:
        expected_binding = obligation_set.binding
    elif isinstance(binding, ImplementationResultBinding):
        expected_binding = binding
    elif isinstance(binding, Mapping):
        if "binding" in binding:
            obligation_set = ImplementationObligationSet.from_dict(binding)
            expected_binding = obligation_set.binding
        else:
            expected_binding = ImplementationResultBinding.from_dict(binding)
    else:
        raise TypeError("binding must be an ImplementationResultBinding or obligation set")
    if obligation is None and obligation_set is not None:
        matches = [
            item
            for item in obligation_set.obligations
            if item.obligation_id == proof.obligation_id
        ]
        obligation = matches[0] if matches else None
    if isinstance(obligation, Mapping):
        obligation = CodeProofObligation.from_dict(obligation)

    reasons: list[str] = []
    stale = False
    contradictory = proof.authoritative_verdict is ProofVerdict.DISPROVED

    def reject(code: str, *, is_stale: bool = False, is_contradictory: bool = False) -> None:
        nonlocal stale, contradictory
        if code not in reasons:
            reasons.append(code)
        stale = stale or is_stale
        contradictory = contradictory or is_contradictory

    if obligation_set is not None and obligation_set.incomplete_reason_codes:
        reject("implementation_obligation_set_incomplete")
        for code in obligation_set.incomplete_reason_codes:
            reject(
                str(code),
                is_stale=(
                    "mismatch" in str(code)
                    or "conservative" in str(code)
                ),
                is_contradictory=(
                    "contradictory" in str(code)
                    or str(code).startswith("failed_")
                ),
            )
    if obligation is None:
        reject("receipt_not_required_by_fresh_obligation_set", is_stale=True)
    else:
        if proof.obligation_id != obligation.obligation_id:
            reject("proof_obligation_mismatch", is_stale=True)
        if proof.ast_scope_ids != obligation.ast_scope_ids:
            reject("proof_scope_mismatch", is_stale=True)
        if proof.premise_ids != expected_binding.assumption_ids:
            reject("proof_assumptions_or_evidence_mismatch", is_stale=True)
        obligation_binding = str(
            obligation.metadata.get("implementation_binding_id") or ""
        )
        if obligation_binding and obligation_binding != expected_binding.binding_id:
            reject("stale_implementation_binding", is_stale=True)
    if proof.plan_id != expected_binding.accepted_plan_id:
        reject("proof_plan_mismatch", is_stale=True)
    if proof.repository_id != expected_binding.repository_id:
        reject("proof_repository_mismatch", is_stale=True)
    if proof.repository_tree_id != expected_binding.repository_tree_id:
        reject("proof_tree_mismatch", is_stale=True)
    if proof.freshness is not EvidenceFreshness.CURRENT:
        reject("stale_proof_receipt", is_stale=True)
        reject("stale_code_proof_receipt", is_stale=True)
    if proof.authoritative_verdict is not ProofVerdict.PROVED:
        reject(
            "code_proof_not_proved",
            is_contradictory=proof.authoritative_verdict is ProofVerdict.DISPROVED,
        )
    required = AssuranceLevel(required_assurance)
    if not assurance_satisfies(proof.authoritative_assurance, required):
        reject("required_code_assurance_not_satisfied")
    metadata_binding = str(
        proof.metadata.get("implementation_binding_id")
        or proof.metadata.get("binding_id")
        or ""
    )
    if metadata_binding != expected_binding.binding_id:
        reject("receipt_binding_mismatch", is_stale=True)
    expected_metadata = expected_binding.receipt_metadata()
    for key, expected in expected_metadata.items():
        if proof.metadata.get(key) != expected:
            reject(
                f"receipt_{key}_mismatch",
                is_contradictory=True,
            )
    if proof.metadata.get("receipt_purpose") != "code_proof":
        reject("receipt_purpose_not_code_proof", is_contradictory=True)

    if plan_assurance is not None:
        plan_id = str(
            plan_assurance.get("plan_id", "")
            if isinstance(plan_assurance, Mapping)
            else getattr(plan_assurance, "plan_id", "")
        )
        consistency = _canonical_strings(
            plan_assurance.get("consistency_receipt_ids", ())
            if isinstance(plan_assurance, Mapping)
            else getattr(plan_assurance, "consistency_receipt_ids", ())
        )
        conformance = _canonical_strings(
            plan_assurance.get("conformance_receipt_ids", ())
            if isinstance(plan_assurance, Mapping)
            else getattr(plan_assurance, "conformance_receipt_ids", ())
        )
        code_receipts = _canonical_strings(
            plan_assurance.get("code_proof_receipt_ids", ())
            if isinstance(plan_assurance, Mapping)
            else getattr(plan_assurance, "code_proof_receipt_ids", ())
        )
        if plan_id != expected_binding.accepted_plan_id:
            reject("plan_assurance_binding_mismatch", is_stale=True)
        if proof.receipt_id in set(consistency) | set(conformance):
            reject("plan_receipt_reused_as_code_proof")
        if proof.receipt_id not in code_receipts:
            reject("receipt_not_declared_as_code_proof")

    if contradictory:
        reject("contradictory_code_proof_receipt", is_contradictory=True)

    return CodeProofReceiptBindingResult(
        receipt_id=proof.receipt_id,
        obligation_id=proof.obligation_id,
        binding_id=expected_binding.binding_id,
        valid=not reasons,
        stale=stale,
        contradictory=contradictory,
        reason_codes=tuple(reasons),
        authoritative_assurance=proof.authoritative_assurance,
        authoritative_verdict=proof.authoritative_verdict,
    )


# Concise compatibility spellings for integration callers.
derive_implementation_obligations = derive_fresh_implementation_obligations
compile_implementation_obligations = derive_fresh_implementation_obligations
validate_code_proof_receipt_binding = validate_code_proof_receipt_bindings
ImplementationBinding = ImplementationResultBinding
ImplementationEvidence = ImplementationResultEvidence
FreshImplementationObligations = ImplementationObligationSet


__all__ = [
    "ASTProofScope",
    "CODE_OBLIGATION_CACHE_KEY_SCHEMA",
    "CODE_OBLIGATION_REQUEST_SCHEMA",
    "CODE_PROOF_BINDING_RESULT_SCHEMA",
    "CandidateChangeKind",
    "CandidateDiffEntry",
    "CandidateFileDiff",
    "CodeObligationRequest",
    "CodeProofObligationRequest",
    "CodeProofScope",
    "CodeProofScopeSet",
    "CompiledProofScopes",
    "DiffChangeKind",
    "FreshImplementationObligations",
    "IMPLEMENTATION_BINDING_SCHEMA",
    "IMPLEMENTATION_EVIDENCE_SCHEMA",
    "IMPLEMENTATION_OBLIGATION_SET_SCHEMA",
    "ImplementationBinding",
    "ImplementationEvidence",
    "ImplementationEvidenceKind",
    "ImplementationObligationKind",
    "ImplementationObligationSet",
    "ImplementationProofObligation",
    "ImplementationResultBinding",
    "ImplementationResultEvidence",
    "CodeProofReceiptBindingResult",
    "PROOF_SCOPE_SCHEMA",
    "PROOF_SCOPE_SET_SCHEMA",
    "ProofScopeCompilationStats",
    "ProofScopeCompilation",
    "ProofScopeKind",
    "ProofObligationRequest",
    "ProofScopeSet",
    "ProofScopeType",
    "TypedASTProofScope",
    "build_code_proof_obligation",
    "build_obligation_cache_key",
    "code_proof_obligation_cache_identity",
    "collect_git_candidate_diff",
    "compile_candidate_diff",
    "compile_candidate_diffs",
    "compile_candidate_diff_scopes",
    "compile_candidate_proof_scopes",
    "compile_code_proof_scopes",
    "compile_ast_proof_scopes",
    "compile_proof_scopes",
    "compile_implementation_obligations",
    "derive_fresh_implementation_obligations",
    "derive_implementation_obligations",
    "materialize_code_proof_obligation",
    "obligation_cache_identity",
    "parse_unified_diff",
    "validate_code_proof_receipt_binding",
    "validate_code_proof_receipt_bindings",
]
