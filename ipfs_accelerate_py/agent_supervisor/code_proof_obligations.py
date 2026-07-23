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
from .formal_verification_contracts import canonical_json, content_identity


PROOF_SCOPE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/code-proof-scope@1"
PROOF_SCOPE_SET_SCHEMA = "ipfs_accelerate_py/agent-supervisor/code-proof-scope-set@1"


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
        return cls(
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


__all__ = [
    "ASTProofScope",
    "CandidateChangeKind",
    "CandidateDiffEntry",
    "CandidateFileDiff",
    "CodeProofScope",
    "CodeProofScopeSet",
    "CompiledProofScopes",
    "DiffChangeKind",
    "PROOF_SCOPE_SCHEMA",
    "PROOF_SCOPE_SET_SCHEMA",
    "ProofScopeCompilationStats",
    "ProofScopeCompilation",
    "ProofScopeKind",
    "ProofScopeSet",
    "ProofScopeType",
    "TypedASTProofScope",
    "collect_git_candidate_diff",
    "compile_candidate_diff",
    "compile_candidate_diffs",
    "compile_candidate_diff_scopes",
    "compile_candidate_proof_scopes",
    "compile_code_proof_scopes",
    "compile_ast_proof_scopes",
    "compile_proof_scopes",
    "parse_unified_diff",
]
