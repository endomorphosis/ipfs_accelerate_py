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

from ..core.conflict_graph import (
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
PROOF_CANDIDATE_NON_AUTHORITY_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proof-candidate-non-authority-evidence@1"
)
STRICT_VALIDATION_PROOF_COMPLETION_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "strict-validation-proof-completion-evidence@1"
)
STRICT_VALIDATION_PARENT_OBJECTIVE_ID = "ASI-G040"
STRICT_VALIDATION_PROOF_GATE_KINDS = ("semantic_proof",)
PROOF_CANDIDATE_NON_AUTHORITY_REQUIREMENT_ID = (
    "006818797857632260116084792540150258746"
)
PROOF_CANDIDATE_NON_AUTHORITY_OBJECTIVE_ID = "ASI-G102"
PROOF_CANDIDATE_NON_AUTHORITY_OBJECTIVE_REVISION = "ASI-G102@asi-070"
PROOF_CANDIDATE_NON_AUTHORITY_COMPLETION_ANALYZER_VERSION = (
    "asi-g102-objective-validation@1"
)
PROOF_CANDIDATE_NON_AUTHORITY_COMPLETION_CONFIGURATION_REVISION = (
    "strict-proof-candidate-non-authority-completion@1"
)
PROOF_CANDIDATE_NON_AUTHORITY_ACCEPTANCE_CRITERIA = (
    (
        "the accepted proposal and population-complete passing validation DAG "
        "remain non-authoritative for code proof and completion"
    ),
    (
        "the canonical provider candidate verdict and assurance are "
        "independently re-derived from typed evidence"
    ),
    (
        "the candidate is rejected against the exact fresh implementation "
        "obligation and required assurance"
    ),
    (
        "strict completion admission remains closed and replays the candidate "
        "binding rejection"
    ),
    (
        "tamper, replay, detached summaries, and forged authority fail closed "
        "across the full current-tree chain"
    ),
    (
        "the exact proof-candidate non-authority requirement is emitted only "
        "by a tamper-evident current-tree witness"
    ),
)
# Concise public spellings used by the objective-completion bridge.  The
# longer names preserve the evidence-record namespace and remain the canonical
# documentation spelling.
PROOF_CANDIDATE_OBJECTIVE_ID = PROOF_CANDIDATE_NON_AUTHORITY_OBJECTIVE_ID
PROOF_CANDIDATE_OBJECTIVE_REVISION = (
    PROOF_CANDIDATE_NON_AUTHORITY_OBJECTIVE_REVISION
)
PROOF_CANDIDATE_COMPLETION_ANALYZER_VERSION = (
    PROOF_CANDIDATE_NON_AUTHORITY_COMPLETION_ANALYZER_VERSION
)
PROOF_CANDIDATE_COMPLETION_CONFIGURATION_REVISION = (
    PROOF_CANDIDATE_NON_AUTHORITY_COMPLETION_CONFIGURATION_REVISION
)
PROOF_CANDIDATE_ACCEPTANCE_CRITERIA = (
    PROOF_CANDIDATE_NON_AUTHORITY_ACCEPTANCE_CRITERIA
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
    effect_scope_map: Mapping[str, Any] = field(default_factory=dict)
    plan_requirement_ids: tuple[str, ...] = ()
    plan_trace_bound: int | None = None
    task_id: str = ""
    goal_id: str = ""
    code_proof_toolchain_id: str = ""
    code_proof_policy_id: str = ""
    proposal_validation_receipt_id: str = ""
    proposal_accepted: bool | None = None
    binding_id: str = ""
    validation_dag_receipt_id: str = ""
    validation_policy_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "accepted_plan_id",
            "repository_id",
            "repository_tree_id",
            "changed_scope_set_id",
            "task_id",
            "goal_id",
            "code_proof_toolchain_id",
            "code_proof_policy_id",
            "proposal_validation_receipt_id",
            "validation_dag_receipt_id",
            "validation_policy_id",
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
        if self.proposal_accepted is not None and not isinstance(
            self.proposal_accepted, bool
        ):
            raise TypeError("proposal_accepted must be boolean or None")
        if self.proposal_validation_receipt_id and self.proposal_accepted is not True:
            raise ValueError(
                "a proposal validation receipt must represent accepted output"
            )
        if bool(self.validation_dag_receipt_id) != bool(self.validation_policy_id):
            raise ValueError(
                "validation DAG receipt and policy identities must be supplied together"
            )
        if self.validation_dag_receipt_id and not self.proposal_validation_receipt_id:
            raise ValueError(
                "a validation DAG receipt must be bound to an accepted proposal receipt"
            )
        object.__setattr__(self, "assumptions", _canonical_mapping(self.assumptions))
        object.__setattr__(self, "validation_bounds", _canonical_mapping(self.validation_bounds))
        effect_scope_map = {
            str(effect_id).strip(): list(_canonical_strings(scope_ids))
            for effect_id, scope_ids in dict(self.effect_scope_map or {}).items()
            if str(effect_id).strip()
        }
        object.__setattr__(
            self,
            "effect_scope_map",
            _canonical_mapping(dict(sorted(effect_scope_map.items()))),
        )
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
        payload = {
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
            "effect_scope_map": self.effect_scope_map,
            "plan_requirement_ids": self.plan_requirement_ids,
            "plan_trace_bound": self.plan_trace_bound,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "code_proof_toolchain_id": self.code_proof_toolchain_id,
            "code_proof_policy_id": self.code_proof_policy_id,
        }
        if self.proposal_validation_receipt_id or self.proposal_accepted is not None:
            payload["proposal_validation_receipt_id"] = (
                self.proposal_validation_receipt_id
            )
            payload["proposal_accepted"] = self.proposal_accepted
        if self.validation_dag_receipt_id:
            payload["validation_dag_receipt_id"] = self.validation_dag_receipt_id
            payload["validation_policy_id"] = self.validation_policy_id
        return payload

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
            "effect_scope_map": self.effect_scope_map,
            "plan_requirement_ids": self.plan_requirement_ids,
            "plan_trace_bound": self.plan_trace_bound,
            "task_id": self.task_id,
        }
        if self.goal_id:
            payload["goal_id"] = self.goal_id
        if self.code_proof_toolchain_id:
            payload["code_proof_toolchain_id"] = self.code_proof_toolchain_id
        if self.code_proof_policy_id:
            payload["code_proof_policy_id"] = self.code_proof_policy_id
        if self.proposal_validation_receipt_id:
            payload["proposal_validation_receipt_id"] = (
                self.proposal_validation_receipt_id
            )
            payload["proposal_accepted"] = self.proposal_accepted
        if self.validation_dag_receipt_id:
            payload["validation_dag_receipt_id"] = self.validation_dag_receipt_id
            payload["validation_policy_id"] = self.validation_policy_id
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
            effect_scope_map=payload.get("effect_scope_map") or {},
            plan_requirement_ids=tuple(payload.get("plan_requirement_ids") or ()),
            plan_trace_bound=payload.get("plan_trace_bound"),
            task_id=str(payload.get("task_id") or ""),
            goal_id=str(payload.get("goal_id") or payload.get("objective_id") or ""),
            code_proof_toolchain_id=str(
                payload.get("code_proof_toolchain_id")
                or payload.get("expected_toolchain_id")
                or ""
            ),
            code_proof_policy_id=str(
                payload.get("code_proof_policy_id")
                or payload.get("expected_proof_policy_id")
                or ""
            ),
            proposal_validation_receipt_id=str(
                payload.get("proposal_validation_receipt_id")
                or payload.get("proposal_receipt_id")
                or ""
            ),
            proposal_accepted=payload.get("proposal_accepted"),
            validation_dag_receipt_id=str(
                payload.get("validation_dag_receipt_id")
                or payload.get("validation_receipt_id")
                or ""
            ),
            validation_policy_id=str(
                payload.get("validation_policy_id") or ""
            ),
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
        required_kinds: set[ImplementationObligationKind] = set()
        if binding.plan_effect_ids or binding.effect_scope_map:
            required_kinds.add(ImplementationObligationKind.EFFECT)
        if binding.test_evidence_ids:
            required_kinds.add(ImplementationObligationKind.TEST)
        if binding.runtime_evidence_ids:
            required_kinds.add(ImplementationObligationKind.RUNTIME_EVIDENCE)
        if binding.static_analysis_evidence_ids:
            required_kinds.add(ImplementationObligationKind.STATIC_ANALYSIS)
        present_kinds = {
            ImplementationObligationKind(value) for value in kinds.values()
        }
        if required_kinds - present_kinds:
            raise ValueError(
                "implementation obligation population omits a binding-required family"
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

    @property
    def proof_authoritative(self) -> bool:
        """Derived obligations describe proof work; they never satisfy it."""

        return False

    @property
    def completion_authoritative(self) -> bool:
        return False

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
        return {
            **self._identity_payload(),
            "set_id": self.set_id,
            "proof_authoritative": False,
            "completion_authoritative": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImplementationObligationSet":
        schema = str(payload.get("schema") or IMPLEMENTATION_OBLIGATION_SET_SCHEMA)
        if schema != IMPLEMENTATION_OBLIGATION_SET_SCHEMA:
            raise ValueError(f"unsupported implementation obligation-set schema: {schema}")
        for name in ("proof_authoritative", "completion_authoritative"):
            if payload.get(name) not in (None, False):
                raise ValueError(
                    f"implementation obligation set cannot claim {name}"
                )
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
    effect_scope_map: Mapping[str, Iterable[str]] | None = None,
    plan_requirement_ids: Iterable[str] = (),
    plan_trace_bound: int | None = None,
    task_id: str = "",
    goal_id: str = "",
    code_proof_toolchain_id: str = "",
    code_proof_policy_id: str = "",
    proposal_validation: Any = None,
    validation_dag: Any = None,
    require_validation_dag: bool = False,
    expected_validation_policy_id: str = "",
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
        plan_goal_ids = _canonical_strings(
            str(
                getattr(item, "goal_id", "")
                or (item.get("goal_id", "") if isinstance(item, Mapping) else "")
            )
            for item in (
                getattr(plan, "goals", ())
                if not isinstance(plan, Mapping)
                else plan.get("goals", ())
            )
        )
        if goal_id and plan_goal_ids and goal_id not in plan_goal_ids:
            raise ValueError("implementation goal does not occur in accepted plan")
        if not goal_id and len(plan_goal_ids) == 1:
            goal_id = plan_goal_ids[0]
        plan_metadata = (
            getattr(plan, "metadata", {})
            if not isinstance(plan, Mapping)
            else plan.get("metadata", {})
        )
        if not isinstance(plan_metadata, Mapping):
            raise ValueError("accepted plan metadata must be a mapping")
        code_proof_toolchain_id = (
            code_proof_toolchain_id
            or str(
                plan_metadata.get("code_proof_toolchain_id")
                or plan_metadata.get("toolchain_id")
                or ""
            ).strip()
        )
        code_proof_policy_id = (
            code_proof_policy_id
            or str(
                plan_metadata.get("code_proof_policy_id")
                or plan_metadata.get("proof_policy_id")
                or ""
            ).strip()
        )
        if effect_scope_map is None:
            plan_effect_scope_map = plan_metadata.get("effect_scope_map")
            if plan_effect_scope_map is not None:
                if not isinstance(plan_effect_scope_map, Mapping):
                    raise ValueError(
                        "accepted plan effect_scope_map must be a mapping"
                    )
                effect_scope_map = plan_effect_scope_map
        plan_preconditions = (
            getattr(plan, "preconditions", ())
            if not isinstance(plan, Mapping)
            else plan.get("preconditions", ())
        )
        extracted_assumptions = [
            str(
                getattr(item, "precondition_id", "")
                or (
                    item.get("precondition_id", "")
                    if isinstance(item, Mapping)
                    else ""
                )
            )
            for item in plan_preconditions
        ]
        assumption_ids = (*assumption_ids, *extracted_assumptions)
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

    proposal_receipt_id = ""
    proposal_accepted: bool | None = None
    proposal_result = None
    if proposal_validation is not None:
        # Kept local to avoid introducing a module-import cycle.
        from ..validation.proposal_validation import ProposalValidationResult

        proposal_result = (
            proposal_validation
            if isinstance(proposal_validation, ProposalValidationResult)
            else ProposalValidationResult.from_dict(proposal_validation)
        )
        proposal_accepted = proposal_result.accepted
        if not proposal_accepted:
            raise ValueError(
                "rejected proposal cannot produce implementation proof obligations"
            )
        if proposal_result.proposal.accepted_plan_id != accepted_plan_id:
            raise ValueError("proposal and implementation plan do not match")
        if proposal_result.proposal.repository_tree_id != repository_tree_id:
            raise ValueError("proposal and implementation tree do not match")
        if tuple(proposal_result.proposal.changed_paths) != tuple(
            scope_set.changed_paths
        ):
            raise ValueError("proposal and implementation changed scopes do not match")
        # Paths alone are not a semantic binding.  Recompile the accepted
        # candidate sources and require the exact AST/interface/effect scope
        # population supplied to this derivation.
        accepted_scopes = compile_candidate_proof_scopes(
            proposal_result.proposal.candidate_diff
        )
        if (
            accepted_scopes.scope_set_id != scope_set.scope_set_id
            or accepted_scopes.scope_ids != scope_set.scope_ids
        ):
            raise ValueError(
                "proposal and implementation AST/interface/effect scopes do not match"
            )
        proposal_goal_id = str(
            proposal_result.proposal.objective_id or ""
        ).strip()
        if goal_id and proposal_goal_id and goal_id != proposal_goal_id:
            raise ValueError("proposal and implementation goals do not match")
        goal_id = goal_id or proposal_goal_id
        proposal_receipt_id = proposal_result.receipt.receipt_id

    validation_dag_receipt_id = ""
    validation_policy_id = ""
    expected_validation_policy_id = str(
        expected_validation_policy_id or ""
    ).strip()
    if validation_dag is None:
        if require_validation_dag or expected_validation_policy_id:
            raise ValueError(
                "validation DAG receipt is required to derive implementation proof obligations"
            )
    else:
        # Kept local to avoid introducing a module-import cycle.
        from ..validation.validation_scheduler import ValidationDAGReceipt

        dag = (
            validation_dag
            if isinstance(validation_dag, ValidationDAGReceipt)
            else ValidationDAGReceipt.from_dict(validation_dag)
        )
        if proposal_result is None:
            raise ValueError(
                "validation DAG requires its accepted proposal validation result"
            )
        if dag.proposal_receipt_id != proposal_receipt_id:
            raise ValueError(
                "validation DAG and proposal validation receipts do not match"
            )
        if dag.repository_tree_id != repository_tree_id:
            raise ValueError("validation DAG and implementation tree do not match")
        if dag.objective_id != proposal_result.proposal.objective_id:
            raise ValueError(
                "validation DAG and proposal objective authorities do not match"
            )
        if tuple(dag.changed_paths) != tuple(scope_set.changed_paths):
            raise ValueError(
                "validation DAG and implementation changed scopes do not match"
            )
        if (
            expected_validation_policy_id
            and dag.policy_id != expected_validation_policy_id
        ):
            raise ValueError(
                "validation DAG policy does not match the expected validation policy"
            )
        if not dag.nodes:
            raise ValueError(
                "empty validation DAG cannot produce implementation proof obligations"
            )
        if dag.uncovered_impact:
            raise ValueError(
                "validation DAG with uncovered impact cannot produce implementation proof obligations"
            )
        coverage_complete = getattr(dag, "coverage_complete", None)
        if coverage_complete is False:
            raise ValueError(
                "incomplete validation DAG cannot produce implementation proof obligations"
            )
        if not dag.passed:
            raise ValueError(
                "failed validation DAG cannot produce implementation proof obligations"
            )
        validation_dag_receipt_id = dag.receipt_id
        validation_policy_id = dag.policy_id

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
        effect_scope_map=effect_scope_map or {},
        plan_requirement_ids=tuple(plan_requirement_ids),
        plan_trace_bound=plan_trace_bound,
        task_id=task_id,
        goal_id=goal_id,
        code_proof_toolchain_id=code_proof_toolchain_id,
        code_proof_policy_id=code_proof_policy_id,
        proposal_validation_receipt_id=proposal_receipt_id,
        proposal_accepted=proposal_accepted,
        validation_dag_receipt_id=validation_dag_receipt_id,
        validation_policy_id=validation_policy_id,
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
    effect_scope_ids = {item.scope_id for item in effects}
    if binding.plan_effect_ids and not effect_scope_ids:
        incomplete.append("planned_effect_scope_omitted")
    if binding.effect_scope_map:
        mapped_effect_ids = set(binding.effect_scope_map)
        planned_effect_id_set = set(binding.plan_effect_ids)
        if mapped_effect_ids != planned_effect_id_set:
            incomplete.append("planned_effect_coverage_mismatch")
        mapped_scope_ids = {
            str(scope_id)
            for scope_ids in binding.effect_scope_map.values()
            for scope_id in scope_ids
        }
        if not mapped_scope_ids.issubset(effect_scope_ids):
            incomplete.append("planned_effect_scope_omitted")
        if not effect_scope_ids.issubset(mapped_scope_ids):
            incomplete.append("changed_effect_scope_unplanned")
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
                "goal_id": binding.goal_id,
                "assumption_ids": binding.assumption_ids,
                "validation_bounds": binding.validation_bounds,
                "evidence_ids": evidence_ids,
                "effect_scope_map": binding.effect_scope_map,
                "code_proof_toolchain_id": binding.code_proof_toolchain_id,
                "code_proof_policy_id": binding.code_proof_policy_id,
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


def transitive_impact_blocks_proof_derivation(
    proposal_validation: Any,
    validation_dag: Any,
) -> bool:
    """Revalidate the G101 adversarial DAG at the code-proof boundary.

    A qualifying transitive-impact witness is deliberately a failed
    validation DAG.  It proves that the defect was found, but must never
    authorize fresh implementation obligations or code-proof work.
    """

    from ..validation.proposal_validation import ProposalValidationResult
    from ..validation.validation_scheduler import (
        REQUIRED_AUTHORITY_GATES,
        TRANSITIVE_IMPACT_OBJECTIVE_ID,
        TRANSITIVE_IMPACT_REQUIREMENT_ID,
        ValidationAuthorityDisposition,
        ValidationDAGReceipt,
    )

    proposal = (
        proposal_validation
        if isinstance(proposal_validation, ProposalValidationResult)
        else ProposalValidationResult.from_dict(proposal_validation)
    )
    dag = (
        validation_dag
        if isinstance(validation_dag, ValidationDAGReceipt)
        else ValidationDAGReceipt.from_dict(validation_dag)
    )
    proposal.require_admitted_binding(
        repository_tree_id=dag.repository_tree_id,
        objective_id=dag.objective_id,
        receipt_id=dag.proposal_receipt_id,
    )
    return bool(
        dag.objective_id == TRANSITIVE_IMPACT_OBJECTIVE_ID
        and not dag.passed
        and dag.coverage_complete
        and not dag.uncovered_impact
        and dag.transitive_evidence is not None
        and dag.transitive_evidence.requirement_id
        == TRANSITIVE_IMPACT_REQUIREMENT_ID
        and {
            gate.gate
            for gate in dag.authority_gates
            if gate.disposition is ValidationAuthorityDisposition.BLOCKED
        }
        == set(REQUIRED_AUTHORITY_GATES)
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

    def __post_init__(self) -> None:
        for name in ("receipt_id", "obligation_id", "binding_id"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ValueError(f"code-proof binding result requires {name}")
            object.__setattr__(self, name, value)
        for name in ("valid", "stale", "contradictory"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        object.__setattr__(
            self, "reason_codes", _canonical_strings(self.reason_codes)
        )
        object.__setattr__(
            self,
            "authoritative_assurance",
            AssuranceLevel(self.authoritative_assurance),
        )
        object.__setattr__(
            self,
            "authoritative_verdict",
            ProofVerdict(self.authoritative_verdict),
        )
        if self.valid and self.reason_codes:
            raise ValueError("valid code-proof binding cannot contain rejection reasons")
        if not self.valid and not self.reason_codes:
            raise ValueError("rejected code-proof binding requires a reason")
        if self.valid and (
            self.authoritative_verdict is not ProofVerdict.PROVED
            or self.authoritative_assurance.rank
            < AssuranceLevel.KERNEL_VERIFIED.rank
        ):
            raise ValueError(
                "valid code-proof binding requires authoritative proved assurance"
            )

    @property
    def result_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    @property
    def proof_authoritative(self) -> bool:
        return self.valid

    @property
    def completion_authoritative(self) -> bool:
        # A valid code-proof binding is an input to the goal-completion policy.
        # It is not, by itself, a complete code-completion decision.
        return False

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
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
            "proof_authoritative": self.proof_authoritative,
            "completion_authoritative": False,
        }
        if include_id:
            payload["result_id"] = self.result_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodeProofReceiptBindingResult":
        schema = str(payload.get("schema") or CODE_PROOF_BINDING_RESULT_SCHEMA)
        if schema != CODE_PROOF_BINDING_RESULT_SCHEMA:
            raise ValueError(f"unsupported code-proof binding-result schema: {schema}")
        result = cls(
            receipt_id=str(payload.get("receipt_id") or ""),
            obligation_id=str(payload.get("obligation_id") or ""),
            binding_id=str(payload.get("binding_id") or ""),
            valid=payload.get("valid", False),
            stale=payload.get("stale", False),
            contradictory=payload.get("contradictory", False),
            reason_codes=_canonical_strings(payload.get("reason_codes") or ()),
            authoritative_assurance=AssuranceLevel(
                payload.get("authoritative_assurance", AssuranceLevel.UNVERIFIED)
            ),
            authoritative_verdict=ProofVerdict(
                payload.get("authoritative_verdict", ProofVerdict.INCONCLUSIVE)
            ),
        )
        if payload.get("proof_authoritative") not in (
            None,
            result.proof_authoritative,
        ):
            raise ValueError("code-proof binding proof authority does not match result")
        if payload.get("completion_authoritative") not in (None, False):
            raise ValueError("code-proof binding result cannot claim completion")
        if payload.get("result_id") and payload["result_id"] != result.result_id:
            raise ValueError("code-proof binding-result identity does not match payload")
        return result


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
    obligation_belongs_to_set = (
        obligation_set is None
        or (
            obligation is not None
            and obligation.obligation_id in obligation_set.obligation_ids
        )
    )

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
    if not obligation_belongs_to_set:
        reject("wrong_theorem_not_in_fresh_obligation_set", is_stale=True)
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
    if (
        expected_binding.code_proof_toolchain_id
        and proof.toolchain_id != expected_binding.code_proof_toolchain_id
    ):
        reject("proof_toolchain_mismatch", is_stale=True)
    if (
        expected_binding.code_proof_policy_id
        and proof.policy_id != expected_binding.code_proof_policy_id
    ):
        reject("proof_policy_mismatch", is_stale=True)
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


@dataclass(frozen=True)
class ProofCandidateNonAuthorityEvidence:
    """Tamper-evident proof that a provider candidate stayed non-authoritative.

    The complete canonical candidate and every authority-bearing input are
    embedded so deserialization can repeat both the code-proof binding check
    and the final completion-admission decision.  Summary fields from a
    provider, scheduler report, or caller are never trusted.
    """

    objective_id: str
    candidate_receipt: ProofReceipt
    obligation_set: ImplementationObligationSet
    proposal_validation: Any
    validation_dag: Any
    required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED
    requirement_id: str = PROOF_CANDIDATE_NON_AUTHORITY_REQUIREMENT_ID
    binding_result: CodeProofReceiptBindingResult | Mapping[str, Any] | None = None
    completion_admission: Any = None
    evidence_id: str = ""

    def __post_init__(self) -> None:
        objective_id = str(self.objective_id or "").strip()
        if not objective_id:
            raise ValueError("proof-candidate evidence requires an objective_id")
        if objective_id != PROOF_CANDIDATE_OBJECTIVE_ID:
            raise ValueError(
                "proof-candidate evidence must bind the ASI-G102 objective"
            )
        object.__setattr__(self, "objective_id", objective_id)
        requirement_id = str(self.requirement_id or "").strip()
        if requirement_id != PROOF_CANDIDATE_NON_AUTHORITY_REQUIREMENT_ID:
            raise ValueError("unexpected proof-candidate non-authority requirement")
        object.__setattr__(self, "requirement_id", requirement_id)
        receipt = (
            self.candidate_receipt
            if isinstance(self.candidate_receipt, ProofReceipt)
            else ProofReceipt.from_dict(self.candidate_receipt)
        )
        object.__setattr__(self, "candidate_receipt", receipt)
        obligations = (
            self.obligation_set
            if isinstance(self.obligation_set, ImplementationObligationSet)
            else ImplementationObligationSet.from_dict(self.obligation_set)
        )
        object.__setattr__(self, "obligation_set", obligations)
        required = AssuranceLevel(self.required_assurance)
        object.__setattr__(self, "required_assurance", required)

        from ..validation.proposal_validation import ProposalValidationResult
        from ..validation.validation_scheduler import ValidationDAGReceipt

        proposal = (
            self.proposal_validation
            if isinstance(self.proposal_validation, ProposalValidationResult)
            else ProposalValidationResult.from_dict(self.proposal_validation)
        )
        dag = (
            self.validation_dag
            if isinstance(self.validation_dag, ValidationDAGReceipt)
            else ValidationDAGReceipt.from_dict(self.validation_dag)
        )
        object.__setattr__(self, "proposal_validation", proposal)
        object.__setattr__(self, "validation_dag", dag)

        binding = obligations.binding
        if not proposal.accepted:
            raise ValueError("candidate-isolation evidence requires an accepted proposal")
        if not dag.passed or not dag.coverage_complete or dag.uncovered_impact:
            raise ValueError("candidate-isolation evidence requires a passing complete DAG")
        if (
            proposal.proposal.objective_id != objective_id
            or dag.objective_id != objective_id
        ):
            raise ValueError("candidate-isolation objective binding mismatch")
        if (
            binding.proposal_validation_receipt_id != proposal.receipt.receipt_id
            or binding.validation_dag_receipt_id != dag.receipt_id
            or binding.validation_policy_id != dag.policy_id
            or binding.repository_tree_id != proposal.proposal.repository_tree_id
            or binding.repository_tree_id != dag.repository_tree_id
            or binding.repository_id != proposal.proposal.repository_id
            or binding.accepted_plan_id != proposal.proposal.accepted_plan_id
        ):
            raise ValueError("candidate-isolation authority chain is inconsistent")

        matching = tuple(
            item
            for item in obligations.obligations
            if item.obligation_id == receipt.obligation_id
        )
        if len(matching) != 1:
            raise ValueError(
                "proof candidate must target exactly one fresh implementation obligation"
            )
        recomputed = validate_code_proof_receipt_bindings(
            receipt,
            obligations,
            obligation=matching[0],
            required_assurance=required,
        )
        supplied_result = self.binding_result
        if supplied_result is not None:
            normalized_result = (
                supplied_result
                if isinstance(supplied_result, CodeProofReceiptBindingResult)
                else CodeProofReceiptBindingResult.from_dict(supplied_result)
            )
            if normalized_result != recomputed:
                raise ValueError("candidate binding result does not match recomputation")
        object.__setattr__(self, "binding_result", recomputed)

        if receipt.authoritative_assurance is not AssuranceLevel.CANDIDATE:
            raise ValueError("receipt is not a proof candidate")
        if receipt.authoritative_verdict is ProofVerdict.PROVED or recomputed.valid:
            raise ValueError("authoritative proof receipts are not candidate evidence")
        if not {
            "code_proof_not_proved",
            "required_code_assurance_not_satisfied",
        }.issubset(recomputed.reason_codes):
            raise ValueError("proof candidate lacks deterministic authority rejection")

        from ..planning.formal_plan_conformance import (
            CompletionAdmissionGate,
            evaluate_completion_admission,
        )

        recomputed_gate = evaluate_completion_admission(
            proposal_validation=proposal,
            validation_dag=dag,
            required=True,
            expected_validation_policy_id=dag.policy_id,
            code_proof_results=(recomputed,),
            require_code_proof=True,
        )
        if self.completion_admission is not None:
            supplied_gate = (
                self.completion_admission
                if isinstance(self.completion_admission, CompletionAdmissionGate)
                else CompletionAdmissionGate.from_dict(self.completion_admission)
            )
            if supplied_gate != recomputed_gate:
                raise ValueError(
                    "candidate completion gate does not match recomputation"
                )
        object.__setattr__(self, "completion_admission", recomputed_gate)
        if recomputed_gate.admitted or not {
            "code_proof_candidate_only",
            "code_proof_not_authoritative",
        }.issubset(recomputed_gate.reason_codes):
            raise ValueError("proof candidate did not close completion authority")

        claimed = str(self.evidence_id or "").strip()
        object.__setattr__(self, "evidence_id", "")
        derived = content_identity(self._identity_payload())
        if claimed and claimed != derived:
            raise ValueError("proof-candidate evidence identity does not match payload")
        object.__setattr__(self, "evidence_id", derived)

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        return (self.requirement_id,)

    @property
    def proof_authoritative(self) -> bool:
        return False

    @property
    def code_proof_authoritative(self) -> bool:
        """A rejection witness cannot itself discharge a code obligation."""

        return False

    @property
    def completion_authoritative(self) -> bool:
        return False

    def evaluate_objective_completion(
        self,
        *,
        current_state: Any = "active",
        evidence: Sequence[Any] = (),
        tasks_complete: bool = False,
        coverage: Any = None,
        analyzer_health: Any = None,
        exhaustion_quorum: Any = None,
        required_exhaustive_receipts: int = 2,
        child_goals: Sequence[Any] = (),
        now: Any = None,
        freshness_seconds: float | None = None,
        clock_skew_seconds: float | None = None,
        analysis_inconclusive: bool = False,
        blocked_reason: str = "",
    ) -> Any:
        """Evaluate ASI-G102 through its closed current-tree proof gate.

        This is intentionally a second phase after the operational
        candidate-isolation witness.  The witness proves that a provider
        candidate did not acquire authority; it cannot mark its own objective
        complete without separately produced, current-tree validation,
        coverage, analyzer-health, and exhaustive-quorum evidence.
        """

        return _evaluate_proof_candidate_objective_completion(
            self,
            current_state=current_state,
            evidence=evidence,
            tasks_complete=tasks_complete,
            coverage=coverage,
            analyzer_health=analyzer_health,
            exhaustion_quorum=exhaustion_quorum,
            required_exhaustive_receipts=required_exhaustive_receipts,
            child_goals=child_goals,
            now=now,
            freshness_seconds=freshness_seconds,
            clock_skew_seconds=clock_skew_seconds,
            analysis_inconclusive=analysis_inconclusive,
            blocked_reason=blocked_reason,
        )

    def strict_validation_completion_evidence(
        self,
    ) -> "StrictValidationProofCompletionEvidence":
        """Project the proof-owned portion of the ASI-G040 parent join."""

        return StrictValidationProofCompletionEvidence(witness=self)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PROOF_CANDIDATE_NON_AUTHORITY_EVIDENCE_SCHEMA,
            "requirement_id": self.requirement_id,
            "objective_id": self.objective_id,
            "candidate_receipt": self.candidate_receipt.to_dict(),
            "obligation_set": self.obligation_set.to_dict(),
            "proposal_validation": self.proposal_validation.to_dict(),
            "validation_dag": self.validation_dag.to_dict(),
            "required_assurance": self.required_assurance.value,
            "binding_result": self.binding_result.to_dict(),
            "completion_admission": self.completion_admission.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "evidence_id": self.evidence_id,
            "candidate_receipt_id": self.candidate_receipt.receipt_id,
            "candidate_authoritative_assurance": (
                self.candidate_receipt.authoritative_assurance.value
            ),
            "candidate_authoritative_verdict": (
                self.candidate_receipt.authoritative_verdict.value
            ),
            "implementation_binding_id": self.obligation_set.binding.binding_id,
            "proposal_receipt_id": self.proposal_validation.receipt.receipt_id,
            "validation_dag_receipt_id": self.validation_dag.receipt_id,
            "proved_requirement_ids": self.proved_requirement_ids,
            "proof_authoritative": False,
            "code_proof_authoritative": False,
            "completion_authoritative": False,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ProofCandidateNonAuthorityEvidence":
        schema = str(
            payload.get("schema")
            or PROOF_CANDIDATE_NON_AUTHORITY_EVIDENCE_SCHEMA
        )
        if schema != PROOF_CANDIDATE_NON_AUTHORITY_EVIDENCE_SCHEMA:
            raise ValueError(f"unsupported proof-candidate evidence schema: {schema}")
        for name in (
            "proof_authoritative",
            "code_proof_authoritative",
            "completion_authoritative",
        ):
            if payload.get(name) not in (None, False):
                raise ValueError(f"proof-candidate evidence cannot claim {name}")
        result = cls(
            objective_id=str(payload.get("objective_id") or ""),
            candidate_receipt=ProofReceipt.from_dict(
                payload.get("candidate_receipt") or {}
            ),
            obligation_set=ImplementationObligationSet.from_dict(
                payload.get("obligation_set") or {}
            ),
            proposal_validation=payload.get("proposal_validation") or {},
            validation_dag=payload.get("validation_dag") or {},
            required_assurance=payload.get(
                "required_assurance", AssuranceLevel.KERNEL_VERIFIED
            ),
            requirement_id=str(payload.get("requirement_id") or ""),
            binding_result=payload.get("binding_result"),
            completion_admission=payload.get("completion_admission"),
            evidence_id=str(payload.get("evidence_id") or ""),
        )
        expected_claims = {
            "candidate_receipt_id": result.candidate_receipt.receipt_id,
            "candidate_authoritative_assurance": (
                result.candidate_receipt.authoritative_assurance.value
            ),
            "candidate_authoritative_verdict": (
                result.candidate_receipt.authoritative_verdict.value
            ),
            "implementation_binding_id": result.obligation_set.binding.binding_id,
            "proposal_receipt_id": result.proposal_validation.receipt.receipt_id,
            "validation_dag_receipt_id": result.validation_dag.receipt_id,
        }
        for name, expected in expected_claims.items():
            if payload.get(name) not in (None, expected):
                raise ValueError(f"proof-candidate evidence {name} is inconsistent")
        claimed_requirements = payload.get("proved_requirement_ids")
        if (
            claimed_requirements is not None
            and tuple(claimed_requirements) != result.proved_requirement_ids
        ):
            raise ValueError(
                "proof-candidate evidence requirement projection is inconsistent"
            )
        return result


@dataclass(frozen=True)
class StrictValidationProofCompletionEvidence:
    """Tamper-evident proof-owned input to the ASI-G040 completion join.

    The embedded G102 witness is reconstructed in full, which replays the
    candidate receipt against its exact fresh implementation obligations and
    re-derives the closed completion-admission gate.  This projection exposes
    that semantic/proof boundary to the parent without acquiring completion
    authority itself.
    """

    witness: ProofCandidateNonAuthorityEvidence
    evidence_id: str = ""

    def __post_init__(self) -> None:
        witness = self.witness
        if not isinstance(witness, ProofCandidateNonAuthorityEvidence):
            if not isinstance(witness, Mapping):
                raise ValueError(
                    "strict validation proof evidence requires a G102 witness"
                )
            witness = ProofCandidateNonAuthorityEvidence.from_dict(witness)
        object.__setattr__(self, "witness", witness)
        if (
            witness.objective_id
            != PROOF_CANDIDATE_NON_AUTHORITY_OBJECTIVE_ID
            or witness.proved_requirement_ids
            != (PROOF_CANDIDATE_NON_AUTHORITY_REQUIREMENT_ID,)
            or witness.binding_result.valid
            or witness.completion_admission.admitted
            or witness.code_proof_authoritative
            or witness.completion_authoritative
        ):
            raise ValueError(
                "G102 witness does not qualify for the strict validation "
                "proof projection"
            )
        claimed = str(self.evidence_id or "").strip()
        object.__setattr__(self, "evidence_id", "")
        derived = content_identity(self._identity_payload())
        if claimed and claimed != derived:
            raise ValueError(
                "strict validation proof evidence identity mismatch"
            )
        object.__setattr__(self, "evidence_id", derived)

    @property
    def objective_id(self) -> str:
        return STRICT_VALIDATION_PARENT_OBJECTIVE_ID

    @property
    def child_objective_id(self) -> str:
        return self.witness.objective_id

    @property
    def repository_id(self) -> str:
        return self.witness.obligation_set.binding.repository_id

    @property
    def repository_tree_id(self) -> str:
        return self.witness.obligation_set.binding.repository_tree_id

    @property
    def validation_policy_id(self) -> str:
        return self.witness.validation_dag.policy_id

    @property
    def operational_receipt_id(self) -> str:
        return self.witness.evidence_id

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        return self.witness.proved_requirement_ids

    @property
    def gate_kinds(self) -> tuple[str, ...]:
        return STRICT_VALIDATION_PROOF_GATE_KINDS

    @property
    def qualifies(self) -> bool:
        return True

    @property
    def completion_authoritative(self) -> bool:
        return False

    @property
    def proof_authoritative(self) -> bool:
        return False

    @property
    def code_proof_authoritative(self) -> bool:
        return False

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": STRICT_VALIDATION_PROOF_COMPLETION_EVIDENCE_SCHEMA,
            "objective_id": self.objective_id,
            "child_objective_id": self.child_objective_id,
            "repository_id": self.repository_id,
            "repository_tree_id": self.repository_tree_id,
            "validation_policy_id": self.validation_policy_id,
            "operational_receipt_id": self.operational_receipt_id,
            "proved_requirement_ids": self.proved_requirement_ids,
            "gate_kinds": self.gate_kinds,
            "qualifies": self.qualifies,
            "proof_authoritative": False,
            "code_proof_authoritative": False,
            "completion_authoritative": False,
            "witness": self.witness.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "evidence_id": self.evidence_id}

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "StrictValidationProofCompletionEvidence":
        required_fields = {
            "schema",
            "objective_id",
            "child_objective_id",
            "repository_id",
            "repository_tree_id",
            "validation_policy_id",
            "operational_receipt_id",
            "proved_requirement_ids",
            "gate_kinds",
            "qualifies",
            "proof_authoritative",
            "code_proof_authoritative",
            "completion_authoritative",
            "witness",
            "evidence_id",
        }
        supplied_fields = {str(name) for name in payload}
        if supplied_fields != required_fields:
            missing = sorted(required_fields - supplied_fields)
            unknown = sorted(supplied_fields - required_fields)
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if unknown:
                details.append("unknown: " + ", ".join(unknown))
            raise ValueError(
                "strict validation proof evidence has an invalid field "
                "population (" + "; ".join(details) + ")"
            )
        if (
            payload.get("schema")
            != STRICT_VALIDATION_PROOF_COMPLETION_EVIDENCE_SCHEMA
        ):
            raise ValueError(
                "unsupported strict validation proof completion schema"
            )
        witness_payload = payload.get("witness")
        if not isinstance(witness_payload, Mapping):
            raise ValueError(
                "strict validation proof evidence is missing its witness"
            )
        result = cls(
            witness=ProofCandidateNonAuthorityEvidence.from_dict(
                witness_payload
            ),
            evidence_id=str(payload.get("evidence_id") or ""),
        )
        expected = result._identity_payload()
        for name, value in expected.items():
            if name == "witness":
                continue
            if canonical_json(payload.get(name)) != canonical_json(value):
                raise ValueError(
                    "strict validation proof projection is inconsistent"
                )
        return result


def _evaluate_proof_candidate_objective_completion(
    witness: ProofCandidateNonAuthorityEvidence,
    *,
    current_state: Any,
    evidence: Sequence[Any],
    tasks_complete: bool,
    coverage: Any,
    analyzer_health: Any,
    exhaustion_quorum: Any,
    required_exhaustive_receipts: int,
    child_goals: Sequence[Any],
    now: Any,
    freshness_seconds: float | None,
    clock_skew_seconds: float | None,
    analysis_inconclusive: bool,
    blocked_reason: str,
) -> Any:
    """Bridge the replayed G102 witness into the goal-completion lifecycle."""

    from ..objectives.goal_completion import evaluate_goal_completion

    binding = witness.obligation_set.binding
    gate = witness.completion_admission
    result = witness.binding_result
    operational_complete = bool(
        witness.objective_id == PROOF_CANDIDATE_NON_AUTHORITY_OBJECTIVE_ID
        and witness.proved_requirement_ids
        == (PROOF_CANDIDATE_NON_AUTHORITY_REQUIREMENT_ID,)
        and witness.proposal_validation.accepted
        and witness.validation_dag.passed
        and witness.validation_dag.coverage_complete
        and not witness.validation_dag.uncovered_impact
        and witness.candidate_receipt.authoritative_assurance
        is AssuranceLevel.CANDIDATE
        and witness.candidate_receipt.authoritative_verdict
        is ProofVerdict.INCONCLUSIVE
        and not result.valid
        and result.binding_id == binding.binding_id
        and result.receipt_id == witness.candidate_receipt.receipt_id
        and result.obligation_id == witness.candidate_receipt.obligation_id
        and {
            "code_proof_not_proved",
            "required_code_assurance_not_satisfied",
        }.issubset(result.reason_codes)
        and not gate.admitted
        and result.result_id in gate.code_proof_result_ids
        and witness.candidate_receipt.receipt_id
        in gate.proof_candidate_receipt_ids
        and {
            "code_proof_candidate_only",
            "code_proof_not_authoritative",
            "code_proof_binding_rejected",
        }.issubset(gate.reason_codes)
    )

    def payload(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        converter = getattr(value, "to_dict", None)
        if callable(converter):
            converted = converter()
            if isinstance(converted, Mapping):
                return dict(converted)
        return {}

    expected_criteria = {
        " ".join(item.lower().split())
        for item in PROOF_CANDIDATE_NON_AUTHORITY_ACCEPTANCE_CRITERIA
    }
    coverage_value = payload(coverage)
    rows_value = coverage_value.get("criteria")
    rows = rows_value if isinstance(rows_value, list) else []
    normalized_rows = [
        " ".join(
            str(
                row.get("criterion", row.get("acceptance_criterion", ""))
                if isinstance(row, Mapping)
                else ""
            )
            .lower()
            .split()
        )
        for row in rows
    ]
    coverage_complete = bool(
        operational_complete
        and len(normalized_rows) == len(expected_criteria)
        and len(normalized_rows) == len(set(normalized_rows))
        and set(normalized_rows) == expected_criteria
        and all(
            isinstance(row, Mapping)
            and bool(str(row.get("implementation") or "").strip())
            and bool(str(row.get("validation") or "").strip())
            for row in rows
        )
    )

    evidence_bound = len(evidence) == len(expected_criteria)
    for item in evidence:
        record = (
            item.to_dict()
            if hasattr(item, "to_dict") and callable(item.to_dict)
            else dict(item)
            if isinstance(item, Mapping)
            else {}
        )
        validation = record.get("validation_receipt")
        validation = validation if isinstance(validation, Mapping) else {}
        evidence_bound = bool(
            evidence_bound
            and validation.get("requirement_id")
            == PROOF_CANDIDATE_NON_AUTHORITY_REQUIREMENT_ID
            and validation.get("objective_id")
            == PROOF_CANDIDATE_NON_AUTHORITY_OBJECTIVE_ID
            and validation.get("repository_id") == binding.repository_id
            and validation.get("tree_id") == binding.repository_tree_id
            and validation.get("operational_receipt_id") == witness.evidence_id
            and validation.get("validation_policy_id")
            == witness.validation_dag.policy_id
        )
    if not coverage_complete or not evidence_bound:
        reasons = coverage_value.get("reason_codes")
        reasons = list(reasons) if isinstance(reasons, (list, tuple)) else []
        if not operational_complete:
            reasons.append("active_operational_evidence_missing")
        if not coverage_complete:
            reasons.append("coverage_missing_implementation_validation_binding")
        if not evidence_bound:
            reasons.append("validation_not_bound_to_operational_witness")
        coverage_value = {
            **coverage_value,
            "verified": False,
            "reason_codes": list(dict.fromkeys(reasons)),
        }

    health_value = payload(analyzer_health)
    if not (
        str(health_value.get("status") or "").lower() == "healthy"
        and health_value.get("healthy") is True
        and health_value.get("safe_for_completion_reasoning") is True
        and health_value.get("analyzer_version")
        == PROOF_CANDIDATE_NON_AUTHORITY_COMPLETION_ANALYZER_VERSION
    ):
        health_value = {
            **health_value,
            "healthy": False,
            "safe_for_completion_reasoning": False,
        }

    expected_binding = {
        "repository_id": binding.repository_id,
        "tree_id": binding.repository_tree_id,
        "objective_id": PROOF_CANDIDATE_NON_AUTHORITY_OBJECTIVE_ID,
        "objective_revision": PROOF_CANDIDATE_NON_AUTHORITY_OBJECTIVE_REVISION,
        "validation_policy_id": witness.validation_dag.policy_id,
        "operational_receipt_id": witness.evidence_id,
        "analyzer_version": (
            PROOF_CANDIDATE_NON_AUTHORITY_COMPLETION_ANALYZER_VERSION
        ),
        "configuration_revision": (
            PROOF_CANDIDATE_NON_AUTHORITY_COMPLETION_CONFIGURATION_REVISION
        ),
    }
    quorum_value = payload(exhaustion_quorum)
    quorum_binding = quorum_value.get("binding")
    quorum_binding = (
        quorum_binding if isinstance(quorum_binding, Mapping) else {}
    )
    members_value = quorum_value.get("members")
    members = members_value if isinstance(members_value, list) else []
    member_receipts = [
        str(member.get("receipt_cid") or "")
        for member in members
        if isinstance(member, Mapping)
    ]
    channels = [
        str(member.get("evidence_channel") or "")
        for member in members
        if isinstance(member, Mapping)
    ]
    quorum_complete = bool(
        not isinstance(required_exhaustive_receipts, bool)
        and isinstance(required_exhaustive_receipts, int)
        and required_exhaustive_receipts >= 2
        and quorum_value.get("required_members") == required_exhaustive_receipts
        and len(members) >= required_exhaustive_receipts
        and all(
            quorum_binding.get(key) == value
            for key, value in expected_binding.items()
        )
        and len(member_receipts) == len(members) == len(set(member_receipts))
        and len(channels) == len(members) == len(set(channels))
        and all(member_receipts)
        and all(channels)
        and all(
            isinstance(member, Mapping)
            and member.get("healthy") is True
            and member.get("safe_for_completion_reasoning") is True
            and str(member.get("scan_mode") or "").lower() == "exhaustive"
            and isinstance(member.get("binding"), Mapping)
            and all(
                member["binding"].get(key) == value
                for key, value in expected_binding.items()
            )
            for member in members
        )
    )
    if not quorum_complete:
        quorum_value = {
            **quorum_value,
            "satisfied": False,
            "quorum_met": False,
        }

    values: dict[str, Any] = {
        "current_state": current_state,
        "acceptance_criteria": (
            PROOF_CANDIDATE_NON_AUTHORITY_ACCEPTANCE_CRITERIA
        ),
        "evidence": evidence,
        "tasks_complete": tasks_complete,
        "repository_tree": binding.repository_tree_id,
        "now": now,
        "analysis_inconclusive": analysis_inconclusive,
        "blocked_reason": blocked_reason,
        "coverage": coverage_value,
        "analyzer_health": health_value,
        "exhaustion_quorum": quorum_value,
        "child_goals": child_goals,
        "analysis_result": None,
        "require_completion_gate": True,
    }
    if freshness_seconds is not None:
        values["freshness_seconds"] = freshness_seconds
    if clock_skew_seconds is not None:
        values["clock_skew_seconds"] = clock_skew_seconds
    return evaluate_goal_completion(**values)


def prove_proof_candidate_non_authority(
    candidate_receipt: ProofReceipt | Mapping[str, Any],
    obligation_set: ImplementationObligationSet | Mapping[str, Any],
    *,
    objective_id: str,
    proposal_validation: Any,
    validation_dag: Any,
    required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED,
) -> ProofCandidateNonAuthorityEvidence:
    """Produce the ASI-G102 witness only after revalidating the full chain."""

    return ProofCandidateNonAuthorityEvidence(
        objective_id=objective_id,
        candidate_receipt=(
            candidate_receipt
            if isinstance(candidate_receipt, ProofReceipt)
            else ProofReceipt.from_dict(candidate_receipt)
        ),
        obligation_set=(
            obligation_set
            if isinstance(obligation_set, ImplementationObligationSet)
            else ImplementationObligationSet.from_dict(obligation_set)
        ),
        proposal_validation=proposal_validation,
        validation_dag=validation_dag,
        required_assurance=required_assurance,
    )


# Concise compatibility spellings for integration callers.
derive_implementation_obligations = derive_fresh_implementation_obligations
compile_implementation_obligations = derive_fresh_implementation_obligations
validate_code_proof_receipt_binding = validate_code_proof_receipt_bindings
ImplementationBinding = ImplementationResultBinding
ImplementationEvidence = ImplementationResultEvidence
FreshImplementationObligations = ImplementationObligationSet


# ---------------------------------------------------------------------------
# CBP-030: obligation compiler with cache-key binding
# ---------------------------------------------------------------------------

CODE_PROOF_OBLIGATION_COMPILATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/code-proof-obligation-compilation@1"
)
COMPILED_CODE_PROOF_ITEM_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/compiled-code-proof-item@1"
)
CODE_PROOF_COMPILE_REQUEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/code-proof-compile-request@1"
)
OBLIGATION_COMPILER_PRODUCER_ID = "producer:code-proof-obligation-compiler@1"
MAX_PREMISE_HANDLE_LENGTH = 256
_PREMISE_HANDLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._/@+=-]{0,255}$")
_FORBIDDEN_PREMISE_MARKERS = (
    "\n",
    "\r",
    "\0",
    "def ",
    "class ",
    "import ",
    "from ",
    "-----begin",
    "gold_ir",
    "gold_body",
    "repository_dump",
    "source_dump",
    "full_source",
    " monorepo",
)
_FORBIDDEN_RESIDUAL_BODY_KEYS = frozenset(
    {
        "gold",
        "gold_ir",
        "gold_body",
        "source",
        "source_text",
        "source_dump",
        "repository_dump",
        "ir_body",
        "secret",
        "private_witness",
        "raw_source",
    }
)
_REPO_WIDE_PREMISE_VALUES = frozenset(
    {
        "*",
        "**",
        ".",
        "./",
        "/",
        "/*",
        "/**",
        "repo",
        "repository",
        "repo:*",
        "repository:*",
        "tree:*",
        "all",
        "entire_repository",
        "repository_wide",
        "full_tree",
    }
)

# Scope kinds that may feed family-specific obligations.
_FAMILY_SCOPE_KINDS: Mapping[str, frozenset[ProofScopeKind]] = {
    "dependency_reachability": frozenset(
        {ProofScopeKind.IMPORT, ProofScopeKind.QUALIFIED_SYMBOL, ProofScopeKind.CALL}
    ),
    "api_contract": frozenset(
        {ProofScopeKind.INTERFACE, ProofScopeKind.QUALIFIED_SYMBOL}
    ),
    "security_property": frozenset(
        {
            ProofScopeKind.QUALIFIED_SYMBOL,
            ProofScopeKind.CALL,
            ProofScopeKind.STATE_TRANSITION,
        }
    ),
    "semantic_equivalence": frozenset(
        {
            ProofScopeKind.QUALIFIED_SYMBOL,
            ProofScopeKind.INTERFACE,
            ProofScopeKind.CALL,
            ProofScopeKind.STATE_TRANSITION,
        }
    ),
    "behavioral_invariant": frozenset(
        {ProofScopeKind.STATE_TRANSITION, ProofScopeKind.QUALIFIED_SYMBOL}
    ),
    "supervisor_lifecycle": frozenset(
        {ProofScopeKind.QUALIFIED_SYMBOL, ProofScopeKind.CALL}
    ),
    "srt_structural": frozenset(
        {ProofScopeKind.QUALIFIED_SYMBOL, ProofScopeKind.INTERFACE}
    ),
    "unsupported": frozenset(
        {
            ProofScopeKind.QUALIFIED_SYMBOL,
            ProofScopeKind.INTERFACE,
            ProofScopeKind.IMPORT,
            ProofScopeKind.CALL,
            ProofScopeKind.STATE_TRANSITION,
        }
    ),
}

# Reviewed template fallbacks when a claim family has no catalog property.
_FAMILY_FALLBACK_TEMPLATE: Mapping[str, str] = {
    "dependency_reachability": "dag-acyclicity",
    "api_contract": "legal-state-transitions",
    "security_property": "lease-uniqueness-and-fencing",
    "semantic_equivalence": "projection-equivalence",
    "behavioral_invariant": "legal-state-transitions",
    "supervisor_lifecycle": "merge-idempotence",
    "srt_structural": "projection-equivalence",
    "unsupported": "unsupported-proof-fail-closed",
}


class PremiseValidationError(ValueError):
    """Raised when a proposed premise is not a typed, content-addressed handle."""


class ObligationCompileStatus(str, Enum):
    """Compiler disposition for one property request.

    ``unsupported`` and ``not_measured`` must never collapse into each other
    or into refutation: the former is a reviewed refusal; the latter means the
    measurement path was not exercised for a still-supported shape.
    """

    OPEN = "open"
    UNSUPPORTED = "unsupported"
    NOT_MEASURED = "not_measured"


def premise_set_digest(premise_ids: Sequence[str] | None = ()) -> str:
    """Content-addressed digest of a closed premise-id set."""

    values = _canonical_strings(premise_ids)
    return content_identity({"schema": "premise-set@1", "premise_ids": list(values)})


def assumption_set_digest(assumption_ids: Sequence[str] | None = ()) -> str:
    """Content-addressed digest of a closed assumption-id set."""

    values = _canonical_strings(assumption_ids)
    return content_identity(
        {"schema": "assumption-set@1", "assumption_ids": list(values)}
    )


def _looks_like_repository_wide_premise(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _REPO_WIDE_PREMISE_VALUES:
        return True
    if lowered.startswith("repository-wide") or lowered.startswith("repo-wide"):
        return True
    if lowered.endswith("/**") or lowered.endswith("/*"):
        return True
    if "repository_dump" in lowered or "source_dump" in lowered:
        return True
    return False


def _looks_like_opaque_source_dump(value: str) -> bool:
    if len(value) > MAX_PREMISE_HANDLE_LENGTH:
        return True
    lowered = value.lower()
    if any(marker in lowered for marker in _FORBIDDEN_PREMISE_MARKERS):
        return True
    # Multi-line or whitespace-heavy blobs are dumps, not handles.
    if any(ch.isspace() and ch not in (" ", "\t") for ch in value):
        return True
    if value.count(" ") > 4:
        return True
    return False


def normalize_premise_ids(
    values: Any,
    *,
    field_name: str = "premise_ids",
) -> tuple[str, ...]:
    """Normalize premise handles and reject repository-wide / source dumps.

    Premises must be short content-addressed ids (for example
    ``premise:sha256:…``).  Full repository source dumps, gold IR bodies, and
    secrets never become premises.
    """

    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, str):
        raw = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        raw = values
    else:
        raise PremiseValidationError(f"{field_name} must be a sequence of handles")

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, Mapping):
            # Allow structured premise refs only when they carry an id handle.
            handle = str(
                item.get("premise_id")
                or item.get("id")
                or item.get("handle")
                or ""
            ).strip()
            forbidden = _FORBIDDEN_RESIDUAL_BODY_KEYS.intersection(
                {str(key).strip().lower() for key in item}
            )
            if forbidden:
                raise PremiseValidationError(
                    f"{field_name} rejects opaque body fields: "
                    + ", ".join(sorted(forbidden))
                )
            if not handle:
                raise PremiseValidationError(
                    f"{field_name} mapping requires premise_id/id handle"
                )
            text = handle
        else:
            text = str(item or "").strip()
        if not text:
            continue
        if _looks_like_repository_wide_premise(text):
            raise PremiseValidationError(
                f"{field_name} rejects repository-wide source dumps: {text!r}"
            )
        if _looks_like_opaque_source_dump(text):
            raise PremiseValidationError(
                f"{field_name} rejects opaque source/gold dumps as premises"
            )
        if not _PREMISE_HANDLE_RE.match(text):
            raise PremiseValidationError(
                f"{field_name} entries must be content-addressed handles "
                f"(got {text!r})"
            )
        if text not in seen:
            seen.add(text)
            normalized.append(text)
    return tuple(sorted(normalized))


def normalize_assumption_ids(
    values: Any,
    *,
    field_name: str = "assumption_ids",
) -> tuple[str, ...]:
    """Normalize assumption handles with the same dump-rejection policy."""

    return normalize_premise_ids(values, field_name=field_name)


def normalize_residual_refs(
    values: Any,
    *,
    field_name: str = "residual_refs",
) -> tuple[str, ...]:
    """Extract content-addressed residual-ref handles; never gold IR bodies."""

    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, str):
        raw = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        raw = values
    else:
        raise PremiseValidationError(f"{field_name} must be a sequence of residual refs")

    handles: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, Mapping):
            keys = {str(key).strip().lower() for key in item}
            forbidden = _FORBIDDEN_RESIDUAL_BODY_KEYS.intersection(keys)
            if forbidden:
                raise PremiseValidationError(
                    f"{field_name} must not embed gold/source bodies; "
                    f"forbidden keys: {', '.join(sorted(forbidden))}"
                )
            handle = str(
                item.get("residual_ref_id")
                or item.get("residual_id")
                or item.get("id")
                or item.get("handle")
                or ""
            ).strip()
            if not handle:
                raise PremiseValidationError(
                    f"{field_name} mapping requires residual_ref_id handle"
                )
        else:
            handle = str(item or "").strip()
        if not handle:
            continue
        if _looks_like_opaque_source_dump(handle) or _looks_like_repository_wide_premise(
            handle
        ):
            raise PremiseValidationError(
                f"{field_name} rejects opaque residual bodies; use handles only"
            )
        if not _PREMISE_HANDLE_RE.match(handle):
            raise PremiseValidationError(
                f"{field_name} entries must be content-addressed handles"
            )
        if handle not in seen:
            seen.add(handle)
            handles.append(handle)
    return tuple(sorted(handles))


def _normalize_plan_effect_ids(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
        raise ValueError("formal_plan_effects must be a sequence")
    ids: list[str] = []
    for item in values:
        if isinstance(item, Mapping):
            effect_id = str(
                item.get("effect_id") or item.get("id") or item.get("handle") or ""
            ).strip()
        else:
            effect_id = str(
                getattr(item, "effect_id", "") or item or ""
            ).strip()
        if effect_id:
            ids.append(effect_id)
    return _canonical_strings(ids)


def _claim_family_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(getattr(value, "value", value) or "").strip().lower().replace("-", "_")
    return text


def _scopes_for_family(
    scope_set: CodeProofScopeSet,
    claim_family: str,
    *,
    requested_scope_ids: Sequence[str] = (),
    plan_effect_scope_ids: Sequence[str] = (),
) -> tuple[CodeProofScope, ...]:
    """Select non-conservative AST scopes relevant to a claim family."""

    if requested_scope_ids:
        return _selected_obligation_scopes(scope_set, requested_scope_ids)

    allowed_kinds = _FAMILY_SCOPE_KINDS.get(
        claim_family,
        frozenset(
            {
                ProofScopeKind.QUALIFIED_SYMBOL,
                ProofScopeKind.INTERFACE,
                ProofScopeKind.IMPORT,
                ProofScopeKind.CALL,
                ProofScopeKind.STATE_TRANSITION,
            }
        ),
    )
    plan_ids = set(plan_effect_scope_ids)
    selected = []
    for scope in scope_set.scopes:
        if scope.kind in (ProofScopeKind.CHANGED_PATH, ProofScopeKind.CONSERVATIVE_FILE):
            continue
        if scope.conservative:
            continue
        if plan_ids and scope.scope_id in plan_ids:
            selected.append(scope)
            continue
        if scope.kind in allowed_kinds:
            selected.append(scope)
    # Prefer exact family match; fall back to any non-conservative AST scope.
    if not selected:
        selected = [
            scope
            for scope in scope_set.scopes
            if scope.kind
            not in (ProofScopeKind.CHANGED_PATH, ProofScopeKind.CONSERVATIVE_FILE)
            and not scope.conservative
        ]
    if plan_ids:
        # Always include plan-mapped scopes when present and non-conservative.
        by_id = {scope.scope_id: scope for scope in scope_set.scopes}
        for scope_id in sorted(plan_ids):
            scope = by_id.get(scope_id)
            if (
                scope is not None
                and not scope.conservative
                and scope.kind
                not in (ProofScopeKind.CHANGED_PATH, ProofScopeKind.CONSERVATIVE_FILE)
                and scope not in selected
            ):
                selected.append(scope)
    return tuple(sorted(selected, key=lambda item: item.scope_id))


@dataclass(frozen=True)
class CodeProofCompileRequest:
    """One property/family request for the obligation compiler."""

    property_id: str = ""
    claim_family: str = ""
    template_id: str = ""
    template_version: str = ""
    code_shape: str = ""
    ast_scope_ids: tuple[str, ...] = ()
    premise_ids: tuple[str, ...] = ()
    residual_ref_ids: tuple[str, ...] = ()
    required_assurance: AssuranceLevel | str = AssuranceLevel.KERNEL_VERIFIED
    force_not_measured: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "property_id",
            "claim_family",
            "template_id",
            "template_version",
            "code_shape",
        ):
            object.__setattr__(
                self, name, str(getattr(self, name) or "").strip()
            )
        if self.claim_family:
            object.__setattr__(
                self,
                "claim_family",
                self.claim_family.lower().replace("-", "_"),
            )
        object.__setattr__(
            self,
            "ast_scope_ids",
            _canonical_strings(self.ast_scope_ids),
        )
        object.__setattr__(
            self,
            "premise_ids",
            normalize_premise_ids(self.premise_ids),
        )
        object.__setattr__(
            self,
            "residual_ref_ids",
            normalize_residual_refs(self.residual_ref_ids),
        )
        assurance = self.required_assurance
        if not isinstance(assurance, AssuranceLevel):
            assurance = AssuranceLevel(str(assurance))
        object.__setattr__(self, "required_assurance", assurance)
        object.__setattr__(self, "force_not_measured", bool(self.force_not_measured))
        object.__setattr__(self, "metadata", _canonical_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CODE_PROOF_COMPILE_REQUEST_SCHEMA,
            "property_id": self.property_id,
            "claim_family": self.claim_family,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "code_shape": self.code_shape,
            "ast_scope_ids": list(self.ast_scope_ids),
            "premise_ids": list(self.premise_ids),
            "residual_ref_ids": list(self.residual_ref_ids),
            "required_assurance": self.required_assurance.value,
            "force_not_measured": self.force_not_measured,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodeProofCompileRequest":
        if not isinstance(payload, Mapping):
            raise TypeError("compile request must be a mapping")
        schema = payload.get("schema")
        if schema not in (None, "", CODE_PROOF_COMPILE_REQUEST_SCHEMA):
            raise ValueError(f"unsupported code-proof compile request schema: {schema}")
        return cls(
            property_id=str(payload.get("property_id") or ""),
            claim_family=str(payload.get("claim_family") or ""),
            template_id=str(payload.get("template_id") or ""),
            template_version=str(payload.get("template_version") or ""),
            code_shape=str(payload.get("code_shape") or ""),
            ast_scope_ids=tuple(payload.get("ast_scope_ids") or ()),
            premise_ids=tuple(payload.get("premise_ids") or ()),
            residual_ref_ids=tuple(
                payload.get("residual_ref_ids") or payload.get("residual_refs") or ()
            ),
            required_assurance=payload.get(
                "required_assurance", AssuranceLevel.KERNEL_VERIFIED
            ),
            force_not_measured=bool(payload.get("force_not_measured", False)),
            metadata=payload.get("metadata") or {},
        )


@dataclass(frozen=True)
class CompiledCodeProofItem:
    """One compiled obligation/claim pair with cache-key identity."""

    status: ObligationCompileStatus
    property_id: str
    claim_family: str
    obligation: CodeProofObligation | None
    claim: Any
    cache_key_id: str
    premise_ids: tuple[str, ...] = ()
    assumption_ids: tuple[str, ...] = ()
    residual_ref_ids: tuple[str, ...] = ()
    invalidation_selectors: tuple[Mapping[str, Any], ...] = ()
    reason_codes: tuple[str, ...] = ()
    template_id: str = ""
    catalog_version: str = ""
    required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = self.status
        if not isinstance(status, ObligationCompileStatus):
            status = ObligationCompileStatus(str(status))
        object.__setattr__(self, "status", status)
        for name in (
            "property_id",
            "claim_family",
            "cache_key_id",
            "template_id",
            "catalog_version",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
        object.__setattr__(self, "premise_ids", _canonical_strings(self.premise_ids))
        object.__setattr__(
            self, "assumption_ids", _canonical_strings(self.assumption_ids)
        )
        object.__setattr__(
            self, "residual_ref_ids", _canonical_strings(self.residual_ref_ids)
        )
        object.__setattr__(
            self, "reason_codes", _canonical_strings(self.reason_codes)
        )
        selectors = tuple(
            dict(item) if isinstance(item, Mapping) else dict(item)
            for item in (self.invalidation_selectors or ())
        )
        object.__setattr__(self, "invalidation_selectors", selectors)
        assurance = self.required_assurance
        if not isinstance(assurance, AssuranceLevel):
            assurance = AssuranceLevel(str(assurance))
        object.__setattr__(self, "required_assurance", assurance)
        object.__setattr__(self, "metadata", _canonical_mapping(self.metadata))
        if self.obligation is not None and not isinstance(
            self.obligation, CodeProofObligation
        ):
            raise TypeError("obligation must be a CodeProofObligation or None")

    @property
    def item_id(self) -> str:
        return content_identity(self.to_dict())

    @property
    def obligation_id(self) -> str:
        if self.obligation is None:
            return ""
        return self.obligation.obligation_id

    @property
    def claim_id(self) -> str:
        claim = self.claim
        if claim is None:
            return ""
        if hasattr(claim, "claim_id"):
            return str(claim.claim_id)
        if isinstance(claim, Mapping):
            return str(claim.get("claim_id") or claim.get("content_id") or "")
        return ""

    def to_dict(self) -> dict[str, Any]:
        claim_payload: Any
        if self.claim is None:
            claim_payload = None
        elif hasattr(self.claim, "to_record"):
            claim_payload = self.claim.to_record()
        elif hasattr(self.claim, "to_dict"):
            claim_payload = self.claim.to_dict()
        elif isinstance(self.claim, Mapping):
            claim_payload = dict(self.claim)
        else:
            claim_payload = {"repr": repr(self.claim)}
        return {
            "schema": COMPILED_CODE_PROOF_ITEM_SCHEMA,
            "status": self.status.value,
            "property_id": self.property_id,
            "claim_family": self.claim_family,
            "obligation": (
                None if self.obligation is None else self.obligation.to_dict()
            ),
            "obligation_id": self.obligation_id,
            "claim": claim_payload,
            "claim_id": self.claim_id,
            "cache_key_id": self.cache_key_id,
            "premise_ids": list(self.premise_ids),
            "assumption_ids": list(self.assumption_ids),
            "residual_ref_ids": list(self.residual_ref_ids),
            "invalidation_selectors": [dict(item) for item in self.invalidation_selectors],
            "reason_codes": list(self.reason_codes),
            "template_id": self.template_id,
            "catalog_version": self.catalog_version,
            "required_assurance": self.required_assurance.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CodeProofObligationCompilation:
    """Result of compiling tree + AST scope (+ plan effects / residual refs)."""

    repository_id: str
    repository_tree_id: str
    catalog_version: str
    catalog_id: str
    scope_set_id: str
    items: tuple[CompiledCodeProofItem, ...]
    premise_digest: str
    assumption_digest: str
    premise_ids: tuple[str, ...] = ()
    assumption_ids: tuple[str, ...] = ()
    plan_effect_ids: tuple[str, ...] = ()
    residual_ref_ids: tuple[str, ...] = ()
    toolchain_id: str = ""
    policy_id: str = ""
    task_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "repository_tree_id",
            "catalog_version",
            "catalog_id",
            "scope_set_id",
            "premise_digest",
            "assumption_digest",
            "toolchain_id",
            "policy_id",
            "task_id",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
        if not self.repository_tree_id:
            raise ValueError("repository_tree_id is required")
        if not isinstance(self.items, tuple):
            object.__setattr__(self, "items", tuple(self.items))
        for item in self.items:
            if not isinstance(item, CompiledCodeProofItem):
                raise TypeError("items must be CompiledCodeProofItem instances")
        object.__setattr__(self, "premise_ids", _canonical_strings(self.premise_ids))
        object.__setattr__(
            self, "assumption_ids", _canonical_strings(self.assumption_ids)
        )
        object.__setattr__(
            self, "plan_effect_ids", _canonical_strings(self.plan_effect_ids)
        )
        object.__setattr__(
            self, "residual_ref_ids", _canonical_strings(self.residual_ref_ids)
        )
        object.__setattr__(self, "metadata", _canonical_mapping(self.metadata))

    @property
    def compilation_id(self) -> str:
        return content_identity(self.to_dict())

    def by_status(
        self, status: ObligationCompileStatus | str
    ) -> tuple[CompiledCodeProofItem, ...]:
        target = (
            status
            if isinstance(status, ObligationCompileStatus)
            else ObligationCompileStatus(str(status))
        )
        return tuple(item for item in self.items if item.status is target)

    def by_family(self, claim_family: str) -> tuple[CompiledCodeProofItem, ...]:
        family = _claim_family_value(claim_family)
        return tuple(
            item for item in self.items if _claim_family_value(item.claim_family) == family
        )

    def open_obligations(self) -> tuple[CodeProofObligation, ...]:
        return tuple(
            item.obligation
            for item in self.items
            if item.status is ObligationCompileStatus.OPEN and item.obligation is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CODE_PROOF_OBLIGATION_COMPILATION_SCHEMA,
            "repository_id": self.repository_id,
            "repository_tree_id": self.repository_tree_id,
            "catalog_version": self.catalog_version,
            "catalog_id": self.catalog_id,
            "scope_set_id": self.scope_set_id,
            "items": [item.to_dict() for item in self.items],
            "premise_digest": self.premise_digest,
            "assumption_digest": self.assumption_digest,
            "premise_ids": list(self.premise_ids),
            "assumption_ids": list(self.assumption_ids),
            "plan_effect_ids": list(self.plan_effect_ids),
            "residual_ref_ids": list(self.residual_ref_ids),
            "toolchain_id": self.toolchain_id,
            "policy_id": self.policy_id,
            "task_id": self.task_id,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


def _resolve_property_and_family(
    request: CodeProofCompileRequest,
    catalog: Any,
):
    """Return (property_or_None, property_id, claim_family, template_id, code_shape, assurance)."""

    from .code_claim_contracts import resolve_claim_family
    from .code_property_catalog import CodeProperty

    prop = None
    property_id = request.property_id
    if property_id and catalog is not None:
        prop = catalog.get(property_id)
    template_id = request.template_id
    code_shape = request.code_shape
    assurance = request.required_assurance
    claim_family = request.claim_family

    if prop is not None:
        property_id = prop.property_id
        template_id = template_id or prop.template_id
        code_shape = code_shape or prop.code_shape
        if assurance is None:
            assurance = prop.required_assurance
        elif isinstance(assurance, AssuranceLevel) and assurance is AssuranceLevel.KERNEL_VERIFIED:
            # Prefer the catalog assurance when the request left the default and
            # the property declares a different bar.
            assurance = prop.required_assurance or assurance
        if not claim_family:
            claim_family = resolve_claim_family(
                template_id=prop.template_id,
                invariant_class=prop.invariant_class,
                property_id=prop.property_id,
                code_shape=prop.code_shape,
            ).value
    elif claim_family:
        # Prefer a catalog property whose family matches.
        if catalog is not None:
            for candidate in getattr(catalog, "properties", ()):
                if not isinstance(candidate, CodeProperty):
                    continue
                family = resolve_claim_family(
                    template_id=candidate.template_id,
                    invariant_class=candidate.invariant_class,
                    property_id=candidate.property_id,
                    code_shape=candidate.code_shape,
                ).value
                if family == claim_family:
                    prop = candidate
                    property_id = candidate.property_id
                    template_id = template_id or candidate.template_id
                    code_shape = code_shape or candidate.code_shape
                    assurance = candidate.required_assurance
                    break
        if not template_id:
            template_id = _FAMILY_FALLBACK_TEMPLATE.get(claim_family, "")
        if not property_id and claim_family:
            property_id = (
                f"property:{template_id}"
                if template_id
                else f"property:family:{claim_family}"
            )
    elif template_id:
        claim_family = resolve_claim_family(template_id=template_id).value
        if not property_id:
            property_id = f"property:{template_id}"
            if catalog is not None:
                matched = catalog.get(property_id)
                if matched is not None:
                    prop = matched
                    code_shape = code_shape or matched.code_shape
                    assurance = matched.required_assurance

    if not claim_family and template_id:
        claim_family = resolve_claim_family(template_id=template_id).value
    if not claim_family and property_id:
        claim_family = resolve_claim_family(property_id=property_id).value
    if not claim_family:
        claim_family = "unsupported"

    return prop, property_id, claim_family, template_id, code_shape, assurance


def _default_resource_budget() -> dict[str, Any]:
    return {
        "wall_time_ms": 30_000,
        "cpu_time_ms": 20_000,
        "memory_bytes": 512 * 1024 * 1024,
        "max_processes": 4,
        "max_premises": 32,
        "network_allowed": False,
    }


def compiled_obligation_cache_identity(
    *,
    property_id: str,
    catalog_version: str,
    catalog_id: str = "",
    repository_tree_id: str,
    ast_scope_ids: Sequence[str],
    premise_ids: Sequence[str],
    assumption_ids: Sequence[str],
    residual_ref_ids: Sequence[str] = (),
    toolchain_id: str,
    policy_id: str,
    required_assurance: AssuranceLevel | str,
    template_id: str = "",
    template_version: str = "",
    template_semantic_hash: str = "",
    obligation_id: str = "",
) -> str:
    """Deterministic cache-key identity for a compiled obligation binding.

    Binds property/catalog version, tree/scope, premise/assumption digests,
    toolchain, policy, and required assurance — the G015/G050 identity surface.
    """

    assurance = (
        required_assurance
        if isinstance(required_assurance, AssuranceLevel)
        else AssuranceLevel(str(required_assurance))
    )
    premises = _canonical_strings(premise_ids)
    assumptions = _canonical_strings(assumption_ids)
    residuals = _canonical_strings(residual_ref_ids)
    scopes = _canonical_strings(ast_scope_ids)
    return content_identity(
        {
            "schema": CODE_OBLIGATION_CACHE_KEY_SCHEMA,
            "property_id": str(property_id or "").strip(),
            "catalog_version": str(catalog_version or "").strip(),
            "catalog_id": str(catalog_id or "").strip(),
            "repository_tree_id": str(repository_tree_id or "").strip(),
            "ast_scope_ids": list(scopes),
            "premise_digest": premise_set_digest(premises),
            "assumption_digest": assumption_set_digest(assumptions),
            "residual_ref_ids": list(residuals),
            "toolchain_id": str(toolchain_id or "").strip(),
            "policy_id": str(policy_id or "").strip(),
            "required_assurance": assurance.value,
            "template_id": str(template_id or "").strip(),
            "template_version": str(template_version or "").strip(),
            "template_semantic_hash": str(template_semantic_hash or "").strip(),
            "obligation_id": str(obligation_id or "").strip(),
        }
    )


def _build_not_measured_or_unsupported_claim(
    *,
    status: ObligationCompileStatus,
    property_id: str,
    claim_family: str,
    repository_id: str,
    repository_tree_id: str,
    scope_ids: Sequence[str],
    premise_ids: Sequence[str],
    assumption_ids: Sequence[str],
    residual_ref_ids: Sequence[str],
    toolchain_id: str,
    policy_id: str,
    catalog_version: str,
    template_id: str,
    required_assurance: AssuranceLevel,
    statement: str,
    metadata: Mapping[str, Any],
):
    from .code_claim_contracts import (
        ClaimFamily,
        ClaimStatus,
        CodeClaimRecord,
        build_invalidation_selectors,
    )

    family = ClaimFamily(claim_family) if claim_family else ClaimFamily.UNSUPPORTED
    claim_status = (
        ClaimStatus.UNSUPPORTED
        if status is ObligationCompileStatus.UNSUPPORTED
        else ClaimStatus.NOT_MEASURED
    )
    # Unsupported family with unsupported status is valid without obligation_id
    # only when family is UNSUPPORTED; otherwise bind a property handle.
    selectors = build_invalidation_selectors(
        repository_tree_id=repository_tree_id,
        scope_ids=scope_ids,
        premise_ids=premise_ids,
        assumption_ids=assumption_ids,
        toolchain_id=toolchain_id,
        policy_id=policy_id,
        catalog_version=catalog_version,
        property_id=property_id,
        producer_id=OBLIGATION_COMPILER_PRODUCER_ID,
        required_assurance=required_assurance,
    )
    meta = dict(metadata)
    if residual_ref_ids:
        meta["residual_ref_ids"] = list(residual_ref_ids)
    meta["compile_status"] = status.value
    return CodeClaimRecord(
        claim_family=family,
        status=claim_status,
        property_id=property_id or f"property:family:{family.value}",
        obligation_id="",
        repository_id=repository_id,
        repository_tree_id=repository_tree_id,
        scope_ids=tuple(scope_ids),
        premise_ids=tuple(premise_ids),
        assumption_ids=tuple(assumption_ids),
        producer_id=OBLIGATION_COMPILER_PRODUCER_ID,
        toolchain_id=toolchain_id,
        policy_id=policy_id,
        catalog_version=catalog_version,
        required_assurance=required_assurance,
        derived_assurance=AssuranceLevel.UNVERIFIED,
        invalidation_selectors=selectors,
        statement=statement,
        template_id=template_id,
        metadata=meta,
    )


def compile_code_proof_obligations(
    scope_set: CodeProofScopeSet | None = None,
    *,
    repository_tree_id: str,
    repository_id: str = "",
    candidate_diff: Any = None,
    property_ids: Sequence[str] = (),
    claim_families: Sequence[str] = (),
    requests: Sequence[CodeProofCompileRequest | Mapping[str, Any]] = (),
    catalog: Any = None,
    formal_plan_effects: Sequence[Any] = (),
    effect_scope_map: Mapping[str, Sequence[str]] | None = None,
    residual_refs: Sequence[Any] = (),
    premise_ids: Sequence[str] = (),
    assumption_ids: Sequence[str] = (),
    toolchain_id: str = "",
    policy_id: str = "",
    translator_id: str = "translator:default",
    solver_id: str = "solver:default",
    kernel_id: str = "kernel:default",
    theorem_registry_id: str = "registry:default",
    resource_budget: Any = None,
    task_id: str = "",
    required_assurance: AssuranceLevel | str | None = None,
    registry: ProofObligationTemplateRegistry = DEFAULT_TEMPLATE_REGISTRY,
    metadata: Mapping[str, Any] | None = None,
) -> CodeProofObligationCompilation:
    """Compile tree + changed AST scope (+ plan effects / residual refs).

    Emits typed claim records with explicit premise/assumption ids and
    invalidation selectors.  Unsupported and not-measured dispositions remain
    distinct.  Cache-key identity binds property/catalog version, tree/scope,
    premise/assumption digests, toolchain, policy, and required assurance.
    Repository-wide source dumps are rejected as premises.
    """

    from .code_claim_contracts import (
        ClaimFamily,
        ClaimStatus,
        claim_from_obligation,
        resolve_claim_family,
    )
    from .code_property_catalog import (
        DEFAULT_CODE_PROPERTY_CATALOG,
        CodePropertyCatalog,
    )

    tree_id = str(repository_tree_id or "").strip()
    if not tree_id:
        raise ValueError("repository_tree_id is required")

    if scope_set is None:
        if candidate_diff is None:
            raise ValueError("scope_set or candidate_diff is required")
        scope_set = compile_candidate_proof_scopes(candidate_diff)
    if not isinstance(scope_set, CodeProofScopeSet):
        raise TypeError("scope_set must be a CodeProofScopeSet")

    if catalog is None:
        catalog = DEFAULT_CODE_PROPERTY_CATALOG
    if not isinstance(catalog, CodePropertyCatalog):
        raise TypeError("catalog must be a CodePropertyCatalog")

    global_premises = normalize_premise_ids(premise_ids)
    global_assumptions = normalize_assumption_ids(assumption_ids)
    residual_handles = normalize_residual_refs(residual_refs)
    plan_effect_ids = _normalize_plan_effect_ids(formal_plan_effects)

    plan_effect_scope_ids: list[str] = []
    if effect_scope_map:
        if not isinstance(effect_scope_map, Mapping):
            raise ValueError("effect_scope_map must be a mapping")
        for effect_id, scope_ids in effect_scope_map.items():
            if plan_effect_ids and str(effect_id).strip() not in set(plan_effect_ids):
                continue
            plan_effect_scope_ids.extend(
                str(scope_id).strip()
                for scope_id in (scope_ids or ())
                if str(scope_id).strip()
            )
    plan_effect_scope_ids_t = _canonical_strings(plan_effect_scope_ids)

    # Plan effect ids are typed handles and may participate as premises.
    if plan_effect_ids:
        global_premises = normalize_premise_ids(
            (*global_premises, *(f"plan-effect:{eid}" for eid in plan_effect_ids))
        )
    if residual_handles:
        # Residual refs are premise handles only — never gold bodies.
        global_premises = normalize_premise_ids(
            (*global_premises, *residual_handles)
        )

    compile_requests: list[CodeProofCompileRequest] = []
    for raw in requests:
        if isinstance(raw, CodeProofCompileRequest):
            compile_requests.append(raw)
        elif isinstance(raw, Mapping):
            compile_requests.append(CodeProofCompileRequest.from_dict(raw))
        else:
            raise TypeError("requests entries must be mappings or CodeProofCompileRequest")

    for property_id in _canonical_strings(property_ids):
        compile_requests.append(CodeProofCompileRequest(property_id=property_id))
    for family in _canonical_strings(claim_families):
        compile_requests.append(
            CodeProofCompileRequest(claim_family=_claim_family_value(family))
        )

    # Residual-ref hook: ensure an SRT structural request when residuals present
    # and the caller did not already request one.
    if residual_handles and not any(
        req.claim_family == "srt_structural" or "residual" in req.property_id
        for req in compile_requests
    ):
        compile_requests.append(
            CodeProofCompileRequest(
                claim_family="srt_structural",
                residual_ref_ids=residual_handles,
                metadata={"residual_ref_hook": True},
            )
        )

    if not compile_requests:
        # Default: open every catalog property against available scopes.
        for prop in catalog.properties:
            compile_requests.append(
                CodeProofCompileRequest(property_id=prop.property_id)
            )

    toolchain = str(toolchain_id or "").strip()
    policy = str(policy_id or "").strip()
    default_assurance = (
        AssuranceLevel(str(required_assurance))
        if required_assurance is not None
        else AssuranceLevel.KERNEL_VERIFIED
    )
    budget = resource_budget if resource_budget is not None else _default_resource_budget()

    items: list[CompiledCodeProofItem] = []
    for request in compile_requests:
        (
            prop,
            property_id,
            claim_family,
            template_id,
            code_shape,
            assurance,
        ) = _resolve_property_and_family(request, catalog)
        if assurance is None:
            assurance = default_assurance
        if not isinstance(assurance, AssuranceLevel):
            assurance = AssuranceLevel(str(assurance))

        item_premises = normalize_premise_ids(
            (*global_premises, *request.premise_ids, *request.residual_ref_ids)
        )
        item_residuals = normalize_residual_refs(
            (*residual_handles, *request.residual_ref_ids)
        )
        item_assumptions = global_assumptions
        reason_codes: list[str] = []

        # Force not-measured path (measurement out of bounds / not executed).
        if request.force_not_measured:
            claim = _build_not_measured_or_unsupported_claim(
                status=ObligationCompileStatus.NOT_MEASURED,
                property_id=property_id,
                claim_family=claim_family,
                repository_id=repository_id,
                repository_tree_id=tree_id,
                scope_ids=request.ast_scope_ids,
                premise_ids=item_premises,
                assumption_ids=item_assumptions,
                residual_ref_ids=item_residuals,
                toolchain_id=toolchain,
                policy_id=policy,
                catalog_version=catalog.catalog_version,
                template_id=template_id,
                required_assurance=assurance,
                statement=f"not measured: {property_id or claim_family}",
                metadata={
                    **dict(request.metadata),
                    "reason": "force_not_measured",
                },
            )
            cache_key_id = compiled_obligation_cache_identity(
                property_id=property_id,
                catalog_version=catalog.catalog_version,
                catalog_id=catalog.catalog_id,
                repository_tree_id=tree_id,
                ast_scope_ids=request.ast_scope_ids,
                premise_ids=item_premises,
                assumption_ids=item_assumptions,
                residual_ref_ids=item_residuals,
                toolchain_id=toolchain,
                policy_id=policy,
                required_assurance=assurance,
                template_id=template_id,
            )
            items.append(
                CompiledCodeProofItem(
                    status=ObligationCompileStatus.NOT_MEASURED,
                    property_id=property_id,
                    claim_family=claim_family,
                    obligation=None,
                    claim=claim,
                    cache_key_id=cache_key_id,
                    premise_ids=item_premises,
                    assumption_ids=item_assumptions,
                    residual_ref_ids=item_residuals,
                    invalidation_selectors=tuple(
                        selector.to_dict() for selector in claim.invalidation_selectors
                    ),
                    reason_codes=("not_measured", "force_not_measured"),
                    template_id=template_id,
                    catalog_version=catalog.catalog_version,
                    required_assurance=assurance,
                    metadata=dict(request.metadata),
                )
            )
            continue

        # Unsupported shape / template path.
        is_unsupported = (
            claim_family == ClaimFamily.UNSUPPORTED.value
            or template_id == "unsupported-proof-fail-closed"
            or (
                prop is not None
                and prop.code_shape
                == ReviewedCodeShape.UNSUPPORTED_PROOF_FAIL_CLOSED.value
            )
        )
        if is_unsupported and not request.ast_scope_ids and not template_id:
            template_id = "unsupported-proof-fail-closed"

        selected_scopes: tuple[CodeProofScope, ...] = ()
        scope_error: str | None = None
        try:
            selected_scopes = _scopes_for_family(
                scope_set,
                claim_family,
                requested_scope_ids=request.ast_scope_ids,
                plan_effect_scope_ids=plan_effect_scope_ids_t,
            )
        except UnsupportedProofTemplateError as exc:
            scope_error = str(exc)
            reason_codes.append("unsupported_scopes")
        except ValueError as exc:
            scope_error = str(exc)
            reason_codes.append("invalid_scopes")

        if not selected_scopes and not is_unsupported:
            # Supported family without measurable scopes → not_measured.
            claim = _build_not_measured_or_unsupported_claim(
                status=ObligationCompileStatus.NOT_MEASURED,
                property_id=property_id,
                claim_family=claim_family,
                repository_id=repository_id,
                repository_tree_id=tree_id,
                scope_ids=(),
                premise_ids=item_premises,
                assumption_ids=item_assumptions,
                residual_ref_ids=item_residuals,
                toolchain_id=toolchain,
                policy_id=policy,
                catalog_version=catalog.catalog_version,
                template_id=template_id,
                required_assurance=assurance,
                statement=(
                    f"not measured: no matching AST scopes for {property_id or claim_family}"
                ),
                metadata={
                    **dict(request.metadata),
                    "reason": scope_error or "no_matching_scopes",
                },
            )
            cache_key_id = compiled_obligation_cache_identity(
                property_id=property_id,
                catalog_version=catalog.catalog_version,
                catalog_id=catalog.catalog_id,
                repository_tree_id=tree_id,
                ast_scope_ids=(),
                premise_ids=item_premises,
                assumption_ids=item_assumptions,
                residual_ref_ids=item_residuals,
                toolchain_id=toolchain,
                policy_id=policy,
                required_assurance=assurance,
                template_id=template_id,
            )
            items.append(
                CompiledCodeProofItem(
                    status=ObligationCompileStatus.NOT_MEASURED,
                    property_id=property_id,
                    claim_family=claim_family,
                    obligation=None,
                    claim=claim,
                    cache_key_id=cache_key_id,
                    premise_ids=item_premises,
                    assumption_ids=item_assumptions,
                    residual_ref_ids=item_residuals,
                    invalidation_selectors=tuple(
                        selector.to_dict() for selector in claim.invalidation_selectors
                    ),
                    reason_codes=tuple(
                        sorted(set(reason_codes) | {"not_measured", "no_matching_scopes"})
                    ),
                    template_id=template_id,
                    catalog_version=catalog.catalog_version,
                    required_assurance=assurance,
                    metadata=dict(request.metadata),
                )
            )
            continue

        if not template_id:
            template_id = _FAMILY_FALLBACK_TEMPLATE.get(claim_family, "")
        if not template_id:
            # Cannot materialize without a reviewed template.
            claim = _build_not_measured_or_unsupported_claim(
                status=ObligationCompileStatus.UNSUPPORTED,
                property_id=property_id or f"property:family:{claim_family}",
                claim_family="unsupported",
                repository_id=repository_id,
                repository_tree_id=tree_id,
                scope_ids=tuple(scope.scope_id for scope in selected_scopes),
                premise_ids=item_premises,
                assumption_ids=item_assumptions,
                residual_ref_ids=item_residuals,
                toolchain_id=toolchain,
                policy_id=policy,
                catalog_version=catalog.catalog_version,
                template_id="",
                required_assurance=assurance,
                statement="unsupported: no reviewed template for claim family",
                metadata={
                    **dict(request.metadata),
                    "reason": "no_reviewed_template",
                    "requested_claim_family": claim_family,
                },
            )
            cache_key_id = compiled_obligation_cache_identity(
                property_id=property_id,
                catalog_version=catalog.catalog_version,
                catalog_id=catalog.catalog_id,
                repository_tree_id=tree_id,
                ast_scope_ids=tuple(scope.scope_id for scope in selected_scopes),
                premise_ids=item_premises,
                assumption_ids=item_assumptions,
                residual_ref_ids=item_residuals,
                toolchain_id=toolchain,
                policy_id=policy,
                required_assurance=assurance,
            )
            items.append(
                CompiledCodeProofItem(
                    status=ObligationCompileStatus.UNSUPPORTED,
                    property_id=property_id or f"property:family:{claim_family}",
                    claim_family="unsupported",
                    obligation=None,
                    claim=claim,
                    cache_key_id=cache_key_id,
                    premise_ids=item_premises,
                    assumption_ids=item_assumptions,
                    residual_ref_ids=item_residuals,
                    invalidation_selectors=tuple(
                        selector.to_dict() for selector in claim.invalidation_selectors
                    ),
                    reason_codes=("unsupported", "no_reviewed_template"),
                    template_id="",
                    catalog_version=catalog.catalog_version,
                    required_assurance=assurance,
                    metadata=dict(request.metadata),
                )
            )
            continue

        # Ensure scopes for unsupported fail-closed path.
        if not selected_scopes:
            try:
                selected_scopes = _selected_obligation_scopes(scope_set, ())
            except UnsupportedProofTemplateError:
                selected_scopes = ()
            if not selected_scopes:
                # Still emit unsupported claim even without scopes.
                claim = _build_not_measured_or_unsupported_claim(
                    status=ObligationCompileStatus.UNSUPPORTED,
                    property_id=property_id
                    or "property:unsupported-proof-fail-closed",
                    claim_family="unsupported",
                    repository_id=repository_id,
                    repository_tree_id=tree_id,
                    scope_ids=(),
                    premise_ids=item_premises,
                    assumption_ids=item_assumptions,
                    residual_ref_ids=item_residuals,
                    toolchain_id=toolchain,
                    policy_id=policy,
                    catalog_version=catalog.catalog_version,
                    template_id=template_id,
                    required_assurance=assurance,
                    statement="unsupported proof shape (fail closed)",
                    metadata=dict(request.metadata),
                )
                cache_key_id = compiled_obligation_cache_identity(
                    property_id=property_id,
                    catalog_version=catalog.catalog_version,
                    catalog_id=catalog.catalog_id,
                    repository_tree_id=tree_id,
                    ast_scope_ids=(),
                    premise_ids=item_premises,
                    assumption_ids=item_assumptions,
                    residual_ref_ids=item_residuals,
                    toolchain_id=toolchain,
                    policy_id=policy,
                    required_assurance=assurance,
                    template_id=template_id,
                )
                items.append(
                    CompiledCodeProofItem(
                        status=ObligationCompileStatus.UNSUPPORTED,
                        property_id=property_id
                        or "property:unsupported-proof-fail-closed",
                        claim_family="unsupported",
                        obligation=None,
                        claim=claim,
                        cache_key_id=cache_key_id,
                        premise_ids=item_premises,
                        assumption_ids=item_assumptions,
                        residual_ref_ids=item_residuals,
                        invalidation_selectors=tuple(
                            selector.to_dict()
                            for selector in claim.invalidation_selectors
                        ),
                        reason_codes=("unsupported", "no_ast_scopes"),
                        template_id=template_id,
                        catalog_version=catalog.catalog_version,
                        required_assurance=assurance,
                        metadata=dict(request.metadata),
                    )
                )
                continue

        obligation_metadata = {
            **dict(request.metadata),
            "property_id": property_id,
            "claim_family": claim_family,
            "catalog_version": catalog.catalog_version,
            "catalog_id": catalog.catalog_id,
            "premise_digest": premise_set_digest(item_premises),
            "assumption_digest": assumption_set_digest(item_assumptions),
            "residual_ref_ids": list(item_residuals),
            "plan_effect_ids": list(plan_effect_ids),
            "semantic_authority": False,
        }
        if code_shape:
            obligation_metadata["code_shape"] = code_shape

        try:
            obligation = materialize_code_proof_obligation(
                scope_set,
                repository_tree_id=tree_id,
                repository_id=repository_id,
                template_id=template_id,
                template_version=request.template_version or None,
                ast_scope_ids=tuple(scope.scope_id for scope in selected_scopes),
                code_shape=code_shape,
                premise_ids=item_premises,
                required_assurance=assurance,
                task_id=task_id,
                metadata=obligation_metadata,
                registry=registry,
            )
        except UnsupportedProofTemplateError as exc:
            # Shape/template mismatch → unsupported (reviewed refusal).
            claim = _build_not_measured_or_unsupported_claim(
                status=ObligationCompileStatus.UNSUPPORTED,
                property_id=property_id,
                claim_family="unsupported"
                if claim_family == "unsupported"
                else claim_family,
                repository_id=repository_id,
                repository_tree_id=tree_id,
                scope_ids=tuple(scope.scope_id for scope in selected_scopes),
                premise_ids=item_premises,
                assumption_ids=item_assumptions,
                residual_ref_ids=item_residuals,
                toolchain_id=toolchain,
                policy_id=policy,
                catalog_version=catalog.catalog_version,
                template_id=template_id,
                required_assurance=assurance,
                statement=f"unsupported: {exc}",
                metadata={**dict(request.metadata), "reason": str(exc)},
            )
            # Keep family distinction: if the request was for a supported family
            # but template refused, still mark compile status unsupported.
            claim = claim.with_updates(
                claim_family=(
                    ClaimFamily.UNSUPPORTED
                    if claim_family == "unsupported"
                    else ClaimFamily(claim_family)
                ),
                status=ClaimStatus.UNSUPPORTED,
            )
            cache_key_id = compiled_obligation_cache_identity(
                property_id=property_id,
                catalog_version=catalog.catalog_version,
                catalog_id=catalog.catalog_id,
                repository_tree_id=tree_id,
                ast_scope_ids=tuple(scope.scope_id for scope in selected_scopes),
                premise_ids=item_premises,
                assumption_ids=item_assumptions,
                residual_ref_ids=item_residuals,
                toolchain_id=toolchain,
                policy_id=policy,
                required_assurance=assurance,
                template_id=template_id,
            )
            items.append(
                CompiledCodeProofItem(
                    status=ObligationCompileStatus.UNSUPPORTED,
                    property_id=property_id,
                    claim_family=claim_family,
                    obligation=None,
                    claim=claim,
                    cache_key_id=cache_key_id,
                    premise_ids=item_premises,
                    assumption_ids=item_assumptions,
                    residual_ref_ids=item_residuals,
                    invalidation_selectors=tuple(
                        selector.to_dict() for selector in claim.invalidation_selectors
                    ),
                    reason_codes=("unsupported", "template_refusal"),
                    template_id=template_id,
                    catalog_version=catalog.catalog_version,
                    required_assurance=assurance,
                    metadata=dict(request.metadata),
                )
            )
            continue

        # Emit typed claim with explicit premise/assumption ids + invalidators.
        family_enum = resolve_claim_family(
            template_id=obligation.template_id,
            invariant_class=obligation.invariant_class,
            property_id=property_id,
            code_shape=code_shape,
            explicit=claim_family,
        )
        claim_status = (
            ClaimStatus.UNSUPPORTED
            if family_enum is ClaimFamily.UNSUPPORTED
            or is_unsupported
            else ClaimStatus.OPEN
        )
        compile_status = (
            ObligationCompileStatus.UNSUPPORTED
            if claim_status is ClaimStatus.UNSUPPORTED
            else ObligationCompileStatus.OPEN
        )
        claim = claim_from_obligation(
            obligation,
            property_id=property_id,
            claim_family=family_enum,
            assumption_ids=item_assumptions,
            producer_id=OBLIGATION_COMPILER_PRODUCER_ID,
            toolchain_id=toolchain,
            policy_id=policy,
            catalog_version=catalog.catalog_version,
            status=claim_status,
            metadata={
                **dict(request.metadata),
                "residual_ref_ids": list(item_residuals),
                "plan_effect_ids": list(plan_effect_ids),
                "premise_digest": premise_set_digest(item_premises),
                "assumption_digest": assumption_set_digest(item_assumptions),
                "catalog_id": catalog.catalog_id,
                "semantic_authority": False,
            },
        )

        cache_key = build_code_proof_cache_key(
            obligation,
            translator_id=translator_id,
            solver_id=solver_id,
            kernel_id=kernel_id,
            toolchain_id=toolchain or "toolchain:default",
            theorem_registry_id=theorem_registry_id,
            policy_id=policy or "policy:default",
            resource_budget=budget,
            candidate_tree=tree_id,
            property_id=property_id,
            catalog_version=catalog.catalog_version,
            catalog_id=catalog.catalog_id,
            assumption_ids=item_assumptions,
            residual_ref_ids=item_residuals,
        )
        # Also expose the compact content-identity used by capsule/query layers.
        compact_id = compiled_obligation_cache_identity(
            property_id=property_id,
            catalog_version=catalog.catalog_version,
            catalog_id=catalog.catalog_id,
            repository_tree_id=tree_id,
            ast_scope_ids=obligation.ast_scope_ids,
            premise_ids=item_premises,
            assumption_ids=item_assumptions,
            residual_ref_ids=item_residuals,
            toolchain_id=toolchain or "toolchain:default",
            policy_id=policy or "policy:default",
            required_assurance=assurance,
            template_id=obligation.template_id,
            template_version=obligation.template_version,
            template_semantic_hash=obligation.template_semantic_hash,
            obligation_id=obligation.obligation_id,
        )
        items.append(
            CompiledCodeProofItem(
                status=compile_status,
                property_id=property_id,
                claim_family=family_enum.value,
                obligation=obligation,
                claim=claim,
                cache_key_id=cache_key.key_id,
                premise_ids=item_premises,
                assumption_ids=item_assumptions,
                residual_ref_ids=item_residuals,
                invalidation_selectors=tuple(
                    selector.to_dict() for selector in claim.invalidation_selectors
                ),
                reason_codes=(
                    ("unsupported",) if compile_status is ObligationCompileStatus.UNSUPPORTED else ()
                ),
                template_id=obligation.template_id,
                catalog_version=catalog.catalog_version,
                required_assurance=assurance,
                metadata={
                    **dict(request.metadata),
                    "compact_cache_identity": compact_id,
                    "proof_cache_key_id": cache_key.key_id,
                },
            )
        )

    # Stable order: family then property then status.
    ordered = tuple(
        sorted(
            items,
            key=lambda item: (
                item.claim_family,
                item.property_id,
                item.status.value,
                item.cache_key_id,
            ),
        )
    )
    return CodeProofObligationCompilation(
        repository_id=str(repository_id or "").strip(),
        repository_tree_id=tree_id,
        catalog_version=catalog.catalog_version,
        catalog_id=catalog.catalog_id,
        scope_set_id=scope_set.scope_set_id,
        items=ordered,
        premise_digest=premise_set_digest(global_premises),
        assumption_digest=assumption_set_digest(global_assumptions),
        premise_ids=global_premises,
        assumption_ids=global_assumptions,
        plan_effect_ids=plan_effect_ids,
        residual_ref_ids=residual_handles,
        toolchain_id=toolchain,
        policy_id=policy,
        task_id=str(task_id or "").strip(),
        metadata=dict(metadata or {}),
    )


# Compatibility spellings for the obligation compiler.
compile_code_proof_obligation_set = compile_code_proof_obligations
compile_obligations_from_scopes = compile_code_proof_obligations


# ---------------------------------------------------------------------------
# CBP-120: supervisor self-properties (lease, merge, DAG, freshness)
# ---------------------------------------------------------------------------

SUPERVISOR_SELF_PROPERTY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/supervisor-self-property@1"
)
SUPERVISOR_SELF_PROPERTY_SELECTION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/supervisor-self-property-selection@1"
)
SUPERVISOR_SELF_PROPERTY_POLICY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/supervisor-self-property-policy@1"
)
SUPERVISOR_SELF_PROPERTY_BUNDLE = "agent-supervisor/codebase-proof/self"
SUPERVISOR_SELF_PROPERTY_PRODUCER_ID = (
    "producer:supervisor-self-property-compiler@1"
)

# Canonical property_ids / shapes that always-on (or policy-gated) self proofs
# must exercise.  Template ids come from the reviewed registry via exact shape.
_SELF_PROPERTY_SHAPE_ORDER: tuple[ReviewedCodeShape, ...] = (
    ReviewedCodeShape.LEASE_UNIQUENESS_AND_FENCING,
    ReviewedCodeShape.MERGE_IDEMPOTENCE,
    ReviewedCodeShape.DAG_ACYCLICITY,
    ReviewedCodeShape.EVIDENCE_FRESHNESS,
)

_SELF_PROPERTY_ID_BY_SHAPE: Mapping[str, str] = {
    ReviewedCodeShape.LEASE_UNIQUENESS_AND_FENCING.value: (
        "property:lease-uniqueness-and-fencing"
    ),
    ReviewedCodeShape.MERGE_IDEMPOTENCE.value: "property:merge-idempotence",
    ReviewedCodeShape.DAG_ACYCLICITY.value: "property:dag-acyclicity",
    ReviewedCodeShape.EVIDENCE_FRESHNESS.value: "property:evidence-freshness",
}


@dataclass(frozen=True)
class SupervisorSelfPropertySpec:
    """One always-on (or policy-gated) supervisor self-property binding."""

    property_id: str
    code_shape: str
    template_id: str
    template_version: str
    template_semantic_hash: str
    invariant_class: str
    always_on: bool = True
    title: str = ""

    def __post_init__(self) -> None:
        for name in (
            "property_id",
            "code_shape",
            "template_id",
            "template_version",
            "template_semantic_hash",
            "invariant_class",
            "title",
        ):
            object.__setattr__(
                self, name, str(getattr(self, name) or "").strip()
            )
        if not self.property_id:
            raise ValueError("property_id is required")
        if not self.code_shape:
            raise ValueError("code_shape is required")
        if not self.template_id:
            raise ValueError("template_id is required")
        object.__setattr__(self, "always_on", bool(self.always_on))
        if not self.title:
            object.__setattr__(
                self, "title", self.template_id.replace("-", " ")
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SUPERVISOR_SELF_PROPERTY_SCHEMA,
            "property_id": self.property_id,
            "code_shape": self.code_shape,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "template_semantic_hash": self.template_semantic_hash,
            "invariant_class": self.invariant_class,
            "always_on": self.always_on,
            "title": self.title,
        }


@dataclass(frozen=True)
class SupervisorSelfPropertyPolicy:
    """Policy gate for supervisor self-property obligations.

    * ``enabled=False`` disables every self-property (no compile/prove).
    * ``always_on=True`` (default) enables every reviewed self-property shape.
    * When ``always_on=False``, only ``enabled_property_ids`` / shapes are
      compiled — empty enable lists then mean none.
    """

    enabled: bool = True
    always_on: bool = True
    enabled_property_ids: tuple[str, ...] = ()
    enabled_code_shapes: tuple[str, ...] = ()
    required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "always_on", bool(self.always_on))
        object.__setattr__(
            self,
            "enabled_property_ids",
            _canonical_strings(self.enabled_property_ids),
        )
        shapes = tuple(
            str(getattr(item, "value", item) or "").strip()
            for item in (self.enabled_code_shapes or ())
            if str(getattr(item, "value", item) or "").strip()
        )
        object.__setattr__(self, "enabled_code_shapes", tuple(sorted(set(shapes))))
        assurance = self.required_assurance
        if not isinstance(assurance, AssuranceLevel):
            assurance = AssuranceLevel(str(assurance))
        object.__setattr__(self, "required_assurance", assurance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SUPERVISOR_SELF_PROPERTY_POLICY_SCHEMA,
            "enabled": self.enabled,
            "always_on": self.always_on,
            "enabled_property_ids": list(self.enabled_property_ids),
            "enabled_code_shapes": list(self.enabled_code_shapes),
            "required_assurance": self.required_assurance.value,
        }

    @classmethod
    def always_on_default(
        cls,
        *,
        required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED,
    ) -> "SupervisorSelfPropertyPolicy":
        return cls(
            enabled=True,
            always_on=True,
            required_assurance=required_assurance,
        )

    @classmethod
    def from_value(
        cls,
        value: "SupervisorSelfPropertyPolicy | bool | Mapping[str, Any] | None",
    ) -> "SupervisorSelfPropertyPolicy":
        if value is None:
            return cls.always_on_default()
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls(enabled=value, always_on=value)
        if isinstance(value, Mapping):
            return cls(
                enabled=bool(value.get("enabled", True)),
                always_on=bool(value.get("always_on", True)),
                enabled_property_ids=tuple(
                    value.get("enabled_property_ids") or ()
                ),
                enabled_code_shapes=tuple(
                    value.get("enabled_code_shapes") or ()
                ),
                required_assurance=value.get(
                    "required_assurance", AssuranceLevel.KERNEL_VERIFIED
                ),
            )
        raise TypeError(
            "supervisor self-property policy must be bool, mapping, "
            "SupervisorSelfPropertyPolicy, or None"
        )


@dataclass(frozen=True)
class SupervisorSelfPropertySelection:
    """Exact ReviewedCodeShape → reviewed template selection for self proofs."""

    specs: tuple[SupervisorSelfPropertySpec, ...]
    registry_version: str = ""
    policy: SupervisorSelfPropertyPolicy = field(
        default_factory=SupervisorSelfPropertyPolicy.always_on_default
    )

    def __post_init__(self) -> None:
        if not isinstance(self.specs, tuple):
            object.__setattr__(self, "specs", tuple(self.specs))
        ordered = tuple(
            sorted(self.specs, key=lambda item: item.property_id)
        )
        object.__setattr__(self, "specs", ordered)
        object.__setattr__(
            self, "registry_version", str(self.registry_version or "").strip()
        )
        if not isinstance(self.policy, SupervisorSelfPropertyPolicy):
            object.__setattr__(
                self, "policy", SupervisorSelfPropertyPolicy.from_value(self.policy)
            )

    @property
    def property_ids(self) -> tuple[str, ...]:
        return tuple(spec.property_id for spec in self.specs)

    @property
    def code_shapes(self) -> tuple[str, ...]:
        return tuple(spec.code_shape for spec in self.specs)

    @property
    def template_ids(self) -> tuple[str, ...]:
        return tuple(spec.template_id for spec in self.specs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SUPERVISOR_SELF_PROPERTY_SELECTION_SCHEMA,
            "bundle": SUPERVISOR_SELF_PROPERTY_BUNDLE,
            "registry_version": self.registry_version,
            "property_ids": list(self.property_ids),
            "code_shapes": list(self.code_shapes),
            "template_ids": list(self.template_ids),
            "specs": [spec.to_dict() for spec in self.specs],
            "policy": self.policy.to_dict(),
        }


def default_supervisor_self_property_shapes() -> tuple[ReviewedCodeShape, ...]:
    """Return the closed always-on self-property shape population (CBP-120)."""

    return _SELF_PROPERTY_SHAPE_ORDER


def default_supervisor_self_property_ids() -> tuple[str, ...]:
    """Return catalog property ids for the closed self-property population."""

    return tuple(
        _SELF_PROPERTY_ID_BY_SHAPE[shape.value]
        for shape in _SELF_PROPERTY_SHAPE_ORDER
    )


def _normalize_self_code_shapes(
    shapes: Sequence[str | ReviewedCodeShape] | None,
) -> tuple[str, ...]:
    if shapes is None:
        return tuple(shape.value for shape in _SELF_PROPERTY_SHAPE_ORDER)
    normalized: list[str] = []
    seen: set[str] = set()
    for item in shapes:
        value = str(getattr(item, "value", item) or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return tuple(normalized)


def select_supervisor_self_templates(
    registry: ProofObligationTemplateRegistry = DEFAULT_TEMPLATE_REGISTRY,
    *,
    code_shapes: Sequence[str | ReviewedCodeShape] | None = None,
    policy: SupervisorSelfPropertyPolicy | bool | Mapping[str, Any] | None = None,
    catalog: Any = None,
) -> SupervisorSelfPropertySelection:
    """Select reviewed templates for supervisor self-properties by exact shape.

    Shape membership is exact: unknown or ambiguous shapes fail closed via the
    template registry.  Catalog property ids are preferred when present.
    """

    resolved_policy = SupervisorSelfPropertyPolicy.from_value(policy)
    if not resolved_policy.enabled:
        return SupervisorSelfPropertySelection(
            specs=(),
            registry_version=str(getattr(registry, "registry_version", "") or ""),
            policy=resolved_policy,
        )

    requested_shapes = _normalize_self_code_shapes(code_shapes)
    if not resolved_policy.always_on:
        allowed_shapes = set(resolved_policy.enabled_code_shapes)
        allowed_ids = set(resolved_policy.enabled_property_ids)
        if not allowed_shapes and not allowed_ids:
            return SupervisorSelfPropertySelection(
                specs=(),
                registry_version=str(
                    getattr(registry, "registry_version", "") or ""
                ),
                policy=resolved_policy,
            )
        filtered: list[str] = []
        for shape in requested_shapes:
            property_id = _SELF_PROPERTY_ID_BY_SHAPE.get(shape, f"property:{shape}")
            if allowed_shapes and shape in allowed_shapes:
                filtered.append(shape)
            elif allowed_ids and property_id in allowed_ids:
                filtered.append(shape)
            elif not allowed_shapes and property_id in allowed_ids:
                filtered.append(shape)
        requested_shapes = tuple(filtered)

    specs: list[SupervisorSelfPropertySpec] = []
    for shape in requested_shapes:
        selection = registry.select_for_code_shape(shape)
        template = selection.require_supported()
        property_id = _SELF_PROPERTY_ID_BY_SHAPE.get(
            shape, f"property:{template.template_id}"
        )
        if catalog is not None:
            matched = catalog.get(property_id)
            if matched is not None:
                property_id = matched.property_id
                # Catalog binding must still match the exact reviewed shape.
                if str(matched.code_shape or "") != shape:
                    raise UnsupportedProofTemplateError(
                        f"catalog property {property_id!r} code_shape "
                        f"{matched.code_shape!r} does not match exact shape "
                        f"{shape!r}"
                    )
                if str(matched.template_id or "") != template.template_id:
                    raise UnsupportedProofTemplateError(
                        f"catalog property {property_id!r} template "
                        f"{matched.template_id!r} does not match shape-selected "
                        f"template {template.template_id!r}"
                    )
        specs.append(
            SupervisorSelfPropertySpec(
                property_id=property_id,
                code_shape=shape,
                template_id=template.template_id,
                template_version=str(template.version),
                template_semantic_hash=str(template.semantic_hash),
                invariant_class=str(template.invariant_class or ""),
                always_on=resolved_policy.always_on,
                title=template.template_id.replace("-", " "),
            )
        )

    return SupervisorSelfPropertySelection(
        specs=tuple(specs),
        registry_version=str(getattr(registry, "registry_version", "") or ""),
        policy=resolved_policy,
    )


def evaluate_supervisor_self_property_mutations(
    registry: ProofObligationTemplateRegistry = DEFAULT_TEMPLATE_REGISTRY,
    *,
    code_shapes: Sequence[str | ReviewedCodeShape] | None = None,
    policy: SupervisorSelfPropertyPolicy | bool | Mapping[str, Any] | None = None,
) -> dict[str, dict[str, bool]]:
    """Run reviewed mutation cases for each selected self-property template.

    Returns ``{template_id: {case_id: passed}}``.  Every case must evaluate to
    its declared expected result for the self-property wiring to be sound.
    """

    selection = select_supervisor_self_templates(
        registry, code_shapes=code_shapes, policy=policy
    )
    outcomes: dict[str, dict[str, bool]] = {}
    for spec in selection.specs:
        template = registry.require(spec.template_id, spec.template_version or None)
        if not template.supports_code_shape(spec.code_shape):
            raise UnsupportedProofTemplateError(
                f"template {template.template_id!r} does not support exact "
                f"code shape {spec.code_shape!r}"
            )
        results = template.verify_mutation_cases()
        outcomes[spec.template_id] = dict(results)
        failed = [case_id for case_id, ok in results.items() if not ok]
        if failed:
            raise UnsupportedProofTemplateError(
                f"self-property mutation cases failed for "
                f"{spec.template_id!r}: {', '.join(sorted(failed))}"
            )
    return outcomes


def _self_property_compile_requests(
    selection: SupervisorSelfPropertySelection,
) -> list[CodeProofCompileRequest]:
    requests: list[CodeProofCompileRequest] = []
    for spec in selection.specs:
        requests.append(
            CodeProofCompileRequest(
                property_id=spec.property_id,
                template_id=spec.template_id,
                template_version=spec.template_version,
                code_shape=spec.code_shape,
                required_assurance=selection.policy.required_assurance,
                metadata={
                    "supervisor_self_property": True,
                    "bundle": SUPERVISOR_SELF_PROPERTY_BUNDLE,
                    "always_on": spec.always_on,
                    "producer_id": SUPERVISOR_SELF_PROPERTY_PRODUCER_ID,
                    "code_shape": spec.code_shape,
                },
            )
        )
    return requests


def compile_supervisor_self_properties(
    scope_set: CodeProofScopeSet | None = None,
    *,
    repository_tree_id: str,
    repository_id: str = "",
    candidate_diff: Any = None,
    policy: SupervisorSelfPropertyPolicy | bool | Mapping[str, Any] | None = None,
    code_shapes: Sequence[str | ReviewedCodeShape] | None = None,
    catalog: Any = None,
    premise_ids: Sequence[str] = (),
    assumption_ids: Sequence[str] = (),
    residual_refs: Sequence[Any] = (),
    formal_plan_effects: Sequence[Any] = (),
    effect_scope_map: Mapping[str, Sequence[str]] | None = None,
    toolchain_id: str = "",
    policy_id: str = "",
    translator_id: str = "translator:default",
    solver_id: str = "solver:default",
    kernel_id: str = "kernel:default",
    theorem_registry_id: str = "registry:default",
    resource_budget: Any = None,
    task_id: str = "",
    required_assurance: AssuranceLevel | str | None = None,
    registry: ProofObligationTemplateRegistry = DEFAULT_TEMPLATE_REGISTRY,
    metadata: Mapping[str, Any] | None = None,
    verify_mutation_cases: bool = True,
) -> CodeProofObligationCompilation:
    """Compile always-on / policy-gated supervisor self-property obligations.

    Templates are selected only by exact :class:`ReviewedCodeShape` membership.
    Each open item binds the selected template semantic hash and code shape so
    the trust-aware proof cache can warm-hit on re-proof and invalidate on
    tree/premise/toolchain/policy mutations.
    """

    from .code_property_catalog import DEFAULT_CODE_PROPERTY_CATALOG

    resolved_policy = SupervisorSelfPropertyPolicy.from_value(policy)
    if required_assurance is not None:
        assurance = (
            required_assurance
            if isinstance(required_assurance, AssuranceLevel)
            else AssuranceLevel(str(required_assurance))
        )
        resolved_policy = SupervisorSelfPropertyPolicy(
            enabled=resolved_policy.enabled,
            always_on=resolved_policy.always_on,
            enabled_property_ids=resolved_policy.enabled_property_ids,
            enabled_code_shapes=resolved_policy.enabled_code_shapes,
            required_assurance=assurance,
        )

    if catalog is None:
        catalog = DEFAULT_CODE_PROPERTY_CATALOG

    selection = select_supervisor_self_templates(
        registry,
        code_shapes=code_shapes,
        policy=resolved_policy,
        catalog=catalog,
    )
    if verify_mutation_cases and selection.specs:
        evaluate_supervisor_self_property_mutations(
            registry,
            code_shapes=selection.code_shapes,
            policy=resolved_policy,
        )

    meta = {
        "bundle": SUPERVISOR_SELF_PROPERTY_BUNDLE,
        "producer_id": SUPERVISOR_SELF_PROPERTY_PRODUCER_ID,
        "supervisor_self_properties": True,
        "self_property_policy": resolved_policy.to_dict(),
        "selected_code_shapes": list(selection.code_shapes),
        "selected_template_ids": list(selection.template_ids),
        **dict(metadata or {}),
    }

    if not selection.specs:
        # Policy disabled / empty enable list: emit an empty compilation bound
        # to the candidate scopes so callers still get a stable tree handle.
        if scope_set is None:
            if candidate_diff is None:
                raise ValueError("scope_set or candidate_diff is required")
            scope_set = compile_candidate_proof_scopes(candidate_diff)
        tree_id = str(repository_tree_id or "").strip()
        if not tree_id:
            raise ValueError("repository_tree_id is required")
        return CodeProofObligationCompilation(
            repository_id=str(repository_id or "").strip(),
            repository_tree_id=tree_id,
            catalog_version=str(getattr(catalog, "catalog_version", "") or ""),
            catalog_id=str(getattr(catalog, "catalog_id", "") or ""),
            scope_set_id=scope_set.scope_set_id,
            items=(),
            premise_digest=premise_set_digest(normalize_premise_ids(premise_ids)),
            assumption_digest=assumption_set_digest(
                normalize_assumption_ids(assumption_ids)
            ),
            premise_ids=normalize_premise_ids(premise_ids),
            assumption_ids=normalize_assumption_ids(assumption_ids),
            plan_effect_ids=_normalize_plan_effect_ids(formal_plan_effects),
            residual_ref_ids=normalize_residual_refs(residual_refs),
            toolchain_id=str(toolchain_id or "").strip(),
            policy_id=str(policy_id or "").strip(),
            task_id=str(task_id or "").strip(),
            metadata=meta,
        )

    compilation = compile_code_proof_obligations(
        scope_set,
        repository_tree_id=repository_tree_id,
        repository_id=repository_id,
        candidate_diff=candidate_diff,
        requests=_self_property_compile_requests(selection),
        catalog=catalog,
        formal_plan_effects=formal_plan_effects,
        effect_scope_map=effect_scope_map,
        residual_refs=residual_refs,
        premise_ids=premise_ids,
        assumption_ids=assumption_ids,
        toolchain_id=toolchain_id,
        policy_id=policy_id,
        translator_id=translator_id,
        solver_id=solver_id,
        kernel_id=kernel_id,
        theorem_registry_id=theorem_registry_id,
        resource_budget=resource_budget,
        task_id=task_id,
        # Pass the enum value string: compile_code_proof_obligations coerces via
        # AssuranceLevel(str(...)), which rejects str(enum_member) name forms.
        required_assurance=resolved_policy.required_assurance.value,
        registry=registry,
        metadata=meta,
    )
    # Ensure every selected shape survived compile with exact shape binding.
    by_property = {item.property_id: item for item in compilation.items}
    for spec in selection.specs:
        item = by_property.get(spec.property_id)
        if item is None:
            raise UnsupportedProofTemplateError(
                f"self-property {spec.property_id!r} missing from compilation"
            )
        if item.template_id and item.template_id != spec.template_id:
            raise UnsupportedProofTemplateError(
                f"self-property {spec.property_id!r} compiled with template "
                f"{item.template_id!r}, expected {spec.template_id!r}"
            )
        if item.obligation is not None:
            bound_shape = str(
                (item.obligation.metadata or {}).get("code_shape") or ""
            )
            if bound_shape and bound_shape != spec.code_shape:
                raise UnsupportedProofTemplateError(
                    f"self-property {spec.property_id!r} bound shape "
                    f"{bound_shape!r}, expected exact {spec.code_shape!r}"
                )
    return compilation


def prove_supervisor_self_properties(
    cache: Any,
    compilation: CodeProofObligationCompilation,
    *,
    prove: Callable[[Any, Any], ProofReceipt],
    previous: CodeProofObligationCompilation | None = None,
    metrics: "ProofCacheMetrics | None" = None,
    translator_id: str = "translator:default",
    solver_id: str = "solver:default",
    kernel_id: str = "kernel:default",
    theorem_registry_id: str = "registry:default",
    resource_budget: Any = None,
    changed_paths: Sequence[str] = (),
    dependency_edge_changed: bool = False,
    prefer_cache_before_provider: bool = True,
) -> Any:
    """Prove / re-prove supervisor self-property obligations cache-first.

    Cold path invokes ``prove`` once per open obligation; warm re-proof of an
    unchanged binding must hit the trust-aware proof cache without a second
    provider call.  Binding mutations invalidate and force re-solve.
    """

    # Local import keeps the CBP-050 re-export graph stable for type checkers.
    from .code_proof_reproof import reprove_code_proof_compilation as _reprove

    if not isinstance(compilation, CodeProofObligationCompilation):
        raise TypeError(
            "compilation must be a CodeProofObligationCompilation"
        )
    return _reprove(
        cache,
        compilation,
        prove=prove,
        previous=previous,
        metrics=metrics,
        translator_id=translator_id,
        solver_id=solver_id,
        kernel_id=kernel_id,
        theorem_registry_id=theorem_registry_id,
        resource_budget=resource_budget,
        changed_paths=changed_paths,
        dependency_edge_changed=dependency_edge_changed,
        prefer_cache_before_provider=prefer_cache_before_provider,
    )


# ---------------------------------------------------------------------------

# CBP-015: cache-first prove path over TrustAwareProofCache
# ---------------------------------------------------------------------------

from collections import Counter
from collections.abc import Callable

from .formal_verification_cache import (
    CacheLookupStatus,
    CacheRejectionReason,
    FormalVerificationCache,
    ProofCacheKey,
    TrustAwareProofCache,
    build_proof_cache_key,
)


@dataclass
class ProofCacheMetrics:
    """Process-local hit/miss/reject counters for the cache-first prove path."""

    hits: int = 0
    misses: int = 0
    rejects: int = 0
    puts: int = 0
    put_failures: int = 0
    single_flight_calls: int = 0
    reject_reasons: Counter[str] = field(default_factory=Counter)

    def record_reject(self, *reason_codes: str) -> None:
        self.rejects += 1
        for code in reason_codes:
            if code:
                self.reject_reasons[str(code)] += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "rejects": self.rejects,
            "puts": self.puts,
            "put_failures": self.put_failures,
            "single_flight_calls": self.single_flight_calls,
            "reject_reasons": dict(sorted(self.reject_reasons.items())),
            "hit_rate": (
                self.hits / (self.hits + self.misses)
                if (self.hits + self.misses)
                else 0.0
            ),
        }


@dataclass(frozen=True)
class CachedProveResult:
    """Outcome of :func:`prove_code_obligation_with_cache`."""

    status: str
    from_cache: bool
    receipt: ProofReceipt | None
    reason_codes: tuple[str, ...] = ()
    key_id: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def authoritative(self) -> bool:
        return self.receipt is not None and self.receipt.authoritative_assurance not in (
            AssuranceLevel.UNVERIFIED,
            AssuranceLevel.CANDIDATE,
        )


def build_code_proof_cache_key(
    obligation: CodeProofObligation,
    *,
    translator_id: str = "translator:default",
    solver_id: str = "solver:default",
    kernel_id: str = "kernel:default",
    toolchain_id: str = "toolchain:default",
    theorem_registry_id: str = "registry:default",
    policy_id: str = "policy:default",
    resource_budget: Any = None,
    candidate_tree: str | None = None,
    property_id: str = "",
    catalog_version: str = "",
    catalog_id: str = "",
    assumption_ids: Sequence[str] = (),
    residual_ref_ids: Sequence[str] = (),
) -> ProofCacheKey:
    """Build a :class:`ProofCacheKey` from a typed code-proof obligation.

    The key binds obligation identity (which already includes template semantics),
    property/catalog version, tree/scope, premise and assumption digests,
    toolchain/policy identities, required assurance, and candidate tree.
    """

    if not isinstance(obligation, CodeProofObligation):
        raise TypeError("obligation must be a CodeProofObligation")
    budget = resource_budget
    if budget is None:
        budget = {
            "wall_time_ms": 30_000,
            "cpu_time_ms": 20_000,
            "memory_bytes": 512 * 1024 * 1024,
            "max_processes": 4,
            "max_premises": 32,
            "network_allowed": False,
        }
    if hasattr(budget, "to_dict"):
        budget = budget.to_dict()
    tree = candidate_tree if candidate_tree is not None else obligation.repository_tree_id
    premises = tuple(obligation.premise_ids)
    assumptions = _canonical_strings(assumption_ids)
    residuals = _canonical_strings(residual_ref_ids)
    # Include required assurance, catalog, and property in the obligation
    # component so raising the bar or catalog drift cannot reuse a weaker
    # cached receipt under the same key.
    obligation_component = {
        "obligation_id": obligation.obligation_id,
        "repository_tree_id": obligation.repository_tree_id,
        "ast_scope_ids": list(obligation.ast_scope_ids),
        "premise_ids": list(premises),
        "premise_digest": premise_set_digest(premises),
        "assumption_ids": list(assumptions),
        "assumption_digest": assumption_set_digest(assumptions),
        "residual_ref_ids": list(residuals),
        "template_id": obligation.template_id,
        "template_version": obligation.template_version,
        "template_semantic_hash": obligation.template_semantic_hash,
        "required_assurance": obligation.required_assurance.value,
        "property_id": str(property_id or "").strip(),
        "catalog_version": str(catalog_version or "").strip(),
        "catalog_id": str(catalog_id or "").strip(),
        "policy_id": policy_id,
        "toolchain_id": toolchain_id,
    }
    return build_proof_cache_key(
        obligation=obligation_component,
        premises=premises,
        translator=translator_id,
        solver=solver_id,
        kernel=kernel_id,
        toolchain=toolchain_id,
        theorem_registry=theorem_registry_id,
        policy=policy_id,
        resource_budget=budget,
        candidate_tree=tree,
    )


def _map_binding_reason(
    key: ProofCacheKey,
    receipt: ProofReceipt,
    reasons: Iterable[str],
) -> tuple[str, ...]:
    """Augment cache reasons with stale_tree / toolchain_drift aliases."""

    codes = {str(code) for code in reasons if code}
    if receipt.repository_tree_id and str(key.candidate_tree) not in (
        str(receipt.repository_tree_id),
        receipt.repository_tree_id,
    ):
        if str(key.candidate_tree) != str(receipt.repository_tree_id):
            codes.add("stale_tree")
    if receipt.toolchain_id and str(key.toolchain) != str(receipt.toolchain_id):
        codes.add("toolchain_drift")
    if receipt.authoritative_assurance in (
        AssuranceLevel.UNVERIFIED,
        AssuranceLevel.CANDIDATE,
    ):
        codes.add("candidate_only")
    return tuple(sorted(codes))


def prove_code_obligation_with_cache(
    cache: FormalVerificationCache | TrustAwareProofCache,
    key: ProofCacheKey,
    *,
    prove: Callable[[], ProofReceipt],
    required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED,
    metrics: ProofCacheMetrics | None = None,
    prefer_cache_before_provider: bool = True,
    store_on_success: bool = True,
) -> CachedProveResult:
    """Lookup-before-provider prove path with single-flight and metrics.

    Hits re-derive assurance inside :meth:`FormalVerificationCache.lookup`.
    Candidate-only receipts are never stored as authoritative cache entries.
    """

    if not callable(prove):
        raise TypeError("prove must be callable")
    stats = metrics if metrics is not None else ProofCacheMetrics()

    if prefer_cache_before_provider:
        lookup = cache.lookup(key, required_assurance=required_assurance)
        if lookup.status is CacheLookupStatus.HIT and lookup.receipt is not None:
            # Re-derive is already performed; refuse candidate-only hits.
            if lookup.receipt.authoritative_assurance in (
                AssuranceLevel.UNVERIFIED,
                AssuranceLevel.CANDIDATE,
            ):
                stats.record_reject("candidate_only")
                return CachedProveResult(
                    status="rejected",
                    from_cache=True,
                    receipt=None,
                    reason_codes=("candidate_only",),
                    key_id=key.key_id,
                    metrics=stats.snapshot(),
                )
            stats.hits += 1
            return CachedProveResult(
                status="hit",
                from_cache=True,
                receipt=lookup.receipt,
                reason_codes=(),
                key_id=key.key_id,
                metrics=stats.snapshot(),
            )
        if lookup.status is CacheLookupStatus.REJECTED:
            stats.record_reject(*lookup.reason_codes)
            # Fall through to re-prove on recoverable rejections (stale/miss-like).
            recoverable = {
                CacheRejectionReason.STALE_ENTRY.value,
                CacheRejectionReason.CACHE_MISS.value,
                CacheRejectionReason.FRESHNESS_NOT_SATISFIED.value,
            }
            if not set(lookup.reason_codes) & recoverable and lookup.reason_codes:
                # Still attempt prove for insufficient assurance / poisoned after
                # clearing is caller responsibility; treat as miss path.
                pass
        else:
            stats.misses += 1
    else:
        stats.misses += 1

    stats.single_flight_calls += 1

    def _execute() -> dict[str, Any]:
        receipt = prove()
        if not isinstance(receipt, ProofReceipt):
            raise TypeError("prove() must return a ProofReceipt")
        # Single-flight outcomes must be DAG-JSON-safe public values.
        return receipt.to_dict()

    payload = cache.single_flight(key, _execute)
    if isinstance(payload, ProofReceipt):
        receipt = payload
    elif isinstance(payload, Mapping):
        receipt = ProofReceipt.from_dict(payload)
    else:
        raise TypeError(
            "single_flight must return a ProofReceipt or receipt mapping"
        )
    if receipt.authoritative_assurance in (
        AssuranceLevel.UNVERIFIED,
        AssuranceLevel.CANDIDATE,
    ):
        reasons = _map_binding_reason(key, receipt, ("candidate_only",))
        stats.record_reject(*reasons)
        return CachedProveResult(
            status="rejected",
            from_cache=False,
            receipt=receipt,
            reason_codes=reasons,
            key_id=key.key_id,
            metrics=stats.snapshot(),
        )

    if not receipt.satisfies(required_assurance):
        reasons = _map_binding_reason(
            key,
            receipt,
            (CacheRejectionReason.INSUFFICIENT_ASSURANCE.value,),
        )
        stats.record_reject(*reasons)
        return CachedProveResult(
            status="rejected",
            from_cache=False,
            receipt=receipt,
            reason_codes=reasons,
            key_id=key.key_id,
            metrics=stats.snapshot(),
        )

    if store_on_success:
        stored = cache.put(key, receipt)
        if stored.stored:
            stats.puts += 1
        else:
            stats.put_failures += 1
            reasons = _map_binding_reason(key, receipt, stored.reason_codes)
            if "private_material" in str(stored.reason_codes) or any(
                "private" in str(code) for code in stored.reason_codes
            ):
                reasons = tuple(sorted(set(reasons) | {"private_material"}))
            stats.record_reject(*reasons)
            return CachedProveResult(
                status="rejected",
                from_cache=False,
                receipt=receipt,
                reason_codes=reasons,
                key_id=key.key_id,
                metrics=stats.snapshot(),
            )

    return CachedProveResult(
        status="proved",
        from_cache=False,
        receipt=receipt,
        reason_codes=(),
        key_id=key.key_id,
        metrics=stats.snapshot(),
    )


# Compatibility spellings.
prove_code_obligation = prove_code_obligation_with_cache
lookup_or_prove_code_obligation = prove_code_obligation_with_cache


# CBP-050 re-exports (implementation lives in code_proof_reproof).
from .code_proof_reproof import (  # noqa: E402
    InvalidationReason,
    ReproofDisposition,
    ReproofReport,
    invalidation_reasons,
    plan_reproof_from_delta,
    reprove_code_proof_compilation,
)

__all__ = [
    "ASTProofScope",
    "CODE_OBLIGATION_CACHE_KEY_SCHEMA",
    "CODE_OBLIGATION_REQUEST_SCHEMA",
    "CODE_PROOF_BINDING_RESULT_SCHEMA",
    "PROOF_CANDIDATE_NON_AUTHORITY_ACCEPTANCE_CRITERIA",
    "PROOF_CANDIDATE_NON_AUTHORITY_COMPLETION_ANALYZER_VERSION",
    "PROOF_CANDIDATE_NON_AUTHORITY_COMPLETION_CONFIGURATION_REVISION",
    "PROOF_CANDIDATE_NON_AUTHORITY_EVIDENCE_SCHEMA",
    "PROOF_CANDIDATE_NON_AUTHORITY_OBJECTIVE_ID",
    "PROOF_CANDIDATE_NON_AUTHORITY_OBJECTIVE_REVISION",
    "PROOF_CANDIDATE_NON_AUTHORITY_REQUIREMENT_ID",
    "PROOF_CANDIDATE_ACCEPTANCE_CRITERIA",
    "PROOF_CANDIDATE_COMPLETION_ANALYZER_VERSION",
    "PROOF_CANDIDATE_COMPLETION_CONFIGURATION_REVISION",
    "PROOF_CANDIDATE_OBJECTIVE_ID",
    "PROOF_CANDIDATE_OBJECTIVE_REVISION",
    "STRICT_VALIDATION_PARENT_OBJECTIVE_ID",
    "STRICT_VALIDATION_PROOF_COMPLETION_EVIDENCE_SCHEMA",
    "STRICT_VALIDATION_PROOF_GATE_KINDS",
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
    "ProofCandidateNonAuthorityEvidence",
    "StrictValidationProofCompletionEvidence",
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
    "build_code_proof_cache_key",
    "build_obligation_cache_key",
    "CachedProveResult",
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
    "lookup_or_prove_code_obligation",
    "materialize_code_proof_obligation",
    "obligation_cache_identity",
    "parse_unified_diff",
    "ProofCacheMetrics",
    "prove_code_obligation",
    "prove_code_obligation_with_cache",
    "prove_proof_candidate_non_authority",
    "transitive_impact_blocks_proof_derivation",
    "validate_code_proof_receipt_binding",
    "validate_code_proof_receipt_bindings",
    "ObligationCompileStatus",
    "PremiseValidationError",
    "CodeProofCompileRequest",
    "CompiledCodeProofItem",
    "CodeProofObligationCompilation",
    "compile_code_proof_obligations",
    "compiled_obligation_cache_identity",
    "normalize_premise_ids",
    "normalize_assumption_ids",
    "normalize_residual_refs",
    "premise_set_digest",
    "assumption_set_digest",
    "SUPERVISOR_SELF_PROPERTY_SCHEMA",
    "SUPERVISOR_SELF_PROPERTY_SELECTION_SCHEMA",
    "SUPERVISOR_SELF_PROPERTY_POLICY_SCHEMA",
    "SUPERVISOR_SELF_PROPERTY_BUNDLE",
    "SUPERVISOR_SELF_PROPERTY_PRODUCER_ID",
    "SupervisorSelfPropertySpec",
    "SupervisorSelfPropertyPolicy",
    "SupervisorSelfPropertySelection",
    "default_supervisor_self_property_shapes",
    "default_supervisor_self_property_ids",
    "select_supervisor_self_templates",
    "evaluate_supervisor_self_property_mutations",
    "compile_supervisor_self_properties",
    "prove_supervisor_self_properties",
    "InvalidationReason",
    "ReproofDisposition",
    "ReproofReport",
    "invalidation_reasons",
    "plan_reproof_from_delta",
    "reprove_code_proof_compilation",
]
