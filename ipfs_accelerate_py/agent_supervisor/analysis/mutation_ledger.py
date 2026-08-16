"""Before/after code mutation ledger with AST edit lineage (MutationLedger@1).

DQP-022 / Interfaces: ``MutationLedger@1``, ``MutationSet@1``,
``MutationFile@1``, ``ASTMutation@1``
============================================================================

Records admitted code mutations as content-addressed before/after evidence
bound to task, attempt, plan, operator, provider, daemon, session, worktree,
lease, and fence identities.  Every admitted byte change must carry exactly
one AST edit lineage; incomplete or unparseable changes are rejected or
quarantined rather than silently accepted.

Acceptance properties
---------------------
* Every admitted byte change has one lineage or is rejected/quarantined.
* Line-number churn alone does not forge a distinct semantic mutation.
* A stale fence or mismatched before snapshot cannot record an accepted
  mutation.
* Rollback restoration is independently verified against before digests.

Cold import of this module performs no filesystem, database, network,
provider, or process action.  Opening a ledger is the first I/O boundary.
"""

from __future__ import annotations

import ast
import hashlib
import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ..core.conflict_graph import (
    AST_BLOB_RECORD_SCHEMA_VERSION,
    ASTBlobRecord,
    build_python_ast_blob_record,
)
from ..task_sources.duckdb_state import open_duckdb_connection

# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

MUTATION_LEDGER_INTERFACE: Final[str] = "MutationLedger@1"
MUTATION_SET_INTERFACE: Final[str] = "MutationSet@1"
MUTATION_FILE_INTERFACE: Final[str] = "MutationFile@1"
AST_MUTATION_INTERFACE: Final[str] = "ASTMutation@1"

MUTATION_LEDGER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mutation-ledger@1"
)
MUTATION_SET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mutation-set@1"
)
MUTATION_FILE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mutation-file@1"
)
AST_MUTATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ast-mutation@1"
)
MUTATION_LINEAGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mutation-lineage@1"
)
MUTATION_HUNK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mutation-hunk@1"
)
ROLLBACK_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mutation-rollback@1"
)
FENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mutation-fence@1"
)

DEFAULT_PARSER_ID: Final[str] = (
    f"python-ast@schema-{AST_BLOB_RECORD_SCHEMA_VERSION}"
)
DEFAULT_LEDGER_VERSION: Final[str] = "mutation-ledger@1"
AUTHORITY_CLASS: Final[str] = "derived_evidence"

MAX_PATH_BYTES: Final[int] = 4_096
MAX_REASON_BYTES: Final[int] = 1_024
MAX_BODY_JSON_BYTES: Final[int] = 262_144
MAX_SOURCE_BYTES: Final[int] = 16 * 1024 * 1024
MAX_FILES_PER_MUTATION: Final[int] = 10_000
MAX_HUNKS_PER_FILE: Final[int] = 4_096
MAX_HUNK_LINES: Final[int] = 512
MAX_EDIT_OPS: Final[int] = 4_096

_LANGUAGE_BY_SUFFIX: Final[Mapping[str, str]] = MappingProxyType(
    {
        ".py": "python",
        ".pyi": "python",
        ".js": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".jsx": "jsx",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".mts": "typescript",
        ".cts": "typescript",
        ".json": "json",
    }
)
_SUPPORTED_PARSE_LANGUAGES: Final[frozenset[str]] = frozenset({"python"})

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS mutation_ledger_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS mutation_fences (
    fence_id VARCHAR PRIMARY KEY,
    worktree_id VARCHAR NOT NULL,
    lease_id VARCHAR NOT NULL DEFAULT '',
    session_id VARCHAR NOT NULL DEFAULT '',
    generation BIGINT NOT NULL,
    token_digest VARCHAR NOT NULL,
    before_snapshot_id VARCHAR NOT NULL DEFAULT '',
    before_tree_id VARCHAR NOT NULL DEFAULT '',
    status VARCHAR NOT NULL,
    registered_at VARCHAR NOT NULL,
    superseded_at VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS mutation_fences_active_uidx
    ON mutation_fences(worktree_id, generation);
CREATE INDEX IF NOT EXISTS mutation_fences_worktree_status_idx
    ON mutation_fences(worktree_id, status);

CREATE TABLE IF NOT EXISTS mutations (
    mutation_id VARCHAR PRIMARY KEY,
    mutation_set_id VARCHAR NOT NULL,
    task_id VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL DEFAULT '',
    plan_id VARCHAR NOT NULL DEFAULT '',
    operator_id VARCHAR NOT NULL DEFAULT '',
    provider_id VARCHAR NOT NULL DEFAULT '',
    daemon_id VARCHAR NOT NULL DEFAULT '',
    session_id VARCHAR NOT NULL DEFAULT '',
    worktree_id VARCHAR NOT NULL,
    lease_id VARCHAR NOT NULL DEFAULT '',
    fence_id VARCHAR NOT NULL,
    fence_generation BIGINT NOT NULL,
    before_snapshot_id VARCHAR NOT NULL,
    after_snapshot_id VARCHAR NOT NULL DEFAULT '',
    before_tree_id VARCHAR NOT NULL DEFAULT '',
    after_tree_id VARCHAR NOT NULL DEFAULT '',
    repository_id VARCHAR NOT NULL DEFAULT '',
    status VARCHAR NOT NULL,
    disposition VARCHAR NOT NULL,
    semantic_mutation_id VARCHAR NOT NULL,
    structural_identity VARCHAR NOT NULL,
    declared_effects_json VARCHAR NOT NULL,
    validation_outcome VARCHAR NOT NULL DEFAULT '',
    proof_outcome VARCHAR NOT NULL DEFAULT '',
    merge_outcome VARCHAR NOT NULL DEFAULT '',
    rollback_outcome VARCHAR NOT NULL DEFAULT '',
    reason VARCHAR NOT NULL DEFAULT '',
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS mutations_task_idx
    ON mutations(task_id, attempt_id);
CREATE INDEX IF NOT EXISTS mutations_worktree_idx
    ON mutations(worktree_id, recorded_at);
CREATE INDEX IF NOT EXISTS mutations_status_idx
    ON mutations(status, disposition);
CREATE INDEX IF NOT EXISTS mutations_semantic_idx
    ON mutations(semantic_mutation_id);

CREATE TABLE IF NOT EXISTS mutation_files (
    mutation_file_id VARCHAR PRIMARY KEY,
    mutation_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    prior_path VARCHAR NOT NULL DEFAULT '',
    change_kind VARCHAR NOT NULL,
    before_blob_id VARCHAR NOT NULL DEFAULT '',
    after_blob_id VARCHAR NOT NULL DEFAULT '',
    before_content_digest VARCHAR NOT NULL DEFAULT '',
    after_content_digest VARCHAR NOT NULL DEFAULT '',
    before_structural_id VARCHAR NOT NULL DEFAULT '',
    after_structural_id VARCHAR NOT NULL DEFAULT '',
    language VARCHAR NOT NULL DEFAULT '',
    byte_delta BIGINT NOT NULL DEFAULT 0,
    semantic_changed INTEGER NOT NULL DEFAULT 0,
    formatting_only INTEGER NOT NULL DEFAULT 0,
    parse_status VARCHAR NOT NULL DEFAULT '',
    lineage_id VARCHAR NOT NULL DEFAULT '',
    disposition VARCHAR NOT NULL,
    reason VARCHAR NOT NULL DEFAULT '',
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS mutation_files_path_uidx
    ON mutation_files(mutation_id, path);
CREATE INDEX IF NOT EXISTS mutation_files_mutation_idx
    ON mutation_files(mutation_id);

CREATE TABLE IF NOT EXISTS mutation_hunks (
    hunk_id VARCHAR PRIMARY KEY,
    mutation_id VARCHAR NOT NULL,
    mutation_file_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    hunk_index BIGINT NOT NULL,
    old_start BIGINT NOT NULL,
    old_count BIGINT NOT NULL,
    new_start BIGINT NOT NULL,
    new_count BIGINT NOT NULL,
    header VARCHAR NOT NULL DEFAULT '',
    lines_json VARCHAR NOT NULL,
    content_digest VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS mutation_hunks_file_idx
    ON mutation_hunks(mutation_file_id, hunk_index);

CREATE TABLE IF NOT EXISTS ast_mutations (
    ast_mutation_id VARCHAR PRIMARY KEY,
    mutation_id VARCHAR NOT NULL,
    mutation_file_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    before_record_id VARCHAR NOT NULL DEFAULT '',
    after_record_id VARCHAR NOT NULL DEFAULT '',
    before_structural_id VARCHAR NOT NULL DEFAULT '',
    after_structural_id VARCHAR NOT NULL DEFAULT '',
    edit_script_json VARCHAR NOT NULL,
    symbols_added_json VARCHAR NOT NULL,
    symbols_removed_json VARCHAR NOT NULL,
    symbols_changed_json VARCHAR NOT NULL,
    parse_status VARCHAR NOT NULL,
    semantic_changed INTEGER NOT NULL DEFAULT 0,
    recorded_at VARCHAR NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ast_mutations_file_uidx
    ON ast_mutations(mutation_id, mutation_file_id);
CREATE INDEX IF NOT EXISTS ast_mutations_mutation_idx
    ON ast_mutations(mutation_id);

CREATE TABLE IF NOT EXISTS mutation_lineages (
    lineage_id VARCHAR PRIMARY KEY,
    mutation_id VARCHAR NOT NULL,
    mutation_file_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    before_content_digest VARCHAR NOT NULL DEFAULT '',
    after_content_digest VARCHAR NOT NULL DEFAULT '',
    before_structural_id VARCHAR NOT NULL DEFAULT '',
    after_structural_id VARCHAR NOT NULL DEFAULT '',
    ast_mutation_id VARCHAR NOT NULL DEFAULT '',
    hunk_count BIGINT NOT NULL DEFAULT 0,
    byte_changed INTEGER NOT NULL DEFAULT 0,
    semantic_changed INTEGER NOT NULL DEFAULT 0,
    disposition VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS mutation_lineages_file_uidx
    ON mutation_lineages(mutation_id, mutation_file_id);
CREATE INDEX IF NOT EXISTS mutation_lineages_mutation_idx
    ON mutation_lineages(mutation_id);

CREATE TABLE IF NOT EXISTS mutation_quarantine (
    quarantine_id VARCHAR PRIMARY KEY,
    mutation_id VARCHAR NOT NULL DEFAULT '',
    worktree_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL DEFAULT '',
    reason VARCHAR NOT NULL,
    fence_id VARCHAR NOT NULL DEFAULT '',
    before_snapshot_id VARCHAR NOT NULL DEFAULT '',
    after_snapshot_id VARCHAR NOT NULL DEFAULT '',
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS mutation_quarantine_worktree_idx
    ON mutation_quarantine(worktree_id, recorded_at);

CREATE TABLE IF NOT EXISTS mutation_rollbacks (
    rollback_id VARCHAR PRIMARY KEY,
    mutation_id VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    verified INTEGER NOT NULL DEFAULT 0,
    expected_digests_json VARCHAR NOT NULL,
    observed_digests_json VARCHAR NOT NULL,
    mismatch_json VARCHAR NOT NULL,
    reason VARCHAR NOT NULL DEFAULT '',
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS mutation_rollbacks_mutation_idx
    ON mutation_rollbacks(mutation_id);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MutationLedgerError(RuntimeError):
    """Base error for mutation ledger failures."""


class MutationLedgerNotOpenError(MutationLedgerError):
    """Operation requires an open ledger."""


class MutationLedgerIntegrityError(MutationLedgerError, ValueError):
    """Identity, path, fence, or payload integrity failure."""


class MutationLedgerBoundsError(MutationLedgerError, ValueError):
    """A resource or payload bound was exceeded."""


class MutationLedgerConflictError(MutationLedgerError):
    """Duplicate identity with a conflicting payload."""


class MutationLedgerAdmissionError(MutationLedgerError):
    """Mutation could not be admitted as accepted."""


class DuckDBUnavailableError(MutationLedgerError):
    """Optional DuckDB dependency is not installed."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class MutationStatus(str, Enum):
    """Lifecycle status of one recorded mutation set."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"
    NO_OP = "no_op"
    PARTIAL = "partial"
    ROLLED_BACK = "rolled_back"


class MutationDisposition(str, Enum):
    """Admission disposition for a mutation set or file."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"
    NO_OP = "no_op"
    FORMATTING_ONLY = "formatting_only"
    PARSE_FAILED = "parse_failed"
    STALE_FENCE = "stale_fence"
    SNAPSHOT_MISMATCH = "snapshot_mismatch"
    MISSING_LINEAGE = "missing_lineage"
    PARTIAL_WRITE = "partial_write"


class FileChangeKind(str, Enum):
    """Per-path change vocabulary."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    NO_OP = "no_op"
    FORMATTING_ONLY = "formatting_only"
    PARSE_FAILED = "parse_failed"


class FenceStatus(str, Enum):
    """Fence lifecycle."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RELEASED = "released"


class ParseOutcome(str, Enum):
    """Bounded parse outcome for a mutation path."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    NOT_APPLICABLE = "not_applicable"
    BOTH_EMPTY = "both_empty"


class RollbackStatus(str, Enum):
    """Rollback recording outcome."""

    VERIFIED = "verified"
    FAILED = "failed"
    PENDING = "pending"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def duckdb_available() -> bool:
    """Return whether the optional duckdb package can be imported."""

    try:
        import duckdb  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _utc_iso(*, coarse: bool = False) -> str:
    now = datetime.now(timezone.utc)
    if coarse:
        return now.replace(microsecond=0).isoformat()
    return now.isoformat()


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if "\x00" in text:
        raise MutationLedgerIntegrityError(f"{name} contains NUL")
    if required and not text:
        raise MutationLedgerIntegrityError(f"{name} is required")
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MutationLedgerBoundsError(f"{name} must be a non-negative integer")
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise MutationLedgerIntegrityError(
            "values must be canonical JSON"
        ) from exc


def _identity(prefix: str, value: Any) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    return f"{prefix}:sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8", errors="surrogatepass"))


def _normalize_digest(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if ":" not in text:
        text = f"sha256:{text}"
    return text


def _repo_path(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    if not raw:
        raise MutationLedgerIntegrityError("repository path is required")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "\x00" in raw:
        raise MutationLedgerIntegrityError(
            f"repository path escapes its root: {value!r}"
        )
    normalized = path.as_posix()
    if len(normalized.encode("utf-8")) > MAX_PATH_BYTES:
        raise MutationLedgerBoundsError(
            f"path exceeds {MAX_PATH_BYTES} bytes: {normalized}"
        )
    return normalized


def _bounded_text(value: Any, maximum: int) -> str:
    text = str(value or "")
    encoded = text.encode("utf-8", "replace")
    if len(encoded) <= maximum:
        return text
    marker = "…[truncated]"
    budget = max(0, maximum - len(marker.encode("utf-8")))
    return encoded[:budget].decode("utf-8", "ignore") + marker


def _row_mapping(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return {str(key): row[key] for key in row}
    try:
        keys = list(row.keys())  # type: ignore[attr-defined]
    except Exception:
        return {}
    return {str(key): row[key] for key in keys}


def _split_sql_statements(sql_text: str) -> list[str]:
    statements: list[str] = []
    for chunk in str(sql_text).split(";"):
        statement = chunk.strip()
        if not statement:
            continue
        lines = [
            line
            for line in statement.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        if lines:
            statements.append("\n".join(lines))
    return statements


def _as_bytes(content: bytes | str | None) -> bytes | None:
    if content is None:
        return None
    if isinstance(content, bytes):
        payload = content
    else:
        payload = str(content).encode("utf-8", errors="surrogatepass")
    if len(payload) > MAX_SOURCE_BYTES:
        raise MutationLedgerBoundsError(
            f"source exceeds {MAX_SOURCE_BYTES} bytes"
        )
    return payload


def _as_text(content: bytes | str | None) -> str | None:
    if content is None:
        return None
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="surrogatepass")
    return str(content)


def language_for_path(path: str) -> str:
    """Return a language tag for a repository-relative path."""

    normalized = _repo_path(path)
    suffix = PurePosixPath(normalized).suffix.casefold()
    return _LANGUAGE_BY_SUFFIX.get(suffix, "")


def content_digest_of(content: bytes | str | None) -> str:
    """Return a sha256 digest for content, or empty when content is absent."""

    payload = _as_bytes(content)
    if payload is None:
        return ""
    return _sha256_bytes(payload)


def structural_identity_of(
    content: bytes | str | None,
    *,
    language: str = "",
    path: str = "",
) -> tuple[str, ParseOutcome, ASTBlobRecord | None]:
    """Return structural identity independent of line-number attributes.

    Structural identity is derived from the AST dump (no location attributes)
    plus symbol hashes.  Pure line-number or whitespace formatting churn that
    does not change the AST therefore yields an identical structural id.
    """

    text = _as_text(content)
    if text is None:
        return "", ParseOutcome.NOT_APPLICABLE, None
    lang = str(language or (language_for_path(path) if path else "")).strip()
    if not text.strip():
        empty_id = _identity(
            "structural",
            {"language": lang or "unknown", "empty": True, "digest": _sha256_text(text)},
        )
        return empty_id, ParseOutcome.BOTH_EMPTY, None
    if lang and lang not in _SUPPORTED_PARSE_LANGUAGES:
        # Unsupported languages still get a content-bound structural id so
        # byte-identical unsupported files do not invent new semantics, but
        # they cannot claim AST lineage success.
        digest = _sha256_text(text)
        return (
            _identity(
                "structural",
                {"language": lang, "unsupported": True, "digest": digest},
            ),
            ParseOutcome.UNSUPPORTED,
            None,
        )
    if not lang or lang == "python":
        record = build_python_ast_blob_record(text)
        if record.parse_error:
            # Failed parse: structural id still content-bound so the lineage
            # can quarantine without inventing AST edits.
            return (
                _identity(
                    "structural",
                    {
                        "language": "python",
                        "parse_failed": True,
                        "digest": record.source_sha256,
                        "error": record.parse_error,
                    },
                ),
                ParseOutcome.FAILED,
                record,
            )
        # Exclude line numbers from structural identity deliberately.
        payload = {
            "language": "python",
            "qualified_symbols": list(record.qualified_symbols),
            "imports": list(record.imports),
            "calls": list(record.calls),
            "state_transitions": list(record.state_transitions),
            "interfaces": list(record.interfaces),
            "symbol_hashes": dict(record.symbol_hashes),
            "record_schema_version": record.record_schema_version,
        }
        # Module-level structural dump without lineno/col_offset.
        try:
            tree = ast.parse(text)
            module_dump = ast.dump(
                tree, annotate_fields=True, include_attributes=False
            )
        except (SyntaxError, ValueError):
            module_dump = ""
        payload["module_dump"] = module_dump
        return _identity("structural", payload), ParseOutcome.SUCCEEDED, record
    digest = _sha256_text(text)
    return (
        _identity("structural", {"language": lang or "unknown", "digest": digest}),
        ParseOutcome.UNSUPPORTED,
        None,
    )


def _compute_hunks(
    before_text: str | None,
    after_text: str | None,
    *,
    path: str,
) -> list[dict[str, Any]]:
    """Compute bounded unified-diff style hunks without retaining full bodies."""

    import difflib

    before_lines = (before_text or "").splitlines(keepends=True)
    after_lines = (after_text or "").splitlines(keepends=True)
    if before_lines == after_lines:
        return []

    matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines)
    hunks: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if len(hunks) >= MAX_HUNKS_PER_FILE:
            break
        old_slice = before_lines[i1:i2]
        new_slice = after_lines[j1:j2]
        # Bound retained line previews.
        preview_old = old_slice[:MAX_HUNK_LINES]
        preview_new = new_slice[:MAX_HUNK_LINES]
        lines: list[str] = []
        for line in preview_old:
            prefix = "-" if tag in {"replace", "delete"} else " "
            lines.append(prefix + line.rstrip("\n"))
        for line in preview_new:
            prefix = "+" if tag in {"replace", "insert"} else " "
            lines.append(prefix + line.rstrip("\n"))
        header = f"@@ -{i1 + 1},{i2 - i1} +{j1 + 1},{j2 - j1} @@"
        hunks.append(
            {
                "hunk_index": len(hunks),
                "path": path,
                "old_start": i1 + 1,
                "old_count": i2 - i1,
                "new_start": j1 + 1,
                "new_count": j2 - j1,
                "header": header,
                "tag": tag,
                "lines": lines,
                "truncated": len(old_slice) > MAX_HUNK_LINES
                or len(new_slice) > MAX_HUNK_LINES,
            }
        )
    return hunks


def _ast_edit_script(
    before_record: ASTBlobRecord | None,
    after_record: ASTBlobRecord | None,
    *,
    before_structural_id: str,
    after_structural_id: str,
    change_kind: FileChangeKind,
) -> dict[str, Any]:
    """Build a bounded AST edit script from before/after blob records."""

    before_symbols = set(before_record.qualified_symbols) if before_record else set()
    after_symbols = set(after_record.qualified_symbols) if after_record else set()
    before_hashes = dict(before_record.symbol_hashes) if before_record else {}
    after_hashes = dict(after_record.symbol_hashes) if after_record else {}

    added = sorted(after_symbols - before_symbols)
    removed = sorted(before_symbols - after_symbols)
    common = before_symbols & after_symbols
    changed = sorted(
        name
        for name in common
        if before_hashes.get(name) != after_hashes.get(name)
    )
    unchanged = sorted(
        name
        for name in common
        if before_hashes.get(name) == after_hashes.get(name)
    )

    ops: list[dict[str, Any]] = []
    if change_kind is FileChangeKind.ADDED:
        ops.append({"op": "add_file", "symbols": added[:MAX_EDIT_OPS]})
    elif change_kind is FileChangeKind.DELETED:
        ops.append({"op": "delete_file", "symbols": removed[:MAX_EDIT_OPS]})
    elif change_kind is FileChangeKind.RENAMED:
        ops.append(
            {
                "op": "rename_file",
                "semantic_changed": before_structural_id != after_structural_id,
            }
        )
    for name in added[:MAX_EDIT_OPS]:
        ops.append(
            {
                "op": "add_symbol",
                "symbol": name,
                "after_hash": after_hashes.get(name, ""),
            }
        )
    for name in removed[:MAX_EDIT_OPS]:
        ops.append(
            {
                "op": "remove_symbol",
                "symbol": name,
                "before_hash": before_hashes.get(name, ""),
            }
        )
    for name in changed[:MAX_EDIT_OPS]:
        ops.append(
            {
                "op": "replace_symbol",
                "symbol": name,
                "before_hash": before_hashes.get(name, ""),
                "after_hash": after_hashes.get(name, ""),
            }
        )

    before_imports = set(before_record.imports) if before_record else set()
    after_imports = set(after_record.imports) if after_record else set()
    for item in sorted(after_imports - before_imports)[:MAX_EDIT_OPS]:
        ops.append({"op": "add_import", "import": item})
    for item in sorted(before_imports - after_imports)[:MAX_EDIT_OPS]:
        ops.append({"op": "remove_import", "import": item})

    if (
        before_structural_id
        and after_structural_id
        and before_structural_id == after_structural_id
        and not ops
    ):
        ops.append({"op": "identity", "reason": "structural_unchanged"})

    semantic_changed = before_structural_id != after_structural_id
    return {
        "schema": "ast-edit-script@1",
        "ops": ops[:MAX_EDIT_OPS],
        "symbols_added": added,
        "symbols_removed": removed,
        "symbols_changed": changed,
        "symbols_unchanged": unchanged,
        "semantic_changed": semantic_changed,
        "before_structural_id": before_structural_id,
        "after_structural_id": after_structural_id,
    }


def semantic_mutation_identity(
    file_entries: Sequence[Mapping[str, Any]],
) -> str:
    """Compute semantic mutation identity independent of line-number churn.

    Two mutations with identical per-path structural before/after identities
    and change kinds share one semantic mutation id even when textual line
    numbers or whitespace formatting differ.
    """

    members: list[dict[str, Any]] = []
    for entry in file_entries:
        path = str(entry.get("path") or "").strip()
        if not path:
            continue
        members.append(
            {
                "path": path,
                "prior_path": str(entry.get("prior_path") or ""),
                "change_kind": str(entry.get("change_kind") or ""),
                "before_structural_id": str(
                    entry.get("before_structural_id") or ""
                ),
                "after_structural_id": str(
                    entry.get("after_structural_id") or ""
                ),
                "semantic_changed": bool(entry.get("semantic_changed")),
            }
        )
    members.sort(key=lambda item: (item["path"], item["prior_path"]))
    # Formatting-only / no-op members do not forge a distinct semantic id.
    semantic_members = [
        item
        for item in members
        if item["semantic_changed"]
        or item["change_kind"]
        in {
            FileChangeKind.ADDED.value,
            FileChangeKind.DELETED.value,
            FileChangeKind.RENAMED.value,
            FileChangeKind.MODIFIED.value,
        }
        and item["before_structural_id"] != item["after_structural_id"]
    ]
    if not semantic_members:
        return _identity("semantic-mutation", {"empty": True, "members": members})
    return _identity("semantic-mutation", {"members": semantic_members})


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MutationFileSpec:
    """One path's before/after content for mutation admission."""

    path: str
    before_content: bytes | str | None = None
    after_content: bytes | str | None = None
    before_content_digest: str = ""
    after_content_digest: str = ""
    prior_path: str = ""
    language: str = ""
    partial: bool = False
    parse_failed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _repo_path(self.path))
        prior = str(self.prior_path or "").strip()
        if prior:
            object.__setattr__(self, "prior_path", _repo_path(prior))
        else:
            object.__setattr__(self, "prior_path", "")
        before_payload = _as_bytes(self.before_content)
        after_payload = _as_bytes(self.after_content)
        before_digest = _normalize_digest(self.before_content_digest)
        after_digest = _normalize_digest(self.after_content_digest)
        if before_payload is not None:
            actual = _sha256_bytes(before_payload)
            if before_digest and before_digest != actual:
                raise MutationLedgerIntegrityError(
                    f"before content digest mismatch for {self.path}"
                )
            before_digest = actual
        if after_payload is not None:
            actual = _sha256_bytes(after_payload)
            if after_digest and after_digest != actual:
                raise MutationLedgerIntegrityError(
                    f"after content digest mismatch for {self.path}"
                )
            after_digest = actual
        object.__setattr__(self, "before_content", before_payload)
        object.__setattr__(self, "after_content", after_payload)
        object.__setattr__(self, "before_content_digest", before_digest)
        object.__setattr__(self, "after_content_digest", after_digest)
        language = str(self.language or language_for_path(self.path)).strip()
        object.__setattr__(self, "language", language)

    @property
    def before_text(self) -> str | None:
        return _as_text(self.before_content)

    @property
    def after_text(self) -> str | None:
        return _as_text(self.after_content)

    @property
    def byte_changed(self) -> bool:
        return self.before_content_digest != self.after_content_digest


@dataclass(frozen=True)
class MutationContext:
    """Binding identities for one mutation admission attempt."""

    task_id: str
    worktree_id: str
    fence_id: str
    before_snapshot_id: str
    attempt_id: str = ""
    plan_id: str = ""
    operator_id: str = ""
    provider_id: str = ""
    daemon_id: str = ""
    session_id: str = ""
    lease_id: str = ""
    repository_id: str = ""
    before_tree_id: str = ""
    after_tree_id: str = ""
    after_snapshot_id: str = ""
    declared_effects: Mapping[str, Any] | Sequence[Any] | None = None
    validation_outcome: str = ""
    proof_outcome: str = ""
    merge_outcome: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id")
        )
        object.__setattr__(self, "fence_id", _text(self.fence_id, "fence_id"))
        object.__setattr__(
            self,
            "before_snapshot_id",
            _text(self.before_snapshot_id, "before_snapshot_id"),
        )
        for name in (
            "attempt_id",
            "plan_id",
            "operator_id",
            "provider_id",
            "daemon_id",
            "session_id",
            "lease_id",
            "repository_id",
            "before_tree_id",
            "after_tree_id",
            "after_snapshot_id",
            "validation_outcome",
            "proof_outcome",
            "merge_outcome",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )


@dataclass(frozen=True)
class MutationFence:
    """Active or historical mutation fence for a worktree."""

    fence_id: str
    worktree_id: str
    generation: int
    token_digest: str
    status: FenceStatus | str
    lease_id: str = ""
    session_id: str = ""
    before_snapshot_id: str = ""
    before_tree_id: str = ""
    registered_at: str = ""
    superseded_at: str = ""
    schema: str = FENCE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id")
        )
        object.__setattr__(
            self, "generation", _nonneg_int(int(self.generation), "generation")
        )
        if self.generation <= 0:
            raise MutationLedgerBoundsError("generation must be positive")
        token = _normalize_digest(self.token_digest)
        if not token.startswith("sha256:") or len(token) != len("sha256:") + 64:
            raise MutationLedgerIntegrityError(
                "token_digest must be sha256:<64-hex>"
            )
        object.__setattr__(self, "token_digest", token)
        object.__setattr__(self, "status", FenceStatus(self.status))
        for name in (
            "lease_id",
            "session_id",
            "before_snapshot_id",
            "before_tree_id",
            "superseded_at",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        object.__setattr__(
            self,
            "registered_at",
            _text(self.registered_at or _utc_iso(), "registered_at"),
        )
        computed = _identity(
            "mutation-fence",
            {
                "schema": self.schema,
                "worktree_id": self.worktree_id,
                "generation": self.generation,
                "token_digest": self.token_digest,
                "lease_id": self.lease_id,
                "session_id": self.session_id,
            },
        )
        claimed = str(self.fence_id or "").strip()
        if claimed and claimed != computed:
            raise MutationLedgerIntegrityError(
                "fence identity does not match payload"
            )
        object.__setattr__(self, "fence_id", claimed or computed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "fence_id": self.fence_id,
            "worktree_id": self.worktree_id,
            "generation": self.generation,
            "token_digest": self.token_digest,
            "status": self.status.value
            if isinstance(self.status, FenceStatus)
            else str(self.status),
            "lease_id": self.lease_id,
            "session_id": self.session_id,
            "before_snapshot_id": self.before_snapshot_id,
            "before_tree_id": self.before_tree_id,
            "registered_at": self.registered_at,
            "superseded_at": self.superseded_at,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class MutationHunk:
    """One textual hunk for a mutation file."""

    hunk_id: str
    mutation_id: str
    mutation_file_id: str
    path: str
    hunk_index: int
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    header: str = ""
    lines: tuple[str, ...] = ()
    content_digest: str = ""
    schema: str = MUTATION_HUNK_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "hunk_id": self.hunk_id,
            "mutation_id": self.mutation_id,
            "mutation_file_id": self.mutation_file_id,
            "path": self.path,
            "hunk_index": self.hunk_index,
            "old_start": self.old_start,
            "old_count": self.old_count,
            "new_start": self.new_start,
            "new_count": self.new_count,
            "header": self.header,
            "lines": list(self.lines),
            "content_digest": self.content_digest,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class ASTMutation:
    """AST-level edit lineage for one mutated path.

    Interface: ``ASTMutation@1``.
    """

    ast_mutation_id: str
    mutation_id: str
    mutation_file_id: str
    path: str
    edit_script: Mapping[str, Any]
    before_record_id: str = ""
    after_record_id: str = ""
    before_structural_id: str = ""
    after_structural_id: str = ""
    symbols_added: tuple[str, ...] = ()
    symbols_removed: tuple[str, ...] = ()
    symbols_changed: tuple[str, ...] = ()
    parse_status: ParseOutcome | str = ParseOutcome.SUCCEEDED
    semantic_changed: bool = False
    recorded_at: str = ""
    schema: str = AST_MUTATION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _repo_path(self.path))
        object.__setattr__(
            self, "mutation_id", _text(self.mutation_id, "mutation_id")
        )
        object.__setattr__(
            self,
            "mutation_file_id",
            _text(self.mutation_file_id, "mutation_file_id"),
        )
        object.__setattr__(
            self, "parse_status", ParseOutcome(self.parse_status)
        )
        object.__setattr__(
            self,
            "recorded_at",
            _text(self.recorded_at or _utc_iso(), "recorded_at"),
        )
        script = dict(self.edit_script or {})
        object.__setattr__(self, "edit_script", MappingProxyType(script))
        object.__setattr__(
            self, "symbols_added", tuple(str(x) for x in self.symbols_added)
        )
        object.__setattr__(
            self, "symbols_removed", tuple(str(x) for x in self.symbols_removed)
        )
        object.__setattr__(
            self, "symbols_changed", tuple(str(x) for x in self.symbols_changed)
        )
        computed = _identity(
            "ast-mutation",
            {
                "schema": self.schema,
                "mutation_id": self.mutation_id,
                "mutation_file_id": self.mutation_file_id,
                "path": self.path,
                "before_structural_id": self.before_structural_id,
                "after_structural_id": self.after_structural_id,
                "edit_script": script,
            },
        )
        claimed = str(self.ast_mutation_id or "").strip()
        if claimed and claimed != computed:
            raise MutationLedgerIntegrityError(
                "AST mutation identity does not match payload"
            )
        object.__setattr__(self, "ast_mutation_id", claimed or computed)

    @property
    def interface(self) -> str:
        return AST_MUTATION_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": AST_MUTATION_INTERFACE,
            "ast_mutation_id": self.ast_mutation_id,
            "mutation_id": self.mutation_id,
            "mutation_file_id": self.mutation_file_id,
            "path": self.path,
            "before_record_id": self.before_record_id,
            "after_record_id": self.after_record_id,
            "before_structural_id": self.before_structural_id,
            "after_structural_id": self.after_structural_id,
            "edit_script": dict(self.edit_script),
            "symbols_added": list(self.symbols_added),
            "symbols_removed": list(self.symbols_removed),
            "symbols_changed": list(self.symbols_changed),
            "parse_status": self.parse_status.value
            if isinstance(self.parse_status, ParseOutcome)
            else str(self.parse_status),
            "semantic_changed": bool(self.semantic_changed),
            "recorded_at": self.recorded_at,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class MutationFile:
    """One path's before/after mutation record.

    Interface: ``MutationFile@1``.
    """

    mutation_file_id: str
    mutation_id: str
    path: str
    change_kind: FileChangeKind | str
    disposition: MutationDisposition | str
    prior_path: str = ""
    before_blob_id: str = ""
    after_blob_id: str = ""
    before_content_digest: str = ""
    after_content_digest: str = ""
    before_structural_id: str = ""
    after_structural_id: str = ""
    language: str = ""
    byte_delta: int = 0
    semantic_changed: bool = False
    formatting_only: bool = False
    parse_status: ParseOutcome | str = ParseOutcome.NOT_APPLICABLE
    lineage_id: str = ""
    reason: str = ""
    recorded_at: str = ""
    schema: str = MUTATION_FILE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _repo_path(self.path))
        if self.prior_path:
            object.__setattr__(self, "prior_path", _repo_path(self.prior_path))
        object.__setattr__(
            self, "mutation_id", _text(self.mutation_id, "mutation_id")
        )
        object.__setattr__(
            self, "change_kind", FileChangeKind(self.change_kind)
        )
        object.__setattr__(
            self, "disposition", MutationDisposition(self.disposition)
        )
        object.__setattr__(
            self, "parse_status", ParseOutcome(self.parse_status)
        )
        object.__setattr__(
            self,
            "before_content_digest",
            _normalize_digest(self.before_content_digest),
        )
        object.__setattr__(
            self,
            "after_content_digest",
            _normalize_digest(self.after_content_digest),
        )
        object.__setattr__(
            self,
            "recorded_at",
            _text(self.recorded_at or _utc_iso(), "recorded_at"),
        )
        object.__setattr__(
            self, "reason", _bounded_text(self.reason, MAX_REASON_BYTES)
        )
        computed = _identity(
            "mutation-file",
            {
                "schema": self.schema,
                "mutation_id": self.mutation_id,
                "path": self.path,
                "prior_path": self.prior_path,
                "before_content_digest": self.before_content_digest,
                "after_content_digest": self.after_content_digest,
                "change_kind": self.change_kind.value
                if isinstance(self.change_kind, FileChangeKind)
                else str(self.change_kind),
            },
        )
        claimed = str(self.mutation_file_id or "").strip()
        if claimed and claimed != computed:
            raise MutationLedgerIntegrityError(
                "mutation file identity does not match payload"
            )
        object.__setattr__(self, "mutation_file_id", claimed or computed)

    @property
    def interface(self) -> str:
        return MUTATION_FILE_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": MUTATION_FILE_INTERFACE,
            "mutation_file_id": self.mutation_file_id,
            "mutation_id": self.mutation_id,
            "path": self.path,
            "prior_path": self.prior_path,
            "change_kind": self.change_kind.value
            if isinstance(self.change_kind, FileChangeKind)
            else str(self.change_kind),
            "disposition": self.disposition.value
            if isinstance(self.disposition, MutationDisposition)
            else str(self.disposition),
            "before_blob_id": self.before_blob_id,
            "after_blob_id": self.after_blob_id,
            "before_content_digest": self.before_content_digest,
            "after_content_digest": self.after_content_digest,
            "before_structural_id": self.before_structural_id,
            "after_structural_id": self.after_structural_id,
            "language": self.language,
            "byte_delta": self.byte_delta,
            "semantic_changed": bool(self.semantic_changed),
            "formatting_only": bool(self.formatting_only),
            "parse_status": self.parse_status.value
            if isinstance(self.parse_status, ParseOutcome)
            else str(self.parse_status),
            "lineage_id": self.lineage_id,
            "reason": self.reason,
            "recorded_at": self.recorded_at,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class MutationLineage:
    """Exactly-one lineage binding for an admitted byte change."""

    lineage_id: str
    mutation_id: str
    mutation_file_id: str
    path: str
    before_content_digest: str = ""
    after_content_digest: str = ""
    before_structural_id: str = ""
    after_structural_id: str = ""
    ast_mutation_id: str = ""
    hunk_count: int = 0
    byte_changed: bool = False
    semantic_changed: bool = False
    disposition: MutationDisposition | str = MutationDisposition.ACCEPTED
    recorded_at: str = ""
    schema: str = MUTATION_LINEAGE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _repo_path(self.path))
        object.__setattr__(
            self, "disposition", MutationDisposition(self.disposition)
        )
        object.__setattr__(
            self,
            "recorded_at",
            _text(self.recorded_at or _utc_iso(), "recorded_at"),
        )
        computed = _identity(
            "mutation-lineage",
            {
                "schema": self.schema,
                "mutation_id": self.mutation_id,
                "mutation_file_id": self.mutation_file_id,
                "path": self.path,
                "before_content_digest": self.before_content_digest,
                "after_content_digest": self.after_content_digest,
                "ast_mutation_id": self.ast_mutation_id,
            },
        )
        claimed = str(self.lineage_id or "").strip()
        if claimed and claimed != computed:
            raise MutationLedgerIntegrityError(
                "mutation lineage identity does not match payload"
            )
        object.__setattr__(self, "lineage_id", claimed or computed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "lineage_id": self.lineage_id,
            "mutation_id": self.mutation_id,
            "mutation_file_id": self.mutation_file_id,
            "path": self.path,
            "before_content_digest": self.before_content_digest,
            "after_content_digest": self.after_content_digest,
            "before_structural_id": self.before_structural_id,
            "after_structural_id": self.after_structural_id,
            "ast_mutation_id": self.ast_mutation_id,
            "hunk_count": self.hunk_count,
            "byte_changed": bool(self.byte_changed),
            "semantic_changed": bool(self.semantic_changed),
            "disposition": self.disposition.value
            if isinstance(self.disposition, MutationDisposition)
            else str(self.disposition),
            "recorded_at": self.recorded_at,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class MutationSet:
    """One admitted (or rejected/quarantined) mutation set.

    Interface: ``MutationSet@1``.
    """

    mutation_id: str
    mutation_set_id: str
    task_id: str
    worktree_id: str
    fence_id: str
    fence_generation: int
    before_snapshot_id: str
    status: MutationStatus | str
    disposition: MutationDisposition | str
    semantic_mutation_id: str
    structural_identity: str
    attempt_id: str = ""
    plan_id: str = ""
    operator_id: str = ""
    provider_id: str = ""
    daemon_id: str = ""
    session_id: str = ""
    lease_id: str = ""
    repository_id: str = ""
    after_snapshot_id: str = ""
    before_tree_id: str = ""
    after_tree_id: str = ""
    declared_effects: Mapping[str, Any] = field(default_factory=dict)
    validation_outcome: str = ""
    proof_outcome: str = ""
    merge_outcome: str = ""
    rollback_outcome: str = ""
    reason: str = ""
    recorded_at: str = ""
    file_count: int = 0
    lineage_count: int = 0
    schema: str = MUTATION_SET_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id")
        )
        object.__setattr__(self, "fence_id", _text(self.fence_id, "fence_id"))
        object.__setattr__(
            self,
            "before_snapshot_id",
            _text(self.before_snapshot_id, "before_snapshot_id"),
        )
        object.__setattr__(
            self,
            "fence_generation",
            _nonneg_int(int(self.fence_generation), "fence_generation"),
        )
        object.__setattr__(self, "status", MutationStatus(self.status))
        object.__setattr__(
            self, "disposition", MutationDisposition(self.disposition)
        )
        object.__setattr__(
            self,
            "semantic_mutation_id",
            _text(self.semantic_mutation_id, "semantic_mutation_id"),
        )
        object.__setattr__(
            self,
            "structural_identity",
            _text(self.structural_identity, "structural_identity"),
        )
        object.__setattr__(
            self,
            "recorded_at",
            _text(self.recorded_at or _utc_iso(), "recorded_at"),
        )
        object.__setattr__(
            self, "reason", _bounded_text(self.reason, MAX_REASON_BYTES)
        )
        effects = dict(self.declared_effects or {})
        object.__setattr__(self, "declared_effects", MappingProxyType(effects))
        set_id = str(self.mutation_set_id or "").strip()
        if not set_id:
            set_id = _identity(
                "mutation-set",
                {
                    "schema": self.schema,
                    "task_id": self.task_id,
                    "attempt_id": self.attempt_id,
                    "worktree_id": self.worktree_id,
                    "fence_id": self.fence_id,
                    "before_snapshot_id": self.before_snapshot_id,
                    "after_snapshot_id": self.after_snapshot_id,
                    "semantic_mutation_id": self.semantic_mutation_id,
                },
            )
        object.__setattr__(self, "mutation_set_id", set_id)
        # Identity is immutable admission binding only. Status/rollback may
        # change later without forging a new mutation_id.
        computed = _identity(
            "mutation",
            {
                "schema": self.schema,
                "mutation_set_id": self.mutation_set_id,
                "task_id": self.task_id,
                "attempt_id": self.attempt_id,
                "plan_id": self.plan_id,
                "worktree_id": self.worktree_id,
                "fence_id": self.fence_id,
                "fence_generation": self.fence_generation,
                "before_snapshot_id": self.before_snapshot_id,
                "after_snapshot_id": self.after_snapshot_id,
                "semantic_mutation_id": self.semantic_mutation_id,
                "structural_identity": self.structural_identity,
                "recorded_at": self.recorded_at,
            },
        )
        claimed = str(self.mutation_id or "").strip()
        if claimed and claimed != computed:
            raise MutationLedgerIntegrityError(
                "mutation identity does not match payload"
            )
        object.__setattr__(self, "mutation_id", claimed or computed)

    @property
    def interface(self) -> str:
        return MUTATION_SET_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": MUTATION_SET_INTERFACE,
            "mutation_id": self.mutation_id,
            "mutation_set_id": self.mutation_set_id,
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "plan_id": self.plan_id,
            "operator_id": self.operator_id,
            "provider_id": self.provider_id,
            "daemon_id": self.daemon_id,
            "session_id": self.session_id,
            "worktree_id": self.worktree_id,
            "lease_id": self.lease_id,
            "fence_id": self.fence_id,
            "fence_generation": self.fence_generation,
            "before_snapshot_id": self.before_snapshot_id,
            "after_snapshot_id": self.after_snapshot_id,
            "before_tree_id": self.before_tree_id,
            "after_tree_id": self.after_tree_id,
            "repository_id": self.repository_id,
            "status": self.status.value
            if isinstance(self.status, MutationStatus)
            else str(self.status),
            "disposition": self.disposition.value
            if isinstance(self.disposition, MutationDisposition)
            else str(self.disposition),
            "semantic_mutation_id": self.semantic_mutation_id,
            "structural_identity": self.structural_identity,
            "declared_effects": dict(self.declared_effects),
            "validation_outcome": self.validation_outcome,
            "proof_outcome": self.proof_outcome,
            "merge_outcome": self.merge_outcome,
            "rollback_outcome": self.rollback_outcome,
            "reason": self.reason,
            "recorded_at": self.recorded_at,
            "file_count": self.file_count,
            "lineage_count": self.lineage_count,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class RollbackReceipt:
    """Independent verification of rollback restoration."""

    rollback_id: str
    mutation_id: str
    worktree_id: str
    status: RollbackStatus | str
    verified: bool
    expected_digests: Mapping[str, str]
    observed_digests: Mapping[str, str]
    mismatches: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    reason: str = ""
    recorded_at: str = ""
    schema: str = ROLLBACK_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mutation_id", _text(self.mutation_id, "mutation_id")
        )
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id")
        )
        object.__setattr__(self, "status", RollbackStatus(self.status))
        object.__setattr__(
            self,
            "expected_digests",
            MappingProxyType(
                {str(k): _normalize_digest(v) for k, v in dict(self.expected_digests).items()}
            ),
        )
        object.__setattr__(
            self,
            "observed_digests",
            MappingProxyType(
                {str(k): _normalize_digest(v) for k, v in dict(self.observed_digests).items()}
            ),
        )
        object.__setattr__(
            self,
            "mismatches",
            MappingProxyType(
                {
                    str(k): MappingProxyType({str(a): str(b) for a, b in dict(v).items()})
                    for k, v in dict(self.mismatches).items()
                }
            ),
        )
        object.__setattr__(
            self,
            "recorded_at",
            _text(self.recorded_at or _utc_iso(), "recorded_at"),
        )
        object.__setattr__(
            self, "reason", _bounded_text(self.reason, MAX_REASON_BYTES)
        )
        computed = _identity(
            "mutation-rollback",
            {
                "schema": self.schema,
                "mutation_id": self.mutation_id,
                "worktree_id": self.worktree_id,
                "status": self.status.value
                if isinstance(self.status, RollbackStatus)
                else str(self.status),
                "expected_digests": dict(self.expected_digests),
                "observed_digests": dict(self.observed_digests),
                "recorded_at": self.recorded_at,
            },
        )
        claimed = str(self.rollback_id or "").strip()
        if claimed and claimed != computed:
            raise MutationLedgerIntegrityError(
                "rollback identity does not match payload"
            )
        object.__setattr__(self, "rollback_id", claimed or computed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "rollback_id": self.rollback_id,
            "mutation_id": self.mutation_id,
            "worktree_id": self.worktree_id,
            "status": self.status.value
            if isinstance(self.status, RollbackStatus)
            else str(self.status),
            "verified": bool(self.verified),
            "expected_digests": dict(self.expected_digests),
            "observed_digests": dict(self.observed_digests),
            "mismatches": {
                key: dict(value) for key, value in self.mismatches.items()
            },
            "reason": self.reason,
            "recorded_at": self.recorded_at,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class MutationRecordResult:
    """Outcome of attempting to record one mutation set."""

    mutation: MutationSet
    files: tuple[MutationFile, ...]
    lineages: tuple[MutationLineage, ...]
    ast_mutations: tuple[ASTMutation, ...]
    hunks: tuple[MutationHunk, ...]
    admitted: bool
    quarantined: bool
    rejected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutation": self.mutation.to_dict(),
            "files": [item.to_dict() for item in self.files],
            "lineages": [item.to_dict() for item in self.lineages],
            "ast_mutations": [item.to_dict() for item in self.ast_mutations],
            "hunks": [item.to_dict() for item in self.hunks],
            "admitted": bool(self.admitted),
            "quarantined": bool(self.quarantined),
            "rejected": bool(self.rejected),
            "authority": AUTHORITY_CLASS,
        }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class MutationLedger:
    """Persist before/after mutations with AST edit lineage in DuckDB.

    Interface: ``MutationLedger@1``.
    """

    INTERFACE: Final[str] = MUTATION_LEDGER_INTERFACE
    SCHEMA: Final[str] = MUTATION_LEDGER_SCHEMA

    def __init__(
        self,
        database_path: Path | str,
        *,
        parser_id: str = DEFAULT_PARSER_ID,
        ledger_version: str = DEFAULT_LEDGER_VERSION,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for MutationLedger; install the optional "
                "duckdb dependency"
            )
        self._path = Path(database_path)
        self._parser_id = _text(parser_id or DEFAULT_PARSER_ID, "parser_id")
        self._ledger_version = _text(
            ledger_version or DEFAULT_LEDGER_VERSION, "ledger_version"
        )
        self._connection: Any | None = None
        self._lock = threading.RLock()
        self._closed = True

    # -- lifecycle -----------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def parser_id(self) -> str:
        return self._parser_id

    @property
    def ledger_version(self) -> str:
        return self._ledger_version

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    def open(self) -> "MutationLedger":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self._path)
            for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                connection.execute(statement)
            for key, value in (
                ("interface", MUTATION_LEDGER_INTERFACE),
                ("schema", MUTATION_LEDGER_SCHEMA),
                ("parser_id", self._parser_id),
                ("ledger_version", self._ledger_version),
                ("authority", AUTHORITY_CLASS),
            ):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO mutation_ledger_metadata(key, value)
                    VALUES (?, ?)
                    """,
                    [key, value],
                )
            self._connection = connection
            self._closed = False
            return self

    def close(self) -> None:
        with self._lock:
            connection = self._connection
            self._connection = None
            self._closed = True
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def __enter__(self) -> "MutationLedger":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require(self) -> Any:
        if not self.is_open or self._connection is None:
            raise MutationLedgerNotOpenError("MutationLedger is not open")
        return self._connection

    def _commit_if_idle(self, connection: Any) -> None:
        if getattr(connection, "in_transaction", False):
            return
        commit = getattr(connection, "commit", None)
        if callable(commit):
            try:
                commit()
            except Exception:
                pass

    def metadata(self) -> dict[str, Any]:
        connection = self._require()
        with self._lock:
            rows = connection.execute(
                "SELECT key, value FROM mutation_ledger_metadata"
            ).fetchall()
            meta = {str(row[0]): str(row[1]) for row in rows}
            meta["database_path"] = str(self._path)
            meta["is_open"] = True
            return meta

    # -- fences --------------------------------------------------------------

    def register_fence(
        self,
        *,
        worktree_id: str,
        token: str | bytes,
        generation: int | None = None,
        lease_id: str = "",
        session_id: str = "",
        before_snapshot_id: str = "",
        before_tree_id: str = "",
        supersede_active: bool = True,
    ) -> MutationFence:
        """Register a new active fence for a worktree."""

        connection = self._require()
        wt = _text(worktree_id, "worktree_id")
        if isinstance(token, bytes):
            token_digest = _sha256_bytes(token)
        else:
            raw = str(token or "").strip()
            if raw.startswith("sha256:") and len(raw) == len("sha256:") + 64:
                token_digest = raw
            else:
                token_digest = _sha256_text(raw)
        with self._lock:
            if generation is None:
                row = connection.execute(
                    """
                    SELECT COALESCE(MAX(generation), 0)
                    FROM mutation_fences
                    WHERE worktree_id = ?
                    """,
                    [wt],
                ).fetchone()
                generation = int(row[0] if row else 0) + 1
            generation = _nonneg_int(int(generation), "generation")
            if generation <= 0:
                raise MutationLedgerBoundsError("generation must be positive")
            if supersede_active:
                connection.execute(
                    """
                    UPDATE mutation_fences
                    SET status = ?, superseded_at = ?
                    WHERE worktree_id = ? AND status = ?
                    """,
                    [
                        FenceStatus.SUPERSEDED.value,
                        _utc_iso(),
                        wt,
                        FenceStatus.ACTIVE.value,
                    ],
                )
            fence = MutationFence(
                fence_id="",
                worktree_id=wt,
                generation=generation,
                token_digest=token_digest,
                status=FenceStatus.ACTIVE,
                lease_id=lease_id,
                session_id=session_id,
                before_snapshot_id=before_snapshot_id,
                before_tree_id=before_tree_id,
            )
            existing = connection.execute(
                "SELECT fence_id FROM mutation_fences WHERE fence_id = ?",
                [fence.fence_id],
            ).fetchone()
            if existing:
                raise MutationLedgerConflictError(
                    f"fence already exists: {fence.fence_id}"
                )
            connection.execute(
                """
                INSERT INTO mutation_fences(
                    fence_id, worktree_id, lease_id, session_id, generation,
                    token_digest, before_snapshot_id, before_tree_id, status,
                    registered_at, superseded_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    fence.fence_id,
                    fence.worktree_id,
                    fence.lease_id,
                    fence.session_id,
                    fence.generation,
                    fence.token_digest,
                    fence.before_snapshot_id,
                    fence.before_tree_id,
                    fence.status.value,
                    fence.registered_at,
                    fence.superseded_at,
                    _canonical_json(fence.to_dict()),
                ],
            )
            self._commit_if_idle(connection)
            return fence

    def get_active_fence(self, worktree_id: str) -> MutationFence | None:
        connection = self._require()
        wt = _text(worktree_id, "worktree_id")
        with self._lock:
            row = connection.execute(
                """
                SELECT fence_id, worktree_id, lease_id, session_id, generation,
                       token_digest, before_snapshot_id, before_tree_id, status,
                       registered_at, superseded_at
                FROM mutation_fences
                WHERE worktree_id = ? AND status = ?
                ORDER BY generation DESC
                LIMIT 1
                """,
                [wt, FenceStatus.ACTIVE.value],
            ).fetchone()
            if row is None:
                return None
            return self._fence_from_row(row)

    def get_fence(self, fence_id: str) -> MutationFence | None:
        connection = self._require()
        fid = _text(fence_id, "fence_id")
        with self._lock:
            row = connection.execute(
                """
                SELECT fence_id, worktree_id, lease_id, session_id, generation,
                       token_digest, before_snapshot_id, before_tree_id, status,
                       registered_at, superseded_at
                FROM mutation_fences
                WHERE fence_id = ?
                """,
                [fid],
            ).fetchone()
            if row is None:
                return None
            return self._fence_from_row(row)

    def supersede_fence(self, fence_id: str) -> MutationFence | None:
        connection = self._require()
        fid = _text(fence_id, "fence_id")
        with self._lock:
            fence = self.get_fence(fid)
            if fence is None:
                return None
            if fence.status is FenceStatus.ACTIVE:
                connection.execute(
                    """
                    UPDATE mutation_fences
                    SET status = ?, superseded_at = ?
                    WHERE fence_id = ?
                    """,
                    [
                        FenceStatus.SUPERSEDED.value,
                        _utc_iso(),
                        fid,
                    ],
                )
                self._commit_if_idle(connection)
                return self.get_fence(fid)
            return fence

    def _fence_from_row(self, row: Any) -> MutationFence:
        mapping = _row_mapping(row)
        if not mapping:
            # positional
            return MutationFence(
                fence_id=str(row[0]),
                worktree_id=str(row[1]),
                lease_id=str(row[2] or ""),
                session_id=str(row[3] or ""),
                generation=int(row[4]),
                token_digest=str(row[5]),
                before_snapshot_id=str(row[6] or ""),
                before_tree_id=str(row[7] or ""),
                status=str(row[8]),
                registered_at=str(row[9] or ""),
                superseded_at=str(row[10] or ""),
            )
        return MutationFence(
            fence_id=str(mapping["fence_id"]),
            worktree_id=str(mapping["worktree_id"]),
            lease_id=str(mapping.get("lease_id") or ""),
            session_id=str(mapping.get("session_id") or ""),
            generation=int(mapping["generation"]),
            token_digest=str(mapping["token_digest"]),
            before_snapshot_id=str(mapping.get("before_snapshot_id") or ""),
            before_tree_id=str(mapping.get("before_tree_id") or ""),
            status=str(mapping["status"]),
            registered_at=str(mapping.get("registered_at") or ""),
            superseded_at=str(mapping.get("superseded_at") or ""),
        )

    # -- mutation admission --------------------------------------------------

    def record_mutation(
        self,
        context: MutationContext,
        files: Sequence[MutationFileSpec],
        *,
        require_active_fence: bool = True,
    ) -> MutationRecordResult:
        """Record a mutation set with AST edit lineage (fail-closed).

        Admission rules:
        * Fence must be active (unless require_active_fence is False for
          explicit reject/quarantine recording of stale attempts).
        * Before snapshot must match the fence's bound before snapshot when
          the fence declares one.
        * Every byte-changing path must receive exactly one lineage or the
          set is rejected/quarantined (never accepted without lineage).
        * Partial writes quarantine the set.
        * Parse failures on changed Python paths quarantine rather than
          accept incomplete AST lineage.
        """

        connection = self._require()
        if not isinstance(context, MutationContext):
            raise MutationLedgerIntegrityError("MutationContext is required")
        file_list = list(files)
        if len(file_list) > MAX_FILES_PER_MUTATION:
            raise MutationLedgerBoundsError(
                f"mutation exceeds {MAX_FILES_PER_MUTATION} files"
            )

        with self._lock:
            fence = self.get_fence(context.fence_id)
            recorded_at = _utc_iso()
            reject_disposition: MutationDisposition | None = None
            reject_reason = ""
            reject_status = MutationStatus.REJECTED

            if fence is None:
                reject_disposition = MutationDisposition.STALE_FENCE
                reject_reason = "fence_not_found"
            elif require_active_fence and fence.status is not FenceStatus.ACTIVE:
                reject_disposition = MutationDisposition.STALE_FENCE
                reject_reason = f"fence_status:{fence.status.value}"
            elif fence.worktree_id != context.worktree_id:
                reject_disposition = MutationDisposition.STALE_FENCE
                reject_reason = "fence_worktree_mismatch"
            elif (
                fence.before_snapshot_id
                and fence.before_snapshot_id != context.before_snapshot_id
            ):
                reject_disposition = MutationDisposition.SNAPSHOT_MISMATCH
                reject_reason = "before_snapshot_mismatch"
                reject_status = MutationStatus.REJECTED

            fence_generation = fence.generation if fence is not None else 0

            analyzed = [
                self._analyze_file(spec, recorded_at=recorded_at)
                for spec in file_list
            ]

            # Partial writes force quarantine even when fence is valid.
            if any(item["partial"] for item in analyzed):
                if reject_disposition is None:
                    reject_disposition = MutationDisposition.PARTIAL_WRITE
                    reject_reason = "partial_write"
                    reject_status = MutationStatus.QUARANTINED

            # Parse failures on byte-changing supported paths quarantine.
            parse_failures = [
                item
                for item in analyzed
                if item["byte_changed"]
                and item["parse_status"] is ParseOutcome.FAILED
            ]
            if parse_failures and reject_disposition is None:
                reject_disposition = MutationDisposition.PARSE_FAILED
                reject_reason = "parse_failure:" + ",".join(
                    item["path"] for item in parse_failures[:8]
                )
                reject_status = MutationStatus.QUARANTINED

            # Missing lineage for content/path mutations is never accepted.
            missing_lineage = [
                item
                for item in analyzed
                if item["requires_lineage"] and not item["has_lineage"]
            ]
            if missing_lineage and reject_disposition is None:
                reject_disposition = MutationDisposition.MISSING_LINEAGE
                reject_reason = "missing_lineage:" + ",".join(
                    item["path"] for item in missing_lineage[:8]
                )
                reject_status = MutationStatus.QUARANTINED

            semantic_entries = [
                {
                    "path": item["path"],
                    "prior_path": item["prior_path"],
                    "change_kind": item["change_kind"].value,
                    "before_structural_id": item["before_structural_id"],
                    "after_structural_id": item["after_structural_id"],
                    "semantic_changed": item["semantic_changed"],
                }
                for item in analyzed
            ]
            semantic_id = semantic_mutation_identity(semantic_entries)
            structural_id = _identity(
                "mutation-structural",
                {
                    "members": [
                        {
                            "path": item["path"],
                            "prior_path": item["prior_path"],
                            "before": item["before_structural_id"],
                            "after": item["after_structural_id"],
                        }
                        for item in sorted(analyzed, key=lambda x: x["path"])
                    ]
                },
            )

            byte_changes = [item for item in analyzed if item["byte_changed"]]
            path_changes = [item for item in analyzed if item["path_changed"]]
            lineage_required = [
                item for item in analyzed if item["requires_lineage"]
            ]
            semantic_changes = [
                item for item in analyzed if item["semantic_changed"]
            ]
            formatting_only = (
                bool(byte_changes)
                and not semantic_changes
                and not path_changes
            )
            no_op = not lineage_required

            if reject_disposition is not None:
                status = reject_status
                disposition = reject_disposition
                reason = reject_reason
            elif no_op:
                status = MutationStatus.NO_OP
                disposition = MutationDisposition.NO_OP
                reason = "no_byte_change"
            elif formatting_only:
                status = MutationStatus.ACCEPTED
                disposition = MutationDisposition.FORMATTING_ONLY
                reason = "formatting_only"
            else:
                status = MutationStatus.ACCEPTED
                disposition = MutationDisposition.ACCEPTED
                reason = ""

            effects: dict[str, Any]
            if context.declared_effects is None:
                effects = {}
            elif isinstance(context.declared_effects, Mapping):
                effects = dict(context.declared_effects)
            else:
                effects = {"items": list(context.declared_effects)}

            # Provisional mutation identity uses recorded_at; build files with
            # a temporary mutation_id then re-bind after MutationSet creation.
            provisional_mutation_id = _identity(
                "mutation-provisional",
                {
                    "task_id": context.task_id,
                    "attempt_id": context.attempt_id,
                    "fence_id": context.fence_id,
                    "recorded_at": recorded_at,
                    "semantic_mutation_id": semantic_id,
                },
            )

            mutation_files: list[MutationFile] = []
            lineages: list[MutationLineage] = []
            ast_mutations: list[ASTMutation] = []
            hunks: list[MutationHunk] = []

            for item in analyzed:
                file_disposition = disposition
                file_reason = reason
                if item["partial"]:
                    file_disposition = MutationDisposition.PARTIAL_WRITE
                    file_reason = "partial_write"
                elif item["byte_changed"] and item["parse_status"] is ParseOutcome.FAILED:
                    file_disposition = MutationDisposition.PARSE_FAILED
                    file_reason = "parse_failed"
                elif item["formatting_only"]:
                    file_disposition = MutationDisposition.FORMATTING_ONLY
                    file_reason = "formatting_only"
                elif not item["requires_lineage"]:
                    file_disposition = MutationDisposition.NO_OP
                    file_reason = "no_byte_change"
                elif (
                    status is MutationStatus.ACCEPTED
                    and item["semantic_changed"]
                ):
                    file_disposition = MutationDisposition.ACCEPTED
                    file_reason = ""

                mutation_file = MutationFile(
                    mutation_file_id="",
                    mutation_id=provisional_mutation_id,
                    path=item["path"],
                    prior_path=item["prior_path"],
                    change_kind=item["change_kind"],
                    disposition=file_disposition,
                    before_blob_id=item["before_content_digest"],
                    after_blob_id=item["after_content_digest"],
                    before_content_digest=item["before_content_digest"],
                    after_content_digest=item["after_content_digest"],
                    before_structural_id=item["before_structural_id"],
                    after_structural_id=item["after_structural_id"],
                    language=item["language"],
                    byte_delta=item["byte_delta"],
                    semantic_changed=item["semantic_changed"],
                    formatting_only=item["formatting_only"],
                    parse_status=item["parse_status"],
                    reason=file_reason,
                    recorded_at=recorded_at,
                )

                edit_script = item["edit_script"]
                ast_mut = ASTMutation(
                    ast_mutation_id="",
                    mutation_id=provisional_mutation_id,
                    mutation_file_id=mutation_file.mutation_file_id,
                    path=item["path"],
                    edit_script=edit_script,
                    before_record_id=item["before_record_id"],
                    after_record_id=item["after_record_id"],
                    before_structural_id=item["before_structural_id"],
                    after_structural_id=item["after_structural_id"],
                    symbols_added=tuple(edit_script.get("symbols_added") or ()),
                    symbols_removed=tuple(
                        edit_script.get("symbols_removed") or ()
                    ),
                    symbols_changed=tuple(
                        edit_script.get("symbols_changed") or ()
                    ),
                    parse_status=item["parse_status"],
                    semantic_changed=item["semantic_changed"],
                    recorded_at=recorded_at,
                )

                file_hunks: list[MutationHunk] = []
                for hunk_data in item["hunks"]:
                    lines = tuple(str(line) for line in hunk_data.get("lines") or ())
                    content_digest = _sha256_text(
                        _canonical_json(
                            {
                                "header": hunk_data.get("header"),
                                "lines": list(lines),
                            }
                        )
                    )
                    hunk = MutationHunk(
                        hunk_id=_identity(
                            "mutation-hunk",
                            {
                                "mutation_file_id": mutation_file.mutation_file_id,
                                "hunk_index": hunk_data["hunk_index"],
                                "content_digest": content_digest,
                            },
                        ),
                        mutation_id=provisional_mutation_id,
                        mutation_file_id=mutation_file.mutation_file_id,
                        path=item["path"],
                        hunk_index=int(hunk_data["hunk_index"]),
                        old_start=int(hunk_data["old_start"]),
                        old_count=int(hunk_data["old_count"]),
                        new_start=int(hunk_data["new_start"]),
                        new_count=int(hunk_data["new_count"]),
                        header=str(hunk_data.get("header") or ""),
                        lines=lines,
                        content_digest=content_digest,
                    )
                    file_hunks.append(hunk)

                lineage: MutationLineage | None = None
                if item["requires_lineage"] and item["has_lineage"]:
                    lineage = MutationLineage(
                        lineage_id="",
                        mutation_id=provisional_mutation_id,
                        mutation_file_id=mutation_file.mutation_file_id,
                        path=item["path"],
                        before_content_digest=item["before_content_digest"],
                        after_content_digest=item["after_content_digest"],
                        before_structural_id=item["before_structural_id"],
                        after_structural_id=item["after_structural_id"],
                        ast_mutation_id=ast_mut.ast_mutation_id,
                        hunk_count=len(file_hunks),
                        byte_changed=bool(item["byte_changed"] or item["path_changed"]),
                        semantic_changed=item["semantic_changed"],
                        disposition=file_disposition
                        if status is not MutationStatus.ACCEPTED
                        else (
                            MutationDisposition.FORMATTING_ONLY
                            if item["formatting_only"]
                            else MutationDisposition.ACCEPTED
                        ),
                        recorded_at=recorded_at,
                    )
                    # Re-bind mutation_file with lineage_id.
                    mutation_file = MutationFile(
                        mutation_file_id=mutation_file.mutation_file_id,
                        mutation_id=provisional_mutation_id,
                        path=mutation_file.path,
                        prior_path=mutation_file.prior_path,
                        change_kind=mutation_file.change_kind,
                        disposition=mutation_file.disposition,
                        before_blob_id=mutation_file.before_blob_id,
                        after_blob_id=mutation_file.after_blob_id,
                        before_content_digest=mutation_file.before_content_digest,
                        after_content_digest=mutation_file.after_content_digest,
                        before_structural_id=mutation_file.before_structural_id,
                        after_structural_id=mutation_file.after_structural_id,
                        language=mutation_file.language,
                        byte_delta=mutation_file.byte_delta,
                        semantic_changed=mutation_file.semantic_changed,
                        formatting_only=mutation_file.formatting_only,
                        parse_status=mutation_file.parse_status,
                        lineage_id=lineage.lineage_id,
                        reason=mutation_file.reason,
                        recorded_at=recorded_at,
                    )

                mutation_files.append(mutation_file)
                if lineage is not None:
                    lineages.append(lineage)
                ast_mutations.append(ast_mut)
                hunks.extend(file_hunks)

            # Final acceptance gate: every content/path mutation in an accepted
            # set must have exactly one lineage.
            if status is MutationStatus.ACCEPTED:
                required_paths = {
                    item["path"] for item in analyzed if item["requires_lineage"]
                }
                lineage_paths = {item.path for item in lineages}
                if required_paths != lineage_paths:
                    status = MutationStatus.QUARANTINED
                    disposition = MutationDisposition.MISSING_LINEAGE
                    reason = "lineage_coverage_gap"
                    # Update file dispositions for quarantine.
                    mutation_files = [
                        MutationFile(
                            mutation_file_id=mf.mutation_file_id,
                            mutation_id=mf.mutation_id,
                            path=mf.path,
                            prior_path=mf.prior_path,
                            change_kind=mf.change_kind,
                            disposition=MutationDisposition.MISSING_LINEAGE
                            if mf.path in required_paths and not mf.lineage_id
                            else MutationDisposition.QUARANTINED,
                            before_blob_id=mf.before_blob_id,
                            after_blob_id=mf.after_blob_id,
                            before_content_digest=mf.before_content_digest,
                            after_content_digest=mf.after_content_digest,
                            before_structural_id=mf.before_structural_id,
                            after_structural_id=mf.after_structural_id,
                            language=mf.language,
                            byte_delta=mf.byte_delta,
                            semantic_changed=mf.semantic_changed,
                            formatting_only=mf.formatting_only,
                            parse_status=mf.parse_status,
                            lineage_id=mf.lineage_id,
                            reason=reason,
                            recorded_at=recorded_at,
                        )
                        for mf in mutation_files
                    ]

            mutation = MutationSet(
                mutation_id="",
                mutation_set_id="",
                task_id=context.task_id,
                attempt_id=context.attempt_id,
                plan_id=context.plan_id,
                operator_id=context.operator_id,
                provider_id=context.provider_id,
                daemon_id=context.daemon_id,
                session_id=context.session_id,
                worktree_id=context.worktree_id,
                lease_id=context.lease_id,
                fence_id=context.fence_id,
                fence_generation=fence_generation,
                before_snapshot_id=context.before_snapshot_id,
                after_snapshot_id=context.after_snapshot_id,
                before_tree_id=context.before_tree_id or (
                    fence.before_tree_id if fence else ""
                ),
                after_tree_id=context.after_tree_id,
                repository_id=context.repository_id,
                status=status,
                disposition=disposition,
                semantic_mutation_id=semantic_id,
                structural_identity=structural_id,
                declared_effects=effects,
                validation_outcome=context.validation_outcome,
                proof_outcome=context.proof_outcome,
                merge_outcome=context.merge_outcome,
                reason=reason,
                recorded_at=recorded_at,
                file_count=len(mutation_files),
                lineage_count=len(lineages),
            )

            # Rebind child identities to the final mutation_id.
            final_files: list[MutationFile] = []
            final_lineages: list[MutationLineage] = []
            final_ast: list[ASTMutation] = []
            final_hunks: list[MutationHunk] = []
            for mf, am, lin in self._zip_optional(
                mutation_files, ast_mutations, lineages, analyzed
            ):
                new_mf = MutationFile(
                    mutation_file_id="",  # recompute with final mutation_id
                    mutation_id=mutation.mutation_id,
                    path=mf.path,
                    prior_path=mf.prior_path,
                    change_kind=mf.change_kind,
                    disposition=mf.disposition,
                    before_blob_id=mf.before_blob_id,
                    after_blob_id=mf.after_blob_id,
                    before_content_digest=mf.before_content_digest,
                    after_content_digest=mf.after_content_digest,
                    before_structural_id=mf.before_structural_id,
                    after_structural_id=mf.after_structural_id,
                    language=mf.language,
                    byte_delta=mf.byte_delta,
                    semantic_changed=mf.semantic_changed,
                    formatting_only=mf.formatting_only,
                    parse_status=mf.parse_status,
                    reason=mf.reason,
                    recorded_at=mf.recorded_at,
                )
                new_am = ASTMutation(
                    ast_mutation_id="",
                    mutation_id=mutation.mutation_id,
                    mutation_file_id=new_mf.mutation_file_id,
                    path=am.path,
                    edit_script=dict(am.edit_script),
                    before_record_id=am.before_record_id,
                    after_record_id=am.after_record_id,
                    before_structural_id=am.before_structural_id,
                    after_structural_id=am.after_structural_id,
                    symbols_added=am.symbols_added,
                    symbols_removed=am.symbols_removed,
                    symbols_changed=am.symbols_changed,
                    parse_status=am.parse_status,
                    semantic_changed=am.semantic_changed,
                    recorded_at=am.recorded_at,
                )
                new_lineage: MutationLineage | None = None
                if lin is not None:
                    new_lineage = MutationLineage(
                        lineage_id="",
                        mutation_id=mutation.mutation_id,
                        mutation_file_id=new_mf.mutation_file_id,
                        path=lin.path,
                        before_content_digest=lin.before_content_digest,
                        after_content_digest=lin.after_content_digest,
                        before_structural_id=lin.before_structural_id,
                        after_structural_id=lin.after_structural_id,
                        ast_mutation_id=new_am.ast_mutation_id,
                        hunk_count=lin.hunk_count,
                        byte_changed=lin.byte_changed,
                        semantic_changed=lin.semantic_changed,
                        disposition=lin.disposition,
                        recorded_at=lin.recorded_at,
                    )
                    new_mf = MutationFile(
                        mutation_file_id=new_mf.mutation_file_id,
                        mutation_id=mutation.mutation_id,
                        path=new_mf.path,
                        prior_path=new_mf.prior_path,
                        change_kind=new_mf.change_kind,
                        disposition=new_mf.disposition,
                        before_blob_id=new_mf.before_blob_id,
                        after_blob_id=new_mf.after_blob_id,
                        before_content_digest=new_mf.before_content_digest,
                        after_content_digest=new_mf.after_content_digest,
                        before_structural_id=new_mf.before_structural_id,
                        after_structural_id=new_mf.after_structural_id,
                        language=new_mf.language,
                        byte_delta=new_mf.byte_delta,
                        semantic_changed=new_mf.semantic_changed,
                        formatting_only=new_mf.formatting_only,
                        parse_status=new_mf.parse_status,
                        lineage_id=new_lineage.lineage_id,
                        reason=new_mf.reason,
                        recorded_at=new_mf.recorded_at,
                    )
                # Rebind hunks that belonged to this file.
                for hunk in hunks:
                    if hunk.mutation_file_id != mf.mutation_file_id:
                        continue
                    final_hunks.append(
                        MutationHunk(
                            hunk_id=_identity(
                                "mutation-hunk",
                                {
                                    "mutation_file_id": new_mf.mutation_file_id,
                                    "hunk_index": hunk.hunk_index,
                                    "content_digest": hunk.content_digest,
                                },
                            ),
                            mutation_id=mutation.mutation_id,
                            mutation_file_id=new_mf.mutation_file_id,
                            path=hunk.path,
                            hunk_index=hunk.hunk_index,
                            old_start=hunk.old_start,
                            old_count=hunk.old_count,
                            new_start=hunk.new_start,
                            new_count=hunk.new_count,
                            header=hunk.header,
                            lines=hunk.lines,
                            content_digest=hunk.content_digest,
                        )
                    )
                final_files.append(new_mf)
                final_ast.append(new_am)
                if new_lineage is not None:
                    final_lineages.append(new_lineage)

            # Persist
            connection.execute(
                """
                INSERT INTO mutations(
                    mutation_id, mutation_set_id, task_id, attempt_id, plan_id,
                    operator_id, provider_id, daemon_id, session_id, worktree_id,
                    lease_id, fence_id, fence_generation, before_snapshot_id,
                    after_snapshot_id, before_tree_id, after_tree_id,
                    repository_id, status, disposition, semantic_mutation_id,
                    structural_identity, declared_effects_json,
                    validation_outcome, proof_outcome, merge_outcome,
                    rollback_outcome, reason, recorded_at, body_json
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    mutation.mutation_id,
                    mutation.mutation_set_id,
                    mutation.task_id,
                    mutation.attempt_id,
                    mutation.plan_id,
                    mutation.operator_id,
                    mutation.provider_id,
                    mutation.daemon_id,
                    mutation.session_id,
                    mutation.worktree_id,
                    mutation.lease_id,
                    mutation.fence_id,
                    mutation.fence_generation,
                    mutation.before_snapshot_id,
                    mutation.after_snapshot_id,
                    mutation.before_tree_id,
                    mutation.after_tree_id,
                    mutation.repository_id,
                    mutation.status.value,
                    mutation.disposition.value,
                    mutation.semantic_mutation_id,
                    mutation.structural_identity,
                    _canonical_json(dict(mutation.declared_effects)),
                    mutation.validation_outcome,
                    mutation.proof_outcome,
                    mutation.merge_outcome,
                    mutation.rollback_outcome,
                    mutation.reason,
                    mutation.recorded_at,
                    _canonical_json(mutation.to_dict()),
                ],
            )

            for mf in final_files:
                connection.execute(
                    """
                    INSERT INTO mutation_files(
                        mutation_file_id, mutation_id, path, prior_path,
                        change_kind, before_blob_id, after_blob_id,
                        before_content_digest, after_content_digest,
                        before_structural_id, after_structural_id, language,
                        byte_delta, semantic_changed, formatting_only,
                        parse_status, lineage_id, disposition, reason,
                        recorded_at, body_json
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?
                    )
                    """,
                    [
                        mf.mutation_file_id,
                        mf.mutation_id,
                        mf.path,
                        mf.prior_path,
                        mf.change_kind.value,
                        mf.before_blob_id,
                        mf.after_blob_id,
                        mf.before_content_digest,
                        mf.after_content_digest,
                        mf.before_structural_id,
                        mf.after_structural_id,
                        mf.language,
                        mf.byte_delta,
                        1 if mf.semantic_changed else 0,
                        1 if mf.formatting_only else 0,
                        mf.parse_status.value,
                        mf.lineage_id,
                        mf.disposition.value,
                        mf.reason,
                        mf.recorded_at,
                        _canonical_json(mf.to_dict()),
                    ],
                )

            for am in final_ast:
                connection.execute(
                    """
                    INSERT INTO ast_mutations(
                        ast_mutation_id, mutation_id, mutation_file_id, path,
                        before_record_id, after_record_id,
                        before_structural_id, after_structural_id,
                        edit_script_json, symbols_added_json,
                        symbols_removed_json, symbols_changed_json,
                        parse_status, semantic_changed, recorded_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    [
                        am.ast_mutation_id,
                        am.mutation_id,
                        am.mutation_file_id,
                        am.path,
                        am.before_record_id,
                        am.after_record_id,
                        am.before_structural_id,
                        am.after_structural_id,
                        _canonical_json(dict(am.edit_script)),
                        _canonical_json(list(am.symbols_added)),
                        _canonical_json(list(am.symbols_removed)),
                        _canonical_json(list(am.symbols_changed)),
                        am.parse_status.value,
                        1 if am.semantic_changed else 0,
                        am.recorded_at,
                    ],
                )

            for lin in final_lineages:
                connection.execute(
                    """
                    INSERT INTO mutation_lineages(
                        lineage_id, mutation_id, mutation_file_id, path,
                        before_content_digest, after_content_digest,
                        before_structural_id, after_structural_id,
                        ast_mutation_id, hunk_count, byte_changed,
                        semantic_changed, disposition, recorded_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    [
                        lin.lineage_id,
                        lin.mutation_id,
                        lin.mutation_file_id,
                        lin.path,
                        lin.before_content_digest,
                        lin.after_content_digest,
                        lin.before_structural_id,
                        lin.after_structural_id,
                        lin.ast_mutation_id,
                        lin.hunk_count,
                        1 if lin.byte_changed else 0,
                        1 if lin.semantic_changed else 0,
                        lin.disposition.value,
                        lin.recorded_at,
                    ],
                )

            for hunk in final_hunks:
                connection.execute(
                    """
                    INSERT INTO mutation_hunks(
                        hunk_id, mutation_id, mutation_file_id, path,
                        hunk_index, old_start, old_count, new_start, new_count,
                        header, lines_json, content_digest
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        hunk.hunk_id,
                        hunk.mutation_id,
                        hunk.mutation_file_id,
                        hunk.path,
                        hunk.hunk_index,
                        hunk.old_start,
                        hunk.old_count,
                        hunk.new_start,
                        hunk.new_count,
                        hunk.header,
                        _canonical_json(list(hunk.lines)),
                        hunk.content_digest,
                    ],
                )

            if status in {
                MutationStatus.QUARANTINED,
                MutationStatus.REJECTED,
            }:
                quarantine_id = _identity(
                    "mutation-quarantine",
                    {
                        "mutation_id": mutation.mutation_id,
                        "reason": reason,
                        "recorded_at": recorded_at,
                    },
                )
                connection.execute(
                    """
                    INSERT INTO mutation_quarantine(
                        quarantine_id, mutation_id, worktree_id, path, reason,
                        fence_id, before_snapshot_id, after_snapshot_id,
                        recorded_at, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        quarantine_id,
                        mutation.mutation_id,
                        mutation.worktree_id,
                        "",
                        reason,
                        mutation.fence_id,
                        mutation.before_snapshot_id,
                        mutation.after_snapshot_id,
                        recorded_at,
                        _canonical_json(
                            {
                                "status": status.value,
                                "disposition": disposition.value,
                                "paths": [mf.path for mf in final_files],
                            }
                        ),
                    ],
                )

            self._commit_if_idle(connection)

            admitted = status is MutationStatus.ACCEPTED
            return MutationRecordResult(
                mutation=mutation,
                files=tuple(final_files),
                lineages=tuple(final_lineages),
                ast_mutations=tuple(final_ast),
                hunks=tuple(final_hunks),
                admitted=admitted,
                quarantined=status is MutationStatus.QUARANTINED,
                rejected=status is MutationStatus.REJECTED,
            )

    def _zip_optional(
        self,
        mutation_files: Sequence[MutationFile],
        ast_mutations: Sequence[ASTMutation],
        lineages: Sequence[MutationLineage],
        analyzed: Sequence[Mapping[str, Any]],
    ) -> list[tuple[MutationFile, ASTMutation, MutationLineage | None]]:
        lineage_by_path = {item.path: item for item in lineages}
        # Match by original provisional mutation_file order via analyzed paths.
        result: list[tuple[MutationFile, ASTMutation, MutationLineage | None]] = []
        am_by_path = {item.path: item for item in ast_mutations}
        mf_by_path = {item.path: item for item in mutation_files}
        for item in analyzed:
            path = str(item["path"])
            mf = mf_by_path[path]
            am = am_by_path[path]
            result.append((mf, am, lineage_by_path.get(path)))
        return result

    def _analyze_file(
        self,
        spec: MutationFileSpec,
        *,
        recorded_at: str,
    ) -> dict[str, Any]:
        del recorded_at  # reserved for future provenance stamping
        before_digest = spec.before_content_digest
        after_digest = spec.after_content_digest
        byte_changed = before_digest != after_digest

        before_struct, before_parse, before_record = structural_identity_of(
            spec.before_content,
            language=spec.language,
            path=spec.path,
        )
        after_struct, after_parse, after_record = structural_identity_of(
            spec.after_content,
            language=spec.language,
            path=spec.path,
        )

        # Prefer failure if either side failed for a present payload.
        if before_parse is ParseOutcome.FAILED or after_parse is ParseOutcome.FAILED:
            parse_status = ParseOutcome.FAILED
        elif (
            before_parse is ParseOutcome.UNSUPPORTED
            or after_parse is ParseOutcome.UNSUPPORTED
        ):
            parse_status = ParseOutcome.UNSUPPORTED
        elif (
            before_parse is ParseOutcome.NOT_APPLICABLE
            and after_parse is ParseOutcome.NOT_APPLICABLE
        ):
            parse_status = ParseOutcome.NOT_APPLICABLE
        elif (
            before_parse is ParseOutcome.BOTH_EMPTY
            and after_parse is ParseOutcome.BOTH_EMPTY
        ):
            parse_status = ParseOutcome.BOTH_EMPTY
        else:
            parse_status = ParseOutcome.SUCCEEDED

        if spec.parse_failed:
            parse_status = ParseOutcome.FAILED

        semantic_changed = before_struct != after_struct
        formatting_only = byte_changed and not semantic_changed

        if spec.prior_path and spec.prior_path != spec.path:
            if not before_digest and after_digest:
                # rename-as-add without before is still rename if prior declared
                change_kind = FileChangeKind.RENAMED
            elif before_digest and after_digest:
                change_kind = FileChangeKind.RENAMED
            elif before_digest and not after_digest:
                change_kind = FileChangeKind.DELETED
            else:
                change_kind = FileChangeKind.RENAMED
        elif not before_digest and after_digest:
            change_kind = FileChangeKind.ADDED
        elif before_digest and not after_digest:
            change_kind = FileChangeKind.DELETED
        elif not byte_changed:
            change_kind = FileChangeKind.NO_OP
        elif formatting_only:
            change_kind = FileChangeKind.FORMATTING_ONLY
        elif parse_status is ParseOutcome.FAILED:
            change_kind = FileChangeKind.PARSE_FAILED
        else:
            change_kind = FileChangeKind.MODIFIED

        edit_script = _ast_edit_script(
            before_record,
            after_record,
            before_structural_id=before_struct,
            after_structural_id=after_struct,
            change_kind=change_kind,
        )
        hunks = _compute_hunks(spec.before_text, spec.after_text, path=spec.path)

        before_len = len(spec.before_content or b"")
        after_len = len(spec.after_content or b"")
        byte_delta = after_len - before_len

        # Path renames are mutations even when content digests match.  Lineage
        # is available when we can bind digests + AST edit script for a content
        # or path change. Unsupported languages still get a structural id and
        # empty AST ops so lineage can exist without claiming full AST facts.
        path_changed = bool(spec.prior_path and spec.prior_path != spec.path)
        requires_lineage = (byte_changed or path_changed) and not spec.partial
        has_lineage = False
        if requires_lineage:
            if parse_status is ParseOutcome.FAILED and byte_changed:
                has_lineage = False
            else:
                has_lineage = True

        return {
            "path": spec.path,
            "prior_path": spec.prior_path,
            "language": spec.language,
            "before_content_digest": before_digest,
            "after_content_digest": after_digest,
            "before_structural_id": before_struct,
            "after_structural_id": after_struct,
            "before_record_id": before_record.record_id if before_record else "",
            "after_record_id": after_record.record_id if after_record else "",
            "byte_changed": byte_changed,
            "path_changed": path_changed,
            "requires_lineage": requires_lineage,
            "semantic_changed": semantic_changed or (
                path_changed and change_kind is FileChangeKind.RENAMED
            ),
            "formatting_only": formatting_only and not path_changed,
            "parse_status": parse_status,
            "change_kind": change_kind,
            "edit_script": edit_script,
            "hunks": hunks,
            "byte_delta": byte_delta,
            "has_lineage": has_lineage,
            "partial": bool(spec.partial),
        }

    # -- queries -------------------------------------------------------------

    def get_mutation(self, mutation_id: str) -> MutationSet | None:
        connection = self._require()
        mid = _text(mutation_id, "mutation_id")
        with self._lock:
            row = connection.execute(
                """
                SELECT body_json, mutation_id, mutation_set_id, task_id,
                       attempt_id, plan_id, operator_id, provider_id, daemon_id,
                       session_id, worktree_id, lease_id, fence_id,
                       fence_generation, before_snapshot_id, after_snapshot_id,
                       before_tree_id, after_tree_id, repository_id, status,
                       disposition, semantic_mutation_id, structural_identity,
                       declared_effects_json, validation_outcome, proof_outcome,
                       merge_outcome, rollback_outcome, reason, recorded_at
                FROM mutations
                WHERE mutation_id = ?
                """,
                [mid],
            ).fetchone()
            if row is None:
                return None
            return self._mutation_from_row(row)

    def list_mutations(
        self,
        *,
        task_id: str = "",
        worktree_id: str = "",
        status: MutationStatus | str | None = None,
        limit: int = 100,
    ) -> tuple[MutationSet, ...]:
        connection = self._require()
        limit = max(1, min(int(limit), 10_000))
        clauses: list[str] = []
        params: list[Any] = []
        if task_id:
            clauses.append("task_id = ?")
            params.append(_text(task_id, "task_id"))
        if worktree_id:
            clauses.append("worktree_id = ?")
            params.append(_text(worktree_id, "worktree_id"))
        if status is not None:
            clauses.append("status = ?")
            params.append(
                status.value if isinstance(status, MutationStatus) else str(status)
            )
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        with self._lock:
            rows = connection.execute(
                f"""
                SELECT body_json, mutation_id, mutation_set_id, task_id,
                       attempt_id, plan_id, operator_id, provider_id, daemon_id,
                       session_id, worktree_id, lease_id, fence_id,
                       fence_generation, before_snapshot_id, after_snapshot_id,
                       before_tree_id, after_tree_id, repository_id, status,
                       disposition, semantic_mutation_id, structural_identity,
                       declared_effects_json, validation_outcome, proof_outcome,
                       merge_outcome, rollback_outcome, reason, recorded_at
                FROM mutations
                {where}
                ORDER BY recorded_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return tuple(self._mutation_from_row(row) for row in rows)

    def list_mutation_files(
        self, mutation_id: str
    ) -> tuple[MutationFile, ...]:
        connection = self._require()
        mid = _text(mutation_id, "mutation_id")
        with self._lock:
            rows = connection.execute(
                """
                SELECT body_json, mutation_file_id, mutation_id, path, prior_path,
                       change_kind, before_blob_id, after_blob_id,
                       before_content_digest, after_content_digest,
                       before_structural_id, after_structural_id, language,
                       byte_delta, semantic_changed, formatting_only,
                       parse_status, lineage_id, disposition, reason, recorded_at
                FROM mutation_files
                WHERE mutation_id = ?
                ORDER BY path
                """,
                [mid],
            ).fetchall()
            return tuple(self._mutation_file_from_row(row) for row in rows)

    def list_ast_mutations(
        self, mutation_id: str
    ) -> tuple[ASTMutation, ...]:
        connection = self._require()
        mid = _text(mutation_id, "mutation_id")
        with self._lock:
            rows = connection.execute(
                """
                SELECT ast_mutation_id, mutation_id, mutation_file_id, path,
                       before_record_id, after_record_id,
                       before_structural_id, after_structural_id,
                       edit_script_json, symbols_added_json,
                       symbols_removed_json, symbols_changed_json,
                       parse_status, semantic_changed, recorded_at
                FROM ast_mutations
                WHERE mutation_id = ?
                ORDER BY path
                """,
                [mid],
            ).fetchall()
            results: list[ASTMutation] = []
            for row in rows:
                m = _row_mapping(row)
                if m:
                    results.append(
                        ASTMutation(
                            ast_mutation_id=str(m["ast_mutation_id"]),
                            mutation_id=str(m["mutation_id"]),
                            mutation_file_id=str(m["mutation_file_id"]),
                            path=str(m["path"]),
                            edit_script=json.loads(str(m["edit_script_json"] or "{}")),
                            before_record_id=str(m.get("before_record_id") or ""),
                            after_record_id=str(m.get("after_record_id") or ""),
                            before_structural_id=str(
                                m.get("before_structural_id") or ""
                            ),
                            after_structural_id=str(
                                m.get("after_structural_id") or ""
                            ),
                            symbols_added=tuple(
                                json.loads(str(m.get("symbols_added_json") or "[]"))
                            ),
                            symbols_removed=tuple(
                                json.loads(
                                    str(m.get("symbols_removed_json") or "[]")
                                )
                            ),
                            symbols_changed=tuple(
                                json.loads(
                                    str(m.get("symbols_changed_json") or "[]")
                                )
                            ),
                            parse_status=str(m["parse_status"]),
                            semantic_changed=bool(int(m.get("semantic_changed") or 0)),
                            recorded_at=str(m.get("recorded_at") or ""),
                        )
                    )
                else:
                    results.append(
                        ASTMutation(
                            ast_mutation_id=str(row[0]),
                            mutation_id=str(row[1]),
                            mutation_file_id=str(row[2]),
                            path=str(row[3]),
                            edit_script=json.loads(str(row[8] or "{}")),
                            before_record_id=str(row[4] or ""),
                            after_record_id=str(row[5] or ""),
                            before_structural_id=str(row[6] or ""),
                            after_structural_id=str(row[7] or ""),
                            symbols_added=tuple(json.loads(str(row[9] or "[]"))),
                            symbols_removed=tuple(
                                json.loads(str(row[10] or "[]"))
                            ),
                            symbols_changed=tuple(
                                json.loads(str(row[11] or "[]"))
                            ),
                            parse_status=str(row[12]),
                            semantic_changed=bool(int(row[13] or 0)),
                            recorded_at=str(row[14] or ""),
                        )
                    )
            return tuple(results)

    def list_lineages(self, mutation_id: str) -> tuple[MutationLineage, ...]:
        connection = self._require()
        mid = _text(mutation_id, "mutation_id")
        with self._lock:
            rows = connection.execute(
                """
                SELECT lineage_id, mutation_id, mutation_file_id, path,
                       before_content_digest, after_content_digest,
                       before_structural_id, after_structural_id,
                       ast_mutation_id, hunk_count, byte_changed,
                       semantic_changed, disposition, recorded_at
                FROM mutation_lineages
                WHERE mutation_id = ?
                ORDER BY path
                """,
                [mid],
            ).fetchall()
            results: list[MutationLineage] = []
            for row in rows:
                m = _row_mapping(row)
                if m:
                    results.append(
                        MutationLineage(
                            lineage_id=str(m["lineage_id"]),
                            mutation_id=str(m["mutation_id"]),
                            mutation_file_id=str(m["mutation_file_id"]),
                            path=str(m["path"]),
                            before_content_digest=str(
                                m.get("before_content_digest") or ""
                            ),
                            after_content_digest=str(
                                m.get("after_content_digest") or ""
                            ),
                            before_structural_id=str(
                                m.get("before_structural_id") or ""
                            ),
                            after_structural_id=str(
                                m.get("after_structural_id") or ""
                            ),
                            ast_mutation_id=str(m.get("ast_mutation_id") or ""),
                            hunk_count=int(m.get("hunk_count") or 0),
                            byte_changed=bool(int(m.get("byte_changed") or 0)),
                            semantic_changed=bool(
                                int(m.get("semantic_changed") or 0)
                            ),
                            disposition=str(m["disposition"]),
                            recorded_at=str(m.get("recorded_at") or ""),
                        )
                    )
                else:
                    results.append(
                        MutationLineage(
                            lineage_id=str(row[0]),
                            mutation_id=str(row[1]),
                            mutation_file_id=str(row[2]),
                            path=str(row[3]),
                            before_content_digest=str(row[4] or ""),
                            after_content_digest=str(row[5] or ""),
                            before_structural_id=str(row[6] or ""),
                            after_structural_id=str(row[7] or ""),
                            ast_mutation_id=str(row[8] or ""),
                            hunk_count=int(row[9] or 0),
                            byte_changed=bool(int(row[10] or 0)),
                            semantic_changed=bool(int(row[11] or 0)),
                            disposition=str(row[12]),
                            recorded_at=str(row[13] or ""),
                        )
                    )
            return tuple(results)

    def list_hunks(self, mutation_id: str) -> tuple[MutationHunk, ...]:
        connection = self._require()
        mid = _text(mutation_id, "mutation_id")
        with self._lock:
            rows = connection.execute(
                """
                SELECT hunk_id, mutation_id, mutation_file_id, path, hunk_index,
                       old_start, old_count, new_start, new_count, header,
                       lines_json, content_digest
                FROM mutation_hunks
                WHERE mutation_id = ?
                ORDER BY path, hunk_index
                """,
                [mid],
            ).fetchall()
            results: list[MutationHunk] = []
            for row in rows:
                m = _row_mapping(row)
                if m:
                    results.append(
                        MutationHunk(
                            hunk_id=str(m["hunk_id"]),
                            mutation_id=str(m["mutation_id"]),
                            mutation_file_id=str(m["mutation_file_id"]),
                            path=str(m["path"]),
                            hunk_index=int(m["hunk_index"]),
                            old_start=int(m["old_start"]),
                            old_count=int(m["old_count"]),
                            new_start=int(m["new_start"]),
                            new_count=int(m["new_count"]),
                            header=str(m.get("header") or ""),
                            lines=tuple(json.loads(str(m.get("lines_json") or "[]"))),
                            content_digest=str(m.get("content_digest") or ""),
                        )
                    )
                else:
                    results.append(
                        MutationHunk(
                            hunk_id=str(row[0]),
                            mutation_id=str(row[1]),
                            mutation_file_id=str(row[2]),
                            path=str(row[3]),
                            hunk_index=int(row[4]),
                            old_start=int(row[5]),
                            old_count=int(row[6]),
                            new_start=int(row[7]),
                            new_count=int(row[8]),
                            header=str(row[9] or ""),
                            lines=tuple(json.loads(str(row[10] or "[]"))),
                            content_digest=str(row[11] or ""),
                        )
                    )
            return tuple(results)

    def list_quarantine(
        self, *, worktree_id: str = "", limit: int = 100
    ) -> tuple[dict[str, Any], ...]:
        connection = self._require()
        limit = max(1, min(int(limit), 10_000))
        with self._lock:
            if worktree_id:
                rows = connection.execute(
                    """
                    SELECT quarantine_id, mutation_id, worktree_id, path, reason,
                           fence_id, before_snapshot_id, after_snapshot_id,
                           recorded_at, body_json
                    FROM mutation_quarantine
                    WHERE worktree_id = ?
                    ORDER BY recorded_at DESC
                    LIMIT ?
                    """,
                    [_text(worktree_id, "worktree_id"), limit],
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT quarantine_id, mutation_id, worktree_id, path, reason,
                           fence_id, before_snapshot_id, after_snapshot_id,
                           recorded_at, body_json
                    FROM mutation_quarantine
                    ORDER BY recorded_at DESC
                    LIMIT ?
                    """,
                    [limit],
                ).fetchall()
            results: list[dict[str, Any]] = []
            for row in rows:
                m = _row_mapping(row)
                if m:
                    results.append(
                        {
                            "quarantine_id": str(m["quarantine_id"]),
                            "mutation_id": str(m.get("mutation_id") or ""),
                            "worktree_id": str(m["worktree_id"]),
                            "path": str(m.get("path") or ""),
                            "reason": str(m.get("reason") or ""),
                            "fence_id": str(m.get("fence_id") or ""),
                            "before_snapshot_id": str(
                                m.get("before_snapshot_id") or ""
                            ),
                            "after_snapshot_id": str(
                                m.get("after_snapshot_id") or ""
                            ),
                            "recorded_at": str(m.get("recorded_at") or ""),
                            "body": json.loads(str(m.get("body_json") or "{}")),
                            "authority": AUTHORITY_CLASS,
                        }
                    )
                else:
                    results.append(
                        {
                            "quarantine_id": str(row[0]),
                            "mutation_id": str(row[1] or ""),
                            "worktree_id": str(row[2]),
                            "path": str(row[3] or ""),
                            "reason": str(row[4] or ""),
                            "fence_id": str(row[5] or ""),
                            "before_snapshot_id": str(row[6] or ""),
                            "after_snapshot_id": str(row[7] or ""),
                            "recorded_at": str(row[8] or ""),
                            "body": json.loads(str(row[9] or "{}")),
                            "authority": AUTHORITY_CLASS,
                        }
                    )
            return tuple(results)

    # -- rollback ------------------------------------------------------------

    def record_rollback(
        self,
        *,
        mutation_id: str,
        restored_files: Sequence[MutationFileSpec] | Mapping[str, bytes | str | None],
        worktree_id: str = "",
    ) -> RollbackReceipt:
        """Independently verify rollback restoration against before digests.

        ``restored_files`` is the observed post-rollback content (or digests
        via MutationFileSpec.after_content / after_content_digest).  Expected
        digests are the original mutation's before_content_digest values.
        """

        connection = self._require()
        mid = _text(mutation_id, "mutation_id")
        mutation = self.get_mutation(mid)
        if mutation is None:
            raise MutationLedgerIntegrityError(f"unknown mutation: {mid}")
        wt = _text(worktree_id or mutation.worktree_id, "worktree_id")
        files = self.list_mutation_files(mid)
        expected: dict[str, str] = {}
        for mf in files:
            if mf.change_kind is FileChangeKind.ADDED:
                # Added files should be absent after rollback.
                expected[mf.path] = ""
            elif (
                mf.change_kind is FileChangeKind.RENAMED and mf.prior_path
            ):
                # Rename rollback restores the prior path and removes the new
                # path. Content identity is the pre-mutation blob.
                expected[mf.prior_path] = (
                    mf.before_content_digest or mf.after_content_digest
                )
                expected[mf.path] = ""
            elif mf.change_kind is FileChangeKind.DELETED:
                # Deleted paths are restored to their before digest.
                expected[mf.path] = mf.before_content_digest
            else:
                expected[mf.path] = mf.before_content_digest

        observed: dict[str, str] = {}
        if isinstance(restored_files, Mapping):
            for path, content in restored_files.items():
                normalized = _repo_path(path)
                if content is None:
                    observed[normalized] = ""
                else:
                    observed[normalized] = content_digest_of(content)
        else:
            for spec in restored_files:
                if not isinstance(spec, MutationFileSpec):
                    raise MutationLedgerIntegrityError(
                        "restored_files must be MutationFileSpec or mapping"
                    )
                # Prefer after_content as "observed restored state".
                if spec.after_content is not None or spec.after_content_digest:
                    observed[spec.path] = (
                        spec.after_content_digest
                        or content_digest_of(spec.after_content)
                    )
                elif spec.before_content is not None or spec.before_content_digest:
                    observed[spec.path] = (
                        spec.before_content_digest
                        or content_digest_of(spec.before_content)
                    )
                else:
                    observed[spec.path] = ""

        mismatches: dict[str, dict[str, str]] = {}
        for path, expected_digest in expected.items():
            actual = observed.get(path, "")
            if _normalize_digest(actual) != _normalize_digest(expected_digest):
                mismatches[path] = {
                    "expected": _normalize_digest(expected_digest),
                    "observed": _normalize_digest(actual),
                }

        verified = not mismatches
        status = (
            RollbackStatus.VERIFIED if verified else RollbackStatus.FAILED
        )
        reason = "" if verified else "digest_mismatch:" + ",".join(
            sorted(mismatches)[:8]
        )
        recorded_at = _utc_iso()
        receipt = RollbackReceipt(
            rollback_id="",
            mutation_id=mid,
            worktree_id=wt,
            status=status,
            verified=verified,
            expected_digests=expected,
            observed_digests=observed,
            mismatches=mismatches,
            reason=reason,
            recorded_at=recorded_at,
        )

        with self._lock:
            connection.execute(
                """
                INSERT INTO mutation_rollbacks(
                    rollback_id, mutation_id, worktree_id, status, verified,
                    expected_digests_json, observed_digests_json, mismatch_json,
                    reason, recorded_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    receipt.rollback_id,
                    receipt.mutation_id,
                    receipt.worktree_id,
                    receipt.status.value,
                    1 if receipt.verified else 0,
                    _canonical_json(dict(receipt.expected_digests)),
                    _canonical_json(dict(receipt.observed_digests)),
                    _canonical_json(
                        {k: dict(v) for k, v in receipt.mismatches.items()}
                    ),
                    receipt.reason,
                    receipt.recorded_at,
                    _canonical_json(receipt.to_dict()),
                ],
            )
            connection.execute(
                """
                UPDATE mutations
                SET rollback_outcome = ?, status = CASE
                    WHEN ? = 1 THEN ?
                    ELSE status
                END
                WHERE mutation_id = ?
                """,
                [
                    receipt.status.value,
                    1 if verified else 0,
                    MutationStatus.ROLLED_BACK.value,
                    mid,
                ],
            )
            self._commit_if_idle(connection)
        return receipt

    def get_rollback(self, mutation_id: str) -> RollbackReceipt | None:
        connection = self._require()
        mid = _text(mutation_id, "mutation_id")
        with self._lock:
            row = connection.execute(
                """
                SELECT rollback_id, mutation_id, worktree_id, status, verified,
                       expected_digests_json, observed_digests_json,
                       mismatch_json, reason, recorded_at
                FROM mutation_rollbacks
                WHERE mutation_id = ?
                ORDER BY recorded_at DESC
                LIMIT 1
                """,
                [mid],
            ).fetchone()
            if row is None:
                return None
            m = _row_mapping(row)
            if m:
                return RollbackReceipt(
                    rollback_id=str(m["rollback_id"]),
                    mutation_id=str(m["mutation_id"]),
                    worktree_id=str(m["worktree_id"]),
                    status=str(m["status"]),
                    verified=bool(int(m.get("verified") or 0)),
                    expected_digests=json.loads(
                        str(m.get("expected_digests_json") or "{}")
                    ),
                    observed_digests=json.loads(
                        str(m.get("observed_digests_json") or "{}")
                    ),
                    mismatches=json.loads(str(m.get("mismatch_json") or "{}")),
                    reason=str(m.get("reason") or ""),
                    recorded_at=str(m.get("recorded_at") or ""),
                )
            return RollbackReceipt(
                rollback_id=str(row[0]),
                mutation_id=str(row[1]),
                worktree_id=str(row[2]),
                status=str(row[3]),
                verified=bool(int(row[4] or 0)),
                expected_digests=json.loads(str(row[5] or "{}")),
                observed_digests=json.loads(str(row[6] or "{}")),
                mismatches=json.loads(str(row[7] or "{}")),
                reason=str(row[8] or ""),
                recorded_at=str(row[9] or ""),
            )

    # -- row helpers ---------------------------------------------------------

    def _mutation_from_row(self, row: Any) -> MutationSet:
        m = _row_mapping(row)
        if m and m.get("body_json"):
            try:
                body = json.loads(str(m["body_json"]))
                if isinstance(body, dict) and body.get("mutation_id"):
                    # Prefer live columns for status/rollback which may update.
                    return MutationSet(
                        mutation_id=str(m.get("mutation_id") or body["mutation_id"]),
                        mutation_set_id=str(
                            m.get("mutation_set_id")
                            or body.get("mutation_set_id")
                            or ""
                        ),
                        task_id=str(m.get("task_id") or body.get("task_id") or ""),
                        attempt_id=str(
                            m.get("attempt_id") or body.get("attempt_id") or ""
                        ),
                        plan_id=str(m.get("plan_id") or body.get("plan_id") or ""),
                        operator_id=str(
                            m.get("operator_id") or body.get("operator_id") or ""
                        ),
                        provider_id=str(
                            m.get("provider_id") or body.get("provider_id") or ""
                        ),
                        daemon_id=str(
                            m.get("daemon_id") or body.get("daemon_id") or ""
                        ),
                        session_id=str(
                            m.get("session_id") or body.get("session_id") or ""
                        ),
                        worktree_id=str(
                            m.get("worktree_id") or body.get("worktree_id") or ""
                        ),
                        lease_id=str(
                            m.get("lease_id") or body.get("lease_id") or ""
                        ),
                        fence_id=str(
                            m.get("fence_id") or body.get("fence_id") or ""
                        ),
                        fence_generation=int(
                            m.get("fence_generation")
                            or body.get("fence_generation")
                            or 0
                        ),
                        before_snapshot_id=str(
                            m.get("before_snapshot_id")
                            or body.get("before_snapshot_id")
                            or ""
                        ),
                        after_snapshot_id=str(
                            m.get("after_snapshot_id")
                            or body.get("after_snapshot_id")
                            or ""
                        ),
                        before_tree_id=str(
                            m.get("before_tree_id")
                            or body.get("before_tree_id")
                            or ""
                        ),
                        after_tree_id=str(
                            m.get("after_tree_id")
                            or body.get("after_tree_id")
                            or ""
                        ),
                        repository_id=str(
                            m.get("repository_id")
                            or body.get("repository_id")
                            or ""
                        ),
                        status=str(m.get("status") or body.get("status") or ""),
                        disposition=str(
                            m.get("disposition") or body.get("disposition") or ""
                        ),
                        semantic_mutation_id=str(
                            m.get("semantic_mutation_id")
                            or body.get("semantic_mutation_id")
                            or ""
                        ),
                        structural_identity=str(
                            m.get("structural_identity")
                            or body.get("structural_identity")
                            or ""
                        ),
                        declared_effects=json.loads(
                            str(
                                m.get("declared_effects_json")
                                or _canonical_json(
                                    body.get("declared_effects") or {}
                                )
                            )
                        ),
                        validation_outcome=str(
                            m.get("validation_outcome")
                            or body.get("validation_outcome")
                            or ""
                        ),
                        proof_outcome=str(
                            m.get("proof_outcome")
                            or body.get("proof_outcome")
                            or ""
                        ),
                        merge_outcome=str(
                            m.get("merge_outcome")
                            or body.get("merge_outcome")
                            or ""
                        ),
                        rollback_outcome=str(
                            m.get("rollback_outcome")
                            or body.get("rollback_outcome")
                            or ""
                        ),
                        reason=str(m.get("reason") or body.get("reason") or ""),
                        recorded_at=str(
                            m.get("recorded_at") or body.get("recorded_at") or ""
                        ),
                        file_count=int(body.get("file_count") or 0),
                        lineage_count=int(body.get("lineage_count") or 0),
                    )
            except (TypeError, ValueError, json.JSONDecodeError, MutationLedgerError):
                pass
        # Positional fallback
        if m:
            return MutationSet(
                mutation_id=str(m["mutation_id"]),
                mutation_set_id=str(m.get("mutation_set_id") or ""),
                task_id=str(m["task_id"]),
                attempt_id=str(m.get("attempt_id") or ""),
                plan_id=str(m.get("plan_id") or ""),
                operator_id=str(m.get("operator_id") or ""),
                provider_id=str(m.get("provider_id") or ""),
                daemon_id=str(m.get("daemon_id") or ""),
                session_id=str(m.get("session_id") or ""),
                worktree_id=str(m["worktree_id"]),
                lease_id=str(m.get("lease_id") or ""),
                fence_id=str(m["fence_id"]),
                fence_generation=int(m.get("fence_generation") or 0),
                before_snapshot_id=str(m["before_snapshot_id"]),
                after_snapshot_id=str(m.get("after_snapshot_id") or ""),
                before_tree_id=str(m.get("before_tree_id") or ""),
                after_tree_id=str(m.get("after_tree_id") or ""),
                repository_id=str(m.get("repository_id") or ""),
                status=str(m["status"]),
                disposition=str(m["disposition"]),
                semantic_mutation_id=str(m["semantic_mutation_id"]),
                structural_identity=str(m["structural_identity"]),
                declared_effects=json.loads(
                    str(m.get("declared_effects_json") or "{}")
                ),
                validation_outcome=str(m.get("validation_outcome") or ""),
                proof_outcome=str(m.get("proof_outcome") or ""),
                merge_outcome=str(m.get("merge_outcome") or ""),
                rollback_outcome=str(m.get("rollback_outcome") or ""),
                reason=str(m.get("reason") or ""),
                recorded_at=str(m.get("recorded_at") or ""),
            )
        return MutationSet(
            mutation_id=str(row[1]),
            mutation_set_id=str(row[2]),
            task_id=str(row[3]),
            attempt_id=str(row[4] or ""),
            plan_id=str(row[5] or ""),
            operator_id=str(row[6] or ""),
            provider_id=str(row[7] or ""),
            daemon_id=str(row[8] or ""),
            session_id=str(row[9] or ""),
            worktree_id=str(row[10]),
            lease_id=str(row[11] or ""),
            fence_id=str(row[12]),
            fence_generation=int(row[13] or 0),
            before_snapshot_id=str(row[14]),
            after_snapshot_id=str(row[15] or ""),
            before_tree_id=str(row[16] or ""),
            after_tree_id=str(row[17] or ""),
            repository_id=str(row[18] or ""),
            status=str(row[19]),
            disposition=str(row[20]),
            semantic_mutation_id=str(row[21]),
            structural_identity=str(row[22]),
            declared_effects=json.loads(str(row[23] or "{}")),
            validation_outcome=str(row[24] or ""),
            proof_outcome=str(row[25] or ""),
            merge_outcome=str(row[26] or ""),
            rollback_outcome=str(row[27] or ""),
            reason=str(row[28] or ""),
            recorded_at=str(row[29] or ""),
        )

    def _mutation_file_from_row(self, row: Any) -> MutationFile:
        m = _row_mapping(row)
        if m and m.get("body_json"):
            try:
                body = json.loads(str(m["body_json"]))
                if isinstance(body, dict) and body.get("mutation_file_id"):
                    return MutationFile(
                        mutation_file_id=str(
                            m.get("mutation_file_id") or body["mutation_file_id"]
                        ),
                        mutation_id=str(
                            m.get("mutation_id") or body.get("mutation_id") or ""
                        ),
                        path=str(m.get("path") or body.get("path") or ""),
                        prior_path=str(
                            m.get("prior_path") or body.get("prior_path") or ""
                        ),
                        change_kind=str(
                            m.get("change_kind") or body.get("change_kind") or ""
                        ),
                        disposition=str(
                            m.get("disposition") or body.get("disposition") or ""
                        ),
                        before_blob_id=str(
                            m.get("before_blob_id")
                            or body.get("before_blob_id")
                            or ""
                        ),
                        after_blob_id=str(
                            m.get("after_blob_id")
                            or body.get("after_blob_id")
                            or ""
                        ),
                        before_content_digest=str(
                            m.get("before_content_digest")
                            or body.get("before_content_digest")
                            or ""
                        ),
                        after_content_digest=str(
                            m.get("after_content_digest")
                            or body.get("after_content_digest")
                            or ""
                        ),
                        before_structural_id=str(
                            m.get("before_structural_id")
                            or body.get("before_structural_id")
                            or ""
                        ),
                        after_structural_id=str(
                            m.get("after_structural_id")
                            or body.get("after_structural_id")
                            or ""
                        ),
                        language=str(
                            m.get("language") or body.get("language") or ""
                        ),
                        byte_delta=int(
                            m.get("byte_delta")
                            if m.get("byte_delta") is not None
                            else body.get("byte_delta") or 0
                        ),
                        semantic_changed=bool(
                            int(
                                m.get("semantic_changed")
                                if m.get("semantic_changed") is not None
                                else (1 if body.get("semantic_changed") else 0)
                            )
                        ),
                        formatting_only=bool(
                            int(
                                m.get("formatting_only")
                                if m.get("formatting_only") is not None
                                else (1 if body.get("formatting_only") else 0)
                            )
                        ),
                        parse_status=str(
                            m.get("parse_status")
                            or body.get("parse_status")
                            or ParseOutcome.NOT_APPLICABLE.value
                        ),
                        lineage_id=str(
                            m.get("lineage_id") or body.get("lineage_id") or ""
                        ),
                        reason=str(m.get("reason") or body.get("reason") or ""),
                        recorded_at=str(
                            m.get("recorded_at") or body.get("recorded_at") or ""
                        ),
                    )
            except (TypeError, ValueError, json.JSONDecodeError, MutationLedgerError):
                pass
        if m:
            return MutationFile(
                mutation_file_id=str(m["mutation_file_id"]),
                mutation_id=str(m["mutation_id"]),
                path=str(m["path"]),
                prior_path=str(m.get("prior_path") or ""),
                change_kind=str(m["change_kind"]),
                disposition=str(m["disposition"]),
                before_blob_id=str(m.get("before_blob_id") or ""),
                after_blob_id=str(m.get("after_blob_id") or ""),
                before_content_digest=str(m.get("before_content_digest") or ""),
                after_content_digest=str(m.get("after_content_digest") or ""),
                before_structural_id=str(m.get("before_structural_id") or ""),
                after_structural_id=str(m.get("after_structural_id") or ""),
                language=str(m.get("language") or ""),
                byte_delta=int(m.get("byte_delta") or 0),
                semantic_changed=bool(int(m.get("semantic_changed") or 0)),
                formatting_only=bool(int(m.get("formatting_only") or 0)),
                parse_status=str(
                    m.get("parse_status") or ParseOutcome.NOT_APPLICABLE.value
                ),
                lineage_id=str(m.get("lineage_id") or ""),
                reason=str(m.get("reason") or ""),
                recorded_at=str(m.get("recorded_at") or ""),
            )
        return MutationFile(
            mutation_file_id=str(row[1]),
            mutation_id=str(row[2]),
            path=str(row[3]),
            prior_path=str(row[4] or ""),
            change_kind=str(row[5]),
            disposition=str(row[18]),
            before_blob_id=str(row[6] or ""),
            after_blob_id=str(row[7] or ""),
            before_content_digest=str(row[8] or ""),
            after_content_digest=str(row[9] or ""),
            before_structural_id=str(row[10] or ""),
            after_structural_id=str(row[11] or ""),
            language=str(row[12] or ""),
            byte_delta=int(row[13] or 0),
            semantic_changed=bool(int(row[14] or 0)),
            formatting_only=bool(int(row[15] or 0)),
            parse_status=str(row[16] or ParseOutcome.NOT_APPLICABLE.value),
            lineage_id=str(row[17] or ""),
            reason=str(row[19] or ""),
            recorded_at=str(row[20] or ""),
        )


def open_mutation_ledger(
    database_path: Path | str,
    *,
    parser_id: str = DEFAULT_PARSER_ID,
    ledger_version: str = DEFAULT_LEDGER_VERSION,
) -> MutationLedger:
    """Open (or create) a mutation ledger store."""

    return MutationLedger(
        database_path,
        parser_id=parser_id,
        ledger_version=ledger_version,
    ).open()


__all__ = [
    "AST_MUTATION_INTERFACE",
    "AST_MUTATION_SCHEMA",
    "ASTMutation",
    "AUTHORITY_CLASS",
    "DEFAULT_LEDGER_VERSION",
    "DEFAULT_PARSER_ID",
    "DuckDBUnavailableError",
    "FENCE_SCHEMA",
    "FenceStatus",
    "FileChangeKind",
    "MUTATION_FILE_INTERFACE",
    "MUTATION_FILE_SCHEMA",
    "MUTATION_LEDGER_INTERFACE",
    "MUTATION_LEDGER_SCHEMA",
    "MUTATION_LINEAGE_SCHEMA",
    "MUTATION_SET_INTERFACE",
    "MUTATION_SET_SCHEMA",
    "MutationContext",
    "MutationDisposition",
    "MutationFence",
    "MutationFile",
    "MutationFileSpec",
    "MutationHunk",
    "MutationLedger",
    "MutationLedgerAdmissionError",
    "MutationLedgerBoundsError",
    "MutationLedgerConflictError",
    "MutationLedgerError",
    "MutationLedgerIntegrityError",
    "MutationLedgerNotOpenError",
    "MutationLineage",
    "MutationRecordResult",
    "MutationSet",
    "MutationStatus",
    "ParseOutcome",
    "ROLLBACK_RECEIPT_SCHEMA",
    "RollbackReceipt",
    "RollbackStatus",
    "content_digest_of",
    "duckdb_available",
    "language_for_path",
    "open_mutation_ledger",
    "semantic_mutation_identity",
    "structural_identity_of",
]
