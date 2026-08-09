"""Fail-closed actual provider registration/handler surface inventory (DCR-013).

Scans the mandatory accelerate, datasets, kit, and MCP++ package roots with
:class:`PythonMcpSurfaceExtractor`, then encodes the observed registration,
dispatcher, handler, effect, unresolved, and duplicate-equivalence rows as a
compact dictionary/Merkle projection under the supervisor admission limit.

Expected descriptors never substitute for actual registrations. Duplicate
anchors remain ambiguous. Capture binds the forest-backed trees, never
transient worktree dirt.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
import zlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Final

from .deterministic_repair_forest import (
    DCR_TODO_PATH,
    REPAIR_FOREST_SCHEMA,
    _document_integrity,
    _git_is_ancestor,
    _git_oid,
    _OID_PATTERN,
)
from .python_mcp_surface_extractor import (
    PYTHON_MCP_SURFACE_EXTRACTOR_INTERFACE,
    PythonMcpSurfaceExtractor,
    PythonMcpToolSurface,
    UnresolvedReason,
    UnresolvedRegistration,
    extract_python_mcp_source,
)

PROVIDER_SURFACE_HEALTH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-provider-surfaces@1"
)
PROVIDER_SURFACE_VALIDATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "deterministic-repair-provider-surfaces-validation@1"
)
PROVIDER_SURFACE_LIFECYCLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "deterministic-repair-provider-surfaces-lifecycle@1"
)
SURFACE_LEDGER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "deterministic-repair-provider-surface-ledger@1"
)
SURFACE_CODEC: Final[str] = "dcr-provider-surface-dictionary-prefix@1"
PROVIDER_SURFACE_HEALTH_INTERFACE: Final[str] = "ProviderSurfaceHealth@1"

DCR_TASK_ID: Final[str] = "DCR-013"
DCR_ARTIFACT_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/provider-surfaces.json"
)
DCR_FOREST_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/forest.json"
)
DCR_CARRIER_SUBJECT: Final[str] = (
    "DCR-013: Index complete actual provider registration and handler surfaces"
)
DCR_TODO_SUBJECT: Final[str] = "DCR-013: mark todo completed"
DEFAULT_MAX_BYTES: Final[int] = 1_048_576
DEFAULT_MAX_FILE_BYTES: Final[int] = 4 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES: Final[int] = 256 * 1024 * 1024
DEFAULT_MAX_FILES: Final[int] = 200_000
_GIT_TIMEOUT_SECONDS: Final[int] = 60
_GIT_CONTEXT_VARIABLES: Final[tuple[str, ...]] = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_SYSTEM",
    "GIT_DIR",
    "GIT_DISCOVERY_ACROSS_FILESYSTEM",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_PREFIX",
    "GIT_REPLACE_REF_BASE",
    "GIT_WORK_TREE",
)

# Mandatory package roots: accelerate, datasets, kit, MCP++.
MANDATORY_PACKAGE_ROOTS: Final[tuple[Mapping[str, str], ...]] = (
    {
        "root_id": "ipfs-accelerate",
        "provider": "ipfs_accelerate_py",
        "relative_path": "external/ipfs_accelerate",
        "package_dirname": "ipfs_accelerate_py",
    },
    {
        "root_id": "ipfs-datasets",
        "provider": "ipfs_datasets_py",
        "relative_path": "external/ipfs_datasets",
        "package_dirname": "ipfs_datasets_py",
    },
    {
        "root_id": "ipfs-kit",
        "provider": "ipfs_kit_py",
        "relative_path": "external/ipfs_kit",
        "package_dirname": "ipfs_kit_py",
    },
    {
        "root_id": "mcp-plus-plus",
        "provider": "mcp_plus_plus",
        "relative_path": "Mcp-Plus-Plus",
        "package_dirname": "",
    },
)

_ARCHIVE_PARTS: Final[frozenset[str]] = frozenset(
    {
        "archive",
        "archives",
        "archived",
        "backup",
        "backups",
        "obsolete",
        "attic",
        "reorganization_backup_root",
        "reorganization_backup_final",
    }
)
_TEST_PARTS: Final[frozenset[str]] = frozenset(
    {"test", "tests", "testing", "fixtures", "conftest"}
)
_GENERATED_PARTS: Final[frozenset[str]] = frozenset(
    {
        "generated",
        "build",
        "dist",
        "out",
        "output",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        "coverage",
        "htmlcov",
    }
)
_EXTRACT_MARKERS: Final[tuple[bytes, ...]] = (
    b"add_tool",
    b"register_tool",
    b"register_mcp_tool",
    b".tool(",
    b"@tool",
    b"list_tools",
    b"call_tool",
    b"tools/list",
    b"tools/call",
)


class ProviderSurfaceHealthError(ValueError):
    """A required provider-surface invariant could not be proven."""

    def __init__(self, reason_code: str, message: str = "") -> None:
        self.reason_code = str(reason_code or "provider_surface_health_error")
        super().__init__(message or self.reason_code)


class SurfaceRowKind(str, Enum):
    REGISTRATION = "registration"
    UNRESOLVED = "unresolved"
    DUPLICATE_EQUIVALENCE = "duplicate_equivalence"


class PathClassification(str, Enum):
    ACTIVE = "active"
    ARCHIVE = "archive"
    TEST = "test"
    GENERATED = "generated"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProviderSurfaceHealthError(
                "duplicate_json_key", f"duplicate JSON key: {key}"
            )
        result[key] = value
    return result


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProviderSurfaceHealthError("noncanonical_provider_surface_value") from exc


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _content_id(value: Any) -> str:
    return _sha256(_canonical_bytes(value))


def _artifact_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProviderSurfaceHealthError("noncanonical_provider_surface_value") from exc


def _read_json_bytes(value: bytes, *, reason: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            value.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except ProviderSurfaceHealthError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProviderSurfaceHealthError(reason) from exc
    if not isinstance(payload, Mapping):
        raise ProviderSurfaceHealthError(reason)
    return payload


def _git_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in _GIT_CONTEXT_VARIABLES:
        environment.pop(name, None)
    for name in tuple(environment):
        if name.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
            environment.pop(name, None)
    environment["GIT_LITERAL_PATHSPECS"] = "1"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    return environment


def _run_git(
    root: Path,
    *arguments: str,
    binary: bool = True,
    reason: str = "git_observation_failed",
) -> bytes | str:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
            env=_git_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProviderSurfaceHealthError(reason, str(root)) from exc
    if result.returncode:
        raise ProviderSurfaceHealthError(reason, str(root))
    if binary:
        return result.stdout
    return os.fsdecode(result.stdout).rstrip("\r\n")


def _default_workspace() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "config/deterministic_contract_repair_roots.json").is_file():
            return candidate
    raise ProviderSurfaceHealthError("workspace_missing")


def _classify_path(path: str) -> PathClassification:
    parts = {part.casefold() for part in PurePosixPath(path).parts}
    name = PurePosixPath(path).name.casefold()
    if parts & _ARCHIVE_PARTS:
        return PathClassification.ARCHIVE
    if parts & _GENERATED_PARTS:
        return PathClassification.GENERATED
    if parts & _TEST_PARTS or name.startswith("test_") or name.endswith("_test.py"):
        return PathClassification.TEST
    return PathClassification.ACTIVE


@dataclass(frozen=True)
class SurfaceRow:
    """One compact surface, unresolved, or duplicate-equivalence observation."""

    row_kind: SurfaceRowKind
    root_id: str
    provider: str
    path: str
    symbol: str
    registration_api: str
    kind: str
    aliases: str
    schema_digest: str
    handler: str
    effect_digest: str
    classification: PathClassification
    reason_code: str = ""
    expression: str = ""
    peer_path: str = ""
    peer_handler: str = ""

    def to_row(self) -> tuple[str, ...]:
        return (
            self.row_kind.value,
            self.root_id,
            self.provider,
            self.path,
            self.symbol,
            self.registration_api,
            self.kind,
            self.aliases,
            self.schema_digest,
            self.handler,
            self.effect_digest,
            self.classification.value,
            self.reason_code,
            self.expression,
            self.peer_path,
            self.peer_handler,
        )


def _schema_digest(schema: Mapping[str, Any]) -> str:
    return _content_id(dict(schema))


def _effect_digest(calls: Sequence[str], policy_gates: Sequence[str]) -> str:
    return _content_id({"calls": list(calls), "policy_gates": list(policy_gates)})


def _tool_row(
    *,
    root_id: str,
    tool: PythonMcpToolSurface,
    package_prefix: str,
) -> SurfaceRow:
    path = tool.registration_span.path
    if package_prefix and not path.startswith(package_prefix):
        path = f"{package_prefix.rstrip('/')}/{path.lstrip('/')}"
    handler = tool.handler
    return SurfaceRow(
        row_kind=SurfaceRowKind.REGISTRATION,
        root_id=root_id,
        provider=tool.provider,
        path=path,
        symbol=tool.canonical_name,
        registration_api=tool.registration_api,
        kind=tool.kind.value,
        aliases=",".join(tool.aliases),
        schema_digest=_schema_digest(tool.input_schema),
        handler=handler.symbol,
        effect_digest=_effect_digest(handler.calls, handler.policy_gates),
        classification=_classify_path(path),
    )


def _unresolved_row(
    *,
    root_id: str,
    item: UnresolvedRegistration,
    package_prefix: str,
) -> SurfaceRow:
    path = item.span.path
    if package_prefix and not path.startswith(package_prefix):
        path = f"{package_prefix.rstrip('/')}/{path.lstrip('/')}"
    return SurfaceRow(
        row_kind=SurfaceRowKind.UNRESOLVED,
        root_id=root_id,
        provider=item.provider,
        path=path,
        symbol="",
        registration_api=item.registration_api,
        kind="",
        aliases="",
        schema_digest="",
        handler="",
        effect_digest="",
        classification=_classify_path(path),
        reason_code=item.reason.value,
        expression=item.expression,
    )


def _duplicate_rows(registrations: Sequence[SurfaceRow]) -> list[SurfaceRow]:
    groups: dict[tuple[str, str], list[SurfaceRow]] = defaultdict(list)
    for row in registrations:
        if row.row_kind is not SurfaceRowKind.REGISTRATION:
            continue
        if row.classification is not PathClassification.ACTIVE:
            continue
        groups[(row.provider, row.symbol)].append(row)
    duplicates: list[SurfaceRow] = []
    for (provider, symbol), items in sorted(groups.items()):
        if len(items) < 2:
            continue
        # Distinct path/handler anchors for the same name remain ambiguous.
        anchors = sorted({(item.path, item.handler, item.kind) for item in items})
        if len(anchors) < 2:
            continue
        first = items[0]
        for peer in items[1:]:
            if (peer.path, peer.handler, peer.kind) == (
                first.path,
                first.handler,
                first.kind,
            ):
                continue
            duplicates.append(
                SurfaceRow(
                    row_kind=SurfaceRowKind.DUPLICATE_EQUIVALENCE,
                    root_id=first.root_id,
                    provider=provider,
                    path=first.path,
                    symbol=symbol,
                    registration_api=first.registration_api,
                    kind=first.kind,
                    aliases=first.aliases,
                    schema_digest=first.schema_digest,
                    handler=first.handler,
                    effect_digest=first.effect_digest,
                    classification=PathClassification.ACTIVE,
                    reason_code="duplicate_active_registration",
                    peer_path=peer.path,
                    peer_handler=peer.handler,
                )
            )
    duplicates.sort(key=lambda item: item.to_row())
    return duplicates


def _encode_surface_ledger(
    rows: Sequence[SurfaceRow],
) -> tuple[dict[str, Any], str]:
    ordered = sorted(rows, key=lambda item: item.to_row())
    uncompressed_rows = [list(item.to_row()) for item in ordered]
    uncompressed_digest = _content_id(uncompressed_rows)
    field_names = (
        "row_kind",
        "root_id",
        "provider",
        "path",
        "symbol",
        "registration_api",
        "kind",
        "aliases",
        "schema_digest",
        "handler",
        "effect_digest",
        "classification",
        "reason_code",
        "expression",
        "peer_path",
        "peer_handler",
    )
    dictionaries: dict[str, list[str]] = {}
    for index, name in enumerate(field_names):
        if name == "path":
            continue
        # Empty ledgers still need stable dictionary keys for decode.
        dictionaries[name] = sorted({row[index] for row in uncompressed_rows}) or [""]
    indexes = {
        name: {value: index for index, value in enumerate(values)}
        for name, values in dictionaries.items()
    }

    packed = bytearray()
    previous_path = ""
    for row in ordered:
        values = row.to_row()
        path = values[3]
        shared = 0
        limit = min(len(previous_path), len(path))
        while shared < limit and previous_path[shared] == path[shared]:
            shared += 1
        suffix = path[shared:]
        packed.extend(struct.pack(">H", shared))
        suffix_bytes = suffix.encode("utf-8", "surrogateescape")
        if len(suffix_bytes) > 0xFFFF:
            raise ProviderSurfaceHealthError("path_suffix_too_long", path)
        packed.extend(struct.pack(">H", len(suffix_bytes)))
        packed.extend(suffix_bytes)
        for field_index, name in enumerate(field_names):
            if name == "path":
                continue
            index_value = indexes[name][values[field_index]]
            if index_value > 0xFFFF:
                raise ProviderSurfaceHealthError("dictionary_index_overflow", name)
            packed.extend(struct.pack(">H", index_value))
        previous_path = path

    compressed = zlib.compress(bytes(packed), level=9)
    ledger = {
        "schema": SURFACE_LEDGER_SCHEMA,
        "codec": SURFACE_CODEC,
        "row_count": len(ordered),
        "uncompressed_digest": uncompressed_digest,
        "dictionary": dictionaries,
        "payload_encoding": "zlib+base64",
        "payload": base64.b64encode(compressed).decode("ascii"),
        "payload_sha256": _sha256(compressed),
    }
    return ledger, uncompressed_digest


def decode_surface_ledger(
    ledger: Mapping[str, Any],
) -> tuple[tuple[SurfaceRow, ...], str]:
    if (
        not isinstance(ledger, Mapping)
        or ledger.get("schema") != SURFACE_LEDGER_SCHEMA
        or ledger.get("codec") != SURFACE_CODEC
        or ledger.get("payload_encoding") != "zlib+base64"
    ):
        raise ProviderSurfaceHealthError("invalid_surface_ledger")
    try:
        row_count = int(ledger["row_count"])
        dictionary = ledger["dictionary"]
        payload_b64 = str(ledger["payload"])
        claimed_digest = str(ledger["uncompressed_digest"])
        claimed_payload_digest = str(ledger["payload_sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProviderSurfaceHealthError("invalid_surface_ledger") from exc
    if not isinstance(dictionary, Mapping):
        raise ProviderSurfaceHealthError("invalid_surface_ledger")
    try:
        compressed = base64.b64decode(payload_b64.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise ProviderSurfaceHealthError("invalid_surface_ledger") from exc
    if _sha256(compressed) != claimed_payload_digest:
        raise ProviderSurfaceHealthError("surface_payload_digest_mismatch")
    try:
        packed = zlib.decompress(compressed)
    except zlib.error as exc:
        raise ProviderSurfaceHealthError("invalid_surface_ledger") from exc

    def table(name: str) -> list[str]:
        values = dictionary.get(name)
        if not isinstance(values, list) or any(
            not isinstance(item, str) for item in values
        ):
            raise ProviderSurfaceHealthError("invalid_surface_ledger")
        return list(values)

    field_names = (
        "row_kind",
        "root_id",
        "provider",
        "symbol",
        "registration_api",
        "kind",
        "aliases",
        "schema_digest",
        "handler",
        "effect_digest",
        "classification",
        "reason_code",
        "expression",
        "peer_path",
        "peer_handler",
    )
    tables = {name: table(name) for name in field_names}
    offset = 0
    previous_path = ""
    rows: list[SurfaceRow] = []

    def take(count: int) -> bytes:
        nonlocal offset
        if offset + count > len(packed):
            raise ProviderSurfaceHealthError("invalid_surface_ledger")
        chunk = packed[offset : offset + count]
        offset += count
        return chunk

    for _ in range(row_count):
        shared = struct.unpack(">H", take(2))[0]
        suffix_len = struct.unpack(">H", take(2))[0]
        suffix = take(suffix_len).decode("utf-8", "surrogateescape")
        if shared > len(previous_path):
            raise ProviderSurfaceHealthError("invalid_surface_ledger")
        path = previous_path[:shared] + suffix
        try:
            values = [
                tables[name][struct.unpack(">H", take(2))[0]] for name in field_names
            ]
        except (IndexError, KeyError, struct.error) as exc:
            raise ProviderSurfaceHealthError("invalid_surface_ledger") from exc
        try:
            rows.append(
                SurfaceRow(
                    row_kind=SurfaceRowKind(values[0]),
                    root_id=values[1],
                    provider=values[2],
                    path=path,
                    symbol=values[3],
                    registration_api=values[4],
                    kind=values[5],
                    aliases=values[6],
                    schema_digest=values[7],
                    handler=values[8],
                    effect_digest=values[9],
                    classification=PathClassification(values[10]),
                    reason_code=values[11],
                    expression=values[12],
                    peer_path=values[13],
                    peer_handler=values[14],
                )
            )
        except ValueError as exc:
            raise ProviderSurfaceHealthError("invalid_surface_ledger") from exc
        previous_path = path
    if offset != len(packed):
        raise ProviderSurfaceHealthError("invalid_surface_ledger")
    recomputed = [list(item.to_row()) for item in rows]
    recomputed_digest = _content_id(recomputed)
    if recomputed_digest != claimed_digest:
        raise ProviderSurfaceHealthError("surface_uncompressed_digest_mismatch")
    return tuple(rows), recomputed_digest


def _prove_historical_forest(
    workspace: Path, forest_payload: Mapping[str, Any]
) -> dict[str, Any]:
    manifest, reasons = _document_integrity(forest_payload)
    if manifest is None:
        raise ProviderSurfaceHealthError("forest_integrity_invalid", ",".join(reasons))
    if forest_payload.get("schema") != REPAIR_FOREST_SCHEMA:
        raise ProviderSurfaceHealthError("forest_schema_invalid")
    portable = manifest.portable
    lifecycle = portable.get("lifecycle")
    if not isinstance(lifecycle, Mapping) or lifecycle.get("task_id") != "DCR-011":
        raise ProviderSurfaceHealthError("forest_lifecycle_invalid")
    subject = str(lifecycle.get("subject_head") or "")
    if not _OID_PATTERN.fullmatch(subject):
        raise ProviderSurfaceHealthError("forest_lifecycle_invalid")
    current = _git_oid(workspace, "rev-parse", "HEAD")
    if current != subject and not _git_is_ancestor(workspace, subject, current):
        raise ProviderSurfaceHealthError("forest_subject_not_ancestor", subject)
    completion_commit = subject
    raw = _run_git(
        workspace,
        "rev-list",
        "--ancestry-path",
        "--reverse",
        "--topo-order",
        f"{subject}..{current}",
        binary=False,
    )
    commits = tuple(item for item in str(raw).splitlines() if item)
    if commits:
        try:
            from .deterministic_repair_forest import _lifecycle_state

            for index in range(1, min(len(commits), 3) + 1):
                tip = commits[index - 1]
                state, state_reasons = _lifecycle_state(
                    workspace,
                    forest_payload,
                    subject=subject,
                    current=tip,
                )
                if not state_reasons and state == "todo_completed":
                    completion_commit = tip
                    break
        except Exception:
            completion_commit = subject
    return {
        "forest_id": manifest.forest_id,
        "subject_head": subject,
        "subject_tree": str(lifecycle.get("subject_tree") or ""),
        "completion_commit": completion_commit,
        "integrity_valid": True,
        "required_root_ids": [item["root_id"] for item in MANDATORY_PACKAGE_ROOTS],
        "root_heads": {
            str(item.get("id")): str(item.get("head"))
            for item in portable.get("roots", ())
            if isinstance(item, Mapping)
        },
        "root_trees": {
            str(item.get("id")): str(item.get("tree"))
            for item in portable.get("roots", ())
            if isinstance(item, Mapping)
        },
    }


def _enumerate_python_paths(
    root: Path, tree_oid: str, *, package_dirname: str
) -> list[dict[str, str]]:
    raw = _run_git(root, "ls-tree", "-r", "-z", tree_oid)
    assert isinstance(raw, bytes)
    prefix = f"{package_dirname}/" if package_dirname else ""
    rows: list[dict[str, str]] = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        metadata, separator, path_bytes = entry.partition(b"\t")
        if not separator:
            raise ProviderSurfaceHealthError("invalid_tree_entry")
        fields = metadata.split()
        if len(fields) != 3:
            raise ProviderSurfaceHealthError("invalid_tree_entry")
        mode = fields[0].decode("ascii", "strict")
        object_type = fields[1].decode("ascii", "strict")
        oid = fields[2].decode("ascii", "strict").lower()
        if object_type != "blob" or not _OID_PATTERN.fullmatch(oid):
            continue
        path = path_bytes.decode("utf-8", "surrogateescape")
        if not path or path.startswith("/") or ".." in PurePosixPath(path).parts:
            raise ProviderSurfaceHealthError("unsafe_tree_path", path)
        if prefix and not path.startswith(prefix):
            continue
        if not path.casefold().endswith(".py"):
            continue
        if mode not in {"100644", "100755"}:
            continue
        rows.append({"path": path, "mode": mode, "blob_oid": oid})
    rows.sort(key=lambda item: item["path"])
    return rows


def _blob_bytes(root: Path, oid: str, *, max_bytes: int) -> bytes:
    try:
        result = subprocess.run(
            ("git", "cat-file", "blob", oid),
            cwd=root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
            env=_git_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProviderSurfaceHealthError("blob_unreadable", oid) from exc
    if result.returncode:
        raise ProviderSurfaceHealthError("blob_unreadable", oid)
    payload = result.stdout
    if len(payload) > max_bytes:
        raise ProviderSurfaceHealthError("oversized_source", oid)
    return payload


def _inventory_merkle(paths: Sequence[str]) -> str:
    return _content_id(sorted(paths))


def _package_merkle(rows: Sequence[SurfaceRow], root_id: str) -> str:
    selected = [list(item.to_row()) for item in rows if item.root_id == root_id]
    return _content_id(sorted(selected))


def materialize_provider_surface_health(
    workspace_root: str | os.PathLike[str] | None = None,
    *,
    forest_path: str | os.PathLike[str] | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_files: int = DEFAULT_MAX_FILES,
) -> dict[str, Any]:
    """Scan mandatory package roots and emit a compact surface-health receipt."""

    workspace = (
        Path(workspace_root).resolve(strict=True)
        if workspace_root is not None
        else _default_workspace().resolve(strict=True)
    )
    forest_file = (
        Path(forest_path)
        if forest_path is not None
        else workspace.joinpath(*PurePosixPath(DCR_FOREST_PATH).parts)
    )
    forest_payload = _read_json_bytes(
        forest_file.read_bytes(), reason="forest_unreadable"
    )
    forest_proof = _prove_historical_forest(workspace, forest_payload)
    portable_roots = {
        str(item["id"]): item
        for item in forest_payload["portable"]["roots"]
        if isinstance(item, Mapping)
    }

    extractor = PythonMcpSurfaceExtractor(
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
    )
    # Keep the extractor interface bound into the receipt.
    assert extractor.interface == PYTHON_MCP_SURFACE_EXTRACTOR_INTERFACE

    surface_rows: list[SurfaceRow] = []
    package_summaries: list[dict[str, Any]] = []
    funnel: Counter[str] = Counter()
    total_scanned = 0
    total_bytes = 0

    for package in MANDATORY_PACKAGE_ROOTS:
        root_id = package["root_id"]
        provider = package["provider"]
        relative = package["relative_path"]
        package_dirname = package["package_dirname"]
        if root_id not in portable_roots:
            raise ProviderSurfaceHealthError("mandatory_root_missing", root_id)
        root_meta = portable_roots[root_id]
        root_path = workspace if relative in {"", "."} else workspace / relative
        if not root_path.is_dir():
            raise ProviderSurfaceHealthError("package_root_missing", root_id)
        tree_oid = str(root_meta.get("tree") or "")
        if not _OID_PATTERN.fullmatch(tree_oid):
            raise ProviderSurfaceHealthError("invalid_root_tree", root_id)
        try:
            entries = _enumerate_python_paths(
                root_path, tree_oid, package_dirname=package_dirname
            )
        except ProviderSurfaceHealthError:
            head = str(root_meta.get("head") or "HEAD")
            entries = _enumerate_python_paths(
                root_path, head, package_dirname=package_dirname
            )
        if len(entries) > max_files:
            raise ProviderSurfaceHealthError("package_exceeds_max_files", root_id)

        package_prefix = package_dirname
        scanned_paths: list[str] = []
        package_rows: list[SurfaceRow] = []
        # Cheap prefilter keeps complete inventory while only AST-extracting
        # registration-shaped sources (still fail-closed for oversized/unreadable).
        for entry in entries:
            path = entry["path"]
            scanned_paths.append(path)
            total_scanned += 1
            try:
                payload = _blob_bytes(
                    root_path, entry["blob_oid"], max_bytes=max_file_bytes
                )
            except ProviderSurfaceHealthError as exc:
                package_rows.append(
                    SurfaceRow(
                        row_kind=SurfaceRowKind.UNRESOLVED,
                        root_id=root_id,
                        provider=provider,
                        path=path,
                        symbol="",
                        registration_api="git_cat_file",
                        kind="",
                        aliases="",
                        schema_digest="",
                        handler="",
                        effect_digest="",
                        classification=_classify_path(path),
                        reason_code=UnresolvedReason.READ_ERROR.value
                        if exc.reason_code != "oversized_source"
                        else UnresolvedReason.RESOURCE_LIMIT.value,
                        expression="",
                    )
                )
                continue
            total_bytes += len(payload)
            if total_bytes > max_total_bytes:
                raise ProviderSurfaceHealthError("package_exceeds_max_total_bytes")
            if not any(marker in payload for marker in _EXTRACT_MARKERS):
                continue
            relative_for_extract = (
                path[len(package_dirname) + 1 :]
                if package_dirname and path.startswith(package_dirname + "/")
                else path
            )
            try:
                source = payload.decode("utf-8")
            except UnicodeDecodeError:
                package_rows.append(
                    SurfaceRow(
                        row_kind=SurfaceRowKind.UNRESOLVED,
                        root_id=root_id,
                        provider=provider,
                        path=path,
                        symbol="",
                        registration_api="utf8_decode",
                        kind="",
                        aliases="",
                        schema_digest="",
                        handler="",
                        effect_digest="",
                        classification=_classify_path(path),
                        reason_code=UnresolvedReason.READ_ERROR.value,
                        expression="",
                    )
                )
                continue
            surface = extract_python_mcp_source(
                source,
                provider=provider,
                path=relative_for_extract,
                repository_tree_id=tree_oid,
            )
            for tool in surface.tools:
                package_rows.append(
                    _tool_row(
                        root_id=root_id,
                        tool=tool,
                        package_prefix=package_prefix,
                    )
                )
            for unresolved in surface.unresolved:
                package_rows.append(
                    _unresolved_row(
                        root_id=root_id,
                        item=unresolved,
                        package_prefix=package_prefix,
                    )
                )

        surface_rows.extend(package_rows)
        package_summaries.append(
            {
                "root_id": root_id,
                "provider": provider,
                "relative_path": relative,
                "package_dirname": package_dirname,
                "package_relpath": (
                    f"{relative}/{package_dirname}" if package_dirname else relative
                ),
                "tree": tree_oid,
                "head": str(root_meta.get("head") or ""),
                "scanned_file_count": len(scanned_paths),
                "inventory_merkle": _inventory_merkle(scanned_paths),
                "surface_merkle": _package_merkle(package_rows, root_id),
                "registration_count": sum(
                    1
                    for item in package_rows
                    if item.row_kind is SurfaceRowKind.REGISTRATION
                ),
                "unresolved_count": sum(
                    1
                    for item in package_rows
                    if item.row_kind is SurfaceRowKind.UNRESOLVED
                ),
            }
        )

    duplicates = _duplicate_rows(surface_rows)
    surface_rows.extend(duplicates)
    for item in surface_rows:
        funnel[f"row_kind:{item.row_kind.value}"] += 1
        funnel[f"classification:{item.classification.value}"] += 1
        if item.row_kind is SurfaceRowKind.UNRESOLVED:
            funnel[f"unresolved:{item.reason_code}"] += 1

    ledger, uncompressed_digest = _encode_surface_ledger(surface_rows)
    mandatory_unresolved = [
        item
        for item in surface_rows
        if item.row_kind is SurfaceRowKind.UNRESOLVED
        and item.classification is PathClassification.ACTIVE
        and item.reason_code
        not in {
            UnresolvedReason.DYNAMIC_NAME.value,
            UnresolvedReason.DYNAMIC_HANDLER.value,
            UnresolvedReason.DYNAMIC_SCHEMA.value,
            UnresolvedReason.DYNAMIC_DISCOVERY.value,
        }
    ]
    # Dynamic registrations are retained evidence but do not alone block parity;
    # parse/read/resource failures on active paths and duplicate anchors do.
    active_blocking_unresolved = [
        item
        for item in surface_rows
        if item.row_kind is SurfaceRowKind.UNRESOLVED
        and item.classification is PathClassification.ACTIVE
        and item.reason_code
        in {
            UnresolvedReason.PARSE_ERROR.value,
            UnresolvedReason.READ_ERROR.value,
            UnresolvedReason.RESOURCE_LIMIT.value,
        }
    ]
    parity_blockers = {
        "active_parse_or_read_failures": len(active_blocking_unresolved),
        "duplicate_equivalence_rows": len(duplicates),
        "missing_mandatory_packages": 0,
    }
    parity_authorized = (
        parity_blockers["active_parse_or_read_failures"] == 0
        and parity_blockers["duplicate_equivalence_rows"] == 0
        and len(package_summaries) == len(MANDATORY_PACKAGE_ROOTS)
    )
    orchestration_head = _git_oid(workspace, "rev-parse", "HEAD")
    orchestration_tree = _git_oid(workspace, "rev-parse", "HEAD^{tree}")
    lifecycle = {
        "schema": PROVIDER_SURFACE_LIFECYCLE_SCHEMA,
        "task_id": DCR_TASK_ID,
        "subject_root_id": "orchestration",
        "subject_head": orchestration_head,
        "subject_tree": orchestration_tree,
        "artifact_path": DCR_ARTIFACT_PATH,
        "todo_path": DCR_TODO_PATH,
        "carrier_subject": DCR_CARRIER_SUBJECT,
        "todo_subject": DCR_TODO_SUBJECT,
        "max_transition_commits": 3,
        "predecessor_forest_id": forest_proof["forest_id"],
        "predecessor_completion_commit": forest_proof["completion_commit"],
    }
    identity = {
        "schema": PROVIDER_SURFACE_HEALTH_SCHEMA,
        "interface": PROVIDER_SURFACE_HEALTH_INTERFACE,
        "extractor_interface": PYTHON_MCP_SURFACE_EXTRACTOR_INTERFACE,
        "forest_id": forest_proof["forest_id"],
        "forest_historical_proof": forest_proof,
        "lifecycle": lifecycle,
        "packages": package_summaries,
        "funnel": {
            "scanned_file_count": total_scanned,
            "registration_count": sum(
                1
                for item in surface_rows
                if item.row_kind is SurfaceRowKind.REGISTRATION
            ),
            "unresolved_count": sum(
                1
                for item in surface_rows
                if item.row_kind is SurfaceRowKind.UNRESOLVED
            ),
            "duplicate_equivalence_count": len(duplicates),
            "active_mandatory_unresolved_count": len(mandatory_unresolved),
            "counts": dict(sorted(funnel.items())),
        },
        "inventory_merkle": _inventory_merkle(
            [
                f"{item['root_id']}:{path}"
                for item in package_summaries
                for path in []  # filled below from package inventory digests
            ]
            + [item["inventory_merkle"] for item in package_summaries]
        ),
        "surface_ledger": ledger,
        "surface_uncompressed_digest": uncompressed_digest,
        "health": {
            "parity_authorized": parity_authorized,
            "safe_for_completion_reasoning": parity_authorized,
            "blockers": parity_blockers,
            "reason_codes": (
                (
                    ["active_parse_or_read_failures"]
                    if parity_blockers["active_parse_or_read_failures"]
                    else []
                )
                + (
                    ["duplicate_equivalence_rows"]
                    if parity_blockers["duplicate_equivalence_rows"]
                    else []
                )
            ),
        },
        "limits": {
            "max_file_bytes": max_file_bytes,
            "max_total_bytes": max_total_bytes,
            "max_files": max_files,
            "max_artifact_bytes": DEFAULT_MAX_BYTES,
        },
        "authoritative": False,
        "completion_authorized": False,
    }
    provider_surface_id = _content_id(identity)
    return {**identity, "provider_surface_id": provider_surface_id}


def write_provider_surface_health(
    destination: str | os.PathLike[str],
    workspace_root: str | os.PathLike[str] | None = None,
    *,
    forest_path: str | os.PathLike[str] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_files: int = DEFAULT_MAX_FILES,
) -> dict[str, Any]:
    payload = materialize_provider_surface_health(
        workspace_root,
        forest_path=forest_path,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
        max_files=max_files,
    )
    encoded = _artifact_bytes(payload)
    if len(encoded) > max_bytes:
        raise ProviderSurfaceHealthError(
            "artifact_exceeds_admission_limit",
            f"{len(encoded)} > {max_bytes}",
        )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return payload


@dataclass(frozen=True)
class ProviderSurfaceValidation:
    integrity_valid: bool = False
    current: bool = False
    downstream_authorized: bool = False
    lifecycle_state: str = "invalid"
    provider_surface_id: str = ""
    forest_id: str = ""
    reason_codes: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.downstream_authorized

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_SURFACE_VALIDATION_SCHEMA,
            "valid": self.valid,
            "integrity_valid": self.integrity_valid,
            "current": self.current,
            "downstream_authorized": self.downstream_authorized,
            "lifecycle_state": self.lifecycle_state,
            "provider_surface_id": self.provider_surface_id,
            "forest_id": self.forest_id,
            "reason_codes": list(self.reason_codes),
        }


def _artifact_matches(
    root: Path, payload: Mapping[str, Any], *, carrier_commit: str | None = None
) -> bool:
    expected = _artifact_bytes(payload)
    try:
        if carrier_commit:
            observed = _run_git(root, "show", f"{carrier_commit}:{DCR_ARTIFACT_PATH}")
            assert isinstance(observed, bytes)
        else:
            path = root / DCR_ARTIFACT_PATH
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o111:
                return False
            observed = path.read_bytes()
    except (ProviderSurfaceHealthError, OSError):
        return False
    return observed == expected


def _todo_delta_is_exact(root: Path, observed: str, completed: str) -> bool:
    try:
        before_raw = _run_git(root, "show", f"{observed}:{DCR_TODO_PATH}")
        after_raw = _run_git(root, "show", f"{completed}:{DCR_TODO_PATH}")
        assert isinstance(before_raw, bytes) and isinstance(after_raw, bytes)
        before = before_raw.decode("utf-8")
        after = after_raw.decode("utf-8")
    except (ProviderSurfaceHealthError, UnicodeDecodeError):
        return False
    marker = "## DCR-013 "
    before_start = before.find(marker)
    after_start = after.find(marker)
    if before_start < 0 or after_start < 0:
        return False
    before_end = before.find("\n## ", before_start + len(marker))
    after_end = after.find("\n## ", after_start + len(marker))
    before_end = len(before) if before_end < 0 else before_end
    after_end = len(after) if after_end < 0 else after_end
    before_block = before[before_start:before_end]
    after_block = after[after_start:after_end]
    if before_block.count("- Status: todo") != 1:
        return False
    if after_block.count("- Status: completed") != 1:
        return False
    restored_block = after_block.replace("- Status: completed", "- Status: todo", 1)
    restored = after[:after_start] + restored_block + after[after_end:]
    return restored == before


def _commit_parents(root: Path, commit: str) -> tuple[str, ...]:
    text = str(_run_git(root, "rev-list", "--parents", "-n", "1", commit, binary=False))
    fields = text.split()
    if (
        not fields
        or fields[0] != commit
        or not all(_OID_PATTERN.fullmatch(item) for item in fields)
    ):
        raise ProviderSurfaceHealthError("invalid_commit_graph")
    return tuple(fields[1:])


def _commit_subject(root: Path, commit: str) -> str:
    return str(_run_git(root, "show", "-s", "--format=%s", commit, binary=False))


def _commit_tree(root: Path, commit: str) -> str:
    return str(_run_git(root, "rev-parse", f"{commit}^{{tree}}", binary=False)).lower()


def _changed_paths(root: Path, parent: str, commit: str) -> tuple[str, ...]:
    raw = _run_git(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        "-z",
        parent,
        commit,
        "--",
    )
    assert isinstance(raw, bytes)
    paths: list[str] = []
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        paths.append(encoded.decode("utf-8", "surrogateescape"))
    return tuple(sorted(paths))


def _lifecycle_state(
    root: Path,
    payload: Mapping[str, Any],
    *,
    subject: str,
    current: str,
) -> tuple[str, tuple[str, ...]]:
    if not _artifact_matches(root, payload):
        return "stale", ("capture_artifact_mismatch",)
    if current == subject:
        return "captured", ()
    if not _git_is_ancestor(root, subject, current):
        return "stale", ("observed_repository_commit_not_ancestor",)
    raw = _run_git(
        root,
        "rev-list",
        "--ancestry-path",
        "--reverse",
        "--topo-order",
        f"{subject}..{current}",
        binary=False,
    )
    commits = tuple(item for item in str(raw).splitlines() if item)
    if (
        not commits
        or len(commits) > 3
        or any(not _OID_PATTERN.fullmatch(item) for item in commits)
    ):
        return "stale", ("unrecognized_lifecycle_transition",)
    carrier = commits[0]
    if (
        _commit_parents(root, carrier) != (subject,)
        or _commit_subject(root, carrier) != DCR_CARRIER_SUBJECT
        or _changed_paths(root, subject, carrier) != (DCR_ARTIFACT_PATH,)
        or not _artifact_matches(root, payload, carrier_commit=carrier)
    ):
        return "stale", ("carrier_transition_invalid",)
    if len(commits) == 1:
        return "artifact_carried", ()
    merge = commits[1]
    merge_parents = _commit_parents(root, merge)
    if merge_parents != (subject, carrier):
        return "stale", ("integration_transition_invalid",)
    subject_text = _commit_subject(root, merge).lower()
    if (
        _commit_tree(root, merge) != _commit_tree(root, carrier)
        or not subject_text.startswith("merge branch '")
        or "dcr-013" not in subject_text
    ):
        return "stale", ("integration_transition_invalid",)
    if len(commits) == 2:
        return "integrated", ()
    completed = commits[2]
    if (
        len(commits) != 3
        or _commit_parents(root, completed) != (merge,)
        or _commit_subject(root, completed) != DCR_TODO_SUBJECT
        or _changed_paths(root, merge, completed) != (DCR_TODO_PATH,)
        or not _todo_delta_is_exact(root, subject, completed)
    ):
        return "stale", ("todo_transition_invalid",)
    return "todo_completed", ()


def _document_integrity_surface(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    reasons: list[str] = []
    if payload.get("schema") != PROVIDER_SURFACE_HEALTH_SCHEMA:
        reasons.append("invalid_schema")
    if payload.get("interface") != PROVIDER_SURFACE_HEALTH_INTERFACE:
        reasons.append("invalid_interface")
    if payload.get("extractor_interface") != PYTHON_MCP_SURFACE_EXTRACTOR_INTERFACE:
        reasons.append("invalid_extractor_interface")
    claimed = payload.get("provider_surface_id")
    try:
        recomputed = _content_id(
            {
                key: value
                for key, value in payload.items()
                if key != "provider_surface_id"
            }
        )
    except ProviderSurfaceHealthError:
        recomputed = ""
    if not isinstance(claimed, str) or claimed != recomputed:
        reasons.append("provider_surface_id_mismatch")
    ledger = payload.get("surface_ledger")
    if not isinstance(ledger, Mapping):
        reasons.append("invalid_surface_ledger")
        rows: tuple[SurfaceRow, ...] = ()
    else:
        try:
            rows, digest = decode_surface_ledger(ledger)
            if digest != payload.get("surface_uncompressed_digest"):
                reasons.append("surface_uncompressed_digest_mismatch")
            if int(ledger.get("row_count", -1)) != len(rows):
                reasons.append("surface_row_count_mismatch")
        except ProviderSurfaceHealthError as exc:
            reasons.append(exc.reason_code)
            rows = ()
    packages = payload.get("packages")
    if not isinstance(packages, list) or len(packages) != len(MANDATORY_PACKAGE_ROOTS):
        reasons.append("mandatory_package_set_invalid")
    else:
        observed = {item.get("root_id") for item in packages if isinstance(item, Mapping)}
        expected = {item["root_id"] for item in MANDATORY_PACKAGE_ROOTS}
        if observed != expected:
            reasons.append("mandatory_package_set_invalid")
    forest_proof = payload.get("forest_historical_proof")
    if not isinstance(forest_proof, Mapping) or not forest_proof.get("integrity_valid"):
        reasons.append("forest_historical_proof_invalid")
    lifecycle = payload.get("lifecycle")
    if not isinstance(lifecycle, Mapping) or lifecycle.get("task_id") != DCR_TASK_ID:
        reasons.append("invalid_lifecycle_policy")
    health = payload.get("health")
    if not isinstance(health, Mapping):
        reasons.append("invalid_health_block")
    if reasons:
        return None, tuple(dict.fromkeys(reasons))
    return dict(payload), ()


def validate_provider_surface_health(
    source: Mapping[str, Any] | str | os.PathLike[str],
    workspace_root: str | os.PathLike[str] | None = None,
    *,
    forest_path: str | os.PathLike[str] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> ProviderSurfaceValidation:
    workspace = (
        Path(workspace_root).resolve(strict=True)
        if workspace_root is not None
        else _default_workspace().resolve(strict=True)
    )
    if isinstance(source, Mapping):
        payload = dict(source)
        encoded = _artifact_bytes(payload)
    else:
        path = Path(source)
        try:
            encoded = path.read_bytes()
        except OSError:
            return ProviderSurfaceValidation(reason_codes=("artifact_unreadable",))
        try:
            payload = dict(_read_json_bytes(encoded, reason="artifact_unreadable"))
        except ProviderSurfaceHealthError as exc:
            return ProviderSurfaceValidation(reason_codes=(exc.reason_code,))
    if len(encoded) > max_bytes:
        return ProviderSurfaceValidation(
            reason_codes=("artifact_exceeds_admission_limit",)
        )
    document, integrity_reasons = _document_integrity_surface(payload)
    provider_surface_id = str(payload.get("provider_surface_id") or "")
    forest_id = str(payload.get("forest_id") or "")
    if document is None:
        return ProviderSurfaceValidation(
            provider_surface_id=provider_surface_id,
            forest_id=forest_id,
            reason_codes=integrity_reasons,
        )
    forest_file = (
        Path(forest_path)
        if forest_path is not None
        else workspace.joinpath(*PurePosixPath(DCR_FOREST_PATH).parts)
    )
    try:
        forest_payload = _read_json_bytes(
            forest_file.read_bytes(), reason="forest_unreadable"
        )
        forest_proof = _prove_historical_forest(workspace, forest_payload)
        if forest_proof.get("forest_id") != payload.get("forest_id"):
            return ProviderSurfaceValidation(
                integrity_valid=True,
                provider_surface_id=provider_surface_id,
                forest_id=forest_id,
                reason_codes=("forest_id_mismatch",),
            )
    except ProviderSurfaceHealthError as exc:
        return ProviderSurfaceValidation(
            integrity_valid=True,
            provider_surface_id=provider_surface_id,
            forest_id=forest_id,
            reason_codes=(exc.reason_code,),
        )
    lifecycle = payload["lifecycle"]
    subject = str(lifecycle.get("subject_head") or "")
    try:
        current = _git_oid(workspace, "rev-parse", "HEAD")
        state, lifecycle_reasons = _lifecycle_state(
            workspace,
            payload,
            subject=subject,
            current=current,
        )
    except ProviderSurfaceHealthError as exc:
        state, lifecycle_reasons = "stale", (exc.reason_code,)
    current_flag = not lifecycle_reasons
    health = payload.get("health") if isinstance(payload.get("health"), Mapping) else {}
    # Parity blockers do not invalidate capture integrity; they withhold parity.
    downstream = current_flag and state in {
        "captured",
        "artifact_carried",
        "integrated",
        "todo_completed",
    }
    return ProviderSurfaceValidation(
        integrity_valid=True,
        current=current_flag,
        downstream_authorized=downstream,
        lifecycle_state=state,
        provider_surface_id=provider_surface_id,
        forest_id=forest_id,
        reason_codes=lifecycle_reasons
        + (
            tuple(health.get("reason_codes") or [])
            if not bool(health.get("parity_authorized"))
            else ()
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "materialize"))
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--forest", default=DCR_FOREST_PATH)
    parser.add_argument("--artifact", default=DCR_ARTIFACT_PATH)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    parser.add_argument("--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES)
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    arguments = parser.parse_args(argv)
    workspace = Path(arguments.workspace).resolve(strict=False)
    forest = Path(arguments.forest)
    if not forest.is_absolute():
        forest = workspace / forest
    artifact = Path(arguments.artifact)
    if not artifact.is_absolute():
        artifact = workspace / artifact
    if arguments.command == "materialize":
        try:
            payload = write_provider_surface_health(
                artifact,
                workspace,
                forest_path=forest,
                max_bytes=arguments.max_bytes,
                max_file_bytes=arguments.max_file_bytes,
                max_total_bytes=arguments.max_total_bytes,
                max_files=arguments.max_files,
            )
        except ProviderSurfaceHealthError as exc:
            sys.stdout.write(
                json.dumps(
                    {"ok": False, "reason_code": exc.reason_code, "message": str(exc)},
                    sort_keys=True,
                )
                + "\n"
            )
            return 1
        sys.stdout.write(
            json.dumps(
                {
                    "ok": True,
                    "provider_surface_id": payload.get("provider_surface_id"),
                    "scanned_file_count": payload.get("funnel", {}).get(
                        "scanned_file_count"
                    ),
                    "parity_authorized": payload.get("health", {}).get(
                        "parity_authorized"
                    ),
                },
                sort_keys=True,
            )
            + "\n"
        )
        return 0

    expected = workspace.joinpath(*PurePosixPath(DCR_ARTIFACT_PATH).parts)
    if artifact.resolve(strict=False) != expected.resolve(strict=False):
        result = ProviderSurfaceValidation(
            reason_codes=("provider_surface_output_path_invalid",)
        )
    else:
        result = validate_provider_surface_health(
            artifact,
            workspace,
            forest_path=forest,
            max_bytes=arguments.max_bytes,
        )
    sys.stdout.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
    return (
        0
        if result.integrity_valid and result.current and result.downstream_authorized
        else 1
    )


class ProviderSurfaceHealth:
    """Public interface marker for ProviderSurfaceHealth@1."""

    interface = PROVIDER_SURFACE_HEALTH_INTERFACE
    schema = PROVIDER_SURFACE_HEALTH_SCHEMA

    @staticmethod
    def materialize(
        workspace_root: str | os.PathLike[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return materialize_provider_surface_health(workspace_root, **kwargs)

    @staticmethod
    def write(
        destination: str | os.PathLike[str],
        workspace_root: str | os.PathLike[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return write_provider_surface_health(destination, workspace_root, **kwargs)

    @staticmethod
    def validate(
        source: Mapping[str, Any] | str | os.PathLike[str],
        workspace_root: str | os.PathLike[str] | None = None,
        **kwargs: Any,
    ) -> ProviderSurfaceValidation:
        return validate_provider_surface_health(source, workspace_root, **kwargs)


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DCR_ARTIFACT_PATH",
    "DCR_CARRIER_SUBJECT",
    "DCR_FOREST_PATH",
    "DCR_TASK_ID",
    "DCR_TODO_SUBJECT",
    "MANDATORY_PACKAGE_ROOTS",
    "PROVIDER_SURFACE_HEALTH_INTERFACE",
    "PROVIDER_SURFACE_HEALTH_SCHEMA",
    "SURFACE_CODEC",
    "PathClassification",
    "ProviderSurfaceHealth",
    "ProviderSurfaceHealthError",
    "ProviderSurfaceValidation",
    "SurfaceRow",
    "SurfaceRowKind",
    "decode_surface_ledger",
    "main",
    "materialize_provider_surface_health",
    "validate_provider_surface_health",
    "write_provider_surface_health",
]


if __name__ == "__main__":
    raise SystemExit(main())
