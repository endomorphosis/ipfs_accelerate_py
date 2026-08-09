"""Fail-closed SwissKnife desktop expected-contract index for DCR-014.

Catalogs desktop MCP consumers, registries, descriptors, manifests, types,
UI/UX IR, ORB/IDL, tests, and call sites with explicit authority classes.
Large inventories are stored as compact dictionary/Merkle projections rather
than raw extractor dumps.  Inferred prose and archive/test/generated sources
cannot silently override reviewed declarations.
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Final

from .deterministic_repair_forest import (
    DCR_ROOT_IDS,
    DCR_TODO_PATH,
    REPAIR_FOREST_SCHEMA,
    _document_integrity,
    _git_is_ancestor,
    _git_oid,
    _OID_PATTERN,
)
from .mcp_contract_catalog import (
    CATALOG_VERSION,
    MCP_IDL_INTERFACE,
    ORB_INTERFACE,
    UIIR_DOCUMENT_INTERFACE,
    ContractSourceKind,
    McpContractCatalog,
    ReviewState,
    SourceAuthorityClass,
    authority_for_source_kind,
)
from .swissknife_contract_extractor import (
    SWISSKNIFE_EXTRACTOR_VERSION,
    SourceRole,
    SwissKnifeContractExtractor,
    SwissKnifeContractExtractorError,
    SwissKnifeSource,
)

DESKTOP_EXPECTATIONS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "deterministic-repair-desktop-expectations@1"
)
DESKTOP_EXPECTATIONS_VALIDATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "deterministic-repair-desktop-expectations-validation@1"
)
DESKTOP_EXPECTATIONS_LIFECYCLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "deterministic-repair-desktop-expectations-lifecycle@1"
)
DESKTOP_INVENTORY_LEDGER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "deterministic-repair-desktop-inventory-ledger@1"
)
DESKTOP_CONSUMER_LEDGER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "deterministic-repair-desktop-consumer-ledger@1"
)
DESKTOP_EXPECTATION_LEDGER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "deterministic-repair-desktop-expectation-ledger@1"
)
INVENTORY_CODEC: Final[str] = "dcr-desktop-inventory-dictionary-prefix@1"
CONSUMER_CODEC: Final[str] = "dcr-desktop-consumer-dictionary-prefix@1"
EXPECTATION_CODEC: Final[str] = "dcr-desktop-expectation-dictionary-prefix@1"
DESKTOP_EXPECTATION_INTERFACE: Final[str] = "DesktopExpectationIndex@1"
MCP_CONTRACT_CATALOG_INTERFACE: Final[str] = "McpContractCatalog@1"

DCR_TASK_ID: Final[str] = "DCR-014"
DCR_ARTIFACT_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/desktop-expectations.json"
)
DCR_FOREST_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/forest.json"
)
DCR_CARRIER_SUBJECT: Final[str] = (
    "DCR-014: Index SwissKnife desktop expected contracts and UI bindings"
)
DCR_TODO_SUBJECT: Final[str] = "DCR-014: mark todo completed"
DEFAULT_MAX_BYTES: Final[int] = 1_048_576
DEFAULT_MAX_FILE_BYTES: Final[int] = 4 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES: Final[int] = 64 * 1024 * 1024
DEFAULT_MAX_FILES: Final[int] = 4096
SWISSKNIFE_ROOT_ID: Final[str] = "swissknife"
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

_SOURCE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json"}
)
_DEFAULT_INCLUDE_GLOBS: Final[tuple[str, ...]] = (
    "src/services/mcp/**/*",
    "src/services/apps/**/*",
    "src/services/ipfs/**/*",
    "src/client/**/*",
    "contracts/**/*",
    "test/mcp-plus-plus/**/*",
    "tests/mcp-plus-plus/**/*",
    "scripts/**/*mcp*",
)
_ARCHIVE_MARKERS: Final[tuple[str, ...]] = (
    "/archive/",
    "/archives/",
    "/archived/",
    "/obsolete/",
    "/attic/",
    "/backup/",
    "/backups/",
    ".bak.",
    ".deps-bak/",
)
_GENERATED_MARKERS: Final[tuple[str, ...]] = (
    "/generated/",
    ".generated.",
    "_generated.",
    "auto-generated",
)
_TEST_MARKERS: Final[tuple[str, ...]] = (
    "/test/",
    "/tests/",
    "/__tests__/",
    ".test.",
    ".spec.",
    "_test.",
)
_UIIR_MARKERS: Final[tuple[str, ...]] = (
    "ui-ux-ir",
    "ui_ux_ir",
    "uiir",
    "mcp-ui-profile",
    "mcp-schema-ui",
)
_ORB_MARKERS: Final[tuple[str, ...]] = (
    "mcp-orb",
    "orb-idl",
    "orb_",
    "/orb.",
    "orb_idl",
)
_IDL_MARKERS: Final[tuple[str, ...]] = (
    "mcp-idl",
    "mcp_idl",
    "interface-descriptor",
    "interop-descriptor",
    "mcp-plus-plus.ts",
    "mcp-plus-plus-interop",
)


class DeterministicDesktopExpectationsError(ValueError):
    """Desktop expectation projection failed closed."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        message = reason_code if not detail else f"{reason_code}: {detail}"
        super().__init__(message)


class ConsumerClassification(str, Enum):
    ACTIVE = "active"
    TEST = "test"
    ARCHIVE = "archive"
    GENERATED = "generated"
    OTHER = "other"


class DeclarationKind(str, Enum):
    DESCRIPTOR = "descriptor"
    REGISTRY = "registry"
    MANIFEST = "manifest"
    SCHEMA = "schema"
    CONNECTOR = "connector"
    UI_IR = "ui_ir"
    ORB = "orb"
    IDL = "idl"
    TYPE = "type"
    CONTRACT_TEST = "contract_test"
    CALL_SITE = "call_site"
    OTHER = "other"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DeterministicDesktopExpectationsError(
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
        raise DeterministicDesktopExpectationsError(
            "noncanonical_desktop_expectations_value"
        ) from exc


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
        raise DeterministicDesktopExpectationsError(
            "noncanonical_desktop_expectations_value"
        ) from exc


def _read_json_bytes(value: bytes, *, reason: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            value.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except DeterministicDesktopExpectationsError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DeterministicDesktopExpectationsError(reason) from exc
    if not isinstance(payload, Mapping):
        raise DeterministicDesktopExpectationsError(reason)
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
        raise DeterministicDesktopExpectationsError(reason, str(root)) from exc
    if result.returncode:
        raise DeterministicDesktopExpectationsError(reason, str(root))
    if binary:
        return result.stdout
    return os.fsdecode(result.stdout).rstrip("\r\n")


def _default_workspace() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "config/deterministic_contract_repair_roots.json").is_file():
            return candidate
    raise DeterministicDesktopExpectationsError("workspace_missing")


def _classify_path(relative: str) -> ConsumerClassification:
    lowered = f"/{relative.replace(chr(92), '/').lower()}/"
    if any(marker in lowered for marker in _ARCHIVE_MARKERS):
        return ConsumerClassification.ARCHIVE
    if any(marker in lowered for marker in _GENERATED_MARKERS):
        return ConsumerClassification.GENERATED
    if any(marker in lowered for marker in _TEST_MARKERS):
        return ConsumerClassification.TEST
    return ConsumerClassification.ACTIVE


def _declaration_kind_for(relative: str, role: SourceRole) -> DeclarationKind:
    lowered = relative.replace("\\", "/").lower()
    if any(marker in lowered for marker in _UIIR_MARKERS):
        return DeclarationKind.UI_IR
    if any(marker in lowered for marker in _ORB_MARKERS):
        return DeclarationKind.ORB
    if any(marker in lowered for marker in _IDL_MARKERS):
        return DeclarationKind.IDL
    if role is SourceRole.DESCRIPTOR:
        return DeclarationKind.DESCRIPTOR
    if role is SourceRole.CAPABILITY_REGISTRY:
        return DeclarationKind.REGISTRY
    if role is SourceRole.MANIFEST:
        return DeclarationKind.MANIFEST
    if role is SourceRole.SCHEMA:
        return DeclarationKind.SCHEMA
    if role is SourceRole.CONNECTOR:
        return DeclarationKind.CONNECTOR
    if role is SourceRole.CONTRACT_TEST:
        return DeclarationKind.CONTRACT_TEST
    if role is SourceRole.APP_BINDING:
        return DeclarationKind.CALL_SITE
    if role is SourceRole.POLICY_MEDIATOR:
        return DeclarationKind.TYPE
    return DeclarationKind.OTHER


def _role_for_path(relative: str) -> SourceRole:
    lowered = relative.replace("\\", "/").lower()
    if any(marker in lowered for marker in _TEST_MARKERS):
        return SourceRole.CONTRACT_TEST
    if "capability-registry" in lowered or "capability_registry" in lowered:
        return SourceRole.CAPABILITY_REGISTRY
    if "manifest" in lowered:
        return SourceRole.MANIFEST
    if lowered.endswith(".json") or "schema" in lowered:
        return SourceRole.SCHEMA
    if "connector" in lowered:
        return SourceRole.CONNECTOR
    if "policy" in lowered or "mediator" in lowered or "mediation" in lowered:
        return SourceRole.POLICY_MEDIATOR
    if "app-binding" in lowered or "app_binding" in lowered or "/apps/" in lowered:
        return SourceRole.APP_BINDING
    if "descriptor" in lowered or "interop" in lowered or "mcp-plus-plus" in lowered:
        return SourceRole.DESCRIPTOR
    return SourceRole.OTHER


def _source_kind_for_role(role: SourceRole) -> ContractSourceKind:
    return {
        SourceRole.DESCRIPTOR: ContractSourceKind.MCP_IDL,
        SourceRole.SCHEMA: ContractSourceKind.JSON_SCHEMA,
        SourceRole.CAPABILITY_REGISTRY: ContractSourceKind.REGISTRATION,
        SourceRole.CONNECTOR: ContractSourceKind.TYPED_INTERFACE,
        SourceRole.POLICY_MEDIATOR: ContractSourceKind.POLICY_CONTRACT,
        SourceRole.APP_BINDING: ContractSourceKind.REGISTRATION,
        SourceRole.CONTRACT_TEST: ContractSourceKind.CONFORMANCE_TEST,
        SourceRole.MANIFEST: ContractSourceKind.MANIFEST,
        SourceRole.OTHER: ContractSourceKind.INFERRED_PROSE,
    }[role]


def _authority_for_classification(
    classification: ConsumerClassification,
    role: SourceRole,
) -> SourceAuthorityClass:
    if classification in {
        ConsumerClassification.ARCHIVE,
        ConsumerClassification.GENERATED,
        ConsumerClassification.TEST,
    }:
        # Tests may be conformance; archive/generated never authorize.
        if (
            classification is ConsumerClassification.TEST
            and role is SourceRole.CONTRACT_TEST
        ):
            return SourceAuthorityClass.CONFORMANCE
        return SourceAuthorityClass.NONE
    return authority_for_source_kind(_source_kind_for_role(role))


@dataclass(frozen=True)
class InventoryRow:
    path: str
    role: str
    classification: str
    declaration_kind: str
    content_digest: str
    byte_count: int

    def to_row(self) -> tuple[str, ...]:
        return (
            self.path,
            self.role,
            self.classification,
            self.declaration_kind,
            self.content_digest,
            str(self.byte_count),
        )


@dataclass(frozen=True)
class ConsumerRow:
    path: str
    role: str
    classification: str
    declaration_kind: str
    authority_class: str
    subject_count: int
    content_digest: str

    def to_row(self) -> tuple[str, ...]:
        return (
            self.path,
            self.role,
            self.classification,
            self.declaration_kind,
            self.authority_class,
            str(self.subject_count),
            self.content_digest,
        )


@dataclass(frozen=True)
class ExpectationRow:
    subject: str
    field_path: str
    declaration_kind: str
    authority_class: str
    review_state: str
    source_kind: str
    path: str
    transport: str
    ui_action: str
    version: str
    value_digest: str

    def to_row(self) -> tuple[str, ...]:
        return (
            self.subject,
            self.field_path,
            self.declaration_kind,
            self.authority_class,
            self.review_state,
            self.source_kind,
            self.path,
            self.transport,
            self.ui_action,
            self.version,
            self.value_digest,
        )


def _encode_prefix_ledger(
    *,
    schema: str,
    codec: str,
    rows: Sequence[tuple[str, ...]],
    path_index: int,
    dictionary_fields: Sequence[tuple[str, int]],
) -> tuple[dict[str, Any], str]:
    ordered = sorted(rows)
    uncompressed_digest = _content_id([list(item) for item in ordered])
    dictionaries: dict[str, list[str]] = {}
    indexes: dict[str, dict[str, int]] = {}
    for name, field_index in dictionary_fields:
        values = sorted({item[field_index] for item in ordered})
        dictionaries[name] = values
        indexes[name] = {value: index for index, value in enumerate(values)}

    packed = bytearray()
    previous_path = ""
    for item in ordered:
        path = item[path_index]
        shared = 0
        limit = min(len(previous_path), len(path))
        while shared < limit and previous_path[shared] == path[shared]:
            shared += 1
        suffix = path[shared:]
        packed.extend(struct.pack(">H", shared))
        suffix_bytes = suffix.encode("utf-8", "surrogateescape")
        packed.extend(struct.pack(">H", len(suffix_bytes)))
        packed.extend(suffix_bytes)
        for name, field_index in dictionary_fields:
            if field_index == path_index:
                continue
            value = item[field_index]
            if name in indexes:
                packed.append(indexes[name][value])
            else:
                encoded = value.encode("utf-8", "surrogateescape")
                packed.extend(struct.pack(">H", len(encoded)))
                packed.extend(encoded)
        # Remaining non-dictionary, non-path fields as length-prefixed strings.
        for field_index, value in enumerate(item):
            if field_index == path_index:
                continue
            if any(field_index == idx for _, idx in dictionary_fields):
                continue
            encoded = value.encode("utf-8", "surrogateescape")
            packed.extend(struct.pack(">H", len(encoded)))
            packed.extend(encoded)
        previous_path = path

    compressed = zlib.compress(bytes(packed), level=9)
    ledger = {
        "schema": schema,
        "codec": codec,
        "row_count": len(ordered),
        "uncompressed_digest": uncompressed_digest,
        "dictionary": dictionaries,
        "path_index": path_index,
        "dictionary_fields": [
            {"name": name, "index": index} for name, index in dictionary_fields
        ],
        "field_count": len(ordered[0]) if ordered else 0,
        "payload_encoding": "zlib+base64",
        "payload": base64.b64encode(compressed).decode("ascii"),
        "payload_sha256": _sha256(compressed),
    }
    return ledger, uncompressed_digest


def decode_prefix_ledger(
    ledger: Mapping[str, Any],
    *,
    expected_schema: str,
    expected_codec: str,
) -> tuple[tuple[tuple[str, ...], ...], str]:
    if (
        not isinstance(ledger, Mapping)
        or ledger.get("schema") != expected_schema
        or ledger.get("codec") != expected_codec
        or ledger.get("payload_encoding") != "zlib+base64"
    ):
        raise DeterministicDesktopExpectationsError("invalid_desktop_ledger")
    try:
        row_count = int(ledger["row_count"])
        dictionary = ledger["dictionary"]
        path_index = int(ledger["path_index"])
        field_count = int(ledger["field_count"])
        payload_b64 = str(ledger["payload"])
        claimed_digest = str(ledger["uncompressed_digest"])
        claimed_payload_digest = str(ledger["payload_sha256"])
        dict_fields_raw = ledger.get("dictionary_fields") or []
    except (KeyError, TypeError, ValueError) as exc:
        raise DeterministicDesktopExpectationsError("invalid_desktop_ledger") from exc
    if not isinstance(dictionary, Mapping) or not isinstance(dict_fields_raw, list):
        raise DeterministicDesktopExpectationsError("invalid_desktop_ledger")
    dictionary_fields = [
        (str(item["name"]), int(item["index"]))
        for item in dict_fields_raw
        if isinstance(item, Mapping)
    ]
    try:
        compressed = base64.b64decode(payload_b64.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise DeterministicDesktopExpectationsError("invalid_desktop_ledger") from exc
    if _sha256(compressed) != claimed_payload_digest:
        raise DeterministicDesktopExpectationsError("desktop_payload_digest_mismatch")
    try:
        packed = zlib.decompress(compressed)
    except zlib.error as exc:
        raise DeterministicDesktopExpectationsError("invalid_desktop_ledger") from exc

    def table(name: str) -> list[str]:
        values = dictionary.get(name)
        if not isinstance(values, list) or any(
            not isinstance(item, str) for item in values
        ):
            raise DeterministicDesktopExpectationsError("invalid_desktop_ledger")
        return list(values)

    tables = {name: table(name) for name, _ in dictionary_fields}
    offset = 0
    previous_path = ""
    rows: list[tuple[str, ...]] = []

    def take(count: int) -> bytes:
        nonlocal offset
        if offset + count > len(packed):
            raise DeterministicDesktopExpectationsError("invalid_desktop_ledger")
        chunk = packed[offset : offset + count]
        offset += count
        return chunk

    for _ in range(row_count):
        shared = struct.unpack(">H", take(2))[0]
        suffix_len = struct.unpack(">H", take(2))[0]
        suffix = take(suffix_len).decode("utf-8", "surrogateescape")
        if shared > len(previous_path):
            raise DeterministicDesktopExpectationsError("invalid_desktop_ledger")
        path = previous_path[:shared] + suffix
        values: list[str | None] = [None] * field_count
        values[path_index] = path
        for name, field_index in dictionary_fields:
            if field_index == path_index:
                continue
            index = take(1)[0]
            try:
                values[field_index] = tables[name][index]
            except IndexError as exc:
                raise DeterministicDesktopExpectationsError(
                    "invalid_desktop_ledger"
                ) from exc
        for field_index in range(field_count):
            if values[field_index] is not None:
                continue
            length = struct.unpack(">H", take(2))[0]
            values[field_index] = take(length).decode("utf-8", "surrogateescape")
        rows.append(tuple(str(item) for item in values))
        previous_path = path
    if offset != len(packed):
        raise DeterministicDesktopExpectationsError("invalid_desktop_ledger")
    recomputed = _content_id([list(item) for item in rows])
    if recomputed != claimed_digest:
        raise DeterministicDesktopExpectationsError(
            "desktop_uncompressed_digest_mismatch"
        )
    return tuple(rows), recomputed


def _merkle_root(rows: Sequence[tuple[str, ...]]) -> str:
    digests = [_content_id(list(row)) for row in sorted(rows)]
    if not digests:
        return _content_id([])
    while len(digests) > 1:
        nxt: list[str] = []
        for index in range(0, len(digests), 2):
            left = digests[index]
            right = digests[index + 1] if index + 1 < len(digests) else left
            nxt.append(_content_id([left, right]))
        digests = nxt
    return digests[0]


def _prove_historical_forest(
    workspace: Path, forest_payload: Mapping[str, Any]
) -> dict[str, Any]:
    manifest, reasons = _document_integrity(forest_payload)
    if manifest is None:
        raise DeterministicDesktopExpectationsError(
            "forest_integrity_invalid", ",".join(reasons)
        )
    if forest_payload.get("schema") != REPAIR_FOREST_SCHEMA:
        raise DeterministicDesktopExpectationsError("forest_schema_invalid")
    portable = manifest.portable
    lifecycle = portable.get("lifecycle")
    if not isinstance(lifecycle, Mapping) or lifecycle.get("task_id") != "DCR-011":
        raise DeterministicDesktopExpectationsError("forest_lifecycle_invalid")
    subject = str(lifecycle.get("subject_head") or "")
    if not _OID_PATTERN.fullmatch(subject):
        raise DeterministicDesktopExpectationsError("forest_lifecycle_invalid")
    current = _git_oid(workspace, "rev-parse", "HEAD")
    if current != subject and not _git_is_ancestor(workspace, subject, current):
        raise DeterministicDesktopExpectationsError(
            "forest_subject_not_ancestor", subject
        )
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
        "required_root_ids": list(DCR_ROOT_IDS),
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


def _swissknife_root(
    workspace: Path, forest_payload: Mapping[str, Any]
) -> tuple[Path, str]:
    manifest, reasons = _document_integrity(forest_payload)
    if manifest is None:
        raise DeterministicDesktopExpectationsError(
            "forest_integrity_invalid", ",".join(reasons)
        )
    roots = manifest.portable.get("roots")
    if not isinstance(roots, Sequence):
        raise DeterministicDesktopExpectationsError("swissknife_root_missing")
    for item in roots:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("id")) != SWISSKNIFE_ROOT_ID:
            continue
        relative = str(
            item.get("relative_path")
            or item.get("configured_path")
            or item.get("path")
            or "swissknife"
        )
        head = str(item.get("head") or "")
        root_path = workspace.joinpath(*PurePosixPath(relative).parts)
        if not root_path.is_dir():
            raise DeterministicDesktopExpectationsError(
                "swissknife_root_missing", relative
            )
        return root_path, head
    raise DeterministicDesktopExpectationsError("swissknife_root_missing")


def _collect_sources(
    swissknife_root: Path,
    *,
    include_paths: Sequence[str] | None = None,
    max_files: int = DEFAULT_MAX_FILES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> tuple[tuple[SwissKnifeSource, ...], tuple[InventoryRow, ...]]:
    patterns = tuple(include_paths or _DEFAULT_INCLUDE_GLOBS)
    selected: dict[str, Path] = {}
    for pattern in patterns:
        candidate = swissknife_root / pattern
        matches = (
            [candidate]
            if candidate.is_file()
            else list(swissknife_root.glob(pattern))
        )
        for match in matches:
            if not match.is_file() or match.suffix.lower() not in _SOURCE_SUFFIXES:
                continue
            if match.is_symlink():
                continue
            try:
                relative = match.resolve().relative_to(swissknife_root.resolve())
            except ValueError:
                continue
            selected[relative.as_posix()] = match
    if len(selected) > max_files:
        raise DeterministicDesktopExpectationsError(
            "file_limit_exceeded", f"{len(selected)} > {max_files}"
        )
    inventory: list[InventoryRow] = []
    sources: list[SwissKnifeSource] = []
    total = 0
    for relative, path in sorted(selected.items()):
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise DeterministicDesktopExpectationsError(
                "source_unreadable", relative
            ) from exc
        size = len(payload)
        if size > max_file_bytes:
            raise DeterministicDesktopExpectationsError(
                "file_byte_limit_exceeded", relative
            )
        total += size
        if total > max_total_bytes:
            raise DeterministicDesktopExpectationsError("total_source_byte_limit_exceeded")
        digest = _sha256(payload)
        role = _role_for_path(relative)
        classification = _classify_path(relative)
        declaration = _declaration_kind_for(relative, role)
        inventory.append(
            InventoryRow(
                path=relative,
                role=role.value,
                classification=classification.value,
                declaration_kind=declaration.value,
                content_digest=digest,
                byte_count=size,
            )
        )
        sources.append(
            SwissKnifeSource(
                path=relative,
                source=payload,
                source_version=digest,
                role=role,
            )
        )
    return tuple(sources), tuple(inventory)


def _expectation_rows(
    extraction: Any,
    inventory_by_path: Mapping[str, InventoryRow],
) -> tuple[ExpectationRow, ...]:
    rows: list[ExpectationRow] = []
    catalog: McpContractCatalog = extraction.catalog
    source_by_id = {source.source_id: source for source in catalog.sources}
    for contract in catalog.contracts:
        authority = contract.authority_class
        state = contract.review_state
        path = ""
        source_kind = ContractSourceKind.INFERRED_PROSE
        best_rank = 10**9
        for source_id in contract.source_ids:
            source = source_by_id.get(source_id)
            if source is None:
                continue
            if source.path and (
                not path or source.authority_class.rank < best_rank
            ):
                path = source.path
                source_kind = source.kind
                best_rank = source.authority_class.rank
        inventory = inventory_by_path.get(path)
        declaration = (
            inventory.declaration_kind
            if inventory is not None
            else DeclarationKind.OTHER.value
        )
        # Archive/generated inventory never authorizes reviewed contracts.
        if inventory is not None and inventory.classification in {
            ConsumerClassification.ARCHIVE.value,
            ConsumerClassification.GENERATED.value,
        }:
            authority = SourceAuthorityClass.NONE
            if state is ReviewState.REVIEWED:
                state = ReviewState.NOMINATED
        rows.append(
            ExpectationRow(
                subject=contract.subject,
                field_path=contract.claim_family.value,
                declaration_kind=declaration,
                authority_class=authority.value,
                review_state=state.value,
                source_kind=source_kind.value,
                path=path,
                transport=str(contract.metadata.get("transport") or ""),
                ui_action=str(contract.metadata.get("ui_action") or ""),
                version=contract.source_version,
                value_digest=contract.contract_id,
            )
        )
    for edge in extraction.invocation_edges:
        path = edge.source_span.path if edge.source_span is not None else ""
        inventory = inventory_by_path.get(path)
        classification = (
            inventory.classification
            if inventory is not None
            else ConsumerClassification.OTHER.value
        )
        role = SourceRole(inventory.role) if inventory is not None else SourceRole.OTHER
        authority = _authority_for_classification(
            ConsumerClassification(classification)
            if classification in {item.value for item in ConsumerClassification}
            else ConsumerClassification.OTHER,
            role,
        )
        rows.append(
            ExpectationRow(
                subject=f"invocation-edge:{edge.edge_id}",
                field_path="edge",
                declaration_kind=(
                    inventory.declaration_kind
                    if inventory is not None
                    else DeclarationKind.CALL_SITE.value
                ),
                authority_class=authority.value,
                review_state=(
                    ReviewState.REVIEWED.value
                    if authority.may_authorize_reviewed_contract
                    else ReviewState.NOMINATED.value
                ),
                source_kind=_source_kind_for_role(role).value,
                path=path,
                transport=str(edge.transport or ""),
                ui_action=str(getattr(edge, "kind", "") and edge.kind.value),
                version=str(extraction.source_versions.get(path) or ""),
                value_digest=edge.edge_id,
            )
        )
    return tuple(sorted(rows, key=lambda item: item.to_row()))


def _consumer_rows(
    inventory: Sequence[InventoryRow],
    extraction: Any,
) -> tuple[ConsumerRow, ...]:
    subjects_by_path: dict[str, set[str]] = {}
    for expectation in extraction.expectations:
        path = expectation.source_span.path
        subjects_by_path.setdefault(path, set()).add(expectation.subject)
    for descriptor in extraction.descriptors:
        path = descriptor.source_span.path if descriptor.source_span else ""
        if path:
            subjects_by_path.setdefault(path, set()).add(
                f"descriptor:{descriptor.package_id}:{descriptor.name}"
            )
    for edge in extraction.invocation_edges:
        path = edge.source_span.path if edge.source_span is not None else ""
        if path:
            subjects_by_path.setdefault(path, set()).add(f"edge:{edge.edge_id}")
    rows: list[ConsumerRow] = []
    for item in inventory:
        role = SourceRole(item.role)
        classification = ConsumerClassification(item.classification)
        # A consumer is "active" only when classified active and participates
        # in at least one declaration, edge, or expectation subject.
        subject_count = len(subjects_by_path.get(item.path, ()))
        if classification is ConsumerClassification.ACTIVE and subject_count == 0:
            # Still count path inventory, but only surface-bearing paths are
            # active consumers for parity accounting.
            continue
        if classification is not ConsumerClassification.ACTIVE and subject_count == 0:
            continue
        authority = _authority_for_classification(classification, role)
        rows.append(
            ConsumerRow(
                path=item.path,
                role=item.role,
                classification=item.classification,
                declaration_kind=item.declaration_kind,
                authority_class=authority.value,
                subject_count=subject_count,
                content_digest=item.content_digest,
            )
        )
    return tuple(sorted(rows, key=lambda item: item.to_row()))


def materialize_desktop_expectations(
    workspace_root: str | os.PathLike[str] | None = None,
    *,
    forest_path: str | os.PathLike[str] | None = None,
    include_paths: Sequence[str] | None = None,
    max_files: int = DEFAULT_MAX_FILES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> dict[str, Any]:
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
    swissknife_root, swissknife_head = _swissknife_root(workspace, forest_payload)
    sources, inventory = _collect_sources(
        swissknife_root,
        include_paths=include_paths,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
    )
    extractor = SwissKnifeContractExtractor(
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
    )
    try:
        extraction = extractor.extract(
            sources,
            repository_tree_id=swissknife_head or forest_proof["forest_id"],
        )
    except SwissKnifeContractExtractorError as exc:
        raise DeterministicDesktopExpectationsError(
            "swissknife_extraction_failed", str(exc)
        ) from exc

    inventory_by_path = {item.path: item for item in inventory}
    consumers = _consumer_rows(inventory, extraction)
    expectations = _expectation_rows(extraction, inventory_by_path)

    inventory_rows = tuple(item.to_row() for item in inventory)
    consumer_rows = tuple(item.to_row() for item in consumers)
    expectation_rows = tuple(item.to_row() for item in expectations)

    inventory_ledger, inventory_digest = _encode_prefix_ledger(
        schema=DESKTOP_INVENTORY_LEDGER_SCHEMA,
        codec=INVENTORY_CODEC,
        rows=inventory_rows,
        path_index=0,
        dictionary_fields=(
            ("role", 1),
            ("classification", 2),
            ("declaration_kind", 3),
        ),
    )
    consumer_ledger, consumer_digest = _encode_prefix_ledger(
        schema=DESKTOP_CONSUMER_LEDGER_SCHEMA,
        codec=CONSUMER_CODEC,
        rows=consumer_rows,
        path_index=0,
        dictionary_fields=(
            ("role", 1),
            ("classification", 2),
            ("declaration_kind", 3),
            ("authority_class", 4),
        ),
    )
    expectation_ledger, expectation_digest = _encode_prefix_ledger(
        schema=DESKTOP_EXPECTATION_LEDGER_SCHEMA,
        codec=EXPECTATION_CODEC,
        rows=expectation_rows,
        path_index=6,
        dictionary_fields=(
            ("declaration_kind", 2),
            ("authority_class", 3),
            ("review_state", 4),
            ("source_kind", 5),
        ),
    )

    active_consumers = [
        item for item in consumers if item.classification == ConsumerClassification.ACTIVE.value
    ]
    active_paths = {item.path for item in active_consumers}
    expectation_paths = {
        item.path for item in expectations if item.path and item.path in active_paths
    }
    # Every active consumer must appear in at least one expectation/edge row.
    missing_active = sorted(active_paths - expectation_paths)
    # Archive/generated must not appear as authorizing reviewed rows.
    forged_authority = [
        item
        for item in expectations
        if item.authority_class
        in {
            SourceAuthorityClass.AUTHORITATIVE.value,
            SourceAuthorityClass.CONFORMANCE.value,
            SourceAuthorityClass.REGISTRATION.value,
            SourceAuthorityClass.MANIFEST.value,
        }
        and inventory_by_path.get(item.path) is not None
        and inventory_by_path[item.path].classification
        in {
            ConsumerClassification.ARCHIVE.value,
            ConsumerClassification.GENERATED.value,
        }
    ]
    contradiction_count = len(extraction.catalog.contradictions)
    blocking: list[str] = []
    if missing_active:
        blocking.append("active_consumer_unaccounted")
    if forged_authority:
        blocking.append("archive_or_generated_authority_override")
    # Contradictions are retained, not fatal for the index itself.
    parity_ok = not blocking
    orchestration_head = _git_oid(workspace, "rev-parse", "HEAD")
    orchestration_tree = _git_oid(workspace, "rev-parse", "HEAD^{tree}")
    lifecycle = {
        "schema": DESKTOP_EXPECTATIONS_LIFECYCLE_SCHEMA,
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
    interfaces = {
        "mcp_contract_catalog": MCP_CONTRACT_CATALOG_INTERFACE,
        "uiir_document": UIIR_DOCUMENT_INTERFACE,
        "mcp_idl": MCP_IDL_INTERFACE,
        "orb": ORB_INTERFACE,
        "desktop_expectation_index": DESKTOP_EXPECTATION_INTERFACE,
    }
    identity = {
        "schema": DESKTOP_EXPECTATIONS_SCHEMA,
        "interface": DESKTOP_EXPECTATION_INTERFACE,
        "interfaces": interfaces,
        "forest_id": forest_proof["forest_id"],
        "forest_historical_proof": forest_proof,
        "lifecycle": lifecycle,
        "swissknife_head": swissknife_head,
        "extractor_version": SWISSKNIFE_EXTRACTOR_VERSION,
        "catalog_version": CATALOG_VERSION,
        "inventory": {
            "scanned_file_count": len(inventory),
            "inventory_merkle_root": _merkle_root(inventory_rows),
            "ledger": inventory_ledger,
            "uncompressed_digest": inventory_digest,
        },
        "consumers": {
            "active_count": len(active_consumers),
            "total_count": len(consumers),
            "classification_counts": {
                kind.value: sum(1 for item in consumers if item.classification == kind.value)
                for kind in ConsumerClassification
            },
            "consumer_merkle_root": _merkle_root(consumer_rows),
            "ledger": consumer_ledger,
            "uncompressed_digest": consumer_digest,
        },
        "expectations": {
            "row_count": len(expectations),
            "expectation_merkle_root": _merkle_root(expectation_rows),
            "ledger": expectation_ledger,
            "uncompressed_digest": expectation_digest,
            "contradiction_count": contradiction_count,
            "catalog_id": extraction.catalog.catalog_id,
            "extraction_id": extraction.extraction_id,
            "descriptor_count": len(extraction.descriptors),
            "invocation_edge_count": len(extraction.invocation_edges),
            "schema_count": len(extraction.schemas),
            "unresolved_count": len(extraction.unresolved_values),
        },
        "authority": {
            "precedence": [
                SourceAuthorityClass.AUTHORITATIVE.value,
                SourceAuthorityClass.CONFORMANCE.value,
                SourceAuthorityClass.REGISTRATION.value,
                SourceAuthorityClass.MANIFEST.value,
                SourceAuthorityClass.NOMINATING.value,
                SourceAuthorityClass.NONE.value,
            ],
            "nominating_cannot_override_reviewed": True,
            "archive_generated_cannot_authorize": True,
            "contradictions_remain_unresolved": True,
        },
        "parity": {
            "active_consumers_accounted": not missing_active,
            "missing_active_consumers": missing_active,
            "blocking_reason_codes": blocking,
            "safe_for_completion_reasoning": parity_ok,
        },
        "authoritative": False,
        "completion_authorized": False,
    }
    desktop_expectations_id = _content_id(identity)
    return {
        **identity,
        "desktop_expectations_id": desktop_expectations_id,
    }


def _compact_expectation_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Drop invocation-edge rows while retaining contract expectation rows."""

    expectations = payload.get("expectations")
    consumers = payload.get("consumers")
    if not isinstance(expectations, Mapping) or not isinstance(consumers, Mapping):
        return dict(payload)
    ledger = expectations.get("ledger")
    consumer_ledger = consumers.get("ledger")
    if not isinstance(ledger, Mapping) or not isinstance(consumer_ledger, Mapping):
        return dict(payload)
    rows, _ = decode_prefix_ledger(
        ledger,
        expected_schema=DESKTOP_EXPECTATION_LEDGER_SCHEMA,
        expected_codec=EXPECTATION_CODEC,
    )
    consumer_rows, _ = decode_prefix_ledger(
        consumer_ledger,
        expected_schema=DESKTOP_CONSUMER_LEDGER_SCHEMA,
        expected_codec=CONSUMER_CODEC,
    )
    retained = tuple(
        row for row in rows if not str(row[0]).startswith("invocation-edge:")
    )
    compacted_ledger, digest = _encode_prefix_ledger(
        schema=DESKTOP_EXPECTATION_LEDGER_SCHEMA,
        codec=EXPECTATION_CODEC,
        rows=retained,
        path_index=6,
        dictionary_fields=(
            ("declaration_kind", 2),
            ("authority_class", 3),
            ("review_state", 4),
            ("source_kind", 5),
        ),
    )
    active_paths = {
        row[0]
        for row in consumer_rows
        if row[2] == ConsumerClassification.ACTIVE.value
    }
    expectation_paths = {row[6] for row in retained if row[6]}
    missing_active = sorted(active_paths - expectation_paths)
    blocking = list((payload.get("parity") or {}).get("blocking_reason_codes") or [])
    blocking = [code for code in blocking if code != "active_consumer_unaccounted"]
    if missing_active:
        blocking.append("active_consumer_unaccounted")
    blocking = list(dict.fromkeys(blocking))
    next_payload = dict(payload)
    next_expectations = dict(expectations)
    next_expectations["ledger"] = compacted_ledger
    next_expectations["uncompressed_digest"] = digest
    next_expectations["row_count"] = len(retained)
    next_expectations["expectation_merkle_root"] = _merkle_root(retained)
    next_expectations["compacted"] = True
    next_payload["expectations"] = next_expectations
    next_payload["parity"] = {
        "active_consumers_accounted": not missing_active,
        "missing_active_consumers": missing_active,
        "blocking_reason_codes": blocking,
        "safe_for_completion_reasoning": not blocking,
    }
    identity = {
        key: value
        for key, value in next_payload.items()
        if key != "desktop_expectations_id"
    }
    next_payload["desktop_expectations_id"] = _content_id(identity)
    return next_payload


def write_desktop_expectations(
    destination: str | os.PathLike[str],
    workspace_root: str | os.PathLike[str] | None = None,
    *,
    forest_path: str | os.PathLike[str] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    include_paths: Sequence[str] | None = None,
    max_files: int = DEFAULT_MAX_FILES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> dict[str, Any]:
    payload = materialize_desktop_expectations(
        workspace_root,
        forest_path=forest_path,
        include_paths=include_paths,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
    )
    # Prefer contract rows when the full edge+contract projection exceeds the
    # supervisor admission limit; active-consumer parity is preserved either way.
    encoded = _artifact_bytes(payload)
    if len(encoded) > max_bytes:
        payload = _compact_expectation_projection(payload)
        encoded = _artifact_bytes(payload)
    if len(encoded) > max_bytes:
        raise DeterministicDesktopExpectationsError(
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
class DesktopExpectationsValidation:
    integrity_valid: bool = False
    current: bool = False
    downstream_authorized: bool = False
    lifecycle_state: str = "invalid"
    desktop_expectations_id: str = ""
    forest_id: str = ""
    reason_codes: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.downstream_authorized

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DESKTOP_EXPECTATIONS_VALIDATION_SCHEMA,
            "valid": self.valid,
            "integrity_valid": self.integrity_valid,
            "current": self.current,
            "downstream_authorized": self.downstream_authorized,
            "lifecycle_state": self.lifecycle_state,
            "desktop_expectations_id": self.desktop_expectations_id,
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
    except (DeterministicDesktopExpectationsError, OSError):
        return False
    return observed == expected


def _todo_delta_is_exact(root: Path, observed: str, completed: str) -> bool:
    try:
        before_raw = _run_git(root, "show", f"{observed}:{DCR_TODO_PATH}")
        after_raw = _run_git(root, "show", f"{completed}:{DCR_TODO_PATH}")
        assert isinstance(before_raw, bytes) and isinstance(after_raw, bytes)
        before = before_raw.decode("utf-8")
        after = after_raw.decode("utf-8")
    except (DeterministicDesktopExpectationsError, UnicodeDecodeError):
        return False
    marker = "## DCR-014 "
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
        raise DeterministicDesktopExpectationsError("invalid_commit_graph")
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
        or "dcr-014" not in subject_text
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


def _document_integrity_desktop(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    reasons: list[str] = []
    if payload.get("schema") != DESKTOP_EXPECTATIONS_SCHEMA:
        reasons.append("invalid_schema")
    claimed = payload.get("desktop_expectations_id")
    try:
        recomputed = _content_id(
            {
                key: value
                for key, value in payload.items()
                if key != "desktop_expectations_id"
            }
        )
    except DeterministicDesktopExpectationsError:
        recomputed = ""
    if not isinstance(claimed, str) or claimed != recomputed:
        reasons.append("desktop_expectations_id_mismatch")
    for section_name, schema, codec in (
        ("inventory", DESKTOP_INVENTORY_LEDGER_SCHEMA, INVENTORY_CODEC),
        ("consumers", DESKTOP_CONSUMER_LEDGER_SCHEMA, CONSUMER_CODEC),
        ("expectations", DESKTOP_EXPECTATION_LEDGER_SCHEMA, EXPECTATION_CODEC),
    ):
        section = payload.get(section_name)
        if not isinstance(section, Mapping):
            reasons.append(f"invalid_{section_name}_section")
            continue
        ledger = section.get("ledger")
        if not isinstance(ledger, Mapping):
            reasons.append(f"invalid_{section_name}_ledger")
            continue
        try:
            rows, digest = decode_prefix_ledger(
                ledger, expected_schema=schema, expected_codec=codec
            )
            if digest != section.get("uncompressed_digest"):
                reasons.append(f"{section_name}_uncompressed_digest_mismatch")
            if int(ledger.get("row_count", -1)) != len(rows):
                reasons.append(f"{section_name}_row_count_mismatch")
        except DeterministicDesktopExpectationsError as exc:
            reasons.append(exc.reason_code)
    forest_proof = payload.get("forest_historical_proof")
    if not isinstance(forest_proof, Mapping) or not forest_proof.get("integrity_valid"):
        reasons.append("forest_historical_proof_invalid")
    lifecycle = payload.get("lifecycle")
    if not isinstance(lifecycle, Mapping) or lifecycle.get("task_id") != DCR_TASK_ID:
        reasons.append("invalid_lifecycle_policy")
    parity = payload.get("parity")
    if not isinstance(parity, Mapping):
        reasons.append("invalid_parity_block")
    if reasons:
        return None, tuple(dict.fromkeys(reasons))
    return dict(payload), ()


def validate_desktop_expectations(
    source: Mapping[str, Any] | str | os.PathLike[str],
    workspace_root: str | os.PathLike[str] | None = None,
    *,
    forest_path: str | os.PathLike[str] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> DesktopExpectationsValidation:
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
            return DesktopExpectationsValidation(reason_codes=("artifact_unreadable",))
        try:
            payload = dict(_read_json_bytes(encoded, reason="artifact_unreadable"))
        except DeterministicDesktopExpectationsError as exc:
            return DesktopExpectationsValidation(reason_codes=(exc.reason_code,))
    if len(encoded) > max_bytes:
        return DesktopExpectationsValidation(
            reason_codes=("artifact_exceeds_admission_limit",)
        )
    document, integrity_reasons = _document_integrity_desktop(payload)
    desktop_id = str(payload.get("desktop_expectations_id") or "")
    forest_id = str(payload.get("forest_id") or "")
    if document is None:
        return DesktopExpectationsValidation(
            desktop_expectations_id=desktop_id,
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
            return DesktopExpectationsValidation(
                integrity_valid=True,
                desktop_expectations_id=desktop_id,
                forest_id=forest_id,
                reason_codes=("forest_id_mismatch",),
            )
    except DeterministicDesktopExpectationsError as exc:
        return DesktopExpectationsValidation(
            integrity_valid=True,
            desktop_expectations_id=desktop_id,
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
    except DeterministicDesktopExpectationsError as exc:
        state, lifecycle_reasons = "stale", (exc.reason_code,)
    current_flag = not lifecycle_reasons
    parity = payload.get("parity") if isinstance(payload.get("parity"), Mapping) else {}
    safe = bool(parity.get("safe_for_completion_reasoning"))
    completion_reasons: tuple[str, ...] = ()
    if safe is True and list(parity.get("blocking_reason_codes") or []):
        completion_reasons = ("completion_safe_claim_forged",)
    if safe is True and parity.get("active_consumers_accounted") is not True:
        completion_reasons = completion_reasons + ("active_consumers_not_accounted",)
    downstream = (
        current_flag
        and not completion_reasons
        and state
        in {
            "captured",
            "artifact_carried",
            "integrated",
            "todo_completed",
        }
    )
    return DesktopExpectationsValidation(
        integrity_valid=True,
        current=current_flag,
        downstream_authorized=downstream,
        lifecycle_state=state,
        desktop_expectations_id=desktop_id,
        forest_id=forest_id,
        reason_codes=lifecycle_reasons + completion_reasons,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "materialize"))
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--forest", default=DCR_FOREST_PATH)
    parser.add_argument("--artifact", default=DCR_ARTIFACT_PATH)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
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
            payload = write_desktop_expectations(
                artifact,
                workspace,
                forest_path=forest,
                max_bytes=arguments.max_bytes,
            )
        except DeterministicDesktopExpectationsError as exc:
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
                    "desktop_expectations_id": payload.get("desktop_expectations_id"),
                    "scanned_file_count": payload.get("inventory", {}).get(
                        "scanned_file_count"
                    ),
                    "active_count": payload.get("consumers", {}).get("active_count"),
                    "safe_for_completion_reasoning": payload.get("parity", {}).get(
                        "safe_for_completion_reasoning"
                    ),
                },
                sort_keys=True,
            )
            + "\n"
        )
        return 0

    expected = workspace.joinpath(*PurePosixPath(DCR_ARTIFACT_PATH).parts)
    if artifact.resolve(strict=False) != expected.resolve(strict=False):
        result = DesktopExpectationsValidation(
            reason_codes=("desktop_output_path_invalid",)
        )
    else:
        result = validate_desktop_expectations(
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


__all__ = [
    "CONSUMER_CODEC",
    "DCR_ARTIFACT_PATH",
    "DCR_CARRIER_SUBJECT",
    "DCR_FOREST_PATH",
    "DCR_TASK_ID",
    "DCR_TODO_SUBJECT",
    "DEFAULT_MAX_BYTES",
    "DESKTOP_EXPECTATION_INTERFACE",
    "DESKTOP_EXPECTATIONS_SCHEMA",
    "EXPECTATION_CODEC",
    "INVENTORY_CODEC",
    "ConsumerClassification",
    "DeclarationKind",
    "DesktopExpectationsValidation",
    "DeterministicDesktopExpectationsError",
    "decode_prefix_ledger",
    "main",
    "materialize_desktop_expectations",
    "validate_desktop_expectations",
    "write_desktop_expectations",
]


if __name__ == "__main__":
    raise SystemExit(main())
