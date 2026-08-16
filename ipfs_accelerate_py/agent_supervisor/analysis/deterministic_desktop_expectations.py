"""DCR-014 static inventory of SwissKnife desktop MCP expectations.

This module reads source bytes only.  It neither imports application modules nor
executes a client, UI, registry, ORB/IDL, test, or network operation.  The
result is deliberately an inventory, not an authority grant: unresolved
consumers and contradictory declarations remain typed blockers.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

DESKTOP_EXPECTATIONS_INTERFACE: Final = "DeterministicDesktopExpectations@1"
DESKTOP_EXPECTATIONS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-desktop-expectations@1"
)
_SOURCE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        ".cjs",
        ".idl",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".mjs",
        ".orb",
        ".py",
        ".ts",
        ".tsx",
        ".yaml",
        ".yml",
    }
)
_EXCLUDED_PARTS: Final[frozenset[str]] = frozenset({".git", "node_modules", "vendor"})
_OPERATION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"['\"]?(?:operation|method|tool|name)['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"]|"
    r"(?:callTool|tools/call|tools\.call|invoke)\s*\(\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)
_VERSION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"['\"]?(?:version|protocolVersion|apiVersion)['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)
_FIELD_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:request|params|input|result|response|error)\s*[:=]\s*['\"]?([A-Za-z0-9_./{}\[\]-]+)",
    re.IGNORECASE,
)
_TRANSPORT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(http|https|stdio|websocket|ws|libp2p|ipc|orb)\b", re.IGNORECASE
)
_UI_ACTION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:onClick|action|command|menuItem|button)\s*[:=]\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)


class DesktopExpectationError(ValueError):
    """The static inventory input is unsafe or incomplete."""


class DesktopAuthorityClass(Enum):
    REVIEWED_DECLARATION = "reviewed_declaration"
    REGISTRATION = "registration"
    CONFORMANCE_TEST = "conformance_test"
    INFERRED_PROSE = "inferred_prose"
    GENERATED = "generated"
    ARCHIVE = "archive"

    @property
    def rank(self) -> int:
        return {
            self.REVIEWED_DECLARATION: 1,
            self.REGISTRATION: 2,
            self.CONFORMANCE_TEST: 3,
            self.INFERRED_PROSE: 4,
            self.GENERATED: 5,
            self.ARCHIVE: 6,
        }[self]


@dataclass(frozen=True)
class SourceSpan:
    """Exact one-line source evidence, including content binding."""

    root: str
    path: str
    digest: str
    line: int
    start_column: int
    end_column: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "path": self.path,
            "sha256": self.digest,
            "line": self.line,
            "start_column": self.start_column,
            "end_column": self.end_column,
        }


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _source_files(root: Path) -> Iterable[Path]:
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file() or candidate.suffix.lower() not in _SOURCE_SUFFIXES:
            continue
        relative = candidate.relative_to(root)
        if any(part.lower() in _EXCLUDED_PARTS for part in relative.parts):
            continue
        yield candidate


def _kind_and_authority(relative: str) -> tuple[str, DesktopAuthorityClass]:
    lowered = relative.lower()
    if "/archive/" in "/" + lowered or lowered.startswith("archive/"):
        return "archive", DesktopAuthorityClass.ARCHIVE
    if "generated" in lowered or "/dist/" in "/" + lowered:
        return "generated", DesktopAuthorityClass.GENERATED
    if any(marker in lowered for marker in (".idl", "/idl/", "/orb/")):
        return "orb_idl", DesktopAuthorityClass.REVIEWED_DECLARATION
    if any(marker in lowered for marker in ("descriptor", "schema", "/types/")):
        return "descriptor", DesktopAuthorityClass.REVIEWED_DECLARATION
    if any(marker in lowered for marker in ("registry", "manifest")):
        return "registry", DesktopAuthorityClass.REGISTRATION
    if any(marker in lowered for marker in ("/test/", "/tests/", ".test.", ".spec.")):
        return "contract_test", DesktopAuthorityClass.CONFORMANCE_TEST
    if any(marker in lowered for marker in ("/docs/", ".md", "readme")):
        return "prose", DesktopAuthorityClass.INFERRED_PROSE
    if any(marker in lowered for marker in ("/ui/", "/ux/", "component", "desktop")):
        return "ui_ir", DesktopAuthorityClass.REGISTRATION
    return "call_site", DesktopAuthorityClass.REGISTRATION


def _field_value(line: str, name: str) -> str:
    match = re.search(
        rf"['\"]?{name}['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9_./{{}}\[\]-]+)", line, re.IGNORECASE
    )
    return match.group(1) if match else ""


def _first_match(pattern: re.Pattern[str], line: str) -> str:
    match = pattern.search(line)
    if not match:
        return ""
    return next((item for item in match.groups() if item), "")


def _record_from_line(
    *,
    root_name: str,
    relative: str,
    digest: str,
    line_number: int,
    line: str,
    kind: str,
    authority: DesktopAuthorityClass,
) -> dict[str, Any] | None:
    operation = _first_match(_OPERATION_PATTERN, line)
    ui_action = _first_match(_UI_ACTION_PATTERN, line)
    looks_like_mcp = "mcp" in line.lower() or operation or ui_action
    if not looks_like_mcp:
        return None
    start = next(
        (
            index
            for index in (line.lower().find("mcp"), line.find(operation), line.find(ui_action))
            if index >= 0
        ),
        0,
    )
    transport = _first_match(_TRANSPORT_PATTERN, line).lower()
    return {
        "declaration_kind": kind,
        "operation": operation,
        "version": _first_match(_VERSION_PATTERN, line),
        "request": _field_value(line, "request")
        or _field_value(line, "params")
        or _field_value(line, "input"),
        "result": _field_value(line, "result") or _field_value(line, "response"),
        "error": _field_value(line, "error"),
        "transport": transport,
        "ui_action": ui_action,
        "authority_class": authority.value,
        "source_span": SourceSpan(
            root=root_name,
            path=relative,
            digest=digest,
            line=line_number,
            start_column=start + 1,
            end_column=len(line) + 1,
        ).to_dict(),
    }


def _signature(record: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(record.get(field, ""))
        for field in ("version", "request", "result", "error", "transport")
    )


def _consumer_record(root_name: str, relative: str, digest: str) -> dict[str, str]:
    return {"root": root_name, "path": relative, "sha256": digest}


def capture_desktop_expectations(
    *, swissknife_root: Path | str, mcp_plus_plus_root: Path | str
) -> dict[str, Any]:
    """Return deterministic static evidence and fail-closed blockers.

    Both roots must exist.  Active sources are all supported source files
    outside archive/generated/vendor trees; every file mentioning both desktop
    or UI vocabulary and MCP vocabulary is retained as a consumer, even when
    its invocation cannot be resolved.
    """

    roots = (("swissknife", Path(swissknife_root)), ("mcp-plus-plus", Path(mcp_plus_plus_root)))
    evidence: list[dict[str, Any]] = []
    consumers: list[dict[str, str]] = []
    blockers: list[dict[str, Any]] = []
    for root_name, supplied in roots:
        root = supplied.expanduser().resolve(strict=True)
        if not root.is_dir():
            raise DesktopExpectationError(f"{root_name} is not a readable source root")
        for source in _source_files(root):
            data = source.read_bytes()
            digest = _sha256(data)
            relative = source.relative_to(root).as_posix()
            text = data.decode("utf-8", "replace")
            kind, authority = _kind_and_authority(relative)
            is_consumer = "mcp" in text.lower() and any(
                marker in relative.lower()
                for marker in ("desktop", "ui", "ux", "app", "client", "connector")
            )
            if is_consumer:
                consumers.append(_consumer_record(root_name, relative, digest))
            before = len(evidence)
            for line_number, line in enumerate(text.splitlines(), start=1):
                record = _record_from_line(
                    root_name=root_name,
                    relative=relative,
                    digest=digest,
                    line_number=line_number,
                    line=line,
                    kind=kind,
                    authority=authority,
                )
                if record is not None:
                    evidence.append(record)
            if is_consumer and not any(
                item["source_span"]["path"] == relative and item["operation"]
                for item in evidence[before:]
            ):
                blockers.append(
                    {
                        "kind": "unresolved_desktop_mcp_consumer",
                        "consumer": _consumer_record(root_name, relative, digest),
                    }
                )
    by_operation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in evidence:
        if record["operation"]:
            by_operation[record["operation"]].append(record)
    effective: list[dict[str, Any]] = []
    for operation, records in sorted(by_operation.items()):
        ordered = sorted(
            records,
            key=lambda item: (
                DesktopAuthorityClass(item["authority_class"]).rank,
                item["source_span"]["root"],
                item["source_span"]["path"],
                item["source_span"]["line"],
            ),
        )
        effective.append(ordered[0])
        signatures = {_signature(item) for item in ordered}
        if len(signatures) > 1:
            blockers.append(
                {
                    "kind": "contradictory_desktop_expectation",
                    "operation": operation,
                    "sources": [item["source_span"] for item in ordered],
                }
            )
    payload = {
        "schema": DESKTOP_EXPECTATIONS_SCHEMA,
        "interface": DESKTOP_EXPECTATIONS_INTERFACE,
        "authoritative": False,
        "scan_mode": "static_source_only",
        "roots": [name for name, _path in roots],
        "consumers": sorted(consumers, key=lambda item: (item["root"], item["path"])),
        "evidence": sorted(
            evidence,
            key=lambda item: (
                item["source_span"]["root"],
                item["source_span"]["path"],
                item["source_span"]["line"],
            ),
        ),
        "effective_expectations": effective,
        "blockers": sorted(blockers, key=lambda item: json.dumps(item, sort_keys=True)),
    }
    payload["identity"] = _sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return payload


__all__ = [
    "DESKTOP_EXPECTATIONS_INTERFACE",
    "DESKTOP_EXPECTATIONS_SCHEMA",
    "DesktopAuthorityClass",
    "DesktopExpectationError",
    "SourceSpan",
    "capture_desktop_expectations",
]
