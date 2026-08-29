"""Stable, identity-preserving machine output for the proof-context CLI.

This module is a projection layer: callers supply a canonical status and
observed identities, and the renderer admits them without rewriting semantic
status or artifact identity. Diagnostic data is redacted and bounded before it
can reach either JSON or human output.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.proof_context.adapters.models import admit_cid
from ipfs_accelerate_py.proof_context.errors import (
    REDACTED,
    SECRET_DETAIL_KEYS,
    MalformedError,
    SchemaMismatchError,
    UnknownFieldError,
    redact_text,
)
from ipfs_accelerate_py.proof_context.results import (
    PROVENANCES,
    STATUSES,
    admit_provenance,
    admit_status,
)

SCHEMA_VERSION: Final[str] = "ipfs-accelerate.proof-context.v0.1/command-result@1"
CONTRACT_VERSION: Final[str] = "0.1"
LEGACY_CLI_RESULT_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/cli-result"
CLI_INTERFACE: Final[str] = "ProofContextCLI@0.1"

MAX_TEXT_LENGTH: Final[int] = 240
MAX_IDENTITY_LENGTH: Final[int] = 240
MAX_COLLECTION_ITEMS: Final[int] = 64
MAX_MAPPING_ITEMS: Final[int] = 64
MAX_DEPTH: Final[int] = 8
MISSING_EVIDENCE: Final[str] = "missing_evidence"
OMITTED_UNBOUNDED: Final[str] = "[omitted: unbounded source/log]"

IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "repository_id",
    "task_id",
    "run_id",
    "patch_id",
)
OPTIONAL_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "repository_state_cid",
    "trace_id",
    "evidence_cid",
    "artifact_id",
    "contract_version",
)
CID_IDENTITY_FIELDS: Final[frozenset[str]] = frozenset({"repository_state_cid", "evidence_cid"})

REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "contract_version",
    "command",
    "status",
    "exit_code",
    "trace_id",
    "correlation_id",
    "identities",
    "artifact_cids",
    "provenance",
    "details",
    "error",
)
COMPATIBILITY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "interface",
        "policy",
        "output_mode",
        "artifact_cid",
        "contract",
        "payload",
        "provider_bound",
        "sibling_layout_required",
    }
)
RESULT_FIELDS: Final[frozenset[str]] = frozenset(REQUIRED_FIELDS) | COMPATIBILITY_FIELDS

EXIT_CODES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "succeeded": 0,
        "invalid": 2,
        "rejected": 3,
        "simulated": 4,
        "unavailable": 5,
        "stale": 6,
        "timeout": 1,
        "cancelled": 1,
        "verification_failed": 1,
        "proof_failed": 1,
        "assurance_failed": 1,
        "context_insufficient": 1,
        "model_escalation_required": 1,
        "human_review_required": 1,
        "infrastructure_failure": 1,
        "partial_effect": 1,
        "repair_required": 1,
    }
)

_UNBOUNDED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "diff",
        "log",
        "logs",
        "patch_text",
        "raw",
        "raw_output",
        "source",
        "stack",
        "stderr",
        "stdout",
        "traceback",
    }
)


def _normalized_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _secret_key(value: Any) -> bool:
    normalized = _normalized_key(value)
    return normalized in SECRET_DETAIL_KEYS or any(
        token in normalized for token in ("secret", "password", "token", "credential")
    )


def _unbounded_key(value: Any) -> bool:
    return _normalized_key(value) in _UNBOUNDED_KEYS


def _bounded_text(value: Any) -> str:
    text = redact_text(str(value))
    if len(text) > MAX_TEXT_LENGTH:
        return text[: MAX_TEXT_LENGTH - 1] + "…"
    return text


def _admit_identity(value: Any, *, field: str, optional: bool = True) -> str | None:
    if value is None or value == "":
        if optional:
            return None
        raise MalformedError(f"{field} is required")
    if not isinstance(value, str):
        raise MalformedError(f"{field} must be a string")
    if not value.strip():
        raise MalformedError(f"{field} must not be blank")
    if len(value) > MAX_IDENTITY_LENGTH:
        raise MalformedError(f"{field} exceeds the frozen identity bound")
    # Identity bytes must never be silently rewritten. Reject any value the
    # canonical redactor would alter instead of emitting a different identity.
    if redact_text(value) != value:
        raise MalformedError(f"{field} contains secret-shaped or unsafe text")
    return value


def redact_and_bound(value: Any, *, depth: int = 0) -> Any:
    """Return deterministic JSON-safe data without secrets or raw dumps."""

    if depth >= MAX_DEPTH:
        return OMITTED_UNBOUNDED
    if isinstance(value, Mapping):
        rendered: dict[str, Any] = {}
        for index, (raw_key, item) in enumerate(value.items()):
            if index >= MAX_MAPPING_ITEMS:
                rendered["truncated_items"] = len(value) - MAX_MAPPING_ITEMS
                break
            key = _bounded_text(raw_key)
            if _secret_key(raw_key):
                rendered[key] = REDACTED
            elif _unbounded_key(raw_key):
                rendered[key] = OMITTED_UNBOUNDED
            else:
                rendered[key] = redact_and_bound(item, depth=depth + 1)
        return rendered
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = [
            redact_and_bound(item, depth=depth + 1) for item in list(value)[:MAX_COLLECTION_ITEMS]
        ]
        if len(value) > MAX_COLLECTION_ITEMS:
            items.append("[truncated: additional items omitted]")
        return items
    if isinstance(value, (set, frozenset)):
        items = sorted(value, key=str)
        return redact_and_bound(items, depth=depth)
    if isinstance(value, bytes):
        return "[binary content omitted]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _bounded_text(value)


def normalized_identities(identities: Mapping[str, Any] | None) -> dict[str, Any]:
    """Admit identities exactly and make the four required slots explicit."""

    if identities is not None and not isinstance(identities, Mapping):
        raise MalformedError("identities must be a mapping")
    source = dict(identities or {})
    allowed = set(IDENTITY_FIELDS) | set(OPTIONAL_IDENTITY_FIELDS)
    extra = set(source) - allowed
    if extra:
        raise UnknownFieldError(f"unknown identity field {sorted(extra)[0]!r}")

    result: dict[str, Any] = {
        field: _admit_identity(source.get(field), field=f"identities.{field}")
        for field in IDENTITY_FIELDS
    }
    for field in OPTIONAL_IDENTITY_FIELDS:
        if field not in source:
            continue
        if field == "contract_version":
            if source[field] != CONTRACT_VERSION:
                raise SchemaMismatchError(
                    f"identity contract version {source[field]!r} is not {CONTRACT_VERSION}"
                )
            result[field] = CONTRACT_VERSION
            continue
        value = source.get(field)
        if value in (None, ""):
            result[field] = None
        elif field in CID_IDENTITY_FIELDS:
            result[field] = admit_cid(value, field=f"identities.{field}")
        else:
            result[field] = _admit_identity(value, field=f"identities.{field}")
    return result


def normalized_artifact_cids(artifact_cids: Sequence[str] | None) -> list[str]:
    if artifact_cids is None:
        return []
    if not isinstance(artifact_cids, Sequence) or isinstance(
        artifact_cids, (str, bytes, bytearray)
    ):
        raise MalformedError("artifact_cids must be an array")
    if len(artifact_cids) > MAX_COLLECTION_ITEMS:
        raise MalformedError("artifact_cids exceeds the frozen item bound")
    return [
        admit_cid(value, field=f"artifact_cids[{index}]")
        for index, value in enumerate(artifact_cids)
    ]


def exit_code_for(status: str, *, provenance: str | None = "live") -> int:
    """Map only canonical statuses and provenances onto the stable exit matrix."""

    admitted_status = admit_status(status)
    admitted_provenance = admit_provenance(provenance) if provenance is not None else None
    if admitted_status == "succeeded" and admitted_provenance != "live":
        return EXIT_CODES["simulated"] if admitted_provenance == "simulated" else 1
    return EXIT_CODES[admitted_status]


def _compatibility_mapping(
    raw: Mapping[str, Any] | None,
    *,
    artifact_cids: Sequence[str],
) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise MalformedError("compatibility fields must be a mapping")
    source = dict(raw)
    extra = set(source) - COMPATIBILITY_FIELDS
    if extra:
        raise UnknownFieldError(f"unknown compatibility field {sorted(extra)[0]!r}")

    rendered: dict[str, Any] = {}
    if "schema" in source:
        if source["schema"] != LEGACY_CLI_RESULT_SCHEMA:
            raise SchemaMismatchError(
                f"legacy schema {source['schema']!r} is not {LEGACY_CLI_RESULT_SCHEMA}"
            )
        rendered["schema"] = LEGACY_CLI_RESULT_SCHEMA
    if "interface" in source:
        if source["interface"] != CLI_INTERFACE:
            raise SchemaMismatchError(
                f"CLI interface {source['interface']!r} is not {CLI_INTERFACE}"
            )
        rendered["interface"] = CLI_INTERFACE
    for field in ("policy", "contract"):
        if field in source:
            rendered[field] = _admit_identity(source[field], field=field, optional=True)
    if "output_mode" in source:
        if source["output_mode"] not in {"json", "human"}:
            raise MalformedError("output_mode must be json or human")
        rendered["output_mode"] = source["output_mode"]
    if "artifact_cid" in source:
        artifact = source["artifact_cid"]
        admitted = admit_cid(artifact, field="artifact_cid") if artifact not in (None, "") else None
        primary = artifact_cids[0] if artifact_cids else None
        if admitted != primary:
            raise MalformedError("artifact_cid must equal artifact_cids[0]")
        rendered["artifact_cid"] = admitted
    if "payload" in source:
        if not isinstance(source["payload"], Mapping):
            raise MalformedError("payload must be a mapping")
        rendered["payload"] = redact_and_bound(source["payload"])
    for field in ("provider_bound", "sibling_layout_required"):
        if field in source:
            if not isinstance(source[field], bool):
                raise MalformedError(f"{field} must be a boolean")
            rendered[field] = source[field]
    return rendered


def command_result(
    *,
    status: str,
    correlation_id: str,
    identities: Mapping[str, Any] | None,
    artifact_cids: Sequence[str] | None = None,
    command: str | None = None,
    trace_id: str | None = None,
    provenance: str | None = "live",
    details: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | str | None = None,
    compatibility: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one admitted envelope without inventing missing evidence."""

    admitted_status = admit_status(status)
    admitted_provenance = admit_provenance(provenance) if provenance is not None else None
    identity_map = normalized_identities(identities)
    artifacts = normalized_artifact_cids(artifact_cids)
    correlation = _admit_identity(correlation_id, field="correlation_id", optional=False)
    observed_trace = trace_id or identity_map.get("trace_id") or correlation
    trace = _admit_identity(observed_trace, field="trace_id", optional=False)
    if details is not None and not isinstance(details, Mapping):
        raise MalformedError("details must be a mapping")
    if error is not None and not isinstance(error, (Mapping, str)):
        raise MalformedError("error must be a mapping, string, or null")

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "command": _bounded_text(command) if command else None,
        "status": admitted_status,
        "exit_code": exit_code_for(admitted_status, provenance=admitted_provenance),
        "trace_id": trace,
        "correlation_id": correlation,
        "identities": identity_map,
        "artifact_cids": artifacts,
        "provenance": admitted_provenance,
        "details": redact_and_bound(details or {}),
        "error": redact_and_bound(error) if error is not None else None,
    }
    result.update(_compatibility_mapping(compatibility, artifact_cids=artifacts))
    return result


def admit_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Re-admit a mapping and reject schema, exit-code, or field drift."""

    if not isinstance(result, Mapping):
        raise MalformedError("command result must be a mapping")
    missing = [field for field in REQUIRED_FIELDS if field not in result]
    if missing:
        raise MalformedError("command result missing required fields: " + ", ".join(missing))
    extra = set(result) - RESULT_FIELDS
    if extra:
        raise UnknownFieldError(f"unknown command result field {sorted(extra)[0]!r}")
    if result["schema_version"] != SCHEMA_VERSION:
        raise SchemaMismatchError(
            f"schema version {result['schema_version']!r} is not {SCHEMA_VERSION}"
        )
    if result["contract_version"] != CONTRACT_VERSION:
        raise SchemaMismatchError(
            f"contract version {result['contract_version']!r} is not {CONTRACT_VERSION}"
        )
    compatibility = {field: result[field] for field in COMPATIBILITY_FIELDS if field in result}
    admitted = command_result(
        status=result["status"],
        correlation_id=result["correlation_id"],
        identities=result["identities"],
        artifact_cids=result["artifact_cids"],
        command=result["command"],
        trace_id=result["trace_id"],
        provenance=result["provenance"],
        details=result["details"],
        error=result["error"],
        compatibility=compatibility,
    )
    if result["exit_code"] != admitted["exit_code"]:
        raise MalformedError("exit_code does not match canonical status/provenance")
    return admitted


def serialize(result: Mapping[str, Any]) -> str:
    """Serialize a fully admitted, redacted envelope deterministically."""

    admitted = admit_result(result)
    return (
        json.dumps(
            admitted,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


__all__ = [
    "CLI_INTERFACE",
    "COMPATIBILITY_FIELDS",
    "CONTRACT_VERSION",
    "EXIT_CODES",
    "IDENTITY_FIELDS",
    "LEGACY_CLI_RESULT_SCHEMA",
    "MAX_COLLECTION_ITEMS",
    "MAX_IDENTITY_LENGTH",
    "MAX_TEXT_LENGTH",
    "MISSING_EVIDENCE",
    "OMITTED_UNBOUNDED",
    "OPTIONAL_IDENTITY_FIELDS",
    "PROVENANCES",
    "REQUIRED_FIELDS",
    "RESULT_FIELDS",
    "SCHEMA_VERSION",
    "STATUSES",
    "admit_result",
    "command_result",
    "exit_code_for",
    "normalized_artifact_cids",
    "normalized_identities",
    "redact_and_bound",
    "serialize",
]
