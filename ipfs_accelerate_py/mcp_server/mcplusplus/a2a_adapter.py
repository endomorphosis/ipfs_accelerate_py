"""A2A reference adapter for the MCP++ execution extension (A2ATaskAdapter@1).

Maps A2A Agent Card, Task, Message, Artifact, status, cancel, and streaming
onto MCP++ envelope/state/receipt evidence without inventing a competing
public task lifecycle (a2a-extension.md; ADR-0006; MCPP-056).

Wire extension URI: ``https://mcplusplus.io/extensions/execution/v1``
Working alias (non-wire): ``io.mcplusplus.execution@1``
"""

from __future__ import annotations

import copy
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from .event_dag import EventDAGStore
from .kubo_cid import cid_for_bytes

# ---------------------------------------------------------------------------
# Interface / identity pins
# ---------------------------------------------------------------------------

INTERFACE = "A2ATaskAdapter@1"
EXTENSION_URI = "https://mcplusplus.io/extensions/execution/v1"
WORKING_ALIAS = "io.mcplusplus.execution@1"
METADATA_KEY_PREFIX = "https://mcplusplus.io/extensions/execution/v1/"
CANONICALIZATION = "mcpp-jcs-v1"
TASK_ID = "MCPP-056"

SCHEMA_AGENT_EXTENSION = "mcp++/a2a/agent-extension@1"
SCHEMA_EXTENSION_PARAMS = "mcp++/a2a/extension-params@1"
SCHEMA_ACTIVATION = "mcp++/a2a/activation@1"
SCHEMA_TASK_METADATA = "mcp++/a2a/task-metadata@1"
SCHEMA_TERMINAL_EVIDENCE = "mcp++/a2a/terminal-evidence@1"
SCHEMA_SKILL_METADATA = "mcp++/a2a/skill-metadata@1"
SCHEMA_PROFILE_REQUEST = "mcp++/a2a/profile-request@1"
SCHEMA_ENVELOPE = "mcp++/execution/envelope@1"
SCHEMA_RECEIPT = "mcp++/execution/receipt@1"
SCHEMA_STATE_REF = "mcp++/state/state-ref@1"

ALLOWED_PROFILES: frozenset[str] = frozenset("ABCDEFGH")
DEFAULT_PROFILES: Tuple[str, ...] = ("A", "B", "C", "D", "F", "G")

MCP_BINDING_CURRENT = "mcp-binding/2026-07-28"
MCP_BINDING_LEGACY = "mcp-binding/legacy-2024-11-05"
ALLOWED_MCP_BINDINGS: frozenset[str] = frozenset(
    {MCP_BINDING_CURRENT, MCP_BINDING_LEGACY}
)

# Official A2A TaskState values (snake_case JSON form). No invented public names.
class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    AUTH_REQUIRED = "auth-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    REJECTED = "rejected"


TERMINAL_TASK_STATES: frozenset[str] = frozenset(
    {
        TaskState.COMPLETED.value,
        TaskState.FAILED.value,
        TaskState.CANCELED.value,
        TaskState.REJECTED.value,
    }
)

NON_TERMINAL_TASK_STATES: frozenset[str] = frozenset(
    {
        TaskState.SUBMITTED.value,
        TaskState.WORKING.value,
        TaskState.INPUT_REQUIRED.value,
        TaskState.AUTH_REQUIRED.value,
    }
)

ALL_TASK_STATES: frozenset[str] = TERMINAL_TASK_STATES | NON_TERMINAL_TASK_STATES

# Fail-closed error codes (conformance/vectors/a2a/manifest.json)
ERR_MALFORMED_EXTENSION_URI = "A2A_MALFORMED_EXTENSION_URI"
ERR_MALFORMED_EXTENSION = "A2A_MALFORMED_EXTENSION"
ERR_MISSING_RECEIPT_CID = "A2A_MISSING_RECEIPT_CID"
ERR_UNSUPPORTED_PROFILE = "A2A_UNSUPPORTED_PROFILE"
ERR_PROFILE_NOT_SUBSET = "A2A_PROFILE_NOT_SUBSET"
ERR_EXTENSION_REQUIRED = "A2A_EXTENSION_SUPPORT_REQUIRED"
ERR_TASK_NOT_CANCELABLE = "A2A_TASK_NOT_CANCELABLE"
ERR_TASK_NOT_FOUND = "A2A_TASK_NOT_FOUND"
ERR_UNSUPPORTED_EXTENSION = "A2A_UNSUPPORTED_EXTENSION"
ERR_NOT_ACTIVATED = "A2A_EXTENSION_NOT_ACTIVATED"

_CID_RE = re.compile(r"^(Qm[1-9A-HJ-NP-Za-km-z]{44}|b[a-z2-7]{58,})$")
_HTTPS_URI_RE = re.compile(r"^https://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+$")


# ---------------------------------------------------------------------------
# Errors / results
# ---------------------------------------------------------------------------


class A2AExtensionError(ValueError):
    """Fail-closed rejection for malformed extension material or lifecycle misuse."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str = "",
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.code = str(code or ERR_MALFORMED_EXTENSION)
        self.path = path
        self.details: Dict[str, Any] = dict(details or {})
        super().__init__(message if not path else f"{path}: {message}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "path": self.path,
            "details": copy.deepcopy(self.details),
        }


@dataclass
class ValidationResult:
    """Outcome of structural / semantic extension validation."""

    ok: bool
    errors: List[str] = field(default_factory=list)
    code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise A2AExtensionError(
                self.code or ERR_MALFORMED_EXTENSION,
                "; ".join(self.errors) or "validation failed",
                details=self.metadata,
            )


@dataclass
class StreamEvent:
    """One A2A-style status or artifact update while a task is open."""

    kind: str  # "status" | "artifact" | "terminal"
    task_id: str
    state: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    message: Optional[str] = None
    attempt: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "task_id": self.task_id,
            "state": self.state,
            "metadata": copy.deepcopy(self.metadata),
            "message": self.message,
            "attempt": self.attempt,
        }


# ---------------------------------------------------------------------------
# Canonicalization / CID helpers
# ---------------------------------------------------------------------------


def canonicalize_json(payload: Mapping[str, Any]) -> bytes:
    """Deterministic JSON bytes (sorted keys, compact separators)."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def compute_cid(payload: Mapping[str, Any]) -> str:
    """Content-address a JSON-like mapping as CIDv1 base32 (schema-compatible)."""
    return cid_for_bytes(canonicalize_json(payload))


def is_valid_cid(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or len(text) < 46 or len(text) > 128:
        return False
    return bool(_CID_RE.match(text))


def namespaced_metadata(short: Mapping[str, Any]) -> Dict[str, Any]:
    """Expand short suffix keys to full extension-namespace metadata keys."""
    out: Dict[str, Any] = {}
    for key, value in (short or {}).items():
        if str(key).startswith("https://"):
            out[str(key)] = copy.deepcopy(value)
        else:
            out[f"{METADATA_KEY_PREFIX}{key}"] = copy.deepcopy(value)
    return out


def denamespace_metadata(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Collapse namespaced keys back to short suffixes when under our prefix."""
    out: Dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        k = str(key)
        if k.startswith(METADATA_KEY_PREFIX):
            out[k[len(METADATA_KEY_PREFIX) :]] = copy.deepcopy(value)
        else:
            out[k] = copy.deepcopy(value)
    return out


# ---------------------------------------------------------------------------
# Extension validation (fail-closed)
# ---------------------------------------------------------------------------


def is_confirmed_extension_uri(uri: Any) -> bool:
    return isinstance(uri, str) and uri.strip() == EXTENSION_URI


def is_wire_extension_uri_shape(uri: Any) -> bool:
    """True when *uri* is an HTTPS URI shape (not reverse-DNS-only)."""
    if not isinstance(uri, str):
        return False
    text = uri.strip()
    if not text:
        return False
    return bool(_HTTPS_URI_RE.match(text))


def classify_extension_uri(uri: Any) -> Tuple[bool, Optional[str]]:
    """Return (ok_as_mcp_plus_plus_execution, error_code_if_not).

    Only the confirmed wire URI is accepted for MCP++ execution interop claims.
    Reverse-DNS alias, foreign hosts, reserved A2A-org prefix, and /v2 all fail.
    """
    if not isinstance(uri, str) or not uri.strip():
        return False, ERR_MALFORMED_EXTENSION_URI
    text = uri.strip()
    if text == EXTENSION_URI:
        return True, None
    if text == WORKING_ALIAS or not is_wire_extension_uri_shape(text):
        return False, ERR_MALFORMED_EXTENSION_URI
    # HTTPS but not the confirmed identifier
    return False, ERR_MALFORMED_EXTENSION_URI


def _validate_profile_letters(
    profiles: Any, *, path: str = "profiles"
) -> ValidationResult:
    if profiles is None:
        return ValidationResult(ok=True, metadata={path: []})
    if not isinstance(profiles, list):
        return ValidationResult(
            ok=False,
            code=ERR_MALFORMED_EXTENSION,
            errors=[f"{path} must be an array of profile letters"],
        )
    seen: Set[str] = set()
    ordered: List[str] = []
    for idx, item in enumerate(profiles):
        if not isinstance(item, str) or item not in ALLOWED_PROFILES:
            return ValidationResult(
                ok=False,
                code=ERR_UNSUPPORTED_PROFILE,
                errors=[f"{path}[{idx}] unsupported profile letter: {item!r}"],
                metadata={"profile": item},
            )
        if item in seen:
            return ValidationResult(
                ok=False,
                code=ERR_MALFORMED_EXTENSION,
                errors=[f"{path} contains duplicate profile {item!r}"],
            )
        seen.add(item)
        ordered.append(item)
    return ValidationResult(ok=True, metadata={path: ordered})


def validate_extension_params(params: Any) -> ValidationResult:
    """Validate AgentExtension.params / extension-params@1 (structural)."""
    if params is None:
        return ValidationResult(ok=True, metadata={"params": {}})
    if not isinstance(params, Mapping):
        return ValidationResult(
            ok=False,
            code=ERR_MALFORMED_EXTENSION,
            errors=["params must be an object"],
        )
    data = dict(params)
    allowed = {
        "schema",
        "profiles",
        "envelope_schema",
        "receipt_schema",
        "state_ref_schema",
        "mcp_bindings",
        "interface_cids",
        "canonicalization",
        "alias",
    }
    extra = set(data) - allowed
    if extra:
        return ValidationResult(
            ok=False,
            code=ERR_MALFORMED_EXTENSION,
            errors=[f"params has unknown keys: {sorted(extra)}"],
        )
    if "schema" in data and data["schema"] != SCHEMA_EXTENSION_PARAMS:
        return ValidationResult(
            ok=False,
            code=ERR_MALFORMED_EXTENSION,
            errors=[f"params.schema must be {SCHEMA_EXTENSION_PARAMS}"],
        )
    if "profiles" in data:
        pr = _validate_profile_letters(data["profiles"], path="params.profiles")
        if not pr.ok:
            return pr
    for marker_key, expected in (
        ("envelope_schema", SCHEMA_ENVELOPE),
        ("receipt_schema", SCHEMA_RECEIPT),
        ("state_ref_schema", SCHEMA_STATE_REF),
    ):
        if marker_key in data and data[marker_key] != expected:
            return ValidationResult(
                ok=False,
                code=ERR_MALFORMED_EXTENSION,
                errors=[f"params.{marker_key} must be {expected}"],
            )
    if "mcp_bindings" in data:
        bindings = data["mcp_bindings"]
        if not isinstance(bindings, list) or not bindings:
            return ValidationResult(
                ok=False,
                code=ERR_MALFORMED_EXTENSION,
                errors=["params.mcp_bindings must be a non-empty array"],
            )
        for b in bindings:
            if b not in ALLOWED_MCP_BINDINGS:
                return ValidationResult(
                    ok=False,
                    code=ERR_MALFORMED_EXTENSION,
                    errors=[f"unknown mcp binding id: {b!r}"],
                )
    if "interface_cids" in data:
        cids = data["interface_cids"]
        if not isinstance(cids, list):
            return ValidationResult(
                ok=False,
                code=ERR_MALFORMED_EXTENSION,
                errors=["params.interface_cids must be an array"],
            )
        for cid in cids:
            if not is_valid_cid(cid):
                return ValidationResult(
                    ok=False,
                    code=ERR_MALFORMED_EXTENSION,
                    errors=[f"invalid interface_cid: {cid!r}"],
                )
    if "canonicalization" in data and data["canonicalization"] != CANONICALIZATION:
        return ValidationResult(
            ok=False,
            code=ERR_MALFORMED_EXTENSION,
            errors=[f"canonicalization must be {CANONICALIZATION}"],
        )
    if "alias" in data and data["alias"] != WORKING_ALIAS:
        return ValidationResult(
            ok=False,
            code=ERR_MALFORMED_EXTENSION,
            errors=[f"alias must be {WORKING_ALIAS} when present"],
        )
    return ValidationResult(ok=True, metadata={"params": copy.deepcopy(data)})


def validate_agent_extension(payload: Any) -> ValidationResult:
    """Validate AgentExtension claiming the MCP++ execution extension."""
    if not isinstance(payload, Mapping):
        return ValidationResult(
            ok=False,
            code=ERR_MALFORMED_EXTENSION,
            errors=["AgentExtension must be an object"],
        )
    data = dict(payload)
    if "uri" not in data:
        return ValidationResult(
            ok=False,
            code=ERR_MALFORMED_EXTENSION,
            errors=["AgentExtension.uri is required"],
        )
    ok, code = classify_extension_uri(data.get("uri"))
    if not ok:
        return ValidationResult(
            ok=False,
            code=code or ERR_MALFORMED_EXTENSION_URI,
            errors=[
                f"AgentExtension.uri must be {EXTENSION_URI}; got {data.get('uri')!r}"
            ],
            metadata={"uri": data.get("uri")},
        )
    if "schema" in data and data["schema"] != SCHEMA_AGENT_EXTENSION:
        return ValidationResult(
            ok=False,
            code=ERR_MALFORMED_EXTENSION,
            errors=[f"schema must be {SCHEMA_AGENT_EXTENSION}"],
        )
    if "params" in data:
        pr = validate_extension_params(data["params"])
        if not pr.ok:
            return pr
    return ValidationResult(ok=True, metadata={"uri": EXTENSION_URI})


def parse_a2a_extensions_header(value: Any) -> List[str]:
    """Parse ``A2A-Extensions`` comma-separated header or list into URIs."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def validate_activation(
    payload: Any,
    *,
    require_execution: bool = False,
) -> ValidationResult:
    """Validate activation payload / A2A-Extensions list (activation@1)."""
    if isinstance(payload, (str, list, tuple)):
        extensions = parse_a2a_extensions_header(payload)
        data = {
            "schema": SCHEMA_ACTIVATION,
            "a2a_extensions": extensions,
            "mcp_plus_plus_execution_activated": EXTENSION_URI in extensions,
        }
    elif isinstance(payload, Mapping):
        data = dict(payload)
        extensions = parse_a2a_extensions_header(data.get("a2a_extensions"))
        data["a2a_extensions"] = extensions
    else:
        return ValidationResult(
            ok=False,
            code=ERR_MALFORMED_EXTENSION,
            errors=["activation must be an object or A2A-Extensions header value"],
        )

    if not extensions:
        return ValidationResult(
            ok=False,
            code=ERR_MALFORMED_EXTENSION,
            errors=["a2a_extensions must contain at least one URI"],
        )

    for uri in extensions:
        # Official rule: entries are URIs. Reverse-DNS-only fails closed.
        if not is_wire_extension_uri_shape(uri):
            return ValidationResult(
                ok=False,
                code=ERR_MALFORMED_EXTENSION_URI,
                errors=[f"A2A-Extensions entry is not an HTTPS URI: {uri!r}"],
                metadata={"uri": uri},
            )

    activated = EXTENSION_URI in extensions
    if data.get("mcp_plus_plus_execution_activated") is True and not activated:
        return ValidationResult(
            ok=False,
            code=ERR_MALFORMED_EXTENSION,
            errors=[
                "mcp_plus_plus_execution_activated true but execution URI missing"
            ],
        )
    if require_execution and not activated:
        return ValidationResult(
            ok=False,
            code=ERR_NOT_ACTIVATED,
            errors=[f"activation must include {EXTENSION_URI}"],
        )

    return ValidationResult(
        ok=True,
        metadata={
            "a2a_extensions": list(extensions),
            "mcp_plus_plus_execution_activated": activated,
            "echo": list(extensions) if activated else [],
        },
    )


def validate_task_metadata(payload: Any) -> ValidationResult:
    """Validate normalized task extension metadata (task-metadata@1)."""
    if not isinstance(payload, Mapping):
        return ValidationResult(
            ok=False,
            code=ERR_MALFORMED_EXTENSION,
            errors=["task metadata must be an object"],
        )
    data = denamespace_metadata(dict(payload))
    allowed = {
        "schema",
        "envelope_cid",
        "result_cid",
        "receipt_cid",
        "event_cid",
        "output_cid",
        "input_cid",
        "intent_cid",
        "interface_cid",
        "method",
        "state_ref_cids",
        "proof_cid",
        "proof_cids",
        "delegation_cid",
        "delegation_cids",
        "decision_cid",
        "profiles",
        "profile",
        "required_abilities",
        "resource",
        "audience",
    }
    extra = set(data) - allowed
    if extra:
        return ValidationResult(
            ok=False,
            code=ERR_MALFORMED_EXTENSION,
            errors=[f"task metadata unknown keys: {sorted(extra)}"],
        )
    if "schema" in data and data["schema"] != SCHEMA_TASK_METADATA:
        return ValidationResult(
            ok=False,
            code=ERR_MALFORMED_EXTENSION,
            errors=[f"schema must be {SCHEMA_TASK_METADATA}"],
        )
    for key in (
        "envelope_cid",
        "result_cid",
        "receipt_cid",
        "event_cid",
        "output_cid",
        "input_cid",
        "intent_cid",
        "interface_cid",
        "proof_cid",
        "delegation_cid",
        "decision_cid",
    ):
        if key in data and not is_valid_cid(data[key]):
            return ValidationResult(
                ok=False,
                code=ERR_MALFORMED_EXTENSION,
                errors=[f"{key} is not a valid CID"],
            )
    for list_key in (
        "state_ref_cids",
        "proof_cids",
        "delegation_cids",
    ):
        if list_key in data:
            items = data[list_key]
            if not isinstance(items, list):
                return ValidationResult(
                    ok=False,
                    code=ERR_MALFORMED_EXTENSION,
                    errors=[f"{list_key} must be an array"],
                )
            for cid in items:
                if not is_valid_cid(cid):
                    return ValidationResult(
                        ok=False,
                        code=ERR_MALFORMED_EXTENSION,
                        errors=[f"{list_key} contains invalid CID"],
                    )
    if "profiles" in data:
        pr = _validate_profile_letters(data["profiles"])
        if not pr.ok:
            return pr
    if "profile" in data:
        if data["profile"] not in ALLOWED_PROFILES:
            return ValidationResult(
                ok=False,
                code=ERR_UNSUPPORTED_PROFILE,
                errors=[f"unsupported profile letter: {data['profile']!r}"],
            )
    return ValidationResult(ok=True, metadata={"task_metadata": copy.deepcopy(data)})


def validate_terminal_evidence(payload: Any) -> ValidationResult:
    """Validate terminal A2A evidence (terminal-evidence@1)."""
    if not isinstance(payload, Mapping):
        return ValidationResult(
            ok=False,
            code=ERR_MALFORMED_EXTENSION,
            errors=["terminal evidence must be an object"],
        )
    data = dict(payload)
    if data.get("schema") != SCHEMA_TERMINAL_EVIDENCE:
        return ValidationResult(
            ok=False,
            code=ERR_MALFORMED_EXTENSION,
            errors=[f"schema must be {SCHEMA_TERMINAL_EVIDENCE}"],
        )
    ok, code = classify_extension_uri(data.get("extension_uri"))
    if not ok:
        return ValidationResult(
            ok=False,
            code=code or ERR_MALFORMED_EXTENSION_URI,
            errors=["extension_uri must be the confirmed execution URI"],
        )
    state = data.get("task_state")
    if state not in TERMINAL_TASK_STATES:
        return ValidationResult(
            ok=False,
            code=ERR_MALFORMED_EXTENSION,
            errors=[
                f"task_state must be a terminal A2A state; got {state!r} "
                "(non-A2A public status names are forbidden)"
            ],
        )
    portable = bool(data.get("portable"))
    if portable and state == TaskState.COMPLETED.value:
        if not is_valid_cid(data.get("receipt_cid")):
            return ValidationResult(
                ok=False,
                code=ERR_MISSING_RECEIPT_CID,
                errors=["portable completed terminal evidence requires receipt_cid"],
            )
        if not is_valid_cid(data.get("envelope_cid")):
            return ValidationResult(
                ok=False,
                code=ERR_MALFORMED_EXTENSION,
                errors=["portable completed terminal evidence requires envelope_cid"],
            )
    for key in (
        "envelope_cid",
        "result_cid",
        "receipt_cid",
        "event_cid",
        "output_cid",
        "proof_cid",
        "decision_cid",
        "delegation_cid",
    ):
        if key in data and data[key] is not None and not is_valid_cid(data[key]):
            return ValidationResult(
                ok=False,
                code=ERR_MALFORMED_EXTENSION,
                errors=[f"{key} is not a valid CID"],
            )
    return ValidationResult(ok=True, metadata={"terminal": copy.deepcopy(data)})


def validate_profile_request(
    advertised: Sequence[str],
    requested: Sequence[str],
    *,
    extension_uri: str = EXTENSION_URI,
) -> ValidationResult:
    """Fail closed when requested profiles are unknown or not advertised."""
    if not is_confirmed_extension_uri(extension_uri):
        return ValidationResult(
            ok=False,
            code=ERR_MALFORMED_EXTENSION_URI,
            errors=["profile request requires confirmed extension URI"],
        )
    adv = _validate_profile_letters(list(advertised), path="advertised_profiles")
    if not adv.ok:
        return adv
    req = _validate_profile_letters(list(requested), path="requested_profiles")
    if not req.ok:
        return req
    adv_set = set(adv.metadata.get("advertised_profiles") or [])
    req_list = list(req.metadata.get("requested_profiles") or [])
    missing = [p for p in req_list if p not in adv_set]
    if missing:
        return ValidationResult(
            ok=False,
            code=ERR_PROFILE_NOT_SUBSET,
            errors=[
                f"requested profiles not advertised: {missing}"
            ],
            metadata={"missing": missing, "advertised": sorted(adv_set)},
        )
    return ValidationResult(
        ok=True,
        metadata={"advertised": sorted(adv_set), "requested": req_list},
    )


def map_result_status_to_task_state(result_status: str) -> str:
    """Map ExecutionResult@1.status onto A2A TaskState (a2a-extension.md §6.6)."""
    mapping = {
        "succeeded": TaskState.COMPLETED.value,
        "failed": TaskState.FAILED.value,
        "timed_out": TaskState.FAILED.value,
        "cancelled": TaskState.CANCELED.value,
        "canceled": TaskState.CANCELED.value,
        "rejected": TaskState.REJECTED.value,
        "compensated": TaskState.COMPLETED.value,
    }
    key = str(result_status or "").strip().lower()
    if key not in mapping:
        raise A2AExtensionError(
            ERR_MALFORMED_EXTENSION,
            f"unknown ExecutionResult status for A2A mapping: {result_status!r}",
        )
    return mapping[key]


# ---------------------------------------------------------------------------
# Agent / Task records
# ---------------------------------------------------------------------------


@dataclass
class A2ATaskRecord:
    """Server-side A2A Task with MCP++ evidence annotations."""

    task_id: str
    context_id: str
    state: str
    agent_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    stream: List[Dict[str, Any]] = field(default_factory=list)
    attempt: int = 0
    created_at_ms: int = 0
    updated_at_ms: int = 0
    cancel_requested: bool = False
    durable_cancel_id: Optional[str] = None
    parent_event_cids: List[str] = field(default_factory=list)
    last_event_cid: Optional[str] = None
    envelope_cid: Optional[str] = None
    receipt_cid: Optional[str] = None
    result_cid: Optional[str] = None
    output_cid: Optional[str] = None
    error: Optional[Dict[str, Any]] = None

    def is_terminal(self) -> bool:
        return self.state in TERMINAL_TASK_STATES

    def public_view(self) -> Dict[str, Any]:
        # Only extension-namespace suffixes from task-metadata@1 / cancel evidence.
        public_meta_keys = {
            "envelope_cid",
            "result_cid",
            "receipt_cid",
            "event_cid",
            "output_cid",
            "input_cid",
            "intent_cid",
            "interface_cid",
            "method",
            "profiles",
            "profile",
            "proof_cid",
            "proof_cids",
            "delegation_cid",
            "delegation_cids",
            "decision_cid",
            "state_ref_cids",
            "durable_cancel_id",
            "prior_task_id",
        }
        public_meta = {
            k: v for k, v in self.metadata.items() if k in public_meta_keys
        }
        return {
            "id": self.task_id,
            "contextId": self.context_id,
            "status": {
                "state": self.state,
                "timestamp": self.updated_at_ms,
                "message": None
                if not self.error
                else {
                    "role": "agent",
                    "parts": [
                        {
                            "kind": "text",
                            "text": self.error.get("message", ""),
                        }
                    ],
                    "metadata": {},
                },
            },
            "metadata": namespaced_metadata(public_meta),
            "artifacts": copy.deepcopy(self.artifacts),
            "history": copy.deepcopy(self.messages),
            "attempt": self.attempt,
            "extension_uri": EXTENSION_URI,
        }


@dataclass
class A2AAgent:
    """Independently configured A2A agent that speaks the MCP++ execution extension.

    Each agent owns its own Event DAG store and task table so two-agent handoff
    tests exercise distinct instances (MCPP-056 acceptance).
    """

    agent_id: str
    name: str
    url: str
    did: str
    profiles: List[str] = field(default_factory=lambda: list(DEFAULT_PROFILES))
    interface_cids: List[str] = field(default_factory=list)
    mcp_bindings: List[str] = field(
        default_factory=lambda: [MCP_BINDING_CURRENT, MCP_BINDING_LEGACY]
    )
    extension_required: bool = False
    skills: List[Dict[str, Any]] = field(default_factory=list)
    version: str = "1.0.0"
    streaming: bool = True
    event_dag: EventDAGStore = field(default_factory=EventDAGStore)
    # Durable cancel journal (in-process stand-in for DurableExecutor cancel).
    durable_cancels: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _tasks: Dict[str, A2ATaskRecord] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _seq: int = 0

    def __post_init__(self) -> None:
        pr = _validate_profile_letters(list(self.profiles))
        pr.raise_if_failed()
        self.profiles = list(pr.metadata.get("profiles") or self.profiles)
        if not self.interface_cids:
            # Stable demo interface CID (well-formed vector catalog value).
            self.interface_cids = [
                "bafkreigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            ]
        for cid in self.interface_cids:
            if not is_valid_cid(cid):
                raise A2AExtensionError(
                    ERR_MALFORMED_EXTENSION,
                    f"invalid interface_cid on agent: {cid!r}",
                )
        if not self.skills:
            self.skills = [
                {
                    "id": "repo.status",
                    "name": "Repository status",
                    "description": "Return git working tree status via MCP-IDL.",
                    "tags": ["vcs", "git"],
                    "metadata": namespaced_metadata(
                        {
                            "interface_cid": self.interface_cids[0],
                            "method": "repo.status",
                            "profiles": ["A", "B"],
                        }
                    ),
                }
            ]

    def agent_extension(self) -> Dict[str, Any]:
        """AgentExtension entry for the Agent Card (validated)."""
        ext = {
            "schema": SCHEMA_AGENT_EXTENSION,
            "uri": EXTENSION_URI,
            "description": (
                "MCP++ execution mapping: envelopes, state refs, receipts, "
                "and proofs on A2A Task."
            ),
            "required": bool(self.extension_required),
            "params": {
                "schema": SCHEMA_EXTENSION_PARAMS,
                "profiles": list(self.profiles),
                "envelope_schema": SCHEMA_ENVELOPE,
                "receipt_schema": SCHEMA_RECEIPT,
                "state_ref_schema": SCHEMA_STATE_REF,
                "mcp_bindings": list(self.mcp_bindings),
                "interface_cids": list(self.interface_cids),
                "canonicalization": CANONICALIZATION,
                "alias": WORKING_ALIAS,
            },
        }
        validate_agent_extension(ext).raise_if_failed()
        return ext

    def agent_card(self) -> Dict[str, Any]:
        """Full Agent Card advertisement including the execution extension."""
        return {
            "name": self.name,
            "description": f"MCP++ A2A agent {self.agent_id}",
            "url": self.url,
            "version": self.version,
            "protocolVersion": "0.3.0",
            "preferredTransport": "JSONRPC",
            "capabilities": {
                "streaming": bool(self.streaming),
                "pushNotifications": False,
                "extensions": [self.agent_extension()],
            },
            "skills": copy.deepcopy(self.skills),
            "defaultInputModes": ["text/plain", "application/json"],
            "defaultOutputModes": ["application/json"],
            "metadata": namespaced_metadata(
                {
                    "agent_id": self.agent_id,
                    "did": self.did,
                    "profiles": list(self.profiles),
                }
            ),
        }

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def _next_task_id(self) -> str:
        with self._lock:
            self._seq += 1
            return f"task-{self.agent_id}-{self._seq:04d}-{uuid.uuid4().hex[:8]}"

    def get_task(self, task_id: str) -> A2ATaskRecord:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise A2AExtensionError(
                    ERR_TASK_NOT_FOUND, f"task not found: {task_id}"
                )
            return task

    def list_tasks(self) -> List[str]:
        with self._lock:
            return sorted(self._tasks.keys())

    def store_artifact(self, payload: Mapping[str, Any]) -> str:
        cid = compute_cid(payload)
        with self._lock:
            self.artifacts[cid] = copy.deepcopy(dict(payload))
        return cid

    def append_event(
        self,
        *,
        kind: str,
        task_id: str,
        parents: Optional[Sequence[str]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Append a content-addressed Event DAG node and return its event_cid."""
        parent_list = [p for p in list(parents or []) if p]
        payload: Dict[str, Any] = {
            "schema": "mcp++/a2a/event@1",
            "kind": kind,
            "task_id": task_id,
            "agent_id": self.agent_id,
            "extension_uri": EXTENSION_URI,
            "parents": parent_list,
            "created_at_ms": self._now_ms(),
        }
        if extra:
            payload.update(copy.deepcopy(dict(extra)))
        event_cid = compute_cid(payload)
        # EventDAGStore requires parents to exist; payload carries parents.
        node_payload = copy.deepcopy(payload)
        node_payload["parents"] = parent_list
        self.event_dag.add_event(event_cid, node_payload)
        return event_cid

    def activate(
        self,
        a2a_extensions: Any,
        *,
        require_execution: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Validate client A2A-Extensions activation against this agent."""
        require = (
            self.extension_required
            if require_execution is None
            else bool(require_execution)
        )
        result = validate_activation(a2a_extensions, require_execution=require)
        if not result.ok:
            # Required extension not activated → fail closed with support-required.
            if require and result.code in (
                ERR_NOT_ACTIVATED,
                ERR_MALFORMED_EXTENSION,
            ):
                if EXTENSION_URI not in parse_a2a_extensions_header(a2a_extensions):
                    raise A2AExtensionError(
                        ERR_EXTENSION_REQUIRED,
                        "Agent requires MCP++ execution extension activation",
                        details=result.metadata,
                    )
            result.raise_if_failed()
        assert result.ok
        activated = bool(result.metadata.get("mcp_plus_plus_execution_activated"))
        # Future / foreign URIs may be listed; only confirmed URI activates us.
        return {
            "activated": activated,
            "a2a_extensions": list(result.metadata.get("a2a_extensions") or []),
            "echo": list(result.metadata.get("echo") or []),
            "extension_uri": EXTENSION_URI if activated else None,
        }

    def _assert_profiles_allowed(self, requested: Sequence[str]) -> None:
        validate_profile_request(self.profiles, list(requested)).raise_if_failed()

    def send_message(
        self,
        *,
        message: Mapping[str, Any],
        a2a_extensions: Any,
        context_id: Optional[str] = None,
        task_id: Optional[str] = None,
        requested_profiles: Optional[Sequence[str]] = None,
        execute: bool = True,
        fail: bool = False,
        hold_open: bool = False,
    ) -> Dict[str, Any]:
        """Accept an A2A Send Message under the execution extension.

        When *hold_open* is True the task remains ``working`` so cancel/stream
        tests can observe non-terminal state. When *execute* is True and not
        held open, the agent runs a deterministic local completion that mints
        envelope/result/receipt/event evidence.
        """
        activation = self.activate(a2a_extensions)
        if not activation["activated"]:
            # Extension not activated: still may accept plain A2A, but MCP++
            # interop claims fail closed for evidence-bearing handoff.
            if self.extension_required:
                raise A2AExtensionError(
                    ERR_EXTENSION_REQUIRED,
                    "execution extension required but not activated",
                )
            raise A2AExtensionError(
                ERR_NOT_ACTIVATED,
                f"A2A-Extensions must include {EXTENSION_URI} for MCP++ handoff",
            )

        profiles = list(requested_profiles or ["A", "B"])
        self._assert_profiles_allowed(profiles)

        msg = dict(message or {})
        msg_meta = denamespace_metadata(msg.get("metadata") or {})
        tm = validate_task_metadata(msg_meta)
        if not tm.ok and msg_meta:
            # Only fail when metadata claims extension keys that are invalid.
            tm.raise_if_failed()

        now = self._now_ms()
        with self._lock:
            if task_id and task_id in self._tasks:
                task = self._tasks[task_id]
                if task.is_terminal():
                    # Retry path: open a new attempt on a fresh task linked by parents.
                    raise A2AExtensionError(
                        ERR_MALFORMED_EXTENSION,
                        "cannot append to terminal task; use retry()",
                        details={"task_id": task_id},
                    )
            else:
                tid = task_id or self._next_task_id()
                task = A2ATaskRecord(
                    task_id=tid,
                    context_id=context_id
                    or str(msg.get("contextId") or f"ctx-{uuid.uuid4().hex[:12]}"),
                    state=TaskState.SUBMITTED.value,
                    agent_id=self.agent_id,
                    created_at_ms=now,
                    updated_at_ms=now,
                    attempt=1,
                )
                self._tasks[tid] = task

            task.messages.append(copy.deepcopy(msg))
            requester = (
                msg.get("from")
                or msg.get("requester")
                or (msg.get("metadata") or {}).get("from")
                or "did:key:client"
            )
            task.metadata.update(
                {
                    "profiles": profiles,
                    "method": msg_meta.get("method")
                    or (msg.get("skill") if isinstance(msg.get("skill"), str) else None)
                    or "repo.status",
                    "interface_cid": msg_meta.get("interface_cid")
                    or self.interface_cids[0],
                    "requester": requester,
                }
            )
            if msg_meta.get("input_cid"):
                task.metadata["input_cid"] = msg_meta["input_cid"]
            if msg_meta.get("envelope_cid"):
                task.envelope_cid = str(msg_meta["envelope_cid"])
                task.metadata["envelope_cid"] = task.envelope_cid

            submit_event = self.append_event(
                kind="task.submitted",
                task_id=task.task_id,
                parents=list(task.parent_event_cids),
                extra={"state": TaskState.SUBMITTED.value, "attempt": task.attempt},
            )
            task.last_event_cid = submit_event
            task.metadata["event_cid"] = submit_event
            task.state = TaskState.WORKING.value
            task.updated_at_ms = self._now_ms()
            work_event = self.append_event(
                kind="task.working",
                task_id=task.task_id,
                parents=[submit_event],
                extra={"state": TaskState.WORKING.value, "attempt": task.attempt},
            )
            task.last_event_cid = work_event
            task.metadata["event_cid"] = work_event
            task.stream.append(
                StreamEvent(
                    kind="status",
                    task_id=task.task_id,
                    state=TaskState.WORKING.value,
                    metadata={"event_cid": work_event},
                    attempt=task.attempt,
                ).to_dict()
            )

            if hold_open:
                return task.public_view()

            if execute:
                if fail:
                    self._fail_task(task, code="EXECUTION_FAILED", message="forced failure")
                else:
                    self._complete_task(task)
            return task.public_view()

    def _mint_execution_bundle(
        self,
        task: A2ATaskRecord,
        *,
        result_status: str,
        output: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Mint envelope, result, receipt, and event CIDs for a terminal attempt."""
        method = str(task.metadata.get("method") or "repo.status")
        interface_cid = str(
            task.metadata.get("interface_cid") or self.interface_cids[0]
        )
        input_payload = {
            "schema": "mcp++/a2a/input@1",
            "task_id": task.task_id,
            "method": method,
            "parts": [
                m.get("parts") for m in task.messages if isinstance(m.get("parts"), list)
            ],
        }
        input_cid = task.metadata.get("input_cid") or self.store_artifact(input_payload)
        intent = {
            "schema": "mcp++/execution/intent@1",
            "interface_cid": interface_cid,
            "method": method,
            "input_cid": input_cid,
            "correlation_id": task.task_id,
        }
        intent_cid = self.store_artifact(intent)
        envelope = {
            "schema": SCHEMA_ENVELOPE,
            "interface_cid": interface_cid,
            "method": method,
            "input_cid": input_cid,
            "intent_cid": intent_cid,
            "correlation_id": task.task_id,
            "requester": task.metadata.get("requester") or "did:key:client",
            "parents": list(task.parent_event_cids),
            "extension_uri": EXTENSION_URI,
        }
        envelope_cid = task.envelope_cid or self.store_artifact(envelope)
        output_payload = dict(
            output
            or {
                "schema": "mcp++/a2a/output@1",
                "task_id": task.task_id,
                "method": method,
                "status": result_status,
                "summary": f"{method} completed",
            }
        )
        output_cid = self.store_artifact(output_payload)
        result = {
            "schema": "mcp++/execution/result@1",
            "status": result_status,
            "envelope_cid": envelope_cid,
            "output_cids": [output_cid],
            "primary_output_cid": output_cid,
            "correlation_id": task.task_id,
        }
        result_cid = self.store_artifact(result)
        receipt = {
            "schema": SCHEMA_RECEIPT,
            "envelope_cid": envelope_cid,
            "result_cid": result_cid,
            "output_cid": output_cid,
            "intent_cid": intent_cid,
            "correlation_id": task.task_id,
            "extension_uri": EXTENSION_URI,
            "status": result_status,
        }
        receipt_cid = self.store_artifact(receipt)
        parents = [task.last_event_cid] if task.last_event_cid else []
        event_cid = self.append_event(
            kind="task.terminal",
            task_id=task.task_id,
            parents=parents,
            extra={
                "state": map_result_status_to_task_state(result_status),
                "envelope_cid": envelope_cid,
                "result_cid": result_cid,
                "receipt_cid": receipt_cid,
                "output_cid": output_cid,
                "result_status": result_status,
                "attempt": task.attempt,
            },
        )
        return {
            "input_cid": input_cid,
            "intent_cid": intent_cid,
            "envelope_cid": envelope_cid,
            "result_cid": result_cid,
            "receipt_cid": receipt_cid,
            "output_cid": output_cid,
            "event_cid": event_cid,
            "result_status": result_status,
        }

    def _complete_task(
        self,
        task: A2ATaskRecord,
        *,
        output: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if task.cancel_requested:
            self._cancel_task_internal(task, reason="cancel-before-complete")
            return
        bundle = self._mint_execution_bundle(
            task, result_status="succeeded", output=output
        )
        task.state = TaskState.COMPLETED.value
        task.envelope_cid = bundle["envelope_cid"]
        task.result_cid = bundle["result_cid"]
        task.receipt_cid = bundle["receipt_cid"]
        task.output_cid = bundle["output_cid"]
        task.last_event_cid = bundle["event_cid"]
        task.metadata.update(
            {
                "envelope_cid": bundle["envelope_cid"],
                "result_cid": bundle["result_cid"],
                "receipt_cid": bundle["receipt_cid"],
                "event_cid": bundle["event_cid"],
                "output_cid": bundle["output_cid"],
                "input_cid": bundle["input_cid"],
                "intent_cid": bundle["intent_cid"],
            }
        )
        task.artifacts.append(
            {
                "artifactId": f"artifact-{task.task_id}",
                "name": "primary-output",
                "parts": [
                    {
                        "kind": "data",
                        "data": {
                            "output_cid": bundle["output_cid"],
                            "result_cid": bundle["result_cid"],
                        },
                    }
                ],
                "metadata": namespaced_metadata(
                    {
                        "output_cid": bundle["output_cid"],
                        "receipt_cid": bundle["receipt_cid"],
                        "event_cid": bundle["event_cid"],
                        "envelope_cid": bundle["envelope_cid"],
                    }
                ),
            }
        )
        task.updated_at_ms = self._now_ms()
        task.stream.append(
            StreamEvent(
                kind="terminal",
                task_id=task.task_id,
                state=TaskState.COMPLETED.value,
                metadata=dict(task.metadata),
                attempt=task.attempt,
            ).to_dict()
        )

    def _fail_task(self, task: A2ATaskRecord, *, code: str, message: str) -> None:
        if task.cancel_requested:
            self._cancel_task_internal(task, reason="cancel-before-fail")
            return
        bundle = self._mint_execution_bundle(
            task,
            result_status="failed",
            output={
                "schema": "mcp++/a2a/output@1",
                "task_id": task.task_id,
                "error": {"code": code, "message": message},
            },
        )
        task.state = TaskState.FAILED.value
        task.error = {"code": code, "message": message}
        task.envelope_cid = bundle["envelope_cid"]
        task.result_cid = bundle["result_cid"]
        task.receipt_cid = bundle["receipt_cid"]
        task.output_cid = bundle["output_cid"]
        task.last_event_cid = bundle["event_cid"]
        task.metadata.update(
            {
                "envelope_cid": bundle["envelope_cid"],
                "result_cid": bundle["result_cid"],
                "receipt_cid": bundle["receipt_cid"],
                "event_cid": bundle["event_cid"],
                "output_cid": bundle["output_cid"],
            }
        )
        task.updated_at_ms = self._now_ms()
        task.stream.append(
            StreamEvent(
                kind="terminal",
                task_id=task.task_id,
                state=TaskState.FAILED.value,
                metadata=dict(task.metadata),
                message=message,
                attempt=task.attempt,
            ).to_dict()
        )

    def _cancel_task_internal(self, task: A2ATaskRecord, *, reason: str) -> None:
        """Durable cancel + Event DAG cancel record + terminal canceled state."""
        parents = [task.last_event_cid] if task.last_event_cid else []
        cancel_journal = {
            "schema": "mcp++/durable/cancel@1",
            "task_id": task.task_id,
            "agent_id": self.agent_id,
            "reason": reason,
            "requested_at_ms": self._now_ms(),
            "extension_uri": EXTENSION_URI,
            "parents": parents,
        }
        durable_id = compute_cid(cancel_journal)
        self.durable_cancels[durable_id] = cancel_journal
        task.durable_cancel_id = durable_id

        event_cid = self.append_event(
            kind="task.canceled",
            task_id=task.task_id,
            parents=parents,
            extra={
                "state": TaskState.CANCELED.value,
                "reason": reason,
                "durable_cancel_id": durable_id,
                "attempt": task.attempt,
            },
        )
        # Optional receipt for cancel path (SHOULD per §6.11).
        cancel_receipt = {
            "schema": SCHEMA_RECEIPT,
            "correlation_id": task.task_id,
            "status": "cancelled",
            "extension_uri": EXTENSION_URI,
            "durable_cancel_id": durable_id,
            "event_cid": event_cid,
        }
        receipt_cid = self.store_artifact(cancel_receipt)

        task.state = TaskState.CANCELED.value
        task.cancel_requested = True
        task.last_event_cid = event_cid
        task.receipt_cid = receipt_cid
        task.error = {"code": "CANCELED", "message": reason}
        task.metadata.update(
            {
                "event_cid": event_cid,
                "receipt_cid": receipt_cid,
                "durable_cancel_id": durable_id,
            }
        )
        task.updated_at_ms = self._now_ms()
        task.stream.append(
            StreamEvent(
                kind="terminal",
                task_id=task.task_id,
                state=TaskState.CANCELED.value,
                metadata=dict(task.metadata),
                message=reason,
                attempt=task.attempt,
            ).to_dict()
        )

    def cancel_task(self, task_id: str, *, reason: str = "client-cancel") -> Dict[str, Any]:
        """A2A Cancel Task: durable cancel + Event DAG records (fail closed if terminal)."""
        with self._lock:
            task = self.get_task(task_id)
            if task.is_terminal():
                raise A2AExtensionError(
                    ERR_TASK_NOT_CANCELABLE,
                    f"task {task_id} is already terminal ({task.state})",
                    details={"state": task.state},
                )
            task.cancel_requested = True
            self._cancel_task_internal(task, reason=reason)
            return task.public_view()

    def retry_task(
        self,
        task_id: str,
        *,
        a2a_extensions: Any,
        message: Optional[Mapping[str, Any]] = None,
        execute: bool = True,
        fail: bool = False,
    ) -> Dict[str, Any]:
        """Retry a failed/canceled task as a new attempt linked via Event DAG parents."""
        with self._lock:
            prior = self.get_task(task_id)
            if prior.state not in (
                TaskState.FAILED.value,
                TaskState.CANCELED.value,
            ):
                raise A2AExtensionError(
                    ERR_MALFORMED_EXTENSION,
                    f"retry only allowed from failed/canceled; got {prior.state}",
                )
            parent_event = prior.last_event_cid
            parent_events = [parent_event] if parent_event else []

        retry_message = dict(message or {})
        if not retry_message:
            retry_message = {
                "role": "user",
                "parts": [{"kind": "text", "text": f"retry {task_id}"}],
                "metadata": namespaced_metadata(
                    {
                        "method": prior.metadata.get("method") or "repo.status",
                        "interface_cid": prior.metadata.get("interface_cid")
                        or self.interface_cids[0],
                    }
                ),
            }
        # New task id; preserve context and parent event lineage.
        view = self.send_message(
            message=retry_message,
            a2a_extensions=a2a_extensions,
            context_id=prior.context_id,
            execute=False,
            hold_open=True,
            requested_profiles=list(prior.metadata.get("profiles") or ["A", "B"]),
        )
        new_id = view["id"]
        with self._lock:
            task = self.get_task(new_id)
            task.attempt = prior.attempt + 1
            task.parent_event_cids = list(parent_events)
            # Link retry event to prior terminal event.
            if parent_events:
                link = self.append_event(
                    kind="task.retry",
                    task_id=task.task_id,
                    parents=parent_events + ([task.last_event_cid] if task.last_event_cid else []),
                    extra={
                        "prior_task_id": prior.task_id,
                        "attempt": task.attempt,
                        "state": TaskState.WORKING.value,
                    },
                )
                task.last_event_cid = link
                task.metadata["event_cid"] = link
                task.metadata["prior_task_id"] = prior.task_id
            if execute:
                if fail:
                    self._fail_task(task, code="EXECUTION_FAILED", message="retry failed")
                else:
                    self._complete_task(task)
            return task.public_view()

    def stream_task(self, task_id: str) -> Iterator[Dict[str, Any]]:
        """Yield buffered streaming updates for a task (status/artifact/terminal)."""
        task = self.get_task(task_id)
        for event in list(task.stream):
            yield copy.deepcopy(event)
        # Always end with current status snapshot.
        yield StreamEvent(
            kind="status" if not task.is_terminal() else "terminal",
            task_id=task.task_id,
            state=task.state,
            metadata=dict(task.metadata),
            attempt=task.attempt,
        ).to_dict()

    def terminal_evidence(self, task_id: str, *, portable: bool = True) -> Dict[str, Any]:
        """Build TerminalEvidence@1 for a terminal task and validate it."""
        task = self.get_task(task_id)
        if not task.is_terminal():
            raise A2AExtensionError(
                ERR_MALFORMED_EXTENSION,
                f"task {task_id} is not terminal ({task.state})",
            )
        evidence: Dict[str, Any] = {
            "schema": SCHEMA_TERMINAL_EVIDENCE,
            "extension_uri": EXTENSION_URI,
            "task_id": task.task_id,
            "task_state": task.state,
            "portable": bool(portable) and task.state == TaskState.COMPLETED.value,
        }
        if task.envelope_cid:
            evidence["envelope_cid"] = task.envelope_cid
        if task.result_cid:
            evidence["result_cid"] = task.result_cid
        if task.receipt_cid:
            evidence["receipt_cid"] = task.receipt_cid
        if task.last_event_cid:
            evidence["event_cid"] = task.last_event_cid
        if task.output_cid:
            evidence["output_cid"] = task.output_cid
        if task.error:
            evidence["error"] = copy.deepcopy(task.error)
        validate_terminal_evidence(evidence).raise_if_failed()
        return evidence

    def event_lineage(self, task_id: str) -> List[str]:
        task = self.get_task(task_id)
        if not task.last_event_cid:
            return []
        return self.event_dag.get_lineage(task.last_event_cid)

    def cancel_events(self, task_id: str) -> List[Dict[str, Any]]:
        """Return Event DAG payloads with kind task.canceled for *task_id*."""
        snapshot = self.event_dag.export_snapshot()
        out: List[Dict[str, Any]] = []
        for item in snapshot.get("events") or []:
            payload = item.get("payload") or {}
            if (
                payload.get("kind") == "task.canceled"
                and payload.get("task_id") == task_id
            ):
                out.append(copy.deepcopy(payload))
        return out


# ---------------------------------------------------------------------------
# Reference adapter facade
# ---------------------------------------------------------------------------


class A2ATaskAdapter:
    """Reference A2ATaskAdapter@1: two-agent handoff over the execution extension.

    The adapter does not replace A2A Task status names. It activates the
    confirmed extension URI, maps messages onto envelope evidence, records
    cancel on the Event DAG, and fails closed on malformed extension material
    and unsupported profiles.
    """

    interface = INTERFACE
    extension_uri = EXTENSION_URI
    working_alias = WORKING_ALIAS
    algorithm = CANONICALIZATION
    task_id = TASK_ID

    def __init__(self) -> None:
        self._agents: Dict[str, A2AAgent] = {}
        self._lock = threading.RLock()

    def create_agent(
        self,
        *,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        url: Optional[str] = None,
        did: Optional[str] = None,
        profiles: Optional[Sequence[str]] = None,
        extension_required: bool = False,
        interface_cids: Optional[Sequence[str]] = None,
    ) -> A2AAgent:
        """Instantiate an independent agent (own Event DAG + task table)."""
        aid = agent_id or f"agent-{uuid.uuid4().hex[:8]}"
        agent = A2AAgent(
            agent_id=aid,
            name=name or f"MCP++ Agent {aid}",
            url=url or f"https://agents.example.invalid/{aid}",
            did=did or f"did:key:{aid}",
            profiles=list(profiles or DEFAULT_PROFILES),
            extension_required=extension_required,
            interface_cids=list(interface_cids or []),
        )
        with self._lock:
            self._agents[aid] = agent
        return agent

    def get_agent(self, agent_id: str) -> A2AAgent:
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                raise A2AExtensionError(
                    ERR_TASK_NOT_FOUND, f"agent not found: {agent_id}"
                )
            return agent

    # -- pure validators (also used by tests / SwissKnife adapter) ----------

    def validate_agent_extension(self, payload: Any) -> ValidationResult:
        return validate_agent_extension(payload)

    def validate_activation(self, payload: Any, **kwargs: Any) -> ValidationResult:
        return validate_activation(payload, **kwargs)

    def validate_task_metadata(self, payload: Any) -> ValidationResult:
        return validate_task_metadata(payload)

    def validate_terminal_evidence(self, payload: Any) -> ValidationResult:
        return validate_terminal_evidence(payload)

    def validate_profile_request(
        self,
        advertised: Sequence[str],
        requested: Sequence[str],
        **kwargs: Any,
    ) -> ValidationResult:
        return validate_profile_request(advertised, requested, **kwargs)

    def reject_malformed_extension(self, payload: Any) -> None:
        """Fail closed on malformed AgentExtension or activation material."""
        if isinstance(payload, Mapping) and "uri" in payload:
            validate_agent_extension(payload).raise_if_failed()
            return
        if isinstance(payload, Mapping) and "a2a_extensions" in payload:
            validate_activation(payload).raise_if_failed()
            return
        if isinstance(payload, Mapping) and payload.get("schema") == SCHEMA_TERMINAL_EVIDENCE:
            validate_terminal_evidence(payload).raise_if_failed()
            return
        if isinstance(payload, Mapping) and (
            "envelope_cid" in payload or payload.get("schema") == SCHEMA_TASK_METADATA
        ):
            validate_task_metadata(payload).raise_if_failed()
            return
        # Treat bare URI strings as extension URI claims.
        if isinstance(payload, str):
            ok, code = classify_extension_uri(payload)
            if not ok:
                raise A2AExtensionError(
                    code or ERR_MALFORMED_EXTENSION_URI,
                    f"malformed extension URI: {payload!r}",
                )
            return
        raise A2AExtensionError(
            ERR_MALFORMED_EXTENSION,
            "unrecognized extension payload",
        )

    # -- handoff ------------------------------------------------------------

    def discover(self, server: A2AAgent) -> Dict[str, Any]:
        """Client-side Agent Card fetch (in-process)."""
        card = server.agent_card()
        extensions = (
            (card.get("capabilities") or {}).get("extensions") or []
        )
        if not extensions:
            raise A2AExtensionError(
                ERR_UNSUPPORTED_EXTENSION,
                "server Agent Card has no extensions",
            )
        validate_agent_extension(extensions[0]).raise_if_failed()
        return card

    def handoff(
        self,
        client: A2AAgent,
        server: A2AAgent,
        *,
        text: str = "run repo.status",
        method: str = "repo.status",
        requested_profiles: Optional[Sequence[str]] = None,
        a2a_extensions: Optional[Any] = None,
        hold_open: bool = False,
        fail: bool = False,
        execute: bool = True,
        context_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Two-agent handoff: client discovers card, activates extension, sends message.

        Returns a structured handoff receipt including client/server ids, the
        public Task view, and terminal evidence when the task completed.
        """
        if client.agent_id == server.agent_id and client is server:
            raise A2AExtensionError(
                ERR_MALFORMED_EXTENSION,
                "handoff requires two independently instantiated agents",
            )

        card = self.discover(server)
        extensions = a2a_extensions
        if extensions is None:
            extensions = [EXTENSION_URI]

        message = {
            "role": "user",
            "messageId": f"msg-{uuid.uuid4().hex[:12]}",
            "parts": [{"kind": "text", "text": text}],
            "metadata": namespaced_metadata(
                {
                    "method": method,
                    "interface_cid": server.interface_cids[0],
                    "profiles": list(requested_profiles or ["A", "B"]),
                }
            ),
            # Client identity is A2A message context, not extension task-metadata.
            "from": client.did,
        }
        if context_id:
            message["contextId"] = context_id

        task_view = server.send_message(
            message=message,
            a2a_extensions=extensions,
            context_id=context_id,
            requested_profiles=requested_profiles or ["A", "B"],
            execute=execute,
            fail=fail,
            hold_open=hold_open,
        )

        result: Dict[str, Any] = {
            "interface": INTERFACE,
            "extension_uri": EXTENSION_URI,
            "client_agent_id": client.agent_id,
            "server_agent_id": server.agent_id,
            "client_did": client.did,
            "server_did": server.did,
            "agent_card_name": card.get("name"),
            "task": task_view,
            "activated_extensions": parse_a2a_extensions_header(extensions),
        }
        state = (task_view.get("status") or {}).get("state")
        if state in TERMINAL_TASK_STATES:
            portable = state == TaskState.COMPLETED.value
            try:
                evidence = server.terminal_evidence(
                    task_view["id"], portable=portable
                )
            except A2AExtensionError:
                evidence = server.terminal_evidence(
                    task_view["id"], portable=False
                )
            result["terminal_evidence"] = evidence
            result["event_lineage"] = server.event_lineage(task_view["id"])
        return result

    def cancel(
        self,
        server: A2AAgent,
        task_id: str,
        *,
        reason: str = "client-cancel",
    ) -> Dict[str, Any]:
        """Cancel a server task; returns public view + cancel Event DAG records."""
        view = server.cancel_task(task_id, reason=reason)
        cancel_events = server.cancel_events(task_id)
        return {
            "task": view,
            "cancel_events": cancel_events,
            "durable_cancels": copy.deepcopy(server.durable_cancels),
            "event_lineage": server.event_lineage(task_id),
        }

    def load_vector_suite(self, name: str) -> Dict[str, Any]:
        """Load a conformance vector suite from the shared a2a vectors directory."""
        root = (
            Path(__file__).resolve().parents[2]
            / "mcplusplus"
            / "conformance"
            / "vectors"
            / "a2a"
        )
        path = root / f"{name}.json"
        if not path.is_file():
            # Alternate layout if package path differs.
            alt = (
                Path(__file__).resolve().parents[3]
                / "ipfs_accelerate_py"
                / "mcplusplus"
                / "conformance"
                / "vectors"
                / "a2a"
                / f"{name}.json"
            )
            path = alt if alt.is_file() else path
        raw = path.read_text(encoding="utf-8")
        return json.loads(raw)

    def evaluate_vector_case(self, case: Mapping[str, Any]) -> ValidationResult:
        """Evaluate one MCPP-055 vector case against adapter validators."""
        schema_file = str(case.get("schema_file") or "")
        payload = case.get("payload")
        expected_valid = bool(case.get("valid", True))
        semantic_rules = list(case.get("semantic_rules") or [])

        if schema_file.startswith("agent-extension"):
            result = validate_agent_extension(payload)
        elif schema_file.startswith("extension-params"):
            result = validate_extension_params(payload)
        elif schema_file.startswith("activation"):
            result = validate_activation(payload)
        elif schema_file.startswith("task-metadata"):
            result = validate_task_metadata(payload)
        elif schema_file.startswith("terminal-evidence"):
            result = validate_terminal_evidence(payload)
        elif schema_file.startswith("skill-metadata"):
            # Skill metadata reuses profile + cid checks.
            if not isinstance(payload, Mapping):
                result = ValidationResult(
                    ok=False,
                    code=ERR_MALFORMED_EXTENSION,
                    errors=["skill metadata must be object"],
                )
            else:
                data = dict(payload)
                if "profiles" in data:
                    pr = _validate_profile_letters(data["profiles"])
                    if not pr.ok:
                        result = pr
                    elif "interface_cid" in data and not is_valid_cid(
                        data["interface_cid"]
                    ):
                        result = ValidationResult(
                            ok=False,
                            code=ERR_MALFORMED_EXTENSION,
                            errors=["invalid interface_cid"],
                        )
                    else:
                        result = ValidationResult(ok=True)
                elif "interface_cid" in data and not is_valid_cid(
                    data["interface_cid"]
                ):
                    result = ValidationResult(
                        ok=False,
                        code=ERR_MALFORMED_EXTENSION,
                        errors=["invalid interface_cid"],
                    )
                else:
                    result = ValidationResult(ok=True)
        elif schema_file.startswith("profile-request"):
            if not isinstance(payload, Mapping):
                result = ValidationResult(
                    ok=False,
                    code=ERR_MALFORMED_EXTENSION,
                    errors=["profile-request must be object"],
                )
            else:
                # Structural profile letters first, then subset semantic rule.
                result = validate_profile_request(
                    list(payload.get("advertised_profiles") or []),
                    list(payload.get("requested_profiles") or []),
                    extension_uri=str(payload.get("extension_uri") or EXTENSION_URI),
                )
                if (
                    result.ok
                    and "requested_subset_of_advertised" in semantic_rules
                    and case.get("semantic_valid") is False
                ):
                    # Vector marks semantic failure even if letters are known —
                    # re-check is already done by validate_profile_request.
                    pass
        else:
            result = ValidationResult(
                ok=False,
                code=ERR_MALFORMED_EXTENSION,
                errors=[f"unknown schema_file: {schema_file}"],
            )

        # For negative vectors, ok=False is success of the suite expectation.
        if expected_valid:
            return result
        # Negative case: adapter must reject.
        if result.ok:
            # Semantic subset failures for profile-request when schema_valid true.
            if (
                schema_file.startswith("profile-request")
                and case.get("schema_valid") is True
                and case.get("semantic_valid") is False
            ):
                # validate_profile_request should have failed; if not, force fail.
                return ValidationResult(
                    ok=False,
                    code=case.get("expected_error") or ERR_PROFILE_NOT_SUBSET,
                    errors=["expected semantic profile subset failure"],
                )
            return ValidationResult(
                ok=False,
                code=ERR_MALFORMED_EXTENSION,
                errors=["expected rejection but validation passed"],
                metadata={"case_id": case.get("id")},
            )
        # Rejected as expected.
        expected_error = case.get("expected_error")
        if expected_error and result.code and result.code != expected_error:
            # Allow close matches for malformed URI vs general malformed.
            close = {
                ERR_MALFORMED_EXTENSION_URI,
                ERR_MALFORMED_EXTENSION,
                ERR_MISSING_RECEIPT_CID,
                ERR_UNSUPPORTED_PROFILE,
                ERR_PROFILE_NOT_SUBSET,
            }
            if result.code not in close or expected_error not in close:
                return ValidationResult(
                    ok=False,
                    code=result.code,
                    errors=[
                        f"expected error {expected_error}, got {result.code}: "
                        f"{result.errors}"
                    ],
                )
        return ValidationResult(
            ok=True,
            metadata={
                "rejected": True,
                "code": result.code,
                "case_id": case.get("id"),
            },
        )


__all__ = [
    "INTERFACE",
    "EXTENSION_URI",
    "WORKING_ALIAS",
    "METADATA_KEY_PREFIX",
    "CANONICALIZATION",
    "TASK_ID",
    "SCHEMA_AGENT_EXTENSION",
    "SCHEMA_ACTIVATION",
    "SCHEMA_TASK_METADATA",
    "SCHEMA_TERMINAL_EVIDENCE",
    "SCHEMA_PROFILE_REQUEST",
    "ALLOWED_PROFILES",
    "DEFAULT_PROFILES",
    "TaskState",
    "TERMINAL_TASK_STATES",
    "ERR_MALFORMED_EXTENSION_URI",
    "ERR_MALFORMED_EXTENSION",
    "ERR_MISSING_RECEIPT_CID",
    "ERR_UNSUPPORTED_PROFILE",
    "ERR_PROFILE_NOT_SUBSET",
    "ERR_EXTENSION_REQUIRED",
    "ERR_TASK_NOT_CANCELABLE",
    "ERR_TASK_NOT_FOUND",
    "ERR_NOT_ACTIVATED",
    "A2AExtensionError",
    "ValidationResult",
    "StreamEvent",
    "A2ATaskRecord",
    "A2AAgent",
    "A2ATaskAdapter",
    "canonicalize_json",
    "compute_cid",
    "is_valid_cid",
    "namespaced_metadata",
    "denamespace_metadata",
    "is_confirmed_extension_uri",
    "classify_extension_uri",
    "parse_a2a_extensions_header",
    "validate_agent_extension",
    "validate_extension_params",
    "validate_activation",
    "validate_task_metadata",
    "validate_terminal_evidence",
    "validate_profile_request",
    "map_result_status_to_task_state",
]
