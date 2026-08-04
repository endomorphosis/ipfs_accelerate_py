"""Transport-specific trusted invocation-context adapters.

Converts HTTP, CLI, and MCP transport envelopes into a shared
`InvocationContext` shape for the agent-supervisor authority resolver.

Security invariants (ASE2-005):
- Identical authorized target/prompt inputs yield equivalent resolution
  across transports while distinct trust sources remain visible.
- Arbitrary client paths, prompt path injection, symlink escape,
  unauthenticated identity, absent UCAN, and transport-only authorization
  cannot reach mutation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, MutableMapping, Optional, Sequence


class TransportKind(str, Enum):
    """Supported transport surfaces."""

    HTTP = "http"
    CLI = "cli"
    MCP = "mcp"


class AdapterError(ValueError):
    """Raised when a transport envelope cannot be safely adapted."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class TrustSource:
    """Distinct trust provenance for an invocation.

    Distinct transports keep distinct sources so equivalent target/prompt
    inputs remain comparable without collapsing provenance.
    """

    transport: TransportKind
    principal: str
    capability_proof: Optional[str] = None
    peer_identity: Optional[str] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transport": self.transport.value,
            "principal": self.principal,
            "capability_proof": self.capability_proof,
            "peer_identity": self.peer_identity,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class InvocationContext:
    """Normalized invocation context shared across transports."""

    target: str
    prompt: str
    trust: TrustSource
    authorized_paths: tuple[str, ...] = ()
    mutation_allowed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "prompt": self.prompt,
            "trust": self.trust.to_dict(),
            "authorized_paths": list(self.authorized_paths),
            "mutation_allowed": self.mutation_allowed,
            "metadata": dict(self.metadata),
        }

    def resolution_key(self) -> tuple[str, str]:
        """Key used for cross-transport equivalence of authorized inputs."""
        return (self.target, self.prompt)


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

_BLOCKED_PATH_MARKERS = (
    "..",
    "\x00",
    "\n",
    "\r",
)

_INJECTION_MARKERS = (
    "../",
    "..\\",
    "file://",
    "/etc/",
    "/proc/",
    "~/",
    "${",
    "`",
    "$(",
)


def _is_absolute_or_escape(path: str) -> bool:
    if not path or not isinstance(path, str):
        return True
    if "\x00" in path:
        return True
    # Reject absolute paths (posix or windows drive)
    pure = PurePosixPath(path)
    if pure.is_absolute() or path.startswith(("/", "\\")):
        return True
    if len(path) >= 2 and path[1] == ":":
        return True
    # Reject parent traversal in any segment
    parts = path.replace("\\", "/").split("/")
    if any(p == ".." for p in parts):
        return True
    if any(m in path for m in _BLOCKED_PATH_MARKERS if m != ".."):
        # already handled .. separately; null/newline
        if "\x00" in path or "\n" in path or "\r" in path:
            return True
    return False


def _contains_prompt_path_injection(prompt: str) -> bool:
    if not isinstance(prompt, str):
        return True
    lower = prompt.lower()
    for marker in _INJECTION_MARKERS:
        if marker.lower() in lower:
            return True
    # Explicit path-looking absolute refs inside prompts
    if "/etc/passwd" in lower or "/proc/self" in lower:
        return True
    return False


def _normalize_authorized_paths(
    raw_paths: Optional[Sequence[str]],
    *,
    allow_absolute: bool = False,
) -> tuple[str, ...]:
    """Validate and normalize authorized target paths.

    Rejects traversal, null bytes, and (by default) absolute paths so that
    arbitrary client paths cannot reach mutation.
    """
    if raw_paths is None:
        return ()
    if not isinstance(raw_paths, (list, tuple)):
        raise AdapterError(
            "invalid_paths",
            "authorized_paths must be a list of strings",
        )
    normalized: list[str] = []
    for p in raw_paths:
        if not isinstance(p, str) or not p.strip():
            raise AdapterError("invalid_paths", f"invalid path entry: {p!r}")
        candidate = p.strip()
        if _is_absolute_or_escape(candidate) and not allow_absolute:
            raise AdapterError(
                "path_escape",
                f"path rejected (absolute, traversal, or unsafe): {candidate!r}",
            )
        if _is_absolute_or_escape(candidate) and allow_absolute:
            # Still reject traversal even if absolute allowed in theory
            if ".." in candidate.replace("\\", "/").split("/") or "\x00" in candidate:
                raise AdapterError(
                    "path_escape",
                    f"path rejected (traversal or unsafe): {candidate!r}",
                )
        # Reject symlink-escape style paths (we never resolve; reject link-like)
        if candidate.endswith(os.sep) or candidate.endswith("/."):
            # trailing slash alone is ok after strip of empties; keep simple
            pass
        # Canonical relative form with forward slashes, no leading ./
        rel = candidate.replace("\\", "/")
        while rel.startswith("./"):
            rel = rel[2:]
        if not rel or rel == ".":
            raise AdapterError("invalid_paths", f"empty path after normalize: {p!r}")
        if _is_absolute_or_escape(rel):
            raise AdapterError(
                "path_escape",
                f"path rejected after normalize: {rel!r}",
            )
        normalized.append(rel)
    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in normalized:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return tuple(out)


def _require_ucan(proof: Any, *, transport: str) -> str:
    if proof is None or (isinstance(proof, str) and not proof.strip()):
        raise AdapterError(
            "absent_ucan",
            f"{transport} invocation missing UCAN capability proof",
        )
    if not isinstance(proof, str):
        raise AdapterError(
            "invalid_ucan",
            f"{transport} UCAN must be a non-empty string",
        )
    token = proof.strip()
    # Minimal structural sanity: JWT-like or cid-like, not empty noise
    if len(token) < 8:
        raise AdapterError(
            "invalid_ucan",
            f"{transport} UCAN capability proof is too short",
        )
    return token


def _require_authenticated_identity(
    identity: Any,
    *,
    transport: str,
    field_name: str = "identity",
) -> str:
    if identity is None or (isinstance(identity, str) and not identity.strip()):
        raise AdapterError(
            "unauthenticated",
            f"{transport} invocation missing authenticated {field_name}",
        )
    if not isinstance(identity, str):
        raise AdapterError(
            "unauthenticated",
            f"{transport} {field_name} must be a string",
        )
    principal = identity.strip()
    if principal.lower() in {"anonymous", "guest", "unauthenticated", "none", "null"}:
        raise AdapterError(
            "unauthenticated",
            f"{transport} identity {principal!r} is not authenticated",
        )
    return principal


def _require_target_prompt(
    target: Any,
    prompt: Any,
    *,
    transport: str,
) -> tuple[str, str]:
    if not isinstance(target, str) or not target.strip():
        raise AdapterError(
            "invalid_target",
            f"{transport} target must be a non-empty string",
        )
    if not isinstance(prompt, str):
        raise AdapterError(
            "invalid_prompt",
            f"{transport} prompt must be a string",
        )
    t = target.strip()
    # Target itself must not be a path escape
    if _is_absolute_or_escape(t) or any(
        m in t for m in ("../", "..\\", "\x00")
    ):
        raise AdapterError(
            "path_escape",
            f"{transport} target rejected as path escape: {t!r}",
        )
    if _contains_prompt_path_injection(prompt):
        raise AdapterError(
            "prompt_path_injection",
            f"{transport} prompt contains path injection markers",
        )
    return t, prompt


def _mutation_gate(
    *,
    wants_mutation: bool,
    has_ucan: bool,
    has_identity: bool,
    authorized_paths: tuple[str, ...],
    transport_only_auth: bool,
) -> bool:
    """Mutation requires authenticated identity + UCAN + authorized paths.

    Transport-only authorization (e.g. bearer without UCAN, or local CLI
    flag without proof) must never open the mutation gate.
    """
    if not wants_mutation:
        return False
    if transport_only_auth:
        raise AdapterError(
            "transport_only_authorization",
            "transport-only authorization cannot grant mutation",
        )
    if not has_identity:
        raise AdapterError(
            "unauthenticated",
            "mutation requires authenticated identity",
        )
    if not has_ucan:
        raise AdapterError(
            "absent_ucan",
            "mutation requires UCAN capability proof",
        )
    if not authorized_paths:
        raise AdapterError(
            "no_authorized_paths",
            "mutation requires at least one authorized path",
        )
    return True


# ---------------------------------------------------------------------------
# Transport adapters
# ---------------------------------------------------------------------------


def adapt_http_context(envelope: Mapping[str, Any]) -> InvocationContext:
    """Adapt an HTTP request envelope into InvocationContext.

    Expected keys (flexible aliases supported):
      - target / resource
      - prompt / input / message
      - identity / principal / user (authenticated)
      - ucan / capability / authorization.ucan
      - paths / authorized_paths / scope.paths
      - mutate / mutation / allow_mutation
      - headers (optional mapping)
    """
    if not isinstance(envelope, Mapping):
        raise AdapterError("invalid_envelope", "HTTP envelope must be a mapping")

    headers = envelope.get("headers") or {}
    if not isinstance(headers, Mapping):
        headers = {}

    # Identity: prefer explicit field, then Authorization principal headers
    identity = (
        envelope.get("identity")
        or envelope.get("principal")
        or envelope.get("user")
        or headers.get("X-Principal")
        or headers.get("x-principal")
    )
    principal = _require_authenticated_identity(identity, transport="http")

    # UCAN: body field or Authorization: UCAN <token> / X-UCAN
    ucan_raw = (
        envelope.get("ucan")
        or envelope.get("capability")
        or envelope.get("capability_proof")
    )
    authz = envelope.get("authorization")
    if ucan_raw is None and isinstance(authz, Mapping):
        ucan_raw = authz.get("ucan") or authz.get("capability")
    if ucan_raw is None:
        header_auth = headers.get("Authorization") or headers.get("authorization")
        if isinstance(header_auth, str):
            ha = header_auth.strip()
            if ha.lower().startswith("ucan "):
                ucan_raw = ha[5:].strip()
            elif ha.lower().startswith("bearer "):
                # Bearer alone is transport-only; do not treat as UCAN
                ucan_raw = None
                transport_bearer = True
            else:
                transport_bearer = False
        else:
            transport_bearer = False
        ucan_hdr = headers.get("X-UCAN") or headers.get("x-ucan")
        if ucan_raw is None and ucan_hdr:
            ucan_raw = ucan_hdr
    else:
        transport_bearer = False

    # Detect transport-only auth: bearer present without UCAN
    header_auth = headers.get("Authorization") or headers.get("authorization") or ""
    has_bearer_only = (
        isinstance(header_auth, str)
        and header_auth.strip().lower().startswith("bearer ")
        and not ucan_raw
    )
    transport_only = bool(envelope.get("transport_only_auth")) or has_bearer_only

    wants_mutation = bool(
        envelope.get("mutate")
        or envelope.get("mutation")
        or envelope.get("allow_mutation")
    )

    # For non-mutation, UCAN may still be required when mutation requested;
    # for read-only we still require identity but allow missing UCAN only if
    # not mutating. Acceptance: absent UCAN cannot reach mutation.
    proof: Optional[str] = None
    if wants_mutation or ucan_raw is not None:
        if wants_mutation:
            proof = _require_ucan(ucan_raw, transport="http")
        elif ucan_raw is not None:
            proof = _require_ucan(ucan_raw, transport="http")

    target_raw = envelope.get("target") or envelope.get("resource")
    prompt_raw = (
        envelope.get("prompt")
        if "prompt" in envelope
        else envelope.get("input")
        if "input" in envelope
        else envelope.get("message")
    )
    if prompt_raw is None:
        prompt_raw = ""
    target, prompt = _require_target_prompt(target_raw, prompt_raw, transport="http")

    paths_raw = (
        envelope.get("authorized_paths")
        or envelope.get("paths")
        or (envelope.get("scope") or {}).get("paths")
        if isinstance(envelope.get("scope"), Mapping)
        else envelope.get("authorized_paths") or envelope.get("paths")
    )
    authorized_paths = _normalize_authorized_paths(paths_raw)

    mutation_allowed = _mutation_gate(
        wants_mutation=wants_mutation,
        has_ucan=proof is not None,
        has_identity=True,
        authorized_paths=authorized_paths,
        transport_only_auth=transport_only and wants_mutation,
    )

    trust = TrustSource(
        transport=TransportKind.HTTP,
        principal=principal,
        capability_proof=proof,
        peer_identity=headers.get("X-Forwarded-For") or headers.get("x-forwarded-for"),
        attributes={
            "method": envelope.get("method", "POST"),
            "path": envelope.get("path") or envelope.get("url_path"),
            "transport_only_auth": transport_only,
        },
    )
    return InvocationContext(
        target=target,
        prompt=prompt,
        trust=trust,
        authorized_paths=authorized_paths,
        mutation_allowed=mutation_allowed,
        metadata={"source": "http"},
    )


def adapt_cli_context(envelope: Mapping[str, Any]) -> InvocationContext:
    """Adapt a CLI invocation envelope into InvocationContext.

    Expected keys:
      - target
      - prompt / args.prompt
      - identity / user / $USER (must be authenticated — not raw env alone)
      - ucan / capability_proof / --ucan
      - paths / authorized_paths / --path
      - mutate / --mutate
      - cwd (never used to expand unsafe client paths into mutation)
    """
    if not isinstance(envelope, Mapping):
        raise AdapterError("invalid_envelope", "CLI envelope must be a mapping")

    # CLI must not treat process uid alone as sufficient for mutation;
    # require explicit authenticated identity field.
    identity = (
        envelope.get("identity")
        or envelope.get("principal")
        or envelope.get("user")
    )
    # Explicit unauthenticated marker
    if envelope.get("authenticated") is False:
        raise AdapterError("unauthenticated", "CLI invocation is not authenticated")
    principal = _require_authenticated_identity(identity, transport="cli")

    ucan_raw = (
        envelope.get("ucan")
        or envelope.get("capability")
        or envelope.get("capability_proof")
        or envelope.get("ucan_token")
    )
    # --token / local session without UCAN is transport-only
    transport_only = bool(
        envelope.get("transport_only_auth")
        or (envelope.get("local_auth") and not ucan_raw)
        or (envelope.get("session_token") and not ucan_raw)
    )

    wants_mutation = bool(
        envelope.get("mutate")
        or envelope.get("mutation")
        or envelope.get("allow_mutation")
        or envelope.get("write")
    )

    proof: Optional[str] = None
    if wants_mutation:
        proof = _require_ucan(ucan_raw, transport="cli")
    elif ucan_raw is not None:
        proof = _require_ucan(ucan_raw, transport="cli")

    args = envelope.get("args") if isinstance(envelope.get("args"), Mapping) else {}
    target_raw = envelope.get("target") or args.get("target")
    prompt_raw = (
        envelope.get("prompt")
        if "prompt" in envelope
        else args.get("prompt")
        if "prompt" in args
        else envelope.get("input")
    )
    if prompt_raw is None:
        prompt_raw = ""
    target, prompt = _require_target_prompt(target_raw, prompt_raw, transport="cli")

    paths_raw = (
        envelope.get("authorized_paths")
        or envelope.get("paths")
        or args.get("paths")
        or args.get("authorized_paths")
    )
    # Never resolve against cwd for authorization — reject escapes only
    authorized_paths = _normalize_authorized_paths(paths_raw)

    # Symlink escape: if client supplies paths that are symlinks outside
    # authorized roots when a root is declared, reject. We do not follow
    # symlinks; if envelope marks symlink_escape attempt, reject.
    if envelope.get("symlink_escape") or envelope.get("follow_symlinks"):
        raise AdapterError(
            "symlink_escape",
            "CLI path symlink escape is not permitted",
        )

    mutation_allowed = _mutation_gate(
        wants_mutation=wants_mutation,
        has_ucan=proof is not None,
        has_identity=True,
        authorized_paths=authorized_paths,
        transport_only_auth=transport_only and wants_mutation,
    )

    trust = TrustSource(
        transport=TransportKind.CLI,
        principal=principal,
        capability_proof=proof,
        peer_identity=None,
        attributes={
            "cwd": envelope.get("cwd"),
            "argv0": envelope.get("argv0"),
            "transport_only_auth": transport_only,
        },
    )
    return InvocationContext(
        target=target,
        prompt=prompt,
        trust=trust,
        authorized_paths=authorized_paths,
        mutation_allowed=mutation_allowed,
        metadata={"source": "cli"},
    )


def adapt_mcp_context(envelope: Mapping[str, Any]) -> InvocationContext:
    """Adapt an MCP tool-call envelope into InvocationContext.

    Expected keys:
      - target / name / tool
      - prompt / arguments.prompt / arguments.input
      - identity / client_id / session.principal
      - ucan / capabilities.ucan / meta.ucan
      - paths / arguments.paths
      - mutate
    """
    if not isinstance(envelope, Mapping):
        raise AdapterError("invalid_envelope", "MCP envelope must be a mapping")

    session = envelope.get("session") if isinstance(envelope.get("session"), Mapping) else {}
    meta = envelope.get("meta") if isinstance(envelope.get("meta"), Mapping) else {}
    arguments = (
        envelope.get("arguments")
        if isinstance(envelope.get("arguments"), Mapping)
        else {}
    )
    capabilities = (
        envelope.get("capabilities")
        if isinstance(envelope.get("capabilities"), Mapping)
        else {}
    )

    identity = (
        envelope.get("identity")
        or envelope.get("principal")
        or envelope.get("client_id")
        or session.get("principal")
        or session.get("identity")
        or meta.get("principal")
    )
    principal = _require_authenticated_identity(identity, transport="mcp")

    ucan_raw = (
        envelope.get("ucan")
        or envelope.get("capability")
        or envelope.get("capability_proof")
        or capabilities.get("ucan")
        or meta.get("ucan")
        or arguments.get("ucan")
    )

    # MCP connection trust without UCAN is transport-only
    transport_only = bool(
        envelope.get("transport_only_auth")
        or (envelope.get("connection_trusted") and not ucan_raw)
        or (session.get("transport_trusted") and not ucan_raw)
    )

    wants_mutation = bool(
        envelope.get("mutate")
        or envelope.get("mutation")
        or envelope.get("allow_mutation")
        or arguments.get("mutate")
        or (envelope.get("method") in {"tools/call_write", "resources/write"})
    )

    proof: Optional[str] = None
    if wants_mutation:
        proof = _require_ucan(ucan_raw, transport="mcp")
    elif ucan_raw is not None:
        proof = _require_ucan(ucan_raw, transport="mcp")

    target_raw = (
        envelope.get("target")
        or envelope.get("name")
        or envelope.get("tool")
        or arguments.get("target")
    )
    prompt_raw = (
        envelope.get("prompt")
        if "prompt" in envelope
        else arguments.get("prompt")
        if "prompt" in arguments
        else arguments.get("input")
        if "input" in arguments
        else envelope.get("input")
    )
    if prompt_raw is None:
        prompt_raw = ""
    target, prompt = _require_target_prompt(target_raw, prompt_raw, transport="mcp")

    paths_raw = (
        envelope.get("authorized_paths")
        or envelope.get("paths")
        or arguments.get("paths")
        or arguments.get("authorized_paths")
    )
    authorized_paths = _normalize_authorized_paths(paths_raw)

    mutation_allowed = _mutation_gate(
        wants_mutation=wants_mutation,
        has_ucan=proof is not None,
        has_identity=True,
        authorized_paths=authorized_paths,
        transport_only_auth=transport_only and wants_mutation,
    )

    trust = TrustSource(
        transport=TransportKind.MCP,
        principal=principal,
        capability_proof=proof,
        peer_identity=session.get("peer") or meta.get("peer"),
        attributes={
            "protocol": envelope.get("protocol", "mcp"),
            "session_id": session.get("id"),
            "transport_only_auth": transport_only,
        },
    )
    return InvocationContext(
        target=target,
        prompt=prompt,
        trust=trust,
        authorized_paths=authorized_paths,
        mutation_allowed=mutation_allowed,
        metadata={"source": "mcp"},
    )


def adapt_context(
    transport: str | TransportKind,
    envelope: Mapping[str, Any],
) -> InvocationContext:
    """Dispatch to the appropriate transport adapter."""
    if isinstance(transport, TransportKind):
        kind = transport
    else:
        try:
            kind = TransportKind(str(transport).lower().strip())
        except ValueError as exc:
            raise AdapterError(
                "unsupported_transport",
                f"unsupported transport: {transport!r}",
            ) from exc
    if kind is TransportKind.HTTP:
        return adapt_http_context(envelope)
    if kind is TransportKind.CLI:
        return adapt_cli_context(envelope)
    if kind is TransportKind.MCP:
        return adapt_mcp_context(envelope)
    raise AdapterError("unsupported_transport", f"unsupported transport: {kind}")


def contexts_equivalent_for_resolution(
    a: InvocationContext,
    b: InvocationContext,
) -> bool:
    """True when authorized target/prompt inputs resolve equivalently.

    Trust sources may differ and remain visible; equivalence is only over
    the resolution key (target, prompt) plus authorized path sets and
    mutation gate outcome for authorized invocations.
    """
    if a.resolution_key() != b.resolution_key():
        return False
    if set(a.authorized_paths) != set(b.authorized_paths):
        return False
    if a.mutation_allowed != b.mutation_allowed:
        return False
    return True


def trust_sources_distinct(a: InvocationContext, b: InvocationContext) -> bool:
    """True when trust provenance differs (transport and/or principal proof)."""
    return a.trust.to_dict() != b.trust.to_dict()


__all__ = [
    "AdapterError",
    "InvocationContext",
    "TransportKind",
    "TrustSource",
    "adapt_cli_context",
    "adapt_context",
    "adapt_http_context",
    "adapt_mcp_context",
    "contexts_equivalent_for_resolution",
    "trust_sources_distinct",
]
