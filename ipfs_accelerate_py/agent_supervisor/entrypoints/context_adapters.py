"""Transport-specific trusted invocation-context adapters for Agent Supervisor.

Adapters normalize HTTP, CLI, and MCP invocation envelopes into a single
TrustedInvocationContext shape for AuthorityResolver.resolve_mutation_authority.
Client-supplied paths, identities, and authorization claims are never trusted
as authority; only transport-attested material is used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence, Tuple

from ipfs_accelerate_py.agent_supervisor.entrypoints.authority_resolver import (
    AuthorityDecision,
    AuthorityResolver,
    MutationAuthorityRequest,
    TrustSource,
)


class TransportKind(str, Enum):
    """Supported trusted transport surfaces."""

    HTTP = "http"
    CLI = "cli"
    MCP = "mcp"


class AdapterRejectReason(str, Enum):
    """Machine-stable reject reasons for untrusted or incomplete envelopes."""

    UNAUTHENTICATED_IDENTITY = "unauthenticated_identity"
    ABSENT_UCAN = "absent_ucan"
    TRANSPORT_ONLY_AUTHORIZATION = "transport_only_authorization"
    ARBITRARY_CLIENT_PATH = "arbitrary_client_path"
    PROMPT_PATH_INJECTION = "prompt_path_injection"
    SYMLINK_ESCAPE = "symlink_escape"
    MISSING_TARGET = "missing_target"
    INVALID_ENVELOPE = "invalid_envelope"
    PATH_OUTSIDE_REPO = "path_outside_repo"


@dataclass(frozen=True)
class TrustedInvocationContext:
    """Normalized trusted context produced by a transport adapter.

    Trust sources remain distinct so downstream resolution and audits can see
    which surface attested which claims. Client-claimed paths/identity are
    never copied into trusted fields.
    """

    transport: TransportKind
    principal_id: str
    capability_token: str
    target_path: str
    prompt: str
    repository_root: str
    trust_sources: Tuple[TrustSource, ...]
    attested_identity: bool
    attested_ucan: bool
    allowed_write_paths: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterResult:
    """Outcome of adapting a transport envelope."""

    accepted: bool
    context: Optional[TrustedInvocationContext] = None
    reject_reason: Optional[AdapterRejectReason] = None
    detail: str = ""

    @classmethod
    def accept(cls, context: TrustedInvocationContext) -> "AdapterResult":
        return cls(accepted=True, context=context)

    @classmethod
    def reject(
        cls,
        reason: AdapterRejectReason,
        detail: str = "",
    ) -> "AdapterResult":
        return cls(accepted=False, reject_reason=reason, detail=detail)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _nonempty_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text if text else None


def _normalize_repo_relative(path: str, repository_root: str) -> Optional[str]:
    """Normalize a path relative to repository_root; reject escapes."""
    if not path or not isinstance(path, str):
        return None
    raw = path.strip()
    if not raw:
        return None
    # Reject absolute paths and drive-style paths from clients.
    if os.path.isabs(raw) or (len(raw) >= 2 and raw[1] == ":"):
        return None
    # Reject null bytes and control injection.
    if "\x00" in raw or any(ord(c) < 32 and c not in "\t" for c in raw):
        return None
    posix = PurePosixPath(raw.replace("\\", "/"))
    if any(part == ".." for part in posix.parts):
        return None
    if posix.is_absolute():
        return None
    normalized = str(posix).lstrip("/")
    if not normalized or normalized == ".":
        return None
    # Physical containment check against repository root when available.
    root = Path(repository_root).resolve()
    candidate = (root / normalized).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return normalized


def _detect_prompt_path_injection(prompt: str, claimed_paths: Sequence[str]) -> bool:
    """Detect prompt text that tries to smuggle path write authority."""
    if not prompt:
        return False
    lower = prompt.lower()
    injection_markers = (
        "write to /",
        "write_path=",
        "target_path=",
        "../",
        "..\\",
        "file://",
        "path injection",
        "ignore previous",
        "authorized path:",
        "mutate path",
    )
    if any(marker in lower for marker in injection_markers):
        return True
    for claimed in claimed_paths:
        if not claimed:
            continue
        # Explicit path claim embedded as authority instruction.
        if f"authorize {claimed}".lower() in lower:
            return True
        if f"write {claimed}".lower() in lower and "path" in lower:
            return True
    return False


def _symlink_escape(
    repository_root: str,
    relative_path: str,
    *,
    follow_symlinks_for_check: bool = True,
) -> bool:
    """Return True if resolving relative_path escapes repository_root via symlink."""
    root = Path(repository_root)
    if not root.exists():
        return False
    root_resolved = root.resolve()
    candidate = root / relative_path
    # If any path component is a symlink pointing outside root, reject.
    parts = PurePosixPath(relative_path).parts
    accum = root_resolved
    for part in parts:
        accum = accum / part
        if accum.is_symlink():
            try:
                link_target = accum.resolve()
            except (OSError, RuntimeError):
                return True
            try:
                link_target.relative_to(root_resolved)
            except ValueError:
                return True
        if not accum.exists():
            # Non-existent leaf is fine if parents stayed inside root.
            break
    if follow_symlinks_for_check and candidate.exists():
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root_resolved)
        except (ValueError, OSError, RuntimeError):
            return True
    return False


def _client_path_claims(envelope: Mapping[str, Any]) -> Tuple[str, ...]:
    """Collect client-supplied path claims that must not become authority."""
    claims: list[str] = []
    for key in (
        "path",
        "target_path",
        "write_path",
        "write_paths",
        "client_path",
        "client_paths",
        "requested_path",
    ):
        val = envelope.get(key)
        if isinstance(val, str) and val.strip():
            claims.append(val.strip())
        elif isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, str) and item.strip():
                    claims.append(item.strip())
    client = _as_mapping(envelope.get("client"))
    for key in ("path", "paths", "target_path", "write_paths"):
        val = client.get(key)
        if isinstance(val, str) and val.strip():
            claims.append(val.strip())
        elif isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, str) and item.strip():
                    claims.append(item.strip())
    body = _as_mapping(envelope.get("body"))
    for key in ("path", "target_path", "write_paths"):
        val = body.get(key)
        if isinstance(val, str) and val.strip():
            claims.append(val.strip())
        elif isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, str) and item.strip():
                    claims.append(item.strip())
    return tuple(claims)


def _resolve_attested_target(
    *,
    repository_root: str,
    transport_attested_path: Optional[str],
    allowed_write_paths: Sequence[str],
    client_claims: Sequence[str],
    prompt: str,
) -> Tuple[Optional[str], Optional[AdapterRejectReason], str]:
    """Pick a trusted target path; never elevate client claims."""
    if _detect_prompt_path_injection(prompt, client_claims):
        return None, AdapterRejectReason.PROMPT_PATH_INJECTION, "prompt path injection"

    # Client-only path with no transport attestation cannot authorize mutation.
    if not transport_attested_path:
        if client_claims:
            return (
                None,
                AdapterRejectReason.ARBITRARY_CLIENT_PATH,
                "client path without transport attestation",
            )
        return None, AdapterRejectReason.MISSING_TARGET, "no attested target path"

    normalized = _normalize_repo_relative(transport_attested_path, repository_root)
    if normalized is None:
        return (
            None,
            AdapterRejectReason.PATH_OUTSIDE_REPO,
            f"attested path escapes repository: {transport_attested_path!r}",
        )

    if _symlink_escape(repository_root, normalized):
        return (
            None,
            AdapterRejectReason.SYMLINK_ESCAPE,
            f"symlink escape for path: {normalized}",
        )

    # If client claims a different path than attested, reject elevation attempt.
    for claim in client_claims:
        claim_norm = _normalize_repo_relative(claim, repository_root)
        if claim_norm is not None and claim_norm != normalized:
            return (
                None,
                AdapterRejectReason.ARBITRARY_CLIENT_PATH,
                "client path diverges from transport-attested path",
            )
        if claim_norm is None and claim.strip():
            # Malicious absolute/escape claim
            if os.path.isabs(claim) or ".." in claim.replace("\\", "/").split("/"):
                return (
                    None,
                    AdapterRejectReason.ARBITRARY_CLIENT_PATH,
                    "arbitrary client path claim",
                )

    if allowed_write_paths:
        allowed_norm = []
        for p in allowed_write_paths:
            n = _normalize_repo_relative(p, repository_root)
            if n is not None:
                allowed_norm.append(n)
        if allowed_norm and normalized not in allowed_norm:
            # Prefix allow: attested path may be under an allowed directory.
            if not any(
                normalized == a or normalized.startswith(a.rstrip("/") + "/")
                for a in allowed_norm
            ):
                return (
                    None,
                    AdapterRejectReason.ARBITRARY_CLIENT_PATH,
                    "attested path not in transport allow-list",
                )

    return normalized, None, ""


def _require_identity_and_ucan(
    *,
    principal_id: Optional[str],
    capability_token: Optional[str],
    identity_attested: bool,
    ucan_attested: bool,
    transport_auth_only: bool,
) -> Optional[AdapterResult]:
    if transport_auth_only:
        return AdapterResult.reject(
            AdapterRejectReason.TRANSPORT_ONLY_AUTHORIZATION,
            "transport authentication alone cannot authorize mutation",
        )
    if not identity_attested or not principal_id:
        return AdapterResult.reject(
            AdapterRejectReason.UNAUTHENTICATED_IDENTITY,
            "identity not transport-attested",
        )
    if not ucan_attested or not capability_token:
        return AdapterResult.reject(
            AdapterRejectReason.ABSENT_UCAN,
            "UCAN capability token absent or not attested",
        )
    return None


class HttpContextAdapter:
    """Adapt an HTTP request envelope into TrustedInvocationContext.

    Trusted material:
    - principal from verified auth middleware (e.g. request.state.principal)
    - UCAN from verified capability header after gateway validation
    - target path from server-bound route/scope, not raw JSON body path alone
    """

    transport = TransportKind.HTTP

    def adapt(
        self,
        envelope: Mapping[str, Any],
        *,
        repository_root: str,
    ) -> AdapterResult:
        if not isinstance(envelope, Mapping):
            return AdapterResult.reject(
                AdapterRejectReason.INVALID_ENVELOPE, "http envelope must be a mapping"
            )

        headers = _as_mapping(envelope.get("headers"))
        state = _as_mapping(envelope.get("state"))
        body = _as_mapping(envelope.get("body"))
        scope = _as_mapping(envelope.get("scope"))

        # Identity: only middleware-attested principal is trusted.
        principal = _nonempty_str(state.get("principal_id")) or _nonempty_str(
            state.get("principal")
        )
        identity_attested = bool(state.get("identity_verified")) or bool(
            state.get("authenticated")
        )
        # Header-only identity without state attestation is untrusted.
        header_principal = _nonempty_str(headers.get("x-principal-id")) or _nonempty_str(
            headers.get("X-Principal-Id")
        )
        if not identity_attested and header_principal:
            return AdapterResult.reject(
                AdapterRejectReason.UNAUTHENTICATED_IDENTITY,
                "principal header without verified auth state",
            )
        if not identity_attested and body.get("principal_id"):
            return AdapterResult.reject(
                AdapterRejectReason.UNAUTHENTICATED_IDENTITY,
                "body principal_id is not a trust source",
            )

        # UCAN: must be marked verified by gateway; raw header alone is insufficient.
        raw_ucan = _nonempty_str(headers.get("authorization")) or _nonempty_str(
            headers.get("Authorization")
        )
        if raw_ucan and raw_ucan.lower().startswith("bearer "):
            raw_ucan = raw_ucan[7:].strip()
        verified_ucan = _nonempty_str(state.get("verified_ucan")) or _nonempty_str(
            state.get("capability_token")
        )
        ucan_attested = bool(state.get("ucan_verified")) and bool(verified_ucan)
        if raw_ucan and not ucan_attested:
            # Present but unverified token → absent trusted UCAN.
            return AdapterResult.reject(
                AdapterRejectReason.ABSENT_UCAN,
                "UCAN header present but not gateway-verified",
            )

        # Transport-only auth: session/API key without capability proof.
        transport_auth_only = bool(state.get("transport_auth_only")) or (
            bool(state.get("api_key_authenticated") or state.get("session_authenticated"))
            and not ucan_attested
        )

        rejected = _require_identity_and_ucan(
            principal_id=principal,
            capability_token=verified_ucan,
            identity_attested=identity_attested,
            ucan_attested=ucan_attested,
            transport_auth_only=transport_auth_only,
        )
        if rejected is not None:
            return rejected

        prompt = _nonempty_str(body.get("prompt")) or _nonempty_str(envelope.get("prompt")) or ""

        # Target: server scope / route binding, not sole client body path.
        attested_path = (
            _nonempty_str(scope.get("target_path"))
            or _nonempty_str(scope.get("effect_path"))
            or _nonempty_str(state.get("bound_target_path"))
            or _nonempty_str(envelope.get("route_target_path"))
        )
        allowed = scope.get("allowed_write_paths") or state.get("allowed_write_paths") or ()
        if isinstance(allowed, str):
            allowed = (allowed,)
        allowed_t = tuple(p for p in allowed if isinstance(p, str))

        client_claims = _client_path_claims(envelope)
        # Body path alone is a client claim unless it matches server-bound scope.
        body_path = _nonempty_str(body.get("path")) or _nonempty_str(body.get("target_path"))
        if body_path and body_path not in client_claims:
            client_claims = client_claims + (body_path,)

        target, reason, detail = _resolve_attested_target(
            repository_root=repository_root,
            transport_attested_path=attested_path,
            allowed_write_paths=allowed_t,
            client_claims=client_claims,
            prompt=prompt,
        )
        if reason is not None:
            return AdapterResult.reject(reason, detail)

        trust_sources: Tuple[TrustSource, ...] = (
            TrustSource.UCAN_CAPABILITY,
            TrustSource.TRANSPORT_ATTESTATION,
        )

        ctx = TrustedInvocationContext(
            transport=self.transport,
            principal_id=principal or "",
            capability_token=verified_ucan or "",
            target_path=target or "",
            prompt=prompt,
            repository_root=str(Path(repository_root).resolve()),
            trust_sources=trust_sources,
            attested_identity=True,
            attested_ucan=True,
            allowed_write_paths=allowed_t,
            metadata={
                "adapter": "http",
                "request_id": state.get("request_id") or headers.get("x-request-id"),
            },
        )
        return AdapterResult.accept(ctx)


class CliContextAdapter:
    """Adapt a CLI invocation envelope into TrustedInvocationContext.

    Trusted material comes from the process supervisor / local agent host,
    not from argv flags that impersonate identity or expand write scope.
    """

    transport = TransportKind.CLI

    def adapt(
        self,
        envelope: Mapping[str, Any],
        *,
        repository_root: str,
    ) -> AdapterResult:
        if not isinstance(envelope, Mapping):
            return AdapterResult.reject(
                AdapterRejectReason.INVALID_ENVELOPE, "cli envelope must be a mapping"
            )

        host = _as_mapping(envelope.get("host"))
        args = _as_mapping(envelope.get("args"))
        flags = _as_mapping(envelope.get("flags"))

        principal = _nonempty_str(host.get("principal_id")) or _nonempty_str(
            host.get("agent_id")
        )
        identity_attested = bool(host.get("identity_verified")) or bool(
            host.get("local_agent_bound")
        )
        # CLI flag spoofing identity is rejected.
        if not identity_attested and (
            flags.get("as_principal") or args.get("principal_id") or flags.get("user")
        ):
            return AdapterResult.reject(
                AdapterRejectReason.UNAUTHENTICATED_IDENTITY,
                "CLI identity flags are not trust sources",
            )

        verified_ucan = _nonempty_str(host.get("verified_ucan")) or _nonempty_str(
            host.get("capability_token")
        )
        ucan_attested = bool(host.get("ucan_verified")) and bool(verified_ucan)
        if (flags.get("ucan") or args.get("ucan")) and not ucan_attested:
            return AdapterResult.reject(
                AdapterRejectReason.ABSENT_UCAN,
                "CLI --ucan flag without host verification",
            )

        transport_auth_only = bool(host.get("transport_auth_only")) or (
            bool(host.get("local_process_trusted")) and not ucan_attested
        )

        rejected = _require_identity_and_ucan(
            principal_id=principal,
            capability_token=verified_ucan,
            identity_attested=identity_attested,
            ucan_attested=ucan_attested,
            transport_auth_only=transport_auth_only,
        )
        if rejected is not None:
            return rejected

        prompt = (
            _nonempty_str(args.get("prompt"))
            or _nonempty_str(flags.get("prompt"))
            or _nonempty_str(envelope.get("prompt"))
            or ""
        )

        attested_path = (
            _nonempty_str(host.get("bound_target_path"))
            or _nonempty_str(host.get("effect_path"))
            or _nonempty_str(envelope.get("workspace_target_path"))
        )
        allowed = host.get("allowed_write_paths") or ()
        if isinstance(allowed, str):
            allowed = (allowed,)
        allowed_t = tuple(p for p in allowed if isinstance(p, str))

        client_claims = list(_client_path_claims(envelope))
        for key in ("path", "target_path", "write_path", "file"):
            for src in (args, flags):
                v = _nonempty_str(src.get(key))
                if v:
                    client_claims.append(v)

        target, reason, detail = _resolve_attested_target(
            repository_root=repository_root,
            transport_attested_path=attested_path,
            allowed_write_paths=allowed_t,
            client_claims=tuple(client_claims),
            prompt=prompt,
        )
        if reason is not None:
            return AdapterResult.reject(reason, detail)

        trust_sources: Tuple[TrustSource, ...] = (
            TrustSource.UCAN_CAPABILITY,
            TrustSource.TRANSPORT_ATTESTATION,
            TrustSource.LOCAL_HOST_BINDING,
        )

        ctx = TrustedInvocationContext(
            transport=self.transport,
            principal_id=principal or "",
            capability_token=verified_ucan or "",
            target_path=target or "",
            prompt=prompt,
            repository_root=str(Path(repository_root).resolve()),
            trust_sources=trust_sources,
            attested_identity=True,
            attested_ucan=True,
            allowed_write_paths=allowed_t,
            metadata={"adapter": "cli", "pid": host.get("pid")},
        )
        return AdapterResult.accept(ctx)


class McpContextAdapter:
    """Adapt an MCP tool-call envelope into TrustedInvocationContext.

    Trusted material is bound by the MCP session/server attestation layer.
    Tool arguments alone cannot supply identity, UCAN, or write paths.
    """

    transport = TransportKind.MCP

    def adapt(
        self,
        envelope: Mapping[str, Any],
        *,
        repository_root: str,
    ) -> AdapterResult:
        if not isinstance(envelope, Mapping):
            return AdapterResult.reject(
                AdapterRejectReason.INVALID_ENVELOPE, "mcp envelope must be a mapping"
            )

        session = _as_mapping(envelope.get("session"))
        server = _as_mapping(envelope.get("server"))
        arguments = _as_mapping(envelope.get("arguments"))
        params = _as_mapping(envelope.get("params"))
        args = arguments or params

        principal = _nonempty_str(session.get("principal_id")) or _nonempty_str(
            server.get("principal_id")
        )
        identity_attested = bool(session.get("identity_verified")) or bool(
            server.get("session_bound")
        )
        if not identity_attested and (
            args.get("principal_id") or args.get("as_user") or envelope.get("principal_id")
        ):
            return AdapterResult.reject(
                AdapterRejectReason.UNAUTHENTICATED_IDENTITY,
                "MCP tool argument principal is not a trust source",
            )

        verified_ucan = _nonempty_str(session.get("verified_ucan")) or _nonempty_str(
            server.get("capability_token")
        )
        ucan_attested = bool(session.get("ucan_verified") or server.get("ucan_verified")) and bool(
            verified_ucan
        )
        if args.get("ucan") and not ucan_attested:
            return AdapterResult.reject(
                AdapterRejectReason.ABSENT_UCAN,
                "MCP argument UCAN without session verification",
            )

        transport_auth_only = bool(session.get("transport_auth_only")) or (
            bool(session.get("mcp_connected") or server.get("transport_connected"))
            and not ucan_attested
        )

        rejected = _require_identity_and_ucan(
            principal_id=principal,
            capability_token=verified_ucan,
            identity_attested=identity_attested,
            ucan_attested=ucan_attested,
            transport_auth_only=transport_auth_only,
        )
        if rejected is not None:
            return rejected

        prompt = (
            _nonempty_str(args.get("prompt"))
            or _nonempty_str(envelope.get("prompt"))
            or ""
        )

        attested_path = (
            _nonempty_str(session.get("bound_target_path"))
            or _nonempty_str(server.get("effect_path"))
            or _nonempty_str(session.get("effect_path"))
        )
        allowed = session.get("allowed_write_paths") or server.get("allowed_write_paths") or ()
        if isinstance(allowed, str):
            allowed = (allowed,)
        allowed_t = tuple(p for p in allowed if isinstance(p, str))

        client_claims = list(_client_path_claims(envelope))
        for key in ("path", "target_path", "write_path", "file", "uri"):
            v = _nonempty_str(args.get(key))
            if v:
                client_claims.append(v)

        target, reason, detail = _resolve_attested_target(
            repository_root=repository_root,
            transport_attested_path=attested_path,
            allowed_write_paths=allowed_t,
            client_claims=tuple(client_claims),
            prompt=prompt,
        )
        if reason is not None:
            return AdapterResult.reject(reason, detail)

        trust_sources: Tuple[TrustSource, ...] = (
            TrustSource.UCAN_CAPABILITY,
            TrustSource.TRANSPORT_ATTESTATION,
            TrustSource.MCP_SESSION_BINDING,
        )

        ctx = TrustedInvocationContext(
            transport=self.transport,
            principal_id=principal or "",
            capability_token=verified_ucan or "",
            target_path=target or "",
            prompt=prompt,
            repository_root=str(Path(repository_root).resolve()),
            trust_sources=trust_sources,
            attested_identity=True,
            attested_ucan=True,
            allowed_write_paths=allowed_t,
            metadata={
                "adapter": "mcp",
                "tool": envelope.get("tool") or envelope.get("name"),
                "session_id": session.get("session_id"),
            },
        )
        return AdapterResult.accept(ctx)


class ContextAdapterRegistry:
    """Dispatch envelopes to the correct transport adapter."""

    def __init__(self) -> None:
        self._adapters = {
            TransportKind.HTTP: HttpContextAdapter(),
            TransportKind.CLI: CliContextAdapter(),
            TransportKind.MCP: McpContextAdapter(),
        }

    def adapt(
        self,
        transport: TransportKind | str,
        envelope: Mapping[str, Any],
        *,
        repository_root: str,
    ) -> AdapterResult:
        if isinstance(transport, str):
            try:
                transport = TransportKind(transport.lower())
            except ValueError:
                return AdapterResult.reject(
                    AdapterRejectReason.INVALID_ENVELOPE,
                    f"unknown transport: {transport!r}",
                )
        adapter = self._adapters.get(transport)
        if adapter is None:
            return AdapterResult.reject(
                AdapterRejectReason.INVALID_ENVELOPE,
                f"no adapter for transport: {transport}",
            )
        return adapter.adapt(envelope, repository_root=repository_root)

    def to_mutation_request(
        self,
        context: TrustedInvocationContext,
    ) -> MutationAuthorityRequest:
        """Map trusted context into AuthorityResolver input."""
        return MutationAuthorityRequest(
            principal_id=context.principal_id,
            capability_token=context.capability_token,
            target_path=context.target_path,
            prompt=context.prompt,
            repository_root=context.repository_root,
            trust_sources=context.trust_sources,
            allowed_write_paths=context.allowed_write_paths,
            transport=context.transport.value,
            metadata=dict(context.metadata),
        )

    def resolve_via(
        self,
        resolver: AuthorityResolver,
        transport: TransportKind | str,
        envelope: Mapping[str, Any],
        *,
        repository_root: str,
    ) -> Tuple[AdapterResult, Optional[AuthorityDecision]]:
        """Adapt then resolve; rejects never reach the resolver as authorized."""
        result = self.adapt(transport, envelope, repository_root=repository_root)
        if not result.accepted or result.context is None:
            return result, None
        decision = resolver.resolve_mutation_authority(
            self.to_mutation_request(result.context)
        )
        return result, decision


def adapt_http(envelope: Mapping[str, Any], *, repository_root: str) -> AdapterResult:
    return HttpContextAdapter().adapt(envelope, repository_root=repository_root)


def adapt_cli(envelope: Mapping[str, Any], *, repository_root: str) -> AdapterResult:
    return CliContextAdapter().adapt(envelope, repository_root=repository_root)


def adapt_mcp(envelope: Mapping[str, Any], *, repository_root: str) -> AdapterResult:
    return McpContextAdapter().adapt(envelope, repository_root=repository_root)


__all__ = [
    "AdapterRejectReason",
    "AdapterResult",
    "CliContextAdapter",
    "ContextAdapterRegistry",
    "HttpContextAdapter",
    "McpContextAdapter",
    "TransportKind",
    "TrustedInvocationContext",
    "adapt_cli",
    "adapt_http",
    "adapt_mcp",
]
