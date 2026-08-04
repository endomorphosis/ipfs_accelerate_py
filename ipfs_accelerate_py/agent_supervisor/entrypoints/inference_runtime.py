"""Ambient inference runtime: collect trusted ambient evidence and orchestrate prompt-only resolution.

This entrypoint gathers trusted ambient evidence from the local environment
(CWD, installed signed profiles, authenticated server context) and resolves
inference targets without requiring low-level target/profile flags when ambient
evidence is sufficient. Prompt text is never allowed to populate security-sensitive
fields (allowlist, caller, policy, provider, validation argv, or authority).
Material ambiguity never launches; unchanged evidence yields an identical receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence

# Fields that prompt text must never populate.
PROMPT_FORBIDDEN_FIELDS = frozenset({
    "allowlist",
    "caller",
    "policy",
    "provider",
    "validation_argv",
    "authority",
})


class AmbientInferenceError(Exception):
    """Base error for ambient inference runtime failures."""


class MaterialAmbiguityError(AmbientInferenceError):
    """Raised when material ambiguity would prevent a safe launch."""


class PromptContaminationError(AmbientInferenceError):
    """Raised when prompt text attempts to populate forbidden fields."""


@dataclass(frozen=True)
class AmbientEvidence:
    """Trusted ambient evidence collected from the local environment."""

    cwd: str
    profile_path: Optional[str] = None
    profile_signed: bool = False
    server_authenticated: bool = False
    server_context: Optional[Mapping[str, Any]] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Stable fingerprint of this evidence for receipt identity."""
        payload = {
            "cwd": self.cwd,
            "profile_path": self.profile_path,
            "profile_signed": self.profile_signed,
            "server_authenticated": self.server_authenticated,
            "server_context": dict(self.server_context) if self.server_context else None,
            "extra": dict(self.extra) if self.extra else {},
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def is_sufficient_for_prompt_only(self) -> bool:
        """True when ambient evidence alone supports prompt-only resolution.

        Local CWD plus an installed signed profile OR authenticated server
        context is sufficient; no low-level target/profile flags are required.
        """
        has_cwd = bool(self.cwd)
        has_signed_profile = bool(self.profile_path) and self.profile_signed
        has_auth_server = self.server_authenticated and self.server_context is not None
        return has_cwd and (has_signed_profile or has_auth_server)


@dataclass(frozen=True)
class ResolutionReceipt:
    """Deterministic receipt for an ambient resolution attempt."""

    evidence_fingerprint: str
    resolved: bool
    launch_authorized: bool
    target: Optional[str] = None
    profile: Optional[str] = None
    reason: Optional[str] = None
    prompt_hash: Optional[str] = None
    policy: Optional[Mapping[str, Any]] = None
    provider: Optional[str] = None
    caller: Optional[str] = None
    allowlist: Optional[Sequence[str]] = None
    authority: Optional[Mapping[str, Any]] = None
    validation_argv: Optional[Sequence[str]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_fingerprint": self.evidence_fingerprint,
            "resolved": self.resolved,
            "launch_authorized": self.launch_authorized,
            "target": self.target,
            "profile": self.profile,
            "reason": self.reason,
            "prompt_hash": self.prompt_hash,
            "policy": dict(self.policy) if self.policy else None,
            "provider": self.provider,
            "caller": self.caller,
            "allowlist": list(self.allowlist) if self.allowlist else None,
            "authority": dict(self.authority) if self.authority else None,
            "validation_argv": list(self.validation_argv) if self.validation_argv else None,
        }

    def identity(self) -> str:
        """Stable identity of this receipt; identical for unchanged evidence."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def collect_ambient_evidence(
    *,
    cwd: Optional[str] = None,
    profile_path: Optional[str] = None,
    profile_signed: Optional[bool] = None,
    server_context: Optional[Mapping[str, Any]] = None,
    server_authenticated: Optional[bool] = None,
    profile_search_paths: Optional[Sequence[str]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> AmbientEvidence:
    """Collect trusted ambient evidence from the local environment.

    Prefers explicit trusted inputs when provided; otherwise inspects CWD and
    well-known profile locations. Does not trust prompt text.
    """
    resolved_cwd = os.path.abspath(cwd if cwd is not None else os.getcwd())

    resolved_profile: Optional[str] = None
    resolved_signed = False

    if profile_path is not None:
        resolved_profile = os.path.abspath(profile_path)
        if profile_signed is not None:
            resolved_signed = bool(profile_signed)
        else:
            resolved_signed = _looks_signed(resolved_profile)
    else:
        search = list(profile_search_paths) if profile_search_paths else _default_profile_search_paths(resolved_cwd)
        for candidate in search:
            path = Path(candidate)
            if path.is_file():
                resolved_profile = str(path.resolve())
                resolved_signed = _looks_signed(resolved_profile) if profile_signed is None else bool(profile_signed)
                break
        if profile_signed is not None and resolved_profile is not None:
            resolved_signed = bool(profile_signed)

    resolved_server = dict(server_context) if server_context else None
    if server_authenticated is not None:
        resolved_auth = bool(server_authenticated)
    else:
        resolved_auth = bool(resolved_server and resolved_server.get("authenticated"))

    return AmbientEvidence(
        cwd=resolved_cwd,
        profile_path=resolved_profile,
        profile_signed=resolved_signed,
        server_authenticated=resolved_auth,
        server_context=resolved_server,
        extra=dict(extra) if extra else {},
    )


def _default_profile_search_paths(cwd: str) -> list[str]:
    home = Path.home()
    return [
        str(Path(cwd) / ".agent-supervisor" / "profile.signed.json"),
        str(Path(cwd) / "profile.signed.json"),
        str(home / ".agent-supervisor" / "profile.signed.json"),
        str(home / ".config" / "agent-supervisor" / "profile.signed.json"),
    ]


def _looks_signed(path: str) -> bool:
    """Heuristic: signed profiles carry a signature marker in content or name."""
    p = Path(path)
    if not p.is_file():
        return False
    name = p.name.lower()
    if "signed" in name or name.endswith(".sig") or name.endswith(".signed.json"):
        return True
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
        data = json.loads(text)
        if isinstance(data, dict) and (data.get("signature") or data.get("signed") or data.get("sig")):
            return True
    except (OSError, json.JSONDecodeError, UnicodeError):
        pass
    return False


def sanitize_prompt_bindings(
    prompt: str,
    bindings: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Ensure prompt text cannot populate security-sensitive fields.

    Returns a cleaned bindings dict. Raises PromptContaminationError if the
    prompt body or untrusted bindings attempt to set forbidden fields.
    """
    cleaned: dict[str, Any] = {}
    if bindings:
        for key, value in bindings.items():
            if key in PROMPT_FORBIDDEN_FIELDS:
                raise PromptContaminationError(
                    f"prompt bindings must not populate forbidden field: {key}"
                )
            cleaned[key] = value

    # Detect structured attempts inside prompt text (e.g. JSON/key=value).
    lowered = prompt.lower()
    for field_name in PROMPT_FORBIDDEN_FIELDS:
        # Common injection patterns: "allowlist:", '"allowlist":', "allowlist="
        markers = (
            f"{field_name}:",
            f'"{field_name}":',
            f"'{field_name}':",
            f"{field_name}=",
            f"{field_name} ",
        )
        for marker in markers:
            if marker in lowered:
                # Only treat as contamination when it looks like an assignment,
                # not mere discussion of the word in free text without structure.
                # Require JSON-ish or key=value style near the marker.
                idx = lowered.find(marker)
                snippet = prompt[idx : idx + len(marker) + 80]
                if _looks_like_field_assignment(snippet, field_name):
                    raise PromptContaminationError(
                        f"prompt text must not populate forbidden field: {field_name}"
                    )
    return cleaned


def _looks_like_field_assignment(snippet: str, field_name: str) -> bool:
    s = snippet.strip()
    fl = field_name.lower()
    sl = s.lower()
    if sl.startswith(f'"{fl}":') or sl.startswith(f"'{fl}':"):
        return True
    if sl.startswith(f"{fl}=") or sl.startswith(f"{fl}:"):
        rest = s[len(field_name) + 1 :].lstrip()
        if not rest:
            return False
        # Reject if it's prose ("allowlist is important") vs assignment.
        if rest[0] in '"\'[{' or rest[:4].lower() in ("true", "fals", "null") or rest[0].isdigit():
            return True
        if rest.startswith("[") or rest.startswith("{"):
            return True
        # key=value with non-space token
        token = [REDACTED] if rest.split() else ""
        if token and not token.endswith((".", ",", "!", "?")):
            # Heuristic: bare word after colon often prose; require JSON/list-like
            if "=" in s[: len(field_name) + 1 + len(token)]:
                return True
    return False


def resolve_prompt_only(
    prompt: str,
    evidence: AmbientEvidence,
    *,
    trusted_bindings: Optional[Mapping[str, Any]] = None,
    prompt_bindings: Optional[Mapping[str, Any]] = None,
    target: Optional[str] = None,
    profile: Optional[str] = None,
    require_no_low_level_flags: bool = True,
) -> ResolutionReceipt:
    """Orchestrate prompt-only resolution from trusted ambient evidence.

    - When evidence is sufficient (CWD + signed profile or auth server),
      low-level target/profile flags are not required.
    - Prompt text cannot populate allowlist, caller, policy, provider,
      validation_argv, or authority.
    - Material ambiguity never launches.
    - Unchanged evidence (+ same inputs) yields an identical receipt.
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")

    # Strip any forbidden fields from prompt-sourced bindings.
    sanitize_prompt_bindings(prompt, prompt_bindings)

    # Trusted bindings may carry policy/provider/etc.; prompt bindings must not.
    trusted: MutableMapping[str, Any] = dict(trusted_bindings) if trusted_bindings else {}

    fp = evidence.fingerprint()
    prompt_hash = _hash_prompt(prompt)

    sufficient = evidence.is_sufficient_for_prompt_only()

    # Ambiguity: multiple conflicting resolution paths without clear authority.
    ambiguities: list[str] = []

    resolved_target = target
    resolved_profile = profile

    if sufficient:
        # No low-level flags required; ambient evidence drives resolution.
        if resolved_profile is None and evidence.profile_path:
            resolved_profile = evidence.profile_path
        if resolved_target is None:
            if evidence.server_context and evidence.server_context.get("target"):
                resolved_target = str(evidence.server_context["target"])
            elif evidence.extra.get("default_target"):
                resolved_target = str(evidence.extra["default_target"])
            elif evidence.profile_path:
                # Target may be implied by signed profile path basename.
                resolved_target = "ambient:" + Path(evidence.profile_path).stem
    else:
        # Insufficient ambient evidence: low-level flags may be needed.
        if require_no_low_level_flags and not (resolved_target or resolved_profile):
            ambiguities.append(
                "insufficient ambient evidence and no target/profile flags provided"
            )
        if not resolved_profile and evidence.profile_path and not evidence.profile_signed:
            ambiguities.append("profile present but not signed and no authenticated server")
        if not resolved_target and not resolved_profile:
            ambiguities.append("cannot resolve target or profile from ambient evidence")

    # Conflicting explicit target vs server target is material ambiguity.
    if (
        target
        and evidence.server_context
        and evidence.server_context.get("target")
        and str(evidence.server_context["target"]) != str(target)
    ):
        ambiguities.append("explicit target conflicts with authenticated server target")

    if (
        profile
        and evidence.profile_path
        and os.path.abspath(profile) != os.path.abspath(evidence.profile_path)
        and evidence.profile_signed
    ):
        ambiguities.append("explicit profile conflicts with installed signed profile")

    if ambiguities:
        return ResolutionReceipt(
            evidence_fingerprint=fp,
            resolved=False,
            launch_authorized=False,
            target=resolved_target,
            profile=resolved_profile,
            reason="; ".join(ambiguities),
            prompt_hash=prompt_hash,
            policy=trusted.get("policy"),
            provider=trusted.get("provider"),
            caller=trusted.get("caller"),
            allowlist=trusted.get("allowlist"),
            authority=trusted.get("authority"),
            validation_argv=trusted.get("validation_argv"),
        )

    if not resolved_target and not resolved_profile:
        return ResolutionReceipt(
            evidence_fingerprint=fp,
            resolved=False,
            launch_authorized=False,
            reason="material ambiguity: no resolvable target or profile",
            prompt_hash=prompt_hash,
            policy=trusted.get("policy"),
            provider=trusted.get("provider"),
            caller=trusted.get("caller"),
            allowlist=trusted.get("allowlist"),
            authority=trusted.get("authority"),
            validation_argv=trusted.get("validation_argv"),
        )

    return ResolutionReceipt(
        evidence_fingerprint=fp,
        resolved=True,
        launch_authorized=True,
        target=resolved_target,
        profile=resolved_profile,
        reason="prompt-only resolution from trusted ambient evidence",
        prompt_hash=prompt_hash,
        policy=trusted.get("policy"),
        provider=trusted.get("provider"),
        caller=trusted.get("caller"),
        allowlist=trusted.get("allowlist"),
        authority=trusted.get("authority"),
        validation_argv=trusted.get("validation_argv"),
    )


def launch_if_authorized(receipt: ResolutionReceipt) -> ResolutionReceipt:
    """Launch only when authorized; material ambiguity never launches."""
    if not receipt.launch_authorized:
        raise MaterialAmbiguityError(
            receipt.reason or "launch denied: material ambiguity or unresolved target"
        )
    return receipt


def orchestrate(
    prompt: str,
    *,
    cwd: Optional[str] = None,
    profile_path: Optional[str] = None,
    profile_signed: Optional[bool] = None,
    server_context: Optional[Mapping[str, Any]] = None,
    server_authenticated: Optional[bool] = None,
    trusted_bindings: Optional[Mapping[str, Any]] = None,
    prompt_bindings: Optional[Mapping[str, Any]] = None,
    target: Optional[str] = None,
    profile: Optional[str] = None,
    launch: bool = False,
) -> ResolutionReceipt:
    """Collect ambient evidence and orchestrate prompt-only resolution.

    Primary public entrypoint for ASE2-001.
    """
    evidence = collect_ambient_evidence(
        cwd=cwd,
        profile_path=profile_path,
        profile_signed=profile_signed,
        server_context=server_context,
        server_authenticated=server_authenticated,
    )
    receipt = resolve_prompt_only(
        prompt,
        evidence,
        trusted_bindings=trusted_bindings,
        prompt_bindings=prompt_bindings,
        target=target,
        profile=profile,
    )
    if launch:
        return launch_if_authorized(receipt)
    return receipt
