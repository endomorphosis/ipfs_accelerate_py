"""Claude CLI and Gemini CLI quota / balance observation (side-effect free).

Reuses :mod:`ipfs_accelerate_py.endpoint_usage.adapters` structured CLI
metadata parsing and adds text classifiers for Anthropic Claude Code and
Google Gemini CLI failure envelopes.  Probes never invoke a model; they only
inspect PATH, well-known config paths, and optional env credentials.

Hard vs transient:

* **hard_quota_exhausted** — billing / credit / plan exhaustion (wait-for-reset
  does not restore service without operator action or a long calendar horizon).
* **capacity_restricted** — rate / usage-window limits with a declared or
  default retry horizon.

These observations feed :mod:`implementation_provider_auto` and the daemon's
provider-capacity latch families.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


CLI_PROVIDER_BALANCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/cli-provider-balance@1"
)

CLAUDE_PROVIDER_ID = "claude"
GEMINI_PROVIDER_ID = "gemini"

# Preference among secondary implementation backends after preferred Grok
# hard-quota exhaustion (higher index = lower preference).
SECONDARY_IMPLEMENTATION_PREFERENCE: tuple[str, ...] = (
    "codex",
    CLAUDE_PROVIDER_ID,
    GEMINI_PROVIDER_ID,
)

_CLAUDE_BIN_ENVS = (
    "IPFS_ACCELERATE_AGENT_CLAUDE_BIN",
    "ipfs_accelerate_py_CLAUDE_CODE_CLI_CMD",
    "CLAUDE_BIN",
    "ANTHROPIC_CLI_BIN",
)
_GEMINI_BIN_ENVS = (
    "IPFS_ACCELERATE_AGENT_GEMINI_BIN",
    "ipfs_accelerate_py_GEMINI_CLI_CMD",
    "GEMINI_BIN",
    "GOOGLE_GEMINI_CLI_BIN",
)
_CLAUDE_AUTH_ENVS = (
    "ANTHROPIC_API_KEY",
    "CLAUDE_API_KEY",
    "ipfs_accelerate_py_ANTHROPIC_API_KEY",
    "IPFS_ACCELERATE_PY_ANTHROPIC_API_KEY",
)
_GEMINI_AUTH_ENVS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_GENAI_API_KEY",
    "ipfs_accelerate_py_GEMINI_API_KEY",
    "IPFS_ACCELERATE_PY_GEMINI_API_KEY",
)

# Hard billing / credit stop (does not clear on short retry).
_CLAUDE_HARD_QUOTA = re.compile(
    r"(?:"
    r"credit\s+balance\s+is\s+too\s+low|"
    r"insufficient[_ ](?:credits?|quota|balance)|"
    r"billing[_ ]?(?:error|not[_ ]active|hard[_ ]limit)|"
    r"payment[_ ]required|"
    r"plan\s+(?:does\s+not|doesn't)\s+include|"
    r"out\s+of\s+credits?|"
    r"add\s+(?:more\s+)?credits?|"
    r"purchase\s+(?:more\s+)?credits?|"
    r"your\s+organization\s+has\s+been\s+disabled"
    r")",
    re.IGNORECASE,
)
_CLAUDE_RATE_LIMIT = re.compile(
    r"(?:"
    r"rate[_ ]?limit(?:ed|_error|_exceeded)?|"
    r"usage[_ ]?limit|"
    r"you've\s+hit\s+your\s+(?:usage\s+)?limit|"
    r"you\s+have\s+hit\s+your\s+(?:usage\s+)?limit|"
    r"too\s+many\s+requests|"
    r"overloaded[_ ]?error|"
    r"429\b|"
    r"anthropic[_-]?ratelimit"
    r")",
    re.IGNORECASE,
)
_GEMINI_HARD_QUOTA = re.compile(
    r"(?:"
    r"billing[_ ]?(?:disabled|not[_ ]enabled|account)|"
    r"insufficient[_ ](?:credits?|quota)|"
    r"consumer[_ ](?:invalid|suspended)|"
    r"payment[_ ]required|"
    r"free[_ ]tier\s+quota\s+exhausted|"
    r"out\s+of\s+credits?"
    r")",
    re.IGNORECASE,
)
_GEMINI_AUTH = re.compile(
    r"(?:"
    r"api\s+key\s+not\s+valid|"
    r"unauthenticated|"
    r"permission[_ ]denied|"
    r"invalid[_ ]api[_ ]key|"
    r"missing\s+api\s+key|"
    r"login\s+required"
    r")",
    re.IGNORECASE,
)
_GEMINI_RATE_LIMIT = re.compile(
    r"(?:"
    r"resource[_ ]?exhausted|"
    r"rate[_ ]?limit(?:ed|_exceeded)?|"
    r"usage[_ ]?limit|"
    r"too\s+many\s+requests|"
    r"429\b|"
    r"quota\s+metric|"
    r"exceeded\s+your\s+current\s+quota|"
    r"generativelanguage\.googleapis\.com.*quota"
    r")",
    re.IGNORECASE,
)

_RETRY_SECONDS = re.compile(
    r"(?:retry(?:\s+after)?|resets?\s+in|try\s+again\s+in)\s*[:=]?\s*"
    r"(?P<seconds>\d+(?:\.\d+)?)\s*(?:s(?:ec(?:ond)?s?)?)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CliQuotaClassification:
    """Bounded classification of one CLI stderr/stdout envelope."""

    provider_id: str
    failure_class: str
    hard_quota_exhausted: bool
    capacity_restricted: bool
    authenticated_failure: bool
    retry_after_seconds: int | None = None
    reason_codes: tuple[str, ...] = ()
    kind: str = ""  # endpoint_usage CLI metadata kind
    evidence_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CLI_PROVIDER_BALANCE_SCHEMA,
            "provider_id": self.provider_id,
            "failure_class": self.failure_class,
            "hard_quota_exhausted": self.hard_quota_exhausted,
            "capacity_restricted": self.capacity_restricted,
            "authenticated_failure": self.authenticated_failure,
            "retry_after_seconds": self.retry_after_seconds,
            "reason_codes": list(self.reason_codes),
            "kind": self.kind,
            "evidence_sha256": self.evidence_sha256,
        }


def _env_nonempty(*names: str) -> str:
    for name in names:
        value = str(os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def _sha256_text(text: str) -> str:
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return f"sha256:{digest}"


def _first_path_component(command: str) -> str:
    """Extract the executable token from a command template."""

    raw = str(command or "").strip()
    if not raw:
        return ""
    # Templates like "npx @google/gemini-cli {prompt}" or "claude {prompt}".
    token = raw.split()[0]
    return token.replace("{prompt}", "").replace("{model}", "").strip()


def resolve_claude_cli_binary() -> str | None:
    """Locate Claude Code CLI without executing it."""

    for env_name in _CLAUDE_BIN_ENVS:
        configured = str(os.environ.get(env_name) or "").strip()
        if not configured:
            continue
        token = _first_path_component(configured)
        if not token:
            continue
        path = Path(token).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        found = shutil.which(token)
        if found:
            return found
    return shutil.which("claude")


def resolve_gemini_cli_binary() -> str | None:
    """Locate Gemini CLI without executing it."""

    for env_name in _GEMINI_BIN_ENVS:
        configured = str(os.environ.get(env_name) or "").strip()
        if not configured:
            continue
        token = _first_path_component(configured)
        if not token:
            continue
        # npx-based templates: treat as available when npx exists.
        if token in {"npx", "npm"}:
            if shutil.which(token):
                return token
            continue
        path = Path(token).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        found = shutil.which(token)
        if found:
            return found
    for candidate in ("gemini", "google-gemini"):
        found = shutil.which(candidate)
        if found:
            return found
    # Official install often uses npx @google/gemini-cli
    if shutil.which("npx"):
        return "npx"
    return None


def claude_cli_auth_available() -> bool:
    """Return whether Claude CLI auth material is present (no network)."""

    if _env_nonempty(*_CLAUDE_AUTH_ENVS):
        return True
    home = Path.home()
    candidates = (
        home / ".claude" / ".credentials.json",
        home / ".claude" / "credentials.json",
        home / ".config" / "claude" / "credentials.json",
        home / ".config" / "claude" / ".credentials.json",
    )
    for path in candidates:
        try:
            if path.is_file() and path.stat().st_size > 0:
                return True
        except OSError:
            continue
    # Directory presence after `claude login` is a weak positive signal.
    claude_dir = home / ".claude"
    try:
        if claude_dir.is_dir() and any(claude_dir.iterdir()):
            return True
    except OSError:
        pass
    return False


def gemini_cli_auth_available() -> bool:
    """Return whether Gemini CLI auth material is present (no network)."""

    if _env_nonempty(*_GEMINI_AUTH_ENVS):
        return True
    home = Path.home()
    candidates = (
        home / ".gemini" / "oauth_creds.json",
        home / ".gemini" / "settings.json",
        home / ".config" / "gemini" / "oauth_creds.json",
        home / ".config" / "gemini" / "settings.json",
        home / ".config" / "gcloud" / "application_default_credentials.json",
    )
    for path in candidates:
        try:
            if path.is_file() and path.stat().st_size > 0:
                return True
        except OSError:
            continue
    gemini_dir = home / ".gemini"
    try:
        if gemini_dir.is_dir() and any(gemini_dir.iterdir()):
            return True
    except OSError:
        pass
    return False


def _extract_retry_seconds(text: str) -> int | None:
    match = _RETRY_SECONDS.search(text)
    if match is None:
        return None
    try:
        seconds = float(match.group("seconds"))
    except (TypeError, ValueError):
        return None
    if seconds < 0 or seconds > 31_536_000:
        return None
    return int(seconds)


def classify_claude_cli_text(text: str) -> CliQuotaClassification:
    """Classify Claude Code CLI output for quota / rate-limit / auth."""

    body = str(text or "")
    digest = _sha256_text(body[-8_192:])
    retry = _extract_retry_seconds(body)
    if _CLAUDE_HARD_QUOTA.search(body):
        return CliQuotaClassification(
            provider_id=CLAUDE_PROVIDER_ID,
            failure_class="hard_quota_exhausted",
            hard_quota_exhausted=True,
            capacity_restricted=True,
            authenticated_failure=False,
            retry_after_seconds=retry,
            reason_codes=("billing.exhausted", "cli.claude"),
            kind="quota_exceeded",
            evidence_sha256=digest,
        )
    if _CLAUDE_RATE_LIMIT.search(body):
        return CliQuotaClassification(
            provider_id=CLAUDE_PROVIDER_ID,
            failure_class="rate_limited",
            hard_quota_exhausted=False,
            capacity_restricted=True,
            authenticated_failure=False,
            retry_after_seconds=retry,
            reason_codes=("subscription.usage_limit", "cli.claude"),
            kind="usage_limit",
            evidence_sha256=digest,
        )
    if re.search(r"authentication|unauthorized|invalid.?api.?key|not logged in", body, re.I):
        return CliQuotaClassification(
            provider_id=CLAUDE_PROVIDER_ID,
            failure_class="authentication",
            hard_quota_exhausted=False,
            capacity_restricted=False,
            authenticated_failure=True,
            reason_codes=("auth.failed", "cli.claude"),
            kind="authentication",
            evidence_sha256=digest,
        )
    return CliQuotaClassification(
        provider_id=CLAUDE_PROVIDER_ID,
        failure_class="unknown",
        hard_quota_exhausted=False,
        capacity_restricted=False,
        authenticated_failure=False,
        reason_codes=("cli.claude",),
        evidence_sha256=digest,
    )


def classify_gemini_cli_text(text: str) -> CliQuotaClassification:
    """Classify Gemini CLI output for quota / rate-limit / auth."""

    body = str(text or "")
    digest = _sha256_text(body[-8_192:])
    retry = _extract_retry_seconds(body)
    if _GEMINI_AUTH.search(body):
        return CliQuotaClassification(
            provider_id=GEMINI_PROVIDER_ID,
            failure_class="authentication",
            hard_quota_exhausted=False,
            capacity_restricted=False,
            authenticated_failure=True,
            reason_codes=("auth.failed", "cli.gemini"),
            kind="authentication",
            evidence_sha256=digest,
        )
    if _GEMINI_HARD_QUOTA.search(body):
        return CliQuotaClassification(
            provider_id=GEMINI_PROVIDER_ID,
            failure_class="hard_quota_exhausted",
            hard_quota_exhausted=True,
            capacity_restricted=True,
            authenticated_failure=False,
            retry_after_seconds=retry,
            reason_codes=("billing.exhausted", "cli.gemini"),
            kind="quota_exceeded",
            evidence_sha256=digest,
        )
    if _GEMINI_RATE_LIMIT.search(body):
        return CliQuotaClassification(
            provider_id=GEMINI_PROVIDER_ID,
            failure_class="rate_limited",
            hard_quota_exhausted=False,
            capacity_restricted=True,
            authenticated_failure=False,
            retry_after_seconds=retry,
            reason_codes=("subscription.usage_limit", "cli.gemini"),
            kind="rate_limit",
            evidence_sha256=digest,
        )
    return CliQuotaClassification(
        provider_id=GEMINI_PROVIDER_ID,
        failure_class="unknown",
        hard_quota_exhausted=False,
        capacity_restricted=False,
        authenticated_failure=False,
        reason_codes=("cli.gemini",),
        evidence_sha256=digest,
    )


def classify_cli_provider_text(provider_id: str, text: str) -> CliQuotaClassification:
    """Dispatch to the Claude or Gemini classifier."""

    pid = str(provider_id or "").strip().lower().replace("-", "_")
    if pid in {"claude", "claude_code", "claude_cli", "anthropic"}:
        return classify_claude_cli_text(text)
    if pid in {"gemini", "gemini_cli", "google_gemini", "google"}:
        return classify_gemini_cli_text(text)
    raise ValueError(f"unsupported cli provider {provider_id!r}")


def parse_cli_balance_observation(
    provider_id: str,
    *,
    kind: str,
    resets_in_seconds: int | None = None,
    usage: Mapping[str, Any] | None = None,
    error_text: str = "",
) -> dict[str, Any]:
    """Normalize structured CLI metadata via endpoint_usage adapters.

    Returns a compact dict suitable for ``usage_observations`` overlays in
    :func:`implementation_provider_auto.probe_llm_router_backends`.
    Falls back to text classification when structured parse fails.
    """

    pid = str(provider_id or "").strip().lower()
    if pid in {"claude", "claude_code", "anthropic"}:
        family = CLAUDE_PROVIDER_ID
    elif pid in {"gemini", "gemini_cli"}:
        family = GEMINI_PROVIDER_ID
    else:
        family = pid

    # Prefer structured adapter when possible.
    try:
        from ipfs_accelerate_py.endpoint_usage.adapters import parse_cli_observation
        from ipfs_accelerate_py.endpoint_usage.identity import (
            credential_configuration_pseudonym,
            stable_id,
        )
        from ipfs_accelerate_py.endpoint_usage.schema import (
            EndpointUsageScope,
            ProtocolKind,
        )

        provider_scope_id = stable_id("provider", family)
        config_ref = (
            "env:ANTHROPIC_API_KEY"
            if family == CLAUDE_PROVIDER_ID
            else "env:GEMINI_API_KEY"
            if family == GEMINI_PROVIDER_ID
            else f"env:{family.upper()}_API_KEY"
        )
        credential = credential_configuration_pseudonym(
            config_ref,
            key_id=f"{family}-balance",
        )
        scope = EndpointUsageScope(
            provider_id=provider_scope_id,
            protocol=ProtocolKind.CLI,
            operation="text.generate",
            deployment_id=stable_id(
                "deployment", provider_scope_id, "cli", "local"
            ),
            credential_pseudonym=credential,
        )
        metadata: dict[str, Any] = {
            "provider": family,
            "kind": str(kind or "usage_limit"),
        }
        if resets_in_seconds is not None:
            metadata["resets_in_seconds"] = int(resets_in_seconds)
        if isinstance(usage, Mapping):
            metadata["usage"] = dict(usage)
        observation = parse_cli_observation(
            {
                "scope": scope,
                "request_id": stable_id("req", family, "balance"),
                "observed_at": datetime.now(timezone.utc),
                "now": datetime.now(timezone.utc),
                "cli_metadata": metadata,
            }
        )
        reason_codes = tuple(observation.reason_codes or ())
        hard = any("billing.exhausted" in code for code in reason_codes)
        restricted = hard or any(
            "usage_limit" in code or "rate" in code for code in reason_codes
        )
        headroom = None
        for limit in observation.limits or ():
            remaining = getattr(limit, "remaining", None)
            kind_name = str(
                getattr(getattr(remaining, "kind", None), "name", "") or ""
            )
            if (
                kind_name == "FINITE"
                and getattr(remaining, "value", None) is not None
            ):
                headroom = int(remaining.value)
                break
        retry_after_seconds = None
        if observation.retry_after_ms is not None:
            retry_after_seconds = int(observation.retry_after_ms) // 1000
        elif resets_in_seconds is not None:
            retry_after_seconds = int(resets_in_seconds)
        return {
            "provider_id": family,
            "hard_quota_exhausted": hard,
            "capacity_restricted": restricted and not hard,
            "capacity_latched": restricted and not hard,
            "request_headroom": headroom,
            "retry_after_seconds": retry_after_seconds,
            "reason_codes": list(reason_codes),
            "source": "endpoint_usage.cli",
        }
    except Exception:
        pass

    # Text fallback. Prefer explicit error_text; do not treat bare kind tokens
    # as Claude/Gemini prose unless they map to known classifier vocabulary.
    text_for_classifier = str(error_text or "").strip()
    if not text_for_classifier:
        kind_token = str(kind or "").strip().casefold()
        if kind_token in {
            "quota_exceeded",
            "billing",
            "insufficient_quota",
        }:
            text_for_classifier = (
                "billing hard limit: credit balance is too low / quota exceeded"
                if family == CLAUDE_PROVIDER_ID
                else "billing account disabled; payment required; free tier quota exhausted"
            )
        elif kind_token in {"usage_limit", "rate_limit", "capacity"}:
            text_for_classifier = (
                "rate_limit_error: you've hit your usage limit retry after 60s"
                if family == CLAUDE_PROVIDER_ID
                else "429 RESOURCE_EXHAUSTED rate limit exceeded"
            )
    classified = classify_cli_provider_text(family, text_for_classifier)
    return {
        "provider_id": family,
        "hard_quota_exhausted": classified.hard_quota_exhausted,
        "capacity_restricted": classified.capacity_restricted
        and not classified.hard_quota_exhausted,
        "capacity_latched": classified.capacity_restricted
        and not classified.hard_quota_exhausted,
        "retry_after_seconds": (
            classified.retry_after_seconds
            if classified.retry_after_seconds is not None
            else (
                int(resets_in_seconds) if resets_in_seconds is not None else None
            )
        ),
        "reason_codes": list(classified.reason_codes),
        "failure_class": classified.failure_class,
        "source": "cli_text_classifier",
    }


def probe_claude_cli_readiness() -> dict[str, Any]:
    """Non-charging readiness snapshot for Claude Code CLI."""

    binary = resolve_claude_cli_binary()
    authenticated = claude_cli_auth_available()
    return {
        "provider_id": CLAUDE_PROVIDER_ID,
        "binary_available": bool(binary),
        "binary_path": binary or "",
        "authenticated": authenticated,
        "ready": bool(binary and authenticated),
        "source": "cli_probe",
    }


def probe_gemini_cli_readiness() -> dict[str, Any]:
    """Non-charging readiness snapshot for Gemini CLI."""

    binary = resolve_gemini_cli_binary()
    authenticated = gemini_cli_auth_available()
    return {
        "provider_id": GEMINI_PROVIDER_ID,
        "binary_available": bool(binary),
        "binary_path": binary or "",
        "authenticated": authenticated,
        "ready": bool(binary and authenticated),
        "source": "cli_probe",
    }


def probe_claude_and_gemini_readiness() -> dict[str, dict[str, Any]]:
    """Return readiness for both CLIs keyed by provider id."""

    return {
        CLAUDE_PROVIDER_ID: probe_claude_cli_readiness(),
        GEMINI_PROVIDER_ID: probe_gemini_cli_readiness(),
    }


__all__ = [
    "CLAUDE_PROVIDER_ID",
    "CLI_PROVIDER_BALANCE_SCHEMA",
    "CliQuotaClassification",
    "GEMINI_PROVIDER_ID",
    "SECONDARY_IMPLEMENTATION_PREFERENCE",
    "classify_claude_cli_text",
    "classify_cli_provider_text",
    "classify_gemini_cli_text",
    "claude_cli_auth_available",
    "gemini_cli_auth_available",
    "parse_cli_balance_observation",
    "probe_claude_and_gemini_readiness",
    "probe_claude_cli_readiness",
    "probe_gemini_cli_readiness",
    "resolve_claude_cli_binary",
    "resolve_gemini_cli_binary",
]
