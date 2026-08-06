"""CLI provider quota / balance observation (side-effect free).

Reuses :mod:`ipfs_accelerate_py.endpoint_usage.adapters` structured CLI
metadata parsing and adds text classifiers for:

* Anthropic Claude Code CLI
* Google Gemini CLI
* Meta Muse Spark (via Goose)
* Mistral Vibe CLI
* GitHub Copilot CLI

Probes never invoke a model; they only inspect PATH, well-known config paths,
and optional env credentials.

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
META_SPARK_PROVIDER_ID = "meta_spark"
MISTRAL_PROVIDER_ID = "mistral"
COPILOT_PROVIDER_ID = "copilot"

# Preference among secondary implementation backends after preferred Grok
# hard-quota exhaustion (higher index = lower preference).
SECONDARY_IMPLEMENTATION_PREFERENCE: tuple[str, ...] = (
    "codex",
    CLAUDE_PROVIDER_ID,
    GEMINI_PROVIDER_ID,
    COPILOT_PROVIDER_ID,
    META_SPARK_PROVIDER_ID,
    MISTRAL_PROVIDER_ID,
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

_META_SPARK_HARD_QUOTA = re.compile(
    r"(?:"
    r"insufficient[_ ]quota|"
    r"quota[_ ]exceeded|"
    r"billing[_ ]?(?:not[_ ]active|hard[_ ]limit)|"
    r"payment[_ ]required|"
    r"credit\s+balance|"
    r"out\s+of\s+credits?|"
    r"account\s+is\s+not\s+active"
    r")",
    re.IGNORECASE,
)
_META_SPARK_RATE_LIMIT = re.compile(
    r"(?:"
    r"meta\s+ai\s+http\s+(?:401|403|429)|"
    r"rate[_ ]?limit(?:ed|s|_exceeded)?|"
    r"usage[_ ]?limit|"
    r"model\s+is\s+currently\s+overloaded|"
    r"too\s+many\s+requests|"
    r"429\b|"
    r"provider.*exhausted|"
    r"goose.*(?:rate\s+limit|quota|usage\s+limit)"
    r")",
    re.IGNORECASE,
)
_META_SPARK_AUTH = re.compile(
    r"(?:"
    r"invalid[_ ]api[_ ]key|"
    r"unauthorized|"
    r"authentication\s+failed|"
    r"missing\s+api\s+key|"
    r"meta_ai_api_key|"
    r"not\s+logged\s+in"
    r")",
    re.IGNORECASE,
)

_MISTRAL_HARD_QUOTA = re.compile(
    r"(?:"
    r"insufficient[_ ](?:credits?|quota)|"
    r"quota[_ ]exceeded|"
    r"billing[_ ]?(?:error|not[_ ]active)|"
    r"payment[_ ]required|"
    r"out\s+of\s+credits?|"
    r"credit\s+balance|"
    r"plan\s+limit\s+reached"
    r")",
    re.IGNORECASE,
)
_MISTRAL_RATE_LIMIT = re.compile(
    r"(?:"
    r"rate[_ ]?limit(?:ed|_exceeded)?|"
    r"usage[_ ]?limit|"
    r"too\s+many\s+requests|"
    r"429\b|"
    r"capacity|"
    r"overloaded|"
    r"service\s+unavailable"
    r")",
    re.IGNORECASE,
)
_MISTRAL_AUTH = re.compile(
    r"(?:"
    r"invalid[_ ]api[_ ]key|"
    r"unauthorized|"
    r"authentication|"
    r"missing\s+api\s+key|"
    r"mistral_api_key|"
    r"run\s+vibe\s+--setup"
    r")",
    re.IGNORECASE,
)

_COPILOT_HARD_QUOTA = re.compile(
    r"(?:"
    r"billing|"
    r"payment[_ ]required|"
    r"subscription\s+(?:required|expired|inactive)|"
    r"copilot\s+is\s+not\s+available|"
    r"not\s+included\s+in\s+your\s+plan|"
    r"upgrade\s+your\s+plan"
    r")",
    re.IGNORECASE,
)
_COPILOT_RATE_LIMIT = re.compile(
    r"(?:"
    r"you(?:'|\u2019)?ve\s+reached\s+your\s+additional\s+usage\s+limit|"
    r"usage[_ ]?limit|"
    r"rate[_ ]?limit(?:ed|_exceeded)?|"
    r"too\s+many\s+requests|"
    r"429\b|"
    r"quota|"
    r"try\s+again\s+later"
    r")",
    re.IGNORECASE,
)
_COPILOT_AUTH = re.compile(
    r"(?:"
    r"not\s+logged\s+in|"
    r"authentication|"
    r"unauthorized|"
    r"gh\s+auth|"
    r"github\s+token|"
    r"please\s+run\s+.*login"
    r")",
    re.IGNORECASE,
)

_META_SPARK_AUTH_ENVS = (
    "MODEL_API_KEY",
    "META_AI_API_KEY",
    "ipfs_accelerate_py_META_AI_API_KEY",
    "IPFS_ACCELERATE_PY_META_AI_API_KEY",
    "OPENAI_API_KEY",  # Goose OpenAI-compatible Meta host often uses this
)
_MISTRAL_AUTH_ENVS = (
    "MISTRAL_API_KEY",
    "IPFS_ACCELERATE_MISTRAL_API_KEY",
    "ipfs_accelerate_py_MISTRAL_API_KEY",
    "IPFS_ACCELERATE_PY_MISTRAL_API_KEY",
)
_COPILOT_AUTH_ENVS = (
    "COPILOT_GITHUB_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
)
_GOOSE_BIN_ENVS = (
    "IPFS_ACCELERATE_AGENT_GOOSE_BIN",
    "IPFS_ACCELERATE_GOOSE_PATH",
    "ipfs_accelerate_py_GOOSE_BIN",
    "GOOSE_BIN",
)
_MISTRAL_BIN_ENVS = (
    "IPFS_ACCELERATE_MISTRAL_VIBE_CLI_CMD",
    "ipfs_accelerate_py_MISTRAL_VIBE_CLI_CMD",
    "IPFS_ACCELERATE_PY_MISTRAL_VIBE_CLI_CMD",
    "MISTRAL_VIBE_BIN",
    "VIBE_BIN",
)
_COPILOT_BIN_ENVS = (
    "IPFS_ACCELERATE_AGENT_COPILOT_BIN",
    "COPILOT_BIN",
    "GITHUB_COPILOT_BIN",
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


def resolve_goose_cli_binary() -> str | None:
    """Locate Goose CLI (Meta Spark agent surface) without executing it."""

    try:
        from ipfs_accelerate_py.llm_router import find_goose_cli

        found = find_goose_cli()
        if found:
            return found
    except Exception:
        pass
    for env_name in _GOOSE_BIN_ENVS:
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
    return shutil.which("goose")


def resolve_mistral_cli_binary() -> str | None:
    """Locate Mistral Vibe CLI without executing or installing it."""

    for env_name in _MISTRAL_BIN_ENVS:
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
    for candidate in ("vibe", "mistral-vibe", "mistral"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def resolve_copilot_cli_binary() -> str | None:
    """Locate GitHub Copilot CLI without executing it."""

    for env_name in _COPILOT_BIN_ENVS:
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
    for candidate in ("copilot", "github-copilot"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def meta_spark_auth_available() -> bool:
    """Return whether Meta Muse Spark credentials are configured (no network)."""

    if _env_nonempty(*_META_SPARK_AUTH_ENVS):
        return True
    try:
        from ipfs_accelerate_py.common.meta_model_api import resolve_meta_model_api_key

        return bool(str(resolve_meta_model_api_key() or "").strip())
    except Exception:
        return False


def mistral_cli_auth_available() -> bool:
    """Return whether Mistral Vibe auth is present (no network, no install)."""

    if _env_nonempty(*_MISTRAL_AUTH_ENVS):
        return True
    try:
        from ipfs_accelerate_py.utils.mistral_vibe import mistral_vibe_auth_available

        return bool(mistral_vibe_auth_available())
    except Exception:
        pass
    home = Path.home()
    for path in (
        home / ".config" / "mistral-vibe",
        home / ".mistral-vibe",
        home / ".vibe",
    ):
        try:
            if path.is_dir() and any(path.iterdir()):
                return True
        except OSError:
            continue
    return False


def copilot_cli_auth_available() -> bool:
    """Return whether Copilot CLI non-interactive auth is available.

    Prefers env tokens so probes stay side-effect free.  When only ``gh`` is
    installed, a status check is used (no network login).
    """

    if _env_nonempty(*_COPILOT_AUTH_ENVS):
        return True
    gh = shutil.which("gh")
    if not gh:
        return False
    try:
        import subprocess

        completed = subprocess.run(
            [gh, "auth", "status"],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
        return completed.returncode == 0
    except Exception:
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


def classify_meta_spark_cli_text(text: str) -> CliQuotaClassification:
    """Classify Meta Spark / Goose CLI output for quota / rate-limit / auth."""

    body = str(text or "")
    digest = _sha256_text(body[-8_192:])
    retry = _extract_retry_seconds(body)
    if _META_SPARK_AUTH.search(body) and not _META_SPARK_HARD_QUOTA.search(body):
        return CliQuotaClassification(
            provider_id=META_SPARK_PROVIDER_ID,
            failure_class="authentication",
            hard_quota_exhausted=False,
            capacity_restricted=False,
            authenticated_failure=True,
            reason_codes=("auth.failed", "cli.meta_spark"),
            kind="authentication",
            evidence_sha256=digest,
        )
    if _META_SPARK_HARD_QUOTA.search(body):
        return CliQuotaClassification(
            provider_id=META_SPARK_PROVIDER_ID,
            failure_class="hard_quota_exhausted",
            hard_quota_exhausted=True,
            capacity_restricted=True,
            authenticated_failure=False,
            retry_after_seconds=retry,
            reason_codes=("billing.exhausted", "cli.meta_spark"),
            kind="quota_exceeded",
            evidence_sha256=digest,
        )
    if _META_SPARK_RATE_LIMIT.search(body):
        return CliQuotaClassification(
            provider_id=META_SPARK_PROVIDER_ID,
            failure_class="rate_limited",
            hard_quota_exhausted=False,
            capacity_restricted=True,
            authenticated_failure=False,
            retry_after_seconds=retry,
            reason_codes=("subscription.usage_limit", "cli.meta_spark"),
            kind="rate_limit",
            evidence_sha256=digest,
        )
    return CliQuotaClassification(
        provider_id=META_SPARK_PROVIDER_ID,
        failure_class="unknown",
        hard_quota_exhausted=False,
        capacity_restricted=False,
        authenticated_failure=False,
        reason_codes=("cli.meta_spark",),
        evidence_sha256=digest,
    )


def classify_mistral_cli_text(text: str) -> CliQuotaClassification:
    """Classify Mistral Vibe CLI output for quota / rate-limit / auth."""

    body = str(text or "")
    digest = _sha256_text(body[-8_192:])
    retry = _extract_retry_seconds(body)
    if _MISTRAL_AUTH.search(body) and not _MISTRAL_HARD_QUOTA.search(body):
        return CliQuotaClassification(
            provider_id=MISTRAL_PROVIDER_ID,
            failure_class="authentication",
            hard_quota_exhausted=False,
            capacity_restricted=False,
            authenticated_failure=True,
            reason_codes=("auth.failed", "cli.mistral"),
            kind="authentication",
            evidence_sha256=digest,
        )
    if _MISTRAL_HARD_QUOTA.search(body):
        return CliQuotaClassification(
            provider_id=MISTRAL_PROVIDER_ID,
            failure_class="hard_quota_exhausted",
            hard_quota_exhausted=True,
            capacity_restricted=True,
            authenticated_failure=False,
            retry_after_seconds=retry,
            reason_codes=("billing.exhausted", "cli.mistral"),
            kind="quota_exceeded",
            evidence_sha256=digest,
        )
    if _MISTRAL_RATE_LIMIT.search(body):
        return CliQuotaClassification(
            provider_id=MISTRAL_PROVIDER_ID,
            failure_class="rate_limited",
            hard_quota_exhausted=False,
            capacity_restricted=True,
            authenticated_failure=False,
            retry_after_seconds=retry,
            reason_codes=("subscription.usage_limit", "cli.mistral"),
            kind="usage_limit",
            evidence_sha256=digest,
        )
    return CliQuotaClassification(
        provider_id=MISTRAL_PROVIDER_ID,
        failure_class="unknown",
        hard_quota_exhausted=False,
        capacity_restricted=False,
        authenticated_failure=False,
        reason_codes=("cli.mistral",),
        evidence_sha256=digest,
    )


def classify_copilot_cli_text(text: str) -> CliQuotaClassification:
    """Classify GitHub Copilot CLI output for quota / rate-limit / auth."""

    body = str(text or "")
    digest = _sha256_text(body[-8_192:])
    retry = _extract_retry_seconds(body)
    if _COPILOT_AUTH.search(body) and not _COPILOT_RATE_LIMIT.search(body):
        return CliQuotaClassification(
            provider_id=COPILOT_PROVIDER_ID,
            failure_class="authentication",
            hard_quota_exhausted=False,
            capacity_restricted=False,
            authenticated_failure=True,
            reason_codes=("auth.failed", "cli.copilot"),
            kind="authentication",
            evidence_sha256=digest,
        )
    if _COPILOT_HARD_QUOTA.search(body) and not _COPILOT_RATE_LIMIT.search(body):
        return CliQuotaClassification(
            provider_id=COPILOT_PROVIDER_ID,
            failure_class="hard_quota_exhausted",
            hard_quota_exhausted=True,
            capacity_restricted=True,
            authenticated_failure=False,
            retry_after_seconds=retry,
            reason_codes=("billing.exhausted", "cli.copilot"),
            kind="quota_exceeded",
            evidence_sha256=digest,
        )
    if _COPILOT_RATE_LIMIT.search(body):
        return CliQuotaClassification(
            provider_id=COPILOT_PROVIDER_ID,
            failure_class="rate_limited",
            hard_quota_exhausted=False,
            capacity_restricted=True,
            authenticated_failure=False,
            retry_after_seconds=retry,
            reason_codes=("subscription.usage_limit", "cli.copilot"),
            kind="usage_limit",
            evidence_sha256=digest,
        )
    return CliQuotaClassification(
        provider_id=COPILOT_PROVIDER_ID,
        failure_class="unknown",
        hard_quota_exhausted=False,
        capacity_restricted=False,
        authenticated_failure=False,
        reason_codes=("cli.copilot",),
        evidence_sha256=digest,
    )


def classify_cli_provider_text(provider_id: str, text: str) -> CliQuotaClassification:
    """Dispatch to the matching CLI classifier."""

    pid = str(provider_id or "").strip().lower().replace("-", "_")
    if pid in {"claude", "claude_code", "claude_cli", "anthropic"}:
        return classify_claude_cli_text(text)
    if pid in {"gemini", "gemini_cli", "google_gemini", "google"}:
        return classify_gemini_cli_text(text)
    if pid in {
        "meta_spark",
        "meta",
        "goose",
        "goose_meta",
        "muse",
        "muse_spark",
        "spark",
    }:
        return classify_meta_spark_cli_text(text)
    if pid in {"mistral", "mistral_vibe", "vibe", "mistral_cli"}:
        return classify_mistral_cli_text(text)
    if pid in {"copilot", "github_copilot", "github-copilot"}:
        return classify_copilot_cli_text(text)
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

    pid = str(provider_id or "").strip().lower().replace("-", "_")
    if pid in {"claude", "claude_code", "anthropic"}:
        family = CLAUDE_PROVIDER_ID
    elif pid in {"gemini", "gemini_cli"}:
        family = GEMINI_PROVIDER_ID
    elif pid in {"meta_spark", "meta", "goose", "muse", "muse_spark", "spark"}:
        family = META_SPARK_PROVIDER_ID
    elif pid in {"mistral", "mistral_vibe", "vibe"}:
        family = MISTRAL_PROVIDER_ID
    elif pid in {"copilot", "github_copilot"}:
        family = COPILOT_PROVIDER_ID
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
        config_ref = {
            CLAUDE_PROVIDER_ID: "env:ANTHROPIC_API_KEY",
            GEMINI_PROVIDER_ID: "env:GEMINI_API_KEY",
            META_SPARK_PROVIDER_ID: "env:MODEL_API_KEY",
            MISTRAL_PROVIDER_ID: "env:MISTRAL_API_KEY",
            COPILOT_PROVIDER_ID: "env:GH_TOKEN",
        }.get(family, f"env:{family.upper()}_API_KEY")
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
    """Return readiness for Claude and Gemini CLIs keyed by provider id."""

    return {
        CLAUDE_PROVIDER_ID: probe_claude_cli_readiness(),
        GEMINI_PROVIDER_ID: probe_gemini_cli_readiness(),
    }


def probe_meta_spark_readiness() -> dict[str, Any]:
    """Non-charging readiness snapshot for Meta Spark via Goose."""

    binary = resolve_goose_cli_binary()
    authenticated = meta_spark_auth_available()
    return {
        "provider_id": META_SPARK_PROVIDER_ID,
        "binary_available": bool(binary),
        "binary_path": binary or "",
        "authenticated": authenticated,
        "ready": bool(binary and authenticated),
        "source": "cli_probe",
        "family": "goose",
    }


def probe_mistral_cli_readiness() -> dict[str, Any]:
    """Non-charging readiness snapshot for Mistral Vibe CLI."""

    binary = resolve_mistral_cli_binary()
    authenticated = mistral_cli_auth_available()
    return {
        "provider_id": MISTRAL_PROVIDER_ID,
        "binary_available": bool(binary),
        "binary_path": binary or "",
        "authenticated": authenticated,
        "ready": bool(binary and authenticated),
        "source": "cli_probe",
    }


def probe_copilot_cli_readiness() -> dict[str, Any]:
    """Non-charging readiness snapshot for GitHub Copilot CLI."""

    binary = resolve_copilot_cli_binary()
    authenticated = copilot_cli_auth_available()
    return {
        "provider_id": COPILOT_PROVIDER_ID,
        "binary_available": bool(binary),
        "binary_path": binary or "",
        "authenticated": authenticated,
        "ready": bool(binary and authenticated),
        "source": "cli_probe",
    }


def probe_all_cli_provider_readiness() -> dict[str, dict[str, Any]]:
    """Return readiness for every supported CLI implementer family."""

    return {
        CLAUDE_PROVIDER_ID: probe_claude_cli_readiness(),
        GEMINI_PROVIDER_ID: probe_gemini_cli_readiness(),
        META_SPARK_PROVIDER_ID: probe_meta_spark_readiness(),
        MISTRAL_PROVIDER_ID: probe_mistral_cli_readiness(),
        COPILOT_PROVIDER_ID: probe_copilot_cli_readiness(),
    }


__all__ = [
    "CLAUDE_PROVIDER_ID",
    "CLI_PROVIDER_BALANCE_SCHEMA",
    "COPILOT_PROVIDER_ID",
    "CliQuotaClassification",
    "GEMINI_PROVIDER_ID",
    "META_SPARK_PROVIDER_ID",
    "MISTRAL_PROVIDER_ID",
    "SECONDARY_IMPLEMENTATION_PREFERENCE",
    "classify_claude_cli_text",
    "classify_cli_provider_text",
    "classify_copilot_cli_text",
    "classify_gemini_cli_text",
    "classify_meta_spark_cli_text",
    "classify_mistral_cli_text",
    "claude_cli_auth_available",
    "copilot_cli_auth_available",
    "gemini_cli_auth_available",
    "meta_spark_auth_available",
    "mistral_cli_auth_available",
    "parse_cli_balance_observation",
    "probe_all_cli_provider_readiness",
    "probe_claude_and_gemini_readiness",
    "probe_claude_cli_readiness",
    "probe_copilot_cli_readiness",
    "probe_gemini_cli_readiness",
    "probe_meta_spark_readiness",
    "probe_mistral_cli_readiness",
    "resolve_claude_cli_binary",
    "resolve_copilot_cli_binary",
    "resolve_gemini_cli_binary",
    "resolve_goose_cli_binary",
    "resolve_mistral_cli_binary",
]
