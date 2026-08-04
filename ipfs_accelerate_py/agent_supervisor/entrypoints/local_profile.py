"""One-time signed local-development profile initialization.

Creates a host-local signed profile that subsequent prompt-only
isolated-worktree edit/test runs can load. Profiles that are unsigned,
tampered, permissive, bound to the wrong repository, revoked, or
derived from prompt text fail closed. Merge, push, deploy, destructive
cleanup, arbitrary secrets/network, and current-checkout rewrite stay
denied under this profile.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Set

# Capabilities allowed for local-dev isolated worktree edit/test runs.
ALLOWED_LOCAL_CAPABILITIES: frozenset[str] = frozenset(
    {
        "edit",
        "test",
        "isolated_worktree",
        "read",
        "write_worktree",
    }
)

# Explicitly denied under the local profile (fail closed).
DENIED_LOCAL_CAPABILITIES: frozenset[str] = frozenset(
    {
        "merge",
        "push",
        "deploy",
        "destructive_cleanup",
        "arbitrary_secrets",
        "arbitrary_network",
        "current_checkout_rewrite",
        "secrets",
        "network",
    }
)

PROFILE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/local-dev-profile@1"
PROFILE_FILENAME = "local_dev_profile.json"
SIGNATURE_FILENAME = "local_dev_profile.sig"
REVOKE_MARKER = "local_dev_profile.revoked"
DEFAULT_PROFILE_DIR_ENV = "AGENT_SUPERVISOR_LOCAL_PROFILE_DIR"
SIGNING_KEY_ENV = "AGENT_SUPERVISOR_LOCAL_PROFILE_KEY"


class LocalProfileError(Exception):
    """Base error for local profile operations."""


class LocalProfileDenied(LocalProfileError):
    """Raised when a profile must fail closed."""


class LocalProfileTampered(LocalProfileDenied):
    """Signature missing, invalid, or content does not match."""


class LocalProfileRevoked(LocalProfileDenied):
    """Profile was explicitly revoked."""


class LocalProfilePermissive(LocalProfileDenied):
    """Profile grants capabilities outside the local-dev allow-list."""


class LocalProfileWrongRepository(LocalProfileDenied):
    """Profile is bound to a different repository."""


class LocalProfilePromptDerived(LocalProfileDenied):
    """Profile appears to have been supplied or derived from prompt text."""


@dataclass(frozen=True)
class LocalDevProfile:
    """Validated, signed local-development authority profile."""

    schema: str
    repository_cid: str
    baseline_commit: str
    capabilities: frozenset[str]
    created_at: float
    profile_id: str
    revoked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "repository_cid": self.repository_cid,
            "baseline_commit": self.baseline_commit,
            "capabilities": sorted(self.capabilities),
            "created_at": self.created_at,
            "profile_id": self.profile_id,
            "revoked": self.revoked,
        }

    def allows(self, capability: str) -> bool:
        if capability in DENIED_LOCAL_CAPABILITIES:
            return False
        return capability in self.capabilities


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _sign(payload: Mapping[str, Any], key: bytes) -> str:
    digest = hmac.new(key, _canonical_bytes(payload), hashlib.sha256).hexdigest()
    return digest


def _verify_signature(payload: Mapping[str, Any], signature: str, key: bytes) -> bool:
    if not signature or not isinstance(signature, str):
        return False
    expected = _sign(payload, key)
    return hmac.compare_digest(expected, signature)


def _default_signing_key() -> bytes:
    env_key = os.environ.get(SIGNING_KEY_ENV)
    if env_key:
        return env_key.encode("utf-8")
    # Host-local fallback: not for production authority; local-dev only.
    host = os.environ.get("HOSTNAME", "local")
    user = os.environ.get("USER", "user")
    material = f"agent-supervisor-local-dev:{host}:{user}".encode("utf-8")
    return hashlib.sha256(material).digest()


def _profile_dir(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(DEFAULT_PROFILE_DIR_ENV)
    if env:
        return Path(env)
    return Path.home() / ".ipfs_accelerate" / "agent_supervisor" / "local_profile"


def _assert_not_prompt_derived(source: Optional[str], payload: Mapping[str, Any]) -> None:
    """Fail closed if the profile was injected via prompt text."""
    if source is None:
        return
    lowered = source.strip().lower()
    if lowered in {"prompt", "prompt_text", "user_prompt", "chat", "message"}:
        raise LocalProfilePromptDerived(
            "local-dev profiles must not be derived from prompt text"
        )
    # Detect payload fields that look like free-form prompt smuggling.
    for key in ("prompt", "prompt_text", "user_message", "from_prompt"):
        if key in payload:
            raise LocalProfilePromptDerived(
                f"local-dev profile must not contain prompt-derived field {key!r}"
            )


def _validate_capabilities(capabilities: Sequence[str]) -> frozenset[str]:
    caps = frozenset(capabilities)
    if not caps:
        raise LocalProfilePermissive("local-dev profile must declare at least one capability")
    denied = caps & DENIED_LOCAL_CAPABILITIES
    if denied:
        raise LocalProfilePermissive(
            f"local-dev profile grants denied capabilities: {sorted(denied)}"
        )
    unknown = caps - ALLOWED_LOCAL_CAPABILITIES
    if unknown:
        raise LocalProfilePermissive(
            f"local-dev profile grants non-allow-listed capabilities: {sorted(unknown)}"
        )
    return caps


def initialize_local_profile(
    *,
    repository_cid: str,
    baseline_commit: str,
    capabilities: Optional[Sequence[str]] = None,
    profile_dir: Optional[Path] = None,
    signing_key: Optional[bytes] = None,
    force: bool = False,
) -> LocalDevProfile:
    """One-time explicit setup for a signed local-development profile.

    Subsequent prompt-only isolated-worktree edit/test runs load this
    profile; they do not recreate it. Re-initialization requires force=True
    or prior revocation.
    """
    if not repository_cid or not isinstance(repository_cid, str):
        raise LocalProfileError("repository_cid is required")
    if not baseline_commit or not isinstance(baseline_commit, str):
        raise LocalProfileError("baseline_commit is required")

    caps = _validate_capabilities(
        list(capabilities)
        if capabilities is not None
        else ["edit", "test", "isolated_worktree", "read", "write_worktree"]
    )

    directory = _profile_dir(profile_dir)
    directory.mkdir(parents=True, exist_ok=True)
    profile_path = directory / PROFILE_FILENAME
    sig_path = directory / SIGNATURE_FILENAME
    revoke_path = directory / REVOKE_MARKER

    if profile_path.exists() and not force:
        if revoke_path.exists():
            pass  # allow re-init after revoke
        else:
            # Load existing rather than silently rewriting.
            existing = load_local_profile(
                repository_cid=repository_cid,
                profile_dir=directory,
                signing_key=signing_key,
            )
            return existing

    created_at = time.time()
    profile_id = hashlib.sha256(
        f"{repository_cid}:{baseline_commit}:{created_at}".encode("utf-8")
    ).hexdigest()[:32]

    profile = LocalDevProfile(
        schema=PROFILE_SCHEMA,
        repository_cid=repository_cid,
        baseline_commit=baseline_commit,
        capabilities=caps,
        created_at=created_at,
        profile_id=profile_id,
        revoked=False,
    )
    payload = profile.to_dict()
    key = signing_key if signing_key is not None else _default_signing_key()
    signature = _sign(payload, key)

    profile_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sig_path.write_text(signature + "\n", encoding="utf-8")
    if revoke_path.exists():
        revoke_path.unlink()

    return profile


def revoke_local_profile(*, profile_dir: Optional[Path] = None) -> None:
    """Revoke the on-disk local profile; subsequent loads fail closed."""
    directory = _profile_dir(profile_dir)
    revoke_path = directory / REVOKE_MARKER
    directory.mkdir(parents=True, exist_ok=True)
    revoke_path.write_text("revoked\n", encoding="utf-8")


def load_local_profile(
    *,
    repository_cid: str,
    profile_dir: Optional[Path] = None,
    signing_key: Optional[bytes] = None,
    source: Optional[str] = None,
    prompt_payload: Optional[Mapping[str, Any]] = None,
) -> LocalDevProfile:
    """Load and verify a signed local-development profile.

    Fails closed on: missing/unsigned, tampered, permissive, wrong
    repository, revoked, or prompt-derived profiles.
    """
    if prompt_payload is not None:
        raise LocalProfilePromptDerived(
            "local-dev profiles must not be supplied via prompt payload"
        )
    _assert_not_prompt_derived(source, {})

    directory = _profile_dir(profile_dir)
    profile_path = directory / PROFILE_FILENAME
    sig_path = directory / SIGNATURE_FILENAME
    revoke_path = directory / REVOKE_MARKER

    if revoke_path.exists():
        raise LocalProfileRevoked("local-dev profile has been revoked")

    if not profile_path.exists():
        raise LocalProfileTampered("local-dev profile is missing; run initialize_local_profile")

    if not sig_path.exists():
        raise LocalProfileTampered("local-dev profile signature is missing (unsigned)")

    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalProfileTampered(f"local-dev profile unreadable: {exc}") from exc

    if not isinstance(raw, dict):
        raise LocalProfileTampered("local-dev profile must be a JSON object")

    _assert_not_prompt_derived(source, raw)

    signature = sig_path.read_text(encoding="utf-8").strip()
    key = signing_key if signing_key is not None else _default_signing_key()
    if not _verify_signature(raw, signature, key):
        raise LocalProfileTampered("local-dev profile signature invalid or content tampered")

    schema = raw.get("schema")
    if schema != PROFILE_SCHEMA:
        raise LocalProfileTampered(f"unsupported local-dev profile schema: {schema!r}")

    bound_repo = raw.get("repository_cid")
    if not bound_repo or bound_repo != repository_cid:
        raise LocalProfileWrongRepository(
            f"profile bound to {bound_repo!r}, expected {repository_cid!r}"
        )

    if raw.get("revoked") is True:
        raise LocalProfileRevoked("local-dev profile marked revoked in payload")

    caps_raw = raw.get("capabilities")
    if not isinstance(caps_raw, list):
        raise LocalProfileTampered("capabilities must be a list")
    caps = _validate_capabilities([str(c) for c in caps_raw])

    baseline = raw.get("baseline_commit")
    if not baseline or not isinstance(baseline, str):
        raise LocalProfileTampered("baseline_commit missing")

    profile_id = raw.get("profile_id")
    if not profile_id or not isinstance(profile_id, str):
        raise LocalProfileTampered("profile_id missing")

    created_at = raw.get("created_at")
    if not isinstance(created_at, (int, float)):
        raise LocalProfileTampered("created_at missing")

    return LocalDevProfile(
        schema=PROFILE_SCHEMA,
        repository_cid=str(bound_repo),
        baseline_commit=baseline,
        capabilities=caps,
        created_at=float(created_at),
        profile_id=profile_id,
        revoked=False,
    )


def assert_capability_allowed(profile: LocalDevProfile, capability: str) -> None:
    """Fail closed if capability is denied or not granted."""
    if capability in DENIED_LOCAL_CAPABILITIES:
        raise LocalProfileDenied(
            f"capability {capability!r} is permanently denied under local-dev profile"
        )
    if not profile.allows(capability):
        raise LocalProfileDenied(f"capability {capability!r} is not granted by local-dev profile")


def local_profile_authority_view(profile: LocalDevProfile) -> dict[str, Any]:
    """Compact authority view for resolvers: edit/test worktree only."""
    return {
        "kind": "local_dev_profile",
        "profile_id": profile.profile_id,
        "repository_cid": profile.repository_cid,
        "baseline_commit": profile.baseline_commit,
        "capabilities": sorted(profile.capabilities),
        "denied": sorted(DENIED_LOCAL_CAPABILITIES),
        "completion_authoritative": False,
        "proof_authoritative": False,
        "repository_write_allowed": False,
        "isolated_worktree_only": True,
        "current_checkout_rewrite": False,
    }
