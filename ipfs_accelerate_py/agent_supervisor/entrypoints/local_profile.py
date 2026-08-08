"""Explicit, host-local authority for prompt-only isolated worktrees.

This is deliberately a small persistence boundary.  A prompt can *use* an
installed profile, but it cannot create, replace, widen, or select one.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ALLOWED_LOCAL_CAPABILITIES = frozenset({"read", "edit", "test", "isolated_worktree", "write_worktree"})
DENIED_LOCAL_CAPABILITIES = frozenset({"current_checkout_rewrite", "merge", "push", "deploy", "destructive_cleanup", "arbitrary_secrets", "arbitrary_network", "secrets", "network"})
PROFILE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/local-dev-profile@2"
PROFILE_FILENAME = "local_dev_profile.json"
SIGNATURE_FILENAME = "local_dev_profile.sig"
KEY_FILENAME = "local_dev_profile.key"
REVOKE_MARKER = "local_dev_profile.revoked"
DEFAULT_PROFILE_DIR_ENV = "AGENT_SUPERVISOR_LOCAL_PROFILE_DIR"
SIGNING_KEY_ENV = "AGENT_SUPERVISOR_LOCAL_PROFILE_KEY"

class LocalProfileError(ValueError): pass
class LocalProfileDenied(LocalProfileError): pass
class LocalProfileTampered(LocalProfileDenied): pass
class LocalProfileRevoked(LocalProfileDenied): pass
class LocalProfilePermissive(LocalProfileDenied): pass
class LocalProfileWrongRepository(LocalProfileDenied): pass
class LocalProfilePromptDerived(LocalProfileDenied): pass

def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()

def _path(directory: Path | None) -> Path:
    if directory is not None: return Path(directory)
    return Path(os.environ.get(DEFAULT_PROFILE_DIR_ENV, Path.home() / ".ipfs_accelerate" / "agent_supervisor" / "local_profile"))

def _atomic_write(path: Path, data: bytes, mode: int) -> None:
    temporary = path.with_name("." + path.name + ".tmp-" + secrets.token_hex(8))
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        os.chmod(path, mode)
    finally:
        if temporary.exists(): temporary.unlink()

def _check_private(path: Path) -> None:
    try: mode = stat.S_IMODE(path.stat().st_mode)
    except OSError as exc: raise LocalProfileTampered("local signing identity is missing") from exc
    if mode & 0o077: raise LocalProfileTampered("local signing identity must be mode 0600")

def _key(directory: Path, supplied: bytes | None, *, create: bool) -> bytes:
    if supplied is not None:
        if not isinstance(supplied, bytes) or not supplied: raise LocalProfileError("signing_key must be non-empty bytes")
        return supplied
    env = os.environ.get(SIGNING_KEY_ENV)
    if env: return env.encode()
    path = directory / KEY_FILENAME
    if path.exists():
        _check_private(path); return path.read_bytes()
    if not create: raise LocalProfileTampered("local signing identity is missing")
    material = secrets.token_bytes(32); _atomic_write(path, material, 0o600); return material

def _caps(values: Sequence[str]) -> frozenset[str]:
    caps = frozenset(values)
    if not caps or caps & DENIED_LOCAL_CAPABILITIES or caps - ALLOWED_LOCAL_CAPABILITIES:
        raise LocalProfilePermissive("profile capabilities are not the bounded local allowlist")
    return caps

@dataclass(frozen=True)
class SignedSupervisorProfile:
    schema: str; repository_cid: str; baseline_commit: str; capabilities: frozenset[str]; created_at: float; profile_id: str; revoked: bool = False
    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "repository_cid": self.repository_cid, "baseline_commit": self.baseline_commit, "capabilities": sorted(self.capabilities), "created_at": self.created_at, "profile_id": self.profile_id, "revoked": self.revoked}
    def allows(self, capability: str) -> bool: return capability in self.capabilities and capability not in DENIED_LOCAL_CAPABILITIES
    @property
    def content_id(self) -> str: return "sha256:" + hashlib.sha256(_canonical(self.to_dict())).hexdigest()

LocalDevProfile = SignedSupervisorProfile

@dataclass(frozen=True)
class ProfileRotationReceipt:
    old_profile_id: str; new_profile_id: str; repository_cid: str; rotated_at: float
    @property
    def content_id(self) -> str:
        return "sha256:" + hashlib.sha256(_canonical(self.__dict__)).hexdigest()

def _reject_prompt(source: str | None, payload: Mapping[str, Any] | None = None) -> None:
    if source and source.strip().casefold() in {"prompt", "prompt_text", "user_prompt", "chat", "message"}: raise LocalProfilePromptDerived("prompt text cannot supply local authority")
    if payload and any(key in payload for key in ("prompt", "prompt_text", "user_message", "from_prompt")): raise LocalProfilePromptDerived("prompt-derived profile rejected")

def initialize_local_profile(*, repository_cid: str, baseline_commit: str, capabilities: Sequence[str] | None = None, profile_dir: Path | None = None, signing_key: bytes | None = None, force: bool = False) -> SignedSupervisorProfile:
    if not isinstance(repository_cid, str) or not repository_cid.strip() or not isinstance(baseline_commit, str) or not baseline_commit.strip(): raise LocalProfileError("repository_cid and baseline_commit are required")
    directory = _path(profile_dir); directory.mkdir(mode=0o700, parents=True, exist_ok=True); os.chmod(directory, 0o700)
    profile_file = directory / PROFILE_FILENAME
    if profile_file.exists() and not force and not (directory / REVOKE_MARKER).exists(): return load_local_profile(repository_cid=repository_cid, profile_dir=directory, signing_key=signing_key)
    caps = _caps(capabilities or tuple(ALLOWED_LOCAL_CAPABILITIES)); now = time.time()
    profile = SignedSupervisorProfile(PROFILE_SCHEMA, repository_cid, baseline_commit, caps, now, hashlib.sha256(f"{repository_cid}\0{baseline_commit}\0{now}\0{secrets.token_hex(16)}".encode()).hexdigest()[:32])
    data = profile.to_dict(); signature = hmac.new(_key(directory, signing_key, create=True), _canonical(data), hashlib.sha256).hexdigest().encode() + b"\n"
    _atomic_write(profile_file, _canonical(data) + b"\n", 0o600); _atomic_write(directory / SIGNATURE_FILENAME, signature, 0o600)
    (directory / REVOKE_MARKER).unlink(missing_ok=True)
    return profile

def load_local_profile(*, repository_cid: str, profile_dir: Path | None = None, signing_key: bytes | None = None, source: str | None = None, prompt_payload: Mapping[str, Any] | None = None) -> SignedSupervisorProfile:
    _reject_prompt(source, prompt_payload); directory = _path(profile_dir)
    if (directory / REVOKE_MARKER).exists(): raise LocalProfileRevoked("local profile is revoked")
    try:
        raw = json.loads((directory / PROFILE_FILENAME).read_text()); signature = (directory / SIGNATURE_FILENAME).read_text().strip()
    except (OSError, json.JSONDecodeError) as exc: raise LocalProfileTampered("signed local profile is missing or unreadable") from exc
    if not isinstance(raw, dict): raise LocalProfileTampered("profile must be an object")
    _reject_prompt(source, raw)
    expected = hmac.new(_key(directory, signing_key, create=False), _canonical(raw), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature): raise LocalProfileTampered("profile signature invalid")
    if raw.get("schema") not in {PROFILE_SCHEMA, "ipfs_accelerate_py/agent-supervisor/local-dev-profile@1"}: raise LocalProfileTampered("unsupported profile schema")
    if raw.get("repository_cid") != repository_cid: raise LocalProfileWrongRepository("profile is bound to a different repository")
    if raw.get("revoked") is True: raise LocalProfileRevoked("profile is revoked")
    try: return SignedSupervisorProfile(PROFILE_SCHEMA, repository_cid, str(raw["baseline_commit"]), _caps(raw["capabilities"]), float(raw["created_at"]), str(raw["profile_id"]))
    except (KeyError, TypeError, ValueError) as exc: raise LocalProfileTampered("profile fields invalid") from exc

def revoke_local_profile(*, profile_dir: Path | None = None) -> None:
    directory = _path(profile_dir); directory.mkdir(mode=0o700, parents=True, exist_ok=True); _atomic_write(directory / REVOKE_MARKER, b"revoked\n", 0o600)

def rotate_local_profile(**kwargs: Any) -> ProfileRotationReceipt:
    directory = _path(kwargs.get("profile_dir")); old = load_local_profile(repository_cid=kwargs["repository_cid"], profile_dir=directory, signing_key=kwargs.get("signing_key")); new = initialize_local_profile(**{**kwargs, "force": True}); return ProfileRotationReceipt(old.profile_id, new.profile_id, new.repository_cid, time.time())

class LocalProfileInitializer:
    initialize = staticmethod(initialize_local_profile); load = staticmethod(load_local_profile); verify = staticmethod(load_local_profile); revoke = staticmethod(revoke_local_profile); rotate = staticmethod(rotate_local_profile)

def assert_capability_allowed(profile: SignedSupervisorProfile, capability: str) -> None:
    if not profile.allows(capability): raise LocalProfileDenied(f"capability {capability!r} is denied by local profile")

def local_profile_authority_view(profile: SignedSupervisorProfile) -> dict[str, Any]:
    return {"kind":"local_dev_profile", "profile_id":profile.profile_id, "repository_cid":profile.repository_cid, "baseline_commit":profile.baseline_commit, "capabilities":sorted(profile.capabilities), "denied":sorted(DENIED_LOCAL_CAPABILITIES), "completion_authoritative":False, "proof_authoritative":False, "repository_write_allowed":False, "isolated_worktree_only":True, "current_checkout_rewrite":False}

def inspect_local_profile(*, repository_cid: str, profile_dir: Path | None = None, signing_key: bytes | None = None) -> dict[str, Any]:
    """Return the bounded authority projection after signature verification."""
    return local_profile_authority_view(load_local_profile(repository_cid=repository_cid, profile_dir=profile_dir, signing_key=signing_key))
