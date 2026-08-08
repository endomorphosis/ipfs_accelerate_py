"""Fail-closed, signed local authority profiles.

The local profile is an effect capability, not configuration.  Its Ed25519
identity is deliberately an owned regular 0600 file and profile lifecycle
records are signed by that identity so a copied pre-revocation key cannot be
silently presented as current authority.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding, NoEncryption, PrivateFormat, PublicFormat,
)

ALLOWED_LOCAL_CAPABILITIES = frozenset({"read", "edit", "test", "isolated_worktree", "write_worktree"})
DENIED_LOCAL_CAPABILITIES = frozenset({"current_checkout_rewrite", "merge", "push", "deploy", "destructive_cleanup", "arbitrary_secrets", "arbitrary_network", "secrets", "network"})
PROFILE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/local-dev-profile@3"
PROFILE_FILENAME = "local_dev_profile.json"
SIGNATURE_FILENAME = "local_dev_profile.sig"
KEY_FILENAME = "local_dev_profile.key"
REVOKE_MARKER = "local_dev_profile.revoked"
LIFECYCLE_FILENAME = "local_dev_profile.lifecycle.jsonl"
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
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()

def _path(directory: Path | None) -> Path:
    return Path(directory) if directory is not None else Path(os.environ.get(DEFAULT_PROFILE_DIR_ENV, Path.home() / ".ipfs_accelerate" / "agent_supervisor" / "local_profile"))

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
        try: temporary.unlink()
        except FileNotFoundError: pass

def _owned_regular(path: Path, *, private: bool = True) -> os.stat_result:
    try: metadata = path.lstat()
    except OSError as exc: raise LocalProfileTampered("local authority artifact is missing") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise LocalProfileTampered("local authority artifact must be an owned regular nonsymlink")
    if metadata.st_uid != os.geteuid():
        raise LocalProfileTampered("local authority artifact has the wrong owner")
    if private and stat.S_IMODE(metadata.st_mode) != 0o600:
        raise LocalProfileTampered("local authority artifact must be mode 0600")
    return metadata

def _owned_directory(path: Path) -> None:
    try: metadata = path.lstat()
    except OSError as exc: raise LocalProfileTampered("local authority directory is missing") from exc
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise LocalProfileTampered("local authority directory must be an owned mode-0700 directory")

def _key(directory: Path, supplied: bytes | None, *, create: bool) -> Ed25519PrivateKey:
    if supplied is not None:
        material = supplied
    elif os.environ.get(SIGNING_KEY_ENV):
        material = os.environ[SIGNING_KEY_ENV].encode()
    else:
        path = directory / KEY_FILENAME
        if not path.exists():
            if not create: raise LocalProfileTampered("local signing identity is missing")
            private = Ed25519PrivateKey.generate()
            _atomic_write(path, private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()), 0o600)
            return private
        _owned_regular(path)
        material = path.read_bytes()
    if not isinstance(material, bytes) or len(material) != 32:
        raise LocalProfileTampered("local signing identity is invalid")
    try: return Ed25519PrivateKey.from_private_bytes(material)
    except ValueError as exc: raise LocalProfileTampered("local signing identity is invalid") from exc

def _did(key: Ed25519PrivateKey) -> str:
    public = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return "did:key:z" + base64.urlsafe_b64encode(public).decode().rstrip("=")

def _caps(values: Sequence[str]) -> frozenset[str]:
    caps = frozenset(values)
    if not caps or caps & DENIED_LOCAL_CAPABILITIES or caps - ALLOWED_LOCAL_CAPABILITIES:
        raise LocalProfilePermissive("profile capabilities are not the bounded local allowlist")
    return caps

@dataclass(frozen=True)
class SignedSupervisorProfile:
    schema: str; repository_cid: str; baseline_commit: str; capabilities: frozenset[str]; created_at: float; profile_id: str; identity_did: str = ""; revoked: bool = False
    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "repository_cid": self.repository_cid, "baseline_commit": self.baseline_commit, "capabilities": sorted(self.capabilities), "created_at": self.created_at, "profile_id": self.profile_id, "identity_did": self.identity_did, "revoked": self.revoked}
    def allows(self, capability: str) -> bool: return capability in self.capabilities and capability not in DENIED_LOCAL_CAPABILITIES
    @property
    def content_id(self) -> str: return "sha256:" + hashlib.sha256(_canonical(self.to_dict())).hexdigest()

LocalDevProfile = SignedSupervisorProfile

@dataclass(frozen=True)
class SignedProfileLifecycleReceipt:
    kind: str; profile_id: str; identity_did: str; repository_cid: str; recorded_at: float; signature: str
    @property
    def content_id(self) -> str: return "sha256:" + hashlib.sha256(_canonical(self.__dict__)).hexdigest()

@dataclass(frozen=True)
class AuthLifecycleFinding:
    """Signed lifecycle observation; evidence never creates prompt authority."""
    profile_id: str; identity_did: str; kind: str; recorded_at: float; content_id: str

@dataclass(frozen=True)
class ProfileRotationReceipt:
    old_profile_id: str; new_profile_id: str; repository_cid: str; rotated_at: float
    @property
    def content_id(self) -> str: return "sha256:" + hashlib.sha256(_canonical(self.__dict__)).hexdigest()

class SecureLocalIdentityStore:
    """Small explicit façade used by effect boundaries and tests."""
    initialize = staticmethod(lambda **kwargs: initialize_local_profile(**kwargs))
    load = staticmethod(lambda **kwargs: load_local_profile(**kwargs))
    revoke = staticmethod(lambda **kwargs: revoke_local_profile(**kwargs))

def _reject_prompt(source: str | None, payload: Mapping[str, Any] | None = None) -> None:
    if source and source.strip().casefold() in {"prompt", "prompt_text", "user_prompt", "chat", "message"}: raise LocalProfilePromptDerived("prompt text cannot supply local authority")
    if payload and any(key in payload for key in ("prompt", "prompt_text", "user_message", "from_prompt")): raise LocalProfilePromptDerived("prompt-derived profile rejected")

def _signature(key: Ed25519PrivateKey, payload: Mapping[str, Any]) -> str:
    return base64.b64encode(key.sign(_canonical(payload))).decode()

def _verify(key: Ed25519PrivateKey, payload: Mapping[str, Any], signature: str) -> None:
    try: key.public_key().verify(base64.b64decode(signature.encode(), validate=True), _canonical(payload))
    except Exception as exc: raise LocalProfileTampered("profile signature invalid") from exc

def _append_lifecycle(directory: Path, key: Ed25519PrivateKey, *, kind: str, profile: SignedSupervisorProfile) -> SignedProfileLifecycleReceipt:
    body = {"kind": kind, "profile_id": profile.profile_id, "identity_did": profile.identity_did, "repository_cid": profile.repository_cid, "recorded_at": time.time()}
    receipt = SignedProfileLifecycleReceipt(**body, signature=_signature(key, body))
    path = directory / LIFECYCLE_FILENAME
    previous = b""
    if path.exists(): _owned_regular(path); previous = path.read_bytes()
    _atomic_write(path, previous + _canonical(receipt.__dict__) + b"\n", 0o600)
    return receipt

def _revoked_in_lifecycle(directory: Path, key: Ed25519PrivateKey, profile: SignedSupervisorProfile) -> bool:
    path = directory / LIFECYCLE_FILENAME
    if not path.exists(): return False
    _owned_regular(path)
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            if not isinstance(raw, dict) or set(raw) != {"kind", "profile_id", "identity_did", "repository_cid", "recorded_at", "signature"}: raise ValueError
            signature = str(raw.pop("signature")); _verify(key, raw, signature)
            if raw["kind"] == "revoked" and raw["profile_id"] == profile.profile_id and raw["identity_did"] == profile.identity_did: return True
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LocalProfileTampered("profile lifecycle history is invalid") from exc
    return False

def initialize_local_profile(*, repository_cid: str, baseline_commit: str, capabilities: Sequence[str] | None = None, profile_dir: Path | None = None, signing_key: bytes | None = None, force: bool = False) -> SignedSupervisorProfile:
    if not isinstance(repository_cid, str) or not repository_cid.strip() or not isinstance(baseline_commit, str) or not baseline_commit.strip(): raise LocalProfileError("repository_cid and baseline_commit are required")
    directory = _path(profile_dir)
    if directory.exists():
        _owned_directory(directory)
    else:
        directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        _owned_directory(directory)
    _owned_regular(directory / PROFILE_FILENAME) if (directory / PROFILE_FILENAME).exists() else None
    if (directory / PROFILE_FILENAME).exists() and not force and not (directory / REVOKE_MARKER).exists(): return load_local_profile(repository_cid=repository_cid, profile_dir=directory, signing_key=signing_key)
    key = _key(directory, signing_key, create=True); now = time.time()
    profile = SignedSupervisorProfile(PROFILE_SCHEMA, repository_cid, baseline_commit, _caps(capabilities or tuple(ALLOWED_LOCAL_CAPABILITIES)), now, hashlib.sha256(f"{repository_cid}\0{baseline_commit}\0{now}\0{secrets.token_hex(16)}".encode()).hexdigest()[:32], _did(key))
    _atomic_write(directory / PROFILE_FILENAME, _canonical(profile.to_dict()) + b"\n", 0o600)
    _atomic_write(directory / SIGNATURE_FILENAME, (_signature(key, profile.to_dict()) + "\n").encode(), 0o600)
    try: (directory / REVOKE_MARKER).unlink()
    except FileNotFoundError: pass
    _append_lifecycle(directory, key, kind="rotated" if force else "initialized", profile=profile)
    return profile

def load_local_profile(*, repository_cid: str, profile_dir: Path | None = None, signing_key: bytes | None = None, source: str | None = None, prompt_payload: Mapping[str, Any] | None = None) -> SignedSupervisorProfile:
    _reject_prompt(source, prompt_payload); directory = _path(profile_dir); _owned_directory(directory)
    if (directory / REVOKE_MARKER).exists():
        _owned_regular(directory / REVOKE_MARKER); raise LocalProfileRevoked("local profile is revoked")
    try:
        _owned_regular(directory / PROFILE_FILENAME); _owned_regular(directory / SIGNATURE_FILENAME)
        raw = json.loads((directory / PROFILE_FILENAME).read_text()); signature = (directory / SIGNATURE_FILENAME).read_text().strip()
    except (OSError, json.JSONDecodeError) as exc: raise LocalProfileTampered("signed local profile is missing or unreadable") from exc
    if not isinstance(raw, dict): raise LocalProfileTampered("profile must be an object")
    _reject_prompt(source, raw); key = _key(directory, signing_key, create=False); _verify(key, raw, signature)
    if raw.get("schema") not in {PROFILE_SCHEMA, "ipfs_accelerate_py/agent-supervisor/local-dev-profile@2", "ipfs_accelerate_py/agent-supervisor/local-dev-profile@1"}: raise LocalProfileTampered("unsupported profile schema")
    if raw.get("repository_cid") != repository_cid: raise LocalProfileWrongRepository("profile is bound to a different repository")
    if raw.get("revoked") is True: raise LocalProfileRevoked("profile is revoked")
    try:
        profile = SignedSupervisorProfile(PROFILE_SCHEMA, repository_cid, str(raw["baseline_commit"]), _caps(raw["capabilities"]), float(raw["created_at"]), str(raw["profile_id"]), str(raw.get("identity_did") or _did(key)))
    except (KeyError, TypeError, ValueError) as exc: raise LocalProfileTampered("profile fields invalid") from exc
    if profile.identity_did != _did(key): raise LocalProfileTampered("profile identity does not match signing key")
    if _revoked_in_lifecycle(directory, key, profile): raise LocalProfileRevoked("local profile is revoked")
    return profile

def revoke_local_profile(*, profile_dir: Path | None = None) -> None:
    directory = _path(profile_dir); profile = load_local_profile(repository_cid=json.loads((directory / PROFILE_FILENAME).read_text())["repository_cid"], profile_dir=directory)
    key = _key(directory, None, create=False); _append_lifecycle(directory, key, kind="revoked", profile=profile); _atomic_write(directory / REVOKE_MARKER, b"revoked\n", 0o600)

def rotate_local_profile(**kwargs: Any) -> ProfileRotationReceipt:
    directory = _path(kwargs.get("profile_dir")); old = load_local_profile(repository_cid=kwargs["repository_cid"], profile_dir=directory, signing_key=kwargs.get("signing_key")); new = initialize_local_profile(**{**kwargs, "force": True}); return ProfileRotationReceipt(old.profile_id, new.profile_id, new.repository_cid, time.time())

class LocalProfileInitializer:
    initialize = staticmethod(initialize_local_profile); load = staticmethod(load_local_profile); verify = staticmethod(load_local_profile); revoke = staticmethod(revoke_local_profile); rotate = staticmethod(rotate_local_profile)

def assert_capability_allowed(profile: SignedSupervisorProfile, capability: str) -> None:
    if not profile.allows(capability): raise LocalProfileDenied(f"capability {capability!r} is denied by local profile")

def local_profile_authority_view(profile: SignedSupervisorProfile) -> dict[str, Any]:
    return {"kind":"local_dev_profile", "profile_id":profile.profile_id, "identity_did":profile.identity_did, "repository_cid":profile.repository_cid, "baseline_commit":profile.baseline_commit, "capabilities":sorted(profile.capabilities), "denied":sorted(DENIED_LOCAL_CAPABILITIES), "completion_authoritative":False, "proof_authoritative":False, "repository_write_allowed":False, "isolated_worktree_only":True, "current_checkout_rewrite":False}

def inspect_local_profile(*, repository_cid: str, profile_dir: Path | None = None, signing_key: bytes | None = None) -> dict[str, Any]:
    return local_profile_authority_view(load_local_profile(repository_cid=repository_cid, profile_dir=profile_dir, signing_key=signing_key))
