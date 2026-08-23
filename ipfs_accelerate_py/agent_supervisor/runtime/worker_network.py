"""Fail-closed network profiles for external-agent task containers.

Task containers never receive a routable Docker network.  They attach to one
pre-created ``--internal`` network and address a CONNECT proxy by its literal
RFC1918 address.  The proxy, rather than the worker, owns DNS and enforces the
provider hostname allowlist.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import stat
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    LocalProfileTampered,
    verify_did_key_signature,
)

WORKER_NETWORK_PROFILE_SCHEMA = "ipfs_accelerate_py/eaaef-worker-network-profile@1"
WORKER_NETWORK_AUTHORIZATION_SCHEMA = (
    "ipfs_accelerate_py/eaaef-worker-network-authorization@1"
)
PROVIDER_HOSTNAME_ALLOWLISTS: Mapping[str, tuple[str, ...]] = {
    "grok": ("api.x.ai",),
    # Codex may use either API-key transport or the ChatGPT Codex backend.
    "codex": ("api.openai.com", "chatgpt.com"),
}
_RESERVED_DOCKER_NETWORKS = frozenset({"bridge", "default", "host", "none"})
_NETWORK_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,62}")
_CONTAINER_NAME_RE = re.compile(r"ipfs-accelerate-(?:grok|codex)-[0-9]+-[0-9a-f]{32}")
_LEASE_ID_RE = re.compile(r"asref-(?:grok|codex)-container-[a-z0-9_-]+")
_APPROVAL_IDENTITY_RE = re.compile(r"eaaef-network-approval:[A-Za-z0-9_.:-]{1,128}")
_CID_RE = re.compile(r"sha256:[0-9a-f]{64}")
_DOCKER_ID_RE = re.compile(r"[0-9a-f]{64}")
_NONCE_RE = re.compile(r"[A-Za-z0-9_.:-]{16,160}")
_MAX_AUTHORIZATION_BYTES = 64 * 1024
_MAX_AUTHORIZATION_LIFETIME_MS = 15 * 60 * 1000
_MAX_CLOCK_SKEW_MS = 30 * 1000
_PROXY_VARIABLES = frozenset(
    {
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
)
_RFC1918_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)


def _canonical_cid(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def derived_worker_network_name(worktree_id: str) -> str:
    """Return the only network name admitted for one signed worktree."""

    value = str(worktree_id).strip()
    if not value:
        raise ValueError("worker network worktree identity is missing")
    return "eaaef-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def worker_network_authorization_relative_path(
    invocation_id: str,
    provider: str,
) -> Path:
    invocation_digest = hashlib.sha256(
        str(invocation_id).encode("utf-8")
    ).hexdigest()
    normalized_provider = str(provider).strip().lower()
    if normalized_provider not in PROVIDER_HOSTNAME_ALLOWLISTS:
        raise ValueError("worker network provider is not configured")
    return Path("network-authorizations") / invocation_digest / (
        normalized_provider + ".json"
    )


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate worker network authorization key: {key}")
        result[key] = value
    return result


def _stable_private_json(
    path: Path,
    *,
    trusted_root: Path,
) -> tuple[dict[str, Any], str]:
    """Read one owner-only regular file without following a pathname swap."""

    candidate = path.expanduser()
    root = trusted_root.expanduser()
    if (
        not candidate.is_absolute()
        or not root.is_absolute()
        or ".." in candidate.parts
        or ".." in root.parts
    ):
        raise ValueError("worker network authorization path is not absolute")
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("worker network authorization escapes its profile root") from exc
    if not relative.parts or relative.name in {"", ".", ".."}:
        raise ValueError("worker network authorization path is invalid")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ValueError("worker network authorization no-follow is unavailable")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | nofollow
    )
    directory_descriptors: list[int] = []
    directory_records: list[tuple[int, os.stat_result, int | None, str]] = []
    directory_descriptor = os.open(root, directory_flags)
    directory_descriptors.append(directory_descriptor)
    try:
        root_metadata = os.fstat(directory_descriptor)
        directory_records.append((directory_descriptor, root_metadata, None, ""))
        for component in relative.parts[:-1]:
            child = os.open(
                component,
                directory_flags,
                dir_fd=directory_descriptor,
            )
            directory_descriptors.append(child)
            child_metadata = os.fstat(child)
            directory_records.append(
                (child, child_metadata, directory_descriptor, component)
            )
            directory_descriptor = child
        descriptor = os.open(
            relative.name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow,
            dir_fd=directory_descriptor,
        )
    except BaseException:
        for opened in reversed(directory_descriptors):
            os.close(opened)
        raise
    try:
        for _opened, metadata, _parent, _component in directory_records:
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink < 1
                or stat.S_IMODE(metadata.st_mode) & 0o022
            ):
                raise ValueError(
                    "worker network authorization parent is not trusted"
                )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_AUTHORIZATION_BYTES
            or stat.S_IMODE(before.st_mode) & 0o077
        ):
            raise ValueError("worker network authorization is not private")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        pathname = os.stat(
            relative.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        directory_after = [
            (record, os.fstat(opened))
            for opened, *record in directory_records
        ]
        root_pathname = os.stat(root, follow_symlinks=False)
        child_pathnames = [
            (metadata, os.stat(component, dir_fd=parent, follow_symlinks=False))
            for _opened, metadata, parent, component in directory_records[1:]
            if parent is not None
        ]
    finally:
        os.close(descriptor)
        for opened in reversed(directory_descriptors):
            os.close(opened)

    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_uid,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    raw = b"".join(chunks)
    def directory_identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_uid,
            item.st_nlink,
        )

    parents_stable = (
        directory_identity(directory_records[0][1])
        == directory_identity(root_pathname)
        and all(
            directory_identity(before_directory)
            == directory_identity(after_directory)
            for (
                before_directory,
                _parent,
                _component,
            ), after_directory in directory_after
        )
        and all(
            directory_identity(before_directory)
            == directory_identity(pathname_directory)
            for before_directory, pathname_directory in child_pathnames
        )
    )
    if (
        len(raw) != before.st_size
        or identity(before) != identity(after)
        or identity(before) != identity(pathname)
        or stat.S_ISLNK(pathname.st_mode)
        or not parents_stable
    ):
        raise ValueError("worker network authorization changed during read")
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("worker network authorization JSON is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("worker network authorization is not an object")
    return value, "sha256:" + hashlib.sha256(raw).hexdigest()


def _normalize_workspace(workspace: Path) -> str:
    resolved = workspace.expanduser().resolve(strict=False)
    if not resolved.is_absolute() or resolved == Path(resolved.anchor):
        raise ValueError("worker network workspace must be a scoped absolute path")
    return str(resolved)


def _validate_proxy_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or parsed.hostname is None
        or parsed.port is None
    ):
        raise ValueError("worker proxy endpoint must be an exact HTTP authority")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError as exc:
        raise ValueError("worker proxy endpoint must use a literal RFC1918 address") from exc
    if not isinstance(address, ipaddress.IPv4Address) or not any(
        address in network for network in _RFC1918_NETWORKS
    ):
        raise ValueError("worker proxy endpoint must use a literal RFC1918 address")
    if parsed.port < 1 or parsed.port > 65535:
        raise ValueError("worker proxy endpoint port is invalid")
    return f"http://{address.compressed}:{parsed.port}"


def _profile_approval_payload(
    *,
    provider: str,
    docker_network: str,
    proxy_endpoint: str,
    allowed_hostnames: Sequence[str],
    approval_identity: str,
    effect_cid: str,
    workspace: str,
    container_name: str,
    lease_id: str,
    lease_root: str,
) -> dict[str, object]:
    return {
        "schema": WORKER_NETWORK_PROFILE_SCHEMA,
        "provider": provider,
        "docker_network": docker_network,
        "docker_network_internal": True,
        "proxy_endpoint": proxy_endpoint,
        "allowed_hostnames": list(allowed_hostnames),
        "approval_identity": approval_identity,
        "effect_cid": effect_cid,
        "workspace": workspace,
        "container_name": container_name,
        "lease_id": lease_id,
        "lease_root": lease_root,
    }


def worker_network_approval_cid(
    *,
    provider: str,
    docker_network: str,
    proxy_endpoint: str,
    approval_identity: str,
    effect_cid: str,
    workspace: Path,
    container_name: str,
    lease_id: str,
    lease_root: Path,
) -> str:
    """Return the CID an approver must authorize for an exact worker effect."""

    normalized_provider = str(provider).strip().lower()
    allowed = PROVIDER_HOSTNAME_ALLOWLISTS.get(normalized_provider)
    if allowed is None:
        raise ValueError("worker network provider is not configured")
    normalized_proxy = _validate_proxy_endpoint(str(proxy_endpoint).strip())
    normalized_workspace = _normalize_workspace(workspace)
    normalized_lease_root = _normalize_workspace(lease_root)
    return _canonical_cid(
        _profile_approval_payload(
            provider=normalized_provider,
            docker_network=str(docker_network).strip(),
            proxy_endpoint=normalized_proxy,
            allowed_hostnames=allowed,
            approval_identity=str(approval_identity).strip(),
            effect_cid=str(effect_cid).strip(),
            workspace=normalized_workspace,
            container_name=str(container_name).strip(),
            lease_id=str(lease_id).strip(),
            lease_root=normalized_lease_root,
        )
    )


_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema",
        "authorization_id",
        "invocation_binding_id",
        "logical_attempt_id",
        "task_id",
        "worktree_id",
        "control_plane_capsule_id",
        "effect_cid",
        "provider",
        "route_id",
        "workspace",
        "container_name",
        "lease_id",
        "lease_root",
        "docker_network",
        "docker_network_id",
        "docker_network_internal",
        "proxy_endpoint",
        "proxy_container_id",
        "proxy_image_id",
        "allowed_hostnames",
        "issued_at_ms",
        "expires_at_ms",
        "one_use_nonce",
        "signer_did",
        "worker_principal_did",
        "provider_principal_did",
        "signature",
    }
)


@dataclass(frozen=True)
class WorkerNetworkAuthorization:
    """Independently signed authority for one exact provider attempt."""

    authorization_id: str
    invocation_binding_id: str
    logical_attempt_id: str
    task_id: str
    worktree_id: str
    control_plane_capsule_id: str
    effect_cid: str
    provider: str
    route_id: str
    workspace: Path
    container_name: str
    lease_id: str
    lease_root: Path
    docker_network: str
    docker_network_id: str
    docker_network_internal: bool
    proxy_endpoint: str
    proxy_container_id: str
    proxy_image_id: str
    allowed_hostnames: tuple[str, ...]
    issued_at_ms: int
    expires_at_ms: int
    one_use_nonce: str
    signer_did: str
    worker_principal_did: str
    provider_principal_did: str
    signature: str
    artifact_cid: str
    source_path: Path
    schema: str = WORKER_NETWORK_AUTHORIZATION_SCHEMA

    def signed_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "authorization_id": self.authorization_id,
            "invocation_binding_id": self.invocation_binding_id,
            "logical_attempt_id": self.logical_attempt_id,
            "task_id": self.task_id,
            "worktree_id": self.worktree_id,
            "control_plane_capsule_id": self.control_plane_capsule_id,
            "effect_cid": self.effect_cid,
            "provider": self.provider,
            "route_id": self.route_id,
            "workspace": str(self.workspace),
            "container_name": self.container_name,
            "lease_id": self.lease_id,
            "lease_root": str(self.lease_root),
            "docker_network": self.docker_network,
            "docker_network_id": self.docker_network_id,
            "docker_network_internal": self.docker_network_internal,
            "proxy_endpoint": self.proxy_endpoint,
            "proxy_container_id": self.proxy_container_id,
            "proxy_image_id": self.proxy_image_id,
            "allowed_hostnames": list(self.allowed_hostnames),
            "issued_at_ms": self.issued_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "one_use_nonce": self.one_use_nonce,
            "signer_did": self.signer_did,
            "worker_principal_did": self.worker_principal_did,
            "provider_principal_did": self.provider_principal_did,
        }


@dataclass(frozen=True)
class WorkerNetworkAuthorizationDecision:
    valid: bool
    blockers: tuple[str, ...]
    authorization_cid: str
    reviewer_did: str
    authority_mutated: bool = False
    process_started: bool = False
    authorization: WorkerNetworkAuthorization | None = None


def load_worker_network_authorization(
    *,
    invocation_binding: object,
    provider: str,
    workspace: Path,
    now_ms: int | None = None,
    expected_artifact_cid: str = "",
    expected_container_name: str = "",
    expected_lease_root: Path | None = None,
    expected_worker_principal_did: str = "",
    expected_provider_principal_did: str = "",
) -> WorkerNetworkAuthorization:
    """Stable-read and verify authority rooted in a verified invocation profile."""

    normalized_provider = str(provider).strip().lower()
    invocation_id = str(getattr(invocation_binding, "invocation_id", ""))
    profile_dir = Path(str(getattr(invocation_binding, "profile_dir", "")))
    relative = worker_network_authorization_relative_path(
        invocation_id, normalized_provider
    )
    path = profile_dir / relative
    value, artifact_cid = _stable_private_json(
        path,
        trusted_root=profile_dir,
    )
    if expected_artifact_cid and artifact_cid != expected_artifact_cid:
        raise ValueError("worker network authorization artifact CID drifted")
    if set(value) != _AUTHORIZATION_FIELDS:
        raise ValueError("worker network authorization fields are invalid")
    if value.get("schema") != WORKER_NETWORK_AUTHORIZATION_SCHEMA:
        raise ValueError("worker network authorization schema is invalid")
    hosts = value.get("allowed_hostnames")
    if not isinstance(hosts, list) or any(not isinstance(item, str) for item in hosts):
        raise ValueError("worker network authorization hosts are invalid")
    issued = value.get("issued_at_ms")
    expires = value.get("expires_at_ms")
    internal = value.get("docker_network_internal")
    if (
        isinstance(issued, bool)
        or not isinstance(issued, int)
        or isinstance(expires, bool)
        or not isinstance(expires, int)
        or type(internal) is not bool
    ):
        raise ValueError("worker network authorization types are invalid")
    source_path = Path(os.path.abspath(os.fspath(path)))
    candidate_workspace = Path(str(value.get("workspace") or ""))
    candidate_lease_root = Path(str(value.get("lease_root") or ""))
    authorization = WorkerNetworkAuthorization(
        **{name: str(value.get(name) or "") for name in (
            "authorization_id", "invocation_binding_id", "logical_attempt_id",
            "task_id", "worktree_id", "control_plane_capsule_id", "effect_cid",
            "provider", "route_id", "container_name", "lease_id",
            "docker_network", "docker_network_id", "proxy_endpoint",
            "proxy_container_id", "proxy_image_id", "one_use_nonce", "signer_did",
            "worker_principal_did", "provider_principal_did", "signature",
        )},
        workspace=Path(_normalize_workspace(candidate_workspace)),
        lease_root=Path(_normalize_workspace(candidate_lease_root)),
        docker_network_internal=internal,
        allowed_hostnames=tuple(hosts),
        issued_at_ms=issued,
        expires_at_ms=expires,
        artifact_cid=artifact_cid,
        source_path=source_path,
    )
    unsigned = authorization.signed_payload()
    claimed_id = str(unsigned.pop("authorization_id"))
    clock = int(now_ms if now_ms is not None else time.time() * 1000)
    expected_hosts = PROVIDER_HOSTNAME_ALLOWLISTS.get(normalized_provider)
    invocation_expires = int(getattr(invocation_binding, "expires_at_ms", 0))
    invocation_content_id = str(getattr(invocation_binding, "content_id", ""))
    reviewer_did = str(getattr(invocation_binding, "reviewer_identity", ""))
    profile_did = str(getattr(invocation_binding, "profile_identity_did", ""))
    control_plane = getattr(invocation_binding, "control_plane", None)
    capsule_id = str(getattr(control_plane, "capsule_id", ""))
    route_id = str(getattr(invocation_binding, "route_id", ""))
    provider_ids = {
        str(getattr(invocation_binding, "primary_provider_id", "")),
        str(getattr(invocation_binding, "fallback_provider_id", "")),
    }
    provider_matches_route = (
        (normalized_provider == "grok" and "grok_cli" in provider_ids)
        or (normalized_provider == "codex" and "codex" in provider_ids)
    )
    expected_lease = expected_lease_root.resolve(strict=False) if expected_lease_root else None
    if (
        not expected_worker_principal_did
        or not expected_provider_principal_did
        or claimed_id != _canonical_cid(unsigned)
        or authorization.provider != normalized_provider
        or not provider_matches_route
        or authorization.invocation_binding_id != invocation_content_id
        or authorization.logical_attempt_id
        != str(getattr(invocation_binding, "logical_attempt_id", ""))
        or authorization.task_id != str(getattr(invocation_binding, "task_id", ""))
        or authorization.worktree_id
        != str(getattr(invocation_binding, "worktree_id", ""))
        or authorization.control_plane_capsule_id != capsule_id
        or authorization.effect_cid != invocation_content_id
        or authorization.route_id != route_id
        or authorization.workspace != workspace.resolve(strict=False)
        or source_path.is_relative_to(workspace.resolve(strict=False))
        or _CONTAINER_NAME_RE.fullmatch(authorization.container_name) is None
        or _LEASE_ID_RE.fullmatch(authorization.lease_id) is None
        or authorization.docker_network
        != derived_worker_network_name(authorization.worktree_id)
        or not authorization.docker_network_internal
        or _NETWORK_NAME_RE.fullmatch(authorization.docker_network) is None
        or _DOCKER_ID_RE.fullmatch(authorization.docker_network_id) is None
        or _DOCKER_ID_RE.fullmatch(authorization.proxy_container_id) is None
        or _CID_RE.fullmatch(authorization.proxy_image_id) is None
        or authorization.allowed_hostnames != expected_hosts
        or _NONCE_RE.fullmatch(authorization.one_use_nonce) is None
        or authorization.signer_did != reviewer_did
        or authorization.signer_did != profile_did
        or not authorization.worker_principal_did.startswith("did:key:z")
        or not authorization.provider_principal_did.startswith("did:key:z")
        or authorization.signer_did
        in {authorization.worker_principal_did, authorization.provider_principal_did}
        or (
            expected_worker_principal_did
            and authorization.worker_principal_did != expected_worker_principal_did
        )
        or (
            expected_provider_principal_did
            and authorization.provider_principal_did != expected_provider_principal_did
        )
        or authorization.issued_at_ms <= 0
        or authorization.issued_at_ms >= authorization.expires_at_ms
        or authorization.expires_at_ms - authorization.issued_at_ms
        > _MAX_AUTHORIZATION_LIFETIME_MS
        or authorization.issued_at_ms > clock + _MAX_CLOCK_SKEW_MS
        or not authorization.issued_at_ms <= clock < authorization.expires_at_ms
        or authorization.expires_at_ms > invocation_expires
        or (expected_container_name and authorization.container_name != expected_container_name)
        or (expected_lease is not None and authorization.lease_root != expected_lease)
        or authorization.lease_root.name != authorization.lease_id
    ):
        raise ValueError("worker network authorization binding is invalid")
    _validate_proxy_endpoint(authorization.proxy_endpoint)
    try:
        verify_did_key_signature(
            identity_did=authorization.signer_did,
            payload=authorization.signed_payload(),
            signature=authorization.signature,
        )
    except LocalProfileTampered as exc:
        raise ValueError("worker network authorization signature is invalid") from exc
    return authorization


def verify_worker_network_authorization(
    *,
    invocation_binding: object,
    provider: str,
    workspace: Path,
    now_ms: int | None = None,
    expected_artifact_cid: str = "",
    expected_container_name: str = "",
    expected_lease_root: Path | None = None,
    expected_worker_principal_did: str = "",
    expected_provider_principal_did: str = "",
) -> WorkerNetworkAuthorizationDecision:
    """Return a non-effecting typed decision for a child-birth gate."""

    try:
        authorization = load_worker_network_authorization(
            invocation_binding=invocation_binding,
            provider=provider,
            workspace=workspace,
            now_ms=now_ms,
            expected_artifact_cid=expected_artifact_cid,
            expected_container_name=expected_container_name,
            expected_lease_root=expected_lease_root,
            expected_worker_principal_did=expected_worker_principal_did,
            expected_provider_principal_did=expected_provider_principal_did,
        )
    except (OSError, ValueError) as exc:
        message = str(exc).strip().lower().replace(" ", "_")
        blocker = re.sub(r"[^a-z0-9_]+", "_", message).strip("_")
        return WorkerNetworkAuthorizationDecision(
            valid=False,
            blockers=(blocker or "worker_network_authorization_invalid",),
            authorization_cid="",
            reviewer_did=str(getattr(invocation_binding, "reviewer_identity", "")),
        )
    return WorkerNetworkAuthorizationDecision(
        valid=True,
        blockers=(),
        authorization_cid=authorization.authorization_id,
        reviewer_did=authorization.signer_did,
        authorization=authorization,
    )


@dataclass(frozen=True)
class WorkerNetworkProfile:
    """Effect-bound approval for one provider container and one lease."""

    provider: str
    docker_network: str
    proxy_endpoint: str
    allowed_hostnames: tuple[str, ...]
    approval_identity: str
    approval_cid: str
    effect_cid: str
    workspace: Path
    container_name: str
    lease_id: str
    lease_root: Path
    authorization: WorkerNetworkAuthorization | None = None
    schema: str = WORKER_NETWORK_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        provider = self.provider.strip().lower()
        expected_hosts = PROVIDER_HOSTNAME_ALLOWLISTS.get(provider)
        if expected_hosts is None:
            raise ValueError("worker network provider is not configured")
        if self.schema != WORKER_NETWORK_PROFILE_SCHEMA:
            raise ValueError("worker network profile schema is invalid")
        network = self.docker_network.strip()
        if (
            _NETWORK_NAME_RE.fullmatch(network) is None
            or network.lower() in _RESERVED_DOCKER_NETWORKS
        ):
            raise ValueError("worker network must name a dedicated Docker network")
        endpoint = _validate_proxy_endpoint(self.proxy_endpoint.strip())
        hosts = tuple(self.allowed_hostnames)
        if hosts != expected_hosts:
            raise ValueError("worker network provider hostname allowlist is not exact")
        for hostname in hosts:
            try:
                ipaddress.ip_address(hostname)
            except ValueError:
                pass
            else:
                raise ValueError("worker network hostname allowlist forbids IP literals")
            if hostname != hostname.lower() or hostname.endswith("."):
                raise ValueError("worker network hostname allowlist is not canonical")
        identity = self.approval_identity.strip()
        if _APPROVAL_IDENTITY_RE.fullmatch(identity) is None:
            raise ValueError("worker network approval identity is invalid")
        effect_cid = self.effect_cid.strip()
        if _CID_RE.fullmatch(effect_cid) is None:
            raise ValueError("worker network effect CID is invalid")
        workspace = _normalize_workspace(self.workspace)
        container_name = self.container_name.strip()
        if _CONTAINER_NAME_RE.fullmatch(container_name) is None:
            raise ValueError("worker network container binding is invalid")
        lease_id = self.lease_id.strip()
        if _LEASE_ID_RE.fullmatch(lease_id) is None:
            raise ValueError("worker network lease binding is invalid")
        lease_root = _normalize_workspace(self.lease_root)
        if Path(lease_root).name != lease_id:
            raise ValueError("worker network lease identity does not match its path")
        if self.authorization is not None and (
            self.authorization.provider != provider
            or self.authorization.docker_network != network
            or self.authorization.proxy_endpoint != endpoint
            or self.authorization.allowed_hostnames != hosts
            or self.authorization.effect_cid != effect_cid
            or self.authorization.workspace != Path(workspace)
            or self.authorization.container_name != container_name
            or self.authorization.lease_id != lease_id
            or self.authorization.lease_root != Path(lease_root)
        ):
            raise ValueError("worker network signed authorization does not bind profile")
        expected_cid = worker_network_approval_cid(
            provider=provider,
            docker_network=network,
            proxy_endpoint=endpoint,
            approval_identity=identity,
            effect_cid=effect_cid,
            workspace=Path(workspace),
            container_name=container_name,
            lease_id=lease_id,
            lease_root=Path(lease_root),
        )
        if self.approval_cid != expected_cid:
            raise ValueError("worker network approval CID does not bind this effect")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "docker_network", network)
        object.__setattr__(self, "proxy_endpoint", endpoint)
        object.__setattr__(self, "workspace", Path(workspace))
        object.__setattr__(self, "lease_root", Path(lease_root))

    def validate_effect_binding(
        self,
        *,
        provider: str,
        workspace: Path,
        container_name: str,
        lease_root: Path,
    ) -> None:
        if (
            provider != self.provider
            or workspace.resolve(strict=False) != self.workspace
            or container_name != self.container_name
            or lease_root.resolve(strict=False) != self.lease_root
        ):
            raise ValueError("worker network profile does not bind this provider effect")

    def docker_arguments(self) -> tuple[str, ...]:
        arguments = (
            f"--network={self.docker_network}",
            "--dns=127.0.0.1",
            "--label",
            f"ipfs_accelerate.worker_network_binding={self.approval_cid}",
            "--label",
            f"ipfs_accelerate.worker_network_effect={self.effect_cid}",
        )
        if self.authorization is not None:
            arguments += (
                "--label",
                "ipfs_accelerate.worker_network_authorization="
                + self.authorization.authorization_id,
            )
        return arguments

    def proxy_environment(self) -> Mapping[str, str]:
        return {
            "HTTP_PROXY": self.proxy_endpoint,
            "HTTPS_PROXY": self.proxy_endpoint,
            "NO_PROXY": "",
            "http_proxy": self.proxy_endpoint,
            "https_proxy": self.proxy_endpoint,
            "no_proxy": "",
        }


EAAEFWorkerNetworkProfile = WorkerNetworkProfile


def diagnostic_network_arguments() -> tuple[str, ...]:
    """Return the network-disabled default for non-provider diagnostics."""

    return ("--network=none",)


def is_proxy_variable(name: str) -> bool:
    lowered = name.lower()
    return lowered in _PROXY_VARIABLES or lowered.endswith("_proxy")


def validate_provider_hostname(profile: WorkerNetworkProfile, hostname: str) -> str:
    """Validate a CONNECT destination before any DNS lookup occurs."""

    candidate = hostname.strip()
    try:
        ipaddress.ip_address(candidate.strip("[]"))
    except ValueError:
        pass
    else:
        raise ValueError("provider proxy destinations cannot be IP literals")
    if candidate != candidate.lower() or candidate.endswith("."):
        raise ValueError("provider proxy destination is not canonical")
    if candidate not in profile.allowed_hostnames:
        raise ValueError("provider proxy destination is not approved")
    return candidate


def validate_worker_network_inspection(
    inspection: object,
    *,
    authorization: WorkerNetworkAuthorization,
    worker_container_id: str = "",
) -> None:
    """Verify the exact internal network and its only admitted peers."""

    if not isinstance(inspection, list) or len(inspection) != 1:
        raise ValueError("worker Docker network inspection is invalid")
    item = inspection[0]
    if not isinstance(item, Mapping):
        raise ValueError("worker Docker network inspection is invalid")
    containers = item.get("Containers")
    ipam = item.get("IPAM")
    config = ipam.get("Config") if isinstance(ipam, Mapping) else None
    if not isinstance(containers, Mapping) or not isinstance(config, list):
        raise ValueError("worker Docker network inspection is incomplete")
    if (
        item.get("Name") != authorization.docker_network
        or item.get("Id") != authorization.docker_network_id
        or item.get("Internal") is not True
        or item.get("Ingress") is not False
        or item.get("Driver") != "bridge"
        or item.get("Scope") != "local"
        or item.get("Attachable") is not False
    ):
        raise ValueError("worker Docker network identity or isolation drifted")
    proxy_address = ipaddress.ip_address(
        urlsplit(authorization.proxy_endpoint).hostname or ""
    )
    subnets: list[Any] = []
    for entry in config:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("Subnet"), str):
            continue
        try:
            subnets.append(ipaddress.ip_network(entry["Subnet"], strict=False))
        except ValueError as exc:
            raise ValueError("worker Docker network subnet is invalid") from exc
    if not subnets or not any(proxy_address in subnet for subnet in subnets):
        raise ValueError("worker proxy endpoint is outside its signed network")
    expected_ids = {authorization.proxy_container_id}
    if worker_container_id:
        if _DOCKER_ID_RE.fullmatch(worker_container_id) is None:
            raise ValueError("worker container identity is invalid")
        expected_ids.add(worker_container_id)
    if set(str(key) for key in containers) != expected_ids:
        raise ValueError("worker Docker network has an unexpected peer")
    proxy = containers.get(authorization.proxy_container_id)
    if not isinstance(proxy, Mapping):
        raise ValueError("worker proxy container is absent")
    raw_proxy_ip = str(proxy.get("IPv4Address") or "").partition("/")[0]
    if raw_proxy_ip != str(proxy_address):
        raise ValueError("worker proxy container address drifted")


def validate_provider_worker_command(
    command: Sequence[str],
    *,
    profile: WorkerNetworkProfile,
    expected_image: str,
    additional_labels: Sequence[str] = (),
    container_execution_profile: object | None = None,
) -> None:
    """Reject command drift at the final Docker construction boundary."""

    try:
        create_index = command.index("create")
        image_index = command.index(expected_image, create_index + 1)
    except ValueError as exc:
        raise ValueError("provider worker Docker image boundary is invalid") from exc
    if command.count(expected_image) != 1 or image_index <= create_index + 1:
        raise ValueError("provider worker Docker image boundary is invalid")
    docker_args = list(command[create_index + 1 : image_index])
    valued_options = {
        "--tmpfs",
        "--name",
        "--cidfile",
        "--label",
        "--user",
        "--workdir",
        "--env",
        "--mount",
    }
    legacy_resource_flags = {
        "--pids-limit=1024",
        "--cpus=4",
        "--memory=16g",
        "--memory-swap=16g",
    }
    execution_resource_flags: set[str] = set()
    if container_execution_profile is not None:
        try:
            execution_resource_flags = {
                f"--pids-limit={int(container_execution_profile.pids_limit)}",
                f"--cpus={float(container_execution_profile.cpu_limit):g}",
                (
                    "--memory="
                    f"{int(container_execution_profile.memory_limit_bytes)}"
                ),
                (
                    "--memory-swap="
                    f"{int(container_execution_profile.memory_limit_bytes)}"
                ),
                (
                    "--storage-opt=size="
                    f"{int(container_execution_profile.disk_limit_bytes)}"
                ),
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "provider worker container execution profile is incomplete"
            ) from exc
    resource_flags = (
        execution_resource_flags
        if container_execution_profile is not None
        else legacy_resource_flags
    )
    allowed_flags = {
        "--pull=never",
        "--interactive",
        "--read-only",
        f"--network={profile.docker_network}",
        "--dns=127.0.0.1",
        "--runtime=runc",
        "--entrypoint=/usr/bin/env",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        *resource_flags,
    }
    parsed_flags: dict[str, int] = {}
    cursor = 0
    while cursor < len(docker_args):
        item = docker_args[cursor]
        if item in valued_options:
            if cursor + 1 >= len(docker_args):
                raise ValueError("provider worker Docker option value is missing")
            cursor += 2
            continue
        if item not in allowed_flags:
            raise ValueError("provider worker contains an unapproved Docker option")
        parsed_flags[item] = parsed_flags.get(item, 0) + 1
        if parsed_flags[item] != 1:
            raise ValueError("provider worker contains a duplicate Docker option")
        cursor += 1
    network_args = [item for item in docker_args if item.startswith("--network=")]
    if network_args != [f"--network={profile.docker_network}"]:
        raise ValueError("provider worker has an invalid Docker network")
    exact_singletons = {
        "--dns=127.0.0.1",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        *resource_flags,
    }
    if any(docker_args.count(item) != 1 for item in exact_singletons):
        raise ValueError("provider worker must disable direct external DNS")
    try:
        if docker_args.count("--user") != 1:
            raise ValueError
        user = docker_args[docker_args.index("--user") + 1]
    except (IndexError, ValueError) as exc:
        raise ValueError("provider worker must use an explicit nonroot account") from exc
    if re.fullmatch(r"[1-9][0-9]*:[1-9][0-9]*", user) is None:
        raise ValueError("provider worker cannot run as root")
    if (
        container_execution_profile is not None
        and user != str(getattr(container_execution_profile, "nonroot_user", ""))
    ):
        raise ValueError("provider worker nonroot account binding drifted")
    expected_workdir = str(profile.workspace)
    expected_mounts: dict[str, bool] | None = None
    if container_execution_profile is not None:
        if (
            str(getattr(container_execution_profile, "network_mode", ""))
            != "policy_proxy_only"
            or getattr(container_execution_profile, "read_only_base", None) is not True
            or tuple(getattr(container_execution_profile, "cap_drop", ())) != ("ALL",)
            or getattr(container_execution_profile, "no_new_privileges", None)
            is not True
        ):
            raise ValueError("provider worker security profile is not exact")
        mount_projector = getattr(
            container_execution_profile,
            "mounts_for_provider",
            None,
        )
        mounts = (
            tuple(mount_projector(profile.provider))
            if callable(mount_projector)
            else tuple(getattr(container_execution_profile, "mounts", ()))
        )
        worktrees = tuple(
            mount for mount in mounts if getattr(mount, "kind", "") == "worktree"
        )
        if len(worktrees) != 1:
            raise ValueError("provider worker worktree mount binding is invalid")
        expected_workdir = str(getattr(worktrees[0], "target", ""))
        expected_mounts = {
            str(getattr(mount, "target", "")): bool(
                getattr(mount, "read_only", False)
            )
            for mount in mounts
        }
        if (
            not expected_workdir.startswith("/")
            or len(expected_mounts) != len(mounts)
            or "" in expected_mounts
        ):
            raise ValueError("provider worker signed mount projection is invalid")
        expected_host = "--host=" + str(
            getattr(container_execution_profile, "engine_endpoint", "")
        )
        host_args = [str(item) for item in command[:create_index] if str(item).startswith("--host=")]
        if host_args != [expected_host]:
            raise ValueError("provider worker engine endpoint binding drifted")
    singleton_values = {
        "--name": profile.container_name,
        "--cidfile": str(profile.lease_root / "container.cid"),
        "--workdir": expected_workdir,
    }
    for option, expected_value in singleton_values.items():
        positions = [
            index for index, item in enumerate(docker_args[:-1]) if item == option
        ]
        if (
            len(positions) != 1
            or docker_args[positions[0] + 1] != expected_value
        ):
            raise ValueError("provider worker Docker singleton binding drifted")
    labels = [
        docker_args[index + 1]
        for index, item in enumerate(docker_args[:-1])
        if item == "--label"
    ]
    extra_labels = tuple(str(item) for item in additional_labels)
    if (
        len(set(extra_labels)) != len(extra_labels)
        or any(
            not item
            or "=" not in item
            or "\x00" in item
            or len(item.encode("utf-8")) > 512
            for item in extra_labels
        )
    ):
        raise ValueError("provider worker additional labels are invalid")
    expected_labels = {
        f"ipfs_accelerate.worker_network_binding={profile.approval_cid}",
        f"ipfs_accelerate.worker_network_effect={profile.effect_cid}",
        (
            "ipfs_accelerate.grok_isolation=true"
            if profile.provider == "grok"
            else "ipfs_accelerate.codex_fallback_isolation=true"
        ),
        *extra_labels,
    }
    if profile.authorization is not None:
        expected_labels.add(
            "ipfs_accelerate.worker_network_authorization="
            + profile.authorization.authorization_id
        )
    if len(labels) != len(expected_labels) or set(labels) != expected_labels:
        raise ValueError("provider worker Docker labels are not exact")
    forbidden_exact = {"--privileged", "-P"}
    forbidden_prefixes = (
        "--net", "--add-host", "--dns-search", "--expose", "--publish", "-p",
        "--volume", "-v", "--device", "--cap-add", "--userns", "--pid", "--ipc",
    )
    for item in docker_args:
        if item == f"--network={profile.docker_network}" or item in exact_singletons:
            continue
        if item in forbidden_exact or any(item.startswith(prefix) for prefix in forbidden_prefixes):
            raise ValueError("provider worker contains an alternate authority route")
        for constrained in (
            "--dns=", "--cap-drop=", "--security-opt=", "--pids-limit=",
            "--cpus=", "--memory=", "--memory-swap=", "--storage-opt=",
        ):
            if item.startswith(constrained) and item not in exact_singletons:
                raise ValueError("provider worker boundary contains an override")
    if "--network" in docker_args or "--dns" in docker_args:
        raise ValueError("provider worker contains an alternate network route")
    proxy_values: dict[str, str] = {}
    for index, item in enumerate(docker_args):
        if item != "--env" or index + 1 >= len(docker_args):
            continue
        assignment = docker_args[index + 1]
        name, separator, value = assignment.partition("=")
        if is_proxy_variable(name):
            if not separator:
                raise ValueError("provider worker inherits an arbitrary proxy variable")
            if name in proxy_values:
                raise ValueError("provider worker contains a duplicate proxy variable")
            proxy_values[name] = value
    if proxy_values != dict(profile.proxy_environment()):
        raise ValueError("provider worker proxy environment is not exact")
    observed_mounts: dict[str, bool] = {}
    for index, item in enumerate(docker_args[:-1]):
        if item != "--mount":
            continue
        mount = docker_args[index + 1]
        lowered_mount = mount.lower()
        if (
            "type=volume" in lowered_mount
            or "type=tmpfs" in lowered_mount
            or ".sock" in lowered_mount
            or "containerd" in lowered_mount
            or "podman" in lowered_mount
            or "docker" in lowered_mount and "sock" in lowered_mount
        ):
            raise ValueError("provider worker cannot mount an alternate authority")
        if expected_mounts is None:
            continue
        fields = mount.split(",")
        if (
            len(fields) not in {3, 4}
            or fields[0] != "type=bind"
            or not fields[1].startswith("src=/")
            or not fields[2].startswith("dst=/")
            or (len(fields) == 4 and fields[3] != "readonly")
        ):
            raise ValueError("provider worker signed bind mount is invalid")
        target = fields[2].removeprefix("dst=")
        if target in observed_mounts:
            raise ValueError("provider worker signed bind mount is duplicated")
        observed_mounts[target] = len(fields) == 4
    if expected_mounts is not None and observed_mounts != expected_mounts:
        raise ValueError("provider worker signed mount binding drifted")


__all__ = [
    "EAAEFWorkerNetworkProfile",
    "PROVIDER_HOSTNAME_ALLOWLISTS",
    "WORKER_NETWORK_AUTHORIZATION_SCHEMA",
    "WORKER_NETWORK_PROFILE_SCHEMA",
    "WorkerNetworkProfile",
    "WorkerNetworkAuthorization",
    "WorkerNetworkAuthorizationDecision",
    "derived_worker_network_name",
    "diagnostic_network_arguments",
    "is_proxy_variable",
    "load_worker_network_authorization",
    "validate_provider_hostname",
    "validate_provider_worker_command",
    "validate_worker_network_inspection",
    "verify_worker_network_authorization",
    "worker_network_authorization_relative_path",
    "worker_network_approval_cid",
]
