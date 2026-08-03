"""Explicit development bootstrap for legacy landed-review authorities.

This module is intentionally separate from supervisor startup.  Starting a
daemon must never infer, generate, or replace signing authority.  An operator
may run this module once, after the implementation tree is clean, to create
two distinct Ed25519 development keys and an exact policy pinned to that Git
``HEAD`` and tree.

The operation is restart-safe and idempotent.  Existing keys are loaded but
never replaced; an existing policy must be byte-identical to the policy that
would be generated now.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ..merge.checkout_lock import (
    CheckoutMaintenanceLease,
    checkout_lock_metadata,
    checkout_mutation_lock_path,
)
from ..proof.formal_verification_contracts import canonical_json_bytes
from .legacy_landed_attestation import LegacyLandedReviewAuthority
from .legacy_landed_review import (
    MAX_POLICY_BYTES,
    build_exact_eight_legacy_landed_policy,
    load_legacy_landed_review_policy,
)
from .production_provider_attestation import (
    ProductionProviderReviewAuthority,
)

PRODUCTION_REVIEW_KEY_NAME: Final = (
    "production-provider-review-authority.ed25519"
)
LEGACY_REVIEW_KEY_NAME: Final = "legacy-landed-review-authority.ed25519"
LEGACY_REVIEW_POLICY_NAME: Final = "legacy-landed-review-policy.json"
_BOOTSTRAP_LOCK_NAME: Final = ".legacy-landed-review-bootstrap.lock"
_BOOTSTRAP_LOCK_TIMEOUT_SECONDS: Final = 30.0
_CHECKOUT_LOCK_MAX_HOLD_SECONDS: Final = 120.0
_BOOTSTRAP_THREAD_LOCKS: dict[str, threading.RLock] = {}
_BOOTSTRAP_THREAD_LOCKS_GUARD = threading.Lock()


def _private_directory(path: str | Path) -> Path:
    """Create/validate one non-symlink, user-only authority directory."""

    directory = Path(os.path.abspath(os.fspath(path)))
    anchor = Path(directory.anchor)
    current = anchor
    for component in directory.parts[1:]:
        current = current / component
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            try:
                os.mkdir(current, 0o700)
            except FileExistsError:
                pass
            info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("authority directory path cannot contain a symlink")
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("authority directory path contains a non-directory")
    info = os.lstat(directory)
    if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
        raise ValueError("authority directory owner is invalid")
    if stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("authority directory permissions must be 0700")
    return directory


def _git_head(repo_root: Path) -> str:
    environment = dict(os.environ)
    environment.update(
        {
            "LC_ALL": "C",
            "LANG": "C",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
        }
    )
    for name in (
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_DIR",
        "GIT_EXTERNAL_DIFF",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_WORK_TREE",
    ):
        environment.pop(name, None)
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
        ],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    head = result.stdout.strip().lower()
    if result.returncode != 0 or len(head) != 40 or any(
        character not in "0123456789abcdef" for character in head
    ):
        raise ValueError("repository HEAD cannot be resolved exactly")
    return head


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _BOOTSTRAP_THREAD_LOCKS_GUARD:
        return _BOOTSTRAP_THREAD_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _bootstrap_lock(authority_directory: Path) -> Iterator[None]:
    """Serialize one complete bootstrap across threads and processes."""

    lock_path = authority_directory / _BOOTSTRAP_LOCK_NAME
    thread_lock = _thread_lock(lock_path)
    if not thread_lock.acquire(timeout=_BOOTSTRAP_LOCK_TIMEOUT_SECONDS):
        raise TimeoutError("timed out acquiring legacy bootstrap thread lock")
    descriptor: int | None = None
    acquired = False
    try:
        descriptor = os.open(
            lock_path,
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise ValueError("legacy bootstrap lock is unsafe")
        deadline = time.monotonic() + _BOOTSTRAP_LOCK_TIMEOUT_SECONDS
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "timed out acquiring legacy bootstrap process lock"
                    ) from exc
                time.sleep(0.01)
        yield
    finally:
        if descriptor is not None:
            try:
                if acquired:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        thread_lock.release()


def _read_existing_policy_bytes(path: Path) -> bytes | None:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            return None
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError("legacy policy target is unsafe") from exc
        raise
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("legacy policy target is unsafe")
        if hasattr(os, "geteuid") and before.st_uid != os.geteuid():
            raise ValueError("legacy policy owner is invalid")
        if before.st_nlink != 1:
            raise ValueError("legacy policy cannot be hard-linked")
        if stat.S_IMODE(before.st_mode) & 0o077:
            raise ValueError(
                "legacy policy permissions must be 0600 or stricter"
            )
        if before.st_size > MAX_POLICY_BYTES:
            raise ValueError("legacy policy is too large")
        raw = b""
        while len(raw) <= MAX_POLICY_BYTES:
            chunk = os.read(
                descriptor,
                min(65536, MAX_POLICY_BYTES + 1 - len(raw)),
            )
            if not chunk:
                break
            raw += chunk
        after = os.fstat(descriptor)
        if (
            len(raw) > MAX_POLICY_BYTES
            or before.st_size != len(raw)
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
        ):
            raise ValueError("legacy policy changed while being read")
        return raw
    finally:
        os.close(descriptor)


def _publish_policy_once(path: Path, payload: bytes) -> bool:
    """Publish bytes without an overwrite window; return whether created."""

    existing = _read_existing_policy_bytes(path)
    if existing is not None:
        if existing != payload:
            raise ValueError("existing legacy policy does not match final HEAD")
        return False

    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{os.urandom(12).hex()}.tmp"
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing legacy policy")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        try:
            os.link(temporary, path, follow_symlinks=False)
            created = True
            # The destination must have one link before strict read-back.
            temporary.unlink()
        except FileExistsError:
            created = False
        existing = _read_existing_policy_bytes(path)
        if existing != payload:
            raise ValueError("concurrent legacy policy publication conflicted")
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return created
    finally:
        temporary.unlink(missing_ok=True)


def _rollback_new_policy(path: Path, payload: bytes) -> None:
    """Remove only the exact policy published by the current locked call."""

    existing = _read_existing_policy_bytes(path)
    if existing != payload:
        raise RuntimeError("new legacy policy changed before rollback")
    path.unlink()
    directory_descriptor = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


@dataclass(frozen=True, slots=True)
class LegacyLandedBootstrapResult:
    authority_directory: Path
    production_key_path: Path
    legacy_key_path: Path
    policy_path: Path
    production_issuer_key_id: str
    legacy_issuer_key_id: str
    policy_id: str
    current_head: str
    current_tree_id: str
    production_key_created: bool
    legacy_key_created: bool
    policy_created: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority_directory": str(self.authority_directory),
            "production_key_path": str(self.production_key_path),
            "legacy_key_path": str(self.legacy_key_path),
            "policy_path": str(self.policy_path),
            "production_issuer_key_id": self.production_issuer_key_id,
            "legacy_issuer_key_id": self.legacy_issuer_key_id,
            "policy_id": self.policy_id,
            "current_head": self.current_head,
            "current_tree_id": self.current_tree_id,
            "production_key_created": self.production_key_created,
            "legacy_key_created": self.legacy_key_created,
            "policy_created": self.policy_created,
        }


def bootstrap_legacy_landed_review(
    *,
    repo_root: str | Path,
    authority_directory: str | Path,
    enabled: bool = True,
) -> LegacyLandedBootstrapResult:
    """Create/load the two dev keys and publish the exact-eight policy."""

    repo = Path(repo_root).resolve(strict=True)
    if not repo.is_dir():
        raise ValueError("repository root must be a directory")
    authority_dir = _private_directory(authority_directory)
    with _bootstrap_lock(authority_dir):
        lease = CheckoutMaintenanceLease(
            checkout_mutation_lock_path(repo),
            checkout_lock_metadata(
                kind="merge",
                repo_root=repo,
                task_id="legacy-landed-review-bootstrap",
                owner_script=Path(sys.argv[0]).name,
                extra={"operation": "legacy_landed_review_policy_bootstrap"},
            ),
            max_hold_seconds=_CHECKOUT_LOCK_MAX_HOLD_SECONDS,
        )
        with lease.exclusive_section():
            return _bootstrap_legacy_landed_review_locked(
                repo=repo,
                authority_dir=authority_dir,
                enabled=enabled,
            )


def _bootstrap_legacy_landed_review_locked(
    *,
    repo: Path,
    authority_dir: Path,
    enabled: bool,
) -> LegacyLandedBootstrapResult:
    """Perform bootstrap while authority and checkout locks are both held."""

    production_key_path = authority_dir / PRODUCTION_REVIEW_KEY_NAME
    legacy_key_path = authority_dir / LEGACY_REVIEW_KEY_NAME
    policy_path = authority_dir / LEGACY_REVIEW_POLICY_NAME

    if policy_path.exists() and not legacy_key_path.exists():
        raise ValueError("existing legacy policy has no paired authority key")

    production_existed = production_key_path.exists()
    legacy_existed = legacy_key_path.exists()
    production_authority = ProductionProviderReviewAuthority.load_or_create(
        production_key_path
    )
    # Both authority contracts deliberately use raw 32-byte Ed25519 keys.  We
    # reuse the hardened atomic creator, then load through the legacy contract.
    ProductionProviderReviewAuthority.load_or_create(legacy_key_path)
    legacy_authority = LegacyLandedReviewAuthority.from_private_key_path(
        legacy_key_path
    )
    if production_authority.issuer_key_id == legacy_authority.issuer_key_id:
        raise ValueError("production and legacy authorities must be distinct")

    head = _git_head(repo)
    payload = build_exact_eight_legacy_landed_policy(
        repo,
        current_head=head,
        issuer_key_id=legacy_authority.issuer_key_id,
        enabled=enabled,
    )
    policy_bytes = canonical_json_bytes(payload)
    policy_created = False
    try:
        # Fence once more immediately before the no-overwrite publication.
        if _git_head(repo) != head:
            raise ValueError("repository HEAD changed during authority bootstrap")
        policy_created = _publish_policy_once(policy_path, policy_bytes)
        loaded = load_legacy_landed_review_policy(policy_path)
        if (
            loaded.policy_id != payload["policy_id"]
            or loaded.issuer_key_id != legacy_authority.issuer_key_id
            or loaded.current_head != head
            or loaded.current_tree_id != payload["current_tree_id"]
            or loaded.enabled is not bool(enabled)
        ):
            raise ValueError("legacy policy round-trip verification failed")
        # Re-resolve after persistence so an uncooperative external checkout
        # mutation cannot leave a freshly written policy already stale.
        if _git_head(repo) != head:
            raise ValueError("repository HEAD changed during authority bootstrap")
    except BaseException:
        if policy_created:
            _rollback_new_policy(policy_path, policy_bytes)
        raise

    return LegacyLandedBootstrapResult(
        authority_directory=authority_dir,
        production_key_path=production_key_path,
        legacy_key_path=legacy_key_path,
        policy_path=policy_path,
        production_issuer_key_id=production_authority.issuer_key_id,
        legacy_issuer_key_id=legacy_authority.issuer_key_id,
        policy_id=loaded.policy_id,
        current_head=head,
        current_tree_id=loaded.current_tree_id,
        production_key_created=not production_existed,
        legacy_key_created=not legacy_existed,
        policy_created=policy_created,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create distinct development review keys and an exact legacy "
            "policy pinned to the current clean Git HEAD."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--authority-directory", type=Path, required=True)
    parser.add_argument(
        "--disabled",
        action="store_true",
        help="Generate a staged policy with review explicitly disabled.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = bootstrap_legacy_landed_review(
        repo_root=args.repo_root,
        authority_directory=args.authority_directory,
        enabled=not args.disabled,
    )
    print(json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())


__all__ = [
    "LEGACY_REVIEW_KEY_NAME",
    "LEGACY_REVIEW_POLICY_NAME",
    "PRODUCTION_REVIEW_KEY_NAME",
    "LegacyLandedBootstrapResult",
    "bootstrap_legacy_landed_review",
    "main",
    "parse_args",
]
