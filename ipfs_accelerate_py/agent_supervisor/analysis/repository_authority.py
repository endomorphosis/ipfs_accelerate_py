"""Canonical repository authority for SwissKnife contract-assurance snapshots.

The integration repository and its SwissKnife checkout are separate
observations.  Each observation receives its own CID before they are combined
under one authority root.  The integration repository's indexed gitlink is
the default program authority; a checkout HEAD never silently replaces it.

Authority-bound cache, proof, and artifact records may only be joined when
their complete authority roots are identical.  This deliberately makes
cross-checkout and stale-cache joins fail closed.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Final, Iterable, Mapping, Sequence

from .content_identity_bridge import identify_strict_artifact, require_multiformats


CHECKOUT_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/repository-checkout-binding@1"
)
STATE_DIGEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/repository-state-digest@1"
)
REVIEWED_AUTHORITY_OVERRIDE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/reviewed-authority-override@1"
)
FRESHNESS_WORK_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/repository-freshness-work@1"
)
SNAPSHOT_AUTHORITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/snapshot-authority@1"
)
AUTHORITY_BOUND_REFERENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/authority-bound-reference@1"
)

_EMPTY_SHA256: Final = hashlib.sha256(b"").hexdigest()
_HEX = frozenset("0123456789abcdef")


class RepositoryAuthorityError(ValueError):
    """Base error for repository-authority failures."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "repository_authority_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


class GitlinkAuthorityError(RepositoryAuthorityError):
    """The integration index does not contain the required stage-zero gitlink."""


class ReviewedEvidenceError(RepositoryAuthorityError):
    """A requested authority override lacks valid reviewed evidence."""


class AuthorityRootMismatchError(RepositoryAuthorityError):
    """Records from different authority roots were presented for a join."""


class AuthoritySource(str, Enum):
    """How the authoritative SwissKnife program commit was selected."""

    INTEGRATION_GITLINK = "integration_gitlink"
    REVIEWED_EVIDENCE = "reviewed_evidence"


class FreshnessWorkKind(str, Enum):
    """Typed reconciliation work emitted for checkout/authority divergence."""

    CHECKOUT_MISSING = "checkout_missing"
    CHECKOUT_DIRTY = "checkout_dirty"
    CHECKOUT_AHEAD = "checkout_ahead"
    CHECKOUT_BEHIND = "checkout_behind"
    CHECKOUT_DIVERGED = "checkout_diverged"
    AUTHORITY_COMMIT_UNAVAILABLE = "authority_commit_unavailable"


class AuthorityJoinKind(str, Enum):
    """Supported consumers of repository authority."""

    CACHE = "cache"
    PROOF = "proof"
    ARTIFACT = "artifact"


def _require_nonempty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RepositoryAuthorityError(
            f"{field_name} must be a nonempty string",
            reason_code="invalid_authority_field",
            details={"field": field_name},
        )
    return value


def _require_oid(value: str, field_name: str, *, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return value
    if (
        not isinstance(value, str)
        or len(value) not in (40, 64)
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise RepositoryAuthorityError(
            f"{field_name} must be a lowercase Git object id",
            reason_code="invalid_git_object_id",
            details={"field": field_name, "value": value},
        )
    return value


def _require_sha256(value: str, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise RepositoryAuthorityError(
            f"{field_name} must be a lowercase SHA-256 digest",
            reason_code="invalid_sha256_digest",
            details={"field": field_name},
        )
    return value


def _strict_cid(payload: Mapping[str, Any]) -> str:
    return identify_strict_artifact(payload).cid


def _require_cid(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.lower():
        raise RepositoryAuthorityError(
            f"{field_name} must be a lowercase CID",
            reason_code="invalid_cid",
            details={"field": field_name},
        )
    try:
        require_multiformats()
        from multiformats import CID  # type: ignore[attr-defined]

        CID.decode(value)
    except Exception as exc:
        raise RepositoryAuthorityError(
            f"{field_name} is not a decodable CID",
            reason_code="invalid_cid",
            details={"field": field_name, "cause": repr(exc)},
        ) from exc
    return value


def _assert_stored_cid(
    stored: str,
    payload: Mapping[str, Any],
    field_name: str,
) -> None:
    expected = _strict_cid(payload)
    if stored != expected:
        raise RepositoryAuthorityError(
            f"{field_name} does not match its canonical preimage",
            reason_code="authority_cid_mismatch",
            details={"field": field_name, "expected": expected, "actual": stored},
        )


def _require_schema(
    value: Mapping[str, Any],
    expected: str,
    field_name: str,
) -> None:
    if value.get("schema") != expected:
        raise RepositoryAuthorityError(
            f"{field_name} has an unsupported schema",
            reason_code="unsupported_authority_schema",
            details={
                "field": field_name,
                "expected": expected,
                "actual": value.get("schema"),
            },
        )


def _git_bytes(root: Path, *arguments: str, check: bool = True) -> bytes:
    command = ("git", "-C", os.fspath(root), *arguments)
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and completed.returncode:
        raise RepositoryAuthorityError(
            "Git command failed while binding repository authority",
            reason_code="git_command_failed",
            details={
                "command": list(command),
                "returncode": completed.returncode,
                "stderr": completed.stderr.decode("utf-8", "replace").strip(),
            },
        )
    return completed.stdout


def _git_text(root: Path, *arguments: str, check: bool = True) -> str:
    return _git_bytes(root, *arguments, check=check).decode("utf-8", "strict").strip()


def _repository_present(root: Path) -> bool:
    return root.is_dir() and bool(
        _git_text(root, "rev-parse", "--is-inside-work-tree", check=False)
        == "true"
    )


def _framed_untracked_state(root: Path, paths: Sequence[bytes]) -> bytes:
    """Return an unambiguous content stream for untracked paths."""

    framed = bytearray()
    for raw_path in sorted(paths):
        if not raw_path:
            continue
        relative = os.fsdecode(raw_path)
        candidate = root / relative
        metadata = candidate.lstat()
        framed.extend(len(raw_path).to_bytes(8, "big"))
        framed.extend(raw_path)
        if stat.S_ISLNK(metadata.st_mode):
            kind = b"symlink"
            content = os.fsencode(os.readlink(candidate))
        elif stat.S_ISREG(metadata.st_mode):
            kind = b"file"
            content = candidate.read_bytes()
        elif stat.S_ISDIR(metadata.st_mode):
            kind = b"directory"
            content = b""
        else:
            kind = b"special"
            content = str(stat.S_IFMT(metadata.st_mode)).encode("ascii")
        framed.extend(len(kind).to_bytes(2, "big"))
        framed.extend(kind)
        framed.extend(len(content).to_bytes(8, "big"))
        framed.extend(content)
    return bytes(framed)


@dataclass(frozen=True, slots=True)
class StateDigestBinding:
    """CID-bound digest of raw Git/index or worktree observation bytes."""

    domain: str
    sha256: str
    byte_length: int
    cid: str = ""

    def __post_init__(self) -> None:
        _require_nonempty(self.domain, "domain")
        _require_sha256(self.sha256, "sha256")
        if not isinstance(self.byte_length, int) or self.byte_length < 0:
            raise RepositoryAuthorityError(
                "byte_length must be a nonnegative integer",
                reason_code="invalid_authority_field",
                details={"field": "byte_length"},
            )
        payload = self._content_dict()
        if self.cid:
            _assert_stored_cid(self.cid, payload, "state_digest.cid")
        else:
            object.__setattr__(self, "cid", _strict_cid(payload))

    @classmethod
    def from_bytes(cls, domain: str, value: bytes) -> "StateDigestBinding":
        return cls(
            domain=domain,
            sha256=hashlib.sha256(value).hexdigest(),
            byte_length=len(value),
        )

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema": STATE_DIGEST_SCHEMA,
            "domain": self.domain,
            "algorithm": "sha256",
            "sha256": self.sha256,
            "byte_length": self.byte_length,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "cid": self.cid}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StateDigestBinding":
        _require_schema(value, STATE_DIGEST_SCHEMA, "state_digest")
        return cls(
            domain=str(value.get("domain", "")),
            sha256=str(value.get("sha256", "")),
            byte_length=value.get("byte_length", -1),
            cid=str(value.get("cid", "")),
        )


@dataclass(frozen=True, slots=True)
class CheckoutSnapshotBinding:
    """Independent content binding for one checkout observation."""

    checkout_id: str
    repository_path: str
    present: bool
    head_commit: str
    head_tree: str
    index_state: StateDigestBinding
    worktree_state: StateDigestBinding
    dirty: bool
    checkout_cid: str = ""

    def __post_init__(self) -> None:
        _require_nonempty(self.checkout_id, "checkout_id")
        _require_nonempty(self.repository_path, "repository_path")
        if not isinstance(self.present, bool) or not isinstance(self.dirty, bool):
            raise RepositoryAuthorityError(
                "present and dirty must be booleans",
                reason_code="invalid_authority_field",
            )
        _require_oid(self.head_commit, "head_commit", allow_empty=not self.present)
        _require_oid(self.head_tree, "head_tree", allow_empty=not self.present)
        if self.present and (not self.head_commit or not self.head_tree):
            raise RepositoryAuthorityError(
                "present checkout must have a HEAD commit and tree",
                reason_code="incomplete_checkout_binding",
            )
        if not self.present and (
            self.head_commit
            or self.head_tree
            or self.dirty
            or self.index_state.sha256 != _EMPTY_SHA256
            or self.worktree_state.sha256 != _EMPTY_SHA256
        ):
            raise RepositoryAuthorityError(
                "missing checkout must carry the canonical empty observation",
                reason_code="invalid_missing_checkout_binding",
            )
        payload = self._content_dict()
        if self.checkout_cid:
            _assert_stored_cid(
                self.checkout_cid, payload, f"{self.checkout_id}.checkout_cid"
            )
        else:
            object.__setattr__(self, "checkout_cid", _strict_cid(payload))

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema": CHECKOUT_BINDING_SCHEMA,
            "checkout_id": self.checkout_id,
            "repository_path": self.repository_path,
            "present": self.present,
            "head_commit": self.head_commit,
            "head_tree": self.head_tree,
            "index_state": self.index_state.to_dict(),
            "worktree_state": self.worktree_state.to_dict(),
            "dirty": self.dirty,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "checkout_cid": self.checkout_cid}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckoutSnapshotBinding":
        _require_schema(value, CHECKOUT_BINDING_SCHEMA, "checkout")
        index_state = value.get("index_state")
        worktree_state = value.get("worktree_state")
        if not isinstance(index_state, Mapping) or not isinstance(
            worktree_state, Mapping
        ):
            raise RepositoryAuthorityError(
                "checkout state bindings must be objects",
                reason_code="invalid_authority_document",
            )
        return cls(
            checkout_id=str(value.get("checkout_id", "")),
            repository_path=str(value.get("repository_path", "")),
            present=value.get("present"),
            head_commit=str(value.get("head_commit", "")),
            head_tree=str(value.get("head_tree", "")),
            index_state=StateDigestBinding.from_dict(index_state),
            worktree_state=StateDigestBinding.from_dict(worktree_state),
            dirty=value.get("dirty"),
            checkout_cid=str(value.get("checkout_cid", "")),
        )


def bind_checkout(
    root: str | os.PathLike[str],
    *,
    checkout_id: str,
    repository_path: str | None = None,
) -> CheckoutSnapshotBinding:
    """Bind a checkout's HEAD, index, tracked diff, and untracked content."""

    checkout_root = Path(root).resolve()
    stable_path = repository_path or checkout_id
    if not _repository_present(checkout_root):
        empty_index = StateDigestBinding.from_bytes(
            f"{checkout_id}:git-index", b""
        )
        empty_worktree = StateDigestBinding.from_bytes(
            f"{checkout_id}:worktree", b""
        )
        return CheckoutSnapshotBinding(
            checkout_id=checkout_id,
            repository_path=stable_path,
            present=False,
            head_commit="",
            head_tree="",
            index_state=empty_index,
            worktree_state=empty_worktree,
            dirty=False,
        )

    head_commit = _git_text(checkout_root, "rev-parse", "HEAD")
    head_tree = _git_text(checkout_root, "rev-parse", "HEAD^{tree}")
    index_bytes = _git_bytes(checkout_root, "ls-files", "--stage", "-z")
    tracked_diff = _git_bytes(
        checkout_root,
        "diff",
        "--binary",
        "--no-ext-diff",
        "--submodule=short",
        "HEAD",
        "--",
    )
    untracked_output = _git_bytes(
        checkout_root, "ls-files", "--others", "--exclude-standard", "-z"
    )
    untracked_paths = tuple(
        path for path in untracked_output.split(b"\0") if path
    )
    untracked_state = _framed_untracked_state(checkout_root, untracked_paths)
    worktree_bytes = (
        len(tracked_diff).to_bytes(8, "big")
        + tracked_diff
        + len(untracked_state).to_bytes(8, "big")
        + untracked_state
    )
    return CheckoutSnapshotBinding(
        checkout_id=checkout_id,
        repository_path=stable_path,
        present=True,
        head_commit=_require_oid(head_commit, "head_commit"),
        head_tree=_require_oid(head_tree, "head_tree"),
        index_state=StateDigestBinding.from_bytes(
            f"{checkout_id}:git-index", index_bytes
        ),
        worktree_state=StateDigestBinding.from_bytes(
            f"{checkout_id}:worktree", worktree_bytes
        ),
        dirty=bool(tracked_diff or untracked_paths),
    )


@dataclass(frozen=True, slots=True)
class ReviewedAuthorityOverride:
    """Explicit reviewed evidence authorizing a non-gitlink program commit."""

    program_commit: str
    supersedes_gitlink_commit: str
    reviewer: str
    reviewed_at: str
    evidence: Mapping[str, Any]
    evidence_cid: str = ""

    def __post_init__(self) -> None:
        _require_oid(self.program_commit, "program_commit")
        _require_oid(
            self.supersedes_gitlink_commit, "supersedes_gitlink_commit"
        )
        _require_nonempty(self.reviewer, "reviewer")
        _require_nonempty(self.reviewed_at, "reviewed_at")
        if not isinstance(self.evidence, Mapping) or not self.evidence:
            raise ReviewedEvidenceError(
                "reviewed authority override requires nonempty evidence",
                reason_code="reviewed_evidence_missing",
            )
        evidence_payload = {
            "schema": REVIEWED_AUTHORITY_OVERRIDE_SCHEMA,
            "program_commit": self.program_commit,
            "supersedes_gitlink_commit": self.supersedes_gitlink_commit,
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
            "evidence": dict(self.evidence),
        }
        if self.evidence_cid:
            try:
                _assert_stored_cid(
                    self.evidence_cid, evidence_payload, "evidence_cid"
                )
            except RepositoryAuthorityError as exc:
                raise ReviewedEvidenceError(
                    "reviewed evidence CID does not match its preimage",
                    reason_code="reviewed_evidence_cid_mismatch",
                    details=exc.details,
                ) from exc
        else:
            object.__setattr__(self, "evidence_cid", _strict_cid(evidence_payload))

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema": REVIEWED_AUTHORITY_OVERRIDE_SCHEMA,
            "program_commit": self.program_commit,
            "supersedes_gitlink_commit": self.supersedes_gitlink_commit,
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
            "evidence": dict(self.evidence),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "evidence_cid": self.evidence_cid}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReviewedAuthorityOverride":
        _require_schema(
            value,
            REVIEWED_AUTHORITY_OVERRIDE_SCHEMA,
            "reviewed_override",
        )
        evidence = value.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ReviewedEvidenceError(
                "reviewed evidence must be an object",
                reason_code="reviewed_evidence_missing",
            )
        return cls(
            program_commit=str(value.get("program_commit", "")),
            supersedes_gitlink_commit=str(
                value.get("supersedes_gitlink_commit", "")
            ),
            reviewer=str(value.get("reviewer", "")),
            reviewed_at=str(value.get("reviewed_at", "")),
            evidence=dict(evidence),
            evidence_cid=str(value.get("evidence_cid", "")),
        )


@dataclass(frozen=True, slots=True)
class FreshnessWork:
    """One typed reconciliation obligation."""

    kind: FreshnessWorkKind
    checkout_id: str
    authority_commit: str
    checkout_commit: str = ""
    work_cid: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, FreshnessWorkKind):
            object.__setattr__(self, "kind", FreshnessWorkKind(self.kind))
        _require_nonempty(self.checkout_id, "checkout_id")
        _require_oid(self.authority_commit, "authority_commit")
        _require_oid(
            self.checkout_commit, "checkout_commit", allow_empty=True
        )
        payload = self._content_dict()
        if self.work_cid:
            _assert_stored_cid(self.work_cid, payload, "freshness_work.work_cid")
        else:
            object.__setattr__(self, "work_cid", _strict_cid(payload))

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema": FRESHNESS_WORK_SCHEMA,
            "kind": self.kind.value,
            "checkout_id": self.checkout_id,
            "authority_commit": self.authority_commit,
            "checkout_commit": self.checkout_commit,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "work_cid": self.work_cid}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FreshnessWork":
        _require_schema(value, FRESHNESS_WORK_SCHEMA, "freshness_work")
        return cls(
            kind=FreshnessWorkKind(str(value.get("kind", ""))),
            checkout_id=str(value.get("checkout_id", "")),
            authority_commit=str(value.get("authority_commit", "")),
            checkout_commit=str(value.get("checkout_commit", "")),
            work_cid=str(value.get("work_cid", "")),
        )


def _read_gitlink(root: Path, gitlink_path: str) -> str:
    pure_path = PurePosixPath(gitlink_path)
    if (
        not gitlink_path
        or pure_path.is_absolute()
        or ".." in pure_path.parts
        or os.fspath(pure_path) != gitlink_path
    ):
        raise GitlinkAuthorityError(
            "gitlink path must be a normalized relative POSIX path",
            reason_code="invalid_gitlink_path",
            details={"gitlink_path": gitlink_path},
        )
    output = _git_bytes(root, "ls-files", "--stage", "-z", "--", gitlink_path)
    records = [record for record in output.split(b"\0") if record]
    if len(records) != 1:
        raise GitlinkAuthorityError(
            "integration index must contain exactly one gitlink record",
            reason_code="gitlink_record_missing",
            details={"gitlink_path": gitlink_path, "record_count": len(records)},
        )
    metadata, separator, raw_path = records[0].partition(b"\t")
    fields = metadata.decode("ascii", "strict").split()
    if (
        not separator
        or raw_path != os.fsencode(gitlink_path)
        or len(fields) != 3
        or fields[0] != "160000"
        or fields[2] != "0"
    ):
        raise GitlinkAuthorityError(
            "integration index record is not a stage-zero gitlink",
            reason_code="invalid_gitlink_record",
            details={"gitlink_path": gitlink_path},
        )
    return _require_oid(fields[1], "integration_gitlink_commit")


def _is_ancestor(root: Path, ancestor: str, descendant: str) -> bool | None:
    result = subprocess.run(
        ("git", "-C", os.fspath(root), "merge-base", "--is-ancestor", ancestor, descendant),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    return None


def _freshness_work(
    checkout_root: Path,
    checkout: CheckoutSnapshotBinding,
    authority_commit: str,
) -> tuple[FreshnessWork, ...]:
    work: list[FreshnessWork] = []
    if not checkout.present:
        return (
            FreshnessWork(
                kind=FreshnessWorkKind.CHECKOUT_MISSING,
                checkout_id=checkout.checkout_id,
                authority_commit=authority_commit,
            ),
        )
    if checkout.head_commit != authority_commit:
        authority_is_ancestor = _is_ancestor(
            checkout_root, authority_commit, checkout.head_commit
        )
        checkout_is_ancestor = _is_ancestor(
            checkout_root, checkout.head_commit, authority_commit
        )
        if authority_is_ancestor is None or checkout_is_ancestor is None:
            kind = FreshnessWorkKind.AUTHORITY_COMMIT_UNAVAILABLE
        elif authority_is_ancestor:
            kind = FreshnessWorkKind.CHECKOUT_AHEAD
        elif checkout_is_ancestor:
            kind = FreshnessWorkKind.CHECKOUT_BEHIND
        else:
            kind = FreshnessWorkKind.CHECKOUT_DIVERGED
        work.append(
            FreshnessWork(
                kind=kind,
                checkout_id=checkout.checkout_id,
                authority_commit=authority_commit,
                checkout_commit=checkout.head_commit,
            )
        )
    if checkout.dirty:
        work.append(
            FreshnessWork(
                kind=FreshnessWorkKind.CHECKOUT_DIRTY,
                checkout_id=checkout.checkout_id,
                authority_commit=authority_commit,
                checkout_commit=checkout.head_commit,
            )
        )
    return tuple(work)


def _validate_freshness_work(
    checkout: CheckoutSnapshotBinding,
    authority_commit: str,
    work: Sequence[FreshnessWork],
) -> None:
    """Require the complete, typed reconciliation set for a checkout."""

    relation_kinds = {
        FreshnessWorkKind.CHECKOUT_AHEAD,
        FreshnessWorkKind.CHECKOUT_BEHIND,
        FreshnessWorkKind.CHECKOUT_DIVERGED,
        FreshnessWorkKind.AUTHORITY_COMMIT_UNAVAILABLE,
    }
    kinds: list[FreshnessWorkKind] = []
    for item in work:
        if not isinstance(item, FreshnessWork):
            raise RepositoryAuthorityError(
                "freshness work must contain only typed records",
                reason_code="untyped_freshness_work",
            )
        if (
            item.checkout_id != checkout.checkout_id
            or item.authority_commit != authority_commit
            or item.checkout_commit != checkout.head_commit
        ):
            raise RepositoryAuthorityError(
                "freshness work does not bind the selected checkout and authority",
                reason_code="freshness_work_scope_mismatch",
                details={
                    "checkout_id": checkout.checkout_id,
                    "authority_commit": authority_commit,
                    "checkout_commit": checkout.head_commit,
                    "work_cid": item.work_cid,
                },
            )
        kinds.append(item.kind)

    if len(kinds) != len(set(kinds)):
        raise RepositoryAuthorityError(
            "freshness work contains duplicate obligations",
            reason_code="duplicate_freshness_work",
        )

    actual = set(kinds)
    if not checkout.present:
        expected = {FreshnessWorkKind.CHECKOUT_MISSING}
    else:
        expected: set[FreshnessWorkKind] = set()
        if checkout.dirty:
            expected.add(FreshnessWorkKind.CHECKOUT_DIRTY)
        relations = actual & relation_kinds
        if checkout.head_commit == authority_commit:
            if relations:
                raise RepositoryAuthorityError(
                    "matching checkout and authority cannot require reconciliation",
                    reason_code="invalid_freshness_work",
                )
        elif len(relations) != 1:
            raise RepositoryAuthorityError(
                "commit mismatch requires exactly one typed freshness obligation",
                reason_code="incomplete_freshness_work",
            )
        expected.update(relations)

    if actual != expected:
        raise RepositoryAuthorityError(
            "freshness work is not the complete checkout obligation set",
            reason_code="incomplete_freshness_work",
            details={
                "expected": sorted(kind.value for kind in expected),
                "actual": sorted(kind.value for kind in actual),
            },
        )


@dataclass(frozen=True, slots=True)
class SnapshotAuthority:
    """Complete CID-bound authority for one integration/SwissKnife observation."""

    integration_checkout: CheckoutSnapshotBinding
    swissknife_checkout: CheckoutSnapshotBinding
    integration_gitlink_path: str
    integration_gitlink_commit: str
    authority_source: AuthoritySource
    program_commit: str
    freshness_work: tuple[FreshnessWork, ...] = ()
    reviewed_override: ReviewedAuthorityOverride | None = None
    authority_root_cid: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.integration_checkout, CheckoutSnapshotBinding):
            raise RepositoryAuthorityError(
                "integration checkout must be a typed binding",
                reason_code="invalid_authority_document",
            )
        if not isinstance(self.swissknife_checkout, CheckoutSnapshotBinding):
            raise RepositoryAuthorityError(
                "SwissKnife checkout must be a typed binding",
                reason_code="invalid_authority_document",
            )
        if (
            self.integration_checkout.checkout_id != "integration"
            or not self.integration_checkout.present
        ):
            raise RepositoryAuthorityError(
                "authority requires a present integration checkout binding",
                reason_code="invalid_integration_checkout_binding",
            )
        if (
            self.swissknife_checkout.checkout_id != "swissknife"
            or self.swissknife_checkout.repository_path
            != self.integration_gitlink_path
        ):
            raise RepositoryAuthorityError(
                "SwissKnife checkout binding does not match the integration gitlink",
                reason_code="invalid_swissknife_checkout_binding",
            )
        if not isinstance(self.authority_source, AuthoritySource):
            object.__setattr__(
                self, "authority_source", AuthoritySource(self.authority_source)
            )
        _require_nonempty(self.integration_gitlink_path, "integration_gitlink_path")
        _require_oid(
            self.integration_gitlink_commit, "integration_gitlink_commit"
        )
        _require_oid(self.program_commit, "program_commit")
        object.__setattr__(self, "freshness_work", tuple(self.freshness_work))
        if self.authority_source is AuthoritySource.INTEGRATION_GITLINK:
            if self.reviewed_override is not None:
                raise ReviewedEvidenceError(
                    "gitlink authority cannot carry a reviewed override",
                    reason_code="unexpected_reviewed_override",
                )
            if self.program_commit != self.integration_gitlink_commit:
                raise RepositoryAuthorityError(
                    "default program authority must equal the indexed gitlink",
                    reason_code="gitlink_authority_mismatch",
                )
        else:
            override = self.reviewed_override
            if override is None:
                raise ReviewedEvidenceError(
                    "reviewed authority requires reviewed evidence",
                    reason_code="reviewed_evidence_missing",
                )
            if (
                override.supersedes_gitlink_commit
                != self.integration_gitlink_commit
                or override.program_commit != self.program_commit
            ):
                raise ReviewedEvidenceError(
                    "reviewed evidence does not bind the selected gitlink and commit",
                    reason_code="reviewed_evidence_scope_mismatch",
                )
        _validate_freshness_work(
            self.swissknife_checkout,
            self.program_commit,
            self.freshness_work,
        )
        payload = self._content_dict()
        if self.authority_root_cid:
            _assert_stored_cid(
                self.authority_root_cid, payload, "authority_root_cid"
            )
        else:
            object.__setattr__(self, "authority_root_cid", _strict_cid(payload))

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema": SNAPSHOT_AUTHORITY_SCHEMA,
            "integration_checkout": self.integration_checkout.to_dict(),
            "swissknife_checkout": self.swissknife_checkout.to_dict(),
            "integration_gitlink": {
                "path": self.integration_gitlink_path,
                "commit": self.integration_gitlink_commit,
            },
            "authority_source": self.authority_source.value,
            "program_commit": self.program_commit,
            "reviewed_override": (
                self.reviewed_override.to_dict()
                if self.reviewed_override is not None
                else None
            ),
            "freshness_work": [item.to_dict() for item in self.freshness_work],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "authority_root_cid": self.authority_root_cid}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SnapshotAuthority":
        _require_schema(value, SNAPSHOT_AUTHORITY_SCHEMA, "snapshot_authority")
        integration = value.get("integration_checkout")
        swissknife = value.get("swissknife_checkout")
        gitlink = value.get("integration_gitlink")
        freshness = value.get("freshness_work")
        if (
            not isinstance(integration, Mapping)
            or not isinstance(swissknife, Mapping)
            or not isinstance(gitlink, Mapping)
            or not isinstance(freshness, list)
        ):
            raise RepositoryAuthorityError(
                "snapshot-authority document has invalid structure",
                reason_code="invalid_authority_document",
            )
        raw_override = value.get("reviewed_override")
        if raw_override is not None and not isinstance(raw_override, Mapping):
            raise ReviewedEvidenceError(
                "reviewed_override must be an object or null",
                reason_code="reviewed_evidence_missing",
            )
        if any(not isinstance(item, Mapping) for item in freshness):
            raise RepositoryAuthorityError(
                "freshness_work entries must be objects",
                reason_code="invalid_authority_document",
            )
        return cls(
            integration_checkout=CheckoutSnapshotBinding.from_dict(integration),
            swissknife_checkout=CheckoutSnapshotBinding.from_dict(swissknife),
            integration_gitlink_path=str(gitlink.get("path", "")),
            integration_gitlink_commit=str(gitlink.get("commit", "")),
            authority_source=AuthoritySource(str(value.get("authority_source", ""))),
            program_commit=str(value.get("program_commit", "")),
            freshness_work=tuple(
                FreshnessWork.from_dict(item) for item in freshness
            ),
            reviewed_override=(
                ReviewedAuthorityOverride.from_dict(raw_override)
                if isinstance(raw_override, Mapping)
                else None
            ),
            authority_root_cid=str(value.get("authority_root_cid", "")),
        )


def build_repository_authority(
    integration_root: str | os.PathLike[str],
    *,
    swissknife_checkout: str | os.PathLike[str] | None = None,
    gitlink_path: str = "swissknife",
    reviewed_override: ReviewedAuthorityOverride | None = None,
) -> SnapshotAuthority:
    """Observe both repositories and select the canonical program authority."""

    integration = Path(integration_root).resolve()
    integration_binding = bind_checkout(
        integration,
        checkout_id="integration",
        repository_path="integration",
    )
    if not integration_binding.present:
        raise RepositoryAuthorityError(
            "integration repository checkout is missing",
            reason_code="integration_checkout_missing",
        )
    gitlink_commit = _read_gitlink(integration, gitlink_path)
    child_root = (
        Path(swissknife_checkout).resolve()
        if swissknife_checkout is not None
        else integration.joinpath(*PurePosixPath(gitlink_path).parts)
    )
    swissknife_binding = bind_checkout(
        child_root,
        checkout_id="swissknife",
        repository_path=gitlink_path,
    )

    if reviewed_override is None:
        source = AuthoritySource.INTEGRATION_GITLINK
        program_commit = gitlink_commit
    else:
        if reviewed_override.supersedes_gitlink_commit != gitlink_commit:
            raise ReviewedEvidenceError(
                "reviewed evidence does not supersede the indexed gitlink",
                reason_code="reviewed_evidence_scope_mismatch",
                details={
                    "indexed_gitlink_commit": gitlink_commit,
                    "evidence_gitlink_commit": (
                        reviewed_override.supersedes_gitlink_commit
                    ),
                },
            )
        source = AuthoritySource.REVIEWED_EVIDENCE
        program_commit = reviewed_override.program_commit

    freshness = _freshness_work(child_root, swissknife_binding, program_commit)
    return SnapshotAuthority(
        integration_checkout=integration_binding,
        swissknife_checkout=swissknife_binding,
        integration_gitlink_path=gitlink_path,
        integration_gitlink_commit=gitlink_commit,
        authority_source=source,
        program_commit=program_commit,
        reviewed_override=reviewed_override,
        freshness_work=freshness,
    )


@dataclass(frozen=True, slots=True)
class AuthorityBoundReference:
    """A cache, proof, or artifact key bound to one exact authority root."""

    kind: AuthorityJoinKind
    reference_id: str
    payload_cid: str
    authority_root_cid: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AuthorityJoinKind):
            object.__setattr__(self, "kind", AuthorityJoinKind(self.kind))
        _require_nonempty(self.reference_id, "reference_id")
        _require_cid(self.payload_cid, "payload_cid")
        _require_cid(self.authority_root_cid, "authority_root_cid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": AUTHORITY_BOUND_REFERENCE_SCHEMA,
            "kind": self.kind.value,
            "reference_id": self.reference_id,
            "payload_cid": self.payload_cid,
            "authority_root_cid": self.authority_root_cid,
        }


def bind_authority_reference(
    authority: SnapshotAuthority,
    *,
    kind: AuthorityJoinKind | str,
    reference_id: str,
    payload: Mapping[str, Any],
) -> AuthorityBoundReference:
    """Create a consumer reference bound to the authority's exact root."""

    return AuthorityBoundReference(
        kind=AuthorityJoinKind(kind),
        reference_id=reference_id,
        payload_cid=_strict_cid(payload),
        authority_root_cid=authority.authority_root_cid,
    )


def join_authority_bound_references(
    authority: SnapshotAuthority,
    references: Iterable[AuthorityBoundReference],
) -> tuple[AuthorityBoundReference, ...]:
    """Validate a cross-consumer join, rejecting every foreign authority root."""

    accepted = tuple(references)
    for reference in accepted:
        if not isinstance(reference, AuthorityBoundReference):
            raise RepositoryAuthorityError(
                "authority join accepts only typed references",
                reason_code="untyped_authority_reference",
            )
        if reference.authority_root_cid != authority.authority_root_cid:
            raise AuthorityRootMismatchError(
                "authority-bound reference belongs to a different root",
                reason_code="authority_root_mismatch",
                details={
                    "kind": reference.kind.value,
                    "reference_id": reference.reference_id,
                    "expected": authority.authority_root_cid,
                    "actual": reference.authority_root_cid,
                },
            )
    return accepted


def load_snapshot_authority(
    path: str | os.PathLike[str],
) -> SnapshotAuthority:
    """Load and fully revalidate a checked-in authority document."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RepositoryAuthorityError(
            "could not load snapshot-authority document",
            reason_code="invalid_authority_document",
            details={"path": os.fspath(path), "cause": repr(exc)},
        ) from exc
    if not isinstance(value, Mapping):
        raise RepositoryAuthorityError(
            "snapshot-authority document must be a JSON object",
            reason_code="invalid_authority_document",
        )
    return SnapshotAuthority.from_dict(value)


def dump_snapshot_authority(authority: SnapshotAuthority) -> str:
    """Return stable, newline-terminated JSON for durable authority state."""

    return json.dumps(
        authority.to_dict(),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"
