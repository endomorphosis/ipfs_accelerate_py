"""Typed authority for externally executed objective completion.

The supervisor may discover implementation markers in repository text, but an
operational run happened outside that repository.  This module deliberately
keeps those two facts separate.  It accepts only content identities and
bounded public metadata, validates them against the current clean Git source,
and adapts a valid receipt into :class:`CompletionEvidence`.

No artifact path, label, manifest, proof-obligation text, holdout material, or
external content is accepted by the schema.  Gitlink locations are represented
by opaque CIDs derived during source inspection rather than persisted paths.
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..goal_completion import (
    DEFAULT_CLOCK_SKEW_SECONDS,
    DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
    CompletionEvidence,
)
from ..task_sources.task_identity import canonical_content_cid


EXTERNAL_GITLINK_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.external-gitlink-identity.v1"
)
EXTERNAL_SOURCE_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.external-source-identity.v1"
)
EXTERNAL_ARTIFACT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.external-artifact-identity.v1"
)
EXTERNAL_COMPLETION_REQUIREMENT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.external-completion-requirement.v1"
)
EXTERNAL_COMPLETION_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.external-operational-completion.v1"
)
EXTERNAL_COMPLETION_AUTHORITY_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.external-completion-authority.v1"
)
EXTERNAL_COMPLETION_VALIDATION_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.external-completion-validation.v1"
)
EXTERNAL_COMPLETION_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.external-completion-evidence.v1"
)

_GIT_OBJECT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_RECEIPT_STATUSES = frozenset(
    {"cancelled", "completed", "failed", "partial", "timed_out"}
)


def HSSLEV2398A61() -> str:
    """Return implementation evidence for typed external completion."""

    return (
        "canonical CID-bound external operational completion authority "
        "with clean recursive Git source and sticky fail-closed governance"
    )


def _nonempty(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _strict_keys(
    payload: Mapping[str, Any],
    *,
    allowed: Iterable[str],
    context: str,
) -> None:
    allowed_keys = set(allowed)
    unknown = sorted(str(key) for key in payload if key not in allowed_keys)
    if unknown:
        # Do not echo an untrusted field name: the key itself could contain a
        # path, secret, or external payload fragment.
        raise ValueError(f"{context} contains unsupported fields")


def _encode_varint(value: int) -> bytes:
    if type(value) is not int or value < 0:
        raise ValueError("CID varint must be a non-negative integer")
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _read_varint(raw: bytes, offset: int) -> tuple[int, int]:
    start = offset
    value = 0
    shift = 0
    while offset < len(raw):
        byte = raw[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if raw[start:offset] != _encode_varint(value):
                raise ValueError("non-canonical CID varint")
            return value, offset
        shift += 7
        if shift > 63:
            break
    raise ValueError("malformed CID varint")


def validate_cid(value: Any, *, field_name: str) -> str:
    """Return one canonical CIDv1/base32 identity or raise ``ValueError``."""

    text = _nonempty(value, field_name=field_name)
    if text != text.lower() or not re.fullmatch(r"b[a-z2-7]+", text):
        raise ValueError(f"{field_name} must be a lowercase base32 CIDv1")
    body = text[1:]
    padding = "=" * ((8 - len(body) % 8) % 8)
    try:
        raw = base64.b32decode(body.upper() + padding, casefold=True)
        canonical = (
            "b"
            + base64.b32encode(raw)
            .decode("ascii")
            .rstrip("=")
            .lower()
        )
        if canonical != text:
            raise ValueError("non-canonical CID base32 encoding")
        version, offset = _read_varint(raw, 0)
        codec, offset = _read_varint(raw, offset)
        hash_code, offset = _read_varint(raw, offset)
        digest_length, offset = _read_varint(raw, offset)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field_name} must be a valid CIDv1") from exc
    if (
        version != 1
        or codec <= 0
        or hash_code <= 0
        or digest_length <= 0
        or offset + digest_length != len(raw)
    ):
        raise ValueError(f"{field_name} must be a complete CIDv1 multihash")
    return text


def _git_object(value: Any, *, field_name: str) -> str:
    text = _nonempty(value, field_name=field_name).lower()
    if not _GIT_OBJECT_RE.fullmatch(text):
        raise ValueError(f"{field_name} must be a full Git object identity")
    return text


def _timestamp(
    value: datetime | str | None,
    *,
    field_name: str,
    required: bool = True,
) -> datetime | None:
    if value in (None, ""):
        if required:
            raise ValueError(f"{field_name} must be present")
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _timestamp_text(value: datetime | None) -> str:
    return value.isoformat() if value is not None else ""


def _now(value: datetime | str | None) -> datetime:
    return _timestamp(
        value if value is not None else datetime.now(timezone.utc),
        field_name="now",
    ) or datetime.now(timezone.utc)


@dataclass(frozen=True)
class ExternalGitlinkIdentity:
    """Opaque recursive gitlink identity without a persisted checkout path."""

    gitlink_id: str
    commit: str
    tree: str
    parent_gitlink_id: str = ""
    depth: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gitlink_id",
            validate_cid(self.gitlink_id, field_name="gitlink_id"),
        )
        object.__setattr__(
            self,
            "commit",
            _git_object(self.commit, field_name="gitlink commit"),
        )
        object.__setattr__(
            self,
            "tree",
            _git_object(self.tree, field_name="gitlink tree"),
        )
        parent = str(self.parent_gitlink_id or "").strip()
        if parent:
            parent = validate_cid(parent, field_name="parent_gitlink_id")
        depth = int(self.depth)
        if depth < 0:
            raise ValueError("gitlink depth must be non-negative")
        if depth == 0 and parent:
            raise ValueError("top-level gitlinks cannot name a parent_gitlink_id")
        if depth > 0 and not parent:
            raise ValueError("nested gitlinks require parent_gitlink_id")
        object.__setattr__(self, "parent_gitlink_id", parent)
        object.__setattr__(self, "depth", depth)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXTERNAL_GITLINK_SCHEMA,
            "gitlink_id": self.gitlink_id,
            "commit": self.commit,
            "tree": self.tree,
            "parent_gitlink_id": self.parent_gitlink_id,
            "depth": self.depth,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExternalGitlinkIdentity":
        _strict_keys(
            payload,
            allowed={
                "schema",
                "gitlink_id",
                "commit",
                "tree",
                "parent_gitlink_id",
                "depth",
            },
            context="external gitlink identity",
        )
        schema = str(payload.get("schema") or EXTERNAL_GITLINK_SCHEMA)
        if schema != EXTERNAL_GITLINK_SCHEMA:
            raise ValueError("unsupported external gitlink identity schema")
        return cls(
            gitlink_id=payload.get("gitlink_id", ""),
            commit=payload.get("commit", ""),
            tree=payload.get("tree", ""),
            parent_gitlink_id=payload.get("parent_gitlink_id", ""),
            depth=int(payload.get("depth", 0)),
        )


@dataclass(frozen=True)
class ExternalSourceIdentity:
    """Clean outer Git identity and complete opaque recursive gitlink map."""

    outer_commit: str
    outer_tree: str
    recursive_gitlinks: tuple[ExternalGitlinkIdentity, ...] = ()
    clean: bool = True
    recursive_gitlinks_complete: bool = True
    submodule_map_cid: str = ""
    source_identity_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outer_commit",
            _git_object(self.outer_commit, field_name="outer_commit"),
        )
        object.__setattr__(
            self,
            "outer_tree",
            _git_object(self.outer_tree, field_name="outer_tree"),
        )
        if not isinstance(self.clean, bool):
            raise ValueError("source clean must be a boolean")
        if not isinstance(self.recursive_gitlinks_complete, bool):
            raise ValueError("recursive_gitlinks_complete must be a boolean")
        entries = tuple(
            item
            if isinstance(item, ExternalGitlinkIdentity)
            else ExternalGitlinkIdentity.from_dict(item)
            for item in self.recursive_gitlinks
        )
        entries = tuple(sorted(entries, key=lambda item: item.gitlink_id))
        ids = [item.gitlink_id for item in entries]
        if len(ids) != len(set(ids)):
            raise ValueError("recursive gitlink identities must be unique")
        by_id = {item.gitlink_id: item for item in entries}
        for item in entries:
            if not item.parent_gitlink_id:
                continue
            parent = by_id.get(item.parent_gitlink_id)
            if parent is None or item.depth != parent.depth + 1:
                raise ValueError(
                    "recursive gitlink parent identity/depth is inconsistent"
                )
        object.__setattr__(self, "recursive_gitlinks", entries)
        map_material = {
            "schema": EXTERNAL_SOURCE_SCHEMA + "/submodule-map",
            "entries": [item.to_dict() for item in entries],
        }
        map_cid = canonical_content_cid(map_material)
        supplied_map_cid = str(self.submodule_map_cid or "").strip()
        if supplied_map_cid:
            supplied_map_cid = validate_cid(
                supplied_map_cid,
                field_name="submodule_map_cid",
            )
            if supplied_map_cid != map_cid:
                raise ValueError(
                    "submodule_map_cid does not match recursive_gitlinks"
                )
        object.__setattr__(self, "submodule_map_cid", map_cid)
        identity_material = {
            "schema": EXTERNAL_SOURCE_SCHEMA,
            "outer_commit": self.outer_commit,
            "outer_tree": self.outer_tree,
            "clean": self.clean,
            "recursive_gitlinks_complete": self.recursive_gitlinks_complete,
            "submodule_map_cid": map_cid,
        }
        identity_cid = canonical_content_cid(identity_material)
        supplied_identity_cid = str(self.source_identity_cid or "").strip()
        if supplied_identity_cid:
            supplied_identity_cid = validate_cid(
                supplied_identity_cid,
                field_name="source_identity_cid",
            )
            if supplied_identity_cid != identity_cid:
                raise ValueError(
                    "source_identity_cid does not match source identity"
                )
        object.__setattr__(self, "source_identity_cid", identity_cid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXTERNAL_SOURCE_SCHEMA,
            "outer_commit": self.outer_commit,
            "outer_tree": self.outer_tree,
            "clean": self.clean,
            "recursive_gitlinks_complete": self.recursive_gitlinks_complete,
            "recursive_gitlinks": [
                item.to_dict() for item in self.recursive_gitlinks
            ],
            "submodule_map_cid": self.submodule_map_cid,
            "source_identity_cid": self.source_identity_cid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExternalSourceIdentity":
        _strict_keys(
            payload,
            allowed={
                "schema",
                "outer_commit",
                "outer_tree",
                "clean",
                "recursive_gitlinks_complete",
                "recursive_gitlinks",
                "submodule_map_cid",
                "source_identity_cid",
            },
            context="external source identity",
        )
        schema = str(payload.get("schema") or EXTERNAL_SOURCE_SCHEMA)
        if schema != EXTERNAL_SOURCE_SCHEMA:
            raise ValueError("unsupported external source identity schema")
        raw_gitlinks = payload.get("recursive_gitlinks", ())
        if not isinstance(raw_gitlinks, list):
            raise ValueError("recursive_gitlinks must be a list")
        if any(not isinstance(item, Mapping) for item in raw_gitlinks):
            raise ValueError("recursive_gitlinks entries must be objects")
        return cls(
            outer_commit=payload.get("outer_commit", ""),
            outer_tree=payload.get("outer_tree", ""),
            clean=payload.get("clean", False),
            recursive_gitlinks_complete=payload.get(
                "recursive_gitlinks_complete",
                False,
            ),
            recursive_gitlinks=tuple(
                ExternalGitlinkIdentity.from_dict(item)
                for item in raw_gitlinks
                if isinstance(item, Mapping)
            ),
            submodule_map_cid=payload.get("submodule_map_cid", ""),
            source_identity_cid=payload.get("source_identity_cid", ""),
        )


@dataclass(frozen=True)
class ExternalSourceInspection:
    """Current source identity plus path-free inspection diagnostics."""

    identity: ExternalSourceIdentity
    reason_codes: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return bool(
            self.identity.clean
            and self.identity.recursive_gitlinks_complete
            and not self.reason_codes
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_identity_cid": self.identity.source_identity_cid,
            "outer_commit": self.identity.outer_commit,
            "outer_tree": self.identity.outer_tree,
            "submodule_map_cid": self.identity.submodule_map_cid,
            "clean": self.identity.clean,
            "recursive_gitlinks_complete": (
                self.identity.recursive_gitlinks_complete
            ),
            "reason_codes": list(self.reason_codes),
        }


def _git(
    repo_root: Path,
    *arguments: str,
    binary: bool = False,
) -> tuple[int, str | bytes]:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            text=not binary,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 1, b"" if binary else ""
    output: str | bytes = completed.stdout
    if not binary:
        output = str(output).strip()
    return completed.returncode, output


def _status_is_clean(
    repo_root: Path,
) -> tuple[bool, bool]:
    arguments = [
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    ]
    returncode, output = _git(repo_root, *arguments, binary=True)
    return returncode == 0, returncode == 0 and not bool(output)


def _gitlinks_at_commit(
    repo_root: Path,
    commit: str,
) -> tuple[bool, list[tuple[str, str]]]:
    returncode, output = _git(
        repo_root,
        "ls-tree",
        "-r",
        "-z",
        commit,
        binary=True,
    )
    if returncode != 0 or not isinstance(output, bytes):
        return False, []
    rows: list[tuple[str, str]] = []
    for raw in output.split(b"\0"):
        if not raw or b"\t" not in raw:
            continue
        metadata, raw_path = raw.split(b"\t", 1)
        parts = metadata.split()
        if len(parts) != 3 or parts[0] != b"160000":
            continue
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        commit_id = parts[2].decode("ascii", errors="replace")
        rows.append((relative, commit_id))
    return True, sorted(rows)


def inspect_external_source(
    repo_root: Path,
    *,
    objective_path: Path | None = None,
    max_depth: int = 16,
) -> ExternalSourceInspection:
    """Inspect only Git metadata and return an identity-only source snapshot.

    Every tracked and untracked path participates in the outer cleanliness
    check, including the objective heap.  ``objective_path`` is accepted to
    verify that reconciliation is scoped to this repository, never as a
    cleanliness exclusion.  Submodule contents are never opened; recursive
    identities come exclusively from Git commits, trees, status, and gitlink
    entries.
    """

    root = repo_root.resolve()
    top_status, top_output = _git(root, "rev-parse", "--show-toplevel")
    if top_status != 0 or not str(top_output):
        raise ValueError("external completion requires a Git repository")
    top = Path(str(top_output)).resolve()
    head_status, head_output = _git(top, "rev-parse", "HEAD")
    tree_status, tree_output = _git(top, "rev-parse", "HEAD^{tree}")
    if head_status != 0 or tree_status != 0:
        raise ValueError("external completion requires a committed Git source")
    outer_commit = _git_object(head_output, field_name="outer_commit")
    outer_tree = _git_object(tree_output, field_name="outer_tree")

    if objective_path is not None:
        try:
            objective_path.resolve().relative_to(top)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                "external completion objective_path must be inside the repository"
            ) from exc

    reasons: list[str] = []
    status_ok, outer_clean = _status_is_clean(top)
    if not status_ok:
        reasons.append("outer_status_unavailable")
    elif not outer_clean:
        reasons.append("current_source_dirty")

    entries: list[ExternalGitlinkIdentity] = []
    complete = True
    clean = bool(status_ok and outer_clean)
    visited: set[tuple[str, str]] = set()

    def inspect_repo(
        checkout: Path,
        commit: str,
        *,
        parent_gitlink_id: str = "",
        depth: int = 0,
    ) -> None:
        nonlocal clean, complete
        if depth > max(0, int(max_depth)):
            complete = False
            reasons.append("recursive_gitlink_depth_exceeded")
            return
        repository_key = (str(checkout.resolve()), commit)
        if repository_key in visited:
            complete = False
            reasons.append("recursive_gitlink_cycle")
            return
        visited.add(repository_key)
        listed, gitlinks = _gitlinks_at_commit(checkout, commit)
        if not listed:
            complete = False
            reasons.append("recursive_gitlink_map_unavailable")
            return
        for relative, recorded_commit in gitlinks:
            link_id = canonical_content_cid(
                {
                    "schema": EXTERNAL_GITLINK_SCHEMA + "/location",
                    "parent_commit": commit,
                    "location": relative,
                }
            )
            candidate = checkout / relative
            try:
                resolved_candidate = candidate.resolve()
                resolved_candidate.relative_to(top)
            except (OSError, RuntimeError, ValueError):
                complete = False
                clean = False
                reasons.append("gitlink_checkout_outside_repository")
                continue
            child_head_status, child_head_output = _git(
                candidate,
                "rev-parse",
                "HEAD",
            )
            child_tree_status, child_tree_output = _git(
                candidate,
                "rev-parse",
                "HEAD^{tree}",
            )
            if child_head_status != 0 or child_tree_status != 0:
                complete = False
                clean = False
                reasons.append("gitlink_checkout_unavailable")
                continue
            child_head = _git_object(
                child_head_output,
                field_name="gitlink checkout commit",
            )
            child_tree = _git_object(
                child_tree_output,
                field_name="gitlink checkout tree",
            )
            child_status_ok, child_clean = _status_is_clean(candidate)
            if not child_status_ok:
                complete = False
                clean = False
                reasons.append("gitlink_status_unavailable")
            elif not child_clean:
                clean = False
                reasons.append("gitlink_checkout_dirty")
            if child_head != recorded_commit:
                clean = False
                reasons.append("gitlink_head_mismatch")
            entries.append(
                ExternalGitlinkIdentity(
                    gitlink_id=link_id,
                    commit=recorded_commit,
                    tree=child_tree,
                    parent_gitlink_id=parent_gitlink_id,
                    depth=depth,
                )
            )
            inspect_repo(
                candidate,
                child_head,
                parent_gitlink_id=link_id,
                depth=depth + 1,
            )

    inspect_repo(top, outer_commit)
    identity = ExternalSourceIdentity(
        outer_commit=outer_commit,
        outer_tree=outer_tree,
        recursive_gitlinks=tuple(entries),
        clean=clean,
        recursive_gitlinks_complete=complete,
    )
    return ExternalSourceInspection(
        identity=identity,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


@dataclass(frozen=True)
class ExternalArtifactIdentity:
    """One required artifact slot and its immutable external content CID."""

    artifact_id: str
    artifact_cid: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_id",
            validate_cid(self.artifact_id, field_name="artifact_id"),
        )
        object.__setattr__(
            self,
            "artifact_cid",
            validate_cid(self.artifact_cid, field_name="artifact_cid"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "schema": EXTERNAL_ARTIFACT_SCHEMA,
            "artifact_id": self.artifact_id,
            "artifact_cid": self.artifact_cid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExternalArtifactIdentity":
        _strict_keys(
            payload,
            allowed={"schema", "artifact_id", "artifact_cid"},
            context="external artifact identity",
        )
        schema = str(payload.get("schema") or EXTERNAL_ARTIFACT_SCHEMA)
        if schema != EXTERNAL_ARTIFACT_SCHEMA:
            raise ValueError("unsupported external artifact identity schema")
        return cls(
            artifact_id=payload.get("artifact_id", ""),
            artifact_cid=payload.get("artifact_cid", ""),
        )


@dataclass(frozen=True)
class ExternalCompletionRequirement:
    """Supervisor-owned expected identities for one goal/evidence binding."""

    goal_id: str
    evidence_term: str
    source_identity_cid: str
    run_plan_cid: str
    parent_ledger_cid: str
    required_artifact_ids: tuple[str, ...]
    expected_producer_id: str
    expected_validator_id: str
    requirement_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "goal_id",
            _nonempty(self.goal_id, field_name="requirement goal_id"),
        )
        object.__setattr__(
            self,
            "evidence_term",
            _nonempty(
                self.evidence_term,
                field_name="requirement evidence_term",
            ),
        )
        for name in (
            "source_identity_cid",
            "run_plan_cid",
            "parent_ledger_cid",
            "expected_producer_id",
            "expected_validator_id",
        ):
            object.__setattr__(
                self,
                name,
                validate_cid(getattr(self, name), field_name=name),
            )
        artifact_ids = tuple(
            sorted(
                validate_cid(item, field_name="required_artifact_id")
                for item in self.required_artifact_ids
            )
        )
        if not artifact_ids:
            raise ValueError(
                "external completion requires at least one artifact identity"
            )
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("required artifact identities must be unique")
        if self.expected_producer_id == self.expected_validator_id:
            raise ValueError(
                "external producer and independent validator must differ"
            )
        object.__setattr__(self, "required_artifact_ids", artifact_ids)
        requirement_cid = canonical_content_cid(self.identity_payload())
        supplied = str(self.requirement_cid or "").strip()
        if supplied:
            supplied = validate_cid(supplied, field_name="requirement_cid")
            if supplied != requirement_cid:
                raise ValueError(
                    "requirement_cid does not match completion requirement"
                )
        object.__setattr__(self, "requirement_cid", requirement_cid)

    @property
    def binding(self) -> tuple[str, str]:
        return self.goal_id, self.evidence_term

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": EXTERNAL_COMPLETION_REQUIREMENT_SCHEMA,
            "goal_id": self.goal_id,
            "evidence_term": self.evidence_term,
            "source_identity_cid": self.source_identity_cid,
            "run_plan_cid": self.run_plan_cid,
            "parent_ledger_cid": self.parent_ledger_cid,
            "required_artifact_ids": list(self.required_artifact_ids),
            "expected_producer_id": self.expected_producer_id,
            "expected_validator_id": self.expected_validator_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "requirement_cid": self.requirement_cid}

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "ExternalCompletionRequirement":
        _strict_keys(
            payload,
            allowed={
                "schema",
                "goal_id",
                "evidence_term",
                "source_identity_cid",
                "run_plan_cid",
                "parent_ledger_cid",
                "required_artifact_ids",
                "expected_producer_id",
                "expected_validator_id",
                "requirement_cid",
            },
            context="external completion requirement",
        )
        schema = str(
            payload.get("schema") or EXTERNAL_COMPLETION_REQUIREMENT_SCHEMA
        )
        if schema != EXTERNAL_COMPLETION_REQUIREMENT_SCHEMA:
            raise ValueError("unsupported external completion requirement schema")
        artifact_ids = payload.get("required_artifact_ids", ())
        if not isinstance(artifact_ids, list):
            raise ValueError("required_artifact_ids must be a list")
        return cls(
            goal_id=payload.get("goal_id", ""),
            evidence_term=payload.get("evidence_term", ""),
            source_identity_cid=payload.get("source_identity_cid", ""),
            run_plan_cid=payload.get("run_plan_cid", ""),
            parent_ledger_cid=payload.get("parent_ledger_cid", ""),
            required_artifact_ids=tuple(artifact_ids),
            expected_producer_id=payload.get("expected_producer_id", ""),
            expected_validator_id=payload.get("expected_validator_id", ""),
            requirement_cid=payload.get("requirement_cid", ""),
        )


@dataclass(frozen=True)
class ExternalOperationalCompletionReceipt:
    """Content-addressed external completion assertion for one evidence term."""

    goal_id: str
    evidence_term: str
    source: ExternalSourceIdentity
    run_plan_cid: str
    parent_ledger_cid: str
    artifacts: tuple[ExternalArtifactIdentity, ...]
    producer_id: str
    validator_id: str
    validator_receipt_cid: str
    observed_at: datetime | str
    fresh_until: datetime | str | None = None
    status: str = "completed"
    receipt_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "goal_id",
            _nonempty(self.goal_id, field_name="receipt goal_id"),
        )
        object.__setattr__(
            self,
            "evidence_term",
            _nonempty(self.evidence_term, field_name="receipt evidence_term"),
        )
        source = (
            self.source
            if isinstance(self.source, ExternalSourceIdentity)
            else ExternalSourceIdentity.from_dict(self.source)
        )
        object.__setattr__(self, "source", source)
        for name in (
            "run_plan_cid",
            "parent_ledger_cid",
            "producer_id",
            "validator_id",
            "validator_receipt_cid",
        ):
            object.__setattr__(
                self,
                name,
                validate_cid(getattr(self, name), field_name=name),
            )
        artifacts = tuple(
            item
            if isinstance(item, ExternalArtifactIdentity)
            else ExternalArtifactIdentity.from_dict(item)
            for item in self.artifacts
        )
        artifacts = tuple(sorted(artifacts, key=lambda item: item.artifact_id))
        if not artifacts:
            raise ValueError("external receipt must contain artifact identities")
        artifact_ids = [item.artifact_id for item in artifacts]
        artifact_cids = [item.artifact_cid for item in artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("external receipt contains duplicate artifact_id")
        if len(artifact_cids) != len(set(artifact_cids)):
            raise ValueError("external receipt contains duplicate artifact_cid")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(
            self,
            "observed_at",
            _timestamp(self.observed_at, field_name="observed_at"),
        )
        object.__setattr__(
            self,
            "fresh_until",
            _timestamp(
                self.fresh_until,
                field_name="fresh_until",
                required=False,
            ),
        )
        status = _nonempty(self.status, field_name="receipt status").lower()
        if status not in _RECEIPT_STATUSES:
            raise ValueError("unsupported external receipt status")
        object.__setattr__(self, "status", status)
        receipt_cid = canonical_content_cid(self.identity_payload())
        supplied = str(self.receipt_cid or "").strip()
        if supplied:
            supplied = validate_cid(supplied, field_name="receipt_cid")
            if supplied != receipt_cid:
                raise ValueError(
                    "receipt_cid does not match external completion receipt"
                )
        object.__setattr__(self, "receipt_cid", receipt_cid)

    @property
    def binding(self) -> tuple[str, str]:
        return self.goal_id, self.evidence_term

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": EXTERNAL_COMPLETION_RECEIPT_SCHEMA,
            "goal_id": self.goal_id,
            "evidence_term": self.evidence_term,
            "source": self.source.to_dict(),
            "run_plan_cid": self.run_plan_cid,
            "parent_ledger_cid": self.parent_ledger_cid,
            "artifacts": [item.to_dict() for item in self.artifacts],
            "producer_id": self.producer_id,
            "validator_id": self.validator_id,
            "validator_receipt_cid": self.validator_receipt_cid,
            "observed_at": _timestamp_text(self.observed_at),
            "fresh_until": _timestamp_text(self.fresh_until),
            "status": self.status,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "receipt_cid": self.receipt_cid}

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "ExternalOperationalCompletionReceipt":
        _strict_keys(
            payload,
            allowed={
                "schema",
                "goal_id",
                "evidence_term",
                "source",
                "run_plan_cid",
                "parent_ledger_cid",
                "artifacts",
                "producer_id",
                "validator_id",
                "validator_receipt_cid",
                "observed_at",
                "fresh_until",
                "status",
                "receipt_cid",
            },
            context="external operational completion receipt",
        )
        schema = str(
            payload.get("schema") or EXTERNAL_COMPLETION_RECEIPT_SCHEMA
        )
        if schema != EXTERNAL_COMPLETION_RECEIPT_SCHEMA:
            raise ValueError(
                "unsupported external operational completion receipt schema"
            )
        source = payload.get("source")
        artifacts = payload.get("artifacts")
        if not isinstance(source, Mapping):
            raise ValueError("external receipt source must be an object")
        if not isinstance(artifacts, list):
            raise ValueError("external receipt artifacts must be a list")
        if any(not isinstance(item, Mapping) for item in artifacts):
            raise ValueError("external receipt artifact entries must be objects")
        return cls(
            goal_id=payload.get("goal_id", ""),
            evidence_term=payload.get("evidence_term", ""),
            source=ExternalSourceIdentity.from_dict(source),
            run_plan_cid=payload.get("run_plan_cid", ""),
            parent_ledger_cid=payload.get("parent_ledger_cid", ""),
            artifacts=tuple(
                ExternalArtifactIdentity.from_dict(item)
                for item in artifacts
                if isinstance(item, Mapping)
            ),
            producer_id=payload.get("producer_id", ""),
            validator_id=payload.get("validator_id", ""),
            validator_receipt_cid=payload.get("validator_receipt_cid", ""),
            observed_at=payload.get("observed_at"),
            fresh_until=payload.get("fresh_until"),
            status=payload.get("status", ""),
            receipt_cid=payload.get("receipt_cid", ""),
        )


@dataclass(frozen=True)
class ExternalCompletionAuthority:
    """Explicit trusted requirements and untrusted external receipt set."""

    requirements: tuple[ExternalCompletionRequirement, ...]
    receipts: tuple[ExternalOperationalCompletionReceipt, ...] = ()
    authority_cid: str = ""

    def __post_init__(self) -> None:
        requirements = tuple(
            item
            if isinstance(item, ExternalCompletionRequirement)
            else ExternalCompletionRequirement.from_dict(item)
            for item in self.requirements
        )
        requirements = tuple(
            sorted(requirements, key=lambda item: item.binding)
        )
        if not requirements:
            raise ValueError(
                "external completion authority requires at least one requirement"
            )
        requirement_bindings = [item.binding for item in requirements]
        if len(requirement_bindings) != len(set(requirement_bindings)):
            raise ValueError("external completion requirements must be unique")
        receipts = tuple(
            item
            if isinstance(item, ExternalOperationalCompletionReceipt)
            else ExternalOperationalCompletionReceipt.from_dict(item)
            for item in self.receipts
        )
        receipts = tuple(sorted(receipts, key=lambda item: item.receipt_cid))
        receipt_cids = [item.receipt_cid for item in receipts]
        if len(receipt_cids) != len(set(receipt_cids)):
            raise ValueError("external completion receipt CIDs must be unique")
        requirement_binding_set = set(requirement_bindings)
        if any(
            item.binding not in requirement_binding_set
            for item in receipts
        ):
            raise ValueError(
                "external completion receipt binding has no matching requirement"
            )
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(self, "receipts", receipts)
        authority_cid = canonical_content_cid(self.identity_payload())
        supplied = str(self.authority_cid or "").strip()
        if supplied:
            supplied = validate_cid(supplied, field_name="authority_cid")
            if supplied != authority_cid:
                raise ValueError(
                    "authority_cid does not match external completion authority"
                )
        object.__setattr__(self, "authority_cid", authority_cid)

    @property
    def governed_goal_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted({item.goal_id for item in self.requirements})
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": EXTERNAL_COMPLETION_AUTHORITY_SCHEMA,
            "requirements": [item.to_dict() for item in self.requirements],
            "receipts": [item.to_dict() for item in self.receipts],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "authority_cid": self.authority_cid}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExternalCompletionAuthority":
        _strict_keys(
            payload,
            allowed={"schema", "requirements", "receipts", "authority_cid"},
            context="external completion authority",
        )
        schema = str(
            payload.get("schema") or EXTERNAL_COMPLETION_AUTHORITY_SCHEMA
        )
        if schema != EXTERNAL_COMPLETION_AUTHORITY_SCHEMA:
            raise ValueError("unsupported external completion authority schema")
        requirements = payload.get("requirements")
        receipts = payload.get("receipts", [])
        if not isinstance(requirements, list):
            raise ValueError(
                "external completion authority requirements must be a list"
            )
        if not isinstance(receipts, list):
            raise ValueError(
                "external completion authority receipts must be a list"
            )
        if any(not isinstance(item, Mapping) for item in requirements):
            raise ValueError(
                "external completion authority requirement entries must be objects"
            )
        if any(not isinstance(item, Mapping) for item in receipts):
            raise ValueError(
                "external completion authority receipt entries must be objects"
            )
        return cls(
            requirements=tuple(
                ExternalCompletionRequirement.from_dict(item)
                for item in requirements
                if isinstance(item, Mapping)
            ),
            receipts=tuple(
                ExternalOperationalCompletionReceipt.from_dict(item)
                for item in receipts
                if isinstance(item, Mapping)
            ),
            authority_cid=payload.get("authority_cid", ""),
        )


def load_external_completion_authority(
    path: Path,
) -> ExternalCompletionAuthority:
    """Load one explicit identity-only authority file, failing closed."""

    if not path.exists() or not path.is_file():
        raise ValueError(
            "external completion receipt path must name an existing JSON file"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "external completion receipt path must contain valid JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("external completion authority must be a JSON object")
    return ExternalCompletionAuthority.from_dict(payload)


@dataclass(frozen=True)
class ExternalReceiptValidationResult:
    """Identity-only validation result safe for supervisor persistence."""

    goal_id: str
    evidence_term: str
    valid: bool
    reason_codes: tuple[str, ...] = ()
    receipt_cid: str = ""
    requirement_cid: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXTERNAL_COMPLETION_VALIDATION_SCHEMA,
            "goal_id": self.goal_id,
            "evidence_term": self.evidence_term,
            "valid": self.valid,
            "reason_codes": list(self.reason_codes),
            "receipt_cid": self.receipt_cid,
            "requirement_cid": self.requirement_cid,
        }


@dataclass(frozen=True)
class ExternalCompletionEvaluation:
    """Validated evidence records plus bounded authority diagnostics."""

    authority_cid: str
    governed_goal_ids: tuple[str, ...]
    evidence_records: Mapping[str, tuple[CompletionEvidence, ...]]
    results: tuple[ExternalReceiptValidationResult, ...]
    source_inspection: ExternalSourceInspection

    @property
    def valid_receipt_cids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                item.receipt_cid
                for item in self.results
                if item.valid and item.receipt_cid
            )
        )

    def results_for_goal(self, goal_id: str) -> tuple[dict[str, Any], ...]:
        return tuple(
            item.to_dict()
            for item in self.results
            if item.goal_id == goal_id
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXTERNAL_COMPLETION_VALIDATION_SCHEMA + "/authority",
            "authority_cid": self.authority_cid,
            "governed_goal_ids": list(self.governed_goal_ids),
            "valid_receipt_cids": list(self.valid_receipt_cids),
            "source_inspection": self.source_inspection.to_dict(),
            "results": [item.to_dict() for item in self.results],
        }


def _validate_receipt(
    receipt: ExternalOperationalCompletionReceipt,
    requirement: ExternalCompletionRequirement,
    source_inspection: ExternalSourceInspection,
    *,
    now: datetime,
    freshness_seconds: float,
    clock_skew_seconds: float,
) -> tuple[str, ...]:
    reasons: list[str] = []

    def reject(code: str) -> None:
        if code not in reasons:
            reasons.append(code)

    current_source = source_inspection.identity
    if receipt.status != "completed":
        reject("external_run_not_completed")
    if receipt.goal_id != requirement.goal_id:
        reject("external_goal_mismatch")
    if receipt.evidence_term != requirement.evidence_term:
        reject("external_evidence_term_mismatch")
    if receipt.run_plan_cid != requirement.run_plan_cid:
        reject("external_run_plan_mismatch")
    if receipt.parent_ledger_cid != requirement.parent_ledger_cid:
        reject("external_parent_ledger_mismatch")
    if receipt.producer_id != requirement.expected_producer_id:
        reject("external_producer_mismatch")
    if receipt.validator_id != requirement.expected_validator_id:
        reject("external_validator_mismatch")
    if receipt.producer_id == receipt.validator_id:
        reject("external_validator_not_independent")
    if receipt.validator_receipt_cid in {
        receipt.producer_id,
        receipt.validator_id,
        receipt.run_plan_cid,
        receipt.parent_ledger_cid,
        *(item.artifact_cid for item in receipt.artifacts),
    }:
        reject("external_validator_receipt_not_independent")

    if not source_inspection.valid:
        for code in source_inspection.reason_codes:
            reject(code)
    if not current_source.clean:
        reject("current_source_dirty")
    if not current_source.recursive_gitlinks_complete:
        reject("current_recursive_gitlinks_incomplete")
    if not receipt.source.clean:
        reject("receipt_source_dirty")
    if not receipt.source.recursive_gitlinks_complete:
        reject("receipt_recursive_gitlinks_incomplete")
    if receipt.source.source_identity_cid != requirement.source_identity_cid:
        reject("external_source_policy_mismatch")
    if current_source.source_identity_cid != requirement.source_identity_cid:
        reject("current_source_policy_mismatch")
    if receipt.source.outer_commit != current_source.outer_commit:
        reject("external_source_commit_mismatch")
    if receipt.source.outer_tree != current_source.outer_tree:
        reject("external_source_tree_mismatch")
    if receipt.source.submodule_map_cid != current_source.submodule_map_cid:
        reject("external_recursive_gitlinks_mismatch")
    if receipt.source.source_identity_cid != current_source.source_identity_cid:
        reject("external_source_identity_mismatch")

    required_artifact_ids = set(requirement.required_artifact_ids)
    supplied_artifact_ids = {item.artifact_id for item in receipt.artifacts}
    if required_artifact_ids - supplied_artifact_ids:
        reject("external_artifacts_missing")
    if supplied_artifact_ids - required_artifact_ids:
        reject("external_artifacts_unexpected")

    max_age = timedelta(seconds=max(0.0, float(freshness_seconds)))
    clock_skew = timedelta(seconds=max(0.0, float(clock_skew_seconds)))
    if receipt.observed_at > now + clock_skew:
        reject("external_receipt_from_future")
    elif now - receipt.observed_at > max_age:
        reject("external_receipt_stale")
    if receipt.fresh_until is not None:
        if receipt.fresh_until < receipt.observed_at:
            reject("external_freshness_window_invalid")
        elif now > receipt.fresh_until:
            reject("external_receipt_stale")
    return tuple(reasons)


def _completion_evidence(
    *,
    authority: ExternalCompletionAuthority,
    requirement: ExternalCompletionRequirement,
    receipt: ExternalOperationalCompletionReceipt,
) -> CompletionEvidence:
    return CompletionEvidence(
        acceptance_criterion=receipt.evidence_term,
        producing_task_or_scan=receipt.producer_id,
        producer_id=receipt.producer_id,
        producer_kind="task",
        validation_receipt={
            "schema": EXTERNAL_COMPLETION_EVIDENCE_SCHEMA,
            "attempted": True,
            "passed": True,
            "status": "verified",
            "tree_id": receipt.source.outer_tree,
            "receipt_cid": receipt.validator_receipt_cid,
            "external_operational_receipt_cid": receipt.receipt_cid,
        },
        validation_passed=True,
        repository_tree=receipt.source.outer_tree,
        repository_id="",
        freshness=True,
        observed_at=receipt.observed_at,
        fresh_until=receipt.fresh_until,
        provenance_cid=receipt.receipt_cid,
        metadata={
            "external_operational_completion": True,
            "external_authority_cid": authority.authority_cid,
            "external_requirement_cid": requirement.requirement_cid,
            "source_identity_cid": receipt.source.source_identity_cid,
            "source_commit": receipt.source.outer_commit,
            "source_tree": receipt.source.outer_tree,
            "recursive_gitlinks_cid": receipt.source.submodule_map_cid,
            "run_plan_cid": receipt.run_plan_cid,
            "parent_ledger_cid": receipt.parent_ledger_cid,
            "artifact_identities": [
                item.to_dict() for item in receipt.artifacts
            ],
            "validator_id": receipt.validator_id,
            "validator_receipt_cid": receipt.validator_receipt_cid,
        },
    )


def evaluate_external_completion_authority(
    authority: ExternalCompletionAuthority | Mapping[str, Any],
    *,
    repo_root: Path,
    objective_path: Path,
    goal_evidence_terms: Mapping[str, Sequence[str]],
    now: datetime | str | None = None,
    freshness_seconds: float = DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
    clock_skew_seconds: float = DEFAULT_CLOCK_SKEW_SECONDS,
) -> ExternalCompletionEvaluation:
    """Validate all governed bindings and produce generic completion evidence."""

    normalized = (
        authority
        if isinstance(authority, ExternalCompletionAuthority)
        else ExternalCompletionAuthority.from_dict(authority)
    )
    current = _now(now)
    source_inspection = inspect_external_source(
        repo_root,
        objective_path=objective_path,
    )
    requirements = {item.binding: item for item in normalized.requirements}
    receipts_by_binding: dict[
        tuple[str, str],
        list[ExternalOperationalCompletionReceipt],
    ] = {}
    for receipt in normalized.receipts:
        receipts_by_binding.setdefault(receipt.binding, []).append(receipt)

    results: list[ExternalReceiptValidationResult] = []
    evidence_records: dict[str, list[CompletionEvidence]] = {
        goal_id: [] for goal_id in normalized.governed_goal_ids
    }
    for goal_id in normalized.governed_goal_ids:
        expected_terms = tuple(
            str(item).strip()
            for item in goal_evidence_terms.get(goal_id, ())
            if str(item).strip()
        )
        expected_set = set(expected_terms)
        requirement_terms = {
            item.evidence_term
            for item in normalized.requirements
            if item.goal_id == goal_id
        }
        for missing_term in sorted(expected_set - requirement_terms):
            results.append(
                ExternalReceiptValidationResult(
                    goal_id=goal_id,
                    evidence_term=missing_term,
                    valid=False,
                    reason_codes=("external_requirement_missing",),
                )
            )
        for unexpected_term in sorted(requirement_terms - expected_set):
            requirement = requirements[(goal_id, unexpected_term)]
            results.append(
                ExternalReceiptValidationResult(
                    goal_id=goal_id,
                    evidence_term=unexpected_term,
                    valid=False,
                    reason_codes=("external_requirement_evidence_mismatch",),
                    requirement_cid=requirement.requirement_cid,
                )
            )

    for binding, requirement in sorted(requirements.items()):
        candidates = receipts_by_binding.get(binding, [])
        if not candidates:
            results.append(
                ExternalReceiptValidationResult(
                    goal_id=requirement.goal_id,
                    evidence_term=requirement.evidence_term,
                    valid=False,
                    reason_codes=("external_receipt_missing",),
                    requirement_cid=requirement.requirement_cid,
                )
            )
            continue
        if len(candidates) != 1:
            results.append(
                ExternalReceiptValidationResult(
                    goal_id=requirement.goal_id,
                    evidence_term=requirement.evidence_term,
                    valid=False,
                    reason_codes=("external_receipt_duplicate",),
                    requirement_cid=requirement.requirement_cid,
                )
            )
            continue
        receipt = candidates[0]
        reason_codes = list(
            _validate_receipt(
                receipt,
                requirement,
                source_inspection,
                now=current,
                freshness_seconds=freshness_seconds,
                clock_skew_seconds=clock_skew_seconds,
            )
        )
        if requirement.evidence_term not in {
            str(item).strip()
            for item in goal_evidence_terms.get(requirement.goal_id, ())
            if str(item).strip()
        }:
            reason_codes.append("external_requirement_evidence_mismatch")
        reason_codes = list(dict.fromkeys(reason_codes))
        valid = not reason_codes
        results.append(
            ExternalReceiptValidationResult(
                goal_id=requirement.goal_id,
                evidence_term=requirement.evidence_term,
                valid=valid,
                reason_codes=tuple(reason_codes),
                receipt_cid=receipt.receipt_cid,
                requirement_cid=requirement.requirement_cid,
            )
        )
        if valid:
            evidence_records[requirement.goal_id].append(
                _completion_evidence(
                    authority=normalized,
                    requirement=requirement,
                    receipt=receipt,
                )
            )

    return ExternalCompletionEvaluation(
        authority_cid=normalized.authority_cid,
        governed_goal_ids=normalized.governed_goal_ids,
        evidence_records={
            goal_id: tuple(records)
            for goal_id, records in evidence_records.items()
        },
        results=tuple(
            sorted(
                results,
                key=lambda item: (
                    item.goal_id,
                    item.evidence_term,
                    item.receipt_cid,
                    item.reason_codes,
                ),
            )
        ),
        source_inspection=source_inspection,
    )


__all__ = [
    "EXTERNAL_ARTIFACT_SCHEMA",
    "EXTERNAL_COMPLETION_AUTHORITY_SCHEMA",
    "EXTERNAL_COMPLETION_EVIDENCE_SCHEMA",
    "EXTERNAL_COMPLETION_RECEIPT_SCHEMA",
    "EXTERNAL_COMPLETION_REQUIREMENT_SCHEMA",
    "EXTERNAL_COMPLETION_VALIDATION_SCHEMA",
    "EXTERNAL_GITLINK_SCHEMA",
    "EXTERNAL_SOURCE_SCHEMA",
    "HSSLEV2398A61",
    "ExternalArtifactIdentity",
    "ExternalCompletionAuthority",
    "ExternalCompletionEvaluation",
    "ExternalCompletionRequirement",
    "ExternalGitlinkIdentity",
    "ExternalOperationalCompletionReceipt",
    "ExternalReceiptValidationResult",
    "ExternalSourceIdentity",
    "ExternalSourceInspection",
    "evaluate_external_completion_authority",
    "inspect_external_source",
    "load_external_completion_authority",
    "validate_cid",
]
