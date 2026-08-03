"""Content binding from a reviewed provider proposal to an exact Git effect.

A provider response digest proves which bytes the model returned, but it does
not by itself prove which bytes the supervisor wrote or which Git commit later
carried them.  ``ProductionReviewedEffectBinding@1`` closes that gap in three
steps: capture the fenced workspace immediately after the writer, compare the
same workspace after validation, then bind the immutable implementation commit
and tree.  Completion verification reconstructs the task, packet, diff, modes,
and blobs from Git rather than trusting queue metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity
from .contract_packet_provider_router import (
    PROVIDER_EXECUTION_RECEIPT_INTERFACE,
    PROVIDER_EXECUTION_RECEIPT_SCHEMA,
    ProductionContractPacket,
    ProviderReason,
    ProviderRole,
    ReviewPresence,
    RouteStatus,
    _packet_content_id,
    review_chain_content_digest,
)
from .llm import LLM_CHILD_RESULT_SCHEMA
from .production_context_slice import (
    PRODUCTION_CONTEXT_SLICE_INTERFACE,
    PRODUCTION_CONTEXT_SLICE_SCHEMA,
)
from .production_provider_cli import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_GROK_MODEL,
    PRODUCTION_CLI_EXECUTION_SCHEMA,
)

PRODUCTION_REVIEWED_EFFECT_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/production-reviewed-effect-binding@1"
)
PRODUCTION_REVIEWED_EFFECT_BINDING_INTERFACE: Final = "ProductionReviewedEffectBinding@1"

# Git treats these ambient variables as repository-routing authority.  A
# long-lived supervisor must not let its service environment redirect proof
# reconstruction to another worktree, index, common directory, or object
# database.  Isolated patch reconstruction adds back only its supervisor-
# created index/object paths after this list has been removed.
_GIT_ROUTING_ENVIRONMENT_KEYS: Final = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_PARAMETERS",
        "GIT_DIR",
        "GIT_DIFF_OPTS",
        "GIT_EXTERNAL_DIFF",
        "GIT_GRAFT_FILE",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PREFIX",
        "GIT_QUARANTINE_PATH",
        "GIT_REPLACE_REF_BASE",
        "GIT_WORK_TREE",
    }
)

_PATH_EFFECT_KEYS: Final = frozenset(
    {
        "path",
        "status",
        "baseline_mode",
        "baseline_blob_oid",
        "applied_git_mode",
        "applied_filesystem_mode",
        "applied_blob_oid",
        "applied_sha256",
        "applied_bytes",
    }
)
_BINDING_KEYS: Final = frozenset(
    {
        "schema",
        "interface",
        "binding_id",
        "task_identity",
        "task_contract_cid",
        "packet_id",
        "packet_task_id",
        "packet_cid",
        "packet_payload",
        "snapshot_id",
        "baseline_commit",
        "baseline_tree_id",
        "context_manifest_cid",
        "context_task_cid",
        "context_snapshot_id",
        "context_scope_cid",
        "provider_policy_id",
        "provider_receipt_cid",
        "provider_receipt",
        "review_chain_digest",
        "selected_proposal_digest",
        "selected_proposal_payload_cid",
        "selected_proposal_payload",
        "implementation_proposal_digest",
        "review_proposal_digest",
        "review_proposal_payload_cid",
        "review_proposal_payload",
        "writer_lease_id",
        "changed_paths",
        "path_effects",
        "implementation_commit",
        "implementation_tree_id",
        "implementation_diff_sha256",
        "implementation_diff_bytes",
        "completion_authoritative",
        "proof_authoritative",
    }
)
_PROVIDER_RECEIPT_KEYS: Final = frozenset(
    {
        "schema",
        "interface",
        "receipt_id",
        "status",
        "reason_code",
        "provider",
        "packet",
        "review_chain",
        "review_presence",
        "admission",
        "attempts",
        "writer_lease_id",
        "write_performed",
        "fallback",
        "selected_proposal_digest",
        "implementation_proposal_digest",
        "review_proposal_digest",
        "proof_authoritative",
        "completion_authoritative",
    }
)
_REVIEW_CHAIN_STEP_KEYS: Final = frozenset(
    {
        "role",
        "status",
        "reason_code",
        "admitted",
        "response_digest",
        "prompt_bytes",
        "prompt_tokens",
        "response_bytes",
    }
)
_PROVIDER_ATTEMPT_KEYS: Final = frozenset(
    {
        "role",
        "status",
        "reason_code",
        "prompt_bytes",
        "prompt_tokens",
        "response_bytes",
        "prompt_digest",
        "response_digest",
        "execution_schema",
        "execution_policy_id",
        "execution_request_id",
        "configured_provider",
        "effective_provider",
        "configured_model",
        "child_result_schema",
        "child_result_status",
        "child_exit_code",
        "prompt_embedded",
        "response_embedded",
    }
)


def _json_detach(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )


def _canonical_path(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("effect path must be a string")
    path = PurePosixPath(value)
    if (
        not value
        or value != value.strip()
        or "\x00" in value
        or "\\" in value
        or unicodedata.normalize("NFC", value) != value
        or path.is_absolute()
        or value in {".", ".."}
        or any(not part or part == "." for part in value.split("/"))
        or ".." in path.parts
        or any(part.casefold() == ".git" for part in path.parts)
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or path.as_posix() != value
    ):
        raise ValueError("effect path must be canonical and repository-relative")
    return value


def _repository_root(value: str | Path) -> Path:
    """Resolve one exact, non-symlinked Git worktree top-level."""

    supplied = Path(value)
    try:
        absolute = Path(os.path.abspath(os.fspath(supplied)))
        current = Path(absolute.anchor)
        for part in absolute.parts[1:]:
            current = current / part
            info = os.lstat(current)
            if stat.S_ISLNK(info.st_mode):
                raise ValueError("repository root cannot contain symlink components")
        if not stat.S_ISDIR(os.lstat(absolute).st_mode):
            raise ValueError("repository root must be a directory")
        reported = (
            _run_git(absolute, ["rev-parse", "--show-toplevel"])
            .decode("utf-8", errors="strict")
            .strip()
        )
        top = Path(os.path.abspath(reported))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("repository root is unavailable or unsafe") from exc
    if top != absolute:
        raise ValueError("repository root must be the exact Git worktree top-level")
    return absolute


def _text(value: Any) -> str:
    return str(value or "").strip()


def _mapping(value: Any) -> dict[str, Any]:
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        value = converter()
    if not isinstance(value, Mapping):
        raise TypeError("value must be a mapping or expose to_dict()")
    detached = _json_detach(dict(value))
    if not isinstance(detached, dict):
        raise TypeError("value must detach to a JSON object")
    return detached


def _task_identity_payload(value: Any) -> dict[str, str]:
    payload = _mapping(value)
    result = {
        "canonical_task_key": _text(payload.get("canonical_task_key")),
        "canonical_task_cid": _text(payload.get("canonical_task_cid")),
        "display_task_id": _text(payload.get("display_task_id")),
        "board_namespace": _text(payload.get("board_namespace")) or "default",
    }
    if not result["canonical_task_key"] or not result["canonical_task_cid"]:
        raise ValueError("canonical task key and CID are required")
    return result


def production_task_contract(task: Any, task_identity: Any) -> dict[str, Any]:
    """Return the immutable task facts which authorize a reviewed write."""

    identity = _task_identity_payload(task_identity)
    metadata = getattr(task, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("task metadata must be a mapping")
    outputs = [_canonical_path(path) for path in (getattr(task, "outputs", ()) or ())]
    if not outputs or len(outputs) != len(set(outputs)):
        raise ValueError("task outputs must be explicit and unique")
    contract = {
        "task_id": _text(getattr(task, "task_id", "")),
        "title": str(getattr(task, "title", "") or ""),
        "priority": str(getattr(task, "priority", "") or ""),
        "track": str(getattr(task, "track", "") or ""),
        "depends_on": [str(value) for value in (getattr(task, "depends_on", ()) or ())],
        "outputs": outputs,
        "validation": [str(value) for value in (getattr(task, "validation", ()) or ())],
        "acceptance": str(getattr(task, "acceptance", "") or ""),
        "metadata": _json_detach(dict(metadata)),
        "canonical_task_key": identity["canonical_task_key"],
        "canonical_task_cid": identity["canonical_task_cid"],
        "board_namespace": identity["board_namespace"],
    }
    if not contract["task_id"]:
        raise ValueError("task_id is required")
    return contract


def production_task_contract_cid(task: Any, task_identity: Any) -> str:
    return content_identity(production_task_contract(task, task_identity))


def _sanitized_git_environment() -> dict[str, str]:
    """Return an environment with no ambient Git repository redirection."""

    environment = dict(os.environ)
    for key in _GIT_ROUTING_ENVIRONMENT_KEYS:
        environment.pop(key, None)
    # GIT_CONFIG_COUNT can inject indexed key/value variables.  Remove those
    # even when the count itself is malformed or absent.
    for key in tuple(environment):
        if key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
            environment.pop(key, None)
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return environment


def _run_git(
    repo_root: Path,
    arguments: Sequence[str],
    *,
    input_bytes: bytes | None = None,
) -> bytes:
    result = subprocess.run(
        ["git", "--literal-pathspecs", *arguments],
        cwd=repo_root,
        env=_sanitized_git_environment(),
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace")
        raise ValueError(f"Git fact reconstruction failed: {detail[-500:]}")
    return bytes(result.stdout)


def _resolve_commit(repo_root: Path, value: str) -> str:
    candidate = _text(value)
    if not candidate:
        raise ValueError("Git commit identity is required")
    resolved = (
        _run_git(
            repo_root,
            ["rev-parse", "--verify", "--end-of-options", f"{candidate}^{{commit}}"],
        )
        .decode("ascii", errors="strict")
        .strip()
    )
    if not resolved:
        raise ValueError("Git commit identity did not resolve")
    return resolved


def _tree_id(repo_root: Path, commit: str) -> str:
    tree = (
        _run_git(
            repo_root,
            ["rev-parse", "--verify", "--end-of-options", f"{commit}^{{tree}}"],
        )
        .decode("ascii", errors="strict")
        .strip()
    )
    return f"git-tree:{tree}" if tree else ""


def _nul_paths(value: bytes) -> tuple[str, ...]:
    paths: list[str] = []
    for raw in value.split(b"\x00"):
        if not raw:
            continue
        path = _canonical_path(raw.decode("utf-8", errors="strict"))
        if path not in paths:
            paths.append(path)
    return tuple(paths)


def _workspace_changed_paths(repo_root: Path, baseline: str) -> tuple[str, ...]:
    tracked = _nul_paths(
        _run_git(
            repo_root,
            [
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--name-only",
                "--no-renames",
                "-z",
                baseline,
                "--",
            ],
        )
    )
    untracked = _nul_paths(
        _run_git(
            repo_root,
            ["ls-files", "--others", "--exclude-standard", "-z", "--"],
        )
    )
    return tuple(sorted(set((*tracked, *untracked))))


def _commit_changed_paths(
    repo_root: Path,
    baseline: str,
    implementation_commit: str,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            _nul_paths(
                _run_git(
                    repo_root,
                    [
                        "diff",
                        "--no-ext-diff",
                        "--no-textconv",
                        "--name-only",
                        "--no-renames",
                        "-z",
                        baseline,
                        implementation_commit,
                        "--",
                    ],
                )
            )
        )
    )


def _commit_diff(
    repo_root: Path,
    baseline: str,
    implementation_commit: str,
    paths: Sequence[str],
) -> bytes:
    return _run_git(
        repo_root,
        [
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--binary",
            "--full-index",
            "--no-renames",
            baseline,
            implementation_commit,
            "--",
            *paths,
        ],
    )


def _patch_index_effect_failures(
    binding: ProductionReviewedEffectBinding,
    *,
    repo_root: Path,
    patch: str,
) -> list[str]:
    """Apply exact reviewed patch to an isolated baseline index and compare blobs."""

    failures: list[str] = []
    raw_objects = (
        _run_git(repo_root, ["rev-parse", "--git-path", "objects"])
        .decode("utf-8", errors="strict")
        .strip()
    )
    repository_objects = Path(raw_objects)
    if not repository_objects.is_absolute():
        repository_objects = repo_root / repository_objects
    repository_objects = repository_objects.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="reviewed-effect-") as directory_name:
        temporary_root = Path(directory_name)
        os.chmod(temporary_root, 0o700)
        index_path = temporary_root / "index"
        temporary_objects = temporary_root / "objects"
        temporary_objects.mkdir(mode=0o700)
        environment = _sanitized_git_environment()
        environment["GIT_INDEX_FILE"] = str(index_path)
        environment["GIT_OBJECT_DIRECTORY"] = str(temporary_objects)
        environment["GIT_ALTERNATE_OBJECT_DIRECTORIES"] = str(repository_objects)

        def run(
            arguments: Sequence[str],
            *,
            input_bytes: bytes | None = None,
        ) -> bytes:
            result = subprocess.run(
                ["git", "--literal-pathspecs", *arguments],
                cwd=repo_root,
                env=environment,
                input=input_bytes,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).decode("utf-8", errors="replace")
                raise ValueError("isolated reviewed patch reconstruction failed: " + detail[-500:])
            return bytes(result.stdout)

        try:
            run(["read-tree", binding.baseline_commit])
            run(
                ["apply", "--cached", "--whitespace=nowarn", "-"],
                input_bytes=patch.encode("utf-8"),
            )
            changed = _nul_paths(
                run(
                    [
                        "diff",
                        "--no-ext-diff",
                        "--no-textconv",
                        "--cached",
                        "--name-only",
                        "--no-renames",
                        "-z",
                        binding.baseline_commit,
                        "--",
                    ]
                )
            )
            if tuple(sorted(changed)) != binding.changed_paths:
                failures.append("reviewed_effect_grok_patch_path_set_mismatch")
                return failures
            for effect in binding.path_effects:
                output = run(["ls-files", "--stage", "-z", "--", effect.path])
                records = [record for record in output.split(b"\x00") if record]
                if effect.status == "deleted":
                    if records:
                        failures.append(f"reviewed_effect_grok_patch_blob_mismatch:{effect.path}")
                    continue
                if len(records) != 1 or b"\t" not in records[0]:
                    failures.append(f"reviewed_effect_grok_patch_blob_mismatch:{effect.path}")
                    continue
                header, raw_path = records[0].split(b"\t", 1)
                parts = header.decode("ascii", errors="strict").split()
                if (
                    raw_path.decode("utf-8", errors="strict") != effect.path
                    or len(parts) != 3
                    or parts[2] != "0"
                ):
                    failures.append(f"reviewed_effect_grok_patch_blob_mismatch:{effect.path}")
                    continue
                mode, oid, _stage = parts
                content = run(["cat-file", "blob", oid])
                if (
                    mode != effect.applied_git_mode
                    or oid != effect.applied_blob_oid
                    or len(content) != effect.applied_bytes
                    or "sha256:" + hashlib.sha256(content).hexdigest() != effect.applied_sha256
                ):
                    failures.append(f"reviewed_effect_grok_patch_blob_mismatch:{effect.path}")
        except (OSError, TypeError, UnicodeError, ValueError):
            failures.append("reviewed_effect_grok_patch_reconstruction_failed")
    return failures


def _tree_blob(
    repo_root: Path,
    commit: str,
    path: str,
) -> tuple[str, str, bytes] | None:
    output = _run_git(repo_root, ["ls-tree", "-z", commit, "--", path])
    if not output:
        return None
    records = [record for record in output.split(b"\x00") if record]
    if len(records) != 1 or b"\t" not in records[0]:
        raise ValueError(f"Git tree path is ambiguous: {path}")
    header, raw_path = records[0].split(b"\t", 1)
    if raw_path.decode("utf-8", errors="strict") != path:
        raise ValueError(f"Git tree path identity mismatch: {path}")
    parts = header.decode("ascii", errors="strict").split()
    if len(parts) != 3 or parts[1] != "blob":
        raise ValueError(f"reviewed effect path is not a Git blob: {path}")
    mode, _kind, oid = parts
    blob = _run_git(repo_root, ["cat-file", "blob", oid])
    return mode, oid, blob


def _workspace_blob(repo_root: Path, path: str) -> tuple[str, int, str, bytes] | None:
    parts = PurePosixPath(_canonical_path(path)).parts
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptors: list[int] = []
    try:
        root_descriptor = os.open(repo_root, directory_flags)
        descriptors.append(root_descriptor)
        parent_descriptor = root_descriptor
        for part in parts[:-1]:
            parent_descriptor = os.open(
                part,
                directory_flags,
                dir_fd=parent_descriptor,
            )
            descriptors.append(parent_descriptor)
            if any(name.casefold() == ".git" for name in os.listdir(parent_descriptor)):
                raise ValueError(f"reviewed effect crosses a nested Git repository: {path}")
        try:
            descriptor = os.open(
                parts[-1],
                file_flags,
                dir_fd=parent_descriptor,
            )
        except FileNotFoundError:
            return None
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"reviewed effect target is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 131_072)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in stable_fields):
            raise ValueError(f"reviewed effect target changed during read: {path}")
        content = b"".join(chunks)
        info = after
    except OSError as exc:
        raise ValueError(f"reviewed effect target is unsafe: {path}") from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    oid = (
        _run_git(repo_root, ["hash-object", "--stdin"], input_bytes=content)
        .decode("ascii", errors="strict")
        .strip()
    )
    git_mode = "100755" if info.st_mode & 0o111 else "100644"
    return git_mode, stat.S_IMODE(info.st_mode), oid, content


@dataclass(frozen=True, slots=True)
class ProductionPathEffect:
    path: str
    status: str
    baseline_mode: str
    baseline_blob_oid: str
    applied_git_mode: str
    applied_filesystem_mode: int
    applied_blob_oid: str
    applied_sha256: str
    applied_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "status": self.status,
            "baseline_mode": self.baseline_mode,
            "baseline_blob_oid": self.baseline_blob_oid,
            "applied_git_mode": self.applied_git_mode,
            "applied_filesystem_mode": self.applied_filesystem_mode,
            "applied_blob_oid": self.applied_blob_oid,
            "applied_sha256": self.applied_sha256,
            "applied_bytes": self.applied_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProductionPathEffect:
        payload = _mapping(value)
        if set(payload) != _PATH_EFFECT_KEYS:
            raise ValueError("production path effect shape is invalid")
        path = _canonical_path(payload.get("path"))
        status_value = payload.get("status")
        if status_value not in {"added", "modified", "deleted"}:
            raise ValueError("production path effect status is invalid")
        mode = payload.get("applied_filesystem_mode")
        size = payload.get("applied_bytes")
        if (
            isinstance(mode, bool)
            or not isinstance(mode, int)
            or mode < 0
            or mode > 0o7777
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ValueError("production path effect numeric fields are invalid")
        return cls(
            path=path,
            status=status_value,
            baseline_mode=_text(payload.get("baseline_mode")),
            baseline_blob_oid=_text(payload.get("baseline_blob_oid")),
            applied_git_mode=_text(payload.get("applied_git_mode")),
            applied_filesystem_mode=mode,
            applied_blob_oid=_text(payload.get("applied_blob_oid")),
            applied_sha256=_text(payload.get("applied_sha256")),
            applied_bytes=size,
        )


def _path_effect(repo_root: Path, baseline: str, path: str) -> ProductionPathEffect:
    before = _tree_blob(repo_root, baseline, path)
    after = _workspace_blob(repo_root, path)
    if before is None and after is None:
        raise ValueError(f"changed effect path is absent before and after: {path}")
    if before is None:
        status_value = "added"
    elif after is None:
        status_value = "deleted"
    else:
        status_value = "modified"
    baseline_mode, baseline_oid = (before[0], before[1]) if before else ("", "")
    if after is None:
        applied_mode, filesystem_mode, applied_oid, content = "", 0, "", b""
    else:
        applied_mode, filesystem_mode, applied_oid, content = after
    if before is not None and after is not None:
        if baseline_mode == applied_mode and baseline_oid == applied_oid:
            raise ValueError(f"effect path has no byte or mode change: {path}")
    return ProductionPathEffect(
        path=path,
        status=status_value,
        baseline_mode=baseline_mode,
        baseline_blob_oid=baseline_oid,
        applied_git_mode=applied_mode,
        applied_filesystem_mode=filesystem_mode,
        applied_blob_oid=applied_oid,
        applied_sha256=("sha256:" + hashlib.sha256(content).hexdigest()) if after else "",
        applied_bytes=len(content),
    )


@dataclass(frozen=True, slots=True)
class ProductionReviewedEffectBinding:
    binding_id: str
    task_identity: Mapping[str, str]
    task_contract_cid: str
    packet_id: str
    packet_task_id: str
    packet_cid: str
    packet_payload: Mapping[str, Any]
    snapshot_id: str
    baseline_commit: str
    baseline_tree_id: str
    context_manifest_cid: str
    context_task_cid: str
    context_snapshot_id: str
    context_scope_cid: str
    provider_policy_id: str
    provider_receipt_cid: str
    provider_receipt: Mapping[str, Any]
    review_chain_digest: str
    selected_proposal_digest: str
    selected_proposal_payload_cid: str
    selected_proposal_payload: Mapping[str, Any]
    implementation_proposal_digest: str
    review_proposal_digest: str
    review_proposal_payload_cid: str
    review_proposal_payload: Mapping[str, Any]
    writer_lease_id: str
    changed_paths: tuple[str, ...]
    path_effects: tuple[ProductionPathEffect, ...]
    implementation_commit: str = ""
    implementation_tree_id: str = ""
    implementation_diff_sha256: str = ""
    implementation_diff_bytes: int = 0

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema": PRODUCTION_REVIEWED_EFFECT_BINDING_SCHEMA,
            "interface": PRODUCTION_REVIEWED_EFFECT_BINDING_INTERFACE,
            "task_identity": dict(self.task_identity),
            "task_contract_cid": self.task_contract_cid,
            "packet_id": self.packet_id,
            "packet_task_id": self.packet_task_id,
            "packet_cid": self.packet_cid,
            "packet_payload": dict(self.packet_payload),
            "snapshot_id": self.snapshot_id,
            "baseline_commit": self.baseline_commit,
            "baseline_tree_id": self.baseline_tree_id,
            "context_manifest_cid": self.context_manifest_cid,
            "context_task_cid": self.context_task_cid,
            "context_snapshot_id": self.context_snapshot_id,
            "context_scope_cid": self.context_scope_cid,
            "provider_policy_id": self.provider_policy_id,
            "provider_receipt_cid": self.provider_receipt_cid,
            "provider_receipt": dict(self.provider_receipt),
            "review_chain_digest": self.review_chain_digest,
            "selected_proposal_digest": self.selected_proposal_digest,
            "selected_proposal_payload_cid": self.selected_proposal_payload_cid,
            "selected_proposal_payload": dict(self.selected_proposal_payload),
            "implementation_proposal_digest": self.implementation_proposal_digest,
            "review_proposal_digest": self.review_proposal_digest,
            "review_proposal_payload_cid": self.review_proposal_payload_cid,
            "review_proposal_payload": dict(self.review_proposal_payload),
            "writer_lease_id": self.writer_lease_id,
            "changed_paths": list(self.changed_paths),
            "path_effects": [effect.to_dict() for effect in self.path_effects],
            "implementation_commit": self.implementation_commit,
            "implementation_tree_id": self.implementation_tree_id,
            "implementation_diff_sha256": self.implementation_diff_sha256,
            "implementation_diff_bytes": self.implementation_diff_bytes,
            "completion_authoritative": False,
            "proof_authoritative": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "binding_id": self.binding_id}

    @classmethod
    def create(cls, **values: Any) -> ProductionReviewedEffectBinding:
        candidate = cls(binding_id="", **values)
        return replace(candidate, binding_id=content_identity(candidate.unsigned_dict()))

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> ProductionReviewedEffectBinding:
        payload = _mapping(value)
        if set(payload) != _BINDING_KEYS:
            raise ValueError("production reviewed effect binding shape is invalid")
        if payload.get("schema") != PRODUCTION_REVIEWED_EFFECT_BINDING_SCHEMA:
            raise ValueError("production reviewed effect binding schema is invalid")
        if payload.get("interface") != PRODUCTION_REVIEWED_EFFECT_BINDING_INTERFACE:
            raise ValueError("production reviewed effect binding interface is invalid")
        if payload.get("completion_authoritative") is not False:
            raise ValueError("reviewed effect cannot claim completion authority")
        if payload.get("proof_authoritative") is not False:
            raise ValueError("reviewed effect cannot claim proof authority")
        changed = tuple(_canonical_path(path) for path in payload.get("changed_paths", ()))
        effects = tuple(
            ProductionPathEffect.from_dict(item) for item in payload.get("path_effects", ())
        )
        if (
            not changed
            or changed != tuple(sorted(set(changed)))
            or tuple(effect.path for effect in effects) != changed
        ):
            raise ValueError("production reviewed effect path set is invalid")
        candidate = cls(
            binding_id=_text(payload.get("binding_id")),
            task_identity=_task_identity_payload(payload.get("task_identity")),
            task_contract_cid=_text(payload.get("task_contract_cid")),
            packet_id=_text(payload.get("packet_id")),
            packet_task_id=_text(payload.get("packet_task_id")),
            packet_cid=_text(payload.get("packet_cid")),
            packet_payload=_mapping(payload.get("packet_payload")),
            snapshot_id=_text(payload.get("snapshot_id")),
            baseline_commit=_text(payload.get("baseline_commit")),
            baseline_tree_id=_text(payload.get("baseline_tree_id")),
            context_manifest_cid=_text(payload.get("context_manifest_cid")),
            context_task_cid=_text(payload.get("context_task_cid")),
            context_snapshot_id=_text(payload.get("context_snapshot_id")),
            context_scope_cid=_text(payload.get("context_scope_cid")),
            provider_policy_id=_text(payload.get("provider_policy_id")),
            provider_receipt_cid=_text(payload.get("provider_receipt_cid")),
            provider_receipt=_mapping(payload.get("provider_receipt")),
            review_chain_digest=_text(payload.get("review_chain_digest")),
            selected_proposal_digest=_text(payload.get("selected_proposal_digest")),
            selected_proposal_payload_cid=_text(payload.get("selected_proposal_payload_cid")),
            selected_proposal_payload=_mapping(payload.get("selected_proposal_payload")),
            implementation_proposal_digest=_text(payload.get("implementation_proposal_digest")),
            review_proposal_digest=_text(payload.get("review_proposal_digest")),
            review_proposal_payload_cid=_text(payload.get("review_proposal_payload_cid")),
            review_proposal_payload=_mapping(payload.get("review_proposal_payload")),
            writer_lease_id=_text(payload.get("writer_lease_id")),
            changed_paths=changed,
            path_effects=effects,
            implementation_commit=_text(payload.get("implementation_commit")),
            implementation_tree_id=_text(payload.get("implementation_tree_id")),
            implementation_diff_sha256=_text(payload.get("implementation_diff_sha256")),
            implementation_diff_bytes=payload.get("implementation_diff_bytes"),
        )
        if (
            isinstance(candidate.implementation_diff_bytes, bool)
            or not isinstance(candidate.implementation_diff_bytes, int)
            or candidate.implementation_diff_bytes < 0
        ):
            raise ValueError("production reviewed effect diff size is invalid")
        if not all(
            (
                candidate.binding_id,
                candidate.task_contract_cid,
                candidate.packet_id,
                candidate.packet_task_id,
                candidate.packet_cid,
                candidate.snapshot_id,
                candidate.baseline_commit,
                candidate.baseline_tree_id,
                candidate.context_manifest_cid,
                candidate.context_task_cid,
                candidate.context_snapshot_id,
                candidate.context_scope_cid,
                candidate.provider_policy_id,
                candidate.provider_receipt_cid,
                candidate.review_chain_digest,
                candidate.selected_proposal_digest,
                candidate.selected_proposal_payload_cid,
                candidate.implementation_proposal_digest,
                candidate.review_proposal_digest,
                candidate.review_proposal_payload_cid,
                candidate.writer_lease_id,
            )
        ):
            raise ValueError("production reviewed effect binding is incomplete")
        finalized_fields = (
            candidate.implementation_commit,
            candidate.implementation_tree_id,
            candidate.implementation_diff_sha256,
        )
        if any(finalized_fields) != all(finalized_fields):
            raise ValueError("production reviewed effect finalization is incomplete")
        if all(finalized_fields) != (candidate.implementation_diff_bytes > 0):
            raise ValueError("production reviewed effect final diff is incomplete")
        if candidate.binding_id != content_identity(candidate.unsigned_dict()):
            raise ValueError("production reviewed effect binding CID is invalid")
        return candidate


@dataclass(frozen=True, slots=True)
class ProductionReviewedEffectVerification:
    verified: bool
    reason_codes: tuple[str, ...]
    binding_id: str = ""
    implementation_commit: str = ""
    implementation_tree_id: str = ""

    @property
    def admitted(self) -> bool:
        return self.verified and not self.reason_codes

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "admitted": self.admitted,
            "reason_codes": list(self.reason_codes),
            "binding_id": self.binding_id,
            "implementation_commit": self.implementation_commit,
            "implementation_tree_id": self.implementation_tree_id,
            "completion_authoritative": False,
            "proof_authoritative": False,
        }


def _packet_payload(packet: ProductionContractPacket | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(packet, ProductionContractPacket):
        return _mapping(packet.payload)
    payload = _mapping(packet)
    nested = payload.get("payload")
    return _mapping(nested) if isinstance(nested, Mapping) else payload


def _context_binding_failures(
    binding: ProductionReviewedEffectBinding,
) -> list[str]:
    failures: list[str] = []
    context = binding.packet_payload.get("context_slice")
    context = dict(context) if isinstance(context, Mapping) else {}
    if (
        context.get("schema") != PRODUCTION_CONTEXT_SLICE_SCHEMA
        or context.get("interface") != PRODUCTION_CONTEXT_SLICE_INTERFACE
    ):
        return ["reviewed_effect_context_manifest_missing_or_invalid"]
    unsigned = {key: value for key, value in context.items() if key != "manifest_cid"}
    manifest_cid = _text(context.get("manifest_cid"))
    if not manifest_cid or manifest_cid != content_identity(unsigned):
        failures.append("reviewed_effect_context_manifest_cid_mismatch")
    task_binding = context.get("task_binding")
    task_binding = dict(task_binding) if isinstance(task_binding, Mapping) else {}
    repository = context.get("repository_binding")
    repository = dict(repository) if isinstance(repository, Mapping) else {}
    scope = context.get("scope")
    scope = dict(scope) if isinstance(scope, Mapping) else {}
    expected = {
        "context_manifest_cid": manifest_cid,
        "context_task_cid": _text(task_binding.get("canonical_task_cid")),
        "context_snapshot_id": _text(repository.get("snapshot_id")),
        "context_scope_cid": _text(scope.get("scope_cid")),
    }
    for field_name, expected_value in expected.items():
        if not expected_value or getattr(binding, field_name) != expected_value:
            failures.append(f"reviewed_effect_{field_name}_mismatch")
    if task_binding.get("task_id") != binding.packet_task_id:
        failures.append("reviewed_effect_context_task_id_mismatch")
    if binding.context_task_cid != binding.task_contract_cid:
        failures.append("reviewed_effect_context_task_contract_mismatch")
    if binding.context_snapshot_id != binding.snapshot_id:
        failures.append("reviewed_effect_context_snapshot_mismatch")
    if repository.get("baseline_commit") != binding.baseline_commit:
        failures.append("reviewed_effect_context_baseline_mismatch")
    if f"git-tree:{_text(repository.get('baseline_tree'))}" != binding.baseline_tree_id:
        failures.append("reviewed_effect_context_baseline_tree_mismatch")
    effect_paths = scope.get("effect_paths")
    if not isinstance(effect_paths, list):
        failures.append("reviewed_effect_context_scope_invalid")
    else:
        try:
            canonical_effects = tuple(sorted(_canonical_path(path) for path in effect_paths))
        except ValueError:
            canonical_effects = ()
            failures.append("reviewed_effect_context_scope_invalid")
        if not set(binding.changed_paths).issubset(canonical_effects):
            failures.append("reviewed_effect_context_effect_scope_mismatch")
    return failures


def _execution_metadata_failures(
    payload: Mapping[str, Any],
    attempt: Mapping[str, Any],
    *,
    role: ProviderRole,
) -> list[str]:
    failures: list[str] = []
    execution = payload.get("supervisor_provider_execution")
    execution = dict(execution) if isinstance(execution, Mapping) else {}
    pairs = {
        "schema": "execution_schema",
        "policy_id": "execution_policy_id",
        "request_id": "execution_request_id",
        "configured_provider": "configured_provider",
        "effective_provider": "effective_provider",
        "configured_model": "configured_model",
        "child_result_schema": "child_result_schema",
        "child_result_status": "child_result_status",
        "child_exit_code": "child_exit_code",
    }
    for execution_key, attempt_key in pairs.items():
        if execution.get(execution_key) != attempt.get(attempt_key):
            failures.append(f"reviewed_effect_execution_payload_mismatch:{role.value}")
            break
    if execution.get("role") != role.value:
        failures.append(f"reviewed_effect_execution_role_mismatch:{role.value}")
    expected_false = (
        "model_output_authored_receipt",
        "repository_checkout_used_as_working_directory",
        "operating_system_filesystem_confinement",
        "completion_authoritative",
        "proof_authoritative",
    )
    if any(execution.get(key) is not False for key in expected_false):
        failures.append(f"reviewed_effect_execution_authority_invalid:{role.value}")
    return failures


def _provider_receipt_failures(
    binding: ProductionReviewedEffectBinding,
) -> list[str]:
    failures: list[str] = []
    receipt = dict(binding.provider_receipt)
    if set(receipt) != _PROVIDER_RECEIPT_KEYS:
        failures.append("reviewed_effect_provider_receipt_shape_invalid")
    if receipt.get("schema") != PROVIDER_EXECUTION_RECEIPT_SCHEMA:
        failures.append("reviewed_effect_provider_receipt_schema_invalid")
    if receipt.get("interface") != PROVIDER_EXECUTION_RECEIPT_INTERFACE:
        failures.append("reviewed_effect_provider_receipt_interface_invalid")
    unsigned = dict(receipt)
    unsigned.pop("receipt_id", None)
    receipt_cid = _text(receipt.get("receipt_id"))
    if (
        not receipt_cid
        or receipt_cid != _packet_content_id(unsigned)
        or binding.provider_receipt_cid != receipt_cid
    ):
        failures.append("reviewed_effect_provider_receipt_cid_mismatch")
    if (
        receipt.get("status") != RouteStatus.SUCCEEDED.value
        or receipt.get("reason_code") != ProviderReason.ROUTED.value
        or receipt.get("provider") != ProviderRole.GROK_IMPLEMENT.value
        or receipt.get("fallback") is not False
        or receipt.get("write_performed") is not True
        or receipt.get("review_presence") != ReviewPresence.INDEPENDENT.value
        or receipt.get("completion_authoritative") is not False
        or receipt.get("proof_authoritative") is not False
        or receipt.get("writer_lease_id") != binding.writer_lease_id
    ):
        failures.append("reviewed_effect_provider_receipt_disposition_invalid")
    expected_admission = {
        "proposal_only": True,
        "repository_write_allowed": True,
        "completion_authoritative": False,
        "proof_authoritative": False,
        "provider_result_admitted": True,
        "independent_review": True,
        "review_presence": ReviewPresence.INDEPENDENT.value,
        "self_review": False,
        "writer_lease_bound": True,
    }
    if receipt.get("admission") != expected_admission:
        failures.append("reviewed_effect_provider_receipt_admission_invalid")
    packet = receipt.get("packet")
    packet = dict(packet) if isinstance(packet, Mapping) else {}
    encoded_packet = json.dumps(
        dict(binding.packet_payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    expected_packet = {
        "packet_id": binding.packet_id,
        "packet_cid": binding.packet_cid,
        "packet_bytes": len(encoded_packet),
        "snapshot_id": binding.snapshot_id,
        "task_id": binding.packet_task_id,
    }
    if packet != expected_packet:
        failures.append("reviewed_effect_provider_receipt_packet_mismatch")
    chain = receipt.get("review_chain")
    chain = list(chain) if isinstance(chain, list) else []
    attempts = receipt.get("attempts")
    attempts = list(attempts) if isinstance(attempts, list) else []
    roles = (ProviderRole.GROK_IMPLEMENT, ProviderRole.CODEX_REVIEW)
    if len(chain) != 2 or len(attempts) != 2:
        failures.append("reviewed_effect_provider_execution_chain_invalid")
        return failures
    expected_executions = (
        ("grok_cli", DEFAULT_GROK_MODEL, binding.selected_proposal_payload),
        ("codex_cli", DEFAULT_CODEX_MODEL, binding.review_proposal_payload),
    )
    response_digests = (
        binding.implementation_proposal_digest,
        binding.review_proposal_digest,
    )
    request_ids: list[str] = []
    for index, role in enumerate(roles):
        step = dict(chain[index]) if isinstance(chain[index], Mapping) else {}
        attempt = dict(attempts[index]) if isinstance(attempts[index], Mapping) else {}
        if set(step) != _REVIEW_CHAIN_STEP_KEYS:
            failures.append(f"reviewed_effect_review_chain_shape_invalid:{index}")
        if set(attempt) != _PROVIDER_ATTEMPT_KEYS:
            failures.append(f"reviewed_effect_provider_attempt_shape_invalid:{index}")
        provider_name, model_name, proposal_payload = expected_executions[index]
        if (
            step.get("role") != role.value
            or step.get("status") != "succeeded"
            or step.get("admitted") is not True
            or step.get("response_digest") != response_digests[index]
            or attempt.get("role") != role.value
            or attempt.get("status") != "succeeded"
            or attempt.get("reason_code") != ProviderReason.ROUTED.value
            or attempt.get("response_digest") != response_digests[index]
            or not _text(attempt.get("prompt_digest"))
            or attempt.get("prompt_embedded") is not False
            or attempt.get("response_embedded") is not False
            or attempt.get("execution_schema") != PRODUCTION_CLI_EXECUTION_SCHEMA
            or attempt.get("execution_policy_id") != binding.provider_policy_id
            or not _text(attempt.get("execution_request_id"))
            or attempt.get("configured_provider") != provider_name
            or attempt.get("effective_provider") != provider_name
            or attempt.get("configured_model") != model_name
            or attempt.get("child_result_schema") != LLM_CHILD_RESULT_SCHEMA
            or attempt.get("child_result_status") != "ok"
            or type(attempt.get("child_exit_code")) is not int
            or attempt.get("child_exit_code") != 0
        ):
            failures.append(f"reviewed_effect_provider_execution_invalid:{index}")
        request_ids.append(_text(attempt.get("execution_request_id")))
        failures.extend(_execution_metadata_failures(proposal_payload, attempt, role=role))
    if not all(request_ids) or len(set(request_ids)) != 2:
        failures.append("reviewed_effect_provider_request_ids_invalid")
    if receipt.get("selected_proposal_digest") != binding.selected_proposal_digest:
        failures.append("reviewed_effect_selected_proposal_digest_mismatch")
    if receipt.get("implementation_proposal_digest") != binding.implementation_proposal_digest:
        failures.append("reviewed_effect_implementation_proposal_digest_mismatch")
    if receipt.get("review_proposal_digest") != binding.review_proposal_digest:
        failures.append("reviewed_effect_review_proposal_digest_mismatch")
    if binding.selected_proposal_digest != binding.implementation_proposal_digest:
        failures.append("reviewed_effect_selected_proposal_not_grok_implementation")
    if binding.review_chain_digest != review_chain_content_digest(chain):
        failures.append("reviewed_effect_review_chain_digest_mismatch")
    return failures


def _proposal_payload_failures(
    binding: ProductionReviewedEffectBinding,
    *,
    repo_root: Path | None = None,
) -> list[str]:
    failures: list[str] = []
    if binding.selected_proposal_payload_cid != content_identity(
        dict(binding.selected_proposal_payload)
    ):
        failures.append("reviewed_effect_selected_proposal_payload_cid_mismatch")
    if binding.review_proposal_payload_cid != content_identity(
        dict(binding.review_proposal_payload)
    ):
        failures.append("reviewed_effect_review_proposal_payload_cid_mismatch")
    review = dict(binding.review_proposal_payload)
    if (
        review.get("decision") != "approve"
        or not isinstance(review.get("findings"), list)
        or review.get("proposal") not in (None, {})
    ):
        failures.append("reviewed_effect_codex_approval_invalid")
    selected = dict(binding.selected_proposal_payload)
    body = selected.get("proposal")
    body = dict(body) if isinstance(body, Mapping) else selected
    declared = body.get("declared_paths")
    files = body.get("files")
    patch_value = body.get("patch")
    if not isinstance(declared, list):
        failures.append("reviewed_effect_grok_declared_paths_invalid")
        return failures
    try:
        declared_paths = tuple(sorted(_canonical_path(path) for path in declared))
    except ValueError:
        declared_paths = ()
        failures.append("reviewed_effect_grok_declared_paths_invalid")
    if declared_paths != binding.changed_paths:
        failures.append("reviewed_effect_grok_path_set_mismatch")
    has_files = isinstance(files, list) and bool(files)
    has_patch = isinstance(patch_value, str) and bool(patch_value.strip())
    if has_files == has_patch:
        failures.append("reviewed_effect_grok_apply_representation_invalid")
        return failures
    if has_patch:
        if repo_root is None:
            failures.append("reviewed_effect_grok_patch_repository_missing")
        else:
            failures.extend(
                _patch_index_effect_failures(
                    binding,
                    repo_root=repo_root,
                    patch=patch_value,
                )
            )
        return failures
    file_content: dict[str, bytes] = {}
    for item in files:
        if not isinstance(item, Mapping) or not isinstance(item.get("content"), str):
            failures.append("reviewed_effect_grok_file_payload_invalid")
            continue
        try:
            path = _canonical_path(item.get("path"))
        except ValueError:
            failures.append("reviewed_effect_grok_file_payload_invalid")
            continue
        if path in file_content:
            failures.append("reviewed_effect_grok_file_payload_invalid")
            continue
        file_content[path] = item["content"].encode("utf-8")
    if tuple(sorted(file_content)) != binding.changed_paths:
        failures.append("reviewed_effect_grok_path_set_mismatch")
    context = binding.packet_payload.get("context_slice")
    context = dict(context) if isinstance(context, Mapping) else {}
    sources = context.get("sources")
    sources = list(sources) if isinstance(sources, list) else []
    fully_visible = {
        _text(record.get("path"))
        for record in sources
        if isinstance(record, Mapping) and record.get("full_visible_coverage") is True
    }
    scope = context.get("scope")
    scope = dict(scope) if isinstance(scope, Mapping) else {}
    absences = scope.get("absence_proofs")
    absences = list(absences) if isinstance(absences, list) else []
    absent_paths = {_text(record.get("path")) for record in absences if isinstance(record, Mapping)}
    for effect in binding.path_effects:
        content = file_content.get(effect.path)
        if effect.status == "deleted" or content is None:
            failures.append(f"reviewed_effect_grok_bytes_mismatch:{effect.path}")
            continue
        if (effect.status == "added" and effect.path not in absent_paths) or (
            effect.status != "added" and effect.path not in fully_visible
        ):
            failures.append(
                f"reviewed_effect_grok_full_replacement_context_incomplete:{effect.path}"
            )
        if (
            len(content) != effect.applied_bytes
            or "sha256:" + hashlib.sha256(content).hexdigest() != effect.applied_sha256
        ):
            failures.append(f"reviewed_effect_grok_bytes_mismatch:{effect.path}")
    return failures


def _packet_contract_failures(
    binding: ProductionReviewedEffectBinding,
    *,
    task: Any,
    task_identity: Any,
    repo_root: Path | None = None,
) -> list[str]:
    failures: list[str] = []
    try:
        identity = _task_identity_payload(task_identity)
        contract = production_task_contract(task, identity)
    except (TypeError, ValueError):
        return ["reviewed_effect_task_invalid"]
    if dict(binding.task_identity) != identity:
        failures.append("reviewed_effect_task_identity_mismatch")
    if binding.task_contract_cid != content_identity(contract):
        failures.append("reviewed_effect_task_contract_mismatch")
    if binding.packet_task_id != contract["task_id"]:
        failures.append("reviewed_effect_packet_task_identity_mismatch")
    payload = dict(binding.packet_payload)
    if binding.packet_cid != content_identity(payload):
        failures.append("reviewed_effect_packet_cid_mismatch")
    if binding.snapshot_id != f"git-commit:{binding.baseline_commit}":
        failures.append("reviewed_effect_snapshot_baseline_mismatch")
    goal = payload.get("goal") if isinstance(payload.get("goal"), Mapping) else {}
    scope = payload.get("scope") if isinstance(payload.get("scope"), Mapping) else {}
    acceptance = payload.get("acceptance") if isinstance(payload.get("acceptance"), Mapping) else {}
    if goal.get("task_id") != contract["task_id"]:
        failures.append("reviewed_effect_packet_task_mismatch")
    for key in ("title", "priority", "track"):
        if goal.get(key) != contract[key]:
            failures.append(f"reviewed_effect_packet_goal_mismatch:{key}")
    if scope.get("write_paths") != contract["outputs"]:
        failures.append("reviewed_effect_packet_write_scope_mismatch")
    if acceptance.get("validation_commands") != contract["validation"]:
        failures.append("reviewed_effect_packet_validation_mismatch")
    if acceptance.get("criteria") != contract["acceptance"]:
        failures.append("reviewed_effect_packet_acceptance_mismatch")
    if not set(binding.changed_paths).issubset(contract["outputs"]):
        failures.append("reviewed_effect_changed_paths_outside_task_scope")
    failures.extend(_context_binding_failures(binding))
    failures.extend(_provider_receipt_failures(binding))
    failures.extend(_proposal_payload_failures(binding, repo_root=repo_root))
    return failures


def capture_production_reviewed_effect(
    *,
    repo_root: str | Path,
    task: Any,
    task_identity: Any,
    packet: ProductionContractPacket | Mapping[str, Any],
    route_result: Any,
    baseline_ref: str,
) -> ProductionReviewedEffectBinding:
    """Capture the exact supervisor-applied effect immediately after writing."""

    root = _repository_root(repo_root)
    baseline = _resolve_commit(root, baseline_ref)
    head = _resolve_commit(root, "HEAD")
    if head != baseline:
        raise ValueError("reviewed effect capture requires HEAD at the baseline")
    if getattr(route_result, "status", None) is not RouteStatus.SUCCEEDED:
        raise ValueError("reviewed effect requires a succeeded provider route")
    if not getattr(route_result, "write_performed", False):
        raise ValueError("reviewed effect requires a supervisor-observed write")
    selected = getattr(route_result, "selected_proposal", None)
    implementation = getattr(route_result, "implementation_proposal", None)
    review = getattr(route_result, "review_proposal", None)
    if (
        selected is None
        or implementation is None
        or review is None
        or getattr(selected, "role", None) is not ProviderRole.GROK_IMPLEMENT
        or getattr(implementation, "role", None) is not ProviderRole.GROK_IMPLEMENT
        or getattr(review, "role", None) is not ProviderRole.CODEX_REVIEW
        or getattr(selected, "admitted", None) is not True
        or getattr(implementation, "admitted", None) is not True
        or getattr(review, "admitted", None) is not True
        or selected.response_digest != implementation.response_digest
        or dict(selected.payload) != dict(implementation.payload)
        or review.payload.get("decision") != "approve"
        or not isinstance(review.payload.get("findings"), list)
        or review.payload.get("proposal") not in (None, {})
    ):
        raise ValueError("reviewed effect requires Grok final bytes and Codex approval")
    body: Mapping[str, Any] = selected.payload
    nested = selected.payload.get("proposal")
    if isinstance(nested, Mapping):
        body = nested
    declared = body.get("declared_paths")
    if not isinstance(declared, (list, tuple)):
        raise ValueError("reviewed proposal requires declared_paths")
    declared_paths = tuple(sorted(_canonical_path(path) for path in declared))
    if not declared_paths or len(declared_paths) != len(set(declared_paths)):
        raise ValueError("reviewed proposal paths must be explicit and unique")
    changed_paths = _workspace_changed_paths(root, baseline)
    if changed_paths != declared_paths:
        raise ValueError("reviewed proposal paths do not match the applied Git effect")
    effects = tuple(_path_effect(root, baseline, path) for path in changed_paths)
    packet_mapping = _mapping(packet)
    packet_payload = _packet_payload(packet)
    packet_id = _text(getattr(packet, "packet_id", "") or packet_mapping.get("packet_id"))
    snapshot_id = _text(getattr(packet, "snapshot_id", "") or packet_mapping.get("snapshot_id"))
    packet_task_id = _text(getattr(packet, "task_id", "") or packet_mapping.get("task_id"))
    route_packet = getattr(route_result, "packet", None)
    route_packet_cid = _text(getattr(route_packet, "packet_cid", ""))
    if (
        not packet_id
        or packet_task_id != _text(getattr(task, "task_id", ""))
        or snapshot_id != f"git-commit:{baseline}"
        or _text(getattr(route_packet, "packet_id", "")) != packet_id
        or _text(getattr(route_packet, "task_id", "")) != packet_task_id
        or _text(getattr(route_packet, "snapshot_id", "")) != snapshot_id
        or route_packet_cid != content_identity(packet_payload)
    ):
        raise ValueError("reviewed effect packet/baseline identity is invalid")
    receipt = route_result.provider_receipt
    receipt_payload = _mapping(receipt)
    attempts = tuple(getattr(route_result, "attempts", ()) or ())
    policy_ids = {_text(getattr(attempt, "execution_policy_id", "")) for attempt in attempts}
    if len(attempts) != 2 or len(policy_ids) != 1 or not next(iter(policy_ids), ""):
        raise ValueError("reviewed effect requires one bound production provider policy")
    provider_policy_id = next(iter(policy_ids))
    context = packet_payload.get("context_slice")
    context = dict(context) if isinstance(context, Mapping) else {}
    task_binding = context.get("task_binding")
    task_binding = dict(task_binding) if isinstance(task_binding, Mapping) else {}
    repository_binding = context.get("repository_binding")
    repository_binding = dict(repository_binding) if isinstance(repository_binding, Mapping) else {}
    context_scope = context.get("scope")
    context_scope = dict(context_scope) if isinstance(context_scope, Mapping) else {}
    selected_payload = _mapping(selected.payload)
    review_payload = _mapping(review.payload)
    writer_lease_id = _text(getattr(route_result, "writer_lease_id", ""))
    if not writer_lease_id:
        raise ValueError("reviewed effect requires the supervisor writer lease")
    binding = ProductionReviewedEffectBinding.create(
        task_identity=_task_identity_payload(task_identity),
        task_contract_cid=production_task_contract_cid(task, task_identity),
        packet_id=packet_id,
        packet_task_id=packet_task_id,
        packet_cid=route_packet_cid,
        packet_payload=packet_payload,
        snapshot_id=snapshot_id,
        baseline_commit=baseline,
        baseline_tree_id=_tree_id(root, baseline),
        context_manifest_cid=_text(context.get("manifest_cid")),
        context_task_cid=_text(task_binding.get("canonical_task_cid")),
        context_snapshot_id=_text(repository_binding.get("snapshot_id")),
        context_scope_cid=_text(context_scope.get("scope_cid")),
        provider_policy_id=provider_policy_id,
        provider_receipt_cid=receipt.receipt_id,
        provider_receipt=receipt_payload,
        review_chain_digest=review_chain_content_digest(receipt.review_chain),
        selected_proposal_digest=selected.response_digest,
        selected_proposal_payload_cid=content_identity(selected_payload),
        selected_proposal_payload=selected_payload,
        implementation_proposal_digest=implementation.response_digest,
        review_proposal_digest=review.response_digest,
        review_proposal_payload_cid=content_identity(review_payload),
        review_proposal_payload=review_payload,
        writer_lease_id=writer_lease_id,
        changed_paths=changed_paths,
        path_effects=effects,
    )
    failures = _packet_contract_failures(
        binding,
        task=task,
        task_identity=task_identity,
        repo_root=root,
    )
    if failures:
        raise ValueError("reviewed effect task/packet mismatch: " + ",".join(failures))
    return ProductionReviewedEffectBinding.from_dict(binding.to_dict())


def verify_production_reviewed_workspace(
    binding: ProductionReviewedEffectBinding | Mapping[str, Any],
    *,
    repo_root: str | Path,
    task: Any,
    task_identity: Any,
    allowed_head_commit: str = "",
) -> ProductionReviewedEffectVerification:
    """Compare post-validation workspace facts with the post-write capture."""

    try:
        value = (
            binding
            if isinstance(binding, ProductionReviewedEffectBinding)
            else ProductionReviewedEffectBinding.from_dict(binding)
        )
    except (TypeError, ValueError):
        return ProductionReviewedEffectVerification(False, ("reviewed_effect_invalid",))
    try:
        root = _repository_root(repo_root)
    except ValueError:
        return ProductionReviewedEffectVerification(
            False,
            ("reviewed_effect_repository_root_invalid",),
            binding_id=value.binding_id,
            implementation_commit=value.implementation_commit,
            implementation_tree_id=value.implementation_tree_id,
        )
    failures = _packet_contract_failures(
        value,
        task=task,
        task_identity=task_identity,
        repo_root=root,
    )
    try:
        baseline = _resolve_commit(root, value.baseline_commit)
        head = _resolve_commit(root, "HEAD")
        expected_head = (
            _resolve_commit(root, allowed_head_commit) if _text(allowed_head_commit) else baseline
        )
        if baseline != value.baseline_commit or head != expected_head:
            failures.append("reviewed_effect_workspace_baseline_mismatch")
        if _tree_id(root, baseline) != value.baseline_tree_id:
            failures.append("reviewed_effect_workspace_baseline_tree_mismatch")
        changed = _workspace_changed_paths(root, baseline)
        if changed != value.changed_paths:
            failures.append("reviewed_effect_workspace_path_set_changed")
        else:
            observed = tuple(_path_effect(root, baseline, path) for path in changed)
            if tuple(effect.to_dict() for effect in observed) != tuple(
                effect.to_dict() for effect in value.path_effects
            ):
                failures.append("reviewed_effect_workspace_bytes_or_modes_changed")
    except (OSError, TypeError, ValueError):
        failures.append("reviewed_effect_workspace_reconstruction_failed")
    reasons = tuple(dict.fromkeys(failures))
    return ProductionReviewedEffectVerification(
        not reasons,
        reasons,
        binding_id=value.binding_id,
        implementation_commit=value.implementation_commit,
        implementation_tree_id=value.implementation_tree_id,
    )


def finalize_production_reviewed_effect(
    binding: ProductionReviewedEffectBinding | Mapping[str, Any],
    *,
    repo_root: str | Path,
    task: Any,
    task_identity: Any,
    implementation_commit: str,
) -> ProductionReviewedEffectBinding:
    """Bind a validated, unchanged capture to its exact commit and tree."""

    value = (
        binding
        if isinstance(binding, ProductionReviewedEffectBinding)
        else ProductionReviewedEffectBinding.from_dict(binding)
    )
    root = _repository_root(repo_root)
    commit = _resolve_commit(root, implementation_commit)
    workspace = verify_production_reviewed_workspace(
        value,
        repo_root=root,
        task=task,
        task_identity=task_identity,
        allowed_head_commit=commit,
    )
    if not workspace.admitted:
        raise ValueError(
            "post-validation reviewed effect changed: " + ",".join(workspace.reason_codes)
        )
    finalized = replace(
        value,
        binding_id="",
        implementation_commit=commit,
        implementation_tree_id=_tree_id(root, commit),
        implementation_diff_sha256="",
        implementation_diff_bytes=0,
    )
    diff = _commit_diff(root, value.baseline_commit, commit, value.changed_paths)
    finalized = replace(
        finalized,
        implementation_diff_sha256="sha256:" + hashlib.sha256(diff).hexdigest(),
        implementation_diff_bytes=len(diff),
    )
    finalized = replace(
        finalized,
        binding_id=content_identity(finalized.unsigned_dict()),
    )
    verification = verify_finalized_production_reviewed_effect(
        finalized,
        repo_root=root,
        task=task,
        task_identity=task_identity,
        expected_implementation_commit=commit,
        expected_implementation_tree_id=finalized.implementation_tree_id,
    )
    if not verification.admitted:
        raise ValueError(
            "implementation commit does not match reviewed effect: "
            + ",".join(verification.reason_codes)
        )
    return finalized


def verify_finalized_production_reviewed_effect(
    binding: ProductionReviewedEffectBinding | Mapping[str, Any] | None,
    *,
    repo_root: str | Path,
    task: Any,
    task_identity: Any,
    expected_implementation_commit: str,
    expected_implementation_tree_id: str,
) -> ProductionReviewedEffectVerification:
    """Recompute task and Git facts for an immutable reviewed-effect binding."""

    try:
        value = (
            binding
            if isinstance(binding, ProductionReviewedEffectBinding)
            else ProductionReviewedEffectBinding.from_dict(binding or {})
        )
    except (TypeError, ValueError):
        return ProductionReviewedEffectVerification(False, ("reviewed_effect_invalid",))
    try:
        root = _repository_root(repo_root)
    except ValueError:
        return ProductionReviewedEffectVerification(
            False,
            ("reviewed_effect_repository_root_invalid",),
            binding_id=value.binding_id,
            implementation_commit=_text(expected_implementation_commit),
            implementation_tree_id=_text(expected_implementation_tree_id),
        )
    failures = _packet_contract_failures(
        value,
        task=task,
        task_identity=task_identity,
        repo_root=root,
    )
    try:
        baseline = _resolve_commit(root, value.baseline_commit)
        commit = _resolve_commit(root, expected_implementation_commit)
        tree_id = _tree_id(root, commit)
        if baseline != value.baseline_commit:
            failures.append("reviewed_effect_baseline_not_canonical")
        if _tree_id(root, baseline) != value.baseline_tree_id:
            failures.append("reviewed_effect_baseline_tree_mismatch")
        if value.implementation_commit != commit:
            failures.append("reviewed_effect_implementation_commit_mismatch")
        if value.implementation_tree_id != tree_id or tree_id != _text(
            expected_implementation_tree_id
        ):
            failures.append("reviewed_effect_implementation_tree_mismatch")
        ancestry = subprocess.run(
            [
                "git",
                "--literal-pathspecs",
                "merge-base",
                "--is-ancestor",
                baseline,
                commit,
            ],
            cwd=root,
            env=_sanitized_git_environment(),
            capture_output=True,
            check=False,
        )
        if ancestry.returncode != 0:
            failures.append("reviewed_effect_baseline_not_ancestor")
        changed = _commit_changed_paths(root, baseline, commit)
        if changed != value.changed_paths:
            failures.append("reviewed_effect_commit_path_set_mismatch")
        else:
            for effect in value.path_effects:
                before = _tree_blob(root, baseline, effect.path)
                after = _tree_blob(root, commit, effect.path)
                if effect.status == "added":
                    if before is not None or effect.baseline_mode or effect.baseline_blob_oid:
                        failures.append(f"reviewed_effect_baseline_blob_mismatch:{effect.path}")
                elif (
                    before is None
                    or before[0] != effect.baseline_mode
                    or before[1] != effect.baseline_blob_oid
                ):
                    failures.append(f"reviewed_effect_baseline_blob_mismatch:{effect.path}")
                if effect.status == "deleted":
                    if (
                        after is not None
                        or effect.applied_git_mode
                        or effect.applied_blob_oid
                        or effect.applied_sha256
                        or effect.applied_bytes != 0
                        or effect.applied_filesystem_mode != 0
                    ):
                        failures.append(f"reviewed_effect_commit_blob_mismatch:{effect.path}")
                    continue
                if after is None:
                    failures.append(f"reviewed_effect_commit_blob_mismatch:{effect.path}")
                    continue
                mode, oid, content = after
                if (
                    effect.applied_git_mode not in {"100644", "100755"}
                    or (effect.applied_git_mode == "100755")
                    != bool(effect.applied_filesystem_mode & 0o111)
                    or mode != effect.applied_git_mode
                    or oid != effect.applied_blob_oid
                    or len(content) != effect.applied_bytes
                    or "sha256:" + hashlib.sha256(content).hexdigest() != effect.applied_sha256
                ):
                    failures.append(f"reviewed_effect_commit_blob_mismatch:{effect.path}")
        diff = _commit_diff(root, baseline, commit, value.changed_paths)
        if (
            value.implementation_diff_sha256 != "sha256:" + hashlib.sha256(diff).hexdigest()
            or value.implementation_diff_bytes != len(diff)
            or not diff
        ):
            failures.append("reviewed_effect_implementation_diff_mismatch")
    except (OSError, TypeError, ValueError):
        failures.append("reviewed_effect_git_reconstruction_failed")
        commit = _text(expected_implementation_commit)
        tree_id = _text(expected_implementation_tree_id)
    reasons = tuple(dict.fromkeys(failures))
    return ProductionReviewedEffectVerification(
        not reasons,
        reasons,
        binding_id=value.binding_id,
        implementation_commit=commit,
        implementation_tree_id=tree_id,
    )


__all__ = [
    "PRODUCTION_REVIEWED_EFFECT_BINDING_INTERFACE",
    "PRODUCTION_REVIEWED_EFFECT_BINDING_SCHEMA",
    "ProductionPathEffect",
    "ProductionReviewedEffectBinding",
    "ProductionReviewedEffectVerification",
    "capture_production_reviewed_effect",
    "finalize_production_reviewed_effect",
    "production_task_contract",
    "production_task_contract_cid",
    "verify_finalized_production_reviewed_effect",
    "verify_production_reviewed_workspace",
]
