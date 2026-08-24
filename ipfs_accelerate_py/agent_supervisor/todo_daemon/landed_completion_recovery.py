"""Closed Git proofs for database landed-completion revalidation.

The receipts in this module grant only one operation: rearming a failed
database task so a newer fenced claim can run the task's declared validations
against the current target tree.  They never grant task completion, merge
queue settlement, or provider-effect authority.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Final

from ..task_sources.intent_repository import MAX_BODY_BYTES as _MAX_TASK_BODY_JSON_BYTES

DATABASE_LANDED_COMPLETION_RECOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/" "database-landed-completion-revalidation@1"
)
DATABASE_LANDED_COMPLETION_CLAIM_SEED_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/" "database-landed-completion-claim-seed@1"
)

_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_MAX_OUTPUTS = 256
_MAX_PATH_BYTES = 1024
_MAX_MERGES = 4096
_MAX_ALIAS_CANDIDATES = 64

# A recovery receipt is nested inside both a retry/claim status receipt and the
# task's existing body.  Keep it far below IntentRepository's hard task-body
# bound so ordinary control fields and the original task projection retain
# substantial headroom.  The claim seed adds another identity wrapper and is
# bounded separately before it can enter a Portal projection.
_MAX_RECOVERY_RECEIPT_BYTES = _MAX_TASK_BODY_JSON_BYTES // 8
_MAX_CLAIM_SEED_BYTES = _MAX_TASK_BODY_JSON_BYTES // 4
_PLACEHOLDER_DIGEST = "sha256:" + ("0" * 64)

_RECOVERY_FIELDS = frozenset(
    {
        "schema",
        "disposition",
        "reason",
        "task_cid",
        "task_alias",
        "source_attempt_id",
        "source_claim_id",
        "source_lease_id",
        "source_owner_session_id",
        "source_attempt_number",
        "source_fencing_token",
        "source_fence_epoch",
        "source_execution_revision",
        "source_execution_finished_at_ms",
        "source_control_revision",
        "candidate_commit",
        "candidate_parent",
        "candidate_tree",
        "integrating_merge",
        "integrating_first_parent",
        "integrating_tree",
        "target_ref",
        "target_commit_at_rearm",
        "declared_outputs",
        "proof_id",
    }
)

_CLAIM_SEED_FIELDS = frozenset(
    {
        "schema",
        "recovery_receipt",
        "target_task_cid",
        "target_task_alias",
        "target_attempt_id",
        "target_claim_id",
        "target_owner_session_id",
        "target_attempt_number",
        "target_fencing_token",
        "target_fence_epoch",
        "target_lease_id",
        "validated_target_commit",
        "validated_target_tree",
        "seed_id",
    }
)


class LandedCompletionRecoveryError(RuntimeError):
    """The proposed landed candidate did not satisfy the closed proof."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(dict(value))).hexdigest()


def _require_serialized_bound(
    value: Mapping[str, Any],
    *,
    noun: str,
    maximum: int,
) -> None:
    if len(_canonical_json(dict(value))) > maximum:
        raise LandedCompletionRecoveryError(
            f"{noun} exceeds its conservative serialized byte bound"
        )


def _safe_output(value: Any) -> str:
    if type(value) is not str:
        raise LandedCompletionRecoveryError("declared output is not a string")
    path = PurePosixPath(value or ".")
    if (
        not value
        or value != value.strip()
        or len(value.encode("utf-8", errors="surrogatepass")) > _MAX_PATH_BYTES
        or "\\" in value
        or path.is_absolute()
        or bool(PureWindowsPath(value).drive)
        or path == PurePosixPath(".")
        or path.as_posix() != value
        or ".." in path.parts
        or any(ord(character) < 32 for character in value)
    ):
        raise LandedCompletionRecoveryError("declared output is unsafe or noncanonical")
    return value


def _outputs(values: Any) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        raise LandedCompletionRecoveryError("declared outputs are not a sequence")
    if not values or len(values) > _MAX_OUTPUTS:
        raise LandedCompletionRecoveryError(
            "declared outputs are empty, duplicated, or outside the closed bound"
        )
    result = tuple(_safe_output(value) for value in values)
    if len(set(result)) != len(result):
        raise LandedCompletionRecoveryError(
            "declared outputs are empty, duplicated, or outside the closed bound"
        )
    return result


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LandedCompletionRecoveryError(f"{field} is not a positive integer")
    return int(value)


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LandedCompletionRecoveryError(f"{field} is not a nonnegative integer")
    return int(value)


def _commit(value: Any, field: str) -> str:
    selected = value if type(value) is str else ""
    if _COMMIT.fullmatch(selected) is None:
        raise LandedCompletionRecoveryError(f"{field} is not an exact commit")
    return selected


def _run_git(repo_root: Path, argv: Sequence[str], *, timeout: int = 15) -> str:
    try:
        result = subprocess.run(
            ["git", *argv],
            cwd=repo_root,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LandedCompletionRecoveryError(
            "Git proof command was unavailable"
        ) from exc
    if result.returncode != 0:
        raise LandedCompletionRecoveryError(
            f"Git proof command failed: {' '.join(argv[:2])}"
        )
    return result.stdout.strip()


def _resolve_commit(repo_root: Path, ref: str) -> str:
    return _commit(
        _run_git(repo_root, ["rev-parse", "--verify", f"{ref}^{{commit}}"]),
        "resolved_commit",
    )


def _resolve_tree(repo_root: Path, commit: str) -> str:
    return _commit(
        _run_git(repo_root, ["rev-parse", "--verify", f"{commit}^{{tree}}"]),
        "resolved_tree",
    )


def _parents(repo_root: Path, commit: str) -> tuple[str, ...]:
    row = _run_git(repo_root, ["rev-list", "--parents", "-n", "1", commit])
    values = tuple(row.split())
    if not values or values[0] != commit:
        raise LandedCompletionRecoveryError("commit parent identity changed")
    return tuple(_commit(value, "parent") for value in values[1:])


def _changed_paths(repo_root: Path, before: str, after: str) -> tuple[str, ...]:
    raw = _run_git(
        repo_root,
        [
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            "--no-renames",
            before,
            after,
        ],
    )
    values = tuple(line for line in raw.splitlines() if line)
    return tuple(_safe_output(value) for value in values)


def _is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _task_alias_in_candidate(repo_root: Path, candidate: str, alias: str) -> bool:
    if not alias or any(ord(character) < 32 for character in alias):
        return False
    message = _run_git(repo_root, ["show", "-s", "--format=%B", candidate])
    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    return pattern.search(message) is not None


def _same_output_blobs(
    repo_root: Path,
    candidate: str,
    integration: str,
    outputs: Sequence[str],
) -> bool:
    for output in outputs:
        try:
            candidate_blob = _run_git(
                repo_root,
                ["rev-parse", "--verify", f"{candidate}:{output}"],
            )
            integration_blob = _run_git(
                repo_root,
                ["rev-parse", "--verify", f"{integration}:{output}"],
            )
        except LandedCompletionRecoveryError:
            return False
        if candidate_blob != integration_blob:
            return False
    return True


def verify_landed_completion_recovery_receipt(
    raw: Mapping[str, Any] | None,
    *,
    task_cid: str = "",
    task_alias: str = "",
    source_attempt_id: str = "",
) -> dict[str, Any]:
    """Verify the static, content-addressed recovery receipt."""

    if not isinstance(raw, Mapping) or set(raw) != _RECOVERY_FIELDS:
        raise LandedCompletionRecoveryError(
            "landed recovery receipt has unknown or missing fields"
        )
    value = dict(raw)
    proof_id = value.pop("proof_id", None)
    outputs = _outputs(value.get("declared_outputs"))
    _require_serialized_bound(
        {**value, "proof_id": proof_id},
        noun="landed recovery receipt",
        maximum=_MAX_RECOVERY_RECEIPT_BYTES,
    )
    for field in (
        "candidate_commit",
        "candidate_parent",
        "candidate_tree",
        "integrating_merge",
        "integrating_first_parent",
        "integrating_tree",
        "target_commit_at_rearm",
    ):
        _commit(value.get(field), field)
    for field in (
        "source_attempt_number",
        "source_fencing_token",
        "source_execution_revision",
        "source_control_revision",
    ):
        _positive_int(value.get(field), field)
    _nonnegative_int(value.get("source_fence_epoch"), "source_fence_epoch")
    _positive_int(
        value.get("source_execution_finished_at_ms"),
        "source_execution_finished_at_ms",
    )
    identity_invalid = bool(
        value.get("schema") != DATABASE_LANDED_COMPLETION_RECOVERY_SCHEMA
        or value.get("disposition") != "fresh_validation"
        or value.get("reason") != "landed_candidate_requires_fresh_validation"
        or type(value.get("task_cid")) is not str
        or not value.get("task_cid")
        or type(value.get("task_alias")) is not str
        or not value.get("task_alias")
        or type(value.get("source_attempt_id")) is not str
        or not value.get("source_attempt_id")
        or type(value.get("source_claim_id")) is not str
        or not value.get("source_claim_id")
        or type(value.get("source_lease_id")) is not str
        or not value.get("source_lease_id")
        or type(value.get("source_owner_session_id")) is not str
        or not value.get("source_owner_session_id")
        or type(value.get("target_ref")) is not str
        or not value.get("target_ref")
        or list(outputs) != value.get("declared_outputs")
        or (task_cid and value.get("task_cid") != task_cid)
        or (task_alias and value.get("task_alias") != task_alias)
        or (source_attempt_id and value.get("source_attempt_id") != source_attempt_id)
        or proof_id != _digest(value)
    )
    if identity_invalid:
        raise LandedCompletionRecoveryError(
            "landed recovery receipt failed identity verification"
        )
    return {**value, "proof_id": str(proof_id)}


def revalidate_landed_completion_repository(
    raw: Mapping[str, Any],
    *,
    repo_root: Path | str,
    target_ref: str = "",
) -> dict[str, str]:
    """Reproduce the Git proof against the current target without mutation."""

    receipt = verify_landed_completion_recovery_receipt(raw)
    root = Path(repo_root).resolve(strict=True)
    selected_target = str(target_ref or receipt["target_ref"])
    if selected_target != receipt["target_ref"]:
        raise LandedCompletionRecoveryError(
            "landed recovery target differs from configured merge target"
        )
    current_target = _resolve_commit(root, selected_target)
    candidate = _resolve_commit(root, str(receipt["candidate_commit"]))
    integration = _resolve_commit(root, str(receipt["integrating_merge"]))
    candidate_parents = _parents(root, candidate)
    integration_parents = _parents(root, integration)
    outputs = tuple(receipt["declared_outputs"])
    if (
        candidate != receipt["candidate_commit"]
        or integration != receipt["integrating_merge"]
        or candidate_parents != (receipt["candidate_parent"],)
        or integration_parents != (receipt["integrating_first_parent"], candidate)
        or _resolve_tree(root, candidate) != receipt["candidate_tree"]
        or _resolve_tree(root, integration) != receipt["integrating_tree"]
        or set(_changed_paths(root, candidate_parents[0], candidate)) != set(outputs)
        or set(_changed_paths(root, integration_parents[0], integration))
        != set(outputs)
        or not _task_alias_in_candidate(root, candidate, str(receipt["task_alias"]))
        or not _same_output_blobs(root, candidate, integration, outputs)
        or not _is_ancestor(
            root, str(receipt["target_commit_at_rearm"]), current_target
        )
        or not _is_ancestor(root, integration, current_target)
    ):
        raise LandedCompletionRecoveryError(
            "landed candidate no longer satisfies its repository proof"
        )
    return {
        "current_target_commit": current_target,
        "current_target_tree": _resolve_tree(root, current_target),
    }


def discover_landed_completion_recovery(
    *,
    repo_root: Path | str,
    target_ref: str,
    task_cid: str,
    task_alias: str,
    declared_outputs: Sequence[str],
    source_attempt_id: str,
    source_claim_id: str,
    source_lease_id: str,
    source_owner_session_id: str,
    source_attempt_number: int,
    source_fencing_token: int,
    source_fence_epoch: int,
    source_execution_revision: int,
    source_execution_finished_at_ms: int,
    source_control_revision: int,
) -> dict[str, Any] | None:
    """Discover one unique already-integrated candidate for fresh validation.

    Ambiguity, malformed history, missing outputs, or any Git error returns no
    authority.  The caller may leave the task blocked and try again later.
    """

    try:
        outputs = _outputs(tuple(declared_outputs))
        root = Path(repo_root).resolve(strict=True)
        target_commit = _resolve_commit(root, target_ref)
        raw_candidates = _run_git(
            root,
            [
                "log",
                "--format=%H",
                "--fixed-strings",
                "--regexp-ignore-case",
                f"--grep={task_alias}",
                f"--max-count={_MAX_ALIAS_CANDIDATES + 1}",
                target_commit,
            ],
            timeout=15,
        )
        candidate_lines = tuple(line for line in raw_candidates.splitlines() if line)
        if len(candidate_lines) > _MAX_ALIAS_CANDIDATES:
            # The sentinel row proves additional alias history exists outside
            # the closed scan.  Do not mistake uniqueness in a truncated
            # prefix for global uniqueness.
            return None
        candidate_rows: dict[str, tuple[str, str]] = {}
        for candidate in candidate_lines:
            candidate = _commit(candidate, "candidate_commit")
            candidate_parents = _parents(root, candidate)
            if len(candidate_parents) != 1:
                continue
            if not _task_alias_in_candidate(root, candidate, task_alias):
                continue
            if set(_changed_paths(root, candidate_parents[0], candidate)) != set(
                outputs
            ):
                continue
            candidate_rows[candidate] = (
                candidate_parents[0],
                _resolve_tree(root, candidate),
            )
        if not candidate_rows:
            return None

        # One bounded parent projection replaces a subprocess per merge.  We
        # then inspect only merges whose second parent is one of the already
        # qualified task-alias candidates (normally exactly one).
        raw_merges = _run_git(
            root,
            [
                "rev-list",
                "--merges",
                "--parents",
                f"--max-count={_MAX_MERGES + 1}",
                target_commit,
            ],
            timeout=30,
        )
        merge_lines = tuple(line for line in raw_merges.splitlines() if line)
        if len(merge_lines) > _MAX_MERGES:
            # As above, the extra row is a truncation sentinel.  Recovery is
            # fail-closed until the configured exhaustive window can cover the
            # target's merge history.
            return None
        matches: list[dict[str, Any]] = []
        for row in merge_lines:
            values = tuple(row.split())
            if len(values) != 3:
                continue
            integration, first_parent, candidate = (
                _commit(values[0], "integrating_merge"),
                _commit(values[1], "integrating_first_parent"),
                _commit(values[2], "candidate_commit"),
            )
            candidate_row = candidate_rows.get(candidate)
            if candidate_row is None:
                continue
            if set(_changed_paths(root, first_parent, integration)) != set(outputs):
                continue
            if not _same_output_blobs(root, candidate, integration, outputs):
                continue
            matches.append(
                {
                    "candidate_commit": candidate,
                    "candidate_parent": candidate_row[0],
                    "candidate_tree": candidate_row[1],
                    "integrating_merge": integration,
                    "integrating_first_parent": first_parent,
                    "integrating_tree": _resolve_tree(root, integration),
                }
            )
        unique = {
            (item["candidate_commit"], item["integrating_merge"]): item
            for item in matches
        }
        if len(unique) != 1:
            return None
        landed = next(iter(unique.values()))
        body: dict[str, Any] = {
            "schema": DATABASE_LANDED_COMPLETION_RECOVERY_SCHEMA,
            "disposition": "fresh_validation",
            "reason": "landed_candidate_requires_fresh_validation",
            "task_cid": str(task_cid),
            "task_alias": str(task_alias),
            "source_attempt_id": str(source_attempt_id),
            "source_claim_id": str(source_claim_id),
            "source_lease_id": str(source_lease_id),
            "source_owner_session_id": str(source_owner_session_id),
            "source_attempt_number": _positive_int(
                source_attempt_number, "source_attempt_number"
            ),
            "source_fencing_token": _positive_int(
                source_fencing_token, "source_fencing_token"
            ),
            "source_fence_epoch": _nonnegative_int(
                source_fence_epoch, "source_fence_epoch"
            ),
            "source_execution_revision": _positive_int(
                source_execution_revision, "source_execution_revision"
            ),
            "source_execution_finished_at_ms": _positive_int(
                source_execution_finished_at_ms,
                "source_execution_finished_at_ms",
            ),
            "source_control_revision": _positive_int(
                source_control_revision, "source_control_revision"
            ),
            **landed,
            "target_ref": str(target_ref),
            "target_commit_at_rearm": target_commit,
            "declared_outputs": list(outputs),
        }
        _require_serialized_bound(
            {**body, "proof_id": _PLACEHOLDER_DIGEST},
            noun="landed recovery receipt",
            maximum=_MAX_RECOVERY_RECEIPT_BYTES,
        )
        body["proof_id"] = _digest(body)
        receipt = verify_landed_completion_recovery_receipt(
            body,
            task_cid=task_cid,
            task_alias=task_alias,
            source_attempt_id=source_attempt_id,
        )
        revalidate_landed_completion_repository(
            receipt,
            repo_root=root,
            target_ref=target_ref,
        )
        return receipt
    except (LandedCompletionRecoveryError, OSError, ValueError):
        return None


def build_landed_completion_claim_seed(
    recovery_receipt: Mapping[str, Any],
    *,
    target_task_cid: str,
    target_task_alias: str,
    target_attempt_id: str,
    target_claim_id: str,
    target_owner_session_id: str,
    target_attempt_number: int,
    target_fencing_token: int,
    target_fence_epoch: int,
    target_lease_id: str,
    validated_target_commit: str,
    validated_target_tree: str,
) -> dict[str, Any]:
    recovery = verify_landed_completion_recovery_receipt(
        recovery_receipt,
        task_cid=target_task_cid,
        task_alias=target_task_alias,
    )
    body: dict[str, Any] = {
        "schema": DATABASE_LANDED_COMPLETION_CLAIM_SEED_SCHEMA,
        "recovery_receipt": recovery,
        "target_task_cid": str(target_task_cid),
        "target_task_alias": str(target_task_alias),
        "target_attempt_id": str(target_attempt_id),
        "target_claim_id": str(target_claim_id),
        "target_owner_session_id": str(target_owner_session_id),
        "target_attempt_number": _positive_int(
            target_attempt_number, "target_attempt_number"
        ),
        "target_fencing_token": _positive_int(
            target_fencing_token, "target_fencing_token"
        ),
        "target_fence_epoch": _nonnegative_int(
            target_fence_epoch, "target_fence_epoch"
        ),
        "target_lease_id": str(target_lease_id),
        "validated_target_commit": _commit(
            validated_target_commit, "validated_target_commit"
        ),
        "validated_target_tree": _commit(
            validated_target_tree, "validated_target_tree"
        ),
    }
    _require_serialized_bound(
        {**body, "seed_id": _PLACEHOLDER_DIGEST},
        noun="landed recovery claim seed",
        maximum=_MAX_CLAIM_SEED_BYTES,
    )
    body["seed_id"] = _digest(body)
    return verify_landed_completion_claim_seed(body)


def verify_landed_completion_claim_seed(
    raw: Mapping[str, Any] | None,
    *,
    task_cid: str = "",
    task_alias: str = "",
    target_attempt_id: str = "",
    target_claim_id: str = "",
    target_owner_session_id: str = "",
) -> dict[str, Any]:
    """Verify the recovery receipt bound to one newer database claim."""

    if not isinstance(raw, Mapping) or set(raw) != _CLAIM_SEED_FIELDS:
        raise LandedCompletionRecoveryError(
            "landed recovery claim seed has unknown or missing fields"
        )
    value = dict(raw)
    seed_id = value.pop("seed_id", None)
    _require_serialized_bound(
        {**value, "seed_id": seed_id},
        noun="landed recovery claim seed",
        maximum=_MAX_CLAIM_SEED_BYTES,
    )
    recovery = verify_landed_completion_recovery_receipt(
        value.get("recovery_receipt"),
        task_cid=str(value.get("target_task_cid") or ""),
        task_alias=str(value.get("target_task_alias") or ""),
    )
    source_number = _positive_int(
        recovery.get("source_attempt_number"), "source_attempt_number"
    )
    source_fencing_token = _positive_int(
        recovery.get("source_fencing_token"), "source_fencing_token"
    )
    source_fence_epoch = _nonnegative_int(
        recovery.get("source_fence_epoch"), "source_fence_epoch"
    )
    target_number = _positive_int(
        value.get("target_attempt_number"), "target_attempt_number"
    )
    target_fencing_token = _positive_int(
        value.get("target_fencing_token"), "target_fencing_token"
    )
    target_fence_epoch = _nonnegative_int(
        value.get("target_fence_epoch"), "target_fence_epoch"
    )
    _commit(value.get("validated_target_commit"), "validated_target_commit")
    _commit(value.get("validated_target_tree"), "validated_target_tree")
    invalid = bool(
        value.get("schema") != DATABASE_LANDED_COMPLETION_CLAIM_SEED_SCHEMA
        or value.get("recovery_receipt") != recovery
        or type(value.get("target_task_cid")) is not str
        or not value.get("target_task_cid")
        or type(value.get("target_task_alias")) is not str
        or not value.get("target_task_alias")
        or type(value.get("target_attempt_id")) is not str
        or not value.get("target_attempt_id")
        or type(value.get("target_claim_id")) is not str
        or not value.get("target_claim_id")
        or type(value.get("target_owner_session_id")) is not str
        or not value.get("target_owner_session_id")
        or type(value.get("target_lease_id")) is not str
        or not value.get("target_lease_id")
        or target_number <= source_number
        or target_fencing_token <= source_fencing_token
        or target_fence_epoch < source_fence_epoch
        or (task_cid and value.get("target_task_cid") != task_cid)
        or (task_alias and value.get("target_task_alias") != task_alias)
        or (target_attempt_id and value.get("target_attempt_id") != target_attempt_id)
        or target_claim_id
        and value.get("target_claim_id") != target_claim_id
        or (
            target_owner_session_id
            and value.get("target_owner_session_id") != target_owner_session_id
        )
        or seed_id != _digest(value)
    )
    if invalid:
        raise LandedCompletionRecoveryError(
            "landed recovery claim seed failed identity verification"
        )
    return {**value, "seed_id": str(seed_id)}


__all__ = (
    "DATABASE_LANDED_COMPLETION_CLAIM_SEED_SCHEMA",
    "DATABASE_LANDED_COMPLETION_RECOVERY_SCHEMA",
    "LandedCompletionRecoveryError",
    "build_landed_completion_claim_seed",
    "discover_landed_completion_recovery",
    "revalidate_landed_completion_repository",
    "verify_landed_completion_claim_seed",
    "verify_landed_completion_recovery_receipt",
)
