"""Fail-closed verification for operator-issued manual completion seals.

Manual task status is scheduling metadata, not authority.  This module verifies
an external, content-addressed receipt before a scheduler may activate artifacts
that become protected after an operator review task completes.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class ManualCompletionSealError(ValueError):
    """Raised when an operator seal cannot be reconstructed exactly."""


_RECEIPT_FIELDS = {
    "artifacts",
    "board_namespace",
    "decision",
    "grant",
    "interface",
    "operator",
    "policy_revision",
    "receipt_id",
    "receipt_version",
    "reviewed_base",
    "schema",
    "task_id",
}
_REVIEWED_BASE_FIELDS = {
    "commit",
    "git_object_format",
    "relation_to_activation_head",
    "tree",
}
_ARTIFACT_FIELDS = {"path", "role", "sha256", "size_bytes"}
INTERACTIVE_OPERATOR = {
    "identity": "interactive_user",
    "authority_basis": "interactive_user_delegation",
    "candidate": False,
    "model": False,
    "automatic_controller": False,
}
# Explicit scheduler-delegated completion: honest automatic controller, not a
# forged interactive-user identity.  Callers must opt in via
# ``allow_delegated_operator`` / scheduler policy.
DELEGATED_SUPERVISOR_OPERATOR = {
    "identity": "delegated_supervisor",
    "authority_basis": "scheduler_delegated_operator_completion@1",
    "candidate": False,
    "model": False,
    "automatic_controller": True,
}
_OPERATOR = INTERACTIVE_OPERATOR
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_GIT_SHA1_PATTERN = re.compile(r"[0-9a-f]{40}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManualCompletionSealError(
                f"duplicate JSON key in manual completion seal: {key!r}"
            )
        result[key] = value
    return result


def _reject_noncanonical_number(value: str) -> Any:
    raise ManualCompletionSealError(
        f"non-canonical number in manual completion seal: {value!r}"
    )


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, Mapping):
        for key in sorted(value):
            yield from _walk(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ManualCompletionSealError(
            "manual completion seal is not canonical JSON"
        ) from exc


def _receipt_identity(receipt: Mapping[str, Any]) -> str:
    body = {
        key: copy.deepcopy(value)
        for key, value in receipt.items()
        if key != "receipt_id"
    }
    return "sha256:" + hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _safe_file(root: Path, relative: str, *, label: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or relative.startswith(("/", "\\"))
        or relative.endswith(("/", "\\"))
        or "\\" in relative
        or "\0" in relative
        or "://" in relative
        or re.match(r"^[A-Za-z]:", relative)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ManualCompletionSealError(
            f"{label} must be a safe exact repository-relative file"
        )
    unresolved = root / relative
    if unresolved.is_symlink():
        raise ManualCompletionSealError(f"{label} must not be a symlink")
    try:
        target = unresolved.resolve(strict=True)
        target.relative_to(root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise ManualCompletionSealError(
            f"{label} is missing or escapes the repository"
        ) from exc
    if not target.is_file():
        raise ManualCompletionSealError(f"{label} must be an exact file")
    return target


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ManualCompletionSealError(
            "cannot reconstruct reviewed Git base"
        ) from exc


def load_strict_manual_completion_seal(path: Path) -> dict[str, Any]:
    """Load one unique-key, integer-only UTF-8 seal object."""

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_float=_reject_noncanonical_number,
            parse_constant=_reject_noncanonical_number,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManualCompletionSealError(
            "manual completion seal is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise ManualCompletionSealError(
            "manual completion seal root must be an object"
        )
    return payload


def build_manual_completion_seal(
    *,
    repo_root: Path,
    task_id: str,
    board_namespace: str,
    schema: str,
    interface: str,
    policy_revision: str,
    artifact_paths: Mapping[str, str],
    grant_type: str,
    grant_action: str,
    reviewed_base_claims: Mapping[str, Any] | None = None,
    grant_claims: Mapping[str, Any] | None = None,
    operator: Mapping[str, Any] | None = None,
    commit: str | None = None,
    tree: str | None = None,
) -> dict[str, Any]:
    """Build one activation-only seal for the current repository tree."""

    root = repo_root.resolve()
    if commit is None or tree is None:
        head = _git(root, "rev-parse", "HEAD")
        head_tree = _git(root, "rev-parse", "HEAD^{tree}")
        if head.returncode != 0 or head_tree.returncode != 0:
            raise ManualCompletionSealError(
                "cannot resolve HEAD for manual completion seal"
            )
        commit = head.stdout.strip()
        tree = head_tree.stdout.strip()
    if (
        not isinstance(commit, str)
        or not _GIT_SHA1_PATTERN.fullmatch(commit)
        or not isinstance(tree, str)
        or not _GIT_SHA1_PATTERN.fullmatch(tree)
    ):
        raise ManualCompletionSealError(
            "manual completion seal reviewed Git IDs are malformed"
        )
    artifacts: list[dict[str, Any]] = []
    for role, relative in artifact_paths.items():
        payload = _safe_file(
            root,
            relative,
            label=f"sealed artifact {role}",
        ).read_bytes()
        artifacts.append(
            {
                "role": role,
                "path": relative,
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    extra_grant_claims = dict(grant_claims or {})
    operator_payload = dict(operator or INTERACTIVE_OPERATOR)
    receipt: dict[str, Any] = {
        "schema": schema,
        "interface": interface,
        "receipt_version": "1",
        "task_id": task_id,
        "board_namespace": board_namespace,
        "decision": "sealed",
        "policy_revision": policy_revision,
        "reviewed_base": {
            "commit": commit,
            "tree": tree,
            "git_object_format": "sha1",
            "relation_to_activation_head": "equal_or_ancestor",
            **dict(reviewed_base_claims or {}),
        },
        "artifacts": artifacts,
        "operator": operator_payload,
        "grant": {
            "type": grant_type,
            "allowed_actions": [grant_action],
            **extra_grant_claims,
            "board_namespace": board_namespace,
            "policy_revision": policy_revision,
            "delegable": False,
            "mutation_authority": False,
            "completion_authority": False,
            "promotion_authority": False,
            "task_status_authority": False,
            "protected_anchor_write_authority": False,
        },
    }
    receipt["receipt_id"] = _receipt_identity(receipt)
    return receipt


def write_manual_completion_seal(
    receipt_path: str,
    receipt: Mapping[str, Any],
    *,
    repo_root: Path,
) -> Path:
    """Atomically write one seal object as canonical sorted JSON."""

    root = repo_root.resolve()
    relative = str(receipt_path)
    if (
        not relative
        or relative.startswith(("/", "\\"))
        or ".." in Path(relative).parts
    ):
        raise ManualCompletionSealError(
            "manual completion seal path must be a safe repository-relative file"
        )
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(dict(receipt), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    )
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)
    return target


def verify_manual_completion_seal(
    receipt_path: str,
    *,
    repo_root: Path,
    task_id: str,
    board_namespace: str,
    schema: str,
    interface: str,
    policy_revision: str,
    expected_receipt_id: str,
    artifact_paths: Mapping[str, str],
    grant_type: str,
    grant_action: str,
    reviewed_base_claims: Mapping[str, Any] | None = None,
    grant_claims: Mapping[str, Any] | None = None,
    allow_delegated_operator: bool = False,
) -> dict[str, Any]:
    """Reconstruct an activation-only operator receipt from current bytes."""

    root = repo_root.resolve()
    receipt_file = _safe_file(
        root,
        receipt_path,
        label="manual completion seal",
    )
    receipt = load_strict_manual_completion_seal(receipt_file)
    if set(receipt) != _RECEIPT_FIELDS:
        raise ManualCompletionSealError(
            "manual completion seal fields do not match the closed schema"
        )
    if any(isinstance(value, float) for value in _walk(receipt)):
        raise ManualCompletionSealError(
            "floating-point manual completion seal values are forbidden"
        )
    exact_scalars = {
        "schema": schema,
        "interface": interface,
        "receipt_version": "1",
        "task_id": task_id,
        "board_namespace": board_namespace,
        "decision": "sealed",
        "policy_revision": policy_revision,
    }
    for field_name, expected in exact_scalars.items():
        if receipt.get(field_name) != expected:
            raise ManualCompletionSealError(
                f"manual completion seal {field_name} mismatch"
            )
    if (
        not isinstance(expected_receipt_id, str)
        or not _SHA256_PATTERN.fullmatch(expected_receipt_id)
    ):
        raise ManualCompletionSealError(
            "pinned manual completion seal identity is malformed"
        )
    receipt_id = receipt.get("receipt_id")
    if (
        not isinstance(receipt_id, str)
        or not _SHA256_PATTERN.fullmatch(receipt_id)
    ):
        raise ManualCompletionSealError(
            "manual completion seal identity is malformed"
        )
    if not hmac.compare_digest(receipt_id, expected_receipt_id):
        raise ManualCompletionSealError(
            "manual completion seal does not match its protected pinned identity"
        )
    if not hmac.compare_digest(receipt_id, _receipt_identity(receipt)):
        raise ManualCompletionSealError(
            "manual completion seal identity mismatch"
        )

    reviewed_base = receipt.get("reviewed_base")
    expected_reviewed_claims = dict(reviewed_base_claims or {})
    if (
        not isinstance(reviewed_base, Mapping)
        or set(reviewed_base)
        != _REVIEWED_BASE_FIELDS | set(expected_reviewed_claims)
        or reviewed_base.get("git_object_format") != "sha1"
        or reviewed_base.get("relation_to_activation_head")
        != "equal_or_ancestor"
    ):
        raise ManualCompletionSealError(
            "manual completion seal reviewed base is invalid"
        )
    for field_name, expected in expected_reviewed_claims.items():
        if reviewed_base.get(field_name) != expected:
            raise ManualCompletionSealError(
                f"manual completion seal reviewed-base claim {field_name} "
                "mismatch"
            )
    commit = reviewed_base.get("commit")
    tree = reviewed_base.get("tree")
    if (
        not isinstance(commit, str)
        or not _GIT_SHA1_PATTERN.fullmatch(commit)
        or not isinstance(tree, str)
        or not _GIT_SHA1_PATTERN.fullmatch(tree)
    ):
        raise ManualCompletionSealError(
            "manual completion seal reviewed Git IDs are malformed"
        )
    commit_type = _git(root, "cat-file", "-t", commit)
    tree_type = _git(root, "cat-file", "-t", tree)
    commit_tree = _git(root, "rev-parse", f"{commit}^{{tree}}")
    ancestor = _git(root, "merge-base", "--is-ancestor", commit, "HEAD")
    if (
        commit_type.returncode != 0
        or commit_type.stdout.strip() != "commit"
        or tree_type.returncode != 0
        or tree_type.stdout.strip() != "tree"
        or commit_tree.returncode != 0
        or commit_tree.stdout.strip() != tree
        or ancestor.returncode != 0
    ):
        raise ManualCompletionSealError(
            "manual completion seal reviewed Git base is unavailable or stale"
        )

    artifacts = receipt.get("artifacts")
    if (
        not isinstance(artifacts, list)
        or len(artifacts) != len(artifact_paths)
    ):
        raise ManualCompletionSealError(
            "manual completion seal artifact population mismatch"
        )
    by_role: dict[str, Mapping[str, Any]] = {}
    seen_paths: set[str] = set()
    for item in artifacts:
        if not isinstance(item, Mapping) or set(item) != _ARTIFACT_FIELDS:
            raise ManualCompletionSealError(
                "manual completion seal artifact fields are invalid"
            )
        role = item.get("role")
        relative = item.get("path")
        if (
            not isinstance(role, str)
            or role in by_role
            or not isinstance(relative, str)
            or relative in seen_paths
        ):
            raise ManualCompletionSealError(
                "manual completion seal artifact roles and paths must be unique"
            )
        by_role[role] = item
        seen_paths.add(relative)
    if set(by_role) != set(artifact_paths):
        raise ManualCompletionSealError(
            "manual completion seal artifact roles mismatch"
        )
    if seen_paths != set(artifact_paths.values()):
        raise ManualCompletionSealError(
            "manual completion seal artifact paths mismatch"
        )
    for role, relative in artifact_paths.items():
        item = by_role[role]
        if item.get("path") != relative:
            raise ManualCompletionSealError(
                "manual completion seal artifact role/path mismatch"
            )
        payload = _safe_file(
            root,
            relative,
            label=f"sealed artifact {role}",
        ).read_bytes()
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        if item.get("sha256") != digest:
            raise ManualCompletionSealError(
                f"sealed artifact digest mismatch for {relative}"
            )
        if (
            type(item.get("size_bytes")) is not int
            or item.get("size_bytes") != len(payload)
        ):
            raise ManualCompletionSealError(
                f"sealed artifact byte count mismatch for {relative}"
            )

    allowed_operators = [INTERACTIVE_OPERATOR]
    if allow_delegated_operator:
        allowed_operators.append(DELEGATED_SUPERVISOR_OPERATOR)
    if receipt.get("operator") not in allowed_operators:
        raise ManualCompletionSealError(
            "manual completion seal operator or delegation basis mismatch"
        )
    extra_grant_claims = dict(grant_claims or {})
    protected_grant_fields = {
        "allowed_actions",
        "board_namespace",
        "completion_authority",
        "delegable",
        "mutation_authority",
        "policy_revision",
        "promotion_authority",
        "protected_anchor_write_authority",
        "task_status_authority",
        "type",
    }
    if set(extra_grant_claims) & protected_grant_fields:
        raise ManualCompletionSealError(
            "manual completion seal extra grant claims overlap authority fields"
        )
    expected_grant = {
        "type": grant_type,
        "allowed_actions": [grant_action],
        **extra_grant_claims,
        "board_namespace": board_namespace,
        "policy_revision": policy_revision,
        "delegable": False,
        "mutation_authority": False,
        "completion_authority": False,
        "promotion_authority": False,
        "task_status_authority": False,
        "protected_anchor_write_authority": False,
    }
    if receipt.get("grant") != expected_grant:
        raise ManualCompletionSealError(
            "manual completion seal grant is not activation-only"
        )
    return receipt


__all__ = [
    "DELEGATED_SUPERVISOR_OPERATOR",
    "INTERACTIVE_OPERATOR",
    "ManualCompletionSealError",
    "build_manual_completion_seal",
    "load_strict_manual_completion_seal",
    "verify_manual_completion_seal",
    "write_manual_completion_seal",
]
