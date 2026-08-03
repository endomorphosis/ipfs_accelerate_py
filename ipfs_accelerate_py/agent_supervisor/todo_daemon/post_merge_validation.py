"""Content-bound evidence for validation of an exact merged repository tree.

The validation runner and merge train are deliberately outside this module.
They own process execution and Git mutation respectively.  This module owns
only the small, deterministic receipt exchanged with authoritative task
completion, so an arbitrary non-empty ``validation_receipt_id`` cannot be
mistaken for evidence that validation ran.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..proof.formal_verification_contracts import (
    ContractValidationError,
    content_identity,
)

POST_MERGE_VALIDATION_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/post-merge-validation-evidence@1"
)

_RECEIPT_IDENTITY_FIELDS = (
    "schema",
    "task_id",
    "target_commit",
    "validated_commit",
    "repository_tree_id",
    "validation_scope",
    "attempted",
    "passed",
    "returncode",
    "stale",
    "validation_result_cid",
)
_RESERVED_RESULT_FIELDS = frozenset(
    {
        *_RECEIPT_IDENTITY_FIELDS,
        "validation_receipt_id",
        "validation_result",
    }
)
_MAX_PROJECTION_DEPTH = 8
_MAX_PROJECTION_ITEMS = 256
_MAX_TEXT_LENGTH = 4096


def _canonical_projection(
    value: Any,
    *,
    depth: int = 0,
) -> Any:
    """Return a bounded DAG-JSON-compatible diagnostic projection."""

    if depth >= _MAX_PROJECTION_DEPTH:
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:_MAX_TEXT_LENGTH]
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return _canonical_projection(enum_value, depth=depth + 1)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ContractValidationError(
                "post-merge validation result keys must be strings"
            )
        projected: dict[str, Any] = {}
        for key in sorted(value)[:_MAX_PROJECTION_ITEMS]:
            bounded_key = key[:_MAX_TEXT_LENGTH]
            if bounded_key in projected:
                raise ContractValidationError(
                    "post-merge validation result keys collide after bounding"
                )
            projected[bounded_key] = _canonical_projection(
                value[key], depth=depth + 1
            )
        return projected
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            _canonical_projection(item, depth=depth + 1)
            for item in value[:_MAX_PROJECTION_ITEMS]
        ]
    raise ContractValidationError(
        "unsupported post-merge validation result value: "
        f"{type(value).__name__}"
    )


def _returncode(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _receipt_identity_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {field: value.get(field) for field in _RECEIPT_IDENTITY_FIELDS}


def build_post_merge_validation_evidence(
    *,
    task_id: str,
    target_commit: str,
    repository_tree_id: str,
    validation_result: Mapping[str, Any],
    validated_commit: str = "",
) -> dict[str, Any]:
    """Build one CIDv1 receipt for validation run against an exact merge.

    ``validation_result`` is retained as a bounded diagnostic projection and
    separately content-addressed.  Authority-bearing binding fields are
    daemon supplied and cannot be overridden by similarly named result keys.
    A successful verdict requires an actually attempted, zero-return-code,
    non-stale run against the target commit.
    """

    if not isinstance(validation_result, Mapping):
        raise TypeError("validation_result must be a mapping")
    if not all(isinstance(key, str) for key in validation_result):
        raise ContractValidationError(
            "post-merge validation result keys must be strings"
        )
    task = str(task_id or "").strip()
    target = str(target_commit or "").strip()
    validated = str(validated_commit or target).strip()
    tree = str(repository_tree_id or "").strip()
    if not task:
        raise ValueError("task_id is required")
    if not target:
        raise ValueError("target_commit is required")
    if not tree:
        raise ValueError("repository_tree_id is required")

    raw_result = {
        key: value
        for key, value in validation_result.items()
        if key not in _RESERVED_RESULT_FIELDS
    }
    projected = _canonical_projection(raw_result)
    if not isinstance(projected, dict):  # defensive; mappings project to dicts
        raise ContractValidationError(
            "post-merge validation result projection must be a mapping"
        )
    attempted = validation_result.get("attempted") is True
    returncode = _returncode(validation_result.get("returncode"))
    stale = bool(
        validation_result.get("stale")
        or validation_result.get("validation_stale")
        or validation_result.get("freshness_authoritative") is False
    )
    passed = bool(
        validation_result.get("passed") is True
        and attempted
        and returncode == 0
        and not stale
        and validated == target
    )
    result_cid = content_identity(projected)
    evidence: dict[str, Any] = {
        "schema": POST_MERGE_VALIDATION_EVIDENCE_SCHEMA,
        "task_id": task,
        "target_commit": target,
        "validated_commit": validated,
        "repository_tree_id": tree,
        "validation_scope": "post_merge",
        "attempted": attempted,
        "passed": passed,
        "returncode": returncode,
        "stale": stale,
        "validation_result_cid": result_cid,
        "validation_result": projected,
    }
    evidence["validation_receipt_id"] = content_identity(
        _receipt_identity_payload(evidence)
    )
    return evidence


def verify_post_merge_validation_evidence(
    value: Mapping[str, Any] | None,
    *,
    expected_task_id: str = "",
    expected_target_commit: str = "",
    expected_repository_tree_id: str = "",
) -> tuple[bool, tuple[str, ...]]:
    """Verify receipt integrity and its optional caller-supplied bindings."""

    if not isinstance(value, Mapping):
        return False, ("post_merge_validation_evidence_missing",)

    reasons: list[str] = []
    if value.get("schema") != POST_MERGE_VALIDATION_EVIDENCE_SCHEMA:
        reasons.append("post_merge_validation_schema_invalid")
    if value.get("validation_scope") != "post_merge":
        reasons.append("post_merge_validation_scope_invalid")
    task_id = str(value.get("task_id") or "")
    target_commit = str(value.get("target_commit") or "")
    validated_commit = str(value.get("validated_commit") or "")
    repository_tree_id = str(value.get("repository_tree_id") or "")
    if not task_id:
        reasons.append("post_merge_validation_task_missing")
    if not target_commit:
        reasons.append("post_merge_validation_target_missing")
    if validated_commit != target_commit:
        reasons.append("post_merge_validation_target_mismatch")
    if not repository_tree_id:
        reasons.append("post_merge_validation_tree_missing")
    if expected_task_id and task_id != expected_task_id:
        reasons.append("post_merge_validation_task_mismatch")
    if expected_target_commit and target_commit != expected_target_commit:
        reasons.append("post_merge_validation_commit_binding_mismatch")
    if (
        expected_repository_tree_id
        and repository_tree_id != expected_repository_tree_id
    ):
        reasons.append("post_merge_validation_tree_binding_mismatch")

    for field in ("attempted", "passed", "stale"):
        if not isinstance(value.get(field), bool):
            reasons.append(f"post_merge_validation_{field}_invalid")
    if value.get("passed") is True and value.get("attempted") is not True:
        reasons.append("post_merge_validation_passed_without_attempt")
    returncode = value.get("returncode")
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        reasons.append("post_merge_validation_returncode_invalid")
    elif value.get("passed") is True and returncode != 0:
        reasons.append("post_merge_validation_passed_with_failure_returncode")
    if value.get("passed") is True and value.get("stale") is not False:
        reasons.append("post_merge_validation_passed_while_stale")

    raw_result = value.get("validation_result")
    if not isinstance(raw_result, Mapping):
        reasons.append("post_merge_validation_result_missing")
    else:
        try:
            projected_result = _canonical_projection(raw_result)
            expected_result_cid = content_identity(projected_result)
        except (ContractValidationError, TypeError, ValueError):
            reasons.append("post_merge_validation_result_invalid")
        else:
            if value.get("validation_result_cid") != expected_result_cid:
                reasons.append("post_merge_validation_result_cid_mismatch")

    try:
        expected_receipt_id = content_identity(
            _receipt_identity_payload(value)
        )
    except (ContractValidationError, TypeError, ValueError):
        reasons.append("post_merge_validation_receipt_payload_invalid")
    else:
        if value.get("validation_receipt_id") != expected_receipt_id:
            reasons.append("post_merge_validation_receipt_id_mismatch")
    return not reasons, tuple(dict.fromkeys(reasons))


__all__ = [
    "POST_MERGE_VALIDATION_EVIDENCE_SCHEMA",
    "build_post_merge_validation_evidence",
    "verify_post_merge_validation_evidence",
]
