"""Fail-closed admission of a typed :class:`PatchPlan` before merge."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .patch_plan import PatchPlan, PatchPlanError, normalize_repository_path, patch_digest

_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----", re.I),
    re.compile(r"\b(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[^\s'\"]{8,}", re.I),
    re.compile(r"\b(?:ghp|github_pat|sk|AKIA)[A-Za-z0-9_-]{12,}\b"),
)
_GENERATED_PATH_PARTS = frozenset({"__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build"})
_GENERATED_SUFFIXES = (".pyc", ".pyo", ".egg-info")
_GENERATED_CONTENT = re.compile(r"(?:^|\n)\s*(?:#|//|/\*)\s*(?:auto-)?generated\b", re.I)


class PatchAdmissionDisposition(str, Enum):
    ADMITTED = "admitted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ValidationReceipt:
    """The current-tree validation assertion required to permit a merge."""

    receipt_id: str
    tree_id: str
    patch_digest: str
    passed: bool
    plan_digest: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ValidationReceipt":
        if not isinstance(value, Mapping):
            raise ValueError("validation receipt must be an object")
        allowed = {"receipt_id", "tree_id", "patch_digest", "passed", "plan_digest"}
        if not {"receipt_id", "tree_id", "patch_digest", "passed"} <= set(value) or set(value) - allowed:
            raise ValueError("validation receipt has missing or unexpected fields")
        return cls(
            receipt_id=value["receipt_id"], tree_id=value["tree_id"],
            patch_digest=value["patch_digest"], passed=value["passed"],
            plan_digest=value.get("plan_digest"),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.receipt_id, str) or not self.receipt_id.strip():
            raise ValueError("validation receipt_id must be non-empty")
        if not isinstance(self.tree_id, str) or not self.tree_id.strip():
            raise ValueError("validation tree_id must be non-empty")
        if not isinstance(self.patch_digest, str) or not self.patch_digest.startswith("sha256:"):
            raise ValueError("validation patch_digest must be typed sha256")
        if type(self.passed) is not bool:
            raise ValueError("validation passed must be boolean")
        if self.plan_digest is not None and (not isinstance(self.plan_digest, str) or not self.plan_digest.startswith("sha256:")):
            raise ValueError("validation plan_digest must be typed sha256")


@dataclass(frozen=True, slots=True)
class PatchAdmissionReceipt:
    disposition: PatchAdmissionDisposition
    plan_digest: str | None
    patch_digest: str | None
    current_tree: str
    validation_receipt_id: str | None
    reason_codes: tuple[str, ...]

    @property
    def admitted(self) -> bool:
        return self.disposition is PatchAdmissionDisposition.ADMITTED


def _changed_paths_from_patch(patch: str) -> tuple[str, ...]:
    new_paths: list[str] = []
    old_paths: list[str] = []
    for line in patch.splitlines():
        if not line.startswith(("+++ ", "--- ")):
            continue
        candidate = line[4:].split("\t", 1)[0]
        if candidate == "/dev/null":
            continue
        if candidate.startswith(("a/", "b/")):
            candidate = candidate[2:]
        normalized = normalize_repository_path(candidate, field="patch path")
        (new_paths if line.startswith("+++ ") else old_paths).append(normalized)
    # A deletion has no ``+++`` side.  For rename/modify patches the new side
    # is the authoritative planned path, as it is the path that would exist
    # after merge.
    return tuple(dict.fromkeys(new_paths or old_paths))


def _has_semantic_change(patch: str) -> bool:
    return any(
        line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        for line in patch.splitlines()
    )


def _in_scope(path: str, scope: Sequence[str]) -> bool:
    return any(path == item or path.startswith(item + "/") for item in scope)


def _secret_or_generated_reasons(patch: str, paths: Sequence[str]) -> set[str]:
    reasons: set[str] = set()
    for path in paths:
        components = path.split("/")
        if (any(part in _GENERATED_PATH_PARTS for part in components)
                or path.endswith(_GENERATED_SUFFIXES)
                or any(part.endswith(".egg-info") for part in components)):
            reasons.add("generated_artifact_forbidden")
    if _GENERATED_CONTENT.search(patch):
        reasons.add("generated_artifact_forbidden")
    if any(pattern.search(patch) for pattern in _SECRET_PATTERNS):
        reasons.add("secret_change_forbidden")
    return reasons


def admit_patch_plan(
    plan: PatchPlan | Mapping[str, Any],
    *,
    patch: bytes | str,
    current_tree: str,
    validation_receipt: ValidationReceipt | Mapping[str, Any] | None,
    changed_paths: Sequence[str] | None = None,
    conflict_paths: Sequence[str] = (),
) -> PatchAdmissionReceipt:
    """Admit only a complete, current, validated patch matching its plan.

    ``changed_paths`` is optional only for a standard unified diff.  Callers
    with a binary or otherwise opaque patch must supply its exact changed paths.
    All reported reasons are stable codes and no untrusted patch content is
    returned in the receipt.
    """

    reasons: set[str] = set()
    parsed_plan: PatchPlan | None = None
    try:
        parsed_plan = plan if isinstance(plan, PatchPlan) else PatchPlan.from_dict(plan)
    except (PatchPlanError, TypeError, ValueError):
        reasons.add("invalid_patch_plan")
    if not isinstance(current_tree, str) or not current_tree.strip():
        reasons.add("invalid_current_tree")

    if isinstance(patch, bytes):
        try:
            patch_text = patch.decode("utf-8")
        except UnicodeDecodeError:
            patch_text = ""
            reasons.add("non_text_patch_requires_explicit_semantic_evidence")
    elif isinstance(patch, str):
        patch_text = patch
    else:
        patch_text = ""
        reasons.add("invalid_patch")
    actual_digest = patch_digest(patch) if isinstance(patch, (bytes, str)) else None

    try:
        paths = (
            tuple(dict.fromkeys(normalize_repository_path(item, field="changed_paths item") for item in changed_paths))
            if changed_paths is not None else _changed_paths_from_patch(patch_text)
        )
    except (PatchPlanError, TypeError):
        paths = ()
        reasons.add("invalid_changed_path")
    try:
        conflicts = tuple(normalize_repository_path(item, field="conflict_paths item") for item in conflict_paths)
    except (PatchPlanError, TypeError):
        conflicts = ("invalid",)
    if conflicts:
        reasons.add("merge_conflict_detected")
    if not paths or not _has_semantic_change(patch_text):
        reasons.add("semantically_empty_patch")
    reasons.update(_secret_or_generated_reasons(patch_text, paths))

    receipt: ValidationReceipt | None = None
    if validation_receipt is None:
        reasons.add("current_validation_receipt_missing")
    else:
        try:
            receipt = validation_receipt if isinstance(validation_receipt, ValidationReceipt) else ValidationReceipt.from_dict(validation_receipt)
        except (TypeError, ValueError):
            reasons.add("invalid_validation_receipt")

    if parsed_plan is not None:
        if parsed_plan.base_tree != current_tree:
            reasons.add("stale_base_tree")
        if actual_digest != parsed_plan.patch_digest:
            reasons.add("patch_digest_mismatch")
        if any(not _in_scope(path, parsed_plan.scope_limit) for path in paths):
            reasons.add("out_of_scope_path")
        if set(paths) != set(parsed_plan.planned_paths):
            reasons.add("files_and_symbols_mismatch")
        if receipt is not None:
            if not receipt.passed:
                reasons.add("validation_failed")
            if receipt.tree_id != current_tree:
                reasons.add("stale_validation_receipt")
            if receipt.patch_digest != actual_digest:
                reasons.add("validation_patch_digest_mismatch")
            if receipt.plan_digest is not None and receipt.plan_digest != parsed_plan.plan_digest:
                reasons.add("validation_plan_digest_mismatch")
    elif receipt is not None and not receipt.passed:
        reasons.add("validation_failed")

    ordered = tuple(sorted(reasons))
    return PatchAdmissionReceipt(
        disposition=(PatchAdmissionDisposition.ADMITTED if not ordered else PatchAdmissionDisposition.REJECTED),
        plan_digest=parsed_plan.plan_digest if parsed_plan is not None else None,
        patch_digest=actual_digest,
        current_tree=current_tree if isinstance(current_tree, str) else "",
        validation_receipt_id=receipt.receipt_id if receipt is not None else None,
        reason_codes=ordered,
    )


admit = admit_patch_plan

__all__ = [
    "PatchAdmissionDisposition", "PatchAdmissionReceipt", "ValidationReceipt",
    "admit", "admit_patch_plan",
]
