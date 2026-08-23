"""Isolated patch-sketch and test-sketch candidates.

Worktree, renderer, repair, merge, effect, authority, and validation systems
remain canonical.  This adapter only nominates bounded PatchSketchIR and
TestSketchIR values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final

from .contracts import (
    ExpertDisposition,
    ResidualIntelligenceError,
    canonical_id,
    reject_candidate_authority,
    required_text,
    text_tuple,
)
from .local_experts import IndependentValidationReceipt
from .procedure_experts import ProcedureHoleResolution

PATCH_SCOPE_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-patch-scope-policy@1"
)
PATCH_SKETCH_IR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-patch-sketch-ir@1"
)
TEST_SKETCH_IR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-test-sketch-ir@1"
)
PATCH_EXPERT_ADAPTER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-patch-expert-adapter@1"
)
FORBIDDEN_PATH_PARTS: Final[frozenset[str]] = frozenset(
    {".git", "secrets", "credentials", "id_rsa", "authorized_keys"}
)
FORBIDDEN_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        "delete_test",
        "weaken_validation",
        "shell",
        "chmod",
        "write_key",
        "authority_edit",
        "binary_replace",
    }
)
ALLOWED_OPERATIONS: Final[tuple[str, ...]] = (
    "replace_function",
    "replace_method",
    "insert_statement",
    "update_binding",
    "add_guard",
    "narrow_type",
    "repair_call",
    "add_test",
)
REASON_PATH_OUT_OF_SCOPE: Final = "path_out_of_scope"
REASON_MAX_LINES: Final = "changed_lines_exceed_bound"
REASON_BINARY_FORBIDDEN: Final = "binary_patch_forbidden"
REASON_TEST_DELETION: Final = "test_deletion_forbidden"
REASON_VALIDATION_WEAKENING: Final = "validation_weakening_forbidden"
REASON_ISOLATED_WORKTREE_REQUIRED: Final = "isolated_worktree_required"


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ResidualIntelligenceError(f"{name} must be boolean")
    return value


def _relative_paths(values: Any, name: str) -> tuple[str, ...]:
    items = text_tuple(values, name, allow_empty=False, max_items=64)
    cleaned: list[str] = []
    for item in items:
        parsed = PurePosixPath(item)
        if parsed.is_absolute() or ".." in parsed.parts:
            raise ResidualIntelligenceError(f"{name} must be a relative path")
        lowered = parsed.as_posix().lower()
        if any(part in FORBIDDEN_PATH_PARTS for part in parsed.parts) or lowered.endswith(
            (".pem", ".key", ".p12", ".bin", ".so", ".dll")
        ):
            raise ResidualIntelligenceError(REASON_PATH_OUT_OF_SCOPE)
        cleaned.append(parsed.as_posix())
    return tuple(cleaned)


@dataclass(frozen=True)
class PatchScopePolicy:
    allowed_paths: tuple[str, ...]
    maximum_changed_lines: int
    isolated_worktree_required: bool = True
    allow_test_deletion: bool = False
    allow_validation_weakening: bool = False
    schema: str = PATCH_SCOPE_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PATCH_SCOPE_POLICY_SCHEMA:
            raise ResidualIntelligenceError("unsupported patch scope policy schema")
        object.__setattr__(
            self, "allowed_paths", _relative_paths(self.allowed_paths, "allowed_paths")
        )
        if type(self.maximum_changed_lines) is not int or self.maximum_changed_lines < 1:
            raise ResidualIntelligenceError("maximum_changed_lines must be a positive integer")
        if self.maximum_changed_lines > 10_000:
            raise ResidualIntelligenceError(REASON_MAX_LINES)
        object.__setattr__(
            self,
            "isolated_worktree_required",
            _require_bool(self.isolated_worktree_required, "isolated_worktree_required"),
        )
        object.__setattr__(
            self, "allow_test_deletion", _require_bool(self.allow_test_deletion, "allow_test_deletion")
        )
        object.__setattr__(
            self,
            "allow_validation_weakening",
            _require_bool(self.allow_validation_weakening, "allow_validation_weakening"),
        )
        if self.allow_test_deletion or self.allow_validation_weakening:
            raise ResidualIntelligenceError("scope policy cannot authorize prohibited effects")

    def permits(self, path: str) -> bool:
        return path in self.allowed_paths


@dataclass(frozen=True)
class PatchSketchIR:
    base_tree_cid: str
    paths: tuple[str, ...]
    symbol_ids: tuple[str, ...]
    operations: tuple[str, ...]
    changed_lines: int
    candidate_only: bool = True
    schema: str = PATCH_SKETCH_IR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PATCH_SKETCH_IR_SCHEMA:
            raise ResidualIntelligenceError("unsupported patch sketch schema")
        object.__setattr__(self, "base_tree_cid", required_text(self.base_tree_cid, "base_tree_cid"))
        object.__setattr__(self, "paths", _relative_paths(self.paths, "paths"))
        object.__setattr__(
            self, "symbol_ids", text_tuple(self.symbol_ids, "symbol_ids", allow_empty=False, max_items=64)
        )
        operations = text_tuple(self.operations, "operations", allow_empty=False, max_items=16)
        forbidden = [item for item in operations if item in FORBIDDEN_OPERATIONS]
        if forbidden:
            raise ResidualIntelligenceError(f"prohibited patch operation: {forbidden[0]}")
        unknown = [item for item in operations if item not in ALLOWED_OPERATIONS]
        if unknown:
            raise ResidualIntelligenceError(f"unknown patch operation: {unknown[0]}")
        object.__setattr__(self, "operations", operations)
        if type(self.changed_lines) is not int or self.changed_lines < 1:
            raise ResidualIntelligenceError("changed_lines must be a positive integer")
        object.__setattr__(self, "changed_lines", self.changed_lines)
        if type(self.candidate_only) is not bool or self.candidate_only is not True:
            raise ResidualIntelligenceError("patch sketches must remain candidate_only")
        reject_candidate_authority(self.to_dict(include_id=False))

    @property
    def sketch_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "base_tree_cid": self.base_tree_cid,
            "paths": self.paths,
            "symbol_ids": self.symbol_ids,
            "operations": self.operations,
            "changed_lines": self.changed_lines,
            "candidate_only": True,
        }
        if include_id:
            payload["sketch_id"] = self.sketch_id
        return payload


@dataclass(frozen=True)
class TestSketchIR:
    __test__ = False
    validation_ids: tuple[str, ...]
    added_tests: tuple[str, ...] = ()
    deleted_tests: tuple[str, ...] = ()
    candidate_only: bool = True
    schema: str = TEST_SKETCH_IR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TEST_SKETCH_IR_SCHEMA:
            raise ResidualIntelligenceError("unsupported test sketch schema")
        object.__setattr__(
            self,
            "validation_ids",
            text_tuple(self.validation_ids, "validation_ids", allow_empty=False, max_items=32),
        )
        object.__setattr__(
            self, "added_tests", text_tuple(self.added_tests, "added_tests", max_items=32)
        )
        deleted = text_tuple(self.deleted_tests, "deleted_tests", max_items=32)
        if deleted:
            raise ResidualIntelligenceError(REASON_TEST_DELETION)
        object.__setattr__(self, "deleted_tests", ())
        if type(self.candidate_only) is not bool or self.candidate_only is not True:
            raise ResidualIntelligenceError("test sketches must remain candidate_only")
        reject_candidate_authority(self.to_dict(include_id=False))

    @property
    def sketch_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "validation_ids": self.validation_ids,
            "added_tests": self.added_tests,
            "deleted_tests": (),
            "candidate_only": True,
        }
        if include_id:
            payload["sketch_id"] = self.sketch_id
        return payload


@dataclass(frozen=True)
class PatchExpertAdapter:
    policy: PatchScopePolicy
    isolated_worktree: bool
    schema: str = PATCH_EXPERT_ADAPTER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PATCH_EXPERT_ADAPTER_SCHEMA:
            raise ResidualIntelligenceError("unsupported patch expert adapter schema")
        if not isinstance(self.policy, PatchScopePolicy):
            raise ResidualIntelligenceError("adapter requires PatchScopePolicy")
        object.__setattr__(
            self, "isolated_worktree", _require_bool(self.isolated_worktree, "isolated_worktree")
        )

    def nominate(
        self,
        sketch: PatchSketchIR,
        tests: TestSketchIR,
        *,
        hole: ProcedureHoleResolution | None = None,
        validation: IndependentValidationReceipt | None = None,
    ) -> dict[str, Any]:
        reasons: list[str] = []
        if self.policy.isolated_worktree_required and not self.isolated_worktree:
            raise ResidualIntelligenceError(REASON_ISOLATED_WORKTREE_REQUIRED)
        if any(not self.policy.permits(path) for path in sketch.paths):
            raise ResidualIntelligenceError(REASON_PATH_OUT_OF_SCOPE)
        if sketch.changed_lines > self.policy.maximum_changed_lines:
            raise ResidualIntelligenceError(REASON_MAX_LINES)
        if "weaken_validation" in sketch.operations:
            raise ResidualIntelligenceError(REASON_VALIDATION_WEAKENING)
        disposition = ExpertDisposition.VALIDATION_REQUIRED
        if validation is None:
            reasons.append("independent_validator_decides")
        elif validation.accepted is not True:
            disposition = ExpertDisposition.REJECT_INPUT
            reasons.append("independent_validator_decides")
        else:
            disposition = ExpertDisposition.ACCEPT
            reasons.append("independent_validator_decides")
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/residual-patch-nomination@1",
            "patch": sketch.to_dict(),
            "tests": tests.to_dict(),
            "hole_resolution": None if hole is None else hole.to_dict(),
            "disposition": disposition.value,
            "reason_codes": tuple(reasons),
            "rollback": {
                "mode": "discard_fenced_worktree",
                "base_tree_cid": sketch.base_tree_cid,
            },
            "candidate_only": True,
        }
