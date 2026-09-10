"""Typed, tamper-evident plans for one proposed repository patch.

``PatchPlan@1`` deliberately records the evidence a merge admission needs; it
is not an authorization to apply, validate, or merge the patch.  The plan's
content digest is reconstructed on every decode so a serialized plan cannot
silently change its declared base, intent, scope, or proof obligations.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import PurePosixPath
from typing import Any, Final

PATCH_PLAN_SCHEMA: Final[str] = "aseh/patch-plan@1"
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")


class PatchPlanValidationError(ValueError):
    """Raised when a plan is incomplete, malformed, or has been tampered with."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _digest(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def patch_digest(patch_text: str) -> str:
    """Return the exact UTF-8 patch payload identity used by admission."""

    if not isinstance(patch_text, str):
        raise PatchPlanValidationError("patch text must be a string")
    return "sha256:" + hashlib.sha256(patch_text.encode("utf-8")).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise PatchPlanValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _repository_path(value: Any, name: str) -> str:
    path = _text(value, name).replace("\\", "/")
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or path != pure.as_posix()
        or path in {".", ".."}
        or ".." in pure.parts
        or any(part in {"", ".", ".git"} for part in pure.parts)
    ):
        raise PatchPlanValidationError(f"{name} must be a safe repository-relative path")
    return path


def _text_items(value: Any, name: str, *, path_items: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PatchPlanValidationError(f"{name} must be a non-empty list")
    items = tuple(
        _repository_path(item, name) if path_items else _text(item, name) for item in value
    )
    if not items or len(set(items)) != len(items):
        raise PatchPlanValidationError(f"{name} must contain unique non-empty values")
    return items


def _tuple_field(value: Any, name: str) -> tuple[Any, ...]:
    """Decode an array field without allowing Python's string iteration."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PatchPlanValidationError(f"{name} must be a non-empty list")
    return tuple(value)


@dataclass(frozen=True)
class PatchScope:
    """The complete write allowlist for a plan.

    An entry ending in ``/**`` admits descendants.  Other entries are exact
    paths or shell-style patterns, always relative to the repository root.
    """

    allowed_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        normalized = _text_items(self.allowed_paths, "scope.allowed_paths", path_items=True)
        object.__setattr__(self, "allowed_paths", normalized)

    def admits(self, path: str) -> bool:
        candidate = _repository_path(path, "changed path")
        for pattern in self.allowed_paths:
            if pattern.endswith("/**") and candidate.startswith(pattern[:-2]):
                return True
            if fnmatchcase(candidate, pattern):
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {"allowed_paths": list(self.allowed_paths)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PatchScope":
        if not isinstance(value, Mapping) or set(value) != {"allowed_paths"}:
            raise PatchPlanValidationError("scope must contain only allowed_paths")
        return cls(allowed_paths=_tuple_field(value["allowed_paths"], "scope.allowed_paths"))


@dataclass(frozen=True)
class PatchPlan:
    """A closed declaration of one patch's semantic and validation contract."""

    plan_id: str
    base_tree: str
    semantic_intent: str
    files: tuple[str, ...]
    symbols: tuple[str, ...]
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]
    invariants: tuple[str, ...]
    tests: tuple[str, ...]
    proofs: tuple[str, ...]
    scope: PatchScope
    patch_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _text(self.plan_id, "plan_id"))
        object.__setattr__(self, "base_tree", _text(self.base_tree, "base_tree"))
        object.__setattr__(self, "semantic_intent", _text(self.semantic_intent, "semantic_intent"))
        object.__setattr__(self, "files", _text_items(self.files, "files", path_items=True))
        for field_name in (
            "symbols",
            "preconditions",
            "postconditions",
            "invariants",
            "tests",
            "proofs",
        ):
            object.__setattr__(self, field_name, _text_items(getattr(self, field_name), field_name))
        if not isinstance(self.scope, PatchScope):
            raise PatchPlanValidationError("scope must be a PatchScope")
        if not _SHA256_RE.fullmatch(self.patch_digest):
            raise PatchPlanValidationError("patch_digest must be a lowercase sha256 digest")
        escaped = [path for path in self.files if not self.scope.admits(path)]
        if escaped:
            raise PatchPlanValidationError("files escape the declared scope: " + ", ".join(escaped))

    @classmethod
    def create(cls, *, patch_text: str, **kwargs: Any) -> "PatchPlan":
        """Build a plan and bind it to ``patch_text`` without caller hashing."""

        return cls(patch_digest=patch_digest(patch_text), **kwargs)

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema": PATCH_PLAN_SCHEMA,
            "plan_id": self.plan_id,
            "base_tree": self.base_tree,
            "semantic_intent": self.semantic_intent,
            "files": list(self.files),
            "symbols": list(self.symbols),
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
            "invariants": list(self.invariants),
            "tests": list(self.tests),
            "proofs": list(self.proofs),
            "scope": self.scope.to_dict(),
            "patch_digest": self.patch_digest,
        }

    @property
    def digest(self) -> str:
        """Digest over every semantic plan field, excluding only itself."""

        return _digest(self._unsigned_dict())

    @property
    def plan_digest(self) -> str:
        """Explicit alias used by validation receipts."""

        return self.digest

    def to_dict(self) -> dict[str, Any]:
        result = self._unsigned_dict()
        result["plan_digest"] = self.digest
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PatchPlan":
        if not isinstance(value, Mapping):
            raise PatchPlanValidationError("patch plan must be an object")
        required = {
            "schema", "plan_id", "base_tree", "semantic_intent", "files", "symbols",
            "preconditions", "postconditions", "invariants", "tests", "proofs", "scope",
            "patch_digest", "plan_digest",
        }
        if set(value) != required:
            raise PatchPlanValidationError("patch plan fields are incomplete or unsupported")
        if value["schema"] != PATCH_PLAN_SCHEMA:
            raise PatchPlanValidationError("unsupported patch plan schema")
        plan = cls(
            plan_id=value["plan_id"], base_tree=value["base_tree"],
            semantic_intent=value["semantic_intent"], files=_tuple_field(value["files"], "files"),
            symbols=_tuple_field(value["symbols"], "symbols"),
            preconditions=_tuple_field(value["preconditions"], "preconditions"),
            postconditions=_tuple_field(value["postconditions"], "postconditions"),
            invariants=_tuple_field(value["invariants"], "invariants"),
            tests=_tuple_field(value["tests"], "tests"),
            proofs=_tuple_field(value["proofs"], "proofs"),
            scope=PatchScope.from_dict(value["scope"]), patch_digest=value["patch_digest"],
        )
        if value["plan_digest"] != plan.digest:
            raise PatchPlanValidationError("patch plan digest does not match its contents")
        return plan
