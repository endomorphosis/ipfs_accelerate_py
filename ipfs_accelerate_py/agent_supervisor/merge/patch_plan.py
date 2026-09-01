"""Canonical, exact-tree patch planning contracts.

``PatchPlan`` deliberately contains no authority to merge.  It is a stable,
digest-bound description of the one patch which a separate admission gate may
consider.  Parsing is strict so a lossy JSON round trip cannot turn missing
evidence into an accepted plan.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final


PATCH_PLAN_SCHEMA: Final = "aseh/patch-plan@1"
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class PatchPlanValidationError(ValueError):
    """Raised when a PatchPlan is incomplete, ambiguous, or tampered with."""


def sha256_digest(value: bytes | str) -> str:
    """Return the tagged SHA-256 identity used by PatchPlan evidence."""
    if isinstance(value, str):
        value = value.encode("utf-8")
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _require_text(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise PatchPlanValidationError(f"{field_name} is required")
    return text


def normalize_path(value: object) -> str:
    """Accept only portable relative repository paths.

    Paths are identity material; accepting ``a/../b`` would make scope and
    digest checks disagree between callers, so it is rejected rather than
    normalized.
    """
    path = _require_text(value, "path").replace("\\", "/")
    if path.startswith("/") or path == "." or "//" in path:
        raise PatchPlanValidationError(f"invalid repository path: {path!r}")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise PatchPlanValidationError(f"invalid repository path: {path!r}")
    return path


def _canonical_texts(values: Iterable[object], field_name: str) -> tuple[str, ...]:
    result = tuple(sorted({_require_text(value, field_name) for value in values}))
    if not result:
        raise PatchPlanValidationError(f"{field_name} must be nonempty")
    return result


def _canonical_scope(values: Iterable[object]) -> tuple[str, ...]:
    scope: set[str] = set()
    for value in values:
        raw = _require_text(value, "scope")
        is_tree = raw.endswith("/")
        normalized = normalize_path(raw[:-1] if is_tree else raw)
        scope.add(normalized + "/" if is_tree else normalized)
    if not scope:
        raise PatchPlanValidationError("scope must be nonempty")
    return tuple(sorted(scope))


def path_within_scope(path: str, scope: Sequence[str]) -> bool:
    """Whether *path* is covered by an exact path or a declared directory."""
    normalized = normalize_path(path)
    return any(
        normalized == allowed or (allowed.endswith("/") and normalized.startswith(allowed))
        for allowed in scope
    )


@dataclass(frozen=True, slots=True)
class PatchFile:
    """A source file and the symbols whose behavior the patch changes."""

    path: str
    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", normalize_path(self.path))
        object.__setattr__(self, "symbols", _canonical_texts(self.symbols, "symbols"))

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "symbols": list(self.symbols)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PatchFile":
        if set(value) - {"path", "symbols"}:
            raise PatchPlanValidationError("PatchFile has unknown fields")
        symbols = value.get("symbols")
        if not isinstance(symbols, Sequence) or isinstance(symbols, (str, bytes)):
            raise PatchPlanValidationError("PatchFile.symbols must be an array")
        return cls(path=value.get("path", ""), symbols=tuple(symbols))


@dataclass(frozen=True, slots=True)
class PatchPlan:
    """A closed, canonical proposal for exactly one source patch."""

    base_tree_id: str
    semantic_intent: str
    files: tuple[PatchFile, ...]
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]
    invariants: tuple[str, ...]
    tests_proofs: tuple[str, ...]
    scope: tuple[str, ...]
    patch_digest: str
    proposer_id: str
    plan_digest: str = field(default="")

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_tree_id", _require_text(self.base_tree_id, "base_tree_id"))
        object.__setattr__(self, "semantic_intent", _require_text(self.semantic_intent, "semantic_intent"))
        object.__setattr__(self, "proposer_id", _require_text(self.proposer_id, "proposer_id"))
        if not SHA256_RE.fullmatch(self.patch_digest):
            raise PatchPlanValidationError("patch_digest must be a sha256 digest")
        files = tuple(sorted((item if isinstance(item, PatchFile) else PatchFile.from_dict(item) for item in self.files), key=lambda item: item.path))
        if not files:
            raise PatchPlanValidationError("files must be nonempty")
        if len({item.path for item in files}) != len(files):
            raise PatchPlanValidationError("files must not repeat paths")
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "preconditions", _canonical_texts(self.preconditions, "preconditions"))
        object.__setattr__(self, "postconditions", _canonical_texts(self.postconditions, "postconditions"))
        object.__setattr__(self, "invariants", _canonical_texts(self.invariants, "invariants"))
        object.__setattr__(self, "tests_proofs", _canonical_texts(self.tests_proofs, "tests_proofs"))
        scope = _canonical_scope(self.scope)
        if not all(path_within_scope(item.path, scope) for item in files):
            raise PatchPlanValidationError("declared file lies outside PatchPlan scope")
        object.__setattr__(self, "scope", scope)
        calculated = self.calculated_digest()
        if self.plan_digest and self.plan_digest != calculated:
            raise PatchPlanValidationError("PatchPlan digest mismatch")
        object.__setattr__(self, "plan_digest", calculated)

    @classmethod
    def create(
        cls,
        *,
        base_tree_id: str,
        semantic_intent: str,
        files: Iterable[PatchFile | Mapping[str, Any]],
        preconditions: Iterable[str],
        postconditions: Iterable[str],
        invariants: Iterable[str],
        tests_proofs: Iterable[str],
        scope: Iterable[str],
        patch: bytes | str,
        proposer_id: str,
    ) -> "PatchPlan":
        """Build a plan whose patch digest is derived from the supplied bytes."""
        return cls(
            base_tree_id=base_tree_id,
            semantic_intent=semantic_intent,
            files=tuple(files),
            preconditions=tuple(preconditions),
            postconditions=tuple(postconditions),
            invariants=tuple(invariants),
            tests_proofs=tuple(tests_proofs),
            scope=tuple(scope),
            patch_digest=sha256_digest(patch),
            proposer_id=proposer_id,
        )

    def _body(self) -> dict[str, object]:
        return {
            "schema": PATCH_PLAN_SCHEMA,
            "base_tree_id": self.base_tree_id,
            "semantic_intent": self.semantic_intent,
            "files": [item.to_dict() for item in self.files],
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
            "invariants": list(self.invariants),
            "tests_proofs": list(self.tests_proofs),
            "scope": list(self.scope),
            "patch_digest": self.patch_digest,
            "proposer_id": self.proposer_id,
        }

    def calculated_digest(self) -> str:
        return sha256_digest(json.dumps(self._body(), sort_keys=True, separators=(",", ":")))

    def to_dict(self) -> dict[str, object]:
        return {**self._body(), "plan_digest": self.plan_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PatchPlan":
        expected = {
            "schema", "base_tree_id", "semantic_intent", "files", "preconditions",
            "postconditions", "invariants", "tests_proofs", "scope", "patch_digest",
            "proposer_id", "plan_digest",
        }
        unknown = set(value) - expected
        if unknown:
            raise PatchPlanValidationError(f"PatchPlan has unknown fields: {sorted(unknown)}")
        if value.get("schema") != PATCH_PLAN_SCHEMA:
            raise PatchPlanValidationError("unsupported PatchPlan schema")
        def array(name: str) -> tuple[Any, ...]:
            raw = value.get(name)
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise PatchPlanValidationError(f"{name} must be an array")
            return tuple(raw)
        file_values = array("files")
        if not all(isinstance(item, Mapping) for item in file_values):
            raise PatchPlanValidationError("files must contain PatchFile objects")
        return cls(
            base_tree_id=value.get("base_tree_id", ""),
            semantic_intent=value.get("semantic_intent", ""),
            files=tuple(PatchFile.from_dict(item) for item in file_values),
            preconditions=array("preconditions"), postconditions=array("postconditions"),
            invariants=array("invariants"), tests_proofs=array("tests_proofs"), scope=array("scope"),
            patch_digest=value.get("patch_digest", ""), proposer_id=value.get("proposer_id", ""),
            plan_digest=value.get("plan_digest", ""),
        )
