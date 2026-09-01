"""Typed, canonical PatchPlan contracts for merge admission.

A plan is deliberately a closed description of the change *before* a patch is
considered for merge.  It is not an approval: ``patch_admission`` binds this
description to the actual patch, current tree, and validation receipt.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

PATCH_PLAN_SCHEMA = "ipfs_accelerate_py/agent-supervisor/merge/patch-plan@1"
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class PatchPlanError(ValueError):
    """A PatchPlan was malformed or its cryptographic binding was broken."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one JSON representation used for plan and patch digests."""

    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PatchPlanError("value is not canonical-JSON serializable") from exc


def sha256_digest(value: bytes | str) -> str:
    """Return a typed sha256 digest without accepting ambiguous encodings."""

    if isinstance(value, str):
        value = value.encode("utf-8")
    if not isinstance(value, bytes):
        raise TypeError("digest input must be bytes or str")
    return "sha256:" + hashlib.sha256(value).hexdigest()


def patch_digest(patch: bytes | str) -> str:
    """Digest exact patch bytes; normalization would make the binding unsafe."""

    return sha256_digest(patch)


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PatchPlanError(f"{name} must be a non-empty string")
    if "\x00" in value:
        raise PatchPlanError(f"{name} must not contain NUL")
    return value


def normalize_repository_path(value: object, *, field: str = "path") -> str:
    """Accept only a normalized relative POSIX repository path."""

    path = _required_text(value, field)
    if "\\" in path or any(ord(character) < 32 for character in path):
        raise PatchPlanError(f"{field} must use POSIX separators")
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or path in {".", ".."} or ".." in candidate.parts:
        raise PatchPlanError(f"{field} must be a relative repository path")
    normalized = str(candidate)
    if normalized != path or path.startswith("./"):
        raise PatchPlanError(f"{field} must be normalized")
    return path


def _text_tuple(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PatchPlanError(f"{name} must be a non-empty array of strings")
    result = tuple(_required_text(item, f"{name} item") for item in value)
    if not result:
        raise PatchPlanError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise PatchPlanError(f"{name} must not contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class FileSymbol:
    """One planned repository file and the symbols the change is expected to affect."""

    path: str
    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", normalize_repository_path(self.path))
        if isinstance(self.symbols, (str, bytes)) or not isinstance(self.symbols, tuple):
            raise PatchPlanError("symbols must be a tuple of non-empty strings")
        symbols = tuple(_required_text(item, "symbol") for item in self.symbols)
        if not symbols or len(set(symbols)) != len(symbols):
            raise PatchPlanError("symbols must be non-empty and unique")
        object.__setattr__(self, "symbols", symbols)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FileSymbol":
        if not isinstance(value, Mapping) or set(value) != {"path", "symbols"}:
            raise PatchPlanError("files_and_symbols entries must contain path and symbols")
        symbols = _text_tuple(value["symbols"], "symbols")
        return cls(path=normalize_repository_path(value["path"]), symbols=symbols)

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "symbols": list(self.symbols)}


@dataclass(frozen=True, slots=True)
class PatchPlan:
    """Closed semantic and scope declaration for a patch proposal.

    ``patch_digest`` is the SHA-256 of the exact proposed patch bytes.  The
    independent ``plan_digest`` property identifies the complete declaration.
    """

    base_tree: str
    intended_semantic_change: str
    files_and_symbols: tuple[FileSymbol, ...]
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]
    expected_invariants: tuple[str, ...]
    required_tests_and_proofs: tuple[str, ...]
    scope_limit: tuple[str, ...]
    patch_digest: str
    schema: str = PATCH_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PATCH_PLAN_SCHEMA:
            raise PatchPlanError("unsupported PatchPlan schema")
        object.__setattr__(self, "base_tree", _required_text(self.base_tree, "base_tree"))
        object.__setattr__(
            self, "intended_semantic_change",
            _required_text(self.intended_semantic_change, "intended_semantic_change"),
        )
        if not isinstance(self.files_and_symbols, tuple) or not self.files_and_symbols:
            raise PatchPlanError("files_and_symbols must be a non-empty tuple")
        files = tuple(self.files_and_symbols)
        if any(not isinstance(item, FileSymbol) for item in files):
            raise PatchPlanError("files_and_symbols must contain FileSymbol entries")
        paths = tuple(item.path for item in files)
        if len(set(paths)) != len(paths):
            raise PatchPlanError("files_and_symbols must not repeat a path")
        object.__setattr__(self, "files_and_symbols", files)
        for name in (
            "preconditions", "postconditions", "expected_invariants",
            "required_tests_and_proofs",
        ):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                raise PatchPlanError(f"{name} must be a tuple")
            object.__setattr__(self, name, _text_tuple(value, name))
        if not isinstance(self.scope_limit, tuple) or not self.scope_limit:
            raise PatchPlanError("scope_limit must be a non-empty tuple")
        scope = tuple(normalize_repository_path(item, field="scope_limit item") for item in self.scope_limit)
        if len(set(scope)) != len(scope):
            raise PatchPlanError("scope_limit must not contain duplicates")
        if any(
            not any(path == limit or path.startswith(limit + "/") for limit in scope)
            for path in paths
        ):
            raise PatchPlanError("scope_limit must cover every files_and_symbols path")
        object.__setattr__(self, "scope_limit", scope)
        if not isinstance(self.patch_digest, str) or not _SHA256_RE.fullmatch(self.patch_digest):
            raise PatchPlanError("patch_digest must be sha256:<64 lowercase hex characters>")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PatchPlan":
        if not isinstance(value, Mapping):
            raise PatchPlanError("PatchPlan must be an object")
        required = {
            "schema", "base_tree", "intended_semantic_change", "files_and_symbols",
            "preconditions", "postconditions", "expected_invariants",
            "required_tests_and_proofs", "scope_limit", "patch_digest",
        }
        if set(value) != required:
            raise PatchPlanError("PatchPlan has missing or unexpected fields")
        raw_files = value["files_and_symbols"]
        if isinstance(raw_files, (str, bytes)) or not isinstance(raw_files, Sequence):
            raise PatchPlanError("files_and_symbols must be an array")
        raw_scope = value["scope_limit"]
        if isinstance(raw_scope, (str, bytes)) or not isinstance(raw_scope, Sequence):
            raise PatchPlanError("scope_limit must be an array")
        return cls(
            schema=value["schema"], base_tree=value["base_tree"],
            intended_semantic_change=value["intended_semantic_change"],
            files_and_symbols=tuple(FileSymbol.from_dict(item) for item in raw_files),
            preconditions=_text_tuple(value["preconditions"], "preconditions"),
            postconditions=_text_tuple(value["postconditions"], "postconditions"),
            expected_invariants=_text_tuple(value["expected_invariants"], "expected_invariants"),
            required_tests_and_proofs=_text_tuple(value["required_tests_and_proofs"], "required_tests_and_proofs"),
            scope_limit=tuple(normalize_repository_path(item, field="scope_limit item") for item in raw_scope),
            patch_digest=value["patch_digest"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "base_tree": self.base_tree,
            "intended_semantic_change": self.intended_semantic_change,
            "files_and_symbols": [item.to_dict() for item in self.files_and_symbols],
            "preconditions": list(self.preconditions), "postconditions": list(self.postconditions),
            "expected_invariants": list(self.expected_invariants),
            "required_tests_and_proofs": list(self.required_tests_and_proofs),
            "scope_limit": list(self.scope_limit), "patch_digest": self.patch_digest,
        }

    @property
    def plan_digest(self) -> str:
        return sha256_digest(canonical_json_bytes(self.to_dict()))

    @property
    def planned_paths(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.files_and_symbols)

    @classmethod
    def for_patch(cls, *, patch: bytes | str, base_tree: str,
                  intended_semantic_change: str, files_and_symbols: Sequence[FileSymbol],
                  preconditions: Sequence[str], postconditions: Sequence[str],
                  expected_invariants: Sequence[str], required_tests_and_proofs: Sequence[str],
                  scope_limit: Sequence[str]) -> "PatchPlan":
        """Build a plan whose patch binding is calculated from exact bytes."""

        return cls(
            base_tree=base_tree, intended_semantic_change=intended_semantic_change,
            files_and_symbols=tuple(files_and_symbols), preconditions=tuple(preconditions),
            postconditions=tuple(postconditions), expected_invariants=tuple(expected_invariants),
            required_tests_and_proofs=tuple(required_tests_and_proofs),
            scope_limit=tuple(scope_limit), patch_digest=patch_digest(patch),
        )


__all__ = [
    "PATCH_PLAN_SCHEMA", "FileSymbol", "PatchPlan", "PatchPlanError",
    "canonical_json_bytes", "normalize_repository_path", "patch_digest", "sha256_digest",
]
