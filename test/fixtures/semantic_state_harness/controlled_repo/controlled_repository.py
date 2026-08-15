"""ControlledSemanticRepository@1 — deterministic fixture tree + mutation matrix.

Provides base and mutated file maps, optional git tree materialization, and
pack-constraint checks. Never imports or executes target-tree modules.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .mutation_case import FixtureCorpusError, MutationCase, PathOperation
from .recipes import (
    REQUIRED_CATEGORIES,
    SOURCE_RACE_MARKER,
    SOURCE_RACE_PATH,
    base_tree_files,
    mutation_cases,
)

CONTROLLED_REPO_INTERFACE = "ControlledSemanticRepository@1"
CONTROLLED_REPO_SCHEMA = (
    "ipfs_accelerate_py/semantic-state/controlled-semantic-repository@1"
)
CORPUS_ID = "semantic-state-controlled-repo-v1"

# Maximum number of path operations for a case declared change_is_bounded.
BOUNDED_CHANGE_MAX_OPS = 3
# Maximum total changed bytes (sum of replaced/added content sizes) when bounded.
BOUNDED_CHANGE_MAX_BYTES = 4096

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "sch-fixture",
    "GIT_AUTHOR_EMAIL": "sch-fixture@example.invalid",
    "GIT_COMMITTER_NAME": "sch-fixture",
    "GIT_COMMITTER_EMAIL": "sch-fixture@example.invalid",
    "GIT_AUTHOR_DATE": "2000-01-01T00:00:00 +0000",
    "GIT_COMMITTER_DATE": "2000-01-01T00:00:00 +0000",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "EMAIL": "sch-fixture@example.invalid",
    "LANG": "C",
    "LC_ALL": "C",
}


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def content_digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def tree_digest(files: Mapping[str, str]) -> str:
    """Deterministic digest over a path->text mapping (not a Git object id)."""

    ordered = {path: files[path] for path in sorted(files)}
    payload = {
        "schema": "sch-fixture/tree-digest@1",
        "files": {
            path: {"sha256": content_digest(body)[7:], "size": len(body.encode())}
            for path, body in ordered.items()
        },
    }
    return "sha256:" + hashlib.sha256(_canonical_json(payload)).hexdigest()


def apply_operations(
    base: Mapping[str, str], operations: tuple[PathOperation, ...]
) -> dict[str, str]:
    """Apply path operations to a base tree; fail closed on conflicts."""

    result = dict(base)
    for operation in operations:
        if operation.op == "replace":
            if operation.path not in result:
                raise FixtureCorpusError(
                    f"replace target missing in base tree: {operation.path}"
                )
            assert operation.content is not None
            result[operation.path] = operation.content
        elif operation.op == "add":
            if operation.path in result:
                raise FixtureCorpusError(
                    f"add path already present in tree: {operation.path}"
                )
            assert operation.content is not None
            result[operation.path] = operation.content
        elif operation.op == "delete":
            if operation.path not in result:
                raise FixtureCorpusError(
                    f"delete path missing in tree: {operation.path}"
                )
            del result[operation.path]
        elif operation.op == "rename":
            assert operation.from_path is not None
            if operation.from_path not in result:
                raise FixtureCorpusError(
                    f"rename source missing in tree: {operation.from_path}"
                )
            if operation.path in result:
                raise FixtureCorpusError(
                    f"rename destination already present: {operation.path}"
                )
            result[operation.path] = result.pop(operation.from_path)
        else:  # pragma: no cover - validated at construction
            raise FixtureCorpusError(f"unsupported op {operation.op}")
    return result


def changed_paths(
    base: Mapping[str, str], mutated: Mapping[str, str]
) -> tuple[str, ...]:
    paths = set(base) | set(mutated)
    changed = [
        path
        for path in sorted(paths)
        if base.get(path) != mutated.get(path)
    ]
    return tuple(changed)


def write_tree(files: Mapping[str, str], destination: Path) -> Path:
    """Write a path->text mapping to disk. Does not import written modules."""

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for path, body in sorted(files.items()):
        target = destination / path
        if ".." in Path(path).parts or Path(path).is_absolute():
            raise FixtureCorpusError(f"refusing to write escaping path: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8", newline="\n")
    return destination


def materialize_git_tree(files: Mapping[str, str], destination: Path) -> dict[str, str]:
    """Materialize files and create a deterministic Git commit.

    Returns commit and tree object ids. Requires ``git`` on PATH. Never runs
    target code.
    """

    root = write_tree(files, destination)
    env = os.environ.copy()
    env.update(_GIT_ENV)
    # Isolate from user gitconfig.
    env["HOME"] = str(root / ".sch-fixture-home")
    Path(env["HOME"]).mkdir(parents=True, exist_ok=True)

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    git("init")
    git("config", "user.name", _GIT_ENV["GIT_AUTHOR_NAME"])
    git("config", "user.email", _GIT_ENV["GIT_AUTHOR_EMAIL"])
    git("add", "-A")
    # Allow empty only if needed; base tree is non-empty.
    git("commit", "-m", "sch-fixture snapshot")
    commit = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    return {"commit": commit, "tree": tree, "root": str(root)}


def read_tree_bytes(root: Path) -> dict[str, bytes]:
    """Read all regular files under root as bytes (scan-safe; no import)."""

    root = Path(root)
    result: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".git/") or rel.startswith(".sch-fixture-home/"):
            continue
        result[rel] = path.read_bytes()
    return result


def pack_candidate_paths(
    case: MutationCase, *, include_source_race_paths: bool = False
) -> tuple[str, ...]:
    """Declare paths eligible for a context pack for this mutation.

    Post-scan source-race paths are excluded unless explicitly requested (which
    tests use only to prove exclusion).
    """

    paths = set(case.oracle.merkle.affected_path_ids)
    if not include_source_race_paths:
        paths.difference_update(case.pack_excluded_paths)
        if case.category == "post_scan_source_race":
            paths.discard(SOURCE_RACE_PATH)
    return tuple(sorted(paths))


def pack_contains_source_race_bytes(
    pack_path_to_bytes: Mapping[str, bytes],
) -> bool:
    """Return True if forbidden source-race marker bytes appear in pack payloads."""

    for payload in pack_path_to_bytes.values():
        if SOURCE_RACE_MARKER in payload:
            return True
    return False


def bounded_change_stats(
    base: Mapping[str, str], case: MutationCase
) -> dict[str, int]:
    mutated = apply_operations(base, case.operations)
    paths = changed_paths(base, mutated)
    changed_bytes = 0
    for path in paths:
        before = base.get(path, "")
        after = mutated.get(path, "")
        # Approximate edit size as max of lengths when content differs.
        if before != after:
            changed_bytes += max(len(before.encode()), len(after.encode()))
    return {
        "changed_path_count": len(paths),
        "operation_count": len(case.operations),
        "changed_bytes": changed_bytes,
    }


@dataclass(frozen=True)
class ControlledSemanticRepository:
    """In-memory controlled fixture repository (ControlledSemanticRepository@1)."""

    corpus_id: str
    interface: str
    schema: str
    base_files: Mapping[str, str]
    mutations: tuple[MutationCase, ...]

    @classmethod
    def load(cls) -> "ControlledSemanticRepository":
        raw = base_tree_files()
        base = {path: raw[path] for path in sorted(raw)}
        mutations = mutation_cases()
        repo = cls(
            corpus_id=CORPUS_ID,
            interface=CONTROLLED_REPO_INTERFACE,
            schema=CONTROLLED_REPO_SCHEMA,
            base_files=base,
            mutations=mutations,
        )
        repo.validate()
        return repo

    def validate(self) -> None:
        if self.interface != CONTROLLED_REPO_INTERFACE:
            raise FixtureCorpusError("interface mismatch")
        if self.schema != CONTROLLED_REPO_SCHEMA:
            raise FixtureCorpusError("schema mismatch")
        if self.corpus_id != CORPUS_ID:
            raise FixtureCorpusError("corpus_id mismatch")
        if not self.base_files:
            raise FixtureCorpusError("base tree is empty")
        # Base paths must be sorted-unique relative paths.
        paths = list(self.base_files)
        if paths != sorted(paths):
            raise FixtureCorpusError("base tree keys must be sorted")
        if len(set(paths)) != len(paths):
            raise FixtureCorpusError("base tree has duplicate paths")
        for path in paths:
            if path.startswith("/") or ".." in path.split("/"):
                raise FixtureCorpusError(f"illegal base path {path}")

        ids = [case.case_id for case in self.mutations]
        if len(set(ids)) != len(ids):
            raise FixtureCorpusError("mutation case_id values must be unique")
        if ids != sorted(ids):
            raise FixtureCorpusError("mutations must be ordered by case_id")

        categories = {case.category for case in self.mutations}
        missing = set(REQUIRED_CATEGORIES) - categories
        if missing:
            raise FixtureCorpusError(f"missing required categories: {sorted(missing)}")

        for case in self.mutations:
            # Ensure operations apply cleanly.
            mutated = apply_operations(self.base_files, case.operations)
            # Every mutation must change the tree or declare a harness scenario
            # that still produces a distinct tree digest via operations.
            if tree_digest(mutated) == tree_digest(self.base_files):
                raise FixtureCorpusError(
                    f"{case.case_id}: mutated tree digest equals base"
                )
            # Independent oracle facets must all be present (enforced by types).
            oracle = case.oracle
            assert oracle.changed_symbol.symbol_ids
            assert oracle.merkle.changed_node_ids or oracle.merkle.affected_path_ids
            # Receipt freshness always declared.
            assert oracle.receipt_freshness.disposition
            # Confidence always declared.
            assert oracle.confidence.confidence

            if case.change_is_bounded:
                stats = bounded_change_stats(self.base_files, case)
                if stats["operation_count"] > BOUNDED_CHANGE_MAX_OPS:
                    raise FixtureCorpusError(
                        f"{case.case_id}: bounded change exceeds op limit"
                    )
                if stats["changed_bytes"] > BOUNDED_CHANGE_MAX_BYTES:
                    raise FixtureCorpusError(
                        f"{case.case_id}: bounded change exceeds byte limit"
                    )

            if case.category == "post_scan_source_race":
                if SOURCE_RACE_PATH not in case.pack_excluded_paths:
                    raise FixtureCorpusError(
                        "post_scan_source_race must exclude race path from packs"
                    )
                if not case.source_race_bytes_forbidden:
                    raise FixtureCorpusError(
                        "post_scan_source_race must forbid source-race bytes in packs"
                    )
                race_body = mutated[SOURCE_RACE_PATH].encode("utf-8")
                if SOURCE_RACE_MARKER not in race_body:
                    raise FixtureCorpusError(
                        "post_scan_source_race mutated tree missing race marker"
                    )
                # Pack paths must not include race marker.
                allowed = pack_candidate_paths(case, include_source_race_paths=False)
                pack_bytes = {
                    path: mutated[path].encode("utf-8")
                    for path in allowed
                    if path in mutated
                }
                if pack_contains_source_race_bytes(pack_bytes):
                    raise FixtureCorpusError(
                        "source-race marker leaked into pack candidate paths"
                    )

            if case.category == "unrelated_formatting":
                if not case.change_is_bounded:
                    raise FixtureCorpusError("formatting change must be bounded")
                if case.oracle.invalidation.invalidation_symbol_ids:
                    raise FixtureCorpusError(
                        "unrelated formatting must not invalidate symbols"
                    )

    def get_mutation(self, case_id: str) -> MutationCase:
        for case in self.mutations:
            if case.case_id == case_id:
                return case
        raise KeyError(case_id)

    def base_tree(self) -> dict[str, str]:
        return dict(self.base_files)

    def mutated_tree(self, case_id: str) -> dict[str, str]:
        case = self.get_mutation(case_id)
        return apply_operations(self.base_files, case.operations)

    def base_tree_digest(self) -> str:
        return tree_digest(self.base_files)

    def mutated_tree_digest(self, case_id: str) -> str:
        return tree_digest(self.mutated_tree(case_id))

    def materialize_base(self, destination: Path, *, git: bool = False) -> dict[str, str]:
        if git:
            return materialize_git_tree(self.base_files, destination)
        write_tree(self.base_files, destination)
        return {
            "root": str(destination),
            "tree_digest": self.base_tree_digest(),
        }

    def materialize_mutation(
        self, case_id: str, destination: Path, *, git: bool = False
    ) -> dict[str, str]:
        files = self.mutated_tree(case_id)
        if git:
            result = materialize_git_tree(files, destination)
            result["tree_digest"] = tree_digest(files)
            result["case_id"] = case_id
            return result
        write_tree(files, destination)
        return {
            "root": str(destination),
            "tree_digest": tree_digest(files),
            "case_id": case_id,
        }

    def declared_pack_paths(self, case_id: str) -> tuple[str, ...]:
        return pack_candidate_paths(self.get_mutation(case_id))

    def to_manifest(self) -> dict[str, Any]:
        """Compact manifest: digests and oracles, not full file bodies."""

        return {
            "schema": self.schema,
            "interface": self.interface,
            "corpus_id": self.corpus_id,
            "base_tree_digest": self.base_tree_digest(),
            "base_path_count": len(self.base_files),
            "base_paths": sorted(self.base_files),
            "required_categories": list(REQUIRED_CATEGORIES),
            "mutations": [
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "description": case.description,
                    "mutated_tree_digest": self.mutated_tree_digest(case.case_id),
                    "changed_paths": list(
                        changed_paths(
                            self.base_files,
                            apply_operations(self.base_files, case.operations),
                        )
                    ),
                    "oracle": case.oracle.to_dict(),
                    "source_race_bytes_forbidden": case.source_race_bytes_forbidden,
                    "change_is_bounded": case.change_is_bounded,
                    "pack_excluded_paths": list(case.pack_excluded_paths),
                    "harness_scenario": case.harness_scenario,
                    "production_eligible": case.production_eligible,
                    "pack_paths": list(self.declared_pack_paths(case.case_id)),
                }
                for case in self.mutations
            ],
        }

    def manifest_digest(self) -> str:
        return "sha256:" + hashlib.sha256(
            _canonical_json(self.to_manifest())
        ).hexdigest()
