"""SemanticGovernorFixtureCorpus@1 — partitioned controlled fixture corpus.

Provides a shared base tree, calibration/development/held-out partitions,
scanner-view and omission/outcome oracles, materialisation helpers, and a
compact content-addressed manifest. Never imports or executes target modules.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .case_record import (
    ADVERSARIAL_SCENARIOS,
    PARTITIONS,
    TASK_FAMILIES,
    FixtureCase,
    FixtureCorpusError,
    PathOperation,
)
from .recipes import (
    REQUIRED_PARTITION_FAMILY_PAIRS,
    base_tree_files,
    fixture_cases,
)

FIXTURE_CORPUS_INTERFACE = "SemanticGovernorFixtureCorpus@1"
FIXTURE_CORPUS_SCHEMA = "scg/partitioned-corpus@1"
CORPUS_ID = "semantic-governor-partitioned-corpus-v1"
EVIDENCE_ID = "scg/partitioned-corpus@1"
TASK_ID = "SCG-040"

# Forbidden artifact kinds in controlled fixture data (conflict policy).
FORBIDDEN_PAYLOAD_MARKERS = (
    "model_output",
    "completion_receipt",
    "state.db",
    "duckdb",
    "provider_response",
)


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
        "schema": "scg-fixture/tree-digest@1",
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


def read_tree_bytes(root: Path) -> dict[str, bytes]:
    """Read all regular files under root as bytes (scan-safe; no import)."""

    root = Path(root)
    result: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        result[rel] = path.read_bytes()
    return result


@dataclass(frozen=True)
class SemanticGovernorFixtureCorpus:
    """In-memory partitioned fixture corpus (SemanticGovernorFixtureCorpus@1)."""

    corpus_id: str
    interface: str
    schema: str
    evidence_id: str
    task_id: str
    base_files: Mapping[str, str]
    cases: tuple[FixtureCase, ...]

    @classmethod
    def load(cls) -> "SemanticGovernorFixtureCorpus":
        raw = base_tree_files()
        base = {path: raw[path] for path in sorted(raw)}
        cases = fixture_cases()
        corpus = cls(
            corpus_id=CORPUS_ID,
            interface=FIXTURE_CORPUS_INTERFACE,
            schema=FIXTURE_CORPUS_SCHEMA,
            evidence_id=EVIDENCE_ID,
            task_id=TASK_ID,
            base_files=base,
            cases=cases,
        )
        corpus.validate()
        return corpus

    def validate(self) -> None:
        if self.interface != FIXTURE_CORPUS_INTERFACE:
            raise FixtureCorpusError("interface mismatch")
        if self.schema != FIXTURE_CORPUS_SCHEMA:
            raise FixtureCorpusError("schema mismatch")
        if self.corpus_id != CORPUS_ID:
            raise FixtureCorpusError("corpus_id mismatch")
        if self.evidence_id != EVIDENCE_ID:
            raise FixtureCorpusError("evidence_id mismatch")
        if self.task_id != TASK_ID:
            raise FixtureCorpusError("task_id mismatch")
        if not self.base_files:
            raise FixtureCorpusError("base tree is empty")

        paths = list(self.base_files)
        if paths != sorted(paths):
            raise FixtureCorpusError("base tree keys must be sorted")
        if len(set(paths)) != len(paths):
            raise FixtureCorpusError("base tree has duplicate paths")
        for path in paths:
            if path.startswith("/") or ".." in path.split("/"):
                raise FixtureCorpusError(f"illegal base path {path}")

        ids = [case.case_id for case in self.cases]
        if len(set(ids)) != len(ids):
            raise FixtureCorpusError("case_id values must be unique")
        if ids != sorted(ids):
            raise FixtureCorpusError("cases must be ordered by case_id")

        # Partitions are disjoint by construction (case belongs to one
        # partition) and must all be non-empty.
        by_partition: dict[str, list[str]] = {name: [] for name in PARTITIONS}
        for case in self.cases:
            by_partition[case.partition].append(case.case_id)
        for name in PARTITIONS:
            if not by_partition[name]:
                raise FixtureCorpusError(f"partition {name!r} is empty")

        # Pairwise disjoint membership (explicit check).
        sets = {name: set(members) for name, members in by_partition.items()}
        for left in PARTITIONS:
            for right in PARTITIONS:
                if left >= right:
                    continue
                overlap = sets[left] & sets[right]
                if overlap:
                    raise FixtureCorpusError(
                        f"partitions {left!r} and {right!r} overlap: "
                        f"{sorted(overlap)}"
                    )

        present_pairs = {(case.partition, case.family) for case in self.cases}
        missing_pairs = set(REQUIRED_PARTITION_FAMILY_PAIRS) - present_pairs
        if missing_pairs:
            raise FixtureCorpusError(
                f"missing partition/family pairs: {sorted(missing_pairs)}"
            )

        adversarial = {
            case.adversarial_scenario
            for case in self.cases
            if case.adversarial_scenario is not None
        }
        missing_adv = set(ADVERSARIAL_SCENARIOS) - adversarial
        if missing_adv:
            raise FixtureCorpusError(
                f"missing adversarial scenarios: {sorted(missing_adv)}"
            )
        # Adversarial cases live only in held_out.
        for case in self.cases:
            if case.adversarial_scenario and case.partition != "held_out":
                raise FixtureCorpusError(
                    f"{case.case_id}: adversarial scenario outside held_out"
                )

        for case in self.cases:
            mutated = apply_operations(self.base_files, case.operations)
            if tree_digest(mutated) == tree_digest(self.base_files):
                raise FixtureCorpusError(
                    f"{case.case_id}: mutated tree digest equals base"
                )
            actual_changed = set(changed_paths(self.base_files, mutated))
            declared = set(case.scanner_view.changed_paths)
            if not declared.issubset(actual_changed):
                raise FixtureCorpusError(
                    f"{case.case_id}: scanner changed_paths not subset of "
                    f"actual tree delta: {sorted(declared - actual_changed)}"
                )
            # Conflict policy: no model outputs / receipts / state DB markers.
            for path, body in mutated.items():
                lowered = f"{path}\n{body}".lower()
                for marker in FORBIDDEN_PAYLOAD_MARKERS:
                    if marker in lowered:
                        raise FixtureCorpusError(
                            f"{case.case_id}: forbidden marker {marker!r} in {path}"
                        )

    def get_case(self, case_id: str) -> FixtureCase:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(case_id)

    def cases_for_partition(self, partition: str) -> tuple[FixtureCase, ...]:
        if partition not in PARTITIONS:
            raise FixtureCorpusError(f"unsupported partition {partition!r}")
        return tuple(case for case in self.cases if case.partition == partition)

    def cases_for_family(self, family: str) -> tuple[FixtureCase, ...]:
        if family not in TASK_FAMILIES:
            raise FixtureCorpusError(f"unsupported family {family!r}")
        return tuple(case for case in self.cases if case.family == family)

    def base_tree(self) -> dict[str, str]:
        return dict(self.base_files)

    def mutated_tree(self, case_id: str) -> dict[str, str]:
        case = self.get_case(case_id)
        return apply_operations(self.base_files, case.operations)

    def base_tree_digest(self) -> str:
        return tree_digest(self.base_files)

    def mutated_tree_digest(self, case_id: str) -> str:
        return tree_digest(self.mutated_tree(case_id))

    def materialize_base(self, destination: Path) -> dict[str, str]:
        write_tree(self.base_files, destination)
        return {
            "root": str(destination),
            "tree_digest": self.base_tree_digest(),
        }

    def materialize_case(self, case_id: str, destination: Path) -> dict[str, str]:
        files = self.mutated_tree(case_id)
        write_tree(files, destination)
        return {
            "root": str(destination),
            "tree_digest": tree_digest(files),
            "case_id": case_id,
        }

    def partition_membership(self) -> dict[str, list[str]]:
        return {
            name: [case.case_id for case in self.cases_for_partition(name)]
            for name in PARTITIONS
        }

    def to_manifest(self) -> dict[str, Any]:
        """Compact manifest: digests and oracles, not full file bodies."""

        membership = self.partition_membership()
        cases_payload = []
        for case in self.cases:
            mutated = apply_operations(self.base_files, case.operations)
            cases_payload.append(
                {
                    "case_id": case.case_id,
                    "partition": case.partition,
                    "family": case.family,
                    "description": case.description,
                    "adversarial_scenario": case.adversarial_scenario,
                    "production_eligible": case.production_eligible,
                    "mutated_tree_digest": tree_digest(mutated),
                    "changed_paths": list(changed_paths(self.base_files, mutated)),
                    "scanner_view": case.scanner_view.to_dict(),
                    "omission": case.omission.to_dict(),
                    "outcome": case.outcome.to_dict(),
                }
            )
        payload = {
            "schema": self.schema,
            "interface": self.interface,
            "evidence_id": self.evidence_id,
            "task_id": self.task_id,
            "corpus_id": self.corpus_id,
            "base_tree_digest": self.base_tree_digest(),
            "base_path_count": len(self.base_files),
            "base_paths": sorted(self.base_files),
            "partitions": list(PARTITIONS),
            "task_families": list(TASK_FAMILIES),
            "adversarial_scenarios": list(ADVERSARIAL_SCENARIOS),
            "partition_membership": membership,
            "partition_counts": {
                name: len(members) for name, members in membership.items()
            },
            "case_count": len(self.cases),
            "cases": cases_payload,
        }
        # Self-identity: digest over payload without corpus_digest field.
        payload["corpus_digest"] = (
            "sha256:" + hashlib.sha256(_canonical_json(payload)).hexdigest()
        )
        return payload

    def manifest_digest(self) -> str:
        return str(self.to_manifest()["corpus_digest"])
