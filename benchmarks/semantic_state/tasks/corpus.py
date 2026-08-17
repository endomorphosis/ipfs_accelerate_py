"""BenchmarkCorpus loader for SemanticStateBenchmarkCorpus@1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .recipes import DEFAULT_BASELINE_RETRIEVAL, build_tasks
from .task_record import (
    REQUIRED_CATEGORY_COUNTS,
    TASK_CATEGORIES,
    BaselineRetrievalPolicy,
    BenchmarkCorpusError,
    BenchmarkTask,
)

BENCHMARK_CORPUS_INTERFACE = "SemanticStateBenchmarkCorpus@1"
BENCHMARK_CORPUS_SCHEMA = "ipfs_accelerate_py/semantic-state/benchmark-corpus@1"
CORPUS_ID = "semantic-state-benchmark-corpus-v1"
EXPECTED_TASK_COUNT = 40

PACKAGE_DIR = Path(__file__).resolve().parent
CORPUS_JSON_PATH = PACKAGE_DIR.parent / "corpus.json"
FIXTURE_PACKAGE_PATH = "test/fixtures/semantic_state_harness/controlled_repo"
FIXTURE_CORPUS_ID = "semantic-state-controlled-repo-v1"


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def content_digest(payload: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class BenchmarkCorpus:
    """Exactly-40-task offline benchmark corpus."""

    corpus_id: str
    interface: str
    schema: str
    fixture_corpus_id: str
    fixture_package_path: str
    baseline_retrieval_defaults: BaselineRetrievalPolicy
    tasks: tuple[BenchmarkTask, ...]

    @classmethod
    def load(cls) -> "BenchmarkCorpus":
        tasks = build_tasks()
        corpus = cls(
            corpus_id=CORPUS_ID,
            interface=BENCHMARK_CORPUS_INTERFACE,
            schema=BENCHMARK_CORPUS_SCHEMA,
            fixture_corpus_id=FIXTURE_CORPUS_ID,
            fixture_package_path=FIXTURE_PACKAGE_PATH,
            baseline_retrieval_defaults=DEFAULT_BASELINE_RETRIEVAL,
            tasks=tasks,
        )
        corpus.validate()
        return corpus

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkCorpus":
        schema = payload.get("schema")
        if schema is not None and schema != BENCHMARK_CORPUS_SCHEMA:
            raise BenchmarkCorpusError(f"unsupported corpus schema {schema!r}")
        interface = str(payload["interface"])
        if interface != BENCHMARK_CORPUS_INTERFACE:
            raise BenchmarkCorpusError(f"unsupported interface {interface!r}")
        tasks = tuple(
            BenchmarkTask.from_dict(item) for item in payload["tasks"]
        )
        corpus = cls(
            corpus_id=str(payload["corpus_id"]),
            interface=interface,
            schema=BENCHMARK_CORPUS_SCHEMA,
            fixture_corpus_id=str(payload["fixture_corpus_id"]),
            fixture_package_path=str(payload["fixture_package_path"]),
            baseline_retrieval_defaults=BaselineRetrievalPolicy.from_dict(
                payload["baseline_retrieval_defaults"]
            ),
            tasks=tasks,
        )
        corpus.validate()
        return corpus

    @classmethod
    def load_checked_in_json(
        cls, path: Path | None = None
    ) -> "BenchmarkCorpus":
        target = Path(path) if path is not None else CORPUS_JSON_PATH
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise BenchmarkCorpusError("corpus.json must be an object")
        return cls.from_dict(payload)

    def validate(self) -> None:
        if self.interface != BENCHMARK_CORPUS_INTERFACE:
            raise BenchmarkCorpusError("interface mismatch")
        if self.schema != BENCHMARK_CORPUS_SCHEMA:
            raise BenchmarkCorpusError("schema mismatch")
        if self.corpus_id != CORPUS_ID:
            raise BenchmarkCorpusError("corpus_id mismatch")
        if self.fixture_corpus_id != FIXTURE_CORPUS_ID:
            raise BenchmarkCorpusError("fixture_corpus_id mismatch")
        if self.fixture_package_path != FIXTURE_PACKAGE_PATH:
            raise BenchmarkCorpusError("fixture_package_path mismatch")

        if len(self.tasks) != EXPECTED_TASK_COUNT:
            raise BenchmarkCorpusError(
                f"expected {EXPECTED_TASK_COUNT} tasks, got {len(self.tasks)}"
            )

        ids = [task.task_id for task in self.tasks]
        if len(set(ids)) != len(ids):
            raise BenchmarkCorpusError("task_id values must be unique")
        if ids != sorted(ids):
            raise BenchmarkCorpusError("tasks must be ordered by task_id")

        counts: dict[str, int] = {category: 0 for category in TASK_CATEGORIES}
        multi_file_count = 0
        frontier_count = 0
        for task in self.tasks:
            counts[task.category] = counts.get(task.category, 0) + 1
            if task.multi_file:
                multi_file_count += 1
            if task.frontier_or_human:
                frontier_count += 1
            if task.candidate.production_eligible:
                raise BenchmarkCorpusError(
                    f"{task.task_id}: production_eligible must be false"
                )
            if (
                task.baseline_retrieval.fixture_corpus_id
                != self.fixture_corpus_id
            ):
                raise BenchmarkCorpusError(
                    f"{task.task_id}: baseline fixture corpus mismatch"
                )
            if task.oracle.oracle_authority not in {
                "reviewed_fixture_authority",
                "controlled_fixture_oracle",
            }:
                raise BenchmarkCorpusError(
                    f"{task.task_id}: invalid oracle authority"
                )
            # Never accept production for corpus fixtures.
            if task.oracle.production_acceptance not in {
                "not_applicable",
                "rejected",
                "blocked",
            }:
                raise BenchmarkCorpusError(
                    f"{task.task_id}: invalid production_acceptance"
                )

        for category, expected in REQUIRED_CATEGORY_COUNTS.items():
            actual = counts.get(category, 0)
            if actual != expected:
                raise BenchmarkCorpusError(
                    f"category {category}: expected {expected}, got {actual}"
                )
        if multi_file_count < 6:
            raise BenchmarkCorpusError(
                f"expected at least 6 multi-file tasks, got {multi_file_count}"
            )
        if frontier_count < 2:
            raise BenchmarkCorpusError(
                "expected frontier/human cases in rejection/escalation cohort"
            )
        # Category multi_file_refactor contributes 6 multi-file tasks; others may add more.
        if counts["multi_file_refactor"] != 6:
            raise BenchmarkCorpusError("multi_file_refactor count must be 6")
        if counts["rejection_or_escalation"] != 6:
            raise BenchmarkCorpusError(
                "rejection_or_escalation count must be 6"
            )

    def get_task(self, task_id: str) -> BenchmarkTask:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(task_id)

    def category_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {category: 0 for category in sorted(TASK_CATEGORIES)}
        for task in self.tasks:
            counts[task.category] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "corpus_id": self.corpus_id,
            "task_count": len(self.tasks),
            "category_counts": self.category_counts(),
            "fixture_corpus_id": self.fixture_corpus_id,
            "fixture_package_path": self.fixture_package_path,
            "baseline_retrieval_defaults": (
                self.baseline_retrieval_defaults.to_dict()
            ),
            "required_category_counts": dict(REQUIRED_CATEGORY_COUNTS),
            "task_ids": [task.task_id for task in self.tasks],
            "tasks": [task.to_dict() for task in self.tasks],
            "notes": [
                "Checked-in candidate patches are oracle/replay fixtures only.",
                "production_eligible is always false; candidates never advance a production root.",
                "Candidate verification outcome is separate from production acceptance.",
                "Oracles are reviewed fixture authority, not benchmark implementation output.",
                "Tasks are runnable offline against the pinned controlled fixture tree.",
            ],
        }

    def manifest_digest(self) -> str:
        return content_digest(self.to_dict())

    def write_corpus_json(self, path: Path | None = None) -> Path:
        target = Path(path) if path is not None else CORPUS_JSON_PATH
        payload = self.to_dict()
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
        if not text.endswith("\n"):
            text += "\n"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target
