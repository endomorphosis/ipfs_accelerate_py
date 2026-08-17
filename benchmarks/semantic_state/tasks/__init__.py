"""Exactly-40-task semantic-state benchmark corpus package.

Interface: ``SemanticStateBenchmarkCorpus@1``
Corpus: ``semantic-state-benchmark-corpus-v1``
Task: SCH-016 / ``sch/benchmark-corpus@1``
"""

from __future__ import annotations

from .corpus import (
    BENCHMARK_CORPUS_INTERFACE,
    BENCHMARK_CORPUS_SCHEMA,
    CORPUS_ID,
    CORPUS_JSON_PATH,
    EXPECTED_TASK_COUNT,
    FIXTURE_CORPUS_ID,
    FIXTURE_PACKAGE_PATH,
    BenchmarkCorpus,
)
from .task_record import (
    REQUIRED_CATEGORY_COUNTS,
    TASK_CATEGORIES,
    BaselineRetrievalPolicy,
    BenchmarkCorpusError,
    BenchmarkTask,
    CandidatePatch,
    TaskOracle,
)

__all__ = [
    "BENCHMARK_CORPUS_INTERFACE",
    "BENCHMARK_CORPUS_SCHEMA",
    "CORPUS_ID",
    "CORPUS_JSON_PATH",
    "EXPECTED_TASK_COUNT",
    "FIXTURE_CORPUS_ID",
    "FIXTURE_PACKAGE_PATH",
    "REQUIRED_CATEGORY_COUNTS",
    "TASK_CATEGORIES",
    "BaselineRetrievalPolicy",
    "BenchmarkCorpus",
    "BenchmarkCorpusError",
    "BenchmarkTask",
    "CandidatePatch",
    "TaskOracle",
]
