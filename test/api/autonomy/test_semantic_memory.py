from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
    AuthorityClass,
    MemoryClass,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.semantic_memory import (
    MEMORY_ENTRY_INTERFACE,
    SEMANTIC_MEMORY_INTERFACE,
    MemoryEntry,
    SemanticMemory,
    SemanticMemoryError,
)


def _entry(**overrides: object) -> MemoryEntry:
    values: dict[str, object] = {
        "artifact_id": "artifact-rule-1",
        "memory_class": MemoryClass.REPOSITORY_PATTERN,
        "evidence_class": AuthorityClass.VERIFIED,
        "created_at_ms": 1_000,
        "ttl_ms": 10_000,
        "dependency_ids": ("contract-1",),
        "scope_ids": ("repo-ipfs_accelerate_py",),
        "frequency": 0,
        "retained_kind": "pattern",
    }
    values.update(overrides)
    return MemoryEntry(**values)


def test_interfaces_are_versioned() -> None:
    assert SEMANTIC_MEMORY_INTERFACE == "SemanticMemory@1"
    assert MEMORY_ENTRY_INTERFACE == "MemoryEntry@1"
    assert SemanticMemory.INTERFACE == SEMANTIC_MEMORY_INTERFACE
    assert MemoryEntry.INTERFACE == MEMORY_ENTRY_INTERFACE


def test_memory_rejects_prompt_source_and_private_reasoning() -> None:
    with pytest.raises(SemanticMemoryError, match="private reasoning|prompt|source"):
        MemoryEntry(
            artifact_id="artifact-1",
            memory_class=MemoryClass.TASK_EPISODE,
            evidence_class=AuthorityClass.ADVISORY,
            created_at_ms=1,
            ttl_ms=10,
            retained_kind="prompt_body",
        )
    with pytest.raises(SemanticMemoryError, match="raw prompt|source|private"):
        MemoryEntry(
            artifact_id="raw_prompt-secret",
            memory_class=MemoryClass.TASK_EPISODE,
            evidence_class=AuthorityClass.ADVISORY,
            created_at_ms=1,
            ttl_ms=10,
            retained_kind="outcome",
        )


def test_frequency_changes_rank_but_not_authority() -> None:
    memory = SemanticMemory()
    first = memory.admit(_entry(artifact_id="pattern-a", frequency=0))
    second = memory.admit(
        _entry(
            artifact_id="pattern-b",
            frequency=0,
            created_at_ms=2_000,
        )
    )
    observed = memory.observe(second.entry_id)
    assert observed.evidence_class is AuthorityClass.VERIFIED
    assert observed.frequency == 1
    ranked = memory.retrieve(now_ms=3_000)
    assert ranked[0].artifact_id == "pattern-b"
    assert ranked[0].evidence_class is ranked[1].evidence_class


def test_ttl_invalidation_and_compaction_are_targeted() -> None:
    memory = SemanticMemory()
    ephemeral = memory.admit(
        _entry(
            artifact_id="attempt-1",
            memory_class=MemoryClass.EPHEMERAL_ATTEMPT,
            evidence_class=AuthorityClass.ADVISORY,
            ttl_ms=5,
            retained_kind="artifact",
            dependency_ids=(),
        )
    )
    stable = memory.admit(
        _entry(
            artifact_id="rule-1",
            memory_class=MemoryClass.CROSS_REPOSITORY_RULE,
            retained_kind="rule",
            dependency_ids=("contract-1",),
        )
    )
    counterexample = memory.admit(
        _entry(
            artifact_id="cex-1",
            memory_class=MemoryClass.AUTHORITATIVE_CURRENT,
            retained_kind="counterexample",
            created_at_ms=1_500,
            dependency_ids=(),
        )
    )
    compacted = memory.compact(now_ms=20_000)
    ids = {item.entry_id for item in compacted}
    assert ephemeral.entry_id not in ids
    assert stable.entry_id in ids
    assert counterexample.entry_id in ids
    receipt = memory.invalidate("contract-1")
    assert stable.entry_id in receipt.withdrawn_entry_ids
    assert counterexample.entry_id in receipt.remaining_entry_ids
    assert memory.retrieve(now_ms=20_000)[0].artifact_id == "cex-1"


def test_withdrawn_entries_cannot_be_revived_as_authority() -> None:
    memory = SemanticMemory()
    entry = memory.admit(_entry())
    memory.invalidate(entry.artifact_id)
    with pytest.raises(SemanticMemoryError, match="withdrawn"):
        memory.admit(entry)
