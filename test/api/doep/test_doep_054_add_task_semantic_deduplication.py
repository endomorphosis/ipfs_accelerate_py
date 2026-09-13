"""Independent current-tree checks for DOEP-054 task semantic deduplication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.semantic_refill import (
    AUTOMATIC_BOUNDED_TASK_REFILL_BINDING,
    SEMANTIC_REFILL_BINDING,
    SEMANTIC_REFILL_INTERFACE,
    SEMANTIC_REFILL_SCHEMA,
    TASK_SEMANTIC_DEDUPLICATION_BINDING,
    TASK_SEMANTIC_DEDUPLICATION_CONSUMES,
    TASK_SEMANTIC_DEDUPLICATION_INTERFACE,
    TASK_SEMANTIC_DEDUPLICATION_SCHEMA,
    RefillError,
    SemanticDeduplicationDisposition,
    SemanticDeduplicationError,
    SemanticDeduplicationResult,
    SemanticRefill,
    deduplicate_refill_tasks,
    propose_refill,
    semantically_deduplicate_tasks,
    task_semantic_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_identity import (
    TASK_IDENTITY_SCHEMA,
    canonical_task_identity,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = (
    ACCELERATE_ROOT
    / "ipfs_accelerate_py/agent_supervisor/task_sources/semantic_refill.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-054.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-054.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/task_sources/semantic_refill.py",
    "test/api/doep/test_doep_054_add_task_semantic_deduplication.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-054.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-054.json",
)
TASK_CID = "sha256:f289b3dbbe1e738901c8a0e832da215387b2250747a9fab18ea3bfdb934fc61b"
PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
BASE_REPOSITORIES = {
    "ipfs_accelerate_py": {
        "commit": "87715e9295626e7918f7fc8a7b1a1531ab04208f",
        "tree": "1c9a399cc7a599d5904e5be2ae58c6be3650cff7",
    },
    "ipfs_datasets_py": {
        "commit": "3668b8857a9aa7b1a3c847be12725b5cd057d2e7",
        "tree": "456e09b51d6a07a3a5873436df24054768195320",
    },
    "ipfs_kit_py": {
        "commit": "b6c65ba732733d7e33852713ba18aa3b12235668",
        "tree": "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2",
    },
    "lift_coding": {
        "commit": "bb8869ed72eb7002434345d9969efee729c4f7f6",
        "tree": "99e85bfe584b7688ffbeff86da1e612dd6893a42",
    },
}


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _task(
    task_id: str,
    *,
    title: str = "Add a durable task ledger",
    outputs: tuple[str, ...] = ("src/ledger.py", "tests/test_ledger.py"),
    acceptance: str = "Retries and receipts retain canonical identity.",
    goal: str = "G9.S1",
    lifecycle: str = "unstarted",
    **extra: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "task_id": task_id,
        "title": title,
        "outputs": list(outputs),
        "acceptance": acceptance,
        "lifecycle": lifecycle,
        "metadata": {"goal id": goal},
    }
    payload.update(extra)
    return payload


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_binding_extends_semantic_refill_without_competing_subsystem() -> None:
    assert SEMANTIC_REFILL_INTERFACE == "SemanticRefill@1"
    assert SEMANTIC_REFILL_BINDING == SEMANTIC_REFILL_INTERFACE
    assert SEMANTIC_REFILL_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/semantic-refill@1"
    )
    assert TASK_SEMANTIC_DEDUPLICATION_INTERFACE == "TaskSemanticDeduplication@1"
    assert TASK_SEMANTIC_DEDUPLICATION_BINDING == TASK_SEMANTIC_DEDUPLICATION_INTERFACE
    assert TASK_SEMANTIC_DEDUPLICATION_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/task-semantic-deduplication@1"
    )
    assert TASK_SEMANTIC_DEDUPLICATION_CONSUMES == (
        TASK_IDENTITY_SCHEMA,
        AUTOMATIC_BOUNDED_TASK_REFILL_BINDING,
    )
    assert SemanticRefill.DEDUPLICATION_INTERFACE == (
        TASK_SEMANTIC_DEDUPLICATION_INTERFACE
    )
    assert SemanticRefill.BINDING == SEMANTIC_REFILL_BINDING
    assert semantically_deduplicate_tasks.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.task_sources.semantic_refill"
    )
    assert propose_refill.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.task_sources.semantic_refill"
    )
    assert task_semantic_identity.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.task_sources.semantic_refill"
    )
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "class SemanticRefill" in source
    assert "def propose_refill(" in source
    assert "def semantically_deduplicate_tasks(" in source
    assert "def deduplicate_refill_tasks(" in source
    assert "canonical_task_identity" in source
    assert "not a second identity owner" in source
    assert "never write DuckDB" in source
    assert "cannot skip deduplication" in source
    assert "class CompetingDeduplicator" not in source
    assert "class SemanticDedupEngine" not in source
    assert "class CompetingRefill" not in source
    assert "CREATE TABLE" not in source
    assert "import duckdb" not in source.casefold()
    assert "connect(" not in source


def test_existing_propose_refill_remains_proposal_only() -> None:
    result = propose_refill({"bound": 1, "max_bound": 4})
    assert result["accepted"] is False
    assert result["proposal"] is True
    assert result["bound"] == 1
    assert result["completion_authoritative"] is False
    assert result["empty_queue_is_completion"] is False
    assert result["database_write"] is False
    with pytest.raises(RefillError, match="accepted history is immutable"):
        propose_refill({"rewrite_accepted": True})
    with pytest.raises(RefillError, match="refill exceeds bound"):
        propose_refill({"bound": 5, "max_bound": 4})


def test_display_id_and_cosmetic_rewrite_cannot_escape_semantic_dedup() -> None:
    original = _task("REF-001")
    cosmetic = _task(
        "LOCAL-987",
        title="  ADD A DURABLE TASK LEDGER  ",
    )
    identity = task_semantic_identity(
        original, board_namespace="board-a"
    )
    alias = task_semantic_identity(
        cosmetic, board_namespace="board-b"
    )
    assert identity.canonical_task_cid == alias.canonical_task_cid
    assert identity.canonical_task_cid == canonical_task_identity(original).canonical_task_cid
    result = semantically_deduplicate_tasks(
        (original, cosmetic),
        worker_assertion=True,
    )
    assert result.admitted_task_ids == ("REF-001",)
    suppressed = result.suppressed
    assert len(suppressed) == 1
    assert suppressed[0].disposition is SemanticDeduplicationDisposition.DUPLICATE_WAVE
    assert suppressed[0].candidate_id == "LOCAL-987"
    assert suppressed[0].canonical_task_cid == identity.canonical_task_cid


def test_existing_open_and_immutable_history_suppress_refill() -> None:
    existing_open = _task("OPEN-1", lifecycle="ready")
    claimed = _task(
        "CLAIM-1",
        title="Add dependency-aware task scheduling",
        outputs=("src/scheduler.py", "tests/test_scheduler.py"),
        acceptance="Scheduler preserves canonical identity.",
        lifecycle="claimed",
    )
    accepted = _task(
        "DONE-1",
        title="Seal current baseline",
        outputs=("artifacts/baseline.json",),
        acceptance="Baseline tree is sealed.",
        lifecycle="accepted",
    )
    refill_open = _task("REFILL-OPEN", lifecycle="unstarted")
    refill_claimed = _task(
        "REFILL-CLAIM",
        title="Add dependency-aware task scheduling",
        outputs=("src/scheduler.py", "tests/test_scheduler.py"),
        acceptance="Scheduler preserves canonical identity.",
        lifecycle="unstarted",
    )
    refill_accepted = _task(
        "REFILL-DONE",
        title="Seal current baseline",
        outputs=("artifacts/baseline.json",),
        acceptance="Baseline tree is sealed.",
        lifecycle="unstarted",
    )
    novel = _task(
        "NOVEL-1",
        title="Add event-driven reassessment",
        outputs=("src/reassess.py", "tests/test_reassess.py"),
        acceptance="Reassessment is model-free.",
    )
    result = deduplicate_refill_tasks(
        (refill_open, refill_claimed, refill_accepted, novel),
        existing_tasks=(existing_open, claimed, accepted),
        worker_assertion=True,
    )
    assert result.admitted_task_ids == ("NOVEL-1",)
    dispositions = {
        item.candidate_id: item.disposition for item in result.decisions
    }
    assert dispositions["REFILL-OPEN"] is SemanticDeduplicationDisposition.DUPLICATE_EXISTING
    assert dispositions["REFILL-CLAIM"] is SemanticDeduplicationDisposition.IMMUTABLE_HISTORY
    assert dispositions["REFILL-DONE"] is SemanticDeduplicationDisposition.IMMUTABLE_HISTORY
    assert dispositions["NOVEL-1"] is SemanticDeduplicationDisposition.ADMITTED
    payload = result.to_dict()
    assert payload["authorizes_append"] is False
    assert payload["authorizes_completion"] is False
    assert payload["database_write"] is False
    assert payload["completion_authoritative"] is False
    assert payload["worker_assertion_is_authority"] is False
    assert payload["empty_queue_is_completion"] is False
    assert payload["identity_owner"] == TASK_IDENTITY_SCHEMA
    assert payload["binding"] == TASK_SEMANTIC_DEDUPLICATION_BINDING
    assert payload["carrier"] == "SemanticRefill"
    assert '"completion_authoritative": true' not in json.dumps(payload)


def test_distinct_semantic_work_is_not_collapsed() -> None:
    first = _task("A")
    second = _task(
        "B",
        title="Add dependency-aware task scheduling",
        outputs=("src/scheduler.py", "tests/test_scheduler.py"),
        acceptance="A different implementation contract.",
    )
    result = semantically_deduplicate_tasks((first, second))
    assert result.admitted_task_ids == ("A", "B")
    assert result.reason == "admitted"
    assert not result.suppressed


def test_idle_capacity_alone_cannot_generate_work() -> None:
    candidate = _task("IDLE-1")
    result = semantically_deduplicate_tasks(
        (candidate,),
        idle_capacity=True,
        worker_assertion=True,
    )
    assert result.admitted_task_ids == ()
    assert result.reason == "idle_capacity_is_not_a_refill_trigger"
    assert result.decisions[0].disposition is SemanticDeduplicationDisposition.IDLE_CAPACITY
    proposed = propose_refill(
        {
            "bound": 1,
            "idle_capacity": True,
            "candidates": [candidate],
            "worker_assertion": True,
        }
    )
    assert proposed["accepted"] is False
    assert proposed["admitted_task_ids"] == []
    assert proposed["reason"] == "idle_capacity_is_not_a_refill_trigger"


def test_worker_assertion_cannot_bypass_dedup_or_complete_empty_queue() -> None:
    existing = _task("OPEN-1", lifecycle="ready")
    duplicate = _task("ALIAS-1")
    empty = semantically_deduplicate_tasks(
        (),
        existing_tasks=(existing,),
        worker_assertion=True,
    )
    assert empty.admitted_task_ids == ()
    assert empty.to_dict()["empty_queue_is_completion"] is False
    assert empty.to_dict()["completion_authoritative"] is False
    blocked = SemanticRefill().deduplicate(
        (duplicate,),
        existing_tasks=(existing,),
        worker_assertion=True,
    )
    assert blocked.admitted_task_ids == ()
    assert blocked.suppressed[0].disposition is (
        SemanticDeduplicationDisposition.DUPLICATE_EXISTING
    )


def test_propose_refill_applies_dedup_and_never_accepts() -> None:
    proposed = SemanticRefill().propose(
        {
            "bound": 2,
            "max_bound": 4,
            "trigger": "semantic",
            "candidates": [
                _task("KEEP"),
                _task("DROP", title="  add a durable task ledger  "),
            ],
            "existing_tasks": (),
            "worker_assertion": True,
        }
    )
    assert proposed["accepted"] is False
    assert proposed["proposal"] is True
    assert proposed["admitted_task_ids"] == ["KEEP"]
    assert proposed["binding"] == SEMANTIC_REFILL_BINDING
    assert proposed["carrier"] == "SemanticRefill"
    assert proposed["deduplication"]["binding"] == TASK_SEMANTIC_DEDUPLICATION_BINDING
    assert proposed["database_write"] is False
    assert isinstance(proposed, type(propose_refill({"bound": 1})))


def test_missing_semantic_metadata_fails_closed() -> None:
    with pytest.raises(SemanticDeduplicationError, match="semantic work metadata"):
        semantically_deduplicate_tasks(({"task_id": "BARE"},))
    with pytest.raises(SemanticDeduplicationError, match="must be mappings"):
        propose_refill("not-a-record")  # type: ignore[arg-type]


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-054"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["entrypoint"] == (
        "semantically_deduplicate_tasks"
    )
    assert manifest["canonical_extension"]["carrier"] == "SemanticRefill"
    assert manifest["canonical_extension"]["binding"] == (
        TASK_SEMANTIC_DEDUPLICATION_BINDING
    )
    assert manifest["canonical_extension"]["authority"] == (
        "model_free_canonical_identity_dedup_proposal_only"
    )
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(SOURCE_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == (
        "pending_independent_fenced_supervisor"
    )
    assert SemanticDeduplicationResult().model_free is True
