"""EAAEF-100: in-memory admitted-path merge loop."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.external_merge_loop import (
    MERGE_LOOP_STEPS,
    ExternalMergeLoop,
    ForgedReceiptError,
    MergeLoopReceipt,
    MissingReceiptError,
    WorkerSelfMergeError,
    canonical_tree_identity,
    issue_receipt,
    merge_accepted_result,
    patch_identity,
)


WORKER = "principal:worker"
REVIEWER = "principal:reviewer"
VERIFIER = "principal:verifier"
TASK_ID = "EAAEF-100"


def _loop() -> ExternalMergeLoop:
    return ExternalMergeLoop(files={"README.md": "base\n"})


def _patch() -> dict[str, str]:
    return {"owned.py": "print('accepted')\n"}


def _receipts(
    *,
    patch_id: str,
    reviewer: str = REVIEWER,
    worker: str = WORKER,
) -> dict[str, dict[str, object]]:
    return {
        "patch": issue_receipt(
            kind="patch",
            task_id=TASK_ID,
            patch_id=patch_id,
            principal_id=worker,
        ),
        "review": issue_receipt(
            kind="review",
            task_id=TASK_ID,
            patch_id=patch_id,
            principal_id=reviewer,
            reviewer_principal_id=reviewer,
            decision="accept",
        ),
        "test": issue_receipt(
            kind="test",
            task_id=TASK_ID,
            patch_id=patch_id,
            principal_id=VERIFIER,
            collected=1,
            passed=1,
            failed=0,
            skipped=0,
        ),
    }


def _proposal(
    loop: ExternalMergeLoop,
    *,
    reviewer: str = REVIEWER,
    worker: str = WORKER,
    principal: str = REVIEWER,
    files: dict[str, str] | None = None,
    receipts: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    patch = files if files is not None else _patch()
    patch_id = patch_identity(patch)
    return {
        "task_id": TASK_ID,
        "worker_principal_id": worker,
        "reviewer_principal_id": reviewer,
        "principal_id": principal,
        "base_tree_id": loop.tree_id,
        "files": patch,
        "receipts": receipts if receipts is not None else _receipts(patch_id=patch_id, reviewer=reviewer, worker=worker),
    }


def test_happy_path_merges_and_settles_queue() -> None:
    loop = _loop()
    before = loop.tree_id
    patch = _patch()
    receipt = loop.merge(_proposal(loop, files=patch))

    assert receipt.status == "merged"
    assert receipt.queue_status == "settled"
    assert receipt.queue_length == 0
    assert loop.queue_length == 0
    assert receipt.steps == MERGE_LOOP_STEPS
    assert receipt.worker_principal_id == WORKER
    assert receipt.reviewer_principal_id == REVIEWER
    assert receipt.reviewer_principal_id != receipt.worker_principal_id
    assert receipt.base_tree_id == before
    assert receipt.tree_id == loop.tree_id
    assert receipt.tree_id != before
    assert receipt.tree_id == canonical_tree_identity(loop.files)
    assert loop.files["owned.py"] == "print('accepted')\n"
    assert receipt.patch_id == patch_identity(patch)
    assert receipt.content_id.startswith("b")
    payload = dict(receipt.to_dict())
    restored = MergeLoopReceipt(
        status=payload["status"],
        task_id=payload["task_id"],
        worker_principal_id=payload["worker_principal_id"],
        reviewer_principal_id=payload["reviewer_principal_id"],
        principal_id=payload["principal_id"],
        base_tree_id=payload["base_tree_id"],
        tree_id=payload["tree_id"],
        patch_id=payload["patch_id"],
        queue_status=payload["queue_status"],
        queue_length=payload["queue_length"],
        steps=tuple(payload["steps"]),
    )
    assert restored.content_id == receipt.content_id
    clone_loop = ExternalMergeLoop(files={"README.md": "base\n"})
    clone = merge_accepted_result(_proposal(clone_loop, files=patch), loop=clone_loop)
    assert clone.tree_id == receipt.tree_id


def test_worker_self_merge_rejected() -> None:
    loop = _loop()
    before = loop.tree_id
    with pytest.raises(WorkerSelfMergeError, match="self-merge") as same_reviewer:
        loop.merge(_proposal(loop, reviewer=WORKER, principal=WORKER))
    assert same_reviewer.value.reason_code == "worker_self_merge"

    with pytest.raises(WorkerSelfMergeError, match="self-merge") as worker_caller:
        loop.merge(_proposal(loop, reviewer=REVIEWER, principal=WORKER))
    assert worker_caller.value.reason_code == "worker_self_merge"

    with pytest.raises(WorkerSelfMergeError, match="independent reviewer") as missing:
        loop.merge(_proposal(loop, reviewer="", principal=REVIEWER))
    assert missing.value.reason_code == "missing_reviewer"

    assert loop.tree_id == before
    assert loop.queue_length == 0
    assert dict(loop.files) == {"README.md": "base\n"}


def test_empty_queue_settle() -> None:
    loop = _loop()
    before = loop.tree_id
    receipt = loop.settle()

    assert receipt.status == "settled"
    assert receipt.queue_status == "settled"
    assert receipt.queue_length == 0
    assert receipt.steps == ("settle_queue",)
    assert receipt.tree_id == before
    assert receipt.tree_id == loop.tree_id
    assert loop.queue_length == 0
    assert receipt.task_id == ""
    assert receipt.patch_id == ""


def test_forged_receipt_rejected() -> None:
    loop = _loop()
    before = loop.tree_id
    patch = _patch()
    receipts = _receipts(patch_id=patch_identity(patch))
    forged = dict(receipts["review"])
    forged["content_id"] = issue_receipt(
        kind="review",
        task_id=TASK_ID,
        patch_id=patch_identity({"forged.py": "no\n"}),
        principal_id=REVIEWER,
        decision="accept",
    )["content_id"]
    receipts["review"] = forged

    with pytest.raises(ForgedReceiptError, match="forged receipt") as err:
        loop.merge(_proposal(loop, files=patch, receipts=receipts))
    assert err.value.reason_code == "forged_receipt"
    assert loop.tree_id == before
    assert loop.queue_length == 0
    assert "owned.py" not in loop.files


def test_missing_receipt_fails_closed() -> None:
    loop = _loop()
    before = loop.tree_id
    patch = _patch()
    receipts = _receipts(patch_id=patch_identity(patch))
    receipts.pop("review")

    with pytest.raises(MissingReceiptError, match="review") as err:
        loop.merge(_proposal(loop, files=patch, receipts=receipts))
    assert err.value.reason_code == "missing_receipt"
    assert loop.tree_id == before
    assert loop.queue_length == 0
    assert "owned.py" not in loop.files
