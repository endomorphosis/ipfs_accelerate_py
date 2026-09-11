"""Canonical migration coverage must preserve the train's actual read keys."""
import pytest
from ipfs_accelerate_py.agent_supervisor.merge.legacy_train_imports import (
    PUBLICATION_LEDGER_PATH as PATH,
    validate_train_import_coverage as validate,
)


def entry(path=PATH, key="distributed-publications", revision=1):
    return {"path": path, "receipt_key": key, "revision": revision}


@pytest.mark.parametrize("receipts", [[], [entry(key="archive")], [entry(), entry()],
    [entry(), entry("history.json", revision=2)],
    [entry(), entry("history.json", revision=1)]])
def test_ledger_cannot_disappear_or_be_rebound(receipts):
    with pytest.raises(ValueError, match="publication ledger"):
        validate(file_names=[PATH], receipt_imports=receipts, cursor_imports=[])


def test_current_ledger_with_explicit_older_history():
    validate(file_names=[PATH, "history.json"], receipt_imports=[
        entry("history.json"), entry(revision=2)], cursor_imports=[])


@pytest.mark.parametrize("path", ["train/receipts/acceptance-abc.json",
    "train/post-merge-recovery-cursors/abc.json"])
def test_existing_evidence_still_requires_import(path):
    with pytest.raises(ValueError, match="lacks explicit import"):
        validate(file_names=[path], receipt_imports=[], cursor_imports=[])


def test_no_inference_for_arbitrary_archived_files_or_receipt_keys():
    validate(file_names=["notes.json", "train/receipts/lossy-name.json"],
             receipt_imports=[entry("train/receipts/lossy-name.json", "explicit:key")],
             cursor_imports=[])
