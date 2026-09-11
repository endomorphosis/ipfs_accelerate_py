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
             receipt_imports=[entry("train/receipts/lossy-name.json", "lossy-:name")],
             cursor_imports=[])


@pytest.mark.parametrize("receipts", [
    [entry("train/receipts/callback.json", "unrelated")],
    [entry("train/receipts/callback.json", "callback"),
     entry("train/receipts/callback.json", "call:back")],
    [entry("train/receipts/callback.json", "callback"),
     entry("history.json", "callback", 2)],
    [entry("train/receipts/callback.json", "callback"),
     entry("history.json", "callback", 1)],
])
def test_canonical_callback_receipt_retains_exact_key_and_head(receipts):
    with pytest.raises(ValueError, match="canonical train receipt"):
        validate(file_names=["train/receipts/callback.json", "history.json"],
                 receipt_imports=receipts, cursor_imports=[])


def test_canonical_callback_receipt_allows_explicit_older_history():
    validate(file_names=["train/receipts/callback.json", "history.json"],
             receipt_imports=[entry("history.json", "callback"),
                              entry("train/receipts/callback.json", "callback", 2)],
             cursor_imports=[])


@pytest.mark.parametrize("key", ["acceptance:abc", "évidence-ß", "k" * 190])
def test_forward_path_matches_actual_legacy_train_writer(key, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.merge.merge_train import MergeTrain
    train = object.__new__(MergeTrain)
    train.receipt_dir = tmp_path / "train/receipts"
    path = train._receipt_path(key).relative_to(tmp_path).as_posix()
    validate(file_names=[path], receipt_imports=[entry(path, key)], cursor_imports=[])


@pytest.mark.parametrize("revision", [True, 0, -1, "2"])
def test_canonical_callback_revision_is_positive_integer(revision):
    with pytest.raises(ValueError, match="canonical train receipt revision"):
        validate(file_names=["train/receipts/callback.json"],
                 receipt_imports=[entry("train/receipts/callback.json", "callback", revision)],
                 cursor_imports=[])
