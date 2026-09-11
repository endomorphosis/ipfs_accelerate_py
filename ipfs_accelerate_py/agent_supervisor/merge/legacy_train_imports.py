"""Coverage checks for explicitly mapped, preserved legacy train evidence.

This validates a migration input, not capture coherence, consumer closure,
source admission or semantic acceptance. It never reads files or infers keys
from lossy legacy receipt filenames.
"""
from collections.abc import Iterable, Mapping
from typing import Any

PUBLICATION_LEDGER_PATH = "train/distributed-publications.json"
PUBLICATION_LEDGER_KEY = "distributed-publications"


def validate_train_import_coverage(
    *,
    file_names: Iterable[str],
    receipt_imports: Iterable[Mapping[str, Any]],
    cursor_imports: Iterable[Mapping[str, Any]],
) -> None:
    """Require explicit imports for canonical evidence in a validated manifest.

    Callers validate their closed manifest and verify all bytes, content IDs,
    contiguous revisions and scope bindings separately. Copying bytes into an
    archive does not make them available to an owner-backed train.
    """
    names = set(file_names)
    receipts = tuple(receipt_imports)
    cursors = tuple(cursor_imports)
    if PUBLICATION_LEDGER_PATH in names:
        ledger = [r for r in receipts if r["path"] == PUBLICATION_LEDGER_PATH]
        if len(ledger) != 1 or ledger[0]["receipt_key"] != PUBLICATION_LEDGER_KEY:
            raise ValueError(
                "preserved publication ledger requires one explicit canonical import"
            )
        if any(
            type(r["revision"]) is not int or r["revision"] < 1
            for r in receipts if r["receipt_key"] == PUBLICATION_LEDGER_KEY
        ):
            raise ValueError("preserved publication ledger revision is invalid")
        # The canonical legacy file is the current head. An independently
        # supplied older history may precede it, but cannot replace that head.
        if any(
            r["receipt_key"] == PUBLICATION_LEDGER_KEY
            and r is not ledger[0]
            and r["revision"] >= ledger[0]["revision"]
            for r in receipts
        ):
            raise ValueError("preserved publication ledger must remain the receipt head")
    required_cursors = {
        name for name in names
        if name.startswith("train/post-merge-recovery-cursors/")
        and name.endswith(".json")
    }
    required_receipts = {
        name for name in names
        if name.startswith("train/receipts/") and name.endswith(".json")
    }
    if not required_cursors.issubset({r["path"] for r in cursors}) or not required_receipts.issubset(
        {r["path"] for r in receipts}
    ):
        raise ValueError("preserved canonical cursor or receipt lacks explicit import")
