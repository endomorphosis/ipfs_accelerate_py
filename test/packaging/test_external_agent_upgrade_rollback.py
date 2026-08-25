"""EAAEF-163: deployment doc covers migration, backup, DuckLake, rollback."""

from __future__ import annotations

from pathlib import Path

DOC = Path(__file__).resolve().parents[2] / "docs/deployment/EXTERNAL_AGENT_FABRIC.md"

REQUIRED = (
    "exclusive Quack owner",
    "later epoch",
    "DuckLake is history only",
    "immutable image digests",
    "Mutable tags are not authority",
)


def test_doc_covers_upgrade_and_rollback_gates() -> None:
    text = DOC.read_text(encoding="utf-8")
    for needle in REQUIRED:
        assert needle in text
