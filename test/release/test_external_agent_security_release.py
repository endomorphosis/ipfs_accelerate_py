"""EAAEF-173: security review document covers required boundaries."""

from __future__ import annotations

from pathlib import Path

REVIEW = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "security"
    / "EXTERNAL_AGENT_FABRIC_REVIEW.md"
)

REQUIRED = (
    "workers cannot self-approve",
    "Docker socket",
    "Quack owner",
    "DuckLake",
    "1.5.5",
    "no-go",
)


def test_review_covers_fail_closed_boundaries() -> None:
    text = REVIEW.read_text(encoding="utf-8")
    for needle in REQUIRED:
        assert needle in text
