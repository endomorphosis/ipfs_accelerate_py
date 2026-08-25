"""EAAEF-052: image sources exist; mutable tags are not authority."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = (
    ROOT / "containers/external-agent/supervisor.Containerfile",
    ROOT / "containers/external-agent/python-worker.Containerfile",
    ROOT / "containers/external-agent/prover.Containerfile",
)


def test_containerfiles_exist_and_forbid_tag_authority() -> None:
    for path in FILES:
        text = path.read_text(encoding="utf-8")
        assert "USER 65532:65532" in text
        assert "tags are not authority" in text.lower() or "mutable tags" in text.lower() or "digest" in text.lower()
