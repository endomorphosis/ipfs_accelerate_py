"""EAAEF-161: image plans require digests, SBOMs, signatures; no tag authority."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.release.build_external_agent_images import ImagePlanError, plan_images


def test_plan_requires_tree_digest_and_forbids_tag_authority() -> None:
    plan = plan_images(
        source_tree="sha256:" + "a" * 64,
        architectures=("amd64", "arm64"),
        mutable_tags_as_authority=False,
    )
    assert plan["requires_sbom"] is True
    assert plan["requires_signature"] is True
    assert plan["roles"] == ["supervisor", "python-worker", "prover"]
    with pytest.raises(ImagePlanError, match="tags"):
        plan_images(
            source_tree="sha256:" + "a" * 64,
            architectures=("amd64",),
            mutable_tags_as_authority=True,
        )
