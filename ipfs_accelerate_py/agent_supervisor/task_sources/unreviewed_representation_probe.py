"""Measure the graph-review refusal that keeps representation learning unreviewed.

The review owner authenticates bytes and still refuses an accepted view.
No pilot README is present, no omission audit runs, and no model is trained.
A changed owner or a present pilot document returns None.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .dependency_block_resolution import (
    EXPECTED_UNREVIEWED_REPRESENTATION_PROBE,
)

_REVIEW_MARKERS = (
    "without admitting legal meaning",
    "does not establish a human's identity, competence, independence or the truth of a review",
    "cannot qualify an accepted view",
    "All semantic, corpus and Lean admission remains downstream.",
    '"accepted_view_eligible": False',
    "downstream_semantic_validation_required",
    "downstream_Lean_kernel_admission_required",
)
_AUTHORITY_KEYS = (
    "admitted",
    "formalized",
    "human_identity_verified",
    "human_independence_verified",
    "proof_authority",
    "signature_establishes_legal_truth",
    "source_semantics_verified",
    "training_qualified",
)


def measure_unreviewed_representation_probe() -> dict[str, object] | None:
    """Return the unreviewed-representation contract, or None on disagreement."""

    try:
        measured, digest = _measure()
    except Exception:
        return None
    if measured != EXPECTED_UNREVIEWED_REPRESENTATION_PROBE:
        return None
    if len(digest) != 64:
        return None
    return {**measured, "review_owner_sha256": digest}


def _measure() -> tuple[dict[str, object], str]:
    from ipfs_datasets_py.logic.autoformal import legal_graph_review as review

    path = Path(review.__file__).resolve()
    raw = path.read_bytes()
    text = " ".join(raw.decode("utf-8").split())
    readme = (
        path.parents[3]
        / "docs/autoencoders/pilots/uscode_definition_component/README.md"
    )
    if readme.is_file() or any(marker not in text for marker in _REVIEW_MARKERS):
        raise ValueError("graph review owner no longer refuses an accepted view")
    authority = review.FALSE_AUTHORITY
    if any(value is not False for value in authority.values()):
        raise ValueError("graph review authority flag is set")
    if any(authority.get(key) is not False for key in _AUTHORITY_KEYS):
        raise ValueError("graph review authority key missing")
    return dict(EXPECTED_UNREVIEWED_REPRESENTATION_PROBE), hashlib.sha256(raw).hexdigest()
