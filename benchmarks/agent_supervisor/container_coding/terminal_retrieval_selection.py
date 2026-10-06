"""Bind optional learned retrieval to the archive's explicit model revision."""
from __future__ import annotations

import re


def validate_model_revision(value: object) -> str:
    if type(value) is not str or (value and re.fullmatch(r"[0-9a-f]{40}", value) is None):
        raise ValueError("retrieval requires an exact pinned model revision")
    return value


def selected_retrieval_revision(manifest: dict, arm: str) -> str:
    """No-index never selects a model; full requires an explicit archive pin."""
    if arm not in {"full", "no-index"}:
        raise ValueError("unknown supervisor ablation")
    if type(manifest) is not dict:
        raise ValueError("runtime retrieval manifest required")
    if arm == "no-index":
        return ""
    requirements = manifest.get("learned_requirements", [])
    if type(requirements) is not list or any(type(item) is not str or not item for item in requirements):
        raise ValueError("runtime learned retrieval selection is malformed")
    revision = manifest.get("model_snapshot_revision")
    if not requirements:
        if revision is not None:
            raise ValueError("retrieval model revision lacks selected learned requirements")
        return ""
    revision = validate_model_revision(revision)
    if not revision:
        raise ValueError("runtime archive lacks its pinned retrieval model revision")
    return revision


def require_retrieval_revision(manifest: dict, arm: str, configured: object) -> str:
    """Reject a stale active pin before deployment or dispatch; never infer one.

    Older lexical/no-index configs carried an unused model_revision default.
    That field remains inert when the archive/arm selects no learned retrieval.
    """
    selected = selected_retrieval_revision(manifest, arm)
    if selected and validate_model_revision(configured) != selected:
        raise ValueError("configured retrieval model revision differs from runtime archive")
    return selected
