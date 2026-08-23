"""Digest-pinned supervisor/worker/prover image plan (EAAEF-161)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final


IMAGE_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-agent-image-plan@1"
)
ROLES: Final[tuple[str, ...]] = ("supervisor", "python-worker", "prover")


class ImagePlanError(ValueError):
    """Image plan is not releasable."""


def plan_images(
    *,
    source_tree: str,
    architectures: Sequence[str],
    mutable_tags_as_authority: bool,
) -> Mapping[str, Any]:
    if not str(source_tree).startswith("sha256:"):
        raise ImagePlanError("source_tree must be sha256:...")
    if mutable_tags_as_authority:
        raise ImagePlanError("mutable tags are not authority")
    if not architectures:
        raise ImagePlanError("architectures are required")
    return MappingProxyType(
        {
            "schema": IMAGE_PLAN_SCHEMA,
            "source_tree": source_tree,
            "roles": list(ROLES),
            "architectures": list(architectures),
            "requires_sbom": True,
            "requires_signature": True,
            "mutable_tags_as_authority": False,
        }
    )
