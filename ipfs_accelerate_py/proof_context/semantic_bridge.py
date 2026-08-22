"""Accelerator consumer of datasets-owned ContextPack authority (PCCE-012).

This module does not reconstruct packs. Token/budget accounting remains a
consumer concern in the legacy ContextPacker compatibility surface.
"""

from __future__ import annotations

from typing import Any

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    CONTEXT_PACK_V01_AUTHORITY,
)
from ipfs_accelerate_py.proof_context.dependencies import DependencyUnavailable


def build_v01_context_pack(**kwargs: Any) -> Any:
    try:
        from ipfs_datasets_py.proof_context.context_pack import build_context_pack
    except ImportError as exc:
        raise DependencyUnavailable(
            "datasets v0.1 ContextPack authority is unavailable; "
            "accelerator must not reconstruct packs"
        ) from exc
    return build_context_pack(**kwargs)


def datasets_is_v01_authority() -> str:
    return CONTEXT_PACK_V01_AUTHORITY


def compatibility_packer_is_not_authority() -> bool:
    from ipfs_accelerate_py.agent_supervisor.semantic_state.context_pack import (
        ContextPacker,
    )

    return ContextPacker.V01_PRODUCTION_AUTHORITY is False
