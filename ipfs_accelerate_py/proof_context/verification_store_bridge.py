"""Accelerator scheduling adapter for kit verification persistence (PCCE-013).

Accelerator does not keep a second production writer. Writes go through kit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ipfs_accelerate_py.proof_context.dependencies import DependencyUnavailable

AUTHORITY = "ipfs_kit_py.proof_context.verification_store"


class VerificationWriteForbidden(RuntimeError):
    """Raised if accelerator attempts a local production write."""


def open_kit_verification_store(root: str | Path) -> Any:
    try:
        from ipfs_kit_py.proof_context.verification_store import (
            open_verification_store,
        )
    except ImportError as exc:
        raise DependencyUnavailable(
            "kit v0.1 verification store is unavailable; "
            "accelerator must not write receipts locally"
        ) from exc
    return open_verification_store(root)


def kit_is_verification_authority() -> str:
    return AUTHORITY


def reject_local_production_writer() -> None:
    raise VerificationWriteForbidden(
        "accelerator is scheduler-only; production verification writes are kit-owned"
    )
