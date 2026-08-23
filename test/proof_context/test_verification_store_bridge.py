"""PCCE-013: accelerator schedules; kit writes verification receipts."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.proof_context.dependencies import DependencyUnavailable
from ipfs_accelerate_py.proof_context.verification_store_bridge import (
    AUTHORITY,
    VerificationWriteForbidden,
    kit_is_verification_authority,
    open_kit_verification_store,
    reject_local_production_writer,
)


def test_authority_is_kit() -> None:
    assert kit_is_verification_authority() == AUTHORITY
    with pytest.raises(VerificationWriteForbidden):
        reject_local_production_writer()


def test_open_delegates_or_fails_closed(tmp_path: Path) -> None:
    try:
        store = open_kit_verification_store(tmp_path)
    except DependencyUnavailable:
        return
    payload = b'{"kind":"proof_receipt","via":"bridge"}'
    ref = store.put_verification_receipt(payload)
    assert store.get_verification_receipt(ref) == payload
