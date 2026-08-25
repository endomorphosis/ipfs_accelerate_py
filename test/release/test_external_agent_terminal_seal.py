"""EAAEF-175: worker self-seal and missing roots fail; independent sealer may seal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.seal_external_agent_release import (
    RECEIPT_PATH,
    SealError,
    seal_release,
    validate_seal_request,
)


SOURCE_ROOT = "sha256:" + "a" * 64
SEMANTIC_ROOT = "sha256:" + "b" * 64
CURSOR = "sha256:" + "c" * 64
SEALER = "sealer:independent-reviewer"


def _valid_request(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "source_root": SOURCE_ROOT,
        "semantic_root": SEMANTIC_ROOT,
        "tests": ["test/release/test_external_agent_terminal_seal.py"],
        "proofs": ["content-identity"],
        "claims": [],
        "merge_queue": [],
        "ducklake_cursor": CURSOR,
        "sealer_id": SEALER,
        "worker_principal_id": "worker:slot-0",
    }
    payload.update(changes)
    return payload


def test_worker_principal_cannot_self_seal() -> None:
    with pytest.raises(SealError, match="self-seal") as worker:
        validate_seal_request(_valid_request(sealer_id="worker:slot-0"))
    assert worker.value.reason_code == "worker_self_seal"
    with pytest.raises(SealError, match="self-seal") as digest:
        validate_seal_request(_valid_request(principal_id="sha256:" + "d" * 64, sealer_id=""))
    assert digest.value.reason_code == "worker_self_seal"


def test_missing_roots_fail_closed() -> None:
    with pytest.raises(SealError, match="source_root") as missing_source:
        validate_seal_request(_valid_request(source_root=""))
    assert missing_source.value.reason_code == "missing_field"
    with pytest.raises(SealError, match="semantic_root") as missing_semantic:
        validate_seal_request(_valid_request(semantic_root=""))
    assert missing_semantic.value.reason_code == "missing_field"
    with pytest.raises(SealError, match="tests") as missing_tests:
        validate_seal_request(_valid_request(tests=[]))
    assert missing_tests.value.reason_code == "missing_tests"
    with pytest.raises(SealError, match="proofs") as missing_proofs:
        validate_seal_request(_valid_request(proofs=[]))
    assert missing_proofs.value.reason_code == "missing_proofs"
    with pytest.raises(SealError, match="claims") as claims:
        validate_seal_request(_valid_request(claims=["open-claim"]))
    assert claims.value.reason_code == "claims_not_empty"
    with pytest.raises(SealError, match="merge queue") as queue:
        validate_seal_request(_valid_request(merge_queue=["merge-1"]))
    assert queue.value.reason_code == "merge_queue_not_empty"


def test_independent_sealer_writes_content_addressed_receipt() -> None:
    sealed = seal_release(_valid_request(), receipt_path=RECEIPT_PATH)
    assert sealed["sealer_id"] == SEALER
    assert sealed["worker_self_seal"] is False
    assert sealed["source_root"] == SOURCE_ROOT
    assert sealed["semantic_root"] == SEMANTIC_ROOT
    assert sealed["ducklake_cursor"] == CURSOR
    assert sealed["claims"] == []
    assert sealed["merge_queue"] == []
    assert sealed["content_id"].startswith("sha256:")
    assert sealed["terminal_report_id"] == sealed["content_id"]
    assert sealed["evidence_mode"] == "contract_fail_closed"
    assert sealed["live_eight_container_qualification"] is False
    saved = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert saved["content_id"] == sealed["content_id"]
    assert saved["sealer_id"] == SEALER
