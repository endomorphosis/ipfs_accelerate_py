"""Proof-cache and ZKP certificates live in DuckDB and project to DuckLake."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.proof_certificate_database import (
    ProofCertificateDatabase,
    ProofCertificateDatabaseError,
    persist_proof_certificate_record,
    records_for_decision,
)


def test_refuses_extra_gate_control_duckdb(tmp_path) -> None:
    store = ProofCertificateDatabase(tmp_path / "control.duckdb")
    with pytest.raises(ProofCertificateDatabaseError, match="control.duckdb"):
        store.persist({"kind": "proof_cache", "key_id": "k", "receipt_id": "r"})


def test_persist_proof_cache_and_query_for_decision(tmp_path) -> None:
    store = ProofCertificateDatabase(tmp_path / "proof_certificates.duckdb")
    stored = store.persist(
        {
            "kind": "proof_cache",
            "key_id": "proof-cache-key:sha256:abc",
            "receipt_id": "receipt:1",
            "verdict": "proved",
            "proposal_only": True,
            "completion_authority": False,
        }
    )
    assert stored["stored"] is True
    assert stored["completion_authority"] is False
    query = store.records_for_decision(kind="proof_cache", key_id="proof-cache-key:sha256:abc")
    assert query["n"] == 1
    assert query["records"][0]["kind"] == "proof_cache"
    assert query["completion_authority"] is False
    assert query["decision_authority"] is False
    assert query["ducklake_authoritative"] is False


def test_persist_zkp_certificate(tmp_path) -> None:
    store = ProofCertificateDatabase(tmp_path / "proof_certificates.duckdb")
    stored = store.persist(
        {
            "kind": "zkp_certificate",
            "key_id": "key:1",
            "receipt_id": "receipt:1",
            "circuit_id": "circuit:trace",
            "public_input_digest": "sha256:pub",
            "simulated": False,
            "completion_authority": False,
        }
    )
    assert stored["stored"] is True
    query = store.records_for_decision(kind="zkp_certificate", receipt_id="receipt:1")
    assert query["n"] == 1
    assert query["records"][0]["payload"]["circuit_id"] == "circuit:trace"
    assert query["decision_authority"] is False


def test_cannot_persist_witness_or_admitted_completion(tmp_path) -> None:
    store = ProofCertificateDatabase(tmp_path / "proof_certificates.duckdb")
    with pytest.raises(ProofCertificateDatabaseError, match="admitted"):
        store.persist(
            {"kind": "proof_cache", "key_id": "k", "receipt_id": "r", "completion_authority": True}
        )
    with pytest.raises(ProofCertificateDatabaseError, match="witness"):
        store.persist(
            {
                "kind": "zkp_certificate",
                "key_id": "k",
                "receipt_id": "r",
                "witness": {"secret": "no"},
            }
        )


def test_unconfigured_store_is_skip(monkeypatch) -> None:
    monkeypatch.delenv("IPFS_ACCELERATE_PROOF_CERTIFICATE_DUCKDB", raising=False)
    skipped = persist_proof_certificate_record(
        {"kind": "proof_cache", "key_id": "k", "receipt_id": "r"}
    )
    assert skipped["status"] == "skip"
    empty = records_for_decision(kind="proof_cache")
    assert empty["n"] == 0
    assert empty["decision_authority"] is False


def test_ducklake_projection_is_observational(tmp_path) -> None:
    store = ProofCertificateDatabase(
        tmp_path / "proof_certificates.duckdb",
        ducklake_root=tmp_path / "proof_certificate_ducklake",
    )
    store.persist(
        {
            "kind": "zkp_certificate",
            "key_id": "key:lake",
            "receipt_id": "receipt:lake",
            "completion_authority": False,
        }
    )
    projected = store.project_ducklake()
    assert projected["completion_authority"] is False
    assert projected["authoritative"] is False
    assert projected["status"] in {"projected", "unavailable"}
    if projected["status"] == "projected":
        assert projected["stored_records"] >= 1
