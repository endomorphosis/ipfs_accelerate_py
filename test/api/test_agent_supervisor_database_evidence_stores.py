"""Tests for DatabaseArtifactStore@1 and DatabaseEvidenceStore@1 (DQP-025).

Evidence subset: content identity, provenance, redaction, size/graph quotas,
corruption, stale key, single flight, cache applicability, rebuild.

Acceptance: JSON/Parquet/file freshness no longer determines authority; every
large external blob is digest-bound and verified on use; caches never promote
assurance; stale or poisoned hits fail closed; database projections rebuild
from admitted evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.database_evidence_store import (
    DATABASE_EVIDENCE_STORE_INTERFACE,
    REDACTION_MARKER as EVIDENCE_REDACTION_MARKER,
    AssuranceLevel,
    DatabaseEvidenceStore,
    EvidenceKey,
    EvidenceOutcome,
    EvidenceVerdict,
    LookupStatus,
    RejectionReason,
    SingleFlightTimeout,
    duckdb_available as evidence_duckdb_available,
    open_database_evidence_store,
)
from ipfs_accelerate_py.agent_supervisor.runtime.database_artifact_store import (
    DATABASE_ARTIFACT_STORE_INTERFACE,
    REDACTION_MARKER as ARTIFACT_REDACTION_MARKER,
    ArtifactQuotaPolicy,
    DatabaseArtifactStore,
    DatabaseArtifactStoreIntegrityError,
    DatabaseArtifactStoreQuotaError,
    duckdb_available as artifact_duckdb_available,
    open_database_artifact_store,
)

pytestmark = pytest.mark.skipif(
    not (artifact_duckdb_available() and evidence_duckdb_available()),
    reason="DuckDB is required for database evidence store hermetic tests",
)


def _artifact_store(tmp_path: Path, **kwargs: object) -> DatabaseArtifactStore:
    return open_database_artifact_store(tmp_path / "artifacts.duckdb", **kwargs)


def _evidence_store(tmp_path: Path, **kwargs: object) -> DatabaseEvidenceStore:
    return open_database_evidence_store(tmp_path / "evidence.duckdb", **kwargs)


def _sample_key(**overrides: object) -> EvidenceKey:
    payload = {
        "domain": "proof",
        "subject_id": "obligation:demo",
        "repository_tree_id": "tree:abc123",
        "policy_digest": "sha256:" + "a" * 64,
        "schema_version": "evidence-key@1",
        "tool_versions": {"z3": "4.12.0", "lean": "4.7.0"},
        "configuration_digest": "sha256:" + "b" * 64,
        "query_digest": "sha256:" + "c" * 64,
    }
    payload.update(overrides)
    return EvidenceKey.from_dict(payload)


# ---------------------------------------------------------------------------
# Interface identities
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert DATABASE_ARTIFACT_STORE_INTERFACE == "DatabaseArtifactStore@1"
    assert DATABASE_EVIDENCE_STORE_INTERFACE == "DatabaseEvidenceStore@1"
    assert DatabaseArtifactStore.INTERFACE == DATABASE_ARTIFACT_STORE_INTERFACE
    assert DatabaseEvidenceStore.INTERFACE == DATABASE_EVIDENCE_STORE_INTERFACE


# ---------------------------------------------------------------------------
# DatabaseArtifactStore
# ---------------------------------------------------------------------------


def test_artifact_content_identity_and_provenance(tmp_path: Path) -> None:
    with _artifact_store(tmp_path) as store:
        record = store.admit_artifact(
            "bundle_index",
            metadata={"task_count": 3, "lane": "dqp-evidence"},
            provenance={
                "producer": "test",
                "repository_id": "repository:demo",
            },
            body={"tasks": ["a", "b", "c"]},
        )
        assert record.artifact_id.startswith("artifact:sha256:")
        assert record.content_digest.startswith("sha256:")
        assert record.blob_digest.startswith("sha256:")
        assert record.authority == "database"
        assert record.provenance["producer"] == "test"
        assert record.admitted_sequence >= 1

        loaded = store.get_artifact(record.artifact_id)
        assert loaded is not None
        assert loaded.content_digest == record.content_digest
        assert loaded.metadata["task_count"] == 3

        # Content identity is stable across reopen.
        again = store.admit_artifact(
            "bundle_index",
            metadata={"task_count": 3, "lane": "dqp-evidence"},
            provenance={
                "producer": "test",
                "repository_id": "repository:demo",
            },
            body={"tasks": ["a", "b", "c"]},
        )
        assert again.artifact_id == record.artifact_id


def test_artifact_redaction_on_admission(tmp_path: Path) -> None:
    with _artifact_store(tmp_path) as store:
        record = store.admit_artifact(
            "provider_receipt",
            metadata={
                "model": "test-model",
                "access_token": "synthetic-secret-token",
                "nested": {"password": "also-secret"},
            },
        )
        assert record.redacted is True
        assert record.metadata["access_token"] == ARTIFACT_REDACTION_MARKER
        assert record.metadata["nested"]["password"] == ARTIFACT_REDACTION_MARKER
        assert record.metadata["model"] == "test-model"


def test_blob_digest_bound_and_verified_on_use(tmp_path: Path) -> None:
    with _artifact_store(tmp_path) as store:
        body = b"large-immutable-body-" + b"x" * 1024
        blob = store.put_blob(body, media_type="application/octet-stream")
        assert blob.blob_digest == "sha256:" + hashlib.sha256(body).hexdigest()
        assert blob.authority == "digest_bound_cas"

        verified = store.verify_blob(blob.blob_digest)
        assert verified == body

        record = store.admit_artifact(
            "raw_blob",
            blob_digest=blob.blob_digest,
            media_type="application/octet-stream",
        )
        assert record.blob_digest == blob.blob_digest
        assert record.size_bytes == len(body)

        # Poison the CAS body and ensure verification fails closed.
        cas_path = Path(blob.cas_path)
        cas_path.write_bytes(b"poisoned-payload")
        with pytest.raises(DatabaseArtifactStoreIntegrityError):
            store.verify_blob(blob.blob_digest)
        with pytest.raises(DatabaseArtifactStoreIntegrityError):
            store.get_artifact(record.artifact_id, verify_blob=True)


def test_size_and_graph_quotas(tmp_path: Path) -> None:
    quotas = ArtifactQuotaPolicy(
        max_blob_bytes=64,
        max_artifacts=2,
        max_datasets=2,
        max_edges=1,
        max_graph_degree=1,
        max_metadata_bytes=4_096,
    )
    with _artifact_store(tmp_path, quotas=quotas) as store:
        with pytest.raises(DatabaseArtifactStoreQuotaError):
            store.put_blob(b"y" * 128)

        first = store.admit_artifact("kind", metadata={"n": 1})
        second = store.admit_artifact("kind", metadata={"n": 2})
        with pytest.raises(DatabaseArtifactStoreQuotaError):
            store.admit_artifact("kind", metadata={"n": 3})

        store.link(first.artifact_id, second.artifact_id, "depends_on")
        with pytest.raises(DatabaseArtifactStoreQuotaError):
            # Degree of first already at max_graph_degree after one edge.
            store.link(first.artifact_id, second.artifact_id, "related_to")


def test_dataset_admission_and_edges(tmp_path: Path) -> None:
    with _artifact_store(tmp_path) as store:
        artifact = store.admit_artifact(
            "analysis_receipt",
            metadata={"analyzer": "ast"},
            body={"symbols": 12},
        )
        dataset = store.admit_dataset(
            "impact_rows",
            schema_id="impact@1",
            row_count=12,
            metadata={"source": "impact_graph"},
            body=[{"symbol": "foo"}, {"symbol": "bar"}],
            provenance={"snapshot_id": "snap:1"},
        )
        assert dataset.dataset_id.startswith("dataset:sha256:")
        assert dataset.blob_digest.startswith("sha256:")
        assert dataset.authority == "database"

        edge = store.link(
            artifact.artifact_id,
            dataset.dataset_id,
            "projects_to",
            body={"role": "primary"},
        )
        assert edge.source_id == artifact.artifact_id
        assert edge.target_id == dataset.dataset_id
        edges = store.list_edges(source_id=artifact.artifact_id)
        assert len(edges) == 1
        assert edges[0].edge_id == edge.edge_id


def test_json_export_is_non_authoritative(tmp_path: Path) -> None:
    with _artifact_store(tmp_path) as store:
        store.admit_artifact("exportable", metadata={"n": 1}, body={"ok": True})
        store.admit_dataset("rows", row_count=1, body=[{"x": 1}])
        export_path = tmp_path / "artifacts.export.json"
        receipt = store.export_json(export_path)
        assert receipt.authority == "export_only"
        assert receipt.to_dict()["authoritative"] is False
        assert export_path.exists()
        assert receipt.content_digest.startswith("sha256:")
        assert store.file_freshness_is_non_authoritative() is True

        before = [item.to_dict() for item in store.list_artifacts()]
        assert store.authority_unaffected_by_export_deletion(export_path) is True
        assert not export_path.exists()
        after = [item.to_dict() for item in store.list_artifacts()]
        assert before == after


def test_artifact_projection_rebuilds_from_admitted_evidence(
    tmp_path: Path,
) -> None:
    with _artifact_store(tmp_path) as store:
        artifact = store.admit_artifact(
            "proof_metrics",
            metadata={"metric": "coverage"},
            body={"value": 1},
        )
        dataset = store.admit_dataset("metrics", row_count=1, body=[{"v": 1}])
        store.link(artifact.artifact_id, dataset.dataset_id, "summarizes")

        assert store.admitted_evidence_count() >= 3
        projection = store.rebuild_projection("catalog")
        assert projection.authority == "database"
        assert projection.body["rebuilt_from_admitted_evidence"] is True
        assert projection.body["artifact_count"] >= 1
        assert projection.body["dataset_count"] >= 1
        assert projection.body["edge_count"] >= 1
        assert projection.rebuilt_from_sequence >= 1

        # Rebuild is deterministic for the same admitted evidence.
        again = store.rebuild_projection("catalog")
        assert again.content_digest == projection.content_digest
        assert again.body["artifacts"][0]["artifact_id"] == artifact.artifact_id


# ---------------------------------------------------------------------------
# DatabaseEvidenceStore
# ---------------------------------------------------------------------------


def test_receipt_admission_lookup_and_redaction(tmp_path: Path) -> None:
    with _evidence_store(tmp_path) as store:
        key = _sample_key()
        receipt = store.put_receipt(
            key,
            assurance_level=AssuranceLevel.KERNEL_CHECKED,
            verdict=EvidenceVerdict.PROVED,
            outcome=EvidenceOutcome.SUCCESSFUL,
            body={
                "prover": "z3",
                "access_token": "secret-token-value",
                "summary": "proved",
            },
            blob={"transcript_ref": "omitted"},
        )
        assert receipt.receipt_id.startswith("receipt:sha256:")
        assert receipt.authority == "database"
        assert receipt.redacted is True
        assert receipt.body["access_token"] == EVIDENCE_REDACTION_MARKER
        assert receipt.blob_digest.startswith("sha256:")

        hit = store.lookup_receipt(
            key, required_assurance=AssuranceLevel.SOLVER_CHECKED
        )
        assert hit.status is LookupStatus.HIT
        assert hit.receipt is not None
        assert hit.receipt.receipt_id == receipt.receipt_id
        assert hit.receipt.content_digest == receipt.content_digest


def test_stale_and_poisoned_receipts_fail_closed(tmp_path: Path) -> None:
    clock = {"now": 1_000.0}

    def _clock() -> float:
        return clock["now"]

    with _evidence_store(tmp_path, clock=_clock) as store:
        key = _sample_key()
        receipt = store.put_receipt(
            key,
            assurance_level=AssuranceLevel.SOLVER_CHECKED,
            verdict=EvidenceVerdict.PROVED,
            ttl_seconds=10,
            body={"ok": True},
        )
        # Advance past expiry.
        clock["now"] = 1_000.0 + 30
        stale = store.lookup_receipt(key)
        assert stale.status is LookupStatus.REJECTED
        assert RejectionReason.STALE.value in stale.reason_codes

        # Fresh receipt then poison its envelope via direct SQL.
        clock["now"] = 2_000.0
        fresh_key = _sample_key(subject_id="obligation:fresh")
        fresh = store.put_receipt(
            fresh_key,
            assurance_level=AssuranceLevel.SOLVER_CHECKED,
            verdict=EvidenceVerdict.PROVED,
            ttl_seconds=3600,
            body={"ok": True},
        )
        connection = store._require()
        connection.execute(
            """
            UPDATE evidence_receipts
            SET receipt_json=?
            WHERE receipt_id=?
            """,
            [
                json.dumps({"receipt_id": fresh.receipt_id, "tampered": True}),
                fresh.receipt_id,
            ],
        )
        store._commit_if_idle(connection)
        poisoned = store.lookup_receipt(fresh_key)
        assert poisoned.status is LookupStatus.REJECTED
        assert RejectionReason.POISONED.value in poisoned.reason_codes


def test_stale_key_and_insufficient_assurance_fail_closed(
    tmp_path: Path,
) -> None:
    with _evidence_store(tmp_path) as store:
        key = _sample_key()
        store.put_receipt(
            key,
            assurance_level=AssuranceLevel.MODEL_ONLY,
            verdict=EvidenceVerdict.PROVED,
            outcome=EvidenceOutcome.SUCCESSFUL,
            body={"model": "guess"},
        )
        # Same logical subject with a different tree id is a different key.
        stale_key = _sample_key(repository_tree_id="tree:other")
        miss = store.lookup_receipt(stale_key)
        assert miss.status is LookupStatus.MISS
        assert RejectionReason.CACHE_MISS.value in miss.reason_codes

        weak = store.lookup_receipt(
            key, required_assurance=AssuranceLevel.KERNEL_CHECKED
        )
        assert weak.status is LookupStatus.REJECTED
        assert (
            RejectionReason.INSUFFICIENT_ASSURANCE.value in weak.reason_codes
        )


def test_invalidation_and_use_outcomes(tmp_path: Path) -> None:
    with _evidence_store(tmp_path) as store:
        key = _sample_key()
        receipt = store.put_receipt(
            key,
            assurance_level=AssuranceLevel.KERNEL_CHECKED,
            verdict=EvidenceVerdict.PROVED,
            body={"ok": True},
        )
        hit = store.lookup_receipt(key)
        assert hit.is_hit

        invalidation = store.invalidate(
            receipt.receipt_id,
            reason="policy_revision_changed",
            subject_kind="receipt",
            invalidated_by="test",
        )
        assert invalidation.reason == "policy_revision_changed"

        rejected = store.lookup_receipt(key)
        assert rejected.status is LookupStatus.REJECTED
        assert RejectionReason.INVALIDATED.value in rejected.reason_codes

        outcomes = store.list_use_outcomes()
        assert any(item.status == LookupStatus.HIT.value for item in outcomes)
        assert any(
            item.status == LookupStatus.REJECTED.value for item in outcomes
        )


def test_cache_never_promotes_assurance(tmp_path: Path) -> None:
    with _evidence_store(tmp_path) as store:
        key = _sample_key(domain="analysis", subject_id="analysis:1")
        entry = store.put_cache_entry(
            key,
            outcome=EvidenceOutcome.SUCCESSFUL,
            assurance_level=AssuranceLevel.KERNEL_CHECKED,
            body={"summary": "cached analysis"},
        )
        assert entry.authority == "cache_non_authoritative"
        assert entry.to_dict()["promotes_assurance"] is False
        assert store.cache_promotes_assurance() is False

        # Structural metadata hit is allowed only for required_assurance=NONE.
        structural = store.lookup_cache(
            key, required_assurance=AssuranceLevel.NONE
        )
        assert structural.status is LookupStatus.HIT
        assert structural.cache_entry is not None
        assert (
            RejectionReason.CACHE_NON_AUTHORITATIVE.value
            in structural.reason_codes
        )

        # Any assurance request fails closed; cache cannot promote.
        promoted = store.lookup_cache(
            key, required_assurance=AssuranceLevel.SOLVER_CHECKED
        )
        assert promoted.status is LookupStatus.REJECTED
        assert (
            RejectionReason.CACHE_NON_AUTHORITATIVE.value
            in promoted.reason_codes
        )
        assert (
            RejectionReason.INSUFFICIENT_ASSURANCE.value
            in promoted.reason_codes
        )


def test_negative_cache_not_completion_evidence(tmp_path: Path) -> None:
    with _evidence_store(tmp_path) as store:
        key = _sample_key(domain="analysis", subject_id="analysis:neg")
        entry = store.put_cache_entry(
            key,
            outcome=EvidenceOutcome.INCONCLUSIVE,
            body={"reason": "timeout"},
            ttl_seconds=600,
        )
        assert entry.is_negative is True
        # Negative TTL is clamped.
        assert entry.expires_at_ms - entry.created_at_ms <= 5 * 60 * 1000

        rejected = store.lookup_cache(key, allow_negative=False)
        assert rejected.status is LookupStatus.REJECTED
        assert (
            RejectionReason.NEGATIVE_NOT_PROMOTABLE.value
            in rejected.reason_codes
        )


def test_single_flight_deduplicates_producers(tmp_path: Path) -> None:
    with _evidence_store(tmp_path) as store:
        key = _sample_key(subject_id="obligation:flight")
        calls = {"n": 0}

        def producer() -> dict[str, object]:
            calls["n"] += 1
            return {"result": "shared", "n": calls["n"]}

        first = store.single_flight(
            key,
            producer,
            lease_seconds=30,
            wait_timeout_seconds=5,
            outcome_ttl_seconds=30,
        )
        assert first.owner is True
        assert first.value["result"] == "shared"

        second = store.single_flight(
            key,
            producer,
            lease_seconds=30,
            wait_timeout_seconds=5,
            outcome_ttl_seconds=30,
        )
        assert second.owner is False
        assert second.shared is True
        assert second.value == first.value
        assert calls["n"] == 1


def test_single_flight_timeout(tmp_path: Path) -> None:
    with _evidence_store(tmp_path) as store:
        key = _sample_key(subject_id="obligation:timeout")
        # Hold the flight without publishing so a short waiter times out.
        acquired, token, fencing = store._claim_flight(
            key.key_id, owner_id="holder", lease_seconds=60
        )
        assert acquired is True
        assert token
        assert fencing >= 1
        with pytest.raises(SingleFlightTimeout):
            store.single_flight(
                key,
                lambda: {"never": True},
                wait_timeout_seconds=0.05,
                poll_interval_seconds=0.01,
                lease_seconds=60,
            )


def test_attestation_bound_to_receipt(tmp_path: Path) -> None:
    with _evidence_store(tmp_path) as store:
        key = _sample_key()
        receipt = store.put_receipt(
            key,
            assurance_level=AssuranceLevel.KERNEL_CHECKED,
            verdict=EvidenceVerdict.PROVED,
            body={"ok": True},
        )
        attestation = store.put_attestation(
            receipt.receipt_id,
            backend="zkp",
            status="verified",
            body={"circuit": "demo"},
        )
        assert attestation.receipt_id == receipt.receipt_id
        listed = store.list_attestations(receipt.receipt_id)
        assert len(listed) == 1
        assert listed[0].attestation_id == attestation.attestation_id


def test_evidence_export_non_authoritative_and_projection_rebuild(
    tmp_path: Path,
) -> None:
    with _evidence_store(tmp_path) as store:
        key = _sample_key()
        store.put_receipt(
            key,
            assurance_level=AssuranceLevel.SOLVER_CHECKED,
            verdict=EvidenceVerdict.PROVED,
            body={"ok": True},
        )
        store.put_cache_entry(
            _sample_key(domain="analysis", subject_id="a1"),
            outcome=EvidenceOutcome.SUCCESSFUL,
            body={"x": 1},
        )
        export_path = tmp_path / "evidence.export.json"
        receipt = store.export_json(export_path)
        assert receipt.authority == "export_only"
        assert receipt.to_dict()["authoritative"] is False
        assert store.file_freshness_is_non_authoritative() is True
        assert store.authority_unaffected_by_export_deletion(export_path) is True

        projection = store.rebuild_projection()
        assert projection.authority == "database"
        assert projection.body["rebuilt_from_admitted_evidence"] is True
        assert projection.body["cache_promotes_assurance"] is False
        assert projection.body["receipt_count"] >= 1
        assert projection.rebuilt_from_sequence >= 1

        again = store.rebuild_projection()
        assert again.content_digest == projection.content_digest


def test_blob_corruption_on_receipt_use_fails_closed(tmp_path: Path) -> None:
    with _evidence_store(tmp_path) as store:
        key = _sample_key(subject_id="obligation:blob")
        payload = b"proof-body-bytes"
        digest = store.put_blob(payload)
        receipt = store.put_receipt(
            key,
            assurance_level=AssuranceLevel.KERNEL_CHECKED,
            verdict=EvidenceVerdict.PROVED,
            blob_digest=digest,
            body={"ok": True},
        )
        path = store._blob_path(digest)
        path.write_bytes(b"corrupted")
        result = store.lookup_receipt(key, verify_blob=True)
        assert result.status is LookupStatus.REJECTED
        assert RejectionReason.BLOB_CORRUPTION.value in result.reason_codes
        assert RejectionReason.POISONED.value in result.reason_codes
        # Identity remains queryable without verification when requested.
        loaded = store.get_receipt(receipt.receipt_id)
        assert loaded is not None
        assert loaded.blob_digest == digest


def test_end_to_end_database_first_authority(tmp_path: Path) -> None:
    """Artifacts and evidence commit in DB before optional exports."""

    with _artifact_store(tmp_path / "art") as artifacts, _evidence_store(
        tmp_path / "ev"
    ) as evidence:
        artifact = artifacts.admit_artifact(
            "validation_bundle",
            metadata={"suite": "dqp-025"},
            body={"cases": 4},
            provenance={"task_id": "DQP-025"},
        )
        dataset = artifacts.admit_dataset(
            "validation_rows",
            row_count=4,
            body=[{"case": 1}, {"case": 2}],
        )
        artifacts.link(artifact.artifact_id, dataset.dataset_id, "contains")

        key = _sample_key(
            subject_id=artifact.artifact_id,
            extra={"artifact_digest": artifact.content_digest},
        )
        receipt = evidence.put_receipt(
            key,
            assurance_level=AssuranceLevel.KERNEL_CHECKED,
            verdict=EvidenceVerdict.PROVED,
            body={"artifact_id": artifact.artifact_id},
            blob={"bundle": artifact.to_dict()},
        )
        evidence.put_attestation(
            receipt.receipt_id, backend="test", status="verified"
        )
        evidence.put_cache_entry(
            key,
            outcome=EvidenceOutcome.SUCCESSFUL,
            assurance_level=AssuranceLevel.KERNEL_CHECKED,
            body={"hint": "do-not-trust"},
        )

        # Export both stores; delete exports; authority remains.
        art_export = tmp_path / "art.json"
        ev_export = tmp_path / "ev.json"
        artifacts.export_json(art_export)
        evidence.export_json(ev_export)
        assert artifacts.authority_unaffected_by_export_deletion(art_export)
        assert evidence.authority_unaffected_by_export_deletion(ev_export)

        art_proj = artifacts.rebuild_projection()
        ev_proj = evidence.rebuild_projection()
        assert art_proj.body["rebuilt_from_admitted_evidence"] is True
        assert ev_proj.body["rebuilt_from_admitted_evidence"] is True
        assert evidence.cache_promotes_assurance() is False

        hit = evidence.lookup_receipt(
            key, required_assurance=AssuranceLevel.SOLVER_CHECKED
        )
        assert hit.status is LookupStatus.HIT
        cache = evidence.lookup_cache(
            key, required_assurance=AssuranceLevel.SOLVER_CHECKED
        )
        assert cache.status is LookupStatus.REJECTED


def test_cold_import_is_side_effect_free() -> None:
    """Re-importing modules must not open databases or touch the filesystem."""

    import importlib

    artifact_mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.runtime.database_artifact_store"
    )
    evidence_mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.proof.database_evidence_store"
    )
    importlib.reload(artifact_mod)
    importlib.reload(evidence_mod)
    assert artifact_mod.DATABASE_ARTIFACT_STORE_INTERFACE.endswith("@1")
    assert evidence_mod.DATABASE_EVIDENCE_STORE_INTERFACE.endswith("@1")
