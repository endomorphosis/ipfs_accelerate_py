"""Tests for DatabaseArtifactStore@1 and DatabaseEvidenceStore@1 (DQP-025).

Evidence subset: content identity, provenance, redaction, size/graph quotas,
corruption, stale key, single flight, cache applicability, rebuild.

Acceptance:

* JSON/Parquet/file freshness no longer determines authority
* Every large external blob is digest-bound and verified on use
* Caches never promote assurance
* Stale or poisoned hits fail closed
* Database projections rebuild from admitted evidence
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.database_evidence_store import (
    AUTHORITY_CLASS as EVIDENCE_AUTHORITY,
    CACHE_ASSURANCE_POLICY,
    DATABASE_EVIDENCE_STORE_INTERFACE,
    REDACTION_MARKER as EVIDENCE_REDACTION_MARKER,
    AssuranceLevel,
    DatabaseEvidenceStore,
    EvidenceKey,
    EvidenceKind,
    EvidenceVerdict,
    LookupStatus,
    RejectionReason,
    SingleFlightExecutionError,
    duckdb_available as evidence_duckdb_available,
    open_database_evidence_store,
)
from ipfs_accelerate_py.agent_supervisor.runtime.database_artifact_store import (
    AUTHORITY_CLASS as ARTIFACT_AUTHORITY,
    DATABASE_ARTIFACT_STORE_INTERFACE,
    EXPORT_AUTHORITY,
    REDACTION_MARKER as ARTIFACT_REDACTION_MARKER,
    ArtifactKind,
    ArtifactStoreQuotas,
    DatabaseArtifactStore,
    DatabaseArtifactStoreIntegrityError,
    DatabaseArtifactStoreQuotaError,
    EdgeKind,
    duckdb_available as artifact_duckdb_available,
    open_database_artifact_store,
)


pytestmark = pytest.mark.skipif(
    not (artifact_duckdb_available() and evidence_duckdb_available()),
    reason="DuckDB is required for database evidence store hermetic tests",
)


def _artifact_store(
    tmp_path: Path,
    *,
    quotas: ArtifactStoreQuotas | None = None,
) -> DatabaseArtifactStore:
    return open_database_artifact_store(
        tmp_path / "artifacts.duckdb",
        quotas=quotas,
        blob_root=tmp_path / "blobs",
    )


def _evidence_store(
    tmp_path: Path,
    *,
    clock=None,
    default_ttl_seconds: int = 3600,
) -> DatabaseEvidenceStore:
    kwargs = {
        "default_ttl_seconds": default_ttl_seconds,
    }
    if clock is not None:
        kwargs["clock"] = clock
    return open_database_evidence_store(
        tmp_path / "evidence.duckdb",
        **kwargs,
    )


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Interface identities
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert DATABASE_ARTIFACT_STORE_INTERFACE == "DatabaseArtifactStore@1"
    assert DATABASE_EVIDENCE_STORE_INTERFACE == "DatabaseEvidenceStore@1"
    assert DatabaseArtifactStore.INTERFACE == DATABASE_ARTIFACT_STORE_INTERFACE
    assert DatabaseEvidenceStore.INTERFACE == DATABASE_EVIDENCE_STORE_INTERFACE
    assert ARTIFACT_AUTHORITY == "database_authority"
    assert EVIDENCE_AUTHORITY == "database_authority"
    assert CACHE_ASSURANCE_POLICY == "never_promote_assurance"
    assert ARTIFACT_REDACTION_MARKER == EVIDENCE_REDACTION_MARKER


# ---------------------------------------------------------------------------
# Content identity + provenance + redaction
# ---------------------------------------------------------------------------


def test_artifact_content_identity_and_provenance(tmp_path: Path) -> None:
    with _artifact_store(tmp_path) as store:
        first = store.put_artifact(
            kind=ArtifactKind.RECEIPT,
            body={"task": "task:1", "status": "ok"},
            provenance={"source": "validation", "run": "run:1"},
            metadata={"lane": "evidence"},
        )
        assert first.admitted is True
        assert first.digest.startswith("sha256:")
        assert first.provenance["source"] == "validation"
        assert first.to_dict()["authority"] == ARTIFACT_AUTHORITY

        again = store.put_artifact(
            kind=ArtifactKind.RECEIPT,
            body={"task": "task:1", "status": "ok"},
            provenance={"source": "validation", "run": "run:1"},
            metadata={"lane": "evidence"},
            artifact_id=first.artifact_id,
        )
        assert again.artifact_id == first.artifact_id
        assert again.digest == first.digest

        edge = store.put_edge(
            first.artifact_id,
            "artifact:upstream",
            EdgeKind.DERIVES_FROM,
            reason="validated from upstream receipt",
        )
        assert edge.edge_kind == EdgeKind.DERIVES_FROM.value
        edges = store.list_edges(artifact_id=first.artifact_id)
        assert len(edges) == 1
        assert edges[0].source_artifact_id == first.artifact_id


def test_redaction_on_artifact_and_evidence_metadata(tmp_path: Path) -> None:
    """Redaction replaces secret-bearing keys with the classification marker.

    Values below are synthetic canaries used only to prove redaction. They are
    not credentials and must never be treated as durable secret material.
    """

    with _artifact_store(tmp_path) as store:
        artifact = store.put_artifact(
            kind=ArtifactKind.GENERIC,
            body={"note": "public"},
            metadata={
                "access_token": "synthetic-access-token-value",
                "nested": {"password": "also-secret"},
                "lane": "safe",
            },
        )
        assert artifact.redacted is True
        assert artifact.metadata["access_token"] == ARTIFACT_REDACTION_MARKER
        assert artifact.metadata["nested"]["password"] == ARTIFACT_REDACTION_MARKER
        assert artifact.metadata["lane"] == "safe"
        assert "synthetic-access-token-value" not in str(artifact.to_dict())

    with _evidence_store(tmp_path) as store:
        key = EvidenceKey.create(
            kind=EvidenceKind.VALIDATION,
            subject_id="task:redact",
            semantic_roots={"tree": "tree:1"},
        )
        receipt = store.put(
            key,
            verdict=EvidenceVerdict.PASS,
            assurance_level=AssuranceLevel.VALIDATED,
            body={
                "access_token": "synthetic-access-token-value",
                "detail": "ok",
            },
            metadata={"api_key": "synthetic-api-key-value"},
        )
        assert receipt.redacted is True
        assert receipt.body["access_token"] == EVIDENCE_REDACTION_MARKER
        assert receipt.metadata["api_key"] == EVIDENCE_REDACTION_MARKER
        assert receipt.body["detail"] == "ok"


# ---------------------------------------------------------------------------
# Digest-bound blobs, corruption, quotas
# ---------------------------------------------------------------------------


def test_blob_digest_bound_verified_on_use_and_corruption(tmp_path: Path) -> None:
    payload = b"large-immutable-body-for-cas"
    with _artifact_store(tmp_path) as store:
        reference = store.put_blob(payload, media_type="application/octet-stream")
        assert reference.digest == _digest(payload)
        assert reference.verified is True
        assert reference.blob_id == f"blob:{reference.digest}"

        loaded = store.verify_blob(reference.digest)
        assert loaded == payload

        artifact = store.put_artifact(
            kind=ArtifactKind.BLOB,
            blob_digest=reference.digest,
            provenance={"origin": "cas"},
        )
        assert artifact.digest == reference.digest
        assert artifact.size_bytes == len(payload)

        # Tamper with the external body: verification must fail closed.
        blob_path = store.blob_root / reference.digest.removeprefix("sha256:")[:2]
        blob_path = blob_path / f"{reference.digest.removeprefix('sha256:')}.blob"
        blob_path.write_bytes(b"corrupted-body-not-matching-digest")
        with pytest.raises(DatabaseArtifactStoreIntegrityError):
            store.verify_blob(reference.digest)


def test_size_and_graph_quotas(tmp_path: Path) -> None:
    quotas = ArtifactStoreQuotas(
        max_artifacts=2,
        max_edges=2,
        max_datasets=1,
        max_blob_bytes=32,
        max_total_blob_bytes=64,
        max_graph_degree=2,
    )
    with _artifact_store(tmp_path, quotas=quotas) as store:
        store.put_artifact(kind=ArtifactKind.GENERIC, body={"n": 1})
        store.put_artifact(kind=ArtifactKind.GENERIC, body={"n": 2})
        with pytest.raises(DatabaseArtifactStoreQuotaError):
            store.put_artifact(kind=ArtifactKind.GENERIC, body={"n": 3})

        with pytest.raises(DatabaseArtifactStoreQuotaError):
            store.put_blob(b"x" * 64)

        a = store.list_artifacts()[0]
        store.put_edge(a.artifact_id, "artifact:t1", EdgeKind.REFERENCES)
        store.put_edge(a.artifact_id, "artifact:t2", EdgeKind.REFERENCES)
        with pytest.raises(DatabaseArtifactStoreQuotaError):
            store.put_edge(a.artifact_id, "artifact:t3", EdgeKind.REFERENCES)

        store.put_dataset(name="ds-one", rows=[{"x": 1}])
        with pytest.raises(DatabaseArtifactStoreQuotaError):
            store.put_dataset(name="ds-two", rows=[{"x": 2}])


# ---------------------------------------------------------------------------
# Export non-authority + rebuild
# ---------------------------------------------------------------------------


def test_export_is_non_authoritative_and_projection_rebuilds(
    tmp_path: Path,
) -> None:
    with _artifact_store(tmp_path) as store:
        store.put_artifact(
            kind=ArtifactKind.RECEIPT,
            body={"task": "task:export"},
        )
        dataset = store.put_dataset(
            name="scan-details",
            rows=[{"path": "a.py", "status": "ok"}],
            provenance={"scanner": "objective"},
        )
        export_path = tmp_path / "export.json"
        receipt = store.export_snapshot(export_path)
        assert receipt.authority == EXPORT_AUTHORITY
        assert export_path.is_file()
        assert receipt.artifact_count >= 1
        assert receipt.dataset_count == 1

        # Deleting the export must not affect database authority.
        export_path.unlink()
        assert not export_path.exists()
        assert store.get_dataset(dataset.dataset_id) is not None
        artifacts = store.list_artifacts()
        assert len(artifacts) >= 1
        rebuilt = store.rebuild_projection("admitted_artifacts")
        assert rebuilt.rebuilt_from == "admitted_evidence"
        assert rebuilt.row_count == len(artifacts)
        assert rebuilt.digest.startswith("sha256:")
        assert store.stats()["authority"] == ARTIFACT_AUTHORITY


# ---------------------------------------------------------------------------
# Evidence: cache applicability, never promote assurance, stale/poisoned
# ---------------------------------------------------------------------------


def test_evidence_lookup_hit_and_key_applicability(tmp_path: Path) -> None:
    with _evidence_store(tmp_path) as store:
        key = EvidenceKey.create(
            kind=EvidenceKind.PROOF,
            subject_id="obligation:alpha",
            semantic_roots={"tree": "tree:abc", "policy": "pol:1"},
            policy_id="policy:proof@1",
        )
        receipt = store.put(
            key,
            verdict=EvidenceVerdict.PASS,
            assurance_level=AssuranceLevel.KERNEL_VERIFIED,
            body={"prover": "lean", "status": "proved"},
        )
        assert receipt.assurance_level == AssuranceLevel.KERNEL_VERIFIED.value

        hit = store.lookup(
            key,
            required_assurance=AssuranceLevel.SOLVER_CHECKED,
        )
        assert hit.status is LookupStatus.HIT
        assert hit.receipt is not None
        assert hit.receipt.receipt_id == receipt.receipt_id
        assert hit.use_id

        # Different semantic roots => different key => miss (stale key).
        stale_key = EvidenceKey.create(
            kind=EvidenceKind.PROOF,
            subject_id="obligation:alpha",
            semantic_roots={"tree": "tree:CHANGED", "policy": "pol:1"},
            policy_id="policy:proof@1",
        )
        miss = store.lookup(stale_key)
        assert miss.status is LookupStatus.MISS
        assert miss.reason is RejectionReason.CACHE_MISS


def test_cache_never_promotes_assurance(tmp_path: Path) -> None:
    with _evidence_store(tmp_path) as store:
        key = EvidenceKey.create(
            kind=EvidenceKind.PROOF,
            subject_id="obligation:beta",
            semantic_roots={"forest": "f:1"},
        )
        store.put(
            key,
            verdict=EvidenceVerdict.PASS,
            assurance_level=AssuranceLevel.HEURISTIC,
            body={"note": "model-only heuristic"},
        )
        result = store.lookup(
            key,
            required_assurance=AssuranceLevel.KERNEL_VERIFIED,
        )
        assert result.status is LookupStatus.REJECTED
        assert result.reason is RejectionReason.ASSURANCE_PROMOTION_FORBIDDEN
        assert result.receipt is not None
        assert (
            result.receipt.assurance_level == AssuranceLevel.HEURISTIC.value
        )


def test_stale_and_poisoned_hits_fail_closed(tmp_path: Path) -> None:
    clock = {"now": 1_000.0}

    def _clock() -> float:
        return clock["now"]

    with _evidence_store(
        tmp_path, clock=_clock, default_ttl_seconds=10
    ) as store:
        key = EvidenceKey.create(
            kind=EvidenceKind.ANALYSIS,
            subject_id="analysis:gamma",
            semantic_roots={"ast": "ast:1"},
        )
        receipt = store.put(
            key,
            verdict=EvidenceVerdict.PASS,
            assurance_level=AssuranceLevel.VALIDATED,
            body={"score": 1},
            ttl_seconds=10,
        )
        assert store.lookup(key).status is LookupStatus.HIT

        # Advance past expiry => stale fail-closed.
        clock["now"] = 1_000.0 + 11
        stale = store.lookup(key)
        assert stale.status is LookupStatus.REJECTED
        assert stale.reason is RejectionReason.STALE

        # Fresh put then poison.
        clock["now"] = 2_000.0
        key2 = EvidenceKey.create(
            kind=EvidenceKind.ANALYSIS,
            subject_id="analysis:delta",
            semantic_roots={"ast": "ast:2"},
        )
        receipt2 = store.put(
            key2,
            verdict=EvidenceVerdict.PASS,
            assurance_level=AssuranceLevel.VALIDATED,
            body={"score": 2},
        )
        store.mark_poisoned(receipt2.receipt_id)
        poisoned = store.lookup(key2)
        assert poisoned.status is LookupStatus.REJECTED
        assert poisoned.reason in {
            RejectionReason.POISONED,
            RejectionReason.INVALIDATED,
        }

        # Explicit invalidation.
        key3 = EvidenceKey.create(
            kind=EvidenceKind.VALIDATION,
            subject_id="validation:epsilon",
            semantic_roots={"run": "run:9"},
        )
        store.put(
            key3,
            verdict=EvidenceVerdict.PASS,
            assurance_level=AssuranceLevel.VALIDATED,
            body={"ok": True},
        )
        changed = store.invalidate(key=key3, reason="semantic-root-changed")
        assert changed >= 1
        invalidated = store.lookup(key3)
        assert invalidated.status is LookupStatus.REJECTED
        assert invalidated.reason is RejectionReason.INVALIDATED

        # Original receipt still addressable for audit, not as a cache hit.
        loaded = store.get_receipt(receipt.receipt_id)
        assert loaded is not None
        assert loaded.receipt_id == receipt.receipt_id


def test_inconclusive_not_promoted_to_hit(tmp_path: Path) -> None:
    with _evidence_store(tmp_path) as store:
        key = EvidenceKey.create(
            kind=EvidenceKind.PROOF,
            subject_id="obligation:inconclusive",
            semantic_roots={"tree": "tree:z"},
        )
        store.put(
            key,
            verdict=EvidenceVerdict.INCONCLUSIVE,
            assurance_level=AssuranceLevel.SOLVER_CHECKED,
            body={"reason": "timeout"},
        )
        rejected = store.lookup(key)
        assert rejected.status is LookupStatus.REJECTED
        assert rejected.reason is RejectionReason.INCONCLUSIVE

        allowed = store.lookup(key, allow_inconclusive=True)
        assert allowed.status is LookupStatus.HIT


# ---------------------------------------------------------------------------
# Single flight + rebuild
# ---------------------------------------------------------------------------


def test_single_flight_owner_and_shared_outcome(tmp_path: Path) -> None:
    with _evidence_store(tmp_path) as store:
        key = EvidenceKey.create(
            kind=EvidenceKind.CACHE,
            subject_id="work:single-flight",
            semantic_roots={"input": "digest:1"},
        )
        calls = {"n": 0}

        def producer() -> dict[str, int]:
            calls["n"] += 1
            return {"answer": 42}

        first = store.single_flight(key, producer, wait_seconds=5, lease_seconds=30)
        assert first.owner is True
        assert first.value == {"answer": 42}
        assert calls["n"] == 1

        second = store.single_flight(key, producer, wait_seconds=5, lease_seconds=30)
        assert second.owner is False
        assert second.shared is True
        assert second.value == {"answer": 42}
        # Shared outcome reused; producer not re-invoked.
        assert calls["n"] == 1


def test_single_flight_error_propagates(tmp_path: Path) -> None:
    with _evidence_store(tmp_path) as store:
        key = EvidenceKey.create(
            kind=EvidenceKind.CACHE,
            subject_id="work:error",
            semantic_roots={"input": "digest:err"},
        )

        def boom() -> None:
            raise RuntimeError("producer exploded")

        with pytest.raises(SingleFlightExecutionError):
            store.single_flight(key, boom, wait_seconds=2, lease_seconds=10)


def test_evidence_projection_rebuilds_from_admitted_only(tmp_path: Path) -> None:
    with _evidence_store(tmp_path) as store:
        good = EvidenceKey.create(
            kind=EvidenceKind.VALIDATION,
            subject_id="task:good",
            semantic_roots={"tree": "t:1"},
        )
        bad = EvidenceKey.create(
            kind=EvidenceKind.VALIDATION,
            subject_id="task:bad",
            semantic_roots={"tree": "t:2"},
        )
        store.put(
            good,
            verdict=EvidenceVerdict.PASS,
            assurance_level=AssuranceLevel.VALIDATED,
            body={"ok": True},
        )
        poisoned = store.put(
            bad,
            verdict=EvidenceVerdict.PASS,
            assurance_level=AssuranceLevel.VALIDATED,
            body={"ok": False},
        )
        store.mark_poisoned(poisoned.receipt_id)

        rebuilt = store.rebuild_projection("admitted_receipts")
        assert rebuilt.rebuilt_from == "admitted_evidence"
        assert rebuilt.row_count == 1
        assert rebuilt.digest.startswith("sha256:")
        stats = store.stats()
        assert stats["cache_assurance_policy"] == CACHE_ASSURANCE_POLICY
        assert stats["receipt_count"] == 2
        assert stats["invalidation_count"] >= 1


def test_attestation_requires_admitted_receipt(tmp_path: Path) -> None:
    with _evidence_store(tmp_path) as store:
        key = EvidenceKey.create(
            kind=EvidenceKind.ATTESTATION,
            subject_id="proof:attested",
            semantic_roots={"kernel": "k:1"},
        )
        receipt = store.put(
            key,
            verdict=EvidenceVerdict.PASS,
            assurance_level=AssuranceLevel.KERNEL_VERIFIED,
            body={"proof": "ok"},
        )
        digest = _digest(b"attestation-body")
        attestation = store.put_attestation(
            receipt.receipt_id,
            content_digest=digest,
            backend="local-repro",
            body={"reproduced": True},
        )
        assert attestation["receipt_id"] == receipt.receipt_id
        assert attestation["content_digest"] == digest

        with pytest.raises(Exception):
            store.put_attestation(
                "receipt:missing",
                content_digest=digest,
            )


def test_dataset_digest_identity_without_file_authority(tmp_path: Path) -> None:
    with _artifact_store(tmp_path) as store:
        dataset = store.put_dataset(
            name="proof-scope-index",
            rows=[
                {"scope": "s1", "status": "active"},
                {"scope": "s2", "status": "active"},
            ],
            provenance={"builder": "proof-scope"},
        )
        assert dataset.row_count == 2
        assert dataset.digest.startswith("sha256:")
        assert dataset.admitted is True
        # Re-admit by digest alone without re-supplying rows.
        again = store.put_dataset(
            name="proof-scope-index",
            digest=dataset.digest,
            row_count=2,
            byte_count=dataset.byte_count,
            dataset_id=dataset.dataset_id,
        )
        assert again.dataset_id == dataset.dataset_id
