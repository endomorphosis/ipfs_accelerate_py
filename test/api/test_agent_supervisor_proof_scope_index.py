from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.dataset_store import ObjectiveDatasetStore
from ipfs_accelerate_py.agent_supervisor.proof_scope_index import (
    ProofInputKind,
    ProofScopeIndex,
    build_proof_scope_index,
)


def _snapshot(blob_a: str = "blob:a1") -> dict[str, list[dict[str, object]]]:
    return {
        "scope_blobs": [
            {
                "path": "src/api.py",
                "blob_id": blob_a,
                "scopes": [
                    {
                        "scope_id": f"scope:{blob_a}",
                        "path": "src/api.py",
                        "qualified_symbol": "pkg.api.Service.run",
                        "interfaces": ["pkg.api.Service"],
                        "assumptions": ["assumption:lease-is-current"],
                    }
                ],
            },
            {
                "path": "src/consumer.py",
                "blob_id": "blob:b1",
                "scopes": [
                    {
                        "scope_id": "scope:b1",
                        "path": "src/consumer.py",
                        "qualified_symbol": "pkg.consumer.consume",
                    }
                ],
            },
        ],
        "obligations": [
            {
                "obligation_id": "obligation:api",
                "ast_scope_ids": [f"scope:{blob_a}"],
                "template_id": "lease-uniqueness",
                "template_version": "2",
                "template_semantic_hash": "sha256:template-v2",
                "premise_ids": [],
            },
            {
                "obligation_id": "obligation:consumer",
                "ast_scope_ids": ["scope:b1"],
                "template_id": "projection-equivalence",
                "template_version": "1",
                "template_semantic_hash": "sha256:projection-v1",
            },
        ],
        "receipts": [
            {
                "receipt_id": "receipt:api",
                "obligation_id": "obligation:api",
                "ast_scope_ids": [f"scope:{blob_a}"],
                "toolchain_id": "lean-4.19",
                "policy_id": "protected-paths-v3",
            },
            {
                "receipt_id": "receipt:consumer",
                "obligation_id": "obligation:consumer",
                "ast_scope_ids": ["scope:b1"],
                "toolchain_id": "lean-4.19",
                "policy_id": "protected-paths-v3",
            },
        ],
        "proof_plans": [
            {
                "plan_id": "plan:one",
                "policy_id": "protected-paths-v3",
                "steps": [
                    {
                        "step_id": "step:api",
                        "obligation_id": "obligation:api",
                        "provider_id": "hammer",
                    },
                    {
                        "step_id": "step:consumer",
                        "obligation_id": "obligation:consumer",
                        "provider_id": "kernel",
                        "depends_on": ["step:api"],
                    },
                ],
            }
        ],
    }


def test_scope_dimensions_map_to_dependent_obligations_and_receipts() -> None:
    index = build_proof_scope_index(**_snapshot())

    assert index.obligations_for_scope(
        ProofInputKind.FILE, "src/api.py"
    ) == ("obligation:api", "obligation:consumer")
    assert index.receipts_for_scope(
        "qualified_symbol", "pkg.api.Service.run"
    ) == ("receipt:api", "receipt:consumer")
    assert index.dependents(
        "interface", "pkg.api.Service"
    ).obligation_ids == ("obligation:api", "obligation:consumer")
    assert index.dependents(
        "assumption", "assumption:lease-is-current"
    ).receipt_ids == ("receipt:api", "receipt:consumer")
    assert index.obligations_for_scope(
        "template", "lease-uniqueness"
    ) == ("obligation:api", "obligation:consumer")
    assert index.receipts_for_scope(
        "toolchain", "lean-4.19"
    ) == ("receipt:api", "receipt:consumer")
    assert index.obligations_for_scope(
        "policy", "protected-paths-v3"
    ) == ("obligation:api", "obligation:consumer")
    assert index.active_receipt_ids == ("receipt:api", "receipt:consumer")
    assert ProofScopeIndex.from_json(index.to_json()) == index


def test_explicit_template_toolchain_and_policy_changes_invalidate_dependents() -> None:
    index = build_proof_scope_index(**_snapshot())

    template_changed = index.invalidate(
        [("template", "lease-uniqueness")], max_reason_chain=3
    )
    assert template_changed.active_receipt_ids == ()
    assert template_changed.reasons_for("obligation:api")[0].changed_input is not None
    assert template_changed.invalidate(
        [("template", "lease-uniqueness")], max_reason_chain=3
    ) == template_changed

    toolchain_changed = index.invalidate(["toolchain:lean-4.19"])
    policy_changed = build_proof_scope_index(
        **_snapshot(), changed_inputs=[{"kind": "policy", "value": "protected-paths-v3"}]
    )
    assert toolchain_changed.active_obligation_ids == ()
    assert policy_changed.active_receipt_ids == ()


def test_blob_cache_reuses_unchanged_content_and_rename_invalidates_old_path() -> None:
    calls: list[str] = []

    def parser(blob: dict[str, object]):
        calls.append(str(blob["path"]))
        return [{"qualified_symbol": "pkg.api.Service.run"}]

    cold = build_proof_scope_index(
        scope_blobs=[
            {"path": "src/api.py", "blob_id": "blob:same", "source": "def run(): pass"}
        ],
        parser=parser,
    )
    assert calls == ["src/api.py"]
    assert cold.stats.parsed_blob_count == 1

    warm = build_proof_scope_index(
        scope_blobs=[
            {"path": "src/api.py", "blob_id": "blob:same", "source": "def run(): pass"}
        ],
        previous=cold,
        parser=parser,
    )
    assert calls == ["src/api.py"]
    assert warm.stats.reused_blob_count == 1
    assert warm.scope_records == cold.scope_records

    renamed_probe = build_proof_scope_index(
        scope_blobs=[
            {
                "path": "src/renamed_api.py",
                "blob_id": "blob:same",
                "source": "def run(): pass",
            }
        ],
        previous=cold,
        parser=parser,
    )
    assert calls == ["src/api.py"]
    assert renamed_probe.stats.reused_blob_count == 1
    assert renamed_probe.stats.renamed_blob_count == 1
    assert renamed_probe.scope_records[0].path == "src/renamed_api.py"
    assert renamed_probe.scope_records[0].scope_id != cold.scope_records[0].scope_id

    renamed_scope_id = renamed_probe.scope_records[0].scope_id
    old_scope_id = cold.scope_records[0].scope_id
    with_evidence = build_proof_scope_index(
        scope_blobs=[
            {"path": "src/api.py", "blob_id": "blob:same", "source": "def run(): pass"}
        ],
        obligations=[
            {"obligation_id": "old-obligation", "ast_scope_ids": [old_scope_id]}
        ],
        receipts=[
            {
                "receipt_id": "old-receipt",
                "obligation_id": "old-obligation",
                "ast_scope_ids": [old_scope_id],
            }
        ],
        parser=parser,
    )
    renamed = build_proof_scope_index(
        scope_blobs=[
            {
                "path": "src/renamed_api.py",
                "blob_id": "blob:same",
                "source": "def run(): pass",
            }
        ],
        obligations=[
            {
                "obligation_id": "old-obligation",
                "ast_scope_ids": [renamed_scope_id],
            }
        ],
        receipts=[
            {
                "receipt_id": "old-receipt",
                "obligation_id": "old-obligation",
                "ast_scope_ids": [renamed_scope_id],
            }
        ],
        previous=with_evidence,
        parser=parser,
    )
    assert renamed.stats.renamed_blob_count == 1
    assert renamed.invalidated_obligation_ids == ("old-obligation",)
    assert renamed.invalidated_receipt_ids == ("old-receipt",)
    assert renamed.reasons_for("old-obligation")[0].reason_code == "scope_renamed"
    assert not renamed.obligations_for_scope(
        "file", "src/renamed_api.py", active_only=True
    )


def test_invalidation_is_transitive_and_reason_chains_are_bounded() -> None:
    original = build_proof_scope_index(**_snapshot())
    changed = _snapshot(blob_a="blob:a2")
    # Retained evidence still names the preceding immutable AST scope.  The
    # exhaustive path must detect this without relying on scan history.
    changed["obligations"][0]["ast_scope_ids"] = ["scope:blob:a1"]
    changed["receipts"][0]["ast_scope_ids"] = ["scope:blob:a1"]

    incremental = build_proof_scope_index(
        **changed,
        previous=original,
        max_reason_chain=3,
    )
    exhaustive = build_proof_scope_index(
        **changed,
        exhaustive=True,
        max_reason_chain=3,
    )

    assert incremental.invalidated_obligation_ids == (
        "obligation:api",
        "obligation:consumer",
    )
    assert incremental.invalidated_receipt_ids == (
        "receipt:api",
        "receipt:consumer",
    )
    consumer_reason = incremental.reasons_for("obligation:consumer")[0]
    assert consumer_reason.reason_code == "dependency_invalidated"
    assert len(consumer_reason.reason_chain) <= 3
    assert incremental.reasons_for("receipt:consumer")[0].chain_truncated
    assert incremental.active_obligation_ids == exhaustive.active_obligation_ids
    assert incremental.active_receipt_ids == exhaustive.active_receipt_ids == ()


def test_delete_invalidates_stale_evidence_and_unrelated_evidence_stays_active() -> None:
    # This case isolates deletion from dependency propagation; the preceding
    # test covers the proof-plan edge explicitly.
    seed = _snapshot()
    seed["proof_plans"] = []
    original = build_proof_scope_index(**seed)
    current = _snapshot()
    current["scope_blobs"] = [current["scope_blobs"][1]]
    current["proof_plans"] = []
    # Keep the append-only evidence catalog to prove stale records fail closed.
    deleted = build_proof_scope_index(**current, previous=original)

    assert deleted.stats.deleted_blob_count == 1
    assert deleted.is_obligation_active("obligation:consumer")
    assert deleted.is_receipt_active("receipt:consumer")
    assert not deleted.is_obligation_active("obligation:api")
    assert not deleted.is_receipt_active("receipt:api")
    assert deleted.reasons_for("obligation:api")[0].reason_code in {
        "missing_scope",
        "scope_deleted",
    }


def test_dataset_store_round_trips_content_addressed_index(tmp_path: Path) -> None:
    index = build_proof_scope_index(**_snapshot())
    store = ObjectiveDatasetStore(tmp_path / "datasets")

    artifact = store.persist_proof_scope_index(index, index_name="repo/tree:one")
    again = store.persist_proof_scope_index(index, index_name="repo/tree:one")

    assert artifact.artifact_id == again.artifact_id
    assert artifact.index_id == index.index_id
    assert artifact.active_receipt_count == 2
    assert artifact.sha256 in artifact.json_path.name
    assert store.load_proof_scope_index("repo/tree:one") == index
    assert store.load_proof_scope_index(artifact) == index
    assert store.load_proof_scope_index(artifact.manifest_path) == index
    assert store.load_proof_scope_index_manifest("repo/tree:one")[
        "artifact_id"
    ] == artifact.artifact_id
