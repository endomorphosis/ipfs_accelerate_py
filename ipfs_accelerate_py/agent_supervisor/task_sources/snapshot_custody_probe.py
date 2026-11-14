"""Measure snapshot custody without promoting a candidate assertion.

The bundle commits its own payload files under a caller-supplied manifest
hash and refuses an incomplete directory. External source, interpretation,
and evidence payloads named by those records stay unauthenticated. A missing
declared dependency stays explicit and is not reused.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from ipfs_datasets_py.utils.cid_utils import canonical_dag_json_bytes

from .dependency_block_resolution import EXPECTED_SNAPSHOT_CUSTODY_PROBE


def measure_snapshot_custody_probe() -> dict[str, object] | None:
    """Return the fixed custody contract, or None when the owners disagree."""

    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_SNAPSHOT_CUSTODY_PROBE:
        return None
    return measured


def _cid(seal_record, label: str) -> str:
    return seal_record({"schema": "snapshot-custody-probe/v1", "label": label})["record_cid"]


def _measure() -> dict[str, object]:
    from ipfs_datasets_py.logic.autoformal import corpus_graph as graph
    from ipfs_datasets_py.logic.autoformal import corpus_graph_exchange as exchange
    from ipfs_datasets_py.logic.autoformal import legal_graph_dependencies as deps
    from ipfs_datasets_py.logic.autoformal.corpus_contracts import (
        CorpusContractError,
        seal_record,
    )

    def cid(label: str) -> str:
        return _cid(seal_record, label)

    source = graph.build_node(
        node_type="source_version",
        stable_id="source",
        content_cid=cid("source-content"),
    )
    interpretation = graph.build_node(
        node_type="interpretation",
        stable_id="interpretation",
        content_cid=cid("interpretation-content"),
    )
    reading = graph.build_node(
        node_type="norm_assertion",
        stable_id="reading",
        content_cid=cid("reading-content"),
        source_occurrence_cids=[source["record_cid"]],
        interpretation_cids=[interpretation["record_cid"]],
    )
    snapshot = graph.validate_graph_snapshot(
        graph.build_graph_snapshot([source, interpretation, reading], [])
    )

    def refusal(call) -> str:
        try:
            call()
        except (CorpusContractError, deps.GraphDependencyError) as exc:
            return str(exc)
        raise RuntimeError("snapshot custody accepted a refused shape")

    with tempfile.TemporaryDirectory() as directory:
        bundle = Path(directory) / "bundle"
        manifest = exchange.write_graph_bundle(snapshot, bundle)
        digest = hashlib.sha256(canonical_dag_json_bytes(manifest)).hexdigest()
        loaded = exchange.load_graph_bundle(
            bundle, expected_manifest_sha256=digest
        )
        names = sorted(os.listdir(bundle))
        if names != sorted(exchange.BUNDLE_FILES):
            raise RuntimeError("bundle filenames changed")
        external = {
            source["content_cid"],
            source["record_cid"],
            interpretation["content_cid"],
            interpretation["record_cid"],
            reading["content_cid"],
        }
        external_in_bundle = sorted(name for name in names if name in external)
        wrong_manifest = refusal(
            lambda: exchange.load_graph_bundle(
                bundle, expected_manifest_sha256="0" * 64
            )
        )
        payload = bundle / "nodes.parquet"
        removed = payload.read_bytes()
        payload.unlink()
        incomplete = refusal(
            lambda: exchange.load_graph_bundle(
                bundle, expected_manifest_sha256=digest
            )
        )
        payload.write_bytes(removed)
        existing = refusal(lambda: exchange.write_graph_bundle(snapshot, bundle))
    if (
        not existing.startswith("could not write local graph bundle:")
        or "File exists" not in existing
    ):
        raise RuntimeError("existing directory refusal changed")
    if loaded.get("snapshot") != snapshot:
        raise RuntimeError("custody reload changed the snapshot")
    by_id = {node["stable_id"]: node for node in loaded["snapshot"]["nodes"]}
    pins = {kind: [] for kind in deps.PIN_KINDS}
    artifact = {
        "artifact_id": "snapshot",
        "artifact_sha256": "a" * 64,
        "dependency_pins": pins,
    }
    edge = {"dependent_id": "snapshot", "dependency_id": "missing-record"}
    built = deps.build_dependency_manifest(
        [artifact], [edge], expected_artifact_ids=["snapshot"]
    )
    checked = deps.validate_dependency_manifest(
        built, expected_manifest_sha256=built["manifest_sha256"]
    )
    identity = refusal(
        lambda: deps.validate_dependency_manifest(
            built, expected_manifest_sha256="b" * 64
        )
    )
    promoted = {
        key: value for key, value in built.items() if key != "manifest_sha256"
    }
    promoted["admitted"] = True
    promoted["manifest_sha256"] = deps._digest(
        {key: value for key, value in promoted.items() if key != "manifest_sha256"}
    )
    promoted_refusal = refusal(
        lambda: deps.validate_dependency_manifest(
            promoted, expected_manifest_sha256=promoted["manifest_sha256"]
        )
    )
    report = deps.invalidate_dependencies(
        checked,
        expected_manifest_sha256=checked["manifest_sha256"],
        observed_artifact_sha256s={"snapshot": "a" * 64},
        observed_pins={kind: {} for kind in deps.PIN_KINDS},
    )
    return {
        "admitted": manifest["admitted"],
        "bundle_files": names,
        "bundle_records_present": True,
        "bundle_roundtrip": True,
        "candidate_assertions_promoted": manifest["admitted"] is True,
        "candidate_view": manifest["view"],
        "coverage_scope": report["coverage_scope"],
        "dependency_admitted": report["admitted"],
        "edge_count": manifest["edge_count"],
        "existing_directory_refused": "could not write local graph bundle:",
        "external_payloads_in_bundle": len(external_in_bundle),
        "external_referenced_records_authenticated": (
            manifest["external_references_verified"] is True
            or len(external_in_bundle) > 0
        ),
        "external_references_verified": manifest["external_references_verified"],
        "formalized": manifest["formalized"],
        "incomplete_bundle_refused": incomplete,
        "interpretation_reference_retained": (
            by_id["reading"]["interpretation_cids"]
            == [interpretation["record_cid"]]
        ),
        "legal_dependency_inventory_exhaustive": report[
            "legal_dependency_inventory_exhaustive"
        ],
        "manifest_identity_refused": identity,
        "missing_dependency_count": len(report["missing_dependencies"]),
        "missing_dependency_id": report["missing_dependencies"][0]["dependency_id"],
        "node_count": manifest["node_count"],
        "payload_count": len(exchange.PAYLOAD_FILES),
        "population_complete": report["population_complete"],
        "promoted_manifest_refused": promoted_refusal,
        "proof_authority": report["proof_authority"],
        "reusable_under_declared_dependencies": list(
            report["reusable_under_declared_dependencies"]
        ),
        "snapshot_view": loaded["snapshot"]["view"],
        "source_reference_retained": (
            by_id["reading"]["source_occurrence_cids"] == [source["record_cid"]]
        ),
        "source_semantics_verified": manifest["source_semantics_verified"],
        "stale_artifact_ids": list(report["stale_artifact_ids"]),
        "wrong_manifest_refused": wrong_manifest,
    }
