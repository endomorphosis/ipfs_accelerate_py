"""Cold chunk projection with explicit, provenance-bound analysis limitations.

This consumer re-verifies Git content and rebuilds the limitation index. It
does not admit serialized projection claims or grant analysis/completion
authority. The existing captured-snapshot consumer remains unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .reconstruction import Reconstruction, ReconstructionError, ReconstructionLimits

MAX_BUNDLE_BLOCK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ChunkedReconstruction(Reconstruction):
    chunked_snapshot_cid: str
    analysis_limitation_cids: tuple[str, ...]

    def observation(self) -> dict[str, Any]:
        return {**super().observation(),
                "schema": "ipfs_accelerate_py/chunked-semantic-reconstruction@1",
                "chunked_snapshot_cid": self.chunked_snapshot_cid,
                "analysis_limitation_count": len(self.analysis_limitation_cids),
                "analysis_limitation_cids": list(self.analysis_limitation_cids),
                "analysis_coverage": "incomplete" if self.analysis_limitation_cids else "not_established",
                "complete_analysis_authority": False}


@dataclass(frozen=True)
class _LimitedIndex:
    state: Any
    producer: Any
    limitations: tuple[Any, ...]

    def __getattr__(self, name):
        if name in {"repository_id", "symbols", "artifacts", "edges", "schema",
                    "extractor_name", "extractor_version"}:
            return getattr(self.state, name)
        raise AttributeError(name)


def _bound_limitations(state, snapshot, chunked, limits, manifest_cid):
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
    from ipfs_datasets_py.logic.software_contracts.semantic_index.models import ArtifactRecord, RepositoryState
    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import AnalysisLimitation, ArtifactFactNode

    entries = {entry.raw_path_hex: entry for entry in snapshot.entries}
    paths = {entry.path: entry for entry in snapshot.entries}
    population = {entry.raw_path_hex: entry for entry in chunked.entries}
    snapshot_cid = snapshot.snapshot_cid
    limitations, provenance, bound_artifacts = [], [], []
    covered = set()
    for artifact in state.artifacts:
        if artifact.confidence != "opaque":
            continue
        raw = artifact.metadata.get("raw_path_hex")
        entry = entries.get(raw) if raw is not None else paths.get(artifact.path)
        if entry is None or artifact.source_cid != entry.source_cid:
            raise ReconstructionError("opaque analysis artifact is not bound to the projected population")
        member = population[entry.raw_path_hex]
        if entry.is_opaque and artifact.metadata.get("opaque_reason") != entry.opaque_reason:
            raise ReconstructionError("opaque analysis artifact changes the projected limitation")
        covered.add(entry.raw_path_hex)
        reason = entry.opaque_reason or artifact.metadata.get("opaque_reason") or "scanner_analysis_opaque"
        payload = {
            "schema": "ipfs_accelerate_py/chunked-analysis-limitation-provenance@1",
            "repository_id": snapshot.repository_id, "git_commit": snapshot.git_commit,
            "git_tree": snapshot.git_tree, "population_cid": chunked.population_cid,
            "chunked_snapshot_cid": manifest_cid, "snapshot_cid": snapshot_cid,
            "raw_path_hex": entry.raw_path_hex, "snapshot_entry_cid": entry.entry_cid,
            "git_object_oid": member.git_object_oid, "git_mode": member.git_mode,
            "source_cid": entry.source_cid, "size_bytes": entry.size_bytes,
            "opaque_artifact_id": artifact.artifact_id,
            "opaque_artifact_fact_cid": ArtifactFactNode(artifact).fact_cid,
            "reason": reason, "projection_limits": asdict(limits),
            "stream_limits": asdict(chunked.limits),
            "source_content_verified": member.object_type == "blob" and entry.source_cid is not None,
            "semantics_analyzed": False,
        }
        cid = cid_for_structured(payload)
        evidence = ArtifactRecord("artifact:analysis-limitation:" + cid, "analysis-limitation-evidence",
                                  "@analysis-limitation/" + cid, entry.source_cid, "opaque", payload)
        limitation = AnalysisLimitation(
            code="chunked_projection." + reason,
            message="Committed input semantics are unavailable; the subject binds the acquisition evidence.",
            subject_id=evidence.artifact_id, confidence="opaque")
        limitations.append(limitation)
        provenance.append(evidence)
        bound_artifacts.append(artifact)
    if {entry.raw_path_hex for entry in snapshot.entries if entry.is_opaque} - covered:
        raise ReconstructionError("scanner omitted a projected opaque input")
    if len({entry.artifact_id for entry in provenance}) != len(provenance):
        raise ReconstructionError("duplicate limitation provenance")
    augmented = RepositoryState(state.repository_id, state.symbols, (*state.artifacts, *provenance),
                                state.edges, state.extractor_name, state.extractor_version, state.schema)
    return augmented, tuple(sorted(limitations, key=lambda item: item.limitation_cid)), tuple(provenance), tuple(bound_artifacts)


def _verify_bound_bundle(bundle, producer, limitations, provenance, bound_artifacts, manifest_blocks):
    """Check both index membership and leaves; a CID claim alone is insufficient."""
    from ipfs_datasets_py.logic.software_contracts.content import canonical_dag_json_bytes
    from ipfs_datasets_py.logic.software_contracts.semantic_state import verify_semantic_state_bundle
    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
        ArtifactFactNode, SemanticStateBundle, SortedPairIndex,
    )

    if not isinstance(bundle, SemanticStateBundle):
        raise ReconstructionError("verification requires a datasets semantic-state bundle")
    if any(len(data) > MAX_BUNDLE_BLOCK_BYTES for data in bundle.blocks.values()):
        raise ReconstructionError("semantic bundle exceeds fixed metadata frame bound")
    root = verify_semantic_state_bundle(bundle)
    expected = SortedPairIndex([(item.limitation_cid, item.limitation_cid) for item in limitations])
    if root.producer != producer or root.analysis_limitation_index_cid != expected.index_cid:
        raise ReconstructionError("bundle changes bound producer or formal analysis limitations")
    required = {expected.index_cid: canonical_dag_json_bytes(expected.identity_payload()), **manifest_blocks}
    for item in limitations:
        required[item.limitation_cid] = canonical_dag_json_bytes(item.identity_payload())
    for artifact in (*provenance, *bound_artifacts):
        fact = ArtifactFactNode(artifact)
        required[fact.fact_cid] = canonical_dag_json_bytes(fact.identity_payload())
    if any(bundle.blocks.get(cid) != data for cid, data in required.items()):
        raise ReconstructionError("bundle omits or changes limitation/provenance/manifest evidence")
    members = dict(json.loads(bundle.blocks[root.artifact_fact_index_cid])["pairs"])
    if any(members.get(artifact.artifact_id) != ArtifactFactNode(artifact).fact_cid
           for artifact in (*provenance, *bound_artifacts)):
        raise ReconstructionError("limitation subjects are not present in the artifact index")
    return root


def reconstruct_chunked_semantic_state(
    repository: str | Path, chunked: Any, *, expected_commit: str, expected_tree: str,
    expected_chunked_snapshot_cid: str, repository_id: str,
    limits: ReconstructionLimits = ReconstructionLimits(), nominated_bundle: Any = None,
) -> ChunkedReconstruction:
    """Reproject the exact committed content, then bind every known opaque input.

    Projection preserves the fixed 4 MiB per-file and 128 MiB retained-source
    ceilings. Chunked work limits and producer revisions must separately be
    qualified by the native launcher. An empty limitation index is not proof
    of complete semantic analysis; this return value grants no such authority.
    """
    from ipfs_datasets_py.logic.software_contracts.semantic_index.chunked_snapshot import (
        ChunkedRepositorySnapshot, project_chunked_repository,
    )
    from ipfs_datasets_py.logic.software_contracts.semantic_index.committed_snapshot import preflight_committed_repository
    from ipfs_datasets_py.logic.software_contracts.semantic_index.scanner import RepositoryScanner
    from ipfs_datasets_py.logic.software_contracts.semantic_index.snapshot import SnapshotError
    from ipfs_datasets_py.logic.software_contracts.semantic_state import build_semantic_state
    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import SemanticStateBundle, SemanticStateProducer

    if any(not isinstance(oid, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", oid)
           for oid in (expected_commit, expected_tree)):
        raise ReconstructionError("request requires full commit and tree object ids")
    if not isinstance(repository_id, str) or not repository_id.strip():
        raise ReconstructionError("request requires an explicit repository identity")
    if not isinstance(limits, ReconstructionLimits) or not isinstance(chunked, ChunkedRepositorySnapshot):
        raise ReconstructionError("chunk reconstruction requires typed limits and manifest")
    if (chunked.repository_id, chunked.git_commit, chunked.git_tree) != (repository_id, expected_commit, expected_tree):
        raise ReconstructionError("chunked manifest differs from requested committed population")
    if (chunked.limits.max_entries > limits.max_entries
            or chunked.limits.max_metadata_bytes > limits.max_metadata_bytes):
        raise ReconstructionError("chunked manifest exceeds reconstruction metadata/entry limits")
    root = Path(repository).resolve(strict=True)
    request = dict(expected_commit=expected_commit, expected_tree=expected_tree,
                   repository_id=repository_id, **asdict(limits))
    try:
        manifest_cid, manifest_blocks = chunked.manifest_blocks()
        if manifest_cid != expected_chunked_snapshot_cid:
            raise ReconstructionError("chunked manifest differs from requested snapshot CID")
        plan = preflight_committed_repository(root, **request)
        if plan.population_cid != chunked.population_cid:
            raise ReconstructionError("chunked manifest changes the committed population")
        projection = project_chunked_repository(root, chunked, max_file_bytes=limits.max_file_bytes,
                                                max_total_bytes=limits.max_total_bytes)
    except SnapshotError as exc:
        raise ReconstructionError(str(exc)) from exc
    snapshot = projection.snapshot
    if (snapshot.mode, snapshot.repository_id, snapshot.git_commit, snapshot.git_tree) != (
            "git-clean", repository_id, expected_commit, expected_tree):
        raise ReconstructionError("projected snapshot differs from committed request")
    if ({entry.raw_path_hex for entry in snapshot.entries} != {entry.raw_path_hex for entry in plan.entries}
            or projection.chunked_snapshot_cid != manifest_cid):
        raise ReconstructionError("projected snapshot changes the exact committed population")
    members = {entry.raw_path_hex: entry for entry in plan.entries}
    blobs = {blob.git_object_oid: blob for blob in chunked.blobs}
    for entry in snapshot.entries:
        member = members[entry.raw_path_hex]
        source_cid = blobs[member.git_object_oid].source_cid if member.object_type == "blob" else None
        if (entry.git_blob_oid != member.git_object_oid or entry.head_blob_oid != member.git_object_oid
                or entry.size_bytes != member.size_bytes or entry.source_cid != source_cid
                or entry.disposition != "clean"):
            raise ReconstructionError("projected source identity differs from the verified chunk manifest")
    retained = [entry for entry in snapshot.entries if entry.captured_bytes is not None]
    if (any(entry.size_bytes > limits.max_file_bytes for entry in retained)
            or sum(entry.size_bytes for entry in retained) > limits.max_total_bytes):
        raise ReconstructionError("projected content exceeds reconstruction retained limits")
    state = RepositoryScanner(repository_id=repository_id).scan_snapshot(
        snapshot, {entry.source_key: entry.captured_bytes for entry in snapshot.entries
                   if entry.captured_bytes is not None and not entry.is_opaque})
    state, limitations, provenance, bound_artifacts = _bound_limitations(state, snapshot, chunked, limits, manifest_cid)
    producer = SemanticStateProducer(state.state_cid, snapshot.snapshot_cid, expected_commit, expected_tree,
                                     manifest_cid, state.schema, state.extractor_name, state.extractor_version)
    bundle = build_semantic_state(_LimitedIndex(state, producer, limitations))
    blocks = dict(bundle.blocks)
    for cid, data in manifest_blocks.items():
        if cid in blocks and blocks[cid] != data:
            raise ReconstructionError("conflicting chunk manifest block")
        blocks[cid] = data
    bundle = SemanticStateBundle(bundle.root, blocks)
    _verify_bound_bundle(bundle, producer, limitations, provenance, bound_artifacts, manifest_blocks)
    matched = None
    if nominated_bundle is not None:
        nominated = _verify_bound_bundle(nominated_bundle, producer, limitations, provenance,
                                         bound_artifacts, manifest_blocks)
        if nominated.root_cid != bundle.root.root_cid:
            raise ReconstructionError("nominated root differs from cold chunk reconstruction")
        matched = True
    try:
        after = preflight_committed_repository(root, **request)
    except SnapshotError as exc:
        raise ReconstructionError(str(exc)) from exc
    if after.population_cid != plan.population_cid:
        raise ReconstructionError("committed population changed during chunk reconstruction")
    configuration = {"schema": "ipfs_accelerate_py/chunked-reconstruction-config@1",
                     "repository_id": repository_id, "population_cid": plan.population_cid,
                     "chunked_snapshot_cid": manifest_cid, "limits": asdict(limits),
                     "stream_limits": asdict(chunked.limits), "population_scope": "complete-committed",
                     "analysis_limitation_index_cid": bundle.root.analysis_limitation_index_cid,
                     "reuse": "cold", "environment_bindings": []}
    digest = "sha256:" + hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return ChunkedReconstruction(bundle, snapshot.snapshot_cid, state.state_cid, expected_commit,
                                 expected_tree, digest,
                                 tuple((entry.path, entry.opaque_reason) for entry in snapshot.entries if entry.is_opaque),
                                 matched, plan.population_cid, manifest_cid,
                                 tuple(item.limitation_cid for item in limitations))


__all__ = ["ChunkedReconstruction", "reconstruct_chunked_semantic_state"]
