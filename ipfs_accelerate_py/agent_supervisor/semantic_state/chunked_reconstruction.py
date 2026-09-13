"""Cold chunk projection with explicit, provenance-bound analysis limitations.

This consumer re-verifies Git content and rebuilds the limitation index. It
does not admit serialized projection claims or grant analysis/completion
authority. The existing captured-snapshot consumer remains unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import hashlib
from itertools import chain
import json
from pathlib import Path
import re
from typing import Any

from .reconstruction import Reconstruction, ReconstructionError, ReconstructionLimits

MAX_BUNDLE_BLOCK_BYTES = 1024 * 1024


class StreamingReconstructionError(ReconstructionError):
    """A structured bounded-analysis refusal, preserving producer diagnostics."""
    def __init__(self, refusal):
        super().__init__("streaming reconstruction refused: " + refusal["code"])
        self.refusal = dict(refusal)

    def observation(self):
        return dict(self.refusal)


@dataclass(frozen=True)
class ChunkedReconstruction(Reconstruction):
    chunked_snapshot_cid: str
    analysis_limitation_cids: tuple[str, ...]
    paged_snapshot_cid: str
    streaming_scan_observation: Any = None
    git_decoder_observation: Any = None

    def observation(self) -> dict[str, Any]:
        return {**super().observation(),
                "schema": ("ipfs_accelerate_py/chunked-semantic-reconstruction@4"
                           if self.git_decoder_observation is not None else
                           "ipfs_accelerate_py/chunked-semantic-reconstruction@3"
                           if self.streaming_scan_observation is not None else
                           "ipfs_accelerate_py/chunked-semantic-reconstruction@2"),
                "chunked_snapshot_cid": self.chunked_snapshot_cid,
                "paged_snapshot_cid": self.paged_snapshot_cid,
                "analysis_limitation_count": len(self.analysis_limitation_cids),
                "analysis_limitation_cids": list(self.analysis_limitation_cids),
                "analysis_coverage": "incomplete" if self.analysis_limitation_cids else "not_established",
                "complete_analysis_authority": False,
                **({"streaming_scan": self.streaming_scan_observation}
                   if self.streaming_scan_observation is not None else {}),
                **({"git_decoder": self.git_decoder_observation}
                   if self.git_decoder_observation is not None else {})}


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


def _bound_limitations(state, snapshot, chunked, limits, manifest_cid, paged_snapshot_cid):
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
            "schema": "ipfs_accelerate_py/chunked-analysis-limitation-provenance@2",
            "repository_id": snapshot.repository_id, "git_commit": snapshot.git_commit,
            "git_tree": snapshot.git_tree, "population_cid": chunked.population_cid,
            "chunked_snapshot_cid": manifest_cid, "snapshot_cid": snapshot_cid,
            "paged_snapshot_cid": paged_snapshot_cid,
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


def _verify_bound_bundle(bundle, producer, limitations, provenance, bound_artifacts, manifest_blocks, snapshot_blocks):
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
    required = {expected.index_cid: canonical_dag_json_bytes(expected.identity_payload()),
                **manifest_blocks, **snapshot_blocks}
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
    admission_limits: Any = None, streaming_limits: Any = None,
    decoder_profile: Any = None, decoder_budget: Any = None,
) -> ChunkedReconstruction:
    """Reproject the exact committed content, then bind every known opaque input.

    Projection preserves the fixed 4 MiB per-file and 128 MiB retained-source
    ceilings. Chunked work limits and producer revisions must separately be
    qualified by the native launcher. An empty limitation index is not proof
    of complete semantic analysis; this return value grants no such authority.
    """
    from ipfs_datasets_py.logic.software_contracts.semantic_index.chunked_snapshot import (
        ChunkedRepositorySnapshot, ChunkedSnapshotLimits, project_chunked_repository,
    )
    from ipfs_datasets_py.logic.software_contracts.semantic_index.git_decoder_profile import (
        DEFAULT_DECODER_PROFILE, DEFAULT_DECODER_BUDGET, admit_decoder_profile, require_decoder_profile,
    )
    from ipfs_datasets_py.logic.software_contracts.semantic_index.paged_snapshot import (
        admit_chunked_snapshot_manifest, page_snapshot_evidence, parse_paged_snapshot_evidence,
    )
    from ipfs_datasets_py.logic.software_contracts.semantic_index.streaming_scanner import (
        StreamingScanLimits, StreamingAnalysisError, scan_chunked_repository_streaming,
    )
    from ipfs_datasets_py.logic.software_contracts.semantic_index.committed_snapshot import preflight_committed_repository
    from ipfs_datasets_py.logic.software_contracts.semantic_index.scanner import RepositoryScanner
    from ipfs_datasets_py.logic.software_contracts.semantic_index.snapshot import SnapshotError
    from ipfs_datasets_py.logic.software_contracts.semantic_state import build_semantic_state
    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import SemanticStateBundle, SemanticStateProducer

    decoder_profile = DEFAULT_DECODER_PROFILE if decoder_profile is None else decoder_profile
    decoder_budget = DEFAULT_DECODER_BUDGET if decoder_budget is None else decoder_budget
    try:
        admit_decoder_profile(decoder_profile, decoder_budget)
    except SnapshotError as exc:
        raise ReconstructionError(str(exc)) from exc
    if any(not isinstance(oid, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", oid)
           for oid in (expected_commit, expected_tree)):
        raise ReconstructionError("request requires full commit and tree object ids")
    if not isinstance(repository_id, str) or not repository_id.strip():
        raise ReconstructionError("request requires an explicit repository identity")
    if not isinstance(limits, ReconstructionLimits) or not isinstance(chunked, (ChunkedRepositorySnapshot, Mapping)):
        raise ReconstructionError("chunk reconstruction requires typed limits and manifest")
    if isinstance(chunked, ChunkedRepositorySnapshot) and (
            chunked.repository_id, chunked.git_commit, chunked.git_tree) != (repository_id, expected_commit, expected_tree):
        raise ReconstructionError("chunked manifest differs from requested committed population")
    admission_limits = admission_limits if admission_limits is not None else ChunkedSnapshotLimits()
    if not isinstance(admission_limits, ChunkedSnapshotLimits):
        raise ReconstructionError("chunk reconstruction requires typed admission limits")
    if (admission_limits.max_entries > limits.max_entries
            or admission_limits.max_metadata_bytes > limits.max_metadata_bytes):
        raise ReconstructionError("chunked manifest exceeds reconstruction metadata/entry limits")
    if isinstance(chunked, ChunkedRepositorySnapshot):
        try:
            require_decoder_profile(chunked.decoder_profile, decoder_profile, decoder_budget)
        except SnapshotError as exc:
            raise ReconstructionError(str(exc)) from exc
        # Object callers must not serialize under their own unqualified budgets.
        if (not isinstance(chunked.limits, ChunkedSnapshotLimits)
                or any(getattr(chunked.limits, name) > value for name, value in asdict(admission_limits).items())
                or len(chunked.entries) > admission_limits.max_entries
                or len(chunked.blobs) > admission_limits.max_entries
                or sum(len(blob.chunks) for blob in chunked.blobs) > admission_limits.max_chunks):
            raise ReconstructionError("manifest object exceeds qualified admission limits before serialization")
        references = chain((chunked.repository_id, chunked.population_cid),
                           (entry.raw_path_hex for entry in chunked.entries),
                           (blob.source_cid for blob in chunked.blobs),
                           (frame.source_cid for blob in chunked.blobs for frame in blob.chunks))
        if any(type(value) is not str or len(value) > MAX_BUNDLE_BLOCK_BYTES for value in references):
            raise ReconstructionError("manifest object reference exceeds frame bounds before serialization")
    if streaming_limits is not None and not isinstance(streaming_limits, StreamingScanLimits):
        raise ReconstructionError("streaming reconstruction requires typed analysis limits")
    streaming = None
    root = Path(repository).resolve(strict=True)
    request = dict(expected_commit=expected_commit, expected_tree=expected_tree,
                   repository_id=repository_id, **asdict(limits))
    try:
        manifest_cid, manifest_blocks = (chunked.manifest_blocks() if isinstance(chunked, ChunkedRepositorySnapshot)
                                         else (expected_chunked_snapshot_cid, chunked))
        if manifest_cid != expected_chunked_snapshot_cid:
            raise ReconstructionError("chunked manifest differs from requested snapshot CID")
        chunked = admit_chunked_snapshot_manifest(root, manifest_cid, manifest_blocks,
                                                   repository_id=repository_id, expected_commit=expected_commit,
                                                   expected_tree=expected_tree, limits=admission_limits,
                                                   decoder_profile=decoder_profile, decoder_budget=decoder_budget)
        manifest_cid, manifest_blocks = chunked.manifest_blocks()
        if streaming_limits is None:
            projection = project_chunked_repository(root, chunked, max_file_bytes=limits.max_file_bytes,
                                                    max_total_bytes=limits.max_total_bytes,
                                                    decoder_profile=decoder_profile, decoder_budget=decoder_budget)
        else:
            streaming = scan_chunked_repository_streaming(root, chunked, max_file_bytes=limits.max_file_bytes,
                                                          limits=streaming_limits,
                                                          decoder_profile=decoder_profile, decoder_budget=decoder_budget)
            projection = streaming.projection
    except StreamingAnalysisError as exc:
        raise StreamingReconstructionError(exc.observation()) from exc
    except SnapshotError as exc:
        raise ReconstructionError(str(exc)) from exc
    snapshot = projection.snapshot
    if (snapshot.mode, snapshot.repository_id, snapshot.git_commit, snapshot.git_tree) != (
            "git-clean", repository_id, expected_commit, expected_tree):
        raise ReconstructionError("projected snapshot differs from committed request")
    if ({entry.raw_path_hex for entry in snapshot.entries} != {entry.raw_path_hex for entry in chunked.entries}
            or projection.chunked_snapshot_cid != manifest_cid):
        raise ReconstructionError("projected snapshot changes the exact committed population")
    members = {entry.raw_path_hex: entry for entry in chunked.entries}
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
    try:
        paged = page_snapshot_evidence(snapshot, manifest_cid, max_metadata_bytes=limits.max_metadata_bytes)
        _, paged = parse_paged_snapshot_evidence(
            paged.root_cid, paged.blocks, expected_snapshot_cid=snapshot.snapshot_cid,
            expected_source_manifest_cid=manifest_cid, repository_id=repository_id,
            expected_commit=expected_commit, expected_tree=expected_tree,
            max_entries=limits.max_entries, max_metadata_bytes=limits.max_metadata_bytes)
    except SnapshotError as exc:
        raise ReconstructionError(str(exc)) from exc
    state = (streaming.state if streaming is not None else RepositoryScanner(repository_id=repository_id).scan_snapshot(
        snapshot, {entry.source_key: entry.captured_bytes for entry in snapshot.entries
                   if entry.captured_bytes is not None and not entry.is_opaque}))
    # Keep the existing scanner/captured schema unchanged. Replace its
    # acquisition artifact before any semantic-state blocks are serialized.
    evidence = paged.artifact()
    if sum(artifact.artifact_id == evidence.artifact_id for artifact in state.artifacts) != 1:
        raise ReconstructionError("scanner acquisition evidence is missing or duplicated")
    state = replace(state, artifacts=tuple(artifact for artifact in state.artifacts
                                          if artifact.artifact_id != evidence.artifact_id) + (evidence,))
    state, limitations, provenance, bound_artifacts = _bound_limitations(
        state, snapshot, chunked, limits, manifest_cid, paged.root_cid)
    bound_artifacts = (*bound_artifacts, evidence)
    producer = SemanticStateProducer(state.state_cid, snapshot.snapshot_cid, expected_commit, expected_tree,
                                     manifest_cid, state.schema, state.extractor_name, state.extractor_version)
    bundle = build_semantic_state(_LimitedIndex(state, producer, limitations))
    blocks = dict(bundle.blocks)
    for cid, data in chain(manifest_blocks.items(), paged.blocks.items()):
        if cid in blocks and blocks[cid] != data:
            raise ReconstructionError("conflicting chunk manifest block")
        blocks[cid] = data
    bundle = SemanticStateBundle(bundle.root, blocks)
    if streaming is not None and any(len(data) > MAX_BUNDLE_BLOCK_BYTES for data in blocks.values()):
        raise StreamingReconstructionError(StreamingAnalysisError(
            "semantic_bundle_frame", phase="bundle", limit=MAX_BUNDLE_BLOCK_BYTES).observation())
    _verify_bound_bundle(bundle, producer, limitations, provenance, bound_artifacts, manifest_blocks, paged.blocks)
    matched = None
    if nominated_bundle is not None:
        nominated = _verify_bound_bundle(nominated_bundle, producer, limitations, provenance,
                                         bound_artifacts, manifest_blocks, paged.blocks)
        if nominated.root_cid != bundle.root.root_cid:
            raise ReconstructionError("nominated root differs from cold chunk reconstruction")
        matched = True
    try:
        after = preflight_committed_repository(root, **request)
    except SnapshotError as exc:
        raise ReconstructionError(str(exc)) from exc
    if after.population_cid != chunked.population_cid:
        raise ReconstructionError("committed population changed during chunk reconstruction")
    configuration = {"schema": "ipfs_accelerate_py/chunked-reconstruction-config@2",
                     "repository_id": repository_id, "population_cid": chunked.population_cid,
                     "chunked_snapshot_cid": manifest_cid, "limits": asdict(limits),
                     "paged_snapshot_cid": paged.root_cid, "admission_limits": asdict(admission_limits),
                     "stream_limits": asdict(chunked.limits), "population_scope": "complete-committed",
                     "analysis_limitation_index_cid": bundle.root.analysis_limitation_index_cid,
                     "reuse": "cold", "environment_bindings": []}
    if streaming is not None:
        configuration.update(schema="ipfs_accelerate_py/chunked-reconstruction-config@3",
                             analysis_mode="streaming-ordinary", streaming_limits=asdict(streaming_limits),
                             analysis_process_profile=streaming.observation["analysis_process_profile"])
    decoder_observation = None
    if decoder_profile != DEFAULT_DECODER_PROFILE or decoder_budget != DEFAULT_DECODER_BUDGET:
        decoder_observation = {"profile": decoder_profile.payload(), "caller_budget": asdict(decoder_budget)}
        configuration.update(schema="ipfs_accelerate_py/chunked-reconstruction-config@4",
                             git_decoder=decoder_observation)
    digest = "sha256:" + hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return ChunkedReconstruction(bundle, snapshot.snapshot_cid, state.state_cid, expected_commit,
                                 expected_tree, digest,
                                 tuple((entry.path, entry.opaque_reason) for entry in snapshot.entries if entry.is_opaque),
                                 matched, chunked.population_cid, manifest_cid,
                                 tuple(item.limitation_cid for item in limitations), paged.root_cid,
                                 streaming.observation if streaming is not None else None, decoder_observation)


__all__ = ["ChunkedReconstruction", "StreamingReconstructionError", "reconstruct_chunked_semantic_state"]
