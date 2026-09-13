"""Formal limitations are rebuilt from verified content, never nomination claims."""

from dataclasses import replace
import json
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state import chunked_reconstruction as r
from ipfs_datasets_py.logic.software_contracts.content import canonical_dag_json_bytes, cid_for_bytes
from ipfs_datasets_py.logic.software_contracts.semantic_index import chunked_snapshot as c
from ipfs_datasets_py.logic.software_contracts.semantic_state import verify_semantic_state_bundle
from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
    AnalysisLimitation, SemanticStateBundle, SortedPairIndex,
)


def git(root, *args):
    return subprocess.check_output(["git", "-c", "core.hooksPath=/dev/null", "-C", str(root), *args],
                                   stderr=subprocess.DEVNULL).decode().strip()


def commit(root):
    git(root, "add", "-A")
    git(root, "-c", "user.name=Fixture", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    return dict(expected_commit=git(root, "rev-parse", "HEAD"), expected_tree=git(root, "rev-parse", "HEAD^{tree}"),
                repository_id="fixture:formal-limitations")


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    git(root, "init", "-b", "main")
    (root / "module.py").write_bytes(b"def add(a, b):\n    return a + b\n")
    return root, commit(root)


def capture(root, request, frame_bytes=256):
    return c.snapshot_chunked_repository(root, **request, limits=c.ChunkedSnapshotLimits(frame_bytes=frame_bytes))


def reconstruct(root, request, chunked, **kwargs):
    return r.reconstruct_chunked_semantic_state(root, chunked, **request,
                                                expected_chunked_snapshot_cid=chunked.snapshot_cid,
                                                limits=kwargs.pop("limits", r.ReconstructionLimits(max_file_bytes=1024)),
                                                **kwargs)


@pytest.fixture
def oversized(source):
    root, _ = source
    (root / "coverage").mkdir()
    (root / "coverage/large.py").write_bytes(b"x = 1\n" * 1000)
    (root / "large.bin").write_bytes(bytes(range(256)) * 32)
    request = commit(root)
    return root, request, capture(root, request)


def formal_records(result):
    pairs = json.loads(result.bundle.blocks[result.bundle.root.analysis_limitation_index_cid])["pairs"]
    assert all(key == cid for key, cid in pairs)
    return [AnalysisLimitation.from_dict({**json.loads(result.bundle.blocks[cid]), "limitation_cid": cid})
            for _, cid in pairs]


def provenance_for(result, record):
    artifacts = dict(json.loads(result.bundle.blocks[result.bundle.root.artifact_fact_index_cid])["pairs"])
    fact = json.loads(result.bundle.blocks[artifacts[record.subject_id]])
    assert fact["artifact_id"] == record.subject_id
    return fact["artifact"]["metadata"]


def test_oversized_semantics_have_nonempty_provenance_bound_formal_index(oversized):
    root, request, chunked = oversized
    result = reconstruct(root, request, chunked)
    records = formal_records(result)
    assert len(records) == 2
    assert all(record.code == "chunked_projection.analysis_budget_exceeded" and record.confidence == "opaque"
               for record in records)
    by_raw = {provenance_for(result, record)["raw_path_hex"]: provenance_for(result, record) for record in records}
    for path in ("coverage/large.py", "large.bin"):
        evidence = by_raw[path.encode().hex()]
        assert evidence["source_cid"] == cid_for_bytes((root / path).read_bytes())
        assert evidence["git_object_oid"] == git(root, "rev-parse", "HEAD:" + path)
        assert evidence["git_commit"] == request["expected_commit"]
        assert evidence["git_tree"] == request["expected_tree"]
        assert evidence["population_cid"] == chunked.population_cid
        assert evidence["snapshot_cid"] == result.snapshot_cid
        assert evidence["chunked_snapshot_cid"] == chunked.snapshot_cid
        assert evidence["projection_limits"]["max_file_bytes"] == 1024
        assert evidence["source_content_verified"] is True
        assert evidence["semantics_analyzed"] is False
        assert result.bundle.blocks[evidence["opaque_artifact_fact_cid"]]
    assert result.bundle.root.producer.source_manifest_cid == chunked.snapshot_cid
    assert result.paged_snapshot_cid in result.bundle.blocks
    snapshot_fact = dict(json.loads(result.bundle.blocks[result.bundle.root.artifact_fact_index_cid])["pairs"])["artifact:snapshot-evidence"]
    reference = json.loads(result.bundle.blocks[snapshot_fact])["artifact"]["metadata"]["snapshot"]
    assert "entries" not in reference
    assert reference["paged_snapshot_cid"] == result.paged_snapshot_cid
    assert reference["snapshot_cid"] == result.snapshot_cid
    assert reference["entry_count"] == len(chunked.entries)
    assert all(result.bundle.blocks[cid] == data for cid, data in chunked.manifest_blocks()[1].items())
    observed = result.observation()
    assert observed["analysis_coverage"] == "incomplete"
    assert observed["analysis_limitation_count"] == 2
    assert observed["complete_analysis_authority"] is False
    assert observed["semantic_acceptance_authority"] is False
    assert observed["completion_authority"] is False
    again = reconstruct(root, request, chunked, nominated_bundle=result.bundle)
    assert again.nomination_matched is True
    assert again.bundle.blocks == result.bundle.blocks


def test_empty_known_limitation_index_is_never_complete_analysis_authority(source):
    root, request = source
    result = reconstruct(root, request, capture(root, request))
    assert formal_records(result) == []
    assert result.observation()["analysis_coverage"] == "not_established"
    assert result.observation()["complete_analysis_authority"] is False


def test_small_python_analysis_failure_is_also_formally_visible(source):
    root, _ = source
    (root / "module.py").write_bytes(b"def broken(:\n")
    request = commit(root)
    result = reconstruct(root, request, capture(root, request))
    records = formal_records(result)
    assert len(records) == 1
    assert records[0].code == "chunked_projection.scanner_analysis_opaque"
    assert provenance_for(result, records[0])["raw_path_hex"] == b"module.py".hex()


@pytest.mark.parametrize("omitted", ["limitation", "provenance", "opaque_fact", "manifest", "snapshot_page", "snapshot_evidence"])
def test_matching_root_with_missing_bound_leaves_is_refused(oversized, omitted):
    root, request, chunked = oversized
    first = reconstruct(root, request, chunked)
    record = formal_records(first)[0]
    evidence = provenance_for(first, record)
    blocks = dict(first.bundle.blocks)
    if omitted == "limitation":
        cid = record.limitation_cid
    elif omitted == "provenance":
        cid = dict(json.loads(blocks[first.bundle.root.artifact_fact_index_cid])["pairs"])[record.subject_id]
    elif omitted == "opaque_fact":
        cid = evidence["opaque_artifact_fact_cid"]
    elif omitted == "manifest":
        cid = chunked.snapshot_cid
    elif omitted == "snapshot_page":
        cid = json.loads(blocks[first.paged_snapshot_cid])["entry_pages"][0]
    else:
        cid = dict(json.loads(blocks[first.bundle.root.artifact_fact_index_cid])["pairs"])["artifact:snapshot-evidence"]
    del blocks[cid]
    stripped = SemanticStateBundle(first.bundle.root, blocks)
    with pytest.raises(r.ReconstructionError, match="omits or changes"):
        reconstruct(root, request, chunked, nominated_bundle=stripped)


def test_valid_root_with_cleared_formal_index_cannot_replace_cold_result(oversized):
    root, request, chunked = oversized
    first = reconstruct(root, request, chunked)
    empty = SortedPairIndex()
    forged_root = replace(first.bundle.root, analysis_limitation_index_cid=empty.index_cid)
    blocks = {**first.bundle.blocks, empty.index_cid: canonical_dag_json_bytes(empty.identity_payload())}
    cleared = SemanticStateBundle(forged_root, blocks)
    # Generic schema/CID validity is insufficient for this producer's exact
    # limitation obligations, even though the opaque content hashes remain.
    verify_semantic_state_bundle(cleared)
    with pytest.raises(r.ReconstructionError, match="formal analysis limitations"):
        reconstruct(root, request, chunked, nominated_bundle=cleared)


def test_builder_cannot_silently_drop_formal_records(oversized, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts import semantic_state
    root, request, chunked = oversized
    actual = semantic_state.build_semantic_state
    monkeypatch.setattr(semantic_state, "build_semantic_state", lambda index: actual(replace(index, limitations=())))
    with pytest.raises(r.ReconstructionError, match="formal analysis limitations"):
        reconstruct(root, request, chunked)


def test_scanner_cannot_silently_drop_projected_opaque_inputs(oversized, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts.semantic_index.scanner import RepositoryScanner
    root, request, chunked = oversized
    actual = RepositoryScanner.scan_snapshot
    def omit(*args, **kwargs):
        state = actual(*args, **kwargs)
        return replace(state, artifacts=tuple(a for a in state.artifacts if a.confidence != "opaque"))
    monkeypatch.setattr(RepositoryScanner, "scan_snapshot", omit)
    with pytest.raises(r.ReconstructionError, match="omitted a projected opaque input"):
        reconstruct(root, request, chunked)


def test_projection_limits_and_manifest_identity_change_formal_provenance(oversized):
    root, request, chunked = oversized
    original = reconstruct(root, request, chunked)
    bounded = reconstruct(root, request, chunked, limits=r.ReconstructionLimits(max_file_bytes=2048))
    reframed = reconstruct(root, request, capture(root, request, frame_bytes=512))
    assert len({x.bundle.root.analysis_limitation_index_cid for x in (original, bounded, reframed)}) == 3
    assert len({x.configuration_digest for x in (original, bounded, reframed)}) == 3


def test_unbound_manifest_is_rejected_before_content_projection(oversized, monkeypatch):
    root, request, chunked = oversized
    monkeypatch.setattr(c, "project_chunked_repository", lambda *args, **kwargs: pytest.fail("unbound projection"))
    with pytest.raises(r.ReconstructionError, match="requested snapshot CID"):
        r.reconstruct_chunked_semantic_state(root, chunked, **request,
                                             expected_chunked_snapshot_cid=cid_for_bytes(b"tampered"))


@pytest.mark.parametrize("limits,reason", [
    (r.ReconstructionLimits(max_total_bytes=1), "retained source"),
    (r.ReconstructionLimits(max_file_bytes=4194305), "fixed retained"),
    (r.ReconstructionLimits(max_entries=1), "metadata/entry"),
])
def test_existing_limits_remain_enforced(source, limits, reason):
    root, request = source
    with pytest.raises(r.ReconstructionError, match=reason):
        reconstruct(root, request, capture(root, request), limits=limits)


def test_gitlink_limitation_does_not_claim_nested_content_was_verified(source):
    root, request = source
    (root / "nested").mkdir()
    git(root, "update-index", "--add", "--cacheinfo", "160000", request["expected_commit"], "nested")
    request = commit(root)
    result = reconstruct(root, request, capture(root, request))
    records = formal_records(result)
    assert len(records) == 1
    evidence = provenance_for(result, records[0])
    assert evidence["git_mode"] == "160000"
    assert evidence["source_content_verified"] is False
    assert evidence["source_cid"] is None


def test_source_change_after_build_refuses_the_result(oversized, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts import semantic_state
    root, request, chunked = oversized
    actual = semantic_state.build_semantic_state
    def changed(*args, **kwargs):
        bundle = actual(*args, **kwargs)
        (root / "module.py").write_bytes(b"changed = True\n")
        return bundle
    monkeypatch.setattr(semantic_state, "build_semantic_state", changed)
    with pytest.raises(r.ReconstructionError, match="clean checkout"):
        reconstruct(root, request, chunked)


def test_projection_cannot_substitute_unverified_opaque_source_identity(oversized, monkeypatch):
    root, request, chunked = oversized
    actual = c.project_chunked_repository
    def altered(*args, **kwargs):
        projection = actual(*args, **kwargs)
        entries = tuple(replace(entry, source_cid=cid_for_bytes(b"unverified")) if entry.is_opaque else entry
                        for entry in projection.snapshot.entries)
        return replace(projection, snapshot=replace(projection.snapshot, entries=entries))
    monkeypatch.setattr(c, "project_chunked_repository", altered)
    with pytest.raises(r.ReconstructionError, match="source identity differs"):
        reconstruct(root, request, chunked)


def test_provenance_leaf_must_also_be_in_root_artifact_index(oversized):
    root, request, chunked = oversized
    first = reconstruct(root, request, chunked)
    subject = formal_records(first)[0].subject_id
    pairs = json.loads(first.bundle.blocks[first.bundle.root.artifact_fact_index_cid])["pairs"]
    reduced = SortedPairIndex([pair for pair in pairs if pair[0] != subject])
    changed_root = replace(first.bundle.root, artifact_fact_index_cid=reduced.index_cid)
    blocks = {**first.bundle.blocks, reduced.index_cid: canonical_dag_json_bytes(reduced.identity_payload())}
    unindexed = SemanticStateBundle(changed_root, blocks)
    with pytest.raises(r.ReconstructionError, match="subjects are not present"):
        reconstruct(root, request, chunked, nominated_bundle=unindexed)


def test_semantic_metadata_frame_limit_is_not_relaxed(oversized, monkeypatch):
    root, request, chunked = oversized
    monkeypatch.setattr(r, "MAX_BUNDLE_BLOCK_BYTES", 256)
    with pytest.raises(r.ReconstructionError, match="fixed metadata frame"):
        reconstruct(root, request, chunked)


def test_issuer_shaped_nomination_cannot_supply_analysis_authority(source):
    root, request = source
    with pytest.raises(r.ReconstructionError, match="datasets semantic-state bundle"):
        reconstruct(root, request, capture(root, request), nominated_bundle={"accepted": True})


def test_closed_manifest_bytes_and_object_form_rebuild_identically(oversized):
    root, request, chunked = oversized
    expected = reconstruct(root, request, chunked)
    cid, blocks = chunked.manifest_blocks()
    actual = r.reconstruct_chunked_semantic_state(root, blocks, **request,
                                                 expected_chunked_snapshot_cid=cid,
                                                 limits=r.ReconstructionLimits(max_file_bytes=1024))
    assert actual.bundle.blocks == expected.bundle.blocks
    assert actual.paged_snapshot_cid == expected.paged_snapshot_cid


def test_missing_chunk_page_refuses_before_content_projection(oversized, monkeypatch):
    root, request, chunked = oversized
    cid, blocks = chunked.manifest_blocks()
    broken = dict(blocks)
    del broken[json.loads(blocks[cid])["entry_pages"][0]]
    monkeypatch.setattr(c, "project_chunked_repository", lambda *args, **kwargs: pytest.fail("unadmitted content read"))
    with pytest.raises(r.ReconstructionError, match="reference is missing"):
        r.reconstruct_chunked_semantic_state(root, broken, **request, expected_chunked_snapshot_cid=cid)


def test_substituted_population_cannot_pass_closed_admission(oversized, monkeypatch):
    root, request, chunked = oversized
    forged = replace(chunked, entries=chunked.entries[1:])
    monkeypatch.setattr(c, "project_chunked_repository", lambda *args, **kwargs: pytest.fail("substituted population read"))
    with pytest.raises(r.ReconstructionError, match="population CID"):
        reconstruct(root, request, forged)


def test_streaming_admission_limit_must_be_explicitly_qualified(oversized, monkeypatch):
    root, request, chunked = oversized
    monkeypatch.setattr(c, "project_chunked_repository", lambda *args, **kwargs: pytest.fail("unqualified stream read"))
    with pytest.raises(r.ReconstructionError, match="qualified admission limits"):
        reconstruct(root, request, chunked, admission_limits=c.ChunkedSnapshotLimits(max_stream_bytes=1))


def test_captured_consumer_keeps_original_snapshot_evidence_schema(source):
    from ipfs_accelerate_py.agent_supervisor.semantic_state.reconstruction import reconstruct_semantic_state
    from ipfs_datasets_py.logic.software_contracts.semantic_index.snapshot import SNAPSHOT_SCHEMA
    root, request = source
    result = reconstruct_semantic_state(root, **request)
    pairs = dict(json.loads(result.bundle.blocks[result.bundle.root.artifact_fact_index_cid])["pairs"])
    snapshot = json.loads(result.bundle.blocks[pairs["artifact:snapshot-evidence"]])["artifact"]["metadata"]["snapshot"]
    assert snapshot["schema"] == SNAPSHOT_SCHEMA
    assert len(snapshot["entries"]) == 1


@pytest.mark.parametrize("attack", ["declared_limits", "entries", "chunks", "reference"])
def test_object_admission_bounds_apply_before_manifest_serialization(source, monkeypatch, attack):
    root, request = source
    chunked = capture(root, request)
    expected = chunked.snapshot_cid
    if attack == "declared_limits":
        chunked = replace(chunked, limits=replace(chunked.limits, max_metadata_bytes=2**40))
    elif attack == "entries":
        chunked = replace(chunked, entries=chunked.entries * 20001)
    elif attack == "chunks":
        blob = chunked.blobs[0]
        chunked = replace(chunked, blobs=(replace(blob, chunks=blob.chunks * 65537),))
    else:
        member = replace(chunked.entries[0], raw_path_hex="a" * (r.MAX_BUNDLE_BLOCK_BYTES + 1))
        chunked = replace(chunked, entries=(member,))
    def forbidden(*args, **kwargs):
        raise AssertionError("unqualified object reached manifest serialization")
    monkeypatch.setattr(c.ChunkedRepositorySnapshot, "manifest_blocks", forbidden)
    with pytest.raises(r.ReconstructionError, match="before serialization"):
        r.reconstruct_chunked_semantic_state(root, chunked, **request, expected_chunked_snapshot_cid=expected)
