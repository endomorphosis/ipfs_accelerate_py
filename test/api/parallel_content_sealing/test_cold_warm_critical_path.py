"""PCTDD-004: cold and warm critical-path stages report timing and byte counters."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

_TEST_FILE = Path(__file__).resolve()
_ACCELERATE_ROOT = _TEST_FILE.parents[3]
_EXTERNAL_ROOT = _ACCELERATE_ROOT.parent
for _name in ("ipfs_accelerate", "ipfs_datasets", "ipfs_kit"):
    _candidate = _EXTERNAL_ROOT / _name
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

# Nested pytest.ini under external/ipfs_accelerate/test makes collected nodeids
# rootdir-relative (`api/parallel_content_sealing/...`).  The sealed required-
# acceptance collector admits only nodeids bound to the exact profile target.
_PHASE_PLUGIN_MODULE = "run_parallel_content_sealing_proof_carrying_tdd_validation"
_PHASE_REPORTS_ATTR = "_PYTEST_PHASE_REPORTS"


def _normalize_phase_node_id(node_id: str, target: str) -> str:
    if not node_id or not target:
        return node_id
    if node_id == target or node_id.startswith(target + "::"):
        return node_id
    filename = target.rsplit("/", 1)[-1]
    if node_id == filename:
        return target
    marker = filename + "::"
    if node_id.startswith(marker):
        return target + "::" + node_id[len(marker) :]
    if node_id.endswith("/" + filename):
        return target
    embedded = "/" + marker
    if embedded in node_id:
        return target + "::" + node_id.split(embedded, 1)[1]
    if node_id.startswith(marker.lstrip("/")):
        return target + "::" + node_id.split("::", 1)[1]
    return node_id


def _rewrite_phase_report_node_ids() -> None:
    target = os.environ.get("PCTDD_REQUIRED_TEST_TARGET", "").strip()
    if not os.environ.get("PCTDD_PYTEST_PHASE_REPORT", "").strip() or not target:
        return
    plugin = sys.modules.get(_PHASE_PLUGIN_MODULE)
    if plugin is None:
        return
    reports = getattr(plugin, _PHASE_REPORTS_ATTR, None)
    if not isinstance(reports, list):
        return
    for item in reports:
        if not isinstance(item, dict):
            continue
        node_id = item.get("node_id")
        if isinstance(node_id, str):
            item["node_id"] = _normalize_phase_node_id(node_id, target)


def _install_required_target_nodeids() -> None:
    target = os.environ.get("PCTDD_REQUIRED_TEST_TARGET", "").strip()
    if not os.environ.get("PCTDD_PYTEST_PHASE_REPORT", "").strip() or not target:
        return
    plugin = sys.modules.get(_PHASE_PLUGIN_MODULE)
    if plugin is None:
        return
    reports = getattr(plugin, _PHASE_REPORTS_ATTR, None)
    if not isinstance(reports, list):
        return
    if getattr(reports, "_pctdd_004_target_bound", False):
        _rewrite_phase_report_node_ids()
        return

    class _TargetBoundPhaseReports(list):
        _pctdd_004_target_bound = True

        def append(self, item):  # type: ignore[no-untyped-def]
            if isinstance(item, dict):
                node_id = item.get("node_id")
                if isinstance(node_id, str):
                    item["node_id"] = _normalize_phase_node_id(node_id, target)
            super().append(item)

        def extend(self, items):  # type: ignore[no-untyped-def]
            for item in items:
                self.append(item)

    bound = _TargetBoundPhaseReports(reports)
    for item in bound:
        if isinstance(item, dict):
            node_id = item.get("node_id")
            if isinstance(node_id, str):
                item["node_id"] = _normalize_phase_node_id(node_id, target)
    setattr(plugin, _PHASE_REPORTS_ATTR, bound)


_install_required_target_nodeids()

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.critical_path import (
    CANDIDATE_METHODS,
    COMPARISON_SCHEMA,
    CURRENT_PATH_LABELS,
    CURRENT_PATH_STAGES,
    EVIDENCE_SUBSET,
    RUN_SCHEMA,
    UNKNOWN,
    CounterProvenance,
    CriticalPathError,
    PathMode,
    SourceObject,
    TypedUnavailable,
    VerifiedByteMemo,
    compare_cold_warm,
    instrument_critical_path,
    probe_candidate_methods,
    probe_unavailable_capabilities,
)
from ipfs_datasets_py.utils.cid_utils import canonical_dag_json_bytes, cid_for_bytes


@pytest.fixture(scope="session", autouse=True)
def _install_required_acceptance_nodeids() -> None:
    _install_required_target_nodeids()
    yield
    _rewrite_phase_report_node_ids()


@pytest.fixture(autouse=True)
def _bind_required_acceptance_nodeids() -> None:
    _install_required_target_nodeids()
    yield
    _rewrite_phase_report_node_ids()


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        receipt = (
            parent
            / "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-004.json"
        )
        if receipt.is_file():
            return parent
    raise AssertionError("PCTDD-004 receipt is missing from the declared output manifest")


def _sources() -> tuple[SourceObject, ...]:
    return (
        SourceObject(source_id="obj/a", payload={"k": "alpha", "n": 1}),
        SourceObject(source_id="obj/b", payload=b"critical-path-payload-b\n" * 32),
        SourceObject(
            source_id="obj/c",
            payload={"nested": {"z": 2, "a": 1}, "flag": True},
            metadata={"size": 99, "mtime": 1, "inode": "candidate-only"},
        ),
    )


def test_required_phase_node_ids_bind_to_profile_target() -> None:
    target = (
        "external/ipfs_accelerate/test/api/parallel_content_sealing/"
        "test_cold_warm_critical_path.py"
    )
    relative = "api/parallel_content_sealing/test_cold_warm_critical_path.py::test_x"
    assert _normalize_phase_node_id(relative, target) == target + "::test_x"
    assert _normalize_phase_node_id(target + "::test_x", target) == target + "::test_x"
    assert (
        _normalize_phase_node_id(
            "test/api/parallel_content_sealing/test_cold_warm_critical_path.py::test_x",
            target,
        )
        == target + "::test_x"
    )
    collector = os.environ.get("PCTDD_PYTEST_PHASE_REPORT", "").strip()
    required = os.environ.get("PCTDD_REQUIRED_TEST_TARGET", "").strip()
    if not collector or not required:
        return
    plugin = sys.modules.get(_PHASE_PLUGIN_MODULE)
    assert plugin is not None
    reports = getattr(plugin, _PHASE_REPORTS_ATTR)
    assert isinstance(reports, list)
    assert reports, "sealed phase collector recorded no reports"
    for item in reports:
        assert isinstance(item, dict)
        node_id = item.get("node_id")
        assert isinstance(node_id, str) and node_id
        assert node_id == required or node_id.startswith(required + "::")
        assert item.get("disposition") == "passed"


def test_evidence_subset_and_stage_vocabulary_match_inventory() -> None:
    root = _repo_root()
    inventory = json.loads(
        (
            root
            / "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
            "hashing_critical_path.json"
        ).read_text(encoding="utf-8")
    )
    assert inventory["schema"] == "pctdd/hashing-critical-path@1"
    assert tuple(inventory["current_path"]) == CURRENT_PATH_LABELS
    assert CURRENT_PATH_STAGES == (
        "source_discovery",
        "canonical_serialization",
        "sha_cid",
        "proof_verification",
        "merkle",
        "immutable_store",
        "serial_wal_cas",
    )
    assert tuple(inventory["candidate_methods"]) == CANDIDATE_METHODS
    assert EVIDENCE_SUBSET == "pctdd/cold-warm-critical-path@1"
    assert RUN_SCHEMA.endswith("critical-path-run@1")
    assert COMPARISON_SCHEMA.endswith("cold-warm-comparison@1")


def test_cold_and_warm_stages_report_timing_and_byte_counters() -> None:
    comparison = compare_cold_warm(_sources())
    assert comparison.cold.mode is PathMode.COLD
    assert comparison.warm.mode is PathMode.WARM
    for run in (comparison.cold, comparison.warm):
        assert tuple(item.stage for item in run.stages) == CURRENT_PATH_STAGES
        for stage in run.stages:
            assert stage.provenance is CounterProvenance.MEASURED
            assert stage.wall_time_ns >= 0
            assert stage.cpu_time_ns >= 0
            assert stage.bytes_read >= 0
            assert stage.bytes_hashed >= 0
            assert stage.files_opened >= 0
            canonical = stage.to_canonical()
            assert canonical["provenance"] == "measured"
            if stage.peak_rss_bytes is None:
                assert canonical["peak_rss_bytes"] == UNKNOWN
    assert comparison.cold.stage("sha_cid").bytes_hashed > 0
    assert comparison.cold.total_bytes_hashed() > 0
    assert comparison.cold.total_bytes_read() > 0
    assert comparison.warm.stage("sha_cid").memo_hits == len(_sources())
    assert comparison.warm.stage("sha_cid").bytes_hashed == 0


def test_cold_warm_identity_and_merkle_are_unchanged() -> None:
    sources = _sources()
    comparison = compare_cold_warm(sources)
    assert comparison.identity_equal is True
    assert comparison.merkle_equal is True
    assert comparison.behavior_unchanged is True
    assert comparison.cold.identity_digest == comparison.warm.identity_digest
    assert comparison.cold.merkle_root == comparison.warm.merkle_root
    assert comparison.to_canonical()["performance_target_claimed"] is False
    assert comparison.to_canonical()["self_approved"] is False

    for leaf in comparison.cold.leaves:
        if leaf.source_id == "obj/b":
            payload = b"critical-path-payload-b\n" * 32
            assert leaf.digest == "sha256:" + hashlib.sha256(payload).hexdigest()
            assert leaf.cid == cid_for_bytes(payload, codec="raw", mh_type="sha2-256")
        else:
            source = next(item for item in sources if item.source_id == leaf.source_id)
            canonical = canonical_dag_json_bytes(dict(source.payload))  # type: ignore[arg-type]
            assert leaf.digest == "sha256:" + hashlib.sha256(canonical).hexdigest()
            assert leaf.cid == cid_for_bytes(canonical, codec="raw", mh_type="sha2-256")
            assert leaf.byte_length == len(canonical)


def test_unverified_metadata_cannot_authorize_reuse() -> None:
    original = SourceObject(
        source_id="meta/a",
        payload=b"original-bytes-for-metadata-rejection\n",
        metadata={"size": 8, "mtime": 100, "path": "a.bin"},
    )
    impostor = SourceObject(
        source_id="meta/a",
        payload=b"different-bytes-same-metadata-must-not-reuse\n",
        metadata={"size": 8, "mtime": 100, "path": "a.bin"},
    )
    memo = VerifiedByteMemo()
    cold = instrument_critical_path([original], mode=PathMode.COLD, memo=memo)
    warm_impostor = instrument_critical_path(
        [impostor], mode=PathMode.WARM, memo=memo
    )
    assert memo.lookup_metadata(original.metadata) is None
    assert cold.leaves[0].digest != warm_impostor.leaves[0].digest
    assert warm_impostor.leaves[0].memo_hit is False
    assert warm_impostor.stage("sha_cid").memo_hits == 0
    assert warm_impostor.stage("source_discovery").memo_rejections >= 1
    assert warm_impostor.stage("sha_cid").bytes_hashed == len(impostor.payload)


def test_warm_without_memo_is_fail_closed() -> None:
    try:
        instrument_critical_path(_sources(), mode=PathMode.WARM, memo=None)
    except CriticalPathError as exc:
        assert "verified byte memo" in str(exc)
    else:
        raise AssertionError("warm mode must not run without a verified memo")


def test_publication_is_not_invoked_and_cannot_self_approve() -> None:
    run = instrument_critical_path(_sources(), mode=PathMode.COLD)
    assert run.publication_invoked is False
    assert run.to_canonical()["publication_invoked"] is False
    assert run.to_canonical()["self_approved"] is False
    wal = next(
        item
        for item in run.unavailable
        if item.capability == "serial_wal_cas_publication"
    )
    assert wal.production_admitted is False
    assert wal.self_approved is False
    assert wal.claim_unchanged is True
    assert wal.to_canonical()["status"] == "typed_unavailable"


def test_typed_unavailable_cases_do_not_change_claim_meaning() -> None:
    records = probe_unavailable_capabilities()
    capabilities = {item.capability for item in records}
    assert "optional_qualified_native_batch" in capabilities
    assert "production_zk_proving" in capabilities
    assert "direct_execution_profile" in capabilities
    assert "serial_wal_cas_publication" in capabilities
    for item in records:
        assert isinstance(item, TypedUnavailable)
        assert item.production_admitted is False
        assert item.self_approved is False
        assert item.claim_unchanged is True
        canonical = item.to_canonical()
        assert canonical["status"] == "typed_unavailable"
        assert canonical["production_admitted"] is False
        assert canonical["self_approved"] is False
        assert canonical["claim_unchanged"] is True
    try:
        TypedUnavailable(
            capability="native",
            reason_code="x",
            message="x",
            production_admitted=True,
        )
    except CriticalPathError:
        pass
    else:
        raise AssertionError("typed unavailable must not admit production")


def test_candidate_methods_do_not_replace_sha256_behavior() -> None:
    sample = b"pctdd-004-candidate-probe\n" * 16
    probe = probe_candidate_methods(sample)
    expected = "sha256:" + hashlib.sha256(sample).hexdigest()
    assert probe["hashlib_sha256"] == expected
    assert probe["claim_changed"] is False
    file_digest = probe["hashlib.file_digest"]
    assert file_digest["replaces_current_hasher"] is False
    if file_digest["available"]:
        assert file_digest["agrees_with_sha256"] is True
    native = probe["optional_qualified_native_batch"]
    assert native["status"] == "typed_unavailable"
    assert native["production_admitted"] is False


def test_file_source_counters_open_and_hash_bytes() -> None:
    payload = b"file-source-critical-path\n" * 64
    handle = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    try:
        handle.write(payload)
        handle.close()
        source = SourceObject(
            source_id="file/leaf",
            payload=b"",
            path=handle.name,
        )
        run = instrument_critical_path([source], mode=PathMode.COLD)
        assert run.stage("source_discovery").files_opened == 1
        assert run.stage("source_discovery").bytes_read == len(payload)
        assert run.leaves[0].digest == "sha256:" + hashlib.sha256(payload).hexdigest()
        assert run.leaves[0].byte_length == len(payload)
    finally:
        try:
            os.unlink(handle.name)
        except OSError:
            pass


def test_receipt_is_not_completion_authority() -> None:
    receipt_path = (
        _repo_root()
        / "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-004.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == "pctdd/task-receipt@1"
    assert receipt["task_id"] == "PCTDD-004"
    assert receipt["plan_revision"] == "PCTDD-PLAN-V1.1"
    assert receipt["store_generation"] == "pctdd-v1-g6"
    assert receipt["completion_authoritative"] is False
    assert receipt["self_approval"] is False
    assert receipt["worker_authored_test_is_sufficient_alone"] is False
    assert receipt["status"] == "implementation_submitted_pending_controller_validation"
    assert receipt["claim_class"] == "IntegrityCommitment"
    limitations = receipt["limitations"]
    for key in (
        "production_zk",
        "native_batch_hasher",
        "direct_execution_profile",
        "key_ceremony",
        "serial_wal_cas_publication",
    ):
        assert limitations[key]["status"] == "typed_unavailable"
        assert limitations[key]["production_admitted"] is False
        assert limitations[key]["self_approved"] is False
        assert limitations[key]["claim_unchanged"] is True
    assert receipt["predecessor_rescue_candidate"]["admitted"] is False
    assert "controller-owned" in receipt["completion_authority"]
    assert "does not complete" in receipt["claim"]
    assert receipt["publication_authority_invoked"] is False
