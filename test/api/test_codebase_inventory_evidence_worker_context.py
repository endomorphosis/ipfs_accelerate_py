"""Receiving the genuine pure native completed-scan reference contract."""
from copy import deepcopy
import hashlib

import pytest

from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes, cid_for_structured
from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_projection_features as features
from ipfs_accelerate_py.agent_supervisor.runtime import codebase_inventory_evidence_worker_context as worker
from ipfs_accelerate_py.agent_supervisor.runtime import codebase_inventory_evidence_admission as joined


@pytest.fixture
def native_refs():
    """Authored typed historical metadata; no CAS, owner, inference or fitting."""
    source, model_bytes = b"VALUE = 1\n", b"authored-model-identity"
    snapshot = cid_for_structured({"authored-snapshot": 1})
    head = CodebaseHead(repository_id="repository:authored-history", generation=1,
        manifest_cid=cid_for_structured({"authored-manifest": 1}), snapshot_cid=snapshot,
        ast_revision_id="rev:repository:authored-history:snapshot:" + snapshot,
        receipt_cid=cid_for_structured({"authored-receipt": 1})).to_dict()
    raw_hex = b"input.py".hex()
    member = {"source_key": "raw:" + raw_hex, "path": "input.py", "raw_path_hex": raw_hex,
        "entry_cid": cid_for_structured({"authored-entry": 1}), "source_cid": cid_for_bytes(source),
        "ast_cid": None, "parse_status": "unindexed", "source_size_bytes": len(source), "opaque_reason": None}
    artifact = {"sha256": hashlib.sha256(model_bytes).hexdigest(), "bytes": len(model_bytes)}
    model = {"version_id": "version:authored-history", "variant_id": "variant:authored-history",
        "artifact": artifact, "artifact_cid": cid_for_bytes(model_bytes),
        "contract_sha256": "a" * 64, "state_sha256": "b" * 64, "feature_space_sha256": "c" * 64,
        "latent_width": 8, "feature_columns": 1, "projection_ids": ["projection:authored"],
        "projection_widths": {"projection:authored": 1},
        "ancestry": [{"version_id": "version:authored-history", "artifact": artifact}]}
    files = {"authored-test-producer": "d" * 64}
    root_body = {"schema": resume.ROOT_SCHEMA, "head": head, "head_cid": cid_for_structured(head),
        "members": [member], "membership_cid": cid_for_structured([member]), "model": model,
        "limits": resume.CodebaseScanResumeLimits(max_inventory_entries=1, page_entries=1, max_pages=1,
            max_inferred_rows=1).to_dict(), "optimized": True,
        "implementation": {"files": files, "sha256": features.digest(files),
            "scope": "listed_local_files_only_not_execution_attestation"},
        "authority": {name: False for name in worker.AUTHORITY_NAMES}}
    root = resume.CodebaseScanResumeRoot.from_dict(cid_for_structured(root_body), root_body)
    complete_body = {"schema": resume.COMPLETION_SCHEMA, "root_cid": root.artifact_cid,
        "head_cid": root_body["head_cid"], "membership_cid": root_body["membership_cid"],
        "model_artifact_cid": model["artifact_cid"],
        "pages": [{"page_cid": cid_for_bytes(b"authored-page"), "start": 0, "end": 1,
            "membership_cid": root_body["membership_cid"], "inferred_rows": 0, "dispositions": {"unindexed": 1}}],
        "coverage": {"inventory_entries": 1, "inferred_rows": 0, "pages": 1, "dispositions": {"unindexed": 1}},
        "authority": {name: False for name in worker.AUTHORITY_NAMES}}
    complete = resume.CodebaseScanResumeCompletion.from_dict(cid_for_structured(complete_body), complete_body)
    return {"schema": worker.SCHEMA, "scan": complete.advisory_refs(root), "evidence": None,
        "authority": {name: False for name in worker.AUTHORITY_NAMES}}


def test_native_completed_refs_stay_detached_and_advisory(native_refs):
    sources = {"input.py": {"sha256": "e" * 64, "executable": False}}
    original = deepcopy(native_refs)
    validated = worker.validate_inventory_context(native_refs, sources=sources)
    assert validated == original and validated is not native_refs
    assert validated["scan"]["coverage"] == {"inventory_entries": 1, "inferred_rows": 0,
        "pages": 1, "dispositions": {"unindexed": 1}}
    assert all(flag is False for flag in validated["authority"].values())
    validated["scan"]["model"]["state_sha256"] = "f" * 64
    assert native_refs == original


@pytest.mark.parametrize("change", ["completion", "root", "model", "coverage", "page", "authority"])
def test_public_reference_substitution_fails_native_historical_replay(native_refs, change):
    value = deepcopy(native_refs)
    scan = value["scan"]
    if change in {"completion", "root"}:
        scan[change + "_cid"] = cid_for_structured({"foreign": change})
    elif change == "model":
        scan["model"]["state_sha256"] = "f" * 64
    elif change == "coverage":
        scan["coverage"]["inferred_rows"] = 1
    elif change == "page":
        scan["pages"][0]["page_cid"] = cid_for_bytes(b"foreign-page")
    else:
        scan["authority"]["proof_authority"] = True
    with pytest.raises(ValueError):
        worker.validate_inventory_context(value)


def test_scan_population_cannot_omit_signed_baseline_member(native_refs):
    with pytest.raises(ValueError, match="complete scan member population"):
        worker.validate_inventory_context(native_refs, sources={"foreign.py": {"sha256": "a" * 64,
                                                                               "executable": False}})


@pytest.mark.parametrize("change", ["full_context_cid", "member_paths", "completion_cid", "coverage", "authority"])
def test_compact_declaration_cannot_substitute_full_scan_refs(native_refs, change):
    declaration = worker.inventory_declaration(native_refs)
    if change == "full_context_cid":
        declaration[change] = cid_for_structured({"foreign-context": 1})
    elif change == "member_paths":
        declaration["scan"][change] = ["foreign.py"]
    elif change == "completion_cid":
        declaration["scan"][change] = cid_for_structured({"foreign-completion": 1})
    elif change == "coverage":
        declaration["scan"][change]["inventory_entries"] = 2
    else:
        declaration[change]["admission_authority"] = 0
    with pytest.raises(ValueError):
        worker.validate_inventory_declaration(declaration, context=native_refs)


def test_owner_receiver_calls_share_one_real_resource_lease(native_refs, tmp_path, monkeypatch):
    from contextlib import contextmanager
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_receiving as receiving
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
        GlobalResourceScheduler, ResourceSchedulerConfig,
    )
    scan = native_refs["scan"]
    root = resume.CodebaseScanResumeRoot.from_dict(scan["root_cid"], scan["root_record"])
    completion = resume.CodebaseScanResumeCompletion.from_dict(scan["completion_cid"], scan["completion_record"])
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig(state_path=tmp_path / "resources.json",
        total_cpu_slots=2, total_memory_mb=2048, total_child_process_slots=2,
        lane_reservations={}, auto_renew_leases=False))
    calls = []
    @contextmanager
    def receiver(record, index, repository, **arguments):
        # Explicit receiving-protocol double, not native freshness qualification.
        assert record is completion and arguments["root"] is root
        assert not arguments["parent_lease"].released
        assert arguments["timeout_seconds"] > 0
        calls.append(arguments["parent_lease"])
        def close():
            calls.append(arguments["parent_lease"])
            return record
        yield close
    monkeypatch.setattr(receiving, "_paired_current_codebase_scan_completion", receiver)
    with joined._current_scope(root=root, completion=completion, index=object(), repository=tmp_path,
            registry=object(), scheduler=scheduler, timeout_seconds=5) as (context, current):
        assert context == worker.inventory_declaration(native_refs)
        current()
    assert len(calls) == 2 and calls[0] is calls[1]
    assert all(lease.released for lease in calls)


def test_pre_cancelled_inventory_owner_never_calls_receiver(native_refs, tmp_path, monkeypatch):
    import threading
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError
    scan = native_refs["scan"]
    root = resume.CodebaseScanResumeRoot.from_dict(scan["root_cid"], scan["root_record"])
    completion = resume.CodebaseScanResumeCompletion.from_dict(scan["completion_cid"], scan["completion_record"])
    signal = threading.Event()
    signal.set()
    def forbidden(*_, **__):
        raise AssertionError("pre-cancelled receiver executed")
    monkeypatch.setattr(resume, "validate_current_codebase_scan_completion", forbidden)
    with pytest.raises(LeaseCancelledError):
        with joined._current_scope(root=root, completion=completion, index=object(), repository=tmp_path,
                                   registry=object(), cancel_event=signal):
            raise AssertionError("cancelled owner entered")
