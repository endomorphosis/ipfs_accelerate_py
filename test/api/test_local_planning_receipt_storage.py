"""Large native plans retain complete signed receipts in bounded owner storage."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario as scenario
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import (
    original as original, _proposal_graph,
)
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    IntentRepository, MAX_BODY_BYTES,
)


def _stored(scenario):
    admission = local.admit_local_benchmark_plan(
        graph=scenario["graph"], manifest=scenario["manifest"],
    )
    result = local.materialize_local_benchmark_plan(admission=admission, intent=scenario["intent"])
    reference = scenario["intent"].get_plan(result["plan_id"])["body"]["local_planning_receipt_ref"]
    path = scenario["lifecycle"] / "local-planning-receipts" / (reference["sha256"] + ".json")
    return admission, reference, path


def test_native_plan_projects_exact_pending_reference_and_original_task_contract(scenario):
    admission, reference, path = _stored(scenario)
    assert local.load_local_planning_receipt(reference, manifest=admission["manifest"]) == admission["receipt"]
    assert reference["pending_requirements"] == admission["receipt"]["payload"]["pending_requirements"]
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    task = scenario["intent"].get_task(scenario["task_cid"])
    contract, _, _, _ = local._contract(task["body"], task["task_cid"])
    assert contract["manifest"] == admission["manifest"]
    assert contract["planning_receipt_cid"] == local.content_identity(admission["receipt"])
    projected = scenario["intent"].plan_projection(task_cids=[task["task_cid"]])
    assert projected["plans"][0]["body"]["local_planning_receipt_ref"] == reference
    assert reference["completion_authority"] is False


@pytest.mark.timeout(15)
@pytest.mark.parametrize("mutation", ["missing", "bytes", "symlink", "fifo", "public-mode", "wrong-pending", "wrong-root", "alias-type", "foreign-path"])
def test_receipt_reference_rejects_missing_tampered_or_rebound_material(scenario, mutation):
    admission, reference, path = _stored(scenario)
    reference = deepcopy(reference)
    if mutation == "missing":
        path.unlink()
    elif mutation == "bytes":
        raw = path.read_bytes()
        path.write_bytes(raw.replace(b'planning_permitted', b'planning-forbidden', 1))
    elif mutation == "symlink":
        target = path.with_suffix(".other")
        path.rename(target)
        path.symlink_to(target)
    elif mutation == "fifo":
        path.unlink()
        os.mkfifo(path, 0o600)
    elif mutation == "public-mode":
        path.chmod(0o644)
    elif mutation == "wrong-pending":
        reference["pending_requirements"] = []
    elif mutation == "wrong-root":
        reference["source_tree_id"] = local.content_identity({"foreign": True})
    elif mutation == "alias-type":
        reference["completion_authority"] = 0
    else:
        reference["path"] = "/tmp/foreign-receipt.json"
    with pytest.raises((ValueError, OSError)):
        local.load_local_planning_receipt(reference, manifest=admission["manifest"])


def test_even_rehashed_receipt_requires_original_owner_signature(scenario):
    admission, reference, path = _stored(scenario)
    forged = deepcopy(admission["receipt"])
    forged["payload"]["pending_requirements"] = []
    raw = local._receipt_bytes(forged)
    # Updating every untrusted artifact nomination cannot sign altered content.
    reference = local._receipt_reference(forged, raw)
    path = path.parent / (hashlib.sha256(raw).hexdigest() + ".json")
    path.write_bytes(raw)
    path.chmod(0o600)
    with pytest.raises(ValueError):
        local.load_local_planning_receipt(reference, manifest=admission["manifest"])


def test_receipt_cannot_be_loaded_against_different_manifest_signed_by_same_owner(scenario):
    admission, reference, _ = _stored(scenario)
    payload = deepcopy(admission["manifest"]["payload"])
    payload["tasks"][0]["acceptance"][0]["criterion"] = "A different declared acceptance"
    foreign = local._signed(payload, payload)
    local._manifest(foreign)
    with pytest.raises(ValueError, match="bindings differ"):
        local.load_local_planning_receipt(reference, manifest=foreign)


def test_220_source_native_admission_materializes_without_raising_body_bound(original):
    root, instruction, state = original
    inventory = root / "public-inputs"
    inventory.mkdir()
    for index in range(217):
        (inventory / f"document-{index:03}.txt").write_text("Independent public input\n")
    subprocess.run(["git", "-C", str(root), "add", "public-inputs"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "full inventory"], check=True)
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    assert len(prepared["manifest"]["payload"]["sources"]) == 220
    admission = local.admit_local_benchmark_plan(graph=_proposal_graph(prepared), manifest=prepared["manifest"])
    assert len(local._receipt_bytes(admission["receipt"])) > MAX_BODY_BYTES
    with IntentRepository(state / "intent.duckdb") as intent:
        result = local.materialize_local_benchmark_plan(admission=admission, intent=intent)
        plan = intent.get_plan(result["plan_id"])
        assert len(json.dumps(plan["body"]).encode()) < 32768
        reference = plan["body"]["local_planning_receipt_ref"]
        assert local.load_local_planning_receipt(reference, manifest=admission["manifest"]) == admission["receipt"]
        assert intent.get_task(result["task_cids"][0])["status"] == "ready"
        projection = intent.plan_projection(task_cids=result["task_cids"])
        assert len(projection["tasks"]) == 1 and len(projection["goals"]) == 2
