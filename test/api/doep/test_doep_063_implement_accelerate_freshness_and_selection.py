"""Independent current-tree checks for DOEP-063 Accelerate freshness and selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ACCELERATE_CONTEXT_PACK_FORBIDDEN_FIELDS,
    ACCELERATE_CONTEXT_PACK_OWNERSHIP,
    CONTEXT_PACK_ADMISSION_SCHEMA,
    CONTEXT_PACK_FRESHNESS_SCHEMA,
    SUPERVISOR_CONTEXT_PACK_SCHEMA,
    SUPERVISOR_CONTEXT_PACK_SCHEMA_VERSION,
    ContextPackAdmissionDisposition,
    ContextPackAdmissionError,
    ContextPackFreshnessStatus,
    admit_context_pack,
    assess_context_pack_freshness,
    select_and_admit_context_pack,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
COMPILER_PATH = (
    ACCELERATE_ROOT
    / "ipfs_accelerate_py/agent_supervisor/context/context_compiler.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-063.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-063.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/context/context_compiler.py",
    "test/api/doep/test_doep_063_implement_accelerate_freshness_and_selection.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-063.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-063.json",
)
TASK_CID = "sha256:a117a64676549c4a29666f6b0d90e407c08cbaa362d8454ee50a0647094599e9"
PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
BASE_REPOSITORIES = {
    "ipfs_accelerate_py": {
        "commit": "87715e9295626e7918f7fc8a7b1a1531ab04208f",
        "tree": "1c9a399cc7a599d5904e5be2ae58c6be3650cff7",
    },
    "ipfs_datasets_py": {
        "commit": "3668b8857a9aa7b1a3c847be12725b5cd057d2e7",
        "tree": "456e09b51d6a07a3a5873436df24054768195320",
    },
    "ipfs_kit_py": {
        "commit": "b6c65ba732733d7e33852713ba18aa3b12235668",
        "tree": "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2",
    },
    "lift_coding": {
        "commit": "bb8869ed72eb7002434345d9969efee729c4f7f6",
        "tree": "99e85bfe584b7688ffbeff86da1e612dd6893a42",
    },
}
CURRENT_TREE = "16ef68abe8a35a3033dfaf1ed4e8d6132600df8f"


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _pack(**overrides: Any) -> dict[str, Any]:
    payload = {
        "capsule_cids": ["cid:capsule-a"],
        "expansion_required": False,
        "pack_cid": "cid:pack-doep-063",
        "producer": "ipfs_datasets_py.proof_context.context_pack",
        "repository_state_cid": "cid:repo-state",
        "required_source_cids": {
            "surrounding_source": "cid:surrounding",
            "target_source": "cid:target",
            "test_source": "cid:test",
        },
        "scanned_tree_oid": CURRENT_TREE,
        "schema": SUPERVISOR_CONTEXT_PACK_SCHEMA,
        "schema_version": SUPERVISOR_CONTEXT_PACK_SCHEMA_VERSION,
        "sufficiency_state": "sufficient",
        "task_id": "DOEP-063",
    }
    payload.update(overrides)
    return payload


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_current_pack_is_selected_and_admitted_without_competing_subsystem() -> None:
    pack = _pack()
    freshness = assess_context_pack_freshness(
        pack,
        current_tree_oid=CURRENT_TREE,
        durable_pack_cid=pack["pack_cid"],
        current_root_cid=pack["pack_cid"],
    )
    admission = select_and_admit_context_pack(
        pack,
        current_tree_oid=CURRENT_TREE,
        durable_pack_cid=pack["pack_cid"],
        current_root_cid=pack["pack_cid"],
    )
    assert freshness.schema == CONTEXT_PACK_FRESHNESS_SCHEMA
    assert freshness.status is ContextPackFreshnessStatus.CURRENT
    assert freshness.completion_authoritative is False
    assert admission.schema == CONTEXT_PACK_ADMISSION_SCHEMA
    assert admission.disposition is ContextPackAdmissionDisposition.ADMIT
    assert admission.admitted is True
    assert admission.completion_authoritative is False
    assert admission.pack_cid == pack["pack_cid"]
    assert admission.selected_source_cids == pack["required_source_cids"]
    assert admission.selected_capsule_cids == ("cid:capsule-a",)
    assert {item.reference_id for item in admission.selection_decisions} >= set(
        pack["required_source_cids"]
    )
    assert all(
        item.included and item.reason == "required"
        for item in admission.selection_decisions
        if item.kind == "required_source"
    )
    assert dict(admission.ownership) == dict(ACCELERATE_CONTEXT_PACK_OWNERSHIP)
    assert admit_context_pack is select_and_admit_context_pack
    source = COMPILER_PATH.read_text(encoding="utf-8")
    assert "def select_and_admit_context_pack(" in source
    assert "def assess_context_pack_freshness(" in source
    assert "class ContextPacker" not in source


def test_stale_tree_or_root_bindings_fail_closed() -> None:
    pack = _pack()
    stale_tree = assess_context_pack_freshness(
        pack, current_tree_oid="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    )
    assert stale_tree.status is ContextPackFreshnessStatus.STALE
    assert "scanned_tree_oid_mismatch" in stale_tree.reason_codes

    stale_root = select_and_admit_context_pack(
        pack,
        current_tree_oid=CURRENT_TREE,
        current_root_cid="cid:other-root",
    )
    assert stale_root.disposition is ContextPackAdmissionDisposition.REJECT
    assert stale_root.freshness.status is ContextPackFreshnessStatus.STALE
    assert "current_root_mismatch" in stale_root.reason_codes
    assert stale_root.selected_capsule_cids == ()
    assert all(
        item.included
        for item in stale_root.selection_decisions
        if item.kind == "required_source"
    )

    stale_durable = select_and_admit_context_pack(
        pack,
        current_tree_oid=CURRENT_TREE,
        durable_pack_cid="cid:other-pack",
    )
    assert stale_durable.disposition is ContextPackAdmissionDisposition.REJECT
    assert "durable_pack_cid_mismatch" in stale_durable.reason_codes


def test_expansion_required_does_not_silently_fully_admit() -> None:
    pack = _pack(expansion_required=True, sufficiency_state="partial")
    admission = select_and_admit_context_pack(
        pack, current_tree_oid=CURRENT_TREE, durable_pack_cid=pack["pack_cid"]
    )
    assert admission.disposition is ContextPackAdmissionDisposition.EXPAND_REQUIRED
    assert admission.admitted is False
    assert admission.selected_source_cids == pack["required_source_cids"]
    assert admission.selected_capsule_cids == ()
    assert any(
        item.kind == "capsule"
        and item.included is False
        and item.reason == "expansion_candidate"
        for item in admission.selection_decisions
    )


def test_forbidden_authority_fields_and_incomplete_packs_are_rejected() -> None:
    with pytest.raises(ContextPackAdmissionError, match="forbidden"):
        select_and_admit_context_pack(
            {**_pack(), "execution_admission": True},
            current_tree_oid=CURRENT_TREE,
        )
    with pytest.raises(ContextPackAdmissionError, match="forbidden"):
        assess_context_pack_freshness(
            {**_pack(), "completion_authoritative": True},
            current_tree_oid=CURRENT_TREE,
        )
    incomplete = _pack()
    del incomplete["task_id"]
    with pytest.raises(ContextPackAdmissionError, match="missing"):
        select_and_admit_context_pack(incomplete, current_tree_oid=CURRENT_TREE)
    assert "execution_admission" in ACCELERATE_CONTEXT_PACK_FORBIDDEN_FIELDS
    assert ACCELERATE_CONTEXT_PACK_OWNERSHIP["operational_admission"] == (
        "ipfs_accelerate_py"
    )


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-063"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["cross_repository_ownership"] == dict(
        ACCELERATE_CONTEXT_PACK_OWNERSHIP
    )
    assert manifest["canonical_extension"]["module"] == (
        "ipfs_accelerate_py.agent_supervisor.context.context_compiler"
    )
    assert manifest["canonical_extension"]["entrypoint"] == (
        "select_and_admit_context_pack"
    )
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {
        path: True for path in OWNER_RELATIVE_OUTPUTS
    }
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(COMPILER_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert (
        receipt["required_evidence"]["verifier_admission"]
        == "pending_independent_fenced_supervisor"
    )
