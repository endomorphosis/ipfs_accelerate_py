"""Independent current-tree checks for DOEP-063 freshness and selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ACCELERATE_CONTEXT_PACK_CONSUMES,
    ACCELERATE_CONTEXT_PACK_OWNERSHIP,
    ACCELERATE_FRESHNESS_AND_SELECTION_BINDING,
    ACCELERATE_FRESHNESS_AND_SELECTION_INTERFACE,
    ACCELERATE_FRESHNESS_AND_SELECTION_SCHEMA,
    KIT_CURRENT_ROOT_CAS_OPERATION,
    SUPERVISOR_CONTEXT_PACK_WIRE_SCHEMA,
    SUPERVISOR_CONTEXT_PACK_WIRE_SCHEMA_VERSION,
    ChangedTreeContextError,
    ContextCompiler,
    ContextPackFreshnessState,
    ContextPackSelectionDisposition,
    ContextPackSelectionError,
    StaleContextPackError,
    accelerate_freshness_and_selection_are_armed,
    admit_accelerate_context_pack,
    compile_context_capsule,
    evaluate_context_pack_freshness,
    select_smallest_adequate_context_pack,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
    ContextBudget,
    ContextReference,
    ContextTier,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = (
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
CURRENT_ROOT_CID = "bafy-doep-063-current-root"


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _pack(
    pack_cid: str,
    *,
    task_id: str = "DOEP-063",
    tree: str = CURRENT_TREE,
    sufficient: bool = True,
    expansion_required: bool = False,
    capsules: tuple[str, ...] | None = None,
) -> dict[str, object]:
    reference = pack_cid
    return {
        "capsule_cids": list(capsules or (reference,)),
        "expansion_required": expansion_required,
        "pack_cid": pack_cid,
        "producer": "ipfs_datasets_py.proof_context.context_pack",
        "repository_state_cid": reference,
        "required_source_cids": {
            "target_source": f"{reference}-target",
            "surrounding_source": f"{reference}-surrounding",
            "test_source": f"{reference}-test",
        },
        "scanned_tree_oid": tree,
        "schema": SUPERVISOR_CONTEXT_PACK_WIRE_SCHEMA,
        "schema_version": SUPERVISOR_CONTEXT_PACK_WIRE_SCHEMA_VERSION,
        "sufficiency_state": "sufficient" if sufficient else "insufficient",
        "task_id": task_id,
    }


def _candidate(pack_cid: str, **operational: object) -> dict[str, object]:
    pack = _pack(pack_cid, **{
        key: operational.pop(key)
        for key in (
            "task_id",
            "tree",
            "sufficient",
            "expansion_required",
            "capsules",
        )
        if key in operational
    })
    return {"pack": pack, **operational}


def _world() -> dict[str, object]:
    return {
        "current_tree_id": CURRENT_TREE,
        "current_policy_id": "policy:supervisor",
        "current_policy_revision": "sha256:policy",
        "current_plan_epoch": 1,
        "current_root": {
            "namespace": "context-pack/doep-063",
            "root_cid": CURRENT_ROOT_CID,
            "revision": 1,
            "transition_cid": "bafy-transition",
        },
        "current_lease_id": "lease:doep-063",
        "current_fence_id": "fence:doep-063",
    }


def _budget() -> ContextBudget:
    return ContextBudget(
        max_input_tokens=220,
        reserved_output_tokens=40,
        reserved_tool_tokens=10,
        max_items=16,
        max_item_bytes=16_384,
        max_serialized_bytes=262_144,
    )


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_binding_extends_context_compiler_without_competing_subsystem() -> None:
    assert ACCELERATE_FRESHNESS_AND_SELECTION_INTERFACE == (
        "AccelerateContextPackFreshnessAndSelection@1"
    )
    assert (
        ACCELERATE_FRESHNESS_AND_SELECTION_BINDING
        == ACCELERATE_FRESHNESS_AND_SELECTION_INTERFACE
    )
    assert ACCELERATE_FRESHNESS_AND_SELECTION_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/"
        "accelerate-context-pack-freshness-and-selection@1"
    )
    assert ACCELERATE_CONTEXT_PACK_CONSUMES == (
        SUPERVISOR_CONTEXT_PACK_WIRE_SCHEMA,
        "DatasetsSemanticContextBuilder@1",
        KIT_CURRENT_ROOT_CAS_OPERATION,
    )
    assert dict(ACCELERATE_CONTEXT_PACK_OWNERSHIP) == {
        "canonical_semantic_identity": "ipfs_datasets_py",
        "exact_bytes_cid_storage": "ipfs_kit_py",
        "operational_admission": "ipfs_accelerate_py",
    }
    assert ContextCompiler.FRESHNESS_INTERFACE == (
        ACCELERATE_FRESHNESS_AND_SELECTION_INTERFACE
    )
    assert ContextCompiler.FRESHNESS_BINDING == (
        ACCELERATE_FRESHNESS_AND_SELECTION_BINDING
    )
    assert ContextCompiler.FRESHNESS_SCHEMA == (
        ACCELERATE_FRESHNESS_AND_SELECTION_SCHEMA
    )
    assert evaluate_context_pack_freshness.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.context.context_compiler"
    )
    assert select_smallest_adequate_context_pack.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.context.context_compiler"
    )
    assert admit_accelerate_context_pack.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.context.context_compiler"
    )
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "class ContextCompiler" in source
    assert "def evaluate_context_pack_freshness(" in source
    assert "def select_smallest_adequate_context_pack(" in source
    assert "def admit_accelerate_context_pack(" in source
    assert "not a second ContextPack builder" in source
    assert "never writes DuckDB" in source
    assert "cannot skip these controls" in source
    assert "class CompetingFreshnessEngine" not in source
    assert "class ContextPackBus" not in source
    assert "class CompetingContextPackCompiler" not in source
    assert "class CompetingSelectionSubsystem" not in source
    assert "CREATE TABLE" not in source
    assert "import duckdb" not in source.casefold()
    assert accelerate_freshness_and_selection_are_armed()


def test_fresh_smallest_adequate_pack_is_admitted_without_completing() -> None:
    small = _candidate(
        CURRENT_ROOT_CID,
        token_cost=40,
        policy_id="policy:supervisor",
        policy_revision="sha256:policy",
        plan_epoch=1,
        lease_id="lease:doep-063",
        fence_id="fence:doep-063",
    )
    large = _candidate(
        CURRENT_ROOT_CID + "-large",
        token_cost=400,
        durable_cid=CURRENT_ROOT_CID,
        policy_id="policy:supervisor",
        policy_revision="sha256:policy",
        plan_epoch=1,
        lease_id="lease:doep-063",
        fence_id="fence:doep-063",
        capsules=(CURRENT_ROOT_CID + "-a", CURRENT_ROOT_CID + "-b"),
    )
    result = admit_accelerate_context_pack(
        (large, small),
        unresolved_questions=("which symbol changed?",),
        worker_assertion=True,
        model_assertion=True,
        **_world(),
    )
    assert result.admitted is True
    assert result.selected_pack_cid == CURRENT_ROOT_CID
    assert result.expansion_required is False
    assert result.unresolved_questions == ("which symbol changed?",)
    payload = result.to_dict()
    assert payload["completion_authoritative"] is False
    assert payload["worker_assertion_is_authority"] is False
    assert payload["database_write"] is False
    assert payload["empty_queue_is_completion"] is False
    assert payload["authorizes_completion"] is False
    assert payload["ownership"] == dict(ACCELERATE_CONTEXT_PACK_OWNERSHIP)
    assert payload["carrier"] == "ContextCompiler"
    dispositions = {
        item.pack_cid: item.disposition for item in result.decisions
    }
    assert dispositions[CURRENT_ROOT_CID] is ContextPackSelectionDisposition.SELECTED
    assert dispositions[CURRENT_ROOT_CID + "-large"] is (
        ContextPackSelectionDisposition.OMITTED_LARGER
    )
    compiler = ContextCompiler(_budget())
    delegated = compiler.admit_context_pack((small,), **_world())
    assert delegated.selected_pack_cid == result.selected_pack_cid
    assert '"completion_authoritative": true' not in json.dumps(payload)


def test_stale_tree_policy_epoch_root_lease_and_pack_fail_closed() -> None:
    world = _world()
    with pytest.raises(StaleContextPackError):
        admit_accelerate_context_pack(
            (_candidate(CURRENT_ROOT_CID, tree="deadbeef" * 5),),
            **world,
        )
    with pytest.raises(StaleContextPackError):
        admit_accelerate_context_pack(
            (
                _candidate(
                    CURRENT_ROOT_CID,
                    policy_revision="sha256:old-policy",
                    plan_epoch=1,
                ),
            ),
            **world,
        )
    with pytest.raises(StaleContextPackError):
        admit_accelerate_context_pack(
            (_candidate(CURRENT_ROOT_CID, plan_epoch=0),),
            **world,
        )
    with pytest.raises(StaleContextPackError):
        admit_accelerate_context_pack(
            (_candidate("bafy-not-current"),),
            **world,
        )
    with pytest.raises(StaleContextPackError):
        admit_accelerate_context_pack(
            (
                _candidate(
                    CURRENT_ROOT_CID,
                    lease_id="lease:other",
                    plan_epoch=1,
                    policy_revision="sha256:policy",
                ),
            ),
            **world,
        )
    with pytest.raises(StaleContextPackError):
        admit_accelerate_context_pack(
            (_candidate(CURRENT_ROOT_CID, freshness="stale"),),
            **world,
        )
    stale_tree = evaluate_context_pack_freshness(
        _candidate(CURRENT_ROOT_CID, tree="0" * 40),
        current_tree_id=CURRENT_TREE,
    )
    assert stale_tree.state is ContextPackFreshnessState.STALE_TREE
    assert stale_tree.fresh is False
    empty_root = evaluate_context_pack_freshness(
        _candidate(CURRENT_ROOT_CID),
        current_tree_id=CURRENT_TREE,
        current_root={"namespace": "context-pack/doep-063", "root_cid": None, "revision": 0},
    )
    assert empty_root.state is ContextPackFreshnessState.STALE_CURRENT_ROOT


def test_required_pack_cannot_be_auctioned_and_worker_cannot_force_stale() -> None:
    required = _candidate(
        CURRENT_ROOT_CID,
        required=True,
        token_cost=900,
        plan_epoch=1,
        policy_revision="sha256:policy",
        lease_id="lease:doep-063",
        fence_id="fence:doep-063",
    )
    cheaper = _candidate(
        CURRENT_ROOT_CID + "-cheap",
        required=False,
        token_cost=10,
        durable_cid=CURRENT_ROOT_CID,
        plan_epoch=1,
        policy_revision="sha256:policy",
        lease_id="lease:doep-063",
        fence_id="fence:doep-063",
    )
    result = select_smallest_adequate_context_pack(
        (cheaper, required),
        worker_assertion=True,
        **_world(),
    )
    assert result.admitted is True
    assert result.selected_pack_cid == CURRENT_ROOT_CID
    selected = {item.pack_cid: item for item in result.decisions}
    assert selected[CURRENT_ROOT_CID].disposition is (
        ContextPackSelectionDisposition.REQUIRED
    )
    assert selected[CURRENT_ROOT_CID + "-cheap"].disposition is (
        ContextPackSelectionDisposition.OMITTED_NOT_REQUIRED
    )
    with pytest.raises(StaleContextPackError):
        admit_accelerate_context_pack(
            (
                _candidate(
                    CURRENT_ROOT_CID,
                    required=True,
                    freshness="stale",
                    worker_assertion=True,
                    model_assertion=True,
                ),
            ),
            worker_assertion=True,
            model_assertion=True,
            **_world(),
        )


def test_expansion_required_is_named_but_not_admitted() -> None:
    expandable = _candidate(
        CURRENT_ROOT_CID,
        expansion_required=True,
        sufficient=True,
        plan_epoch=1,
        policy_revision="sha256:policy",
        lease_id="lease:doep-063",
        fence_id="fence:doep-063",
    )
    result = select_smallest_adequate_context_pack(
        (expandable,),
        unresolved_questions=("what is the exact changed symbol?",),
        **_world(),
    )
    assert result.admitted is False
    assert result.expansion_required is True
    assert result.selected_pack_cid == CURRENT_ROOT_CID
    assert result.unresolved_questions == ("what is the exact changed symbol?",)
    assert result.to_dict()["execution_admitted"] is False
    assert result.to_dict()["completion_authoritative"] is False


def test_compiler_rejects_stale_evidence_without_a_second_selector() -> None:
    binding = {
        "repository_id": "repo:doep-063",
        "tree_id": CURRENT_TREE,
        "objective_id": "DOEP-G070",
        "objective_revision": "sha256:objective",
        "policy_id": "policy:supervisor",
        "policy_revision": "sha256:policy",
        "caller": "supervisor:doep-063",
        "stage": "planning",
    }
    core = {
        "goal": {"id": "DOEP-G070", "summary": "Admit current ContextPacks"},
        "authority": {"mode": "proposal"},
        "scope": {"paths": ["ipfs_accelerate_py/agent_supervisor/context"]},
        "acceptance": {"criteria": ["stale packs are rejected"]},
    }
    current = ContextReference(
        reference_id="ev-current",
        kind="context-pack",
        tier=ContextTier.EVIDENCE,
        referenced_content_id="sha256:" + "ab" * 32,
        repository_id=binding["repository_id"],
        tree_id=CURRENT_TREE,
        token_count=8,
    )
    compiled = compile_context_capsule(_budget(), evidence=(current,), **binding, **core)
    assert compiled.capsule.tree_id == CURRENT_TREE
    stale_tree = ContextReference(
        reference_id="ev-stale-tree",
        kind="context-pack",
        tier=ContextTier.EVIDENCE,
        referenced_content_id="sha256:" + "cd" * 32,
        repository_id=binding["repository_id"],
        tree_id="0" * 40,
        token_count=8,
    )
    with pytest.raises(ChangedTreeContextError):
        compile_context_capsule(_budget(), evidence=(stale_tree,), **binding, **core)
    stale_flag = ContextReference(
        reference_id="ev-stale-flag",
        kind="context-pack",
        tier=ContextTier.EVIDENCE,
        referenced_content_id="sha256:" + "ef" * 32,
        repository_id=binding["repository_id"],
        tree_id=CURRENT_TREE,
        token_count=8,
        metadata={"freshness": "stale"},
    )
    with pytest.raises(StaleContextPackError):
        compile_context_capsule(_budget(), evidence=(stale_flag,), **binding, **core)


def test_malformed_authority_fields_and_empty_candidates_fail_closed() -> None:
    with pytest.raises(ContextPackSelectionError):
        select_smallest_adequate_context_pack((), **_world())
    malformed = evaluate_context_pack_freshness(
        {**_pack(CURRENT_ROOT_CID), "execution_admission": True},
        current_tree_id=CURRENT_TREE,
    )
    assert malformed.state is ContextPackFreshnessState.MALFORMED
    forbidden = evaluate_context_pack_freshness(
        {**_pack(CURRENT_ROOT_CID), "duckdb": "write"},
        current_tree_id=CURRENT_TREE,
    )
    assert forbidden.state is ContextPackFreshnessState.MALFORMED


def test_manifest_and_candidate_receipt_bind_the_exact_current_tree_outputs() -> None:
    manifest = _load_json(OUTPUT_PATH)
    receipt = _load_json(RECEIPT_PATH)
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-063"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["plan_revision"] == "DOEP-PLAN-V5"
        assert payload["board_namespace"] == (
            "agent-supervisor-direct-objective-and-event-driven-planning-v1"
        )
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["entrypoint"] == (
        "admit_accelerate_context_pack"
    )
    assert manifest["canonical_extension"]["carrier"] == "ContextCompiler"
    assert manifest["canonical_extension"]["binding"] == (
        ACCELERATE_FRESHNESS_AND_SELECTION_BINDING
    )
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["expected_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["write_scope"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {
        path: True for path in OWNER_RELATIVE_OUTPUTS
    }
    assert receipt["base_repositories"] == BASE_REPOSITORIES
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(SOURCE_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    evidence = receipt["required_evidence"]
    assert evidence["source_commit_tree_gitlinks"] == BASE_REPOSITORIES
    assert evidence["verifier_admission"] == (
        "pending_independent_fenced_supervisor"
    )
    digested = {
        relative: _sha256_file(ACCELERATE_ROOT / relative)
        for relative in OWNER_RELATIVE_OUTPUTS[:-1]
    }
    encoded = json.dumps(
        digested, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert evidence["changed_path_digest"] == (
        "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    )
