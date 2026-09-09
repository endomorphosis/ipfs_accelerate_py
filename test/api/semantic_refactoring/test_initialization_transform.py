"""Independent contract tests for SPAR-023 InitializationRewritePlan."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.partition_generators import (
    GeneratorKind,
    ProgramPartitionCandidate,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.transformation_packet import (
    ANALYZER_ID as SPAR019_ANALYZER_ID,
    AdapterKind,
    RefactorTransformationPacket,
    compile_refactor_transformation_packet,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.initialization_transform import (
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    DECLARED_INITIALIZATION_REWRITE_KINDS,
    DECLARED_ORDER_RELATIONS,
    DRY_RUN_IS_DETERMINISTIC,
    DRY_RUN_MUTATES,
    DUCKLAKE_IS_AUTHORITY,
    EXECUTOR_IS_NOMINATION_ONLY,
    GOAL_ID,
    HANDLED_ADAPTER_KINDS,
    IDENTITY_EXCLUDED_FIELDS,
    INITIALIZATION_REWRITE_INTERFACE,
    INITIALIZATION_REWRITE_PLAN_INTERFACE,
    INITIALIZATION_REWRITE_RECEIPT_INTERFACE,
    MARKDOWN_IS_NOT_COMPLETION,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    PLAN_IS_NOMINATION_ONLY,
    PROGRAM,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_SOURCE_REQUIRED,
    REWRITE_CAN_AUTHORIZE_COMPLETION,
    REWRITE_CAN_AUTHORIZE_TRANSITION,
    REWRITE_CAN_CREATE_AUTHORITY,
    REWRITE_CONTRACT_VERSION,
    RUNTIME_OBSERVATION_IS_NOT_STATIC_FACT,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    TRACE_REQUIRED_EFFECTS,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    InitializationRewrite,
    InitializationRewriteKind,
    InitializationRewritePlan,
    InitializationRewriteReceipt,
    InitializationTransformError,
    assert_not_competing_capsule_family,
    compile_initialization_rewrite_plan,
    compile_initialization_rewrite_receipt,
    compile_initialization_rewrites,
    decode_canonical_plan,
    decode_canonical_receipt,
    decode_canonical_rewrite,
    dry_run_initialization_rewrites,
    encode_canonical_plan,
    encode_canonical_receipt,
    encode_canonical_rewrite,
    execute_initialization_rewrite_plan,
    execute_initialization_rewrites,
    initialization_transform_cid_profile,
    provider_free_exports,
    verify_preimages,
    verify_required_traces,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "initialization_transform.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/initialization_transform.py",
    "test/api/semantic_refactoring/test_initialization_transform.py",
)
PROTECTED_PATHS = (
    ".gitignore",
    "benchmarks/agent_supervisor/semantic_refactoring/preregistration.json",
    "config/agent_supervisor_semantic_preserving_remodularization_scheduler.json",
    "config/semantic_preserving_autonomous_remodularization_dependencies.seal.json",
    "docs/architecture/SEMANTIC_PRESERVING_AUTONOMOUS_REMODULARIZATION_PLAN.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization.objectives.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization.todo.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/authority_matrix.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/benchmark_preregistration.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/dynamic_python_risk_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/identity_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/interface_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/overlap_gap_matrix.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/repository_baseline.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/rollout_baseline.json",
    "scripts/materialize_semantic_preserving_remodularization_program.py",
    "scripts/ops/agent_supervisor/semantic_preserving_remodularization.py",
    "scripts/validate_semantic_preserving_remodularization_board.py",
    "scripts/validate_semantic_preserving_remodularization_dependencies.py",
    "test/api/semantic_refactoring/test_bootstrap_controls.py",
)
CAPSULE_TYPES = (
    "FunctionSemanticCapsule",
    "MethodSemanticCapsule",
    "ClassSemanticCapsule",
    "TopLevelBlockCapsule",
    "ModuleSemanticCapsule",
    "PackageSemanticCapsule",
    "CallsiteSemanticCapsule",
    "StateOwnerCapsule",
    "RegistrationCapsule",
    "ResourceLifecycleCapsule",
)
TREE_ID = "fbc6fa1ddefb2f9ecb7b5c718d618e3b60aa3051"
WRITE_PATHS = ("pkg/mod.py", "pkg/extracted.py")
VALIDATION = ("python3 -m pytest -q tests/test_mod.py",)
SYMBOL_ID = "symbol:pkg.mod.Record"
CONSUMER_PLANS = (
    {
        "obligation_id": "obl:cli",
        "consumer_id": "pkg.cli",
        "subject_id": SYMBOL_ID,
        "subject_module": "pkg.mod",
        "kind": "cli",
        "disposition": "migrate",
        "migration_kind": "cli",
        "required": True,
        "target_module_id": "",
    },
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _candidate(**overrides: Any) -> ProgramPartitionCandidate:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "generator_kind": GeneratorKind.SCC,
        "member_ids": ("node:leaf",),
        "admitted": True,
        "consumer_ids": ("pkg.cli",),
    }
    fields.update(overrides)
    return ProgramPartitionCandidate(**fields)


def _contracts(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "source_cid": _cid("source"),
        "partition_cid": "",
        "contract_set_cid": _cid("contracts"),
        "contracts": [
            {
                "edge_id": "edge:cli",
                "source_id": "pkg.cli",
                "target_id": "node:leaf",
                "kind": "import",
                "disposition": "admitted",
                "complete": True,
                "required": True,
                "allowed_effects": ["bounded_source_edit", "isolated_validation"],
                "forbidden_effects": ["network"],
            }
        ],
    }
    fields.update(overrides)
    return fields


def _target_api(candidate: ProgramPartitionCandidate, **overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "plan_cid": _cid("target-api"),
        "modules": [
            {
                "module_id": candidate.candidate_cid,
                "member_ids": list(candidate.member_ids),
                "public_exports": [
                    {
                        "member_id": candidate.member_ids[0],
                        "consumer_ids": ["pkg.cli"],
                    }
                ],
            }
        ],
    }
    fields.update(overrides)
    return fields


def _facade(
    candidate: ProgramPartitionCandidate,
    *,
    migration_kind: str = "cli",
    disposition: str = "migrate",
    required: bool = True,
    facade_required: bool = False,
    kind: str = "cli",
    obligation_id: str = "obl:cli",
    **overrides: Any,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "plan_cid": _cid("facade"),
        "consumer_plans": [
            {
                "obligation_id": obligation_id,
                "consumer_id": "pkg.cli",
                "subject_id": SYMBOL_ID,
                "subject_module": "pkg.mod",
                "kind": kind,
                "disposition": disposition,
                "migration_kind": migration_kind,
                "required": required,
                "target_module_id": candidate.candidate_cid,
            }
        ],
        "subject_facades": [
            {
                "subject_id": SYMBOL_ID,
                "subject_module": "pkg.mod",
                "facade_required": facade_required,
                "consumer_ids": ["pkg.cli"],
                "undispositioned_consumer_ids": [],
                "migration_kinds": [migration_kind],
                "target_module_id": candidate.candidate_cid,
                "can_retire_facade": False,
            }
        ],
    }
    fields.update(overrides)
    return fields


def _preimage(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "repository_id": PROGRAM,
        "environment_cid": _cid("env"),
        "graph_cid": _cid("graph"),
        "source_cids": [_cid("source")],
    }
    fields.update(overrides)
    return fields


def _compile(
    candidate: ProgramPartitionCandidate | None = None,
    **overrides: Any,
) -> RefactorTransformationPacket:
    resolved = candidate or _candidate()
    fields: dict[str, Any] = {
        "boundary_contracts": _contracts(partition_cid=resolved.candidate_cid),
        "target_api_plan": _target_api(resolved),
        "facade_plan": _facade(resolved),
        "candidates": (resolved,),
        "preimage": _preimage(),
        "write_paths": WRITE_PATHS,
        "lease_id": _cid("lease"),
        "fence_id": _cid("fence"),
        "epoch_id": _cid("epoch"),
        "validation_commands": VALIDATION,
        "repository_id": PROGRAM,
    }
    fields.update(overrides)
    return compile_refactor_transformation_packet(**fields)


def _graph(
    *,
    effects: Sequence[str] = ("registration",),
    cycle: bool = False,
    preserve: bool = True,
) -> dict[str, Any]:
    nodes = [
        {
            "node_id": "block:pkg.mod:toplevel:0",
            "order_index": 0,
            "module_name": "pkg.mod",
            "effect_kinds": list(effects),
            "required": True,
        },
        {
            "node_id": "block:pkg.mod:toplevel:1",
            "order_index": 1,
            "module_name": "pkg.mod",
            "effect_kinds": ["decorator"],
            "required": True,
        },
    ]
    edges = [
        {
            "source_id": "block:pkg.mod:toplevel:0",
            "target_id": "block:pkg.mod:toplevel:1",
            "kind": "happens_before",
        }
    ]
    if cycle:
        edges.append(
            {
                "source_id": "block:pkg.mod:toplevel:1",
                "target_id": "block:pkg.mod:toplevel:0",
                "kind": "initialization_order",
            }
        )
    return {
        "nodes": nodes,
        "edges": edges,
        "candidates": [
            {"kind": "preserve_order", "required": preserve},
        ],
        "effects": [],
    }


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-023"
    assert GOAL_ID == "SPAR-G042"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert INITIALIZATION_REWRITE_INTERFACE == "InitializationRewrite@1"
    assert INITIALIZATION_REWRITE_PLAN_INTERFACE == "InitializationRewritePlan@1"
    assert INITIALIZATION_REWRITE_RECEIPT_INTERFACE == "InitializationRewriteReceipt@1"
    assert REWRITE_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("initialization_transform@1")
    assert ANALYZER_ID != SPAR019_ANALYZER_ID
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "partition orchestration"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert REWRITE_CAN_AUTHORIZE_COMPLETION is False
    assert REWRITE_CAN_AUTHORIZE_TRANSITION is False
    assert REWRITE_CAN_CREATE_AUTHORITY is False
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert EXECUTOR_IS_NOMINATION_ONLY is True
    assert PLAN_IS_NOMINATION_ONLY is True
    assert RAW_SOURCE_REQUIRED is True
    assert DRY_RUN_IS_DETERMINISTIC is True
    assert DRY_RUN_MUTATES is False
    assert RUNTIME_OBSERVATION_IS_NOT_STATIC_FACT is True
    profile = initialization_transform_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]
    assert DECLARED_INITIALIZATION_REWRITE_KINDS == {
        "initialization_order",
        "decorator",
        "registration",
        "cli",
        "plugin",
        "signal",
        "atexit",
        "resource",
    }
    assert HANDLED_ADAPTER_KINDS == {
        AdapterKind.CLI.value,
        AdapterKind.PLUGIN.value,
        AdapterKind.REGISTRY.value,
    }
    assert DECLARED_ORDER_RELATIONS == {"happens_before", "initialization_order"}
    assert TRACE_REQUIRED_EFFECTS == {"signal", "atexit", "resource"}


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "InitializationRewritePlan" in names
    assert "InitializationRewrite" in names
    assert "InitializationRewriteReceipt" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "InitializationRewritePlan" in exports
    assert "compile_initialization_rewrite_plan" in exports
    assert "execute_initialization_rewrite_plan" in exports
    assert "dry_run_initialization_rewrites" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_cli_adapter_is_nominated_from_packet() -> None:
    packet = _compile()
    rewrites = compile_initialization_rewrites(packet)
    kinds = {item.rewrite_kind for item in rewrites}
    assert InitializationRewriteKind.CLI.value in kinds
    cli = next(
        item for item in rewrites if item.rewrite_kind == InitializationRewriteKind.CLI.value
    )
    assert cli.subject_id == SYMBOL_ID
    assert cli.source_module == "pkg.mod"
    assert cli.destination_module == packet.expected_delta.destination_module_ids[0]
    assert cli.adapter_kind == AdapterKind.CLI.value
    assert cli.rewrite_is_nomination_only is True
    assert cli.write_paths == WRITE_PATHS


def test_plugin_and_registry_adapters_are_nominated() -> None:
    candidate = _candidate()
    plugin = _compile(
        candidate,
        facade_plan=_facade(
            candidate,
            migration_kind="plugin",
            kind="plugin",
            obligation_id="obl:plugin",
        ),
    )
    plugin_kinds = {
        item.rewrite_kind for item in compile_initialization_rewrites(plugin)
    }
    assert InitializationRewriteKind.PLUGIN.value in plugin_kinds
    registry = _compile(
        candidate,
        facade_plan=_facade(
            candidate,
            migration_kind="registry",
            kind="registry",
            obligation_id="obl:registry",
        ),
    )
    registry_kinds = {
        item.rewrite_kind for item in compile_initialization_rewrites(registry)
    }
    assert InitializationRewriteKind.REGISTRATION.value in registry_kinds


def test_initialization_graph_preserves_order_and_decorators() -> None:
    packet = _compile()
    rewrites = compile_initialization_rewrites(
        packet, initialization_graph=_graph()
    )
    kinds = {item.rewrite_kind for item in rewrites}
    assert InitializationRewriteKind.INITIALIZATION_ORDER.value in kinds
    assert InitializationRewriteKind.DECORATOR.value in kinds
    assert InitializationRewriteKind.REGISTRATION.value in kinds
    ordered = [
        item
        for item in rewrites
        if item.rewrite_kind == InitializationRewriteKind.INITIALIZATION_ORDER.value
    ]
    assert [item.order_index for item in ordered] == [0, 1]
    plan = compile_initialization_rewrite_plan(
        packet, initialization_graph=_graph()
    )
    assert plan.order_preserved is True
    assert plan.cycle_free is True
    assert plan.plan_is_nomination_only is True


def test_preimages_are_verified_against_packet() -> None:
    packet = _compile()
    assert verify_preimages(packet) == packet.preimage.preimage_cid
    assert (
        verify_preimages(packet, claimed_preimage_cid=packet.preimage.preimage_cid)
        == packet.preimage.preimage_cid
    )
    with pytest.raises(InitializationTransformError, match="preimage does not verify"):
        verify_preimages(packet, claimed_preimage_cid=_cid("other-preimage"))
    with pytest.raises(InitializationTransformError, match="preimage does not verify"):
        compile_initialization_rewrites(packet, claimed_preimage_cid=_cid("stale"))


def test_initialization_order_cycles_are_rejected() -> None:
    packet = _compile()
    with pytest.raises(InitializationTransformError, match="cycles"):
        compile_initialization_rewrites(
            packet, initialization_graph=_graph(cycle=True)
        )


def test_undispositioned_consumers_are_typed_terminals() -> None:
    packet = _compile()
    with pytest.raises(InitializationTransformError, match="undispositioned"):
        compile_initialization_rewrites(
            packet,
            undispositioned_consumer_ids=("pkg.ghost",),
        )
    with pytest.raises(InitializationTransformError, match="undispositioned"):
        compile_initialization_rewrites(
            packet,
            consumer_plans=(
                {
                    **CONSUMER_PLANS[0],
                    "disposition": "undispositioned",
                    "required": True,
                },
            ),
        )


def test_required_traces_validate_for_signals_atexit_and_resources() -> None:
    packet = _compile()
    trace = _cid("trace:atexit")
    graph = _graph(effects=("atexit", "signal", "resource"))
    with pytest.raises(InitializationTransformError, match="required traces"):
        compile_initialization_rewrites(packet, initialization_graph=graph)
    rewrites = compile_initialization_rewrites(
        packet,
        initialization_graph=graph,
        required_trace_cids=(trace,),
        observed_trace_cids=(trace,),
    )
    kinds = {item.rewrite_kind for item in rewrites}
    assert InitializationRewriteKind.ATEXIT.value in kinds
    assert InitializationRewriteKind.SIGNAL.value in kinds
    assert InitializationRewriteKind.RESOURCE.value in kinds
    assert verify_required_traces(
        required_trace_cids=(trace,),
        observed_trace_cids=(trace,),
    ) == (trace,)
    with pytest.raises(InitializationTransformError, match="required traces"):
        verify_required_traces(
            required_trace_cids=(trace,),
            observed_trace_cids=(_cid("other-trace"),),
        )


def test_unsupported_required_import_time_effect_is_typed_terminal() -> None:
    packet = _compile()
    with pytest.raises(InitializationTransformError, match="unsupported required"):
        compile_initialization_rewrites(
            packet,
            initialization_graph=_graph(effects=("network",)),
        )


def test_dry_run_is_deterministic_and_does_not_mutate() -> None:
    packet = _compile()
    first = dry_run_initialization_rewrites(packet)
    second = execute_initialization_rewrites(packet)
    assert first.receipt_cid == second.receipt_cid
    assert first.mutated is False
    assert first.deterministic is True
    assert first.can_authorize_transition is False
    assert first.can_authorize_completion is False
    assert first.executor_is_nomination_only is True
    assert first.packet_cid == packet.packet_cid
    assert first.preimage_cid == packet.preimage.preimage_cid
    assert first.preimage_verified is True
    assert first.order_preserved is True
    assert first.traces_validated is True
    assert first.write_paths == WRITE_PATHS
    restored = InitializationRewriteReceipt.from_dict(first.to_dict())
    assert restored == first
    with pytest.raises(InitializationTransformError, match="cannot mutate"):
        execute_initialization_rewrites(packet, mutate=True)


def test_plan_round_trip_is_deterministic() -> None:
    packet = _compile()
    first = compile_initialization_rewrites(packet)
    second = compile_initialization_rewrites(packet)
    assert [item.rewrite_cid for item in first] == [item.rewrite_cid for item in second]
    restored = decode_canonical_rewrite(encode_canonical_rewrite(first[0]))
    assert restored == first[0]
    plan = compile_initialization_rewrite_plan(packet)
    assert decode_canonical_plan(encode_canonical_plan(plan)) == plan
    assert execute_initialization_rewrite_plan(packet) == plan
    receipt = compile_initialization_rewrite_receipt(packet)
    assert decode_canonical_receipt(encode_canonical_receipt(receipt)) == receipt
    assert receipt.plan_cid == plan.plan_cid


def test_identity_excludes_observational_fields() -> None:
    packet = _compile()
    plan = compile_initialization_rewrite_plan(packet)
    payload = plan.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(payload))
    dirty = dict(payload)
    dirty["timestamp"] = "now"
    with pytest.raises(InitializationTransformError, match="observational"):
        InitializationRewritePlan.from_dict(dirty)


def test_plan_cannot_claim_authority_flags() -> None:
    packet = _compile()
    plan = compile_initialization_rewrite_plan(packet)
    payload = plan.to_dict()
    payload["can_authorize_completion"] = True
    payload["plan_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "plan_cid"}
    )
    with pytest.raises(InitializationTransformError, match="can_authorize_completion"):
        InitializationRewritePlan.from_dict(payload)
    payload = plan.to_dict()
    payload["plan_is_nomination_only"] = False
    payload["plan_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "plan_cid"}
    )
    with pytest.raises(InitializationTransformError, match="nomination_only"):
        InitializationRewritePlan.from_dict(payload)


def test_same_source_and_destination_require_preserve_order() -> None:
    packet = _compile()
    with pytest.raises(InitializationTransformError, match="preserve_order"):
        InitializationRewrite(
            rewrite_kind=InitializationRewriteKind.CLI,
            subject_id=SYMBOL_ID,
            source_module="pkg.mod",
            destination_module="pkg.mod",
            write_paths=WRITE_PATHS,
            preimage_cid=packet.preimage.preimage_cid,
            packet_cid=packet.packet_cid,
            tree_id=packet.tree_id,
            preserve_order=False,
        )
    preserved = InitializationRewrite(
        rewrite_kind=InitializationRewriteKind.INITIALIZATION_ORDER,
        subject_id="block:0",
        source_module="pkg.mod",
        destination_module="pkg.mod",
        write_paths=WRITE_PATHS,
        preimage_cid=packet.preimage.preimage_cid,
        packet_cid=packet.packet_cid,
        tree_id=packet.tree_id,
        preserve_order=True,
    )
    assert preserved.preserve_order is True


def test_packet_must_remain_spar019_analyzer() -> None:
    packet = _compile()
    payload = packet.to_dict()
    payload["analyzer_id"] = ANALYZER_ID
    identity = {key: value for key, value in payload.items() if key != "packet_cid"}
    payload["packet_cid"] = cid_for_dag_json(identity)
    with pytest.raises(Exception, match="SPAR-019 analyzer"):
        compile_initialization_rewrites(payload)


def test_empty_write_paths_are_unrestricted_scope() -> None:
    packet = _compile()
    with pytest.raises(InitializationTransformError, match="unrestricted scope"):
        InitializationRewrite(
            rewrite_kind=InitializationRewriteKind.CLI,
            subject_id=SYMBOL_ID,
            source_module="pkg.mod",
            destination_module="pkg.extracted",
            write_paths=(),
            preimage_cid=packet.preimage.preimage_cid,
            packet_cid=packet.packet_cid,
            tree_id=packet.tree_id,
            preserve_order=False,
        )


def test_reexport_only_packet_is_not_an_initialization_transform() -> None:
    candidate = _candidate()
    packet = _compile(
        candidate,
        facade_plan=_facade(
            candidate,
            migration_kind="reexport",
            kind="import_path",
            disposition="preserve",
            obligation_id="obl:import",
        ),
    )
    with pytest.raises(InitializationTransformError, match="requires initialization"):
        compile_initialization_rewrites(packet)
