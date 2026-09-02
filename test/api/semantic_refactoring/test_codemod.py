"""Independent contract tests for SPAR-020 CSTExtractionCodemod@1."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.partition_generators import (
    GeneratorKind,
    ProgramPartitionCandidate,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.transformation_packet import (
    EditKind,
    compile_refactor_transformation_packet,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.codemod import (
    ANALYTICAL_CHANGE_TRANSFORMER_IS_EXECUTOR,
    ANALYZER_ID,
    ASTTOKENS_IS_PARSE_HELPER,
    AUTHORITY,
    AUTHORITY_OWNER,
    BEST_AVAILABLE_CST_BACKEND,
    CODEMOD_CAN_AUTHORIZE_COMPLETION,
    CODEMOD_CAN_AUTHORIZE_TRANSITION,
    CODEMOD_CAN_CREATE_AUTHORITY,
    CODEMOD_CAN_RETIRE_FACADE,
    CODEMOD_CONTRACT_VERSION,
    CODEMOD_IS_NOMINATION_ONLY,
    CODEMOD_RECEIPT_INTERFACE,
    CODEMOD_WRITES_REPOSITORY,
    CSTExtractionCodemod,
    CST_CAPABILITY_PROBE_INTERFACE,
    CST_EXTRACTION_CODEMOD_INTERFACE,
    DECLARED_TARGET_KINDS,
    DRY_RUN_IS_DETERMINISTIC,
    DRY_RUN_MUTATES,
    DUCKLAKE_IS_AUTHORITY,
    EXTRACTION_RESULT_INTERFACE,
    GOAL_ID,
    IDENTITY_EXCLUDED_FIELDS,
    LIBCST_IS_USABLE,
    LIBCST_MUST_NOT_BE_CLAIMED_USABLE,
    MARKDOWN_IS_NOT_COMPLETION,
    MEMBER_LOCATOR_INTERFACE,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    PARSO_IS_PARSE_HELPER,
    PROGRAM,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_SOURCE_REQUIRED,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    SOURCE_MAP_ENTRY_INTERFACE,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    CSTCapabilityProbe,
    CodemodError,
    CodemodReceipt,
    ExtractionResult,
    MemberLocator,
    SourceMapEntry,
    TargetKind,
    apply_cst_extraction,
    assert_not_competing_capsule_family,
    codemod_cid_profile,
    compile_codemod_receipt,
    decode_canonical_receipt,
    decode_canonical_result,
    dry_run_cst_extraction,
    encode_canonical_receipt,
    encode_canonical_result,
    extract_with_cst_codemod,
    probe_cst_capability,
    provider_free_exports,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "codemod.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/codemod.py",
    "test/api/semantic_refactoring/test_codemod.py",
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
SOURCE_WITH_COMMENT = '''"""mod doc"""
from x import y

# keep this comment
def leaf():
    # inner comment
    return 1

class Other:
    pass
'''


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
                "edge_id": "edge:import",
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
    migration_kind: str = "reexport",
    disposition: str = "preserve",
    required: bool = True,
    facade_required: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "plan_cid": _cid("facade"),
        "consumer_plans": [
            {
                "obligation_id": "obl:import",
                "consumer_id": "pkg.cli",
                "subject_id": "symbol:pkg.mod.Record",
                "subject_module": "pkg.mod",
                "kind": "import_path",
                "disposition": disposition,
                "migration_kind": migration_kind,
                "required": required,
                "target_module_id": candidate.candidate_cid,
            }
        ],
        "subject_facades": [
            {
                "subject_id": "symbol:pkg.mod.Record",
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
):
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


def _locator(**overrides: Any) -> MemberLocator:
    fields: dict[str, Any] = {
        "member_id": "node:leaf",
        "path": "pkg/mod.py",
        "symbol": "leaf",
        "kind": TargetKind.FUNCTION,
    }
    fields.update(overrides)
    return MemberLocator(**fields)


def _sources(source: str = SOURCE_WITH_COMMENT, dest: str = "") -> dict[str, str]:
    return {"pkg/mod.py": source, "pkg/extracted.py": dest}


def _extract(source: str = SOURCE_WITH_COMMENT, **overrides: Any) -> ExtractionResult:
    candidate = overrides.pop("candidate", None) or _candidate()
    packet = overrides.pop("packet", None) or _compile(candidate)
    locators = overrides.pop("locators", None) or (_locator(),)
    raw_sources = overrides.pop("raw_sources", None) or _sources(source)
    destination_paths = overrides.pop(
        "destination_paths", {candidate.candidate_cid: "pkg/extracted.py"}
    )
    return extract_with_cst_codemod(
        packet,
        raw_sources=raw_sources,
        locators=locators,
        destination_paths=destination_paths,
        **overrides,
    )


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-020"
    assert GOAL_ID == "SPAR-G041"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert CST_EXTRACTION_CODEMOD_INTERFACE == "CSTExtractionCodemod@1"
    assert MEMBER_LOCATOR_INTERFACE == "MemberLocator@1"
    assert SOURCE_MAP_ENTRY_INTERFACE == "SourceMapEntry@1"
    assert CST_CAPABILITY_PROBE_INTERFACE == "CSTCapabilityProbe@1"
    assert EXTRACTION_RESULT_INTERFACE == "ExtractionResult@1"
    assert CODEMOD_RECEIPT_INTERFACE == "CodemodReceipt@1"
    assert CODEMOD_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("codemod@1")
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "partition orchestration"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert CODEMOD_CAN_AUTHORIZE_COMPLETION is False
    assert CODEMOD_CAN_AUTHORIZE_TRANSITION is False
    assert CODEMOD_CAN_CREATE_AUTHORITY is False
    assert CODEMOD_CAN_RETIRE_FACADE is False
    assert CODEMOD_WRITES_REPOSITORY is False
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert CODEMOD_IS_NOMINATION_ONLY is True
    assert RAW_SOURCE_REQUIRED is True
    assert DRY_RUN_IS_DETERMINISTIC is True
    assert DRY_RUN_MUTATES is False
    assert LIBCST_IS_USABLE is False
    assert LIBCST_MUST_NOT_BE_CLAIMED_USABLE is True
    assert PARSO_IS_PARSE_HELPER is True
    assert ASTTOKENS_IS_PARSE_HELPER is True
    assert ANALYTICAL_CHANGE_TRANSFORMER_IS_EXECUTOR is False
    assert BEST_AVAILABLE_CST_BACKEND == "parso"
    assert DECLARED_TARGET_KINDS == {"function", "class"}
    profile = codemod_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "CSTExtractionCodemod" in names
    assert "MemberLocator" in names
    assert "SourceMapEntry" in names
    assert "ExtractionResult" in names
    assert "CodemodReceipt" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "CSTExtractionCodemod" in exports
    assert "extract_with_cst_codemod" in exports
    assert "dry_run_cst_extraction" in exports
    assert "apply_cst_extraction" in exports
    assert "probe_cst_capability" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_module_does_not_import_libcst_or_analytical_transformer() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".", 1)[0] != "libcst" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module.split(".", 1)[0] != "libcst"
            assert "analytical_change_transforms" not in module


def test_capability_probe_types_libcst_unavailable_and_helpers_present() -> None:
    probe = probe_cst_capability()
    assert probe.libcst == "typed_unavailable"
    assert probe.libcst_usable is False
    assert probe.parso == "present_parse_helper"
    assert probe.asttokens == "present_parse_helper"
    assert probe.parso_is_parse_helper is True
    assert probe.asttokens_is_parse_helper is True
    assert probe.analytical_change_transformer_is_executor is False
    assert probe.best_available == "parso"
    assert importlib.util.find_spec("libcst") is None
    assert importlib.util.find_spec("parso") is not None
    assert importlib.util.find_spec("asttokens") is not None
    restored = CSTCapabilityProbe.from_dict(probe.to_dict())
    assert restored == probe
    dirty = probe.to_dict()
    dirty["libcst_usable"] = True
    with pytest.raises(CodemodError, match="must not be claimed usable"):
        CSTCapabilityProbe.from_dict(dirty)


def test_move_preserves_comments_and_source_maps() -> None:
    result = _extract()
    source = result.sources["pkg/mod.py"]
    dest = result.sources["pkg/extracted.py"]
    assert "def leaf" not in source
    assert "class Other" in source
    assert "from x import y" in source
    assert "def leaf" in dest
    assert "# keep this comment" in dest
    assert "# inner comment" in dest
    assert "class Other" not in dest
    assert result.source_maps
    entry = result.source_maps[0]
    assert entry.member_id == "node:leaf"
    assert entry.origin_path == "pkg/mod.py"
    assert entry.destination_path == "pkg/extracted.py"
    assert entry.comment_prefix_preserved is True
    assert entry.origin_start_line >= 1
    assert entry.destination_start_line >= 1
    assert result.moved_member_ids == ("node:leaf",)
    assert result.mutated is False
    assert result.writes_repository is False
    assert result.libcst_usable is False
    assert result.can_authorize_completion is False
    assert result.can_authorize_transition is False
    kinds = {item.kind for item in _compile().edits}
    assert EditKind.REWRITE.value in kinds
    assert result.deferred_edit_cids
    assert "from pkg.extracted import" not in source


def test_class_and_decorated_targets_are_supported() -> None:
    class_source = "class Record:\n    def method(self):\n        return 1\n"
    class_result = _extract(
        class_source,
        locators=(_locator(symbol="Record", kind=TargetKind.CLASS),),
    )
    assert "class Record" not in class_result.sources["pkg/mod.py"]
    assert "class Record" in class_result.sources["pkg/extracted.py"]
    assert "def method" in class_result.sources["pkg/extracted.py"]
    decorated = "@dec\ndef leaf():\n    return 1\n"
    decorated_result = _extract(decorated)
    assert "@dec" in decorated_result.sources["pkg/extracted.py"]
    assert "def leaf" in decorated_result.sources["pkg/extracted.py"]
    assert "def leaf" not in decorated_result.sources["pkg/mod.py"]


def test_async_and_nested_targets_are_typed_terminals() -> None:
    with pytest.raises(CodemodError, match="typed terminal"):
        _extract("async def leaf():\n    return 1\n")
    nested = "def outer():\n    def leaf():\n        return 1\n"
    with pytest.raises(CodemodError, match="typed terminal"):
        _extract(nested)
    method = "class Other:\n    def leaf(self):\n        return 1\n"
    with pytest.raises(CodemodError, match="typed terminal"):
        _extract(method)


def test_missing_raw_source_and_locator_fail_closed() -> None:
    packet = _compile()
    with pytest.raises(CodemodError, match="raw source"):
        extract_with_cst_codemod(
            packet,
            raw_sources={"pkg/extracted.py": ""},
            locators=(_locator(),),
            destination_paths={packet.edits[0].destination_id: "pkg/extracted.py"},
        )
    with pytest.raises(CodemodError, match="locator"):
        extract_with_cst_codemod(
            packet,
            raw_sources=_sources(),
            locators=(),
            destination_paths={packet.edits[0].destination_id: "pkg/extracted.py"},
        )
    with pytest.raises(CodemodError, match="unrestricted scope"):
        MemberLocator(
            member_id="node:leaf",
            path="../secret.py",
            symbol="leaf",
        )


def test_libcst_backend_is_refused() -> None:
    with pytest.raises(CodemodError, match="must not be claimed usable"):
        CSTExtractionCodemod(backend="libcst")
    with pytest.raises(CodemodError, match="must not be claimed usable"):
        _extract(backend="libcst")
    with pytest.raises(CodemodError, match="typed terminal"):
        _extract(backend="unknown")


def test_dry_run_is_deterministic_and_does_not_mutate_inputs() -> None:
    packet = _compile()
    locators = (_locator(),)
    destination_paths = {packet.edits[0].destination_id: "pkg/extracted.py"}
    raw = _sources()
    snapshot = dict(raw)
    first = dry_run_cst_extraction(
        packet,
        raw_sources=raw,
        locators=locators,
        destination_paths=destination_paths,
    )
    second = dry_run_cst_extraction(
        packet,
        raw_sources=raw,
        locators=locators,
        destination_paths=destination_paths,
    )
    assert raw == snapshot
    assert first.receipt_cid == second.receipt_cid
    assert first.mutated is False
    assert first.deterministic is True
    assert first.writes_repository is False
    assert first.can_authorize_transition is False
    assert first.can_authorize_completion is False
    assert first.libcst_usable is False
    applied = apply_cst_extraction(
        packet,
        raw_sources=raw,
        locators=locators,
        destination_paths=destination_paths,
    )
    assert raw == snapshot
    assert applied.mutated is False
    assert applied.writes_repository is False
    assert applied.receipt().receipt_cid == first.receipt_cid
    adapter = CSTExtractionCodemod()
    via_adapter = adapter.extract(
        packet,
        raw_sources=raw,
        locators=locators,
        destination_paths=destination_paths,
    )
    assert via_adapter.result_cid == applied.result_cid


def test_round_trip_receipt_and_result_are_deterministic() -> None:
    first = _extract()
    second = _extract()
    assert first.result_cid == second.result_cid
    restored = decode_canonical_result(encode_canonical_result(first))
    assert restored == first
    receipt = compile_codemod_receipt(first)
    assert receipt.packet_cid == first.packet_cid
    assert receipt.can_authorize_completion is False
    assert decode_canonical_receipt(encode_canonical_receipt(receipt)) == receipt
    dirty = first.to_dict()
    dirty["timestamp"] = "now"
    with pytest.raises(CodemodError, match="observational"):
        ExtractionResult.from_dict(dirty)


def test_result_cannot_claim_authority_flags() -> None:
    result = _extract()
    payload = result.to_dict()
    payload["can_authorize_completion"] = True
    payload["result_cid"] = cid_for_dag_json(
        {
            key: value
            for key, value in result.identity_payload().items()
        }
    )
    with pytest.raises(CodemodError, match="can_authorize_completion"):
        ExtractionResult.from_dict(payload)
    payload = result.to_dict()
    payload["libcst_usable"] = True
    payload["result_cid"] = cid_for_dag_json(result.identity_payload())
    with pytest.raises(CodemodError, match="must not be claimed usable"):
        ExtractionResult.from_dict(payload)
    payload = result.to_dict()
    payload["writes_repository"] = True
    payload["result_cid"] = cid_for_dag_json(result.identity_payload())
    with pytest.raises(CodemodError, match="cannot write"):
        ExtractionResult.from_dict(payload)


def test_already_applied_move_is_idempotent() -> None:
    first = _extract()
    second = extract_with_cst_codemod(
        _compile(),
        raw_sources=first.sources,
        locators=(_locator(),),
        destination_paths={_compile().edits[0].destination_id: "pkg/extracted.py"},
    )
    assert "def leaf" in second.sources["pkg/extracted.py"]
    assert "def leaf" not in second.sources["pkg/mod.py"]
    assert second.moved_member_ids == ("node:leaf",)


def test_duplicate_definition_is_typed_terminal() -> None:
    dest = "# keep this comment\ndef leaf():\n    # inner comment\n    return 1\n"
    with pytest.raises(CodemodError, match="typed terminal"):
        _extract(SOURCE_WITH_COMMENT, raw_sources=_sources(SOURCE_WITH_COMMENT, dest))


def test_path_outside_scope_is_rejected() -> None:
    packet = _compile()
    with pytest.raises(CodemodError, match="effect_scope"):
        extract_with_cst_codemod(
            packet,
            raw_sources={"pkg/outside.py": "def leaf():\n    return 1\n"},
            locators=(_locator(path="pkg/outside.py"),),
            destination_paths={packet.edits[0].destination_id: "pkg/extracted.py"},
        )
    with pytest.raises(CodemodError, match="effect_scope"):
        extract_with_cst_codemod(
            packet,
            raw_sources=_sources(),
            locators=(_locator(),),
            destination_paths={packet.edits[0].destination_id: "pkg/other.py"},
        )


def test_two_write_path_convention_binds_destination() -> None:
    packet = _compile()
    result = extract_with_cst_codemod(
        packet,
        raw_sources=_sources(),
        locators=(_locator(),),
    )
    assert "def leaf" in result.sources["pkg/extracted.py"]


def test_source_map_round_trip() -> None:
    result = _extract()
    entry = result.source_maps[0]
    restored = SourceMapEntry.from_dict(entry.to_dict())
    assert restored == entry
    locator = _locator()
    assert MemberLocator.from_dict(locator.to_dict()) == locator
