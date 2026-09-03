"""Independent contract tests for SPAR-033 RefactorTransition@1 and exact reuse."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.refactor_memory import (
    ADAPTER_IS_NOMINATION_ONLY,
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    DECLARED_DECISIONS,
    DECLARED_OUTCOMES,
    DIMENSION_MISMATCH_REASONS,
    DRY_RUN_IS_DETERMINISTIC,
    DRY_RUN_MUTATES,
    DUCKLAKE_IS_AUTHORITY,
    EXACT_KEY_REQUIRED_FOR_REUSE,
    FORBIDDEN_REUSE_NAMES,
    GOAL_ID,
    IDENTITY_EXCLUDED_FIELDS,
    MARKDOWN_IS_NOT_COMPLETION,
    MEMORY_CAN_AUTHORIZE_COMPLETION,
    MEMORY_CAN_AUTHORIZE_TRANSITION,
    MEMORY_CAN_CREATE_AUTHORITY,
    MEMORY_CONTRACT_VERSION,
    MISMATCHED_EVIDENCE_REVOKES_REUSE,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    NEGATIVE_EPISODES_RETAINED,
    NETWORK_DENIED,
    NETWORK_DENY,
    PROGRAM,
    PROOF_TRANSFER_ACROSS_STALE_BINDINGS,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_SOURCE_REQUIRED,
    REFACTOR_MEMORY_INTERFACE,
    REFACTOR_MEMORY_RECEIPT_INTERFACE,
    REFACTOR_MEMORY_STORE_INTERFACE,
    REFACTOR_REUSE_DECISION_INTERFACE,
    REFACTOR_REUSE_KEY_INTERFACE,
    REFACTOR_TRANSITION_INTERFACE,
    REUSE_KEY_DIMENSIONS,
    SIMILARITY_YIELDS_CONTEXT_ONLY,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    STALE_EVIDENCE_REVOKES_REUSE,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    TEST_PASS_IS_NOT_PROOF,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    RefactorMemory,
    RefactorMemoryError,
    RefactorMemoryReceipt,
    RefactorMemoryStore,
    RefactorReuseDecision,
    RefactorReuseKey,
    RefactorTransition,
    ReuseDecisionKind,
    RevokeReason,
    TransitionOutcome,
    assert_not_competing_capsule_family,
    compile_memory_receipt,
    compile_refactor_transition,
    compile_reuse_key,
    decode_canonical_decision,
    decode_canonical_key,
    decode_canonical_receipt,
    decode_canonical_store,
    decode_canonical_transition,
    decide_exact_reuse,
    dry_run_exact_reuse,
    encode_canonical_decision,
    encode_canonical_key,
    encode_canonical_receipt,
    encode_canonical_store,
    encode_canonical_transition,
    lookup_exact_reuse,
    mismatch_reasons,
    provider_free_exports,
    refactor_memory_cid_profile,
    refactor_memory_descriptor,
    remember_transition,
    retain_negative_episode,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "refactor_memory.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/refactor_memory.py",
    "test/api/semantic_refactoring/test_refactor_memory.py",
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
OTHER_TREE = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
WRITE_PATHS = ("pkg/mod.py", "pkg/extracted.py")
VALIDATION = ("python3 -m pytest -q tests/test_mod.py",)
TOOLCHAIN_ID = "toolchain:hammer@1"
PROCEDURE_VERSION = "procedure:v1"


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _key(**overrides: Any) -> RefactorReuseKey:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "state_cid": _cid("state"),
        "partition_cid": _cid("partition"),
        "policy_cid": _cid("policy"),
        "environment_cid": _cid("env"),
        "toolchain_id": TOOLCHAIN_ID,
        "obligation_root_cid": _cid("obligations"),
        "validation_cid": _cid("validation"),
        "procedure_version": PROCEDURE_VERSION,
    }
    fields.update(overrides)
    return compile_reuse_key(**fields)


def _transition(**overrides: Any) -> RefactorTransition:
    fields: dict[str, Any] = {
        "reuse_key": _key(),
        "outcome": TransitionOutcome.ACCEPTED.value,
        "wave_receipt_cid": _cid("wave"),
        "validation_result_cid": _cid("tv-result"),
        "proof_receipt_cid": _cid("proof"),
        "packet_cid": _cid("packet"),
        "raw_source_cids": [_cid("source")],
        "write_paths": list(WRITE_PATHS),
        "validation_commands": list(VALIDATION),
        "evidence_class": "transition",
    }
    fields.update(overrides)
    return compile_refactor_transition(**fields)


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-033"
    assert GOAL_ID == "SPAR-G061"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert REFACTOR_TRANSITION_INTERFACE == "RefactorTransition@1"
    assert REFACTOR_REUSE_DECISION_INTERFACE == "RefactorReuseDecision@1"
    assert REFACTOR_REUSE_KEY_INTERFACE == "RefactorReuseKey@1"
    assert REFACTOR_MEMORY_INTERFACE == "RefactorMemory@1"
    assert REFACTOR_MEMORY_STORE_INTERFACE == "RefactorMemoryStore@1"
    assert REFACTOR_MEMORY_RECEIPT_INTERFACE == "RefactorMemoryReceipt@1"
    assert MEMORY_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("refactor_memory@1")
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "exact reuse"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert MEMORY_CAN_AUTHORIZE_COMPLETION is False
    assert MEMORY_CAN_AUTHORIZE_TRANSITION is False
    assert MEMORY_CAN_CREATE_AUTHORITY is False
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert SIMILARITY_YIELDS_CONTEXT_ONLY is True
    assert STALE_EVIDENCE_REVOKES_REUSE is True
    assert MISMATCHED_EVIDENCE_REVOKES_REUSE is True
    assert NEGATIVE_EPISODES_RETAINED is True
    assert EXACT_KEY_REQUIRED_FOR_REUSE is True
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert TEST_PASS_IS_NOT_PROOF is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert ADAPTER_IS_NOMINATION_ONLY is True
    assert RAW_SOURCE_REQUIRED is True
    assert NETWORK_DENIED is True
    assert NETWORK_DENY == "deny"
    assert DRY_RUN_IS_DETERMINISTIC is True
    assert DRY_RUN_MUTATES is False
    assert PROOF_TRANSFER_ACROSS_STALE_BINDINGS is False
    profile = refactor_memory_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "RefactorTransition" in names
    assert "RefactorReuseDecision" in names
    assert "RefactorReuseKey" in names
    assert "RefactorMemory" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "RefactorTransition" in exports
    assert "RefactorReuseDecision" in exports
    assert "decide_exact_reuse" in exports
    assert "compile_reuse_key" in exports
    assert "remember_transition" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_module_does_not_implement_forbidden_reuse_shortcuts() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert not (names & FORBIDDEN_REUSE_NAMES)
    descriptor = refactor_memory_descriptor()
    assert descriptor["interface"] == REFACTOR_MEMORY_INTERFACE
    assert descriptor["exact_key_required_for_reuse"] is True
    assert descriptor["similarity_yields_context_only"] is True
    assert set(descriptor["reuse_key_dimensions"]) == set(REUSE_KEY_DIMENSIONS)
    forbids = set(descriptor["forbids"])
    assert "admit_by_similarity" in forbids
    assert "vector_reuse" in forbids
    assert "reuse_stale_proof" in forbids
    assert "suppress_raw_source" in forbids


def test_reuse_key_covers_all_freshness_dimensions() -> None:
    key = _key()
    assert REUSE_KEY_DIMENSIONS == (
        "tree_id",
        "state_cid",
        "partition_cid",
        "policy_cid",
        "environment_cid",
        "toolchain_id",
        "obligation_root_cid",
        "validation_cid",
        "procedure_version",
    )
    for dimension in REUSE_KEY_DIMENSIONS:
        assert getattr(key, dimension)
    assert key.key_cid == compile_reuse_key(
        tree_id=TREE_ID,
        state_cid=_cid("state"),
        partition_cid=_cid("partition"),
        policy_cid=_cid("policy"),
        environment_cid=_cid("env"),
        toolchain_id=TOOLCHAIN_ID,
        obligation_root_cid=_cid("obligations"),
        validation_cid=_cid("validation"),
        procedure_version=PROCEDURE_VERSION,
    ).key_cid


def test_changing_any_key_dimension_changes_identity() -> None:
    base = _key()
    variants = {
        "tree_id": OTHER_TREE,
        "state_cid": _cid("state-2"),
        "partition_cid": _cid("partition-2"),
        "policy_cid": _cid("policy-2"),
        "environment_cid": _cid("env-2"),
        "toolchain_id": "toolchain:other@1",
        "obligation_root_cid": _cid("obligations-2"),
        "validation_cid": _cid("validation-2"),
        "procedure_version": "procedure:v2",
    }
    seen = {base.key_cid}
    for dimension, value in variants.items():
        changed = _key(**{dimension: value})
        assert changed.key_cid not in seen
        seen.add(changed.key_cid)
        reasons = mismatch_reasons(base, changed)
        assert DIMENSION_MISMATCH_REASONS[dimension] in reasons


def test_incomplete_reuse_key_fails_closed() -> None:
    with pytest.raises(RefactorMemoryError, match="incomplete_key"):
        compile_reuse_key(tree_id=TREE_ID, state_cid=_cid("state"))
    with pytest.raises(RefactorMemoryError, match="observational"):
        compile_reuse_key(
            tree_id=TREE_ID,
            state_cid=_cid("state"),
            partition_cid=_cid("partition"),
            policy_cid=_cid("policy"),
            environment_cid=_cid("env"),
            toolchain_id=TOOLCHAIN_ID,
            obligation_root_cid=_cid("obligations"),
            validation_cid=_cid("validation"),
            procedure_version=PROCEDURE_VERSION,
            timestamp="now",
        )


def test_exact_accepted_episode_reuses() -> None:
    episode = _transition()
    decision = decide_exact_reuse(_key(), (episode,))
    assert decision.decision == ReuseDecisionKind.REUSE.value
    assert decision.exact_match is True
    assert decision.matched_transition_cid == episode.transition_cid
    assert decision.reasons == ()
    assert decision.can_authorize_completion is False
    assert decision.can_authorize_transition is False
    assert decision.similarity_authoritative is False
    assert DECLARED_DECISIONS == {"reuse", "revoke", "context_only"}
    assert DECLARED_OUTCOMES == {"accepted", "rejected"}


def test_stale_tree_revokes_reuse() -> None:
    stored = _transition()
    query = _key(tree_id=OTHER_TREE)
    decision = decide_exact_reuse(query, (stored,))
    assert decision.decision == ReuseDecisionKind.REVOKE.value
    assert decision.exact_match is False
    assert RevokeReason.STALE_TREE.value in decision.reasons
    assert stored.transition_cid in decision.context_transition_cids


def test_each_mismatched_dimension_revokes_reuse() -> None:
    stored = _transition()
    variants = {
        "state_cid": (_cid("state-x"), RevokeReason.MISMATCHED_STATE.value),
        "partition_cid": (_cid("part-x"), RevokeReason.MISMATCHED_PARTITION.value),
        "policy_cid": (_cid("policy-x"), RevokeReason.MISMATCHED_POLICY.value),
        "environment_cid": (_cid("env-x"), RevokeReason.MISMATCHED_ENVIRONMENT.value),
        "toolchain_id": ("toolchain:stale@1", RevokeReason.MISMATCHED_TOOLCHAIN.value),
        "obligation_root_cid": (
            _cid("obl-x"),
            RevokeReason.MISMATCHED_OBLIGATIONS.value,
        ),
        "validation_cid": (_cid("val-x"), RevokeReason.MISMATCHED_VALIDATION.value),
        "procedure_version": (
            "procedure:stale",
            RevokeReason.MISMATCHED_PROCEDURE_VERSION.value,
        ),
    }
    for dimension, (value, reason) in variants.items():
        decision = decide_exact_reuse(_key(**{dimension: value}), (stored,))
        assert decision.decision == ReuseDecisionKind.REVOKE.value
        assert reason in decision.reasons
        assert decision.exact_match is False


def test_similarity_yields_context_only() -> None:
    stored = _transition(reuse_key=_key(state_cid=_cid("other-state")))
    similar = [
        {
            "channel": "vector",
            "evidence_class": "vector_candidate",
            "transition_cid": stored.transition_cid,
        }
    ]
    decision = decide_exact_reuse(_key(), (stored,), similar_hits=similar)
    assert decision.decision == ReuseDecisionKind.CONTEXT_ONLY.value
    assert decision.exact_match is False
    assert RevokeReason.SIMILARITY_NOT_EXACT.value in decision.reasons
    assert stored.transition_cid in decision.context_transition_cids
    assert decision.similarity_authoritative is False
    assert decision.vector_authoritative is False
    assert decision.matched_transition_cid == ""


def test_similarity_cannot_override_exact_negative_episode() -> None:
    rejected = _transition(
        outcome=TransitionOutcome.REJECTED.value,
        evidence_class="countermodel",
    )
    similar = [
        {
            "channel": "lexical",
            "evidence_class": "heuristic",
            "hit_cid": _cid("analog"),
        }
    ]
    decision = decide_exact_reuse(_key(), (rejected,), similar_hits=similar)
    assert decision.decision == ReuseDecisionKind.REVOKE.value
    assert decision.exact_match is True
    assert RevokeReason.NEGATIVE_EPISODE_BLOCKS_REUSE.value in decision.reasons
    assert rejected.transition_cid in decision.retained_negative_cids


def test_negative_episodes_are_retained_and_block_reuse() -> None:
    accepted = _transition()
    rejected = _transition(
        outcome=TransitionOutcome.REJECTED.value,
        evidence_class="countermodel",
        wave_receipt_cid=_cid("wave-rejected"),
    )
    store = remember_transition(None, accepted)
    store = retain_negative_episode(store, rejected)
    assert accepted.transition_cid in store.accepted_transition_cids
    assert rejected.transition_cid in store.rejected_transition_cids
    assert rejected.negative_episode is True
    decision = lookup_exact_reuse(store, _key())
    assert decision.decision == ReuseDecisionKind.REVOKE.value
    assert RevokeReason.NEGATIVE_EPISODE_BLOCKS_REUSE.value in decision.reasons
    assert rejected.transition_cid in decision.retained_negative_cids
    with pytest.raises(RefactorMemoryError, match="rejected outcome"):
        retain_negative_episode(store, accepted)


def test_vector_evidence_cannot_admit_accepted_transition() -> None:
    with pytest.raises(RefactorMemoryError, match="cannot admit"):
        _transition(evidence_class="vector_candidate")
    with pytest.raises(RefactorMemoryError, match="cannot admit"):
        _transition(evidence_class="model_hypothesis")
    rejected = _transition(
        outcome=TransitionOutcome.REJECTED.value,
        evidence_class="heuristic",
    )
    assert rejected.negative_episode is True


def test_raw_source_and_write_scope_are_required() -> None:
    with pytest.raises(RefactorMemoryError, match="raw_source_cids"):
        _transition(raw_source_cids=[])
    with pytest.raises(RefactorMemoryError, match="unrestricted scope"):
        _transition(write_paths=[])
    with pytest.raises(RefactorMemoryError, match="unrestricted scope"):
        _transition(write_paths=["pkg/*.py"])
    with pytest.raises(RefactorMemoryError, match="unrestricted scope"):
        _transition(write_paths=["/tmp/pkg/mod.py"])
    with pytest.raises(RefactorMemoryError, match="unrestricted scope"):
        _transition(write_paths=["pkg/../secret.py"])
    with pytest.raises(RefactorMemoryError, match="validation_commands"):
        _transition(validation_commands=[])


def test_remember_is_idempotent_and_append_only() -> None:
    first = _transition()
    store = remember_transition(None, first)
    again = remember_transition(store, first)
    assert again.store_cid == store.store_cid
    assert len(again.episodes) == 1
    second = _transition(packet_cid=_cid("packet-2"))
    grown = remember_transition(store, second)
    assert grown.store_cid != store.store_cid
    assert len(grown.episodes) == 2


def test_round_trip_and_receipt_are_deterministic() -> None:
    key = _key()
    episode = _transition(reuse_key=key)
    store = remember_transition(None, episode)
    decision = lookup_exact_reuse(store, key)
    restored_key = decode_canonical_key(encode_canonical_key(key))
    assert restored_key == key
    assert restored_key.key_cid == key.key_cid
    restored_episode = decode_canonical_transition(encode_canonical_transition(episode))
    assert restored_episode == episode
    restored_store = decode_canonical_store(encode_canonical_store(store))
    assert restored_store == store
    restored_decision = decode_canonical_decision(encode_canonical_decision(decision))
    assert restored_decision == decision
    receipt = compile_memory_receipt(store, decision)
    assert receipt.can_authorize_completion is False
    assert receipt.can_authorize_transition is False
    assert receipt.mutated is False
    assert receipt.deterministic is True
    restored_receipt = decode_canonical_receipt(encode_canonical_receipt(receipt))
    assert restored_receipt == receipt
    assert RefactorMemoryReceipt.from_dict(receipt.to_dict()).receipt_cid == receipt.receipt_cid


def test_identity_excludes_observational_fields() -> None:
    episode = _transition()
    payload = episode.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(payload))
    dirty = dict(payload)
    dirty["timestamp"] = "now"
    with pytest.raises(RefactorMemoryError, match="observational"):
        RefactorTransition.from_dict(dirty)
    key_payload = _key().to_dict()
    key_payload["wall_clock"] = "now"
    with pytest.raises(RefactorMemoryError, match="observational"):
        RefactorReuseKey.from_dict(key_payload)


def test_transition_cannot_claim_authority_flags() -> None:
    episode = _transition()
    payload = episode.to_dict()
    payload["can_authorize_completion"] = True
    payload["transition_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "transition_cid"}
    )
    with pytest.raises(RefactorMemoryError, match="can_authorize_completion"):
        RefactorTransition.from_dict(payload)
    payload = episode.to_dict()
    payload["adapter_is_nomination_only"] = False
    payload["transition_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "transition_cid"}
    )
    with pytest.raises(RefactorMemoryError, match="nomination_only"):
        RefactorTransition.from_dict(payload)
    payload = episode.to_dict()
    payload["raw_source_required"] = False
    payload["transition_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "transition_cid"}
    )
    with pytest.raises(RefactorMemoryError, match="raw_source_required"):
        RefactorTransition.from_dict(payload)
    payload = episode.to_dict()
    payload["similarity_authoritative"] = True
    payload["transition_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "transition_cid"}
    )
    with pytest.raises(RefactorMemoryError, match="similarity_authoritative"):
        RefactorTransition.from_dict(payload)


def test_empty_memory_revokes_with_no_exact_match() -> None:
    decision = decide_exact_reuse(_key(), ())
    assert decision.decision == ReuseDecisionKind.REVOKE.value
    assert RevokeReason.NO_EXACT_MATCH.value in decision.reasons
    assert decision.exact_match is False


def test_similar_hits_cannot_claim_exact_channel() -> None:
    with pytest.raises(RefactorMemoryError, match="cannot claim exact"):
        decide_exact_reuse(
            _key(),
            (),
            similar_hits=[{"channel": "exact", "transition_cid": _cid("hit")}],
        )


def test_adapter_dry_run_is_deterministic_and_does_not_mutate() -> None:
    adapter = RefactorMemory()
    episode = _transition()
    store = adapter.remember(None, episode)
    first = adapter.dry_run(_key(), store.episodes)
    second = adapter.lookup(store, _key())
    third = dry_run_exact_reuse(_key(), store.episodes)
    assert first.decision_cid == second.decision_cid == third.decision_cid
    assert first.decision == ReuseDecisionKind.REUSE.value
    receipt = adapter.receipt(store, first)
    assert receipt.mutated is False
    assert receipt.deterministic is True
    rejected = _transition(
        outcome=TransitionOutcome.REJECTED.value,
        evidence_class="countermodel",
        wave_receipt_cid=_cid("wave-neg"),
    )
    with_negative = adapter.retain_negative(store, rejected)
    blocked = adapter.decide(_key(), with_negative.episodes)
    assert blocked.decision == ReuseDecisionKind.REVOKE.value
    assert rejected.transition_cid in with_negative.rejected_transition_cids


def test_reuse_requires_exact_match_contract() -> None:
    with pytest.raises(RefactorMemoryError, match="exact key match"):
        RefactorReuseDecision(
            query_key_cid=_cid("query"),
            decision=ReuseDecisionKind.REUSE.value,
            matched_transition_cid=_cid("episode"),
            exact_match=False,
        )
    with pytest.raises(RefactorMemoryError, match="revoke requires"):
        RefactorReuseDecision(
            query_key_cid=_cid("query"),
            decision=ReuseDecisionKind.REVOKE.value,
            reasons=(),
        )
    with pytest.raises(RefactorMemoryError, match="context_only"):
        RefactorReuseDecision(
            query_key_cid=_cid("query"),
            decision=ReuseDecisionKind.CONTEXT_ONLY.value,
            reasons=(RevokeReason.SIMILARITY_NOT_EXACT.value,),
            exact_match=True,
        )


def test_store_cannot_drop_negative_retention_flag() -> None:
    store = remember_transition(None, _transition())
    payload = store.to_dict()
    payload["negative_episodes_retained"] = False
    payload["store_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "store_cid"}
    )
    with pytest.raises(RefactorMemoryError, match="negative episodes"):
        RefactorMemoryStore.from_dict(payload)
