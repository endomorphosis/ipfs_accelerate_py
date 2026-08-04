from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from statistics import median

import pytest
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    DELTA_RETRY_EVIDENCE_ID,
    ChangedTreeContextError,
    ContentAddressedContextStore,
    ContextCompileResult,
    ContextCompiler,
    ContextCompileResult,
    ContextDeltaBudgetError,
    ContextDeltaError,
    ContextDeltaReceipt,
    ContextExpansionCancelled,
    DeltaRetryContextEvidence,
    ExclusionReason,
    InclusionReason,
    MissingContextReferenceError,
    RetryContextCapsule,
    compile_context_delta,
    compile_retry_context,
    expand_context,
    expand_context_references,
    reconstruct_context,
    render_retry_context,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
    ContextBoundsError,
    ContextBudget,
    ContextBoundsError,
    ContextBudget,
    ContextCapsule,
    ContextContractError,
    ContextDeltaCapsule,
    ContextReference,
    ContextTier,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    ImplementationRetryDeferred,
    PortalImplementationDaemon,
    PortalTask,
)

BINDING = {
    "repository_id": "repo:delta",
    "tree_id": "tree:current",
    "objective_id": "ASI-G092",
    "objective_revision": "sha256:objective",
    "policy_id": "policy:supervisor",
    "policy_revision": "sha256:policy",
    "caller": "supervisor:test",
    "stage": "implementation",
}
CORE = {
    "goal": {"id": "ASI-G092", "summary": "Use retry deltas"},
    "authority": {"mode": "proposal", "allowed_paths": ["src"]},
    "scope": {"paths": ["src/context.py"]},
    "acceptance": {"criteria": ["coverage remains complete"]},
}


def _budget() -> ContextBudget:
    return ContextBudget(
        max_input_tokens=600,
        reserved_output_tokens=100,
        reserved_tool_tokens=20,
        max_items=32,
        max_serialized_bytes=262_144,
    )


def _reference(
    reference_id: str,
    content: str,
    tokens: int,
    *,
    required: bool = False,
) -> ContextReference:
    return ContextReference(
        reference_id=reference_id,
        kind="test-evidence",
        tier=ContextTier.INVARIANT if required else ContextTier.EVIDENCE,
        referenced_content_id=f"sha256:{content}",
        repository_id=BINDING["repository_id"],
        tree_id=BINDING["tree_id"],
        token_count=tokens,
        metadata={
            "required": required,
            "coverage_ids": (f"coverage:{reference_id}",),
        },
    )


def _tokenizer(text: str) -> int:
    return max(1, len(text.encode("utf-8")) // 32)


def _compiler() -> ContextCompiler:
    return ContextCompiler(
        _budget(),
        tokenizer=_tokenizer,
        provider_context_window=720,
    )


def _parent() -> tuple[
    ContextCompiler,
    ContextCapsule,
    ContextReference,
    ContextReference,
]:
    compiler = _compiler()
    required = _reference("required", "old-required", 80, required=True)
    optional = _reference("diagnostic", "old-diagnostic", 160)
    result = compiler.compile(
        **BINDING,
        **CORE,
        evidence=(required, optional),
    )
    return compiler, result.capsule, required, optional


def test_delta_transmits_changes_and_preserves_required_coverage() -> None:
    compiler, parent, required, _ = _parent()
    changed = _reference("diagnostic", "new-diagnostic", 30)

    result = compiler.compile_delta(
        parent,
        evidence=(required, changed),
    )

    assert result.delta_capsule.is_delta
    assert ContextDeltaCapsule.from_json(
        result.delta_capsule.to_json()
    ) == result.delta_capsule
    assert result.delta_capsule.parent_capsule_id == parent.capsule_id
    assert tuple(
        item.reference_id for item in result.delta_capsule.evidence
    ) == ("diagnostic",)
    assert {
        item.reference_id for item in result.reconstructed_capsule.evidence
    } == {"required", "diagnostic"}
    assert result.receipt.delta_tokens < result.receipt.full_replay_tokens
    assert result.receipt.delta_tokens == compiler.estimator.estimate(
        result.delta_capsule.to_record()
    )
    assert result.receipt.full_replay_tokens == max(
        result.reconstructed_capsule.input_tokens,
        compiler.estimator.estimate(
            result.reconstructed_capsule.provider_input_payload
        ),
    )
    assert result.receipt.evidence is not None
    assert result.receipt.evidence.requirement_id == DELTA_RETRY_EVIDENCE_ID
    assert result.receipt.evidence_claim_references == (
        DELTA_RETRY_EVIDENCE_ID,
    )
    assert set(result.receipt.evidence.required_coverage_ids).issubset(
        result.receipt.evidence.reconstructed_coverage_ids
    )
    decisions = {item.reference_id: item for item in result.decisions}
    assert decisions["diagnostic"].reason is InclusionReason.CHANGED
    assert decisions["required"].reason is ExclusionReason.UNCHANGED


def test_delta_receipt_and_witness_round_trip_and_reject_forged_claims() -> None:
    compiler, parent, required, _ = _parent()
    result = compiler.compile_delta(
        parent,
        evidence=(
            required,
            _reference("diagnostic", "fixed", 20),
        ),
    )

    assert ContextDeltaReceipt.from_json(
        result.receipt.to_json()
    ) == result.receipt
    assert DeltaRetryContextEvidence.from_json(
        result.receipt.evidence.to_json()  # type: ignore[union-attr]
    ) == result.receipt.evidence

    forged = result.receipt.to_record()
    forged["delta_tokens"] += 1
    with pytest.raises(ContextDeltaError, match="bound|identity"):
        ContextDeltaReceipt.from_dict(forged)

    forged = result.receipt.to_dict()
    forged["evidence_claim_references"] = ()
    with pytest.raises(ContextDeltaError, match="claim"):
        ContextDeltaReceipt.from_dict(forged)

    assert result.receipt.evidence is not None
    forged_evidence = replace(
        result.receipt.evidence,
        artifact_digest="sha256:" + "0" * 64,
    )
    forged_receipt = replace(result.receipt, evidence=forged_evidence)
    with pytest.raises(ContextDeltaError, match="artifact digest"):
        replace(result, receipt=forged_receipt)

    forged_receipt = replace(result.receipt, objective_id="ASI-G999")
    with pytest.raises(ContextDeltaError, match="complete parent"):
        replace(result, receipt=forged_receipt)

    assert result.receipt.evidence is not None
    forged_evidence = replace(
        result.receipt.evidence,
        full_replay_tokens=result.receipt.full_replay_tokens + 1,
    )
    forged_receipt = replace(
        result.receipt,
        full_replay_tokens=result.receipt.full_replay_tokens + 1,
        evidence=forged_evidence,
    )
    with pytest.raises(ContextDeltaError, match="complete parent"):
        replace(result, receipt=forged_receipt)

    forged_decisions = tuple(
        replace(item, reason=InclusionReason.REQUESTED)
        if item.reference_id == "diagnostic"
        else item
        for item in result.decisions
    )
    forged_receipt = replace(result.receipt, decisions=forged_decisions)
    with pytest.raises(ContextDeltaError, match="transmitted evidence"):
        replace(
            result,
            receipt=forged_receipt,
            decisions=forged_decisions,
        )

    forged_evidence = replace(
        result.receipt.evidence,
        delta_tokens=result.receipt.delta_tokens + 1,
    )
    forged_receipt = replace(
        result.receipt,
        delta_tokens=result.receipt.delta_tokens + 1,
        evidence=forged_evidence,
    )
    with pytest.raises(ContextDeltaError, match="not reproducible"):
        replace(result, receipt=forged_receipt)

    stale_parent = replace(parent, objective_revision="sha256:stale")
    with pytest.raises(ContextDeltaError, match="exact reconstruction|not bound"):
        replace(result, parent_capsule=stale_parent)


def test_unchanged_retry_and_required_evidence_loss_fail_closed() -> None:
    compiler, parent, required, optional = _parent()

    with pytest.raises(ContextDeltaError, match="changed or explicitly requested"):
        compiler.compile_delta(
            parent,
            evidence=(required, optional),
        )

    with pytest.raises(ContextDeltaError, match="drops required"):
        compiler.compile_delta(
            parent,
            evidence=(_reference("diagnostic", "new", 20),),
        )


def test_requested_expansion_is_parent_bound_and_deterministic() -> None:
    compiler = _compiler()
    required = _reference("required", "required", 50, required=True)
    omitted = _reference("large", "large", 700)
    parent_result = compiler.compile(
        **BINDING,
        **CORE,
        evidence=(required, omitted),
    )
    parent = parent_result.capsule
    assert parent.expansion_references

    smaller = _reference("large", "large-summary", 20)
    result = expand_context(compiler, parent, (smaller,))

    assert result.delta_capsule.parent_capsule_id == parent.capsule_id
    decision = {
        item.reference_id: item for item in result.decisions
    }["large"]
    assert decision.reason in {
        InclusionReason.CHANGED,
        InclusionReason.REQUESTED,
    }
    rebuilt = reconstruct_context(parent, result.delta_capsule)
    assert rebuilt == result.reconstructed_capsule
    assert {
        item.reference_id for item in rebuilt.evidence
    } == {"required", "large"}
    assert not rebuilt.truncated
    assert rebuilt.omissions == ()


def test_reconstruction_rejects_stale_parent_and_delta_omits_invariant_core() -> None:
    compiler, parent, required, _ = _parent()
    result = compiler.compile_delta(
        parent,
        evidence=(required, _reference("diagnostic", "new", 20)),
    )

    stale_parent = replace(parent, objective_revision="sha256:other")
    with pytest.raises(ContextDeltaError, match="not bound"):
        reconstruct_context(stale_parent, result.delta_capsule)

    wire = result.delta_capsule.to_dict()
    assert {"goal", "authority", "scope", "acceptance"}.isdisjoint(wire)
    wire["goal"] = {"id": "ASI-G092", "summary": "smuggled replay"}
    with pytest.raises(ContextContractError, match="unsupported fields"):
        ContextDeltaCapsule.from_dict(wire)


def test_reconstruction_rejects_undeclared_unchanged_evidence_replay() -> None:
    _, parent, required, _ = _parent()
    replay = ContextDeltaCapsule(
        parent_capsule_id=parent.capsule_id,
        stage=parent.stage,
        evidence=(required,),
        reconstructed_input_tokens=parent.input_tokens,
    )

    with pytest.raises(ContextDeltaError, match="replays unchanged evidence"):
        reconstruct_context(parent, replay)

    requested = replace(
        replay,
        requested_reference_ids=(required.reference_id,),
    )
    assert reconstruct_context(parent, requested) == parent


def test_delta_is_deterministic_across_candidate_and_request_order() -> None:
    compiler, parent, required, optional = _parent()
    first_new = _reference("first-new", "first", 15)
    second_new = _reference("second-new", "second", 15)

    forward = compiler.compile_delta(
        parent,
        evidence=(required, optional, first_new, second_new),
        requested_reference_ids=("diagnostic", "second-new"),
    )
    reverse = compiler.compile_delta(
        parent,
        evidence=(second_new, first_new, optional, required),
        requested_reference_ids=("second-new", "diagnostic"),
    )

    assert reverse.delta_capsule == forward.delta_capsule
    assert reverse.decisions == forward.decisions
    assert reverse.receipt == forward.receipt
    assert reverse.reconstructed_capsule == forward.reconstructed_capsule


def test_requested_unchanged_reference_is_not_masqueraded_as_changed() -> None:
    compiler, parent, required, optional = _parent()

    result = compiler.compile_delta(
        parent,
        evidence=(required, optional),
        requested_reference_ids=("diagnostic",),
    )

    assert result.receipt.evidence is not None
    assert result.receipt.evidence.changed_reference_ids == ()
    assert result.receipt.evidence.requested_reference_ids == ("diagnostic",)
    assert result.delta_capsule.requested_reference_ids == ("diagnostic",)
    decision = {item.reference_id: item for item in result.decisions}
    assert decision["diagnostic"].reason is InclusionReason.REQUESTED


def test_delta_rejects_requiredness_downgrade_and_full_context_overflow() -> None:
    compiler, parent, required, optional = _parent()
    downgraded = ContextReference(
        reference_id=required.reference_id,
        kind=required.kind,
        tier=ContextTier.EVIDENCE,
        referenced_content_id=required.referenced_content_id,
        repository_id=required.repository_id,
        tree_id=required.tree_id,
        token_count=required.token_count,
        metadata={
            "required": False,
            "coverage_ids": required.coverage_ids,
        },
    )
    with pytest.raises(ContextDeltaError, match="downgrades"):
        compiler.compile_delta(
            parent,
            evidence=(downgraded, optional),
            requested_reference_ids=("diagnostic",),
        )
    coverage_losing = ContextReference(
        reference_id=required.reference_id,
        kind=required.kind,
        tier=ContextTier.INVARIANT,
        referenced_content_id="sha256:coverage-losing",
        repository_id=required.repository_id,
        tree_id=required.tree_id,
        token_count=required.token_count,
        metadata={"required": True},
    )
    with pytest.raises(ContextDeltaError, match="loses required coverage"):
        compiler.compile_delta(
            parent,
            evidence=(coverage_losing, optional),
        )

    tight_budget = ContextBudget(
        max_input_tokens=100,
        reserved_output_tokens=0,
        reserved_tool_tokens=0,
    )
    tight = ContextCompiler(tight_budget, tokenizer=_tokenizer)
    base_only = tight.compile(**BINDING, **CORE).capsule.input_tokens
    base_required = _reference("required", "required", 1, required=True)
    tight_parent = tight.compile(
        **BINDING, **CORE, evidence=(base_required,)
    ).capsule
    overflowing = _reference(
        "new-required",
        "new-required",
        100 - base_only + 1,
        required=True,
    )
    with pytest.raises(ContextDeltaBudgetError, match="full context exceeds"):
        tight.compile_delta(
            tight_parent,
            evidence=(base_required, overflowing),
        )


def test_reconstruction_preserves_expansion_handles_and_rejects_token_forgery() -> None:
    compiler = _compiler()
    required = _reference("required", "required", 50, required=True)
    selected_later = _reference("selected-later", "large-a", 700)
    still_deferred = _reference("still-deferred", "large-b", 700)
    parent = compiler.compile(
        **BINDING,
        **CORE,
        evidence=(required, selected_later, still_deferred),
    ).capsule
    result = expand_context(
        compiler,
        parent,
        (_reference("selected-later", "summary-a", 20),),
    )

    assert tuple(
        item.reference_id
        for item in result.reconstructed_capsule.expansion_references
    ) == ("still-deferred",)
    assert result.reconstructed_capsule.truncated
    assert result.reconstructed_capsule.omissions == (
        "still-deferred:token_budget",
    )
    forged = replace(
        result.delta_capsule,
        reconstructed_input_tokens=sum(
            item.token_count for item in result.reconstructed_capsule.evidence
        ),
    )
    with pytest.raises(ContextDeltaError, match="omits inherited core"):
        reconstruct_context(parent, forged)


def test_reconstruction_preserves_colon_reference_omission_reason() -> None:
    compiler = _compiler()
    required = _reference("required", "required", 50, required=True)
    parent = compiler.compile(
        **BINDING,
        **CORE,
        evidence=(required, _reference("deferred", "large", 700)),
    ).capsule
    deferred = replace(
        parent.expansion_references[0],
        reference_id="evidence:still-deferred",
    )
    parent = replace(
        parent,
        expansion_references=(deferred,),
        truncated=True,
        omissions=("evidence:still-deferred:item_limit",),
    )

    result = compiler.compile_delta(
        parent,
        evidence=(
            replace(required, referenced_content_id="sha256:changed"),
        ),
    )

    assert result.reconstructed_capsule.omissions == (
        "evidence:still-deferred:item_limit",
    )


def test_new_required_candidate_is_included_in_witness_coverage() -> None:
    compiler, parent, required, _ = _parent()
    newly_required = _reference("new-required", "new", 20, required=True)

    result = compiler.compile_delta(
        parent,
        evidence=(required, newly_required),
    )

    assert result.receipt.evidence is not None
    assert set(result.receipt.evidence.required_coverage_ids) == {
        "coverage:required",
        "coverage:new-required",
    }


def test_delta_must_be_smaller_than_full_replay() -> None:
    compiler = ContextCompiler(
        _budget(),
        tokenizer=lambda text: (
            100 if "context-delta-capsule@1" in text else 1
        ),
    )
    required = _reference("required", "old", 1, required=True)
    parent = compiler.compile(
        **BINDING,
        **CORE,
        evidence=(required,),
    ).capsule

    with pytest.raises(ContextDeltaError, match="fewer tokens"):
        compiler.compile_delta(
            parent,
            evidence=(_reference("required", "new", 10, required=True),),
        )


def test_top_level_delta_wrapper_binds_the_same_contract() -> None:
    _, parent, required, _ = _parent()
    result = compile_context_delta(
        _budget(),
        parent,
        tokenizer=_tokenizer,
        provider_context_window=720,
        evidence=(required, _reference("diagnostic", "new", 20)),
    )

    assert result.receipt.parent_capsule_id == parent.capsule_id
    assert result.receipt.delta_capsule_id == result.delta_capsule.capsule_id


def test_content_addressed_expansion_resolves_exact_bytes_and_fails_closed() -> None:
    compiler = _compiler()
    store = ContentAddressedContextStore()
    body = ("Focused diagnostic evidence.\n" * 20).strip()
    target = store.put(body)
    required = _reference("required", "required", 50, required=True)
    candidate = ContextReference(
        reference_id="expand-me",
        kind="diagnostic",
        tier=ContextTier.EVIDENCE,
        referenced_content_id=target,
        repository_id=BINDING["repository_id"],
        tree_id=BINDING["tree_id"],
        summary=body,
        byte_count=len(body.encode()),
        token_count=700,
    )
    parent = compiler.compile(
        **BINDING,
        **CORE,
        evidence=(required, candidate),
    ).capsule
    assert tuple(
        item.reference_id for item in parent.expansion_references
    ) == ("expand-me",)

    result = expand_context_references(
        compiler,
        parent,
        ("expand-me",),
        store,
        repository_id=parent.repository_id,
        tree_id=parent.tree_id,
    )

    assert reconstruct_context(parent, result.delta_capsule) == (
        result.reconstructed_capsule
    )
    expanded = {
        item.reference_id: item for item in result.reconstructed_capsule.evidence
    }["expand-me"]
    assert expanded.summary == body
    assert expanded.referenced_content_id == target

    with pytest.raises(MissingContextReferenceError, match="not present"):
        expand_context_references(
            compiler, parent, ("absent",), store
        )
    empty_store = ContentAddressedContextStore()
    with pytest.raises(MissingContextReferenceError, match="unavailable"):
        expand_context_references(
            compiler, parent, ("expand-me",), empty_store
        )
    with pytest.raises(ChangedTreeContextError, match="tree changed"):
        expand_context_references(
            compiler,
            parent,
            ("expand-me",),
            store,
            tree_id="tree:new",
        )
    with pytest.raises(ContextExpansionCancelled, match="cancelled"):
        expand_context_references(
            compiler,
            parent,
            ("expand-me",),
            store,
            cancelled=True,
        )


def test_semantic_retry_capsule_carries_only_delta_repair_context() -> None:
    compiler, parent, required, optional = _parent()
    failure = _reference("failure:new", "failure-v2", 12)

    result = compile_retry_context(
        compiler,
        parent,
        prior_decision_id="decision:previous",
        diagnostic_receipt_id="diagnostic:stable",
        evidence=(required, optional, failure),
        failure_evidence_ids=("failure:new",),
        changed_files=("src/context.py",),
        changed_symbols=("ContextCompiler.compile_delta",),
        unresolved_requirement_ids=("requirement:coverage",),
        repair_round=2,
        max_repair_rounds=3,
    )

    assert RetryContextCapsule.from_json(
        result.capsule.to_json()
    ) == result.capsule
    assert reconstruct_context(
        parent, result.capsule.delta_capsule
    ) == result.reconstructed_capsule
    wire = render_retry_context(result.capsule)
    assert result.capsule.prior_decision_id == "decision:previous"
    assert result.capsule.diagnostic_receipt_id == "diagnostic:stable"
    assert result.capsule.changed_files == ("src/context.py",)
    assert result.capsule.changed_symbols == (
        "ContextCompiler.compile_delta",
    )
    assert result.capsule.unresolved_requirement_ids == (
        "requirement:coverage",
    )
    assert all(
        field not in wire
        for field in ('"goal":', '"authority":', '"scope":', '"acceptance":')
    )

    with pytest.raises(ChangedTreeContextError, match="invalidated"):
        compile_retry_context(
            compiler,
            parent,
            prior_decision_id="decision:previous",
            diagnostic_receipt_id="diagnostic:stable",
            evidence=(required, optional, failure),
            failure_evidence_ids=("failure:new",),
            tree_id="tree:new",
        )
    with pytest.raises(ContextDeltaError, match="round exceeds"):
        replace(result.capsule, repair_round=4)


def test_paired_semantic_retries_reduce_median_tokens_by_at_least_35_percent() -> None:
    retry_tokens: list[int] = []
    replay_tokens: list[int] = []
    for index in range(7):
        compiler = _compiler()
        core = {
            **CORE,
            "goal": {
                "id": "ASI-G092",
                "summary": ("invariant implementation objective " * 160)
                + str(index),
            },
        }
        required = _reference(
            f"required-{index}", f"required-{index}", 25, required=True
        )
        parent = compiler.compile(
            **BINDING, **core, evidence=(required,)
        ).capsule
        failure = _reference(
            f"failure-{index}", f"failure-{index}", 5
        )
        result = compile_retry_context(
            compiler,
            parent,
            prior_decision_id=f"decision:{index}",
            diagnostic_receipt_id=f"diagnostic:{index}",
            evidence=(required, failure),
            failure_evidence_ids=(failure.reference_id,),
            unresolved_requirement_ids=(f"coverage:{required.reference_id}",),
        )
        retry_tokens.append(
            compiler.estimator.estimate(result.capsule.to_record())
        )
        replay_tokens.append(result.receipt.full_replay_tokens)
        assert set(parent.evidence_coverage_ids).issubset(
            result.reconstructed_capsule.evidence_coverage_ids
        )
        assert result.capsule.unresolved_requirement_ids

    assert median(retry_tokens) <= median(replay_tokens) * 0.65


def test_implementation_daemon_dispatches_delta_and_reuses_diagnostic(
    tmp_path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Context Test"],
        cwd=repo,
        check=True,
    )
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    state_dir = repo / "state"
    state_dir.mkdir()
    task = PortalTask(
        task_id="ASI-006",
        title="Add delta retry contexts",
        status="ready",
        completion="manual",
        priority="P1",
        track="token-efficiency",
        outputs=["src/context.py"],
        validation=["pytest test_context.py"],
        acceptance="Retry evidence remains complete.",
        metadata={"ast symbols": "ContextCompiler, FormalReplanner"},
    )
    daemon = PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        implementation_log_dir=state_dir / "logs",
        implementation_context_budget=ContextBudget(
            max_input_tokens=2_000,
            reserved_output_tokens=100,
            reserved_tool_tokens=20,
            max_items=64,
        ),
        implementation_context_tokenizer=_tokenizer,
        implementation_provider_context_window=2_200,
    )

    full_prompt = daemon._build_implementation_prompt(task, attempt=1)
    daemon._persist_implementation_context_receipt(task, attempt=1)
    diagnostic = daemon.record_implementation_failure_context(
        task,
        {
            "kind": "validation_failure",
            "returncode": 1,
            "reason_codes": ["assertion"],
        },
        changed_files=("src/context.py",),
        changed_symbols=("ContextCompiler.compile_delta",),
        unresolved_requirements=("requirement:test",),
    )
    restarted = PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "restarted-state.json",
        strategy_path=state_dir / "restarted-strategy.json",
        events_path=state_dir / "restarted-events.jsonl",
        repo_root=repo,
        implementation_log_dir=state_dir / "logs",
        implementation_context_budget=ContextBudget(
            max_input_tokens=2_000,
            reserved_output_tokens=100,
            reserved_tool_tokens=20,
            max_items=64,
        ),
        implementation_context_tokenizer=_tokenizer,
        implementation_provider_context_window=2_200,
    )
    retry_prompt = restarted._build_implementation_prompt(task, attempt=2)

    wire = json.loads(retry_prompt)
    assert wire["schema"].endswith("retry-context-capsule@1")
    assert wire["diagnostic_receipt_id"] == diagnostic.receipt_id
    assert wire["changed_files"] == ["src/context.py"]
    assert wire["changed_symbols"] == ["ContextCompiler.compile_delta"]
    assert wire["unresolved_requirement_ids"] == ["requirement:test"]
    assert "Retry evidence remains complete." in full_prompt
    assert "Retry evidence remains complete." not in retry_prompt
    assert restarted._last_implementation_retry is not None
    assert reconstruct_context(
        restarted._last_implementation_retry.delta_result.parent_capsule,
        restarted._last_implementation_retry.capsule.delta_capsule,
    ) == restarted._last_implementation_retry.reconstructed_capsule

    repeated = restarted.record_implementation_failure_context(
        task,
        {
            "reason_codes": ["assertion"],
            "returncode": 1,
            "kind": "validation_failure",
        },
        changed_files=("src/context.py",),
        changed_symbols=("ContextCompiler.compile_delta",),
        unresolved_requirements=("requirement:test",),
    )
    assert repeated.receipt_id == diagnostic.receipt_id
    with pytest.raises(ImplementationRetryDeferred, match="backoff"):
        restarted._build_implementation_prompt(task, attempt=3)


def test_implementation_retry_compacts_oversized_diagnostic_projection(
    tmp_path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Context Test"],
        cwd=repo,
        check=True,
    )
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    state_dir = repo / "state"
    state_dir.mkdir()
    task = PortalTask(
        task_id="ASI-007",
        title="Compact a verbose retry diagnosis",
        status="ready",
        completion="manual",
        priority="P1",
        track="token-efficiency",
        outputs=["src/context.py"],
        validation=["pytest test_context.py"],
        acceptance="Retry evidence remains actionable.",
    )
    budget = ContextBudget(
        max_input_tokens=4_096,
        reserved_output_tokens=100,
        reserved_tool_tokens=20,
        max_items=64,
    )
    tokenizer = lambda text: max(1, len(text.encode("utf-8")) // 4)
    daemon = PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        implementation_log_dir=state_dir / "logs",
        implementation_context_budget=budget,
        implementation_context_tokenizer=tokenizer,
        implementation_provider_context_window=4_500,
    )

    daemon._build_implementation_prompt(task, attempt=1)
    daemon._persist_implementation_context_receipt(task, attempt=1)
    denied_paths = [
        f"denied-{index:03d}-" + ("x" * 260)
        for index in range(50)
    ]
    diagnostic = daemon.record_implementation_failure_context(
        task,
        {
            "kind": "validation_failure",
            "returncode": 78,
            "failure_review": {
                "accepted": False,
                "denied_paths": denied_paths,
            },
        },
        changed_files=("src/context.py",),
    )
    restarted = PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "restarted-state.json",
        strategy_path=state_dir / "restarted-strategy.json",
        events_path=state_dir / "restarted-events.jsonl",
        repo_root=repo,
        implementation_log_dir=state_dir / "logs",
        implementation_context_budget=budget,
        implementation_context_tokenizer=tokenizer,
        implementation_provider_context_window=4_500,
    )

    retry_prompt = restarted._build_implementation_prompt(task, attempt=2)
    wire = json.loads(retry_prompt)

    assert wire["schema"].endswith("retry-context-capsule@1")
    assert wire["diagnostic_receipt_id"] == diagnostic.receipt_id
    assert "denied-000-" in retry_prompt
    assert "denied-049-" not in retry_prompt
    assert restarted._last_implementation_retry is not None


def test_implementation_retry_rebases_once_when_immutable_parent_is_full(
    tmp_path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Context Test"],
        cwd=repo,
        check=True,
    )
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    state_dir = repo / "state"
    state_dir.mkdir()
    task = PortalTask(
        task_id="FVT-086",
        title="Certify the SecPAL artifact intake",
        status="ready",
        completion="manual",
        priority="P0",
        track="formal-verification",
        outputs=["src/secpal.py"],
        validation=["pytest tests/test_secpal.py"],
        acceptance=(
            "Preserve exact artifact authority and reject unsupported "
            "execution."
        ),
    )
    budget = ContextBudget(
        max_input_tokens=4_096,
        reserved_output_tokens=100,
        reserved_tool_tokens=20,
        max_items=64,
    )

    def tokenizer(text: str) -> int:
        count = max(1, len(text.encode("utf-8")) // 12)
        if "## Prior failure review (deterministic)" in text:
            count += 1_000
        return count

    daemon = PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        implementation_log_dir=state_dir / "logs",
        implementation_context_budget=budget,
        implementation_context_tokenizer=tokenizer,
        implementation_provider_context_window=4_500,
    )
    monkeypatch.setattr(
        daemon,
        "_render_todo_vector_context",
        lambda _task: "optional vendor evidence " * 6_000,
    )
    monkeypatch.setattr(
        daemon,
        "_load_todo_vector_context",
        lambda _task: {},
    )

    daemon._build_implementation_prompt(task, attempt=1)
    parent_result = daemon._last_implementation_context
    assert isinstance(parent_result, ContextCompileResult)
    parent = parent_result.capsule
    assert parent.input_tokens > 4_000
    assert parent.expansion_references
    diagnostic = daemon.record_implementation_failure_context(
        task,
        {
            "kind": "validation_failure",
            "returncode": 1,
            "reason_codes": ["reason-" + ("r" * 250)] * 4,
            "failed_commands": ["command-" + ("c" * 250)] * 4,
            "failure_review": {
                "accepted": False,
                "reason_codes": ["review-" + ("v" * 250)] * 4,
                "missing_expected_outputs": [
                    "receipt-" + ("p" * 250)
                ]
                * 4,
                "next_attempt_prompt_addendum": "guidance-" + ("g" * 500),
            },
        },
        changed_files=("src/secpal.py",),
        changed_symbols=("certify_secpal",),
        unresolved_requirements=("requirement:secpal-certification",),
    )
    routes: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        daemon,
        "_decision_runtime_route",
        lambda route, payload: routes.append((route, dict(payload))),
    )

    prompt = daemon._build_implementation_prompt(task, attempt=2)
    fresh_result = daemon._last_implementation_context

    assert isinstance(fresh_result, ContextCompileResult)
    assert daemon._last_implementation_retry is None
    fresh = fresh_result.capsule
    assert fresh.repository_id == parent.repository_id
    assert fresh.tree_id == parent.tree_id
    assert fresh.objective_id == parent.objective_id
    assert fresh.objective_revision == parent.objective_revision
    assert fresh.policy_id == parent.policy_id
    assert fresh.policy_revision == parent.policy_revision
    assert fresh.caller == parent.caller
    assert fresh.stage == parent.stage
    assert fresh.invariant_core_id == parent.invariant_core_id
    assert fresh.invariant_core == parent.invariant_core
    assert fresh.authority == parent.authority
    assert fresh.scope == parent.scope
    assert fresh.budget.max_input_tokens <= parent.budget.max_input_tokens
    assert fresh.budget.max_items <= parent.budget.max_items
    assert fresh.input_tokens <= fresh.budget.max_input_tokens
    rescue_reference = next(
        item
        for item in fresh.evidence
        if item.kind == "implementation-fresh-retry-context"
    )
    assert rescue_reference.required
    rescue_binding = json.loads(rescue_reference.summary)
    assert rescue_binding["mode"] == "bounded_fresh_context_rescue"
    assert rescue_binding["parent_capsule_id"] == parent.capsule_id
    assert rescue_binding["parent_invariant_core_id"] == parent.invariant_core_id
    assert rescue_binding["prior_decision_id"] == diagnostic.prior_decision_id
    assert rescue_binding["diagnostic_receipt_id"] == diagnostic.receipt_id
    assert rescue_binding["diagnostic_failure_id"] == diagnostic.failure_id
    assert rescue_binding["repair_round"] == 1
    assert "guidance-" in rescue_reference.summary
    assert "## Prior failure review (deterministic)" not in prompt
    assert json.loads(prompt)["repository_id"] == parent.repository_id
    prompt_tokens, prompt_token_limit = daemon._implementation_prompt_token_usage(
        task,
        prompt,
    )
    assert prompt_tokens <= prompt_token_limit
    retry_route = next(
        payload
        for route, payload in routes
        if route == "retry"
        and payload.get("mode") == "bounded_fresh_context_rescue"
    )
    assert retry_route["diagnostic_projection_attempts"] == [
        "full",
        "compact",
        "minimal",
    ]
    assert retry_route["reason"] == "delta_full_reconstruction_budget"
    assert any(
        route == "implementation_context"
        and payload.get("mode") == "deterministic_addendum_omitted"
        and payload.get("reason") == "receipt_bound_fresh_retry_context"
        for route, payload in routes
    )
    rebound_diagnostic = daemon._implementation_diagnostics[
        daemon._canonical_ref(task)
    ]
    assert rebound_diagnostic.failure_id == diagnostic.failure_id
    assert rebound_diagnostic.receipt_id != diagnostic.receipt_id
    assert (
        rebound_diagnostic.prior_decision_id
        == fresh_result.receipt.receipt_id
    )

    daemon._persist_implementation_context_receipt(task, attempt=2)
    persisted_diagnostic = json.loads(
        (
            state_dir
            / "logs"
            / "fvt-086-diagnostic-receipt.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        persisted_diagnostic["prior_decision_id"]
        == fresh_result.receipt.receipt_id
    )
    restarted = PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "restarted-state.json",
        strategy_path=state_dir / "restarted-strategy.json",
        events_path=state_dir / "restarted-events.jsonl",
        repo_root=repo,
        implementation_log_dir=state_dir / "logs",
        implementation_context_budget=budget,
        implementation_context_tokenizer=tokenizer,
        implementation_provider_context_window=4_500,
    )
    restart_routes: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        restarted,
        "_decision_runtime_route",
        lambda route, payload: restart_routes.append(
            (route, dict(payload))
        ),
    )
    restart_prompt = restarted._build_implementation_prompt(task, attempt=2)
    assert restart_prompt == prompt
    assert "guidance-" in restart_prompt
    assert "## Prior failure review (deterministic)" not in restart_prompt
    assert any(
        route == "retry"
        and payload.get("mode") == "bounded_fresh_context_reuse"
        for route, payload in restart_routes
    )
    reloaded_diagnostic = restarted._implementation_diagnostics[
        restarted._canonical_ref(task)
    ]
    assert reloaded_diagnostic.receipt_id == rebound_diagnostic.receipt_id
    assert isinstance(
        restarted._last_implementation_context,
        ContextCompileResult,
    )
    assert restarted._implementation_parent(task) == (
        fresh,
        fresh_result.receipt.receipt_id,
    )


def test_implementation_fresh_retry_uses_receipt_bound_projection_for_large_core(
    tmp_path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Context Test"],
        cwd=repo,
        check=True,
    )
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    state_dir = repo / "state"
    state_dir.mkdir()
    task = PortalTask(
        task_id="FVT-088",
        title="Audit every deployment axis end to end",
        status="ready",
        completion="manual",
        priority="P0",
        track="formal-verification",
        outputs=["docs/architecture/assurance-matrix.json"],
        validation=["pytest tests/test_assurance_matrix.py -q"],
        acceptance="Preserve the exact authority-bearing core.",
    )
    budget = ContextBudget(
        max_input_tokens=4_096,
        reserved_output_tokens=100,
        reserved_tool_tokens=20,
        max_items=64,
    )
    daemon = PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        implementation_log_dir=state_dir / "logs",
        implementation_context_budget=budget,
        implementation_provider_context_window=4_500,
    )
    repository_id, tree_id = daemon._implementation_repository_and_tree_ids(
        task
    )
    compiler = ContextCompiler(
        budget,
        provider_context_window=4_500,
    )
    parent_required = ContextReference(
        reference_id="required-parent:0001",
        kind="required-parent-contract",
        tier=ContextTier.INVARIANT,
        referenced_content_id="sha256:" + ("d" * 64),
        repository_id=repository_id,
        tree_id=tree_id,
        summary="required parent authority",
        metadata={
            "required": True,
            "priority": 900,
            "coverage_ids": ("requirement:parent-authority",),
        },
    )
    parent_optional = ContextReference(
        reference_id="optional-parent:0001",
        kind="optional-parent-evidence",
        tier=ContextTier.EVIDENCE,
        referenced_content_id="sha256:" + ("e" * 64),
        repository_id=repository_id,
        tree_id=tree_id,
        summary="o" * 2_000,
        metadata={"priority": 100},
    )
    parent_result = compiler.compile(
        repository_id=repository_id,
        tree_id=tree_id,
        objective_id=task.task_id,
        objective_revision=daemon._canonical_ref(task),
        policy_id="policy:implementation-daemon",
        policy_revision="sha256:" + ("c" * 64),
        caller="agent-supervisor:implementation-daemon",
        stage="implementation",
        goal={"task_id": task.task_id, "instruction": "retry safely"},
        authority={
            "padding_a": "x" * 8_000,
            "padding_b": "y" * 4_000,
        },
        scope={"expected_outputs": tuple(task.outputs)},
        acceptance={"criteria": task.acceptance},
        evidence=(parent_required, parent_optional),
    )
    parent = parent_result.capsule
    assert 3_900 < parent.input_tokens <= parent.budget.max_input_tokens
    assert {item.reference_id for item in parent.evidence} == {
        parent_required.reference_id,
        parent_optional.reference_id,
    }
    daemon._implementation_base_contexts[
        daemon._canonical_ref(task)
    ] = parent_result
    diagnostic = daemon.record_implementation_failure_context(
        task,
        {
            "kind": "validation_failure",
            "returncode": 1,
            "failure_review": {
                "accepted": False,
                "reason_codes": ["validation_command_failed"],
                "failed_commands": [
                    "pytest tests/test_assurance_matrix.py -q"
                ],
                "next_attempt_prompt_addendum": (
                    "Re-run the authoritative assurance matrix validation and "
                    "repair the exact failed assertion. "
                    + ("guidance " * 80)
                ),
            },
        },
        changed_files=(
            "docs/architecture/assurance-matrix.json",
            "tests/test_assurance_matrix.py",
        ),
        changed_symbols=("build_assurance_matrix",),
        unresolved_requirements=("requirement:matrix-current",),
    )
    routes: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        daemon,
        "_decision_runtime_route",
        lambda route, payload: routes.append((route, dict(payload))),
    )

    prompt = daemon._build_implementation_prompt(task, attempt=2)
    fresh_result = daemon._last_implementation_context

    assert isinstance(fresh_result, ContextCompileResult)
    fresh = fresh_result.capsule
    assert fresh.invariant_core_id == parent.invariant_core_id
    assert fresh.invariant_core == parent.invariant_core
    assert fresh.budget == parent.budget
    selected_ids = {item.reference_id for item in fresh.evidence}
    assert parent_required.reference_id in selected_ids
    assert parent_optional.reference_id not in selected_ids
    rescue_reference = next(
        item
        for item in fresh.evidence
        if item.kind == "implementation-fresh-retry-context"
    )
    assert rescue_reference.required
    assert diagnostic.failure_id in rescue_reference.coverage_ids
    assert "requirement:matrix-current" in rescue_reference.coverage_ids
    rescue_binding = json.loads(rescue_reference.summary)
    assert rescue_binding["schema"].endswith(
        "implementation-fresh-retry-context@2"
    )
    assert rescue_binding["diagnostic_receipt_id"] == diagnostic.receipt_id
    assert rescue_binding["diagnostic_failure_id"] == diagnostic.failure_id
    assert rescue_binding["parent_capsule_id"] == parent.capsule_id
    assert rescue_binding["parent_invariant_core_id"] == parent.invariant_core_id
    assert rescue_binding["changed_files_id"] == content_identity(
        list(diagnostic.changed_files)
    )
    assert rescue_binding["changed_symbols_id"] == content_identity(
        list(diagnostic.changed_symbols)
    )
    assert rescue_binding["unresolved_requirements_id"] == content_identity(
        list(diagnostic.unresolved_requirements)
    )
    assert "changed_files" not in rescue_binding
    assert "changed_symbols" not in rescue_binding
    assert "unresolved_requirement_ids" not in rescue_binding
    assert rescue_binding["failure"]["kind"] == "validation_failure"
    assert "Re-run the authoritative" in rescue_binding["failure"][
        "next_attempt_prompt_addendum"
    ]
    assert json.loads(prompt)["repository_id"] == parent.repository_id
    prompt_tokens, prompt_limit = daemon._implementation_prompt_token_usage(
        task,
        prompt,
    )
    assert prompt_tokens <= prompt_limit
    retry_route = next(
        payload
        for route, payload in routes
        if route == "retry"
        and payload.get("mode") == "bounded_fresh_context_rescue"
    )
    assert retry_route["rescue_binding_projection"] == (
        "receipt_bound_actionable"
    )
    assert retry_route["rescue_binding_projection_attempts"] == [
        "detailed",
        "receipt_bound_actionable",
    ]


def test_implementation_fresh_retry_rescue_fails_closed_when_required_evidence_does_not_fit(
    tmp_path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Context Test"],
        cwd=repo,
        check=True,
    )
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    state_dir = repo / "state"
    state_dir.mkdir()
    task = PortalTask(
        task_id="FVT-086",
        title="Reject an over-budget retry rescue",
        status="ready",
        completion="manual",
        priority="P0",
        track="formal-verification",
        outputs=["src/secpal.py"],
        validation=["pytest tests/test_secpal.py"],
        acceptance="Never widen context authority.",
    )
    daemon = PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        implementation_log_dir=state_dir / "logs",
        implementation_context_budget=ContextBudget(
            max_input_tokens=4_096,
            reserved_output_tokens=100,
            reserved_tool_tokens=20,
            max_items=64,
        ),
        implementation_context_tokenizer=lambda text: max(
            1, len(text.encode("utf-8")) // 12
        ),
        implementation_provider_context_window=4_500,
    )
    monkeypatch.setattr(
        daemon,
        "_render_todo_vector_context",
        lambda _task: "optional vendor evidence " * 6_000,
    )
    monkeypatch.setattr(
        daemon,
        "_load_todo_vector_context",
        lambda _task: {},
    )
    daemon._build_implementation_prompt(task, attempt=1)
    parent_result = daemon._last_implementation_context
    assert isinstance(parent_result, ContextCompileResult)
    parent = parent_result.capsule
    daemon.record_implementation_failure_context(
        task,
        {"kind": "validation_failure", "returncode": 1},
    )
    monkeypatch.setattr(
        daemon,
        "_implementation_retry_diagnostic_projections",
        lambda _diagnostic: (
            ("mandatory-oversize", {"reason": "x" * 80_000}),
        ),
    )
    routes: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        daemon,
        "_decision_runtime_route",
        lambda route, payload: routes.append((route, dict(payload))),
    )

    with pytest.raises(
        ImplementationRetryDeferred,
        match="implementation retry context budget exhausted",
    ):
        daemon._build_implementation_prompt(task, attempt=2)

    assert daemon._last_implementation_context is parent_result
    assert daemon._last_implementation_retry is None
    assert daemon._implementation_parent(task) == (
        parent,
        parent_result.receipt.receipt_id,
    )
    assert routes == []


def test_implementation_fresh_retry_rescue_translates_max_items_overflow(
    tmp_path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Context Test"],
        cwd=repo,
        check=True,
    )
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    state_dir = repo / "state"
    state_dir.mkdir()
    task = PortalTask(
        task_id="FVT-086",
        title="Bound retry reference count",
        status="ready",
        completion="manual",
        priority="P0",
        track="formal-verification",
        outputs=["src/secpal.py"],
        validation=["pytest tests/test_secpal.py"],
        acceptance="Never widen context item authority.",
    )
    daemon = PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        implementation_log_dir=state_dir / "logs",
        implementation_context_budget=ContextBudget(
            max_input_tokens=4_096,
            reserved_output_tokens=100,
            reserved_tool_tokens=20,
            max_items=64,
            max_serialized_bytes=1_048_576,
        ),
        implementation_context_tokenizer=lambda _text: 1,
        implementation_provider_context_window=4_500,
    )
    monkeypatch.setattr(
        daemon,
        "_render_todo_vector_context",
        lambda _task: (("x" * 6_100) + "\n") * 64,
    )
    monkeypatch.setattr(
        daemon,
        "_load_todo_vector_context",
        lambda _task: {},
    )
    daemon._build_implementation_prompt(task, attempt=1)
    parent_result = daemon._last_implementation_context
    assert isinstance(parent_result, ContextCompileResult)
    assert len(parent_result.capsule.evidence) == 64
    assert not parent_result.capsule.expansion_references
    daemon.record_implementation_failure_context(
        task,
        {"kind": "validation_failure", "returncode": 1},
    )

    with pytest.raises(
        ImplementationRetryDeferred,
        match="implementation retry context budget exhausted",
    ) as caught:
        daemon._build_implementation_prompt(task, attempt=2)

    assert isinstance(caught.value.__cause__, ContextBoundsError)
    assert daemon._last_implementation_context is parent_result
    assert daemon._last_implementation_retry is None


def test_delta_result_exposes_exact_invariant_core_preservation() -> None:
    compiler, parent, required, optional = _parent()
    changed = replace(
        optional,
        referenced_content_id="sha256:changed-diagnostic",
    )
    result = compiler.compile_delta(
        parent,
        evidence=(required, changed),
    )

    assert result.invariant_core_preserved
    assert (
        result.parent_capsule.invariant_core_id
        == result.reconstructed_capsule.invariant_core_id
    )
    assert result.reconstructed_capsule.invariant_core == parent.invariant_core
