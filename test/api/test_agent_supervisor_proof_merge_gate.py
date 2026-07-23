from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
    EvidenceAuthority,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_policy import (
    FormalVerificationPolicy,
    MergeProofGateReceipt,
    OverrideReceipt,
    PolicyValidationError,
    ProofPolicyRule,
    ProofResultStatus,
    RiskLevel,
    RolloutMode,
    ValidationOutcome,
)
from ipfs_accelerate_py.agent_supervisor.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.merge_train import MergeTrain


PROTECTED_PATH = "protected/lease.py"
INVARIANT = "lease_safety"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _candidate_repo(tmp_path: Path, *, advance_target: bool = False) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Proof Merge Gate Test")
    _git(repo, "config", "user.email", "proof-merge-gate@example.invalid")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")

    _git(repo, "switch", "-c", "implementation/proof-gate")
    protected = repo / PROTECTED_PATH
    protected.parent.mkdir(parents=True)
    protected.write_text("def acquire():\n    return 'lease'\n", encoding="utf-8")
    _git(repo, "add", PROTECTED_PATH)
    _git(repo, "commit", "-m", "change protected lease")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    if advance_target:
        (repo / "target.txt").write_text("new target input\n", encoding="utf-8")
        _git(repo, "add", "target.txt")
        _git(repo, "commit", "-m", "advance target")
    return repo, base, candidate


def _policy(
    mode: RolloutMode,
    *,
    allow_fallback: bool = False,
    canary_paths: tuple[str, ...] = (),
) -> FormalVerificationPolicy:
    return FormalVerificationPolicy(
        name="proof-merge-gate-test",
        version="1",
        rollout_mode=mode,
        rules=(
            ProofPolicyRule(
                rule_id="protected-lease",
                path_patterns=("protected/**",),
                minimum_risk=RiskLevel.HIGH,
                invariant_classes=(INVARIANT,),
                required_assurance=AssuranceLevel.KERNEL_VERIFIED,
                fallback_validations=(
                    ("pytest:lease-regression",) if allow_fallback else ()
                ),
                allow_fallback=allow_fallback,
            ),
        ),
        canary_path_patterns=canary_paths,
        canary_percent=0,
        canary_salt="proof-merge-gate-test",
    )


def _enqueue(
    queue: MergeQueue,
    candidate: str,
    base: str,
    *,
    path: str = PROTECTED_PATH,
) -> Any:
    return queue.enqueue(
        branch_name="implementation/proof-gate",
        task_id="REF-268",
        canonical_task_id="canonical-ref-268",
        commit_sha=candidate,
        priority="P0",
        metadata={
            "baseline_ref": base,
            "proof_changed_scopes": [
                {
                    "path": path,
                    "ast_scope_ids": ("function:protected.lease.acquire",),
                    "risk": RiskLevel.CRITICAL.value,
                    "invariant_classes": (INVARIANT,),
                }
            ],
        },
    )


def _plan(policy_id: str, tree_id: str, plan_id: str = "plan:merge-gate") -> dict[str, str]:
    return {
        "plan_id": plan_id,
        "policy_id": policy_id,
        "repository_tree_id": tree_id,
    }


def _proved_receipt(
    *,
    policy_id: str,
    tree_id: str,
    requirement_id: str,
    plan_id: str = "plan:merge-gate",
) -> ProofReceipt:
    obligation_id = "obligation:protected-lease"
    kernel_id = "kernel:lean:4"
    evidence = ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:kernel-check",
        subject_id=obligation_id,
        verifier_id=kernel_id,
        independent=True,
    )
    return ProofReceipt(
        obligation_id=obligation_id,
        plan_id=plan_id,
        attempt_id="attempt:kernel",
        repository_id="repository:proof-merge-gate",
        repository_tree_id=tree_id,
        ast_scope_ids=("function:protected.lease.acquire",),
        premise_ids=("premise:lease-uniqueness",),
        translator_id="translator:python-lean",
        solver_id="solver:z3",
        kernel_id=kernel_id,
        toolchain_id="toolchain:proof-merge-gate",
        policy_id=policy_id,
        resource_budget=ResourceBudget(
            wall_time_ms=10_000,
            cpu_time_ms=8_000,
            memory_bytes=64 * 1024 * 1024,
            max_processes=2,
        ),
        verdict=ProofVerdict.PROVED,
        evidence=(evidence,),
        kernel_receipt_id="kernel-receipt:protected-lease",
        metadata={"requirement_id": requirement_id},
    )


def _proved_packet(**kwargs: Any) -> dict[str, Any]:
    policy = kwargs["policy"]
    selection = kwargs["selection"]
    tree_id = kwargs["repository_tree_id"]
    receipt = _proved_receipt(
        policy_id=policy.policy_id,
        tree_id=tree_id,
        requirement_id=selection.requirements[0].requirement_id,
    )
    return {
        "proof_plan": _plan(policy.policy_id, tree_id),
        "proof_receipts": [receipt],
        "provider_status": {"status": "succeeded"},
    }


def _train(
    repo: Path,
    queue: MergeQueue,
    tmp_path: Path,
    *,
    policy: FormalVerificationPolicy,
    proof_gate: Any,
    merge_callback: Any,
    max_attempts: int = 3,
) -> MergeTrain:
    return MergeTrain(
        repo,
        queue,
        formal_verification_policy=policy,
        proof_gate=proof_gate,
        merge_callback=merge_callback,
        max_attempts=max_attempts,
        state_dir=tmp_path / "train-state",
        proof_cache_dir=tmp_path / "proof-cache",
    )


def test_shadow_missing_assurance_promotes_and_records_exact_would_block(
    tmp_path: Path,
) -> None:
    repo, base, candidate = _candidate_repo(tmp_path)
    queue = MergeQueue(tmp_path / "queue")
    request = _enqueue(queue, candidate, base)
    seen: dict[str, Any] = {}

    def gate(_request: Any, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"provider_status": {"status": "unavailable"}}

    result = _train(
        repo,
        queue,
        tmp_path,
        policy=_policy(RolloutMode.SHADOW),
        proof_gate=gate,
        merge_callback=lambda _request: {"merged": True},
    ).run_once()

    assert result is not None and result["status"] == "merged"
    receipt = MergeProofGateReceipt.from_dict(result["proof_gate"])
    assert receipt.allowed is True
    assert receipt.repository_tree_id == seen["repository_tree_id"]
    assert receipt.selection.requirements[0].path == PROTECTED_PATH
    assert receipt.decision.shadow_would_block == (
        receipt.selection.requirements[0].requirement_id,
    )
    assert receipt.decision.results[0].proof_status is ProofResultStatus.MISSING
    assert receipt.provider_status == {"status": "unavailable"}
    attempt_receipt = next((tmp_path / "train-state/proof-gates/attempts").glob("*.json"))
    persisted = MergeProofGateReceipt.from_dict(
        json.loads(attempt_receipt.read_text())
    )
    assert persisted.receipt_id == receipt.receipt_id
    assert queue.get(request.request_id).status == "completed"  # type: ignore[union-attr]


def test_canary_blocks_configured_path_but_unselected_path_remains_shadow(
    tmp_path: Path,
) -> None:
    blocked_root = tmp_path / "blocked"
    repo, base, candidate = _candidate_repo(blocked_root)
    queue = MergeQueue(blocked_root / "queue")
    _enqueue(queue, candidate, base)
    merge_calls: list[str] = []
    policy = _policy(RolloutMode.CANARY, canary_paths=(PROTECTED_PATH,))
    blocked = _train(
        repo,
        queue,
        blocked_root,
        policy=policy,
        proof_gate=lambda _request, **_kwargs: {},
        merge_callback=lambda _request: merge_calls.append("called") or {"merged": True},
        max_attempts=1,
    ).run_once()

    assert blocked is not None
    assert blocked["reason"] == "proof_gate_blocked"
    assert merge_calls == []
    blocked_receipt = MergeProofGateReceipt.from_dict(blocked["proof_gate"])
    assert blocked_receipt.decision.results[0].effective_mode is RolloutMode.CANARY
    assert blocked_receipt.allowed is False

    shadow_root = tmp_path / "shadow"
    repo, base, candidate = _candidate_repo(shadow_root)
    queue = MergeQueue(shadow_root / "queue")
    _enqueue(queue, candidate, base, path="protected/other.py")
    shadow = _train(
        repo,
        queue,
        shadow_root,
        policy=policy,
        proof_gate=lambda _request, **_kwargs: {},
        merge_callback=lambda _request: {"merged": True},
    ).run_once()

    assert shadow is not None and shadow["status"] == "merged"
    shadow_receipt = MergeProofGateReceipt.from_dict(shadow["proof_gate"])
    assert shadow_receipt.decision.results[0].effective_mode is RolloutMode.SHADOW
    assert shadow_receipt.decision.results[0].would_block is True


@pytest.mark.parametrize("provider_status", (None, "timed_out"))
def test_enforcement_missing_or_timed_out_assurance_blocks_before_callback(
    tmp_path: Path,
    provider_status: str | None,
) -> None:
    root = tmp_path / (provider_status or "missing")
    repo, base, candidate = _candidate_repo(root)
    queue = MergeQueue(root / "queue")
    _enqueue(queue, candidate, base)
    merge_calls: list[str] = []

    def gate(_request: Any, **_kwargs: Any) -> dict[str, Any]:
        return (
            {"provider_status": {"status": provider_status}}
            if provider_status
            else {}
        )

    result = _train(
        repo,
        queue,
        root,
        policy=_policy(RolloutMode.ENFORCEMENT),
        proof_gate=gate,
        merge_callback=lambda _request: merge_calls.append("called") or {"merged": True},
        max_attempts=1,
    ).run_once()

    assert result is not None and result["reason"] == "proof_gate_blocked"
    assert merge_calls == []
    receipt = MergeProofGateReceipt.from_dict(result["proof_gate"])
    assert receipt.allowed is False
    assert receipt.decision.results[0].proof_status is ProofResultStatus.MISSING
    if provider_status:
        assert receipt.provider_status["status"] == provider_status


def test_exact_typed_proof_plan_and_receipt_allow_promotion_and_round_trip(
    tmp_path: Path,
) -> None:
    repo, base, candidate = _candidate_repo(tmp_path)
    queue = MergeQueue(tmp_path / "queue")
    _enqueue(queue, candidate, base)
    policy = _policy(RolloutMode.ENFORCEMENT)

    result = _train(
        repo,
        queue,
        tmp_path,
        policy=policy,
        proof_gate=lambda _request, **kwargs: _proved_packet(**kwargs),
        merge_callback=lambda _request: {"merged": True},
    ).run_once()

    assert result is not None and result["status"] == "merged"
    receipt = MergeProofGateReceipt.from_dict(result["proof_gate"])
    assert receipt.allowed is True
    assert receipt.policy_id == policy.policy_id
    assert receipt.proof_plan_id == "plan:merge-gate"
    assert receipt.proof_receipt_ids == (receipt.proof_receipts[0].receipt_id,)
    assert receipt.decision.results[0].proof_receipt_id == receipt.proof_receipt_ids[0]
    assert receipt.decision.results[0].proof_satisfied is True
    assert receipt.selection_id == receipt.decision.selection_id
    assert receipt.repository_tree_id == result["repository_tree_id"]
    assert MergeProofGateReceipt.from_json(receipt.to_json()) == receipt


def test_fallback_requires_named_passing_validation_with_durable_receipt(
    tmp_path: Path,
) -> None:
    policy = _policy(RolloutMode.ENFORCEMENT, allow_fallback=True)
    tree_id = "git-tree:fallback"
    selection = policy.select(
        [
            {
                "path": PROTECTED_PATH,
                "risk": "critical",
                "invariant_classes": [INVARIANT],
            }
        ],
        repository_tree_id=tree_id,
    )
    unsupported = {
        "requirement_id": selection.requirements[0].requirement_id,
        "status": "unsupported",
    }
    with pytest.raises(PolicyValidationError, match=r"durable .*receipt"):
        MergeProofGateReceipt.build(
            policy=policy,
            selection=selection,
            repository_tree_id=tree_id,
            outcomes=[unsupported],
            validations=[
                ValidationOutcome("pytest:lease-regression", True, receipt_id="")
            ],
            now="2026-07-23T12:00:00Z",
        )

    receipt = MergeProofGateReceipt.build(
        policy=policy,
        selection=selection,
        repository_tree_id=tree_id,
        outcomes=[unsupported],
        validations=[
            ValidationOutcome(
                "pytest:lease-regression",
                True,
                receipt_id="validation-receipt:lease-regression",
            )
        ],
        now="2026-07-23T12:00:00Z",
    )
    assert receipt.allowed is True
    assert receipt.decision.results[0].fallback_satisfied is True
    assert receipt.validation_receipt_ids == (
        "validation-receipt:lease-regression",
    )


def test_bounded_operator_override_is_bound_into_success_receipt(
    tmp_path: Path,
) -> None:
    repo, base, candidate = _candidate_repo(tmp_path)
    queue = MergeQueue(tmp_path / "queue")
    _enqueue(queue, candidate, base)
    policy = _policy(RolloutMode.ENFORCEMENT)

    def gate(_request: Any, **kwargs: Any) -> dict[str, Any]:
        selection = kwargs["selection"]
        now = datetime.now(timezone.utc)
        override = OverrideReceipt.create(
            policy_id=policy.policy_id,
            repository_tree_id=kwargs["repository_tree_id"],
            paths=(PROTECTED_PATH,),
            ast_scope_ids=selection.requirements[0].ast_scope_ids,
            invariant_classes=selection.requirements[0].invariant_classes,
            actor="did:key:merge-operator",
            reason="Bounded provider outage approved under incident INC-268.",
            ticket_id="INC-268",
            ttl_seconds=300,
            now=now,
        )
        return {
            "proof_outcomes": [
                {
                    "requirement_id": selection.requirements[0].requirement_id,
                    "status": "timed_out",
                }
            ],
            "provider_status": {"status": "timed_out"},
            "override": override,
        }

    result = _train(
        repo,
        queue,
        tmp_path,
        policy=policy,
        proof_gate=gate,
        merge_callback=lambda _request: {"merged": True},
    ).run_once()

    assert result is not None and result["status"] == "merged"
    receipt = MergeProofGateReceipt.from_dict(result["proof_gate"])
    assert receipt.allowed is True
    assert receipt.override is not None
    assert receipt.override_receipt_id == receipt.decision.override_receipt_id
    assert receipt.override.actor == "did:key:merge-operator"
    assert receipt.decision.results[0].requirement_satisfied is False
    assert "bounded_override_applied" in receipt.decision.results[0].reason_codes


def test_retry_reuses_exact_valid_cached_receipt_without_reinvoking_provider(
    tmp_path: Path,
) -> None:
    repo, base, candidate = _candidate_repo(tmp_path)
    queue = MergeQueue(tmp_path / "queue", max_attempts=3)
    _enqueue(queue, candidate, base)
    gate_calls = 0
    merge_calls = 0

    def gate(_request: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal gate_calls
        gate_calls += 1
        return _proved_packet(**kwargs)

    def merge(_request: Any) -> dict[str, Any]:
        nonlocal merge_calls
        merge_calls += 1
        if merge_calls == 1:
            return {"merged": False, "reason": "temporary_merge_failure"}
        return {"merged": True}

    train = _train(
        repo,
        queue,
        tmp_path,
        policy=_policy(RolloutMode.ENFORCEMENT),
        proof_gate=gate,
        merge_callback=merge,
        max_attempts=3,
    )
    first = train.run_once()
    second = train.run_once()

    assert first is not None and first["status"] == "retrying"
    assert second is not None and second["status"] == "merged"
    assert gate_calls == 1
    assert merge_calls == 2
    first_receipt = MergeProofGateReceipt.from_dict(first["proof_gate"])
    second_receipt = MergeProofGateReceipt.from_dict(second["proof_gate"])
    assert second_receipt.cache_status["status"] == "hit"
    assert second_receipt.cache_status["reused_receipt_id"] == first_receipt.receipt_id
    assert second_receipt.proof_receipt_ids == first_receipt.proof_receipt_ids
    assert second_receipt.proof_plan_id == first_receipt.proof_plan_id
    assert first["proof_gate"]["policy_id"] == second["proof_gate"]["policy_id"]
    assert first["proof_gate"]["repository_tree_id"] == second["proof_gate"]["repository_tree_id"]


def test_provider_timeout_retry_cannot_repin_candidate_to_weaker_policy(
    tmp_path: Path,
) -> None:
    repo, base, candidate = _candidate_repo(tmp_path)
    queue = MergeQueue(tmp_path / "queue", max_attempts=3)
    _enqueue(queue, candidate, base)
    enforcement = _policy(RolloutMode.ENFORCEMENT)
    first_train = _train(
        repo,
        queue,
        tmp_path,
        policy=enforcement,
        proof_gate=lambda _request, **_kwargs: {
            "provider_status": {"status": "timed_out"}
        },
        merge_callback=lambda _request: {"merged": True},
        max_attempts=3,
    )
    first = first_train.run_once()
    assert first is not None and first["status"] == "retrying"
    assert first["reason"] == "proof_gate_blocked"

    weaker_gate_calls: list[str] = []
    weaker_train = _train(
        repo,
        queue,
        tmp_path,
        policy=_policy(RolloutMode.SHADOW),
        proof_gate=lambda _request, **_kwargs: weaker_gate_calls.append("called") or {},
        merge_callback=lambda _request: {"merged": True},
        max_attempts=3,
    )
    second = weaker_train.run_once()

    assert second is not None
    assert second["reason"] == "proof_gate_identity_invalid"
    assert second["retryable"] is False
    assert weaker_gate_calls == []
    assert second["proof_gate"]["policy_id"] == weaker_train.formal_verification_policy.policy_id


@pytest.mark.parametrize("malformation", ("tree", "plan"))
def test_malformed_tree_or_plan_identity_fails_closed(
    tmp_path: Path,
    malformation: str,
) -> None:
    root = tmp_path / malformation
    repo, base, candidate = _candidate_repo(root)
    queue = MergeQueue(root / "queue")
    _enqueue(queue, candidate, base)
    merge_calls: list[str] = []

    def gate(_request: Any, **kwargs: Any) -> dict[str, Any]:
        if malformation == "tree":
            return {"repository_tree_id": "git-tree:wrong"}
        return {
            "proof_plan": {
                "plan_id": "plan:wrong",
                "policy_id": kwargs["policy"].policy_id,
                "repository_tree_id": "git-tree:wrong",
            }
        }

    result = _train(
        repo,
        queue,
        root,
        policy=_policy(RolloutMode.ENFORCEMENT),
        proof_gate=gate,
        merge_callback=lambda _request: merge_calls.append("called") or {"merged": True},
        max_attempts=1,
    ).run_once()

    assert result is not None
    assert result["reason"] == "proof_gate_identity_invalid"
    assert result["retryable"] is False
    assert merge_calls == []
    assert "provider_error" in result["proof_gate"]


def test_builtin_rebase_refuses_to_promote_a_tree_different_from_proved_tree(
    tmp_path: Path,
) -> None:
    repo, base, candidate = _candidate_repo(tmp_path, advance_target=True)
    target_before = _git(repo, "rev-parse", "main")
    queue = MergeQueue(tmp_path / "queue")
    _enqueue(queue, candidate, base)

    result = _train(
        repo,
        queue,
        tmp_path,
        policy=_policy(RolloutMode.SHADOW),
        proof_gate=lambda _request, **_kwargs: {},
        merge_callback=None,
        max_attempts=1,
    ).run_once()

    assert result is not None
    assert result["reason"] == "proof_gate_tree_mismatch"
    assert result["proof_repository_tree_id"] != result[
        "integration_repository_tree_id"
    ]
    assert _git(repo, "rev-parse", "main") == target_before
