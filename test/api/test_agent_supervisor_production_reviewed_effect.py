"""Reviewed provider bytes stay bound to one exact task, packet, and commit."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.contract_packet_provider_router import (
    ImplementationProviderRouter,
    ProductionContractPacket,
    bind_applied_patch_to_review_chain,
    build_production_contract_packet,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.llm import (
    LLM_USAGE_MODE_ENFORCE,
    LlmChildResultEnvelope,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_context_slice import (
    build_production_context_slice,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_provider_attestation import (
    ProductionProviderReviewAuthority,
    verify_production_provider_review_attestation,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_provider_cli import (
    ProductionCLIProviderPolicy,
    build_production_cli_provider_pair,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_reviewed_effect import (
    ProductionReviewedEffectBinding,
    capture_production_reviewed_effect,
    finalize_production_reviewed_effect,
    production_task_contract,
    verify_finalized_production_reviewed_effect,
    verify_production_reviewed_workspace,
)
from pytest import MonkeyPatch

TASK_ID = "ASE-EFFECT-001"
TARGET = "src/value.py"
BASELINE = "VALUE = 'baseline'\n"
PROPOSAL_A = "VALUE = 'proposal-a'\n"
PROPOSAL_B = "VALUE = 'proposal-b'\n"
NESTED_ROOT = "vendor/lib"
NESTED_TARGET = f"{NESTED_ROOT}/{TARGET}"
OUTER_TARGET = "root_value.py"


@dataclass(frozen=True)
class _Task:
    task_id: str = TASK_ID
    title: str = "Apply the independently reviewed value"
    status: str = "ready"
    completion: str = ""
    priority: str = "P0"
    track: str = "security"
    depends_on: tuple[str, ...] = ()
    outputs: tuple[str, ...] = (TARGET,)
    validation: tuple[str, ...] = (f"python -m py_compile {TARGET}",)
    acceptance: str = "the exact reviewed bytes are committed"
    metadata: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            object.__setattr__(self, "metadata", {"providers": "grok,codex"})


IDENTITY = {
    "canonical_task_key": "default:ase-effect-001",
    "canonical_task_cid": "cidv1:task-revision:ase-effect-001",
    "display_task_id": TASK_ID,
    "board_namespace": "default",
}


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Reviewed Effect Test")
    _git(repo, "config", "user.email", "effect@example.invalid")
    target = repo / TARGET
    target.parent.mkdir()
    target.write_text(BASELINE, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    return repo


def _repo_with_submodule(tmp_path: Path) -> tuple[Path, Path]:
    child_origin = tmp_path / "child-origin"
    child_origin.mkdir()
    _git(child_origin, "init")
    _git(child_origin, "config", "user.name", "Reviewed Effect Test")
    _git(child_origin, "config", "user.email", "effect@example.invalid")
    target = child_origin / TARGET
    target.parent.mkdir()
    target.write_text(BASELINE, encoding="utf-8")
    _git(child_origin, "add", ".")
    _git(child_origin, "commit", "-m", "child baseline")

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Reviewed Effect Test")
    _git(repo, "config", "user.email", "effect@example.invalid")
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child_origin),
        NESTED_ROOT,
    )
    (repo / OUTER_TARGET).write_text(BASELINE, encoding="utf-8")
    _git(repo, "add", OUTER_TARGET)
    _git(repo, "commit", "-m", "superproject baseline")
    return repo, repo / NESTED_ROOT


def _child_receipt(config) -> LlmChildResultEnvelope:
    return LlmChildResultEnvelope(
        usage_mode=LLM_USAGE_MODE_ENFORCE,
        request_id=config.request_id,
        attempt=config.attempt,
        idempotency_key=config.idempotency_key,
        status="ok",
        effective_provider=str(config.provider or ""),
        text_chars=1,
        exit_code=0,
    )


def _route(
    repo: Path,
    *,
    content: str = PROPOSAL_A,
    patch: str = "",
    task: _Task | None = None,
    allowed_nested_repository_roots: tuple[str, ...] = (),
    capture_allowed_nested_repository_roots: tuple[str, ...] | None = None,
):
    task = task or _Task()
    target_paths = tuple(task.outputs)
    contract = production_task_contract(task, IDENTITY)
    baseline = _git(repo, "rev-parse", "HEAD")
    snapshot = f"git-commit:{baseline}"
    context = build_production_context_slice(
        repo_root=repo,
        task_id=task.task_id,
        task_payload=contract,
        read_paths=task.outputs,
        effect_paths=task.outputs,
        allowed_nested_repository_roots=allowed_nested_repository_roots,
    )
    base_packet = build_production_contract_packet(
        task_id=task.task_id,
        snapshot_id=snapshot,
        write_paths=task.outputs,
        read_paths=task.outputs,
        validation_commands=task.validation,
        acceptance_criteria=task.acceptance,
        packet_id="packet:effect:ase-effect-001",
        extra_goal={
            "title": task.title,
            "priority": task.priority,
            "track": task.track,
        },
    )
    packet = ProductionContractPacket(
        packet_id=base_packet.packet_id,
        snapshot_id=base_packet.snapshot_id,
        task_id=base_packet.task_id,
        payload={**dict(base_packet.payload), **context.provider_payload()},
    )
    policy = ProductionCLIProviderPolicy()

    def invoke(_prompt: str, config):
        if config.provider == policy.grok_provider:
            proposal = {"declared_paths": list(target_paths)}
            if patch:
                proposal["patch"] = patch
            else:
                proposal["files"] = [
                    {"path": target_path, "content": content}
                    for target_path in target_paths
                ]
            output = {"proposal": proposal}
        else:
            output = {"decision": "approve", "findings": []}
        return json.dumps(output), _child_receipt(config)

    grok, codex = build_production_cli_provider_pair(policy, invoker=invoke)

    def writer(proposal, lease_id: str) -> None:
        assert lease_id == "lease:effect:1"
        body = proposal.payload["proposal"]
        if body.get("patch"):
            subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "-"],
                cwd=repo,
                input=body["patch"],
                text=True,
                check=True,
            )
        else:
            for item in body["files"]:
                (repo / item["path"]).write_text(
                    item["content"],
                    encoding="utf-8",
                )

    result = ImplementationProviderRouter(
        grok_provider=grok,
        codex_provider=codex,
        admission_gate=lambda _proposal: {
            "accepted": True,
            "reason_code": "supervisor-admitted",
        },
        writer=writer,
    ).route(
        packet,
        current_snapshot_id=snapshot,
        apply=True,
        writer_lease_id="lease:effect:1",
    )
    binding = capture_production_reviewed_effect(
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        packet=packet,
        route_result=result,
        baseline_ref=baseline,
        allowed_nested_repository_roots=(
            allowed_nested_repository_roots
            if capture_allowed_nested_repository_roots is None
            else capture_allowed_nested_repository_roots
        ),
    )
    return task, baseline, packet, result, binding


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "--all")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _recid(
    binding: ProductionReviewedEffectBinding, **changes
) -> ProductionReviewedEffectBinding:
    candidate = replace(binding, binding_id="", **changes)
    return replace(candidate, binding_id=content_identity(candidate.unsigned_dict()))


def _nested_task() -> _Task:
    return replace(
        _Task(),
        outputs=(NESTED_TARGET,),
        validation=(f"python -m py_compile {NESTED_TARGET}",),
    )


def _nested_patch(proposal: str = "proposal-a") -> str:
    return (
        f"diff --git a/{NESTED_TARGET} b/{NESTED_TARGET}\n"
        f"--- a/{NESTED_TARGET}\n"
        f"+++ b/{NESTED_TARGET}\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 'baseline'\n"
        f"+VALUE = '{proposal}'\n"
    )


def _mixed_patch() -> str:
    return (
        f"diff --git a/{OUTER_TARGET} b/{OUTER_TARGET}\n"
        f"--- a/{OUTER_TARGET}\n"
        f"+++ b/{OUTER_TARGET}\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 'baseline'\n"
        "+VALUE = 'proposal-a'\n" + _nested_patch()
    )


def _nested_route(repo: Path):
    return _route(
        repo,
        task=_nested_task(),
        patch=_nested_patch(),
        allowed_nested_repository_roots=(NESTED_ROOT,),
    )


def test_reviewed_effect_round_trip_reconstructs_exact_commit(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    task, _baseline, _packet, _result, captured = _route(repo)

    legacy_payload = captured.to_dict()
    assert legacy_payload["schema"].endswith("production-reviewed-effect-binding@1")
    assert legacy_payload["interface"] == "ProductionReviewedEffectBinding@1"
    assert "nested_repository_effects" not in legacy_payload
    assert (
        ProductionReviewedEffectBinding.from_dict(legacy_payload).to_dict()
        == legacy_payload
    )

    assert verify_production_reviewed_workspace(
        captured,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
    ).admitted
    commit = _commit(repo, "reviewed proposal A")
    tree = f"git-tree:{_git(repo, 'rev-parse', f'{commit}^{{tree}}')}"
    finalized = finalize_production_reviewed_effect(
        captured,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        implementation_commit=commit,
    )

    assert finalized.implementation_diff_sha256.startswith("sha256:")
    assert finalized.implementation_diff_bytes > 0
    assert verify_finalized_production_reviewed_effect(
        finalized.to_dict(),
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        expected_implementation_commit=commit,
        expected_implementation_tree_id=tree,
    ).admitted


def test_reviewed_effect_rejects_approval_with_findings_at_both_boundaries(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    task, baseline, packet, result, captured = _route(repo)
    assert result.review_proposal is not None
    contradictory_payload = {
        **dict(result.review_proposal.payload),
        "findings": ["approval cannot carry a finding"],
    }
    contradictory_review = replace(
        result.review_proposal,
        payload=contradictory_payload,
    )
    contradictory_result = replace(result, review_proposal=contradictory_review)

    try:
        capture_production_reviewed_effect(
            repo_root=repo,
            task=task,
            task_identity=IDENTITY,
            packet=packet,
            route_result=contradictory_result,
            baseline_ref=baseline,
        )
    except ValueError as error:
        assert "Grok final bytes and Codex approval" in str(error)
    else:  # pragma: no cover - a release-critical fail-closed assertion
        raise AssertionError("contradictory approval must fail effect capture")

    commit = _commit(repo, "reviewed proposal A")
    tree = f"git-tree:{_git(repo, 'rev-parse', f'{commit}^{{tree}}')}"
    finalized = finalize_production_reviewed_effect(
        captured,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        implementation_commit=commit,
    )
    contradictory_binding_payload = json.loads(
        json.dumps(finalized.review_proposal_payload)
    )
    contradictory_binding_payload["findings"] = ["approval cannot carry a finding"]
    contradictory_binding = _recid(
        finalized,
        review_proposal_payload=contradictory_binding_payload,
        review_proposal_payload_cid=content_identity(contradictory_binding_payload),
    )
    verification = verify_finalized_production_reviewed_effect(
        contradictory_binding,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        expected_implementation_commit=commit,
        expected_implementation_tree_id=tree,
    )

    assert not verification.admitted
    assert "reviewed_effect_codex_approval_invalid" in verification.reason_codes


def test_validation_mutation_and_proposal_a_commit_b_are_rejected(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    task, _baseline, _packet, _result, captured = _route(repo)

    (repo / TARGET).write_text(PROPOSAL_B, encoding="utf-8")
    verification = verify_production_reviewed_workspace(
        captured,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
    )
    assert not verification.admitted
    assert (
        "reviewed_effect_workspace_bytes_or_modes_changed" in verification.reason_codes
    )

    commit_b = _commit(repo, "unreviewed proposal B")
    try:
        finalize_production_reviewed_effect(
            captured,
            repo_root=repo,
            task=task,
            task_identity=IDENTITY,
            implementation_commit=commit_b,
        )
    except ValueError as error:
        assert "post-validation reviewed effect changed" in str(error)
    else:  # pragma: no cover - a release-critical fail-closed assertion
        raise AssertionError("proposal A must never attest unrelated commit B")


def test_task_packet_provider_and_child_facts_are_reverified(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    task, _baseline, _packet, _result, captured = _route(repo)
    commit = _commit(repo, "reviewed proposal A")
    tree = f"git-tree:{_git(repo, 'rev-parse', f'{commit}^{{tree}}')}"
    finalized = finalize_production_reviewed_effect(
        captured,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        implementation_commit=commit,
    )

    changed_task = replace(task, acceptance="different acceptance")
    assert not verify_finalized_production_reviewed_effect(
        finalized,
        repo_root=repo,
        task=changed_task,
        task_identity=IDENTITY,
        expected_implementation_commit=commit,
        expected_implementation_tree_id=tree,
    ).admitted


def test_signed_attestation_requires_and_reverifies_finalized_effect(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    task, _baseline, _packet, result, captured = _route(repo)
    commit = _commit(repo, "reviewed proposal A")
    tree = f"git-tree:{_git(repo, 'rev-parse', f'{commit}^{{tree}}')}"
    finalized = finalize_production_reviewed_effect(
        captured,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        implementation_commit=commit,
    )
    chain_binding = bind_applied_patch_to_review_chain(
        result,
        implementation_commit=commit,
    )
    assert chain_binding is not None
    authority = ProductionProviderReviewAuthority.generate()
    policy_id = ProductionCLIProviderPolicy().policy_id

    try:
        authority.issue(
            provider_receipt=result.provider_receipt,
            review_chain_binding=chain_binding,
            provider_policy_id=policy_id,
            implementation_commit=commit,
            implementation_tree_id=tree,
        )
    except ValueError as error:
        assert "provider_reviewed_effect_missing" in str(error)
    else:  # pragma: no cover - a release-critical fail-closed assertion
        raise AssertionError("proposal-only review binding must not be signed")

    attestation = authority.issue(
        provider_receipt=result.provider_receipt,
        review_chain_binding=chain_binding,
        provider_policy_id=policy_id,
        implementation_commit=commit,
        implementation_tree_id=tree,
        reviewed_effect_binding=finalized,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        issued_at_ms=1_800_000_000_000,
        nonce="reviewed-effect-test-nonce-0001",
    )
    verification = verify_production_provider_review_attestation(
        attestation,
        trusted_public_keys={
            authority.issuer_key_id: authority.public_key_bytes,
        },
        provider_receipt=result.provider_receipt,
        review_chain_binding=chain_binding,
        reviewed_effect_binding=finalized.to_dict(),
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        expected_task_id=task.task_id,
        expected_snapshot_id=finalized.snapshot_id,
        expected_provider_policy_id=policy_id,
        expected_implementation_commit=commit,
        expected_implementation_tree_id=tree,
    )

    assert verification.admitted
    assert attestation.reviewed_effect_binding_cid == finalized.binding_id


def test_exact_unified_patch_is_reconstructed_and_substitution_is_rejected(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    patch_a = (
        "diff --git a/src/value.py b/src/value.py\n"
        "--- a/src/value.py\n"
        "+++ b/src/value.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 'baseline'\n"
        "+VALUE = 'proposal-a'\n"
    )
    patch_b = patch_a.replace("proposal-a", "proposal-b")
    object_path = Path(_git(repo, "rev-parse", "--git-path", "objects"))
    if not object_path.is_absolute():
        object_path = repo / object_path
    objects_before = {
        path.relative_to(object_path).as_posix()
        for path in object_path.rglob("*")
        if path.is_file()
    }
    task, _baseline, _packet, _result, captured = _route(repo, patch=patch_a)
    objects_after_capture = {
        path.relative_to(object_path).as_posix()
        for path in object_path.rglob("*")
        if path.is_file()
    }
    assert objects_after_capture == objects_before
    commit = _commit(repo, "reviewed unified patch A")
    tree = f"git-tree:{_git(repo, 'rev-parse', f'{commit}^{{tree}}')}"

    poison = tmp_path / "poison-repository"
    poison.mkdir()
    _git(poison, "init")
    poison_git = poison / ".git"
    poisoned_routing = {
        "GIT_DIR": str(poison_git),
        "GIT_WORK_TREE": str(poison),
        "GIT_COMMON_DIR": str(poison_git),
        "GIT_INDEX_FILE": str(poison_git / "index"),
        "GIT_OBJECT_DIRECTORY": str(poison_git / "objects"),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(poison_git / "objects"),
        "GIT_EXTERNAL_DIFF": str(poison / "untrusted-diff"),
        "GIT_DIFF_OPTS": "--unified=0",
    }
    for key, value in poisoned_routing.items():
        monkeypatch.setenv(key, value)

    finalized = finalize_production_reviewed_effect(
        captured,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        implementation_commit=commit,
    )
    assert verify_finalized_production_reviewed_effect(
        finalized,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        expected_implementation_commit=commit,
        expected_implementation_tree_id=tree,
    ).admitted

    substituted_payload = json.loads(json.dumps(finalized.selected_proposal_payload))
    substituted_payload["proposal"]["patch"] = patch_b
    substituted = _recid(
        finalized,
        selected_proposal_payload=substituted_payload,
        selected_proposal_payload_cid=content_identity(substituted_payload),
    )
    verification = verify_finalized_production_reviewed_effect(
        substituted,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        expected_implementation_commit=commit,
        expected_implementation_tree_id=tree,
    )
    assert not verification.admitted
    assert any(
        "grok_patch_blob_mismatch" in reason for reason in verification.reason_codes
    )

    receipt = json.loads(json.dumps(finalized.provider_receipt))
    receipt["attempts"][0]["configured_model"] = "wrong-model"
    unsigned_receipt = dict(receipt)
    unsigned_receipt.pop("receipt_id")
    receipt["receipt_id"] = content_identity(unsigned_receipt)
    wrong_model = _recid(
        finalized,
        provider_receipt=receipt,
        provider_receipt_cid=receipt["receipt_id"],
    )
    verification = verify_finalized_production_reviewed_effect(
        wrong_model,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        expected_implementation_commit=commit,
        expected_implementation_tree_id=tree,
    )
    assert not verification.admitted
    assert any(
        "provider_execution_invalid" in reason for reason in verification.reason_codes
    )

    receipt = json.loads(json.dumps(finalized.provider_receipt))
    receipt["attempts"][1].pop("child_exit_code")
    unsigned_receipt = dict(receipt)
    unsigned_receipt.pop("receipt_id")
    receipt["receipt_id"] = content_identity(unsigned_receipt)
    missing_child = _recid(
        finalized,
        provider_receipt=receipt,
        provider_receipt_cid=receipt["receipt_id"],
    )
    assert not verify_finalized_production_reviewed_effect(
        missing_child,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        expected_implementation_commit=commit,
        expected_implementation_tree_id=tree,
    ).admitted


def test_registered_submodule_effect_round_trip_binds_outer_and_child_commits(
    tmp_path: Path,
) -> None:
    repo, child = _repo_with_submodule(tmp_path)
    task, baseline, _packet, _result, captured = _nested_route(repo)

    payload = captured.to_dict()
    assert payload["schema"].endswith("production-reviewed-effect-binding@2")
    assert payload["interface"] == "ProductionReviewedEffectBinding@2"
    assert captured.changed_paths == (NESTED_TARGET,)
    assert len(captured.nested_repository_effects) == 1
    nested = captured.nested_repository_effects[0]
    assert nested.root == NESTED_ROOT
    assert nested.changed_paths == (NESTED_TARGET,)
    assert nested.baseline_gitlink_commit == _git(child, "rev-parse", "HEAD")
    assert not nested.implementation_gitlink_commit
    assert ProductionReviewedEffectBinding.from_dict(payload).to_dict() == payload

    assert verify_production_reviewed_workspace(
        captured,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        allowed_nested_repository_roots=(NESTED_ROOT,),
    ).admitted
    assert not verify_production_reviewed_workspace(
        captured,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
    ).admitted

    child_commit = _commit(child, "reviewed nested proposal")
    implementation_commit = _commit(repo, "advance reviewed submodule gitlink")
    tree = f"git-tree:{_git(repo, 'rev-parse', f'{implementation_commit}^{{tree}}')}"
    finalized = finalize_production_reviewed_effect(
        captured,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        implementation_commit=implementation_commit,
        allowed_nested_repository_roots=(NESTED_ROOT,),
    )

    nested = finalized.nested_repository_effects[0]
    assert nested.implementation_gitlink_commit == child_commit
    assert nested.implementation_tree_id.startswith("git-tree:")
    assert nested.implementation_diff_sha256.startswith("sha256:")
    assert nested.implementation_diff_bytes > 0
    assert finalized.implementation_diff_bytes > 0
    assert verify_finalized_production_reviewed_effect(
        finalized.to_dict(),
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        expected_implementation_commit=implementation_commit,
        expected_implementation_tree_id=tree,
        allowed_nested_repository_roots=(NESTED_ROOT,),
    ).admitted
    assert not verify_finalized_production_reviewed_effect(
        finalized,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        expected_implementation_commit=implementation_commit,
        expected_implementation_tree_id=tree,
    ).admitted

    substituted_payload = json.loads(json.dumps(finalized.selected_proposal_payload))
    substituted_payload["proposal"]["patch"] = _nested_patch("proposal-b")
    substituted = _recid(
        finalized,
        selected_proposal_payload=substituted_payload,
        selected_proposal_payload_cid=content_identity(substituted_payload),
    )
    substitution = verify_finalized_production_reviewed_effect(
        substituted,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        expected_implementation_commit=implementation_commit,
        expected_implementation_tree_id=tree,
        allowed_nested_repository_roots=(NESTED_ROOT,),
    )
    assert not substitution.admitted
    assert any(
        "grok_patch_blob_mismatch" in reason for reason in substitution.reason_codes
    )
    assert finalized.baseline_commit == baseline


def test_linked_nested_worktree_effect_attests_from_supervisor_checkout(
    tmp_path: Path,
) -> None:
    """Nested immutable objects verify outside the implementation checkouts."""

    supervisor, supervisor_child = _repo_with_submodule(tmp_path)
    baseline = _git(supervisor, "rev-parse", "HEAD")
    child_baseline = _git(supervisor_child, "rev-parse", "HEAD")
    implementation = tmp_path / "implementation-worktree"
    _git(
        supervisor,
        "worktree",
        "add",
        "-b",
        "reviewed-effect-implementation",
        str(implementation),
        baseline,
    )
    implementation_child = implementation / NESTED_ROOT
    if implementation_child.exists():
        implementation_child.rmdir()
    _git(
        supervisor_child,
        "worktree",
        "add",
        "-b",
        "reviewed-effect-child-implementation",
        str(implementation_child),
        child_baseline,
    )

    task, routed_baseline, _packet, result, captured = _nested_route(
        implementation
    )
    assert routed_baseline == baseline
    assert verify_production_reviewed_workspace(
        captured,
        repo_root=implementation,
        task=task,
        task_identity=IDENTITY,
        allowed_nested_repository_roots=(NESTED_ROOT,),
    ).admitted

    child_commit = _commit(
        implementation_child,
        "reviewed linked-worktree nested proposal",
    )
    implementation_commit = _commit(
        implementation,
        "advance linked-worktree reviewed submodule gitlink",
    )
    implementation_tree_id = (
        "git-tree:"
        + _git(
            supervisor,
            "rev-parse",
            f"{implementation_commit}^{{tree}}",
        )
    )
    finalized = finalize_production_reviewed_effect(
        captured,
        repo_root=implementation,
        task=task,
        task_identity=IDENTITY,
        implementation_commit=implementation_commit,
        allowed_nested_repository_roots=(NESTED_ROOT,),
    )
    assert (
        finalized.nested_repository_effects[0].implementation_gitlink_commit
        == child_commit
    )

    chain_binding = bind_applied_patch_to_review_chain(
        result,
        implementation_commit=implementation_commit,
    )
    assert chain_binding is not None
    authority = ProductionProviderReviewAuthority.generate()
    policy_id = ProductionCLIProviderPolicy().policy_id
    assert _git(supervisor_child, "rev-parse", "HEAD") == child_baseline
    attestation = authority.issue(
        provider_receipt=result.provider_receipt,
        review_chain_binding=chain_binding,
        provider_policy_id=policy_id,
        implementation_commit=implementation_commit,
        implementation_tree_id=implementation_tree_id,
        reviewed_effect_binding=finalized,
        repo_root=supervisor,
        task=task,
        task_identity=IDENTITY,
        allowed_nested_repository_roots=(NESTED_ROOT,),
        issued_at_ms=1_800_000_000_000,
        nonce="linked-nested-effect-test-nonce-0001",
    )
    verification = verify_production_provider_review_attestation(
        attestation,
        trusted_public_keys={
            authority.issuer_key_id: authority.public_key_bytes,
        },
        provider_receipt=result.provider_receipt,
        review_chain_binding=chain_binding,
        reviewed_effect_binding=finalized.to_dict(),
        repo_root=supervisor,
        task=task,
        task_identity=IDENTITY,
        expected_task_id=task.task_id,
        expected_snapshot_id=finalized.snapshot_id,
        expected_provider_policy_id=policy_id,
        expected_implementation_commit=implementation_commit,
        expected_implementation_tree_id=implementation_tree_id,
        allowed_nested_repository_roots=(NESTED_ROOT,),
    )

    assert verification.admitted
    assert attestation.reviewed_effect_binding_cid == finalized.binding_id
    assert _git(supervisor_child, "rev-parse", "HEAD") == child_baseline


def test_mixed_outer_and_registered_submodule_effects_flatten_deterministically(
    tmp_path: Path,
) -> None:
    repo, child = _repo_with_submodule(tmp_path)
    task = replace(
        _Task(),
        outputs=(OUTER_TARGET, NESTED_TARGET),
        validation=(f"python -m py_compile {OUTER_TARGET} {NESTED_TARGET}",),
    )
    task, _baseline, _packet, _result, captured = _route(
        repo,
        task=task,
        patch=_mixed_patch(),
        allowed_nested_repository_roots=(NESTED_ROOT,),
    )

    assert captured.changed_paths == (OUTER_TARGET, NESTED_TARGET)
    assert tuple(effect.path for effect in captured.path_effects) == (
        OUTER_TARGET,
        NESTED_TARGET,
    )
    assert captured.nested_repository_effects[0].changed_paths == (NESTED_TARGET,)
    assert verify_production_reviewed_workspace(
        captured,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        allowed_nested_repository_roots=(NESTED_ROOT,),
    ).admitted

    _commit(child, "reviewed nested half")
    implementation_commit = _commit(repo, "reviewed mixed outer and nested effect")
    tree = f"git-tree:{_git(repo, 'rev-parse', f'{implementation_commit}^{{tree}}')}"
    finalized = finalize_production_reviewed_effect(
        captured,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        implementation_commit=implementation_commit,
        allowed_nested_repository_roots=(NESTED_ROOT,),
    )
    assert verify_finalized_production_reviewed_effect(
        finalized,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        expected_implementation_commit=implementation_commit,
        expected_implementation_tree_id=tree,
        allowed_nested_repository_roots=(NESTED_ROOT,),
    ).admitted


def test_nested_effect_capture_rejects_an_unregistered_gitlink(tmp_path: Path) -> None:
    repo, _child = _repo_with_submodule(tmp_path)
    try:
        _route(
            repo,
            task=_nested_task(),
            patch=_nested_patch(),
            allowed_nested_repository_roots=(NESTED_ROOT,),
            capture_allowed_nested_repository_roots=(),
        )
    except ValueError as error:
        assert "paths do not match" in str(error) or "nested" in str(error)
    else:  # pragma: no cover - a release-critical fail-closed assertion
        raise AssertionError("unregistered nested effects must fail capture")


def test_nested_effect_rejects_stale_child_head_and_symlink_tampering(
    tmp_path: Path,
) -> None:
    stale_root = tmp_path / "stale"
    stale_root.mkdir()
    repo, child = _repo_with_submodule(stale_root)
    task, _baseline, _packet, _result, captured = _nested_route(repo)
    _commit(child, "premature child head movement")
    stale = verify_production_reviewed_workspace(
        captured,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        allowed_nested_repository_roots=(NESTED_ROOT,),
    )
    assert not stale.admitted
    assert "reviewed_effect_workspace_reconstruction_failed" in stale.reason_codes

    symlink_root = tmp_path / "symlink"
    symlink_root.mkdir()
    repo, child = _repo_with_submodule(symlink_root)
    task, _baseline, _packet, _result, captured = _nested_route(repo)
    target = child / TARGET
    target.unlink()
    target.symlink_to("/etc/passwd")
    symlinked = verify_production_reviewed_workspace(
        captured,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        allowed_nested_repository_roots=(NESTED_ROOT,),
    )
    assert not symlinked.admitted
    assert "reviewed_effect_workspace_reconstruction_failed" in symlinked.reason_codes


def test_nested_effect_rejects_deeper_repository_and_finalized_gitlink_tamper(
    tmp_path: Path,
) -> None:
    deeper_root = tmp_path / "deeper"
    deeper_root.mkdir()
    repo, child = _repo_with_submodule(deeper_root)
    task, _baseline, _packet, _result, captured = _nested_route(repo)
    deeper = child / "deeper"
    deeper.mkdir()
    _git(deeper, "init")
    _git(deeper, "config", "user.name", "Reviewed Effect Test")
    _git(deeper, "config", "user.email", "effect@example.invalid")
    (deeper / "value.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(deeper, "add", ".")
    _git(deeper, "commit", "-m", "unbound deeper repository")
    nested = verify_production_reviewed_workspace(
        captured,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        allowed_nested_repository_roots=(NESTED_ROOT,),
    )
    assert not nested.admitted

    final_root = tmp_path / "final"
    final_root.mkdir()
    repo, child = _repo_with_submodule(final_root)
    task, _baseline, _packet, _result, captured = _nested_route(repo)
    child_baseline = captured.nested_repository_effects[0].baseline_gitlink_commit
    _commit(child, "reviewed nested proposal")
    implementation_commit = _commit(repo, "advance reviewed submodule gitlink")
    tree = f"git-tree:{_git(repo, 'rev-parse', f'{implementation_commit}^{{tree}}')}"
    finalized = finalize_production_reviewed_effect(
        captured,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        implementation_commit=implementation_commit,
        allowed_nested_repository_roots=(NESTED_ROOT,),
    )
    _git(child, "checkout", "--detach", child_baseline)
    historical = verify_finalized_production_reviewed_effect(
        finalized,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        expected_implementation_commit=implementation_commit,
        expected_implementation_tree_id=tree,
        allowed_nested_repository_roots=(NESTED_ROOT,),
    )
    assert historical.admitted

    tampered_nested = replace(
        finalized.nested_repository_effects[0],
        implementation_gitlink_commit=child_baseline,
    )
    tampered_binding = _recid(
        finalized,
        nested_repository_effects=(tampered_nested,),
    )
    tampered = verify_finalized_production_reviewed_effect(
        tampered_binding,
        repo_root=repo,
        task=task,
        task_identity=IDENTITY,
        expected_implementation_commit=implementation_commit,
        expected_implementation_tree_id=tree,
        allowed_nested_repository_roots=(NESTED_ROOT,),
    )
    assert not tampered.admitted
    assert "reviewed_effect_nested_gitlink_mismatch:vendor/lib" in tampered.reason_codes
