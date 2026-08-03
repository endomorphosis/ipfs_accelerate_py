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


def _route(repo: Path, *, content: str = PROPOSAL_A, patch: str = ""):
    task = _Task()
    contract = production_task_contract(task, IDENTITY)
    baseline = _git(repo, "rev-parse", "HEAD")
    snapshot = f"git-commit:{baseline}"
    context = build_production_context_slice(
        repo_root=repo,
        task_id=task.task_id,
        task_payload=contract,
        read_paths=task.outputs,
        effect_paths=task.outputs,
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
            proposal = {"declared_paths": [TARGET]}
            if patch:
                proposal["patch"] = patch
            else:
                proposal["files"] = [{"path": TARGET, "content": content}]
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
            (repo / body["files"][0]["path"]).write_text(
                body["files"][0]["content"],
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
    )
    return task, baseline, packet, result, binding


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "--all")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _recid(binding: ProductionReviewedEffectBinding, **changes) -> ProductionReviewedEffectBinding:
    candidate = replace(binding, binding_id="", **changes)
    return replace(candidate, binding_id=content_identity(candidate.unsigned_dict()))


def test_reviewed_effect_round_trip_reconstructs_exact_commit(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    task, _baseline, _packet, _result, captured = _route(repo)

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


def test_validation_mutation_and_proposal_a_commit_b_are_rejected(tmp_path: Path) -> None:
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
    assert "reviewed_effect_workspace_bytes_or_modes_changed" in verification.reason_codes

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
    assert any("grok_patch_blob_mismatch" in reason for reason in verification.reason_codes)

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
    assert any("provider_execution_invalid" in reason for reason in verification.reason_codes)

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
