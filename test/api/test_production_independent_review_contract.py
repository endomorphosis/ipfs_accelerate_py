"""Joined offline production routing preserves operator and Git authority.

Provider invokers return bounded authored proposals and child envelopes.  The
daemon, native writer, Git reconstruction, and Ed25519 verifier run unchanged.
No external model transport is called and review alone never completes a task.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.contract_packet_provider_router import (
    ProviderRole,
    ProviderQuotaError,
    RouteStatus,
    bind_applied_patch_to_review_chain,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    PortalTaskState,
    TodoImplementationDaemon,
    parse_task_file,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.llm import (
    LLM_USAGE_MODE_ENFORCE,
    LlmChildResultEnvelope,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_provider_attestation import (
    ProductionProviderReviewAuthority,
    verify_production_provider_review_attestation,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_provider_cli import (
    PRODUCTION_CLI_POLICY_NAME,
    ProductionCLIProviderPolicy,
    build_production_cli_provider_pair,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_reviewed_effect import (
    finalize_production_reviewed_effect,
    verify_production_reviewed_workspace,
)

TARGET = "src/value.py"
BASELINE = "VALUE = 'baseline'\n"
REVIEWED = "VALUE = 'reviewed'\n"


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repo, check=True, text=True,
        capture_output=True,
    ).stdout.strip()


def _fixture(tmp_path: Path, *, metadata: dict[str, Any] | None = None, ephemeral=False):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Joined Review Test")
    _git(repo, "config", "user.email", "joined-review@example.invalid")
    (repo / "src").mkdir()
    (repo / TARGET).write_text(BASELINE, encoding="utf-8")
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    todo = repo / "tasks.todo.md"
    todo.write_text("# Tasks\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    state = tmp_path / "state"
    key = state / "operator-review.ed25519"
    daemon = TodoImplementationDaemon(
        todo_path=todo,
        state_path=state / "task_state.json",
        strategy_path=state / "strategy.json",
        events_path=state / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## JOIN-",
        implement=True,
        use_ephemeral_worktree=ephemeral,
        worktree_root=tmp_path / "worktrees",
        production_provider_policy=PRODUCTION_CLI_POLICY_NAME,
        production_provider_review_authority_key_path=key,
    )
    task = PortalTask(
        task_id="JOIN-001", title="Review the exact value change", status="ready",
        completion="manual", priority="P0", track="provider-review",
        outputs=[TARGET], validation=[f"python -m py_compile {TARGET}"],
        acceptance="the exact independently reviewed bytes are committed",
        metadata={"Provider role": "grok-implement, codex-review", **(metadata or {})},
    )
    return daemon, task, repo, key


def _live_fixture(tmp_path: Path, *, validation: str = ""):
    daemon, _task, repo, key = _fixture(tmp_path, ephemeral=True)
    command = validation or f"python -m py_compile {TARGET}"
    daemon.todo_path.write_text(
        "# Tasks\n\n## JOIN-001 Review the exact value change\n"
        "- Status: ready\n- Completion: manual\n- Priority: P0\n"
        "- Track: provider-review\n"
        f"- Outputs: {TARGET}\n- Validation: {command}\n"
        "- Acceptance: the exact independently reviewed bytes are committed\n"
        "- Provider role: grok-implement, codex-review\n",
        encoding="utf-8",
    )
    _git(repo, "add", "--", "tasks.todo.md")
    _git(repo, "commit", "--amend", "--no-edit")
    task, = parse_task_file(daemon.todo_path, daemon.task_header_prefix)
    return daemon, task, repo, key


def _providers(daemon, task, repo, *, fault: str = "", policy=None, files=None, patch=""):
    calls: list[str] = []
    selected = policy or daemon.production_provider_policy

    def invoke(prompt, config):
        request = json.loads(prompt)
        role = request["role"]
        calls.append(role)
        assert "repository_corpus" not in prompt
        if role == ProviderRole.GROK_IMPLEMENT.value:
            if fault == "source_drift":
                (repo / TARGET).write_text("VALUE = 'external change'\n", encoding="utf-8")
            elif fault == "task_drift":
                task.outputs.append("src/extra.py")
            elif fault == "policy_drift":
                daemon.production_provider_policy = replace(selected, context_budget_tokens=3072)
            replacements = files or [{"path": TARGET, "content": REVIEWED}]
            proposal = {
                "declared_paths": [item["path"] for item in replacements],
                "files": replacements,
            }
            if patch:
                proposal = {"declared_paths": [TARGET], "patch": patch}
            if fault == "undeclared_path":
                proposal["declared_paths"] = ["src/unowned.py"]
                proposal["files"] = [{"path": "src/unowned.py", "content": REVIEWED}]
            output = {"proposal": proposal}
        else:
            assert role == ProviderRole.CODEX_REVIEW.value
            assert set(request["provider_input"]) == {
                "admitted_implementation_proposal", "evidence_slice",
            }
            output = {"decision": "approve", "findings": []}
            if fault == "review_rewrite":
                output["proposal"] = {
                    "declared_paths": [TARGET],
                    "files": [{"path": TARGET, "content": "VALUE = 'reviewer rewrite'\n"}],
                }
            elif fault == "review_decline":
                output = {"decision": "reject", "findings": ["acceptance_unsatisfied"]}
            elif fault == "review_quota":
                raise ProviderQuotaError("independent reviewer capacity exhausted")
        envelope = LlmChildResultEnvelope(
            usage_mode=LLM_USAGE_MODE_ENFORCE,
            request_id=config.request_id, attempt=config.attempt,
            idempotency_key=config.idempotency_key, status="ok",
            effective_provider=str(config.provider or ""), text_chars=1, exit_code=0,
        )
        if fault == "effective_provider" and role == ProviderRole.CODEX_REVIEW.value:
            envelope = replace(envelope, effective_provider="grok_cli")
        return json.dumps(output), envelope

    daemon._production_grok_provider, daemon._production_codex_provider = (
        build_production_cli_provider_pair(selected, invoker=invoke)
    )
    return calls


def _route(daemon, task, repo, **kwargs):
    return daemon.run_production_model_assisted_route(
        task, attempt=1, workspace_path=repo,
        baseline_ref=_git(repo, "rev-parse", "HEAD"), apply=True, **kwargs,
    )


def _denied_route(daemon, task, repo, *, error_match: str = "", **kwargs):
    # Closed entrypoint rejection and explicit pending outcomes are both safe.
    # Unexpected exceptions (including missing integration) fail these tests.
    try:
        outcome = _route(daemon, task, repo, **kwargs)
    except (RuntimeError, ValueError) as error:
        if error_match:
            assert re.search(error_match, str(error)), str(error)
        return
    assert outcome.get("reviewed_effect_binding") is None
    result = outcome.get("route_result")
    assert result is None or result.write_performed is False
    assert daemon.model_assisted_authoritative_completion_allowed(result) is False


def _signed_fixture(tmp_path: Path):
    daemon, task, repo, key = _live_fixture(tmp_path)
    calls = _providers(daemon, task, repo)
    baseline = _git(repo, "rev-parse", "HEAD")
    _route(daemon, task, repo)
    _git(repo, "add", "--", TARGET)
    _git(repo, "commit", "-m", "independently reviewed bytes")
    commit = _git(repo, "rev-parse", "HEAD")
    evidence = daemon._finalize_production_model_assisted_route(
        task, attempt=1, workspace_path=repo, implementation_commit=commit,
    )
    return daemon, task, repo, key, calls, baseline, evidence


def _publication_files(daemon):
    directory = daemon.state_path.parent
    queue = directory / "task_queue.json"
    store = directory / "completion-publications"
    return {
        "queue": queue.read_bytes() if queue.exists() else None,
        "publications": {
            str(path.relative_to(store)): path.read_bytes()
            for path in store.rglob("*") if path.is_file()
        } if store.exists() else {},
    }


def _restart(daemon, key, repo):
    return TodoImplementationDaemon(
        todo_path=daemon.todo_path, state_path=daemon.state_path,
        strategy_path=daemon.strategy_path, events_path=daemon.events_path,
        repo_root=repo, task_header_prefix=daemon.task_header_prefix,
        implement=True, use_ephemeral_worktree=True,
        worktree_root=daemon.worktree_root,
        production_provider_policy=daemon.production_provider_policy,
        production_provider_review_authority_key_path=key,
    )


def test_joined_route_binds_actual_written_and_committed_bytes(tmp_path: Path) -> None:
    daemon, task, repo, key = _fixture(tmp_path)
    calls = _providers(daemon, task, repo)
    outcome = _route(daemon, task, repo)
    route = outcome["route_result"]
    captured = outcome["reviewed_effect_binding"]
    assert route.status is RouteStatus.SUCCEEDED
    assert calls == [ProviderRole.GROK_IMPLEMENT.value, ProviderRole.CODEX_REVIEW.value]
    assert (repo / TARGET).read_text(encoding="utf-8") == REVIEWED
    assert captured.changed_paths == (TARGET,)
    assert captured.provider_policy_id == daemon.production_provider_policy.policy_id
    assert captured.writer_lease_id
    assert daemon.model_assisted_authoritative_completion_allowed(route) is False
    _git(repo, "add", "--", TARGET)
    _git(repo, "commit", "-m", "exact reviewed change")
    commit = _git(repo, "rev-parse", "HEAD")
    tree = "git-tree:" + _git(repo, "rev-parse", "HEAD^{tree}")
    identity = daemon._identity_for_task(task)
    finalized = finalize_production_reviewed_effect(
        captured, repo_root=repo, task=task, task_identity=identity,
        implementation_commit=commit,
    )
    binding = bind_applied_patch_to_review_chain(route, implementation_commit=commit)
    authority = ProductionProviderReviewAuthority.load_or_create(key)
    attestation = authority.issue(
        provider_receipt=route.provider_receipt, review_chain_binding=binding,
        provider_policy_id=daemon.production_provider_policy.policy_id,
        implementation_commit=commit, implementation_tree_id=tree,
        reviewed_effect_binding=finalized, repo_root=repo, task=task,
        task_identity=identity,
    )
    checked = verify_production_provider_review_attestation(
        attestation, trusted_public_keys=daemon._trusted_production_provider_review_keys(),
        provider_receipt=route.provider_receipt, review_chain_binding=binding,
        reviewed_effect_binding=finalized, repo_root=repo, task=task,
        task_identity=identity, expected_task_id=task.task_id,
        expected_snapshot_id=captured.snapshot_id,
        expected_provider_policy_id=daemon.production_provider_policy.policy_id,
        expected_implementation_commit=commit, expected_implementation_tree_id=tree,
    )
    assert checked.admitted is True
    assert attestation.to_dict()["completion_authoritative"] is False


@pytest.mark.parametrize("fault", [
    "source_drift", "task_drift", "policy_drift", "undeclared_path",
    "review_rewrite", "review_decline", "review_quota", "effective_provider",
])
def test_provider_callbacks_cannot_promote_drift_or_rewrite_to_effect(
    tmp_path: Path, fault: str,
) -> None:
    daemon, task, repo, _key = _fixture(tmp_path)
    calls = _providers(daemon, task, repo, fault=fault)
    _denied_route(daemon, task, repo)
    assert calls and calls[0] == ProviderRole.GROK_IMPLEMENT.value
    expected = "VALUE = 'external change'\n" if fault == "source_drift" else BASELINE
    assert (repo / TARGET).read_text(encoding="utf-8") == expected
    assert not (repo / "src/unowned.py").exists()
    assert _git(repo, "log", "--format=%s", "-1") == "baseline"


def test_foreign_policy_pair_is_rejected_before_provider_dispatch(tmp_path: Path) -> None:
    daemon, task, repo, _key = _fixture(tmp_path)
    calls = _providers(
        daemon, task, repo, policy=ProductionCLIProviderPolicy(context_budget_tokens=3072),
    )
    _denied_route(daemon, task, repo, error_match="operator-bound roles and policy")
    assert calls == []
    assert (repo / TARGET).read_text(encoding="utf-8") == BASELINE


def test_role_bound_implementation_cannot_be_installed_as_its_own_reviewer(
    tmp_path: Path,
) -> None:
    daemon, task, repo, _key = _fixture(tmp_path)
    calls = _providers(daemon, task, repo)
    daemon._production_codex_provider = daemon._production_grok_provider
    _denied_route(daemon, task, repo, error_match="operator-bound roles|independent")
    assert calls == []
    assert (repo / TARGET).read_text(encoding="utf-8") == BASELINE


def test_native_second_write_failure_restores_the_first_and_removes_temporaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, task, repo, _key = _fixture(tmp_path)
    second = "src/other.py"
    (repo / second).write_text(BASELINE, encoding="utf-8")
    _git(repo, "add", "--", second)
    _git(repo, "commit", "--amend", "--no-edit")
    task.outputs.append(second)
    calls = _providers(daemon, task, repo, files=[
        {"path": TARGET, "content": REVIEWED},
        {"path": second, "content": REVIEWED},
    ])
    real_write = os.write
    failed = False
    replacement_writes = 0

    def fail_second_replacement(descriptor, value):
        nonlocal failed, replacement_writes
        destination = os.readlink(f"/proc/self/fd/{descriptor}")
        if "/.provider-write-" in destination and not failed:
            replacement_writes += 1
            if replacement_writes == 2:
                failed = True
                raise OSError("injected second replacement write failure")
        return real_write(descriptor, value)

    monkeypatch.setattr(os, "write", fail_second_replacement)
    _denied_route(daemon, task, repo)
    assert failed is True
    assert calls == [ProviderRole.GROK_IMPLEMENT.value, ProviderRole.CODEX_REVIEW.value]
    assert (repo / TARGET).read_text(encoding="utf-8") == BASELINE
    assert (repo / second).read_text(encoding="utf-8") == BASELINE
    assert not list((repo / "src").glob(".provider-write-*"))
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""


def test_native_git_apply_failure_after_postimage_restores_owned_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import production_daemon_route

    daemon, task, repo, _key = _fixture(tmp_path)
    patch = (
        f"diff --git a/{TARGET} b/{TARGET}\n"
        f"--- a/{TARGET}\n+++ b/{TARGET}\n@@ -1 +1 @@\n"
        "-VALUE = 'baseline'\n+VALUE = 'reviewed'\n"
    )
    calls = _providers(daemon, task, repo, patch=patch)
    real_git = production_daemon_route._run_git_bounded
    applied = False

    def fail_after_actual_apply(root, *arguments, **kwargs):
        nonlocal applied
        outcome = real_git(root, *arguments, **kwargs)
        if arguments and arguments[0] == "apply" and "--check" not in arguments:
            assert outcome[0] == 0
            assert (repo / TARGET).read_text(encoding="utf-8") == REVIEWED
            applied = True
            return 1, outcome[1], outcome[2]
        return outcome

    monkeypatch.setattr(production_daemon_route, "_run_git_bounded", fail_after_actual_apply)
    _denied_route(daemon, task, repo)
    assert applied is True
    assert calls == [ProviderRole.GROK_IMPLEMENT.value, ProviderRole.CODEX_REVIEW.value]
    assert (repo / TARGET).read_text(encoding="utf-8") == BASELINE
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""


def test_lost_durable_lease_stops_remaining_and_compensating_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import production_daemon_route

    daemon, task, repo, _key = _fixture(tmp_path)
    second = "src/other.py"
    (repo / second).write_text(BASELINE, encoding="utf-8")
    _git(repo, "add", "--", second)
    _git(repo, "commit", "--amend", "--no-edit")
    task.outputs.append(second)
    calls = _providers(daemon, task, repo, files=[
        {"path": TARGET, "content": REVIEWED},
        {"path": second, "content": REVIEWED},
    ])
    real_replace = production_daemon_route._atomic_replace
    replacements = []
    foreign_lease = None

    def replace_then_rotate_lease(root, name, content, mode):
        nonlocal foreign_lease
        replacements.append(name)
        real_replace(root, name, content, mode)
        if foreign_lease is None:
            lease = daemon._current_checkout_mutation_lease()
            foreign = json.loads(lease.lock_path.read_text())
            foreign["lease_id"] = "foreign-owner-generation"
            temporary = lease.lock_path.with_name(lease.lock_path.name + ".replacement")
            foreign_lease = json.dumps(foreign, sort_keys=True).encode()
            temporary.write_bytes(foreign_lease)
            os.chmod(temporary, 0o600)
            os.replace(temporary, lease.lock_path)

    monkeypatch.setattr(production_daemon_route, "_atomic_replace", replace_then_rotate_lease)
    _denied_route(daemon, task, repo)
    assert replacements == [TARGET]
    assert foreign_lease is not None
    assert (repo / TARGET).read_text(encoding="utf-8") == REVIEWED
    assert (repo / second).read_text(encoding="utf-8") == BASELINE
    assert daemon._checkout_mutation_context.retain_until_protected_clean is True
    assert daemon._production_reviewed_evidence == {}
    assert calls == [ProviderRole.GROK_IMPLEMENT.value, ProviderRole.CODEX_REVIEW.value]


def test_production_entrypoint_does_not_accept_caller_writer_authority(tmp_path: Path) -> None:
    daemon, task, repo, _key = _fixture(tmp_path)
    calls = _providers(daemon, task, repo)
    writes = []
    with pytest.raises(TypeError, match="writer"):
        _route(daemon, task, repo, writer=lambda *_args: writes.append("caller write"))
    assert calls == []
    assert writes == []
    assert (repo / TARGET).read_text(encoding="utf-8") == BASELINE


def test_foreign_git_workspace_cannot_borrow_the_daemon_writer_lease(tmp_path: Path) -> None:
    daemon, task, repo, _key = _fixture(tmp_path)
    calls = _providers(daemon, task, repo)
    foreign = tmp_path / "foreign-repository"
    foreign.mkdir()
    _git(foreign, "init")
    _git(foreign, "config", "user.name", "Foreign Workspace")
    _git(foreign, "config", "user.email", "foreign@example.invalid")
    (foreign / "src").mkdir()
    (foreign / TARGET).write_text(BASELINE, encoding="utf-8")
    _git(foreign, "add", ".")
    _git(foreign, "commit", "-m", "foreign baseline")
    _denied_route(daemon, task, foreign, error_match="repository|workspace|worktree")
    assert calls == []
    assert (foreign / TARGET).read_text(encoding="utf-8") == BASELINE
    assert (repo / TARGET).read_text(encoding="utf-8") == BASELINE


def test_invalid_operator_pinned_launch_receipt_blocks_provider_dispatch(tmp_path: Path) -> None:
    daemon, task, repo, _key = _fixture(tmp_path)
    calls = _providers(daemon, task, repo)
    receipt = tmp_path / "launch-receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    os.chmod(receipt, 0o600)
    daemon.production_provider_launch_authority_receipt_path = receipt
    daemon.production_provider_launch_authority_receipt_content_id = "sha256:" + "0" * 64
    _denied_route(daemon, task, repo, error_match="launch receipt CID differs")
    assert calls == []
    assert (repo / TARGET).read_text(encoding="utf-8") == BASELINE


def test_live_implementation_validates_commits_and_signs_one_reviewed_effect(
    tmp_path: Path,
) -> None:
    daemon, task, repo, key = _live_fixture(tmp_path)
    calls = _providers(daemon, task, repo)
    baseline = _git(repo, "rev-parse", "HEAD")
    result = daemon._run_implementation(task, PortalTaskState())
    assert result["returncode"] == 0, result
    assert calls == [ProviderRole.GROK_IMPLEMENT.value, ProviderRole.CODEX_REVIEW.value]
    retained = daemon._production_reviewed_evidence[daemon._canonical_ref(task)]
    commit = retained["implementation_commit"]
    assert commit != baseline
    assert _git(repo, "show", commit + ":" + TARGET) == REVIEWED.strip()
    assert retained["provider_review_attestation"]["completion_authoritative"] is False
    effect = retained["production_reviewed_effect_binding"]
    assert effect["implementation_commit"] == commit
    assert retained["provider_review_attestation"]["reviewed_effect_binding_cid"] == (
        effect["binding_id"]
    )
    events = [json.loads(line) for line in daemon.events_path.read_text().splitlines()]
    assert any(event.get("type") == "production_provider_review_attested" for event in events)
    validation = result.get("validation_result") or result.get("validation")
    assert validation and validation["passed"] is True
    assert validation["attempted"] is True
    assert validation.get("results")
    assert validation["authoritative"] is False
    merge = result["merge_result"]
    assert merge["merged"] is True
    request = daemon.merge_queue.get(merge["request_id"])
    assert request is not None
    queued_review = request.metadata["production_reviewed_evidence"]
    for name in (
        "provider_execution_receipt", "admitted_review_chain_binding",
        "production_reviewed_effect_binding", "provider_review_attestation",
    ):
        assert queued_review[name] == retained[name]
    restarted = _restart(daemon, key, repo)
    current_task, = parse_task_file(restarted.todo_path, restarted.task_header_prefix)
    target = _git(repo, "rev-parse", "HEAD")
    tree = "git-tree:" + _git(repo, "rev-parse", "HEAD^{tree}")
    gate = restarted._verified_provider_review_gate_evidence(
        task=current_task, implementation_commit=commit,
        merge_commit=target, repository_tree_id=tree, evidence=queued_review,
    )
    assert gate and gate["satisfied"] is True
    tampered = json.loads(json.dumps(queued_review))
    tampered["provider_review_attestation"]["signature"] = "A" * 86 + "=="
    assert restarted._verified_provider_review_gate_evidence(
        task=current_task, implementation_commit=commit,
        merge_commit=target, repository_tree_id=tree, evidence=tampered,
    ) is None


def test_live_failed_validation_does_not_commit_or_issue_review_attestation(
    tmp_path: Path,
) -> None:
    daemon, task, repo, _key = _live_fixture(
        tmp_path, validation="python -c 'raise SystemExit(23)'",
    )
    calls = _providers(daemon, task, repo)
    baseline = _git(repo, "rev-parse", "HEAD")
    result = daemon._run_implementation(task, PortalTaskState())
    assert result["returncode"] != 0, result
    assert calls == [ProviderRole.GROK_IMPLEMENT.value, ProviderRole.CODEX_REVIEW.value]
    assert _git(repo, "rev-parse", "HEAD") == baseline
    assert daemon._canonical_ref(task) not in daemon._production_reviewed_evidence
    events = [json.loads(line) for line in daemon.events_path.read_text().splitlines()]
    assert not any(event.get("type") == "production_provider_review_attested" for event in events)


def test_task_metadata_cannot_install_provider_policy_or_signing_key(tmp_path: Path) -> None:
    attacker_key = tmp_path / "attacker.ed25519"
    metadata = {
        "Production provider policy": "attacker-self-review",
        "Production provider context budget tokens": "1",
        "Production provider review authority key path": str(attacker_key),
        "production_provider_policy": "attacker-self-review",
        "production_provider_review_authority_key_path": str(attacker_key),
    }
    daemon, task, repo, key = _fixture(tmp_path, metadata=metadata)
    original = dict(task.metadata)
    operator_policy = daemon.production_provider_policy.policy_id
    _providers(daemon, task, repo)
    outcome = _route(daemon, task, repo)
    assert outcome["reviewed_effect_binding"].provider_policy_id == operator_policy
    assert task.metadata == original
    assert key.exists()
    assert not attacker_key.exists()
    assert daemon.production_provider_policy.policy_id == operator_policy


def test_post_write_mutation_prevents_commit_bound_review(tmp_path: Path) -> None:
    daemon, task, repo, _key = _fixture(tmp_path)
    _providers(daemon, task, repo)
    outcome = _route(daemon, task, repo)
    captured = outcome["reviewed_effect_binding"]
    (repo / TARGET).write_text("VALUE = 'after review'\n", encoding="utf-8")
    observed = verify_production_reviewed_workspace(
        captured, repo_root=repo, task=task, task_identity=daemon._identity_for_task(task),
    )
    assert observed.admitted is False
    assert "reviewed_effect_workspace_bytes_or_modes_changed" in observed.reason_codes
    _git(repo, "add", "--", TARGET)
    _git(repo, "commit", "-m", "unreviewed changed bytes")
    with pytest.raises(ValueError, match="reviewed effect changed"):
        finalize_production_reviewed_effect(
            captured, repo_root=repo, task=task,
            task_identity=daemon._identity_for_task(task),
            implementation_commit=_git(repo, "rev-parse", "HEAD"),
        )


@pytest.mark.parametrize("change", ["unmerged_same_bytes", "reviewed_bytes", "reviewed_mode"])
def test_completion_target_requires_reviewed_ancestry_and_exact_postimage(
    tmp_path: Path, change: str,
) -> None:
    daemon, task, repo, _key, calls, baseline, evidence = _signed_fixture(tmp_path)
    if change == "unmerged_same_bytes":
        _git(repo, "checkout", "-b", "unrelated-same-bytes", baseline)
        (repo / TARGET).write_text(REVIEWED, encoding="utf-8")
    elif change == "reviewed_bytes":
        (repo / TARGET).write_text("VALUE = 'unreviewed descendant'\n", encoding="utf-8")
    else:
        os.chmod(repo / TARGET, 0o755)
    _git(repo, "add", "--", TARGET)
    _git(repo, "commit", "-m", "completion target outside reviewed binding")
    target = _git(repo, "rev-parse", "HEAD")
    before = _publication_files(daemon)
    with pytest.raises(ValueError, match="has not landed|effect changed"):
        daemon._completion_publication_intent(
            task, merged_tree_id="git-commit:" + target, evidence=evidence,
        )
    assert _publication_files(daemon) == before
    assert calls == [ProviderRole.GROK_IMPLEMENT.value, ProviderRole.CODEX_REVIEW.value]


@pytest.mark.parametrize("historical_target", [False, True])
def test_unrelated_descendant_retains_only_the_verified_provider_review_gate(
    tmp_path: Path, historical_target: bool,
) -> None:
    daemon, task, repo, _key, calls, _baseline, evidence = _signed_fixture(tmp_path)
    (repo / "unrelated.txt").write_text("unrelated descendant\n", encoding="utf-8")
    _git(repo, "add", "--", "unrelated.txt")
    _git(repo, "commit", "-m", "unrelated descendant")
    target = _git(repo, "rev-parse", "HEAD")
    before = _publication_files(daemon)
    intent = daemon._completion_publication_intent(
        task, merged_tree_id="git-commit:" + (
            evidence["implementation_commit"] if historical_target else target
        ), evidence=evidence,
    )
    retained = intent["completion_payload"]["post_merge_evidence"]
    assert retained["provider_review"]["satisfied"] is True
    expected_binding = evidence["implementation_commit"] if historical_target else target
    assert retained["provider_review"]["target_commit"] == expected_binding
    assert retained["provider_review_attestation"]["completion_authoritative"] is False
    assert not retained.get("proof_authoritative")
    assert not retained.get("completion_authoritative")
    assert _publication_files(daemon) == before
    assert calls == [ProviderRole.GROK_IMPLEMENT.value, ProviderRole.CODEX_REVIEW.value]


def test_old_valid_target_cannot_hide_a_changed_current_reviewed_blob(tmp_path: Path) -> None:
    daemon, task, repo, _key, calls, _baseline, evidence = _signed_fixture(tmp_path)
    (repo / TARGET).write_text("VALUE = 'changed current head'\n", encoding="utf-8")
    _git(repo, "add", "--", TARGET)
    _git(repo, "commit", "-m", "change current reviewed blob")
    before = _publication_files(daemon)
    with pytest.raises(ValueError, match="effect changed"):
        daemon._completion_publication_intent(
            task, merged_tree_id="git-commit:" + evidence["implementation_commit"],
            evidence=evidence,
        )
    assert _publication_files(daemon) == before
    assert (repo / TARGET).read_text(encoding="utf-8") == "VALUE = 'changed current head'\n"
    assert calls == [ProviderRole.GROK_IMPLEMENT.value, ProviderRole.CODEX_REVIEW.value]


@pytest.mark.parametrize("fault", ["task_drift", "forged_signature", "missing_review"])
def test_restart_revalidates_pending_intent_before_any_completion_publication(
    tmp_path: Path, fault: str,
) -> None:
    daemon, task, repo, key, calls, _baseline, evidence = _signed_fixture(tmp_path)
    intent = daemon._completion_publication_intent(
        task, merged_tree_id="git-commit:" + evidence["implementation_commit"],
        evidence=evidence,
    )
    if fault == "task_drift":
        daemon.todo_path.write_text(
            daemon.todo_path.read_text().replace(
                "the exact independently reviewed bytes are committed",
                "a different acceptance contract",
            ), encoding="utf-8",
        )
    else:
        evidence = intent["completion_payload"]["post_merge_evidence"]
        if fault == "forged_signature":
            evidence["provider_review_attestation"]["signature"] = "A" * 86 + "=="
        else:
            evidence.pop("provider_review_attestation")
        # Even a coherent outer carrier cannot mint a review trust root.
        intent.pop("intent_id")
        intent["intent_id"] = content_identity(intent)
    restarted = _restart(daemon, key, repo)
    restart_calls = _providers(restarted, task, repo)
    before = _publication_files(restarted)
    with pytest.raises(ValueError, match="exact current task|verified independent review"):
        restarted._publish_completion_intent(intent)
    assert _publication_files(restarted) == before
    assert restart_calls == []
    assert calls == [ProviderRole.GROK_IMPLEMENT.value, ProviderRole.CODEX_REVIEW.value]
