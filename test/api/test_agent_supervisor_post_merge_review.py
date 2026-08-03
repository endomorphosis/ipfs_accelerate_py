from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.runtime.event_log import (
    append_jsonl_event,
    event_log_manifest,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import post_merge_review as review
from ipfs_accelerate_py.agent_supervisor.todo_daemon.authoritative_completion import (
    POST_MERGE_VALIDATION_EVIDENCE_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.llm import (
    LLM_CHILD_ENVELOPE_VERSION,
    LlmChildResultEnvelope,
)
from ipfs_accelerate_py.agent_supervisor.validation.scope_adjudication import (
    ScopeAdjudicationReceipt,
    ScopeExpansionReason,
    ScopeExpansionVerdict,
    ScopePathDecision,
)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True)
    _git(path, "init")
    _git(path, "config", "user.name", "Post Merge Review Test")
    _git(path, "config", "user.email", "post-merge-review@example.invalid")


def _task() -> PortalTask:
    outputs = [
        "external/child/docs/contract.md",
        "external/child/tests/vocabulary.json",
    ]
    validation = ["python3 -m pytest tests/test_contract.py -q"]
    acceptance = "Both exact nested artifacts are complete and coherent."
    return PortalTask(
        task_id="REV-001",
        title="Review a nested implementation",
        status="ready",
        completion="manual",
        priority="P0",
        track="review",
        outputs=outputs,
        validation=validation,
        acceptance=acceptance,
        metadata={
            "status": "ready",
            "completion": "manual",
            "priority": "P0",
            "track": "review",
            "outputs": ", ".join(outputs),
            "validation": validation[0],
            "acceptance": acceptance,
            "provider role": "grok-implement, codex-review",
            "canonical task key": "uiir-test:REV-001",
            "canonical task cid": "sha256:" + "a" * 64,
            "board namespace": "uiir-test",
        },
        canonical_task_key="uiir-test:REV-001",
        canonical_task_cid="sha256:" + "a" * 64,
        board_namespace="uiir-test",
    )


def _tree(repo: Path, commit: str) -> str:
    return f"git-tree:{_git(repo, 'rev-parse', f'{commit}^{{tree}}')}"


def _validation(
    task: PortalTask,
    merge_commit: str,
    repository_tree_id: str,
) -> dict[str, Any]:
    tree = repository_tree_id.removeprefix("git-tree:")
    plan_material = {
        "task_id": task.task_id,
        "target_commit": merge_commit,
        "repository_tree_id": repository_tree_id,
        "validation_scope": "post_merge",
        "declared_commands": list(task.validation),
    }
    material: dict[str, Any] = {
        "schema": POST_MERGE_VALIDATION_EVIDENCE_SCHEMA,
        "task_id": task.task_id,
        "validation_scope": "post_merge",
        "target_commit": merge_commit,
        "target_tree": tree,
        "repository_tree_id": repository_tree_id,
        "attempted": True,
        "passed": True,
        "returncode": 0,
        "stale": False,
        "declared_commands": list(task.validation),
        "validation_plan_id": content_identity(plan_material),
        "reason": "post_merge_validation_passed",
        "results": [],
        "selection": {},
        "validated_commit": merge_commit,
        "validated_tree": tree,
        "validation_dirty_paths": [],
        "workspace_clean": True,
        "workspace_status_porcelain": "",
        "validation_status_returncode": 0,
        "validation_status_stderr": "",
        "freshness_authoritative": True,
    }
    return {
        **material,
        "validation_receipt_id": content_identity(material),
    }


def _response(request: dict[str, Any], decision: str = "approve") -> str:
    findings = (
        []
        if decision == "approve"
        else [
            {
                "code": "review-change",
                "severity": "high",
                "summary": "The exact patch requires a correction.",
            }
        ]
    )
    return json.dumps(
        {
            "schema": review.POST_MERGE_INDEPENDENT_REVIEW_RESPONSE_SCHEMA,
            "decision": decision,
            "task_id": request["task_id"],
            "implementation_commit": request["implementation_commit"],
            "merge_commit": request["merge_commit"],
            "repository_tree_id": request["repository_tree_id"],
            "diff_binding_id": request["diff_binding_id"],
            "review_request_id": request["request_id"],
            "reviewer_provider": review.CODEX_REVIEWER_PROVIDER,
            "implementer_provider": request["implementer_provider"],
            "findings": findings,
            "repository_write_authorized": False,
            "proof_authoritative": False,
            "completion_authoritative": False,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _transport(
    request: dict[str, Any],
    response_text: str,
    *,
    attempt: int | None = None,
    contract_version: int = LLM_CHILD_ENVELOPE_VERSION,
    execution_result_id: str | None = None,
    digest_text: str | None = None,
) -> dict[str, Any]:
    bound_text = response_text if digest_text is None else digest_text
    encoded = bound_text.encode("utf-8")
    selected_attempt = int(
        request["attempt"] if attempt is None else attempt
    )
    execution_material = {
        "request_id": request["request_id"],
        "attempt": selected_attempt,
        "idempotency_key": request["request_id"],
        "effective_provider": review.CODEX_REVIEWER_PROVIDER,
        "text_chars": len(bound_text),
        "text_bytes": len(encoded),
        "text_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    selected_execution_id = execution_result_id
    if selected_execution_id is None:
        selected_execution_id = "sha256:" + hashlib.sha256(
            json.dumps(
                execution_material,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    payload = LlmChildResultEnvelope(
        contract_version=contract_version,
        request_id=request["request_id"],
        attempt=selected_attempt,
        idempotency_key=request["request_id"],
        status="ok",
        execution_result_id=selected_execution_id,
        effective_provider=review.CODEX_REVIEWER_PROVIDER,
        text_chars=len(bound_text),
        text_bytes=len(encoded),
        text_sha256=hashlib.sha256(encoded).hexdigest(),
        exit_code=0,
    ).to_dict()
    return payload


def _reviewer(decision: str = "approve"):
    def invoke(_prompt: str, request: dict[str, Any]) -> review.ReviewerInvocation:
        response_text = _response(request, decision)
        return review.ReviewerInvocation(
            provider_id=review.CODEX_REVIEWER_PROVIDER,
            response_text=response_text,
            transport_receipt=_transport(request, response_text),
        )

    return invoke


def _request_from_prompt(prompt: str) -> dict[str, Any]:
    prefix = "Bound review request:\n"
    suffix = "\n\nExact changed-content Git bindings:"
    encoded_request = prompt.split(prefix, 1)[1].split(suffix, 1)[0]
    request = json.loads(encoded_request)
    assert isinstance(request, dict)
    return request


def _fake_codex_child(decision: str = "approve"):
    def invoke(prompt: str, invocation: Any):
        request = _request_from_prompt(prompt)
        assert invocation.provider == review.CODEX_REVIEWER_PROVIDER
        assert invocation.codex_read_only is True
        assert invocation.request_id == request["request_id"]
        assert invocation.attempt == request["attempt"]
        response_text = _response(request, decision)
        child_receipt = LlmChildResultEnvelope.from_dict(
            _transport(request, response_text)
        )
        return response_text, child_receipt

    return invoke


def _append_review_event(
    events_path: Path,
    outcome: review.PostMergeReviewOutcome,
) -> dict[str, Any]:
    payload = dict(outcome.event)
    event_type = str(payload.pop("type"))
    return append_jsonl_event(events_path, event_type, payload)


def test_strict_ledger_rejects_sibling_path_manifest_replay(
    tmp_path: Path,
) -> None:
    source = tmp_path / "state" / "origin" / "events.jsonl"
    copied = tmp_path / "state" / "consumer" / "events.jsonl"
    append_jsonl_event(
        source,
        "implementation_finished",
        {
            "task_id": "REV-001",
            "attempt": 1,
            "implementation_commit": "a" * 40,
            "returncode": 0,
        },
    )
    copied.parent.mkdir(parents=True)
    copied.write_bytes(source.read_bytes())
    source_manifest = source.with_name(
        f"{source.name}.manifest.json"
    )
    copied_manifest = copied.with_name(
        f"{copied.name}.manifest.json"
    )
    copied_manifest.write_bytes(source_manifest.read_bytes())
    copied_stream_id = event_log_manifest(copied)["stream_id"]
    assert copied_stream_id == event_log_manifest(source)["stream_id"]

    with pytest.raises(review.PostMergeReviewError) as raised:
        review._strict_event_ledger(copied)

    assert (
        raised.value.reason_code
        == "event_ledger_path_binding_invalid"
    )


@pytest.fixture()
def nested_case(tmp_path: Path) -> SimpleNamespace:
    child = tmp_path / "child-source"
    _init_repo(child)
    (child / ".gitignore").write_text(".cache/\n", encoding="utf-8")
    child_baseline = _commit(child, "child baseline")
    (child / "docs").mkdir()
    (child / "docs/contract.md").write_text("# Contract\n", encoding="utf-8")
    child_seed = _commit(child, "seed first output")
    (child / "tests").mkdir()
    (child / "tests/vocabulary.json").write_text(
        '{"version":1}\n',
        encoding="utf-8",
    )
    child_final = _commit(child, "finish second output")
    _git(child, "checkout", "--detach", child_baseline)

    root = tmp_path / "root"
    _init_repo(root)
    todo_path = root / "tasks.todo.md"
    todo_path.write_text(
        "# Review tasks\n\n"
        "## REV-001 Review a nested implementation\n\n"
        "- Status: ready\n"
        "- Completion: manual\n"
        "- Priority: P0\n"
        "- Track: review\n"
        "- Outputs: external/child/docs/contract.md, "
        "external/child/tests/vocabulary.json\n"
        "- Validation: python3 -m pytest tests/test_contract.py -q\n"
        "- Acceptance: Both exact nested artifacts are complete and coherent.\n"
        "- Provider role: grok-implement, codex-review\n"
        "- Canonical task key: uiir-test:REV-001\n"
        f"- Canonical task CID: sha256:{'a' * 64}\n"
        "- Board namespace: uiir-test\n",
        encoding="utf-8",
    )
    (root / ".gitignore").write_text("state/\n", encoding="utf-8")
    _git(
        root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child),
        "external/child",
    )
    _git(root / "external/child", "checkout", "--detach", child_baseline)
    baseline = _commit(root, "root baseline")
    _git(root / "external/child", "checkout", "--detach", child_seed)
    seed = _commit(root, "seed prior attempt")
    _git(root / "external/child", "checkout", "--detach", child_final)
    implementation = _commit(root, "implementation candidate")
    assert _git(root, "rev-parse", f"{implementation}^") == seed

    log_path = Path("state/implementation_logs/rev-001-attempt-3.log")
    (root / log_path).parent.mkdir(parents=True)
    (root / log_path).write_text("grok implementation log\n", encoding="utf-8")
    events_path = root / "state/events.jsonl"
    append_jsonl_event(
        events_path,
        "implementation_started",
        {
            "task_id": "REV-001",
            "attempt": 3,
            "execution_mode": "model-assisted",
            "branch": "implementation/rev-001-attempt-3",
            "log_path": str(log_path),
            "command": [
                "/usr/bin/python3",
                "/opt/ipfs_accelerate_py/agent_supervisor/grok_cli_runner.py",
                "--grok-bin",
                "/usr/bin/grok",
                "--model",
                "grok-4.5",
            ],
        },
    )
    append_jsonl_event(
        events_path,
        "implementation_finished",
        {
            "task_id": "REV-001",
            "attempt": 3,
            "branch": "implementation/rev-001-attempt-3",
            "log_path": str(log_path),
            "implementation_commit": implementation,
            "returncode": 0,
        },
    )
    provenance = review.verified_implementer_provenance_from_ledger(
        events_path,
        repo_root=root,
        expected_task_id="REV-001",
        expected_implementation_attempt=3,
        expected_implementation_commit=implementation,
    )
    task = _task()
    tree = _tree(root, implementation)
    return SimpleNamespace(
        root=root,
        todo_path=todo_path,
        baseline=baseline,
        implementation=implementation,
        merge_commit=implementation,
        repository_tree_id=tree,
        task=task,
        validation=_validation(task, implementation, tree),
        events_path=events_path,
        provenance=provenance,
        receipt_dir=tmp_path / "receipts",
    )


def _perform(
    case: SimpleNamespace,
    *,
    reviewer=None,
    expected_changed_paths: list[str] | None = None,
    scope_authorized_paths: list[str] | None = None,
    scope_adjudication_id: str = "",
) -> review.PostMergeReviewOutcome:
    return review.perform_post_merge_independent_review(
        repo_root=case.root,
        receipt_dir=case.receipt_dir,
        implementation_events_path=case.events_path,
        task=case.task,
        attempt=4,
        implementation_attempt=int(
            getattr(case, "implementation_attempt", 3)
        ),
        baseline_commit=case.baseline,
        implementation_commit=case.implementation,
        merge_commit=case.merge_commit,
        repository_tree_id=case.repository_tree_id,
        validation_result=case.validation,
        expected_changed_paths=(
            case.task.outputs
            if expected_changed_paths is None
            else expected_changed_paths
        ),
        scope_authorized_paths=scope_authorized_paths or (),
        scope_adjudication_id=scope_adjudication_id,
        implementer_provider="grok_cli",
        implementer_provenance=case.provenance,
        reviewer=reviewer,
    )


def test_recursive_submodule_binding_uses_explicit_pre_seed_baseline(
    nested_case: SimpleNamespace,
) -> None:
    binding = review._collect_repository_binding(
        repo_root=nested_case.root,
        task=nested_case.task,
        baseline_commit=nested_case.baseline,
        implementation_commit=nested_case.implementation,
        merge_commit=nested_case.merge_commit,
        repository_tree_id=nested_case.repository_tree_id,
        expected_changed_paths=nested_case.task.outputs,
    )
    assert binding["changed_paths"] == nested_case.task.outputs
    assert len(binding["gitlink_bindings"]) == 1
    gitlink = binding["gitlink_bindings"][0]
    assert gitlink["path"] == "external/child"
    assert gitlink["implementation"] == gitlink["merged"]
    assert binding["patch_bytes"] > 0
    assert binding["task_binding_id"] == review.post_merge_task_binding_id(
        nested_case.task
    )
    assert binding["task_binding_id"] != review.post_merge_task_binding_id(
        replace(nested_case.task, acceptance="Drifted acceptance criteria.")
    )


def test_metadata_declared_proposal_scope_is_consistent_in_post_merge_review(
    nested_case: SimpleNamespace,
) -> None:
    predicted_path = "tests/predicted_contract.py"
    (nested_case.root / "tests").mkdir(exist_ok=True)
    (nested_case.root / predicted_path).write_text(
        "def test_predicted_contract():\n    assert True\n",
        encoding="utf-8",
    )
    implementation = _commit(
        nested_case.root,
        "add metadata-predicted validation artifact",
    )
    metadata = dict(nested_case.task.metadata)
    metadata["predicted files"] = predicted_path
    task = replace(nested_case.task, metadata=metadata)
    expected_paths = sorted([*task.outputs, predicted_path])

    assert review.task_proposal_scope_paths(task) == tuple(
        sorted([*task.outputs, predicted_path])
    )
    binding = review._collect_repository_binding(
        repo_root=nested_case.root,
        task=task,
        baseline_commit=nested_case.baseline,
        implementation_commit=implementation,
        merge_commit=implementation,
        repository_tree_id=_tree(
            nested_case.root,
            implementation,
        ),
        expected_changed_paths=expected_paths,
    )
    assert list(binding["changed_paths"]) == expected_paths
    assert binding["scope_authorized_paths"] == []


def test_scope_adjudication_paths_are_exactly_request_and_receipt_bound(
    nested_case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    companion = "tests/test_contract.py"
    (nested_case.root / "tests").mkdir(exist_ok=True)
    (nested_case.root / companion).write_text(
        "def test_companion_entry():\n    assert True\n",
        encoding="utf-8",
    )
    implementation = _commit(
        nested_case.root,
        "add scope-adjudicated companion test",
    )
    expected_paths = sorted([*nested_case.task.outputs, companion])
    proposal_id = "proposal:scope-bound"
    initial_policy_id = "policy:scope-initial"
    policy_id = "policy:scope-expanded"
    proposal_tree_id = nested_case.baseline
    scope_receipt = ScopeAdjudicationReceipt(
        task_id=nested_case.task.task_id,
        proposal_id=proposal_id,
        initial_policy_id=initial_policy_id,
        repository_id=review._scope_receipt_repository_id(
            nested_case.root
        ),
        repository_tree_id=proposal_tree_id,
        baseline_id=nested_case.baseline,
        original_scope_paths=tuple(nested_case.task.outputs),
        candidate_paths=tuple(expected_paths),
        initial_finding_codes=("path_outside_scope",),
        decisions=(
            ScopePathDecision(
                path=companion,
                verdict=ScopeExpansionVerdict.JUSTIFIED,
                reason_codes=(
                    ScopeExpansionReason.EXPLICIT_VALIDATION_TARGET,
                ),
            ),
        ),
    ).bind_authorized_policy(policy_id)
    scope_receipt_id = scope_receipt.receipt_id
    branch = "implementation/rev-001-attempt-4"
    log_path = Path("state/implementation_logs/rev-001-attempt-4.log")
    (nested_case.root / log_path).write_text(
        "grok companion implementation log\n",
        encoding="utf-8",
    )
    append_jsonl_event(
        nested_case.events_path,
        "implementation_started",
        {
            "task_id": "REV-001",
            "attempt": 4,
            "execution_mode": "model-assisted",
            "branch": branch,
            "log_path": str(log_path),
            "command": [
                "/usr/bin/python3",
                "/opt/ipfs_accelerate_py/agent_supervisor/grok_cli_runner.py",
                "--grok-bin",
                "/usr/bin/grok",
                "--model",
                "grok-4.5",
            ],
        },
    )
    append_jsonl_event(
        nested_case.events_path,
        "implementation_finished",
        {
            "task_id": "REV-001",
            "attempt": 4,
            "branch": branch,
            "log_path": str(log_path),
            "implementation_commit": implementation,
            "baseline_ref": nested_case.baseline,
            "canonical_task_key": (
                nested_case.task.canonical_task_key
            ),
            "canonical_task_cid": (
                nested_case.task.canonical_task_cid
            ),
            "returncode": 0,
            "validation_result": {
                "passed": True,
                "proposal_gate": {
                    "accepted": True,
                    "proposal_id": proposal_id,
                    "policy_id": policy_id,
                    "receipt_id": "proposal-receipt:scope-bound",
                    "repository_tree_id": proposal_tree_id,
                    "changed_paths": expected_paths,
                },
                "scope_adjudication": scope_receipt.to_record(),
            },
        },
    )
    provenance = review.verified_implementer_provenance_from_ledger(
        nested_case.events_path,
        repo_root=nested_case.root,
        expected_task_id="REV-001",
        expected_implementation_attempt=4,
        expected_implementation_commit=implementation,
    )
    tree = _tree(nested_case.root, implementation)
    case = SimpleNamespace(
        **{
            **vars(nested_case),
            "implementation": implementation,
            "implementation_attempt": 4,
            "merge_commit": implementation,
            "repository_tree_id": tree,
            "validation": _validation(
                nested_case.task,
                implementation,
                tree,
            ),
            "provenance": provenance,
        }
    )
    adjudication_id = content_identity(
        {
            "task_binding_id": review.post_merge_task_binding_id(
                nested_case.task
            ),
            "proposal_id": proposal_id,
            "authorized_policy_id": policy_id,
            "receipt_id": scope_receipt_id,
            "repository_tree_id": proposal_tree_id,
            "changed_paths": expected_paths,
            "authorized_paths": [companion],
            "proof_authoritative": False,
            "completion_authoritative": False,
        }
    )

    rejected = _perform(
        case,
        reviewer=_reviewer("approve"),
        expected_changed_paths=expected_paths,
    )
    assert rejected.reason_code == "scope_adjudication_binding_mismatch"
    assert rejected.event == {}

    monkeypatch.setattr(
        review,
        "call_llm_router_with_receipt",
        _fake_codex_child("approve"),
    )
    forged_live = _perform(
        case,
        expected_changed_paths=expected_paths,
        scope_authorized_paths=[companion],
        scope_adjudication_id=content_identity({"forged": True}),
    )
    assert forged_live.admitted is False
    assert (
        forged_live.reason_code
        == "scope_adjudication_binding_mismatch"
    )
    assert forged_live.event == {}
    assert (
        review.mint_gate_from_live_outcome(
            forged_live,
            {},
            events_path=case.events_path,
        )
        == {}
    )

    outcome = _perform(
        case,
        reviewer=_reviewer("approve"),
        expected_changed_paths=expected_paths,
        scope_authorized_paths=[companion],
        scope_adjudication_id=adjudication_id,
    )
    assert outcome.receipt["scope_authorized_paths"] == [companion]
    assert outcome.receipt["scope_adjudication_id"] == adjudication_id
    request = outcome.receipt["review_request"]
    assert request["scope_authorized_paths"] == [companion]
    assert request["scope_adjudication_id"] == adjudication_id
    assert request["scope_authorization_id"] == (
        outcome.receipt["scope_authorization_id"]
    )

    verification = review.verify_post_merge_review_receipt(
        outcome.receipt,
        repo_root=case.root,
        implementation_events_path=case.events_path,
        task=case.task,
        validation_result=case.validation,
        attempt=4,
        implementation_attempt=4,
        baseline_commit=case.baseline,
        implementation_commit=case.implementation,
        merge_commit=case.merge_commit,
        repository_tree_id=case.repository_tree_id,
        expected_changed_paths=expected_paths,
        scope_authorized_paths=[companion],
        scope_adjudication_id=adjudication_id,
        implementer_provenance=case.provenance,
    )
    assert verification.valid is True

    missing_authority = review.verify_post_merge_review_receipt(
        outcome.receipt,
        repo_root=case.root,
        implementation_events_path=case.events_path,
        task=case.task,
        validation_result=case.validation,
        attempt=4,
        implementation_attempt=4,
        baseline_commit=case.baseline,
        implementation_commit=case.implementation,
        merge_commit=case.merge_commit,
        repository_tree_id=case.repository_tree_id,
        expected_changed_paths=expected_paths,
        implementer_provenance=case.provenance,
    )
    assert missing_authority.valid is False
    assert (
        missing_authority.reason_code
        == "scope_adjudication_binding_mismatch"
    )

    forged_log_path = Path(
        "state/implementation_logs/rev-001-attempt-5.log"
    )
    (nested_case.root / forged_log_path).write_text(
        "grok forged scope receipt test log\n",
        encoding="utf-8",
    )
    forged_scope = deepcopy(scope_receipt.to_record())
    forged_scope["decisions"][0]["reason_codes"] = [
        ScopeExpansionReason.CANDIDATE_IMPORTS_DECLARED_PATH.value
    ]
    append_jsonl_event(
        nested_case.events_path,
        "implementation_started",
        {
            "task_id": "REV-001",
            "attempt": 5,
            "execution_mode": "model-assisted",
            "branch": "implementation/rev-001-attempt-5",
            "log_path": str(forged_log_path),
            "command": [
                "/usr/bin/python3",
                "/opt/ipfs_accelerate_py/agent_supervisor/grok_cli_runner.py",
                "--grok-bin",
                "/usr/bin/grok",
                "--model",
                "grok-4.5",
            ],
        },
    )
    append_jsonl_event(
        nested_case.events_path,
        "implementation_finished",
        {
            "task_id": "REV-001",
            "attempt": 5,
            "branch": "implementation/rev-001-attempt-5",
            "log_path": str(forged_log_path),
            "implementation_commit": implementation,
            "baseline_ref": nested_case.baseline,
            "canonical_task_key": nested_case.task.canonical_task_key,
            "canonical_task_cid": nested_case.task.canonical_task_cid,
            "returncode": 0,
            "validation_result": {
                "passed": True,
                "proposal_gate": {
                    "accepted": True,
                    "proposal_id": proposal_id,
                    "policy_id": policy_id,
                    "receipt_id": "proposal-receipt:scope-bound",
                    "repository_tree_id": proposal_tree_id,
                    "changed_paths": expected_paths,
                },
                "scope_adjudication": forged_scope,
            },
        },
    )
    forged_provenance = (
        review.verified_implementer_provenance_from_ledger(
            nested_case.events_path,
            repo_root=nested_case.root,
            expected_task_id="REV-001",
            expected_implementation_attempt=5,
            expected_implementation_commit=implementation,
        )
    )
    forged_case = SimpleNamespace(
        **{
            **vars(case),
            "implementation_attempt": 5,
            "provenance": forged_provenance,
        }
    )
    forged_receipt = _perform(
        forged_case,
        reviewer=_reviewer("approve"),
        expected_changed_paths=expected_paths,
        scope_authorized_paths=[companion],
        scope_adjudication_id=adjudication_id,
    )
    assert forged_receipt.admitted is False
    assert (
        forged_receipt.reason_code
        == "scope_adjudication_receipt_invalid"
    )
    assert forged_receipt.event == {}


def test_ledger_native_provenance_is_unique_and_log_bound(
    nested_case: SimpleNamespace,
) -> None:
    selected = review.verified_implementer_provenance_from_ledger(
        nested_case.events_path,
        repo_root=nested_case.root,
        expected_task_id="REV-001",
        expected_implementation_attempt=3,
        expected_implementation_commit=nested_case.implementation,
    )
    assert selected == nested_case.provenance
    assert selected.log_binding_scope == review.IMPLEMENTER_LOG_BINDING_SCOPE
    assert selected.log_event_anchored is False

    with pytest.raises(
        review.PostMergeReviewError,
        match="no valid implementation start/finish pair",
    ):
        review.verified_implementer_provenance_from_ledger(
            nested_case.events_path,
            repo_root=nested_case.root,
            expected_task_id="REV-001",
            expected_implementation_attempt=3,
            expected_implementation_commit=nested_case.baseline,
        )

    append_jsonl_event(
        nested_case.events_path,
        "implementation_finished",
        {
            "task_id": "REV-001",
            "attempt": 3,
            "branch": selected.branch,
            "log_path": selected.log_path,
            "implementation_commit": nested_case.implementation,
            "returncode": 0,
        },
    )
    with pytest.raises(
        review.PostMergeReviewError,
        match="multiple valid implementation event pairs",
    ):
        review.verified_implementer_provenance_from_ledger(
            nested_case.events_path,
            repo_root=nested_case.root,
            expected_task_id="REV-001",
            expected_implementation_attempt=3,
            expected_implementation_commit=nested_case.implementation,
        )


def test_log_mutation_after_provenance_snapshot_fails_closed(
    nested_case: SimpleNamespace,
) -> None:
    (nested_case.root / nested_case.provenance.log_path).write_text(
        "mutated implementation log\n",
        encoding="utf-8",
    )
    outcome = _perform(nested_case, reviewer=_reviewer("approve"))
    assert outcome.admitted is False
    assert outcome.reason_code == "implementer_event_provenance_mismatch"


def test_provenance_commit_cannot_authorize_a_different_reviewed_commit(
    nested_case: SimpleNamespace,
) -> None:
    child = nested_case.root / "external/child"
    (child / "docs/contract.md").write_text(
        "# Contract\n\nSecond implementation.\n",
        encoding="utf-8",
    )
    _commit(child, "different implementation")
    different_commit = _commit(nested_case.root, "different root candidate")
    different_tree = _tree(nested_case.root, different_commit)
    different_case = SimpleNamespace(**vars(nested_case))
    different_case.implementation = different_commit
    different_case.merge_commit = different_commit
    different_case.repository_tree_id = different_tree
    different_case.validation = _validation(
        different_case.task,
        different_commit,
        different_tree,
    )

    outcome = _perform(different_case, reviewer=_reviewer("approve"))
    assert outcome.admitted is False
    assert outcome.reason_code == "implementer_provenance_binding_invalid"


def test_live_production_review_mints_only_after_durable_head_and_admits(
    nested_case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        review,
        "call_llm_router_with_receipt",
        _fake_codex_child("approve"),
    )
    monkeypatch.setattr(
        review,
        "_production_codex_reviewer",
        lambda *_args, **_kwargs: pytest.fail(
            "replaceable reviewer symbol crossed the canonical live boundary"
        ),
    )
    outcome = _perform(nested_case)
    assert outcome.admitted is True
    assert outcome.receipt["attempt"] == 4
    assert outcome.receipt["implementation_attempt"] == 3
    assert outcome.receipt["baseline_commit"] == nested_case.baseline
    assert outcome.receipt["changed_paths"] == nested_case.task.outputs
    assert outcome.event["task_binding_id"] == review.post_merge_task_binding_id(
        nested_case.task
    )
    assert outcome.event["canonical_task_key"] == "uiir-test:REV-001"
    assert outcome.event["canonical_task_cid"] == "sha256:" + "a" * 64
    assert outcome.event["board_namespace"] == "uiir-test"
    provenance = outcome.receipt["implementer_provenance"]
    assert provenance["log_bytes"] == len(b"grok implementation log\n")
    assert provenance["log_sha256"] == hashlib.sha256(
        b"grok implementation log\n"
    ).hexdigest()
    assert (
        provenance["log_binding_scope"]
        == review.IMPLEMENTER_LOG_BINDING_SCOPE
    )
    assert provenance["log_event_anchored"] is False

    forged = dict(outcome.event)
    forged.update(
        {
            "sequence": 999,
            "stream_id": "forged",
            "snapshot_id": "forged",
            "previous_event_id": "",
        }
    )
    forged["event_id"] = "sha256:" + hashlib.sha256(
        json.dumps(
            forged,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert (
        review.mint_gate_from_live_outcome(
            outcome,
            forged,
            events_path=nested_case.events_path,
        )
        == {}
    )

    malformed_payload = dict(outcome.event)
    malformed_type = str(malformed_payload.pop("type"))
    malformed_payload["attempt"] = []
    malformed = append_jsonl_event(
        nested_case.events_path,
        malformed_type,
        malformed_payload,
    )
    assert (
        review.mint_gate_from_live_outcome(
            outcome,
            malformed,
            events_path=nested_case.events_path,
        )
        == {}
    )

    appended = _append_review_event(nested_case.events_path, outcome)
    gate = review.mint_gate_from_live_outcome(
        outcome,
        appended,
        events_path=nested_case.events_path,
    )
    assert gate["gate_kind"] == "provider_review"
    assert gate["review_presence"] == "independent"
    assert gate["task_binding_id"] == outcome.event["task_binding_id"]
    assert gate["canonical_task_key"] == outcome.event["canonical_task_key"]
    assert gate["canonical_task_cid"] == outcome.event["canonical_task_cid"]
    assert gate["board_namespace"] == outcome.event["board_namespace"]

    daemon = TodoImplementationDaemon(
        todo_path=nested_case.todo_path,
        state_path=nested_case.root / "state/acceptance-state.json",
        strategy_path=nested_case.root / "state/acceptance-strategy.json",
        events_path=nested_case.root / "state/acceptance-events.jsonl",
        repo_root=nested_case.root,
        task_header_prefix="## REV-",
        implement=False,
    )
    monkeypatch.setattr(
        daemon,
        "_decision_runtime_completion",
        lambda *_args, **_kwargs: None,
    )
    result = daemon.apply_post_merge_authoritative_acceptance(
        nested_case.task,
        implementation_commit=nested_case.implementation,
        merge_commit=nested_case.merge_commit,
        repository_tree_id=nested_case.repository_tree_id,
        validation_result=nested_case.validation,
        gate_evidence={"provider_review": gate},
        model_invocation_observed=True,
    )
    assert result["authoritatively_completed"] is True


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("implementation_attempt", 99),
        ("review_receipt_path", "/tmp/forged-receipt.json"),
        ("provider_result_admitted", False),
        ("repository_write_allowed", True),
        ("proof_authoritative", True),
        ("completion_authoritative", True),
        ("task_binding_id", "sha256:" + "b" * 64),
        ("canonical_task_key", "forged-task-key"),
        ("canonical_task_cid", "sha256:" + "c" * 64),
        ("board_namespace", "forged-board"),
    ),
)
def test_live_mint_requires_the_entire_exact_outcome_event(
    nested_case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    replacement: Any,
) -> None:
    monkeypatch.setattr(
        review,
        "call_llm_router_with_receipt",
        _fake_codex_child("approve"),
    )
    outcome = _perform(nested_case)
    payload = dict(outcome.event)
    event_type = str(payload.pop("type"))
    payload[field_name] = replacement
    appended = append_jsonl_event(
        nested_case.events_path,
        event_type,
        payload,
    )
    assert (
        review.mint_gate_from_live_outcome(
            outcome,
            appended,
            events_path=nested_case.events_path,
        )
        == {}
    )


def test_mutable_gate_preview_cannot_redirect_sealed_authority(
    nested_case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        review,
        "call_llm_router_with_receipt",
        _fake_codex_child("approve"),
    )
    outcome = _perform(nested_case)
    appended = _append_review_event(nested_case.events_path, outcome)
    assert isinstance(outcome._gate_evidence, dict)
    outcome._gate_evidence.update(
        {
            "task_id": "forged-task",
            "implementation_commit": nested_case.baseline,
            "merge_commit": nested_case.baseline,
            "repository_tree_id": "git-tree:" + "0" * 40,
            "review_receipt_id": "sha256:" + "0" * 64,
        }
    )
    gate = review.mint_gate_from_live_outcome(
        outcome,
        appended,
        events_path=nested_case.events_path,
    )
    assert gate["task_id"] == nested_case.task.task_id
    assert gate["implementation_commit"] == nested_case.implementation
    assert gate["merge_commit"] == nested_case.merge_commit
    assert gate["repository_tree_id"] == nested_case.repository_tree_id
    assert (
        gate["review_receipt_id"]
        == outcome.receipt["receipt_id"]
    )


@pytest.mark.parametrize("mapping_name", ("receipt", "event"))
def test_public_outcome_mapping_mutation_after_append_fails_closed(
    nested_case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    mapping_name: str,
) -> None:
    monkeypatch.setattr(
        review,
        "call_llm_router_with_receipt",
        _fake_codex_child("approve"),
    )
    outcome = _perform(nested_case)
    appended = _append_review_event(nested_case.events_path, outcome)
    mapping = getattr(outcome, mapping_name)
    assert isinstance(mapping, dict)
    if mapping_name == "receipt":
        response = mapping["review_response"]
        assert isinstance(response, dict)
        response["decision"] = "changes_required"
    else:
        mapping["merge_commit"] = nested_case.baseline
    assert (
        review.mint_gate_from_live_outcome(
            outcome,
            appended,
            events_path=nested_case.events_path,
        )
        == {}
    )


def test_nested_outcome_mutation_before_append_fails_closed(
    nested_case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        review,
        "call_llm_router_with_receipt",
        _fake_codex_child("approve"),
    )
    outcome = _perform(nested_case)
    assert isinstance(outcome.event, dict)
    embedded_receipt = outcome.event["review_receipt"]
    assert isinstance(embedded_receipt, dict)
    embedded_response = embedded_receipt["review_response"]
    assert isinstance(embedded_response, dict)
    embedded_response["findings"] = [
        {
            "code": "forged",
            "severity": "low",
            "summary": "Mutated after the live review.",
        }
    ]
    appended = _append_review_event(nested_case.events_path, outcome)
    assert (
        review.mint_gate_from_live_outcome(
            outcome,
            appended,
            events_path=nested_case.events_path,
        )
        == {}
    )


def test_injected_and_declined_reviews_remain_pending(
    nested_case: SimpleNamespace,
) -> None:
    injected = _perform(nested_case, reviewer=_reviewer("approve"))
    assert injected.admitted is False
    assert injected.receipt["production_review_route"] is False
    assert injected.receipt["provider_result_admitted"] is False
    appended = _append_review_event(nested_case.events_path, injected)
    assert (
        review.mint_gate_from_live_outcome(
            injected,
            appended,
            events_path=nested_case.events_path,
        )
        == {}
    )

    declined = _perform(nested_case, reviewer=_reviewer("changes_required"))
    assert declined.admitted is False
    assert declined.reason_code == "independent_review_changes_required"
    assert declined.acceptance_pending is True
    assert declined._gate_evidence == {}
    assert declined._producer_seal is None
    assert (
        review.post_merge_review_denial_tombstone_from_live_outcome(
            declined,
            target_repository_id="repository:test",
            target_branch="main",
        )
        == {}
    )


@pytest.mark.parametrize(
    "mutation",
    ("same_length_swap", "wrong_attempt", "wrong_version", "blank_execution_id"),
)
def test_tampered_transport_cannot_receive_live_seal(
    nested_case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    def tampered(
        prompt: str,
        invocation: Any,
    ) -> tuple[str, LlmChildResultEnvelope]:
        request = _request_from_prompt(prompt)
        assert invocation.request_id == request["request_id"]
        response_text = _response(request)
        kwargs: dict[str, Any] = {}
        if mutation == "same_length_swap":
            swapped = response_text.replace("approve", "APPROVE")
            assert len(swapped) == len(response_text)
            kwargs["digest_text"] = swapped
        elif mutation == "wrong_attempt":
            kwargs["attempt"] = int(request["attempt"]) + 1
        elif mutation == "blank_execution_id":
            kwargs["execution_result_id"] = ""
        child_receipt = LlmChildResultEnvelope.from_dict(
            _transport(request, response_text, **kwargs)
        )
        if mutation == "wrong_version":
            child_receipt = replace(
                child_receipt,
                contract_version=LLM_CHILD_ENVELOPE_VERSION + 1,
            )
        return response_text, child_receipt

    monkeypatch.setattr(review, "call_llm_router_with_receipt", tampered)
    outcome = _perform(nested_case)
    assert outcome.admitted is False
    assert outcome.reason_code == "reviewer_execution_receipt_invalid"


def test_receipt_tamper_and_unmanifested_forged_ledger_fail_closed(
    nested_case: SimpleNamespace,
) -> None:
    structural = _perform(nested_case, reviewer=_reviewer("approve"))
    tampered = deepcopy(dict(structural.receipt))
    tampered["merge_commit"] = nested_case.baseline
    verification = review.verify_post_merge_review_receipt(
        tampered,
        repo_root=nested_case.root,
        implementation_events_path=nested_case.events_path,
        task=nested_case.task,
        validation_result=nested_case.validation,
        attempt=4,
        implementation_attempt=3,
        baseline_commit=nested_case.baseline,
        implementation_commit=nested_case.implementation,
        merge_commit=nested_case.merge_commit,
        repository_tree_id=nested_case.repository_tree_id,
        expected_changed_paths=nested_case.task.outputs,
        implementer_provenance=nested_case.provenance,
    )
    assert verification.valid is False
    assert verification.reason_code == "review_receipt_content_identity_invalid"

    forged_path = nested_case.events_path.with_name("forged-events.jsonl")
    forged_path.write_text(
        nested_case.events_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    forged_case = SimpleNamespace(**vars(nested_case))
    forged_case.events_path = forged_path
    outcome = _perform(forged_case, reviewer=_reviewer("approve"))
    assert outcome.admitted is False
    assert outcome.reason_code == "event_ledger_manifest_invalid"


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("task_binding_id", "sha256:" + "1" * 64),
        (
            "changed_paths",
            [
                "external/child/tests/vocabulary.json",
                "external/child/docs/contract.md",
            ],
        ),
        ("content_binding_id", "sha256:" + "2" * 64),
        ("gitlink_binding_id", "sha256:" + "3" * 64),
        ("diff_binding_id", "sha256:" + "4" * 64),
        ("validation_receipt_id", "sha256:" + "5" * 64),
        ("review_request_id", "sha256:" + "6" * 64),
    ),
)
def test_reidentified_receipt_cannot_forge_top_level_bindings(
    nested_case: SimpleNamespace,
    field_name: str,
    replacement: Any,
) -> None:
    structural = _perform(nested_case, reviewer=_reviewer("approve"))
    tampered = deepcopy(dict(structural.receipt))
    tampered[field_name] = replacement
    material = dict(tampered)
    material.pop("receipt_id")
    tampered["receipt_id"] = content_identity(material)
    verification = review.verify_post_merge_review_receipt(
        tampered,
        repo_root=nested_case.root,
        implementation_events_path=nested_case.events_path,
        task=nested_case.task,
        validation_result=nested_case.validation,
        attempt=4,
        implementation_attempt=3,
        baseline_commit=nested_case.baseline,
        implementation_commit=nested_case.implementation,
        merge_commit=nested_case.merge_commit,
        repository_tree_id=nested_case.repository_tree_id,
        expected_changed_paths=nested_case.task.outputs,
        implementer_provenance=nested_case.provenance,
    )
    assert verification.valid is False
    assert verification.reason_code == "review_receipt_binding_mismatch"


def test_live_mint_accepts_exact_member_after_later_event(
    nested_case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        review,
        "call_llm_router_with_receipt",
        _fake_codex_child("approve"),
    )
    outcome = _perform(nested_case)
    appended = _append_review_event(nested_case.events_path, outcome)
    append_jsonl_event(
        nested_case.events_path,
        "later_unrelated_event",
        {"task_id": "REV-001"},
    )
    gate = review.mint_gate_from_live_outcome(
        outcome,
        appended,
        events_path=nested_case.events_path,
    )
    assert gate["gate_kind"] == "provider_review"


def test_truncated_ledger_cannot_authenticate_implementer(
    nested_case: SimpleNamespace,
) -> None:
    payload = nested_case.events_path.read_bytes()
    nested_case.events_path.write_bytes(payload[:-8])
    outcome = _perform(nested_case, reviewer=_reviewer("approve"))
    assert outcome.admitted is False
    assert outcome.reason_code == "event_ledger_manifest_invalid"
