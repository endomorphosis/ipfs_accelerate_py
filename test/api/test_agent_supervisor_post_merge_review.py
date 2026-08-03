from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy
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
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.llm import (
    LLM_CHILD_ENVELOPE_VERSION,
    LlmChildProviderCapacityError,
    LlmChildResultEnvelope,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.post_merge_validation import (
    POST_MERGE_VALIDATION_EVIDENCE_SCHEMA,
    build_post_merge_validation_evidence,
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
    canonical = build_post_merge_validation_evidence(
        task_id=task.task_id,
        target_commit=merge_commit,
        repository_tree_id=repository_tree_id,
        validation_result=material,
        validated_commit=merge_commit,
    )
    return {**material, **canonical}


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
        assert invocation.model_name == "gpt-5.6-sol"
        assert invocation.codex_read_only is True
        assert invocation.allow_cross_provider_fallback is False
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
    approved_descendant_gitlinks: dict[str, str] | None = None,
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
        approved_descendant_gitlinks=approved_descendant_gitlinks,
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


def _land_composite_child_with_sibling(
    case: SimpleNamespace,
    *,
    overlap_task_leaf: bool = False,
) -> tuple[str, str]:
    child = case.root / "external/child"
    implementation_child = _git(
        case.root,
        "rev-parse",
        f"{case.implementation}:external/child",
    )
    _git(child, "checkout", "-b", "landed-sibling", implementation_child)
    if overlap_task_leaf:
        (child / "docs/contract.md").write_text(
            "# Contract\n\nOverwritten by sibling work.\n",
            encoding="utf-8",
        )
    else:
        (child / "sibling.txt").write_text(
            "independent landed work\n",
            encoding="utf-8",
        )
    _commit(child, "land sibling child work")
    _git(child, "checkout", "-b", "landed-composite", implementation_child)
    _git(child, "merge", "--no-ff", "landed-sibling", "-m", "merge child work")
    landed_child = _git(child, "rev-parse", "HEAD")
    landed_root = _commit(case.root, "record composite child")
    return landed_child, landed_root


def test_descendant_gitlink_requires_exact_authorization_and_leaf_bytes(
    nested_case: SimpleNamespace,
) -> None:
    landed_child, landed_root = _land_composite_child_with_sibling(
        nested_case
    )

    with pytest.raises(review.PostMergeReviewError) as missing:
        review._collect_repository_binding(
            repo_root=nested_case.root,
            task=nested_case.task,
            baseline_commit=nested_case.baseline,
            implementation_commit=nested_case.implementation,
            merge_commit=landed_root,
            repository_tree_id=_tree(nested_case.root, landed_root),
            expected_changed_paths=nested_case.task.outputs,
        )
    assert missing.value.reason_code == "merged_content_binding_mismatch"

    binding = review._collect_repository_binding(
        repo_root=nested_case.root,
        task=nested_case.task,
        baseline_commit=nested_case.baseline,
        implementation_commit=nested_case.implementation,
        merge_commit=landed_root,
        repository_tree_id=_tree(nested_case.root, landed_root),
        expected_changed_paths=nested_case.task.outputs,
        approved_descendant_gitlinks={"external/child": landed_child},
    )

    assert binding["changed_paths"] == nested_case.task.outputs
    assert binding["approved_descendant_gitlinks"] == {
        "external/child": landed_child,
    }
    root_gitlink = binding["gitlink_bindings"][0]
    assert root_gitlink["landing_relation"] == "approved_descendant"
    assert root_gitlink["merged"]["git_object_id"] == landed_child
    assert all(
        item["implementation"] == item["merged"]
        for item in binding["content_bindings"]
    )
    assert "external/child/sibling.txt" not in binding["changed_paths"]


def test_exact_gitlink_rejects_unused_descendant_authorization(
    nested_case: SimpleNamespace,
) -> None:
    implementation_child = _git(
        nested_case.root,
        "rev-parse",
        f"{nested_case.implementation}:external/child",
    )

    with pytest.raises(review.PostMergeReviewError) as raised:
        review._collect_repository_binding(
            repo_root=nested_case.root,
            task=nested_case.task,
            baseline_commit=nested_case.baseline,
            implementation_commit=nested_case.implementation,
            merge_commit=nested_case.merge_commit,
            repository_tree_id=nested_case.repository_tree_id,
            expected_changed_paths=nested_case.task.outputs,
            approved_descendant_gitlinks={
                "external/child": implementation_child,
            },
        )

    assert (
        raised.value.reason_code
        == "descendant_gitlink_authorization_unused"
    )


@pytest.mark.parametrize(
    "malformed",
    ([], [("external/child", "a" * 40)], ""),
)
def test_descendant_gitlink_rejects_falsey_or_iterable_non_mapping_grant(
    nested_case: SimpleNamespace,
    malformed: Any,
) -> None:
    with pytest.raises(review.PostMergeReviewError) as raised:
        review._collect_repository_binding(
            repo_root=nested_case.root,
            task=nested_case.task,
            baseline_commit=nested_case.baseline,
            implementation_commit=nested_case.implementation,
            merge_commit=nested_case.merge_commit,
            repository_tree_id=nested_case.repository_tree_id,
            expected_changed_paths=nested_case.task.outputs,
            approved_descendant_gitlinks=malformed,
        )

    assert (
        raised.value.reason_code
        == "descendant_gitlink_authorization_invalid"
    )


def test_descendant_gitlink_rejects_sibling_overlap_with_task_leaf(
    nested_case: SimpleNamespace,
) -> None:
    landed_child, landed_root = _land_composite_child_with_sibling(
        nested_case,
        overlap_task_leaf=True,
    )

    with pytest.raises(review.PostMergeReviewError) as raised:
        review._collect_repository_binding(
            repo_root=nested_case.root,
            task=nested_case.task,
            baseline_commit=nested_case.baseline,
            implementation_commit=nested_case.implementation,
            merge_commit=landed_root,
            repository_tree_id=_tree(nested_case.root, landed_root),
            expected_changed_paths=nested_case.task.outputs,
            approved_descendant_gitlinks={"external/child": landed_child},
        )

    assert raised.value.reason_code == "merged_content_binding_mismatch"


def test_descendant_gitlink_rejects_foreign_same_tree_child(
    nested_case: SimpleNamespace,
) -> None:
    child = nested_case.root / "external/child"
    implementation_child = _git(
        nested_case.root,
        "rev-parse",
        f"{nested_case.implementation}:external/child",
    )
    implementation_tree = _git(
        child,
        "rev-parse",
        f"{implementation_child}^{{tree}}",
    )
    foreign_child = _git(
        child,
        "commit-tree",
        implementation_tree,
        "-m",
        "foreign same-tree child",
    )
    _git(child, "checkout", "--detach", foreign_child)
    landed_root = _commit(nested_case.root, "record foreign same-tree child")

    with pytest.raises(review.PostMergeReviewError) as raised:
        review._collect_repository_binding(
            repo_root=nested_case.root,
            task=nested_case.task,
            baseline_commit=nested_case.baseline,
            implementation_commit=nested_case.implementation,
            merge_commit=landed_root,
            repository_tree_id=_tree(nested_case.root, landed_root),
            expected_changed_paths=nested_case.task.outputs,
            approved_descendant_gitlinks={"external/child": foreign_child},
        )

    assert raised.value.reason_code == "submodule_implementation_not_contained"


def test_descendant_gitlink_ignores_replace_refs_at_review_boundary(
    nested_case: SimpleNamespace,
) -> None:
    child = nested_case.root / "external/child"
    implementation_child = _git(
        nested_case.root,
        "rev-parse",
        f"{nested_case.implementation}:external/child",
    )
    (child / "docs/contract.md").write_text(
        "# Contract\n\nMalicious unrelated landing.\n",
        encoding="utf-8",
    )
    _git(child, "add", "docs/contract.md")
    malicious_tree = _git(child, "write-tree")
    unrelated_child = _git(
        child,
        "commit-tree",
        malicious_tree,
        "-m",
        "unrelated malicious child",
    )
    implementation_tree = _git(
        child,
        "rev-parse",
        f"{implementation_child}^{{tree}}",
    )
    synthetic_replacement = _git(
        child,
        "commit-tree",
        implementation_tree,
        "-p",
        implementation_child,
        "-m",
        "forged ancestry and tree",
    )
    _git(child, "replace", unrelated_child, synthetic_replacement)
    _git(child, "checkout", "--detach", unrelated_child)
    landed_root = _commit(
        nested_case.root,
        "record unrelated child hidden by replacement",
    )

    with pytest.raises(review.PostMergeReviewError) as raised:
        review._collect_repository_binding(
            repo_root=nested_case.root,
            task=nested_case.task,
            baseline_commit=nested_case.baseline,
            implementation_commit=nested_case.implementation,
            merge_commit=landed_root,
            repository_tree_id=_tree(nested_case.root, landed_root),
            expected_changed_paths=nested_case.task.outputs,
            approved_descendant_gitlinks={
                "external/child": unrelated_child,
            },
        )

    assert raised.value.reason_code == "submodule_implementation_not_contained"


def test_descendant_gitlink_ignores_legacy_grafts_at_review_boundary(
    nested_case: SimpleNamespace,
) -> None:
    child = nested_case.root / "external/child"
    implementation_child = _git(
        nested_case.root,
        "rev-parse",
        f"{nested_case.implementation}:external/child",
    )
    implementation_tree = _git(
        child,
        "rev-parse",
        f"{implementation_child}^{{tree}}",
    )
    unrelated_child = _git(
        child,
        "commit-tree",
        implementation_tree,
        "-m",
        "unrelated child with the candidate tree",
    )
    raw_git_dir = Path(_git(child, "rev-parse", "--git-dir"))
    git_dir = raw_git_dir if raw_git_dir.is_absolute() else child / raw_git_dir
    grafts_path = git_dir / "info/grafts"
    grafts_path.parent.mkdir(parents=True, exist_ok=True)
    grafts_path.write_text(
        f"{unrelated_child} {implementation_child}\n",
        encoding="ascii",
    )
    _git(
        child,
        "merge-base",
        "--is-ancestor",
        implementation_child,
        unrelated_child,
    )
    _git(child, "checkout", "--detach", unrelated_child)
    landed_root = _commit(
        nested_case.root,
        "record unrelated child hidden by legacy graft",
    )

    with pytest.raises(review.PostMergeReviewError) as raised:
        review._collect_repository_binding(
            repo_root=nested_case.root,
            task=nested_case.task,
            baseline_commit=nested_case.baseline,
            implementation_commit=nested_case.implementation,
            merge_commit=landed_root,
            repository_tree_id=_tree(nested_case.root, landed_root),
            expected_changed_paths=nested_case.task.outputs,
            approved_descendant_gitlinks={
                "external/child": unrelated_child,
            },
        )

    assert raised.value.reason_code == "submodule_implementation_not_contained"


def test_descendant_gitlink_rejects_divergent_implementation_child(
    nested_case: SimpleNamespace,
) -> None:
    child = nested_case.root / "external/child"
    original_child = _git(
        nested_case.root,
        "rev-parse",
        f"{nested_case.implementation}:external/child",
    )
    implementation_tree = _git(
        child,
        "rev-parse",
        f"{original_child}^{{tree}}",
    )
    divergent_child = _git(
        child,
        "commit-tree",
        implementation_tree,
        "-m",
        "divergent implementation child",
    )
    _git(nested_case.root, "checkout", "--detach", nested_case.baseline)
    _git(child, "checkout", "--detach", divergent_child)
    divergent_implementation = _commit(
        nested_case.root,
        "record divergent implementation child",
    )
    landed_child = _git(
        child,
        "commit-tree",
        implementation_tree,
        "-p",
        divergent_child,
        "-m",
        "advance divergent child",
    )
    _git(child, "checkout", "--detach", landed_child)
    landed_root = _commit(nested_case.root, "record advanced divergent child")

    with pytest.raises(review.PostMergeReviewError) as raised:
        review._collect_repository_binding(
            repo_root=nested_case.root,
            task=nested_case.task,
            baseline_commit=nested_case.baseline,
            implementation_commit=divergent_implementation,
            merge_commit=landed_root,
            repository_tree_id=_tree(nested_case.root, landed_root),
            expected_changed_paths=nested_case.task.outputs,
            approved_descendant_gitlinks={"external/child": landed_child},
        )

    assert (
        raised.value.reason_code
        == "submodule_implementation_diverged_from_baseline"
    )


def test_composite_gitlink_still_requires_fresh_validation_and_review(
    nested_case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    landed_child, landed_root = _land_composite_child_with_sibling(
        nested_case
    )
    landed_tree = _tree(nested_case.root, landed_root)
    advanced = SimpleNamespace(**vars(nested_case))
    advanced.merge_commit = landed_root
    advanced.repository_tree_id = landed_tree

    provider_calls: list[str] = []
    child = _fake_codex_child("approve")

    def counted_child(prompt: str, invocation: Any):
        provider_calls.append(str(invocation.request_id))
        return child(prompt, invocation)

    monkeypatch.setattr(
        review,
        "call_llm_router_with_receipt",
        counted_child,
    )
    stale = _perform(
        advanced,
        approved_descendant_gitlinks={"external/child": landed_child},
    )
    assert stale.admitted is False
    assert stale.reason_code == "post_merge_validation_unbound"
    assert provider_calls == []

    advanced.validation = _validation(
        advanced.task,
        landed_root,
        landed_tree,
    )
    fresh = _perform(
        advanced,
        approved_descendant_gitlinks={"external/child": landed_child},
    )

    assert fresh.admitted is True
    assert len(provider_calls) == 1
    assert fresh.acceptance_pending is True
    assert fresh.receipt["repository_write_allowed"] is False
    assert fresh.receipt["proof_authoritative"] is False
    assert fresh.receipt["completion_authoritative"] is False


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


def test_live_codex_capacity_failure_is_typed_and_remains_pending(
    nested_case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def capacity_failure(*_args: Any, **_kwargs: Any) -> Any:
        raise LlmChildProviderCapacityError(
            provider_id="codex_cli",
            reason_codes=("usage_limit", "capacity_unavailable"),
            next_eligible_at="2026-08-10T05:23:00Z",
        )

    monkeypatch.setattr(
        review,
        "call_llm_router_with_receipt",
        capacity_failure,
    )
    outcome = _perform(nested_case)

    assert outcome.admitted is False
    assert outcome.event == {}
    assert outcome.receipt == {}
    assert outcome.reason_code == "reviewer_provider_capacity_unavailable"
    assert outcome.retryable is True
    assert outcome.acceptance_pending is True
    assert outcome.provider_reason_codes == (
        "usage_limit",
        "capacity_unavailable",
    )
    assert outcome.provider_next_eligible_at == "2026-08-10T05:23:00Z"


def test_injected_exception_cannot_forge_production_capacity_metadata(
    nested_case: SimpleNamespace,
) -> None:
    class ForgedCapacityError(RuntimeError):
        reason_code = "reviewer_provider_capacity_unavailable"
        reason_codes = ("usage_limit", "capacity_unavailable")
        next_eligible_at = "2026-08-10T05:23:00Z"

    def forged_reviewer(
        _prompt: str,
        _request: dict[str, Any],
    ) -> review.ReviewerInvocation:
        raise ForgedCapacityError("forged provider metadata")

    outcome = _perform(nested_case, reviewer=forged_reviewer)

    assert outcome.admitted is False
    assert outcome.reason_code == "independent_review_failed"
    assert outcome.provider_reason_codes == ()
    assert outcome.provider_next_eligible_at == ""
    assert outcome.event == {}


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

    consume_kwargs = {
        "task": nested_case.task,
        "implementation_commit": nested_case.implementation,
        "merge_commit": nested_case.merge_commit,
        "repository_tree_id": nested_case.repository_tree_id,
    }
    plain_gate = dict(gate)
    copied_gate = copy(gate)
    reloaded_gate = json.loads(json.dumps(plain_gate))
    assert type(copied_gate) is dict
    for untrusted_gate in (plain_gate, copied_gate, reloaded_gate):
        assert (
            review._consume_live_post_merge_review_gate(
                untrusted_gate,
                **consume_kwargs,
            )
            is None
        )

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
    assert (
        review._consume_live_post_merge_review_gate(
            gate,
            **consume_kwargs,
        )
        is None
    )

    concurrent_gate = review.mint_gate_from_live_outcome(
        outcome,
        appended,
        events_path=nested_case.events_path,
    )

    def consume_concurrently(_index: int) -> dict[str, Any] | None:
        return review._consume_live_post_merge_review_gate(
            concurrent_gate,
            **consume_kwargs,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        concurrent_results = list(pool.map(consume_concurrently, range(32)))
    assert sum(result is not None for result in concurrent_results) == 1


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


@pytest.fixture()
def composite_recovery_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Build the closed Grok -> one-line repair -> advanced target lineage."""

    submodule_path = "external/ipfs_datasets"
    child_paths = (
        "docs/architecture/UI_UX_IR_MCP_IDL_IDENTITY.md",
        "tests/fixtures/ui_ux_ir/v1/mcp_idl_identity_vectors.json",
        "tests/unit/logic/ui_ux_ir/test_mcp_idl_identity_contract.py",
    )
    changed_paths = [f"{submodule_path}/{path}" for path in child_paths]
    restored_symbol = (
        "test_reject_datasets_resource_cost_hints_exclusion"
    )
    provider_symbol = (
        "test_reject_resource_cost_hints_omission_from_verified_identity"
    )
    baseline_test = (
        f"def {restored_symbol}():\n"
        '    assert "baseline-contract"\n'
    )
    provider_test = (
        f"def {provider_symbol}():\n"
        '    assert "provider-contract"\n'
    )
    final_test = (
        f"def {restored_symbol}():\n"
        '    assert "provider-contract"\n'
    )

    child_source = tmp_path / "ipfs-datasets-source"
    _init_repo(child_source)
    for path in child_paths:
        (child_source / path).parent.mkdir(parents=True, exist_ok=True)
    (child_source / child_paths[0]).write_text(
        "# Baseline identity contract\n",
        encoding="utf-8",
    )
    (child_source / child_paths[1]).write_text(
        '{"version":0,"identity":"baseline"}\n',
        encoding="utf-8",
    )
    (child_source / child_paths[2]).write_text(
        baseline_test,
        encoding="utf-8",
    )
    baseline_child = _commit(child_source, "child baseline")
    (child_source / child_paths[0]).write_text(
        "# Provider-authored identity contract\n\nSubstantive content.\n",
        encoding="utf-8",
    )
    (child_source / child_paths[1]).write_text(
        '{"version":1,"identity":"provider"}\n',
        encoding="utf-8",
    )
    (child_source / child_paths[2]).write_text(
        provider_test,
        encoding="utf-8",
    )
    provider_child = _commit(child_source, "Grok provider child")
    (child_source / child_paths[2]).write_text(
        final_test,
        encoding="utf-8",
    )
    final_child = _commit(child_source, "restore exact baseline test symbol")

    root = tmp_path / "accelerator"
    _init_repo(root)
    (root / ".gitignore").write_text("state/\n", encoding="utf-8")
    _git(
        root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child_source),
        submodule_path,
    )
    child = root / submodule_path
    _git(child, "checkout", "--detach", baseline_child)
    baseline = _commit(root, "root baseline")

    _git(root, "checkout", "-b", "provider-source")
    _git(child, "checkout", "--detach", provider_child)
    source_commit = _commit(root, "Grok provider root")
    _git(root, "checkout", "-b", "recovery-seed", baseline)
    _git(child, "checkout", "--detach", final_child)
    seed_commit = _commit(root, "zero-edit recovery seed")
    seed_tree_id = _tree(root, seed_commit)

    _git(root, "checkout", "-b", "main", baseline)
    _git(root, "merge", "--no-ff", "recovery-seed", "-m", "integrate seed")
    integration_boundary = _git(root, "rev-parse", "HEAD")
    integration_boundary_tree = _git(
        root,
        "rev-parse",
        f"{integration_boundary}^{{tree}}",
    )
    (root / "later-unrelated.txt").write_text(
        "target advanced after exact integration\n",
        encoding="utf-8",
    )
    reviewed_target = _commit(root, "advance target after recovery integration")
    reviewed_tree_id = _tree(root, reviewed_target)
    assert reviewed_tree_id != seed_tree_id

    log_path = Path("state/implementation_logs/source-attempt-2.log")
    (root / log_path).parent.mkdir(parents=True, exist_ok=True)
    (root / log_path).write_text(
        "canonical Grok execution log\n",
        encoding="utf-8",
    )
    events_path = root / "state/events.jsonl"
    task_id = "UIIR-042"
    canonical_task_key = "uiir:UIIR-042"
    canonical_task_cid = "sha256:" + ("c" * 64)
    board_namespace = "uiir"
    task = PortalTask(
        task_id=task_id,
        title="Complete the UI/UX IR MCP/IDL identity contract",
        status="ready",
        completion="manual",
        priority="P0",
        track="logic",
        outputs=list(changed_paths),
        validation=["/usr/bin/true"],
        acceptance=(
            "The exact UI/UX IR identity contract and rejection vectors are "
            "implemented without weakening the baseline test vocabulary."
        ),
        metadata={
            "status": "ready",
            "completion": "manual",
            "priority": "P0",
            "track": "logic",
            "outputs": ", ".join(changed_paths),
            "validation": "/usr/bin/true",
            "acceptance": (
                "The exact UI/UX IR identity contract and rejection vectors "
                "are implemented without weakening the baseline test "
                "vocabulary."
            ),
            "provider role": "grok-implement, codex-review",
            "canonical task key": canonical_task_key,
            "canonical task cid": canonical_task_cid,
            "board namespace": board_namespace,
        },
        canonical_task_key=canonical_task_key,
        canonical_task_cid=canonical_task_cid,
        board_namespace=board_namespace,
    )
    task_binding_id = review.post_merge_task_binding_id(task)
    branch = "implementation/uiir-042-attempt-5"
    request_id = "merge-request-uiir-042"
    target_repository_id = "repository:sha256:" + ("d" * 64)
    target_branch = "main"
    denial_id = "denial-uiir-042"
    grant_id = "grant-uiir-042"
    grant_record_id = "grant-record-uiir-042"
    consumption_record_id = "consumption-record-uiir-042"
    repair_task_id = "UIIR-043"
    repair_binding_id = "repair-binding-uiir-043"
    authority_binding_id = "authority-binding-uiir-042"

    source_started = append_jsonl_event(
        events_path,
        "implementation_started",
        {
            "task_id": task_id,
            "attempt": 2,
            "execution_mode": "model-assisted",
            "branch": "implementation/uiir-042-attempt-2",
            "baseline_ref": baseline,
            "log_path": str(log_path),
            "canonical_task_key": canonical_task_key,
            "canonical_task_cid": canonical_task_cid,
            "board_namespace": board_namespace,
            "task_binding_id": task_binding_id,
            "command": [
                "/usr/bin/python3",
                "/opt/ipfs_accelerate_py/grok_cli_runner.py",
                "--workspace",
                str(root),
                "--grok-bin",
                "/usr/bin/grok",
                "--model",
                "grok-4.5",
                "--max-turns",
                "100000",
                "--mode",
                "agent",
            ],
        },
    )
    append_jsonl_event(
        events_path,
        "implementation_finished",
        {
            "task_id": task_id,
            "attempt": 2,
            "branch": "implementation/uiir-042-attempt-2",
            "baseline_ref": baseline,
            "log_path": str(log_path),
            "implementation_commit": source_commit,
            "returncode": 78,
            "attempt_consumed": True,
            "canonical_task_key": canonical_task_key,
            "canonical_task_cid": canonical_task_cid,
            "board_namespace": board_namespace,
            "task_binding_id": task_binding_id,
            "commit_result": {
                "committed": True,
                "commit": source_commit,
                "submodule_results": [
                    {
                        "path": submodule_path,
                        "committed": True,
                        "commit": provider_child,
                    }
                ],
            },
            "validation_result": {
                "attempted": False,
                "passed": False,
                "returncode": 78,
                "reason": "proposal_gate_failed",
                "error": "proposal_validation_failed",
                "proposal_gate": {
                    "attempted": True,
                    "accepted": False,
                    "reason_codes": ["test_weakening_forbidden"],
                    "proof_authoritative": False,
                    "completion_authoritative": False,
                    "repository_tree_id": baseline,
                    "changed_paths": changed_paths,
                },
            },
        },
    )
    seed_fields = {
        "recovery_seed_ref": seed_commit,
        "recovery_seed_tree_id": seed_tree_id,
        "recovery_seed_submodule_path": submodule_path,
        "recovery_seed_submodule_commit": final_child,
    }
    grant_projection = {
        "schema": "post-merge-correction-repair-grant-v1",
        "grant_id": grant_id,
        "denial_id": denial_id,
        "source_task_id": task_id,
        "source_task_binding_id": task_binding_id,
        "source_canonical_task_key": canonical_task_key,
        "source_canonical_task_cid": canonical_task_cid,
        "repair_task_id": repair_task_id,
        "repair_binding_id": repair_binding_id,
        "origin_stream_id": source_started["stream_id"],
        **seed_fields,
    }
    grant_event = append_jsonl_event(
        events_path,
        "task_retry_budget_reset",
        {
            "resets": [
                {
                    "post_merge_correction_repair_grant": grant_projection,
                }
            ]
        },
    )
    authority = {
        "task_id": task_id,
        "task_binding_id": task_binding_id,
        "canonical_task_key": canonical_task_key,
        "canonical_task_cid": canonical_task_cid,
        "board_namespace": board_namespace,
        "authorized_attempt": 5,
        "origin_stream_id": source_started["stream_id"],
        "durable_denial_id": denial_id,
        "authority_id": grant_id,
        "authority_binding_id": authority_binding_id,
        "authority_event_sequence": grant_event["sequence"],
        "durable_authority_head_record_id": grant_record_id,
        "target_repository_id": target_repository_id,
        "target_branch": target_branch,
        "repair_task_id": repair_task_id,
        "repair_binding_id": repair_binding_id,
        **seed_fields,
    }
    recovery_log_path = "state/implementation_logs/recovery-attempt-5.log"
    recovery_started = append_jsonl_event(
        events_path,
        "implementation_started",
        {
            "task_id": task_id,
            "attempt": 5,
            "execution_mode": "recovery-seed-validation",
            "command": ["/usr/bin/true"],
            "branch": branch,
            "baseline_ref": baseline,
            "log_path": recovery_log_path,
            "canonical_task_key": canonical_task_key,
            "canonical_task_cid": canonical_task_cid,
            "board_namespace": board_namespace,
            "task_binding_id": task_binding_id,
            "post_merge_correction_authority": authority,
        },
    )
    recovery_finished = append_jsonl_event(
        events_path,
        "implementation_finished",
        {
            "task_id": task_id,
            "attempt": 5,
            "branch": branch,
            "baseline_ref": baseline,
            "log_path": recovery_log_path,
            "implementation_commit": seed_commit,
            "returncode": 0,
            "attempt_consumed": True,
            "canonical_task_key": canonical_task_key,
            "canonical_task_cid": canonical_task_cid,
            "board_namespace": board_namespace,
            "task_binding_id": task_binding_id,
            "implementation_started_event_id": recovery_started["event_id"],
            "implementation_started_event_sequence": recovery_started[
                "sequence"
            ],
            "commit_result": {
                "committed": True,
                "reason": "existing_commit",
                "commit": seed_commit,
                "baseline_ref": baseline,
                "recovery_seed_zero_edit_promotion_guard": {
                    "allowed": True,
                    "applicable": True,
                    "durable_consumption_verified": True,
                    "reasons": [],
                    "implementation_started_event_id": recovery_started[
                        "event_id"
                    ],
                    "implementation_started_event_sequence": recovery_started[
                        "sequence"
                    ],
                    "validation_changed_paths": changed_paths,
                    **seed_fields,
                },
            },
            "merge_result": {
                "attempted": False,
                "merged": False,
                "queued": True,
                "request_id": request_id,
                "implementation_commit": seed_commit,
                "branch": branch,
            },
            "validation_result": {
                "passed": True,
                "returncode": 0,
                "proposal_gate": {
                    "accepted": True,
                    "changed_paths": changed_paths,
                },
            },
        },
    )
    recovery_material = {
        "schema": review.RECOVERY_SEED_ZERO_EDIT_MERGE_PROVENANCE_SCHEMA,
        "task_id": task_id,
        "task_binding_id": task_binding_id,
        "canonical_task_key": canonical_task_key,
        "canonical_task_cid": canonical_task_cid,
        "board_namespace": board_namespace,
        "implementation_attempt": 5,
        "implementation_commit": seed_commit,
        "branch": branch,
        "baseline_ref": baseline,
        "request_id": request_id,
        "implementation_provider": "",
        "target_already_integrated": True,
        "observed_target_commit": reviewed_target,
        "observed_target_gitlink": final_child,
        "candidate_tree_id": seed_tree_id,
        "target_repository_id": target_repository_id,
        "target_branch": target_branch,
        "denial_id": denial_id,
        "grant_id": grant_id,
        "grant_record_id": grant_record_id,
        "consumption_record_id": consumption_record_id,
        "repair_task_id": repair_task_id,
        "repair_binding_id": repair_binding_id,
        "authority_binding_id": authority_binding_id,
        "grant_event_id": grant_event["event_id"],
        "grant_event_sequence": grant_event["sequence"],
        "started_event_id": recovery_started["event_id"],
        "started_event_sequence": recovery_started["sequence"],
        "finished_event_id": recovery_finished["event_id"],
        "finished_event_sequence": recovery_finished["sequence"],
        "origin_stream_id": source_started["stream_id"],
        "source": "verified_recovery_seed_zero_edit",
        "queue_projection_verified": True,
        "legacy_model_invocation_projection": False,
        "validation_changed_paths": changed_paths,
        "integration_boundary": {
            "commit": integration_boundary,
            "tree": integration_boundary_tree,
            "mode": "exact_seed_no_ff_merge",
        },
        **seed_fields,
    }
    recovery = {
        **recovery_material,
        "evidence_id": content_identity(recovery_material),
    }
    witness_payload = {
        **recovery,
        "queue_attempt": 2,
        "queue_failure_count": 1,
        "request_claim_generation": 7,
        "raw_model_invocation_observed": False,
        "effective_model_invocation_observed": False,
        "model_invocation_observed": False,
        "normalization_reason": (
            "verified_recovery_seed_no_model_execution"
        ),
        "authoritative": False,
        "proof_authoritative": False,
        "completion_authoritative": False,
        "repository_write_authorized": False,
    }
    early_witness = append_jsonl_event(
        events_path,
        review.RECOVERY_SEED_ZERO_EDIT_EXECUTION_VERIFIED_EVENT,
        {**witness_payload, "queue_status": "processing"},
    )
    witness = append_jsonl_event(
        events_path,
        review.RECOVERY_SEED_ZERO_EDIT_EXECUTION_VERIFIED_EVENT,
        {
            **witness_payload,
            "queue_status": "completed",
        },
    )

    correction_material = {
        "kind": "baseline-test-symbol-restoration",
        "path": child_paths[2],
        "root_relative_path": changed_paths[2],
        "line_number": 1,
        "baseline_symbol_line_number": 1,
        "baseline_child_commit": baseline_child,
        "provider_child_commit": provider_child,
        "final_child_commit": final_child,
        "baseline_blob_id": _git(
            child,
            "rev-parse",
            f"{baseline_child}:{child_paths[2]}",
        ),
        "provider_blob_id": _git(
            child,
            "rev-parse",
            f"{provider_child}:{child_paths[2]}",
        ),
        "final_blob_id": _git(
            child,
            "rev-parse",
            f"{final_child}:{child_paths[2]}",
        ),
        "provider_symbol": provider_symbol,
        "restored_symbol": restored_symbol,
        "provider_line_sha256": hashlib.sha256(
            provider_test.splitlines(keepends=True)[0].encode("utf-8")
        ).hexdigest(),
        "final_line_sha256": hashlib.sha256(
            final_test.splitlines(keepends=True)[0].encode("utf-8")
        ).hexdigest(),
        "preserves_all_other_bytes": True,
    }
    correction_id = content_identity(correction_material)
    correction_identity = (
        baseline_child,
        provider_child,
        final_child,
        child_paths[2],
        provider_symbol,
        restored_symbol,
        correction_id,
    )
    monkeypatch.setattr(
        review,
        "COMPOSITE_RECOVERY_DETERMINISTIC_CORRECTIONS",
        frozenset({correction_identity}),
    )
    factory_kwargs = {
        "repo_root": root,
        "expected_task_id": task_id,
        "expected_task_binding_id": task_binding_id,
        "expected_canonical_task_key": canonical_task_key,
        "expected_canonical_task_cid": canonical_task_cid,
        "expected_board_namespace": board_namespace,
        "expected_implementation_attempt": 5,
        "expected_implementation_commit": seed_commit,
        "expected_branch": branch,
        "expected_baseline_ref": baseline,
        "expected_integration_commit": reviewed_target,
        "expected_repository_tree_id": reviewed_tree_id,
        "expected_target_repository_id": target_repository_id,
        "expected_target_branch": target_branch,
        "expected_request_id": request_id,
        "expected_queue_attempt": 2,
        "expected_queue_failure_count": 1,
        "expected_request_claim_generation": 7,
        "recovery_seed_provenance": recovery,
        "recovery_execution_witness": witness,
    }
    return SimpleNamespace(
        root=root,
        child=child,
        events_path=events_path,
        receipt_dir=root / "state/post_merge_receipts",
        task=task,
        changed_paths=changed_paths,
        baseline=baseline,
        recovery=recovery,
        witness=witness,
        early_witness=early_witness,
        factory_kwargs=factory_kwargs,
        seed_commit=seed_commit,
        seed_tree_id=seed_tree_id,
        reviewed_target=reviewed_target,
        reviewed_tree_id=reviewed_tree_id,
        integration_boundary=integration_boundary,
        recovery_finished=recovery_finished,
        correction_id=correction_id,
    )


def test_composite_recovery_provenance_accepts_advanced_review_target(
    composite_recovery_case: SimpleNamespace,
) -> None:
    case = composite_recovery_case
    provenance = (
        review.verified_composite_recovery_implementer_provenance_from_ledger(
            case.events_path,
            **case.factory_kwargs,
        )
    )

    assert isinstance(
        provenance,
        review.VerifiedCompositeRecoveryImplementerProvenance,
    )
    assert case.seed_tree_id != case.reviewed_tree_id
    assert (
        provenance.recovery_execution["integration_boundary_commit"]
        == case.integration_boundary
    )
    assert (
        provenance.recovery_execution["review_target_commit"]
        == case.reviewed_target
    )
    assert (
        provenance.recovery_execution["review_target_tree_id"]
        == case.reviewed_tree_id
    )
    member = review.verified_implementation_finished_event_from_ledger(
        case.events_path,
        provenance,
        repo_root=case.root,
    )
    assert member["event_id"] == case.recovery_finished["event_id"]


@pytest.mark.parametrize(
    ("tamper_target", "replacement"),
    [
        ("witness_event_id", "sha256:" + ("0" * 64)),
        ("witness_queue_attempt", 3),
        ("early_witness", None),
        ("boundary_tree", "0" * 40),
        ("grant_id", "different-grant"),
    ],
)
def test_composite_recovery_provenance_rejects_tampered_bindings(
    composite_recovery_case: SimpleNamespace,
    tamper_target: str,
    replacement: Any,
) -> None:
    case = composite_recovery_case
    kwargs = deepcopy(case.factory_kwargs)
    if tamper_target == "early_witness":
        kwargs["recovery_execution_witness"] = case.early_witness
    elif tamper_target.startswith("witness_"):
        if tamper_target == "witness_event_id":
            witness = deepcopy(case.witness)
            witness["event_id"] = replacement
            kwargs["recovery_execution_witness"] = witness
        else:
            kwargs["expected_queue_attempt"] = replacement
    else:
        recovery = deepcopy(case.recovery)
        if tamper_target == "boundary_tree":
            recovery["integration_boundary"]["tree"] = replacement
        else:
            recovery["grant_id"] = replacement
        material = dict(recovery)
        material.pop("evidence_id")
        recovery["evidence_id"] = content_identity(material)
        kwargs["recovery_seed_provenance"] = recovery

    with pytest.raises(review.PostMergeReviewError):
        review.verified_composite_recovery_implementer_provenance_from_ledger(
            case.events_path,
            **kwargs,
        )


def test_composite_recovery_provenance_rejects_correction_not_allowlisted(
    composite_recovery_case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        review,
        "COMPOSITE_RECOVERY_DETERMINISTIC_CORRECTIONS",
        frozenset(),
    )

    with pytest.raises(review.PostMergeReviewError) as raised:
        review.verified_composite_recovery_implementer_provenance_from_ledger(
            composite_recovery_case.events_path,
            **composite_recovery_case.factory_kwargs,
        )

    assert (
        raised.value.reason_code
        == "composite_recovery_correction_not_authorized"
    )


def test_composite_recovery_denial_round_trip_rejects_nested_witness_tamper(
    composite_recovery_case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = composite_recovery_case
    provenance = (
        review.verified_composite_recovery_implementer_provenance_from_ledger(
            case.events_path,
            **case.factory_kwargs,
        )
    )
    monkeypatch.setattr(
        review,
        "call_llm_router_with_receipt",
        _fake_codex_child("changes_required"),
    )
    outcome = review.perform_post_merge_independent_review(
        repo_root=case.root,
        receipt_dir=case.receipt_dir,
        implementation_events_path=case.events_path,
        task=case.task,
        attempt=2,
        implementation_attempt=5,
        baseline_commit=case.baseline,
        implementation_commit=case.seed_commit,
        merge_commit=case.reviewed_target,
        repository_tree_id=case.reviewed_tree_id,
        validation_result=_validation(
            case.task,
            case.reviewed_target,
            case.reviewed_tree_id,
        ),
        expected_changed_paths=case.changed_paths,
        implementer_provider="grok_cli",
        implementer_provenance=provenance,
    )

    assert outcome.admitted is False
    assert outcome.reason_code == "independent_review_changes_required"
    appended = _append_review_event(case.events_path, outcome)
    corrections = (
        review.verified_post_merge_review_corrections_from_strict_ledger(
            case.events_path,
            require_local_provenance=True,
        )
    )
    assert len(corrections) == 1
    assert corrections[0]["task_id"] == case.task.task_id
    assert corrections[0]["implementation_attempt"] == 5
    assert corrections[0]["target_implementation_attempt"] == 6
    assert corrections[0]["source_event_id"] == appended["event_id"]

    tampered = deepcopy(provenance.to_dict())
    tampered["recovery_execution"]["execution_witness"][
        "grant_record_id"
    ] = "tampered-grant-record"
    assert review._composite_provenance_matches_local_ledger(
        tampered,
        tuple(review._strict_event_ledger(case.events_path)),
        denial_event_sequence=appended["sequence"],
        expected_task_id=case.task.task_id,
        expected_task_binding_id=review.post_merge_task_binding_id(case.task),
        expected_canonical_task_key=case.task.canonical_task_key,
        expected_canonical_task_cid=case.task.canonical_task_cid,
        expected_board_namespace=case.task.board_namespace,
        expected_review_attempt=2,
        expected_implementation_attempt=5,
        expected_implementation_commit=case.seed_commit,
        expected_merge_commit=case.reviewed_target,
        expected_repository_tree_id=case.reviewed_tree_id,
    ) is False
