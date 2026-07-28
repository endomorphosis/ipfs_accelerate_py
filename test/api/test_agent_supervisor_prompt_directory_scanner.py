from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.core import program_behavior
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_directory_scanner import (
    DirectoryResolutionError,
    NestedRepositoryError,
    OptionalAnalysisResult,
    RepositoryAllowlistError,
    ScanSymlinkError,
    SecretLeakageError,
    UnstableDirectoryScanError,
    build_repository_allowlist,
    repository_root_cid,
    scan_prompt_directory,
    scan_prompt_directory_detailed,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    DirectoryScanPolicy,
    EvidenceAuthority,
    OutputMode,
    PromptOutputPolicy,
    PromptPlanningPolicy,
    PromptSource,
    PromptWorkflowBudget,
    PromptWorkflowRequest,
    prompt_workflow_cid,
)


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Scanner Test")
    _git(repository, "config", "user.email", "scanner@example.invalid")
    (repository / "src").mkdir()
    (repository / "src" / "service.py").write_text(
        """class Service:
    def dispatch(self, request):
        return transform(request)
""",
        encoding="utf-8",
    )
    (repository / "test").mkdir()
    (repository / "test" / "test_service.py").write_text(
        "def test_dispatch():\n    assert True\n",
        encoding="utf-8",
    )
    (repository / "README.md").write_text("Fixture documentation.\n", encoding="utf-8")
    (repository / "pyproject.toml").write_text(
        "[build-system]\nrequires = []\n",
        encoding="utf-8",
    )
    (repository / "SECURITY.md").write_text("Security policy.\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "fixture")
    return repository


def _cid(label: str) -> str:
    return prompt_workflow_cid({"fixture": label})


def _budget(**changes: int) -> PromptWorkflowBudget:
    values = {
        "max_files": 100,
        "max_scan_bytes": 2 * 1024 * 1024,
        "max_file_bytes": 256 * 1024,
        "max_symbols": 100,
        "max_prompt_tokens": 1_024,
        "max_provider_tokens": 2_048,
        "max_latency_ms": 60_000,
        "max_goals": 8,
        "max_tasks": 16,
        "max_evidence": 32,
        "max_graph_depth": 8,
        "max_serialized_bytes": 256 * 1024,
        "max_rescue_actions": 4,
    }
    values.update(changes)
    return PromptWorkflowBudget(**values)


def _request(
    repository: Path,
    *,
    directory: Path | None = None,
    budget: PromptWorkflowBudget | None = None,
    scan_policy: DirectoryScanPolicy | None = None,
    output_root: Path | None = None,
    allowed_output_roots: tuple[str, ...] | None = None,
) -> tuple[PromptWorkflowRequest, object]:
    allowlist = build_repository_allowlist((repository,))
    selected_output_root = output_root or repository
    request = PromptWorkflowRequest(
        prompt_source=PromptSource.inline(
            "Improve this directory",
            redacted_metadata={"summary": "Repository improvement request"},
        ),
        repository_root=str(repository),
        directory=str(directory or repository),
        repository_root_cid=repository_root_cid(repository),
        allowlist_cid=allowlist.allowlist_cid,
        scan_policy=scan_policy
        or DirectoryScanPolicy(
            policy_id="scan:default",
            scanner_version="1.0.0",
        ),
        planning_policy=PromptPlanningPolicy(policy_id="planning:deterministic"),
        output_policy=PromptOutputPolicy(
            policy_id="output:test",
            mode=OutputMode.BOTH,
            output_root=str(selected_output_root),
            allowed_output_roots=allowed_output_roots
            or (str(selected_output_root),),
            markdown_path="generated/work.todo.md",
            duckdb_path="generated/work.duckdb",
            board_namespace="scanner-test",
            task_prefix="SCAN",
        ),
        budget=budget or _budget(),
        caller="principal:test",
        program_root=_cid("program-input"),
        intent_ir_root=_cid("intent"),
        legal_ir_root=_cid("legal"),
        security_ir_root=_cid("security"),
        policy_root=_cid("policy"),
    )
    return request, allowlist


def test_scan_is_stable_body_free_and_returns_all_bounded_summary_kinds(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    request, allowlist = _request(repository)

    first = scan_prompt_directory_detailed(
        request, repository_allowlist=allowlist, clock_ms=lambda: 10
    )
    second = scan_prompt_directory_detailed(
        request, repository_allowlist=allowlist, clock_ms=lambda: 20
    )

    assert first.receipt.scan_cid == second.receipt.scan_cid
    assert first.receipt.dirty_worktree_root == second.receipt.dirty_worktree_root
    assert first.receipt.program_root == second.receipt.program_root
    assert first.receipt.started_at_ms != second.receipt.started_at_ms
    assert first.receipt.counts["files"] == 5
    assert first.receipt.counts["tests"] == 1
    assert first.receipt.counts["documents"] >= 2
    assert first.receipt.counts["policies"] == 1
    assert first.receipt.counts["build_files"] == 1
    kinds = {item.source_kind for item in first.receipt.evidence}
    assert {
        "directory_scan_languages",
        "directory_scan_build",
        "directory_scan_interfaces",
        "directory_scan_symbols",
        "directory_scan_tests",
        "directory_scan_documents",
        "directory_scan_policies",
        "directory_scan_worktree",
        "directory_scan_policy",
        "directory_scan_optional_analysis",
    }.issubset(kinds)
    assert all(
        item.authority is EvidenceAuthority.SCAN_ADVISORY
        for item in first.receipt.evidence
    )
    encoded = first.receipt.to_json()
    assert "return transform(request)" not in encoded
    assert "Fixture documentation." not in encoded
    assert first.configuration_root in encoded
    assert all(
        artifact.artifact_handle.startswith("blob:sha256:")
        for artifact in first.artifacts
    )


def test_tracked_staged_modified_deleted_and_untracked_bytes_invalidate_scan(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    request, allowlist = _request(repository)
    clean = scan_prompt_directory(request, repository_allowlist=allowlist)

    service = repository / "src" / "service.py"
    service.write_text("def staged():\n    return 1\n", encoding="utf-8")
    _git(repository, "add", "src/service.py")
    service.write_text("def worktree():\n    return 2\n", encoding="utf-8")
    (repository / "README.md").unlink()
    (repository / "new.py").write_text("NEW = True\n", encoding="utf-8")

    dirty = scan_prompt_directory(request, repository_allowlist=allowlist)

    assert dirty.scan_cid != clean.scan_cid
    assert dirty.dirty_worktree_root != clean.dirty_worktree_root
    assert dirty.counts["staged_and_modified"] == 1
    assert dirty.counts["deleted"] == 1
    assert dirty.counts["untracked"] == 1
    assert dirty.counts["tracked"] == 5

    (repository / "new.py").write_text("NEW = False\n", encoding="utf-8")
    changed_untracked = scan_prompt_directory(
        request, repository_allowlist=allowlist
    )
    assert changed_untracked.scan_cid != dirty.scan_cid


def test_exclusions_bind_ignore_generated_credentials_binary_and_outputs(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    (repository / ".gitignore").write_text(
        "ignored.txt\n",
        encoding="utf-8",
    )
    _git(repository, "add", ".gitignore")
    _git(repository, "commit", "-qm", "ignore policy")
    (repository / "ignored.txt").write_text("ignored bytes\n", encoding="utf-8")
    (repository / ".env").write_text(
        "TOKEN=not-read-by-the-scanner\n", encoding="utf-8"
    )
    (repository / "payload.png").write_bytes(b"\x89PNG\x00opaque")
    (repository / "__pycache__").mkdir()
    (repository / "__pycache__" / "service.pyc").write_bytes(b"cache")
    (repository / "generated").mkdir()
    (repository / "generated" / "work.todo.md").write_text(
        "previous output\n", encoding="utf-8"
    )
    request, allowlist = _request(repository)

    details = scan_prompt_directory_detailed(
        request, repository_allowlist=allowlist
    )

    rendered = "\n".join(details.receipt.exclusions)
    assert ".env: credential_or_key_material" in rendered
    assert "ignored.txt: repository_ignore_policy" in rendered
    assert "payload.png: large_or_binary_default" in rendered
    assert "__pycache__: cache_tree" in rendered
    assert "generated: generated_tree" in rendered
    decisions = next(
        item for item in details.artifacts if item.kind == "scan-decisions"
    ).payload["decisions"]
    decision_by_path = {item["path"]: item for item in decisions}
    assert decision_by_path[".env"]["redactions"] == ["content_not_read"]
    assert decision_by_path["src/service.py"]["redactions"] == [
        "source_body_content_addressed"
    ]
    assert ".env" not in {
        entry.path for entry in details.program_behavior.repository.entries
    }


def test_allowlist_symlink_nested_repository_and_output_escapes_fail_closed(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init", "-q")
    request, allowlist = _request(repository)
    other_allowlist = build_repository_allowlist((other,))

    with pytest.raises(RepositoryAllowlistError, match="allowlist"):
        scan_prompt_directory(request, repository_allowlist=other_allowlist)
    with pytest.raises(RepositoryAllowlistError, match="identity"):
        scan_prompt_directory(
            replace(request, repository_root_cid=_cid("wrong-root")),
            repository_allowlist=allowlist,
        )

    os.symlink("src/service.py", repository / "alias.py")
    with pytest.raises(ScanSymlinkError, match="symlink"):
        scan_prompt_directory(request, repository_allowlist=allowlist)
    (repository / "alias.py").unlink()

    nested = repository / "nested"
    nested.mkdir()
    _git(nested, "init", "-q")
    with pytest.raises(NestedRepositoryError, match="nested"):
        scan_prompt_directory(request, repository_allowlist=allowlist)
    shutil.rmtree(nested / ".git")

    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, repository / "linked-output")
    escaped_request, _ = _request(
        repository,
        output_root=repository / "linked-output",
        allowed_output_roots=(str(repository / "linked-output"),),
    )
    with pytest.raises(DirectoryResolutionError, match="output"):
        scan_prompt_directory(escaped_request, repository_allowlist=allowlist)


def test_secret_content_rejected_without_leaking_value_and_optional_is_lazy(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    secret_value = "sk-" + "z" * 30
    (repository / "src" / "unsafe.py").write_text(
        f'API_KEY = "{secret_value}"\n',
        encoding="utf-8",
    )
    request, allowlist = _request(repository)

    with pytest.raises(SecretLeakageError) as raised:
        scan_prompt_directory(request, repository_allowlist=allowlist)
    assert secret_value not in str(raised.value)

    (repository / "src" / "unsafe.py").unlink()
    calls: list[object] = []

    class BrokenOptional:
        def analyze(self, context):
            calls.append(context)
            raise RuntimeError(secret_value)

    local = scan_prompt_directory_detailed(
        request, repository_allowlist=allowlist
    )
    assert calls == []
    assert local.optional_analysis_status == "not_requested"

    degraded = scan_prompt_directory_detailed(
        request,
        repository_allowlist=allowlist,
        optional_analysis=BrokenOptional(),
    )
    assert len(calls) == 1
    assert degraded.optional_analysis_status.startswith("invocation_failed")
    assert secret_value not in degraded.receipt.to_json()

    authority = scan_prompt_directory_detailed(
        request,
        repository_allowlist=allowlist,
        optional_analysis=lambda context: OptionalAnalysisResult(
            status="available",
            summary="Approximate match.",
            authority=EvidenceAuthority.AUTHORITATIVE,
        ),
    )
    assert authority.optional_analysis_status == "authority_claim_rejected"
    optional_record = next(
        item
        for item in authority.receipt.evidence
        if item.source_kind == "directory_scan_optional_analysis"
    )
    assert optional_record.authority is EvidenceAuthority.SCAN_ADVISORY


def test_budget_truncation_is_exact_and_post_analysis_mutation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    (repository / "src" / "many.py").write_text(
        "\n".join(f"def symbol_{index}(): pass" for index in range(8)) + "\n",
        encoding="utf-8",
    )
    request, allowlist = _request(
        repository, budget=_budget(max_symbols=2)
    )
    bounded = scan_prompt_directory_detailed(
        request, repository_allowlist=allowlist
    )
    assert bounded.receipt.counts["symbols"] == 2
    assert bounded.receipt.truncated is True
    assert any(
        item.startswith("symbol_summary:max_symbols:2-of-")
        for item in bounded.receipt.truncations
    )
    symbol_artifact = next(
        item for item in bounded.artifacts if item.kind == "symbols"
    )
    assert symbol_artifact.payload["summary"]["count"] == 2
    assert symbol_artifact.payload["summary"]["truncated"] is True

    original = program_behavior.build_program_behavior

    def mutate_after_behavior(*args, **kwargs):
        result = original(*args, **kwargs)
        (repository / "src" / "service.py").write_text(
            "def raced():\n    return False\n", encoding="utf-8"
        )
        return result

    monkeypatch.setattr(
        program_behavior, "build_program_behavior", mutate_after_behavior
    )
    with pytest.raises(UnstableDirectoryScanError, match="changed"):
        scan_prompt_directory(request, repository_allowlist=allowlist)
