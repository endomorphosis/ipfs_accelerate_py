"""Tests for PlanningAnalysisFactory@1 (PDR-011)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.planning_analysis_factory import (
    DEFAULT_OPEN_FRONTIERS,
    PLANNING_ANALYSIS_FACTORY_INTERFACE,
    OpenFrontierStatus,
    PlanningAdmissionRequestFactory,
    PlanningAnalysisAdmissionError,
    PlanningAnalysisAllowlistError,
    PlanningAnalysisFactory,
    PlanningAnalysisSecretError,
    PlanningOptionalAnalysisAdapter,
    build_planning_analysis_factory,
    build_planning_analysis_view,
)
from ipfs_accelerate_py.agent_supervisor.analysis.repository_indexer import (
    PLANNING_OPEN_FRONTIER_KINDS,
    planning_category_inventory,
    planning_path_category,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_directory_scanner import (
    OptionalAnalysisResult,
    PromptDirectoryScanner,
    build_prompt_scanner_with_planning_factory,
    build_repository_allowlist,
    repository_root_cid,
    scan_prompt_directory_detailed,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    DirectoryScanPolicy,
    EvidenceAuthority,
    OutputMode,
    PromptOutputPolicy,
    PromptPlanningPolicy,
    PromptSource,
    PromptSupervisorService,
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


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Factory Test")
    _git(repository, "config", "user.email", "factory@example.invalid")
    _write(
        repository / "src" / "service.py",
        "class Service:\n    def dispatch(self, request):\n        return request\n",
    )
    _write(
        repository / "test" / "test_service.py",
        "def test_dispatch():\n    assert True\n",
    )
    _write(repository / "config" / "app.toml", "[app]\nname = 'fixture'\n")
    _write(
        repository / "schemas" / "event.schema.json",
        '{"type":"object","properties":{"id":{"type":"string"}}}\n',
    )
    _write(repository / "docs" / "README.md", "Fixture documentation.\n")
    _write(repository / "SECURITY.md", "Security policy.\n")
    _write(
        repository / "pyproject.toml",
        "[build-system]\nrequires = []\nbuild-backend = 'setuptools.build_meta'\n",
    )
    _write(repository / "native" / "shim.c", "int shim(void) { return 0; }\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "fixture")
    # Dirty overlay: modified tracked file + admitted untracked source.
    _write(
        repository / "src" / "service.py",
        "class Service:\n    def dispatch(self, request):\n        return transform(request)\n",
    )
    _write(repository / "src" / "extra.py", "def extra():\n    return 1\n")
    return repository


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


def _cid(label: str) -> str:
    return prompt_workflow_cid({"fixture": label})


def _request(repository: Path) -> tuple[PromptWorkflowRequest, object]:
    allowlist = build_repository_allowlist((repository,))
    request = PromptWorkflowRequest(
        prompt_source=PromptSource.inline(
            "Analyze the fixture service.",
            redacted_metadata={"summary": "Factory analysis request"},
        ),
        repository_root=str(repository),
        directory=str(repository),
        repository_root_cid=repository_root_cid(repository),
        allowlist_cid=allowlist.allowlist_cid,
        scan_policy=DirectoryScanPolicy(
            policy_id="scan:factory-test",
            scanner_version="1.0.0",
        ),
        planning_policy=PromptPlanningPolicy(policy_id="planning:deterministic"),
        output_policy=PromptOutputPolicy(
            policy_id="output:factory-test",
            mode=OutputMode.BOTH,
            output_root=str(repository),
            allowed_output_roots=(str(repository),),
            markdown_path="generated/work.todo.md",
            duckdb_path="generated/work.duckdb",
            board_namespace="factory-test",
            task_prefix="PDR",
        ),
        budget=_budget(),
        caller="principal:test",
        program_root=_cid("program-input"),
        intent_ir_root=_cid("intent"),
        legal_ir_root=_cid("legal"),
        security_ir_root=_cid("security"),
        policy_root=_cid("policy"),
    )
    return request, allowlist


def test_planning_path_categories_cover_required_surfaces() -> None:
    assert planning_path_category("test/test_service.py") == "tests"
    assert planning_path_category("config/app.toml") == "config"
    assert planning_path_category("pyproject.toml") == "build"
    assert planning_path_category("schemas/event.schema.json") == "schema"
    assert planning_path_category("docs/README.md") == "docs"
    assert planning_path_category("SECURITY.md") == "policies"
    inventory = planning_category_inventory(
        [
            "test/test_service.py",
            "config/app.toml",
            "pyproject.toml",
            "schemas/event.schema.json",
            "docs/README.md",
            "SECURITY.md",
            "src/service.py",
        ]
    )
    for name in ("tests", "config", "build", "schema", "docs", "policies"):
        assert inventory[name]["count"] >= 1


def test_factory_scans_allowlisted_checkout_with_categories_and_frontiers(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    factory = build_planning_analysis_factory(
        repository_allowlist=(repository,),
        index_root=tmp_path / "index",
    )
    view = factory.analyze(repository)

    assert factory.INTERFACE == PLANNING_ANALYSIS_FACTORY_INTERFACE
    assert view.factory_interface == PLANNING_ANALYSIS_FACTORY_INTERFACE
    assert view.completeness in {"complete", "partial_with_frontier"}
    assert view.dirty_overlay_id
    assert view.sca_snapshot.stats.dirty_path_count >= 1
    assert "dirty_overlay_admitted" in view.notes
    assert "target_code_not_imported" in view.notes

    totals = view.category_inventory["totals"]
    assert totals["tests"] >= 1
    assert totals["config"] >= 1
    assert totals["build"] >= 1
    assert totals["schema"] >= 1
    assert totals["docs"] >= 1
    assert totals["policies"] >= 1

    kinds = {item.kind for item in view.open_frontiers}
    assert kinds == set(PLANNING_OPEN_FRONTIER_KINDS)
    assert set(view.open_frontier_ids) == set(DEFAULT_OPEN_FRONTIERS)
    assert all(
        item.status
        in {
            OpenFrontierStatus.OPEN,
            OpenFrontierStatus.DEGRADED,
            OpenFrontierStatus.ABSTAINED,
        }
        for item in view.open_frontiers
    )
    # Native path should be reflected on the native frontier sample.
    native = next(item for item in view.open_frontiers if item.kind == "native")
    assert native.path_count >= 1

    assert view.repository_index is not None
    assert view.reasoning_snapshot.roots.repository_id
    assert view.reasoning_snapshot.roots.tree_id
    assert view.reasoning_snapshot.stability.stable is True
    # Body-free projection.
    payload = view.to_dict()
    assert "body" not in str(payload).casefold()
    assert "source_text" not in str(payload)


def test_factory_recursive_configured_submodules(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    nested = tmp_path / "nested"
    nested.mkdir()
    _git(nested, "init", "-q")
    _git(nested, "config", "user.name", "Nested")
    _git(nested, "config", "user.email", "nested@example.invalid")
    _write(nested / "lib.py", "VALUE = 1\n")
    _git(nested, "add", ".")
    _git(nested, "commit", "-qm", "nested")
    nested_commit = _git(nested, "rev-parse", "HEAD")

    # Register as a gitlink without network submodule add.
    vendor = repository / "vendor"
    vendor.mkdir(exist_ok=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{nested_commit},vendor/nested",
        ),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _git(repository, "commit", "-qm", "add gitlink")
    # Materialize checkout for recursive walk.
    target = repository / "vendor" / "nested"
    if target.exists():
        pass
    else:
        subprocess.run(
            ("cp", "-a", str(nested), str(target)),
            check=True,
        )

    factory = PlanningAnalysisFactory(
        repository_allowlist=(repository,),
        index_root=tmp_path / "index-sub",
        build_index=True,
    )
    view = factory.analyze(repository)
    assert any(item.path == "vendor/nested" for item in view.submodule_closure)
    nested_entry = next(
        item for item in view.submodule_closure if item.path == "vendor/nested"
    )
    assert nested_entry.commit_id == nested_commit.lower()
    assert nested_entry.available is True
    # Reasoning snapshot carries recursive gitlink identity.
    assert any(
        item.path == "vendor/nested"
        for item in view.reasoning_snapshot.recursive_gitlinks()
    )


def test_wrong_tree_and_allowlist_fail_closed(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init", "-q")
    _git(other, "config", "user.name", "Other")
    _git(other, "config", "user.email", "other@example.invalid")
    _write(other / "a.py", "x = 1\n")
    _git(other, "add", ".")
    _git(other, "commit", "-qm", "other")

    factory = PlanningAnalysisFactory(
        repository_allowlist=(repository,),
        index_root=tmp_path / "index-allow",
        build_index=False,
    )
    with pytest.raises(PlanningAnalysisAllowlistError):
        factory.analyze(other)

    # Empty allowlist rejected at construction.
    with pytest.raises(PlanningAnalysisAllowlistError):
        PlanningAnalysisFactory(repository_allowlist=())


def test_secret_material_fails_closed(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    secret_value = "sk-" + "z" * 30
    _write(
        repository / "src" / "unsafe.py",
        f'API_KEY = "{secret_value}"\n',
    )
    factory = PlanningAnalysisFactory(
        repository_allowlist=(repository,),
        index_root=tmp_path / "index-secret",
        build_index=False,
    )
    with pytest.raises(PlanningAnalysisSecretError) as raised:
        factory.analyze(repository)
    assert secret_value not in str(raised.value)


def test_symlink_escape_fails_closed(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    _write(outside / "leak.py", "LEAK = 1\n")
    link = repository / "src" / "escaped.py"
    os.symlink(outside / "leak.py", link)
    # Stage the symlink into the index so the snapshot sees it.
    _git(repository, "add", "src/escaped.py")
    factory = PlanningAnalysisFactory(
        repository_allowlist=(repository,),
        index_root=tmp_path / "index-symlink",
        build_index=False,
    )
    with pytest.raises(Exception) as raised:
        factory.analyze(repository)
    # Either symlink escape or a related path/snapshot rejection.
    message = str(raised.value).casefold()
    assert (
        "symlink" in message
        or "escape" in message
        or raised.type.__name__ in {
            "PlanningAnalysisSymlinkError",
            "SymlinkEscapeError",
            "PlanningAnalysisPathEscapeError",
            "RepositoryPathEscapeError",
            "RepositorySnapshotError",
        }
    )


def test_lazy_optional_provider_loss_degrades_or_abstains(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    def boom() -> None:
        raise RuntimeError("provider exploded")

    factory = PlanningAnalysisFactory(
        repository_allowlist=(repository,),
        index_root=tmp_path / "index-opt",
        build_index=False,
        optional_providers={
            "cfg": boom,
            "dataflow": lambda: None,
            "native": lambda: {"status": "unavailable"},
            "generated": lambda: {"status": "available"},
            # concurrency omitted -> not_requested
        },
    )
    view = factory.analyze(repository)
    assert view.optional_provider_status["cfg"] == "failed"
    assert view.optional_provider_status["dataflow"] == "abstained"
    assert view.optional_provider_status["native"] == "abstained"
    assert view.optional_provider_status["generated"] == "available"
    assert view.optional_provider_status["concurrency"] == "not_requested"

    by_kind = {item.kind: item for item in view.open_frontiers}
    assert by_kind["cfg"].status is OpenFrontierStatus.DEGRADED
    assert by_kind["dataflow"].status is OpenFrontierStatus.ABSTAINED
    assert by_kind["native"].status is OpenFrontierStatus.ABSTAINED
    # Available optional providers still cannot close frontiers without certification.
    assert by_kind["generated"].status is OpenFrontierStatus.DEGRADED
    assert by_kind["concurrency"].status is OpenFrontierStatus.OPEN


def test_wires_default_optional_analysis_and_admission_request_factory(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    factory = PlanningAnalysisFactory(
        repository_allowlist=(repository,),
        index_root=tmp_path / "index-wire",
        build_index=True,
    )
    view = factory.analyze(repository)
    assert view is factory.last_view

    assert isinstance(factory.optional_analysis, PlanningOptionalAnalysisAdapter)
    assert isinstance(
        factory.admission_request_factory, PlanningAdmissionRequestFactory
    )

    service = PromptSupervisorService(repository_allowlist=(str(repository),))
    assert service.optional_analysis is None
    assert service.admission_request_factory is None
    factory.wire_prompt_supervisor(service)
    assert service.optional_analysis is factory.optional_analysis
    assert service.admission_request_factory is factory.admission_request_factory

    scanner = PromptDirectoryScanner((repository,))
    assert scanner.optional_analysis is None
    scanner.wire_planning_analysis_factory(factory)
    assert scanner.optional_analysis is factory.optional_analysis

    # Wired scanner helper.
    scanner2 = build_prompt_scanner_with_planning_factory((repository,), factory)
    assert scanner2.optional_analysis is factory.optional_analysis

    # Optional analysis adapter returns advisory inventory summary.
    class _Ctx:
        request_cid = "req"
        repository_root_cid = "repo"
        dirty_worktree_root = "dirty"
        scanner_policy_cid = "policy"
        program_root = "program"
        ast_root = "ast"
        configuration_root = "config"
        included_paths = ("src/service.py",)
        category_counts = {"tests": 1}
        max_summary_bytes = 4_096

    result = factory.optional_analysis.analyze(_Ctx())
    assert isinstance(result, OptionalAnalysisResult)
    assert result.status == "available"
    assert result.authority is EvidenceAuthority.SCAN_ADVISORY
    assert "tests=" in result.summary
    assert result.artifact_cid == view.view_cid

    # Admission factory is wired but fails closed without IR builder.
    with pytest.raises(PlanningAnalysisAdmissionError):
        factory.admission_request_factory.build(None, None, None)


def test_admission_request_factory_with_builder(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    class _FakeIR:
        pass

    def builder(**kwargs):
        assert "repository_tree_id" in kwargs
        return _FakeIR()

    factory = PlanningAnalysisFactory(
        repository_allowlist=(repository,),
        index_root=tmp_path / "index-adm",
        build_index=False,
        ir_request_builder=builder,
    )
    factory.analyze(repository)

    class _Graph:
        pass

    # PromptPlanAdmissionRequest requires a real graph; bare IR is returned when
    # graph validation would fail.  Exercise builder path with a graph that
    # triggers PromptPlanAdmissionRequest construction only if graph is valid.
    # Here we expect TypeError from PromptPlanAdmissionRequest on fake graph.
    with pytest.raises((TypeError, PlanningAnalysisAdmissionError)):
        factory.admission_request_factory.build(None, None, _Graph())


def test_optional_analysis_through_directory_scanner(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    factory = PlanningAnalysisFactory(
        repository_allowlist=(repository,),
        index_root=tmp_path / "index-scan",
        build_index=True,
    )
    factory.analyze(repository)
    request, allowlist = _request(repository)
    details = scan_prompt_directory_detailed(
        request,
        repository_allowlist=allowlist,
        optional_analysis=factory.optional_analysis,
    )
    assert details.optional_analysis_status in {
        "available",
        "identity_mismatch",
        "degraded",
    }
    # When bindings match empty claims, adapter fills from context — status
    # should not be not_requested.
    assert details.optional_analysis_status != "not_requested"


def test_one_shot_builder(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    view = build_planning_analysis_view(
        repository,
        repository_allowlist=(repository,),
        index_root=tmp_path / "index-once",
        build_index=False,
    )
    assert view.reasoning_snapshot is not None
    assert set(view.open_frontier_ids) == set(DEFAULT_OPEN_FRONTIERS)


def test_factory_does_not_import_target_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = _repository(tmp_path)
    # Write a module that would raise if imported.
    _write(
        repository / "src" / "boom_module.py",
        "raise RuntimeError('target code was imported')\n",
    )
    import sys

    before = set(sys.modules)
    factory = PlanningAnalysisFactory(
        repository_allowlist=(repository,),
        index_root=tmp_path / "index-noimport",
        build_index=True,
    )
    view = factory.analyze(repository)
    after = set(sys.modules)
    leaked = {
        name
        for name in after - before
        if "boom_module" in name or name.endswith("src.service")
    }
    assert not leaked
    assert view.repository_index is not None or "index_degraded" in " ".join(view.notes)
