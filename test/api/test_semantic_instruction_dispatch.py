"""Signed public instructions survive real native dispatch and historical replay.

The provider returns authored text. Manifest admission, semantic indexing,
capsule compilation, Git worktree checks and audit reconstruction run normally.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_router_public_instruction import selected  # noqa: F401
from test.api.test_semantic_router_integration import provider  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, build_text_context_references, render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import router_public_instruction as instruction
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_metadata_view import SemanticMetadataTokenCounter
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit


MODE = "common-bindings-instruction@1"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def expanded_selected(scenario, tmp_path):
    """Bind a substantial instruction before admission; never repin a signed artifact."""
    root = scenario["repository"]
    source = ('"""Public authored requirements.\n' +
        "Preserve all accepted behavior and return the integer two from answer.\n" * 60 +
        '"""\nfrom answer import answer\nassert answer() == 2, "answer must be two"\n')
    (root / "test_answer.py").write_text(source)
    subprocess.run(["git", "-C", str(root), "add", "test_answer.py"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Instruction qualification",
        "-c", "user.email=instruction@example.invalid", "commit", "-qm", "authored public requirement"], check=True)
    profile, lifecycle = tmp_path / "expanded-profile", tmp_path / "expanded-lifecycle"
    Supervisor.init_local(repository=root, consent=True, profile_dir=profile, lifecycle_dir=lifecycle)
    graph = replace(scenario["graph"],
        scan_cid=content_identity({"sources": local._sources(root, ["answer.py", "test_answer.py"])}),
        program_root=content_identity({"git_tree": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True).strip()}))
    manifest = local.author_local_benchmark_manifest(repository=root, profile_dir=profile,
        lifecycle_dir=lifecycle, task_specs=scenario["manifest"]["payload"]["tasks"],
        planning_roots={key: getattr(graph, key) for key in ("request_cid", "scan_cid", "program_root")})
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=manifest)
    raw = (root / "test_answer.py").read_bytes()
    context = instruction.prepare_public_instruction_context(repository=root, admission=admission,
        task_cid=scenario["task_cid"], source_path="test_answer.py", expected_source_sha256=sha(raw))
    workspace = tmp_path / "expanded-allocated"
    subprocess.run(["git", "-C", str(root), "worktree", "add", "--detach", "-q", str(workspace)], check=True)
    request = dict(artifact=Path(context["artifact"]), expected_sha256=context["sha256"],
        task_cid=scenario["task_cid"], prompt="", workspace=workspace)
    return root, admission, raw, context, request


def compose(selection, monkeypatch, *, include_instruction=True):
    root, _, source, _, request = selection
    paths = ["answer.py", "test_answer.py"] if include_instruction else ["answer.py"]
    output = root / ".runtime/semantic"
    prepare_semantic_context(repository=root, paths=paths, required_raw_paths=paths,
        objective="Repair the answer", task_id="LOCAL-TASK", output=output)
    artifact = output / "worker-context.json"
    refs = build_text_context_references(artifact.read_text(), reference_prefix="semantic-context",
        kind="semantic-context", path=artifact.relative_to(root).as_posix(), repository_id="repo:test",
        tree_id="tree:test", required=True, chunk_bytes=1201)
    compiled = ContextCompiler(ContextBudget(max_input_tokens=65536, max_items=128,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(
        repository_id="repo:test", tree_id="tree:test", objective_id="LOCAL-TASK",
        objective_revision="sha256:task", policy_id="policy:test", policy_revision="sha256:policy",
        caller="supervisor:test", stage="implementation", goal={"id": "LOCAL-TASK"},
        authority={"mode": "candidate_only", "completion_authority": False},
        scope={"allowed_paths": ["answer.py"]}, acceptance={"criteria": ["public validation pending"]}, evidence=refs)
    request["prompt"] = render_context_capsule(compiled.capsule)
    monkeypatch.chdir(request["workspace"])
    monkeypatch.setenv("CODEX_HOME", str(root.parent / "empty-codex-home"))
    return dict(root=root, source=source, request=request)


def invoke(case, **options):
    request = case["request"]
    return runner.run(prompt=request["prompt"], provider="codex_cli", model="pinned", timeout=1,
        max_output_tokens=128, semantic_repository=case["root"],
        public_instruction_artifact=request["artifact"], public_instruction_sha256=request["expected_sha256"],
        public_instruction_task_cid=request["task_cid"], **options)


def counter(compact_tokens):
    return SemanticMetadataTokenCounter(
        count=lambda text: compact_tokens if '"schema":"supervisor-semantic-instruction-view@1"' in text else 100,
        provider="codex_cli", model="pinned", tokenizer="authored-counter",
        tokenizer_revision="fixture@1", qualification="estimate")


def historical_replay(case, receipt, actual, monkeypatch, *, token_counter=None):
    parsed = audit._receipt({**receipt, "phase": "coding"}, case["request"]["workspace"].parent)
    (case["root"] / "answer.py").write_text("def answer():\n    return 2\n")
    (case["root"] / "test_answer.py").write_text("subsequently changed public instruction")
    subprocess.run(["git", "-C", str(case["root"]), "worktree", "remove", "--force",
        str(case["request"]["workspace"])], check=True)
    monkeypatch.chdir(case["root"].parent)
    replayed, _, checks = audit._model_projection(rendered=case["request"]["prompt"],
        receipt=parsed, repository=case["root"], token_counter=token_counter)
    assert replayed == actual and all(checks.values()), checks


def test_signed_instruction_selected_once_with_unchanged_native_prompt_and_historical_replay(
        expanded_selected, provider, monkeypatch, record_property):
    case = compose(expanded_selected, monkeypatch)
    prompt = case["request"]["prompt"]
    block, _ = instruction.load_public_instruction(**case["request"])
    _, baseline = invoke(case)
    baseline_actual = provider[1][-1][0]
    _, receipt = invoke(case, semantic_metadata_view=MODE)
    actual = provider[1][-1][0]
    selection = receipt["semantic_metadata_view"]
    for field in ("native_complete_bytes", "candidate_complete_bytes", "selected_complete_bytes",
                  "selected_mode", "fallback_reason"):
        record_property(field, selection[field])
    record_property("measurement_scope", "complete_rendered_authored_coding_input_bytes")
    record_property("live_token_savings_measured", False)
    assert selection["selected_mode"] == selection["requested_mode"] == MODE
    assert len(actual.encode()) < len(baseline_actual.encode())
    assert selection["native_complete_sha256"] == sha(baseline_actual.encode())
    assert selection["selected_complete_sha256"] == sha(actual.encode())
    assert receipt["native_prompt_sha256"] == baseline["native_prompt_sha256"] == sha(prompt.encode())
    assert receipt["native_prompt_bytes"] == len(prompt.encode())
    assert case["request"]["prompt"] == prompt and "semantic_metadata_view" not in baseline
    assert actual.count(block) == baseline_actual.count(block) == 1
    start = receipt["workspace_advisory_bytes"]
    wire, _ = json.JSONDecoder().raw_decode(actual[start:])
    raw_sources = wire["translated_semantic"]["raw_sources"]
    assert raw_sources["test_answer.py"]["$public_instruction"]["source_sha256"] == sha(case["source"])
    assert raw_sources["answer.py"] == (case["root"] / "answer.py").read_text()
    assert receipt["public_instruction"]["semantic_minification_applied"] is False
    assert receipt["public_instruction"]["manifest_signature_verified"] is True
    assert receipt["usage"] == baseline["usage"] and len(provider[1]) == 2
    historical_replay(case, receipt, actual, monkeypatch)


@pytest.mark.parametrize("compact_tokens,expected", [(90, MODE), (200, "legacy")])
def test_counter_can_select_or_refuse_instruction_view_and_replays_after_source_change(
        expanded_selected, provider, monkeypatch, compact_tokens, expected):
    case = compose(expanded_selected, monkeypatch)
    trusted = counter(compact_tokens)
    _, receipt = invoke(case, semantic_metadata_view=MODE, semantic_metadata_token_counter=trusted)
    selection = receipt["semantic_metadata_view"]
    assert selection["schema"] == "supervisor-semantic-metadata-selection@2"
    assert selection["selected_mode"] == expected and selection["requested_mode"] == MODE
    assert selection["selected_complete_tokens"] == (compact_tokens if expected == MODE else 100)
    assert selection["candidate_complete_bytes"] < selection["native_complete_bytes"]
    assert selection["provider_usage_reported"] is False
    historical_replay(case, receipt, provider[1][-1][0], monkeypatch, token_counter=trusted)


def test_small_signed_instruction_falls_back_to_exact_native_input(selected, provider, monkeypatch, record_property):
    case = compose(selected, monkeypatch)
    _, baseline = invoke(case)
    actual = provider[1][-1][0]
    _, receipt = invoke(case, semantic_metadata_view=MODE)
    selection = receipt["semantic_metadata_view"]
    for field in ("native_complete_bytes", "candidate_complete_bytes", "selected_complete_bytes",
                  "selected_mode", "fallback_reason"):
        record_property(field, selection[field])
    record_property("measurement_scope", "complete_rendered_authored_coding_input_bytes")
    record_property("live_token_savings_measured", False)
    assert selection["selected_mode"] == "legacy" and selection["requested_mode"] == MODE
    assert selection["fallback_reason"] == "complete_input_not_smaller_under_bytes_and_proxy"
    assert provider[1][-1][0] == actual
    assert receipt["model_prompt_sha256"] == baseline["model_prompt_sha256"]
    historical_replay(case, receipt, actual, monkeypatch)


@pytest.mark.parametrize("change", ["canonical_instruction", "worker_instruction", "missing_raw_source", "missing_binding"])
def test_instruction_view_rejects_stale_or_absent_requirements_before_provider(
        selected, provider, monkeypatch, change):
    case = compose(selected, monkeypatch, include_instruction=change != "missing_raw_source")
    if change == "canonical_instruction":
        (case["root"] / "test_answer.py").write_text("stale canonical public instruction")
    elif change == "worker_instruction":
        (case["request"]["workspace"] / "test_answer.py").write_text("stale allocated public instruction")
    with pytest.raises(ValueError):
        if change == "missing_binding":
            runner.run(prompt=case["request"]["prompt"], provider="codex_cli", model="pinned", timeout=1,
                max_output_tokens=128, semantic_repository=case["root"], semantic_metadata_view=MODE)
        else:
            invoke(case, semantic_metadata_view=MODE)
    assert provider[1] == []


@pytest.mark.parametrize("token_counted", [False, True])
def test_instruction_census_has_exact_selected_input_without_provider_discovery(
        expanded_selected, provider, monkeypatch, token_counted):
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    case = compose(expanded_selected, monkeypatch)
    trusted = counter(90) if token_counted else None
    _, delivered = invoke(case, semantic_metadata_view=MODE,
        **({"semantic_metadata_token_counter": trusted} if trusted else {}))
    calls_before = len(provider[1])
    def forbidden(*args, **kwargs):
        pytest.fail("census attempted provider discovery")
    monkeypatch.setattr(intelligence_index, "discover_available_providers", forbidden)
    _, census = invoke(case, semantic_metadata_view=MODE, census_only=True,
        **({"semantic_metadata_token_counter": trusted} if trusted else {}))
    assert len(provider[1]) == calls_before
    assert census["provider_calls"] == census["router_calls"] == 0
    assert census["provider_dispatch_observed"] is False
    assert census["provider_token_usage"] is None and census["total_token_savings_measured"] is False
    assert census["census"]["selected_mode"] == MODE
    assert census["census"]["selected_complete_sha256"] == delivered["semantic_metadata_view"]["selected_complete_sha256"]
    assert census["public_instruction_included"] is True
