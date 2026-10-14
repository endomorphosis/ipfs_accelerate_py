"""Real producer source projections over authored files, never model traffic."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, build_text_context_references, render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.runtime import code_source_view as rendering
from ipfs_accelerate_py.agent_supervisor.runtime import source_input_view as view
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as native
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata


def error(code, operation):
    with pytest.raises(view.SourceInputViewError) as caught:
        operation()
    assert caught.value.code == code


@pytest.fixture
def fixture_factory(tmp_path):
    sequence = 0
    def build(source=None, query="compute alpha", limit=8, required_program=False):
        nonlocal sequence
        sequence += 1
        root = tmp_path / f"canonical{sequence}"
        root.mkdir()
        if source is None:
            source = ("# Original comments are retained locally.\n"
                      "CONSTANT = 3\n\n"
                      "def compute_alpha(value):\n"
                      "    # A long authored explanation whose original bytes remain intact.\n"
                      "    # Another detailed comment deliberately exercises rendering only.\n"
                      "    return value + CONSTANT\n\n"
                      "class Holder:\n"
                      "    def compute(self, value):\n"
                      "        # Method commentary is absent from compact reading.\n"
                      "        return value + 2\n")
        (root / "program.py").write_text(source)
        (root / "instruction.txt").write_text("Preserve exact requirements; edit only the authored source.\n")
        with public_replay_without_metadata():
            prepare_semantic_context(repository=root, paths=["program.py", "instruction.txt"],
                program_paths=["program.py"], required_raw_paths=["instruction.txt"] + (["program.py"] if required_program else []),
                objective="Authored compact source presentation", task_id="AUTHORED-SOURCE-VIEW",
                output=root / "semantic", worker_query=query, worker_capsule_limit=limit,
                worker_selector="duckdb-query-symbol-name@1", worker_max_bytes=32768)
            artifact = root / "semantic/worker-context.json"
            references = build_text_context_references(artifact.read_text(), reference_prefix="semantic-context",
                kind="semantic-context", path="semantic/worker-context.json", repository_id="repo:authored",
                tree_id="tree:authored", required=True, chunk_bytes=1201)
            compiled = ContextCompiler(ContextBudget(max_input_tokens=131072, max_items=128,
                max_item_bytes=16384, max_serialized_bytes=262144)).compile(repository_id="repo:authored",
                tree_id="tree:authored", objective_id="AUTHORED-SOURCE-VIEW", objective_revision="sha256:task",
                policy_id="policy:authored", policy_revision="sha256:policy", caller="supervisor:authored",
                stage="implementation", goal={"id":"AUTHORED-SOURCE-VIEW"}, authority={"mode":"candidate_only","completion_authority":False},
                scope={"allowed_paths":["program.py"]}, acceptance={"criteria":["pending authored checks"]}, evidence=references)
            prompt = render_context_capsule(compiled.capsule) + "\nLiteral task suffix.\n"
            encoded = native.encode_semantic_router_prompt(prompt=prompt, repository=root)
        workspace = tmp_path / f"allocated{sequence}"
        workspace.mkdir()
        for path in ("program.py", "instruction.txt"):
            shutil.copyfile(root / path, workspace / path)
        return root, workspace, encoded
    return build


@pytest.fixture
def case(fixture_factory):
    return fixture_factory()


def project(case, **options):
    root, workspace, encoded = case
    with public_replay_without_metadata():
        return view.project_source_input_view(encoded=encoded, repository=root, workspace=workspace, **options)


@pytest.mark.parametrize("format", ["raw", "compact"])
@pytest.mark.parametrize("factor_ast", [False, True])
def test_real_producer_projection_and_exact_restore(case, format, factor_ast):
    root, workspace, encoded = case
    original_context = (root / "semantic/worker-context.json").read_bytes()
    projected = project(case, format=format, factor_ast=factor_ast)
    receipt = projected.receipt
    assert receipt["views_included"] >= 1
    assert receipt["compact_views_included"] > 0 if format == "compact" else receipt["compact_views_included"] == 0
    assert receipt["ast_slots_factored"] == (receipt["views_included"] if factor_ast else 0)
    assert receipt["provider_prompt_sha256"] == hashlib.sha256(projected.provider_prompt.encode()).hexdigest()
    assert receipt["provider_prompt_bytes"] == len(projected.provider_prompt.encode())
    assert (root / "semantic/worker-context.json").read_bytes() == original_context
    assert "return value" in projected.provider_prompt
    assert "return value" not in json.dumps(receipt)
    with public_replay_without_metadata():
        assert view.restore_source_input_view(provider_prompt=projected.provider_prompt, receipt=receipt,
            encoded=encoded, repository=root) == encoded.provider_prompt
    view.assert_current_source_input_view(encoded=encoded, repository=root, workspace=workspace, receipt=receipt)


def test_four_arms_have_same_ordered_selection_and_preserve_evidence(case):
    projections = [project(case, format=format, factor_ast=factored)
                   for format in ("raw", "compact") for factored in (False, True)]
    assert len({item.receipt["ordered_selection_sha256"] for item in projections}) == 1
    original = json.loads(case[2].provider_prompt)
    factored = json.loads(projections[1].provider_prompt)["semantic_transport"]
    changed_slots = {(row["capsule_index"], row["facet_index"]) for row in projections[1].receipt["views"]}
    for index, capsule in enumerate(factored["translated_semantic"]["capsules"]):
        if (index, 0) in changed_slots:
            capsule["metadata"]["facets"][0]["version_evidence"] = original["translated_semantic"]["capsules"][index]["metadata"]["facets"][0]["version_evidence"]
    assert factored == original


def test_historical_replay_after_both_sources_changed_and_workspace_removed(case):
    root, workspace, encoded = case
    projected = project(case, format="compact", factor_ast=True)
    (root / "program.py").write_text("# Later canonical source\nCHANGED = True\n")
    shutil.rmtree(workspace)
    with public_replay_without_metadata():
        replayed = view.replay_source_input_view_for_audit(encoded=encoded, repository=root, receipt=projected.receipt)
    assert replayed.provider_prompt == projected.provider_prompt
    assert replayed.receipt["freshness_checked"] is False
    assert replayed.receipt["workspace_checked"] is False
    error("current_workspace_projection_required", lambda: view.assert_current_source_input_view(
        encoded=encoded, repository=root, workspace=workspace, receipt=replayed.receipt))


@pytest.mark.parametrize("path", ["program.py", "instruction.txt"])
def test_stale_allocated_dependency_rejected(case, path):
    (case[1] / path).write_text("# changed independently\n")
    error("stale_workspace", lambda: project(case, format="raw"))


def test_last_moment_freshness_barrier(case):
    root, workspace, encoded = case
    projected = project(case, format="raw")
    (workspace / "instruction.txt").write_text("A different instruction")
    error("stale_workspace", lambda: view.assert_current_source_input_view(encoded=encoded,
        repository=root, workspace=workspace, receipt=projected.receipt))


@pytest.mark.parametrize("change", ["format", "selection", "count", "authority", "content", "hash", "unknown"])
def test_receipt_tampering_fails(case, change):
    projected = project(case, format="raw", factor_ast=True)
    receipt = deepcopy(projected.receipt)
    if change == "format": receipt["format"] = "compact"
    elif change == "selection": receipt["views"][0]["symbol"] = "foreign"
    elif change == "count": receipt["views_included"] = True
    elif change == "authority": receipt["proof_authority"] = True
    elif change == "content": receipt["views"][0]["content_sha256"] = "0" * 64
    elif change == "hash": receipt["provider_prompt_sha256"] = "0" * 64
    elif change == "unknown": receipt["extra"] = "untrusted"
    with pytest.raises(view.SourceInputViewError), public_replay_without_metadata():
        view.replay_source_input_view_for_audit(encoded=case[2], repository=case[0], receipt=receipt)


def test_provider_payload_tamper_and_marker_tamper_fail(case):
    projected = project(case, format="raw", factor_ast=True)
    changed = json.loads(projected.provider_prompt)
    changed["source_read_view"]["views"][0]["content"] = "SECRET = True"
    with public_replay_without_metadata():
        error("projected_input_mismatch", lambda: view.restore_source_input_view(provider_prompt=json.dumps(changed),
            receipt=projected.receipt, encoded=case[2], repository=case[0]))


def test_required_raw_program_is_never_removed(fixture_factory):
    case = fixture_factory(required_program=True)
    error("no_eligible_source_views", lambda: project(case, format="raw", factor_ast=True))


def test_read_only_assignment_exact_selection_and_unprofitable_fallback(fixture_factory):
    case = fixture_factory("MAX_SHAPES = 4\n", query="MAX_SHAPES", limit=1)
    raw = project(case, format="raw", factor_ast=True)
    assert raw.receipt["views"][0]["symbol"] == "MAX_SHAPES"
    assert raw.receipt["views"][0]["kind"] == "Assign"
    assert raw.receipt["views_included"] == 1
    error("no_compact_source_views", lambda: project(case, format="compact", factor_ast=True))
    with pytest.raises(rendering.SourceViewError):
        rendering.CodeSourceView(case[1]).replace("program.py", "MAX_SHAPES",
            raw.receipt["views"][0]["source_sha256"], "MAX_SHAPES = 9")


def test_selected_module_read_requires_exact_module_identity(fixture_factory):
    case = fixture_factory("# A long explanation remains canonical.\nCONSTANT = 2\ndef function(): return 1\n", query="program", limit=1)
    result = project(case, format="raw", factor_ast=True)
    assert result.receipt["views"][0]["producer_qualified_name"] == "program"
    assert result.receipt["views"][0]["symbol"] == "module"
    assert "def function" in result.provider_prompt


def test_named_module_assignment_does_not_alias_whole_file(fixture_factory):
    case = fixture_factory("module = 1\nUNRELATED = 'never widen this selection'\n", query="module", limit=1)
    error("no_eligible_source_views", lambda: project(case, format="raw"))


def test_pure_rendering_never_needs_original_file(tmp_path):
    source = b"def f(value):\n    # Keep original content off the rendered view.\n    return value + 1\n"
    compact = rendering.render_source_bytes("missing.py", source, "f", "compact")
    assert not (tmp_path / "missing.py").exists()
    assert compact["source_sha256"] == hashlib.sha256(source).hexdigest()
    assert "Keep original" not in compact["content"]
    assert rendering.render_source_bytes("p.py", b"VALUE = 1\n", "VALUE", "raw", read_only_assignment=True)["edit_supported"] is False


def test_invalid_policy_before_verification(case, monkeypatch):
    monkeypatch.setattr(view, "_verified", lambda *args: (_ for _ in ()).throw(AssertionError("verification reached")))
    error("invalid_projection_policy", lambda: project(case, format="unknown"))
    error("invalid_projection_policy", lambda: project(case, format="raw", factor_ast=1))
