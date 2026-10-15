"""Real native worktree integration for the bounded arithmetic code view."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import replace

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import WorktreeLifecycleStore
from ipfs_accelerate_py.agent_supervisor.runtime.python_code_view_worktree import (
    CodeViewWorktreeError,
    apply_code_view_patch,
    capture_worktree_code_view,
    prepare_code_view_expression,
    read_worktree_code_view,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.python_code_view import (
    PythonCodeViewError,
    build_python_code_view,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.worktree import (
    PatchScope,
    PatchValidationError,
    WorktreeFenceError,
    create_isolated_worktree,
)

SOURCE = (
    "# Preserve original names and audit note λ\n"
    "def adjusted_total(base_quantity, adjustment_quantity):\n"
    '    """Retain literal docstring λ."""\n'
    "    intermediate_quantity = base_quantity + adjustment_quantity\n"
    "    return intermediate_quantity * 2\n"
    "\n"
    "def untouched(some_quantity):\n"
    "    return some_quantity - 1\n"
)


def _git(root, *args):
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture
def native_tree(tmp_path, request):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "codeview-fixture@example.com")
    _git(repo, "config", "user.name", "Code view fixture")
    _git(repo, "config", "core.autocrlf", "false")
    original = getattr(request, "param", SOURCE)
    (repo / "calc.py").write_bytes(original.encode())
    (repo / "other.py").write_bytes(b"OTHER = 17\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "fixture")
    store = WorktreeLifecycleStore(repo_root=repo, store_dir=tmp_path / "lifecycle", lease_seconds=300)
    with create_isolated_worktree(repo_root=repo, worktree_path=tmp_path / "isolated",
            task_id="CODEVIEW-INTEGRATION", lifecycle_store=store) as isolated:
        scope = PatchScope(allowed_paths=("calc.py",), effect_paths=("calc.py",),
                           task_owned_paths=("calc.py",))
        yield isolated, scope, repo


def _owner(tree):
    isolated, scope, _ = tree
    return dict(worktree=isolated, scope=scope, lease_id=isolated.lease_id, fence=isolated.fence)


def _capture(tree):
    return capture_worktree_code_view(**_owner(tree), path="calc.py")


def _request(tree, view):
    data = view.to_dict()
    function = next(item for item in data["functions"] if item["name"] == "adjusted_total")
    slot = next(item for item in data["slots"]
                if item["function_id"] == function["function_id"] and item["kind"] == "return")
    alias = next(item["alias"] for item in function["aliases"] if item["original"] == "intermediate_quantity")
    return dict(**_owner(tree), view=view, expected_view_id=view.view_id,
                slot_id=slot["slot_id"], before_sha256=slot["before_sha256"],
                expression=f"({alias} * 3) + 1", visible_sources={"calc.py": data["original_source"]})


def _current(tree):
    return (tree[0].worktree_path / "calc.py").read_bytes()


def test_capture_and_read_preserve_native_bytes(native_tree):
    view = _capture(native_tree)
    packet = read_worktree_code_view(**_owner(native_tree), view=view, expected_view_id=view.view_id)
    assert packet == view.model_packet()
    assert "original_source" not in packet and "token_map" not in packet
    assert packet["compact_source"] != SOURCE
    assert "λ" in packet["compact_source"]
    assert _current(native_tree) == SOURCE.encode()
    assert (native_tree[2] / "calc.py").read_bytes() == SOURCE.encode()


@pytest.mark.parametrize("native_tree", [SOURCE, SOURCE.replace("\n", "\r\n"),
    SOURCE.removesuffix("\n"), SOURCE.replace("\n", "\r\n").removesuffix("\r\n")],
    indirect=True, ids=["lf", "crlf", "lf-no-final-newline", "crlf-no-final-newline"])
def test_native_edit_executes_and_preserves_surrounding_bytes(native_tree):
    original = _current(native_tree).decode()
    view = _capture(native_tree)
    prepared = prepare_code_view_expression(**_request(native_tree, view))
    expected = original.replace("return intermediate_quantity * 2", "return (intermediate_quantity * 3) + 1")
    assert prepared.native_edit.after_source == expected
    assert _current(native_tree) == original.encode(), "prepare must not mutate source"
    result = apply_code_view_patch(**_owner(native_tree), prepared=prepared,
                                  expected_view_id=view.view_id, visible_sources={"calc.py": original})
    assert result.applied, result.to_dict()
    assert _current(native_tree) == expected.encode()
    assert (native_tree[2] / "calc.py").read_bytes() == original.encode()
    assert (native_tree[0].worktree_path / "other.py").read_bytes() == b"OTHER = 17\n"
    # Only reconstructed native code is executed, in the ordinary isolated tree.
    execution = subprocess.run([sys.executable, "-B", "-c",
        "from calc import adjusted_total, untouched; "
        "assert adjusted_total(4, 3) == 22; assert adjusted_total(-5, 4) == -2; "
        "assert adjusted_total(0, 0) == 1; assert untouched(9) == 8"],
        cwd=native_tree[0].worktree_path, text=True, capture_output=True)
    assert execution.returncode == 0, execution.stderr
    receipt = prepared.receipt()
    assert receipt["candidate_only"] and receipt["lean_admits"] == 0
    assert not receipt["execution_authority"] and not receipt["completion_authority"]
    assert not receipt["semantic_acceptance_authority"]
    assert "before_source" not in receipt and "after_source" not in receipt


@pytest.mark.parametrize("visibility", [None, {}, {"calc.py": None}, "compact"])
def test_private_preimage_does_not_grant_native_visibility(native_tree, visibility):
    view = _capture(native_tree)
    request = _request(native_tree, view)
    request["visible_sources"] = {"calc.py": view.compact_source} if visibility == "compact" else visibility
    with pytest.raises((CodeViewWorktreeError, PatchValidationError)):
        prepare_code_view_expression(**request)
    assert _current(native_tree) == SOURCE.encode()


@pytest.mark.parametrize("change", ["outside_expression", "other_function", "selected_expression"])
def test_full_source_drift_is_refused_before_prepare(native_tree, change):
    view = _capture(native_tree)
    replacements = {"outside_expression": ("audit note λ", "audit note μ"),
                    "other_function": ("some_quantity - 1", "some_quantity - 2"),
                    "selected_expression": ("intermediate_quantity * 2", "intermediate_quantity * 4")}
    altered = SOURCE.replace(*replacements[change]).encode()
    (native_tree[0].worktree_path / "calc.py").write_bytes(altered)
    with pytest.raises(PythonCodeViewError, match="preimage"):
        prepare_code_view_expression(**_request(native_tree, view))
    assert _current(native_tree) == altered


def test_drift_after_prepare_refuses_apply(native_tree):
    view = _capture(native_tree)
    prepared = prepare_code_view_expression(**_request(native_tree, view))
    altered = SOURCE.replace("audit note λ", "audit note μ").encode()
    (native_tree[0].worktree_path / "calc.py").write_bytes(altered)
    with pytest.raises(PythonCodeViewError, match="preimage"):
        apply_code_view_patch(**_owner(native_tree), prepared=prepared,
                             expected_view_id=view.view_id, visible_sources={"calc.py": SOURCE})
    assert _current(native_tree) == altered


@pytest.mark.parametrize("field", ["expected_view_id", "slot_id", "before_sha256", "expression", "lease_id", "fence"])
def test_request_identity_scope_and_alias_refusals(native_tree, field):
    view = _capture(native_tree)
    request = _request(native_tree, view)
    request[field] = {"expected_view_id": "wrong", "slot_id": "wrong", "before_sha256": "0" * 64,
                      "expression": "new_identifier + 1", "lease_id": "wrong", "fence": request["fence"] + 1}[field]
    with pytest.raises((PythonCodeViewError, CodeViewWorktreeError, WorktreeFenceError)):
        prepare_code_view_expression(**request)
    assert _current(native_tree) == SOURCE.encode()


@pytest.mark.parametrize("binding", ["repository_id", "tree_id", "task_id"])
def test_cross_owner_view_refused(native_tree, binding):
    view = _capture(native_tree)
    bindings = dict(view.to_dict()["bindings"])
    bindings[binding] = "other-owner"
    foreign = build_python_code_view(source=SOURCE, **bindings)
    with pytest.raises(CodeViewWorktreeError, match="binding"):
        read_worktree_code_view(**_owner(native_tree), view=foreign, expected_view_id=foreign.view_id)
    assert _current(native_tree) == SOURCE.encode()


@pytest.mark.parametrize("path", ["other.py", "../outside.py", "/tmp/outside.py", ".git/config"])
def test_capture_checks_path_scope_before_read(native_tree, path):
    with pytest.raises(CodeViewWorktreeError):
        capture_worktree_code_view(**_owner(native_tree), path=path)
    assert _current(native_tree) == SOURCE.encode()


def test_symlink_source_refused(native_tree, tmp_path):
    path = native_tree[0].worktree_path / "calc.py"
    outside = tmp_path / "private.py"
    outside.write_bytes(SOURCE.encode())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError):
        _capture(native_tree)
    assert outside.read_bytes() == SOURCE.encode()


def test_prepared_diff_cannot_be_replaced(native_tree):
    view = _capture(native_tree)
    prepared = prepare_code_view_expression(**_request(native_tree, view))
    tampered = replace(prepared, patch_text=prepared.patch_text.replace("+ 1", "+ 999"))
    assert tampered != prepared
    with pytest.raises(CodeViewWorktreeError, match="differs"):
        apply_code_view_patch(**_owner(native_tree), prepared=tampered,
                             expected_view_id=view.view_id, visible_sources={"calc.py": SOURCE})
    assert _current(native_tree) == SOURCE.encode()


def test_complete_prompt_cost_probe(native_tree, record_property):
    started = time.perf_counter()
    view = _capture(native_tree)
    capture_ms = (time.perf_counter() - started) * 1000
    packet = read_worktree_code_view(**_owner(native_tree), view=view, expected_view_id=view.view_id)
    common = "Task: change adjusted_total to return three times the intermediate quantity plus one.\n"
    native_prompt = common + "Native source calc.py:\n" + SOURCE
    compact_prompt = common + (
        "Use the existing scoped aliases. Return an explicit slot_id, before_sha256, and "
        "single-line expression replacement. Native patch validation and acceptance remain required.\n"
    ) + json.dumps(packet, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    prepared = prepare_code_view_expression(**_request(native_tree, view))
    probe = {
        "schema": "supervisor-python-code-view-probe@1", "fixture_count": 1,
        "native_source_bytes": len(SOURCE.encode()), "compact_source_bytes": len(view.compact_source.encode()),
        "native_complete_prompt_bytes": len(native_prompt.encode()),
        "compact_complete_prompt_bytes": len(compact_prompt.encode()),
        "native_prompt_sha256": hashlib.sha256(native_prompt.encode()).hexdigest(),
        "compact_prompt_sha256": hashlib.sha256(compact_prompt.encode()).hexdigest(),
        "required_native_visibility_expansion_bytes": len(SOURCE.encode()),
        "private_native_patch_bytes": len(prepared.patch_text.encode()), "capture_elapsed_ms": capture_ms,
        "model_input_tokens": None, "total_session_token_savings": None,
        "provider_calls": 0, "weight_downloads": 0, "lean_admits": 0,
    }
    record_property("python_code_view_probe", json.dumps(probe, sort_keys=True))
    assert probe["compact_source_bytes"] < probe["native_source_bytes"]
    assert _current(native_tree) == SOURCE.encode()
