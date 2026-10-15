"""Packed public views and edit rows through real native worktree gates."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import WorktreeLifecycleStore
from ipfs_accelerate_py.agent_supervisor.runtime.python_code_view_wire_worktree import (
    CodeViewWireWorktreeError,
    build_python_code_view_wire_edit,
    expand_python_code_view_wire_edit,
    parse_python_code_view_wire_edit,
    prepare_worktree_code_view_wire_expression,
    read_worktree_code_view_wire,
    restore_worktree_code_view_wire,
)
from ipfs_accelerate_py.agent_supervisor.runtime.python_code_view_worktree import (
    apply_code_view_patch,
    capture_worktree_code_view,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.python_code_view import (
    MAX_EXPRESSION_BYTES,
    PythonCodeViewError,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.python_code_view_wire import (
    pack_python_code_view_packet,
    unpack_python_code_view_packet,
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


@pytest.fixture
def native_tree(tmp_path, request):
    source = getattr(request, "param", SOURCE)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "wire-fixture@example.com")
    _git(repo, "config", "user.name", "Code view wire fixture")
    _git(repo, "config", "core.autocrlf", "false")
    (repo / "calc.py").write_bytes(source.encode())
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "fixture")
    store = WorktreeLifecycleStore(repo_root=repo, store_dir=tmp_path / "lifecycle", lease_seconds=300)
    with create_isolated_worktree(repo_root=repo, worktree_path=tmp_path / "isolated",
            task_id="CODEVIEW-WIRE-INTEGRATION", lifecycle_store=store) as worktree:
        scope = PatchScope(allowed_paths=("calc.py",), effect_paths=("calc.py",), task_owned_paths=("calc.py",))
        yield worktree, scope, repo, source


def _owner(tree):
    worktree, scope, _, _ = tree
    return dict(worktree=worktree, scope=scope, lease_id=worktree.lease_id, fence=worktree.fence)


def _capture(tree):
    return capture_worktree_code_view(**_owner(tree), path="calc.py")


def _read_args(tree, view):
    return dict(**_owner(tree), view=view, expected_view_id=view.view_id)


def _request(view):
    data = view.to_dict()
    function = data["functions"][0]
    slot = next(item for item in data["slots"]
                if item["kind"] == "return" and item["function_id"] == function["function_id"])
    last_local = [item for item in function["aliases"] if item["kind"] == "local"][-1]
    wire = build_python_code_view_wire_edit(view_id=view.view_id, slot_id=slot["slot_id"],
        before_sha256=slot["before_sha256"], expression=f"({last_local['alias']} * 3) + 1")
    return wire, last_local["original"]


def _current(tree):
    return (tree[0].worktree_path / "calc.py").read_bytes()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def test_current_wire_restores_entire_public_packet(native_tree):
    view = _capture(native_tree)
    wire, stats = read_worktree_code_view_wire(**_read_args(native_tree, view))
    packet = restore_worktree_code_view_wire(**_read_args(native_tree, view), wire_packet=wire)
    assert packet == view.model_packet() == unpack_python_code_view_packet(wire)
    assert stats and len(_json(wire).encode()) < len(_json(packet).encode())
    assert not any(packet["authority"].values())
    assert _current(native_tree) == native_tree[3].encode()
    assert (native_tree[2] / "calc.py").read_bytes() == native_tree[3].encode()


@pytest.mark.parametrize("native_tree", [SOURCE, SOURCE.replace("\n", "\r\n"),
    SOURCE.removesuffix("\n"), SOURCE.replace("\n", "\r\n").removesuffix("\r\n")],
    indirect=True, ids=["lf", "crlf", "lf-no-final-newline", "crlf-no-final-newline"])
def test_wire_json_edit_reconstructs_and_executes_native_source(native_tree):
    view = _capture(native_tree)
    wire, _ = read_worktree_code_view_wire(**_read_args(native_tree, view))
    restored = restore_worktree_code_view_wire(**_read_args(native_tree, view), wire_packet=wire)
    request, original_local = _request(view)
    expanded = expand_python_code_view_wire_edit(request)
    assert expanded["view_id"] == restored["view_id"]
    assert request["edit"][2] == expanded["before_sha256"]
    prepared = prepare_worktree_code_view_wire_expression(**_read_args(native_tree, view),
        wire_request=_json(request), visible_sources={"calc.py": native_tree[3]})
    assert _current(native_tree) == native_tree[3].encode()
    expected = native_tree[3].replace(f"return {original_local} * 2", f"return ({original_local} * 3) + 1", 1)
    assert prepared.native_edit.after_source == expected
    result = apply_code_view_patch(**_owner(native_tree), prepared=prepared,
        expected_view_id=view.view_id, visible_sources={"calc.py": native_tree[3]})
    assert result.applied, result.to_dict()
    assert _current(native_tree) == expected.encode()
    assert (native_tree[2] / "calc.py").read_bytes() == native_tree[3].encode()
    execution = subprocess.run([sys.executable, "-B", "-c",
        "from calc import adjusted_total, untouched; assert adjusted_total(4,3)==22; "
        "assert adjusted_total(-5,4)==-2; assert adjusted_total(0,0)==1; assert untouched(9)==8"],
        cwd=native_tree[0].worktree_path, text=True, capture_output=True)
    assert execution.returncode == 0, execution.stderr
    assert prepared.receipt()["lean_admits"] == 0


@pytest.mark.parametrize("field", ["compact_source", "task_id", "view_id"])
def test_well_shaped_wire_is_not_authenticated_by_syntax(native_tree, field):
    view = _capture(native_tree)
    packet = view.model_packet()
    if field == "task_id":
        packet["bindings"]["task_id"] = "other-task"
    elif field == "view_id":
        # A different real content ID retains valid syntax but lacks this binding.
        other = capture_worktree_code_view(**_owner(native_tree), path="calc.py")
        from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json
        packet["view_id"] = cid_for_dag_json({"different_view": other.view_id})
    else:
        packet["compact_source"] = packet["compact_source"].replace("*2", "*7")
    forged, _ = pack_python_code_view_packet(packet)
    with pytest.raises(CodeViewWireWorktreeError, match="differs"):
        restore_worktree_code_view_wire(**_read_args(native_tree, view), wire_packet=forged)
    assert _current(native_tree) == native_tree[3].encode()


@pytest.mark.parametrize("column", [0, 1, 2])
def test_full_edit_identities_still_bind_native_request(native_tree, column):
    view = _capture(native_tree)
    request, _ = _request(view)
    request["edit"][column] = ["other-view", "s999", "0" * 64][column]
    with pytest.raises((CodeViewWireWorktreeError, PythonCodeViewError)):
        prepare_worktree_code_view_wire_expression(**_read_args(native_tree, view),
            wire_request=request, visible_sources={"calc.py": native_tree[3]})
    assert _current(native_tree) == native_tree[3].encode()


@pytest.mark.parametrize("visibility", [None, {}, {"calc.py": None}, "wire"])
def test_packed_wire_cannot_supply_native_visibility(native_tree, visibility):
    view = _capture(native_tree)
    request, _ = _request(view)
    wire, _ = read_worktree_code_view_wire(**_read_args(native_tree, view))
    if visibility == "wire":
        visibility = {"calc.py": _json(wire)}
    with pytest.raises((ValueError, PatchValidationError)):
        prepare_worktree_code_view_wire_expression(**_read_args(native_tree, view),
            wire_request=request, visible_sources=visibility)
    assert _current(native_tree) == native_tree[3].encode()


@pytest.mark.parametrize("operation", ["read", "restore", "prepare"])
def test_stale_native_source_refuses_transport_operations(native_tree, operation):
    view = _capture(native_tree)
    wire, _ = read_worktree_code_view_wire(**_read_args(native_tree, view))
    request, _ = _request(view)
    altered = native_tree[3].replace("audit note λ", "audit note μ").encode()
    (native_tree[0].worktree_path / "calc.py").write_bytes(altered)
    with pytest.raises(PythonCodeViewError, match="preimage"):
        if operation == "read":
            read_worktree_code_view_wire(**_read_args(native_tree, view))
        elif operation == "restore":
            restore_worktree_code_view_wire(**_read_args(native_tree, view), wire_packet=wire)
        else:
            prepare_worktree_code_view_wire_expression(**_read_args(native_tree, view),
                wire_request=request, visible_sources={"calc.py": native_tree[3]})
    assert _current(native_tree) == altered


@pytest.mark.parametrize("binding", ["lease_id", "fence"])
def test_wire_read_retains_live_fence_check(native_tree, binding):
    view = _capture(native_tree)
    args = _read_args(native_tree, view)
    args[binding] = "wrong-owner" if binding == "lease_id" else args["fence"] + 1
    with pytest.raises(WorktreeFenceError):
        read_worktree_code_view_wire(**args)
    assert _current(native_tree) == native_tree[3].encode()


@pytest.mark.parametrize("fault", ["duplicate_key", "unknown_field", "boolean_id", "long_expression", "newline"])
def test_raw_edit_json_is_strict_before_native_mutation(native_tree, fault):
    view = _capture(native_tree)
    request, _ = _request(view)
    if fault == "unknown_field":
        request["extra"] = True
    elif fault == "boolean_id":
        request["edit"][0] = True
    elif fault == "long_expression":
        request["edit"][3] = "x" * (MAX_EXPRESSION_BYTES + 1)
    elif fault == "newline":
        request["edit"][3] = "a\n+1"
    text = _json(request)
    if fault == "duplicate_key":
        text = '{"schema":"discarded",' + text[1:]
    with pytest.raises(ValueError):
        parse_python_code_view_wire_edit(text)
    with pytest.raises(ValueError):
        prepare_worktree_code_view_wire_expression(**_read_args(native_tree, view),
            wire_request=text, visible_sources={"calc.py": native_tree[3]})
    assert _current(native_tree) == native_tree[3].encode()


def _scaled_source(function_count):
    # Authored arithmetic stress fixtures, not a claim about arbitrary modules.
    lines = ["# Descriptive quantities and literal audit note λ\n"]
    for index in range(function_count):
        base, adjustment = "base_quantity_total_value", "adjustment_quantity_total_value"
        lines.append(f"def adjusted_total_{index}({base}, {adjustment}):\n")
        lines.append('    """Retain literal semantic names λ."""\n')
        previous = None
        for stage in range(6):
            local = f"intermediate_quantity_stage_{stage}"
            expression = " + ".join([base, adjustment] * 6)
            if previous is not None:
                expression += " + " + previous
            lines.append(f"    {local} = {expression}\n")
            previous = local
        lines.append(f"    return {previous} * 2\n\n")
    return "".join(lines)


WIRE_GUIDANCE = (
    "Wire@1 fixed rows: bindings[path,repo,tree,task]; functions[id,name,aliases[original,alias,kind]] "
    "(alias kind 0=parameter,1=local); slots[id,function-index,kind,sha256,editable,refusal] "
    "(slot kind 0=assignment,1=return). Names/source are literal; authority stays false. "
    "Return {schema:supervisor-python-code-view-wire-edit@1,edit:[view_id,slot_id,before_sha256,expression]} "
    "using scoped aliases. Native source visibility, patch validation and acceptance remain required.\n"
)


@pytest.mark.parametrize("native_tree", [SOURCE, _scaled_source(1), _scaled_source(8)],
    indirect=True, ids=["tiny", "repeated-expressions", "eight-functions"])
def test_complete_wire_messages_and_native_edit_probe(native_tree, record_property):
    view = _capture(native_tree)
    packet = view.model_packet()
    started = time.perf_counter()
    wire, _ = pack_python_code_view_packet(packet)
    packing_ms = (time.perf_counter() - started) * 1000
    assert unpack_python_code_view_packet(wire) == packet
    request, original_local = _request(view)
    common = "Task: replace the first function's return with three times its last intermediate quantity plus one.\n"
    native = common + "Native source calc.py:\n" + native_tree[3]
    unpacked = common + (
        "Use scoped aliases. Return an explicit slot_id, before_sha256 and expression replacement. "
        "Native source visibility, patch validation and acceptance remain required.\n"
    ) + _json(packet)
    packed = common + WIRE_GUIDANCE + _json(wire)
    prepared = prepare_worktree_code_view_wire_expression(**_read_args(native_tree, view),
        wire_request=_json(request), visible_sources={"calc.py": native_tree[3]})
    expected = native_tree[3].replace(f"return {original_local} * 2", f"return ({original_local} * 3) + 1", 1)
    assert prepared.native_edit.after_source == expected
    result = apply_code_view_patch(**_owner(native_tree), prepared=prepared,
        expected_view_id=view.view_id, visible_sources={"calc.py": native_tree[3]})
    assert result.applied, result.to_dict()
    assert _current(native_tree) == expected.encode()
    assert (native_tree[2] / "calc.py").read_bytes() == native_tree[3].encode()
    name = packet["functions"][0]["name"]
    expected_positive, expected_negative = (22, -2) if native_tree[3] == SOURCE else (757, -107)
    execution = subprocess.run([sys.executable, "-B", "-c",
        f"from calc import {name}; assert {name}(4,3)=={expected_positive}; "
        f"assert {name}(-5,4)=={expected_negative}; assert {name}(0,0)==1"],
        cwd=native_tree[0].worktree_path, text=True, capture_output=True)
    assert execution.returncode == 0, execution.stderr
    probe = {"schema": "supervisor-python-code-view-wire-probe@1",
        "fixture": "tiny" if native_tree[3] == SOURCE else ("repeated-expressions" if len(packet["functions"]) == 1 else "eight-functions"),
        "function_count": len(packet["functions"]), "native_source_bytes": len(native_tree[3].encode()),
        "compact_source_bytes": len(view.compact_source.encode()),
        "unpacked_public_packet_bytes": len(_json(packet).encode()), "packed_public_packet_bytes": len(_json(wire).encode()),
        "native_complete_message_bytes": len(native.encode()), "unpacked_complete_message_bytes": len(unpacked.encode()),
        "packed_complete_message_bytes": len(packed.encode()), "wire_edit_message_bytes": len(_json(request).encode()),
        "required_native_visibility_expansion_bytes": len(native_tree[3].encode()),
        "native_message_sha256": hashlib.sha256(native.encode()).hexdigest(),
        "unpacked_message_sha256": hashlib.sha256(unpacked.encode()).hexdigest(),
        "packed_message_sha256": hashlib.sha256(packed.encode()).hexdigest(),
        "packing_elapsed_ms": packing_ms, "native_fixture_execution_passed": True,
        "model_input_tokens": None, "total_session_token_savings": None,
        "provider_calls": 0, "weight_downloads": 0, "lean_admits": 0}
    record_property("python_code_view_wire_probe", _json(probe))
    assert probe["packed_public_packet_bytes"] < probe["unpacked_public_packet_bytes"]
