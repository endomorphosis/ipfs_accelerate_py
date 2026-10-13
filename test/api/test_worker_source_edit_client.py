"""Compact edit protocol and private local writes, without model calls."""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_edit_client as client
from ipfs_accelerate_py.agent_supervisor.runtime.reversible_source_view import build_source_view
from ipfs_accelerate_py.agent_supervisor.runtime.source_view_edits import map_source_view_edit
from test.api.test_numerical_obligation_client import descriptor as numerical_descriptor


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def descriptor(workspace):
    value = numerical_descriptor()
    for key in ("task_cid", "context_bundle_sha256", "context_receipt_sha256", "source_tree_id"):
        value.pop(key)
    value.update(schema=client.DESCRIPTOR_SCHEMA, workspace=str(workspace),
        worker_uid=os.geteuid(), owner_uid=os.geteuid() + 1,
        scope_cid="scope:fixture", semantic_root_cid="semantic:fixture",
        semantic_sha256="c" * 64, native_prompt_sha256="d" * 64, source_manifest_sha256="e" * 64,
        profile="python-comments@1", source_scope_sha256="f" * 64,
        view_policy_binding_sha256=None, source_edit_mode=client.MODE)
    return value


def request(edit, *, operation="prepare", **query):
    fields = dict(operation=operation, view_policy_binding_sha256=None, max_bytes=8192)
    fields.update(edit if operation == "prepare" else dict(proposal_sha256="2" * 64))
    fields.update(query)
    return dict(schema=client.REQUEST_SCHEMA, session_id="a" * 32, request_id="b" * 32, query=fields)


def prepared(edit, mapped, query):
    mapping = mapped.receipt
    return dict(schema=client.PREPARED_SCHEMA, status="prepared_validation_pending", proposal_sha256="2" * 64,
        prepare_request_sha256=sha(client.canonical_json(query)), path=edit["path"],
        source_generation=edit["expected_workspace_generation"], before_source_sha256=edit["expected_source_sha256"],
        before_view_sha256=edit["expected_view_sha256"], before_manifest_sha256=edit["expected_manifest_sha256"],
        **{key: mapping[key] for key in ("original_start", "original_end", "preimage_sha256", "preimage_bytes",
            "replacement_sha256", "replacement_bytes", "candidate_sha256", "candidate_bytes", "candidate_view_sha256",
            "candidate_view_bytes", "candidate_manifest_sha256")},
        view_policy_binding_sha256=None, validation_pending=True, owner_writes_performed=False,
        source_postimage_verified=False, proof_authority=False, execution_authority=False,
        publication_authority=False, completion_authority=False, mutation_authority=False,
        semantic_equivalence_claimed=False, source_obligations_discharged=0, provider_calls=0, prover_calls=0)


def response(query, page=None, reason=None):
    return dict(schema=client.RESPONSE_SCHEMA, session_id=query["session_id"], request_id=query["request_id"],
        ok=reason is None, page=page, reason_code=reason)


@pytest.fixture
def case(tmp_path, monkeypatch):
    root = tmp_path / "allocated"
    root.mkdir()
    monkeypatch.chdir(root)
    raw = b"# Authored public fixture\nfirst = 1  # ordinary note\nvalue = 'old'\n"
    path = root / "sample.py"
    path.write_bytes(raw)
    path.chmod(0o640)
    view = build_source_view(raw, profile="python-comments@1")
    start = view.view.index(b"'old'")
    edit = dict(path="sample.py", expected_workspace_generation="1" * 64,
        expected_source_sha256=sha(raw), expected_view_sha256=sha(view.view),
        expected_manifest_sha256=view.manifest_sha256, view_start=start, view_end=start + 5,
        expected_view_preimage="'old'", replacement="'new λ'")
    mapped = map_source_view_edit(original=raw, view=view.view, manifest=view.manifest,
        expected_manifest_sha256=view.manifest_sha256, view_start=start, view_end=start + 5,
        expected_view_preimage=b"'old'", replacement="'new λ'".encode())
    query = request(edit)
    page = prepared(edit, mapped, query)
    return dict(root=root, path=path, raw=raw, view=view, edit=edit, mapped=mapped, request=query,
        prepared=page, descriptor=descriptor(root))


def test_actual_mapper_bindings_roundtrip_without_original_or_replacement_echo(case):
    req, page = case["request"], case["prepared"]
    assert page["original_start"] != req["query"]["view_start"]
    value = response(req, page)
    assert client.validate_request(req) is req
    assert client.validate_response(value, request=req, descriptor=case["descriptor"]) is value
    raw = client.canonical_json(value)
    assert case["raw"] not in raw and b"ordinary note" not in raw
    assert case["edit"]["replacement"].encode() not in raw and b"'old'" not in raw


@pytest.mark.parametrize("field,value", [
    ("path", "../private"), ("path", "/private"), ("path", "a/.git/config"), ("path", "a\\b"),
    ("path", "./sample.py"), ("path", ""), ("expected_workspace_generation", None),
    ("expected_source_sha256", "A" * 64), ("expected_view_sha256", "x"), ("expected_manifest_sha256", True),
    ("view_start", True), ("view_start", -1), ("view_end", 1048577), ("view_end", 0),
    ("expected_view_preimage", "x"), ("expected_view_preimage", None), ("replacement", "'old'"),
    ("replacement", "\0"), ("replacement", "\ud800"), ("replacement", "λ" * 2049),
], ids=["parent-path", "absolute-path", "git-path", "backslash-path", "dot-path", "empty-path",
    "missing-generation", "uppercase-source-pin", "bad-view-pin", "boolean-manifest-pin",
    "boolean-start", "negative-start", "large-end", "backwards-end", "preimage-size", "preimage-type",
    "noop", "nul-replacement", "surrogate-replacement", "large-utf8-replacement"])
def test_closed_edit_refuses_unsafe_paths_types_extents_and_literals(case, field, value):
    edit = dict(case["edit"], **{field: value})
    with pytest.raises(ValueError):
        client.validate_edit(edit)


@pytest.mark.parametrize("mutation", ["extra", "wrong_schema", "session", "request", "query_extra",
    "operation", "bool_budget", "small_budget", "large_budget", "policy", "whole_frame", "confirm_extra"])
def test_request_envelope_and_full_serialized_budget_are_closed(case, mutation):
    query = deepcopy(case["request"])
    if mutation == "extra": query["replacement"] = "leak"
    elif mutation == "wrong_schema": query["schema"] = "unknown"
    elif mutation in {"session", "request"}: query[mutation + "_id"] = "a" * 31
    elif mutation == "query_extra": query["query"]["repository"] = "/owner"
    elif mutation == "operation": query["query"]["operation"] = "apply"
    elif mutation == "bool_budget": query["query"]["max_bytes"] = True
    elif mutation == "small_budget": query["query"]["max_bytes"] = 4095
    elif mutation == "large_budget": query["query"]["max_bytes"] = 8193
    elif mutation == "policy": query["query"]["view_policy_binding_sha256"] = "unknown"
    elif mutation == "whole_frame":
        query["query"].update(view_start=0, view_end=4096, expected_view_preimage="a" * 4096, replacement="b" * 4096)
    elif mutation == "confirm_extra": query["query"]["operation"] = "confirm"
    with pytest.raises(ValueError): client.validate_request(query)


@pytest.mark.parametrize("mutation", ["envelope_extra", "page_extra", "request", "session", "schema", "status",
    "path", "prepare_digest", "generation", "source", "view", "manifest", "original_start", "original_end",
    "preimage", "replacement", "candidate_bytes", "view_bytes", "policy", "validation", "owner_write",
    "postimage", "proof", "mutation", "zero_bool", "candidate_noop", "view_noop", "bad_digest",
    "bad_descriptor_mode", "descriptor_session", "descriptor_policy"])
def test_unbound_or_malformed_owner_response_refuses(case, mutation):
    req, page, selected = deepcopy(case["request"]), deepcopy(case["prepared"]), deepcopy(case["descriptor"])
    value = response(req, page)
    if mutation == "envelope_extra": value["original"] = "private"
    elif mutation == "page_extra": page["replacement"] = "private"
    elif mutation in {"request", "session"}: value[mutation + "_id"] = "f" * 32
    elif mutation == "schema": page["schema"] = client.CONFIRMED_SCHEMA
    elif mutation == "status": page["status"] = "complete"
    elif mutation == "path": page["path"] = "other.py"
    elif mutation == "prepare_digest": page["prepare_request_sha256"] = "f" * 64
    elif mutation == "generation": page["source_generation"] = "f" * 64
    elif mutation in {"source", "view", "manifest"}: page["before_" + mutation + "_sha256"] = "f" * 64
    elif mutation == "original_start": page["original_start"] = True
    elif mutation == "original_end": page["original_end"] += 1
    elif mutation in {"preimage", "replacement"}: page[mutation + "_sha256"] = "f" * 64
    elif mutation == "candidate_bytes": page["candidate_bytes"] = -1
    elif mutation == "view_bytes": page["candidate_view_bytes"] = page["candidate_bytes"] + 1
    elif mutation == "policy": page["view_policy_binding_sha256"] = "f" * 64
    elif mutation == "validation": page["validation_pending"] = False
    elif mutation == "owner_write": page["owner_writes_performed"] = True
    elif mutation == "postimage": page["source_postimage_verified"] = True
    elif mutation in {"proof", "mutation"}: page[mutation + "_authority"] = True
    elif mutation == "zero_bool": page["source_obligations_discharged"] = False
    elif mutation == "candidate_noop": page["candidate_sha256"] = page["before_source_sha256"]
    elif mutation == "view_noop": page["candidate_view_sha256"] = page["before_view_sha256"]
    elif mutation == "bad_digest": page["candidate_manifest_sha256"] = "x"
    elif mutation == "bad_descriptor_mode": selected["source_edit_mode"] = "disabled"
    elif mutation == "descriptor_session": selected["session_id"] = "f" * 32
    elif mutation == "descriptor_policy": selected["view_policy_binding_sha256"] = "f" * 64
    with pytest.raises(ValueError): client.validate_response(value, request=req, descriptor=selected)


def confirmed(page):
    return dict(page, schema=client.CONFIRMED_SCHEMA, status="applied_validation_pending",
        source_postimage_verified=True, workspace_generation="3" * 64, source_population_sha256="4" * 64,
        baseline_evidence_applicable=False)


@pytest.mark.parametrize("mutation", [None, "proposal", "old_generation", "baseline", "missing", "postimage"])
def test_confirmation_is_pending_bound_and_never_a_proof(case, mutation):
    query = request({}, operation="confirm")
    page = confirmed(case["prepared"])
    if mutation == "proposal": page["proposal_sha256"] = "f" * 64
    elif mutation == "old_generation": page["workspace_generation"] = page["source_generation"]
    elif mutation == "baseline": page["baseline_evidence_applicable"] = True
    elif mutation == "missing": page.pop("source_population_sha256")
    elif mutation == "postimage": page["source_postimage_verified"] = False
    if mutation is None:
        assert client.validate_response(response(query, page), request=query, descriptor=case["descriptor"])["ok"]
    else:
        with pytest.raises(ValueError): client.validate_response(response(query, page), request=query)


@pytest.mark.parametrize("reason", sorted(client.REASON_CODES))
def test_closed_owner_refusal_never_echoes_edit(case, reason):
    value = response(case["request"], reason=reason)
    assert client.validate_response(value, request=case["request"], descriptor=case["descriptor"]) == value
    value["reason_code"] = "exception including replacement"
    with pytest.raises(ValueError): client.validate_response(value, request=case["request"])


def rpc(case, monkeypatch, *, confirm_failure=None, before_write=None):
    calls, prepared_page = [], []
    monkeypatch.setattr(client.source, "load_descriptor", lambda *args, **kwargs: dict(case["descriptor"]))
    def exchange(path, selected, query):
        calls.append(query["query"]["operation"])
        if calls == ["prepare"]:
            page = prepared(case["edit"], case["mapped"], query)
            prepared_page.append(page)
            if before_write: before_write()
            result = response(query, page)
        else:
            assert calls == ["prepare", "confirm"]
            if confirm_failure == "transport": raise OSError("DO NOT PRINT private original")
            if confirm_failure == "refusal": result = response(query, reason="confirmation_refused")
            else:
                page = confirmed(prepared_page[0])
                if confirm_failure == "changed_map": page["original_start"] += 1; page["original_end"] += 1
                result = response(query, page)
        return client.validate_response(result, request=query, descriptor=selected)
    monkeypatch.setattr(client, "_exchange", exchange)
    return calls


def test_private_original_read_write_confirm_preserves_hidden_comment_and_mode(case, monkeypatch):
    calls = rpc(case, monkeypatch)
    output = client.edit_source("/routing", "f" * 64, edit=case["edit"])
    assert calls == ["prepare", "confirm"] and output["ok"]
    assert case["path"].read_bytes() == case["mapped"].candidate
    assert b"ordinary note" in case["path"].read_bytes()
    assert stat.S_IMODE(case["path"].stat().st_mode) == 0o640
    assert output["write_mode"] == "atomic_replace" and output["status"] == "applied_validation_pending"
    assert output["local_postimage_verified"] and output["owner_postimage_verified"]
    assert not output["atomic_application_claimed"] and not output["completion_authority"]
    assert b"'old'" not in client.canonical_json(output) and "new λ" not in client.canonical_json(output).decode()
    assert not list(case["root"].glob(".compact-edit-*"))


@pytest.mark.parametrize("mode", ["mapped-span@1", "mapped-span@2"])
@pytest.mark.parametrize("policy", [None, "9" * 64])
def test_descriptor_accepts_only_owner_selected_edit_modes_with_policy(case, mode, policy):
    selected = case["descriptor"] | {"source_edit_mode": mode, "view_policy_binding_sha256": policy}
    assert client.validate_descriptor(selected) is selected


def test_repeated_mode_two_client_invocations_use_fresh_source_mapping(case, monkeypatch):
    selected = case["descriptor"] | {"source_edit_mode": "mapped-span@2"}
    monkeypatch.setattr(client.source, "load_descriptor", lambda *args, **kwargs: dict(selected))
    state = {"generation": case["edit"]["expected_workspace_generation"], "proposal": None, "confirmed": 0}
    calls, observed_prepares = [], []
    def exchange(path, descriptor, query):
        operation = query["query"]["operation"]
        calls.append(operation)
        if operation == "prepare":
            assert state["proposal"] is None
            edit = {key: query["query"][key] for key in client.EDIT_FIELDS}
            raw = case["path"].read_bytes()
            view = build_source_view(raw, profile=descriptor["profile"])
            assert edit["expected_workspace_generation"] == state["generation"]
            assert edit["expected_source_sha256"] == sha(raw)
            assert edit["expected_view_sha256"] == sha(view.view)
            assert edit["expected_manifest_sha256"] == view.manifest_sha256
            mapped = map_source_view_edit(original=raw, view=view.view, manifest=view.manifest,
                expected_manifest_sha256=view.manifest_sha256, view_start=edit["view_start"],
                view_end=edit["view_end"], expected_view_preimage=edit["expected_view_preimage"].encode(),
                replacement=edit["replacement"].encode())
            page = prepared(edit, mapped, query)
            page["proposal_sha256"] = sha(client.canonical_json(query))
            state["proposal"] = (page, mapped.candidate)
            observed_prepares.append(edit)
        else:
            assert operation == "confirm" and state["proposal"] is not None
            page, candidate = state["proposal"]
            assert query["query"]["proposal_sha256"] == page["proposal_sha256"]
            assert case["path"].read_bytes() == candidate
            state["confirmed"] += 1
            state["generation"] = sha(str(state["confirmed"]).encode() + candidate)
            page = confirmed(page) | {"workspace_generation": state["generation"],
                "source_population_sha256": sha(client.canonical_json({"sample.py": sha(candidate)}))}
            state["proposal"] = None
        return client.validate_response(response(query, page), request=query, descriptor=descriptor)
    monkeypatch.setattr(client, "_exchange", exchange)
    first = client.edit_source("/routing", "f" * 64, edit=case["edit"])
    assert first["ok"] and case["path"].read_bytes() == case["mapped"].candidate
    # Simulate the fresh working read, retaining the real regenerated compact
    # view/map. This test authors owner RPCs; socket lifecycle is tested separately.
    raw = case["path"].read_bytes()
    view = build_source_view(raw, profile=selected["profile"])
    preimage, replacement = "'new λ'", "'final π'"
    start = view.view.index(preimage.encode())
    second_edit = dict(path="sample.py", expected_workspace_generation=first["workspace_generation"],
        expected_source_sha256=sha(raw), expected_view_sha256=sha(view.view),
        expected_manifest_sha256=view.manifest_sha256, view_start=start, view_end=start + len(preimage.encode()),
        expected_view_preimage=preimage, replacement=replacement)
    second = client.edit_source("/routing", "f" * 64, edit=second_edit)
    assert first["ok"] and second["ok"] and calls == ["prepare", "confirm", "prepare", "confirm"]
    assert state["confirmed"] == 2 and state["proposal"] is None
    assert case["path"].read_bytes() == case["raw"].replace(b"'old'", replacement.encode())
    assert stat.S_IMODE(case["path"].stat().st_mode) == 0o640
    for key in ("expected_workspace_generation", "expected_source_sha256", "expected_view_sha256",
                "expected_manifest_sha256"):
        assert observed_prepares[0][key] != observed_prepares[1][key]
    for result in (first, second):
        assert result["status"] == "applied_validation_pending" and not result["automatic_retry_performed"]
        assert not result["atomic_application_claimed"] and not result["completion_authority"]
        assert b"ordinary note" not in client.canonical_json(result)
        assert replacement.encode() not in client.canonical_json(result)
    assert first["workspace_generation"] != second["workspace_generation"]
    assert first["proposal_sha256"] != second["proposal_sha256"]


def test_owner_inode_write_policy_preserves_original_inode(case, monkeypatch):
    calls = rpc(case, monkeypatch)
    inode = case["path"].stat().st_ino
    # No chown/elevation: select existing foreign-owner branch while retaining
    # actual OS permissions. The container integration checks real UID separation.
    monkeypatch.setattr(client.os, "geteuid", lambda: case["descriptor"]["owner_uid"])
    output = client.edit_source("/routing", "f" * 64, edit=case["edit"])
    assert output["ok"] and output["write_mode"] == "preserved_owner_inode"
    assert calls == ["prepare", "confirm"] and case["path"].stat().st_ino == inode
    assert case["path"].read_bytes() == case["mapped"].candidate


@pytest.mark.parametrize("mutation", ["source_bytes", "symlink", "hardlink", "directory", "large", "parent_symlink"])
def test_local_source_drift_or_unsupported_file_refuses_without_writing(case, monkeypatch, mutation):
    path = case["path"]
    outside = case["root"].parent / "outside.py"
    outside.write_bytes(case["raw"])
    if mutation == "source_bytes": path.write_bytes(case["raw"] + b"# drift\n")
    elif mutation == "symlink": path.unlink(); path.symlink_to(outside)
    elif mutation == "hardlink": os.link(path, case["root"] / "same.py")
    elif mutation == "directory": path.unlink(); path.mkdir()
    elif mutation == "large": path.write_bytes(b"x" * (client.MAX_SOURCE_BYTES + 1))
    elif mutation == "parent_symlink":
        folder = case["root"] / "nested"; folder.symlink_to(case["root"], target_is_directory=True)
        case["edit"]["path"] = "nested/sample.py"
    initial = path.read_bytes() if path.is_file() else None
    calls = rpc(case, monkeypatch)
    output = client.edit_source("/routing", "f" * 64, edit=case["edit"])
    assert calls == ["prepare"] and not output["ok"] and output["status"] == "not_written"
    assert not output["write_attempted"]
    assert path.read_bytes() == initial if initial is not None else path.is_dir()
    assert outside.read_bytes() == case["raw"]


@pytest.mark.parametrize("mode", ["mapped-span@1", "mapped-span@2"])
@pytest.mark.parametrize("failure", ["transport", "refusal", "changed_map"])
def test_confirmation_failure_after_write_is_unknown_and_never_retries(case, monkeypatch, failure, mode):
    case["descriptor"]["source_edit_mode"] = mode
    calls = rpc(case, monkeypatch, confirm_failure=failure)
    output = client.edit_source("/routing", "f" * 64, edit=case["edit"])
    assert calls == ["prepare", "confirm"] and not output["ok"]
    assert output["status"] == "written_confirmation_unknown" and output["write_attempted"]
    assert output["local_postimage_verified"] and not output["automatic_retry_performed"]
    assert case["path"].read_bytes() == case["mapped"].candidate
    assert b"DO NOT PRINT" not in client.canonical_json(output)


def test_inplace_partial_write_failure_remains_unknown_without_rollback(case, monkeypatch):
    calls = rpc(case, monkeypatch)
    monkeypatch.setattr(client.os, "geteuid", lambda: case["descriptor"]["owner_uid"])
    write = client.os.write
    def interrupted(fd, value):
        write(fd, value[:8])
        raise OSError("private literal text")
    monkeypatch.setattr(client.os, "write", interrupted)
    output = client.edit_source("/routing", "f" * 64, edit=case["edit"])
    assert calls == ["prepare"] and output["status"] == "written_confirmation_unknown"
    assert output["write_attempted"] and not output["local_postimage_verified"]
    assert not output["automatic_retry_performed"] and b"private literal" not in client.canonical_json(output)


def test_disabled_owner_refuses_before_any_socket_or_source_write(case, monkeypatch):
    selected = case["descriptor"]
    selected["schema"] = client.source.DESCRIPTOR_SCHEMA
    for key in ("source_edit_mode", "source_scope_sha256", "view_policy_binding_sha256"):
        selected.pop(key)
    monkeypatch.setattr(client.source, "load_descriptor", lambda *args, **kwargs: selected)
    monkeypatch.setattr(client, "_exchange", lambda *args: pytest.fail("disabled owner contacted"))
    output = client.edit_source("/routing", "f" * 64, edit=case["edit"])
    assert output["status"] == "not_written" and not output["write_attempted"]
    assert case["path"].read_bytes() == case["raw"]


@pytest.mark.parametrize("raw", [b'{"path":"sample.py","path":"other.py"}', b'{"path":NaN}',
    b'{"replacement":"\xff"}', b" " * 8193, b"[]"],
    ids=["duplicate", "nonfinite", "non-utf8", "oversized", "non-object"])
def test_stdin_rejects_ambiguous_or_unbounded_json_without_content_echo(monkeypatch, capsys, raw):
    monkeypatch.setattr(client.sys, "stdin", io.TextIOWrapper(io.BytesIO(raw)))
    assert client.main(["--edit-file", "-", client.source.DESCRIPTOR_FLAG, "/routing",
        client.source.DIGEST_FLAG, "f" * 64]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "not_written" and result["reason_code"] == "invalid_input"
    assert set(result) == {"schema", "ok", "status", "reason_code", "automatic_retry_performed"}


def test_standalone_isolated_cli_needs_no_package_imports_or_provider(case):
    module = Path(client.__file__).resolve()
    result = subprocess.run([sys.executable, "-I", str(module), "--help"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and "--edit-file" in result.stdout
    assert not result.stderr


def test_file_input_rejects_symlink_and_whole_utf8_body_is_bounded(case):
    path = case["root"] / "edit.json"
    path.write_bytes(client.canonical_json(case["edit"]))
    assert client._edit_input(str(path)) == case["edit"]
    link = case["root"] / "edit-link.json"
    link.symlink_to(path)
    with pytest.raises(OSError): client._edit_input(str(link))


def test_guidance_exposes_native_commands_and_unknown_write_fallback():
    text = client.guidance("python-comments@1")
    prefix = "/opt/ipfs-supervisor/source/ipfs_accelerate_py/agent_supervisor/runtime/"
    assert prefix + "worker_source_client.py --path <path> --view compact --working --edit-ready" in text
    assert prefix + "worker_source_edit_client.py --edit-file <JSON-file>" in text
    assert "expected_manifest_sha256" in text and "written_confirmation_unknown" in text
    assert "One mapped compact span edit" in text and "never replay the old edit automatically" in text
    assert "Keep request JSON files outside the checkout" in text and "or use stdin" in text
    assert "Selected source profile: python-comments@1." in text


@pytest.mark.parametrize("profile,expected_sha256", [
    ("identity@1", "6b075852d57784a5520ac0226f6c0a9b12a8d27fe57cd305fb10edf0332e4fbd"),
    ("python-comments@1", "1d5f979fcfa60c5d1fbab2f82987a5f014356474cd2c6449bf75eb0bdd69ea30"),
    ("python-whitespace@1", "0496019305188ef136da6dbbfdd5287029503f8f2cc557e6e5476a7565316708"),
])
def test_default_and_explicit_v1_guidance_preserve_published_bytes(profile, expected_sha256):
    assert sha(client.guidance(profile).encode()) == expected_sha256
    assert client.guidance(profile, mode="mapped-span@1") == client.guidance(profile)


def test_repeated_guidance_requires_fresh_pins_bounded_session_and_no_unknown_retry():
    text = client.guidance("python-comments@1", mode="mapped-span@2")
    assert "Sequential mapped compact span edits" in text and "one outstanding proposal" in text
    assert "at most 16 successful preparations" in text and "Before each edit, read a fresh" in text
    assert "--working --edit-ready" in text and "fresh generation/source/view/manifest pins" in text
    assert "written_confirmation_unknown, mapped editing stops for this session" in text
    assert "never replay the old edit automatically" in text and "ordinary original-source editing" in text
    assert "Keep request JSON files outside the checkout" in text and "or use stdin" in text
    assert "Selected source profile: python-comments@1." in text and "validation-pending" in text


@pytest.mark.parametrize("mode", ["disabled", "mapped-span@3", None])
def test_guidance_refuses_unsupported_edit_mode(mode):
    with pytest.raises(ValueError):
        client.guidance(mode=mode)
