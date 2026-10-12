"""Actual DuckDB/Unix-owner qualification of bounded current source reads."""

import base64
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import current_source_lookup as lookup
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_inventory as sources
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import FakeQuackTransport, build_server
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import OwnerLiveness
from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner as typed
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import canonical_json_bytes, content_identity
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import open_duckdb_connection
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes, cid_for_structured
from test.api.causal_federation.test_bootstrap_runtime import _capability, _migrate
from test.api.test_repository_source_inventory import _repository, _retain, _git, _commit_request, _chunk_cid


DEFAULT_SOURCE = "TEXT = 'caf\u00e9'\r\n# authored exact source\r\nVALUE = 42\r\n".encode()


@contextmanager
def _case(tmp_path, *, files=None, allowed=None, max_response_bytes=16384, max_slice_bytes=4096,
          executable=None, source_root=None, request=None):
    files = files or {b"module.py": DEFAULT_SOURCE, b"other.py": b"OTHER = 0\n"}
    root = source_root or _repository(tmp_path, files)
    if executable is not None:
        (root / executable).chmod(0o755)
    source = _retain(root, tmp_path, request=request,
                     limits=sources.SourceInventoryLimits(frame_bytes=4096))
    selected = allowed if allowed is not None else (b"module.py".hex(),)
    body = {"task_id": source.request.task_id, "title": "Read authored inert source",
            "repository_id": source.request.repository_id, "read_paths": list(selected)}
    task_cid, revision = content_identity({"authored_task": body}), 3
    server = build_server(database_path=tmp_path / "control.duckdb", state_dir=tmp_path / "owner",
        repository_id=source.request.repository_id, store_id="current-source-tests",
        transport=FakeQuackTransport(), capability_probe=_capability, migrate=_migrate,
        connection_factory=open_duckdb_connection, owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
        typed_command_socket_path=tmp_path / "source.sock")
    server.start()
    clients = []
    try:
        connection, gateway = server._connection, server._command_gateway
        connection.execute("INSERT INTO goals (goal_cid,goal_alias,objective_id,parent_goal_cid,ordinal,title,status,created_at,updated_at,revision,body_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ["goal:source-lookup", "SOURCE-GOAL", "objective:source-lookup", "", 1, "Source lookup", "open",
             "1970-01-01T00:00:00Z", "1970-01-01T00:00:00Z", 1, "{}"])
        connection.execute("INSERT INTO tasks (task_cid,task_alias,goal_cid,plan_cid,objective_id,ordinal,status,revision,priority,created_at,updated_at,identity_json,body_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [task_cid, "SOURCE-TASK", "goal:source-lookup", "", "objective:source-lookup", 1, "ready", revision, "P0",
             "1970-01-01T00:00:00Z", "1970-01-01T00:00:00Z", "{}", canonical_json_bytes(body).decode()])
        config = dict(repository=root, cas=source.cas, expected_inventory_cid=source.ref.inventory_cid,
            request=source.request, limits=source.limits, task_cid=task_cid, task_revision=revision,
            expected_task_body_cid=content_identity(body), context_cid=cid_for_structured({"authored_context": task_cid}),
            allowed_paths=tuple(sorted(selected)), max_response_bytes=max_response_bytes, max_slice_bytes=max_slice_bytes)
        binding = server.bind_current_source_lookup(**config)
        case = SimpleNamespace(root=root, source=source, request=source.request, cas=source.cas,
            limits=source.limits, files=files, server=server, gateway=gateway, connection=connection,
            task_cid=task_cid, revision=revision, body=body, config=config, binding=binding,
            service=gateway._current_source_lookup, clients=clients)
        case.client, case.token, case.grant = _client(case)
        yield case
    finally:
        for client in clients:
            client.close()
        server.stop()


def _client(case, *, scopes=None):
    scopes = scopes or {"task_cid": case.task_cid, "repository_id": case.request.repository_id,
                        "tree_id": case.request.expected_tree}
    birth = typed.kernel_process_birth_id()
    token, grant = case.server.issue_typed_client_grant_record(client_id="source-reader-" + str(len(case.clients)),
        process_birth_id=birth, peer_pid=os.getpid(), allowed_operations=("source.get",), entity_scopes=scopes)
    client = typed.TypedStateOwnerConnection(socket_path=case.server.typed_command_socket_path(), token=token,
        client_id=grant.client_id, process_birth_id=birth, store_id=case.gateway.store_id, timeout_seconds=3)
    case.clients.append(client)
    return client, token, grant


@pytest.fixture
def current(tmp_path):
    with _case(tmp_path) as case:
        yield case


def _payload(case, *, name=b"module.py", offset=0, maximum=None, continuation=None):
    return dict(schema=lookup.REQUEST_SCHEMA, binding_cid=case.binding["binding_cid"], raw_path_hex=name.hex(),
        source_cid=cid_for_bytes(case.files[name]), offset=offset,
        max_bytes=maximum if maximum is not None else case.binding["max_slice_bytes"], continuation=continuation)


def _data(response):
    result = response["result"]
    return result["content"].encode() if result["encoding"] == "utf-8" else base64.b64decode(result["content"], validate=True)


def _assert_response(case, payload, response):
    assert set(response) == {"schema", "request_id", "ok", "result"} and response["ok"] is True
    result, raw = response["result"], _data(response)
    assert result["response_bytes"] == len(canonical_json_bytes(response)) + 1 <= case.binding["max_response_bytes"]
    assert result["response_wire_bytes"] == result["response_bytes"] + 3
    assert result["offset"] == payload["offset"]
    assert result["end_offset"] == result["offset"] + len(raw)
    assert result["source_cid"] == payload["source_cid"] and result["slice_cid"] == cid_for_bytes(raw)
    assert result["binding"]["task_cid"] == case.task_cid
    assert result["caller_grant_id"] == case.grant.grant_id
    assert result["working_source_verified"] is True and result["read_only"] is True
    assert all(result[key] is False for key in ("repository_complete", "completion_authority", "proof_authority", "edit_authority", "omission_authority"))
    return result, raw


def test_actual_scoped_owner_returns_exact_source_without_sql_authority(current, monkeypatch):
    monkeypatch.setattr(ImmutableCAS, "__init__", lambda *a, **k: pytest.fail("lookup created a new CAS"))
    payload = _payload(current)
    result, raw = _assert_response(current, payload, current.client.source_get(payload))
    assert raw == DEFAULT_SOURCE and result["source_complete"] is True
    assert result["next_continuation"] is None and result["unresolved_frontier"] == []
    assert "allowed_paths" not in current.binding and "source_limits" not in current.binding
    assert current.binding["allowed_path_count"] == 1
    with pytest.raises(typed.TypedStateOwnerError):
        current.client.execute_operation("load_store_generation")
    with pytest.raises(typed.TypedStateOwnerError):
        current.client.execute("SELECT 1")


@pytest.mark.parametrize("data", ['"\\\n\t\u03bb' * 600, bytes(range(256)) * 12], ids=["escaped-utf8", "binary"])
def test_complete_serialized_envelope_limits_and_continuations_cover_every_byte(tmp_path, data):
    data = data.encode() if isinstance(data, str) else data
    with _case(tmp_path, files={b"module.py": data}, max_response_bytes=4096, max_slice_bytes=4096) as case:
        payload, chunks, ranges, responses = _payload(case), [], [], []
        while True:
            response = case.client.source_get(payload)
            result, raw = _assert_response(case, payload, response)
            chunks.append(raw)
            ranges.append((result["offset"], result["end_offset"]))
            responses.append(response)
            if result["source_complete"]:
                break
            assert result["next_continuation"] is not None and result["unresolved_frontier"]
            payload = _payload(case, offset=result["end_offset"], continuation=result["next_continuation"])
        assert b"".join(chunks) == data
        assert ranges[0][0] == 0 and ranges[-1][1] == len(data)
        assert all(first[1] == second[0] for first, second in zip(ranges, ranges[1:]))
        assert len(responses) > 1 and ranges[0][1] < min(len(data), 4096)


@pytest.mark.parametrize("mutation", ["selected", "omitted", "assume", "head"])
def test_stale_dirty_or_hidden_overlay_refuses_current_source(current, mutation):
    if mutation == "assume":
        _git(current.root, "update-index", "--assume-unchanged", "module.py")
    name = "other.py" if mutation == "omitted" else "module.py"
    (current.root / name).write_bytes(b"VALUE = 'changed overlay'\n")
    if mutation == "head":
        _commit_request(current.root)
    with pytest.raises((typed.TypedStateOwnerError, ValueError)):
        current.client.source_get(_payload(current))


@pytest.mark.parametrize("field", ["revision", "body"])
def test_native_task_row_drift_refuses_unchanged_repository(current, monkeypatch, field):
    if field == "revision":
        original, calls = ImmutableCAS.get_bytes, []
        def mutate(self, cid):
            data = original(self, cid)
            if self is current.cas and not calls:
                calls.append(cid)
                current.connection.execute("UPDATE tasks SET revision=revision+1 WHERE task_cid=?", [current.task_cid])
            return data
        monkeypatch.setattr(ImmutableCAS, "get_bytes", mutate)
    else:
        current.connection.execute("UPDATE tasks SET body_json=? WHERE task_cid=?", ['{"changed":"native task"}', current.task_cid])
    with pytest.raises((typed.TypedStateOwnerError, ValueError)):
        current.client.source_get(_payload(current))


@pytest.mark.parametrize("change", ["binding", "source", "outside", "bool-limit", "missing-cursor"])
def test_foreign_identity_scope_or_invalid_bound_refuses_before_raw_reads(current, monkeypatch, change):
    payload = _payload(current)
    if change == "binding": payload["binding_cid"] = cid_for_structured({"foreign": "binding"})
    elif change == "source": payload["source_cid"] = cid_for_bytes(b"foreign source")
    elif change == "outside": payload = _payload(current, name=b"other.py")
    elif change == "bool-limit": payload["max_bytes"] = True
    else: payload["offset"] = 1
    if change != "source":
        monkeypatch.setattr(ImmutableCAS, "get_bytes", lambda *a, **k: pytest.fail("refused scope read source bytes"))
    with pytest.raises((typed.TypedStateOwnerError, ValueError)):
        current.client.source_get(payload)


@pytest.mark.parametrize("scope", ["task_cid", "tree_id"])
def test_wrong_native_grant_scope_cannot_be_issued(current, scope):
    scopes = {"task_cid": current.task_cid, "repository_id": current.request.repository_id,
              "tree_id": current.request.expected_tree}
    scopes[scope] = "foreign:" + scope
    with pytest.raises(typed.TypedStateOwnerError):
        _client(current, scopes=scopes)


def test_revoked_source_only_grant_cannot_escape_after_verified_bytes_are_read(current, monkeypatch):
    original, calls = ImmutableCAS.get_bytes, []
    def revoke(self, cid):
        data = original(self, cid)
        if self is current.cas and not calls:
            calls.append(cid)
            current.gateway.revoke_grant(current.grant.grant_id)
        return data
    monkeypatch.setattr(ImmutableCAS, "get_bytes", revoke)
    with pytest.raises(typed.TypedStateOwnerError):
        current.client.source_get(_payload(current))


def test_owner_issued_cursor_cannot_be_forged_consumed_twice_or_shared_with_new_grant(current):
    initial = _payload(current, maximum=1)
    manufactured = cid_for_structured(dict(schema="supervisor-source-continuation@1",
        binding_cid=initial["binding_cid"], caller_grant_id=current.grant.grant_id,
        raw_path_hex=initial["raw_path_hex"], source_cid=initial["source_cid"], offset=1, max_bytes=1))
    with pytest.raises(typed.TypedStateOwnerError):
        current.client.source_get(_payload(current, maximum=1, offset=1, continuation=manufactured))
    first = current.client.source_get(initial)["result"]
    next_payload = _payload(current, maximum=1, offset=first["end_offset"], continuation=first["next_continuation"])
    foreign, _, _ = _client(current)
    with pytest.raises(typed.TypedStateOwnerError):
        foreign.source_get(next_payload)
    current.client.source_get(next_payload)
    with pytest.raises(typed.TypedStateOwnerError):
        current.client.source_get(next_payload)


def test_non_utf8_raw_name_never_aliases_malformed_display_label(tmp_path):
    raw = b"invalid-\xff.txt"
    alias = b"@malformed-path/" + raw.hex().encode()
    files = {raw: b"raw byte filename source\n", alias: b"different literal display-like path\n"}
    root = _repository(tmp_path, files)
    first = _commit_request(root)
    os.symlink(raw, os.fsencode(root) + b"/linked.py")
    (root / "nested").mkdir()
    _git(root, "update-index", "--add", "--cacheinfo", "160000", first.expected_commit, "nested")
    request = _commit_request(root)
    scope = tuple(sorted(name.hex() for name in (*files, b"linked.py", b"nested")))
    with _case(tmp_path, files={**files, b"linked.py": raw, b"nested": b""}, allowed=scope,
               source_root=root, request=request) as case:
        for name, expected in files.items():
            response = case.client.source_get(_payload(case, name=name))
            assert response["result"]["raw_path_hex"] == name.hex()
            assert _data(response) == expected
        for name in (b"linked.py", b"nested"):
            with pytest.raises(typed.TypedStateOwnerError):
                case.client.source_get(_payload(case, name=name))


def test_clean_crlf_checkout_is_not_admitted_as_exact_committed_bytes(tmp_path):
    files = {b"module.py": b"VALUE = 1\n", b".gitattributes": b"module.py text eol=crlf\n"}
    with _case(tmp_path, files=files) as case:
        (case.root / "module.py").write_bytes(b"VALUE = 1\r\n")
        # Refresh Git's stat record through the configured clean filter;
        # the admitted tree remains exact while disk bytes are CRLF.
        _git(case.root, "add", "module.py")
        assert _git(case.root, "rev-parse", "HEAD") == case.request.expected_commit
        assert _git(case.root, "write-tree") == case.request.expected_tree
        assert _git(case.root, "rev-parse", ":module.py") == _git(case.root, "rev-parse", "HEAD:module.py")
        assert _git(case.root, "status", "--porcelain") == ""
        with pytest.raises(typed.TypedStateOwnerError):
            case.client.source_get(_payload(case))


def test_group_execute_cannot_replace_git_owner_execute_under_clean_status(tmp_path):
    with _case(tmp_path, files={b"module.py": b"VALUE = 1\n"}, executable="module.py") as case:
        _git(case.root, "config", "core.filemode", "false")
        (case.root / "module.py").chmod(0o650)
        assert _git(case.root, "status", "--porcelain") == ""
        with pytest.raises(typed.TypedStateOwnerError):
            case.client.source_get(_payload(case))


def test_retained_source_corruption_cannot_fall_back_to_unchanged_working_file(current):
    cid = _chunk_cid(current.source)
    path = current.cas.path_for(cid, source=True)
    data = path.read_bytes()
    path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(typed.TypedStateOwnerError):
        current.client.source_get(_payload(current))
    assert (current.root / "module.py").read_bytes() == DEFAULT_SOURCE


@pytest.mark.parametrize("drift", ["inventory", "store", "limits"])
def test_callbacks_after_verified_cas_read_cannot_escape_current_custody(current, monkeypatch, drift):
    replacement_store = sources._BoundCAS(current.cas, current.service._guard)
    original, seen = ImmutableCAS.get_bytes, []
    def change(self, cid):
        raw = original(self, cid)
        if self is current.cas and not seen:
            seen.append(cid)
            if drift == "inventory": current.service._inventory_cid = cid_for_structured({"foreign": "inventory"})
            elif drift == "store": current.service._store = replacement_store
            else: current.service._limits = replace(current.limits, max_file_bytes=current.limits.max_file_bytes + 1)
        return raw
    monkeypatch.setattr(ImmutableCAS, "get_bytes", change)
    with pytest.raises(typed.TypedStateOwnerError):
        current.client.source_get(_payload(current))
    assert len(seen) == 1


def test_intermediate_parent_swap_during_last_working_read_refuses_even_identical_bytes(tmp_path, monkeypatch):
    files = {b"pkg/module.py": DEFAULT_SOURCE}
    with _case(tmp_path, files=files, allowed=(b"pkg/module.py".hex(),)) as case:
        original, reads, changed = os.read, [], []
        def read(fd, length):
            data = original(fd, length)
            try: name = os.readlink(f"/proc/self/fd/{fd}")
            except OSError: name = ""
            if data and name == str(case.root / "pkg/module.py"):
                reads.append(fd)
                if len(reads) == 2:
                    saved = tmp_path / "detached-original-parent"
                    (case.root / "pkg").rename(saved)
                    shutil.copytree(saved, case.root / "pkg")
                    changed.append(True)
            return data
        monkeypatch.setattr(os, "read", read)
        with pytest.raises(typed.TypedStateOwnerError):
            case.client.source_get(_payload(case, name=b"pkg/module.py"))
        assert changed == [True]
        assert _git(case.root, "status", "--porcelain") == ""


def test_final_native_check_cannot_change_clean_filtered_working_bytes(tmp_path, monkeypatch):
    files = {b"module.py": b"VALUE = 1\n", b".gitattributes": b"module.py text eol=crlf\n"}
    with _case(tmp_path, files=files) as case:
        original, calls = case.service.check_native, []
        def check(*args, **kwargs):
            result = original(*args, **kwargs)
            calls.append(True)
            if len(calls) == 2:
                (case.root / "module.py").write_bytes(b"VALUE = 1\r\n")
                _git(case.root, "add", "module.py")
                assert _git(case.root, "rev-parse", "HEAD") == case.request.expected_commit
                assert _git(case.root, "write-tree") == case.request.expected_tree
                assert _git(case.root, "rev-parse", ":module.py") == _git(case.root, "rev-parse", "HEAD:module.py")
                assert _git(case.root, "status", "--porcelain") == ""
            return result
        monkeypatch.setattr(case.service, "check_native", check)
        with pytest.raises(typed.TypedStateOwnerError):
            case.client.source_get(_payload(case))
        assert len(calls) == 2 and _git(case.root, "status", "--porcelain") == ""


def _recount(response):
    result = response["result"]
    for _ in range(20):
        count = len(canonical_json_bytes(response)) + 1
        if result["response_bytes"] == count and result["response_wire_bytes"] == count + 3:
            return
        result["response_bytes"], result["response_wire_bytes"] = count, count + 3
    pytest.fail("authored response count did not stabilize")


@pytest.mark.parametrize("attack", ["unknown-binding-field", "frontier-bool-alias"])
def test_response_validation_refuses_rehashed_closed_binding_or_boolean_frontier_alias(current, attack):
    payload = _payload(current, maximum=1)
    response = deepcopy(current.client.source_get(payload))
    if attack == "unknown-binding-field":
        binding = response["result"]["binding"]
        binding["invented_proof"] = False
        new_cid = cid_for_structured({key: value for key, value in binding.items() if key != "binding_cid"})
        binding["binding_cid"] = response["result"]["binding_cid"] = payload["binding_cid"] = new_cid
        response["result"]["request_cid"] = cid_for_structured(payload)
    else:
        assert response["result"]["end_offset"] == 1
        response["result"]["unresolved_frontier"][0]["offset"] = True
    _recount(response)
    with pytest.raises(lookup.CurrentSourceLookupError):
        lookup.validate_source_response(response, payload=payload,
            owner_identity=current.client.identity, grant=current.client.grant)
