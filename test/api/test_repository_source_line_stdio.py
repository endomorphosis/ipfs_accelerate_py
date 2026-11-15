"""Exact bounded source working sets through the real native MCP child.

Only authored fixture files are read. Local tokenizer counts describe complete
tool frames, not remote provider usage or whole-task token savings.
"""
import hashlib
import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_mcp as mcp
from ipfs_accelerate_py.agent_supervisor.runtime.local_prompt_tokenizer import load_owner_prompt_tokenizer
from test.api.test_repository_source_mcp import configuration, real_client
from test.api.test_repository_source_token_stdio import token_config


def _read(client, **arguments):
    return client.call("repository_read", generation=1, representation="original", **arguments)


def test_actual_child_returns_exact_javascript_lines_and_full_source_binding(tmp_path):
    config, path, digest = configuration(tmp_path)
    source = Path(config["workspace"]) / "app.js"
    raw = b"// retained context\nexport function value(x) {\n  return x + 1;\n}\n"
    source.write_bytes(raw)
    with real_client(path, digest) as client:
        window, failed = _read(client, path="app.js", start_line=2, end_line=3)
        assert not failed, window
        assert window["content"].encode() == b"export function value(x) {\n  return x + 1;\n"
        assert window["source_sha256"] == hashlib.sha256(raw).hexdigest()
        assert window["representation"] == "original" and window["encoding"] == "utf-8"
        assert {key: window[key] for key in (
            "scope", "start_line", "end_line", "total_lines", "omitted_lines")} == {
                "scope": "line_range", "start_line": 2, "end_line": 3,
                "total_lines": 4, "omitted_lines": 2}
        assert "symbol" not in window and "symbol_id" not in window
        assert window["proof_authority"] is False
        whole, failed = _read(client, path="app.js")
        assert not failed and whole["content"].encode() == raw
        assert whole["source_sha256"] == window["source_sha256"]
        assert "scope" not in whole  # Existing whole-file response stays exact.
    audit = mcp.summarize_repository_audit(config["audit_store"])
    assert audit["successful_reads"] == 2 and audit["errors"] == 0
    assert "export function" not in Path(config["audit_store"]).read_text()


def test_actual_child_preserves_unicode_crlf_and_non_lf_separators(tmp_path):
    config, path, digest = configuration(tmp_path)
    raw = "heading\r\nπ\u2028same physical line\r\ntrailer\rlone CR".encode()
    (Path(config["workspace"]) / "notes.txt").write_bytes(raw)
    with real_client(path, digest) as client:
        window, failed = _read(client, path="notes.txt", start_line=2, end_line=3)
        assert not failed, window
        assert window["content"].encode() == raw[len(b"heading\r\n"):]
        assert window["total_lines"] == 3 and window["omitted_lines"] == 1
        assert window["source_sha256"] == hashlib.sha256(raw).hexdigest()


def test_actual_child_rejects_ambiguous_ranges_without_poisoning_session(tmp_path):
    config, path, digest = configuration(tmp_path)
    with real_client(path, digest) as client:
        for changes in (
            {"start_line": 1}, {"end_line": 2},
            {"start_line": 3, "end_line": 1},
            {"start_line": True, "end_line": 2},
            {"start_line": 1, "end_line": 99},
            {"start_line": 1, "end_line": 2, "symbol_id": "foreign-symbol"},
            {"start_line": 1, "end_line": 2, "representation": "compact"},
        ):
            response, failed = client.call("repository_read", **{
                "generation": 1, "path": "calc.py", "representation": "original", **changes})
            assert failed and set(response) == {"error"}, response
        # The compact session default is deliberately not a line coordinate system.
        response, failed = client.call("repository_read", generation=1, path="calc.py",
                                      start_line=1, end_line=2)
        assert failed and response["error"] == "line_range_requires_original"
        healthy, failed = _read(client, path="calc.py", start_line=2, end_line=2)
        assert not failed and healthy["content"] == "def add(value):\n"
    audit = mcp.summarize_repository_audit(config["audit_store"])
    assert audit["errors"] == 8 and audit["successful_reads"] == 1


def test_actual_child_range_checks_other_source_changes_and_successor_generation(tmp_path):
    config, path, digest = configuration(tmp_path)
    source = Path(config["workspace"])
    (source / "app.sh").write_bytes(b"#!/bin/sh\nprintf 'fixture'\n")
    with real_client(path, digest) as client:
        before, failed = _read(client, path="app.sh", start_line=2, end_line=2)
        assert not failed, before
        # Even an unchanged requested file cannot bypass whole-root freshness.
        (source / "calc.py").write_text("def add(value):\n    return value + 2\n")
        stale, failed = _read(client, path="app.sh", start_line=2, end_line=2)
        assert failed and stale["error"] == "stale_source"
        refreshed, failed = client.call("repository_refresh", generation=1)
        assert not failed and refreshed["generation"] == 2
        current, failed = client.call("repository_read", generation=2, path="app.sh",
            representation="original", start_line=2, end_line=2)
        assert not failed and current["content"] == before["content"]
        assert current["source_sha256"] == before["source_sha256"]
        assert current["generation"] == 2


def test_actual_child_rejects_binary_empty_and_oversized_ranges(tmp_path):
    config, path, digest = configuration(tmp_path)
    source = Path(config["workspace"])
    (source / "binary.dat").write_bytes(b"\xff\x00\n")
    (source / "empty.txt").write_bytes(b"")
    (source / "long.txt").write_bytes(b"x" * 65537 + b"\n")
    with real_client(path, digest) as client:
        for name, error in (("binary.dat", "line_range_requires_text"),
                            ("empty.txt", "line_range_out_of_bounds"),
                            ("long.txt", "line_range_byte_limit")):
            response, failed = _read(client, path=name, start_line=1, end_line=1)
            assert failed and response["error"] == error
        whole, failed = _read(client, path="binary.dat")
        assert not failed and whole["encoding"] == "base64"


def test_actual_tokenized_child_delivers_smaller_complete_exact_line_reply(tmp_path, record_property):
    config, path, digest = token_config(tmp_path)
    source = Path(config["workspace"]) / "large.js"
    lines = [f"const fixture_{i} = 'retained authored source {i}';\n" for i in range(600)]
    raw = "".join(lines).encode()
    source.write_bytes(raw)
    with real_client(path, digest) as client:
        def frame(**options):
            response = client.request("tools/call", {"name": "repository_read", "arguments": {
                "generation": 1, "path": "large.js", "representation": "original", **options}})
            assert response["result"]["isError"] is False, response
            payload = json.loads(response["result"]["content"][0]["text"])
            rendered = json.dumps(response, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
            return payload, rendered
        whole, full_frame = frame()
        window, window_frame = frame(start_line=301, end_line=305)
        assert whole["content"].encode() == raw
        assert window["content"] == "".join(lines[300:305])
        assert whole["source_sha256"] == window["source_sha256"] == hashlib.sha256(raw).hexdigest()
        assert window["total_lines"] == 600 and window["omitted_lines"] == 595
        rejected, failed = client.call("repository_read", generation=1, path="large.js",
                                      start_line=301, end_line=305)
        assert failed and rejected["error"] == "line_range_requires_original"
    tokenizer = load_owner_prompt_tokenizer(**config["tokenizer"])
    full_tokens, window_tokens = tokenizer.count(full_frame), tokenizer.count(window_frame)
    assert len(window_frame.encode()) < len(full_frame.encode())
    assert 0 < window_tokens < full_tokens
    observation = {"scope": "authored_local_complete_MCP_reply_only",
        "full_reply_bytes": len(full_frame.encode()), "range_reply_bytes": len(window_frame.encode()),
        "full_local_tokens": full_tokens, "range_local_tokens": window_tokens,
        "provider_usage_measured": False, "whole_task_token_savings_measured": False,
        "tool_schema_and_guidance_cost_included": False}
    record_property("line_read_exposure", json.dumps(observation, sort_keys=True))
    audit = mcp.summarize_repository_audit(config["audit_store"])
    assert audit["successful_reads"] == 2 and audit["errors"] == 1
