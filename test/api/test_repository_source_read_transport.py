"""Real source/stdio/audit lifecycle with an authored native-index adapter.

Source capture, compact rendering, generation fences, MCP framing and private
cleanup execute normally. Only native indexing uses the existing SymbolIndex
fixture; this suite does not qualify DuckDB, model adoption or native tokens.
"""
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_mcp as mcp
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_policy as policy
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_read_observation as observations
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
from test.api.test_repository_source_mcp import Client, configuration, initialization
from test.api.test_repository_source_symbol_session import SymbolIndex


def session_factory(workspace, state_dir, **options):
    return RepositorySourceSession(workspace, state_dir, index_factory=SymbolIndex, **options)


@contextmanager
def authored_index_client(config_path, digest):
    # Constructor injection is an existing Python interface, not a new CLI or
    # an override of resource admission. This child never instantiates a native
    # index or opens a scheduler/database.
    command = [sys.executable, "-B", "-c", """
import sys
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_mcp as mcp
from test.api.test_repository_source_read_transport import session_factory
config = mcp.load_config(Path(sys.argv[1]), sys.argv[2])
server = mcp.RepositoryMCPServer(config, session_factory=session_factory)
try:
    code = mcp.serve(server, sys.stdin.buffer, sys.stdout.buffer)
finally:
    server.close()
raise SystemExit(code)
""", str(config_path), digest]
    with (config_path.parent / "stdio-test.stderr").open("wb") as diagnostics:
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=diagnostics, bufsize=0, cwd="/tmp")
        try:
            client = Client(child)
            client.send(initialization())
            assert "result" in client.receive()
            client.send(dict(jsonrpc="2.0", method="notifications/initialized"))
            yield client
        finally:
            child.stdin.close()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
            child.stdout.close()
        assert child.returncode == 0


def test_stdio_repetition_refresh_binary_and_symbol_scope_survive_cleanup(tmp_path):
    config, _, _ = configuration(tmp_path)
    workspace = Path(config["workspace"])
    source = workspace / "calc.py"
    source.write_text(source.read_text() + "# Unicode fixture: λ\n")
    (workspace / "opaque.bin").write_bytes(b"\xff\x00\x10")
    binding = policy.create_binding(workspace=workspace, requested_policy=policy.POLICIES[2])
    identity = policy.create_runtime_store(binding)
    returned = []
    try:
        with authored_index_client(Path(binding["config_path"]), binding["config_sha256"]) as client:
            def read(generation=1, path="calc.py", representation="original", **extra):
                value, failed = client.call("repository_read", generation=generation, path=path,
                                           representation=representation, **extra)
                assert not failed, value
                returned.append(value)
                return value
            assert not client.call("repository_list", generation=1)[1]
            first = read()
            assert read() == first  # Different JSON-RPC IDs, same source reply.
            compact = read(representation="compact")
            assert read(representation="compact") == compact
            symbol = read(symbol_id="native-add")
            assert read(symbol_id="native-add") == symbol
            assert symbol["scope"] == "python_definition"
            assert read(path="opaque.bin")["encoding"] == "base64"
            (workspace / "unrelated.txt").write_text("generation changes, calc bytes do not\n")
            assert client.call("repository_read", generation=1, path="calc.py")[1]
            assert client.call("repository_refresh", generation=1)[0]["generation"] == 2
            assert client.call("repository_read", generation=1, path="calc.py")[1]
            assert read(generation=2)["content"] == first["content"]
            assert read(generation=2, representation="compact")["content"] == compact["content"]
            source.write_text(source.read_text() + "# A changed canonical source identity.\n")
            assert client.call("repository_refresh", generation=2)[0]["generation"] == 3
            changed = read(generation=3, representation="compact")
            assert changed["content"] == compact["content"]
            assert changed["source_sha256"] != compact["source_sha256"]
        audit = Path(binding["audit_store"]).read_text()
        before = mcp.summarize_repository_audit(binding["audit_store"])
        assert all(text not in audit for text in ("calc.py", "native-add", "return value", "Unicode fixture"))
        assert len(audit.encode()) < mcp.MAX_AUDIT_BYTES
    finally:
        usage = policy.finish_runtime_store(binding, expected_identity=identity)
    assert usage["store_cleanup_complete"] is True
    assert not Path(binding["invocation_dir"]).exists()
    observed = usage["read_observation"]
    assert all("read_observation" not in row for row in returned)
    assert observed == before["read_observation"]
    assert observed["complete"] is True
    assert observed["identified_reads"] == observed["successful_reads"] == 10
    assert observed["read_error_count"] == 2
    assert observed["known_first_identity_reads"] == 5
    assert observed["known_same_generation_repeat_reads"] == 3
    assert observed["known_cross_generation_repeat_reads"] == 2
    assert len(observed["groups"]) == 7
    assert observed["known_content_bytes"] == sum(len(row["content"].encode()) for row in returned)
    assert observed["successful_read_payload_bytes"] > observed["known_content_bytes"]
    assert observed["other_operation_payload_bytes"] > 0
    assert observed["provider_ingestion_verified"] is observed["provider_tokens"] is None
    assert observed["inline_overlap_bytes"] is None


def test_prepared_read_survives_broken_output_without_becoming_ingestion(tmp_path):
    config, _, _ = configuration(tmp_path)
    server = mcp.RepositoryMCPServer(config, session_factory=session_factory)
    class BrokenOutput:
        def write(self, _):
            raise OSError("authored unavailable receiver")
        def flush(self):
            raise AssertionError("write should already fail")
    try:
        assert "result" in server.handle(initialization())
        server.handle(dict(jsonrpc="2.0", method="notifications/initialized"))
        request = dict(jsonrpc="2.0", id=2, method="tools/call", params={"name": "repository_read",
            "arguments": {"generation": 1, "path": "calc.py", "representation": "original"}})
        with pytest.raises(OSError):
            mcp.serve(server, io.BytesIO(json.dumps(request).encode() + b"\n"), BrokenOutput())
    finally:
        server.close()
    observed = mcp.summarize_repository_audit(config["audit_store"])["read_observation"]
    assert observed["identified_reads"] == 1
    assert observed["provider_ingestion_verified"] is None
    assert observed["provider_tokens"] is None


def test_maximum_read_population_fits_unchanged_audit_limit(tmp_path):
    config, _, _ = configuration(tmp_path)
    audit = mcp._Audit(config["audit_store"])
    try:
        audit.record("initialize")
        for index in range(mcp.MAX_REQUESTS):
            value = dict(generation=8, path=f"private-{index}.py", source_sha256=hashlib.sha256(str(index).encode()).hexdigest(),
                representation="original", encoding="utf-8", content="x = 1\n", proof_authority=False)
            event = observations.make_read_event({"generation": 8, "path": value["path"]}, value, True)
            audit.record("tool", tool="repository_read", returned_bytes=len(mcp._json(value)), read_observation=event)
    finally:
        audit.close()
    assert Path(config["audit_store"]).stat().st_size <= mcp.MAX_AUDIT_BYTES
    result = mcp.summarize_repository_audit(config["audit_store"])
    assert result["status"] == "observed"
    assert result["read_observation"]["identified_reads"] == mcp.MAX_REQUESTS
    assert len(result["read_observation"]["groups"]) == mcp.MAX_REQUESTS


@pytest.mark.parametrize("mutation", ["wrong_tool", "failed_read", "compact_flag", "truncated"])
def test_malformed_read_journal_cannot_retain_complete_read_accounting(tmp_path, mutation):
    config, _, _ = configuration(tmp_path)
    audit = mcp._Audit(config["audit_store"])
    value = dict(generation=1, path="calc.py", source_sha256="a" * 64,
                 representation="original", encoding="utf-8", content="x", proof_authority=False)
    try:
        audit.record("initialize")
        audit.record("tool", tool="repository_read", returned_bytes=200,
            read_observation=observations.make_read_event({"generation": 1, "path": "calc.py"}, value, True))
    finally:
        audit.close()
    path = Path(config["audit_store"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "wrong_tool": rows[-1]["tool"] = "repository_status"
    elif mutation == "failed_read": rows[-1]["success"] = False
    elif mutation == "compact_flag": rows[-1]["compact"] = True
    encoded = b"".join(mcp._json(row) + b"\n" for row in rows)
    path.write_bytes(encoded[:-2] if mutation == "truncated" else encoded)
    result = mcp.summarize_repository_audit(path)
    assert result["status"] == "unavailable"
    assert result.get("read_observation") is None
