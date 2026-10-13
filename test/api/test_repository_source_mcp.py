"""Real stdio read/edit/refresh and bounded protocol failure tests."""
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_mcp as mcp


def configuration(tmp_path):
    workspace = tmp_path / "source"
    workspace.mkdir()
    (workspace / "calc.py").write_text(
        "# Original public comment is retained in canonical bytes.\n"
        "def add(value):\n    # This long implementation comment is absent from a compact view.\n"
        "    return value + 1\n")
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    (private / "state").mkdir(mode=0o700)
    config = dict(schema="repository-source-mcp-config@1", workspace=str(workspace),
        state_dir=str(private / "state"), audit_store=str(private / "operations.jsonl"),
        representation="compact")
    path = private / "config.json"
    raw = json.dumps(config).encode()
    path.write_bytes(raw)
    path.chmod(0o600)
    return config, path, hashlib.sha256(raw).hexdigest()


def initialization(identity=1):
    return dict(jsonrpc="2.0", id=identity, method="initialize", params={
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "authored-test", "version": "1"}})


class Client:
    def __init__(self, process):
        self.process = process
        self.sequence = 1

    def send(self, value):
        self.process.stdin.write(json.dumps(value).encode() + b"\n")
        self.process.stdin.flush()

    def receive(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            assert selector.select(60), "MCP response timeout"
        line = self.process.stdout.readline()
        assert line, f"MCP exited: {self.process.poll()}"
        return json.loads(line)

    def request(self, method, params=None):
        self.sequence += 1
        self.send(dict(jsonrpc="2.0", id=self.sequence, method=method, params=params or {}))
        response = self.receive()
        assert response["id"] == self.sequence
        return response

    def call(self, tool, **arguments):
        response = self.request("tools/call", {"name": tool, "arguments": arguments})
        result = response["result"]
        return json.loads(result["content"][0]["text"]), result["isError"]


@contextmanager
def real_client(config_path, digest):
    command = [sys.executable, mcp.__file__, "--config", str(config_path), "--config-sha256", digest]
    # Keep bounded diagnostics in a file so stderr cannot block the child pipe.
    with (config_path.parent / "stderr.log").open("wb") as diagnostics:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=diagnostics, bufsize=0, cwd="/tmp")
        try:
            client = Client(process)
            client.send(initialization())
            response = client.receive()
            assert "result" in response, response
            assert response["result"]["protocolVersion"] == "2025-06-18"
            client.send(dict(jsonrpc="2.0", method="notifications/initialized"))
            yield client
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            process.stdout.close()


def test_real_stdio_canonical_edit_external_change_refresh_and_audit(tmp_path):
    config, path, digest = configuration(tmp_path)
    source = Path(config["workspace"])
    with real_client(path, digest) as client:
        tools = client.request("tools/list")["result"]["tools"]
        assert {tool["name"] for tool in tools} == set(mcp.TOOL_NAMES)
        status, failed = client.call("repository_status")
        assert not failed and status["generation"] == 1 and status["source_current"]
        symbols, failed = client.call("repository_symbols", generation=1, query="add")
        assert not failed and any(row["name"] == "calc.add" for row in symbols["symbols"])
        read, failed = client.call("repository_read", generation=1, path="calc.py")
        assert not failed and read["representation"] == "compact"
        assert "Original public comment" not in read["content"]
        original = (source / "calc.py").read_bytes()
        edited, failed = client.call("repository_replace_symbol", generation=1, path="calc.py",
            symbol="add", expected_sha256=hashlib.sha256(original).hexdigest(),
            replacement="def add(value):\n    return value + 10\n")
        assert not failed and edited["canonical_changed"] is True and edited["generation"] == 2
        assert b"Original public comment" in (source / "calc.py").read_bytes()
        run = subprocess.run([sys.executable, "-B", "-c", "from calc import add; print(add(2))"],
                             cwd=source, capture_output=True, timeout=10, check=True)
        assert run.stdout == b"12\n"  # authored behavior, no target code is run by MCP
        stale, failed = client.call("repository_read", generation=1, path="calc.py")
        assert failed and stale["error"] == "stale_generation"
        (source / "other.py").write_text("def second():\n    return 2\n")
        stale, failed = client.call("repository_symbols", generation=2)
        assert failed and stale["error"] == "stale_source"
        refreshed, failed = client.call("repository_refresh", generation=2)
        assert not failed and refreshed["generation"] == 3
        symbols, failed = client.call("repository_symbols", generation=3, query="second")
        assert not failed and any(row["name"] == "other.second" for row in symbols["symbols"])
        original_read, failed = client.call("repository_read", generation=3, path="calc.py", representation="original")
        assert not failed and original_read["representation"] == "original"
        assert original_read["content"].encode() == (source / "calc.py").read_bytes()
        assert original_read["proof_authority"] is False
    summary = mcp.summarize_repository_audit(config["audit_store"])
    assert summary["status"] == "observed" and summary["initializations"] == 1
    assert summary["successful_reads"] == 2 and summary["compact_reads"] == 1
    assert summary["canonical_edits"] == 1 and summary["refreshes"] == 2
    assert summary["errors"] == 2 and summary["uncertain_edits"] == 0
    audit = Path(config["audit_store"]).read_text()
    assert "calc.py" not in audit and "return value" not in audit and "source_tree_sha256" not in audit


def test_actual_pytest_caches_remain_retained_and_refreshable(tmp_path):
    """Running authored task checks must not permanently fence the native index."""
    import base64
    config, path, digest = configuration(tmp_path)
    source = Path(config["workspace"])
    (source / "test_calc.py").write_text("from calc import add\n\ndef test_add():\n    assert add(2) == 3\n")
    with real_client(path, digest) as client:
        before, failed = client.call("repository_status")
        assert not failed and before["generation"] == 1
        environment = {key: value for key, value in os.environ.items()
                       if key not in {"PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX", "PYTEST_ADDOPTS"}}
        environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        checked = subprocess.run([sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "test_calc.py"],
            cwd=source, env=environment, capture_output=True, text=True, timeout=30, check=False)
        assert checked.returncode == 0, checked.stderr
        assert (source / "__pycache__").is_dir() and (source / ".pytest_cache").is_dir()
        cached = {p.relative_to(source).as_posix(): p.read_bytes()
                  for folder in (source / "__pycache__", source / ".pytest_cache")
                  for p in folder.rglob("*") if p.is_file()}
        stale, failed = client.call("repository_read", generation=1, path="calc.py")
        assert failed and stale["error"] == "stale_source"
        refreshed, failed = client.call("repository_refresh", generation=1)
        assert not failed and refreshed["generation"] == 2
        status, failed = client.call("repository_status")
        assert not failed and status["source_current"] is True
        retained = status["native_index"]["retained_opaque_cache"]
        assert retained["roots_count"] == 2 and "roots" not in retained and "entries" not in retained
        inventory, failed = client.call("repository_cache_inventory", generation=2,
            expected_inventory_sha256=retained["inventory_sha256"])
        assert not failed and inventory["next_offset"] is None
        assert sorted(row["path"] for row in inventory["entries"] if row["cache_root"]) == [".pytest_cache", "__pycache__"]
        entries = {row["path"]: row for row in inventory["entries"]}
        assert retained["native_indexed"] is retained["proof_authority"] is False
        assert retained["file_count"] == len(cached)
        assert status["native_index"]["coverage"]["inventory_entries"] == 2
        for name, original in cached.items():
            assert entries[name]["original_sha256"] == hashlib.sha256(original).hexdigest()
            read, failed = client.call("repository_read", generation=2, path=name, representation="original")
            assert not failed
            actual = read["content"].encode() if read["encoding"] == "utf-8" else base64.b64decode(read["content"])
            assert actual == original
        symbols, failed = client.call("repository_symbols", generation=2)
        assert not failed and all(row["path"] in {"calc.py", "test_calc.py"} for row in symbols["symbols"])
        (source / "calc.py").write_text("def add(value):\n    return value + 2\n")
        stale, failed = client.call("repository_symbols", generation=2)
        assert failed and stale["error"] == "stale_source"
        refreshed, failed = client.call("repository_refresh", generation=2)
        assert not failed and refreshed["generation"] == 3
        read, failed = client.call("repository_read", generation=3, path="calc.py")
        assert not failed and "return value + 2" in read["content"]


class FakeSession:
    def __init__(self, *args, **kwargs):
        pass

    def read(self, **kwargs):
        return {"representation": "compact", "content": "hello"}

    def close(self):
        pass


def test_protocol_validation_bounds_and_no_arbitrary_tools(tmp_path):
    config, _, _ = configuration(tmp_path)
    server = mcp.RepositoryMCPServer(config, session_factory=FakeSession)
    try:
        assert server.handle(dict(jsonrpc="2.0", id=0, method="tools/list"))["error"]["message"] == "not_initialized"
        assert "result" in server.handle(initialization())
        server.handle(dict(jsonrpc="2.0", method="notifications/initialized"))
        assert "error" in server.handle(initialization(2))
        for args in ({"generation": True, "path": "x.py"},
                     {"generation": 1, "path": "x.py", "workspace": "/"},
                     {"generation": 1, "path": "x.py", "representation": []}):
            response = server.handle(dict(jsonrpc="2.0", id=3, method="tools/call",
                params={"name": "repository_read", "arguments": args}))
            assert response["result"]["isError"]
        assert server.handle(dict(jsonrpc="2.0", id=4, method="tools/call", params={"name": "exec"}))["error"]["message"] == "unknown_tool"
        assert server.handle(dict(jsonrpc="2.0", id="\ud800", method="ping"))["error"]["code"] == -32600
        assert "result" in server.handle(dict(jsonrpc="2.0", id=5, method="tools/list", params={"_meta": {}}))
        output = io.BytesIO()
        mcp.serve(server, io.BytesIO(b'{"jsonrpc":"2.0","id":1,"id":2}\n'), output)
        assert json.loads(output.getvalue())["error"]["code"] == -32700
        output = io.BytesIO()
        assert mcp.serve(server, io.BytesIO(b" " * (mcp.MAX_REQUEST_BYTES + 1)), output) == 2
        assert json.loads(output.getvalue())["error"]["message"] == "request_limit"
    finally:
        server.close()


def test_startup_failure_is_observed_without_leaking_exception(tmp_path):
    config, _, _ = configuration(tmp_path)
    def unavailable(*args, **kwargs):
        raise RuntimeError("private source text")
    server = mcp.RepositoryMCPServer(config, session_factory=unavailable)
    try:
        response = server.handle(initialization())
        assert "private source" not in json.dumps(response)
        assert response["error"]["message"] == "repository_session_unavailable"
    finally:
        server.close()
    summary = mcp.summarize_repository_audit(config["audit_store"])
    assert summary["initializations"] == 0 and summary["errors"] == 1


def test_config_hash_scope_symlink_and_audit_validation(tmp_path):
    config, path, digest = configuration(tmp_path)
    assert mcp.load_config(path, digest) == config
    with pytest.raises(ValueError, match="hash"):
        mcp.load_config(path, "0" * 64)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="config_file"):
        mcp.load_config(path, digest)
    path.chmod(0o600)
    link = path.parent / "link.json"
    link.symlink_to(path)
    with pytest.raises(OSError):
        mcp.load_config(link, digest)
    assert mcp.summarize_repository_audit(config["audit_store"])["status"] == "not_observed"
    audit = Path(config["audit_store"])
    audit.write_bytes(b'{"truncated":')
    audit.chmod(0o600)
    assert mcp.summarize_repository_audit(audit)["status"] == "unavailable"
