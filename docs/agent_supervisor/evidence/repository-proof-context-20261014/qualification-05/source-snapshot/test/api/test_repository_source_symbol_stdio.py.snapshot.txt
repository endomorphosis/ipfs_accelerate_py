"""Indexed definition reads through the real worker route and child MCP.

The provider boundary is authored. Wire counts describe this local fixture,
not Terminal Bench success or provider/session token savings.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_mcp as mcp
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_policy as policy
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_token_budget as budget
from test.api.test_local_prompt_tokenizer import make_binding
from test.api.test_repository_source_mcp import real_client
from test.api.test_repository_source_router import route, _run
from test.api.test_repository_source_token_stdio import token_config


def call(client, name, **arguments):
    value, failed = client.call(name, **arguments)
    assert not failed, value
    return value


def test_real_worker_queries_reads_definition_edits_and_refreshes(route, tmp_path):
    workspace, state = route
    source = workspace / "module.py"
    source.write_text(
        "# Public API comment stays byte-identical.\n"
        "class Counter:\n"
        "    @staticmethod\n"
        "    def square(value):\n"
        "        # Deliberately long source comment that is omitted only in a compact read.\n"
        "        return value * value\n\n"
        "# Unselected suffix stays byte-identical.\n"
        "def untouched():\n    return 19\n")
    consumer = workspace / "consumer.py"
    consumer.write_text("from module import Counter\n\ndef run():\n    return Counter.square(3)\n")
    before, consumer_before = source.read_bytes(), consumer.read_bytes()
    artifact, digest, _ = make_binding(tmp_path, provider="codex_cli", model="pinned")

    def worker(binding):
        with real_client(Path(binding["config_path"]), binding["config_sha256"]) as client:
            context = call(client, "repository_context_query", generation=1, query="square", limit=1)
            candidate = context["candidates"][0]
            selected = call(client, "repository_read", generation=1, path=candidate["path"],
                            symbol_id=candidate["symbol_id"])
            assert selected["symbol"] == "Counter.square"
            assert selected["symbol_id"] == candidate["symbol_id"]
            assert selected["scope"] == "python_definition" and selected["edit_supported"] is True
            assert selected["source_sha256"] == hashlib.sha256(before).hexdigest()
            assert selected["representation"] == "compact" and selected["proof_authority"] is False
            assert selected["content"].startswith("@staticmethod\ndef square(")
            assert "class Counter" not in selected["content"] and "untouched" not in selected["content"]
            assert "source comment" not in selected["content"]
            edited = call(client, "repository_replace_symbol", generation=selected["generation"],
                path=selected["path"], symbol=selected["symbol"], expected_sha256=selected["source_sha256"],
                representation=selected["representation"],
                replacement=selected["content"].replace("value * value", "value * value + 1"))
            assert edited["canonical_changed"] is True and edited["generation"] == 2
            assert edited["refresh_required"] is False
            stale, failed = client.call("repository_read", generation=1, path=candidate["path"],
                                         symbol_id=candidate["symbol_id"])
            assert failed and stale == {"error": "stale_generation"}
            current = call(client, "repository_context_query", generation=2, query="square", limit=1)
            assert current["semantic_root_cid"] != context["semantic_root_cid"]
            successor = call(client, "repository_read", generation=2, path="module.py",
                             symbol_id=current["candidates"][0]["symbol_id"], representation="original")
            assert successor["source_sha256"] == edited["source_sha256"]
            assert "value * value + 1" in successor["content"]
            # A change to an unselected dependency still fences targeted reads.
            consumer.write_bytes(consumer_before + b"# External caller edit.\n")
            stale, failed = client.call("repository_read", generation=2, path="module.py",
                                         symbol_id=current["candidates"][0]["symbol_id"])
            assert failed and stale == {"error": "stale_source"}
            refreshed = call(client, "repository_refresh", generation=2)
            full = call(client, "repository_read", generation=refreshed["generation"],
                        path="module.py", representation="original")
            assert "symbol" not in full and full["content"].encode() == source.read_bytes()

    state["callback"] = worker
    _, receipt = _run(repository_source_policy=policy.POLICIES[2],
        prompt_tokenizer_artifact=artifact, prompt_tokenizer_sha256=digest)
    after = source.read_bytes()
    assert after.split(b"    @staticmethod")[0] == before.split(b"    @staticmethod")[0]
    assert after.split(b"# Unselected suffix")[1] == before.split(b"# Unselected suffix")[1]
    assert consumer.read_bytes() == consumer_before + b"# External caller edit.\n"
    checked = subprocess.run([sys.executable, "-B", "-c", "from consumer import run; assert run() == 10"],
                             cwd=workspace, capture_output=True, timeout=10)
    assert checked.returncode == 0, checked.stderr
    usage = receipt["repository_source_policy"]["usage"]
    assert usage["successful_reads"] == usage["token_selection"]["read_count"] == 3
    assert usage["canonical_edits"] == 1 and usage["errors"] == 2
    assert usage["retrieval_observation"]["successful_queries"] == 2
    assert usage["store_cleanup_complete"] is True
    assert not Path(receipt["repository_source_policy"]["invocation_dir"]).exists()
    assert "def square" not in json.dumps(usage)


def test_compact_fallback_original_noop_and_unsupported_full_file_escape(tmp_path):
    config, path, digest = token_config(tmp_path)
    source = Path(config["workspace"]) / "calc.py"
    source.write_text("def tiny(): return 1\n\ndef outer():\n    def inner():\n        return 2\n    return inner()\n")
    before = source.read_bytes()
    with real_client(path, digest) as client:
        symbols = call(client, "repository_symbols", generation=1, path="calc.py")["symbols"]
        tiny = next(row for row in symbols if row["name"] == "calc.tiny")
        read = call(client, "repository_read", generation=1, path="calc.py", symbol_id=tiny["symbol_id"])
        assert read["representation"] == "original" and read["content"] == "def tiny(): return 1"
        # Returned representation, not session default, controls exact no-op recognition.
        edited = call(client, "repository_replace_symbol", generation=1, path="calc.py",
            symbol=read["symbol"], expected_sha256=read["source_sha256"],
            representation=read["representation"], replacement=read["content"])
        assert edited["canonical_changed"] is False and edited["generation"] == 2
        assert source.read_bytes() == before
        symbols = call(client, "repository_symbols", generation=2, path="calc.py")["symbols"]
        nested = next(row for row in symbols if row["name"] == "calc.outer.inner")
        refused, failed = client.call("repository_read", generation=2, path="calc.py", symbol_id=nested["symbol_id"])
        assert failed and "content" not in refused
        assert call(client, "repository_status")["source_current"] is True
        full = call(client, "repository_read", generation=2, path="calc.py", representation="original")
        assert full["content"].encode() == before
        mismatch, failed = client.call("repository_read", generation=2, path="other.py", symbol_id=tiny["symbol_id"])
        assert failed and mismatch == {"error": "symbol_path_differs"}
        assert call(client, "repository_status")["source_current"] is True
    summary = budget.summarize_selection_audit(path.parent / "token-selections.jsonl")
    assert summary["read_count"] == 2 and summary["compact_selected"] == 0


def test_local_wire_read_comparison_counts_full_envelopes(tmp_path, record_property):
    config, path, digest = token_config(tmp_path)
    source = Path(config["workspace"]) / "calc.py"
    source.write_text("\n".join(
        f"def item_{number}(value):\n"
        "    # Long authored implementation commentary omitted from compact views.\n"
        f"    return value + {number}\n" for number in range(32)))
    from ipfs_accelerate_py.agent_supervisor.runtime.local_prompt_tokenizer import load_owner_prompt_tokenizer
    tokenizer = load_owner_prompt_tokenizer(**config["tokenizer"])
    observations = {}
    with real_client(path, digest) as client:
        context = call(client, "repository_context_query", generation=1, query="item_17", limit=1)
        candidate = context["candidates"][0]
        for label, options in (
            ("whole_original", {"representation": "original"}),
            ("whole_compact", {"representation": "compact"}),
            ("definition_compact", {"representation": "compact", "symbol_id": candidate["symbol_id"]}),
        ):
            arguments = dict(generation=1, path="calc.py", **options)
            request = dict(jsonrpc="2.0", id=client.sequence + 1, method="tools/call",
                           params={"name": "repository_read", "arguments": arguments})
            response = client.request("tools/call", request["params"])
            assert response["result"]["isError"] is False
            # Client.send and server.serve use these exact serializations.
            request_frame = json.dumps(request) + "\n"
            response_frame = (mcp._json(response) + b"\n").decode()
            payload = json.loads(response["result"]["content"][0]["text"])
            observations[label] = dict(request_bytes=len(request_frame.encode()),
                reply_bytes=len(response_frame.encode()),
                fixture_request_tokens=tokenizer.count(request_frame),
                fixture_reply_tokens=tokenizer.count(response_frame),
                representation=payload["representation"])
        assert observations["definition_compact"]["reply_bytes"] < observations["whole_compact"]["reply_bytes"]
        assert observations["whole_compact"]["reply_bytes"] < observations["whole_original"]["reply_bytes"]
        assert (observations["definition_compact"]["fixture_reply_tokens"]
                < observations["whole_compact"]["fixture_reply_tokens"])
    evidence = dict(schema="authored-symbol-read-wire-comparison@1", definitions=32,
        scope="one repository_read request and complete reply per arm; identical preceding query excluded",
        tokenizer="authored-tiny-wordlevel-fixture", provider_usage_measured=False,
        total_session_token_savings=None, terminal_bench_success_measured=False, observations=observations)
    record_property("symbol_read_wire_comparison", json.dumps(evidence, sort_keys=True))
