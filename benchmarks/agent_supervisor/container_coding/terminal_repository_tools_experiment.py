"""Offline real-stdio repository tool cycles over reviewed public COPY inputs.

Only disposable copies are edited. No benchmark source is imported/executed, no
provider/container is started, and content-bearing protocol messages stay in
memory. Counts describe actual JSON-RPC wire bytes and a local token proxy, not
provider usage or a claim that a model adopted the tools.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time

from . import terminal_profile_catalog as catalog
from .terminal_source_view_experiment import (
    DEFAULT_TASKS, MAX_INPUTS, MAX_TASKS, MAX_TOTAL_BYTES,
    _relative, _render, _snapshot, local_token_counter,
)
from .terminal_repository_view_experiment import _read_bytes, _ast

SCHEMA = "terminal-repository-tools-experiment@1"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
WIRE_COMPONENTS = ("setup", "status", "list", "symbols", "read", "edit", "refresh", "cache_inventory",
                   "context_query", "capsule_expand")
_TOOL_COMPONENTS = {
    "repository_status": "status", "repository_list": "list", "repository_symbols": "symbols",
    "repository_read": "read", "repository_replace_symbol": "edit", "repository_refresh": "refresh",
    "repository_cache_inventory": "cache_inventory",
    "repository_context_query": "context_query", "repository_capsule_expand": "capsule_expand",
}


def _empty_counts(counter):
    return dict(messages=0, requests=0, responses=0, request_bytes=0, response_bytes=0,
                local_tokenizer_proxy_tokens=0 if counter else None,
                request_local_tokenizer_proxy_tokens=0 if counter else None,
                response_local_tokenizer_proxy_tokens=0 if counter else None)


def _component(method, params):
    if method == "tools/call":
        name = params.get("name") if type(params) is dict else None
        if name not in _TOOL_COMPONENTS:
            raise ValueError("unknown repository tool accounting component")
        return _TOOL_COMPONENTS[name]
    if method not in {"initialize", "notifications/initialized", "tools/list", "ping"}:
        raise ValueError("unknown repository protocol accounting component")
    return "setup"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode() + b"\n"


class _Client:
    """One bounded subprocess; only aggregate counters survive each exchange."""
    def __init__(self, config, counter):
        from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_mcp
        self.counter = counter
        self.sequence = 0
        self.deadline = time.monotonic() + 120
        self.counts = {name: _empty_counts(counter) for name in ("setup", "workflow")}
        self.components = {name: _empty_counts(counter) for name in WIRE_COMPONENTS}
        self.accounting_errors = []
        self.tokenizer_failed = False
        self.closed = False
        self.exit_code = None
        script = Path(repository_source_mcp.__file__).resolve()
        raw = config.read_bytes()
        self.process = subprocess.Popen([sys.executable, str(script), "--config", str(config),
            "--config-sha256", _sha(raw)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, cwd=config.parent,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})

    def _incomplete(self, reason):
        if reason not in self.accounting_errors:
            self.accounting_errors.append(reason)

    def _count(self, phase, component, raw, *, response):
        # One exact wire message belongs to one lifecycle phase and one
        # operation component. These are two partitions, never added together.
        for row in (self.counts[phase], self.components[component]):
            row["messages"] += 1
            row["responses" if response else "requests"] += 1
            row["response_bytes" if response else "request_bytes"] += len(raw)
        # Bytes have already crossed the client boundary. Tokenizer failures
        # must not erase them or leave apparently complete partial token sums.
        if self.counter is not None and not self.tokenizer_failed:
            try:
                tokens = self.counter(raw.decode("utf-8"))
                if type(tokens) is not int or tokens <= 0:
                    raise ValueError("invalid local token count")
            except Exception:
                self.tokenizer_failed = True
                self._incomplete("local_tokenizer_failed")
                for row in (*self.counts.values(), *self.components.values()):
                    for key in row:
                        if key.endswith("tokens"):
                            row[key] = None
                raise ValueError("repository wire tokenizer failed") from None
            for row in (self.counts[phase], self.components[component]):
                row["local_tokenizer_proxy_tokens"] += tokens
                row["response_local_tokenizer_proxy_tokens" if response
                    else "request_local_tokenizer_proxy_tokens"] += tokens

    def request(self, method, params=None, *, phase="workflow", notification=False):
        component = _component(method, params)
        if phase not in self.counts:
            raise ValueError("unknown repository accounting phase")
        self.sequence += 1
        message = dict(jsonrpc="2.0", method=method)
        if params is not None:
            message["params"] = params
        if not notification:
            message["id"] = self.sequence
        raw = _wire(message)
        try:
            try:
                written = self.process.stdin.write(raw)
                self.process.stdin.flush()
            except Exception:
                # A failed buffered write/flush does not reveal how many bytes
                # reached the peer. Do not invent a byte count for that prefix.
                self._incomplete("request_write_unconfirmed")
                raise
            if type(written) is not int or not 0 <= written <= len(raw):
                self._incomplete("request_write_unconfirmed")
                raise ValueError("repository tool subprocess request write")
            if written:
                self._count(phase, component, raw[:written], response=False)
            if written != len(raw):
                self._incomplete("request_write_partial")
                raise ValueError("repository tool subprocess partial request write")
            if notification:
                return None
            remaining = min(45, self.deadline - time.monotonic())
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                self._incomplete("response_deadline")
                raise ValueError("repository tool subprocess response deadline")
            raw = self.process.stdout.readline(MAX_RESPONSE_BYTES + 1)
            if raw:
                self._count(phase, component, raw, response=True)
            if not raw.endswith(b"\n") or len(raw) > MAX_RESPONSE_BYTES:
                self._incomplete("response_framing")
                raise ValueError("repository tool subprocess response framing")
            response = json.loads(raw)
            if response.get("id") != self.sequence or "error" in response:
                raise ValueError("repository tool subprocess protocol failure")
            return response["result"]
        except Exception:
            self._incomplete("exchange_incomplete")
            raise

    def tool(self, name, arguments=None, *, phase="workflow", expect_error=False):
        result = self.request("tools/call", {"name": name, "arguments": arguments or {}}, phase=phase)
        if result.get("isError", False) is not expect_error:
            self._incomplete("unexpected_tool_disposition")
            raise ValueError("repository tool unexpected success/error disposition")
        blocks = result.get("content")
        if type(blocks) is not list or len(blocks) != 1 or blocks[0].get("type") != "text":
            self._incomplete("invalid_tool_response")
            raise ValueError("repository tool response is not one bounded JSON text block")
        try:
            return json.loads(blocks[0]["text"])
        except Exception:
            self._incomplete("invalid_tool_response")
            raise

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            code = self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                code = self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                code = self.process.wait(timeout=5)
        if self.process.stdout:
            self.process.stdout.close()
        self.closed, self.exit_code = True, code
        if code != 0:
            self._incomplete("subprocess_exit_failed")
        return code

    def metrics(self):
        def rendered(counts):
            return {**counts, "rendered_json_bytes": counts["request_bytes"] + counts["response_bytes"]}
        result = {phase: rendered(counts) for phase, counts in self.counts.items()}
        result["total"] = {key: (None if any(result[phase][key] is None for phase in self.counts)
                                 else sum(result[phase][key] for phase in self.counts))
                           for key in result["setup"]}
        result["components"] = {name: rendered(counts) for name, counts in self.components.items()}
        result["component_totals_match"] = all(
            result["total"][key] == (None if any(row[key] is None for row in result["components"].values())
                else sum(row[key] for row in result["components"].values()))
            for key in result["total"])
        result["wire_accounting_complete"] = self.closed and self.exit_code == 0 and not self.accounting_errors
        result["token_accounting_complete"] = bool(result["wire_accounting_complete"]
            and self.counter is not None and not self.tokenizer_failed)
        result["accounting_errors"] = list(self.accounting_errors)
        result["accounting_scope"] = ("Each transmitted JSON-RPC line counted once; phase and component "
            "totals are alternative partitions. Token counts are independent per-message local proxies, "
            "excluding provider framing, hidden tool schemas, history replay and model output.")
        result["traffic_scope"] = ("Confirmed client request writes and response prefixes read by this client; "
            "an incomplete exchange can leave unobserved subprocess traffic. Failed writes have unknown "
            "delivery. Message counts include observed partial frames, not only valid JSON-RPC lines.")
        return result


def _probe(reads):
    """Choose one explicit top-level authored syntax edit, never a solution."""
    for path, response in sorted(reads.items()):
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(_read_bytes(response), type_comments=True)
        except (SyntaxError, ValueError, UnicodeError):
            continue
        node = next((item for item in tree.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
        if node is not None:
            node.body.append(ast.Pass())
            return dict(path=path, symbol=node.name,
                        replacement=ast.unparse(ast.fix_missing_locations(node)) + "\n",
                        expected_ast=ast.dump(tree, include_attributes=False))
    return None


def _arm(*, sources, representation, counter):
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_mcp import summarize_repository_audit
    with tempfile.TemporaryDirectory(prefix="repository-tools-public-", dir="/tmp") as temporary:
        root = Path(temporary)
        workspace, private = root / "workspace", root / "private"
        _snapshot(workspace, sources)
        private.mkdir(mode=0o700)
        (private / "state").mkdir(mode=0o700)
        config = private / "config.json"
        config.write_bytes(_wire(dict(schema="repository-source-mcp-config@1", workspace=str(workspace),
            state_dir=str(private / "state"), audit_store=str(private / "operations.jsonl"),
            representation=representation)))
        config.chmod(0o600)
        client = _Client(config, counter)
        checks, refusals = {}, {}
        status, final_status, final_hashes, failure = {}, {}, {}, None
        edit_count = compacted = 0
        try:
            initialized = client.request("initialize", dict(protocolVersion="2025-06-18", capabilities={},
                clientInfo={"name": "offline-public-tools-qualification", "version": "1"}), phase="setup")
            checks["native_mcp_initialized"] = initialized["serverInfo"]["name"] == "repository_source"
            client.request("notifications/initialized", phase="setup", notification=True)
            advertised = client.request("tools/list", phase="setup")
            checks["repository_tools_advertised"] = {row["name"] for row in advertised["tools"]} == {
                "repository_status", "repository_list", "repository_symbols", "repository_read",
                "repository_replace_symbol", "repository_refresh", "repository_cache_inventory",
                "repository_context_query", "repository_capsule_expand"}
            status = client.tool("repository_status", phase="setup")
            generation = status["generation"]
            checks["initial_native_index_current"] = status["source_current"] and status["native_index"]["source_scope_verified"]
            listed = client.tool("repository_list", dict(generation=generation, limit=100))
            paths = {row["path"] for row in listed["entries"] if row["kind"] == "file"}
            checks["same_reviewed_file_population"] = paths == set(sources) and listed["next_offset"] is None
            symbol_rows = client.tool("repository_symbols", dict(generation=generation, limit=100))
            checks["native_symbols_in_reviewed_scope"] = all(row["path"] in sources for row in symbol_rows["symbols"])
            reads = {path: client.tool("repository_read", dict(generation=generation, path=path)) for path in sorted(sources)}
            checks["reads_bound_to_original_hashes"] = all(row["source_sha256"] == _sha(sources[path]) for path, row in reads.items())
            checks["read_representation_matches"] = all(row["representation"] == representation for row in reads.values())
            checks["raw_reads_exact_or_compact_ast_equal"] = all(
                _read_bytes(row) == sources[path] or (path.endswith(".py") and _ast(_read_bytes(row)) == _ast(sources[path]))
                for path, row in reads.items())
            compacted = sum(_read_bytes(row) != sources[path] for path, row in reads.items())
            probe = _probe(reads)
            edit_count = 0
            changed_path = sorted(sources)[0]
            if probe is not None:
                changed_path = probe["path"]
                edit = client.tool("repository_replace_symbol", dict(generation=generation,
                    path=probe["path"], symbol=probe["symbol"], replacement=probe["replacement"],
                    expected_sha256=reads[probe["path"]]["source_sha256"]))
                edit_count = 1
                checks["authored_edit_refreshed"] = edit["canonical_changed"] is True and not edit["refresh_required"]
                new_generation = edit["generation"]
                stale = client.tool("repository_read", dict(generation=generation, path=changed_path), expect_error=True)
                refusals["old_generation"] = stale["error"]
                checks["old_generation_refused"] = stale["error"] == "stale_generation"
                generation = new_generation
                after = client.tool("repository_read", dict(generation=generation, path=changed_path))
                checks["only_authored_ast_pass_added"] = _ast(_read_bytes(after)) == probe["expected_ast"]
                checks["canonical_edit_matches_view"] = _ast((workspace / changed_path).read_bytes()) == probe["expected_ast"]
            # Observe an outside write to the disposable checkout, then require
            # explicit refresh. This is a source freshness probe, not execution.
            with (workspace / changed_path).open("ab") as stream:
                stream.write(b"\n# Authored external freshness probe.\n")
            stale = client.tool("repository_read", dict(generation=generation, path=changed_path), expect_error=True)
            refusals["external_source_change"] = stale["error"]
            checks["external_change_refused"] = stale["error"] == "stale_source"
            refreshed = client.tool("repository_refresh", dict(generation=generation))
            generation = refreshed["generation"]
            final = client.tool("repository_read", dict(generation=generation, path=changed_path, representation="original"))
            checks["explicit_refresh_delivers_current_original"] = _read_bytes(final) == (workspace / changed_path).read_bytes()
            checks["unchanged_files_byte_exact"] = all((workspace / path).read_bytes() == raw
                for path, raw in sources.items() if path != changed_path)
            final_hashes = {path: _sha((workspace / path).read_bytes()) for path in sorted(sources)}
            final_status = client.tool("repository_status")
        except Exception as error:
            failure = {"error_type": type(error).__name__}
            checks["workflow_completed"] = False
            client._incomplete("workflow_incomplete")
        finally:
            exit_code = client.close()
        audit = summarize_repository_audit(private / "operations.jsonl")
        checks["subprocess_clean_exit"] = exit_code == 0
        checks["actual_reads_observed"] = type(audit["successful_reads"]) is int and audit["successful_reads"] >= len(sources)
        checks["actual_compact_reads_observed"] = representation != "compact" or (
            type(audit["compact_reads"]) is int and audit["compact_reads"] >= len(sources))
        return dict(representation=representation, qualified=all(checks.values()), checks=checks,
            protocol=client.metrics(), audit=audit, refusal_codes=refusals, public_source_edits=edit_count,
            compacted_file_reads=compacted, final_source_hashes=final_hashes,
            native_coverage=status.get("index_coverage"), final_native_coverage=final_status.get("index_coverage"),
            final_ast_stats=final_status.get("native_index", {}).get("ast_stats"), failure=failure,
            content_bearing_messages_retained=False, subprocess_exit_code=exit_code)


def run_experiment(*, dataset: Path, output: Path, tasks=DEFAULT_TASKS, tokenizer_cache: Path | None = None):
    """Read only catalog-approved public metadata and COPY inputs; keep all arms."""
    tasks = tuple(tasks)
    if not tasks or len(tasks) > MAX_TASKS or len(set(tasks)) != len(tasks):
        raise ValueError("tasks must be a bounded, nonempty unique selection")
    dataset = Path(dataset).resolve(strict=True)
    loaded, total = [], 0
    for name in tasks:
        profile = catalog.reviewed_profile(dataset=dataset, task_name=name)
        rows = profile["bindings"]["inputs"]
        if not rows or len(rows) > MAX_INPUTS:
            raise ValueError("public input count must be bounded and nonempty")
        sources = {}
        for row in rows:
            _relative(row["path"])
            raw = catalog._read(dataset / name / "environment" / row["environment_path"], row["sha256"])
            total += len(raw)
            if total > MAX_TOTAL_BYTES or row["path"] in sources:
                raise ValueError("public inputs exceed bounds or contain duplicate paths")
            sources[row["path"]] = raw
        loaded.append((name, profile, sources))
    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("experiment output must be a new canonical directory")
    if output == dataset or dataset in output.parents:
        raise ValueError("experiment output must be outside the source dataset")
    counter, tokenizer = local_token_counter(tokenizer_cache)
    output.mkdir(parents=True)
    results = []
    for name, profile, sources in loaded:
        arms = {}
        for representation in ("original", "compact"):
            try:
                arms[representation] = _arm(sources=sources, representation=representation, counter=counter)
            except Exception as error:
                # Retain failures without saving content-bearing exception text.
                arms[representation] = dict(qualified=False, error_type=type(error).__name__,
                    content_bearing_messages_retained=False)
        paired = all(arm["qualified"] for arm in arms.values()) and (
            arms["original"]["final_source_hashes"] == arms["compact"]["final_source_hashes"])
        results.append(dict(task_name=name, qualified=paired, arms=arms,
            paired_final_canonical_sources_identical=paired, file_count=len(sources),
            input_sha256={path: _sha(raw) for path, raw in sorted(sources.items())},
            dataset_revision_reviewed=profile["dataset_revision_reviewed"],
            public_metadata_sha256=profile["bindings"]["public_metadata"],
            source_boundary=profile["source_boundary"], hidden_verifier_or_solution_read=False))
    unchanged = all(catalog._read(dataset / name / "environment" / row["environment_path"], row["sha256"]) == sources[row["path"]]
        for name, profile, sources in loaded for row in profile["bindings"]["inputs"])
    result = dict(schema=SCHEMA, qualified=unchanged and all(row["qualified"] for row in results),
        tasks=results, original_dataset_unchanged=unchanged, local_tokenizer=tokenizer,
        provider_calls=0, containers_started=0, benchmark_source_executed=False,
        provider_usage_tokens=None, official_reward=None, content_bearing_messages_retained=False,
        measurement_scope="actual JSON-RPC request/response bytes including newlines; phase and operation partitions with transmission counts; per-message local token proxies",
        limitations=["No model solving, provider framing/usage, reasoning, retries, task success or official verifier is measured.",
            "Fixed public source selection and authored syntax edits; neither arm solves the task.",
            "Real stdio tool use is scripted; live model tool adoption and total-session token savings remain unproven.",
            "Native AST cache reuse is measured; semantic graph reconstruction remains fresh.",
            "Independent private session identities produce different hashes; proxy totals include this representation overhead."])
    (output / "qualification.json").write_text(_render(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--tokenizer-cache", type=Path)
    args = parser.parse_args(argv)
    result = run_experiment(dataset=args.dataset, output=args.output, tasks=args.tasks or DEFAULT_TASKS,
                            tokenizer_cache=args.tokenizer_cache)
    print(_render(dict(schema=SCHEMA, qualified=result["qualified"], provider_calls=0,
                       output=str(args.output))), end="")
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
