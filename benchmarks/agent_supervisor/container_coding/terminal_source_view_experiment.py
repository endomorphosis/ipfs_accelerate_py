"""Offline source-view qualification on pinned, public Terminal Bench inputs.

This experiment makes no provider requests and does not solve benchmark tasks.
It compares the same source selections and authored edit in raw/compact views,
including the complete JSON protocol in its byte and optional tokenizer totals.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile

from . import terminal_profile_catalog as catalog

SCHEMA = "terminal-source-view-experiment@1"
DEFAULT_TASKS = ("tune-mjcf", "llm-inference-batching-scheduler")
MAX_TASKS = 8
MAX_INPUTS = 64
MAX_TOTAL_BYTES = 8 * 1024 * 1024
O200K_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
O200K_CACHE_KEY = "fb374d419588a4632f3f557e76b4b70aebbca790"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _render(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _save(path: Path, value: dict) -> None:
    path.write_text(_render(value), encoding="utf-8")


def local_token_counter(cache_file: Path | None = None):
    """Use verified local ranks directly; never call a tokenizer downloader."""
    if cache_file is None:
        cache = os.environ.get("TIKTOKEN_CACHE_DIR") or os.environ.get("DATA_GYM_CACHE_DIR")
        cache_file = Path(cache or Path(tempfile.gettempdir()) / "data-gym-cache") / O200K_CACHE_KEY
        if not cache_file.is_file():
            return None, {"available": False, "reason": "verified_local_o200k_cache_unavailable"}
    cache_file = Path(cache_file)
    if cache_file.is_symlink() or not cache_file.is_file():
        raise ValueError("tokenizer ranks must be a local regular file")
    with cache_file.open("rb") as stream:
        raw = stream.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024 or _sha(raw) != O200K_SHA256:
        raise ValueError("local tokenizer ranks do not match the pinned o200k encoding")
    try:
        import tiktoken
    except ImportError:
        return None, {"available": False, "reason": "tiktoken_not_installed"}
    ranks = {base64.b64decode(token): int(rank) for token, rank in
        (line.split() for line in raw.splitlines() if line)}
    pattern = "|".join([
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}", r" ?[^\s\p{L}\p{N}]+[\r\n/]*", r"\s*[\r\n]+", r"\s+(?!\S)", r"\s+",
    ])
    encoding = tiktoken.Encoding(name="o200k_base_local_proxy", pat_str=pattern,
        mergeable_ranks=ranks, special_tokens={"<|endoftext|>": 199999, "<|endofprompt|>": 200018})
    return lambda text: len(encoding.encode(text, disallowed_special=())), {
        "available": True, "encoding": "o200k_base", "ranks_sha256": O200K_SHA256,
        "scope": "local tokenizer proxy; not provider usage, billing, or message framing",
        "download_performed": False,
    }


def _metrics(values: list[dict], counter) -> dict:
    rendered = [_render(value) for value in values]
    return {"protocol_messages": len(values), "rendered_json_bytes": sum(len(v.encode("utf-8")) for v in rendered),
        "local_tokenizer_proxy_tokens": sum(counter(v) for v in rendered) if counter else None}


def _relative(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or any(part in {".", ".."} or part.startswith(".") for part in path.parts):
        raise ValueError("public input path must be a visible relative path")
    return path


def _snapshot(root: Path, sources: dict[str, bytes]) -> None:
    root.mkdir(parents=True)
    for name, raw in sources.items():
        path = root / _relative(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)


def _ast_equal(left: str | bytes, right: str | bytes) -> bool:
    return ast.dump(ast.parse(left), include_attributes=False) == ast.dump(ast.parse(right), include_attributes=False)


def _request(operation: str, **kwargs) -> dict:
    return {"operation": operation, **kwargs}


def _task_experiment(*, name: str, sources: dict[str, bytes], output: Path, counter, metadata: dict) -> dict:
    from ipfs_accelerate_py.agent_supervisor.runtime.code_source_view import CodeSourceView, SourceViewError

    output.mkdir()
    python_paths = sorted(path for path in sources if path.endswith(".py"))
    original_hashes = {path: _sha(raw) for path, raw in sorted(sources.items())}
    _snapshot(output / "canonical-before", sources)
    result = {"task_name": name, **metadata, "input_sha256": original_hashes,
        "python_paths": python_paths, "unsupported_input_paths": sorted(set(sources) - set(python_paths)),
        "coverage": "all reviewed Python modules; first top-level function per nonempty module for edit qualification",
        "restoration_mechanism": "retained canonical bytes; compact AST rendering is not a standalone lossless encoding",
        "provider_calls": 0, "containers_started": 0, "official_reward": None, "task_solved": None,
        "raw": {}, "compact": {}}
    if not python_paths:
        result["disposition"] = "no_supported_python_input"
        _save(output / "qualification.json", result)
        return result
    replacements = {}
    for path in python_paths:
        node = next((item for item in ast.parse(sources[path]).body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
        if node is not None:
            symbol = node.name
            # This authored pass statement changes the AST without solving the task.
            node.body.append(ast.Pass())
            replacements[path] = (symbol, ast.unparse(ast.fix_missing_locations(node)) + "\n")

    for view_format in ("raw", "compact"):
        workspace = output / view_format / "workspace"
        _snapshot(workspace, sources)
        engine = CodeSourceView(workspace)
        index_response = engine.index(python_paths)
        artifact = {"index": index_response, "module_reads": [], "cycles": []}
        module_messages = [_request("index", paths=python_paths), index_response]
        cycle_messages = [_request("index", paths=python_paths), index_response]
        complete_messages = [_request("index", paths=python_paths), index_response]
        workflow_messages = [_request("index", paths=python_paths), index_response]
        checks = {"byte_exact_initial_copy": all((workspace / path).read_bytes() == raw for path, raw in sources.items()),
            "compact_view_ast_equivalent": True, "no_op_round_trip_byte_exact": True,
            "stale_edits_rejected": True, "refresh_uses_changed_hash": True,
            "public_probe_is_only_pass_append": True, "all_canonical_python_syntax_valid": True,
            "canonical_outside_selected_symbol_preserved": True}
        for path in python_paths:
            response = engine.read(path, symbol="module", format=view_format)
            artifact["module_reads"].append(response)
            messages = [_request("read", path=path, symbol="module", format=view_format), response]
            module_messages.extend(messages)
            complete_messages.extend(messages)
            workflow_messages.extend(messages)
            checks["compact_view_ast_equivalent"] &= _ast_equal(sources[path], response["content"])
        for path, (symbol, replacement) in replacements.items():
            before = engine.read(path, symbol=symbol, format=view_format)
            canonical_before = (workspace / path).read_bytes()
            # Re-submitting the current view must not lose comments/formatting.
            no_op_request = _request("replace", path=path, symbol=symbol,
                expected_sha256=before["source_sha256"], replacement=before["content"], format=view_format)
            no_op = engine.replace(path, symbol=symbol, expected_sha256=before["source_sha256"],
                replacement=before["content"], format=view_format)
            checks["no_op_round_trip_byte_exact"] &= (workspace / path).read_bytes() == canonical_before
            change = engine.replace(path, symbol=symbol, expected_sha256=before["source_sha256"],
                replacement=replacement, format=view_format)
            after = engine.read(path, symbol=symbol, format=view_format)
            checks["refresh_uses_changed_hash"] &= before["source_sha256"] != after["source_sha256"]
            checks["public_probe_is_only_pass_append"] &= _ast_equal(after["content"], replacement)
            original_tree = ast.parse(canonical_before)
            resulting_tree = ast.parse((workspace / path).read_bytes())
            expected_tree = ast.parse(canonical_before)
            next(n for n in expected_tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == symbol).body.append(ast.Pass())
            checks["public_probe_is_only_pass_append"] &= ast.dump(expected_tree, include_attributes=False) == ast.dump(resulting_tree, include_attributes=False)
            node = next(n for n in original_tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == symbol)
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            old_lines = canonical_before.splitlines(keepends=True)
            edited = (workspace / path).read_bytes()
            prefix, suffix = b"".join(old_lines[:start]), b"".join(old_lines[node.end_lineno:])
            checks["canonical_outside_selected_symbol_preserved"] &= edited.startswith(prefix) and edited.endswith(suffix)
            stale_error = None
            try:
                engine.replace(path, symbol=symbol, expected_sha256=before["source_sha256"],
                    replacement=replacement, format=view_format)
            except SourceViewError as exc:
                stale_error = exc.code
            checks["stale_edits_rejected"] &= stale_error == "stale_source"
            artifact["cycles"].append({"path": path, "symbol": symbol, "before": before, "replace": change,
                "after": after, "no_op_round_trip": no_op, "stale_error": stale_error})
            before_messages = [_request("read", path=path, symbol=symbol, format=view_format), before]
            change_request = _request("replace", path=path, symbol=symbol,
                expected_sha256=before["source_sha256"], replacement=replacement, format=view_format)
            change_messages = [change_request, change,
                _request("read", path=path, symbol=symbol, format=view_format), after]
            cycle_messages.extend(before_messages + change_messages)
            workflow_messages.extend(before_messages + change_messages)
            complete_messages.extend(before_messages + [no_op_request, no_op] + change_messages
                + [change_request, {"schema": "code-source-view@1", "error": stale_error,
                    "source_content_disclosed": False}])
        for path in python_paths:
            compile((workspace / path).read_bytes(), path, "exec")
        artifact["complete_messages"] = complete_messages
        (output / view_format / "protocol.json").write_text(_render(artifact), encoding="utf-8")
        result[view_format] = {"module_read_protocol": _metrics(module_messages, counter),
            "edit_cycle_protocol": _metrics(cycle_messages, counter),
            "normal_workflow_protocol": _metrics(workflow_messages, counter),
            "complete_qualification_protocol": _metrics(complete_messages, counter), "checks": checks,
            "edit_cycles": len(replacements), "final_sha256": {path: _sha((workspace / path).read_bytes()) for path in sources}}
    result["paired_final_canonical_bytes_identical"] = result["raw"]["final_sha256"] == result["compact"]["final_sha256"]
    result["retained_original_bytes_unchanged"] = all((output / "canonical-before" / path).read_bytes() == raw for path, raw in sources.items())
    result["qualified"] = result["paired_final_canonical_bytes_identical"] and result["retained_original_bytes_unchanged"] and all(
        all(result[fmt]["checks"].values()) for fmt in ("raw", "compact"))
    _save(output / "qualification.json", result)
    return result


def run_experiment(*, dataset: Path, output: Path, tasks=DEFAULT_TASKS, tokenizer_cache: Path | None = None) -> dict:
    """Qualify all inputs before creating any output, then operate on copies."""
    tasks = tuple(tasks)
    if not tasks or len(tasks) > MAX_TASKS or len(set(tasks)) != len(tasks):
        raise ValueError("tasks must be a bounded, nonempty unique selection")
    selected = [(name, catalog.reviewed_profile(dataset=dataset, task_name=name)) for name in tasks]
    loaded = []
    total = 0
    for name, profile in selected:
        if len(profile["bindings"]["inputs"]) > MAX_INPUTS:
            raise ValueError("public profile exceeds input count bound")
        sources = {}
        for item in profile["bindings"]["inputs"]:
            _relative(item["path"])
            raw = catalog._read(Path(dataset).resolve(strict=True) / name / "environment" / item["environment_path"], item["sha256"])
            total += len(raw)
            if total > MAX_TOTAL_BYTES:
                raise ValueError("public profile aggregate bytes exceed experiment bound")
            if item["path"] in sources:
                raise ValueError("duplicate public input path")
            sources[item["path"]] = raw
        loaded.append((name, profile, sources))
    counter, tokenizer = local_token_counter(tokenizer_cache)
    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("experiment output must be a new canonical directory")
    dataset_root = Path(dataset).resolve(strict=True)
    if output == dataset_root or dataset_root in output.parents:
        raise ValueError("experiment output must be outside the source dataset")
    output.mkdir(parents=True)
    results = [_task_experiment(name=name, sources=sources, output=output / name, counter=counter,
        metadata={"dataset_revision_reviewed": profile["dataset_revision_reviewed"],
            "public_metadata_sha256": profile["bindings"]["public_metadata"],
            "source_boundary": profile["source_boundary"], "hidden_verifier_or_solution_read": False})
        for name, profile, sources in loaded]
    # Independent authored behavior test; no benchmark source is imported/executed.
    authored_source = b"# Preserve this canonical comment.\ndef increment(value):\n    # A stable authored fixture.\n    return value + 1\n\nSENTINEL = 'unchanged'\n"
    authored = output / "authored-behavior"
    _snapshot(authored, {"fixture.py": authored_source})
    from ipfs_accelerate_py.agent_supervisor.runtime.code_source_view import CodeSourceView, SourceViewError
    engine = CodeSourceView(authored)
    before = engine.read("fixture.py", symbol="increment", format="compact")
    change = engine.replace("fixture.py", symbol="increment", expected_sha256=before["source_sha256"],
        replacement="def increment(value):\n    return value + 2\n")
    after = engine.read("fixture.py", symbol="increment", format="compact")
    namespace = {}
    exec(compile((authored / "fixture.py").read_bytes(), "authored-fixture", "exec"), namespace)
    authored_result = {"schema": "terminal-source-view-authored-behavior@1", "source_sha256": _sha(authored_source),
        "behavior_change_verified": namespace["increment"](5) == 7,
        "sentinel_preserved": namespace["SENTINEL"] == "unchanged",
        "refresh_uses_changed_hash": before["source_sha256"] != after["source_sha256"],
        "provider_calls": 0, "benchmark_source_executed": False, "replace": change}
    _save(authored / "qualification.json", authored_result)
    original_unchanged = all(catalog._read(dataset_root / name / "environment" / item["environment_path"], item["sha256"]) == sources[item["path"]]
        for name, profile, sources in loaded for item in profile["bindings"]["inputs"])
    result = {"schema": SCHEMA, "tasks": results, "authored_behavior": authored_result,
        "provider_calls": 0, "containers_started": 0, "original_dataset_unchanged": original_unchanged,
        "local_tokenizer": tokenizer, "provider_usage_tokens": None, "official_reward": None,
        "qualified": original_unchanged and authored_result["behavior_change_verified"] and authored_result["sentinel_preserved"]
            and authored_result["refresh_uses_changed_hash"] and all(row.get("qualified", False) for row in results),
        "measurement_scope": "identical selected modules and symbols; complete sorted compact JSON requests and responses, including index and replacement text",
        "excluded_qualification_costs": "normal workflow/edit-cycle totals exclude safety-only no-op and deliberate stale rejection; complete qualification totals include them",
        "limitations": ["No live model solving, provider tokens, task success rate, or official reward measured.",
            "Python AST-equivalent views retain docstrings and identifiers; unsupported non-Python inputs remain canonical.",
            "Canonical source is retained externally; this does not reconstruct original comments or formatting from compact source alone.",
            "Public edit probes append a pass statement; they do not implement benchmark solutions.",
            "Protocol totals exclude model reasoning, instructions, shell framing, retries and later source expansion.",
            "This isolates source rendering; DuckDB selection, semantic capsule admission and proof solving are not varied."]}
    _save(output / "qualification.json", result)
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
    print(_render({"schema": SCHEMA, "qualified": result["qualified"], "provider_calls": 0,
        "tasks": [row["task_name"] for row in result["tasks"]], "output": str(args.output)}), end="")
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
