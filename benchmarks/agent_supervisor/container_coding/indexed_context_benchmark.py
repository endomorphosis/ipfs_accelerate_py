"""Offline native context/expansion benchmark; never a Terminal Bench score.

Requires an existing o200k_base cache. No tokenizer download or provider call
is permitted. Counts cover worker JSON and expansion responses only.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time


def _offline_counter(cache: Path):
    import tiktoken
    import tiktoken.load

    asset = cache / "fb374d419588a4632f3f557e76b4b70aebbca790"
    expected = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
    if asset.stat().st_size > 8_000_000 or hashlib.sha256(asset.read_bytes()).hexdigest() != expected:
        raise ValueError("an existing hash-verified o200k_base cache is required")
    previous = os.environ.get("TIKTOKEN_CACHE_DIR")
    original_read = tiktoken.load.read_file
    def refuse_download(_):
        raise RuntimeError("offline context benchmark forbids tokenizer downloads")
    try:
        os.environ["TIKTOKEN_CACHE_DIR"] = str(cache)
        tiktoken.load.read_file = refuse_download
        encoding = tiktoken.get_encoding("o200k_base")
    finally:
        tiktoken.load.read_file = original_read
        if previous is None:
            os.environ.pop("TIKTOKEN_CACHE_DIR", None)
        else:
            os.environ["TIKTOKEN_CACHE_DIR"] = previous
    return (lambda text: len(encoding.encode(text, disallowed_special=()))), {
        "encoding": "o200k_base", "tiktoken_version": version("tiktoken"),
        "asset_sha256": expected, "remote_model_equivalence_asserted": False,
    }


def run(*, output: Path, tokenizer_cache: Path) -> dict:
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import (
        INDEXED_SELECTOR, load_semantic_worker_context, prepare_semantic_context,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_capsule_expansion import (
        expand_semantic_capsules,
    )

    count, tokenizer = _offline_counter(tokenizer_cache)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for size in (12, 24, 48):
        root = output / f"symbols-{size}"
        root.mkdir()
        (root / "instruction.txt").write_text("Preserve the public addition behavior of add_7.\n")
        (root / "code.py").write_text("\n".join(
            f"def add_{i}(value: int) -> int:\n    return value + {i}\n" for i in range(size)))
        args = dict(repository=root, paths=["code.py", "instruction.txt"],
            program_paths=["code.py"], required_raw_paths=["instruction.txt"],
            objective="Preserve addition in add_7", task_id=f"context-{size}",
            context_input_tokens=128000)
        variants = {}
        for name in ("full", "indexed"):
            start = time.monotonic()
            prepared = prepare_semantic_context(**args, output=root / name,
                worker_query="add_7" if name == "indexed" else "",
                worker_selector=INDEXED_SELECTOR if name == "indexed" else "lexical-query-symbol-name@1")
            artifact = f"{name}/worker-context.json"
            load_args = dict(repository=root, artifact=artifact,
                expected_sha256=prepared["worker_payload_sha256"], task_id=args["task_id"])
            text = load_semantic_worker_context(**load_args)
            variants[name] = dict(payload=json.loads(text), load_args=load_args,
                tokens=count(text), bytes=len(text.encode()), seconds=time.monotonic() - start)
        full, indexed = variants["full"], variants["indexed"]
        expected = {item["stable_symbol_id"]: item for item in full["payload"]["capsules"]}
        selected = {item["stable_symbol_id"] for item in indexed["payload"]["capsules"]}
        omitted = sorted(set(expected) - selected)
        expansion_tokens, calls = 0, 0
        for offset in range(0, len(omitted), 4):
            response = expand_semantic_capsules(**indexed["load_args"],
                stable_symbol_ids=omitted[offset:offset + 4], max_bytes=65536)
            for capsule in response["capsules"]:
                if capsule != expected[capsule["stable_symbol_id"]]:
                    raise AssertionError("expanded capsule differs from full native context")
            expansion_tokens += count(json.dumps(response, sort_keys=True, ensure_ascii=False,
                separators=(",", ":")))
            calls += 1
        if indexed["payload"]["raw_sources"]["instruction.txt"] != (root / "instruction.txt").read_text():
            raise AssertionError("required public instruction changed")
        rows.append(dict(symbol_functions=size, full_capsules=len(expected),
            initial_capsules=len(selected), omitted_capsules=len(omitted),
            full_worker_tokens=full["tokens"], indexed_worker_tokens=indexed["tokens"],
            initial_worker_reduction_percent=100 * (full["tokens"] - indexed["tokens"]) / full["tokens"],
            full_worker_bytes=full["bytes"], indexed_worker_bytes=indexed["bytes"],
            all_omitted_expansions_exact=True, expansion_responses=calls,
            all_expansion_response_tokens=expansion_tokens,
            indexed_plus_all_expansion_response_tokens=indexed["tokens"] + expansion_tokens,
            full_prepare_and_load_seconds=full["seconds"], indexed_prepare_and_load_seconds=indexed["seconds"]))
    report = dict(schema="supervisor-indexed-context-local-benchmark@1", tokenizer=tokenizer,
        fixtures=rows, provider_calls=0, provider_token_usage=None,
        terminal_bench_success_measured=False,
        limitations=["Synthetic source fixtures; no task-solving or official verifier run.",
            "Local worker/expansion representation counts exclude other context, tools, framing, and later model turns.",
            "Expansion response sums are not cumulative provider usage; repeated history can cost more.",
            "Initial context savings are not end-to-end token savings.",
            "Sequential cold preparations share process caches; timings are diagnostic only."])
    (output / "benchmark.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-cache", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(output=args.output.resolve(), tokenizer_cache=args.tokenizer_cache.resolve()),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
