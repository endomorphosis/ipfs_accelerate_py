"""Measure retained prompt components and counters; never read model thoughts.

Byte measurements refer to the complete recorded task input, not the provider
harness or later internal turns. This producer performs no provider, database,
benchmark, reconstruction, tokenizer, training or network operation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat

MAX_BYTES = 64 * 1024 * 1024


def load(path):
    path = Path(path).absolute()
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.resolve(strict=True) != path or info.st_size > MAX_BYTES:
        raise ValueError("bounded canonical metadata file required")
    raw = path.read_bytes()
    if len(raw) != info.st_size or path.stat().st_mtime_ns != info.st_mtime_ns:
        raise ValueError("metadata changed during read")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate metadata key")
            value[key] = item
        return value
    value = json.loads(raw, object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite metadata")))
    if not isinstance(value, dict):
        raise ValueError("metadata object required")
    return value, {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def count(value):
    if type(value) is not int or value < 0:
        raise ValueError("observed nonnegative counter required")
    return value


def row(receipt):
    if receipt["trial_count"] != 1 or not receipt["complete_single_trial_receipt"]:
        raise ValueError("one completed retained supervisor trial required")
    trial = receipt["trials"][0]
    supervisor = trial["supervisor"]
    phases = []
    for invocation in supervisor["provider_invocations"]:
        native = invocation["native_rollout_usage"]
        if native["task_complete_observed"] is not True or native["usage_available"] is not True:
            raise ValueError("complete native session counters required for this accounting")
        counters = {key: count(native["usage"][key]) for key in
                    ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")}
        if counters["input_tokens"] + counters["output_tokens"] != counters["total_tokens"]:
            raise ValueError("native token totals differ")
        counters["uncached_input_tokens"] = counters["input_tokens"] - counters["cached_input_tokens"]
        if counters["uncached_input_tokens"] < 0:
            raise ValueError("cache count exceeds input")
        component = {
            "phase": invocation["phase"], "native_prompt_bytes": count(invocation["native_prompt_bytes"]),
            "router_prompt_bytes": count(invocation["router_prompt_bytes"]),
            "complete_model_prompt_bytes": count(invocation["model_prompt_bytes"]),
            "workspace_advisory_bytes": count(invocation["workspace_advisory_bytes"]),
            "native_token_count_records": count(native["observed_token_count_records"]),
            "usage": counters,
        }
        translation = invocation.get("semantic_translation")
        if translation is not None:
            doctor = invocation.get("doctor_residual_context") or {}
            instruction = invocation.get("public_instruction") or {}
            translated = count(translation["provider_prompt_bytes"])
            doctor_bytes = count(doctor["advisory_bytes"])
            instruction_bytes = count(instruction["block_bytes"])
            if translated + doctor_bytes + instruction_bytes != component["router_prompt_bytes"]:
                raise ValueError("appended component accounting differs")
            if component["router_prompt_bytes"] + component["workspace_advisory_bytes"] != component["complete_model_prompt_bytes"]:
                raise ValueError("final model input accounting differs")
            component.update(
                semantic_transport_bytes=translated,
                doctor_advisory_bytes=doctor_bytes,
                instruction_and_requirement_context_bytes=instruction_bytes,
                representation_bytes_saved=component["native_prompt_bytes"] - translated,
                net_complete_prompt_bytes_saved=component["native_prompt_bytes"] - component["complete_model_prompt_bytes"],
                identifier_aliases=count(translation["identifier_mappings"]),
                identifier_occurrences=count(translation["identifier_occurrences"]),
            )
        phases.append(component)
    total = {key: sum(phase["usage"][key] for phase in phases) for key in
             ("input_tokens", "cached_input_tokens", "uncached_input_tokens", "output_tokens", "total_tokens")}
    return {"trial": trial["trial"], "planning_strategy": supervisor["planning"]["planning_strategy"],
        "router_sessions": len(phases), "observed_official_reward": trial["reward"],
        "phases": phases, "complete_observed_usage": total,
        "prompt_byte_counts_are_provider_tokens": False,
        "cached_input_already_in_input_total": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--symbolic-ablation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new output required")
    direct, direct_binding = load(args.direct)
    ablation, ablation_binding = load(args.symbolic_ablation)
    primary, secondary = row(direct), row(ablation)
    if primary["planning_strategy"] != "direct" or primary["router_sessions"] != 2:
        raise ValueError("primary accounting must retain both planning and coding")
    payload = {
        "schema": "supervisor-planning-preserved-token-economy-accounting@1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "Reduce total tokens using metadata, capsules, formal results and minification while retaining LLM planning and coding.",
        "historical_direct_baseline": primary, "historical_symbolic_ablation_secondary_only": secondary,
        "inputs": [direct_binding, ablation_binding],
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "new_provider_calls": 0, "new_benchmark_runs": 0, "training_updates": 0,
        "new_context_projection_implemented": False,
        "planning_preserved_token_savings_measured": False,
        "limits": [
            "These are retained receipt measurements, not a new context-compaction experiment.",
            "The removed-planner ablation is not the requested primary optimization.",
            "Representation saving combines JSON unpacking and typed aliases, not pure identifier minification.",
            "Final initial prompt bytes exclude provider harness context and later internal history/tool outputs.",
            "No exact tokenizer estimate or provider dollar cost is claimed here.",
            "One router session can contain multiple internal model turns; token-count records are not API call counts.",
        ],
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"primary_router_sessions": primary["router_sessions"],
        "primary_total_tokens": primary["complete_observed_usage"]["total_tokens"],
        "new_provider_calls": 0, "planning_preserved_savings_measured": False}))


if __name__ == "__main__":
    main()
