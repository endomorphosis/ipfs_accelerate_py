"""Explicit learned GTE retrieval with coding600; preserves attempts 01–08.

Root-run recipes only: each explicit stage creates fresh retained evidence.

No stage runs merely by importing this module. Bundle/profile/prepare have no
model calls; execute launches one authorized Harbor trial with its native verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path("/home/barberb/lift_coding")
OUTPUT = ROOT / "artifacts/grok-recovery-20261006"
SOURCE = ROOT / ".worktrees/grok-recovery-20261006"
DATASETS = ROOT / ".worktrees/terminal-bounded-header-datasets-20261005"
KIT = ROOT / ".worktrees/terminal-supervisor-kit-20261005"
PYTHON = ROOT / ".venvs/terminal-bench-harbor/bin/python"
DATASET = ROOT / ".benchmarks/terminal-bench-2"
PROVIDER_PROFILE = "grok-4.7-cli-1.0.46@1"
RESOURCE_PROFILE = "source384-5cpu-20gib-coding600@1"
EMBEDDING_REVISION = "17e1f347d17fe144873b1201da91788898c639cd"
EMBEDDING_SNAPSHOT = Path("/home/barberb/.cache/huggingface/hub/models--thenlper--gte-small/snapshots") / EMBEDDING_REVISION
BASE = json.loads((ROOT / "artifacts/terminal-supervisor-blockers-20261005/build-03-command.json").read_text())
ENV = {**BASE["env"], "PYTHONPATH": BASE["env"]["PYTHONPATH"].replace(BASE["cwd"], str(SOURCE)).replace(str(ROOT / ".worktrees/ir-supervisor-contracts-datasets-20261004"), str(DATASETS))}
PINS = {str(DATASETS): "5171a632c6b9f0ecb2939d29d2ad74992cbfeb11",
        str(KIT): "a9b98beac1ef14278b4adf4f2289cd509530fb86"}


def snapshot(expected_source_head, *, strict=True):
    expected = {**PINS, str(SOURCE): expected_source_head}
    heads, statuses = {}, {}
    for path, pin in expected.items():
        heads[path] = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
        statuses[path] = subprocess.check_output(["git", "-C", path, "status", "--porcelain=v1", "--untracked-files=all"], text=True)
        if strict and (heads[path] != pin or statuses[path]):
            raise SystemExit("source pin or clean-worktree requirement failed: " + path)
    return {"source_heads": heads, "source_status": statuses}


def write_json(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("bundle", "profile", "prepare", "execute"))
    parser.add_argument("--expected-source-head", required=True)
    parser.add_argument("--task", choices=("tune-mjcf", "largest-eigenval"), default="tune-mjcf")
    parser.add_argument("--bundle-attempt", default="01")
    parser.add_argument("--trial-attempt", default="01")
    parser.add_argument("--intent-requirement-contract", type=Path)
    parser.add_argument("--resource-profile", default=RESOURCE_PROFILE,
                        choices=(RESOURCE_PROFILE,))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_source_head):
        parser.error("an exact committed source SHA is required")
    if not all(re.fullmatch(r"[0-9]{2}", label) for label in (args.bundle_attempt, args.trial_attempt)):
        parser.error("attempt labels must contain exactly two digits")
    source = snapshot(args.expected_source_head)
    bundle = OUTPUT / ("grok-bundle-" + args.bundle_attempt)
    trial = OUTPUT / ("grok-" + args.task + "-" + args.trial_attempt)
    task_profile = OUTPUT / ("grok-" + args.task + "-" + args.trial_attempt + "-profile.json")
    stem = "grok-bundle-" + args.bundle_attempt if args.stage == "bundle" else "grok-" + args.task + "-" + args.trial_attempt + "-" + args.stage
    record = dict(cwd=str(SOURCE), env=ENV, stage=args.stage,
        provider_profile=PROVIDER_PROFILE, resource_profile=args.resource_profile,
        memory_limit_bytes=21_474_836_480, cpu_limit=5,
        resource_changed_from="source384-5cpu-20gib-planner180@1",
        coding_budget_changed_from_seconds=300, same_coding_budget_performance_comparison_claim=False,
        same_resource_performance_comparison_claim=False,
        cross_provider_fallback=False, setup_cache_policy=None,
        retrieval_policy="local-safetensors-symbols@1", retrieval_model_revision=EMBEDDING_REVISION,
        previous_retrieval_policy="lexical-tfidf-symbols@1",
        same_index_configuration_performance_comparison_claim=False,
        embedding_asset_download_calls=0, embedding_training_steps=0,
        driver_seconds=900, work_seconds=840, cleanup_seconds=60,
        planning_seconds=180, coding_seconds=600, coding_watchdog_cap_seconds=660,
        coding_remaining_work_reserve_seconds=25, coding_timeout_clamped_to_remaining_work=True,
        start_allowance_seconds=120, stop_allowance_seconds=20, planning_tools=0, coding_tools=6,
        terminal_bench_revision=subprocess.check_output(["git", "-C", str(DATASET), "rev-parse", "HEAD"], text=True).strip(),
        **source)
    if args.intent_requirement_contract is not None:
        contract_path = args.intent_requirement_contract.resolve(strict=True)
        if args.task != "largest-eigenval" or args.intent_requirement_contract.is_symlink() or not contract_path.is_file():
            raise SystemExit("reviewed regular eigenvalue contract required")
        contract_digest = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        if contract_digest != "3802af85bb0e3382ca9e15ed73995e82ccbf233596087592d25a30d84a1ec983":
            raise SystemExit("reviewed eigenvalue contract bytes changed")
        record.update(intent_requirement_contract_sha256=contract_digest,
                      planning_strategy="intent_symbolic")
    if args.stage == "bundle":
        argv = list(BASE["argv"])
        for flag, value in (("--output", bundle), ("--source", SOURCE), ("--datasets", DATASETS), ("--kit", KIT)):
            argv[argv.index(flag) + 1] = str(value)
        offset = argv.index("--setup-cache-policy")
        del argv[offset:offset + 2]
        argv += ["--grok-binary", "/home/barberb/.grok/downloads/grok-1.0.46-linux-aarch64",
                 "--model-snapshot", str(EMBEDDING_SNAPSHOT)]
        record.update(provider_calls=0, benchmark_executed=False)
    else:
        manifest = json.loads((bundle / "manifest.json").read_text())
        if manifest.get("model_snapshot_revision") != EMBEDDING_REVISION or not manifest.get("learned_requirements"):
            raise SystemExit("exact pinned learned retrieval archive required")
        if "setup_cache" in manifest or manifest.get("grok_cli_assets", {}).get("version") != "1.0.46":
            raise SystemExit("explicit Grok archive without Codex cache policy required")
        built = json.loads((OUTPUT / ("grok-bundle-" + args.bundle_attempt + "-command.json")).read_text())
        if built["source_heads"] != source["source_heads"]:
            raise SystemExit("archive source pins differ from current frozen source")
        review_path = OUTPUT / ("grok-container/archive-review-" + args.bundle_attempt + ".json")
        review = json.loads(review_path.read_text())
        if (review.get("qualified") is not True or review.get("source_heads") != source["source_heads"]
                or review.get("manifest_sha256") != hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()):
            raise SystemExit("fresh independent archive review required")
        record["archive_sha256"] = manifest["archive_sha256"]
        record["archive_review_sha256"] = hashlib.sha256(review_path.read_bytes()).hexdigest()
        if args.stage == "profile":
            if args.task == "largest-eigenval":
                original = ROOT / "artifacts/terminal-suite-eigenval-pilot-20261004/largest-eigenval-profile.json"
                with task_profile.open("xb") as stream:
                    stream.write(original.read_bytes())
                write_json(OUTPUT / (stem + "-command.json"), {**record, "provider_calls": 0,
                    "profile_copy_source": str(original), "profile_sha256": hashlib.sha256(task_profile.read_bytes()).hexdigest()})
                print(json.dumps({"stage": args.stage, "profile": str(task_profile), "provider_calls": 0}))
                return
            argv = [str(PYTHON), "-m", "benchmarks.agent_supervisor.container_coding.terminal_profile_catalog",
                "--dataset", str(DATASET), "--task", args.task, "--output", str(task_profile)]
            record.update(provider_calls=0, benchmark_executed=False)
        elif args.stage == "prepare":
            argv = [str(PYTHON), "-m", "benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark",
                "prepare", "--dataset", str(DATASET), "--task", args.task,
                "--task-profile", str(task_profile), "--archive", str(bundle), "--output", str(trial),
                "--arm", "full", "--resource-profile", args.resource_profile,
                "--source384-config", str(ROOT / "artifacts/terminal-suite-pilot-20261004/source384-config.json"),
                "--provider-profile", PROVIDER_PROFILE]
            if args.intent_requirement_contract is not None:
                argv += ["--intent-requirement-contract", str(contract_path)]
            record.update(provider_calls=0, benchmark_executed=False)
        else:
            argv = [str(PYTHON), "-m", "benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark",
                "execute", "--output", str(trial), "--task", args.task]
            record.update(provider_calls=None, benchmark_executed=True,
                token_accounting="Grok native final-envelope observations; cache excluded from input; billing total unverified")
    record["argv"] = argv
    write_json(OUTPUT / (stem + "-command.json"), record)
    started = time.monotonic()
    with (OUTPUT / (stem + ".log")).open("xb") as stream:
        result = subprocess.run(argv, cwd=SOURCE, env={**os.environ, **ENV}, stdout=stream, stderr=subprocess.STDOUT)
    write_json(OUTPUT / (stem + "-exit.json"), {"exit_code": result.returncode,
        "seconds": time.monotonic() - started, "source_after": snapshot(args.expected_source_head, strict=False),
        "source_unchanged": snapshot(args.expected_source_head, strict=False) == source})
    print(json.dumps({"stage": args.stage, "exit_code": result.returncode,
        "log": str(OUTPUT / (stem + ".log")), "official_reward_must_be_read_from_receipt": args.stage == "execute"}))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
