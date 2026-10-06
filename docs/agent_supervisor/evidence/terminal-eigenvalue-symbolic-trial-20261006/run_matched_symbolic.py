"""Prepare and execute one fresh trial against the retained successful archive.

The only intended benchmark treatment is the reviewed version-2 planning
contract. Task/profile/archive/model/cache/budget inputs are reused verbatim.
Full contracts, raw logs and receipts remain in ignored local artifacts.
No retry, hidden-source inspection, training or candidate reuse is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

HISTORICAL = Path("/home/barberb/lift_coding/artifacts/terminal-supervisor-blockers-20261005")
RUNNER_HEAD = "8dab684ef69f87a88c3dac1585eea07a5bf29a24"
ARCHIVE_SHA256 = "a9f25619d230725ebe9f0f784d74feaf9191905ab78216fcedaee45a95c0cbcc"


def binding(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git(root, *argv):
    return subprocess.run(["git", "-C", str(root), *argv], check=True,
                          capture_output=True, text=True).stdout.strip()


def invoke(argv, *, runner, environment, artifact, phase):
    record = {"argv": argv, "cwd": str(runner), "env": environment,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_head": git(runner, "rev-parse", "HEAD"),
        "producer": binding(Path(__file__)), "phase": phase}
    write(artifact / f"{phase}-command.json", record)
    started = time.monotonic()
    env = os.environ.copy()
    env.update(environment)
    with (artifact / f"{phase}.stdout").open("w") as stdout, (artifact / f"{phase}.stderr").open("w") as stderr:
        result = subprocess.run(argv, cwd=runner, env=env, stdout=stdout, stderr=stderr)
    exit_record = {"exit_code": result.returncode, "seconds": time.monotonic() - started,
                   "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    write(artifact / f"{phase}-exit.json", exit_record)
    print(json.dumps({"phase": phase, **exit_record}), flush=True)
    if result.returncode:
        raise RuntimeError(f"{phase} failed; bounded exit and private logs retained")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--resume-prepared", action="store_true",
                        help="Execute an existing immutable preparation after explicit controls review")
    args = parser.parse_args()
    runner = args.runner.resolve(strict=True)
    artifact = args.artifact.resolve(strict=True)
    assert git(runner, "rev-parse", "HEAD") == RUNNER_HEAD
    assert not git(runner, "status", "--porcelain")
    output = artifact / "run-01"
    assert output.exists() if args.resume_prepared else not output.exists()
    # Source-bearing benchmark contract is local and excluded from publication.
    private_contract = artifact / "private-intent-contract.json"
    if args.resume_prepared:
        assert binding(args.contract) == binding(private_contract)
        assert not (output / "invocation.json").exists(), "one official trial only"
    else:
        assert not private_contract.exists()
        shutil.copyfile(args.contract, private_contract)
    contract = json.loads(private_contract.read_text())
    assert contract["schema"] == "intent-plan-requirement-contract@2"
    prepare = json.loads((HISTORICAL / "prepare-03-command.json").read_text())
    execute = json.loads((HISTORICAL / "execute-03-command.json").read_text())
    assert prepare["archive_sha256"] == execute["archive_sha256"] == ARCHIVE_SHA256
    old_source = prepare["cwd"]
    environment = dict(prepare["env"])
    parts = environment["PYTHONPATH"].split(":")
    assert parts[0] == old_source
    parts[0] = str(runner)
    environment["PYTHONPATH"] = ":".join(parts)
    argv = list(prepare["argv"])
    argv[argv.index("--output") + 1] = str(output)
    argv.extend(["--intent-requirement-contract", str(private_contract)])
    selection = {
        "schema": "largest-eigenval-symbolic-trial-selection@1",
        "historical_trial": "largest-eigenval__iieLP5S", "new_trial_output": str(output),
        "runner_head": RUNNER_HEAD, "archive_sha256": ARCHIVE_SHA256,
        "treatment": "source-bound reviewed administrative @2 symbolic planning contract",
        "private_contract": binding(private_contract), "source_benchmark_excluded_from_training": True,
        "new_code_generated_from_original_public_task": True, "historical_candidate_reused": False,
        "retries": 0, "planned_official_trials": 1,
        "source_semantics_verified": False, "numerical_proof_claimed": False,
        "causal_efficiency_claimed": False,
    }
    if args.resume_prepared:
        assert json.loads((artifact / "selection.json").read_text()) == selection
    else:
        write(artifact / "selection.json", selection)
        invoke(argv, runner=runner, environment=environment, artifact=artifact, phase="prepare")
    # Preparation emits signed immutable controls. Compare actual configs,
    # removing only path identity and the intentional requirement treatment.
    previous = json.loads((HISTORICAL / "largest-eigenval-03/config.json").read_text())
    current = json.loads((output / "config.json").read_text())
    def normalized(value):
        value = json.loads(json.dumps(value))
        value["jobs_dir"] = "<fresh-output>/jobs"
        value["agents"][0]["kwargs"].pop("intent_requirement_contract", None)
        return value
    strict_equal = normalized(previous) == normalized(current)
    def behavior(value):
        value = normalized(value)
        for key in ("exclude_exceptions", "include_exceptions"):
            if isinstance(value["retry"].get(key), list):
                value["retry"][key] = sorted(value["retry"][key])
        return value
    equal = behavior(previous) == behavior(current)
    no_retry = previous["retry"]["max_retries"] == current["retry"]["max_retries"] == 0
    if (artifact / "controls-preflight.json").exists():
        saved = artifact / "controls-preflight-before-retry-normalization.json"
        assert not saved.exists()
        shutil.copyfile(artifact / "controls-preflight.json", saved)
    write(artifact / "controls-preflight.json", {
        "schema": "largest-eigenval-symbolic-matched-controls@1",
        "historical_config": binding(HISTORICAL / "largest-eigenval-03/config.json"),
        "new_config": binding(output / "config.json"),
        "equal_after_output_identity_and_intent_treatment": strict_equal,
        "equal_after_inactive_retry_set_order_normalization": equal,
        "both_retry_policies_disabled": no_retry,
        "serialization_mismatch_retained": not strict_equal,
        "archive_reused_without_rebuild": True, "provider_calls_in_preparation": 0,
        "only_contract_is_intended_treatment": True,
        "limits": ["Historical comparator is not randomized or repeated.",
            "Time, host load, provider state and session caches may differ.",
            "Router sessions are not internal model turns or API request counts."],
    })
    assert equal and no_retry, "Unintended config difference; official trial not started"
    argv = list(execute["argv"])
    argv[argv.index("--output") + 1] = str(output)
    invoke(argv, runner=runner, environment=environment, artifact=artifact, phase="execute")


if __name__ == "__main__":
    main()
