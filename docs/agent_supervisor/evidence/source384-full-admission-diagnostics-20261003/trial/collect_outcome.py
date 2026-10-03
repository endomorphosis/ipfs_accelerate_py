"""Collect existing public receipts only; never rerun the trial or read verifier code."""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
TRIAL = BASE / "trial-01/jobs/supervisor-full-fix-code-vulnerability/fix-code-vulnerability__YF7vcfn"


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def metadata(path):
    body = path.read_bytes()
    return {"path": str(path.relative_to(BASE)), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


receipt = read(BASE / "trial-01/receipt.json")
trial = receipt["trials"][0]
report = read(TRIAL / "agent/supervisor-result.json")
agent = trial["agent_context"]["metadata"]
deployment = read(TRIAL / "agent/deployment/deployment.json")
config = read(BASE / "trial-01/config.json")
preparation = read(BASE / "trial-01/preparation.json")
before = read(TRIAL / "agent/public-output-evidence/before/receipt.json")
after = read(TRIAL / "agent/public-output-evidence/after/receipt.json")
public = read(TRIAL / "agent/public-output-evidence/receipt.json")

assert len(receipt["trials"]) == 1
assert report["error"]["type"] == "LeaseTimeoutError"
assert report["provider_invocations"] == []
assert agent["usage"]["provider_calls"] == 0
assert all(agent["usage"][key] is None for key in ("input_tokens", "output_tokens", "cached_input_tokens", "total_tokens"))
assert deployment["original_inputs"] == deployment["retained_inputs"]
assert before["files"] == after["files"]

copies = {
    "supervisor-result.json": TRIAL / "agent/supervisor-result.json",
    "public-before-receipt.json": TRIAL / "agent/public-output-evidence/before/receipt.json",
    "public-after-receipt.json": TRIAL / "agent/public-output-evidence/after/receipt.json",
    "public-output-receipt.json": TRIAL / "agent/public-output-evidence/receipt.json",
    "supervisor.stdout": TRIAL / "agent/supervisor.stdout",
    "supervisor.stderr": TRIAL / "agent/supervisor.stderr",
    "execute-exit.json": BASE / "execute-exit.json",
}
for name, source in copies.items():
    shutil.copyfile(source, OUT / name)

command = ["docker", "ps", "-a", "--filter", "id=30943288823d", "--format", "{{json .}}"]
observed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
cleanup = subprocess.run(command, capture_output=True, text=True, check=False, timeout=20)
cleanup_receipt = {
    "observed_at": observed_at, "command": command, "returncode": cleanup.returncode,
    "stdout": cleanup.stdout, "stderr": cleanup.stderr,
    "container_absent": cleanup.returncode == 0 and not cleanup.stdout.strip(),
    "scope": "Read-only post-trial observation of this container ID; not a global process inventory.",
}
write("cleanup-observation.json", cleanup_receipt)

env_keys = ("type", "override_cpus", "override_memory_mb", "cpu_enforcement_policy", "memory_enforcement_policy", "delete")
result = {
    "schema": "terminal-full-supervisor-trial-outcome@1",
    "task": receipt["task"], "trial": trial["trial"], "arm": receipt["arm"],
    "trial_count": 1, "parallel_workers": receipt["parallel_workers"],
    "model": receipt["model"], "reasoning_effort": receipt["reasoning_effort"], "cli_version": receipt["cli_version"],
    "reward": trial["reward"], "reward_source": "Harbor result.json verifier_result.rewards (output only; verifier source not inspected)",
    "task_completed": report["task_completed"], "driver_returncode": agent["driver_returncode"],
    "harbor_exception_type": trial["exception_type"], "error": report["error"],
    "failure_phase": "initial_context", "failure_before_planning_and_repair": True,
    "usage": agent["usage"], "observed_provider_invocations": len(report["provider_invocations"]),
    "planning_doctor_proof_and_coding_routes": "not reached; no corresponding report fields or receipts",
    "intent_preplanning": report["intent_preplanning"],
    "source384_selected": receipt["source384_enabled"],
    "source384_inference_receipt_present": False,
    "native_task_worker_started": False,
    "native_setup_qualification": "Deployment qualified; distinct from the coding-task runtime, which was not started.",
    "durations_seconds": {**trial["durations_seconds"], "deployment": deployment["seconds"], "supervisor": report["seconds"], **report["phases"]},
    "deadlines_seconds": {key: report[key] for key in ("max_total_agent_seconds", "reserved_cleanup_seconds", "work_cutoff_seconds")},
    "resource_controls": {
        "profile": receipt["resource_profile"],
        "declared_environment": {key: config["environment"].get(key) for key in env_keys},
        "observed_configuration_status": receipt["comparison_controls"]["status"],
        "configuration_unchanged": receipt["comparison_controls"]["configuration_unchanged"],
        "config_sha256": preparation["config_sha256"],
        "actual_container_cgroup_observation": None,
        "actual_limit_scope": "No per-container cgroup observation retained for this trial. Host resources-before.json is archive-build capacity only.",
    },
    "input_preservation": {
        "all_original_inputs_unchanged_across_deployment": deployment["task_source_preserved"],
        "original_input_count": len(deployment["original_inputs"]["files"]),
        "host_original_task_inputs_unchanged": receipt["original_task_inputs_unchanged"],
        "post_agent_declared_public_files_unchanged": before["files"] == after["files"],
        "public_files": after["files"],
        "public_diff_bytes": {key: value["bytes"] for key, value in public["diffs"].items()},
        "scope": "All 218 originals compared before/after deployment. Post-agent capture covers bottle.py and report.jsonl only; no full 218-file post-agent snapshot is claimed.",
    },
    "runtime_archive_sha256": agent["runtime_archive_sha256"],
    "cleanup": {"worker_cleanup_returncode": report["worker_cleanup_returncode"], "container_absent": cleanup_receipt["container_absent"], "observation": "cleanup-observation.json", "remaining_processes": report["remaining_processes"]},
    "diagnostic_limits": {
        "traceback_retained": False,
        "structured_error_file": str((TRIAL / "agent/supervisor-result.json").relative_to(BASE)),
        "reason": "Driver catches exceptions into error.type/message; stderr is empty, stdout has no traceback. Exact lease acquisition frame is not retained.",
    },
    "benchmark_advantage_claimed": False,
    "matched_baseline_present": False,
    "authority": {"completion": False, "execution": False, "proof": False},
}
write("result.json", result)

reference_paths = list(copies.values()) + [BASE / "trial-01/receipt.json", BASE / "trial-01/config.json", BASE / "trial-01/preparation.json", TRIAL / "result.json", TRIAL / "agent/deployment/deployment.json", TRIAL / "trial.log", BASE / "resources-before.json"]
write("receipt-metadata.json", {
    "original_receipts": [metadata(path) for path in reference_paths],
    "copied_receipts": {name: metadata(source) for name, source in copies.items()},
    "omitted": ["raw source bodies", "diff bodies", "auth material", "verifier inputs and code", "model payloads", "private supervisor state", "large deployment and Harbor metadata bodies"],
})
print(json.dumps({"result": str(OUT / "result.json"), "reward": trial["reward"], "provider_calls": agent["usage"]["provider_calls"], "tokens": None, "container_absent": cleanup_receipt["container_absent"]}))
