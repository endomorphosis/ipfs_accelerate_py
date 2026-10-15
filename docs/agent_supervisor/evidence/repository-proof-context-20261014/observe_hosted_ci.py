"""Read exact published-commit documentation CI; never dispatch or rerun a workflow."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

API = "/repos/endomorphosis/ipfs_accelerate_py"
OUTPUT = Path("/home/barberb/lift_coding/.worktrees/repository-proof-context-root-20261014/") / (
    "artifacts/repository-proof-context-publication-20261014/child-ci-observation.json")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision")
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.revision) is None:
        raise ValueError("Exact independently selected publication SHA required")
    if OUTPUT.exists():
        raise ValueError("Refusing to overwrite hosted-CI evidence")
    observations = []

    def read(endpoint):
        if not endpoint.startswith(API + "/"):
            raise ValueError("Only this repository's read-only API is supported")
        argv = ["gh", "api", endpoint]
        raw = subprocess.check_output(argv, timeout=45)
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError("CI observation exceeds byte bound")
        observations.append(dict(argv=argv, observed_at_utc=now(),
            stdout_sha256=hashlib.sha256(raw).hexdigest()))
        return json.loads(raw)

    listing = read(API + "/actions/workflows/documentation-gates.yml/runs?head_sha="
                   + args.revision + "&per_page=10")
    runs = [row for row in listing["workflow_runs"] if row["head_sha"] == args.revision
            and row["path"] == ".github/workflows/documentation-gates.yml"]
    if not runs or runs[0]["status"] != "completed":
        print(json.dumps(dict(status="not_ready", head_sha=args.revision,
                              exact_runs_observed=len(runs))))
        return 3
    run = runs[0]
    jobs = read(API + "/actions/runs/" + str(run["id"]) + "/jobs")
    if len(jobs["jobs"]) != jobs["total_count"]:
        raise ValueError("Complete CI job population required")
    selected = [row for row in jobs["jobs"] if row["name"] == "documentation-gates"]
    if len(selected) != 1:
        raise ValueError("One exact documentation-gates job required")
    job = selected[0]
    url = job["check_run_url"]
    prefix = "https://api.github.com" + API + "/check-runs/"
    if not url.startswith(prefix) or not url.removeprefix(prefix).isdigit():
        raise ValueError("Exact repository check-run endpoint required")
    endpoint = API + "/check-runs/" + url.removeprefix(prefix)
    check = read(endpoint)
    annotations = read(endpoint + "/annotations?per_page=100")
    if len(annotations) != check["output"]["annotations_count"]:
        raise ValueError("Complete check annotation population required")
    if not (run["head_sha"] == check["head_sha"] == job["head_sha"] == args.revision
            and job["run_id"] == run["id"] and job["id"] == check["id"]
            and job["status"] == check["status"]
            and job["conclusion"] == check["conclusion"]):
        raise ValueError("CI run/job/check identity or outcome differs")
    closing = read(API + "/actions/runs/" + str(run["id"]))
    if not (all(closing[field] == run[field] for field in
                ("id", "head_sha", "path", "run_attempt", "status", "conclusion"))
            and check["status"] == "completed"
            and all(row["status"] == "completed"
                    and row["run_attempt"] == run["run_attempt"]
                    and row["head_sha"] == args.revision
                    and row["run_id"] == run["id"] for row in jobs["jobs"])):
        print(json.dumps(dict(status="not_ready", head_sha=args.revision,
                              reason="run_or_job_attempt_changed")))
        return 3
    run = closing
    steps = any(row["steps"] for row in jobs["jobs"])
    billing = any(row.get("annotation_level") == "failure"
                  and "billing" in row.get("message", "").lower() for row in annotations)
    passed = bool(jobs["jobs"]) and run["conclusion"] == "success" and all(
        row["conclusion"] == "success" and row["runner_id"] > 0 and row["steps"]
        for row in jobs["jobs"])
    record = dict(schema="repository-proof-context-exact-hosted-ci@1",
        observed_at_utc=now(), head_sha=args.revision, run_id=run["id"],
        status=run["status"], conclusion=run["conclusion"], run=run, jobs=jobs,
        check_run=check, annotations=annotations, api_observations=observations,
        hosted_steps_executed=steps, hosted_gates_passed=passed,
        billing_lock_observed=billing, workflow_dispatch_or_rerun_requested=False)
    with OUTPUT.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(dict(status="observed", head_sha=args.revision,
        run_id=run["id"], conclusion=run["conclusion"], hosted_gates_passed=passed,
        hosted_steps_executed=steps, billing_lock_observed=billing)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
