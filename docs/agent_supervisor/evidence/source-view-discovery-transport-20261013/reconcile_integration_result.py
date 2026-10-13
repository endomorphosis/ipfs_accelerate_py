"""Reconcile a verbose collection prefix without changing execution evidence.

This standard-library verifier runs no product imports or tests. --write creates
one separate receipt; --check verifies that receipt without rewriting it.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
ATTEMPT = PHASE / "integration-01"
OUTPUT = ATTEMPT / "result-reconciliation.json"


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def reconcile():
    paths = [ATTEMPT / name for name in (
        "commands.json", "tests.stdout.log", "tests.stderr.log", "junit.xml")]
    raw = {path.name: path.read_bytes() for path in paths}
    record = json.loads(raw["commands.json"])
    need(record["status"] == "fail" and record["collected_tests"] is None
         and record["collected_count_observations"] == []
         and record["collection_execution_count_agrees"] is False,
         "expected original collection-parser failure")
    need(record.get("runner_error") is None
         and not any(name in record for name in (
             "final_source_observation_error", "junit_observation_error")),
         "an execution or observation failure cannot be reconciled")
    need(len(record["invocations"]) == 1 and record["pytest_invocations"] == 1,
         "one retained pytest invocation required")
    call = record["invocations"][0]
    need(call["exit_code"] == 0, "pytest must have exited successfully")
    for prefix in ("stdout", "stderr"):
        data = raw["tests." + prefix + ".log"]
        need(sha(data) == call[prefix + "_sha256"]
             and len(data) == call[prefix + "_bytes"], "execution log binding differs")
    need(sha(raw["junit.xml"]) == record["junit_sha256"], "JUnit binding differs")
    matches = re.findall(r"^(?:collecting \.\.\. )?collected ([0-9]+) items?\s*$",
                         raw["tests.stdout.log"].decode("utf-8"), re.M)
    need(len(matches) == 1 and int(matches[0]) > 0, "unique collection count required")
    count = int(matches[0])
    document = ET.fromstring(raw["junit.xml"])
    suites = [document] if document.tag == "testsuite" else document.findall("testsuite")
    totals = {key: sum(int(suite.get(key, 0)) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    cases = [(item.get("classname"), item.get("name")) for item in document.iter("testcase")]
    need(totals == record["junit"] == dict(tests=count, failures=0, errors=0, skipped=0)
         and count == len(cases) == len(set(cases)) == record["junit_testcase_count"]
         == record["junit_distinct_testcase_count"]
         and all(all(type(part) is str and part for part in pair) for pair in cases),
         "collected and unique successful execution populations differ")
    need(not any(list(document.iter(tag)) for tag in ("failure", "error", "skipped")),
         "JUnit contains an unsuccessful case")
    need(record["before"] == record["after"] and record["source_before_after_equal"] is True,
         "execution source bindings changed")
    datasets = Path(record["datasets_path"])
    need(record["datasets_execution_revision"] == record["datasets_execution_head_after"]
         == record["datasets_parent_pin_reference"]
         and record["datasets_parent_pin_executed"] is True, "datasets execution differs")
    for name, checkout in (("datasets", datasets), ("kit", Path(record["kit_path"]))):
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
        expected = record["datasets_execution_revision"] if name == "datasets" else record["kit_execution_head"]
        need(head == expected, name + " checkout changed")
    need(record["kit_head_before_after_equal"] is True
         and record["kit_execution_head"] == record["kit_execution_head_after"]
         and record["historical_full_runner_unchanged"] is True,
         "kit or historical runner changed")
    need(sha((PHASE / "qualification_runner.py").read_bytes()) == record["historical_full_runner_sha256"],
         "historical runner binding differs")
    need(all(item["unchanged"] is True for item in record["unchanged_historical_codec_and_pure_mapper"]),
         "historical codec changed")
    roots = {"accelerator": ROOT, "datasets_execution_checkout": datasets}
    for row in record["before"]:
        relative = Path(row["path"])
        need(not relative.is_absolute() and ".." not in relative.parts, "relative source path required")
        data = (roots[row["repository"]] / relative).read_bytes()
        need(sha(data) == row["sha256"] and len(data) == row["bytes"], "current source differs")
        if row["repository"] == "datasets_execution_checkout":
            original = subprocess.check_output(["git", "show",
                record["datasets_execution_revision"] + ":" + row["path"]], cwd=datasets)
            need(data == original and row["matches_immutable_git_object"] is True,
                 "datasets source differs from immutable object")
    need(all(record[key] == 0 for key in ("model_calls", "provider_calls", "training_steps", "solver_calls"))
         and record["docker_run"] is False, "unexpected execution scope")
    return dict(schema="source-view-discovery-transport-result-reconciliation@1", status="pass",
        original_runner_status=record["status"], original_runner_receipt_preserved=True,
        reason="collection_parser_missed_standard_verbose_prefix", pytest_exit_code=0,
        collected_tests=count, junit=totals, junit_distinct_testcase_count=len(set(cases)),
        collection_execution_count_agrees=True, source_before_after_equal=True,
        source_bindings_verified=len(record["before"]),
        accelerator_execution_head=record["accelerator_execution_head"],
        datasets_execution_revision=record["datasets_execution_revision"],
        datasets_parent_pin_executed=True, kit_execution_head=record["kit_execution_head"],
        source_binding_scope=record["source_binding_scope"],
        reconciler_sha256=sha(Path(__file__).read_bytes()),
        inputs=[dict(path=str(path.relative_to(ROOT)), bytes=len(raw[path.name]),
                     sha256=sha(raw[path.name])) for path in paths],
        tests_rerun=False, product_imports=False, provider_calls=0, model_calls=0,
        training_steps=0, proof_authority=False, completion_authority=False,
        official_task_result=None, total_token_savings_measured=False)


if __name__ == "__main__":
    need(sys.argv[1:] in (["--write"], ["--check"]), "use --write or --check")
    result = reconcile()
    if sys.argv[1] == "--write":
        need(not OUTPUT.exists(), "reconciliation already exists")
        result["recorded_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        existing = json.loads(OUTPUT.read_text())
        existing.pop("recorded_at_utc")
        need(existing == result, "reconciliation differs from retained evidence")
    print(json.dumps({key: result[key] for key in ("status", "collected_tests", "source_bindings_verified")}))
