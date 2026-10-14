#!/usr/bin/env python3
"""Qualify pinned LegalIR source contracts through native deterministic jobs.

These receipts concern executable source contracts. They do not confer corpus
coverage, reviewed legal meaning, model qualification or Lean kernel authority.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "legal-autoformalization-foundations-qualification/v1"
TASKS = ("LAC-S2-005", "LAC-S2-010")
MAX_BYTES = 1024 * 1024
MAX_FAILURE_CASES = 5
MAX_FAILURE_FIELD_BYTES = 512
MAX_FAILURE_TAIL_BYTES = 1024


class QualificationFailure(ValueError):
    """A refused qualification with bounded diagnostics retained by its caller."""

    def __init__(self, message, diagnostics):
        self.diagnostics = diagnostics
        first = next(iter(diagnostics["failed_cases"]), None)
        detail = f": {first['name']}: {first['message']}" if first else ""
        super().__init__(message + detail)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    need(len(raw) <= MAX_BYTES, "JSON exceeds qualification read bound")
    return json.loads(raw)


def identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    need((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
         "Input changed during hashing")
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": after.st_size}


def relative_file(root, value):
    need(isinstance(value, str) and value, "Missing relative source path")
    parsed = PurePosixPath(value)
    need(not parsed.is_absolute() and ".." not in parsed.parts
         and parsed.as_posix() == value and "\\" not in value,
         "Invalid relative source path")
    path = (root / value).resolve(strict=True)
    need(path.is_relative_to(root) and path.is_file(), "Source escaped pinned repository")
    return path


def git(root, *args):
    run = subprocess.run(["git", "-C", str(root), *args], text=True,
                         capture_output=True, timeout=20, check=False,
                         env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))
    need(run.returncode == 0, "Pinned repository identity unavailable")
    return run.stdout.strip()


def bindings(task, sources_path):
    need(task in TASKS, "Unknown foundation qualification task")
    sources_path = Path(sources_path).resolve(strict=True)
    need(sources_path.is_relative_to(ROOT), "Sources must be repository owned")
    sources = read_json(sources_path)
    need(sources.get("schema") == "legal-autoformalization-foundations-sources/v1",
         "Wrong source manifest schema")
    datasets = Path(sources["datasets"]["root"]).resolve(strict=True)
    commit = git(datasets, "rev-parse", "HEAD")
    need(commit == sources["datasets"]["commit"], "Datasets commit moved")
    need(not git(datasets, "status", "--porcelain=v1", "--untracked-files=no"),
         "Pinned datasets checkout has tracked changes")
    job = sources["jobs"][task]
    tests = job["test_paths"]
    need(isinstance(tests, list) and tests and len(tests) == len(set(tests)),
         "Qualification requires distinct reviewed test files")
    rows = []
    pins = {row["path"]: row["sha256"] for row in sources["immutable_inputs"]
            if row.get("repository", "datasets") == "datasets"}
    for name in dict.fromkeys([*job["code_paths"], *tests]):
        path = relative_file(datasets, name)
        git(datasets, "ls-files", "--error-unmatch", "--", name)
        actual = identity(path)
        need(actual["sha256"] == pins.get(name), "Qualification source pin differs")
        rows.append({**actual, "repository_path": name})
    prior = sources["prior_bootstrap_completion"]
    need(identity(prior["path"])["sha256"] == prior["sha256"], "Bootstrap completion moved")
    need(read_json(prior["path"])["status"] == "inventory_bootstrap_complete",
         "Bootstrap prerequisite is not complete")
    interpreter = identity(job.get("python_executable", sys.executable))
    if job.get("python_executable_sha256"):
        need(interpreter["sha256"] == job["python_executable_sha256"], "Qualification interpreter moved")
    return sources, datasets, job, {"source_manifest_sha256": identity(sources_path)["sha256"],
        "source_manifest_repository_path": sources_path.relative_to(ROOT).as_posix(),
        "driver_sha256": identity(__file__)["sha256"], "datasets_commit": commit,
        "datasets_tree": git(datasets, "rev-parse", "HEAD^{tree}"), "inputs": rows,
        "prior_bootstrap_completion_sha256": prior["sha256"], "interpreter": interpreter}


def command(job):
    return [job.get("python_executable", sys.executable), "-B", "-m", "pytest", "-q",
            *job["test_paths"]]


def execution_environment_policy(datasets):
    """Nonsecret, recorded overrides binding imports and pytest invocation."""
    return {"PYTHONPATH": str(Path(datasets).resolve(strict=True)), "PYTEST_ADDOPTS": "",
            "PYTHONDONTWRITEBYTECODE": "1", "IPFS_AUTO_INSTALL": "false",
            "IPFS_DATASETS_AUTO_INSTALL": "false", "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1",
            "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS": "0", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": ""}


def _bounded_failure_text(value):
    return str(value or "").encode("utf-8", errors="replace")[:MAX_FAILURE_FIELD_BYTES].decode(
        "utf-8", errors="ignore")


def _failure_tail(path):
    with Path(path).open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - MAX_FAILURE_TAIL_BYTES))
        raw = stream.read(MAX_FAILURE_TAIL_BYTES)
    return raw.decode("utf-8", errors="ignore")


def refusal_payload(error):
    payload = {"valid": False, "status": "refused", "error": str(error)}
    if isinstance(error, QualificationFailure):
        payload["diagnostics"] = error.diagnostics
    return payload


def execute(datasets, job):
    environment_policy = execution_environment_policy(datasets)
    env = dict(os.environ, **environment_policy)
    with tempfile.TemporaryDirectory(prefix="legal-foundation-tests-") as temporary:
        junit = Path(temporary) / "junit.xml"
        cases, failed_cases = [], []

        def fail(message, returncode):
            raise QualificationFailure(message, {
                "schema": "legal-autoformalization-qualification-failure/v1",
                "returncode": returncode, "test_count": len(cases),
                "failed_case_count": sum(c["status"] != "passed" for c in cases),
                "failed_cases": failed_cases,
                "stdout_tail": _failure_tail(Path(temporary) / "stdout"),
                "stderr_tail": _failure_tail(Path(temporary) / "stderr"),
                "environment_policy": environment_policy})

        with (Path(temporary) / "stdout").open("wb") as out, \
                (Path(temporary) / "stderr").open("wb") as err:
            try:
                run = subprocess.run([*command(job), "--junitxml", str(junit),
                    "-o", "cache_dir=" + str(Path(temporary) / "cache")],
                    cwd=datasets, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    timeout=min(300, int(job.get("timeout_seconds", 180))), check=False)
            except subprocess.TimeoutExpired:
                out.flush()
                err.flush()
                fail("Qualification tests exceeded the declared timeout", None)
        if not junit.is_file() or junit.stat().st_size > MAX_BYTES:
            fail("Qualification did not produce bounded JUnit evidence", run.returncode)
        try:
            tree = ET.parse(junit)
        except ET.ParseError:
            fail("Qualification produced malformed JUnit evidence", run.returncode)
        for node in tree.iter("testcase"):
            issues = [child for child in node if child.tag in ("failure", "error", "skipped")]
            cases.append({"class": node.get("classname", ""), "name": node.get("name", ""),
                          "status": issues[0].tag if issues else "passed"})
            if issues and len(failed_cases) < MAX_FAILURE_CASES:
                failed_cases.append({"class": _bounded_failure_text(node.get("classname", "")),
                    "name": _bounded_failure_text(node.get("name", "")), "status": issues[0].tag,
                    "message": _bounded_failure_text(issues[0].get("message") or issues[0].text)})
        result = {"command": command(job), "returncode": run.returncode, "cases": cases,
            "ephemeral_options": {"junitxml": "temporary", "cache_dir": "temporary"},
            "environment_policy": environment_policy,
            "test_count": len(cases), "passed_count": sum(c["status"] == "passed" for c in cases),
            "stdout_sha256": identity(Path(temporary) / "stdout")["sha256"],
            "stderr_sha256": identity(Path(temporary) / "stderr")["sha256"]}
        if run.returncode != 0 or not cases or any(c["status"] != "passed" for c in cases):
            fail("Qualification tests failed, errored, or skipped", run.returncode)
        if len(cases) < int(job.get("minimum_tests", 1)):
            fail("Qualification test population shrank", run.returncode)
        return result


def validate(result, task, sources_path):
    sources, datasets, job, current = bindings(task, sources_path)
    need(result.get("schema") == SCHEMA and result.get("task_id") == task
         and result.get("campaign_id") == sources["campaign_id"], "Wrong qualification identity")
    need(result.get("bindings") == current, "Qualification immutable bindings differ")
    need(result.get("scope") == "source-contract behavioral qualification"
         and result.get("original_task_acceptance") == "partial"
         and result.get("legal_semantic_acceptance") is False
         and result.get("corpus_coverage_complete") is False
         and result.get("provider_dispatched") is False, "Qualification exceeds its acceptance scope")
    observed = result["test_execution"]
    need(observed["command"] == command(job) and observed["returncode"] == 0,
         "Qualification command or outcome differs")
    need(observed.get("ephemeral_options") == {"junitxml": "temporary", "cache_dir": "temporary"},
         "Qualification temporary output policy differs")
    need(observed.get("environment_policy") == execution_environment_policy(datasets),
         "Qualification import/pytest environment policy differs")
    cases = observed["cases"]
    need(isinstance(cases, list) and len(cases) >= int(job.get("minimum_tests", 1))
         and all(c["status"] == "passed" for c in cases), "Incomplete passing test population")
    need(len({(c["class"], c["name"]) for c in cases}) == len(cases), "Duplicate test identities")
    need(observed["test_count"] == len(cases) and observed["passed_count"] == len(cases),
         "Qualification test denominator differs")
    need(result.get("test_execution_sha256") == sha(wire(observed)), "Test execution digest differs")
    return {"valid": True, "task_id": task, "test_count": len(cases),
            "datasets_commit": current["datasets_commit"], "legal_semantic_acceptance": False,
            "original_task_acceptance": "partial", "scope": result["scope"]}


def output(task):
    need(task in TASKS, "Unknown foundation qualification task")
    path = ROOT / "data/legal_autoformalization_foundations" / task / "result.json"
    need(path.resolve() == path, "Qualification output path alias refused")
    return path


def materialize(task, sources_path):
    path = output(task)
    if path.exists():
        before = identity(path)
        checked = validate(read_json(path), task, sources_path)
        need(identity(path) == before, "Qualification changed during validation")
        return {**checked, "materialized": False}
    sources, datasets, job, frozen = bindings(task, sources_path)
    observed = execute(datasets, job)
    result = {"schema": SCHEMA, "campaign_id": sources["campaign_id"], "task_id": task,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "source-contract behavioral qualification", "original_task_acceptance": "partial",
        "legal_semantic_acceptance": False, "corpus_coverage_complete": False,
        "provider_dispatched": False, "bindings": frozen, "test_execution": observed,
        "test_execution_sha256": sha(wire(observed))}
    checked = validate(result, task, sources_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".qualification-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(wire(result) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        return materialize(task, sources_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {**checked, "materialized": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("materialize", "validate"))
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--sources", default="config/legal_autoformalization_foundations_sources.json")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    sources_path = ROOT / args.sources
    if args.action == "materialize":
        checked = materialize(args.task, sources_path)
    else:
        result = read_json(output(args.task))
        checked = validate(result, args.task, sources_path)
        if args.replay:
            _, datasets, job, _ = bindings(args.task, sources_path)
            replayed = execute(datasets, job)
            need(replayed["cases"] == result["test_execution"]["cases"],
                 "Independent replay test population differs")
            checked["independent_replay_passed"] = True
    print(json.dumps(checked, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired, ET.ParseError) as error:
        print(json.dumps(refusal_payload(error)), file=sys.stderr)
        raise SystemExit(1)
