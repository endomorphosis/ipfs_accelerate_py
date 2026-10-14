"""Retain one standalone upstream semantic-state API qualification.

Run the full existing dataset test_api module once, from the fixed child cwd
with its pytest config and a fresh private seal/runtime namespace. This is
separate evidence: its counts are never added to the supervisor qualification.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
CHILD_REVISION = "f15c146467c29c95f39ae1f041c8eb84e76d8bc8"
DATASETS = Path("/home/barberb/lift_coding/.worktrees/source-query-replay-datasets-20261014")
DATASETS_REVISION = "f4a823231ad52c9252504810a1b00ed4738346b6"
KIT = Path("/home/barberb/lift_coding/.worktrees/repository-proof-context-kit-20261014")
KIT_REVISION = "40c98c71b95be750509ffdb21e7e5494d91b4e79"
TARGET_NAME = "tests/unit/logic/software_contracts/semantic_state/test_api.py"
TARGET = DATASETS / TARGET_NAME
MAX_BYTES = 16 * 1024 * 1024
sys.dont_write_bytecode = True


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def git(root, *args):
    result = subprocess.run(["git", "--no-optional-locks", *args], cwd=root,
                            capture_output=True, timeout=60)
    need(result.returncode == 0 and len(result.stdout) <= MAX_BYTES,
         "bounded read-only Git observation failed: " + repr(args))
    return result.stdout


def read(path):
    need(path.is_absolute() and path.resolve(strict=True) == path, "canonical input required")
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= MAX_BYTES,
         "bounded regular input required")
    raw, after = path.read_bytes(), path.lstat()
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
    need(len(raw) == before.st_size and all(getattr(before, key) == getattr(after, key) for key in fields),
         "input changed during read")
    return raw


def inputs():
    paths = [
        (DATASETS, "datasets", "ipfs_datasets_py/logic/software_contracts/semantic_state/api.py"),
        (DATASETS, "datasets", "ipfs_datasets_py/logic/software_contracts/semantic_state/models.py"),
        (DATASETS, "datasets", "ipfs_datasets_py/logic/software_contracts/content.py"),
        (DATASETS, "datasets", TARGET_NAME),
        (DATASETS, "datasets", "pytest.ini"),
        (ROOT, "accelerator", "pytest.ini"),
        (ROOT, "accelerator", "conftest.py"),
        (ROOT, "accelerator", "ipfs_accelerate_py/testing/pytest_ast_seal.py"),
        (ROOT, "accelerator", str(Path(__file__).resolve().relative_to(ROOT))),
    ]
    # Review and bind each existing ancestor conftest of the absolute target.
    directory = TARGET.parent
    while True:
        relative = str((directory / "conftest.py").relative_to(DATASETS))
        if (directory / "conftest.py").is_file():
            paths.append((DATASETS, "datasets", relative))
        if directory == DATASETS:
            break
        directory = directory.parent
    need(len({(label, name) for _root, label, name in paths}) == len(paths), "duplicate input binding")
    return sorted(paths, key=lambda row: (row[1], row[2]))


def bindings(selected):
    result = []
    runner = Path(__file__).resolve()
    for root, label, name in selected:
        path, revision = root / name, DATASETS_REVISION if root == DATASETS else CHILD_REVISION
        raw = read(path)
        immutable = path != runner
        if immutable:
            need(raw == git(root, "show", revision + ":" + name), "immutable input object differs: " + name)
        result.append(dict(repository=label, path=name, source_checkout=str(root),
            bytes=len(raw), sha256=sha(raw), matches_immutable_git_object=immutable))
    return result


def states():
    result = {}
    for root, revision in ((ROOT, CHILD_REVISION), (DATASETS, DATASETS_REVISION), (KIT, KIT_REVISION)):
        observed = dict(head=git(root, "rev-parse", "HEAD").decode().strip(),
            index_sha256=sha(git(root, "ls-files", "--stage", "-z")),
            refs_sha256=sha(git(root, "for-each-ref", "--format=%(refname) %(objectname)")))
        need(observed["head"] == revision, "fixed execution HEAD differs: " + str(root))
        result[str(root)] = observed
    return result


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", help="exclusive dependency-NN attempt directory")
    args = parser.parse_args()
    need(re.fullmatch(r"dependency-[0-9]{2,3}", args.attempt), "canonical standalone attempt required")
    attempt = PHASE / args.attempt
    attempt.mkdir(exist_ok=False)
    selected = inputs()
    before, before_git = bindings(selected), states()
    snapshots = []
    for row in before:
        root = DATASETS if row["repository"] == "datasets" else ROOT
        raw = read(root / row["path"])
        need(len(raw) == row["bytes"] and sha(raw) == row["sha256"], "source changed before snapshot")
        snapshot = attempt / "source-snapshot" / row["repository"] / (row["path"] + ".snapshot.txt")
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(raw)
        snapshots.append(dict(row, snapshot=str(snapshot.relative_to(ROOT))))
    private = Path(tempfile.mkdtemp(prefix="sqd-"))
    updates = dict(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONHASHSEED="0",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", IPFS_DATASETS_PY_LAZY_INSTALL_ERGOAI="0", IPFS_KIT_DISABLE="1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
        IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS="0", IPFS_DATASETS_SEMANTIC_INDEX_CACHE="1",
        IPFS_DATASETS_SEMANTIC_IDENTITY_REUSE="1", IPFS_DATASETS_SEMANTIC_BLOCK_CACHE="1",
        PYTHONPATH=os.pathsep.join(map(str, (ROOT, DATASETS, KIT))),
        IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR=str(private / "orchestration"),
        IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=str(private / "seal.duckdb"))
    removed = ("IPFS_ACCELERATE_AGENT_STATE_OWNER_SOCKET", "IPFS_ACCELERATE_AGENT_QUACK_TOKEN",
        "IPFS_ACCELERATE_META_INDEX_DUCKDB", "IPFS_ACCELERATE_META_INDEX_DUCKLAKE",
        "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "IPFS_TEST_PROOF_REUSE_MODE")
    env = os.environ.copy()
    for name in removed:
        env.pop(name, None)
    env.update(updates)
    argv = [sys.executable, "-B", "-m", "pytest", "-c", "pytest.ini", str(TARGET), "--color=no",
        "--basetemp=" + str(private / "pytest-temp"), "-o", "log_cli=false",
        "-o", "cache_dir=" + str(private / "pytest-cache"), "--tb=short", "--maxfail=1",
        "--junitxml=" + str(attempt / "junit.xml")]
    record = dict(schema="source-query-dependency-qualification@1", status="running",
        started_at_utc=utc(), accelerator_execution_head=CHILD_REVISION,
        datasets_execution_revision=DATASETS_REVISION, datasets_execution_path=str(DATASETS),
        kit_execution_revision=KIT_REVISION, cwd=str(ROOT), targets=[str(TARGET)], argv=argv,
        pytest_invocations=1, separate_collection=False, full_upstream_module=True,
        before=before, git_state_before=before_git, source_snapshots=snapshots,
        interpreter=sys.executable, interpreter_version=sys.version, platform=platform.platform(),
        distribution_versions={name: importlib.metadata.version(name) for name in
                               ("pytest", "duckdb", "cryptography", "numpy", "multiformats")},
        environment_updates=updates, inherited_environment_names_removed=list(removed),
        private_runtime_directory=str(private), child_pytest_configuration_selected=True,
        datasets_pytest_configuration_selected=False, datasets_ancestor_conftests_bound=True,
        auto_dependency_install_disabled=True, mandatory_ast_seal_configured=True,
        fresh_private_seal_catalog=True,
        source_binding_scope="listed_standalone_dependencies_not_complete_import_closure",
        complete_import_closure_attested=False, model_calls=0, provider_calls=0, training_steps=0,
        solver_calls=0, target_program_executions=0, child_processes_launched=False,
        native_task_owners_constructed=False, model_weights_or_embeddings_loaded=False,
        docker_run=False, runtime_databases_published=False, official_task_result=None,
        proof_authority=False, execution_authority=False, completion_authority=False,
        worker_uid_qualified=False, supervisor_qualification_counts_added=False,
        previous_counts_added=False, total_session_token_savings_established=False)
    write(attempt / "commands.json", record)
    print("Starting " + args.attempt + ": one full upstream module, " + str(len(before)) + " source bindings", flush=True)
    started = time.monotonic()
    with (attempt / "tests.stdout.log").open("wb") as stdout, (attempt / "tests.stderr.log").open("wb") as stderr:
        result = subprocess.run(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
    record.update(pytest_exit_code=result.returncode, completed_at_utc=utc(),
                  elapsed_seconds=round(time.monotonic() - started, 3))
    record["after"], record["git_state_after"] = bindings(selected), states()
    record["source_before_after_equal"] = record["before"] == record["after"]
    record["fixed_heads_and_indexes_unchanged"] = all(
        all(before_git[root][key] == record["git_state_after"][root][key] for key in ("head", "index_sha256"))
        for root in before_git)
    record["observed_shared_ref_changes"] = [dict(path=root, before=before_git[root]["refs_sha256"],
        after=record["git_state_after"][root]["refs_sha256"], actor_attributed=False)
        for root in before_git if before_git[root]["refs_sha256"] != record["git_state_after"][root]["refs_sha256"]]
    for name in ("tests.stdout.log", "tests.stderr.log", "junit.xml"):
        path = attempt / name
        record[name] = dict(bytes=path.stat().st_size, sha256=sha(path.read_bytes())) if path.is_file() else None
    counts = re.findall(r"^(?:collecting \.\.\. )?collected ([0-9]+) items?\s*$",
                        (attempt / "tests.stdout.log").read_text(errors="replace"), re.M)
    record["collected_tests"] = int(counts[0]) if len(counts) == 1 else None
    cases, outcome = [], None
    if (attempt / "junit.xml").is_file():
        xml = ET.parse(attempt / "junit.xml").getroot()
        cases = [(row.get("classname"), row.get("name")) for row in xml.iter("testcase")]
        outcome = dict(tests=len(cases), failures=sum(1 for _ in xml.iter("failure")),
                       errors=sum(1 for _ in xml.iter("error")), skipped=sum(1 for _ in xml.iter("skipped")))
    record.update(junit=outcome, distinct_junit_cases=len(set(cases)),
                  suites=dict(collections.Counter(suite for suite, _name in cases)))
    passed = (result.returncode == 0 and record["source_before_after_equal"]
        and record["fixed_heads_and_indexes_unchanged"] and len(cases) > 0
        and record["collected_tests"] == len(cases) == len(set(cases))
        and outcome == dict(tests=len(cases), failures=0, errors=0, skipped=0))
    record["status"] = "pass" if passed else "fail"
    write(attempt / "commands.json", record)
    print(json.dumps({key: record[key] for key in
        ("status", "pytest_exit_code", "junit", "distinct_junit_cases", "elapsed_seconds")}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
