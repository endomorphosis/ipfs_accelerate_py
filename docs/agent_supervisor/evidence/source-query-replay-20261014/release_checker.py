"""Check retained query replay evidence and run only the four offline doc gates.

No tests, source programs, native owners, providers, models or provers execute.
Git is read only. Shared refs are observations, never a fixed-head assertion.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
BASE = "b37a248a74d2cbad55dec7b1289aa7cfb367d7e4"
HISTORICAL_DATASETS = Path("/home/barberb/lift_coding/.worktrees/router-source-delivery-fences-datasets-20261014")
HISTORICAL_DATASETS_REVISION = "4b50ae4aac65da9166bd3b24608f32ba7e9fb23d"
DATASETS = Path("/home/barberb/lift_coding/.worktrees/source-query-replay-datasets-20261014")
DATASETS_REVISION = "f4a823231ad52c9252504810a1b00ed4738346b6"
DATASET_PROFILES = {
    str(HISTORICAL_DATASETS): HISTORICAL_DATASETS_REVISION,
    str(DATASETS): DATASETS_REVISION,
}
BASELINE_PROFILES = {
    "baseline-01": (BASE, HISTORICAL_DATASETS, HISTORICAL_DATASETS_REVISION),
    "baseline-02": ("f15c146467c29c95f39ae1f041c8eb84e76d8bc8", DATASETS, DATASETS_REVISION),
}
KIT = Path("/home/barberb/lift_coding/.worktrees/repository-proof-context-kit-20261014")
KIT_REVISION = "40c98c71b95be750509ffdb21e7e5494d91b4e79"
PRODUCTIONS = (
    "ipfs_accelerate_py/agent_supervisor/runtime/current_source_lookup.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/typed_state_owner.py",
)
NEW_TEST = "test/api/test_source_query_replay.py"
DRIVER = str((PHASE / "qualification_runner.py").relative_to(ROOT))
CHANGED = (*PRODUCTIONS, NEW_TEST, DRIVER)
ROSTER = (
    "test/api/test_source_query_replay.py",
    "test/api/test_current_source_lookup.py",
    "test/api/test_source_worker_construction.py",
    "test/api/test_native_source_worker_delivery.py",
    "test/api/test_source_lookup_worker_channel.py",
    "test/api/test_native_source_worker_spawn.py",
    "test/api/test_native_source_worker_spawn_custody.py",
    "test/api/test_native_source_worker_spawn_lock_order.py",
    "test/api/test_native_nominated_source_allocation.py",
    "test/api/test_native_nominated_source_allocation_custody.py",
    "test/api/test_current_source_worktree_allocation.py",
)
NEW_SUITE = NEW_TEST.removesuffix(".py").replace("/", ".")
EXPECTED_TESTS = 250
EXPECTED_NEW_CASES = 27
DEPENDENCY_EXECUTION_HEAD = "f15c146467c29c95f39ae1f041c8eb84e76d8bc8"
DEPENDENCY_TARGET = "tests/unit/logic/software_contracts/semantic_state/test_api.py"
DEPENDENCY_DRIVER = str((PHASE / "dependency_qualification_runner.py").relative_to(ROOT))
HISTORICAL_DOC_RECEIPT_SHA = "f71deac12af40228af46d9e1062d774e6b9422fc94ef8fcc5ea8d1cbf826832f"
BASELINE_NODE = NEW_TEST + "::test_native_query_cas_callback_reenters_native_fences_after_server_release"
BASELINE_DIAGNOSTIC = "native query CAS replay must release SERVER"
DOCUMENTS = (
    str((PHASE / "README.md").relative_to(ROOT)),
    "docs/agent_supervisor/source_lookup_worker.md",
    "docs/agent_supervisor/reversible_codebase_views_plan.md",
)
UTILITY = ROOT / "docs/agent_supervisor/evidence/router-signed-scope-closing-20261014/doc_qualification_runner.py"
UTILITY_SHA = "efefd2fd4764cfe460adbf167a4cf30b2c739f7e064e0d2f3209a1fdce1fce68"
WORKFLOW = ROOT / ".github/workflows/documentation-gates.yml"
WORKFLOW_SHA = "0a46fa5b8905600cb9f74adaaec8b8aeb5f89878612a43fb7a847e586bde3763"
LINK_CHECKER = ROOT / "scripts/docs/check_current_docs_links.py"
LINK_CHECKER_SHA = "596a0038b18d592f2e605643890afc85e457558f7b887932a474557d7586819e"
VOCAB_CHECKER = ROOT / "scripts/docs/check_agent_supervisor_docs.py"
VOCAB_CHECKER_SHA = "09580aa9a8707ce8da278d5b7fef4055e1945e53266c0fc4437e516eda1fed87"
EXPECTED_NAMES = (
    "Supervisor primary-doc vocabulary", "Maintained-surface relative links",
    "Packaging version pin alignment", "Closeout navigation markers",
)


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def git(root, *args):
    return subprocess.check_output(["git", "--no-optional-locks", *args], cwd=root)


def read(path):
    need(path.resolve(strict=True) == path and stat.S_ISREG(path.lstat().st_mode),
         "canonical regular input required: " + str(path))
    return path.read_bytes()


def relative(root, name):
    need(type(name) is str and name and not PurePosixPath(name).is_absolute()
         and PurePosixPath(name).as_posix() == name
         and all(part not in (".", "..") for part in PurePosixPath(name).parts),
         "canonical relative input path required")
    return root / name


def bind(path):
    raw = read(path)
    return dict(path=str(path.relative_to(ROOT)), bytes=len(raw), sha256=sha(raw))


def digest_row(raw, row, label):
    need(type(row["bytes"]) is int and row["bytes"] >= 0
         and type(row["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
         and (len(raw), sha(raw)) == (row["bytes"], row["sha256"]), label)


def index_state(root):
    return dict(head=git(root, "rev-parse", "HEAD").decode().strip(),
                index_sha256=sha(git(root, "ls-files", "--stage", "-z")))


def published_index_sha(root, revision):
    rows = []
    for entry in git(root, "ls-tree", "-r", "-z", revision).split(b"\0"):
        if entry:
            metadata, name = entry.split(b"\t", 1)
            mode, _kind, oid = metadata.split()
            rows.append(mode + b" " + oid + b" 0\t" + name + b"\0")
    return sha(b"".join(rows))


def live_git(execution_head, dataset_root, dataset_revision):
    fixed, refs = {}, {}
    for root, expected in ((ROOT, execution_head), (dataset_root, dataset_revision), (KIT, KIT_REVISION)):
        fixed[str(root)] = index_state(root)
        need(fixed[str(root)]["head"] == expected, "fixed execution HEAD differs: " + str(root))
        refs[str(root)] = sha(git(root, "for-each-ref", "--format=%(refname) %(objectname)"))
    return fixed, refs


def prior_evidence():
    """Compare Git metadata, without reading/hashing the old working files."""
    base = git(ROOT, "ls-tree", "-r", "-z", BASE, "--", "docs/agent_supervisor/evidence")
    staged = {}
    for entry in git(ROOT, "ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, oid, stage = metadata.split()
        if stage == b"0":
            staged[name] = (mode, oid)
    count = 0
    for entry in base.split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, kind, oid = metadata.split()
        need(kind == b"blob" and staged.get(name) == (mode, oid),
             "prior evidence Git mode/blob is not retained in stage zero: " + name.decode())
        count += 1
    need(count > 0, "published prior evidence population is empty")
    return dict(revision=BASE, comparison="published_tree_vs_current_stage_zero_index",
                file_count=count, tree_records_sha256=sha(base), all_modes_and_blob_ids_match=True,
                old_working_files_read=False, complete_import_closure_attested=False)


def inventory():
    rows = json.loads(read(PHASE / "source-inventory.json"))
    need(type(rows) is list and len(rows) == 277, "exact 277 listed dependencies required")
    for row in rows:
        need(type(row) is dict and set(row) == {"repository", "path"}
             and row["repository"] in {"accelerator", "datasets_execution_checkout"},
             "listed dependency identity differs")
        relative(ROOT, row["path"])
    identities = [(row["repository"], row["path"]) for row in rows]
    need(len(set(identities)) == 277
         and sum(row["repository"] == "accelerator" for row in rows) == 226
         and sum(row["repository"] == "datasets_execution_checkout" for row in rows) == 51,
         "listed dependency population differs")
    need(all(("accelerator", path) in identities for path in CHANGED),
         "production, new tests and qualification driver must be listed")
    return rows


def junit(path):
    root = ET.fromstring(read(path))
    cases = list(root.iter("testcase"))
    names = [(case.get("classname"), case.get("name")) for case in cases]
    need(names and all(type(suite) is str and suite and type(name) is str and name
                       for suite, name in names) and len(set(names)) == len(names),
         "unique exact JUnit identities required")
    outcome = dict(tests=len(cases), failures=sum(1 for _ in root.iter("failure")),
                   errors=sum(1 for _ in root.iter("error")), skipped=sum(1 for _ in root.iter("skipped")))
    suites = list(root.iter("testsuite"))
    need(suites and all(sum(int(suite.get(key, "-1")) for suite in suites) == value
                        for key, value in outcome.items()), "JUnit aggregate counters differ")
    return cases, names, outcome


def evidence_limits(report, *, baseline):
    need(report["baseline_control"] is baseline
         and report["actual_inert_child_probes"] is (not baseline)
         and report["authored_listener_and_deployment_metadata_seams"] is True
         and report["new_suite_child_processes_launched"] is False
         and report["retained_suites_include_actual_inert_child_probes"] is (not baseline)
         and report["pytest_invocations"] == 1 and type(report["pytest_invocations"]) is int
         and report["separate_collection"] is False,
         "qualification invocation/authored seam limits differ")
    for name in ("worker_uid_qualified", "production_coding_child_adoption", "universal_cas_callback_safety",
                 "proof_authority", "execution_authority", "completion_authority", "docker_run",
                 "total_session_token_savings_established", "runtime_databases_published", "previous_counts_added"):
        need(report[name] is False, "qualification exceeds its scope: " + name)
    need(report["official_task_result"] is None
         and report["source_binding_scope"] == "listed_dependencies_not_complete_import_closure",
         "official/import closure claim differs")
    for name in ("provider_calls", "model_calls", "training_steps", "solver_calls", "target_program_executions"):
        need(type(report[name]) is int and report[name] == 0, "unexpected execution: " + name)


def dataset_profile(report):
    """Authenticate the historical or fresh checkout, without opening an owner."""
    rows = report["before"]
    need(type(rows) is list and all(type(row) is dict for row in rows), "captured source rows differ")
    selected = [row for row in rows if row.get("repository") == "datasets_execution_checkout"]
    need(len(selected) == 51, "exact captured datasets population required")
    names = {row["source_checkout"] for row in selected}
    need(len(names) == 1, "captured datasets checkout is mixed")
    name = next(iter(names))
    revision = report["datasets_execution_revision"]
    need(DATASET_PROFILES.get(name) == revision
         and report.get("datasets_execution_path", name) == name,
         "dataset checkout/revision is outside the two explicit execution profiles")
    need(report["environment_updates"]["PYTHONPATH"] == os.pathsep.join((str(ROOT), name, str(KIT))),
         "qualification PYTHONPATH differs from its exact captured checkouts")
    return Path(name), revision


def attempt_record(attempt, listed, *, baseline=False):
    report = json.loads(read(attempt / "commands.json"))
    execution_head = report["accelerator_execution_head"]
    need(report["schema"] == "source-query-replay-qualification@1"
         and report["status"] in ({"expected_regression"} if baseline else {"pass", "fail"})
         and report["accelerator_base"] == BASE
         and type(execution_head) is str and re.fullmatch(r"[0-9a-f]{40}", execution_head)
         and report["kit_execution_revision"] == KIT_REVISION
         and report["cwd"] == str(ROOT), "retained qualification identity/status differs")
    git(ROOT, "merge-base", "--is-ancestor", BASE, execution_head)
    datasets, datasets_revision = dataset_profile(report)
    if baseline:
        need(attempt.name in BASELINE_PROFILES, "unknown baseline execution profile")
        expected_head, expected_datasets, expected_revision = BASELINE_PROFILES[attempt.name]
        need((execution_head, datasets, datasets_revision)
             == (expected_head, expected_datasets, expected_revision), "unpatched baseline execution profile differs")
    targets = [BASELINE_NODE] if baseline else list(ROSTER)
    need(report["targets"] == targets, "exact retained target roster differs")
    private = Path(report["private_runtime_directory"])
    need(private.is_absolute() and type(report["interpreter"]) is str
         and Path(report["interpreter"]).is_absolute(), "qualification runtime/interpreter namespace differs")
    expected_argv = [report["interpreter"], "-B", "-m", "pytest", "-c", "pytest.ini", *targets,
        "--color=no", "--basetemp=" + str(private / "pytest-temp"), "-o", "log_cli=false",
        "-o", "cache_dir=" + str(private / "pytest-cache"), "--tb=short", "--maxfail=1",
        "--junitxml=" + str(attempt / "junit.xml")]
    need(report["argv"] == expected_argv, "retained pytest invocation differs from the exact declared roster")
    evidence_limits(report, baseline=baseline)
    need(report["source_before_after_equal"] is True and report["before"] == report["after"]
         and report["fixed_heads_and_indexes_unchanged"] is True, "qualification source/index custody differs")
    before_git, after_git = report["git_state_before"], report["git_state_after"]
    need(set(before_git) == set(after_git) == {str(ROOT), str(datasets), str(KIT)},
         "qualification checkout population differs")
    for root, revision in ((ROOT, execution_head), (datasets, datasets_revision), (KIT, KIT_REVISION)):
        a, b = before_git[str(root)], after_git[str(root)]
        need(a["head"] == b["head"] == revision
             and a["index_sha256"] == b["index_sha256"] == published_index_sha(root, revision),
             "qualification fixed HEAD/index changed")
    observed = [dict(path=root, before=before_git[root]["refs_sha256"],
                     after=after_git[root]["refs_sha256"], actor_attributed=False)
                for root in (str(ROOT), str(datasets), str(KIT))
                if before_git[root]["refs_sha256"] != after_git[root]["refs_sha256"]]
    need(report["observed_shared_ref_changes"] == observed, "shared-ref observations differ")
    bindings = report["before"]
    need(type(bindings) is list and [(row["repository"], row["path"]) for row in bindings]
         == [(row["repository"], row["path"]) for row in listed], "source inventory differs")
    for row in bindings:
        need(set(row) in ({"repository", "path", "source_checkout", "bytes", "sha256"},
                          {"repository", "path", "source_checkout", "bytes", "sha256", "matches_immutable_git_object"}),
             "source binding metadata schema differs")
        owner = ROOT if row["repository"] == "accelerator" else datasets
        need(row["source_checkout"] == str(owner), "source checkout namespace differs")
        relative(owner, row["path"])
        if row["repository"] != "accelerator" or row["path"] not in CHANGED:
            revision = execution_head if owner == ROOT else datasets_revision
            digest_row(git(owner, "show", revision + ":" + row["path"]), row,
                       "immutable listed source differs: " + row["path"])
            need(row.get("matches_immutable_git_object") is True, "immutable source marker differs")
    snapshots = report["frozen_changed_source_snapshots"]
    need(type(snapshots) is list and [row["path"] for row in snapshots] == list(CHANGED),
         "exact production/test/driver snapshots required")
    original_rows = {row["path"]: row for row in bindings if row["repository"] == "accelerator"}
    for row in snapshots:
        expected = attempt / "source-snapshot" / (row["path"] + ".snapshot.txt")
        need(relative(ROOT, row["snapshot"]) == expected, "snapshot path escaped its retained attempt")
        raw = read(expected)
        digest_row(raw, row, "retained snapshot custody differs")
        digest_row(raw, original_rows[row["path"]], "snapshot differs from captured source")
        if baseline and row["path"] in PRODUCTIONS:
            need(raw == git(ROOT, "show", BASE + ":" + row["path"]), "baseline production was patched")
            need(raw == git(ROOT, "show", execution_head + ":" + row["path"]),
                 "baseline production differs from its actual execution commit")
    for name in ("tests.stdout.log", "tests.stderr.log", "junit.xml"):
        raw = read(attempt / name)
        need(report[name] == dict(bytes=len(raw), sha256=sha(raw)), "raw qualification output custody differs")
    stdout = read(attempt / "tests.stdout.log").decode("utf-8", errors="replace")
    counts = re.findall(r"^(?:collecting \.\.\. )?collected ([0-9]+) items?\s*$", stdout, re.M)
    cases, names, outcome = junit(attempt / "junit.xml")
    need(len(counts) == 1 and type(report["collected_tests"]) is int
         and report["collected_tests"] == int(counts[0])
         and type(report["distinct_junit_cases"]) is int
         and report["distinct_junit_cases"] == len(names)
         and report["junit"] == outcome
         and report["suites"] == dict(collections.Counter(suite for suite, _ in names)),
         "raw collection/JUnit identities differ")
    need(type(report["pytest_exit_code"]) is int and type(report["elapsed_seconds"]) in (int, float)
         and report["elapsed_seconds"] >= 0, "qualification exit/time differs")
    if baseline:
        failure = list(cases[0].iter("failure"))
        need(report["pytest_exit_code"] == 1 and report["collected_tests"] == 1
             and outcome == dict(tests=1, failures=1, errors=0, skipped=0)
             and names == [(NEW_SUITE, BASELINE_NODE.split("::", 1)[1])]
             and len(failure) == 1 and BASELINE_DIAGNOSTIC in stdout
             and BASELINE_DIAGNOSTIC in ((failure[0].text or "") + failure[0].get("message", "")),
             "unpatched control did not show the exact expected SERVER-lock regression")
    elif report["status"] == "pass":
        need(report["pytest_exit_code"] == 0 and report["collected_tests"] == len(names) == EXPECTED_TESTS
             and outcome == dict(tests=len(names), failures=0, errors=0, skipped=0)
             and {suite for suite, _ in names} == {path.removesuffix(".py").replace("/", ".") for path in ROSTER}
             and sum(suite == NEW_SUITE for suite, _ in names) == EXPECTED_NEW_CASES,
             "final full qualification requires eleven suites and the exact successful new query cases")
    else:
        need(report["pytest_exit_code"] != 0 or outcome["failures"] or outcome["errors"]
             or outcome["skipped"] or report["collected_tests"] != len(names),
             "retained failed attempt has no recorded unsuccessful result")
    return report


def qualification():
    need(json.loads(read(PHASE / "targets.json")) == list(ROSTER), "exact eleven-suite targets required")
    listed = inventory()
    baselines = sorted(path for path in PHASE.glob("baseline-*") if path.is_dir())
    need([path.name for path in baselines] == ["baseline-01", "baseline-02"],
         "exact historical and fresh baseline controls required")
    baseline_records = [(path, attempt_record(path, listed, baseline=True)) for path in baselines]
    attempts = [path for path in PHASE.glob("qualification-*") if path.is_dir()]
    need(attempts and all(re.fullmatch(r"qualification-[0-9]{2,3}", path.name) for path in attempts),
         "canonical qualification attempts required")
    attempts.sort(key=lambda path: int(path.name.split("-")[1]))
    records = [(path, attempt_record(path, listed)) for path in attempts]
    selected, report = records[-1]
    need(report["status"] == "pass", "latest full qualification attempt must pass")
    datasets, datasets_revision = dataset_profile(report)
    need((datasets, datasets_revision) == (DATASETS, DATASETS_REVISION),
         "latest release requires the freshly qualified dataset profile")
    checked = []
    for row in report["before"]:
        owner = ROOT if row["repository"] == "accelerator" else datasets
        digest_row(read(relative(owner, row["path"])), row, "qualified current source differs: " + row["path"])
        checked.append(row)
    fresh_test = next(row for row in baseline_records[-1][1]["frozen_changed_source_snapshots"]
                      if row["path"] == NEW_TEST)
    selected_test = next(row for row in checked if row["repository"] == "accelerator" and row["path"] == NEW_TEST)
    need((fresh_test["bytes"], fresh_test["sha256"]) == (selected_test["bytes"], selected_test["sha256"]),
         "fresh baseline must capture the final qualified query test source")
    controls = []
    for path, baseline in baseline_records:
        baseline_datasets, baseline_revision = dataset_profile(baseline)
        controls.append(dict(commands_binding=bind(path / "commands.json"), diagnostic=BASELINE_DIAGNOSTIC,
            expected_failures=1, elapsed_seconds=baseline["elapsed_seconds"], unpatched_production_revision=BASE,
            accelerator_execution_head=baseline["accelerator_execution_head"], datasets_execution_revision=baseline_revision,
            datasets_execution_checkout=str(baseline_datasets), promoted_to_pass=False))
    return dict(qualification_path=str(selected.relative_to(ROOT)),
                qualification_sha256=sha(read(selected / "commands.json")),
                qualification_binding=bind(selected / "commands.json"),
                accelerator_base=BASE, accelerator_execution_head=report["accelerator_execution_head"],
                datasets_execution_path=str(datasets), datasets_execution_revision=datasets_revision,
                selected_tests=report["collected_tests"], suites=11, new_cases=EXPECTED_NEW_CASES,
                source_bindings=checked, source_binding_scope="listed_dependencies_not_complete_import_closure",
                baseline_control=controls[-1], baseline_controls=controls,
                attempts=[dict(commands_binding=bind(path / "commands.json"), status=record["status"],
                               accelerator_execution_head=record["accelerator_execution_head"],
                               datasets_execution_path=str(dataset_profile(record)[0]),
                               datasets_execution_revision=record["datasets_execution_revision"],
                               junit=record["junit"], selected_for_release=path == selected) for path, record in records])


def dependency_qualification():
    """Recount the separate cache module without increasing primary counts."""
    attempt = PHASE / "dependency-01"
    report = json.loads(read(attempt / "commands.json"))
    target = str(DATASETS / DEPENDENCY_TARGET)
    need(report["schema"] == "source-query-dependency-qualification@1"
         and report["status"] == "pass" and type(report["pytest_exit_code"]) is int
         and report["pytest_exit_code"] == 0
         and report["accelerator_execution_head"] == DEPENDENCY_EXECUTION_HEAD
         and report["datasets_execution_path"] == str(DATASETS)
         and report["datasets_execution_revision"] == DATASETS_REVISION
         and report["kit_execution_revision"] == KIT_REVISION
         and report["cwd"] == str(ROOT) and report["targets"] == [target],
         "standalone dependency execution identity differs")
    for name in ("full_upstream_module", "child_pytest_configuration_selected", "datasets_ancestor_conftests_bound",
                 "auto_dependency_install_disabled", "mandatory_ast_seal_configured", "fresh_private_seal_catalog",
                 "source_before_after_equal", "fixed_heads_and_indexes_unchanged"):
        need(report[name] is True, "standalone dependency guard differs: " + name)
    for name in ("separate_collection", "datasets_pytest_configuration_selected", "complete_import_closure_attested",
                 "child_processes_launched", "native_task_owners_constructed", "model_weights_or_embeddings_loaded",
                 "docker_run", "runtime_databases_published", "proof_authority", "execution_authority",
                 "completion_authority", "worker_uid_qualified", "supervisor_qualification_counts_added",
                 "previous_counts_added", "total_session_token_savings_established"):
        need(report[name] is False, "standalone dependency exceeds its scope: " + name)
    need(type(report["pytest_invocations"]) is int and report["pytest_invocations"] == 1
         and report["official_task_result"] is None
         and report["source_binding_scope"] == "listed_standalone_dependencies_not_complete_import_closure",
         "standalone dependency invocation/coverage differs")
    for name in ("model_calls", "provider_calls", "training_steps", "solver_calls", "target_program_executions"):
        need(type(report[name]) is int and report[name] == 0, "standalone dependency invoked " + name)
    private = Path(report["private_runtime_directory"])
    need(private.is_absolute() and type(report["interpreter"]) is str
         and Path(report["interpreter"]).is_absolute(), "standalone private execution paths differ")
    need(report["argv"] == [report["interpreter"], "-B", "-m", "pytest", "-c", "pytest.ini", target,
        "--color=no", "--basetemp=" + str(private / "pytest-temp"), "-o", "log_cli=false",
        "-o", "cache_dir=" + str(private / "pytest-cache"), "--tb=short", "--maxfail=1",
        "--junitxml=" + str(attempt / "junit.xml")], "standalone dependency argv differs")
    updates = report["environment_updates"]
    need(updates["PYTHONPATH"] == os.pathsep.join(map(str, (ROOT, DATASETS, KIT)))
         and updates["IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB"] == str(private / "seal.duckdb")
         and updates["IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR"] == str(private / "orchestration"),
         "standalone dependency private environment differs")
    for name in ("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "IPFS_DATASETS_SEMANTIC_INDEX_CACHE",
                 "IPFS_DATASETS_SEMANTIC_IDENTITY_REUSE", "IPFS_DATASETS_SEMANTIC_BLOCK_CACHE"):
        need(updates[name] == "1", "standalone cache/seal setting differs: " + name)
    need(updates["IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS"] == "0", "automatic dependency installation enabled")
    before_git, after_git = report["git_state_before"], report["git_state_after"]
    need(set(before_git) == set(after_git) == {str(ROOT), str(DATASETS), str(KIT)},
         "standalone dependency checkout population differs")
    for root, revision in ((ROOT, DEPENDENCY_EXECUTION_HEAD), (DATASETS, DATASETS_REVISION), (KIT, KIT_REVISION)):
        a, b = before_git[str(root)], after_git[str(root)]
        need(a["head"] == b["head"] == revision
             and a["index_sha256"] == b["index_sha256"] == published_index_sha(root, revision),
             "standalone dependency fixed HEAD/index differs")
    observed = [dict(path=root, before=before_git[root]["refs_sha256"], after=after_git[root]["refs_sha256"],
                     actor_attributed=False) for root in (str(ROOT), str(DATASETS), str(KIT))
                if before_git[root]["refs_sha256"] != after_git[root]["refs_sha256"]]
    need(report["observed_shared_ref_changes"] == observed, "standalone dependency ref observations differ")
    expected = sorted([
        ("accelerator", "conftest.py"), ("accelerator", DEPENDENCY_DRIVER),
        ("accelerator", "ipfs_accelerate_py/testing/pytest_ast_seal.py"), ("accelerator", "pytest.ini"),
        ("datasets", "conftest.py"), ("datasets", "ipfs_datasets_py/logic/software_contracts/content.py"),
        ("datasets", "ipfs_datasets_py/logic/software_contracts/semantic_state/api.py"),
        ("datasets", "ipfs_datasets_py/logic/software_contracts/semantic_state/models.py"),
        ("datasets", "pytest.ini"), ("datasets", "tests/conftest.py"),
        ("datasets", "tests/unit/conftest.py"), ("datasets", DEPENDENCY_TARGET),
    ])
    rows, snapshots = report["before"], report["source_snapshots"]
    need(type(rows) is list and len(rows) == 12 and report["after"] == rows
         and [(row["repository"], row["path"]) for row in rows] == expected
         and type(snapshots) is list and len(snapshots) == 12,
         "exact standalone twelve-input population differs")
    for row, snapshot in zip(rows, snapshots):
        need(set(row) == {"repository", "path", "source_checkout", "bytes", "sha256", "matches_immutable_git_object"},
             "standalone input metadata schema differs")
        owner = ROOT if row["repository"] == "accelerator" else DATASETS
        need(row["source_checkout"] == str(owner), "standalone input source namespace differs")
        raw = read(relative(owner, row["path"]))
        digest_row(raw, row, "standalone current input differs")
        immutable = row["repository"] != "accelerator" or row["path"] != DEPENDENCY_DRIVER
        need(row["matches_immutable_git_object"] is immutable, "standalone immutable input marker differs")
        if immutable:
            revision = DEPENDENCY_EXECUTION_HEAD if owner == ROOT else DATASETS_REVISION
            need(raw == git(owner, "show", revision + ":" + row["path"]), "standalone immutable Git input differs")
        expected_snapshot = attempt / "source-snapshot" / row["repository"] / (row["path"] + ".snapshot.txt")
        need(set(snapshot) == set(row) | {"snapshot"}
             and {key: value for key, value in snapshot.items() if key != "snapshot"} == row
             and relative(ROOT, snapshot["snapshot"]) == expected_snapshot,
             "standalone snapshot identity differs")
        need(read(expected_snapshot) == raw, "standalone frozen snapshot differs")
    for name in ("tests.stdout.log", "tests.stderr.log", "junit.xml"):
        raw = read(attempt / name)
        need(report[name] == dict(bytes=len(raw), sha256=sha(raw)), "standalone raw output custody differs")
    stdout = read(attempt / "tests.stdout.log").decode("utf-8", errors="replace")
    counts = re.findall(r"^(?:collecting \.\.\. )?collected ([0-9]+) items?\s*$", stdout, re.M)
    xml = ET.fromstring(read(attempt / "junit.xml"))
    cases = list(xml.iter("testcase"))
    names = [(case.get("classname"), case.get("name")) for case in cases]
    # Pytest reports an empty classname for this absolute, out-of-child target.
    # Bind the actual blank suite, exact target/argv and nonempty unique names.
    need(len(counts) == 1 and int(counts[0]) == 45
         and type(report["collected_tests"]) is int and report["collected_tests"] == 45
         and type(report["distinct_junit_cases"]) is int and report["distinct_junit_cases"] == 45
         and len(names) == len(set(names)) == 45
         and all(suite == "" and type(name) is str and name for suite, name in names),
         "standalone raw collection/unique identities differ")
    outcome = dict(tests=45, failures=sum(1 for _ in xml.iter("failure")),
                   errors=sum(1 for _ in xml.iter("error")), skipped=sum(1 for _ in xml.iter("skipped")))
    suites = list(xml.iter("testsuite"))
    need(outcome == report["junit"] == dict(tests=45, failures=0, errors=0, skipped=0)
         and report["suites"] == {"": 45}
         and suites and all(sum(int(suite.get(key, "-1")) for suite in suites) == value
                            for key, value in outcome.items()), "standalone JUnit outcome differs")
    need(type(report["elapsed_seconds"]) in (int, float) and report["elapsed_seconds"] >= 0,
         "standalone duration differs")
    return dict(qualification_path=str(attempt.relative_to(ROOT)),
        qualification_sha256=sha(read(attempt / "commands.json")), commands_binding=bind(attempt / "commands.json"),
        status="pass", tests=45, source_bindings=rows, source_snapshots=snapshots, targets=[target],
        accelerator_execution_head=DEPENDENCY_EXECUTION_HEAD, datasets_execution_path=str(DATASETS),
        datasets_execution_revision=DATASETS_REVISION, kit_execution_revision=KIT_REVISION,
        elapsed_seconds=report["elapsed_seconds"], source_binding_scope=report["source_binding_scope"],
        counts_aggregated_with_primary=False)


def historical_documentation():
    """Keep the completed old doc receipt and its original inputs historical."""
    path = PHASE / "documentation-checks.json"
    raw = read(path)
    need(sha(raw) == HISTORICAL_DOC_RECEIPT_SHA, "completed historical documentation receipt changed")
    report = json.loads(raw)
    need(report["status"] == "pass" and report["before"] == report["after"]
         and len(report["steps"]) == 4 and all(row["exit_code"] == 0 for row in report["steps"]),
         "historical documentation outcome differs")
    log = PHASE / "documentation-workflow.log"
    need(bind(log) == report["log_binding"], "historical documentation log changed")
    q2 = PHASE / "qualification-02/commands.json"
    need(sha(read(q2)) == report["before"]["qualification_sha256"]
         and report["before"]["source_bindings"] == json.loads(read(q2))["before"],
         "historical documentation source qualification differs")
    old_checker = next(row for row in report["before"]["read_guards"]
                       if row["path"] == str(Path(__file__).resolve().relative_to(ROOT)))
    inputs = [*report["before"]["documents"], old_checker]
    snapshots = []
    for row in inputs:
        snapshot = PHASE / "historical-documentation/source-snapshot" / (row["path"] + ".snapshot.txt")
        digest_row(read(snapshot), row, "original historical documentation input snapshot differs")
        snapshots.append(bind(snapshot))
    return dict(receipt_binding=bind(path), log_binding=bind(log), source_qualification_binding=bind(q2),
                original_input_snapshots=snapshots, status="historical_pass", workflow_blocks=4,
                relative_links=report["relative_links"], anchors=report["anchors"],
                reexecuted=False, current_source_authority=False)


def load_stdlib_utility(path, expected, name):
    raw = read(path)
    need(sha(raw) == expected and raw == git(ROOT, "show", BASE + ":" + str(path.relative_to(ROOT))),
         "reviewed documentation utility differs")
    # Both pinned utilities contain only standard-library imports and a guarded
    # main. Never call the old phase's guard, qualification or main functions.
    prior = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        need(spec is not None and spec.loader is not None, "documentation utility cannot load")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = prior
    need(module.ROOT == ROOT, "documentation utility root differs")
    return module


def doc_inputs(checker):
    files, missing = checker.expand_allowlist(checker.ALLOWLIST)
    need(not missing, "maintained documentation roots are missing")
    paths = set(files) | {ROOT / name for name in DOCUMENTS}
    for name in ("AGENT_SUPERVISOR_PHILOSOPHY.md", "AGENT_SUPERVISOR_ARCHITECTURE.md"):
        paths.add(ROOT / "docs/architecture" / name)
    for name in ("README.md", "PACKAGE_MAP.md", "FOR_AGENTS.md", "FOR_CONTRIBUTORS.md"):
        paths.add(ROOT / "docs/architecture/agent_supervisor" / name)
    paths.update((ROOT / "docs/architecture/agent_supervisor/packages").rglob("*.md"))
    # The manual check reads Markdown targets when an anchor is present.
    for source in list(paths):
        text = checker.strip_markup_noise(read(source).decode("utf-8"))
        for match in checker.LINK_RE.finditer(text):
            target, fragment = checker.split_target(match.group("target"))
            if fragment and not checker.is_external(target, fragment):
                path = checker.resolve_link(source, target)
                if path is not None and path.is_file() and path.suffix.lower() in checker.MD_SUFFIXES:
                    paths.add(path)
    paths.update((ROOT / "pyproject.toml", ROOT / "setup.py", ROOT / "ipfs_accelerate_py/__init__.py",
                  ROOT / "docs/development/DOCUMENTATION_VALIDATION_2026_08.md"))
    return [bind(path) for path in sorted(paths)]


def guard(checker, excluded):
    selected = qualification()
    fixed, refs = live_git(selected["accelerator_execution_head"],
                          Path(selected["datasets_execution_path"]), selected["datasets_execution_revision"])
    need(sha(read(WORKFLOW)) == WORKFLOW_SHA and sha(read(VOCAB_CHECKER)) == VOCAB_CHECKER_SHA,
         "reviewed workflow/vocabulary utility differs")
    stable = dict(selected, dependency_qualification=dependency_qualification(),
                  historical_documentation=historical_documentation(),
                  fixed_git=fixed, preserved_prior_evidence=prior_evidence(),
                  documents=[bind(ROOT / name) for name in DOCUMENTS], documentation_inputs=doc_inputs(checker),
                  read_guards=[bind(path) for path in (Path(__file__).resolve(), UTILITY, WORKFLOW, LINK_CHECKER, VOCAB_CHECKER)],
                  current_evidence=[bind(path) for path in sorted(PHASE.rglob("*"))
                                    if path.is_file() and path not in excluded])
    return stable, refs


def dump(stream, value):
    stream.seek(0)
    stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    stream.truncate()
    stream.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", help="optional two- or three-digit documentation attempt; original outputs are the default")
    args = parser.parse_args(argv)
    if args.attempt is None:
        receipt, log = PHASE / "documentation-checks.json", PHASE / "documentation-workflow.log"
    else:
        need(re.fullmatch(r"[0-9]{2,3}", args.attempt) is not None, "exact documentation attempt required")
        receipt = PHASE / ("documentation-checks-" + args.attempt + ".json")
        log = PHASE / ("documentation-workflow-" + args.attempt + ".log")
    excluded = {receipt, log, PHASE / "manifest.json"}
    need(not receipt.exists() and not log.exists(), "exclusive documentation outputs already exist")
    utility = load_stdlib_utility(UTILITY, UTILITY_SHA, "query_replay_workflow_extractor")
    checker = load_stdlib_utility(LINK_CHECKER, LINK_CHECKER_SHA, "query_replay_relative_link_checker")
    blocks = utility.extract_workflow_blocks()
    need(tuple(block["name"] for block in blocks) == EXPECTED_NAMES
         and len(blocks) == 4, "exact four reviewed workflow blocks required")
    before, before_refs = guard(checker, excluded)
    report = dict(schema="source-query-replay-documentation@1", status="running", started_at_utc=utc(),
        documentation_attempt=args.attempt,
        before=before, exact_workflow_run_blocks=blocks, workflow_sha256=WORKFLOW_SHA,
        steps=[], source_binding_scope="listed_dependencies_not_complete_import_closure",
        complete_import_closure_attested=False, prior_evidence_check_scope="Git modes/blob IDs at base versus staged index",
        additional_tests=False, native_owners_opened=False, provider_calls=0, model_calls=0, training_steps=0,
        solver_calls=0, target_program_executions=0, docker_run=False, worker_uid_qualified=False,
        production_coding_child_adoption=False, universal_cas_callback_safety=False,
        proof_authority=False, execution_authority=False, completion_authority=False,
        official_task_result=None, total_session_token_savings_established=False, hosted_ci_passed_here=False,
        refs_observed_before=before_refs, shared_ref_changes_are_observational=True,
        manifest_excluded_as_future_output=True, parent_advance_checked_here=False)
    with receipt.open("x", encoding="utf-8") as output, log.open("x", encoding="utf-8") as workflow_log:
        dump(output, report)
        try:
            with tempfile.TemporaryDirectory(prefix="sqr-doc-") as temporary:
                private = Path(temporary)
                (private / "bin").mkdir()
                (private / "tmp").mkdir()
                (private / "cache").mkdir()
                (private / "bin/python").symlink_to(Path(sys.executable).resolve(strict=True))
                env = dict(PATH=str(private / "bin") + os.pathsep + "/usr/bin:/bin",
                    LANG="C.UTF-8", LC_ALL="C.UTF-8", TZ="UTC", PYTHONDONTWRITEBYTECODE="1",
                    PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1", IPFS_KIT_DISABLE="1", IPFS_DATASETS_PY_LAZY_INSTALL_ERGOAI="0",
                    TMPDIR=str(private / "tmp"), XDG_CACHE_HOME=str(private / "cache"))
                report["private_documentation_environment"] = dict(inherited_environment_forwarded=False,
                    names=sorted(env), actual_python_executable=str(Path(sys.executable).resolve()),
                    temporary_directory_removed_after_use=True)
                for block in blocks:
                    argv = ["bash", "--noprofile", "--norc", "-euo", "pipefail", "-c", block["script"]]
                    result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
                    step = dict(name=block["name"], argv=argv, script_sha256=block["script_sha256"],
                                exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
                    report["steps"].append(step)
                    workflow_log.write(json.dumps(step, sort_keys=True) + "\n")
                    workflow_log.flush()
                    dump(output, report)
                    need(result.returncode == 0, "documentation workflow failed: " + block["name"])
            links = []
            for name in DOCUMENTS:
                path = ROOT / name
                text = checker.strip_markup_noise(read(path).decode("utf-8"))
                targets = [checker.split_target(match.group("target")) for match in checker.LINK_RE.finditer(text)]
                relative_targets = [(target, fragment) for target, fragment in targets
                                    if not checker.is_external(target, fragment)]
                links.append(dict(path=name, errors=checker.check_file(path, check_anchors=True),
                                  relative_links=len(relative_targets), anchors=sum(bool(fragment) for _, fragment in relative_targets)))
            report.update(manual_links=links, relative_links=sum(row["relative_links"] for row in links),
                          anchors=sum(row["anchors"] for row in links))
            need(all(not row["errors"] for row in links), "selected documentation links/anchors failed")
            after, after_refs = guard(checker, excluded)
            need(before == after, "documentation source/read/HEAD/index guards changed")
            report.update(after=after, refs_observed_after=after_refs,
                observed_shared_ref_changes=[dict(path=root, before=before_refs[root], after=after_refs[root],
                    actor_attributed=False) for root in before_refs if before_refs[root] != after_refs[root]],
                status="pass", completed_at_utc=utc(), log_binding=bind(log))
            dump(output, report)
        except BaseException as error:
            report.update(status="fail", completed_at_utc=utc(), failure_type=type(error).__name__,
                          failure_message=str(error), log_binding=bind(log))
            dump(output, report)
            raise
    print(json.dumps(dict(status=report["status"], workflow_blocks=len(report["steps"]),
                          relative_links=report["relative_links"], anchors=report["anchors"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
