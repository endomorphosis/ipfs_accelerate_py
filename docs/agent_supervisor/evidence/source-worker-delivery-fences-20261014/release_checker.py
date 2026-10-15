"""Check retained delivery evidence and run only the four offline doc gates.

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
BASE = "09ad10bd0e6694db7b0570614ab289184a0713c5"
DATASETS = Path("/home/barberb/lift_coding/.worktrees/router-source-delivery-fences-datasets-20261014")
DATASETS_REVISION = "4b50ae4aac65da9166bd3b24608f32ba7e9fb23d"
KIT = Path("/home/barberb/lift_coding/.worktrees/repository-proof-context-kit-20261014")
KIT_REVISION = "40c98c71b95be750509ffdb21e7e5494d91b4e79"
PRODUCTION = "ipfs_accelerate_py/agent_supervisor/runtime/source_lookup_worker_broker.py"
NEW_TEST = "test/api/test_native_source_worker_delivery.py"
DRIVER = str((PHASE / "qualification_runner.py").relative_to(ROOT))
CHANGED = (PRODUCTION, NEW_TEST, DRIVER)
ROSTER = (
    NEW_TEST,
    "test/api/test_source_lookup_worker_channel.py",
    "test/api/test_native_source_worker_spawn.py",
    "test/api/test_native_source_worker_spawn_custody.py",
    "test/api/test_native_source_worker_spawn_lock_order.py",
    "test/api/test_native_nominated_source_allocation.py",
    "test/api/test_native_nominated_source_allocation_custody.py",
    "test/api/test_current_source_worktree_allocation.py",
)
NEW_SUITE = NEW_TEST.removesuffix(".py").replace("/", ".")
BASELINE_NODE = NEW_TEST + "::test_broker_selected_cas_callback_reenters_native_fences_after_server_release[prebackend]"
BASELINE_DIAGNOSTIC = "broker-selected replay must release SERVER"
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


def live_git(execution_head):
    fixed, refs = {}, {}
    for root, expected in ((ROOT, execution_head), (DATASETS, DATASETS_REVISION), (KIT, KIT_REVISION)):
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
    need(type(rows) is list and len(rows) == 273, "exact 273 listed dependencies required")
    for row in rows:
        need(type(row) is dict and set(row) == {"repository", "path"}
             and row["repository"] in {"accelerator", "datasets_execution_checkout"},
             "listed dependency identity differs")
        relative(ROOT, row["path"])
    identities = [(row["repository"], row["path"]) for row in rows]
    need(len(set(identities)) == 273
         and sum(row["repository"] == "accelerator" for row in rows) == 222
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


def attempt_record(attempt, listed, *, baseline=False):
    report = json.loads(read(attempt / "commands.json"))
    execution_head = report["accelerator_execution_head"]
    need(report["schema"] == "source-worker-delivery-qualification@1"
         and report["status"] in ({"expected_regression"} if baseline else {"pass", "fail"})
         and report["accelerator_base"] == BASE
         and type(execution_head) is str and re.fullmatch(r"[0-9a-f]{40}", execution_head)
         and report["datasets_execution_revision"] == DATASETS_REVISION
         and report["kit_execution_revision"] == KIT_REVISION
         and report["cwd"] == str(ROOT), "retained qualification identity/status differs")
    git(ROOT, "merge-base", "--is-ancestor", BASE, execution_head)
    need(not baseline or execution_head == BASE, "unpatched baseline execution commit differs")
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
    need(set(before_git) == set(after_git) == {str(ROOT), str(DATASETS), str(KIT)},
         "qualification checkout population differs")
    for root, revision in ((ROOT, execution_head), (DATASETS, DATASETS_REVISION), (KIT, KIT_REVISION)):
        a, b = before_git[str(root)], after_git[str(root)]
        need(a["head"] == b["head"] == revision
             and a["index_sha256"] == b["index_sha256"] == published_index_sha(root, revision),
             "qualification fixed HEAD/index changed")
    observed = [dict(path=root, before=before_git[root]["refs_sha256"],
                     after=after_git[root]["refs_sha256"], actor_attributed=False)
                for root in before_git if before_git[root]["refs_sha256"] != after_git[root]["refs_sha256"]]
    need(report["observed_shared_ref_changes"] == observed, "shared-ref observations differ")
    bindings = report["before"]
    need(type(bindings) is list and [(row["repository"], row["path"]) for row in bindings]
         == [(row["repository"], row["path"]) for row in listed], "source inventory differs")
    for row in bindings:
        need(set(row) in ({"repository", "path", "source_checkout", "bytes", "sha256"},
                          {"repository", "path", "source_checkout", "bytes", "sha256", "matches_immutable_git_object"}),
             "source binding metadata schema differs")
        owner = ROOT if row["repository"] == "accelerator" else DATASETS
        need(row["source_checkout"] == str(owner), "source checkout namespace differs")
        relative(owner, row["path"])
        if row["repository"] != "accelerator" or row["path"] not in CHANGED:
            revision = execution_head if owner == ROOT else DATASETS_REVISION
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
        if baseline and row["path"] == PRODUCTION:
            need(raw == git(ROOT, "show", BASE + ":" + PRODUCTION), "baseline production was patched")
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
        need(report["pytest_exit_code"] == 0 and report["collected_tests"] == len(names)
             and outcome == dict(tests=len(names), failures=0, errors=0, skipped=0)
             and {suite for suite, _ in names} == {path.removesuffix(".py").replace("/", ".") for path in ROSTER}
             and sum(suite == NEW_SUITE for suite, _ in names) == 22,
             "final full qualification requires eight suites and 22 successful new cases")
    else:
        need(report["pytest_exit_code"] != 0 or outcome["failures"] or outcome["errors"]
             or outcome["skipped"] or report["collected_tests"] != len(names),
             "retained failed attempt has no recorded unsuccessful result")
    return report


def qualification():
    need(json.loads(read(PHASE / "targets.json")) == list(ROSTER), "exact eight-suite targets required")
    listed = inventory()
    baselines = sorted(path for path in PHASE.glob("baseline-*") if path.is_dir())
    need([path.name for path in baselines] == ["baseline-01"], "one retained baseline control required")
    baseline = attempt_record(baselines[0], listed, baseline=True)
    attempts = [path for path in PHASE.glob("qualification-*") if path.is_dir()]
    need(attempts and all(re.fullmatch(r"qualification-[0-9]{2,3}", path.name) for path in attempts),
         "canonical qualification attempts required")
    attempts.sort(key=lambda path: int(path.name.split("-")[1]))
    records = [(path, attempt_record(path, listed)) for path in attempts]
    selected, report = records[-1]
    need(report["status"] == "pass", "latest full qualification attempt must pass")
    checked = []
    for row in report["before"]:
        owner = ROOT if row["repository"] == "accelerator" else DATASETS
        digest_row(read(relative(owner, row["path"])), row, "qualified current source differs: " + row["path"])
        checked.append(row)
    return dict(qualification_path=str(selected.relative_to(ROOT)),
                qualification_sha256=sha(read(selected / "commands.json")),
                qualification_binding=bind(selected / "commands.json"),
                accelerator_base=BASE, accelerator_execution_head=report["accelerator_execution_head"],
                selected_tests=report["collected_tests"], suites=8, new_cases=22,
                source_bindings=checked, source_binding_scope="listed_dependencies_not_complete_import_closure",
                baseline_control=dict(commands_binding=bind(baselines[0] / "commands.json"),
                    diagnostic=BASELINE_DIAGNOSTIC, expected_failures=1, elapsed_seconds=baseline["elapsed_seconds"],
                    unpatched_production_revision=BASE, promoted_to_pass=False),
                attempts=[dict(commands_binding=bind(path / "commands.json"), status=record["status"],
                               accelerator_execution_head=record["accelerator_execution_head"],
                               junit=record["junit"], selected_for_release=path == selected) for path, record in records])


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
    fixed, refs = live_git(selected["accelerator_execution_head"])
    need(sha(read(WORKFLOW)) == WORKFLOW_SHA and sha(read(VOCAB_CHECKER)) == VOCAB_CHECKER_SHA,
         "reviewed workflow/vocabulary utility differs")
    stable = dict(selected, fixed_git=fixed, preserved_prior_evidence=prior_evidence(),
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
    utility = load_stdlib_utility(UTILITY, UTILITY_SHA, "delivery_workflow_extractor")
    checker = load_stdlib_utility(LINK_CHECKER, LINK_CHECKER_SHA, "delivery_relative_link_checker")
    blocks = utility.extract_workflow_blocks()
    need(tuple(block["name"] for block in blocks) == EXPECTED_NAMES
         and len(blocks) == 4, "exact four reviewed workflow blocks required")
    before, before_refs = guard(checker, excluded)
    report = dict(schema="source-worker-delivery-documentation@1", status="running", started_at_utc=utc(),
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
            with tempfile.TemporaryDirectory(prefix="swd-doc-") as temporary:
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
