"""Run the four frozen offline documentation gates once after the docs freeze.

This stdlib runner inspects retained qualification; it never invokes pytest,
models, providers, Docker or an official task verifier. Its own file is bound
separately from the qualification's 188 listed source dependencies.
"""
from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
BASE = "5ad92f61bb643d4138749d3df16663a2af1c79c1"
WORKFLOW = ROOT / ".github/workflows/documentation-gates.yml"
WORKFLOW_SHA = "0a46fa5b8905600cb9f74adaaec8b8aeb5f89878612a43fb7a847e586bde3763"
CHECKER = ROOT / "scripts/docs/check_current_docs_links.py"
CHECKER_SHA = "596a0038b18d592f2e605643890afc85e457558f7b887932a474557d7586819e"
EXPECTED_NAMES = (
    "Supervisor primary-doc vocabulary", "Maintained-surface relative links",
    "Packaging version pin alignment", "Closeout navigation markers",
)
DOCUMENTS = (
    ROOT / "docs/agent_supervisor/repository_source_scope.md",
    ROOT / "docs/agent_supervisor/repository_source_inventory.md",
    ROOT / "docs/agent_supervisor/reversible_codebase_views_plan.md",
    PHASE / "README.md",
)
PRIOR_PHASE = ROOT / "docs/agent_supervisor/evidence/benchmark-shared-source-policy-20261013"
RECEIPT = PHASE / "documentation-checks.json"
LOG = PHASE / "documentation-workflow.log"


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def binding(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(ROOT)), bytes=len(raw), sha256=sha(raw))


def tree_bindings(directory):
    return [binding(path) for path in sorted(directory.rglob("*")) if path.is_file()]


def extract_workflow_blocks():
    raw = WORKFLOW.read_bytes()
    need(sha(raw) == WORKFLOW_SHA, "documentation workflow differs from reviewed bytes")
    lines, blocks, index, name = raw.decode("utf-8").splitlines(), [], 0, None
    while index < len(lines):
        line = lines[index]
        if line.startswith("      - name: "):
            name = line.removeprefix("      - name: ")
        match = re.fullmatch(r"        run: (.+)", line)
        if match:
            value = match.group(1)
            if value == "|":
                body = []
                index += 1
                while index < len(lines) and (not lines[index] or lines[index].startswith("          ")):
                    body.append(lines[index][10:] if lines[index] else "")
                    index += 1
                script, form = "\n".join(body).rstrip() + "\n", "indented_block"
            else:
                need(value and not value.startswith(("|", ">")), "unsupported workflow run scalar")
                script, form = value + "\n", "single_command"
                index += 1
            blocks.append(dict(name=name, form=form, script=script,
                script_sha256=sha(script.encode("utf-8"))))
            continue
        need(not line.lstrip().startswith("run:"), "unrecognized workflow run indentation")
        index += 1
    need(tuple(row["name"] for row in blocks) == EXPECTED_NAMES,
         "exact four reviewed workflow run blocks required")
    need([row["form"] for row in blocks] == ["single_command", "single_command", "indented_block", "indented_block"],
         "expected two single commands and two indented blocks")
    return blocks


def qualified_source_bindings(record):
    rows = record["before"]
    need(len(rows) == 188 and rows == record["after"], "retained 188-source custody disagrees")
    need(len({(row["repository"], row["path"]) for row in rows}) == 188,
         "duplicate retained source dependency")
    datasets = Path(record["datasets_execution_path"])
    revision = record["datasets_execution_revision"]
    need(git(datasets, "rev-parse", "HEAD").decode().strip() == revision,
         "datasets execution checkout revision changed")
    checked = []
    for row in rows:
        repository = row["repository"]
        root = ROOT if repository == "accelerator" else datasets
        need(repository in {"accelerator", "datasets_execution_checkout"}, "unknown source repository")
        path = root / row["path"]
        raw = path.read_bytes()
        need(len(raw) == row["bytes"] and sha(raw) == row["sha256"], "qualified source changed: " + row["path"])
        actual = dict(repository=repository, path=row["path"], bytes=len(raw), sha256=sha(raw))
        if repository != "accelerator":
            original = git(datasets, "show", revision + ":" + row["path"])
            need(original == raw, "datasets source differs from its immutable Git object")
            actual.update(immutable_git_revision=revision, immutable_git_object_sha256=sha(original),
                matches_immutable_git_object=True)
        checked.append(actual)
    need(sum(row["repository"] == "accelerator" for row in checked) == 137,
         "expected 137 accelerator dependencies")
    need(sum(row["repository"] != "accelerator" for row in checked) == 51,
         "expected 51 immutable datasets objects")
    return checked


def retained_qualification():
    attempt = PHASE / "qualification-01"
    record = json.loads((attempt / "commands.json").read_bytes())
    need(record["status"] == "pass" and record["pytest_exit_code"] == 0
         and record["pytest_invocations"] == 1 and record["separate_collection"] is False,
         "final qualification has not passed its single execution")
    need(record["source_before_after_equal"] is True and record["collection_execution_agrees"] is True,
         "final qualification custody or collection disagrees")
    need(record["collected_tests"] == record["distinct_junit_cases"] == 219,
         "final qualification is not the reviewed 219 distinct cases")
    document = ET.parse(attempt / "junit.xml").getroot()
    suites = [document] if document.tag == "testsuite" else document.findall("testsuite")
    counts = {key: sum(int(suite.get(key, 0)) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    cases = [(case.get("classname"), case.get("name")) for case in document.iter("testcase")]
    need(counts == {"tests": 219, "failures": 0, "errors": 0, "skipped": 0}
         and len(cases) == len(set(cases)) == 219 and record["junit"] == counts,
         "retained JUnit does not prove 219 unique passing identities")
    text = (attempt / "tests.stdout.log").read_text(errors="replace")
    observations = list(map(int, re.findall(r"^(?:collecting \.\.\. )?collected ([0-9]+) items?\s*$", text, re.M)))
    need(observations == record["collection_count_observations"] == [219],
         "retained collection observations are not unique")
    need(len(re.findall(r"^=+[^\n]*\b219 passed\b[^\n]*=+\s*$", text, re.M)) == 1,
         "retained successful final pytest summary is not unique")
    for name in ("junit.xml", "tests.stdout.log", "tests.stderr.log"):
        raw = (attempt / name).read_bytes()
        need(dict(bytes=len(raw), sha256=sha(raw)) == record[name], "retained test output binding differs")
    return record, counts


def preserved_evidence():
    paths = [*sorted(PRIOR_PHASE.rglob("*")), *sorted((PHASE / "qualification-01").rglob("*")),
             PHASE / "parallel-work-survey.json", PHASE / "review-observations.json"]
    return [binding(path) for path in paths if path.is_file()]


def prior_evidence_git_changes():
    names = git(ROOT, "diff", "--name-only", BASE, "--", "docs/agent_supervisor/evidence").decode().splitlines()
    current = str(PHASE.relative_to(ROOT)) + "/"
    return [name for name in names if not name.startswith(current)]


def manual_links():
    need(sha(CHECKER.read_bytes()) == CHECKER_SHA, "reviewed documentation link checker changed")
    spec = importlib.util.spec_from_file_location("retained_documentation_link_checker", CHECKER)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    need(checker.ROOT == ROOT, "checker root differs from execution checkout")
    results = []
    for path in DOCUMENTS:
        text = checker.strip_markup_noise(path.read_text(encoding="utf-8"))
        links = [checker.split_target(match.group("target")) for match in checker.LINK_RE.finditer(text)]
        relative = [(target, fragment) for target, fragment in links if not checker.is_external(target, fragment)]
        errors = checker.check_file(path, check_anchors=True)
        results.append(dict(**binding(path), status="pass" if not errors else "fail", errors=errors,
            checker_api="check_file(path, check_anchors=True)", check_anchors=True,
            markdown_link_count=len(links), external_link_count=len(links)-len(relative),
            relative_link_count=len(relative), relative_anchor_count=sum(bool(fragment) for _, fragment in relative)))
    return results


def main():
    need(len(sys.argv) == 1, "Usage: documentation_runner.py")
    need(not RECEIPT.exists() and not LOG.exists(), "documentation outputs already exist; refusing overwrite")
    blocks = extract_workflow_blocks()
    qualification, counts = retained_qualification()
    source_before = qualified_source_bindings(qualification)
    evidence_before = preserved_evidence()
    documents_before = [binding(path) for path in DOCUMENTS]
    need(not prior_evidence_git_changes(), "prior phase evidence differs from the published base")
    guard_before = [binding(path) for path in (Path(__file__), WORKFLOW, CHECKER, PHASE / "qualification_runner.py")]
    environment = os.environ.copy()
    removed = ("BASH_ENV", "ENV", "SHELLOPTS", "CDPATH", "PYTHONPATH", "PYTHONSTARTUP")
    for key in removed:
        environment.pop(key, None)
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
        PATH=str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", ""))
    report = dict(schema="repository-source-scope-documentation@1", started_at_utc=utc(),
        cwd=str(ROOT), git_head=git(ROOT, "rev-parse", "HEAD").decode().strip(),
        python_version=sys.version, interpreter=sys.executable,
        workflow_path=str(WORKFLOW.relative_to(ROOT)), workflow_sha256=WORKFLOW_SHA,
        manual_check_api_source=binding(CHECKER), exact_workflow_run_blocks=blocks,
        document_bindings_before=documents_before, qualified_source_binding_checks_before=source_before,
        qualified_source_bindings_count=188, qualified_accelerator_source_bindings_count=137,
        qualified_datasets_source_bindings_count=51, preserved_evidence_before=evidence_before,
        read_guard_bindings_before=guard_before, retained_qualification_status=qualification["status"],
        retained_collected_tests=219, retained_distinct_junit_cases=219, retained_junit_counts=counts,
        retained_pytest_invocations=1, retained_separate_collection=False,
        runner_outside_188_listed_source_inventory=True,
        source_binding_scope="listed_dependencies_not_complete_import_closure_or_toolchain_attestation",
        qualification_executed_by_documentation_checker=False, additional_pytest_run=False,
        provider_calls=0, model_calls=0, training_run=False, solver_run=False, additional_docker_run=False,
        official_task_verifier_run=False, product_edits=False, token_savings_measured=False,
        hosted_ci_status_verified_here=False, proof_authority=False, completion_authority=False,
        inherited_environment_names_removed=list(removed), python_command=shutil.which("python", path=environment["PATH"]),
        steps=[], status="running")
    started = time.monotonic()
    # Exclusive handles create link targets without a second write or overwrite.
    # JSON is written once after observations; the log is streamed to one handle.
    with RECEIPT.open("x", encoding="utf-8") as receipt, LOG.open("x", encoding="utf-8") as log:
        try:
            for block in blocks:
                argv = ["bash", "--noprofile", "--norc", "-euo", "pipefail", "-c", block["script"]]
                result = subprocess.run(argv, cwd=ROOT, env=environment, capture_output=True, text=True,
                                        timeout=120, check=False)
                step = dict(name=block["name"], argv=argv, exit_code=result.returncode,
                    stdout=result.stdout, stderr=result.stderr,
                    stdout_sha256=sha(result.stdout.encode()), stderr_sha256=sha(result.stderr.encode()))
                report["steps"].append(step)
                log.write("== " + block["name"] + " ==\n" + result.stdout + result.stderr
                          + "exit_code=" + str(result.returncode) + "\n")
                log.flush()
                need(result.returncode == 0, "documentation workflow run block failed: " + block["name"])
            report["manual_relative_link_checks"] = manual_links()
            need(all(row["status"] == "pass" for row in report["manual_relative_link_checks"]),
                 "manual relative links or anchors failed")
            report["manual_relative_link_count"] = sum(row["relative_link_count"] for row in report["manual_relative_link_checks"])
            report["manual_relative_anchor_count"] = sum(row["relative_anchor_count"] for row in report["manual_relative_link_checks"])
            final_record, final_counts = retained_qualification()
            report["qualified_source_binding_checks_after"] = qualified_source_bindings(final_record)
            report["preserved_evidence_after"] = preserved_evidence()
            report["document_bindings_after"] = [binding(path) for path in DOCUMENTS]
            report["read_guard_bindings_after"] = [binding(path) for path in (Path(__file__), WORKFLOW, CHECKER, PHASE / "qualification_runner.py")]
            report["all_188_qualified_sources_match_before_and_after_documentation"] = source_before == report["qualified_source_binding_checks_after"]
            report["preserved_evidence_unchanged"] = evidence_before == report["preserved_evidence_after"]
            report["documents_unchanged_during_checks"] = documents_before == report["document_bindings_after"]
            report["read_guards_unchanged"] = guard_before == report["read_guard_bindings_after"]
            report["prior_evidence_git_changes"] = prior_evidence_git_changes()
            need(qualification == final_record and counts == final_counts
                 and report["all_188_qualified_sources_match_before_and_after_documentation"]
                 and report["preserved_evidence_unchanged"] and report["documents_unchanged_during_checks"]
                 and report["read_guards_unchanged"] and not report["prior_evidence_git_changes"],
                 "documentation read guards changed")
            report["status"] = "pass"
        except Exception as error:
            report.update(status="fail", error=dict(type=type(error).__name__, message=str(error)))
            log.write("FAIL: " + type(error).__name__ + ": " + str(error) + "\n")
        log.flush()
        report.update(completed_at_utc=utc(), elapsed_seconds=round(time.monotonic()-started, 3),
            log_path=str(LOG.relative_to(ROOT)), log_bytes=LOG.stat().st_size,
            log_sha256=sha(LOG.read_bytes()))
        json.dump(report, receipt, indent=2, sort_keys=True)
        receipt.write("\n")
    print(json.dumps(dict(status=report["status"], workflow_blocks=len(report["steps"]),
        relative_links=report.get("manual_relative_link_count"), anchors=report.get("manual_relative_anchor_count"))))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
