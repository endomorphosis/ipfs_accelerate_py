"""Publish bounded checker-refusal diagnostics and selected control evidence."""
from pathlib import Path
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET

B = Path(__file__).resolve().parent
P = B / "public-component"
W = B.parent.parent
A = W / ".worktrees/ir-release-accelerate-20261002"
D = W / ".worktrees/ir-pressure-attribution-datasets-20261004"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (P / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    outcome = read(B / "focused-01-exit.json")
    command = read(B / "focused-01-command.json")
    assert outcome["returncode"] == 0 and outcome["source_pins_unchanged"]
    assert outcome["counts"] == dict(passed=91, failed=0, errors=0, skipped=1, executions=92)
    for name, digest in command["source_pins"].items():
        prefix, relative = name.split("/", 1)
        assert sha({"source": A, "datasets": D}[prefix] / relative) == digest
    tree = ET.parse(B / "focused-01.xml")
    assert all(node.tag in {"testsuites", "testsuite", "testcase", "skipped"} for node in tree.iter())
    skips = []
    for case in tree.findall(".//testcase"):
        skipped = case.find("skipped")
        if skipped is not None:
            skips.append({"classname": case.get("classname"), "name": case.get("name"),
                          "reason": skipped.get("message")})
    assert len(skips) == 1
    assert skips[0]["name"] == "test_normal_initialization_with_real_checkpoint_produces_nomination_before_planning"
    P.mkdir(exist_ok=False)
    for name in ("focused-01-command.json", "focused-01-exit.json", "focused-01.xml",
                 "prior-native-diagnosis.json", "original-pins.json", "run_controls.py"):
        shutil.copyfile(B / name, P / name)
    write("observation.json", {
        "schema": "header-checker-failure-recovery-component@1",
        "scope": "A applicability refusal gates, wrapper propagation, driver metadata and prior native admission controls",
        "counts": outcome["counts"], "seconds": outcome["seconds"], "skipped_cases": skips,
        "selected_source_pins_unchanged": True,
        "selected_source_pins": command["source_pins"],
        "pin_scope": "Ten selected A/D owners and test files; not a complete imported-source inventory.",
        "actual_z3_controls": True, "checkpoint_inference_in_this_group": False,
        "source_fixture": "Authored Python header fixture; not hidden benchmark verifier input.",
        "resource_telemetry_scope": "Authored telemetry with real isolated native scheduler mechanics; not a host-pressure recovery measurement.",
        "proof_acceptance_changed": False, "applicability_replay_limit_seconds": 45,
        "planning_aggregate_limit_changed": False,
        "new_diagnostic_field": "failure_header_checker",
        "diagnostic_schemas": ["bounded-header-checker-failure@1", "header-model-check-refusal@1"],
        "exception_cause_walk_limit": 8, "query_status_summary_row_limit": 64,
        "old_trial_exact_checker_outcome_available": False,
        "old_trial_pressure_causality_claimed": False,
        "Docker_qualification_claimed": False, "benchmark_score": None,
        "token_score": None, "benchmark_advantage_claimed": False,
        "provider_calls": 0, "production_scheduler_ledger_mutated": False,
    })
    original_records = {}
    for path in sorted((B / "original").rglob("*")):
        if path.is_file():
            original_records[str(path.relative_to(B / "original"))] = {
                "sha256": sha(path), "bytes": path.stat().st_size, "raw_body_exported": False}
    write("original-source-records.json", original_records)
    related = B.parent / "header-checker-admission-fix-20261004"
    write("related-datasets-controls.json", {
        "scope": "Independent D controls; these results are not counted in the A91-pass total.",
        "records": {str(path): {"sha256": sha(path), "bytes": path.stat().st_size,
                                "raw_body_exported": False}
                    for path in (related / "native-01-command.json", related / "native-01-exit.json",
                                 related / "native-01.xml", related / "contention-01-command.json",
                                 related / "contention-01-exit.json", related / "contention-01.xml")},
    })
    shutil.copyfile(Path(__file__), P / "build_component.py")
    (P / "README.md").write_text("""# Header checker failure propagation

The selected A controls passed91 cases with zero failures/errors and one explicit optional checkpoint/GTE skip, in88.986 seconds. All ten selected A/D source/test pins remained unchanged. Real Z3 and native captured-source/scheduler mechanisms are exercised against authored fixtures. The skipped normal-initialization case requires explicit checkpoint and embedding paths; this host group does not claim checkpoint inference coverage. Mandatory Docker qualification belongs to the next separately recorded exact runtime generation.

The earlier full trial ended in Doctor before START with `invalid symbolic intent planning: actual complete local-model checking required`. Its original report did not retain the exact failed checker condition or query verdict. Neither that condition nor a host-pressure cause can be reconstructed from the later19.69-percent pressure sample. The retained diagnosis distinguishes this evidence limit from confirmed source mechanisms.

The D checker repair separates bounded resource admission from its shared five-second version/query execution window, preserves original leased timeout/cancellation causes, and prevents an unleased version fallback on failure. The existing enclosing deadline still limits both phases. Independent D controls, including actual reservation contention longer than five seconds followed by real bounded Z3 and exhaustion of the original aggregate deadline, are referenced by digest and are not added to this package's91 passing cases.

A now reports which existing checker gate refused: overall result status, execution profile, solver identity, call completeness, counterexample presence, or expectation agreement. Only bounded verdict counts, booleans, and closed reason codes are retained. LocalPlanningError carries the bounded diagnostic while preserving explicit exception causes. The driver exports `failure_header_checker` and recovers the actual native admission observation through at most eight explicit causes. Later host observations remain separate. At most64 query rows contribute to a status-count summary; model, solver output and source bodies are excluded.

Every original proof/counterexample gate remains enforced. Unknown or mismatched results do not authorize a fact or repair. The45-second applicability limit, planner aggregate budget, solver execution limits, signed source checks and publication/completion authority boundaries are unchanged by A. Tests cover actual Z3 replay, altered successful-check metadata refusing the plan, unknown results, bounded/closed diagnostics, native admission causes through wrappers, implicit/cyclic/deep exception rejection, and metadata privacy.

This export contains command/exit/XML records, digests, a bounded prior-failure diagnosis and artifact-only reproduction recipes. It excludes production source bodies, failed solver/model bodies, prompts, credentials, seal/key stores, and scheduler databases. It is neither a new Docker result nor a Terminal Bench score, token score, complete upstream-suite result, or performance advantage claim.
""")
    files = [dict(path=str(path.relative_to(P)), bytes=path.stat().st_size, sha256=sha(path))
             for path in sorted(P.rglob("*")) if path.is_file()]
    write("manifest.json", {"schema": "header-checker-failure-recovery-public-evidence@1", "files": files})
    print(json.dumps({"package": str(P), "members": len(files),
                      "member_bytes": sum(row["bytes"] for row in files),
                      "manifest_sha256": sha(P / "manifest.json")}))


if __name__ == "__main__":
    main()
