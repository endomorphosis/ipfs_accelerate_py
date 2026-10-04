"""Publish bounded controls for the native completion-binding contract repair."""
from pathlib import Path
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET

B = Path(__file__).resolve().parent
P = B / "public-component"
A = B.parent.parent / ".worktrees/ir-release-accelerate-20261002"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (P / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    final = read(B / "regression-02-exit.json")
    assert final["returncode"] == 0 and final["source_pins_unchanged"]
    assert final["counts"] == dict(passed=58, failed=0, errors=0, skipped=0, executions=58)
    command = read(B / "regression-02-command.json")
    assert all(sha(A / name) == digest for name, digest in command["source_pins"].items())
    P.mkdir(exist_ok=False)
    for label in ("before-01", "native-01", "native-02", "native-03", "regression-01", "regression-02"):
        for suffix in ("-command.json", "-exit.json"):
            shutil.copyfile(B / (label + suffix), P / (label + suffix))
    for label in ("native-03", "regression-02"):
        tree = ET.parse(B / (label + ".xml"))
        assert all(node.tag in {"testsuites", "testsuite", "testcase"} for node in tree.iter())
        shutil.copyfile(B / (label + ".xml"), P / (label + ".xml"))
    shutil.copyfile(B / "native-02-diagnostic.json", P / "native-02-diagnostic.json")
    failures = []
    reasons = {
        "before-01": "Actual native implementation constructor reproduced unexpected retirable keyword before START.",
        "native-01": "Native START passed; old fixture's absolute Python validation launcher failed sealed pre-merge validation and task remained in_progress at90 seconds.",
        "native-02": "Retained native rerun established the same declared-validation fixture cause; production policy was preserved.",
        "regression-01": "One test incorrectly reopened a coordinator intentionally retained after install-then-raise; its process lock correctly refused the second connection. Test was corrected to inspect and tear down the retained coordinator.",
    }
    for label, reason in reasons.items():
        path = B / (label + ".xml")
        cases = []
        for case in ET.parse(path).findall(".//testcase"):
            for node in case.findall("failure") + case.findall("error"):
                cases.append({"classname": case.get("classname", ""), "name": case.get("name", ""),
                              "kind": node.tag, "failure_body_sha256": hashlib.sha256((node.text or "").encode()).hexdigest()})
        failures.append({"label": label, "scope": reason, "qualifying": False,
                         "xml_sha256": sha(path), "xml_bytes": path.stat().st_size,
                         "raw_failure_bodies_exported": False, "failures": cases})
    write("prior-failures.json", failures)
    originals = {}
    for path in sorted((B / "original").rglob("*")):
        if path.is_file():
            originals[str(path.relative_to(B / "original"))] = dict(sha256=sha(path), bytes=path.stat().st_size,
                                                                     raw_body_exported=False)
    write("original-source-records.json", originals)
    write("observation.json", {
        "schema": "native-completion-binding-component-evidence@1",
        "qualified_scope": "58 affected host controls including actual implementation and owner-signed completion",
        "final_label": "regression-02", "counts": final["counts"], "seconds": final["seconds"],
        "selected_source_pins_unchanged": True, "selected_final_source_pins": command["source_pins"],
        "pin_scope": "Three production owners and two changed test files; this is not a complete imported-source inventory.",
        "native_implementation": {
            "independently_declared_fixture": True, "implementation_authored_by_test": True,
            "provider_calls": 0, "signed_runtime_constructor": True, "native_START": True,
            "task_completed": True, "immutable_public_check_preserved": True,
            "native_STOP": True, "remaining_native_processes": 0,
            "included_in_final_58": True,
        },
        "ordinary_failure_controls": [
            "Pre-bind refusal: actual empty tree, bootstrap shutdown, persisted run lease released, task ready.",
            "Install-then-raise: bootstrap shutdown, callback and run lease retained, no successful cleanup claim.",
            "Pre-existing duplicate callback: existing callback preserved, only failed new runtime disposed.",
        ],
        "ordinary_binding_retirable": False,
        "finite_and_inventory_paired_cleanup_modified": False,
        "intermediate_native_03": "Actual authored repair passed before final cleanup-edge changes; final joined run re-exercises it on current owners.",
        "benchmark_score": None, "completed_token_score": None, "Docker_qualification_claimed": False,
        "service_changes": False, "production_scheduler_ledger_mutated": False,
    })
    shutil.copyfile(B / "run_controls.py", P / "run_controls.py")
    shutil.copyfile(Path(__file__), P / "build_component.py")
    (P / "README.md").write_text("""# Native completion binding repair

The final `regression-02` run passed all58 affected host controls with zero failures, errors, or skips. Its selected production/test pins stayed unchanged. It exercises an actual admitted `implement=True` runtime: independently declare a small failing repository, bind the owner validation service, START, apply a fixed authored answer.py repair without a model, obtain native task completion while preserving the public check, and STOP with no remaining native process.

The original real constructor failed because the ordinary bridge passed `retirable=False` to a gateway that accepted only its handler argument. Ordinary binding now preserves that original one-argument contract. Optional retirement requires supported bind/unbind APIs before mutation; the gateway implements opaque binding identity, transaction custody, active callback checks, and exact retirement. Existing finite/inventory paired cleanup remains unchanged.

Ordinary constructor refusal now disposes its bootstrap thread/socket and releases its exact run lease only after an actual empty-process observation and confirmation that completion custody did not change. A work interruption after callback installation retains callback/run-lease custody and reports unproven cleanup, while still stopping bootstrap transport. It does not grant ordinary callbacks retirement authority. Separate native cases prove refusal before binding, interruption after binding, and refusal with a pre-existing callback. No artificial STOP receipt is created for an unlaunched constructor.

Earlier observations are retained separately. `before-01` reproduces the original keyword mismatch. `native-01` and `native-02` exposed a fixture declaration using an absolute Python validation launcher, which the native pre-merge policy rejects; the new fixture now declares `python3` before independent admission is signed. `native-03` completed the authored repair before the additional interrupted-binding cleanup fix. `regression-01` had57 passes and one test-side coordinator-reopen error; `regression-02` corrects that inspection and re-exercises the final production owners. Counts from earlier runs are not added to the final58.

Command/exit records, successful XML, explicit selected source hashes, original source hashes, and bounded failure metadata are exported. Raw failed traces, production source bodies, runtime credentials, seal/key stores, private scheduler/coordinator databases, and task bodies are excluded. The selected source-pin set covers three production owners and two changed test files; it is not a complete import inventory. This is host integration evidence, not a new Docker qualification, Terminal Bench reward, token score, performance advantage, or complete upstream-suite result. No model or service was invoked or changed by these controls.
""")
    files = [dict(path=str(path.relative_to(P)), bytes=path.stat().st_size, sha256=sha(path))
             for path in sorted(P.rglob("*")) if path.is_file()]
    write("manifest.json", {"schema": "native-completion-binding-public-evidence@1", "files": files})
    print(json.dumps({"members": len(files), "member_bytes": sum(row["bytes"] for row in files),
                      "manifest_sha256": sha(P / "manifest.json"), "package": str(P)}))


if __name__ == "__main__":
    main()
