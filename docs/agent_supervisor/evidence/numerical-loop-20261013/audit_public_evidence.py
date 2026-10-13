"""Replay retained public numerical reports and audit recorded real-Lake runs."""
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path

from ipfs_datasets_py.logic.software_verification import numerical_logic, numerical_logic_v2, numerical_logic_v3
from ipfs_datasets_py.logic.formalization.autoencoder import structured_source_384, native_family_lake_v4
from ipfs_datasets_py.logic.syntax_core.ast import TypedExpression

R = Path(__file__).resolve().parent
Q = R / "public-02"
PUBLIC = Path("/home/barberb/lift_coding/.benchmarks/terminal-bench-2/largest-eigenval/environment/src")
sha = lambda data: hashlib.sha256(data).hexdigest()
raw = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
read = lambda path: json.loads(path.read_bytes())
qualification, baseline = read(Q / "qualification.json"), read(R / "baseline.json")
checks = []


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


check("qualification_complete", qualification["complete"] is True)
check("baseline_file_identity", qualification["baseline_sha256"] == sha((R / "baseline.json").read_bytes()))
for name, digest in qualification["producer_sha256"].items():
    module = importlib.import_module(name)
    check("current_producer:" + name, sha(Path(module.__file__).read_bytes()) == digest)
for name, digest in baseline["producer_sha256"].items():
    check("historical_producer_unchanged:" + name, qualification["producer_sha256"][name] == digest)
check("structured_checkpoint_implementation_unchanged", structured_source_384._implementation() == baseline["structured_checkpoint_implementation"])
for entry in baseline["files"]:
    data = (PUBLIC / entry["path"]).read_bytes()
    check("baseline_source:" + entry["profile"] + ":" + entry["path"], sha(data) == entry["source_sha256"])
    owner = numerical_logic if entry["profile"] == numerical_logic.PROFILE else numerical_logic_v2
    report = owner.formalize_numerical_source(data.decode(), path=entry["path"])
    check("historical_exact_report:" + entry["profile"] + ":" + entry["path"], sha(raw(report)) == entry["canonical_report_sha256"])

files, retained_reports = [], {}
for entry in qualification["files"]:
    name = entry["path"]
    check("declared_public_path:" + name, name in {"eigen.py", "eval.py"})
    source = (PUBLIC / name).read_bytes()
    report = read(Q / (name + ".report.json"))
    check("public_source_identity:" + name, sha(source) == entry["source_sha256"] == report["source_sha256"])
    check("native_logic_exact_source_replay:" + name,
          numerical_logic_v3.validate_numerical_logic_report(report, source.decode(), path=name) == report)
    check("canonical_report_identity:" + name, sha(raw(report)) == entry["canonical_report_sha256"])
    check("internal_report_identity:" + name, sha(raw({k:v for k,v in report.items() if k != "report_sha256"})) == report["report_sha256"])
    check("pending_count:" + name, len(report["pending_obligations"]) == entry["pending_obligations"])
    check("pending_kind_inventory:" + name, dict(Counter(r["kind"] for r in report["pending_obligations"])) == entry["obligation_kinds"])
    check("function_inventory:" + name, report["summary"]["per_function"] == entry["per_function"])
    check("complete_functions:" + name, all(r["structurally_complete"] for r in entry["per_function"]))
    check("no_source_proof:" + name, report["coverage"]["source_obligations_discharged"] == 0 and not report["source_semantics_verified"])
    obligations = {row["obligation_id"]: row for row in report["pending_obligations"]}
    check("all_pending_without_authority:" + name, all(r["status"] == "pending" and not r["proof_authority"]
        and not r["assumptions_asserted"] and all(r[k] is False for k in numerical_logic_v3._CONTROL_FALSE) for r in obligations.values()))
    for projection in report["projections"]["first_order"]:
        typed = TypedExpression.from_dict(projection["expression"]).to_dict()
        row = obligations[projection["obligation_id"]]
        check("fol_nested_control_binding:" + name + ":" + row["obligation_id"], all(
            typed["metadata"][k] == v == projection[k] for k,v in numerical_logic_v3._control_binding(row).items()))
    for projection in report["projections"]["smt"] + report["projections"]["lean"]["bindings"]:
        row = obligations[projection["obligation_id"]]
        check("backend_control_binding:" + name + ":" + row["obligation_id"], all(
            projection[k] == v for k,v in numerical_logic_v3._control_binding(row).items()))
    native = report["projections"]["program"]
    check("complete_native_control_sidecars:" + name, all(native[k] == report["structural"][k]
        for k in ("guarded_regions", "control_regions", "command_contexts")))
    lean = report["projections"]["lean"]
    check("retained_lean_source:" + name, (Q / (name + ".lean")).read_bytes() == lean["source"].encode()
        and sha(lean["source"].encode()) == entry["lean_source_sha256"] == lean["source_sha256"])
    altered = deepcopy(report)
    altered["pending_obligations"][0]["evaluation_contexts"] = []
    altered["report_sha256"] = sha(raw({k:v for k,v in altered.items() if k != "report_sha256"}))
    refused = False
    try:
        numerical_logic_v3.validate_numerical_logic_report(altered, source.decode(), path=name)
    except ValueError:
        refused = True
    check("rehashed_context_tamper_refused:" + name, refused)
    check("public_source_unchanged_after_audit:" + name, sha((PUBLIC / name).read_bytes()) == entry["source_sha256"])
    retained_reports[name] = report
    files.append(dict(path=name, source_sha256=entry["source_sha256"], structural_status=report["structural"]["status"],
        module_diagnostic_codes=sorted({d["code"] for d in report["structural"]["diagnostics"] if not d["function"]}),
        complete_functions=sum(r["structurally_complete"] for r in entry["per_function"]),
        pending_obligations=len(obligations), compound_contracts=len(report["compound_contracts"]),
        assertion_requirements=len(report["assertion_requirements"]),
        control_regions=len(report["structural"]["control_regions"]),
        proof_phase_obligations=sum("proof_scope" in r for r in obligations.values()),
        first_order_projections=len(report["projections"]["first_order"]), smt_projections=len(report["projections"]["smt"])))

check("module_partial_explicit", retained_reports["eval.py"]["structural"]["status"] == "partial"
    and {"python.module.Assign", "python.module.If"} <= set(files[1]["module_diagnostic_codes"]))
runner_source = (R / "qualify_public_source.py").read_text()
authored = next(ast.literal_eval(n.value) for n in ast.parse(runner_source).body
               if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "AUTHORED" for t in n.targets))
authored_report = read(Q / "authored-control.report.json")
check("authored_control_exact_replay", numerical_logic_v3.validate_numerical_logic_report(
    authored_report, authored, path="authored_control.py") == authored_report)
lake_sources = {name:report["projections"]["lean"]["source"] for name,report in retained_reports.items()}
lake_sources["authored-control"] = authored_report["projections"]["lean"]["source"]
lake_sources["false-source-truth"] = lake_sources["authored-control"] + "\ntheorem unsupported_source_truth : False := by trivial\n"
builds = []
for name, lean_source in lake_sources.items():
    log = read(Q / (name + ".lake.json"))
    positive = name != "false-source-truth"
    check("lake_execution_record:" + name, log["backend_executed"] and not log["timed_out"] and not log["output_truncated"] and not log["workspace_limit_exceeded"])
    check("lake_command:" + name, log["command"][1:] == ["build", "NumericalObligations"])
    check("lake_binary_identity:" + name, sha(Path(log["command"][0]).read_bytes()) == log["executable_sha256"])
    check("lake_exact_input:" + name, sha(lean_source.encode()) == log["input_files_sha256"]["NumericalObligations.lean"])
    check("lake_status:" + name, (log["status"] == "passed" and log["returncode"] == 0) if positive else
          (log["status"] == "failed" and log["returncode"] != 0 and "⊢ False" in log["stdout"]))
    builds.append(dict(name=name, status=log["status"], returncode=log["returncode"],
        toolchain=log["toolchain"], executable_sha256=log["executable_sha256"],
        input_source_sha256=log["input_files_sha256"]["NumericalObligations.lean"]))
check("qualification_excludes_performance_claims", qualification["native_provider_tokens"] is None
    and qualification["official_terminal_bench_score"] is None and qualification["provider_calls"] == 0
    and qualification["model_inference_calls"] == 0 and qualification["source_obligations_discharged"] == 0)
check("launcher_success", read(R / "public-run-02" / "exit.json")["returncode"] == 0)
artifacts = {str(path.relative_to(R)):sha(path.read_bytes()) for path in sorted(Q.iterdir()) if path.is_file()}
for path in (R / "baseline.json", R / "qualify_public_source.py", Path(native_family_lake_v4.__file__), Path(__file__)):
    artifacts[str(path)] = sha(path.read_bytes())
result = dict(schema="numerical-loop-public-evidence-audit@1", passed=True, checks=len(checks),
    check_names=checks, files=files, builds=builds, artifacts_sha256=artifacts,
    producer_sha256=qualification["producer_sha256"], historical_reports_replayed=4,
    current_public_reports_replayed=2, authored_control_reports_replayed=1,
    complete_public_functions=sum(f["complete_functions"] for f in files),
    public_pending_obligations=sum(f["pending_obligations"] for f in files),
    public_compound_contracts=sum(f["compound_contracts"] for f in files),
    public_assertion_requirements=sum(f["assertion_requirements"] for f in files),
    lake_build_evidence=dict(recorded_runs=4, positive_passes=3, false_proposition_rejections=1,
        independently_rerun_by_auditor=False),
    target_source_executed=False, hidden_verifier_read=False, audit_provider_calls=0,
    audit_solver_calls=0, source_obligations_discharged=0, proof_authority=False,
    limitations=["All four public functions are structurally complete; eval.py module-level Assign/If remain partial.",
        "Lake accepted generated definitions and conditional mathematical lemmas; it did not prove Python/NumPy source behavior, loop invariants, termination or the pending obligations.",
        "Recorded real-Lake logs and exact input/binary hashes were audited; this audit did not launch additional Lake processes.",
        "This is deterministic frontend/projection qualification, not learned autoencoder inference or a live supervisor/Terminal-Bench performance run.",
        "No token, billing, benchmark reward or causal performance advantage is established."])
(R / "public-evidence-audit.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
print(json.dumps(dict(passed=True, checks=len(checks), public_functions=result["complete_public_functions"],
                     public_obligations=result["public_pending_obligations"])))
