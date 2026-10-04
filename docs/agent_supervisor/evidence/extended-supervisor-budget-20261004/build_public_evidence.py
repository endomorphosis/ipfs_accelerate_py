"""Export bounded qualification metadata; leave the full trial pending and unsealed."""
from pathlib import Path
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET

B = Path(__file__).resolve().parent
P = B / "public-evidence"
M = B.parent / "local-benchmark-memory-profile-20261004"
EXPORTED = {}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path, exported=False):
    return {"sha256": sha(path), "bytes": path.stat().st_size,
            "raw_body_exported": exported}


def write(name, value):
    path = P / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def copy(path, name=None):
    target = P / (name or path.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    EXPORTED[str(path)] = record(path, True)


def main():
    P.mkdir(exist_ok=False)
    audit = read(B / "build-audit.json")
    q = read(B / "qualification-result.json")
    native = read(B / "qualification-01/source384-result.json")
    end = read(B / "qualification-exit.json")
    controller = read(B / "qualification-controller-exit.json")
    assert end["returncode"] == controller["returncode"] == 0
    assert all(end[k] is True for k in (
        "source_pins_unchanged", "public_task_inputs_unchanged", "cleanup_verified"))
    assert q["qualified"] and native["qualified"]
    assert q["deployment"]["archive_sha256"] == audit["archive_sha256"]
    assert native["resource_profile"] == "source384-5cpu-16gib-extended@1"
    frozen = read(B / "frozen-production-pins.json")
    assert read(B / "qualification-before-pins.json") == frozen
    assert read(B / "qualification-after-pins.json") == frozen
    for name in (
        "build-command.json", "build-exit.json", "build-audit.json",
        "input-review.json", "solver-profile.json", "qualification-command.json",
        "qualification-controller-command.json", "qualification-controller-exit.json",
        "qualification-exit.json", "qualification-public-task-pins.json",
        "qualification-containers-before.json", "qualification-containers-after.json",
        "run-admission.json", "selected-test-summary.json",
    ):
        copy(B / name)
    for name in ("resources.json", "admission-estimate.json"):
        copy(B / "qualification-01" / name, "qualification/" + name)
    copy(B / "bundle/setup-cache-selection.json", "qualification/setup-cache-selection.json")
    keys = (
        "schema", "qualified", "provider_calls", "official_verifier_executed",
        "benchmark_result", "source_qualified_proof_claimed", "training_steps",
        "download_calls", "prepare_seconds", "initial_context_seconds",
        "warm_observation_seconds", "source384_seconds", "context_seconds",
        "context_nonoverlapping_seconds", "seconds", "checkpoint_sha256", "config_sha256",
        "inference_sha256", "intent_requirement_contract_cid", "header_consumer_sha256",
        "source_head", "coverage", "source384_resource_profile", "native_worker_executed",
        "inference_executed", "neural_inference_replayed", "producer", "signed_source_hashes",
        "resource_profile", "execution_budget",
    )
    write("qualification/native-observation.json", {k: native[k] for k in keys if k in native})
    write("qualification/deployment-observation.json", {k: q["deployment"][k] for k in (
        "schema", "qualified", "archive_sha256", "task_source_preserved", "steps",
        "seconds", "provider_calls", "official_verifier_executed", "benchmark_result",
        "credential_contents_recorded")})
    write("qualification/setup-cache-observation.json", {k: q["setup_cache"][k] for k in (
        "completed", "policy", "selection", "admission_authority", "advice_is_best_effort",
        "credential_contents_recorded", "global_drop_caches", "freed_bytes_claimed")})
    write("source-pin-observation.json", {
        "schema": "complete-inventory-source-pin-observation@1",
        "repository_members": len(frozen),
        "namespaces": {ns: sum(p.startswith(ns + "/") for p in frozen)
                       for ns in ("source", "datasets", "kit")},
        "source_revisions": audit["source_revisions"],
        "before_matches_frozen": True, "after_matches_frozen": True,
        "map_bodies_exported": False,
        "records": {n: record(B / n) for n in (
            "frozen-production-pins.json", "qualification-before-pins.json",
            "qualification-after-pins.json")},
        "scope": "Retained qualification maps compared exactly; no new live filesystem scan during full trial.",
    })
    summary = read(B / "selected-test-summary.json")
    cases, groups = {}, []
    for i, name in enumerate(summary["groups"], 1):
        path = Path(name)
        label = path.stem
        prefix = f"controls/{i:02d}-{label}"
        counts = {"passed": 0, "skipped": 0, "failed": 0, "errors": 0}
        root = ET.parse(path)
        for node in root.findall(".//testcase"):
            status = ("failed" if node.find("failure") is not None else
                      "errors" if node.find("error") is not None else
                      "skipped" if node.find("skipped") is not None else "passed")
            counts[status] += 1
            key = (node.get("classname", ""), node.get("name", ""))
            if key in cases:
                assert cases[key] == status, key
            cases[key] = status
        assert counts["failed"] == counts["errors"] == 0
        assert all(n.tag in {"testsuites", "testsuite", "testcase", "skipped"}
                   for n in root.iter())
        for suffix in (".xml", "-command.json", "-exit.json"):
            copy(path.parent / (label + suffix), prefix + suffix)
        groups.append({"source_group": str(path), "export_prefix": prefix, "counts": counts})
    aggregate = {status: sum(v == status for v in cases.values())
                 for status in ("passed", "skipped", "failed", "errors")}
    assert len(cases) == summary["distinct_cases"] == 708
    assert aggregate == {"passed": 706, "skipped": 2, "failed": 0, "errors": 0}
    write("controls/recount.json", {
        "schema": "deduplicated-selected-control-outcomes@1", "distinct_cases": len(cases),
        "counts": aggregate, "deduplication_key": ["classname", "name"], "groups": groups,
        "native_seven_breakdown": {
            "actual_pinned_checkpoint_gte_inference_with_warm_reopened_registry_replay": 1,
            "preparation_and_tamper_cases": 6,
            "same_fixture_used_by_seven_cases": True,
            "fresh_numeric_worker_subprocess": True,
        },
        "source_binding_limits": [
            "Parent transport/context/final-profile commands have no per-run before/after source-pin record.",
            "Driver focused receipt pins the earlier central Harbor930 budget; later final-profile coverage exercises Harbor960.",
            "D controls and native02 receipts retain their explicit source pins; compilation and Docker qualification are distinct scopes.",
        ],
    })
    initial = []
    for label in ("transport-01", "transport-02"):
        path = B / (label + ".xml")
        failures = []
        for case in ET.parse(path).findall(".//testcase"):
            for node in case.findall("failure") + case.findall("error"):
                failures.append({
                    "classname": case.get("classname", ""), "name": case.get("name", ""),
                    "kind": node.tag, "raw_failure_body_exported": False,
                    "failure_text_sha256": hashlib.sha256((node.text or "").encode()).hexdigest(),
                })
        initial.append({"label": label, "outcome": read(B / (label + "-exit.json")),
                        "xml": record(path), "failures": failures,
                        "qualified": False, "superseded_by_passing_scope": "transport-03"})
        for suffix in ("-command.json", "-exit.json"):
            copy(B / (label + suffix), "prior-failures/" + label + suffix)
    write("prior-failures/observation.json", initial)
    live = read(M / "live-readiness-01.json")
    write("memory/local-readiness.json", {k: live[k] for k in (
        "schema", "observed_at", "seconds", "admitted", "host", "policy", "pressure_sources",
        "active_lease_count", "waiting_request_count", "production_scheduler_ledger_mutated",
        "causal_evidence_for_previous_runs")})
    delta = read(M / "host-memory-deltas-01.json")
    attrs = read(M / "host-memory-attribution-01.json")
    limits = read(M / "cgroup-high-limits-01.json")
    gpu_allocations = [int(s["gpu_compute"].split(",")[1].strip())
                       for s in attrs["samples"] if len(s["gpu_compute"].split(",")) == 2]
    write("memory/host-attribution-observation.json", {
        "schema": "bounded-host-memory-attribution-observation@1",
        "samples": len(attrs["samples"]), "seconds": delta["seconds"],
        "gpu_reported_allocation_mib": gpu_allocations,
        "benchmark_container_memory_high_event_delta": delta["benchmark_container_event_delta"]["high"],
        "gpu_service_memory_high_event_delta": delta["gpu_service_event_delta"]["high"],
        "user_ancestor_memory_high_event_delta": delta["user_ancestor_event_delta"]["high"],
        "benchmark_container_oom_delta": delta["benchmark_container_event_delta"]["oom"],
        "user_ancestor_oom_delta": delta["user_ancestor_event_delta"]["oom"],
        "user_ancestor_memory_full_avg10": [float(s["user_ancestor"]["memory_full"]["avg10"])
                                           for s in delta["sample_profiles"]],
        "benchmark_container_memory_full_avg10": [float(s["benchmark_container"]["memory_full"]["avg10"])
                                                  for s in delta["sample_profiles"]],
        "host_available_memory_kib": [s["meminfo"]["MemAvailable"] for s in delta["sample_profiles"]],
        "separate_cgroup_high_limit_snapshot": limits,
        "processes_stopped_during_diagnostic": attrs["processes_stopped"],
        "global_cache_dropped": attrs["cache_dropped"],
        "credentials_or_foreign_argv_read": attrs["credentials_or_foreign_argv_read"],
        "source_checkout_unchanged": attrs["source_checkout_unchanged"],
        "causal_limit": "Contemporaneous allocation and separate cgroup throttling observations; they do not prove the GPU service alone caused pressure, a past refusal, or a benchmark outcome.",
        "source_records": {n: record(M / n) for n in (
            "live-readiness-01.json", "host-memory-deltas-01.json",
            "host-memory-attribution-01.json", "cgroup-high-limits-01.json")},
    })
    write("observation.json", {
        "schema": "extended-budget-qualified-pending-full-trial@1",
        "qualification_passed": True, "full_trial_status": "pending",
        "official_reward": None, "completed_token_score": None, "advantage_claimed": False,
        "resource_profile": native["resource_profile"], "execution_budget": native["execution_budget"],
        "source_revisions": audit["source_revisions"],
        "archive_sha256": audit["archive_sha256"], "manifest_sha256": audit["manifest_sha256"],
        "qualification_provider_calls": native["provider_calls"],
        "qualification_official_verifier_executed": native["official_verifier_executed"],
        "qualification_controller_seconds": controller["seconds"],
        "selected_controls": {"distinct": 708, "passed": 706, "skipped": 2},
        "full_trial_service_pause": "User-authorized Leanstral pause is managed separately by the root controller; outcome and restoration have not been inferred in this package.",
        "manifest_sealed": False,
    })
    retained = [
        "qualification-01/qualification.json", "qualification-01/source384-result.json",
        "qualification-01/source384-context.json", "qualification-01/native-inference.json",
        "source384-config.json", "intent-requirements.json", "bundle/manifest.json",
        "build_fresh.py", "run_qualification.py", "launch_qualification.py", "full_trial.py",
        "run_tests.py", "frozen-production-pins.json", "qualification-before-pins.json",
        "qualification-after-pins.json",
    ]
    write("retained-records.json", {
        **EXPORTED, **{str(B / n): record(B / n) for n in retained},
    })
    copy(Path(__file__), "build_public_evidence.py")
    text = f"""# Extended supervisor budget qualification

The new `source384-5cpu-16gib-extended@1` archive passed ordinary Docker qualification. The full Terminal Bench trial is pending; this package does not report a task reward, completed token score, or advantage over a baseline. Its manifest remains unsealed so the root controller can add the independent full-trial outcome and service-restoration record.

Frozen revisions: A `{audit['source_revisions']['source']}`, D `{audit['source_revisions']['datasets']}`, kit `{audit['source_revisions']['kit']}`. Archive `{audit['archive_sha256']}` and manifest `{audit['manifest_sha256']}` bind the exact runtime. The retained 12,008 repository-member pin maps match before and after qualification; the package exports their hashes and counts instead of three duplicate maps. Public task inputs were unchanged and the qualification container was removed.

The selected profile provides 5 CPUs and 16 GiB, with driver900 seconds, work840, cleanup60, Source384 preparation180, Harbor960, exec910, qualification probe600, qualification exec630, and qualification outer2600. The default profile retains its original285/245/40-second driver/work/cleanup limits and2-percent memory-pressure threshold. The explicit local benchmark admission profile uses10 percent; it retains memory headroom, CPU/IO/PID/resource checks and requires an explicit local scheduler store.

Qualification measured Source38470.179885 seconds, initial context125.951173, warm observation9.091272, total probe146.881438, and controller416.998 seconds. These scopes overlap and must not be added. Native checkpoint/GTE inference executed. Of128 selected units,127 were decoded as unsupported, unverified candidates and one was deferred for the GTE token limit. No learned source-qualified proof authority, provider call, training step, download, or official benchmark verifier execution is claimed by qualification.

Six retained selected-control groups contain706 distinct passing cases and two environment-gated skipped cases after deduplication by test classname/name. The seven D source-unit controls include one actual pinned-checkpoint/GTE inference case with warm/reopened-registry replay and six preparation/tamper cases; the numerical worker uses a fresh subprocess. They are not seven independent inference experiments. The same fixture is shared by the seven cases. Initial transport collection and fixture failures are retained as nonqualifying metadata; corrected transport scope passed. The parent transport/context/final-profile runner did not record per-run source pins. The driver-focused receipt predates the final Harbor930-to960 adjustment; final-profile controls cover the final profile. Source-binding limitations remain explicit in `controls/recount.json`.

The separate live readiness check observed memory pressure3.71 percent against the explicit10-percent local benchmark threshold and admitted a request without modifying the production scheduler ledger. A bounded host diagnostic also observed approximately63,527 MiB of reported GPU allocation and2,677 ancestor `memory.high` events while benchmark-container high events stayed zero. A separate cgroup snapshot identifies other services with finite high limits. These observations support multiple potential contributors; they do not establish that the GPU service alone caused prior failures. No diagnostic stopped processes or dropped global caches. A later user-authorized service pause belongs to the separately retained full-trial controller, whose result is pending here.

This export contains reviewed command/exit/XML records, digests, numeric scope summaries and this artifact-only export recipe. It excludes production source bodies, prompts, raw model/checkpoint payloads, verifier bodies, credentials, seal/key stores, and scheduler databases. No full upstream test-suite or end-to-end performance improvement is claimed.
"""
    (P / "README.md").write_text(text)
    print(json.dumps({"package": str(P), "files": sum(p.is_file() for p in P.rglob("*")),
                      "sealed": False, "full_trial_status": "pending"}))


if __name__ == "__main__":
    main()
