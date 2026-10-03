"""Closed, source-free projection of the retained failed Harbor trial; no rerun."""
from pathlib import Path
import hashlib
import json
import shutil

B = Path(__file__).resolve().parent
R = B.parent / "source384-initial-context-lifetime-20261003"
T = R / "trial-02"
J = T / "jobs/supervisor-full-fix-code-vulnerability/fix-code-vulnerability__aaqEeLD"
P = B / "public-evidence"
assert not P.exists()
P.mkdir()
records = {}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    assert path.is_file() and not path.is_symlink() and path.stat().st_size <= 16*1024**2
    raw = path.read_bytes()
    records[str(path.relative_to(R))] = {"sha256": sha(raw), "bytes": len(raw)}
    return json.loads(raw)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write(name, value):
    path = P / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


receipt = read(T / "receipt.json")
result = read(J / "result.json")
driver = read(J / "agent/supervisor-result.json")
controller = read(R / "execute-02-exit.json")
prepare = read(R / "prepare-02-exit.json")
cleanup = read(T / "root-cleanup-observation.json")
correction = read(R / "host-controller-correction.json")
deployment = read(J / "agent/deployment/deployment.json")
resources = read(J / "agent/setup-cache/resources.json")
archive = read(R / "bundle/manifest.json")
frozen = read(R / "frozen-production-pins.json")
source = driver["initial_context"]["source384_context"]
invocations = driver["provider_invocations"]
assert len(invocations) == 1 and invocations[0]["phase"] == "planning"
invocation = invocations[0]
usage = invocation["native_rollout_usage"]["usage"]
assert usage["input_tokens"] + usage["output_tokens"] == usage["total_tokens"] == 22587
assert result["verifier_result"]["rewards"]["reward"] == 0.0
assert controller["returncode"] == receipt["harbor_returncode"] == 0
assert result["agent_result"]["metadata"]["driver_returncode"] == 1
assert driver["task_completed"] is False and driver["production_activation"] is False
assert driver["error_phase"] == "context" and driver["error"]["message"] == "Source384 context receipt exceeds its byte bound"
assert cleanup["container_removed"] and cleanup["rows"] == [] and cleanup["worker_cleanup_returncode"] == 0
assert len(canonical(source)) == 78563 and source["seconds"] < 90
assert source["summary"]["coverage"]["candidate_statuses"] == {"fail_open_source_contract_unsupported": 127}
assert resources["cpu_max"] == "500000 100000" and resources["memory_max"] == "12884901888"
assert len(frozen) == 33 and correction["source384_runtime_pins_unchanged"]
assert deployment["archive_sha256"] == archive["archive_sha256"] == correction["selected_archive_sha256"]

write("result.json", dict(schema="failed-full-source384-harbor-trial@1", task=receipt["task"],
    arm=receipt["arm"], trial_count=1, status="failed_before_coding_dispatch",
    controller_returncode=0, harbor_returncode=0, driver_returncode=1, official_reward=0.0,
    driver_official_reward=driver["official_reward"], task_completed=False, production_activation=False,
    planning_strategy=receipt["planning_strategy"], provider_invocations=1, coding_invocations=0,
    provider="codex_cli via llm_router", model=receipt["model"], reasoning_effort=receipt["reasoning_effort"],
    parallel_workers=receipt["parallel_workers"], provider_graph={k:driver["planning"]["provider_receipt"]["parse"][k]
        for k in ("goal_count", "task_count", "status", "plan_root_cid")},
    official_verifier_executed=True, benchmark_advantage_claimed=False, proof_authority=False,
    source384_native_phase_returned=True, source384_seconds=source["seconds"],
    original_task_files_at_deployment=deployment["original_inputs"]["files"],
    signed_source_files=len(source["source_hashes"]), coverage=source["summary"]["coverage"],
    source_qualified_proof_claimed=False, container_removed=True))

write("planning-usage.json", dict(schema="failed-full-trial-planning-usage@1", scope="one planning invocation only",
    usage=usage, cached_input_is_subset=True, reasoning_output_is_subset=True,
    total_definition="input_tokens + output_tokens; do not add cache or reasoning subsets",
    billing_total_verified=invocation["native_rollout_usage"]["billing_total_verified"],
    usage_available=invocation["native_rollout_usage"]["usage_available"],
    observed_token_count_records=invocation["native_rollout_usage"]["observed_token_count_records"],
    provider_model_task_complete_observed=invocation["native_rollout_usage"]["task_complete_observed"],
    benchmark_task_complete=False, rollout_sha256=invocation["native_rollout_usage"]["rollout_sha256"],
    native_usage_schema=invocation["native_rollout_usage"]["schema"],
    model=invocation["model"], provider=invocation["provider"], router_calls=invocation["router_calls"],
    model_prompt_bytes=invocation["model_prompt_bytes"], model_prompt_sha256=invocation["model_prompt_sha256"],
    requested_output_tokens=invocation["requested_output_tokens"],
    provider_output_token_cap_enforced=invocation["provider_output_token_cap_enforced"],
    enforced_total_token_budget=False, dollar_cost=receipt["dollar_cost"],
    native_agent_metrics={k:result["agent_result"][k] for k in ("n_input_tokens", "n_cache_tokens", "n_output_tokens", "cost_usd")},
    provider_tool_call_events=invocation["native_rollout_usage"]["tool_diagnostics"]["observed_tool_call_events"],
    raw_prompt_or_rollout_exported=False, efficiency_claim=False))

phases = driver["phases"]
initial_parts = driver["initial_context"]["nonoverlapping_seconds"]
write("timing-audit.json", dict(schema="failed-full-trial-timing-audit@1",
    controller_seconds=controller["seconds"], harbor_invocation_seconds=receipt["invocation_seconds"],
    harbor_stage_seconds=receipt["trials"][0]["durations_seconds"], driver_seconds=driver["seconds"],
    driver_phase_seconds=phases, driver_phase_sum_seconds=sum(phases.values()),
    driver_unassigned_seconds=driver["seconds"]-sum(phases.values()),
    initial_context_component_seconds=initial_parts,
    initial_context_component_sum_seconds=sum(initial_parts.values()),
    initial_context_unassigned_seconds=phases["initial_context_seconds"]-sum(initial_parts.values()),
    source384_native_seconds=source["seconds"], source384_native_deadline_seconds=90,
    planning_receipt_elapsed_seconds=driver["planning"]["elapsed_seconds"],
    planning_provider_invocation_seconds=invocation["seconds"],
    planning_provider_receipt_latency_ms=driver["planning"]["provider_receipt"]["provider"]["latency_ms"],
    prepare_host_seconds=prepare["seconds"], max_total_agent_seconds=driver["max_total_agent_seconds"],
    work_cutoff_seconds=driver["work_cutoff_seconds"], reserved_cleanup_seconds=driver["reserved_cleanup_seconds"],
    planning_and_cold_index_charged_to_agent_time=True,
    interpretation="Nested independently instrumented intervals are not additive. Reported unassigned intervals are arithmetic residuals, not attributed causes. Context ended on a byte-bound ValueError, not an identified timeout. No repeated-run speedup or cost advantage is inferred."))

write("failure.json", dict(schema="failed-full-trial-context-transport@1", error=driver["error"],
    error_phase=driver["error_phase"], traceback=driver["error_traceback"],
    canonical_receipt_bytes=len(canonical(source)), canonical_receipt_sha256=sha(canonical(source)),
    archived_inline_transport_limit_bytes=32768, canonical_observer_receipt_limit_bytes=131072,
    failure_resources=driver["failure_resources"], failure_scheduler=driver["failure_scheduler"],
    observations_are_after_unwind=True, resource_cause_inferred=False,
    boundary="write_task_context_bundle/_source384_receipt, after initial-context and provider planning returned"))
write("cleanup-observation.json", cleanup)
write("resource-observation.json", resources)
write("runtime-binding.json", dict(schema="failed-full-trial-runtime-binding@1",
    archive_sha256=archive["archive_sha256"], manifest_sha256=sha((R/"bundle/manifest.json").read_bytes()),
    manifest_member_count=len(archive["files"]), resource_profile=receipt["resource_profile"],
    setup_cache_policy=archive["setup_cache"]["policy"],
    host_correction=correction, frozen_execution_pins=frozen,
    qualification_evidence=dict(path="docs/agent_supervisor/evidence/source384-lifetime-advice-qualification-20261003/manifest.json",
        sha256="148fb5034b2fc166d0d6e9632c3aec3b67db3c7c1dcf866259ed731157274797"),
    source384_receipt=dict(checkpoint_sha256=source["checkpoint_sha256"], config_sha256=source["config_sha256"],
        inference_sha256=source["inference_sha256"], source_head=source["source_head"],
        producer=source["producer"], version_id=source["version_id"]),
    actual_inference_artifact_exported_in_this_trial_package=False,
    inference_model_load_count_not_independently_recorded_here=True,
    comparison_status={k:receipt["comparison_controls"][k] for k in ("status","configuration_unchanged","reason")},
    comparison_is_matched_outcome_evidence=False))
public = read(J / "agent/public-output-evidence/receipt.json")
write("public-output-observation.json", dict(schema=public["schema"], status=public["status"],
    declared_files=public["declared_files"], files=public["files"],
    diffs={name:{k:v for k,v in row.items() if k not in ("artifact",)} for name,row in public["diffs"].items()},
    before_receipt_sha256=public["before_receipt_sha256"], after_receipt_sha256=public["after_receipt_sha256"],
    implementation_sha256=public["implementation_sha256"], candidate_code_executed=public["candidate_code_executed"],
    private_state_accessed=public["private_state_accessed"], verifier_state_accessed=public["verifier_state_accessed"],
    post_task_scope="Only declared bottle.py and report.jsonl observed; not a post-task comparison of all218 original paths.",
    source_or_diff_bodies_exported=False))
write("intent-status.json", {k:result["agent_result"]["metadata"]["intent_preplanning"][k] for k in
    ("status","before_goal_decomposition","artifact_persisted","supplied_to_router","completion_authority","execution_authority")})
for operation in ("prepare", "execute"):
    original = read(R/f"{operation}-02-command.json")
    write(f"commands/{operation}.json", {k:v for k,v in original.items() if k != "env"})
    write(f"commands/{operation}-exit.json", read(R/f"{operation}-02-exit.json"))
write("original-records.json", dict(schema="retained-original-record-identities@1", records=records,
    original_payloads_included=False, note="Relative to the original source384-initial-context-lifetime-20261003 artifact directory. Original paths are provenance, not package member references."))
write("independent-outcome-audit.json", read(R/"independent-trial-02-outcome-audit.json"))
shutil.copyfile(__file__, P/"build_public_evidence.py")

(P/"README.md").write_text('''# Failed full supervisor trial after ordinary qualification

The single full `fix-code-vulnerability` Harbor trial received official reward
**0**. Controller and Harbor both returned 0 because result collection completed;
the supervisor driver returned 1, and no coding worker was dispatched. Initial
context and provider planning returned, then task-context publication rejected a
78,563-byte Source384 receipt against the archived 32,768-byte inline envelope
limit. The native observer's separate 131,072-byte receipt limit was not changed.
This package freezes that failed generation; subsequent transport fixes are not
part of its execution.

One llm_router → Codex CLI planning call used 21,497 input tokens and 1,090 output
tokens: **22,587 total**. The 11,264 cached-input tokens are already included in
input; 151 reasoning tokens are included in output. Cost is unavailable, billing
totals are unverified, and the requested output cap was not enforced. This is
planning-only usage for an unsuccessful run, not a completed-work token score or
an efficiency advantage. The provider's own `task_complete` event ended its
planning response; the benchmark task remained incomplete.

Source384 returned in 78.254s within its 90s deadline. Initial context took
139.991s, planning 68.953s, and the failing task-context stage 24.264s; the driver
took 244.323s. Harbor measured 536.169s setup and 248.231s agent execution, and
the host controller took 816.954s. These nested measurements must not be added
together. The unchanged profile observed five CPUs and 12,288MiB. Planning and
cold indexing were charged to the agent budget.

The signed source population was 220 files, including the original 218 public
inputs and two supervisor inputs. Across 31 Python files/944 functions, 128 units
were selected, 127 decoded but all remained unsupported by the source-contract
guard, and one exceeded the GTE token bound. No formal property was proved.
Intent preprocessing remained fail-open because no checkpoint was selected.
The provider graph contained two goals and one task. The successful preceding
ordinary qualification is separate evidence, not this task's reward.

The exact run container was absent in the retained cleanup observation and the
worker cleanup command returned 0. The driver's remaining-processes field is
null, so this package does not claim a native STOP/absent-process-tree receipt.
After-task public observation found unchanged `bottle.py` and no `report.jsonl`;
it did not recompare every original source file. Failure resources/scheduler
observations are explicitly after unwind and do not prove a resource cause.

Only selected metrics, hashes, bounded stack-frame metadata and framework
collection code are published. Task/instruction/diff bodies, provider prompts or
rollouts, hidden verifier contents, credentials, checkpoints, archives, databases
and scheduler capabilities are excluded. Original record paths in the identity
table are host-relative provenance, not missing package members. RPI-019 and the
matched benchmark campaign remain open; the overall backlog stays 18/32 closed.
''')
files=[]
for path in sorted(P.rglob('*')):
    if path.is_file():
        raw=path.read_bytes();files.append(dict(path=path.relative_to(P).as_posix(),bytes=len(raw),sha256=sha(raw)))
write("manifest.json",dict(schema="source384-full-trial-receipt-failure-evidence@1",files=files))
print(json.dumps(dict(manifest_sha256=sha((P/'manifest.json').read_bytes()),members=len(files),bytes=sum(r['bytes'] for r in files))))
