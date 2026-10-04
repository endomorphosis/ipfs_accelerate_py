"""One genuine router proposal against independently signed local constraints.

Preparation and bridge preflight make no provider call. ``--live`` permits
exactly one shared-router call, then requires native parsing/local admission.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import time

from ipfs_accelerate_py.agent_supervisor.control.profile_authority import load_local_profile
from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_directory_scanner import (
    RepositoryAllowlist,
    repository_root_cid,
    scan_prompt_directory_detailed,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
    PROMPT_GOAL_PROPOSAL_SCHEMA,
    PromptGoalPlannerConfig,
    _select_evidence,
    build_prompt_goal_provider_request,
    generate_prompt_goal_graph,
    parse_prompt_goal_graph,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    DirectoryScanPolicy,
    LocalFallbackPolicy,
    PromptOutputPolicy,
    PromptPlanningPolicy,
    PromptSource,
    PromptWorkflowBudget,
    PromptWorkflowRequest,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


def prepare(output: Path) -> dict:
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    repository = output / "repository"
    repository.mkdir()
    (repository / "answer.py").write_text("def answer():\n    return 1\n")
    (repository / "test_answer.py").write_text("from answer import answer\nassert answer() == 2\n")
    for args in [
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Local planner",
            "-c",
            "user.email=local@example.invalid",
            "commit",
            "-qm",
            "independent benchmark",
        ],
    ]:
        subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True)
    profile_dir, lifecycle_dir = output / "profile", output / "lifecycle"
    bootstrap = Supervisor.init_local(
        repository=repository, consent=True, profile_dir=profile_dir, lifecycle_dir=lifecycle_dir
    )
    profile = load_local_profile(
        repository_cid=bootstrap["repository_cid"],
        profile_dir=profile_dir,
        lifecycle_dir=lifecycle_dir,
    )
    policy = content_identity(local.LOCAL_POLICY)
    argv = ["python3", "-B", "test_answer.py"]
    acceptance = {
        "criterion_key": "answer-is-two",
        "criterion": "The immutable public answer check passes",
        "evidence_cids": [],
        "validation_keys": ["public-answer"],
    }
    spec = {
        "task_key": "LOCAL-TASK",
        "scope_paths": ["answer.py", "test_answer.py"],
        "dependencies": [],
        "outputs": [{"path": "answer.py", "effect": "modify", "media_type": "text/x-python"}],
        "validations": [
            {
                "validation_key": "public-answer",
                "argv": argv,
                "cwd": ".",
                "expected_exit_codes": [0],
                "policy_cid": policy,
            }
        ],
        "acceptance": [acceptance],
    }
    domains = local.local_planning_domain_declarations(
        repository=repository,
        profile_dir=profile_dir,
        lifecycle_dir=lifecycle_dir,
        task_specs=[spec],
    )
    allowlist = RepositoryAllowlist.from_roots([repository])
    budget = PromptWorkflowBudget(
        max_files=16,
        max_scan_bytes=65536,
        max_file_bytes=8192,
        max_symbols=128,
        max_prompt_tokens=8192,
        max_provider_tokens=4096,
        max_latency_ms=90000,
        max_goals=2,
        max_tasks=1,
        max_evidence=16,
        max_graph_depth=4,
        max_serialized_bytes=262144,
        max_rescue_actions=1,
    )
    request = PromptWorkflowRequest(
        prompt_source=PromptSource.inline(
            "Plan the smallest bounded repair that returns two from answer().",
            redacted_metadata={
                "summary": "Repair answer.py so answer() returns 2; preserve the immutable test. Generate root goal, one child subgoal and LOCAL-TASK with exact signed acceptance, outputs and argv from constraints."
            },
        ),
        repository_root=str(repository),
        directory=str(repository),
        repository_root_cid=repository_root_cid(repository),
        allowlist_cid=allowlist.allowlist_cid,
        scan_policy=DirectoryScanPolicy(
            policy_id="local-bounded-scan", scanner_version="1", include_patterns=("*.py",)
        ),
        planning_policy=PromptPlanningPolicy(
            policy_id="local-bounded-planner",
            provider_preferences=("grok_cli",),
            model_preferences=("grok-4.6",),
            allow_model=True,
            fallback_policy=LocalFallbackPolicy.DISABLED,
        ),
        output_policy=PromptOutputPolicy(
            policy_id="local-planner-preview",
            mode="markdown",
            output_root=str(repository),
            allowed_output_roots=(str(repository),),
            markdown_path=".runtime/planner.todo.md",
        ),
        budget=budget,
        caller=profile.identity_did,
        program_root=local._tree(local._sources(repository, ["answer.py", "test_answer.py"])),
        intent_ir_root=content_identity(domains["intent"]),
        legal_ir_root=content_identity(domains["legal"]),
        security_ir_root=content_identity(domains["security"]),
        policy_root=policy,
    )
    details = scan_prompt_directory_detailed(request, repository_allowlist=allowlist)
    # Bind a fresh request to the actual native producer root, then rescan its
    # exact bytes. Never patch a scan receipt's request identity after the fact.
    request = replace(request, program_root=details.receipt.program_root)
    details = scan_prompt_directory_detailed(
        request, repository_allowlist=allowlist, previous=details
    )
    scan = details.receipt
    config = PromptGoalPlannerConfig(
        repo_root=repository,
        provider="grok_cli",
        model="grok-4.6",
        timeout_seconds=90,
        max_new_tokens=4096,
        allow_local_fallback=False,
        allowed_validation_prefixes=(tuple(argv),),
    )
    evidence = _select_evidence(request, scan, config)
    # Evidence supports the specification's provenance; these handles do not
    # assert that its acceptance check has passed.
    acceptance["evidence_cids"] = [evidence[0].evidence_cid]
    inputs = {
        "request": request.to_dict(),
        "scan": scan.to_dict(),
        "domain_declarations": domains,
        "selected_evidence": [item.to_dict() for item in evidence],
    }
    manifest = local.author_local_benchmark_manifest(
        repository=repository,
        profile_dir=profile_dir,
        lifecycle_dir=lifecycle_dir,
        task_specs=[spec],
        planning_roots={
            "request_cid": request.request_cid,
            "scan_cid": scan.scan_cid,
            "program_root": request.program_root,
        },
        planning_inputs=inputs,
    )
    constraints = {
        "allowed_paths": spec["scope_paths"],
        "validation_commands": [argv],
        "constraint_summaries": [
            "Use exactly two goals: LOCAL-GOAL root and LOCAL-SUBGOAL child, with exactly one task LOCAL-TASK owned by the child. All dependency arrays, risks, assumptions, unresolved_questions and uncertainty_debt must be empty.",
            "All goal and task acceptance arrays must exactly equal: "
            + json.dumps(spec["acceptance"], sort_keys=True),
            "The task outputs/scope/validation must exactly match this independent signed declaration (validation policy_cid is added by the native parser, omit that field in the model proposal): "
            + json.dumps(spec, sort_keys=True),
            "Use source evidence references only as descriptive context. No domain assurance, policy, proof, execution or completion claims. Resource cpu-medium; fallback fail_closed.",
        ],
    }
    provider_prompt = build_prompt_goal_provider_request(
        request, scan, config=config, constraint_summaries=constraints
    )
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    (output / "scan.json").write_text(json.dumps(scan.to_dict(), sort_keys=True, indent=2) + "\n")
    (output / "provider-request.json").write_text(provider_prompt + "\n")
    for index, artifact in enumerate(details.artifacts):
        (output / f"scan-artifact-{index}.json").write_text(
            json.dumps(local._plain(dict(artifact.payload)), sort_keys=True, indent=2) + "\n"
        )
    return dict(
        output=output,
        repository=repository,
        request=request,
        scan=scan,
        config=config,
        constraints=constraints,
        manifest=manifest,
        spec=spec,
        evidence=evidence,
    )


def preflight_proposal(prepared: dict) -> dict:
    """Provider-free native parser qualification; never returned as live output."""
    spec, evidence = prepared["spec"], prepared["evidence"]
    refs = [evidence[0].evidence_cid]
    goal = {
        "goal_key": "LOCAL-GOAL",
        "parent_goal_key": "",
        "dependency_goal_keys": [],
        "title": "Repair answer",
        "objective": "Return two",
        "rationale": "Bounded public acceptance",
        "scope_paths": spec["scope_paths"],
        "acceptance": spec["acceptance"],
        "evidence_cids": refs,
        "risks": [],
        "assumptions": [],
    }
    child = {**goal, "goal_key": "LOCAL-SUBGOAL", "parent_goal_key": "LOCAL-GOAL"}
    task = {
        "task_key": "LOCAL-TASK",
        "goal_key": "LOCAL-SUBGOAL",
        "dependency_task_keys": [],
        "objective": "Return two",
        "rationale": "Satisfy immutable public acceptance",
        "scope_paths": spec["scope_paths"],
        "outputs": spec["outputs"],
        "validations": [
            {key: value for key, value in item.items() if key != "policy_cid"}
            for item in spec["validations"]
        ],
        "acceptance": spec["acceptance"],
        "evidence_cids": refs,
        "priority": "P1",
        "track": "local-benchmark",
        "bundle": "",
        "parallel_lane": "",
        "resource_class": "cpu-medium",
        "predicted_files": ["answer.py"],
        "risks": [],
        "assumptions": [],
        "fallback_behavior": "fail_closed",
    }
    return {
        "schema": PROMPT_GOAL_PROPOSAL_SCHEMA,
        "proposal_version": "1",
        "root_goal_key": "LOCAL-GOAL",
        "goals": [goal, child],
        "tasks": [task],
        "unresolved_questions": [],
        "uncertainty_debt": [],
    }


def run(output: Path, *, live=False, max_turns=1) -> dict:
    prepared = prepare(output)
    proposal = preflight_proposal(prepared)
    graph = parse_prompt_goal_graph(
        json.dumps(proposal),
        prepared["request"],
        prepared["scan"],
        config=prepared["config"],
        constraint_summaries=prepared["constraints"],
    )
    local.admit_local_benchmark_plan(graph=graph, manifest=prepared["manifest"])
    if not live:
        result = {
            "schema": "local-native-planner-bridge-preflight@1",
            "qualified": True,
            "provider_calls": 0,
            "goals": len(graph.goals),
            "tasks": len(graph.tasks),
            "live_planner_exercised": False,
        }
    else:
        from ipfs_accelerate_py.llm_router import generate_text
        from ipfs_accelerate_py.router_deps import RouterDeps
        from ipfs_accelerate_py.cli_runtime.cli_metadata import (
            set_last_cli_observation,
            get_last_cli_observation,
        )

        calls = 0
        observation = {}
        deps = RouterDeps()
        started = time.monotonic()

        def router(prompt):
            nonlocal calls, observation
            if calls:
                raise RuntimeError("one provider call maximum")
            calls += 1
            set_last_cli_observation("grok_cli", {})
            try:
                response = generate_text(
                    prompt,
                    provider="grok_cli",
                    model_name="grok-4.6",
                    allow_local_fallback=False,
                    allow_cross_provider_fallback=False,
                    max_new_tokens=4096,
                    max_tokens=4096,
                    timeout=90,
                    temperature=0,
                    task_kind="planning",
                    allocation_path="cli",
                    deps=deps,
                    grok_max_turns=max_turns,
                    grok_tools="",
                    grok_permission_mode="dontAsk",
                    grok_cli_cmd=[
                        "grok",
                        "--cwd",
                        str(prepared["repository"]),
                        "--deny",
                        "*",
                        "--disallowed-tools",
                        "run_terminal_command,read_file,search_replace,list_dir,grep,kill_command_or_subagent,todo_write,get_command_or_subagent_output,spawn_subagent,scheduler_create,scheduler_delete,scheduler_list,monitor,search_tool,use_tool,workflow,enter_plan_mode,exit_plan_mode,ask_user_question,send_feedback,image_gen,image_edit,image_to_video,reference_to_video,write",
                        "--system-prompt-override",
                        "Return only one JSON object matching the supplied response_schema and exact independent constraints. This is text-only planning. No tools or workspace inspection are permitted or needed. All required input is in the prompt.",
                    ],
                )
                (output / "provider-response.txt").write_text(response)
                return response
            finally:
                observed = get_last_cli_observation("grok_cli")
                observation = {
                    key: observed[key]
                    for key in (
                        "prompt_tokens",
                        "completion_tokens",
                        "cached_tokens",
                        "total_cost_usd",
                        "session_id",
                        "model_id",
                        "num_turns",
                    )
                    if key in observed
                }
                (output / "provider-observation.json").write_text(
                    json.dumps(observation, sort_keys=True, indent=2) + "\n"
                )

        try:
            planning = generate_prompt_goal_graph(
                prepared["request"],
                prepared["scan"],
                router=router,
                config=prepared["config"],
                constraint_summaries=prepared["constraints"],
            )
        except Exception as exc:
            result = {
                "schema": "local-native-live-planner@1",
                "qualified": False,
                "provider_calls": calls,
                "provider": "grok_cli",
                "model": "grok-4.6",
                "provider_observation": observation,
                "provider_receipt": getattr(exc, "provider_receipt", None),
                "failure": {"type": type(exc).__name__, "message": str(exc)},
                "elapsed_seconds": time.monotonic() - started,
                "max_turns": max_turns,
                "timeout_seconds": 90,
                "provider_output_token_cap_enforced": False,
                "fallback_used": False,
                "benchmark_performance_claimed": False,
            }
            (output / "result.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
            return result
        receipt = planning.receipt.to_dict()
        (output / "planner-receipt.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n"
        )
        result = {
            "schema": "local-native-live-planner@1",
            "provider_calls": calls,
            "provider": "grok_cli",
            "model": "grok-4.6",
            "provider_observation": observation,
            "planner_receipt": receipt,
            "elapsed_seconds": time.monotonic() - started,
            "live_planner_exercised": True,
            "qualified": False,
            "completion_authority": False,
            "external_ir_assurance": "unavailable",
            "benchmark_performance_claimed": False,
            "provider_output_token_cap_enforced": False,
            "post_response_token_estimate_bound": 4096,
            "timeout_seconds": 90,
            "max_turns": max_turns,
        }
        try:
            if planning.receipt.outcome != "provider" or planning.receipt.fallback.used:
                raise ValueError(
                    "provider proposal not accepted; deterministic fallback is not qualification"
                )
            admission = local.admit_local_benchmark_plan(
                graph=planning.graph, manifest=prepared["manifest"]
            )
            with IntentRepository(output / "intent.duckdb") as intent:
                materialized = local.materialize_local_benchmark_plan(
                    admission=admission, intent=intent
                )
            (output / "admission.json").write_text(
                json.dumps(admission, sort_keys=True, indent=2) + "\n"
            )
            result.update(
                qualified=True,
                task_cids=materialized["task_cids"],
                goals=len(planning.graph.goals),
                tasks=len(planning.graph.tasks),
                subgoals=sum(bool(goal.parent_goal_cid) for goal in planning.graph.goals),
            )
        except Exception as exc:
            result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    (output / "result.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-turns", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.output.absolute(), live=args.live, max_turns=args.max_turns),
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
