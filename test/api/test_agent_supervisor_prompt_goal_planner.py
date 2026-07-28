from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
    PROMPT_GOAL_PROPOSAL_SCHEMA,
    PROMPT_GOAL_PROVIDER_REQUEST_SCHEMA,
    PromptGoalPlannerConfig,
    PromptGoalProposalError,
    build_prompt_goal_provider_request,
    deterministic_prompt_goal_graph,
    generate_prompt_goal_graph,
    parse_prompt_goal_graph,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    DirectoryScanPolicy,
    DirectoryScanReceipt,
    LocalFallbackPolicy,
    OutputMode,
    PromptEvidenceRecord,
    PromptGoalGraph,
    PromptOutputPolicy,
    PromptPlanningPolicy,
    PromptSource,
    PromptWorkflowBudget,
    PromptWorkflowRequest,
    prompt_workflow_cid,
)


def _cid(name: str) -> str:
    return prompt_workflow_cid({"fixture": name})


def _budget(**changes: int) -> PromptWorkflowBudget:
    values = {
        "max_files": 1_000,
        "max_scan_bytes": 8 * 1024 * 1024,
        "max_file_bytes": 512 * 1024,
        "max_symbols": 10_000,
        "max_prompt_tokens": 8_192,
        "max_provider_tokens": 8_192,
        "max_latency_ms": 30_000,
        "max_goals": 16,
        "max_tasks": 64,
        "max_evidence": 128,
        "max_graph_depth": 8,
        "max_serialized_bytes": 256 * 1024,
        "max_rescue_actions": 4,
    }
    values.update(changes)
    return PromptWorkflowBudget(**values)


def _request(
    *,
    allow_model: bool = True,
    budget: PromptWorkflowBudget | None = None,
) -> PromptWorkflowRequest:
    return PromptWorkflowRequest(
        prompt_source=PromptSource.inline(
            "Improve bounded retry planning",
            redacted_metadata={
                "summary": "Bounded retry planner request",
                "sensitivity": "redacted",
            },
        ),
        repository_root="/workspace/repository",
        directory="/workspace/repository/pkg",
        repository_root_cid=_cid("repository"),
        allowlist_cid=_cid("allowlist"),
        scan_policy=DirectoryScanPolicy(
            policy_id="scan:strict",
            scanner_version="1.0.0",
            include_patterns=("**/*.py",),
            exclude_patterns=(".git/**",),
        ),
        planning_policy=PromptPlanningPolicy(
            policy_id="planning:strict",
            provider_preferences=("test-provider",),
            model_preferences=("test-model",),
            allow_model=allow_model,
            fallback_policy=LocalFallbackPolicy.REQUIRED,
        ),
        output_policy=PromptOutputPolicy(
            policy_id="output:preview",
            mode=OutputMode.MARKDOWN,
            output_root="/workspace/repository",
            allowed_output_roots=("/workspace/repository",),
            markdown_path="plans/prompt.todo.md",
        ),
        budget=budget or _budget(),
        caller="principal:test-suite",
        program_root=_cid("program"),
        intent_ir_root=_cid("intent"),
        legal_ir_root=_cid("legal"),
        security_ir_root=_cid("security"),
        policy_root=_cid("policy"),
    )


def _evidence(
    key: str = "scan:retry-planner",
    *,
    summary: str = "Retry planner implementation and focused tests.",
    paths: tuple[str, ...] = ("pkg/retry_planner.py",),
) -> PromptEvidenceRecord:
    return PromptEvidenceRecord(
        evidence_key=key,
        source_kind="directory_scan",
        artifact_cid=_cid(f"artifact:{key}"),
        summary=summary,
        repository_paths=paths,
        claim_keys=(f"claim:{key}",),
        provenance={"scanner": "fixture"},
    )


def _scan(
    request: PromptWorkflowRequest,
    *,
    evidence: tuple[PromptEvidenceRecord, ...] | None = None,
    counts: dict[str, int] | None = None,
) -> DirectoryScanReceipt:
    return DirectoryScanReceipt(
        request_cid=request.request_cid,
        repository_root=request.repository_root,
        directory=request.directory,
        repository_root_cid=request.repository_root_cid,
        dirty_worktree_root=_cid(
            "dirty:" + ",".join(item.evidence_key for item in (evidence or (_evidence(),)))
        ),
        scanner_policy_cid=request.scan_policy.content_id,
        program_root=request.program_root,
        ast_root=_cid("ast"),
        index_root=_cid("index"),
        budget=request.budget,
        evidence=evidence or (_evidence(),),
        counts=counts or {"files": 1, "scan_bytes": 1_024, "symbols": 8},
    )


def _proposal(scan: DirectoryScanReceipt, **changes: object) -> dict[str, object]:
    evidence_cid = scan.evidence[0].evidence_cid
    acceptance = {
        "criterion_key": "criterion:pytest",
        "criterion": "The focused retry planner tests pass.",
        "evidence_cids": [evidence_cid],
        "validation_keys": ["validation:pytest"],
    }
    values: dict[str, object] = {
        "schema": PROMPT_GOAL_PROPOSAL_SCHEMA,
        "proposal_version": "1",
        "root_goal_key": "goal:root",
        "goals": [
            {
                "goal_key": "goal:root",
                "parent_goal_key": "",
                "dependency_goal_keys": [],
                "title": "Improve bounded retry planning",
                "objective": "Produce an evidence-backed retry planner improvement.",
                "rationale": "The scan identifies the retry planner implementation.",
                "scope_paths": ["pkg"],
                "acceptance": [acceptance],
                "evidence_cids": [evidence_cid],
                "risks": ["Retry behavior may have timing-sensitive edge cases."],
                "assumptions": ["The pinned scan remains current."],
            }
        ],
        "tasks": [
            {
                "task_key": "task:retry-planner",
                "goal_key": "goal:root",
                "dependency_task_keys": [],
                "objective": "Implement bounded retry planner behavior.",
                "rationale": "The root criterion requires a focused implementation task.",
                "scope_paths": ["pkg/retry_planner.py"],
                "outputs": [
                    {
                        "path": "pkg/retry_planner.py",
                        "effect": "modify",
                        "media_type": "text/x-python",
                    }
                ],
                "validations": [
                    {
                        "validation_key": "validation:pytest",
                        "argv": ["python", "-m", "pytest", "pkg/tests", "-q"],
                        "cwd": ".",
                        "expected_exit_codes": [0],
                    }
                ],
                "acceptance": [acceptance],
                "evidence_cids": [evidence_cid],
                "priority": "P0",
                "track": "prompt-goal-planning",
                "bundle": "prompt-workflow/planning",
                "parallel_lane": "retry-planner",
                "resource_class": "cpu-small",
                "predicted_files": ["pkg/retry_planner.py"],
                "risks": ["Retry timing could regress."],
                "assumptions": ["Focused tests cover the requested behavior."],
                "fallback_behavior": "fail_closed",
            }
        ],
        "unresolved_questions": [],
        "uncertainty_debt": ["Admission must verify current retry semantics."],
    }
    values.update(changes)
    return values


def _encoded_proposal(scan: DirectoryScanReceipt, **changes: object) -> str:
    return json.dumps(
        _proposal(scan, **changes),
        sort_keys=True,
        separators=(",", ":"),
    )


def test_provider_request_is_canonical_body_free_bounded_and_handle_only() -> None:
    request = _request()
    scan = _scan(request)

    prompt = build_prompt_goal_provider_request(
        request,
        scan,
        capabilities={
            "available": True,
            "operations": ["plan"],
            "resource_classes": ["cpu-small"],
        },
        constraint_summaries={
            "allowed_paths": ["pkg"],
            "protected_paths": [
                "docs/architecture/agent_supervisor_self_improvement.todo.md"
            ],
            "policy_roots": [request.policy_root],
        },
    )
    payload = json.loads(prompt)

    assert prompt == json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert payload["schema"] == PROMPT_GOAL_PROVIDER_REQUEST_SCHEMA
    assert payload["response_schema"]["$id"] == PROMPT_GOAL_PROPOSAL_SCHEMA
    assert payload["response_schema"]["additionalProperties"] is False
    assert payload["response_schema"]["properties"]["goals"]["maxItems"] == (
        request.budget.max_goals
    )
    assert (
        payload["response_schema"]["definitions"]["task"][
            "additionalProperties"
        ]
        is False
    )
    assert payload["request_core"]["request_cid"] == request.request_cid
    assert payload["request_core"]["scan_cid"] == scan.scan_cid
    assert payload["constraints"]["completion_authoritative"] is False
    assert payload["constraints"]["shell_allowed"] is False
    assert len(payload["evidence_handles"]) <= 64
    assert "Improve bounded retry planning" not in prompt
    assert "transient_body" not in prompt
    assert "source_body" not in prompt


def test_strict_provider_graph_compiles_local_dependencies_to_canonical_cids() -> None:
    request = _request()
    scan = _scan(request)

    graph = parse_prompt_goal_graph(_encoded_proposal(scan), request, scan)

    assert isinstance(graph, PromptGoalGraph)
    assert graph.request_cid == request.request_cid
    assert graph.scan_cid == scan.scan_cid
    assert graph.root_goal.goal_key == "goal:root"
    assert graph.tasks[0].goal_cid == graph.root_goal.goal_cid
    assert graph.tasks[0].outputs[0].path == "pkg/retry_planner.py"
    assert graph.tasks[0].validations[0].argv[:3] == (
        "python",
        "-m",
        "pytest",
    )
    assert set(graph.tasks[0].policy_roots) == {
        request.policy_root,
        request.intent_ir_root,
        request.legal_ir_root,
        request.security_ir_root,
    }
    assert PromptGoalGraph.from_json(graph.to_json()).plan_root_cid == (
        graph.plan_root_cid
    )


def test_provider_success_emits_complete_hash_only_receipt() -> None:
    request = _request()
    scan = _scan(request)
    response = _encoded_proposal(scan)
    seen: list[str] = []

    result = generate_prompt_goal_graph(
        request,
        scan,
        router=lambda prompt: seen.append(prompt) or response,
    )
    receipt = result.receipt.to_dict()
    receipt_text = result.receipt.to_json()

    assert seen
    assert result.provider_succeeded
    assert receipt["outcome"] == "provider"
    assert receipt["provider"]["status"] == "succeeded"
    assert receipt["parse"]["status"] == "succeeded"
    assert receipt["fallback"]["used"] is False
    assert receipt["provider"]["response_bytes"] == len(response.encode())
    assert receipt["provider"]["response_sha256"].startswith("sha256:")
    assert response not in receipt_text
    assert seen[0] not in receipt_text
    assert "Improve bounded retry planning" not in receipt_text


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda payload: {**payload, "authority_claims": {"complete": True}},
            "unknown_or_missing_field",
        ),
        (
            lambda payload: {
                **payload,
                "tasks": [
                    {
                        **payload["tasks"][0],
                        "outputs": [
                            {
                                **payload["tasks"][0]["outputs"][0],
                                "path": "../escape.py",
                            }
                        ],
                    }
                ],
            },
            "invalid_path",
        ),
        (
            lambda payload: {
                **payload,
                "tasks": [
                    {
                        **payload["tasks"][0],
                        "validations": [
                            {
                                **payload["tasks"][0]["validations"][0],
                                "argv": ["bash", "-c", "rm -rf repo"],
                            }
                        ],
                    }
                ],
            },
            "forbidden_instruction",
        ),
        (
            lambda payload: {
                **payload,
                "tasks": [
                    {
                        **payload["tasks"][0],
                        "fallback_behavior": "mark task complete",
                    }
                ],
            },
            "forbidden_instruction",
        ),
        (
            lambda payload: {
                **payload,
                "tasks": [
                    {
                        **payload["tasks"][0],
                        "validations": [],
                    }
                ],
            },
            "output_too_large",
        ),
        (
            lambda payload: {
                **payload,
                "goals": [
                    {
                        **payload["goals"][0],
                        "parent_goal_key": "goal:missing",
                    }
                ],
            },
            "orphan_reference",
        ),
    ],
)
def test_untrusted_fields_paths_shell_completion_and_orphans_fail_closed(
    mutate, reason: str
) -> None:
    request = _request()
    scan = _scan(request)
    payload = mutate(_proposal(scan))

    with pytest.raises(PromptGoalProposalError) as raised:
        parse_prompt_goal_graph(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            request,
            scan,
        )
    assert raised.value.reason_code == reason


def test_prose_wrappers_duplicate_keys_cycles_and_protected_paths_are_rejected() -> None:
    request = _request()
    scan = _scan(request)
    encoded = _encoded_proposal(scan)

    with pytest.raises(PromptGoalProposalError, match="unwrapped") as wrapped:
        parse_prompt_goal_graph(f"```json\n{encoded}\n```", request, scan)
    assert wrapped.value.reason_code == "prose_wrapper"

    duplicate = encoded.replace(
        '"proposal_version":"1"',
        '"proposal_version":"1","proposal_version":"1"',
        1,
    )
    with pytest.raises(PromptGoalProposalError) as duplicated:
        parse_prompt_goal_graph(duplicate, request, scan)
    assert duplicated.value.reason_code == "duplicate_key"

    payload = _proposal(scan)
    original = payload["tasks"][0]
    payload["tasks"] = [
        {**original, "task_key": "task:a", "dependency_task_keys": ["task:b"]},
        {**original, "task_key": "task:b", "dependency_task_keys": ["task:a"]},
    ]
    with pytest.raises(PromptGoalProposalError) as cyclic:
        parse_prompt_goal_graph(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            request,
            scan,
        )
    assert cyclic.value.reason_code == "cycle"

    protected = _proposal(scan)
    protected["tasks"] = [
        {
            **protected["tasks"][0],
            "scope_paths": [
                "pkg/agent_supervisor_self_improvement.todo.md"
            ],
            "outputs": [
                {
                    "path": "pkg/agent_supervisor_self_improvement.todo.md",
                    "effect": "modify",
                    "media_type": "text/markdown",
                }
            ],
            "predicted_files": [
                "pkg/agent_supervisor_self_improvement.todo.md"
            ],
        }
    ]
    config = PromptGoalPlannerConfig(
        protected_paths=("pkg/agent_supervisor_self_improvement.todo.md",)
    )
    with pytest.raises(PromptGoalProposalError) as denied:
        parse_prompt_goal_graph(
            json.dumps(protected, sort_keys=True, separators=(",", ":")),
            request,
            scan,
            config=config,
        )
    assert denied.value.reason_code == "protected_path"


def test_output_byte_budget_is_checked_before_json_decode() -> None:
    request = _request(budget=_budget(max_provider_tokens=32))
    scan = _scan(request)
    response = _encoded_proposal(scan)

    with pytest.raises(PromptGoalProposalError) as raised:
        parse_prompt_goal_graph(response, request, scan)
    assert raised.value.reason_code == "response_over_budget"


def test_request_over_budget_skips_provider_and_uses_fallback_receipts() -> None:
    request = _request()
    scan = _scan(request)
    calls: list[str] = []

    result = generate_prompt_goal_graph(
        request,
        scan,
        router=lambda prompt: calls.append(prompt) or "",
        config=PromptGoalPlannerConfig(max_provider_request_bytes=512),
    )

    assert calls == []
    assert result.used_fallback
    assert result.receipt.provider.attempted is False
    assert result.receipt.provider.status == "over_budget"
    assert result.receipt.provider.request_bytes == 0
    assert result.receipt.provider.request_sha256.startswith("sha256:")
    assert result.receipt.parse.status == "not_attempted"
    assert result.receipt.fallback.reason_code == "request_over_budget"


def test_request_specific_graph_depth_and_capabilities_are_enforced() -> None:
    request = _request(budget=_budget(max_graph_depth=1))
    scan = _scan(request)
    payload = _proposal(scan)
    first = payload["tasks"][0]
    payload["tasks"] = [
        first,
        {
            **first,
            "task_key": "task:dependent",
            "dependency_task_keys": ["task:retry-planner"],
        },
    ]

    with pytest.raises(PromptGoalProposalError) as deep:
        parse_prompt_goal_graph(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            request,
            scan,
        )
    assert deep.value.reason_code == "graph_over_budget"

    with pytest.raises(PromptGoalProposalError) as unavailable:
        parse_prompt_goal_graph(
            _encoded_proposal(scan),
            request,
            scan,
            capabilities={"resource_classes": ["provider-llm"]},
        )
    assert unavailable.value.reason_code == "unsupported_resource"


def test_policy_disabled_path_is_provider_free_and_deterministic() -> None:
    request = _request(allow_model=False)
    scan = _scan(request)
    calls: list[str] = []

    first = generate_prompt_goal_graph(
        request, scan, router=lambda prompt: calls.append(prompt) or ""
    )
    second = generate_prompt_goal_graph(
        request, scan, router=lambda prompt: calls.append(prompt) or ""
    )

    assert calls == []
    assert first.used_fallback
    assert first.receipt.provider.attempted is False
    assert first.receipt.provider.status == "disabled"
    assert first.receipt.parse.status == "not_attempted"
    assert first.receipt.fallback.reason_code == "policy_disabled"
    assert first.graph.to_json() == second.graph.to_json()


@pytest.mark.parametrize(
    ("router", "status"),
    [
        (lambda _prompt: "not-json", "malformed"),
        (
            lambda _prompt: (_ for _ in ()).throw(
                ModuleNotFoundError("llm_router unavailable")
            ),
            "unavailable",
        ),
        (
            lambda _prompt: (_ for _ in ()).throw(
                RuntimeError("llm_router child timed out")
            ),
            "timeout",
        ),
    ],
)
def test_malformed_unavailable_and_timeout_paths_use_deterministic_fallback(
    router, status: str
) -> None:
    request = _request()
    scan = _scan(request)

    result = generate_prompt_goal_graph(request, scan, router=router)

    assert result.used_fallback
    assert result.receipt.provider.status == status
    assert result.receipt.fallback.status == "succeeded"
    assert result.receipt.fallback.plan_root_cid == result.graph.plan_root_cid
    assert result.graph.root_goal.goal_key == "goal:root"
    assert result.graph.tasks[0].validations
    assert result.graph.tasks[0].acceptance


def test_fallback_is_schema_equivalent_and_bounded_under_tenfold_irrelevant_growth() -> None:
    request = _request(allow_model=False)
    relevant = _evidence()
    small_irrelevant = tuple(
        _evidence(
            f"scan:irrelevant-{index:03d}",
            summary="Unrelated generated fixture.",
            paths=(f"pkg/fixtures/unrelated_{index:03d}.txt",),
        )
        for index in range(5)
    )
    large_irrelevant = tuple(
        _evidence(
            f"scan:irrelevant-{index:03d}",
            summary="Unrelated generated fixture.",
            paths=(f"pkg/fixtures/unrelated_{index:03d}.txt",),
        )
        for index in range(50)
    )
    small = _scan(
        request,
        evidence=(relevant, *small_irrelevant),
        counts={"files": 6, "scan_bytes": 6_000, "symbols": 8},
    )
    large = _scan(
        request,
        evidence=(relevant, *large_irrelevant),
        counts={"files": 51, "scan_bytes": 60_000, "symbols": 8},
    )
    config = PromptGoalPlannerConfig(
        max_provider_request_bytes=32 * 1024,
        max_selected_evidence=4,
    )

    small_prompt = build_prompt_goal_provider_request(
        request, small, config=config
    )
    large_prompt = build_prompt_goal_provider_request(
        request, large, config=config
    )
    small_graph = deterministic_prompt_goal_graph(
        request, small, config=config
    )
    large_graph = deterministic_prompt_goal_graph(
        request, large, config=config
    )

    assert len(small_prompt.encode()) <= config.max_provider_request_bytes
    assert len(large_prompt.encode()) <= config.max_provider_request_bytes
    assert len(large_prompt) <= len(small_prompt) + 256
    assert len(json.loads(large_prompt)["evidence_handles"]) <= 4
    assert (
        json.loads(small_prompt)["evidence_handles"]
        == json.loads(large_prompt)["evidence_handles"]
    )
    assert PromptGoalGraph.from_json(small_graph.to_json())
    assert PromptGoalGraph.from_json(large_graph.to_json())
    assert small_graph.root_goal.objective == large_graph.root_goal.objective
    assert small_graph.tasks[0].objective == large_graph.tasks[0].objective
    assert small_graph.tasks[0].predicted_files == (
        large_graph.tasks[0].predicted_files
    )


def test_constraint_command_allowlist_is_exact_when_pinned() -> None:
    request = _request()
    scan = _scan(request)
    constraints = {
        "allowed_paths": ["pkg"],
        "validation_commands": [
            ["python", "-m", "pytest", "pkg/tests", "-q"]
        ],
    }
    accepted = parse_prompt_goal_graph(
        _encoded_proposal(scan),
        request,
        scan,
        constraint_summaries=constraints,
    )
    assert accepted.tasks

    payload = _proposal(scan)
    payload["tasks"][0]["validations"][0]["argv"] = [
        "python",
        "-m",
        "pytest",
        "pkg/other_tests",
        "-q",
    ]
    with pytest.raises(PromptGoalProposalError) as raised:
        parse_prompt_goal_graph(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            request,
            scan,
            constraint_summaries=constraints,
        )
    assert raised.value.reason_code == "forbidden_instruction"


def test_explicit_unavailable_capability_skips_provider() -> None:
    request = _request()
    scan = _scan(request)
    calls: list[str] = []

    result = generate_prompt_goal_graph(
        request,
        scan,
        capabilities={"available": False},
        router=lambda prompt: calls.append(prompt) or _encoded_proposal(scan),
    )

    assert calls == []
    assert result.used_fallback
    assert result.receipt.provider.status == "unavailable"
    assert result.receipt.provider.reason_code == "capability_unavailable"
