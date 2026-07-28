from __future__ import annotations

import json

import pytest

import ipfs_accelerate_py.agent_supervisor.objectives.objective_tracker as objective_tracker_module
from ipfs_accelerate_py.agent_supervisor.objectives.goal_coverage import (
    UNMAPPED_GOAL_ID,
    goal_coverage_work_seeds,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    ObjectiveGenerationLimits,
    ObjectiveGoalMaterializationPolicy,
    ObjectiveWorkKind,
    ObjectiveWorkProposal,
    materialize_bounded_objective_work,
    objective_goal_content_id,
    preview_objective_goal_materialization,
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon import (
    _objective_generation_board_state,
    active_objective_generation_work,
    blocked_review_objective_generation_families,
    completion_evidence_records_from_gate_records,
    load_objective_admission_records,
    load_goal_completion_gate_records,
    load_objective_generation_work,
    materialize_admitted_objective_work,
    materialize_objective_generation_cycle,
    objective_generation_proposals,
    objective_generation_task_findings,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_tracker import (
    ObjectiveMaterializationTransactionState,
    commit_objective_goal_materialization,
    objective_materialization_tree_identity,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_evaluator import (
    AnalysisProposal,
    ObjectiveWorkEvaluationPolicy,
)
from ipfs_accelerate_py.agent_supervisor.planning.task_proposal_router import (
    analysis_proposals_to_objective_work,
)


def _proposal(**overrides: object) -> ObjectiveWorkProposal:
    values: dict[str, object] = {
        "kind": "task",
        "title": "Add API proof",
        "parent_goal_id": "G1",
        "parent_objective_terms": ("API is available",),
        "expected_evidence_delta": ("API validation receipt",),
        "dependencies": ("bootstrap",),
        "predicted_files": ("src/api.py",),
        "predicted_symbols": ("ApiClient",),
        "validation_commands": ("pytest -q",),
        "confidence": 0.8,
        "estimated_cost": 2.0,
        "novelty": 0.9,
        "depth": 1,
        "estimated_tokens": 100,
    }
    values.update(overrides)
    return ObjectiveWorkProposal(**values)  # type: ignore[arg-type]


def test_coverage_and_contradictions_create_complete_hierarchical_work() -> None:
    coverage = {
        "criteria": [
            {
                "goal_id": "G1",
                "criterion_id": "criterion-1",
                "criterion": "The API has current validation proof",
                "status": "uncovered",
                "missing_surfaces": ["predicted_files", "validation_receipts"],
            }
        ],
        "edges": [
            {
                "criterion_id": "criterion-1",
                "surface": "predicted_file",
                "value": "src/api.py",
            },
            {
                "criterion_id": "criterion-1",
                "surface": "ast_symbol",
                "value": "ApiClient",
            },
            {
                "criterion_id": "criterion-1",
                "surface": "validation_command",
                "value": "pytest tests/test_api.py -q",
            },
        ],
        "finding_assignments": [
            {
                "finding_id": "finding-1",
                "goal_id": UNMAPPED_GOAL_ID,
                "confidence": 0.7,
                "finding": {
                    "title": "Cover the unsupported event surface",
                    "missing_evidence": ["event delivery receipt"],
                    "predicted_files": ["src/events.py"],
                    "predicted_symbols": ["EventSink"],
                    "validation": ["pytest tests/test_events.py -q"],
                },
            }
        ],
    }
    goals = [
        {
            "goal_id": "G1",
            "fields": {
                "acceptance": "The API has current validation proof",
                "outputs": "src/api.py",
                "validation": "pytest tests/test_api.py -q",
                "graph_depth": "0",
            },
        }
    ]
    contradictions = [
        {
            "goal_id": "G1",
            "kind": "failed_validation",
            "summary": "The API validation regressed.",
            "impacted_criteria": ["The API has current validation proof"],
            "invalidated_evidence": ["receipt-old"],
            "source_receipt_id": "receipt-failed",
        }
    ]

    first = goal_coverage_work_seeds(
        coverage,
        goals=goals,
        contradictions=contradictions,
    )
    second = goal_coverage_work_seeds(
        coverage,
        goals=goals,
        contradictions=contradictions,
    )

    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert {item.kind for item in first} == {
        ObjectiveWorkKind.GOAL,
        ObjectiveWorkKind.SUBGOAL,
        ObjectiveWorkKind.TASK,
    }
    surface_tasks = [item for item in first if item.parent_goal_id.startswith("objective-work:")]
    assert surface_tasks
    assert all(item.dependencies == (item.parent_goal_id,) for item in surface_tasks)
    for item in first:
        payload = item.to_dict()
        assert payload["canonical_id"].startswith("objective-work:")
        assert payload["semantic_key"].startswith("objective-work/v1/")
        assert "parent_objective_terms" in payload
        assert payload["expected_evidence_delta"]
        assert "dependencies" in payload
        assert "predicted_files" in payload
        assert "predicted_symbols" in payload
        assert "validation" in payload
        assert "confidence" in payload
        assert "cost" in payload
        assert "novelty" in payload


def test_canonical_and_semantic_duplicates_are_suppressed_across_cycles() -> None:
    original = _proposal()
    exact = ObjectiveWorkProposal.from_dict(original.to_dict())
    semantic = _proposal(title="Prove the API with equivalent evidence")

    assert semantic.semantic_key == original.semantic_key
    assert semantic.canonical_id != original.canonical_id

    result = materialize_bounded_objective_work(
        [exact, semantic],
        existing_work=[original],
        limits=ObjectiveGenerationLimits(semantic_similarity_threshold=0.5),
    )

    assert not result.accepted
    assert {item.reason for item in result.rejected} == {
        "canonical_duplicate",
        "semantic_duplicate",
    }


@pytest.mark.parametrize(
    ("proposal_overrides", "limit_overrides", "current_open", "reason"),
    [
        ({"depth": 3}, {"max_depth": 2}, 0, "depth_limit"),
        ({"retry_count": 2}, {"max_retries": 1}, 0, "retry_limit"),
        ({"estimated_tokens": 101}, {"token_budget": 100}, 0, "token_budget"),
        ({}, {"max_open_work": 1}, 1, "open_work_limit"),
        ({}, {"max_breadth_per_parent": 0}, 0, "breadth_limit"),
        ({}, {"max_new_work": 0}, 0, "cycle_limit"),
    ],
)
def test_each_finite_generation_limit_is_fail_closed(
    proposal_overrides: dict[str, object],
    limit_overrides: dict[str, object],
    current_open: int,
    reason: str,
) -> None:
    result = materialize_bounded_objective_work(
        [_proposal(**proposal_overrides)],
        current_open_work=current_open,
        limits=ObjectiveGenerationLimits(**limit_overrides),  # type: ignore[arg-type]
    )
    assert not result.accepted
    assert [item.reason for item in result.rejected] == [reason]
    assert result.exhausted


def test_persisted_work_identity_is_revalidated() -> None:
    payload = _proposal().to_dict()
    payload["semantic_key"] = "objective-work/v1/tampered"
    with pytest.raises(ValueError, match="semantic_key"):
        ObjectiveWorkProposal.from_dict(payload)


def test_llm_router_proposals_preserve_complete_work_metadata() -> None:
    proposal = AnalysisProposal.from_dict(
        {
            "branch": {
                "branch_id": "router-plan",
                "summary": "Implement the uncovered API evidence.",
                "predicted_files": ["src/api.py"],
                "predicted_symbols": ["ApiClient"],
                "dependencies": ["REF-205"],
                "validation_commands": ["pytest tests/test_api.py -q"],
                "validation_proof": ["API receipt is current and provenance-backed"],
                "estimated_cost": 2.5,
                "risk": 0.1,
                "expected_objective_delta": 0.8,
                "source": "llm_router",
            },
            "confidence": 0.9,
            "novelty": 0.85,
            "objective_terms": ["API evidence"],
        }
    )

    (work,) = analysis_proposals_to_objective_work(
        [proposal],
        parent_goal_id="G1",
        depth=2,
        estimated_tokens=512,
        retry_count=1,
    )

    assert work.source == "llm_router"
    assert work.parent_objective_terms == ("API evidence",)
    assert work.expected_evidence_delta == (
        "API receipt is current and provenance-backed",
    )
    assert work.dependencies == ("REF-205",)
    assert work.predicted_files == ("src/api.py",)
    assert work.predicted_symbols == ("ApiClient",)
    assert work.validation_commands == ("pytest tests/test_api.py -q",)
    assert work.confidence == 0.9
    assert work.estimated_cost == 2.5
    assert work.novelty == 0.85
    assert work.estimated_tokens == 512
    assert work.retry_count == 1


def test_daemon_generation_ledger_prevents_cross_cycle_regeneration(tmp_path) -> None:
    artifact = tmp_path / "state" / "objective_generation.json"
    policy = ObjectiveWorkEvaluationPolicy(
        min_confidence=0.0,
        min_novelty=0.0,
        max_proposals=3,
        max_total_cost=10.0,
        max_open_work=10,
        current_open_work=0,
        remaining_token_budget=1000,
    )

    first, first_artifact = materialize_objective_generation_cycle(
        [_proposal()],
        artifact_path=artifact,
        evaluation_policy=policy,
        objective_terms=["API is available"],
    )
    second, second_artifact = materialize_objective_generation_cycle(
        [_proposal(title="Cosmetically renamed API proof")],
        artifact_path=artifact,
        evaluation_policy=policy,
        objective_terms=["API is available"],
    )

    assert len(first.accepted) == 1
    assert not second.accepted
    assert first_artifact["cycle_count"] == 1
    assert second_artifact["cycle_count"] == 2
    assert second_artifact["generated_work_count"] == 1
    assert second_artifact["last_evaluation"]["rejected"][0]["reason"] in {
        "duplicate_canonical_identity",
        "duplicate_semantic_work",
    }
    assert load_objective_generation_work(artifact) == tuple(
        second_artifact["generated_work"]
    )
    assert json.loads(artifact.read_text(encoding="utf-8"))["cycle_count"] == 2


def _documentation_goal_heap(*, outputs: str = "Mcp-Plus-Plus/docs") -> str:
    return f"""# Objective Heap

## DCS-G030 MCP++ documentation

- Status: provisionally_complete
- Parent: DCS-G000
- Goal: Keep MCP++ documentation aligned with executable conformance.
- Acceptance: Runtime claims pass conformance; status boundaries are explicit; the repository receipt passes.
- Evidence: Mcp-Plus-Plus/docs/DOCUMENTATION_INDEX.md, Mcp-Plus-Plus/test-results/documentation-current-state.json
- Outputs: {outputs}
- Validation: python Mcp-Plus-Plus/scripts/validate_documentation_current_state.py
"""


def _documentation_completion_gate(*, receipt_sha256: str = "receipt-refresh-a"):
    binding = {
        "repository_id": "repo",
        "tree_id": "tree-refresh-only",
        "objective_revision": "objective-v1",
        "analyzer_version": "documentation-adapter-v2",
        "configuration_revision": "documentation-policy-v3",
    }
    criteria = [
        {
            "criterion": "Every normative MCP++ claim passes executable conformance.",
            "status": "unverified",
            "verified": False,
            "reason_codes": [
                "semantic_verifier_missing",
                "missing_producer_channel:mcplusplus-runtime-conformance-verifier",
            ],
            "required_producer_channel": "mcplusplus-runtime-conformance-verifier",
            "implementation_paths": [
                "Mcp-Plus-Plus/docs/protocol/runtime.md",
            ],
        },
        {
            "criterion": "Experimental and implemented behavior have explicit status boundaries.",
            "status": "unverified",
            "verified": False,
            "reason_codes": [
                "semantic_verifier_missing",
                "missing_producer_channel:mcplusplus-status-boundary-verifier",
            ],
            "required_producer_channel": "mcplusplus-status-boundary-verifier",
            "affected_document_paths": [
                "Mcp-Plus-Plus/docs/protocol/status-boundaries.md",
            ],
        },
        {
            "criterion": "The repository documentation validator passes.",
            "status": "unverified",
            "verified": False,
            "reason_codes": ["receipt_failed", "validation_command_failed"],
            "required_producer_channel": "repository-validator:DCS-G030",
            "analyzer_implementation_paths": [
                "Mcp-Plus-Plus/scripts/validate_documentation_current_state.py",
            ],
        },
    ]
    return {
        "binding": binding,
        "reason_codes": [
            "receipt_failed",
            "semantic_verifier_missing",
            "independent_evidence_quorum_unsatisfied",
        ],
        "missing_producer_channels": [
            "mcplusplus-runtime-conformance-verifier",
            "mcplusplus-status-boundary-verifier",
            "repository-validator:DCS-G030",
        ],
        "coverage": {
            "binding": binding,
            "verified": False,
            "criteria": criteria,
        },
        "rejected_receipts": [
            {
                "path": "Mcp-Plus-Plus/test-results/documentation-current-state.json",
                "sha256": receipt_sha256,
                "status": "failed",
                "reason_codes": ["receipt_failed", "validation_command_failed"],
                "errors": [
                    "runtime conformance command typescript-vectors did not pass: exit code 1",
                    "runtime conformance command go-vectors did not pass: exit code 1",
                ],
                "validation_returncode": 1,
                "validation_command": [
                    "python",
                    "Mcp-Plus-Plus/scripts/validate_documentation_current_state.py",
                ],
            }
        ],
        # Rejected receipts remain diagnostics. They are deliberately not
        # projected into completion_evidence_records.
        "completion_evidence_records": [],
    }


def test_documentation_gate_emits_focused_direct_channel_tasks(tmp_path) -> None:
    objective = tmp_path / "objective.md"
    objective.write_text(_documentation_goal_heap(), encoding="utf-8")
    gate = _documentation_completion_gate()
    proposals = objective_generation_proposals(
        objective_path=objective,
        completion_gate_records={"DCS-G030": gate},
        completion_decisions={
            "DCS-G030": {
                "verified": False,
                "actionable_reasons": ["Produce completion evidence."],
            }
        },
    )

    typed = [item for item in proposals if item.source == "completion_gate_gap"]
    assert len(typed) == 3
    assert {item.parent_goal_id for item in typed} == {"DCS-G030"}
    assert all(item.kind is ObjectiveWorkKind.TASK for item in typed)
    assert all(item.family_key.startswith("objective-family/v1/") for item in typed)
    assert all(item.instance_key.startswith("objective-instance/v1/") for item in typed)
    assert not [item for item in proposals if item.source == "completion_gate"]
    titles = "\n".join(item.title for item in typed)
    assert "mcplusplus-runtime-conformance-verifier" in titles
    assert "mcplusplus-status-boundary-verifier" in titles
    assert "repository-validator:DCS-G030" in titles
    flattened_diagnostics = "\n".join(
        value
        for item in typed
        for value in (*item.expected_evidence_delta, item.rationale)
    )
    assert "typescript-vectors" in flattened_diagnostics
    assert "go-vectors" in flattened_diagnostics
    assert gate["completion_evidence_records"] == []
    assert not completion_evidence_records_from_gate_records(
        {"DCS-G030": gate}
    ).get("DCS-G030")


def test_documentation_gap_prefers_scoped_row_validation_over_receipt_and_goal(
    tmp_path,
) -> None:
    objective = tmp_path / "objective.md"
    objective.write_text(_documentation_goal_heap(), encoding="utf-8")
    validator = tmp_path / "scripts/probe_runtime.py"
    validator.parent.mkdir(parents=True)
    validator.write_text("# scoped probe\n", encoding="utf-8")
    gate = _documentation_completion_gate()
    runtime_row = gate["coverage"]["criteria"][0]
    runtime_row["implementation_paths"] = ["scripts/probe_runtime.py"]
    runtime_row["validation_commands"] = [
        "python scripts/probe_runtime.py --check-channel runtime"
    ]
    gate["rejected_receipts"][0].update(
        {
            "producer_channel": "mcplusplus-runtime-conformance-verifier",
            "validation_command": ["python", "scripts/broad_validator.py", "--write"],
        }
    )

    proposals = objective_generation_proposals(
        objective_path=objective,
        repo_root=tmp_path,
        completion_gate_records={"DCS-G030": gate},
        default_validation=("git diff --check",),
    )
    runtime = next(
        item
        for item in proposals
        if "mcplusplus-runtime-conformance-verifier" in item.predicted_symbols
    )

    assert set(runtime.validation_commands) == {
        "python scripts/probe_runtime.py --check-channel runtime",
        "git diff --check",
    }
    assert not any("--write" in command for command in runtime.validation_commands)
    assert not any(
        "validate_documentation_current_state.py" in command
        for command in runtime.validation_commands
    )
    assert runtime.predicted_files == ("scripts/probe_runtime.py",)


def test_documentation_gap_prefers_explicit_files_and_alignment_diagnostics(
    tmp_path,
) -> None:
    objective = tmp_path / "objective.md"
    objective.write_text(_documentation_goal_heap(), encoding="utf-8")
    gate = _documentation_completion_gate()
    runtime_row = gate["coverage"]["criteria"][0]
    runtime_row.update(
        {
            "affected_document_paths": [
                "Mcp-Plus-Plus/docs/protocol/runtime.md",
            ],
            "evidence_paths": [
                "Mcp-Plus-Plus/test-results/runtime-conformance.json",
            ],
            "probe_outcome": {
                "status": "failed",
                "probe": "typescript-vectors",
                "generated_at": "refresh-a",
            },
            "documentation_alignment": {
                "claim": "TypeScript runtime conformance",
                "state": "drifted",
            },
            "debt_path": "Mcp-Plus-Plus/docs/debt/runtime.md",
        }
    )
    gate["rejected_receipts"][0].update(
        {
            "affected_document_paths": [
                "Mcp-Plus-Plus/docs/DOCUMENTATION_INDEX.md",
            ],
            "evidence_paths": [
                "Mcp-Plus-Plus/test-results/documentation-current-state.json",
            ],
            "probe_outcome": "validator failed",
        }
    )

    first = objective_generation_proposals(
        objective_path=objective,
        completion_gate_records={"DCS-G030": gate},
    )
    runtime = next(
        item
        for item in first
        if "mcplusplus-runtime-conformance-verifier"
        in item.predicted_symbols
    )
    repository = next(
        item
        for item in first
        if "repository-validator:DCS-G030" in item.predicted_symbols
    )

    assert runtime.predicted_files == (
        "Mcp-Plus-Plus/docs/protocol/runtime.md",
    )
    assert set(repository.predicted_files) == {
        "Mcp-Plus-Plus/scripts/validate_documentation_current_state.py",
        "Mcp-Plus-Plus/docs/DOCUMENTATION_INDEX.md",
    }
    assert "Mcp-Plus-Plus/test-results/runtime-conformance.json" not in (
        runtime.predicted_files
    )
    assert all(
        "Mcp-Plus-Plus/docs" not in item.predicted_files
        for item in first
    )
    assert all(
        item.title.startswith("Align documentation evidence")
        for item in first
    )
    assert "Probe outcome" in runtime.rationale
    assert "Documentation alignment" in runtime.rationale
    assert "Documentation debt path" in runtime.rationale
    assert "Evidence path (read-only)" in runtime.rationale

    refresh_only = json.loads(json.dumps(gate))
    refresh_only["coverage"]["criteria"][0]["probe_outcome"][
        "generated_at"
    ] = "refresh-b"
    refreshed = objective_generation_proposals(
        objective_path=objective,
        completion_gate_records={"DCS-G030": refresh_only},
    )
    refreshed_runtime = next(
        item
        for item in refreshed
        if "mcplusplus-runtime-conformance-verifier"
        in item.predicted_symbols
    )
    assert refreshed_runtime.instance_key == runtime.instance_key

    meaningful_change = json.loads(json.dumps(refresh_only))
    meaningful_change["coverage"]["criteria"][0]["probe_outcome"][
        "status"
    ] = "passed-but-documentation-still-drifted"
    changed = objective_generation_proposals(
        objective_path=objective,
        completion_gate_records={"DCS-G030": meaningful_change},
    )
    changed_runtime = next(
        item
        for item in changed
        if "mcplusplus-runtime-conformance-verifier"
        in item.predicted_symbols
    )
    assert changed_runtime.family_key == runtime.family_key
    assert changed_runtime.instance_key != runtime.instance_key


def test_missing_channels_without_coverage_still_emit_typed_tasks(tmp_path) -> None:
    objective = tmp_path / "objective.md"
    objective.write_text(_documentation_goal_heap(), encoding="utf-8")
    gate = _documentation_completion_gate()
    gate.pop("coverage")

    proposals = objective_generation_proposals(
        objective_path=objective,
        completion_gate_records={"DCS-G030": gate},
    )

    assert len(proposals) == 3
    assert {item.source for item in proposals} == {
        "completion_gate_gap_manual_review"
    }
    assert all(not item.predicted_files for item in proposals)


def test_rejected_receipt_without_coverage_remains_an_actionable_gap(tmp_path) -> None:
    objective = tmp_path / "objective.md"
    objective.write_text(_documentation_goal_heap(), encoding="utf-8")
    gate = _documentation_completion_gate()
    gate.pop("coverage")
    gate.pop("missing_producer_channels")
    gate["rejected_receipts"][0]["producer_channel"] = (
        "repository-validator:DCS-G030"
    )

    proposals = objective_generation_proposals(
        objective_path=objective,
        completion_gate_records={"DCS-G030": gate},
    )

    assert len(proposals) == 1
    assert proposals[0].source == "completion_gate_gap_manual_review"
    assert not proposals[0].predicted_files
    assert "typescript-vectors" in proposals[0].rationale
    assert "Receipt/report path (read-only)" in proposals[0].rationale
    assert not completion_evidence_records_from_gate_records(
        {"DCS-G030": gate}
    ).get("DCS-G030")


def test_gate_coverage_without_channel_never_reuses_raw_goal_outputs(
    tmp_path,
) -> None:
    objective = tmp_path / "objective.md"
    objective.write_text(_documentation_goal_heap(), encoding="utf-8")
    gate = {
        "coverage": {
            "verified": False,
            "criteria": [
                {
                    "goal_id": "DCS-G030",
                    "criterion_id": "legacy-uncovered-criterion",
                    "criterion": "Current behavior has executable proof.",
                    "status": "uncovered",
                }
            ],
        }
    }

    proposals = objective_generation_proposals(
        objective_path=objective,
        repo_root=tmp_path,
        completion_gate_records={"DCS-G030": gate},
    )

    assert len(proposals) == 1
    assert proposals[0].source == "completion_gate_gap_manual_review"
    assert proposals[0].predicted_files == ()
    assert proposals[0].family_key
    assert proposals[0].instance_key
    assert "Mcp-Plus-Plus/docs" not in proposals[0].predicted_files
    assert not [item for item in proposals if item.source == "coverage_rule"]


def test_unverified_decision_without_typed_gate_uses_stable_manual_review_family(
    tmp_path,
) -> None:
    objective = tmp_path / "objective.md"
    objective.write_text(
        _documentation_goal_heap(outputs="Mcp-Plus-Plus/README.md"),
        encoding="utf-8",
    )
    decision = {
        "verified": False,
        "state": "provisionally_complete",
        "actionable_reasons": ["independent_evidence_quorum_unsatisfied"],
    }

    first = objective_generation_proposals(
        objective_path=objective,
        repo_root=tmp_path,
        completion_decisions={"DCS-G030": decision},
    )
    objective.write_text(
        _documentation_goal_heap(outputs="Mcp-Plus-Plus/docs"),
        encoding="utf-8",
    )
    second = objective_generation_proposals(
        objective_path=objective,
        repo_root=tmp_path,
        completion_decisions={"DCS-G030": decision},
    )

    assert len(first) == 1
    assert first[0].source == "completion_gate_gap_manual_review"
    assert first[0].predicted_files == ()
    assert first[0].family_key.startswith("objective-family/v1/")
    assert first[0].instance_key.startswith("objective-instance/v1/")
    assert second[0].family_key == first[0].family_key
    assert second[0].instance_key == first[0].instance_key
    assert not [item for item in (*first, *second) if item.source == "completion_gate"]


def test_unverified_decision_uses_only_explicit_safe_edit_targets(tmp_path) -> None:
    objective = tmp_path / "objective.md"
    objective.write_text(_documentation_goal_heap(), encoding="utf-8")

    proposals = objective_generation_proposals(
        objective_path=objective,
        repo_root=tmp_path,
        completion_decisions={
            "DCS-G030": {
                "verified": False,
                "actionable_reasons": ["validator_diagnostic_missing"],
                "implementation_paths": [
                    "Mcp-Plus-Plus/scripts/validate_documentation_current_state.py"
                ],
            }
        },
    )

    assert len(proposals) == 1
    assert proposals[0].source == "completion_gate_gap"
    assert proposals[0].predicted_files == (
        "Mcp-Plus-Plus/scripts/validate_documentation_current_state.py",
    )
    assert proposals[0].family_key
    assert proposals[0].instance_key


@pytest.mark.parametrize(
    "required_child_goal_ids",
    (
        "DCS-G030.1",
        ["DCS-G030.1", " DCS-G030.1 "],
        [""],
        [42],
    ),
)
def test_gate_loader_rejects_malformed_required_child_goal_ids(
    tmp_path,
    required_child_goal_ids,
) -> None:
    artifact = tmp_path / "completion-gate.json"
    artifact.write_text(
        json.dumps(
            {
                "goals": {
                    "DCS-G030": {
                        "required_child_goal_ids": required_child_goal_ids,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="required_child_goal_ids"):
        load_goal_completion_gate_records(artifact)


def test_gate_loader_normalizes_unique_required_child_goal_ids(tmp_path) -> None:
    artifact = tmp_path / "completion-gate.json"
    artifact.write_text(
        json.dumps(
            {
                "goals": {
                    "DCS-G030": {
                        "required_child_goal_ids": [
                            " DCS-G030.1 ",
                            "DCS-G030.2",
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    records = load_goal_completion_gate_records(artifact)

    assert records["DCS-G030"]["required_child_goal_ids"] == [
        "DCS-G030.1",
        "DCS-G030.2",
    ]


@pytest.mark.parametrize(
    "validation_commands",
    (
        {"command": "pytest -q"},
        ["pytest -q", 42],
        [""],
        ["x" * 513],
        ["pytest -q"] * 9,
    ),
)
def test_gate_loader_rejects_malformed_scoped_validation_commands(
    tmp_path,
    validation_commands,
) -> None:
    artifact = tmp_path / "completion-gate.json"
    artifact.write_text(
        json.dumps(
            {
                "goals": {
                    "DCS-G030": {
                        "coverage": {
                            "criteria": [
                                {
                                    "criterion": "Documentation is aligned.",
                                    "validation_commands": validation_commands,
                                }
                            ]
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="validation_commands"):
        load_goal_completion_gate_records(artifact)


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "/tmp/outside.md",
        "../outside.md",
        "Mcp-Plus-Plus/docs/../../outside.md",
        "C:/outside.md",
        "//server/share.md",
        "unsafe\x00name.md",
        "Mcp-Plus-Plus/docs/",
    ),
)
def test_gate_loader_rejects_nested_unsafe_edit_targets(
    tmp_path,
    unsafe_path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    artifact = repo / "completion-gate.json"
    artifact.write_text(
        json.dumps(
            {
                "goals": {
                    "DCS-G030": {
                        "coverage": {
                            "criteria": [
                                {
                                    "criterion": "Documentation is aligned.",
                                    "implementation_paths": [unsafe_path],
                                }
                            ]
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe or imprecise edit target"):
        load_goal_completion_gate_records(artifact, repo_root=repo)


def test_gate_loader_rejects_symlink_edit_target_escaping_repo(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "escape").symlink_to(outside, target_is_directory=True)
    artifact = repo / "completion-gate.json"
    artifact.write_text(
        json.dumps(
            {
                "goals": {
                    "DCS-G030": {
                        "rejected_receipts": [
                            {
                                "validator_source_path": "escape/validator.py",
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe or imprecise edit target"):
        load_goal_completion_gate_records(artifact, repo_root=repo)


def test_gate_loader_rejects_existing_directory_with_file_like_name(
    tmp_path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "docs.md").mkdir()
    artifact = repo / "completion-gate.json"
    artifact.write_text(
        json.dumps(
            {
                "goals": {
                    "DCS-G030": {
                        "implementation_paths": ["docs.md"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe or imprecise edit target"):
        load_goal_completion_gate_records(artifact, repo_root=repo)


def test_gate_loader_requires_repo_root_for_nested_edit_targets(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    artifact = repo / "completion-gate.json"
    artifact.write_text(
        json.dumps(
            {
                "goals": {
                    "DCS-G030": {
                        "coverage": {
                            "criteria": [
                                {
                                    "validator_source_path": (
                                        "Mcp-Plus-Plus/scripts/validator.py"
                                    ),
                                }
                            ]
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="repo_root is required"):
        load_goal_completion_gate_records(artifact)

    records = load_goal_completion_gate_records(artifact, repo_root=repo)
    assert records["DCS-G030"]["coverage"]["criteria"][0][
        "validator_source_path"
    ] == "Mcp-Plus-Plus/scripts/validator.py"


def test_in_memory_gate_rejects_symlink_edit_target_escaping_repo(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "escape").symlink_to(outside, target_is_directory=True)
    objective = repo / "objective.md"
    objective.write_text(_documentation_goal_heap(), encoding="utf-8")
    gate = _documentation_completion_gate()
    gate["coverage"]["criteria"][0]["implementation_paths"] = [
        "escape/runtime.md"
    ]

    proposals = objective_generation_proposals(
        objective_path=objective,
        repo_root=repo,
        completion_gate_records={"DCS-G030": gate},
    )
    runtime = next(
        item
        for item in proposals
        if "mcplusplus-runtime-conformance-verifier" in item.predicted_symbols
    )

    assert runtime.source == "completion_gate_gap_manual_review"
    assert runtime.predicted_files == ()


def test_documentation_gap_family_ignores_refresh_sha_and_mutable_goal_surfaces(
    tmp_path,
) -> None:
    objective = tmp_path / "objective.md"
    objective.write_text(_documentation_goal_heap(outputs="Mcp-Plus-Plus/docs"), encoding="utf-8")
    first = objective_generation_proposals(
        objective_path=objective,
        completion_gate_records={
            "DCS-G030": _documentation_completion_gate(
                receipt_sha256="receipt-generated-at-a"
            )
        },
    )
    objective.write_text(
        _documentation_goal_heap(
            outputs="Mcp-Plus-Plus/docs, Mcp-Plus-Plus/README.md, mutable-extra.md"
        ).replace("MCP++ documentation", "MCP++ documentation wording changed"),
        encoding="utf-8",
    )
    second = objective_generation_proposals(
        objective_path=objective,
        completion_gate_records={
            "DCS-G030": _documentation_completion_gate(
                receipt_sha256="receipt-generated-at-b"
            )
        },
    )

    first_typed = {item.family_key: item for item in first if item.family_key}
    second_typed = {item.family_key: item for item in second if item.family_key}
    assert set(first_typed) == set(second_typed)
    assert {
        key: item.instance_key for key, item in first_typed.items()
    } == {
        key: item.instance_key for key, item in second_typed.items()
    }


def test_active_typed_gap_bootstraps_missing_family_state_for_terminal_retry(
    tmp_path,
) -> None:
    objective = tmp_path / "objective.md"
    objective.write_text(_documentation_goal_heap(), encoding="utf-8")
    artifact = tmp_path / "state" / "objective-generation.json"
    proposals = objective_generation_proposals(
        objective_path=objective,
        completion_gate_records={"DCS-G030": _documentation_completion_gate()},
    )
    validator = next(
        item
        for item in proposals
        if "repository-validator:DCS-G030" in item.predicted_symbols
    )

    active, active_payload = materialize_objective_generation_cycle(
        [validator],
        artifact_path=artifact,
        limits=ObjectiveGenerationLimits(max_retries=1),
        active_family_keys=(validator.family_key,),
    )
    retry, retry_payload = materialize_objective_generation_cycle(
        [validator],
        artifact_path=artifact,
        limits=ObjectiveGenerationLimits(max_retries=1),
        active_family_keys=(),
        terminal_family_counts={validator.family_key: 1},
    )

    assert not active.accepted
    active_state = active_payload["gap_family_states"][validator.family_key]
    assert active_state["active"] is True
    assert active_state["attempt_count"] == 1
    assert active_state["completed_task_count"] == 0
    assert len(retry.accepted) == 1
    assert retry.accepted[0].source == "completion_gate_gap_retry"
    retry_state = retry_payload["gap_family_states"][validator.family_key]
    assert retry_state["attempt_count"] == 2
    assert retry_state["completed_task_count"] == 1


def test_typed_gap_lifecycle_dedupes_active_and_allows_reappearance(tmp_path) -> None:
    objective = tmp_path / "objective.md"
    objective.write_text(_documentation_goal_heap(), encoding="utf-8")
    artifact = tmp_path / "state" / "objective-generation.json"
    proposals = objective_generation_proposals(
        objective_path=objective,
        completion_gate_records={"DCS-G030": _documentation_completion_gate()},
    )

    first, first_artifact = materialize_objective_generation_cycle(
        proposals,
        artifact_path=artifact,
        limits=ObjectiveGenerationLimits(max_retries=1),
        current_open_work=0,
        active_family_keys=(),
        terminal_family_counts={},
    )
    families = tuple(item.family_key for item in first.accepted)
    second, second_artifact = materialize_objective_generation_cycle(
        proposals,
        artifact_path=artifact,
        limits=ObjectiveGenerationLimits(max_retries=1),
        current_open_work=0,
        active_family_keys=families,
        terminal_family_counts={},
    )
    retry, retry_artifact = materialize_objective_generation_cycle(
        proposals,
        artifact_path=artifact,
        limits=ObjectiveGenerationLimits(max_retries=1),
        current_open_work=0,
        active_family_keys=(),
        terminal_family_counts={family: 1 for family in families},
    )
    repeated_retry, _repeated_retry_artifact = (
        materialize_objective_generation_cycle(
            proposals,
            artifact_path=artifact,
            limits=ObjectiveGenerationLimits(max_retries=1),
            current_open_work=0,
            active_family_keys=(),
            terminal_family_counts={family: 1 for family in families},
        )
    )
    active_retry, _active_retry_artifact = (
        materialize_objective_generation_cycle(
            proposals,
            artifact_path=artifact,
            limits=ObjectiveGenerationLimits(max_retries=1),
            current_open_work=0,
            active_family_keys=families,
            terminal_family_counts={family: 1 for family in families},
        )
    )
    review, review_artifact = materialize_objective_generation_cycle(
        proposals,
        artifact_path=artifact,
        limits=ObjectiveGenerationLimits(max_retries=1),
        current_open_work=0,
        active_family_keys=(),
        terminal_family_counts={family: 2 for family in families},
    )
    repeated_review, _repeated_review_artifact = (
        materialize_objective_generation_cycle(
            proposals,
            artifact_path=artifact,
            limits=ObjectiveGenerationLimits(max_retries=1),
            current_open_work=0,
            active_family_keys=(),
            terminal_family_counts={family: 2 for family in families},
        )
    )
    blocked, blocked_artifact = materialize_objective_generation_cycle(
        proposals,
        artifact_path=artifact,
        limits=ObjectiveGenerationLimits(max_retries=1),
        current_open_work=0,
        active_family_keys=(),
        terminal_family_counts={family: 3 for family in families},
    )
    repeated_blocked, _repeated_blocked_artifact = (
        materialize_objective_generation_cycle(
            proposals,
            artifact_path=artifact,
            limits=ObjectiveGenerationLimits(max_retries=1),
            current_open_work=0,
            active_family_keys=(),
            terminal_family_counts={family: 3 for family in families},
        )
    )
    resolved, resolved_artifact = materialize_objective_generation_cycle(
        (),
        artifact_path=artifact,
        limits=ObjectiveGenerationLimits(max_retries=1),
        current_open_work=0,
        active_family_keys=(),
        terminal_family_counts={family: 3 for family in families},
        observed_gap_goal_ids=("DCS-G030",),
    )
    reappeared, reappeared_artifact = materialize_objective_generation_cycle(
        proposals,
        artifact_path=artifact,
        limits=ObjectiveGenerationLimits(max_retries=1),
        current_open_work=0,
        active_family_keys=(),
        terminal_family_counts={family: 3 for family in families},
    )

    assert len(first.accepted) == 3
    assert not second.accepted
    assert len(retry.accepted) == 3
    assert {item.source for item in retry.accepted} == {
        "completion_gate_gap_retry"
    }
    assert {item.retry_count for item in retry.accepted} == {1}
    assert not repeated_retry.accepted
    assert not active_retry.accepted
    assert len(review.accepted) == 3
    assert {item.source for item in review.accepted} == {
        "completion_gate_gap_review"
    }
    assert not repeated_review.accepted
    assert not blocked.accepted
    assert not repeated_blocked.accepted
    assert all(
        state["outcome"] == "retry" and state["attempt_count"] == 2
        for state in retry_artifact["gap_family_states"].values()
    )
    assert all(
        state["outcome"] == "review_required" and state["review_emitted"]
        for state in review_artifact["gap_family_states"].values()
    )
    assert all(
        state["outcome"] == "blocked_review"
        for state in blocked_artifact["gap_family_states"].values()
    )
    assert not resolved.accepted
    assert all(
        state["resolved"]
        for state in resolved_artifact["gap_family_states"].values()
    )
    assert len(reappeared.accepted) == 3
    assert all(
        state["occurrence"] == 2
        for state in reappeared_artifact["gap_family_states"].values()
    )
    assert first_artifact["generated_work_count"] == 3
    assert second_artifact["generated_work_count"] == 3


def test_blocked_gap_bypasses_retries_for_one_visible_review(tmp_path) -> None:
    objective = tmp_path / "objective.md"
    objective.write_text(_documentation_goal_heap(), encoding="utf-8")
    artifact = tmp_path / "state" / "objective-generation.json"
    proposals = objective_generation_proposals(
        objective_path=objective,
        completion_gate_records={"DCS-G030": _documentation_completion_gate()},
    )
    first, _first_payload = materialize_objective_generation_cycle(
        proposals,
        artifact_path=artifact,
        limits=ObjectiveGenerationLimits(max_retries=2),
        active_family_keys=(),
    )
    families = tuple(item.family_key for item in first.accepted)

    review, review_payload = materialize_objective_generation_cycle(
        proposals,
        artifact_path=artifact,
        limits=ObjectiveGenerationLimits(max_retries=2),
        active_family_keys=(),
        blocked_family_counts={family: 1 for family in families},
    )
    repeated, _repeated_payload = materialize_objective_generation_cycle(
        proposals,
        artifact_path=artifact,
        limits=ObjectiveGenerationLimits(max_retries=2),
        active_family_keys=(),
        blocked_family_counts={family: 1 for family in families},
    )
    findings = objective_generation_task_findings(
        review_payload["generated_work"],
        repo_root=tmp_path,
        objective_path=objective,
        generation_path=artifact,
        gap_family_states=review_payload["gap_family_states"],
    )
    blocked, blocked_payload = materialize_objective_generation_cycle(
        proposals,
        artifact_path=artifact,
        limits=ObjectiveGenerationLimits(max_retries=2),
        active_family_keys=(),
        blocked_family_counts={family: 2 for family in families},
    )

    assert len(review.accepted) == 3
    assert {item.source for item in review.accepted} == {
        "completion_gate_gap_review"
    }
    assert {item.retry_count for item in review.accepted} == {0}
    assert len(findings) == 3
    assert all("manual review" in item.gap_task for item in findings)
    assert not repeated.accepted
    assert not blocked.accepted
    assert all(
        state["attempt_count"] == 1
        and state["outcome"] == "blocked_review"
        for state in blocked_payload["gap_family_states"].values()
    )
    assert blocked_review_objective_generation_families(
        blocked_payload["gap_family_states"]
    ) == tuple(sorted(families))


def test_unresolved_diagnostic_changes_do_not_reset_family_retry_budget(
    tmp_path,
) -> None:
    objective = tmp_path / "objective.md"
    objective.write_text(_documentation_goal_heap(), encoding="utf-8")
    artifact = tmp_path / "state" / "objective-generation.json"
    first_proposals = objective_generation_proposals(
        objective_path=objective,
        completion_gate_records={"DCS-G030": _documentation_completion_gate()},
    )
    first, first_payload = materialize_objective_generation_cycle(
        first_proposals,
        artifact_path=artifact,
        limits=ObjectiveGenerationLimits(max_retries=1),
        active_family_keys=(),
    )
    validator = next(
        item
        for item in first.accepted
        if "repository-validator:DCS-G030" in item.predicted_symbols
    )

    changed_gate = _documentation_completion_gate()
    changed_gate["rejected_receipts"][0]["errors"][0] = (
        "typescript-vectors failed with diagnostic revision two"
    )
    changed_proposals = objective_generation_proposals(
        objective_path=objective,
        completion_gate_records={"DCS-G030": changed_gate},
    )
    retry, retry_payload = materialize_objective_generation_cycle(
        changed_proposals,
        artifact_path=artifact,
        limits=ObjectiveGenerationLimits(max_retries=1),
        active_family_keys=(),
        terminal_family_counts={validator.family_key: 1},
    )

    changed_again_gate = json.loads(json.dumps(changed_gate))
    changed_again_gate["rejected_receipts"][0]["errors"][0] = (
        "typescript-vectors failed with diagnostic revision three"
    )
    changed_again = objective_generation_proposals(
        objective_path=objective,
        completion_gate_records={"DCS-G030": changed_again_gate},
    )
    diagnostic_refresh, diagnostic_refresh_payload = (
        materialize_objective_generation_cycle(
            changed_again,
            artifact_path=artifact,
            limits=ObjectiveGenerationLimits(max_retries=1),
            active_family_keys=(),
            terminal_family_counts={validator.family_key: 1},
        )
    )
    review, review_payload = materialize_objective_generation_cycle(
        changed_again,
        artifact_path=artifact,
        limits=ObjectiveGenerationLimits(max_retries=1),
        active_family_keys=(),
        terminal_family_counts={validator.family_key: 2},
    )

    assert len(first.accepted) == 3
    assert len(retry.accepted) == 1
    assert retry.accepted[0].source == "completion_gate_gap_retry"
    assert not diagnostic_refresh.accepted
    assert len(review.accepted) == 1
    assert review.accepted[0].source == "completion_gate_gap_review"
    retry_state = retry_payload["gap_family_states"][validator.family_key]
    diagnostic_refresh_state = diagnostic_refresh_payload[
        "gap_family_states"
    ][validator.family_key]
    review_state = review_payload["gap_family_states"][validator.family_key]
    assert retry_state["attempt_count"] == 2
    assert retry_state["occurrence"] == 1
    assert diagnostic_refresh_state["attempt_count"] == 2
    assert diagnostic_refresh_state["review_emitted"] is False
    assert review_state["attempt_count"] == 2
    assert review_state["occurrence"] == 1
    assert review_state["review_emitted"] is True
    assert first_payload["gap_family_states"][validator.family_key][
        "attempt_count"
    ] == 1


def test_board_state_distinguishes_completed_and_blocked_families() -> None:
    board = """# Board

## DCS-001 Completed generated work

- Status: completed
- Merge key: objective-family/v1/completed

## DCS-002 Blocked generated work

- Status: blocked
- Merge key: objective-family/v1/blocked

## DCS-003 Active generated work

- Status: in_progress
- Merge key: objective-family/v1/active
"""

    active, completed, blocked = _objective_generation_board_state(
        board,
        task_prefix="DCS-",
    )

    assert active == {"objective-family/v1/active"}
    assert completed == {"objective-family/v1/completed": 1}
    assert blocked == {"objective-family/v1/blocked": 1}


def test_changed_failed_diagnostics_wait_for_terminal_task_before_retry(
    tmp_path,
) -> None:
    objective = tmp_path / "objective.md"
    objective.write_text(_documentation_goal_heap(), encoding="utf-8")
    artifact = tmp_path / "state" / "objective-generation.json"
    first_proposals = objective_generation_proposals(
        objective_path=objective,
        completion_gate_records={"DCS-G030": _documentation_completion_gate()},
    )
    materialize_objective_generation_cycle(
        first_proposals,
        artifact_path=artifact,
        active_family_keys=(),
    )
    changed_gate = _documentation_completion_gate(
        receipt_sha256="receipt-refresh-only-change"
    )
    changed_gate["rejected_receipts"][0]["errors"][0] = (
        "runtime conformance command typescript-vectors did not pass: exit code 2"
    )
    changed_proposals = objective_generation_proposals(
        objective_path=objective,
        completion_gate_records={"DCS-G030": changed_gate},
    )

    result, payload = materialize_objective_generation_cycle(
        changed_proposals,
        artifact_path=artifact,
        active_family_keys=(),
    )
    changed_validator = next(
        item
        for item in changed_proposals
        if "repository-validator:DCS-G030" in item.predicted_symbols
    )
    retry, retry_payload = materialize_objective_generation_cycle(
        changed_proposals,
        artifact_path=artifact,
        active_family_keys=(),
        terminal_family_counts={changed_validator.family_key: 1},
    )

    assert not result.accepted
    state = payload["gap_family_states"][changed_validator.family_key]
    assert state["attempt_count"] == 1
    assert state["completed_task_count"] == 0
    assert state["review_emitted"] is False
    assert state["latest_instance_key"] == changed_validator.instance_key
    assert len(retry.accepted) == 1
    assert retry.accepted[0].family_key == changed_validator.family_key
    assert retry.accepted[0].source == "completion_gate_gap_retry"
    assert "exit code 2" in retry.accepted[0].rationale
    assert retry_payload["gap_family_states"][changed_validator.family_key][
        "attempt_count"
    ] == 2


def test_completed_legacy_generic_work_does_not_suppress_typed_gap(tmp_path) -> None:
    objective = tmp_path / "objective.md"
    objective.write_text(_documentation_goal_heap(), encoding="utf-8")
    artifact = tmp_path / "state" / "objective-generation.json"
    generic = _proposal(
        title="Produce completion evidence for MCP++ documentation",
        parent_goal_id="DCS-G030",
        parent_objective_terms=("Mcp-Plus-Plus/docs/DOCUMENTATION_INDEX.md",),
        expected_evidence_delta=("Produce completion evidence.",),
        predicted_files=("Mcp-Plus-Plus/docs",),
        predicted_symbols=("Mcp-Plus-Plus/docs/DOCUMENTATION_INDEX.md",),
        validation_commands=("test -s Mcp-Plus-Plus/docs/DOCUMENTATION_INDEX.md",),
        source="completion_gate",
    )
    materialize_objective_generation_cycle([generic], artifact_path=artifact)
    typed = objective_generation_proposals(
        objective_path=objective,
        completion_gate_records={"DCS-G030": _documentation_completion_gate()},
    )

    result, _payload = materialize_objective_generation_cycle(
        typed,
        artifact_path=artifact,
        current_open_work=0,
        active_family_keys=(),
    )

    assert len(result.accepted) == 3
    assert {item.source for item in result.accepted} == {"completion_gate_gap"}


def test_only_active_board_generated_work_consumes_open_capacity(tmp_path) -> None:
    objective = tmp_path / "objective.md"
    objective.write_text(_documentation_goal_heap(), encoding="utf-8")
    proposals = objective_generation_proposals(
        objective_path=objective,
        completion_gate_records={"DCS-G030": _documentation_completion_gate()},
    )
    work = [item.to_dict() for item in proposals]
    active_family = proposals[0].family_key
    board = f"""# Board

## DCS-001 Active typed task

- Status: todo
- Merge key: {active_family}

## DCS-002 Completed typed task

- Status: completed
- Merge key: {proposals[1].family_key}
"""

    active = active_objective_generation_work(
        board,
        work,
        task_prefix="DCS-",
    )

    assert [item["family_key"] for item in active] == [active_family]


def test_task_fingerprint_uses_gap_occurrence_not_per_refresh_receipt(tmp_path) -> None:
    objective = tmp_path / "objective.md"
    objective.write_text(_documentation_goal_heap(), encoding="utf-8")
    generation = tmp_path / "state" / "objective-generation.json"
    proposals = objective_generation_proposals(
        objective_path=objective,
        completion_gate_records={"DCS-G030": _documentation_completion_gate()},
    )
    _result, payload = materialize_objective_generation_cycle(
        proposals,
        artifact_path=generation,
        active_family_keys=(),
    )
    first = objective_generation_task_findings(
        payload["generated_work"],
        repo_root=tmp_path,
        objective_path=objective,
        generation_path=generation,
        gap_family_states=payload["gap_family_states"],
    )

    materialize_objective_generation_cycle(
        (),
        artifact_path=generation,
        active_family_keys=(),
    )
    _result, payload = materialize_objective_generation_cycle(
        proposals,
        artifact_path=generation,
        active_family_keys=(),
    )
    second = objective_generation_task_findings(
        payload["generated_work"],
        repo_root=tmp_path,
        objective_path=objective,
        generation_path=generation,
        gap_family_states=payload["gap_family_states"],
        seen_fingerprints=[item.fingerprint for item in first],
    )

    assert len(first) == 3
    assert len(second) == 3
    assert {item.fingerprint for item in first}.isdisjoint(
        item.fingerprint for item in second
    )


def _objective_heap() -> str:
    return """# Objective Heap

## ROOT Frozen objective

- Status: active
- Goal: Deliver the root objective
- Evidence: root proof
- Graph depth: 0
"""


def _hierarchical_goal_work() -> tuple[ObjectiveWorkProposal, ObjectiveWorkProposal]:
    goal = _proposal(
        kind="goal",
        title="Establish API evidence goal",
        parent_goal_id="ROOT",
        parent_objective_terms=("Deliver the root objective",),
        expected_evidence_delta=("API evidence",),
        dependencies=("bootstrap",),
        predicted_files=("src/api.py",),
        predicted_symbols=("ApiClient",),
        validation_commands=("pytest tests/test_api.py -q",),
        depth=1,
        source_id="proposal:api-goal",
    )
    subgoal = _proposal(
        kind="subgoal",
        title="Verify API evidence",
        parent_goal_id=goal.canonical_id,
        parent_objective_terms=("API evidence",),
        expected_evidence_delta=("API validation receipt",),
        dependencies=(goal.canonical_id,),
        predicted_files=("tests/test_api.py",),
        predicted_symbols=("test_api",),
        validation_commands=("pytest tests/test_api.py -q",),
        depth=2,
        source_id="proposal:api-subgoal",
    )
    return goal, subgoal


def test_goal_materialization_is_preview_first_lossless_and_transactional(
    tmp_path,
) -> None:
    objective_path = tmp_path / "repo" / "objective-heap.md"
    objective_path.parent.mkdir()
    objective_path.write_text(_objective_heap(), encoding="utf-8")
    original = objective_path.read_bytes()
    goal, subgoal = _hierarchical_goal_work()

    preview = preview_objective_goal_materialization(
        objective_path.read_text(encoding="utf-8"),
        (subgoal, goal),  # provider order cannot change hierarchy
        policy=ObjectiveGoalMaterializationPolicy(root_goal_id="ROOT"),
    )

    assert preview.ready
    assert objective_path.read_bytes() == original
    assert preview.admitted_proposal_ids == (goal.canonical_id, subgoal.canonical_id)
    result = commit_objective_goal_materialization(
        repo_root=objective_path.parent,
        objective_path=objective_path,
        journal_path=tmp_path / "admission-journal.json",
        preview=preview,
    )

    assert result.state is ObjectiveMaterializationTransactionState.COMMITTED
    assert result.changed
    materialized = {
        item.goal_id: item
        for item in parse_goal_heap(objective_path.read_text(encoding="utf-8"))
    }
    assert materialized[goal.canonical_id].parent_goal_ids == ["ROOT"]
    child = materialized[subgoal.canonical_id]
    assert child.parent_goal_ids == [goal.canonical_id]
    assert child.dependencies == [goal.canonical_id]
    assert child.required_evidence == ["API validation receipt"]
    assert child.semantic_key == subgoal.semantic_key
    assert child.canonical_proposal_id == subgoal.canonical_id
    assert child.lifecycle_owner == "objective_daemon"
    assert child.status == "active"

    replay = commit_objective_goal_materialization(
        repo_root=objective_path.parent,
        objective_path=objective_path,
        journal_path=tmp_path / "admission-journal.json",
        preview=preview,
    )
    assert replay.committed and replay.resumed and not replay.changed
    assert objective_path.read_text(encoding="utf-8").count(
        f"## {subgoal.canonical_id} "
    ) == 1


def test_goal_materialization_binds_epoch_revision_and_exact_goal_mapping(
    tmp_path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    objective_path = repo_root / "objective-heap.md"
    objective_path.write_text(_objective_heap(), encoding="utf-8")
    journal_path = tmp_path / "epoch-objective-journal.json"
    goal, subgoal = _hierarchical_goal_work()
    preview = preview_objective_goal_materialization(
        _objective_heap(),
        (goal, subgoal),
        root_goal_id="ROOT",
    )
    goal_ids = tuple(item.goal.goal_id for item in preview.materialized)

    result = commit_objective_goal_materialization(
        repo_root=repo_root,
        objective_path=objective_path,
        journal_path=journal_path,
        preview=preview,
        epoch_id="refill-epoch:sha256:test",
        expected_objective_revision=preview.base_heap_content_id,
        expected_goal_ids=goal_ids,
    )

    assert result.committed
    assert result.epoch_id == "refill-epoch:sha256:test"
    assert result.mapped_goal_ids == goal_ids
    transaction = json.loads(journal_path.read_text(encoding="utf-8"))[
        "transactions"
    ][result.transaction_id]
    assert transaction["epoch_id"] == result.epoch_id
    assert tuple(transaction["mapped_goal_ids"]) == goal_ids

    # A committed exact replay validates the journal and heap but performs no
    # objective or journal write.
    monkeypatch.setattr(
        objective_tracker_module,
        "_atomic_write_json",
        lambda *_args, **_kwargs: pytest.fail("exact replay wrote its journal"),
    )
    monkeypatch.setattr(
        objective_tracker_module,
        "_atomic_rewrite",
        lambda *_args, **_kwargs: pytest.fail("exact replay wrote its heap"),
    )
    replay = commit_objective_goal_materialization(
        repo_root=repo_root,
        objective_path=objective_path,
        journal_path=journal_path,
        preview=preview,
        epoch_id=result.epoch_id,
        expected_objective_revision=preview.base_heap_content_id,
        expected_goal_ids=goal_ids,
    )
    assert replay.committed and replay.resumed and not replay.changed
    assert replay.mapped_goal_ids == goal_ids

    remap = commit_objective_goal_materialization(
        repo_root=repo_root,
        objective_path=objective_path,
        journal_path=journal_path,
        preview=preview,
        epoch_id="refill-epoch:sha256:different",
        expected_objective_revision=preview.base_heap_content_id,
        expected_goal_ids=goal_ids,
    )
    assert remap.state is ObjectiveMaterializationTransactionState.BLOCKED
    assert remap.reason_codes == ("stale_objective_heap",)


def test_goal_materialization_rejects_epoch_revision_or_mapping_conflict(
    tmp_path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    objective_path = repo_root / "objective-heap.md"
    objective_path.write_text(_objective_heap(), encoding="utf-8")
    goal, _subgoal = _hierarchical_goal_work()
    preview = preview_objective_goal_materialization(
        _objective_heap(),
        (goal,),
        root_goal_id="ROOT",
    )

    stale = commit_objective_goal_materialization(
        repo_root=repo_root,
        objective_path=objective_path,
        journal_path=tmp_path / "stale-epoch-journal.json",
        preview=preview,
        epoch_id="refill-epoch:stale",
        expected_objective_revision="objective:stale",
        expected_goal_ids=(goal.canonical_id,),
    )
    assert stale.state is ObjectiveMaterializationTransactionState.BLOCKED
    assert stale.reason_codes == ("objective_revision_conflict",)

    wrong_mapping = commit_objective_goal_materialization(
        repo_root=repo_root,
        objective_path=objective_path,
        journal_path=tmp_path / "mapping-epoch-journal.json",
        preview=preview,
        epoch_id="refill-epoch:mapping",
        expected_objective_revision=preview.base_heap_content_id,
        expected_goal_ids=("OTHER-GOAL",),
    )
    assert wrong_mapping.state is ObjectiveMaterializationTransactionState.BLOCKED
    assert wrong_mapping.reason_codes == ("goal_mapping_conflict",)
    assert objective_path.read_text(encoding="utf-8") == _objective_heap()


def test_goal_materialization_keeps_semantic_and_structural_breadth_bounds() -> None:
    goal, _subgoal = _hierarchical_goal_work()
    first = preview_objective_goal_materialization(
        _objective_heap(),
        (goal,),
        root_goal_id="ROOT",
    )
    assert first.ready

    semantic_replay = _proposal(
        kind="goal",
        title="Reworded API evidence goal",
        parent_goal_id=goal.parent_goal_id,
        parent_objective_terms=goal.parent_objective_terms,
        expected_evidence_delta=goal.expected_evidence_delta,
        dependencies=goal.dependencies,
        predicted_files=goal.predicted_files,
        predicted_symbols=goal.predicted_symbols,
        validation_commands=goal.validation_commands,
        depth=goal.depth,
        source_id="proposal:different-provider-identity",
    )
    assert semantic_replay.canonical_id != goal.canonical_id
    assert semantic_replay.semantic_key == goal.semantic_key
    duplicate = preview_objective_goal_materialization(
        first.candidate_text,
        (semantic_replay,),
        root_goal_id="ROOT",
    )
    assert not duplicate.ready
    assert duplicate.candidate_text == first.candidate_text
    assert [item.reason for item in duplicate.rejected] == ["semantic_duplicate"]

    terminal_child_heap = (
        _objective_heap()
        + "\n## TERMINAL Historical child\n\n"
        "- Status: verified\n"
        "- Goal: Historical bounded branch\n"
        "- Parents: ROOT\n"
        "- Graph depth: 1\n"
    )
    breadth = preview_objective_goal_materialization(
        terminal_child_heap,
        (goal,),
        policy=ObjectiveGoalMaterializationPolicy(
            root_goal_id="ROOT",
            limits=ObjectiveGenerationLimits(max_breadth_per_parent=1),
        ),
    )
    assert not breadth.ready
    assert breadth.candidate_text == terminal_child_heap
    assert [item.reason for item in breadth.rejected] == ["breadth_limit"]


def test_stale_heap_and_lease_conflict_fail_closed_and_remain_resumable(
    tmp_path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    objective_path = repo_root / "objective-heap.md"
    objective_path.write_text(_objective_heap(), encoding="utf-8")
    goal, _subgoal = _hierarchical_goal_work()
    preview = preview_objective_goal_materialization(
        objective_path.read_text(encoding="utf-8"),
        (goal,),
        root_goal_id="ROOT",
    )

    objective_path.write_text(
        _objective_heap() + "\n## HUMAN Concurrent goal\n\n- Status: active\n- Parent: ROOT\n",
        encoding="utf-8",
    )
    concurrent = objective_path.read_bytes()
    stale = commit_objective_goal_materialization(
        repo_root=repo_root,
        objective_path=objective_path,
        journal_path=tmp_path / "stale-journal.json",
        preview=preview,
    )
    assert stale.state is ObjectiveMaterializationTransactionState.BLOCKED
    assert stale.reason_codes == ("stale_objective_heap",)
    assert objective_path.read_bytes() == concurrent

    objective_path.write_text(_objective_heap(), encoding="utf-8")
    tree_journal = tmp_path / "tree-journal.json"
    tree_preview = preview_objective_goal_materialization(
        _objective_heap(),
        (goal,),
        root_goal_id="ROOT",
    )
    expected_tree = objective_materialization_tree_identity(
        repo_root,
        objective_path=objective_path,
        journal_path=tree_journal,
    ).tree_id
    (repo_root / "concurrent-source.py").write_text(
        "CONCURRENT = True\n", encoding="utf-8"
    )
    stale_tree = commit_objective_goal_materialization(
        repo_root=repo_root,
        objective_path=objective_path,
        journal_path=tree_journal,
        preview=tree_preview,
        expected_repository_tree_id=expected_tree,
    )
    assert stale_tree.state is ObjectiveMaterializationTransactionState.BLOCKED
    assert stale_tree.reason_codes == ("stale_repository_tree",)
    assert objective_path.read_text(encoding="utf-8") == _objective_heap()

    fresh = preview_objective_goal_materialization(
        objective_path.read_text(encoding="utf-8"),
        (goal,),
        root_goal_id="ROOT",
    )
    journal = tmp_path / "lease-journal.json"
    blocked = commit_objective_goal_materialization(
        repo_root=repo_root,
        objective_path=objective_path,
        journal_path=journal,
        preview=fresh,
        lease_guard=lambda token: False,
        expected_lease_token="fence-7",
    )
    assert blocked.state is ObjectiveMaterializationTransactionState.PREPARED
    assert blocked.resumable
    assert objective_path.read_text(encoding="utf-8") == _objective_heap()

    resumed = commit_objective_goal_materialization(
        repo_root=repo_root,
        objective_path=objective_path,
        journal_path=journal,
        preview=fresh,
        lease_guard=lambda token: {"fencing_token": token},
        expected_lease_token="fence-7",
    )
    assert resumed.committed and resumed.resumed
    assert goal.canonical_id in {
        item.goal_id
        for item in parse_goal_heap(objective_path.read_text(encoding="utf-8"))
    }


def test_partial_heap_write_remains_prepared_and_resumes_exactly(
    tmp_path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    objective_path = repo_root / "objective-heap.md"
    objective_path.write_text(_objective_heap(), encoding="utf-8")
    goal, subgoal = _hierarchical_goal_work()
    preview = preview_objective_goal_materialization(
        _objective_heap(),
        (goal, subgoal),
        root_goal_id="ROOT",
    )
    journal = tmp_path / "partial-journal.json"
    partial_text = (
        _objective_heap().rstrip()
        + "\n\n"
        + preview.materialized[0].rendered_block.strip()
        + "\n"
    )
    real_rewrite = objective_tracker_module._atomic_rewrite
    monkeypatch.setattr(
        objective_tracker_module,
        "_atomic_rewrite",
        lambda path, _text: path.write_text(partial_text, encoding="utf-8"),
    )

    partial = commit_objective_goal_materialization(
        repo_root=repo_root,
        objective_path=objective_path,
        journal_path=journal,
        preview=preview,
    )
    assert partial.state is ObjectiveMaterializationTransactionState.PREPARED
    assert partial.resumable
    assert partial.reason_codes[0] == "partial_write"
    assert goal.canonical_id in objective_path.read_text(encoding="utf-8")
    assert subgoal.canonical_id not in objective_path.read_text(encoding="utf-8")

    monkeypatch.setattr(objective_tracker_module, "_atomic_rewrite", real_rewrite)
    resumed = commit_objective_goal_materialization(
        repo_root=repo_root,
        objective_path=objective_path,
        journal_path=journal,
        preview=preview,
    )
    assert resumed.committed and resumed.resumed
    materialized_ids = [
        item.goal_id
        for item in parse_goal_heap(objective_path.read_text(encoding="utf-8"))
    ]
    assert materialized_ids.count(goal.canonical_id) == 1
    assert materialized_ids.count(subgoal.canonical_id) == 1


def test_shadow_never_mutates_and_assist_persists_review_only(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    objective_path = repo_root / "objective-heap.md"
    objective_path.write_text(_objective_heap(), encoding="utf-8")
    generation_path = tmp_path / "state" / "objective-generation.json"
    goal, subgoal = _hierarchical_goal_work()
    original = objective_path.read_bytes()

    shadow = materialize_admitted_objective_work(
        (goal, subgoal),
        repo_root=repo_root,
        objective_path=objective_path,
        generation_path=generation_path,
        mode="shadow",
        root_goal_id="ROOT",
    )
    assert shadow.status == "shadow"
    assert objective_path.read_bytes() == original
    assert not generation_path.exists()

    assist = materialize_admitted_objective_work(
        (goal, subgoal),
        repo_root=repo_root,
        objective_path=objective_path,
        generation_path=generation_path,
        mode="assist",
        root_goal_id="ROOT",
    )
    assert assist.status == "review_required"
    assert assist.review_persisted and assist.resumable
    assert objective_path.read_bytes() == original
    records = load_objective_admission_records(generation_path)
    assert set(records) == {goal.canonical_id, subgoal.canonical_id}
    assert {item["status"] for item in records.values()} == {"review_required"}
    assert all(item["preview"] for item in records.values())


def test_auto_safe_without_bound_authority_fails_closed_and_is_reviewable(
    tmp_path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    objective_path = repo_root / "objective-heap.md"
    objective_path.write_text(_objective_heap(), encoding="utf-8")
    generation_path = tmp_path / "state" / "objective-generation.json"
    goal, _subgoal = _hierarchical_goal_work()
    root = parse_goal_heap(_objective_heap())[0]

    result = materialize_admitted_objective_work(
        (goal,),
        repo_root=repo_root,
        objective_path=objective_path,
        generation_path=generation_path,
        mode="auto_safe",
        root_goal_id="ROOT",
        expected_root_content_id=objective_goal_content_id(root),
        new_assumption_ids=("assumption:hidden",),
        unsupported_semantics=("formula:invented",),
        hard_policy_gates={"scope": False},
    )

    assert result.status == "rejected"
    assert {
        "new_assumptions",
        "unsupported_semantics",
        "hard_policy_gate:scope",
        "unresolved_authoritative_receipts",
    }.issubset(result.reason_codes)
    assert objective_path.read_text(encoding="utf-8") == _objective_heap()
    record = load_objective_admission_records(generation_path)[goal.canonical_id]
    assert record["status"] == "rejected"
    assert record["lifecycle_owner"] == "objective_daemon"
