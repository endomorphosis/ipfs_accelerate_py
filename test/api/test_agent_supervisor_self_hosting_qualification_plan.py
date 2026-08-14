"""Fail-closed invariants for the operator-authored qualification plan."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor import goal_graph, parse_goal_heap
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    external_authority_goal_fence,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    parse_task_file,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = REPO_ROOT / "docs/architecture/SELF_HOSTING_QUALIFICATION_PLAN.md"
OBJECTIVE_PATH = REPO_ROOT / "docs/architecture/self_hosting_qualification.objectives.md"
ACTIVE_TODO_PATH = REPO_ROOT / "docs/architecture/self_hosting_qualification.todo.md"
V1_HISTORY_PATH = REPO_ROOT / "docs/architecture/self_hosting_qualification.v1_history.todo.md"
V2_HISTORY_PATH = REPO_ROOT / "docs/architecture/self_hosting_qualification.v2_history.todo.md"
V3_HISTORY_PATH = REPO_ROOT / "docs/architecture/self_hosting_qualification.v3_history.todo.md"
V4_HISTORY_PATH = REPO_ROOT / "docs/architecture/self_hosting_qualification.v4_history.todo.md"
V5_HISTORY_PATH = REPO_ROOT / "docs/architecture/self_hosting_qualification.v5_history.todo.md"
V6_HISTORY_PATH = REPO_ROOT / "docs/architecture/self_hosting_qualification.v6_history.todo.md"
V7_HISTORY_PATH = REPO_ROOT / "docs/architecture/self_hosting_qualification.v7_history.todo.md"
V8_HISTORY_PATH = REPO_ROOT / "docs/architecture/self_hosting_qualification.v8_history.todo.md"
V9_HISTORY_PATH = REPO_ROOT / "docs/architecture/self_hosting_qualification.v9_history.todo.md"
V10_HISTORY_PATH = REPO_ROOT / "docs/architecture/self_hosting_qualification.v10_history.todo.md"
V11_HISTORY_PATH = REPO_ROOT / "docs/architecture/self_hosting_qualification.v11_history.todo.md"


def _goals():
    return parse_goal_heap(OBJECTIVE_PATH.read_text(encoding="utf-8"))


def _descendant_closure(goals, seed_goal_ids: set[str]) -> set[str]:
    closure = set(seed_goal_ids)
    while True:
        expanded = closure | {
            goal.goal_id for goal in goals if closure.intersection(goal.parent_goal_ids)
        }
        if expanded == closure:
            return closure
        closure = expanded


def test_bootstrap_context_envelope_preserves_the_input_allowance() -> None:
    plan = PLAN_PATH.read_text(encoding="utf-8")
    normalized_plan = " ".join(plan.split())

    assert "model_context_window=49152" in plan
    assert "IPFS_ACCELERATE_AGENT_CODEX_CONTEXT_WINDOW=49152" in plan
    assert "--context-budget-tokens 24576" in plan
    assert "model_context_window=24576" not in plan
    assert "IPFS_ACCELERATE_AGENT_CODEX_CONTEXT_WINDOW=24576" not in plan
    assert 49_152 - 16_384 - 8_192 == 24_576
    assert "operator preflight/detective controls" in normalized_plan


def test_plan_has_one_closed_combined_goal_dag() -> None:
    goals = _goals()
    goal_ids = [goal.goal_id for goal in goals]
    goal_id_set = set(goal_ids)
    hierarchy = goal_graph(goals)

    assert len(goals) == 51
    assert len(goal_id_set) == len(goal_ids)
    assert hierarchy["roots"] == ["SHQ-G000"]
    assert all(
        edge["from"] in goal_id_set and edge["to"] in goal_id_set for edge in hierarchy["edges"]
    )

    # Parent and explicit dependency relations are both prerequisites.  Check
    # the union so a cycle split across the two native fields cannot hide from
    # the hierarchy-only projection.
    prerequisites = {
        goal.goal_id: set(goal.parent_goal_ids) | set(goal.dependencies) for goal in goals
    }
    assert all(
        prerequisite in goal_id_set for values in prerequisites.values() for prerequisite in values
    )
    assert all(goal_id not in values for goal_id, values in prerequisites.items())

    dependents: dict[str, set[str]] = {goal_id: set() for goal_id in goal_ids}
    remaining = {goal_id: set(values) for goal_id, values in prerequisites.items()}
    for goal_id, values in remaining.items():
        for prerequisite in values:
            dependents[prerequisite].add(goal_id)
    ready = sorted(goal_id for goal_id, values in remaining.items() if not values)
    visited: list[str] = []
    while ready:
        goal_id = ready.pop(0)
        visited.append(goal_id)
        for dependent in sorted(dependents[goal_id]):
            remaining[dependent].discard(goal_id)
            if not remaining[dependent] and dependent not in visited and dependent not in ready:
                ready.append(dependent)
        ready.sort()

    assert set(visited) == goal_id_set, {
        goal_id: sorted(values) for goal_id, values in remaining.items() if values
    }


def test_external_admission_and_preregistration_gates_fail_closed() -> None:
    goals = _goals()

    declared_external, blocked = external_authority_goal_fence(
        goals,
        trust_recorded_completion=False,
    )
    expected_external = {"SHQ-G010", "SHQ-G072"}
    expected_blocked = _descendant_closure(goals, expected_external)

    assert declared_external == expected_external
    assert blocked == expected_blocked
    assert external_authority_goal_fence(goals)[1] == expected_blocked
    assert "SHQ-G006" not in blocked
    assert "SHQ-G007" not in blocked
    assert _descendant_closure(goals, {"SHQ-G072"}) == {
        "SHQ-G072",
        "SHQ-G073",
        "SHQ-G074",
        "SHQ-G075",
        "SHQ-G076",
    }


def test_work_units_target_and_repository_ownership_are_explicit() -> None:
    source = OBJECTIVE_PATH.read_text(encoding="utf-8")
    normalized_source = " ".join(source.split())
    goals = _goals()
    by_id = {goal.goal_id: goal for goal in goals}
    work_goals = [goal for goal in goals if goal.required_evidence]
    external_work_goals = [goal for goal in work_goals if goal.requires_external_completion]
    local_work_goals = [goal for goal in work_goals if not goal.requires_external_completion]

    assert len(work_goals) == 43
    assert len(local_work_goals) == 41
    assert "endomorphosis/ipfs_kit_py:ipfs_kit_py/core/wal" in source
    assert "`core.operation_contracts` is a read-only dependency" in normalized_source
    assert {goal.goal_id for goal in external_work_goals} == {
        "SHQ-G010",
        "SHQ-G072",
    }
    assert all(goal.fields.get("gap_task") for goal in work_goals)
    assert all(goal.predicted_files for goal in work_goals)
    assert all(goal.validation_commands for goal in work_goals)

    assert "SelfHostingTaskCorpus" in by_id["SHQ-G032"].fields["interfaces"]
    assert "compare_task_outcomes" in by_id["SHQ-G036"].fields["interfaces"]
    assert "QualificationArtifactStore" in by_id["SHQ-G041"].fields["interfaces"]
    assert "create_qualification_manifest" in by_id["SHQ-G044"].fields["interfaces"]
    assert "QualificationRuntimePort@1" in by_id["SHQ-G031"].fields["interfaces"]
    assert "GovernedCodingAgentRuntime" in by_id["SHQ-G052"].fields["interfaces"]
    assert "SelfHostingQualificationHarness" in by_id["SHQ-G053"].fields["interfaces"]

    assert by_id["SHQ-G005A"].status == "blocked"
    assert by_id["SHQ-G005A"].dependencies == []
    assert by_id["SHQ-G005A"].fields["review_only"] == "true"
    assert by_id["SHQ-G005A"].fields["reviewed_predecessor_producing_task"] == "SHQ-023"
    assert (
        "not objective lifecycle completion authority"
        in (by_id["SHQ-G005A"].fields["reviewed_predecessor_binding"])
    )

    stage_expectations = {
        "SHQ-G006A": {
            "depends": [],
            "priority": "144",
            "bundle": "prerequisite-observer-catalog-bounded-v12",
            "files": [
                "scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py",
                "test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py",
            ],
        },
        "SHQ-G006B": {
            "depends": ["SHQ-G006A"],
            "priority": "233",
            "bundle": "prerequisite-observer-terminal-chain-bounded-v12",
            "files": [
                "scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py",
                "test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py",
            ],
        },
        "SHQ-G006": {
            "depends": ["SHQ-G006B"],
            "priority": "377",
            "bundle": "prerequisite-observer-integration-bounded-v12",
            "files": [
                ".gitignore",
                "scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py",
                "test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py",
            ],
        },
        "SHQ-G007": {
            "depends": ["SHQ-G006"],
            "priority": "610",
            "bundle": "prerequisite-observation-snapshot-bounded-v12",
            "files": [
                "artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json"
            ],
        },
    }
    for goal_id, expected in stage_expectations.items():
        goal = by_id[goal_id]
        assert goal.dependencies == expected["depends"]
        assert goal.fields["fib_priority"] == expected["priority"]
        assert goal.fields["bundle"].endswith(expected["bundle"])
        assert goal.fields["parallel_lane"].endswith(expected["bundle"])
        assert goal.predicted_files == expected["files"]
        assert goal.fields["submodules"] == (
            "ipfs_datasets_py, ipfs_kit_py, ipfs_accelerate_py/mcplusplus"
        )
        assert all(command.startswith("python3 ") for command in goal.validation_commands)

    assert by_id["SHQ-G010"].dependencies == ["SHQ-G007"]

    generated = parse_task_file(ACTIVE_TODO_PATH, "SHQ-")
    assert [
        (
            task.task_id,
            task.metadata.get("goal id", ""),
            tuple(task.depends_on),
            task.metadata.get("bundle", ""),
        )
        for task in generated
    ] == [
        (
            "SHQ-026",
            "SHQ-G006A",
            (),
            "agent-supervisor/self-hosting/prerequisite-observer-catalog-bounded-v12",
        ),
        (
            "SHQ-027",
            "SHQ-G006B",
            ("SHQ-026",),
            "agent-supervisor/self-hosting/prerequisite-observer-terminal-chain-bounded-v12",
        ),
        (
            "SHQ-028",
            "SHQ-G006",
            ("SHQ-027",),
            "agent-supervisor/self-hosting/prerequisite-observer-integration-bounded-v12",
        ),
        (
            "SHQ-029",
            "SHQ-G007",
            ("SHQ-028",),
            "agent-supervisor/self-hosting/prerequisite-observation-snapshot-bounded-v12",
        ),
    ]
    assert all(task.metadata.get("status") == "todo" for task in generated)
    assert all("SHQ-023" not in task.depends_on for task in generated)

    historical_suffixes = (
        (
            V1_HISTORY_PATH,
            "SHQ-001",
            5_845,
            69,
            "2eb0ce62a2536a911754cf54e7ebcd583ac170f23f9922ef7c0a3e61468c0fd7",
        ),
        (
            V2_HISTORY_PATH,
            "SHQ-002",
            12_095,
            133,
            "564ff7b2733aa956cb0b20b11a7d90c53f29082293986c157f0e6ece1c207a86",
        ),
        (
            V3_HISTORY_PATH,
            "SHQ-004",
            12_653,
            133,
            "0eff97f2eaf92e61b01ab7af8f32425af082483cb9417dba8fbdd6793b42023c",
        ),
        (
            V4_HISTORY_PATH,
            "SHQ-006",
            12_974,
            133,
            "7c4027e329873364a3742276d5e4582d3a997826c9b1f12a3cffd04ddb783f50",
        ),
        (
            V5_HISTORY_PATH,
            "SHQ-008",
            17_243,
            133,
            "0fea2882a697e3fb809a3f80f8e194a4978f6b4c07dc95535b71c1fe28d2b2f4",
        ),
        (
            V6_HISTORY_PATH,
            "SHQ-010",
            19_204,
            133,
            "adb335bd3cb4361fdd0bc6476f2c1c519c0df944119206fb4c80ebb54943880d",
        ),
        (
            V7_HISTORY_PATH,
            "SHQ-012",
            22_142,
            133,
            "0e296a248293e339d6c23978e49afffcdd4a24b60fe7bb9790dde9ebd3d8b5b6",
        ),
        (
            V8_HISTORY_PATH,
            "SHQ-014",
            47_553,
            200,
            "49ea659afa99acd42cf3c056ec181033ccae9cbf9dd451407da4f4d6e22e9008",
        ),
        (
            V9_HISTORY_PATH,
            "SHQ-017",
            50_074,
            200,
            "e731e0446f02cd081f41fe1861a33f83b52ff8e5b639544d07188a57e1a18d1f",
        ),
    )
    for path, first_task_id, byte_count, newline_count, expected_sha256 in historical_suffixes:
        history_bytes = path.read_bytes()
        task_bytes = history_bytes[history_bytes.index(f"## {first_task_id} ".encode()) :]
        assert len(task_bytes) == byte_count
        assert task_bytes.count(b"\n") == newline_count
        assert hashlib.sha256(task_bytes).hexdigest() == expected_sha256

    historical_dispositions = {
        V1_HISTORY_PATH: ("SHQ-001", "Historical task: true"),
        V2_HISTORY_PATH: ("SHQ-002: cancelled/retryable", "SHQ-003: never launched"),
        V3_HISTORY_PATH: ("SHQ-004: never launched", "SHQ-005: never launched"),
        V4_HISTORY_PATH: ("SHQ-006: rejected/cancelled retryable", "SHQ-007: never launched"),
        V5_HISTORY_PATH: ("SHQ-008: never launched", "SHQ-009: never launched"),
        V6_HISTORY_PATH: ("SHQ-010: rejected/cancelled retryable", "SHQ-011: never launched"),
        V7_HISTORY_PATH: (
            "SHQ-012: rejected/cancelled retryable",
            "SHQ-013: never leased or launched",
        ),
        V8_HISTORY_PATH: (
            "SHQ-014: rejected/cancelled retryable",
            "SHQ-016: never leased or launched",
        ),
        V9_HISTORY_PATH: (
            "SHQ-017: rejected/cancelled retryable",
            "SHQ-019: never leased or launched",
        ),
    }
    for path, required in historical_dispositions.items():
        text = path.read_text(encoding="utf-8")
        assert all(value in text for value in required)

    v10_history = V10_HISTORY_PATH.read_text(encoding="utf-8")
    v10_blocks = v10_history[v10_history.index("## SHQ-020 ") :]
    assert len(v10_blocks.encode("utf-8")) == 51_699
    assert hashlib.sha256(v10_blocks.encode("utf-8")).hexdigest() == (
        "78695c942525e48a8377057ffd7c0ff6c04b7237a6bfe2c7f2599037e90d4aa5"
    )

    v11_history = V11_HISTORY_PATH.read_text(encoding="utf-8")
    normalized_v11 = " ".join(v11_history.split())
    assert (
        "SHQ-023: completed and merged after the reviewed attempt/fence/token 3/3/3"
        in normalized_v11
    )
    assert "SHQ-024: rejected/cancelled retryable on all four attempts" in normalized_v11
    assert "SHQ-025: registered as coordination task CID" in normalized_v11
    for fact in (
        "0200be041e1c154660ade9c44a552df97b84dec1",
        "bbf8039a67bf2f4dafdd19ef289638d023825e22",
        "17e19a8e5db327a18dc9437a8de2be299599ecf2",
        "389048a0ee4d39b24dc68289e21a78da9ca1c4c9",
        "baguqeera5o6wzpnwezcacp5oiwycvzk5uhvrvadr7e6m6x3qdzh65ff5nktq",
        "baguqeerayifbixgmh227xewfgwza77itadvtynj5oaavccihrdxh5ftkbuoq",
        "baguqeerahs3er2kphhbtexrifryshplgoxyzprzgy5bdk2qfdfezrfzh62ma",
        "baguqeera5sf7lumgi3wgjbkq7z3gsylw7xercadfojzwnwwt5pv2c7czgd3a",
        "baguqeerakwl5iw63ul5mfpnvjhfx6tkdge5d7pkvccego5nz4yjoz63bagda",
        "baguqeeraxafvxe6hh6hzff5e6dgcudl62vij5cirxksiiftrfoubgudd7jlq",
        "baguqeeramuomp3hrb2udku6ddnwi6vahohv72e6flvb4ybqt6nv7mtiyivca",
        "baguqeerazwsoknkmne7a53qcn365itp23oirdmmas6tp66eoukcdwyekmfjq",
        "baguqeerakma3nlvtq3fum5zmu5yxwgaqk6fpvgqoe4cpmv4qn7mdtmudtnqa",
        "3f0114e0040216dafe8af1e2626fc8c82cdac88f17e885a8266f9fdf6c6489e3",
        "165563de58d8983b67990e728c168b69854898c57e0bffb8162243c950c3361f",
        "050ab44624968db8bff758377de18fa1e5c94ec3",
        "1ffa5e8d1868f2822482d9aee33a113c2dd152718a9161c386ddb9aea8a0ca9d",
    ):
        assert fact in v11_history
    assert "Token history ends exactly at 4" in v11_history
    assert "`retry_budget_repair_receipts` is empty" in v11_history
    assert "no attempt-5 claim, lease, receipt, launch, or authorization exists" in normalized_v11
    assert "operator quarantine bundle" in v11_history
    assert "prohibited non-input" in v11_history
    v11_blocks = v11_history[v11_history.index("## SHQ-023 ") :]
    assert len(v11_blocks.encode("utf-8")) == 62_976
    assert v11_blocks.count("\n") == 200
    assert hashlib.sha256(v11_blocks.encode("utf-8")).hexdigest() == (
        "bd13e307f0d0fb3a4bc8fa4ba930b7445242c57c8f9bebd381ab642b13ee53c5"
    )

    datasets_goal_ids = {
        "SHQ-G032",
        "SHQ-G033",
        "SHQ-G034",
        "SHQ-G035",
        "SHQ-G036",
        "SHQ-G037",
        "SHQ-G038",
    }
    assert {by_id[goal_id].fields["bundle"] for goal_id in datasets_goal_ids} == {
        "datasets/self-hosting/corpus"
    }
    assert all(
        by_id[goal_id].fields.get("submodules") == "ipfs_datasets_py"
        for goal_id in datasets_goal_ids
    )
    assert all(
        output.startswith(
            (
                "ipfs_datasets_py/",
                "artifacts/agent_supervisor/self_hosting_qualification/",
            )
        )
        for goal_id in datasets_goal_ids
        for output in by_id[goal_id].predicted_files
    )
    kit_goal_ids = {"SHQ-G041", "SHQ-G042", "SHQ-G043", "SHQ-G044"}
    assert all(
        output.startswith("ipfs_kit_py/")
        for goal_id in kit_goal_ids
        for output in by_id[goal_id].predicted_files
    )


def test_v12_observer_contract_reuses_authorities_and_fails_closed() -> None:
    source = OBJECTIVE_PATH.read_text(encoding="utf-8")
    plan = PLAN_PATH.read_text(encoding="utf-8")
    normalized = " ".join(source.split())
    normalized_plan = " ".join(plan.split())
    by_id = {goal.goal_id: goal for goal in _goals()}

    checkpoint_lead = (
        "This bounded task is neither resumable nor long-running; the generic "
        "durable-checkpoint clause is inapplicable and expressly revoked for the "
        "implementation agent."
    )
    for goal_id in ("SHQ-G005A", "SHQ-G006A", "SHQ-G006B", "SHQ-G006", "SHQ-G007"):
        assert by_id[goal_id].fields["acceptance"].startswith(checkpoint_lead)
        assert by_id[goal_id].fields["refinement"].startswith(checkpoint_lead)
    assert normalized.count(checkpoint_lead) == 10

    for invariant in (
        "every bounded-v11 SHQ-024 attempt 1 through 4",
        "SHQ-025 registration",
        "operator quarantine bundle",
        "prohibited non-input",
        "must not be inspected, enumerated, restored, copied, seeded",
        "freshly committed clean bounded-v12 launch HEAD/tree",
        "must not `git show`, check out, or otherwise reopen the frozen anchor",
        "one semantic implementation attempt",
        "typed transient with null output",
        "no active claim/lease/process/worktree/ref/lock",
        "semantic or contract rejection freezes",
        "never an actor switch",
    ):
        assert invariant in normalized

    ordered_catalog = (
        "IncrementalSemanticIndex",
        "SemanticCapsuleCompiler",
        "ContextPackBuilder",
        "VerificationReceiptCache",
        "IncrementalVerificationPlanner",
        "ModelRoutePlanner",
        "VerifiedGuiOptimizer",
        "IncrementalProofSealer",
        "SemanticCompressionGovernor",
        "AdversarialAssuranceEngine",
    )
    acceptance_a = by_id["SHQ-G006A"].fields["acceptance"]
    for identity in (
        "0200be041e1c154660ade9c44a552df97b84dec1",
        "aea528d467450cf6a70efa36d5ab6f34b4947fc7",
        "bbf8039a67bf2f4dafdd19ef289638d023825e22",
        "00c76524f2f9e1273b89816103a27130a551de85",
        "17e19a8e5db327a18dc9437a8de2be299599ecf2",
        "389048a0ee4d39b24dc68289e21a78da9ca1c4c9",
    ):
        assert identity in acceptance_a
        assert identity in plan
    positions = [acceptance_a.index(name) for name in ordered_catalog]
    assert positions == sorted(positions)
    for literal in (
        "ipfs_datasets_py/logic/software_contracts/semantic_index/index.py",
        "ipfs_datasets_py/logic/software_contracts/semantic_index/__init__.py",
        "scan_repository",
        "diff_repository_states",
        "calculate_invalidation",
        "verify_capsule_compile_result",
        "SemanticCapsuleCompiler@1",
        "ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack.py",
        "ipfs_accelerate_py/agent_supervisor/verification/receipt_cache.py",
        "ipfs_accelerate_py/agent_supervisor/verification/planner.py",
        "ipfs_accelerate_py/agent_supervisor/verification/model_route.py",
        "ipfs_accelerate_py/agent_supervisor/verification/__init__.py",
        "VerificationReceiptCache@1",
        "IncrementalVerificationPlanner@1",
        "ModelRoutePlanner@1",
        "owner_contract_not_declared_on_launch_tree",
        "expected_absent_pending_owner",
        "terminal_chain_not_run",
        "no authoritative filesystem receipt path",
    ):
        assert literal in acceptance_a

    for literal in (
        "ContextPacker(budget=ContextBudget(), policy=ContextCoveragePolicy(), estimator_version=TOKEN_ESTIMATOR_VERSION)",
        "ContextPacker.pack",
        "pack_context",
        "project_admission_to_reference",
        "CapsuleAdmission",
        "token_count=0",
        "ipfs-accelerate.context-pack-result@1",
        "ipfs-accelerate.context-coverage-policy@1",
        "context-compiler-calibrated_utf8@1",
    ):
        assert literal in normalized_plan
        assert literal.split("(")[0] in acceptance_a
    assert (
        "target/surrounding/test source are exact required `INVARIANT` references"
        in normalized_plan
    )
    assert "budget or coverage failure never truncates required coverage" in normalized_plan
    assert "never-compressed target/surrounding/test `INVARIANT` sources" in acceptance_a
    assert "nontruncating budget/coverage escalation" in acceptance_a
    for literal in (
        "dependency_capsule_cids",
        "token_totals",
        "escalation_recommendation",
        "ContextPackResult.to_dict()",
        "schema` and `interface",
        "never serializes the optional `production_slice` object itself",
        "`production_slice`, `production_slice_cid`",
        "`interface` and `receipt_path` are null",
    ):
        assert literal in acceptance_a
        assert literal in normalized_plan
    assert "parent package exports `ContextPack` only" in acceptance_a
    assert "there is no `ContextPackBuilder` facade" in acceptance_a
    assert "same root" not in acceptance_a
    assert "same release/board" not in acceptance_a
    for literal in (
        r"^##[ \t]+(?P<task_id>[A-Z][A-Z0-9_]*-[0-9]+)(?:[ \t]+.*)?$",
        r"^##(?:[ \t]|$)",
        r"^[ \t]*-[ \t]+Status:[ \t]*(?P<value>[^\r\n]*)[ \t]*$",
        "nonzero unique task headings",
        "exactly one status per block",
        "`completed`, `todo`, `blocked`, `in_progress`, `review`, and `cancelled`",
        "terminal iff every status is `completed`",
        "deeper headings",
    ):
        assert literal in acceptance_a
        assert literal in normalized_plan

    for literal in (
        "dependency_admissions=()",
        "counterexample_cids=()",
        "production_slice_builder=None",
        "budget=None",
        "policy=None",
        "estimator_version=TOKEN_ESTIMATOR_VERSION",
        "`pack`, `pack_cid`, `references`, `token_estimate`, `coverage_satisfied`",
        "production_slice_cid",
        "ipfs-accelerate.context-pack-result@1",
        "ipfs-accelerate.context-coverage-policy@1",
        "context-compiler-calibrated_utf8@1",
        "docs/benchmarks/semantic_compression_harness_results.json",
        "ipfs_accelerate_py/semantic-state/benchmark-report@1",
        "artifacts/agent_supervisor/incremental_verification/benchmark.json",
        "ipfs_accelerate_py/agent-supervisor/incremental-verification-benchmark@2",
        "checkout root/interface",
    ):
        assert literal in acceptance_a

    for receipt_literal in (
        "TestReceipt@1",
        "ipfs_accelerate_py/agent-supervisor/verification-test-receipt@1",
        "DirectExecutionObservation@1",
        "ipfs_accelerate_py/agent-supervisor/direct-verification-observation@1",
        "ipfs_accelerate_py/agent-supervisor/verification-process-runner@1",
        "VerificationReceiptKey.key_id",
    ):
        assert receipt_literal in plan

    integration_acceptance = by_id["SHQ-G006"].fields["acceptance"]
    terminal_acceptance = by_id["SHQ-G006B"].fields["acceptance"]
    snapshot_acceptance = by_id["SHQ-G007"].fields["acceptance"]
    for literal in (
        "Build the runtime exactly once",
        "exactly one live same-process `VerificationProcessRunner.run(VerificationCommand)`",
        "`process_started is true`",
        "`disposition == completed`",
        "`exit_code == 0`",
        "`result.ok is true`",
        "`publication_allowed is true`",
        "Before any structural projection",
        "actual contract fields",
        "`receipt_key_cid`, `repository_tree_cid`, `environment_cid`",
        "Only after that run call `VerificationIdentityCompiler.compile_key`",
        "Construct `DirectExecutionObservation`",
        "S1 == S0",
        "simulated",
        "replayed",
    ):
        assert literal in terminal_acceptance
    assert terminal_acceptance.count("VerificationProcessRunner.run(VerificationCommand)") == 1
    assert terminal_acceptance.count("VerificationIdentityCompiler.compile_key") == 1
    assert "zero successful" not in terminal_acceptance
    observation_projection = terminal_acceptance.split(
        "Construct `DirectExecutionObservation` only from its actual contract fields:", 1
    )[1].split("Each stream must", 1)[0]
    for live_only_field in (
        "`executable`",
        "`cwd`",
        "`sandbox`",
        "`network_policy`",
        "`timeout_seconds`",
        "`disposition`",
        "process/lease identity",
    ):
        assert live_only_field not in observation_projection
    for literal in (
        "(sealed_python, '-m', 'pytest', '-q', *selectors)",
        "shlex.join",
        "exact suffix after `--`",
        "build_hermetic_validation_runtime",
        "hermetic_validation_command",
        "--unshare-net",
        'b"bubblewrap 0.9.0\\n"',
        "receipt_kind=TEST",
        "adapter_schema=PROCESS_RUNNER_SCHEMA",
        "selector_argv",
        "tool_name='bwrap'",
        "captured_byte_count == byte_count == len(preview.encode('utf-8'))",
        "TestReceipt.from_dict(receipt.to_record()).to_record() == receipt.to_record()",
        "for_production=True",
        "injected phase reports",
        "cache-only authority",
    ):
        assert literal in terminal_acceptance
    for literal in (
        "O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC",
        "O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC",
        "fchmod` 0644",
        "no-clobber link",
        "Never use `os.replace` or direct target writes",
        "short/zero writes",
        "existing file/symlink and concurrent link races",
        "source races",
        "cleanup durability",
        "--mode verify-existing --artifact <path>",
        "--allow-exact-evidence-projection-child",
        "no general descendant",
        "exact two-parent supervisor merge",
    ):
        assert literal in integration_acceptance
    assert "--mode observe" not in by_id["SHQ-G007"].fields["validation"]
    assert by_id["SHQ-G007"].fields["validation"].count("--mode verify-existing") == 2
    assert "--allow-exact-evidence-projection-child" not in by_id["SHQ-G007"].fields["validation"]
    assert "execute the merged ordinary-observe CLI exactly once" in snapshot_acceptance
    assert "Validation must never invoke ordinary observe again" in snapshot_acceptance
    observe_command = (
        "python3 scripts/ops/agent_supervisor/"
        "self_hosting_qualification_prerequisites.py --repo-root . --mode observe --quiet"
    )
    assert snapshot_acceptance.count(observe_command) == 1
    assert source.count(observe_command) == 1
    assert plan.count(observe_command) == 1
    assert "default canonical target" in snapshot_acceptance

    assert 'SHQ_PROJECTION="$SHQ_DATA/projections/v12"' in plan
    assert (
        "SHQ_RUN=/home/barberb/.local/state/ipfs_accelerate_py/self-hosting-qualification-v12"
    ) in plan
    assert "self_hosting_qualification.v11_history.todo.md" in plan
    assert "projections/v11" not in plan
    assert "self-hosting-qualification-v11" not in plan

    assert plan.count('--protected-output-path "$SHQ_V11_HISTORY_TODO"') == 4
    assert plan.count('--implementation-protected-path "$SHQ_V11_HISTORY_TODO"') == 1
    assert plan.count('--protected-output-path "$SHQ_ACTIVE_TODO"') == 4
    assert plan.count('--implementation-protected-path "$SHQ_ACTIVE_TODO"') == 1
    assert (
        "--protected-output-path docs/architecture/self_hosting_qualification.todo.md" not in plan
    )
    assert (
        "--implementation-protected-path docs/architecture/self_hosting_qualification.todo.md"
        not in plan
    )
    plan_lines = plan.splitlines()
    for index, line in enumerate(plan_lines):
        if line.strip().startswith(
            (
                '--protected-output-path "$SHQ_V10_HISTORY_TODO"',
                '--implementation-protected-path "$SHQ_V10_HISTORY_TODO"',
            )
        ):
            assert plan_lines[index + 1].strip() == line.strip().replace(
                "V10_HISTORY", "V11_HISTORY"
            )

    scope_goals = ("SHQ-G006A", "SHQ-G006B", "SHQ-G006", "SHQ-G007")
    assert "--max-findings 4" in plan
    bootstrap_start = plan.index(
        '( cd "$SHQ_REPO" && "$SHQ_PYTHON" -m '
        "ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon"
    )
    bootstrap_end = plan.index("\n)\n```", bootstrap_start)
    bootstrap_lines = [line.strip() for line in plan[bootstrap_start:bootstrap_end].splitlines()]
    for goal_id in scope_goals:
        assert bootstrap_lines.count(f"--scope-goal-id {goal_id} \\") == 1
        assert bootstrap_lines.count(f"--force-goal-id {goal_id} \\") == 1
    assert not any("SHQ-G005A" in line for line in bootstrap_lines)
    assert "--assume-completed-task-id SHQ-023" not in plan
    assert "must allocate SHQ-026, SHQ-027, SHQ-028, and SHQ-029" in normalized_plan
    assert "G006A, G006B, G006, and G007" in normalized_plan
    assert (
        '-k "catalog or compatibility or api or path or board or forest or nonterminal"'
        in by_id["SHQ-G006A"].fields["validation"]
    )

    assert "prerequisite-observer-catalog-bounded-v12" in plan
    assert "prerequisite-observer-terminal-chain-bounded-v12" in plan
    assert "prerequisite-observer-integration-bounded-v12" in plan
    assert "prerequisite-observation-snapshot-bounded-v12" in plan
    assert plan.count("bounded_v12_runtime.todo.md") == 4
    assert "bounded_v11_runtime.todo.md" not in plan

    assert plan.splitlines().count('  --max-task-attempts "$SHQ_MAX_TASK_ATTEMPTS" \\') == 1
    assert plan.splitlines().count("SHQ_MAX_TASK_ATTEMPTS=1") == 1
    assert 'assert expected_max_task_attempts in {"1", "2"}' in plan
    assert "`--max-task-attempts 1` → `--max-task-attempts 2`" in normalized_plan
    assert "--max-task-attempts 5" not in plan
    assert "default three-round repair ceiling is not extra authority" in normalized_plan
    assert "operator pre-invocation gate" in normalized_plan
    for literal in (
        "implementation_attempts_by_cid[<exact canonical task CID>] == 1",
        "selection_idle_reason == all_selectable_ready_tasks_reached_max_task_attempts",
        "implementation_retry_deferred:*",
        "retry-budget-repair receipt",
    ):
        assert literal in normalized
        assert literal in normalized_plan
    assert "later manual rerun" in normalized_plan
    assert "Every outer supervisor launch uses `--start --once`" in plan
    assert "Those later invocations are successor admissions, not retries" in normalized_plan
    assert "exact predecessor `operator-stage-binding@1` path" in normalized_plan
    assert "A dash is valid only for G006A/SHQ-026." in plan
    assert "sole claimable lane" in normalized_plan
    assert "each successor resets to `--max-task-attempts 1`" in normalized_plan
    assert (
        "`launched_task_cids` equal to the singleton exact Profile-G coordination CID"
        in normalized_plan
    )
    assert "post-G010 qualification scheduler" in normalized_plan
    assert '"${SHQ_BUNDLE_ARGS[@]}" --start --once' in plan
    assert 'test ! -e "$SHQ_REPO/$SHQ_PROJECTION"' in plan
    assert 'test ! -L "$SHQ_REPO/$SHQ_PROJECTION"' in plan
    assert 'test ! -e "$SHQ_RUN"' in plan
    assert 'test ! -L "$SHQ_RUN"' in plan
    assert "SHQ_FROZEN_V11_HEAD=17e19a8e5db327a18dc9437a8de2be299599ecf2" in plan
    assert "SHQ_V12_MIGRATION_HEAD=$(git -C" in plan
    assert "SHQ_V12_MIGRATION_TREE=$(git -C" in plan
    assert "status --porcelain=v1 --untracked-files=all)" in plan
    assert "submodule foreach --recursive" in plan
    assert "rev-parse --verify 'HEAD^1')" in plan
    assert (
        'merge-base --is-ancestor \\\n  "$SHQ_FROZEN_V11_HEAD" "$SHQ_V12_MIGRATION_HEAD"' in plan
    )
    assert '"docs/architecture/self_hosting_qualification.v11_history.todo.md"' in plan
    assert '"test/api/test_agent_supervisor_self_hosting_qualification_plan.py"' in plan
    assert "task_ids_from_artifact_names" in plan
    assert 'expected = {f"SHQ-{number:03d}" for number in range(1, 26)}' in plan
    for task_id, goal_id in zip(
        ("SHQ-026", "SHQ-027", "SHQ-028", "SHQ-029"), scope_goals, strict=True
    ):
        assert f'("{task_id}", "{goal_id}"' in plan

    for literal in (
        "operator-stage-binding@1",
        "operator-stage-start-verification@1",
        "operator-stage-mismatch-cancellation@1",
        '"canonical_task_cid": member_task_cid',
        '"task_spec_cid": task_spec_cid',
        '"coordination_task_cid": coordination_task_cid',
        '"recursive_gitlink_status"',
        '"bundle_index"',
        '"dry_manifest"',
        '"implementation_envelope"',
        '"provider_environment"',
        '"predecessor_lineage": predecessor_lineage',
        '"quiescence"',
        '"captured_at"',
        'manifest.get("launched_task_cids") == [expected_coordination_cid]',
        'pool.get("base_commit") == binding["target"]["head"]',
        'implementation_started.get("command") == shlex.split(',
        'implementation_started.get("baseline_ref") == binding["target"]["head"]',
        '"task_completion": False',
        "set -euo pipefail",
        '("SHQ-G006A", "SHQ-026"): None',
        '("SHQ-G006B", "SHQ-027"): ("SHQ-G006A", "SHQ-026")',
        '("SHQ-G006", "SHQ-028"): ("SHQ-G006B", "SHQ-027")',
        '("SHQ-G007", "SHQ-029"): ("SHQ-G006", "SHQ-028")',
        "SELECT dependency_task_cid FROM task_dependencies WHERE task_cid=?",
        "shq_cancel_mismatched_stage",
        'if ! SHQ_STAGE_START_VERIFICATION=$(shq_verify_started_stage',
        "send_exact(lane_pid, observed_identity, signal.SIGTERM)",
        "send_exact(lane_pid, observed_identity, signal.SIGKILL)",
        'assert receipt_status != "succeeded"',
        'assert lease_state != "completed"',
    ):
        assert literal in plan

    assert "gpt-5.6-terra" in plan
    assert "model_context_window=49152" in plan
    assert "agents.max_threads=1" in plan
    assert "agents.max_depth=0" in plan
    assert "IPFS_ACCELERATE_AGENT_DISABLE_SUBAGENTS=1" in plan
    assert "--context-budget-tokens 24576" in plan
    assert "No fallback provider, wrapper, actor substitution" in normalized_plan
    assert "No rescue-commit or prior-attempt revision is readable" in plan
    assert "63ea88e41227d4d2d424f41051b9e9390c1a1c32" not in plan
    assert plan.count("--mode verify-existing") == 3
    assert plan.count("--allow-exact-evidence-projection-child --quiet") == 2
    assert '( cd "$SHQ_REPO" &&\n  test -z "$(git status' in plan
    assert '"$SHQ_PYTHON" -m pytest -q test/api/' in plan
    assert "--repo-root . --mode verify-existing" in plan
    assert "accepts no general descendant" in plan
    assert "exact native two-parent supervisor merge" in plan
