"""Fail-closed invariants for the operator-authored qualification plan."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor import goal_graph, parse_goal_heap
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    external_authority_goal_fence,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = REPO_ROOT / "docs/architecture/SELF_HOSTING_QUALIFICATION_PLAN.md"
OBJECTIVE_PATH = (
    REPO_ROOT / "docs/architecture/self_hosting_qualification.objectives.md"
)
ACTIVE_TODO_PATH = (
    REPO_ROOT / "docs/architecture/self_hosting_qualification.todo.md"
)
V1_HISTORY_PATH = (
    REPO_ROOT / "docs/architecture/self_hosting_qualification.v1_history.todo.md"
)
V2_HISTORY_PATH = (
    REPO_ROOT / "docs/architecture/self_hosting_qualification.v2_history.todo.md"
)
V3_HISTORY_PATH = (
    REPO_ROOT / "docs/architecture/self_hosting_qualification.v3_history.todo.md"
)
V4_HISTORY_PATH = (
    REPO_ROOT / "docs/architecture/self_hosting_qualification.v4_history.todo.md"
)
V5_HISTORY_PATH = (
    REPO_ROOT / "docs/architecture/self_hosting_qualification.v5_history.todo.md"
)
V6_HISTORY_PATH = (
    REPO_ROOT / "docs/architecture/self_hosting_qualification.v6_history.todo.md"
)
V7_HISTORY_PATH = (
    REPO_ROOT / "docs/architecture/self_hosting_qualification.v7_history.todo.md"
)


def _goals():
    return parse_goal_heap(OBJECTIVE_PATH.read_text(encoding="utf-8"))


def _descendant_closure(goals, seed_goal_ids: set[str]) -> set[str]:
    closure = set(seed_goal_ids)
    while True:
        expanded = closure | {
            goal.goal_id
            for goal in goals
            if closure.intersection(goal.parent_goal_ids)
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

    assert len(goals) == 49
    assert len(goal_id_set) == len(goal_ids)
    assert hierarchy["roots"] == ["SHQ-G000"]
    assert all(
        edge["from"] in goal_id_set and edge["to"] in goal_id_set
        for edge in hierarchy["edges"]
    )

    # Parent and explicit dependency relations are both prerequisites.  Check
    # the union so a cycle split across the two native fields cannot hide from
    # the hierarchy-only projection.
    prerequisites = {
        goal.goal_id: set(goal.parent_goal_ids) | set(goal.dependencies)
        for goal in goals
    }
    assert all(
        prerequisite in goal_id_set
        for values in prerequisites.values()
        for prerequisite in values
    )
    assert all(goal_id not in values for goal_id, values in prerequisites.items())

    dependents: dict[str, set[str]] = {goal_id: set() for goal_id in goal_ids}
    remaining = {
        goal_id: set(values) for goal_id, values in prerequisites.items()
    }
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
        goal_id: sorted(values)
        for goal_id, values in remaining.items()
        if values
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
    external_work_goals = [
        goal for goal in work_goals if goal.requires_external_completion
    ]
    local_work_goals = [
        goal for goal in work_goals if not goal.requires_external_completion
    ]

    assert "endomorphosis/ipfs_kit_py:ipfs_kit_py/core/wal" in source
    assert "`core.operation_contracts` is a read-only dependency" in normalized_source
    assert len(work_goals) == 41
    assert len(local_work_goals) == 39
    assert {goal.goal_id for goal in external_work_goals} == {
        "SHQ-G010",
        "SHQ-G072",
    }
    assert all(goal.fields.get("gap_task") for goal in work_goals)
    assert all(goal.predicted_files for goal in work_goals)
    assert all(goal.validation_commands for goal in work_goals)

    # These interfaces are the ownership seams: datasets owns task meaning and
    # evaluation, kit owns durable evidence/release state, MCP++ owns only the
    # wire schema, and accelerate owns orchestration/runtime behavior.
    assert "SelfHostingTaskCorpus" in by_id["SHQ-G032"].fields["interfaces"]
    assert "compare_task_outcomes" in by_id["SHQ-G036"].fields["interfaces"]
    assert "QualificationArtifactStore" in by_id["SHQ-G041"].fields["interfaces"]
    assert "create_qualification_manifest" in by_id["SHQ-G044"].fields["interfaces"]
    assert "QualificationRuntimePort@1" in by_id["SHQ-G031"].fields["interfaces"]
    assert "GovernedCodingAgentRuntime" in by_id["SHQ-G052"].fields["interfaces"]
    assert "SelfHostingQualificationHarness" in by_id["SHQ-G053"].fields["interfaces"]

    assert by_id["SHQ-G005A"].predicted_files == [
        "ipfs_accelerate_py/agent_supervisor/verification/contracts.py",
        "test/api/test_agent_supervisor_verification_contracts.py",
    ]
    assert by_id["SHQ-G006"].predicted_files == [
        ".gitignore",
        "scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py",
        "test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py",
    ]
    assert by_id["SHQ-G007"].predicted_files == [
        "artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json"
    ]
    assert by_id["SHQ-G006"].dependencies == ["SHQ-G005A"]
    assert by_id["SHQ-G007"].dependencies == ["SHQ-G006"]
    assert by_id["SHQ-G010"].dependencies == ["SHQ-G007"]
    local_bootstrap_goal_ids = ("SHQ-G005A", "SHQ-G006", "SHQ-G007")
    assert len(
        {by_id[goal_id].fields["bundle"] for goal_id in local_bootstrap_goal_ids}
    ) == 3
    assert len(
        {
            by_id[goal_id].fields["parallel_lane"]
            for goal_id in local_bootstrap_goal_ids
        }
    ) == 3
    assert all(
        by_id[goal_id].fields["bundle"].endswith("bounded-v8")
        and by_id[goal_id].fields["parallel_lane"].endswith("bounded-v8")
        for goal_id in local_bootstrap_goal_ids
    )
    assert all(
        command.startswith("python3 ")
        for goal_id in local_bootstrap_goal_ids
        for command in by_id[goal_id].validation_commands
    )

    active_todo = ACTIVE_TODO_PATH.read_text(encoding="utf-8")
    history = V1_HISTORY_PATH.read_text(encoding="utf-8")
    v2_history = V2_HISTORY_PATH.read_text(encoding="utf-8")
    v3_history = V3_HISTORY_PATH.read_text(encoding="utf-8")
    v4_history = V4_HISTORY_PATH.read_text(encoding="utf-8")
    v5_history = V5_HISTORY_PATH.read_text(encoding="utf-8")
    v6_history = V6_HISTORY_PATH.read_text(encoding="utf-8")
    v7_history = V7_HISTORY_PATH.read_text(encoding="utf-8")
    normalized_v5_history = " ".join(v5_history.split())
    normalized_v6_history = " ".join(v6_history.split())
    normalized_v7_history = " ".join(v7_history.split())
    assert "## SHQ-001 " not in active_todo
    assert "## SHQ-001 " in history
    assert "- Status: blocked" in history
    assert "- Completion: superseded:SHQ-002" in history
    assert "- Historical task: true" in history
    assert "- Is schedulable: false" in history
    assert "## SHQ-002 " in v2_history
    assert "## SHQ-003 " in v2_history
    assert "SHQ-002: cancelled/retryable" in v2_history
    assert "superseded by SHQ-004" in v2_history
    assert "SHQ-003: never launched" in v2_history
    assert "superseded by SHQ-005" in v2_history
    assert v2_history.count("- Status: todo") == 2
    assert v2_history.count("- Completion: manual") == 2
    assert v2_history.count("- Is schedulable: true") == 2
    assert "## SHQ-004 " in v3_history
    assert "## SHQ-005 " in v3_history
    assert "SHQ-004: never launched" in v3_history
    assert "SHQ-005: never launched" in v3_history
    assert "Neither task was submitted to coordination" in v3_history
    assert "SHQ-006: rejected/cancelled retryable" in v4_history
    assert "SHQ-007: never launched" in v4_history
    assert "superseded by the bounded-v5 G006 projection" in v4_history
    assert "superseded by the bounded-v5 G007 projection" in v4_history
    v4_blocks = v4_history[v4_history.index("## SHQ-006 ") :]
    assert hashlib.sha256(v4_blocks.encode("utf-8")).hexdigest() == (
        "7c4027e329873364a3742276d5e4582d3a997826c9b1f12a3cffd04ddb783f50"
    )
    assert "SHQ-008: never launched" in normalized_v5_history
    assert "SHQ-009: never launched" in normalized_v5_history
    assert "prelaunch receipt-schema correction" in normalized_v5_history
    assert "superseded by the bounded-v6 G006 projection" in normalized_v5_history
    assert "superseded by the bounded-v6 G007 projection" in normalized_v5_history
    assert "Neither task was submitted to coordination" in normalized_v5_history
    v5_blocks = v5_history[v5_history.index("## SHQ-008 ") :]
    assert hashlib.sha256(v5_blocks.encode("utf-8")).hexdigest() == (
        "0fea2882a697e3fb809a3f80f8e194a4978f6b4c07dc95535b71c1fe28d2b2f4"
    )
    assert "SHQ-010: rejected/cancelled retryable after attempt 1" in normalized_v6_history
    assert "SHQ-011: never launched" in normalized_v6_history
    assert "redirected the authorized rescue commit's `git show` output" in normalized_v6_history
    assert "`/tmp/prior_observer.py` and `/tmp/prior_test.py`" in normalized_v6_history
    assert "superseded by the bounded-v7 G006 projection" in normalized_v6_history
    assert "superseded by the bounded-v7 G007 projection" in normalized_v6_history
    v6_blocks = v6_history[v6_history.index("## SHQ-010 ") :]
    assert hashlib.sha256(v6_blocks.encode("utf-8")).hexdigest() == (
        "adb335bd3cb4361fdd0bc6476f2c1c519c0df944119206fb4c80ebb54943880d"
    )
    assert "SHQ-012: rejected/cancelled retryable after attempt 1" in normalized_v7_history
    assert "SHQ-013: never leased or launched" in normalized_v7_history
    assert "b1ea78f66073b5ceb6c22375cafc4bd80d0e1eec" in v7_history
    assert "baguqeeraokrailmmvgz3vc5tm6lcj2ttovwg6lxmtvgldvfrmlxkicucxsbq" in v7_history
    assert "baguqeerakoa6upvffhceogv5rolwg4bxdwdcdnfwphni6fqsn7nffgt2z2za" in v7_history
    assert "`cancelled:retryable` with null output" in normalized_v7_history
    assert "575c48e0d4ade5b7f38dc330499d7e62a00dcb104a768f7051473c2995ab014a" in v7_history
    assert "1 failed/29 passed" in normalized_v7_history
    assert "unavailable result with an empty receipt key" in normalized_v7_history
    assert "after a successful live process-runner execution" in normalized_v7_history
    assert "No `implementation_finished`, implementation commit, or merge occurred" in normalized_v7_history
    assert "superseded by the bounded-v8 G006 projection" in normalized_v7_history
    assert "superseded by the bounded-v8 G007 projection" in normalized_v7_history
    v7_blocks = v7_history[v7_history.index("## SHQ-012 ") :]
    assert hashlib.sha256(v7_blocks.encode("utf-8")).hexdigest() == (
        "0e296a248293e339d6c23978e49afffcdd4a24b60fe7bb9790dde9ebd3d8b5b6"
    )
    # This test is intentionally valid on both sides of the reviewed tracked
    # v8 migration. Before migration the v7 active blocks must be exact; after
    # migration the board is title-only until v8 allocates SHQ-014/015/016.
    if "## SHQ-012 " in active_todo:
        assert active_todo[active_todo.index("## SHQ-012 ") :] == v7_blocks
    else:
        assert active_todo.strip() == "# Objective Todo" or (
            "## SHQ-014 " in active_todo
            and "## SHQ-015 " in active_todo
            and "## SHQ-016 " in active_todo
            and "## SHQ-012 " not in active_todo
            and "## SHQ-013 " not in active_todo
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
    assert {
        by_id[goal_id].fields["bundle"] for goal_id in datasets_goal_ids
    } == {"datasets/self-hosting/corpus"}
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


def test_v8_observer_contract_reuses_authorities_and_fails_closed() -> None:
    source = OBJECTIVE_PATH.read_text(encoding="utf-8")
    plan = PLAN_PATH.read_text(encoding="utf-8")
    normalized = " ".join(source.split())
    normalized_plan = " ".join(plan.split())

    assert "PrerequisiteTestReceipt@1" not in source
    assert "PYTEST_VERIFICATION_ADAPTER_SCHEMA" not in source
    for authority in (
        "verification.contracts.VerificationIdentityCompiler",
        "verification.process_runner.PROCESS_RUNNER_SCHEMA",
        "verification.process_runner.VerificationProcessRunner",
        "verification.process_runner.VerificationCommand",
        "verification.process_runner.VerificationStreamArtifact",
        "validation.validation_runtime.build_hermetic_validation_runtime",
        "validation.validation_runtime.hermetic_validation_command",
        "verification.contracts.TestReceipt@1",
        "verification.contracts.DirectExecutionObservation@1",
        "verification.receipt_cache.VerificationReceiptCache",
    ):
        assert authority in source
    for alias_invariant in (
        'frozenset({("bwrap", "bubblewrap")})',
        "private immutable closed alias constant",
        "exact standalone `bubblewrap` token",
        'actual raw probe bytes `bubblewrap 0.9.0\\n`',
        "The declared and keyed `tool_name` remains exact `bwrap`",
        "no caller argument, environment value, configuration, adapter, or subclass",
        "`notbubblewrap`, `bubblewrap-helper`, `not-bwrap`",
        "wrong/missing/subtoken versions",
        "Existing pytest, mypy, and all ordinary exact-name behavior remain unchanged",
        "never permits a wrapper or synthetic probe output",
    ):
        assert alias_invariant in normalized
    for invariant in (
        "Prior attempt 1 was hard-rejected for redirecting permitted `git show` stdout to host `/tmp` and rereading it",
        "outside-checkout redirect, tee, copy, save, cache, checkpoint, materialization, or read is an immediate hard rejection",
        "stop before validation",
        "exact authorized seed written straight to its matching declared output path",
        "inspect only the two named blobs at commit `63ea88e41227d4d2d424f41051b9e9390c1a1c32`",
        "Do not access any other revision or path",
        "exact non-empty ordered list of ten unique requested systems",
        "absolute root or `..` component",
        "existing parent and symlink",
        "exact module-level definition or assignment",
        "exact package export",
        "same-process `VerificationProcessRunner.run(VerificationCommand)`",
        "For terminal admission only, the mandatory chain",
        "terminal-admission steps (3)-(8)",
        "skipping steps (3)-(7) but still performing the stable observation-manifest step (8)",
        'exact actually observed `/usr/bin/bwrap --version` bytes `b"bubblewrap 0.9.0\\n"`',
        "exact outer Bubblewrap argv as `selector_argv`",
        "compiler `tool_name` is the bwrap basename, never pytest",
        "TestReceipt.from_dict(receipt.to_record()).to_record() == receipt.to_record()",
        "admit(receipt, for_production=True, require_production_eligible=True)",
        "lookup(key, for_production=True)",
        "authority comes only from the observer's live in-process isolated runner call",
        "Missing Bubblewrap, namespace denial, isolation startup failure",
        "unisolated fallback makes the test evidence unverifiable",
        "live result's executable, cwd, environment, sandbox, network policy, timeout, disposition",
        "observation's stdout/stderr CIDs to equal the live result",
        "injected phase report",
        "present real run result",
        'captured_byte_count == byte_count == len(preview.encode("utf-8"))',
        "rehash the exact preview bytes",
        "do not claim the discarded runner temporary bytes were persisted",
        "digest-mismatched, or CID-mismatched",
        "optional corroboration and insufficient",
        "outer/tree/gitlink/submodule/tracked-content source identity",
        "canonical repository-relative values",
        "git check-ignore -q --no-index -- artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json",
        ".prerequisite_observation.<nonce>.json",
        "final `*.json` ignore rule while excluding the exact target exception",
        "absence from recursive porcelain while the owned temp fd is open",
        "nonignored or target-exception-matching temp is a hard failure",
        "self-contained clean temporary Git fixture",
        "fixture success cannot upgrade the real incomplete forest",
        "terminal:false",
        "identical degraded-closure reasons",
        "final whole-snapshot two-phase revalidation counterexamples",
        "same-directory exclusive temporary file",
        "No incomplete, stale, partially validated, or source-raced artifact",
        "two named blobs at commit `63ea88e41227d4d2d424f41051b9e9390c1a1c32`",
        "two declared outputs are the sole persistence targets",
        "No intermediate or scratch copy is evidence or authority",
        "host path outside the disposable checkout as a discovery source or scratch sink",
        "host `/tmp`, supervisor/checkpoint/state directories, or sibling worktrees",
        "Generic checkpoint instructions grant no task-input authority",
        "internal ephemeral stream capture and the validation namespace's private `/tmp`",
        "neither is a discovery source nor persisted evidence",
        "freshly projected bounded-v8 G006 canonical task CID as the sole predecessor identity",
        "retired display ID, alias, canonical key, CID, worktree, receipt, or merge",
        "The exact fresh predecessor is SHQ-015",
        "not the compatibility task SHQ-014 or retired SHQ-012",
        "O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC",
        "fchmod` 0644",
        "os.link(temp_name, target_name",
        "Never use `os.replace` or direct target writes",
        "unlink target through the dirfd and `fsync` parent",
    ):
        assert invariant in normalized

    assert "PROCESS_RUNNER_SCHEMA" in plan
    assert "PYTEST_VERIFICATION_ADAPTER_SCHEMA" not in plan
    assert 'SHQ_PROJECTION="$SHQ_DATA/projections/v8"' in plan
    assert "self_hosting_qualification.v4_history.todo.md" in plan
    assert "self_hosting_qualification.v5_history.todo.md" in plan
    assert "self_hosting_qualification.v6_history.todo.md" in plan
    assert "self_hosting_qualification.v7_history.todo.md" in plan
    assert "must allocate SHQ-014, SHQ-015, and SHQ-016" in plan
    assert "leave SHQ_ACTIVE_TODO title-only" in plan
    assert "SHQ_PYTHON=/usr/bin/python3.12" in plan
    assert (
        "python3 -m pytest -q "
        "test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py"
    ) in plan
    assert (
        '"$SHQ_PYTHON" -m pytest -q '
        "test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py"
    ) not in plan
    assert "SHQ_RUN=/home/barberb/.local/state/ipfs_accelerate_py/self-hosting-qualification-v8" in plan
    assert "SHQ_RUN=/home/barberb/.local/state/ipfs_accelerate_py/self-hosting-qualification-v7" not in plan
    assert "prerequisite-observer-implementation-bounded-v8" in plan
    assert "prerequisite-observation-snapshot-bounded-v8" in plan
    assert "verification-banner-alias-compatibility-bounded-v8" in plan
    assert "--max-findings 3" in plan
    for goal_id in ("SHQ-G005A", "SHQ-G006", "SHQ-G007"):
        assert f"--scope-goal-id {goal_id}" in plan
        assert f"--force-goal-id {goal_id}" in plan
    assert "Prior attempt 1 was hard-rejected" in normalized_plan
    assert "outside-checkout redirect, tee, copy, save, cache" in normalized_plan
    assert "immediate hard rejection; stop" in normalized_plan
    assert "matching declared output path inside the checkout" in normalized_plan
    assert "sole persistence targets for code" in normalized_plan
    assert "every other revision or path is prohibited" in normalized_plan
    assert "host path outside the disposable checkout as a discovery source or scratch sink" in normalized_plan
    assert "Generic checkpoint instructions grant no task input authority" in normalized_plan
    assert "does not prohibit the required process runner's internal" in normalized_plan
    assert "actual raw `/usr/bin/bwrap --version` bytes" in normalized_plan
    assert "adapter is `PROCESS_RUNNER_SCHEMA`" in normalized_plan
    assert "incomplete committed recursive gitlink closure" in normalized_plan
    assert "ten-row `terminal:false` observation" in normalized_plan
    assert "returns rc1 and writes nothing" in normalized_plan
    assert "self-contained clean temporary Git fixture" in normalized_plan
    assert "isolated, nonpersisted cache" in normalized_plan
    assert "never serialized, injected, or substituted as current checkout evidence" in normalized_plan
    assert ".prerequisite_observation.<nonce>.json" in normalized_plan
    assert "final `*.json` ignore rule" in normalized_plan
    assert "recursive porcelain omits it while its owned fd is open" in normalized_plan
    assert "A nonignored or exception-matching temp fails" in normalized_plan
    assert "negative tests reject a nonignored or target-exception-matching temp" in normalized_plan
    assert 'test ! -e "$SHQ_RUN"' in plan
    assert '--provider-capacity-path "$SHQ_CAPACITY_PATH"' in plan
    assert '--state-root "$SHQ_RUN/state"' in plan
    assert '--coordination-path "$SHQ_RUN/state/coordination.duckdb"' in plan
    assert "bounded-v4/state" not in plan
    assert "bounded-v5/state" not in plan
    assert "bounded-v6/state" not in plan
    assert "bounded-v7/state" not in plan
