from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import (
    configured_board_scheduler as scheduler_module,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    multi_supervisor_runner as multi_runner_module,
)
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
    ConfiguredBoardError,
    configured_board_common_args,
    configured_board_launch_plan,
    load_configured_board,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    DATABASE_PROGRAM_JSON_ENV,
    RUNTIME_REGISTRY_PATH_ENV,
    STATE_AUTHORITY_MODE_ENV,
    STATE_ENDPOINT_SECRET_HANDLE_ENV,
    STATE_FAILOVER_POLICY_ENV,
    STATE_QUACK_ENDPOINT_ENV,
    STATE_QUACK_MUTATION_DIR_ENV,
    STATE_STORE_GENERATION_ENV,
    TASK_SOURCE_KIND_ENV,
    common_args_from_parsed_args,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    build_arg_parser as build_multi_supervisor_arg_parser,
)

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_RELATIVE = Path(
    "config/"
    "agent_supervisor_logic_governed_compositional_verification_fabric_"
    "quack_candidate_scheduler.json"
)
CANDIDATE_PATH = ROOT / CANDIDATE_RELATIVE
CANONICAL_PATH = ROOT / (
    "config/"
    "agent_supervisor_logic_governed_compositional_verification_fabric_"
    "scheduler.json"
)
BOARD_BRANCH = "agent/logic-governed-compositional-verification-fabric-v1"
RUNTIME_ROOT = (
    "data/agent_supervisor/"
    "logic_governed_compositional_verification_fabric/run-v27"
)
QUACK_OWNER = f"{RUNTIME_ROOT}/quack-owner"
QUACK_HANDLE = "env://IPFS_ACCELERATE_AGENT_QUACK_TOKEN"
QUACK_ENDPOINT = "quack:127.0.0.1:24689"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _common_args(argv: list[str]) -> list[str]:
    prefix = "--common-arg="
    return [item[len(prefix) :] for item in argv if item.startswith(prefix)]


def _option_value(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


def test_lgcvf_quack_candidate_is_additive_run_v27_fail_closed_profile() -> None:
    candidate = _load_json(CANDIDATE_PATH)
    canonical = _load_json(CANONICAL_PATH)

    # This candidate reuses the sealed LGCVF plan and board branch without
    # changing or impersonating the frozen run-v17 recovery profile.
    for field in (
        "schema",
        "taskboard_path",
        "objectives_path",
        "plan_path",
        "formal_plan_path",
        "validator_path",
        "task_prefix",
        "goal_prefix",
        "board_namespace",
        "merge_target_branch",
        "source_binding",
        "plan_binding",
        "initial_projection",
        "task_groups",
    ):
        assert candidate[field] == canonical[field]
    assert candidate["merge_target_branch"] == BOARD_BRANCH
    assert "fresh_generation_recovery" not in candidate
    assert "fresh_generation_recovery" in canonical
    assert canonical["database_program"]["authority_mode"] == "embedded"
    assert canonical["database_program"]["store_generation"] == "lgcvf-run-v17"
    assert candidate["validation_runtime"] == canonical["validation_runtime"]
    assert candidate["validation_runtime"] == {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "authority-validation-container-runtime@1"
        ),
        "backend": "authority_validation_container",
        "container_image": (
            "sha256:fbe85c882cbad09dcef78841b5c7cabc1ec0541aca2a8884d018d34c9f1732ae"
        ),
        "required_modules": ["pytest", "z3", "cvc5"],
    }

    runtime = candidate["runtime_paths"]
    program = candidate["database_program"]
    assert runtime["root"] == RUNTIME_ROOT
    assert runtime["quack_owner"] == QUACK_OWNER
    assert program["authority_mode"] == "quack"
    assert program["task_source_kind"] == "duckdb"
    assert program["endpoint_secret_handle"] == QUACK_HANDLE
    assert program["quack_endpoint"] == QUACK_ENDPOINT
    assert program["runtime_registry_path"] == runtime["quack_owner"]
    assert program["store_id"] == f"{RUNTIME_ROOT}/control.duckdb"
    assert program["store_generation"] == "lgcvf-run-v27"
    assert program["export_profile"] == "lgcvf-run-v27"
    assert program["failover_policy"] == "fail_closed"
    assert program["explicit_legacy"] is False
    assert program["claim_policy"] == {
        "schema": "ipfs_accelerate_py/agent-supervisor/database-claim-policy@1",
        "task_prefix": "LGCVF-",
        "task_shard_count": 4,
        "strict_task_sharding": True,
        "idle_lane_work_stealing": "virgin-transfer",
    }
    rendered_candidate = json.dumps(candidate, sort_keys=True)
    assert "run-v17" not in rendered_candidate
    assert "run-v23" not in rendered_candidate

    lanes = candidate["lanes"]
    assert candidate["max_lanes"] == 4
    assert candidate["strict_task_sharding"] is True
    assert candidate["idle_lane_work_stealing"] == "virgin-transfer"
    provider = candidate["provider"]
    assert provider["max_concurrency"] == 4
    assert provider["primary_provider_id"] == "grok_cli"
    assert provider["primary_model_id"] == "grok-4.6"
    assert provider["fallback_provider_id"] == "codex"
    assert provider["fallback_model_id"] == "gpt-5.6-terra"
    assert provider["fallback_trigger"] == "primary_quota_exhausted"
    assert provider["fallback_reasoning_effort"] == "high"
    assert "provider_id" not in provider
    assert [lane["index"] for lane in lanes] == list(range(4))
    assert [lane["strict_shard_remainder"] for lane in lanes] == list(range(4))
    assert len({lane["name"] for lane in lanes}) == 4
    assert candidate["bootstrap_writer_policy"] == {
        "maximum_processes": 1,
        "quack_required": False,
        "offline_single_writer_materialization_permitted": True,
        "quack_required_after_publish": True,
        "direct_multi_process_duckdb_permitted": False,
        "automatic_installation_permitted": False,
    }

    ducklake = candidate["ducklake_projection_program"]
    authority = candidate["authority_policy"]
    projection_root = f"{RUNTIME_ROOT}/ducklake-board-projection"
    assert ducklake["mode"] == "enabled_non_authoritative"
    assert ducklake["authority"] is False
    assert ducklake["scheduling_prerequisite"] is False
    assert ducklake["completion_prerequisite"] is False
    assert ducklake["may_grant_authority"] is False
    assert ducklake["catalog_path"] == f"{projection_root}/lake.ducklake"
    assert ducklake["data_path"] == f"{projection_root}/lake-data"
    assert Path(ducklake["catalog_path"]).parent != Path(
        program["store_id"]
    ).parent
    assert ducklake["root"] == projection_root
    assert authority["ducklake_projection_authoritative"] is False
    assert authority["ducklake_projection_required_for_scheduling"] is False
    assert authority["ducklake_projection_required_for_completion"] is False
    assert authority["ducklake_projection_is_completion_authority"] is False
    assert candidate["completion_policy"][
        "ducklake_outage_cannot_block_core_completion"
    ] is True

    protected = set(candidate["protected_paths"])
    assert CANDIDATE_RELATIVE.as_posix() in protected
    assert CANONICAL_PATH.relative_to(ROOT).as_posix() in protected
    assert {
        "scripts/run_logic_governed_compositional_verification_fabric_quack.py",
        "test/api/test_agent_supervisor_lgcvf_quack_successor.py",
    } <= protected


def test_lgcvf_quack_candidate_loads_and_renders_detached_launch_plan(
    monkeypatch,
) -> None:
    # Rendering must not depend on a live owner, a token, a DuckDB file, or a
    # child process. The handle is carried as a reference for later resolution.
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", raising=False)

    def _unexpected_process_start(*_args, **_kwargs):
        raise AssertionError("launch-plan rendering started a process")

    monkeypatch.setattr(
        scheduler_module.subprocess,
        "Popen",
        _unexpected_process_start,
    )
    board = load_configured_board(CANDIDATE_PATH, repo_root=ROOT)
    plan = configured_board_launch_plan(
        board,
        implement=True,
        detach=True,
        duration_seconds=60,
        stamp="20260824T000000Z",
    )

    assert plan["implementation_branch"] == BOARD_BRANCH
    assert plan["lanes"] == 4
    assert plan["strict_task_sharding"] is True
    assert plan["idle_lane_work_stealing"] == "virgin-transfer"
    assert plan["effective_strict_task_sharding"] is True
    assert plan["effective_idle_lane_work_stealing"] == "virgin-transfer"
    assert plan["detach"] is True
    assert plan["plan_bound_dispatch"] is False
    assert "fresh_generation_recovery_admission" not in plan
    assert plan["database_program"]["authority_mode"] == "quack"
    assert plan["database_program"]["runtime_registry_path"] == QUACK_OWNER

    argv = plan["argv"]
    parsed = build_multi_supervisor_arg_parser().parse_args(argv)
    effective_common = common_args_from_parsed_args(parsed)
    assert effective_common.count("--strict-task-sharding") == 1
    assert effective_common.count("--idle-lane-work-stealing") == 1
    assert _option_value(
        effective_common,
        "--idle-lane-work-stealing",
    ) == "virgin-transfer"
    assert _option_value(argv, "--implementation-supervisor-lanes-per-track") == "4"
    assert "--implementation-supervisor-strict-task-sharding" in argv
    assert _option_value(
        argv,
        "--implementation-supervisor-idle-lane-work-stealing",
    ) == "virgin-transfer"
    assert "--exit-when-all-tracks-terminal" in argv
    assert "--detach" in argv
    common = _common_args(argv)
    assert _option_value(common, "--authority-mode") == "quack"
    assert _option_value(common, "--task-source-kind") == "duckdb"
    assert _option_value(common, "--state-failover-policy") == "fail_closed"
    assert _option_value(common, "--endpoint-secret-handle") == QUACK_HANDLE
    assert _option_value(common, "--quack-endpoint") == QUACK_ENDPOINT
    assert _option_value(common, "--runtime-registry-path") == QUACK_OWNER
    assert _option_value(common, "--merge-target-branch") == BOARD_BRANCH
    assert _option_value(
        common,
        "--idle-lane-work-stealing",
    ) == "virgin-transfer"

    environment = plan["environment"]
    assert environment[STATE_AUTHORITY_MODE_ENV] == "quack"
    assert environment[TASK_SOURCE_KIND_ENV] == "duckdb"
    assert environment[STATE_FAILOVER_POLICY_ENV] == "fail_closed"
    assert environment[STATE_ENDPOINT_SECRET_HANDLE_ENV] == QUACK_HANDLE
    assert environment[STATE_QUACK_ENDPOINT_ENV] == QUACK_ENDPOINT
    assert environment[STATE_STORE_GENERATION_ENV] == "lgcvf-run-v27"
    expected_owner = str((ROOT / QUACK_OWNER).resolve())
    assert environment[RUNTIME_REGISTRY_PATH_ENV] == expected_owner
    assert environment[STATE_QUACK_MUTATION_DIR_ENV] == (
        f"{expected_owner}/mutations"
    )
    assert "IPFS_ACCELERATE_AGENT_QUACK_TOKEN" not in environment
    rendered_program = json.loads(environment[DATABASE_PROGRAM_JSON_ENV])
    assert rendered_program["runtime_registry_path"] == QUACK_OWNER
    assert rendered_program["endpoint_secret_handle"] == QUACK_HANDLE


def _fake_lgcvf_live_context() -> SimpleNamespace:
    candidate_sha256 = hashlib.sha256(CANDIDATE_PATH.read_bytes()).hexdigest()
    return SimpleNamespace(
        capsule_pin=SimpleNamespace(
            candidate_config_sha256=f"sha256:{candidate_sha256}",
        ),
        capsule_descriptor=101,
        capsule_pin_json="sealed-capsule-pin",
        admission=SimpleNamespace(
            lane_names=tuple(
                f"lgcvf-quack-lane-{index}" for index in range(4)
            ),
        ),
        admission_json="sealed-live-admission",
        native_launch_json="sealed-native-launch",
        native_descriptor=102,
    )


def test_lgcvf_live_scheduler_admits_exact_virgin_transfer_policy(
    monkeypatch,
) -> None:
    board = load_configured_board(CANDIDATE_PATH, repo_root=ROOT)
    context = _fake_lgcvf_live_context()
    monkeypatch.setattr(
        scheduler_module,
        "verify_lgcvf_configured_board_live_context",
        lambda **_kwargs: context,
    )
    live_arguments = {
        "configured_board_live_capsule_pin_json": context.capsule_pin_json,
        "configured_board_live_capsule_descriptor": context.capsule_descriptor,
        "configured_board_live_admission_json": context.admission_json,
        "configured_board_live_native_launch_json": context.native_launch_json,
        "configured_board_live_native_descriptor": context.native_descriptor,
    }

    plan = configured_board_launch_plan(
        board,
        implement=True,
        detach=False,
        duration_seconds=60,
        stamp="20260825T000000Z",
        **live_arguments,
    )

    assert plan["idle_lane_work_stealing"] == "virgin-transfer"
    assert plan["effective_idle_lane_work_stealing"] == "virgin-transfer"
    assert _option_value(
        plan["argv"],
        "--implementation-supervisor-idle-lane-work-stealing",
    ) == "virgin-transfer"
    with pytest.raises(
        ConfiguredBoardError,
        match="does not match the exact foreground four-lane Quack board",
    ):
        configured_board_launch_plan(
            replace(board, idle_lane_work_stealing=""),
            implement=True,
            detach=False,
            duration_seconds=60,
            stamp="20260825T000000Z",
            **live_arguments,
        )


def test_lgcvf_live_runner_binds_policy_from_capsule_to_lane_argv(
    monkeypatch,
) -> None:
    candidate = _load_json(CANDIDATE_PATH)
    board = load_configured_board(CANDIDATE_PATH, repo_root=ROOT)
    state_dir = board.path(board.runtime_paths["state"])
    track_spec = multi_runner_module.implementation_supervisor_compact_track_spec(
        name="lgcvf-quack-lane",
        script_path=ROOT / multi_runner_module.PLAN_BOUND_ACCEPTED_ENTRY_PATH,
        state_dir=state_dir,
        state_prefix="lgcvf",
    )
    tracks = multi_runner_module.expand_implementation_track_lanes(
        track_spec,
        stamp="20260825T000000Z",
        lanes_per_track=4,
    )
    common = configured_board_common_args(board, implement=True)
    monkeypatch.setattr(
        multi_runner_module,
        "_lgcvf_configured_board_live_embedded_config",
        lambda _context: candidate,
    )

    admitted = multi_runner_module._verify_lgcvf_configured_board_live_profile(
        tracks=tracks,
        repo_root=ROOT,
        common_args=common,
        context=object(),
    )

    assert tuple(track.name for track in admitted) == tuple(
        f"lgcvf-quack-lane-{index}" for index in range(4)
    )
    assert common.count("--idle-lane-work-stealing") == 1
    assert _option_value(common, "--idle-lane-work-stealing") == (
        "virgin-transfer"
    )
    tampered_common = list(common)
    policy_index = tampered_common.index("--idle-lane-work-stealing") + 1
    tampered_common[policy_index] = ""
    with pytest.raises(
        ValueError,
        match="--idle-lane-work-stealing differs from the capsule",
    ):
        multi_runner_module._verify_lgcvf_configured_board_live_profile(
            tracks=tracks,
            repo_root=ROOT,
            common_args=tampered_common,
            context=object(),
        )

    without_policy = dict(candidate)
    without_policy["idle_lane_work_stealing"] = ""
    monkeypatch.setattr(
        multi_runner_module,
        "_lgcvf_configured_board_live_embedded_config",
        lambda _context: without_policy,
    )
    with pytest.raises(ValueError, match="closed live profile"):
        multi_runner_module._verify_lgcvf_configured_board_live_profile(
            tracks=tracks,
            repo_root=ROOT,
            common_args=common,
            context=object(),
        )
